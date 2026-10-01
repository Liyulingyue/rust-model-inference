"""Convert GLiNER2.5 boundary-family checkpoints to F32 GGUF.

Variant: ``architecture="boundary"`` (BoundaryExtractor). Bundles all
multi-task heads verbatim:

  - encoder (DeBERTa-v3-base for ``gliner2.5-base-v1``)
  - boundary_head (sparse proposal + pair scoring + content pooler)
  - classifier (2-layer head: ``classifier.0.weight (1536, 768)`` +
    ``classifier.3.weight (1, 1536)`` for base-v1)
  - relation_scorer
  - record_decoder

The Rust inference path for the encoder is the same as Decide's
(``src/models/gliner/compute.rs``); this converter emits the GGUF so
that:
  - encoder weights land on the same tensor names as Decide's GGUF
    (``token_embd.weight``, ``blk.X.*``, ``rel_embeddings.weight``,
    ``classifier.0.weight``, ...);
  - all extra heads are emitted with their original safetensors names
    so a future Rust BoundaryExtractor forward can pick them up
    directly (without a name map).

The active classifier layer indices differ from Decide
(``classifier.2.*`` vs ``classifier.3.*``); the GGUF writes them as
``classifier.0.*`` and ``classifier.3.*`` to match the source and
preserves the index so future code can detect the boundary variant.

The ``boundary_head`` settings block from ``config.json`` is transcribed
into flat ``gliner2.boundary.*`` metadata (flags, dims, temperatures) and
cross-checked against the bundled head tensor shapes, so the Rust loader
selects feature sources from the checkpoint instead of guessing them.
See ``src/models/gliner_boundary/`` for the consumer.

Run with:
    models/.venv/bin/python -m tools.converter.gliner.convert_boundary \\
        models/gliner2.5-base-v1 \\
        models/gliner2.5-base-v1/gliner2.5-base-v1-f32.gguf
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

from tools.converter.utils.gguf import GgufWriter, GGML_F32, gguf_dims, open_safetensors

ARCH = "gliner2"

# DeBERTa-v3-base dimensions for ``microsoft/deberta-v3-base`` (base-v1).
# Unlike ``convert_gliner.py`` (which hardcodes DeBERTa-v3-large), this
# converter uses base dims because the boundary-family checkpoints ship
# only with DeBERTa-v3-base + mDeBERTa-v3-base.
ENCODER = {
    "model_name": "microsoft/deberta-v3-base",
    "hidden_size": 768,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "intermediate_size": 3072,
    "hidden_act": "gelu",
    "layer_norm_eps": 1e-7,
    "relative_attention": True,
    "position_buckets": 256,
    "max_position_embeddings": 512,
    "max_relative_positions": -1,
    "position_biased_input": False,
    "type_vocab_size": 0,
    "norm_rel_ebd": "layer_norm",
    "pos_att_type": "p2c|c2p",
    "share_att_key": True,
}

SCALE_DIVISOR = 3

# Bundle tensors that aren't part of the encoder / classifier and
# should be carried through verbatim for the future BoundaryExtractor
# Rust forward. Keyed by the safetensors prefix they share.
BUNDLED_HEAD_PREFIXES = (
    "boundary_head.",
    "relation_scorer.",
    "record_decoder.",
)


# ---------------------------------------------------------------------------
# SPM / tokenizer helpers (mirror convert_gliner.py; we re-implement only the
# minimum needed so this script can stand alone without an import cycle).
# ---------------------------------------------------------------------------

def _varint(buf, pos):
    result, shift = 0, 0
    while True:
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, pos
        shift += 7


def _fields(buf):
    pos = 0
    while pos < len(buf):
        key, pos = _varint(buf, pos)
        field, wire = key >> 3, key & 7
        if wire == 0:
            value, pos = _varint(buf, pos)
        elif wire == 1:
            yield field, buf[pos:pos + 8]
            pos += 8
            continue
        elif wire == 2:
            length, pos = _varint(buf, pos)
            value = buf[pos:pos + length]
            pos += length
        elif wire == 5:
            value, pos = buf[pos:pos + 4], pos + 4
        else:
            raise ValueError(f"unsupported wire {wire}")
        yield field, value


SPECIAL_TOKENS = (
    # [MASK] is part of the SPM vocab at id 128000 for DeBERTa-v3;
    # boundary-family configs (gliner2.5-base-v1) do not list it under
    # extra_special_tokens because it ships with the base tokenizer.
    "[SEP_STRUCT]", "[SEP_TEXT]", "[P]", "[C]", "[E]", "[R]",
    "[L]", "[EXAMPLE]", "[OUTPUT]", "[DESCRIPTION]",
)


def parse_spm(buf):
    pieces, scores, types = [], [], []
    normalizer = {}
    for field, value in _fields(buf):
        if field == 1:  # pieces
            piece, score, kind = [], None, None
            for sub, sub_value in _fields(value):
                if sub == 1:
                    piece.append(sub_value.decode())
                elif sub == 2:
                    score = 0.0
                    for v, _ in _fields(sub_value):
                        score = float(int.from_bytes(sub_value, "little")) / 10 ** 6 if False else 0.0
                elif sub == 3:
                    kind = sub_value.decode()
            if piece:
                pieces.append("".join(piece))
                if score is not None:
                    scores.append(score)
                if kind is not None:
                    types.append(kind)
        elif field == 2:  # TrainerSpec
            byte_fallback = next((v for v, _ in _fields(value) if True), None)
        elif field == 3:  # NormalizerSpec
            for sub, sub_value in _fields(value):
                if sub == 1:
                    normalizer["name"] = sub_value.decode()
                elif sub == 2:
                    normalizer["charsmap"] = sub_value
                elif sub == 3:
                    normalizer["add_dummy_prefix"] = bool(int.from_bytes(sub_value, "little"))
                elif sub == 4:
                    normalizer["remove_extra_whitespaces"] = bool(int.from_bytes(sub_value, "little"))
                elif sub == 5:
                    normalizer["escape_whitespaces"] = bool(int.from_bytes(sub_value, "little"))
    for key, default in (("add_dummy_prefix", True), ("remove_extra_whitespaces", True),
                         ("escape_whitespaces", True), ("treat_whitespace_as_suffix", False),
                         ("byte_fallback", False)):
        normalizer.setdefault(key, default)
    return {"pieces": pieces, "normalizer": normalizer}


def fast_tokenizer_pieces(fast):
    model_block = fast.get("model", {})
    if model_block.get("type") != "Unigram" or len(model_block.get("vocab", [])) != 128000:
        raise ValueError("unsupported tokenizer.json Unigram vocabulary")
    return [entry[0] for entry in model_block["vocab"]]


# ---------------------------------------------------------------------------
# Source / target tensor mapping
# ---------------------------------------------------------------------------

def encoder_source_contracts():
    """Encoder + classifier safetensors → GGUF tensor name. Same shape as
    Decide's converter, just with base-v1 dims."""
    d, f = ENCODER["hidden_size"], ENCODER["intermediate_size"]
    wide = d * 2  # classifier takes (start, end) concat = 2 * encoder_dim
    mapping = {
        "token_embd.weight": "encoder.embeddings.word_embeddings.weight",
        "tok_norm.weight": "encoder.embeddings.LayerNorm.weight",
        "tok_norm.bias": "encoder.embeddings.LayerNorm.bias",
        "rel_embeddings.weight": "encoder.encoder.rel_embeddings.weight",
        "rel_norm.weight": "encoder.encoder.LayerNorm.weight",
        "rel_norm.bias": "encoder.encoder.LayerNorm.bias",
        # base-v1's classifier lives at indices 0 and 3 (with GeLU + dropout
        # in between); Decide's lives at indices 0 and 2 (ReLU in between).
        # Both write to the same `classifier.0.*` / `classifier.<last>.*`
        # tensor names so a single Rust classifier can pick either up.
        "classifier.0.weight": "classifier.0.weight",
        "classifier.0.bias": "classifier.0.bias",
        "classifier.3.weight": "classifier.3.weight",
        "classifier.3.bias": "classifier.3.bias",
    }
    layer = "attention.self.{q}"
    ff = "attention.output.dense"
    for i in range(ENCODER["num_hidden_layers"]):
        src = f"encoder.encoder.layer.{i}."
        dst = f"blk.{i}."
        for role, name in (("q", "query_proj"), ("k", "key_proj"), ("v", "value_proj")):
            mapping[dst + f"attn_{role}.weight"] = (src + layer.format(q=name) + ".weight")
            mapping[dst + f"attn_{role}.bias"] = (src + layer.format(q=name) + ".bias")
        mapping[dst + "attn_output.weight"] = src + ff + ".weight"
        mapping[dst + "attn_output.bias"] = src + ff + ".bias"
        mapping[dst + "attn_out_norm.weight"] = src + "attention.output.LayerNorm.weight"
        mapping[dst + "attn_out_norm.bias"] = src + "attention.output.LayerNorm.bias"
        mapping[dst + "ffn_up.weight"] = src + "intermediate.dense.weight"
        mapping[dst + "ffn_up.bias"] = src + "intermediate.dense.bias"
        mapping[dst + "ffn_down.weight"] = src + "output.dense.weight"
        mapping[dst + "ffn_down.bias"] = src + "output.dense.bias"
        mapping[dst + "output_norm.weight"] = src + "output.LayerNorm.weight"
        mapping[dst + "output_norm.bias"] = src + "output.LayerNorm.bias"
    return mapping


# ---------------------------------------------------------------------------
# ``boundary_head`` settings -> GGUF metadata
# ---------------------------------------------------------------------------

# Scalars the Rust BoundaryExtractor loader must read instead of guessing.
# Everything here comes straight from the checkpoint's ``boundary_head``
# block; the Rust side (``src/models/gliner_boundary/``) keys off the same
# names. Booleans select which optional feature sources exist (span content
# pooler, inside evidence, endpoint difference, query-conditioned inside
# weight), so a loader that invents defaults instead of reading them will
# silently drop contributions — the exact failure mode this block prevents.
BOUNDARY_FLAG_KEYS = (
    "use_inside_evidence",
    "enable_span_content",
    "content_soft_max_pool",
    "query_conditioned_inside_weight",
    "endpoint_difference_features",
    "enable_rotary_endpoints",
    "reranker_endpoint_compat",
    "bidirectional_proposals",
    "adaptive_threshold",
    "hard_negative_keep_all_when_absent",
    "directional_relation_states",
    "relation_biaffine_content",
    "enable_relations",
    "enable_records",
    "enable_count_head",
    "enable_abstention",
)
BOUNDARY_INT_KEYS = (
    "boundary_dim",
    "pair_dim",
    "content_dim",
    "record_dim",
    "multihead_pair_compat_heads",
    "candidate_budget",
    "pool_size",
    "start_top_k",
    "end_top_k",
    "ends_per_start",
    "starts_per_end",
    "end_block_size",
    "boundary_top_k_max",
    "boundary_top_k_bucket",
    "pool_boundary_top_k",
    "min_pool_per_query",
    "record_instance_queries",
    "relation_heads_per_type",
    "relation_tails_per_type",
    "relation_pair_cap",
    "boundary_attention_layers",
    "boundary_attention_heads",
    "boundary_attention_window",
    "boundary_refinement_layers",
    "candidate_attention_heads",
    "candidate_attention_layers",
    "query_attention_layers",
)
BOUNDARY_FLOAT_KEYS = (
    "rotary_base",
    "dropout",
    "boundary_ffn_multiplier",
    "boundary_top_k_alpha",
    "classification_temperature",
    "pair_temperature",
    "record_temperature",
    "relation_temperature",
    "abstention_threshold",
    "relation_argument_proposal_threshold",
    "record_anchor_threshold",
    "record_anchor_proposal_threshold",
    "record_field_threshold",
)
BOUNDARY_STR_KEYS = ("candidate_pool", "boundary_marginal_loss", "loss_reduction", "overlap_policy", "export_mode")
BOUNDARY_INT_KEYS_REQUIRED = (
    "boundary_dim",
    "pair_dim",
    "content_dim",
    "multihead_pair_compat_heads",
    "pool_boundary_top_k",
    "pool_size",
    "min_pool_per_query",
)
BOUNDARY_FLAG_KEYS_REQUIRED = (
    "use_inside_evidence",
    # The relation scorer's *architecture* depends on this one: with it off the
    # three content projections are absent, so a loader that invents the flag
    # would look for weights that are not there (or skip weights that are).
    "relation_biaffine_content",
    "enable_span_content",
    "content_soft_max_pool",
    "query_conditioned_inside_weight",
    "endpoint_difference_features",
    "enable_rotary_endpoints",
    "reranker_endpoint_compat",
)


def boundary_settings_metadata(settings: dict) -> dict:
    """Transcribe the ``boundary_head`` block into flat GGUF metadata values.

    Raises on a missing or wrongly typed key rather than defaulting: a silent
    default here is how the Rust loader ends up running a limited scorer on a
    checkpoint whose published config enables every feature.
    """
    if not isinstance(settings, dict):
        raise ValueError(f"config.json boundary_head must be an object, got {type(settings)!r}")
    out: dict = {}
    for key in BOUNDARY_FLAG_KEYS:
        if key not in settings:
            continue
        value = settings[key]
        if not isinstance(value, bool):
            raise ValueError(f"boundary_head.{key} must be a bool, got {value!r}")
        out[key] = value
    for key in BOUNDARY_INT_KEYS:
        if key not in settings:
            continue
        value = settings[key]
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"boundary_head.{key} must be an int, got {value!r}")
        out[key] = value
    for key in BOUNDARY_FLOAT_KEYS:
        if key not in settings:
            continue
        value = settings[key]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"boundary_head.{key} must be a number, got {value!r}")
        out[key] = float(value)
    for key in BOUNDARY_STR_KEYS:
        if key not in settings:
            continue
        value = settings[key]
        if not isinstance(value, str):
            raise ValueError(f"boundary_head.{key} must be a string, got {value!r}")
        out[key] = value
    missing = [k for k in BOUNDARY_INT_KEYS_REQUIRED + BOUNDARY_FLAG_KEYS_REQUIRED
               if k not in out]
    if missing:
        raise ValueError(f"config.json boundary_head is missing required settings: {missing}")
    if out["multihead_pair_compat_heads"] <= 0 or out["pair_dim"] % out["multihead_pair_compat_heads"]:
        raise ValueError(
            f"pair_dim {out['pair_dim']} is not divisible by "
            f"multihead_pair_compat_heads {out['multihead_pair_compat_heads']}"
        )
    return out


def check_pair_scorer_shapes(settings: dict, shapes: dict, hidden_size: int) -> None:
    """Cross-check the pair-scorer tensor shapes against the declared settings.

    ``SparseBoundaryPairScorer`` only instantiates the optional sub-modules
    when the matching flag is set, so a checkpoint whose config disagrees with
    its own state dict would otherwise fail at inference time (or, worse, drop
    a feature). Doing it here keeps the mismatch at conversion time.
    """
    d = settings["boundary_dim"]
    pair = settings["pair_dim"]
    content = settings["content_dim"]
    content_out = content * (2 if settings["content_soft_max_pool"] else 1)
    heads = settings["multihead_pair_compat_heads"]
    gate_out = pair // 2 if settings["enable_rotary_endpoints"] else pair

    def require(name: str, shape: tuple) -> None:
        actual = shapes.get(name)
        if actual is None:
            raise ValueError(f"settings require {name} but the checkpoint does not carry it")
        if actual != shape:
            raise ValueError(f"{name} has shape {actual}, settings imply {shape}")

    def forbid(name: str, why: str) -> None:
        if name in shapes:
            raise ValueError(f"{why} but the checkpoint carries {name}")

    require("boundary_head.pair_scorer.start_endpoint_projection.weight", (pair, d))
    require("boundary_head.pair_scorer.end_endpoint_projection.weight", (pair, d))
    require("boundary_head.pair_scorer.query_gate.weight", (gate_out, hidden_size))
    require("boundary_head.pair_scorer.compat_mix.weight", (1, heads))
    require("boundary_head.pair_scorer.length_query_projection.weight", (3, hidden_size))
    if settings["query_conditioned_inside_weight"]:
        require("boundary_head.pair_scorer.inside_weight.weight", (1, hidden_size))
    else:
        forbid("boundary_head.pair_scorer.inside_weight.weight",
               "query_conditioned_inside_weight is false")
    if settings["endpoint_difference_features"]:
        require("boundary_head.pair_scorer.endpoint_difference_projection.weight", (1, 2 * pair))
    else:
        forbid("boundary_head.pair_scorer.endpoint_difference_projection.weight",
               "endpoint_difference_features is false")
    if settings["enable_span_content"]:
        require("boundary_head.pair_scorer.content_pooler.value_projection.weight",
                (content, hidden_size))
        require("boundary_head.pair_scorer.content_pooler.layer_norm.weight", (content_out,))
        require("boundary_head.pair_scorer.content_query_projection.weight",
                (content_out, hidden_size))
        require("boundary_head.pair_scorer.content_bias.weight", (1, content_out))
    else:
        for name in (
            "boundary_head.pair_scorer.content_pooler.value_projection.weight",
            "boundary_head.pair_scorer.content_query_projection.weight",
            "boundary_head.pair_scorer.content_bias.weight",
        ):
            forbid(name, "enable_span_content is false")



def check_relation_scorer_shapes(settings: dict, shapes: dict, hidden_size: int) -> None:
    """Cross-check ``relation_scorer``'s shapes against the declared settings.

    Kept apart from :func:`check_pair_scorer_shapes` on purpose. That function's
    local ``d`` is ``boundary_dim`` (128 for base-v1), but the relation scorer
    is built on the *encoder* hidden size (768) — it consumes text states and
    relation query states, not boundary states. Sharing one scope made the two
    widths a one-character mistake apart.
    """
    h = hidden_size
    # ``directional_relation_states`` is the one setting that changes a *shape*
    # rather than a computation: the relation query state is the concatenation
    # of the two role states instead of their mean, so the gate projection and
    # the content linear widen from ``h`` to ``2h``. A checkpoint that disagrees
    # with its own flag cannot be loaded unambiguously.
    relation_dim = 2 * h if settings["directional_relation_states"] else h

    def require(name: str, shape: tuple) -> None:
        actual = shapes.get(name)
        if actual is None:
            raise ValueError(f"settings require {name} but the checkpoint does not carry it")
        if actual != shape:
            raise ValueError(f"{name} has shape {actual}, settings imply {shape}")

    def forbid(name: str, why: str) -> None:
        if name in shapes:
            raise ValueError(f"{why} but the checkpoint carries {name}")

    # Four endpoint states + the relation query + order + normalized distance.
    require("relation_scorer.mlp.0.weight", (h, 4 * h + relation_dim + 2))
    require("relation_scorer.mlp.0.bias", (h,))
    # ``nn.Sequential(Linear, GELU, Dropout, Linear)``: Dropout occupies an
    # index, so the output linear is ``mlp.3``, not ``mlp.2``.
    require("relation_scorer.mlp.3.weight", (1, h))
    require("relation_scorer.mlp.3.bias", (1,))
    if settings["relation_biaffine_content"]:
        require("relation_scorer.head_content_projection.weight", (h, h))
        require("relation_scorer.tail_content_projection.weight", (h, h))
        require("relation_scorer.relation_content_gate.weight", (h, relation_dim))
        require("relation_scorer.content_linear.weight", (1, 2 * h + relation_dim))
    else:
        for name in (
            "relation_scorer.head_content_projection.weight",
            "relation_scorer.tail_content_projection.weight",
            "relation_scorer.relation_content_gate.weight",
            "relation_scorer.content_linear.weight",
        ):
            forbid(name, "relation_biaffine_content is false")


def tensor_contracts():
    d, f = ENCODER["hidden_size"], ENCODER["intermediate_size"]
    wide = d * 2
    shapes = {
        "token_embd.weight": (128011, d),  # 128000 SPM + 11 schema specials (same as Decide)
        "tok_norm.weight": (d,), "tok_norm.bias": (d,),
        "rel_embeddings.weight": (ENCODER["position_buckets"] * 2, d),
        "rel_norm.weight": (d,), "rel_norm.bias": (d,),
        "classifier.0.weight": (wide, d), "classifier.0.bias": (wide,),
        "classifier.3.weight": (1, wide), "classifier.3.bias": (1,),
    }
    for i in range(ENCODER["num_hidden_layers"]):
        p = f"blk.{i}."
        shapes.update({
            p + "attn_q.weight": (d, d), p + "attn_q.bias": (d,),
            p + "attn_k.weight": (d, d), p + "attn_k.bias": (d,),
            p + "attn_v.weight": (d, d), p + "attn_v.bias": (d,),
            p + "attn_output.weight": (d, d), p + "attn_output.bias": (d,),
            p + "attn_out_norm.weight": (d,), p + "attn_out_norm.bias": (d,),
            p + "ffn_up.weight": (f, d), p + "ffn_up.bias": (f,),
            p + "ffn_down.weight": (d, f), p + "ffn_down.bias": (d,),
            p + "output_norm.weight": (d,), p + "output_norm.bias": (d,),
        })
    return shapes


# ---------------------------------------------------------------------------

def convert(model_dir: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    config = json.loads((model_dir / "config.json").read_text())
    if config.get("architecture") != "boundary":
        raise ValueError(f"this converter handles architecture='boundary', got {config.get('architecture')!r}")
    if config.get("model_name") not in ("microsoft/deberta-v3-base", "microsoft/mdeberta-v3-base"):
        raise ValueError(f"this converter hard-codes DeBERTa-v3-base dims; got model_name={config.get('model_name')!r}")
    tokenizer_config = json.loads((model_dir / "tokenizer_config.json").read_text())
    if tokenizer_config.get("tokenizer_class") != "DebertaV2Tokenizer":
        raise ValueError(f"Expected DebertaV2Tokenizer, got {tokenizer_config.get('tokenizer_class')!r}")
    if tokenizer_config.get("vocab_type") != "spm" or tokenizer_config.get("do_lower_case"):
        raise ValueError("Expected a case-sensitive SentencePiece tokenizer")
    tokenizer_json_path = model_dir / "tokenizer.json"
    fast_json = (
        tokenizer_json_path.read_text()
        if tokenizer_json_path.exists() and not (model_dir / "spm.model").exists()
        else None
    )
    fast = json.loads(fast_json) if fast_json is not None else None
    if fast is not None:
        pieces = fast_tokenizer_pieces(fast)
        spm = None
    else:
        spm = parse_spm((model_dir / "spm.model").read_bytes())
        if spm["normalizer"].get("name") != "nmt_nfkc":
            raise ValueError(f"Unsupported normalizer {spm['normalizer'].get('name')!r}")
        pieces = spm["pieces"]
    # Different checkpoints write the schema specials under different
    # keys: Decide / pre-2.5 use `added_tokens_decoder` (a dict with explicit
    # ids); 2.5 boundary-family uses `extra_special_tokens` (a list of names
    # whose ids are the contiguous block right after the SPM vocab, same
    # order as our SPECIAL_TOKENS list).
    declared = {}
    if "added_tokens_decoder" in tokenizer_config:
        for entry in tokenizer_config["added_tokens_decoder"].values():
            declared[entry["content"]] = entry["id"]
        added = {token: declared[token] for token in SPECIAL_TOKENS}
    else:
        declared_list = tokenizer_config.get("extra_special_tokens", [])
        if declared_list != list(SPECIAL_TOKENS):
            raise ValueError(
                f"extra_special_tokens order differs from expected: {declared_list}"
            )
        # SPM vocab (DeBERTa-v3-base) is 128000 tokens with ``[MASK]`` at id
        # 128000 already inside it, so the schema extras start at 128001.
        # (Boundary configs omit ``[MASK]`` from ``extra_special_tokens``.)
        base = len(pieces) + 1
        added = {token: base + index for index, token in enumerate(SPECIAL_TOKENS)}
    vocab_size = max(added.values()) + 1

    contracts = tensor_contracts()
    sources = encoder_source_contracts()

    source = open_safetensors(model_dir / "model.safetensors")
    present = {name for name in source.header if name != "__metadata__"}
    needed = set(sources.values())
    if missing := needed - present:
        raise ValueError(f"checkpoint is missing {sorted(missing)[:4]} ({len(missing)} tensors)")
    # Bundled heads: every tensor outside `needed` that begins with one of
    # the bundled prefixes is carried through to the GGUF unchanged.
    bundled = sorted(
        name for name in present
        if name not in needed and any(name.startswith(p) for p in BUNDLED_HEAD_PREFIXES)
    )
    # Anything else outside `needed` is unexpected — refuse to drop silently.
    unexpected = [
        name for name in (present - needed)
        if not any(name.startswith(p) for p in BUNDLED_HEAD_PREFIXES)
    ]
    if unexpected:
        raise ValueError(f"refusing to drop unexpected tensors: {unexpected[:5]}")

    for name, key in sources.items():
        info = source.header[key]
        shape = tuple(info["shape"])
        if shape != contracts[name]:
            raise ValueError(f"Invalid shape for {key}: {shape} != {contracts[name]}")

    # The pair scorer's optional sub-modules exist only when their flag is set
    # (``SparseBoundaryPairScorer.__init__``), so verify the checkpoint's own
    # shapes agree with the settings we are about to write into the GGUF.
    boundary_settings = boundary_settings_metadata(config["boundary_head"])
    bundled_shapes = {name: tuple(source.header[name]["shape"]) for name in bundled}
    check_pair_scorer_shapes(boundary_settings, bundled_shapes, ENCODER["hidden_size"])
    check_relation_scorer_shapes(boundary_settings, bundled_shapes, ENCODER["hidden_size"])

    tokens = list(pieces) + [token for token, _ in sorted(added.items(), key=lambda kv: kv[1])]
    # ``vocab_size`` and ``len(tokens)`` can differ by 1 when ``[MASK]`` is part
    # of the SPM pieces (boundary-family checkpoints ship MASK in the SPM
    # vocab at id 128000, but do not list it under ``extra_special_tokens``).
    # The actual embedding row count is the source of truth; we don't pin it
    # against ``tokens`` because the SPM vocab string array excludes the
    # MASK row.

    writer = GgufWriter(output)
    writer.add_meta("general.architecture", ARCH)
    writer.add_meta("general.name", "gliner2.5-boundary")
    writer.add_meta("general.file_type", "F32")
    writer.add_meta(f"{ARCH}.block_count", ENCODER["num_hidden_layers"])
    writer.add_meta(f"{ARCH}.embedding_length", ENCODER["hidden_size"])
    writer.add_meta(f"{ARCH}.feed_forward_length", ENCODER["intermediate_size"])
    writer.add_meta(f"{ARCH}.attention.head_count", ENCODER["num_attention_heads"])
    writer.add_meta(f"{ARCH}.attention.head_dim", ENCODER["hidden_size"] // ENCODER["num_attention_heads"])
    writer.add_meta(f"{ARCH}.attention.scale_divisor", SCALE_DIVISOR)
    writer.add_meta(f"{ARCH}.attention.layer_norm_epsilon", ENCODER["layer_norm_eps"])
    writer.add_meta(f"{ARCH}.relative_attention.bucket_size", ENCODER["position_buckets"])
    writer.add_meta(f"{ARCH}.relative_attention.max_relative_positions", ENCODER["max_position_embeddings"])
    writer.add_meta(f"{ARCH}.norm_rel_embeddings", True)
    writer.add_meta(f"{ARCH}.hidden_act", ENCODER["hidden_act"])
    writer.add_meta(f"{ARCH}.vocab_size", vocab_size)
    writer.add_meta(f"{ARCH}.classifier.intermediate_size", ENCODER["hidden_size"] * 2)
    # The reference hardcodes `activation="relu"` in
    # `BoundaryExtractorModel.__init__`'s `create_mlp` call
    # (gliner2/models/boundary/model.py:1163). Decide's classifier also
    # uses relu (layer index 2 there); the boundary variant's
    # LayerNorm-free MLP puts the final linear at index 3.
    writer.add_meta(f"{ARCH}.classifier.activation", "relu")
    writer.add_meta(f"{ARCH}.classifier.last_layer_index", 3)  # boundary classifier lives at index 3
    writer.add_meta(f"{ARCH}.variant", "boundary")
    writer.add_meta(f"{ARCH}.boundary.bundled_heads", json.dumps(BUNDLED_HEAD_PREFIXES))
    writer.add_meta(f"{ARCH}.boundary.bundled_tensor_count", len(bundled))
    for key, value in boundary_settings.items():
        writer.add_meta(f"{ARCH}.boundary.{key}", value)
    writer.add_meta(f"{ARCH}.source_architecture", json.dumps({
        k: config[k] for k in ("architecture", "model_name", "config_version", "token_pooling")
        if k in config
    }))
    writer.add_meta(f"{ARCH}.source_config", json.dumps(ENCODER))
    writer.add_meta("tokenizer.ggml.model", "hf-json" if fast_json is not None else "spm")
    writer.add_meta("tokenizer.ggml.vocab_size", vocab_size)
    if fast_json is not None:
        writer.add_meta(f"{ARCH}.tokenizer_json", fast_json)
        writer.add_meta(f"{ARCH}.tokenizer_sha256", hashlib.sha256(fast_json.encode()).hexdigest())
    else:
        scores = list(spm["scores"]) + [0.0] * len(added)
        types = list(spm["types"]) + [4] * len(added)
        writer.add_meta("tokenizer.ggml.tokens", tokens)
        writer.add_meta("tokenizer.ggml.scores", scores)
        writer.add_meta("tokenizer.ggml.token_type", [3 if t in (1, 2, 3) else 4 if t == 6 else 1 for t in types])
    writer.add_meta(f"{ARCH}.special_tokens", [token for token, _ in sorted(added.items(), key=lambda kv: kv[1])])
    if spm is not None:
        writer.add_meta(f"{ARCH}.spm.piece_count", len(spm["pieces"]))
        writer.add_meta(f"{ARCH}.spm.piece_types", types)
        writer.add_meta(f"{ARCH}.spm.charsmap_b64", base64.b64encode(spm["normalizer"]["charsmap"]).decode())
        for flag in ("byte_fallback", "add_dummy_prefix", "remove_extra_whitespaces",
                     "escape_whitespaces", "treat_whitespace_as_suffix"):
            writer.add_meta(f"{ARCH}.spm.{flag}", bool(spm["normalizer"][flag]))
        writer.add_meta(f"{ARCH}.spm.normalizer", spm["normalizer"]["name"])
    with source.path.open("rb") as stream:
        writer.add_meta(f"{ARCH}.source_sha256", hashlib.file_digest(stream, "sha256").hexdigest())

    def chunks(key, source_path, data_offset, info):
        start, end = info["data_offsets"]
        dtype = "<f2" if info["dtype"] == "F16" else "<f4"
        with source_path.open("rb") as stream:
            stream.seek(data_offset + start)
            remaining = end - start
            while remaining:
                raw = stream.read(min(remaining, 8 * 1024 * 1024))
                if not raw:
                    raise ValueError(f"Truncated tensor {key}")
                values = np.frombuffer(raw, dtype=dtype).astype("<f4")
                if not np.isfinite(values).all():
                    raise ValueError(f"Non-finite tensor {key}")
                yield values.tobytes()
                remaining -= len(raw)

    for name, shape in contracts.items():
        key = sources[name]
        info = source.header[key]
        writer.add_tensor_chunks(name, GGML_F32, gguf_dims(shape), math.prod(shape) * 4,
                                 lambda key=key: chunks(key, source.path, source.data_offset, info))

    # Bundled heads: emit with the original safetensors key as the GGUF name
    # so a future Rust BoundaryExtractor forward can pick them up directly.
    for key in bundled:
        info = source.header[key]
        shape = tuple(info["shape"])
        # Use the safetensors key as the GGUF tensor name (no remap).
        writer.add_tensor_chunks(key, GGML_F32, gguf_dims(shape), math.prod(shape) * 4,
                                 lambda key=key: chunks(key, source.path, source.data_offset, info))

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    writer.path = temporary
    try:
        writer.write()
        temporary.rename(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    print(f"{output} (encoder={len(sources)} tensors, bundled heads={len(bundled)} tensors)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    convert(args.model_dir, args.output)