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
import json
from pathlib import Path

from tools.converter.gliner.common import (
    ARCH, TOKENIZER_CLASSES, add_encoder_meta, add_source_sha256,
    add_tokenizer_meta, emit_tensor, encoder_tensor_contracts,
    fast_tokenizer_pieces, parse_spm, resolve_encoder, tensor_chunks, write_atomic,
)
from tools.converter.utils.gguf import GgufWriter, open_safetensors

# DeBERTa-v3-base dimensions for ``microsoft/deberta-v3-base`` (base-v1).
# Unlike ``convert_gliner.py`` (which hardcodes DeBERTa-v3-large), this
# converter uses base dims because the boundary-family checkpoints ship
# only with DeBERTa-v3-base + mDeBERTa-v3-base.
# Encoder size per ``config.json``'s ``model_name``. The boundary family ships
# with three encoders: DeBERTa-v3-base (base-v1), mDeBERTa-v3-base
# (multi-v1 / multi-Decide) and DeBERTa-v3-xsmall (small-v1). mDeBERTa-v3's
# published config is field-for-field deberta-v3-base apart from `vocab_size`,
# so its numbers repeat; xsmall is a genuinely different width.
#
# Only the size fields vary. The values are read off the published
# ``microsoft/*`` configs, and `encoder_tensor_contracts` re-derives hidden_size,
# num_hidden_layers and intermediate_size from the checkpoint's own shapes, so a
# wrong entry fails the conversion. `num_attention_heads` appears in no tensor
# shape, which makes it the one field only the oracle can catch.
ENCODER_SIZES = {
    "microsoft/deberta-v3-base": {"hidden_size": 768, "num_hidden_layers": 12,
                                  "num_attention_heads": 12, "intermediate_size": 3072},
    "microsoft/mdeberta-v3-base": {"hidden_size": 768, "num_hidden_layers": 12,
                                   "num_attention_heads": 12, "intermediate_size": 3072},
    "microsoft/deberta-v3-xsmall": {"hidden_size": 384, "num_hidden_layers": 12,
                                    "num_attention_heads": 6, "intermediate_size": 1536},
}


def resolve_boundary_encoder(model_name: str) -> dict:
    return resolve_encoder(model_name, ENCODER_SIZES, "boundary")


CLASSIFIER_LAST_LAYER = 3  # GeLU + dropout occupy 1 and 2; the output linear at 3.

# Bundle tensors that aren't part of the encoder / classifier and
# should be carried through verbatim for the future BoundaryExtractor
# Rust forward. Keyed by the safetensors prefix they share.
BUNDLED_HEAD_PREFIXES = (
    "boundary_head.",
    "relation_scorer.",
    "record_decoder.",
)


SPECIAL_TOKENS = (
    # [MASK] is part of the SPM vocab at id 128000 for DeBERTa-v3;
    # boundary-family configs (gliner2.5-base-v1) do not list it under
    # extra_special_tokens because it ships with the base tokenizer.
    "[SEP_STRUCT]", "[SEP_TEXT]", "[P]", "[C]", "[E]", "[R]",
    "[L]", "[EXAMPLE]", "[OUTPUT]", "[DESCRIPTION]",
)

# ---------------------------------------------------------------------------
# Source / target tensor mapping
# ---------------------------------------------------------------------------

def encoder_source_contracts(encoder: dict):
    """Encoder + classifier safetensors → GGUF tensor name."""
    d, f = encoder["hidden_size"], encoder["intermediate_size"]
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
    for i in range(encoder["num_hidden_layers"]):
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


def tensor_contracts(encoder: dict, vocab_size: int):
    return encoder_tensor_contracts(encoder, vocab_size, CLASSIFIER_LAST_LAYER)


# ---------------------------------------------------------------------------

def convert(model_dir: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    config = json.loads((model_dir / "config.json").read_text())
    if config.get("architecture") != "boundary":
        raise ValueError(f"this converter handles architecture='boundary', got {config.get('architecture')!r}")
    encoder = resolve_boundary_encoder(config.get("model_name"))
    tokenizer_config = json.loads((model_dir / "tokenizer_config.json").read_text())
    # fastino ships both the slow and the fast DeBERTa-v2 class over one SPM
    # vocab; the properties that decide the pieces and the normalizer are
    # checked on the next line, so name exactly these two rather than widening.
    if tokenizer_config.get("tokenizer_class") not in TOKENIZER_CLASSES:
        raise ValueError(f"expected one of {sorted(TOKENIZER_CLASSES)}, "
                         f"got {tokenizer_config.get('tokenizer_class')!r}")
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
        pieces = fast_tokenizer_pieces(fast, SPECIAL_TOKENS, 1)
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

    contracts = tensor_contracts(encoder, vocab_size)
    sources = encoder_source_contracts(encoder)

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
    check_pair_scorer_shapes(boundary_settings, bundled_shapes, encoder["hidden_size"])
    check_relation_scorer_shapes(boundary_settings, bundled_shapes, encoder["hidden_size"])

    tokens = list(pieces) + [token for token, _ in sorted(added.items(), key=lambda kv: kv[1])]
    # ``vocab_size`` and ``len(tokens)`` can differ by 1 when ``[MASK]`` is part
    # of the SPM pieces (boundary-family checkpoints ship MASK in the SPM
    # vocab at id 128000, but do not list it under ``extra_special_tokens``).
    # The actual embedding row count is the source of truth; we don't pin it
    # against ``tokens`` because the SPM vocab string array excludes the
    # MASK row.

    writer = GgufWriter(output)
    add_encoder_meta(writer, model_dir.name, encoder, vocab_size)
    # The reference hardcodes `activation="relu"` in
    # `BoundaryExtractorModel.__init__`'s `create_mlp` call
    # (gliner2/models/boundary/model.py:1163). Decide's classifier also
    # uses relu (layer index 2 there); the boundary variant's
    # LayerNorm-free MLP puts the final linear at index 3.
    writer.add_meta(f"{ARCH}.classifier.activation", "relu")
    writer.add_meta(f"{ARCH}.classifier.last_layer_index", CLASSIFIER_LAST_LAYER)
    writer.add_meta(f"{ARCH}.variant", "boundary")
    writer.add_meta(f"{ARCH}.boundary.bundled_heads", json.dumps(BUNDLED_HEAD_PREFIXES))
    writer.add_meta(f"{ARCH}.boundary.bundled_tensor_count", len(bundled))
    for key, value in boundary_settings.items():
        writer.add_meta(f"{ARCH}.boundary.{key}", value)
    writer.add_meta(f"{ARCH}.source_architecture", json.dumps({
        k: config[k] for k in ("architecture", "model_name", "config_version", "token_pooling")
        if k in config
    }))
    writer.add_meta(f"{ARCH}.source_config", json.dumps(encoder))
    add_tokenizer_meta(writer, tokens, vocab_size, spm, fast_json, added)
    add_source_sha256(writer, source.path)

    for name, shape in contracts.items():
        key = sources[name]
        info = source.header[key]
        emit_tensor(writer, name, shape,
                    lambda key=key, info=info: tensor_chunks(
                        source.path, source.data_offset, info, key))

    # Bundled heads: emit with the original safetensors key as the GGUF name
    # so a future Rust BoundaryExtractor forward can pick them up directly.
    for key in bundled:
        info = source.header[key]
        shape = tuple(info["shape"])
        emit_tensor(writer, key, shape,
                    lambda key=key, info=info: tensor_chunks(
                        source.path, source.data_offset, info, key))

    write_atomic(writer, output)
    print(f"{output} (encoder={len(sources)} tensors, bundled heads={len(bundled)} tensors)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    convert(args.model_dir, args.output)