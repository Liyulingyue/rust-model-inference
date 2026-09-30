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

Only the encoder is exposed to the Rust code right now; boundary
detection + pair scoring are not implemented in Rust (see
``glinerTODO.md``). The bundled ``boundary_head`` /
``relation_scorer`` / ``record_decoder`` weights live in the GGUF
for forward compatibility — when the Rust forward catches up, those
tensors are already in place.

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