"""Convert the GLiNER2.5-Decide safetensors checkpoint to F32 GGUF.

The model is a DeBERTa-v3-large encoder with a 2-layer ReLU MLP on top of the
hidden state at each ``[L]`` marker. Only that path is converted: the NER
heads (``span_rep``, ``count_embed``, ``count_pred``) never run in
``classify_text`` and are dropped.

The GGUF is self-contained: it embeds the checkpoint's ``tokenizer.json`` when
present, or the SentencePiece vocabulary and normalizer from ``spm.model``.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import struct
from pathlib import Path

import numpy as np

from tools.converter.utils.gguf import GgufWriter, GGML_F32, gguf_dims, open_safetensors

ARCH = "gliner2"

# Added on top of the 128000 SentencePiece pieces; ids 128000..128010 in
# `tokenizer_config.json`. [MASK] leads because GLiNER2 appends its ten schema
# markers after whatever the base tokenizer already declared.
EXPECTED_CONFIG = {"model_name": "microsoft/deberta-v3-large"}
# Pre-2.5 configs (e.g. fastino/gliner2-large-v1) omit `architecture`,
# `config_version`, and `token_pooling`; the field is present only from
# gliner2.5 onwards. Same encoder + head contract, just an older config schema.
EXPECTED_CONFIG_OPTIONAL = {"architecture": "span", "config_version": 3,
                            "token_pooling": "first"}

# microsoft/deberta-v3-large, as resolved by `AutoConfig.from_pretrained` inside
# `SpanExtractorModel._load_encoder`. `position_buckets * 2` is the row count of
# `rel_embeddings`, and `att_span` (the clamp range inside
# `disentangled_attention_bias`) is the bucket count itself.
ENCODER = {"hidden_size": 1024, "num_hidden_layers": 24, "num_attention_heads": 16,
           "intermediate_size": 4096, "hidden_act": "gelu", "layer_norm_eps": 1e-7,
           "relative_attention": True, "position_buckets": 256,
           "max_position_embeddings": 512, "max_relative_positions": -1,
           "position_biased_input": False, "type_vocab_size": 0,
           "norm_rel_ebd": "layer_norm", "pos_att_type": "p2c|c2p", "share_att_key": True}

# `DisentangledSelfAttention.forward` sets `scale_factor = 1 + |pos_att_type|`
# and divides the content-content *and* both position terms by
# `sqrt(head_dim * scale_factor)`, so this is 3, not 1.
SCALE_DIVISOR = 3

HEADERS = ("piece", "score", "type")
SPECIAL_HEADERS = ("task", "labels", "multi_label", "cls_threshold", "class_act")
SPECIAL_TOKENS = ("[MASK]", "[SEP_STRUCT]", "[SEP_TEXT]", "[P]", "[C]", "[E]", "[R]",
                  "[L]", "[EXAMPLE]", "[OUTPUT]", "[DESCRIPTION]")


# ---------------------------------------------------------------------------
# Minimal protobuf reader (ModelProto), stdlib only
# ---------------------------------------------------------------------------

def _varint(buf: bytes, pos: int) -> tuple[int, int]:
    result = 0
    shift = 0
    while True:
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, pos
        shift += 7


def _fields(buf: bytes):
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
            yield field, buf[pos:pos + 4]
            pos += 4
            continue
        else:
            raise ValueError(f"unsupported protobuf wire type {wire}")
        yield field, value


def _first(buf: bytes, want: int):
    for field, value in _fields(buf):
        if field == want:
            return value
    return None


def parse_spm(blob: bytes) -> dict:
    """Pull the unigram model and its NormalizerSpec out of a `spm.model`."""
    pieces: list[str] = []
    scores: list[float] = []
    types: list[int] = []
    normalizer: dict = {}
    for field, value in _fields(blob):
        if field == 1:
            piece, score, kind = "", 0.0, 1
            for sub, sub_value in _fields(value):
                if sub == 1:
                    piece = sub_value.decode()
                elif sub == 2:
                    score = struct.unpack("<f", sub_value)[0]
                elif sub == 3:
                    kind = sub_value
            pieces.append(piece)
            scores.append(score)
            types.append(kind)
        elif field == 2:  # TrainerSpec
            byte_fallback = _first(value, 35)
            treat_as_suffix = _first(value, 24)
            normalizer["byte_fallback"] = bool(byte_fallback)
            normalizer["treat_whitespace_as_suffix"] = bool(treat_as_suffix)
        elif field == 3:  # NormalizerSpec
            for sub, sub_value in _fields(value):
                if sub == 1:
                    normalizer["name"] = sub_value.decode()
                elif sub == 2:
                    normalizer["charsmap"] = sub_value
                elif sub == 3:
                    normalizer["add_dummy_prefix"] = bool(sub_value)
                elif sub == 4:
                    normalizer["remove_extra_whitespaces"] = bool(sub_value)
                elif sub == 5:
                    normalizer["escape_whitespaces"] = bool(sub_value)
    if not pieces:
        raise ValueError("spm.model has no pieces")
    for key, default in (("add_dummy_prefix", True), ("remove_extra_whitespaces", True),
                         ("escape_whitespaces", True), ("treat_whitespace_as_suffix", False),
                         ("byte_fallback", False)):
        normalizer.setdefault(key, default)
    return {"pieces": pieces, "scores": scores, "types": types, "normalizer": normalizer}


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------

def validate_config(config: dict) -> None:
    for key, value in EXPECTED_CONFIG.items():
        if config.get(key) != value:
            raise ValueError(f"Unsupported config {key}: {config.get(key)!r}; expected {value!r}")
    for key, value in EXPECTED_CONFIG_OPTIONAL.items():
        if config.get(key) not in (value, None):
            raise ValueError(f"Unsupported config {key}: {config.get(key)!r}; expected {value!r} or missing")


def added_tokens(tokenizer_config: dict, spm_pieces: list[str]) -> dict[str, int]:
    """Tokens appended past the SentencePiece vocab, id-ordered.

    `added_tokens_decoder` also lists the four base specials, so those are
    dropped after being checked against the SPM pieces.
    """
    decoder = tokenizer_config.get("added_tokens_decoder") or {}
    declared = {entry["content"]: int(index) for index, entry in decoder.items()}
    for content, index in (("[PAD]", 0), ("[CLS]", 1), ("[SEP]", 2), ("[UNK]", 3)):
        if declared.pop(content, None) != index or spm_pieces[index] != content:
            raise ValueError(f"base special token {content} does not match the SPM piece at {index}")
    missing = [token for token in SPECIAL_TOKENS if token not in declared]
    if missing:
        raise ValueError(f"tokenizer_config is missing added tokens: {missing}")
    base = len(spm_pieces)
    if sorted(declared.values()) != list(range(base, base + len(declared))):
        raise ValueError(f"added tokens are not a contiguous block at {base}: {declared}")
    return declared


def fast_tokenizer_pieces(fast: dict) -> list[str]:
    model = fast.get("model", {})
    if model.get("type") != "Unigram" or len(model.get("vocab", [])) != 128000:
        raise ValueError("unsupported tokenizer.json Unigram vocabulary")
    declared = {entry["content"]: entry["id"] for entry in fast.get("added_tokens", [])}
    if any(declared.get(token) != index for index, token in enumerate(SPECIAL_TOKENS, 128000)):
        raise ValueError("tokenizer.json added token IDs differ from GLiNER2")
    return [entry[0] for entry in model["vocab"]]


def tensor_contracts(vocab_size: int) -> dict[str, tuple]:
    d, f = ENCODER["hidden_size"], ENCODER["intermediate_size"]
    # `create_mlp(input_dim=hidden, intermediate_dims=[hidden * 2], output_dim=1)`.
    wide = d * 2
    shapes = {
        "token_embd.weight": (vocab_size, d),
        "tok_norm.weight": (d,), "tok_norm.bias": (d,),
        "rel_embeddings.weight": (ENCODER["position_buckets"] * 2, d),
        "rel_norm.weight": (d,), "rel_norm.bias": (d,),
        "classifier.0.weight": (wide, d), "classifier.0.bias": (wide,),
        "classifier.2.weight": (1, wide), "classifier.2.bias": (1,),
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


def source_contracts(vocab_size: int) -> dict[str, str]:
    """Map GGUF tensor name -> safetensors key."""
    mapping = {
        "token_embd.weight": "encoder.embeddings.word_embeddings.weight",
        "tok_norm.weight": "encoder.embeddings.LayerNorm.weight",
        "tok_norm.bias": "encoder.embeddings.LayerNorm.bias",
        "rel_embeddings.weight": "encoder.encoder.rel_embeddings.weight",
        "rel_norm.weight": "encoder.encoder.LayerNorm.weight",
        "rel_norm.bias": "encoder.encoder.LayerNorm.bias",
        "classifier.0.weight": "classifier.0.weight",
        "classifier.0.bias": "classifier.0.bias",
        "classifier.2.weight": "classifier.2.weight",
        "classifier.2.bias": "classifier.2.bias",
    }
    layer = "attention.self.{q}.{k}"
    ff = "attention.output.dense"
    for i in range(ENCODER["num_hidden_layers"]):
        src = f"encoder.encoder.layer.{i}."
        dst = f"blk.{i}."
        for role, name in (("q", "query_proj"), ("k", "key_proj"), ("v", "value_proj")):
            mapping[dst + f"attn_{role}.weight"] = (src + layer.format(q=name, k="weight"))
            mapping[dst + f"attn_{role}.bias"] = (src + layer.format(q=name, k="bias"))
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

def convert(model_dir: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    config = json.loads((model_dir / "config.json").read_text())
    validate_config(config)
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
    added = added_tokens(tokenizer_config, pieces)
    vocab_size = max(added.values()) + 1
    contracts = tensor_contracts(vocab_size)
    sources = source_contracts(vocab_size)

    source = open_safetensors(model_dir / "model.safetensors")
    present = {name for name in source.header if name != "__metadata__"}
    needed = set(sources.values())
    if missing := needed - present:
        raise ValueError(f"checkpoint is missing {sorted(missing)[:4]} ({len(missing)} tensors)")
    dropped = present - needed
    if not dropped <= {name for name in dropped if name.startswith(("span_rep.", "count_embed.", "count_pred."))}:
        raise ValueError(f"refusing to drop unexpected tensors: {sorted(dropped - needed)[:4]}")
    for name, key in sources.items():
        info = source.header[key]
        shape = tuple(info["shape"])
        if shape != contracts[name]:
            raise ValueError(f"Invalid shape for {key}: {shape} != {contracts[name]}")

    tokens = list(pieces) + [token for token, _ in sorted(added.items(), key=lambda kv: kv[1])]
    if len(tokens) != vocab_size:
        raise ValueError(f"vocab mismatch: {len(tokens)} tokens vs {vocab_size} ids")

    writer = GgufWriter(output)
    writer.add_meta("general.architecture", ARCH)
    writer.add_meta("general.name", "gliner2.5-decide")
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
    writer.add_meta(f"{ARCH}.classifier.activation", "relu")
    writer.add_meta(f"{ARCH}.source_architecture", json.dumps({k: config.get(k) for k in EXPECTED_CONFIG | EXPECTED_CONFIG_OPTIONAL}))
    writer.add_meta(f"{ARCH}.source_config", json.dumps(ENCODER))
    writer.add_meta("tokenizer.ggml.model", "hf-json" if fast_json is not None else "spm")
    writer.add_meta("tokenizer.ggml.vocab_size", vocab_size)
    if fast_json is not None:
        writer.add_meta(f"{ARCH}.tokenizer_json", fast_json)
        writer.add_meta(f"{ARCH}.tokenizer_sha256", hashlib.sha256(fast_json.encode()).hexdigest())
    else:
        scores = list(spm["scores"]) + [0.0] * len(added)
        types = list(spm["types"]) + [4] * len(added)  # USER_DEFINED
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

    def chunks(key: str):
        info = source.header[key]
        start, end = info["data_offsets"]
        dtype = "<f2" if info["dtype"] == "F16" else "<f4"
        with source.path.open("rb") as stream:
            stream.seek(source.data_offset + start)
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
        writer.add_tensor_chunks(name, GGML_F32, gguf_dims(shape), math.prod(shape) * 4,
                                 lambda key=sources[name]: chunks(key))
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
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    convert(args.model_dir, args.output)
