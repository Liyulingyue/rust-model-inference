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
#
# Pre-2.5 configs (e.g. fastino/gliner2-large-v1) omit `architecture`,
# `config_version`, and `token_pooling`; the field is present only from
# gliner2.5 onwards. Same encoder + head contract, just an older config schema.
EXPECTED_CONFIG_OPTIONAL = {"architecture": "span", "config_version": 3,
                            "token_pooling": "first"}

# Both the slow and the fast DeBERTa-v2 tokenizer classes, over one SPM vocab.
TOKENIZER_CLASSES = ("DebertaV2Tokenizer", "DebertaV2TokenizerFast")

# Encoder size per `config.json`'s `model_name`. Only the four size fields vary
# across the DeBERTa-v3 checkpoints GLiNER2 ships; the relative-attention
# settings in ENCODER_COMMON are what `AutoConfig.from_pretrained` resolves
# identically for every v3 size.
#
# `position_buckets * 2` is the row count of `rel_embeddings`, and `att_span`
# (the clamp range inside `disentangled_attention_bias`) is the bucket count.
#
# Sizes are read off the published `microsoft/deberta-v3-*` configs, not
# inferred. Three of the four are re-derived from the checkpoint's tensor shapes
# by the contracts below, so a wrong entry here fails the conversion;
# `num_attention_heads` appears in no tensor shape, which makes it the one field
# only the byte-exact oracle can catch.
ENCODER_SIZES = {
    "microsoft/deberta-v3-large": {"hidden_size": 1024, "num_hidden_layers": 24,
                                   "num_attention_heads": 16, "intermediate_size": 4096},
    "microsoft/deberta-v3-base": {"hidden_size": 768, "num_hidden_layers": 12,
                                  "num_attention_heads": 12, "intermediate_size": 3072},
    # fastino/gliner2-multi-v1. mDeBERTa-v3 is DeBERTa-v2 with a 250k
    # multilingual SPM vocab; its published config is field-for-field identical
    # to deberta-v3-base apart from `vocab_size` (251000 vs 128100), so every
    # number here is the base row again. The vocab is not in this table: it is
    # derived from the tokenizer below and re-checked against the checkpoint's
    # `word_embeddings` shape.
    "microsoft/mdeberta-v3-base": {"hidden_size": 768, "num_hidden_layers": 12,
                                   "num_attention_heads": 12, "intermediate_size": 3072},
}

# As resolved by `AutoConfig.from_pretrained` inside
# `SpanExtractorModel._load_encoder`, for every size in ENCODER_SIZES.
ENCODER_COMMON = {"hidden_act": "gelu", "layer_norm_eps": 1e-7,
                  "relative_attention": True, "position_buckets": 256,
                  "max_position_embeddings": 512, "max_relative_positions": -1,
                  "position_biased_input": False, "type_vocab_size": 0,
                  "norm_rel_ebd": "layer_norm", "pos_att_type": "p2c|c2p",
                  "share_att_key": True}


def resolve_encoder(model_name: str) -> dict:
    """The encoder dims for `model_name`, or an error naming what is supported.

    An unknown `model_name` is a hard failure rather than a fallback to the
    large sizes: a silently wrong `hidden_size` would either trip the shape
    contracts or, worse, pass them and produce a GGUF that decodes to noise.
    """
    if model_name not in ENCODER_SIZES:
        raise ValueError(f"Unsupported model_name {model_name!r}; "
                         f"expected one of {sorted(ENCODER_SIZES)}")
    return {**ENCODER_COMMON, **ENCODER_SIZES[model_name]}

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

def validate_config(config: dict) -> dict:
    """Check the config against the contract and return the encoder dims.

    `model_type` is the family marker both sizes share; `model_name` selects the
    size and is resolved rather than compared, since more than one is supported.
    """
    if config.get("model_type") != "extractor":
        raise ValueError(f"Unsupported model_type: {config.get('model_type')!r}; expected 'extractor'")
    encoder = resolve_encoder(config.get("model_name"))
    for key, value in EXPECTED_CONFIG_OPTIONAL.items():
        if config.get(key) not in (value, None):
            raise ValueError(f"Unsupported config {key}: {config.get(key)!r}; expected {value!r} or missing")
    return encoder


def added_tokens(tokenizer_config: dict, spm_pieces: list[str],
                 fast: dict | None = None) -> dict[str, int]:
    """Tokens appended past the SentencePiece vocab, id-ordered.

    `added_tokens_decoder` also lists tokens that are *already* in the SPM
    vocab, and those are not a mistake: the four base specials always are, and
    mDeBERTa-v3 (`fastino/gliner2-multi-v1`) additionally declares its 100
    `<extra_id_N>` sentinel pieces there. So declarations are split by id — those
    below the SPM length must match the piece they claim, which is a stricter
    check than ignoring them, and only the rest are the appended block the GGUF
    has to carry.
    """
    base = len(spm_pieces)
    decoder = tokenizer_config.get("added_tokens_decoder") or {}
    declared = {entry["content"]: int(index) for index, entry in decoder.items()}
    # The family ships two declaration styles. Older repos carry an
    # `added_tokens_decoder` in `tokenizer_config.json`; the guardrail repos and
    # the whole boundary family carry only `extra_special_tokens` (ten names, no
    # `[MASK]`) and leave the ids to `tokenizer.json`. When both are present they
    # must agree, which is a stronger check than trusting either alone.
    from_fast = {entry["content"]: int(entry["id"])
                 for entry in (fast or {}).get("added_tokens", [])}
    if declared and from_fast:
        disagree = sorted(token for token in set(declared) & set(from_fast)
                          if declared[token] != from_fast[token])
        if disagree:
            raise ValueError(
                f"tokenizer_config and tokenizer.json disagree on {disagree[:3]}"
            )
    if not declared:
        declared = from_fast
    for content, index in (("[PAD]", 0), ("[CLS]", 1), ("[SEP]", 2), ("[UNK]", 3)):
        if declared.get(content) != index or spm_pieces[index] != content:
            raise ValueError(f"base special token {content} does not match the SPM piece at {index}")
    mismatched = [content for content, index in declared.items()
                  if index < base and spm_pieces[index] != content]
    if mismatched:
        raise ValueError(f"added token {mismatched[0]!r} does not match the SPM piece at its id")
    missing = [token for token in SPECIAL_TOKENS if token not in declared]
    if missing:
        raise ValueError(f"tokenizer_config is missing added tokens: {missing}")
    added = {content: index for content, index in declared.items() if index >= base}
    if sorted(added.values()) != list(range(base, base + len(added))):
        raise ValueError(f"added tokens are not a contiguous block at {base}: {added}")
    return added


def fast_tokenizer_pieces(fast: dict) -> list[str]:
    model = fast.get("model", {})
    # See the boundary converter: Unigram is the structural requirement, the
    # length is not fixed (128000 for DeBERTa-v3, 250101 for mDeBERTa-v3), and
    # the checkpoint's embedding shape is the authority.
    if model.get("type") != "Unigram" or not model.get("vocab"):
        raise ValueError("unsupported tokenizer.json vocabulary")
    pieces = [entry[0] for entry in model["vocab"]]
    # The eleven schema specials occupy a contiguous block starting right after
    # the vocabulary: 128000 for the DeBERTa-v3 tokenizers, 250101 for
    # mDeBERTa-v3's. Deriving the base from the vocabulary length is what makes
    # this work for both, and `added_tokens` cross-checks it against whatever
    # `tokenizer_config.json` declares.
    declared = {entry["content"]: entry["id"] for entry in fast.get("added_tokens", [])}
    base = len(pieces)
    for offset, token in enumerate(SPECIAL_TOKENS):
        if declared.get(token) != base + offset:
            raise ValueError(
                f"tokenizer.json places {token!r} at {declared.get(token)!r}, "
                f"expected {base + offset} (vocab is {base} pieces)"
            )
    return pieces


def tensor_contracts(encoder: dict, vocab_size: int) -> dict[str, tuple]:
    d, f = encoder["hidden_size"], encoder["intermediate_size"]
    # `create_mlp(input_dim=hidden, intermediate_dims=[hidden * 2], output_dim=1)`.
    wide = d * 2
    shapes = {
        "token_embd.weight": (vocab_size, d),
        "tok_norm.weight": (d,), "tok_norm.bias": (d,),
        "rel_embeddings.weight": (encoder["position_buckets"] * 2, d),
        "rel_norm.weight": (d,), "rel_norm.bias": (d,),
        "classifier.0.weight": (wide, d), "classifier.0.bias": (wide,),
        "classifier.2.weight": (1, wide), "classifier.2.bias": (1,),
    }
    for i in range(encoder["num_hidden_layers"]):
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


def source_contracts(encoder: dict, vocab_size: int) -> dict[str, str]:
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
    for i in range(encoder["num_hidden_layers"]):
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
    encoder = validate_config(config)
    tokenizer_config = json.loads((model_dir / "tokenizer_config.json").read_text())
    # fastino ships both the slow `DebertaV2Tokenizer` (gliner2-large-v1,
    # GLiNER2.5-Decide) and the fast `DebertaV2TokenizerFast` (gliner2-base-v1)
    # for what is the same DeBERTa-v2 SPM vocabulary. The class name is only a
    # flavour marker; the properties that decide the pieces and the normalizer
    # are checked on the next line, so accept exactly these two rather than
    # widening to any `Deberta*`.
    if tokenizer_config.get("tokenizer_class") not in TOKENIZER_CLASSES:
        raise ValueError(f"Expected one of {sorted(TOKENIZER_CLASSES)}, "
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
        pieces = fast_tokenizer_pieces(fast)
        spm = None
    else:
        spm = parse_spm((model_dir / "spm.model").read_bytes())
        if spm["normalizer"].get("name") != "nmt_nfkc":
            raise ValueError(f"Unsupported normalizer {spm['normalizer'].get('name')!r}")
        pieces = spm["pieces"]
    added = added_tokens(tokenizer_config, pieces, fast)
    vocab_size = max(added.values()) + 1
    contracts = tensor_contracts(encoder, vocab_size)
    sources = source_contracts(encoder, vocab_size)

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
    writer.add_meta("general.name", model_dir.name)
    writer.add_meta("general.file_type", "F32")
    writer.add_meta(f"{ARCH}.block_count", encoder["num_hidden_layers"])
    writer.add_meta(f"{ARCH}.embedding_length", encoder["hidden_size"])
    writer.add_meta(f"{ARCH}.feed_forward_length", encoder["intermediate_size"])
    writer.add_meta(f"{ARCH}.attention.head_count", encoder["num_attention_heads"])
    writer.add_meta(f"{ARCH}.attention.head_dim", encoder["hidden_size"] // encoder["num_attention_heads"])
    writer.add_meta(f"{ARCH}.attention.scale_divisor", SCALE_DIVISOR)
    writer.add_meta(f"{ARCH}.attention.layer_norm_epsilon", encoder["layer_norm_eps"])
    writer.add_meta(f"{ARCH}.relative_attention.bucket_size", encoder["position_buckets"])
    writer.add_meta(f"{ARCH}.relative_attention.max_relative_positions", encoder["max_position_embeddings"])
    writer.add_meta(f"{ARCH}.norm_rel_embeddings", True)
    writer.add_meta(f"{ARCH}.hidden_act", encoder["hidden_act"])
    writer.add_meta(f"{ARCH}.vocab_size", vocab_size)
    writer.add_meta(f"{ARCH}.classifier.intermediate_size", encoder["hidden_size"] * 2)
    writer.add_meta(f"{ARCH}.classifier.activation", "relu")
    source_keys = ("model_name", "model_type", *EXPECTED_CONFIG_OPTIONAL)
    writer.add_meta(f"{ARCH}.source_architecture", json.dumps({k: config.get(k) for k in source_keys}))
    writer.add_meta(f"{ARCH}.source_config", json.dumps(encoder))
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
