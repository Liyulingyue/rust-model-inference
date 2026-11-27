"""Shared machinery for the two GLiNER GGUF converters.

``convert_gliner.py`` (span family) and ``convert_boundary.py`` (boundary
family) encode the same DeBERTa encoder and the same self-contained tokenizer
block, and both had grown a byte-identical copy of the protobuf ``spm.model``
reader, the encoder metadata block, the tokenizer metadata block, the
streaming tensor writer, and the atomic-write tail. This module is the single
copy of that machinery so a fix lands once.

What stays in the per-variant converters is exactly what differs: how the
size table is keyed, the encoder tensor *name* map (the boundary classifier
lives at index 3, Decide's at 2), the ``boundary_head`` settings transcription
and its shape cross-checks, the bundled-head pass-through, and the variant
metadata.

The byte-exactness of the emitted GGUF is the contract here: these helpers
must keep producing the same metadata keys, in the same order, with the same
values as the copies they replace.
"""
from __future__ import annotations

import base64
import hashlib
import math
import struct

import numpy as np

from tools.converter.utils.gguf import GGML_F32, gguf_dims

ARCH = "gliner2"

# `DisentangledSelfAttention.forward` sets `scale_factor = 1 + |pos_att_type|`
# and divides the content-content *and* both position terms by
# `sqrt(head_dim * scale_factor)`, so this is 3, not 1.
SCALE_DIVISOR = 3

# As resolved by `AutoConfig.from_pretrained` inside the extractor loaders, for
# every DeBERTa-v3 size the family ships. The relative-attention settings are
# what resolves identically for every v3 size; only the four size fields vary.
ENCODER_COMMON = {"hidden_act": "gelu", "layer_norm_eps": 1e-7,
                  "relative_attention": True, "position_buckets": 256,
                  "max_position_embeddings": 512, "max_relative_positions": -1,
                  "position_biased_input": False, "type_vocab_size": 0,
                  "norm_rel_ebd": "layer_norm", "pos_att_type": "p2c|c2p",
                  "share_att_key": True}

# Both the slow and the fast DeBERTa-v2 tokenizer classes, over one SPM vocab.
TOKENIZER_CLASSES = ("DebertaV2Tokenizer", "DebertaV2TokenizerFast")


def resolve_encoder(model_name: str, sizes: dict, family: str) -> dict:
    """The encoder dims for `model_name`, or an error naming what is supported.

    An unknown name raises rather than falling back to any default size: a
    silently wrong ``hidden_size`` would either trip the shape contracts or,
    worse, pass them and produce a GGUF that decodes to noise.
    """
    if model_name not in sizes:
        raise ValueError(f"unsupported {family} model_name {model_name!r}; "
                         f"expected one of {sorted(sizes)}")
    return {**ENCODER_COMMON, **sizes[model_name]}


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
# Encoder tensor shape contract
# ---------------------------------------------------------------------------

def fast_tokenizer_pieces(fast: dict, specials: tuple[str, ...], base_offset: int) -> list[str]:
    """The Unigram pieces out of a `tokenizer.json`, validating the specials.

    Unigram is the structural requirement; the vocabulary *length* is not fixed
    (128000 for DeBERTa-v3, 250101 for mDeBERTa-v3), so the checkpoint's
    `word_embeddings` shape stays the authority. What this does check is that
    the schema specials occupy the contiguous block right after the vocabulary,
    at `len(pieces) + base_offset`: the span family leads with `[MASK]` at the
    vocabulary end (offset 0) and the boundary family omits `[MASK]` from its
    list because that token ships inside the base vocab (offset 1). A tokenizer
    that drifted here would otherwise load with a subtly shifted marker table.
    """
    model = fast.get("model", {})
    if model.get("type") != "Unigram" or not model.get("vocab"):
        raise ValueError("unsupported tokenizer.json vocabulary")
    pieces = [entry[0] for entry in model["vocab"]]
    declared = {entry["content"]: entry["id"] for entry in fast.get("added_tokens", [])}
    base = len(pieces) + base_offset
    for offset, token in enumerate(specials):
        if declared.get(token) != base + offset:
            raise ValueError(
                f"tokenizer.json places {token!r} at {declared.get(token)!r}, "
                f"expected {base + offset} (vocab is {len(pieces)} pieces)"
            )
    return pieces

def encoder_tensor_contracts(encoder: dict, vocab_size: int,
                             last_layer_index: int) -> dict[str, tuple]:
    """GGUF tensor name -> expected shape for the encoder + classifier.

    `hidden_size`, `num_hidden_layers` and `intermediate_size` are re-derived
    from the checkpoint's own shapes through this table, so a wrong entry in a
    variant's size table fails the conversion. `num_attention_heads` appears in
    no tensor shape, which makes it the one field only the byte-exact oracle can
    catch.

    `last_layer_index` is the classifier's output linear: 2 for the span family
    (``create_mlp`` puts ReLU at 1), 3 for the boundary family (GeLU + dropout
    in between). It is the only tensor name here that differs between them.
    """
    d, f = encoder["hidden_size"], encoder["intermediate_size"]
    # `create_mlp(input_dim=hidden, intermediate_dims=[hidden * 2], output_dim=1)`.
    wide = d * 2
    shapes = {
        "token_embd.weight": (vocab_size, d),
        "tok_norm.weight": (d,), "tok_norm.bias": (d,),
        "rel_embeddings.weight": (encoder["position_buckets"] * 2, d),
        "rel_norm.weight": (d,), "rel_norm.bias": (d,),
        "classifier.0.weight": (wide, d), "classifier.0.bias": (wide,),
        f"classifier.{last_layer_index}.weight": (1, wide),
        f"classifier.{last_layer_index}.bias": (1,),
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


# ---------------------------------------------------------------------------
# Metadata blocks
# ---------------------------------------------------------------------------

def add_encoder_meta(writer, name: str, encoder: dict, vocab_size: int) -> None:
    """The encoder metadata block, shared by every variant, in emission order."""
    writer.add_meta("general.architecture", ARCH)
    writer.add_meta("general.name", name)
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


def add_tokenizer_meta(writer, tokens: list[str], vocab_size: int, spm: dict | None,
                       fast_json: str | None, added: dict[str, int]) -> None:
    """The self-contained tokenizer block: either the raw ``tokenizer.json`` or
    the SPM vocabulary plus its normalizer.

    Emitting the whole ``tokenizer.json`` (rather than a re-derived vocabulary)
    is what lets the GGUF carry a tokenizer that needs no sidecar file. The SPM
    path is the pre-2.5 fallback.
    """
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
        writer.add_meta("tokenizer.ggml.token_type",
                        [3 if t in (1, 2, 3) else 4 if t == 6 else 1 for t in types])
    writer.add_meta(f"{ARCH}.special_tokens",
                    [token for token, _ in sorted(added.items(), key=lambda kv: kv[1])])
    if spm is not None:
        writer.add_meta(f"{ARCH}.spm.piece_count", len(spm["pieces"]))
        writer.add_meta(f"{ARCH}.spm.piece_types", types)
        writer.add_meta(f"{ARCH}.spm.charsmap_b64",
                        base64.b64encode(spm["normalizer"]["charsmap"]).decode())
        for flag in ("byte_fallback", "add_dummy_prefix", "remove_extra_whitespaces",
                     "escape_whitespaces", "treat_whitespace_as_suffix"):
            writer.add_meta(f"{ARCH}.spm.{flag}", bool(spm["normalizer"][flag]))
        writer.add_meta(f"{ARCH}.spm.normalizer", spm["normalizer"]["name"])


def add_source_sha256(writer, source_path) -> None:
    with source_path.open("rb") as stream:
        writer.add_meta(f"{ARCH}.source_sha256",
                        hashlib.file_digest(stream, "sha256").hexdigest())


# ---------------------------------------------------------------------------
# Tensor streaming / atomic write
# ---------------------------------------------------------------------------

def tensor_chunks(source_path, data_offset: int, info, key: str):
    """Stream one safetensors tensor as F32 GGUF chunks, rejecting non-finite.

    The finiteness check is the only place a corrupt upstream checkpoint turns
    into a GGUF that loads and decodes to NaN logits instead of failing at
    conversion time.
    """
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


def emit_tensor(writer, name: str, shape, chunks) -> None:
    writer.add_tensor_chunks(name, GGML_F32, gguf_dims(shape), math.prod(shape) * 4, chunks)


def write_atomic(writer, output) -> None:
    """Write through a temporary file so a failed conversion leaves no partial
    GGUF behind for the next run to trip over."""
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
