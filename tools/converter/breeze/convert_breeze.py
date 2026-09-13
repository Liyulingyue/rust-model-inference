#!/usr/bin/env python3
"""Convert Breeze TTS 2 safetensors checkpoints to GGUF.

Precision modes (``--quant`` for the main model):
    bf16    BF16 source (default; matches upstream dtype)
    f16     F16 source (BF16 weights re-encoded as F16)
    f32     F32 source (BF16 weights up-cast)
    q8_0    Q8_0 for learned 2D weight matrices, source precision elsewhere
    q4_0    Q4_0 for learned 2D weight matrices, source precision elsewhere

The codec (audio) GGUF is always F32 by default; pass ``--codec-quant q8_0``
to quantise learned 2D tensors inside the codec as well.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from converter.utils.gguf import (  # noqa: E402
    GGML_BF16,
    GGML_F16,
    GGML_F32,
    GGML_Q4_0,
    GGML_Q8_0,
    GgufWriter,
    bf16_to_f32,
    bf16_to_f16,
    f32_to_bf16,
    f32_to_f16,
    gguf_dims,
    quantize_q4_0,
    quantize_q8_0,
)

_ELEMENT_BYTES = {"BF16": 2, "F32": 4}

_MAIN_QUANTS = {"bf16", "f16", "f32", "q8_0", "q4_0"}
_CODEC_QUANTS = {"f32", "q8_0"}
_QUANT_TO_GGML = {"bf16": GGML_BF16, "f16": GGML_F16, "f32": GGML_F32, "q8_0": GGML_Q8_0, "q4_0": GGML_Q4_0}
_QUANT_TO_SUFFIX = {"bf16": "BF16", "f16": "F16", "f32": "F32", "q8_0": "Q8_0", "q4_0": "Q4_0"}
_LEARNED_LEAFS = {"weight"}


def _is_quantisable_2d_weight(name: str, shape: tuple[int, ...]) -> bool:
    """Match the dots.tts rule plus the GGML Q4_0/Q8_0 block constraint.

    A learned 2D weight row whose width is a multiple of 32 can be
    quantised.  Anything else (e.g. narrow classifiers, bias-shaped tensors)
    keeps its source precision even under ``q8_0``/``q4_0``.
    """
    if len(shape) != 2:
        return False
    leaf = name.rsplit(".", 1)[-1]
    if leaf not in _LEARNED_LEAFS:
        return False
    row_width = shape[1]
    return row_width % 32 == 0


def _is_learned_2d_weight(name: str, shape: tuple[int, ...]) -> bool:
    """Loose learned check (no row-width filter); used to skip codec_model."""
    if len(shape) != 2:
        return False
    leaf = name.rsplit(".", 1)[-1]
    return leaf in _LEARNED_LEAFS


def _parse_json(text: str | bytes, path: Path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"{path}: duplicate JSON key {key}")
            result[key] = value
        return result

    def constant(value):
        raise ValueError(f"{path}: invalid JSON constant {value}")

    try:
        return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError(f"{path}: invalid JSON") from exc


def _json(path: Path) -> tuple[object, str]:
    text = path.read_bytes().decode("utf-8")
    return _parse_json(text, path), text


def _product(shape, name):
    if not isinstance(shape, list) or not 1 <= len(shape) <= 4 or any(type(x) is not int or x <= 0 for x in shape):
        raise ValueError(f"{name}: invalid shape {shape!r}")
    return math.prod(shape)


def _read_shard(path: Path) -> dict[str, tuple[str, tuple[int, ...], int, int]]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"invalid shard path: {path}")
    with path.open("rb") as source:
        prefix = source.read(8)
        if len(prefix) != 8:
            raise ValueError(f"{path}: truncated safetensors header")
        header_len = struct.unpack("<Q", prefix)[0]
        file_size = path.stat().st_size
        if not 2 <= header_len <= min(file_size - 8, 100_000_000):
            raise ValueError(f"{path}: invalid or truncated safetensors header length")
        header_raw = source.read(header_len)
        if len(header_raw) != header_len:
            raise ValueError(f"{path}: truncated safetensors header")
        header = _parse_json(header_raw, path)
        if not isinstance(header, dict):
            raise ValueError(f"{path}: safetensors header must be object")
        data_start, file_size = 8 + header_len, path.stat().st_size
        result = {}
        for name, info in header.items():
            if name == "__metadata__":
                continue
            if not name or not isinstance(info, dict):
                raise ValueError(f"{path}: malformed tensor entry")
            dtype, shape, offsets = info.get("dtype"), info.get("shape"), info.get("data_offsets")
            if dtype not in _ELEMENT_BYTES:
                raise ValueError(f"{path}: unsupported dtype {dtype!r} for {name}")
            count = _product(shape, name)
            if not isinstance(offsets, list) or len(offsets) != 2 or any(type(x) is not int for x in offsets):
                raise ValueError(f"{path}: invalid offsets for {name}")
            start, end = offsets
            if start < 0 or end < start or end - start != count * _ELEMENT_BYTES[dtype]:
                raise ValueError(f"{path}: invalid byte length for {name}")
            if data_start + end > file_size:
                raise ValueError(f"{path}: tensor {name} exceeds file")
            result[name] = (dtype, tuple(shape), start, end)
        spans = sorted((v[2], v[3], n) for n, v in result.items())
        if (not spans or spans[0][0] != 0 or spans[-1][1] != file_size - data_start
                or any(a[1] != b[0] for a, b in zip(spans, spans[1:]))):
            raise ValueError(f"{path}: incomplete or overlapping tensor ranges")
        return result


def _input_path(root: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"shard path escapes model directory: {value}")
    path = (root / candidate).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"shard path escapes model directory: {value}")
    return path


def _load_main(model_dir: Path):
    index_obj, _ = _json(_input_path(model_dir, "model.safetensors.index.json"))
    if not isinstance(index_obj, dict) or not isinstance(index_obj.get("weight_map"), dict):
        raise ValueError("model.safetensors.index.json: missing weight_map")
    weight_map = index_obj["weight_map"]
    declared_size = index_obj.get("metadata", {}).get("total_size") if isinstance(index_obj.get("metadata"), dict) else None
    if type(declared_size) is not int or declared_size <= 0:
        raise ValueError("model.safetensors.index.json: missing metadata.total_size")
    shards = {}
    for name, shard_name in weight_map.items():
        if not isinstance(name, str) or not isinstance(shard_name, str):
            raise ValueError("model.safetensors.index.json: invalid weight_map")
        path = _input_path(model_dir, shard_name)
        if shard_name not in shards:
            shards[shard_name] = (path, _read_shard(path))
    entries = {}
    for shard_name, (path, header) in shards.items():
        for name, entry in header.items():
            if weight_map.get(name) != shard_name:
                raise ValueError(f"{name}: shard index points to the wrong file")
            if name in entries:
                raise ValueError(f"duplicate tensor across shards: {name}")
            entries[name] = (path, entry)
    if set(entries) != set(weight_map):
        raise ValueError("shard index/header tensor membership mismatch")
    if sum(entry[1][3] - entry[1][2] for entry in entries.values()) != declared_size:
        raise ValueError("metadata.total_size does not match tensor bytes")
    return [(name, entries[name][0], entries[name][1]) for name in sorted(entries)]


def _load_audio(model_dir: Path, codec_quant: str):
    """Load codec tensors.  codec_quant in {"f32", "q8_0"}.

    The Mimi codec checkpoints ship every tensor as F32.  We accept that as
    the source dtype; ``q8_0`` only changes the encoded GGUF precision for
    learned 2D weight matrices via ``_build``.
    """
    if codec_quant not in _CODEC_QUANTS:
        raise ValueError(f"unsupported --codec-quant {codec_quant!r}")
    path = _input_path(model_dir, "audio_tokenizer/model.safetensors")
    header = _read_shard(path)
    for name, entry in header.items():
        if entry[0] != "F32":
            raise ValueError(f"codec tensor {name}: dtype expected F32, got {entry[0]}")
    return [(name, path, entry) for name, entry in sorted(header.items())]


def _must_keep_source(name: str) -> bool:
    """Tensors the Breeze Rust loader reads via ``load_f32_tensor`` (F32/BF16
    only).  Keeping their source dtype intact lets ``f16``/``f32``/``q8_0``/
    ``q4_0`` GGUF outputs load under the relaxed preflight without touching
    the per-tensor type for every norm in the model.
    """
    if name == "depth_decoder.codebooks_head.weight":
        return True
    if name == "text_encoder.embed_tokens.eoi_embedding":
        return True
    if name.endswith(".norm.weight"):
        return True
    for prefix, count, norms in [
        ("text_encoder", 26, [
            "pre_self_attn_layernorm", "post_self_attn_layernorm",
            "pre_feedforward_layernorm", "post_feedforward_layernorm",
            "self_attn.q_norm", "self_attn.k_norm",
        ]),
        ("backbone_model", 28, [
            "input_layernorm", "post_attention_layernorm",
            "self_attn.q_norm", "self_attn.k_norm",
        ]),
        ("depth_decoder.model", 12, [
            "input_layernorm", "post_attention_layernorm",
        ]),
    ]:
        for index in range(count):
            for norm in norms:
                if name == f"{prefix}.layers.{index}.{norm}.weight":
                    return True
    return False


def _source_ggml_type(name: str, source_dtype: str, target_quant: str):
    """Pick the GGUF tensor type for a tensor under ``target_quant``.

    The following tensors keep their source dtype regardless of
    ``target_quant``:
      * ``codec_model.*`` — Mimi codec module snapshots (BF16 conv weights,
        F32 codebook initialisation flags).
      * norm weights and codebook/eoi embedding vectors — the Rust loader
        feeds these through ``load_f32_tensor`` (F32/BF16 only); quantising
        them would break the loader even if the underlying kernels
        supported it.

    Other tensors honour ``target_quant``; learned 2D weights additionally
    get Q8_0/Q4_0 encoding via ``_quantized_payload``.
    """
    if name.startswith("codec_model.") or _must_keep_source(name):
        return {"BF16": GGML_BF16, "F32": GGML_F32}[source_dtype]
    if target_quant in {"q8_0", "q4_0"}:
        return {"BF16": GGML_BF16, "F32": GGML_F32}[source_dtype]
    return _QUANT_TO_GGML[target_quant]


def _quantized_payload(
    name: str,
    source_dtype: str,
    shape: tuple[int, ...],
    chunks_iter,
    target_quant: str,
):
    """Yield the GGUF payload bytes for a single tensor under ``target_quant``.

    For ``bf16``/``f16``/``f32`` we re-encode BF16 -> target precision in
    chunks to bound transient memory.  For ``q8_0``/``q4_0`` we materialise
    the F32 view, quantise per 32-element block, and stream the encoded
    blocks back.  Quantised learned weights transposed to ``(row, n_out)``
    so the GGUF dims read as ``out_features x in_features``.
    """
    if target_quant in {"bf16"}:
        if source_dtype == "BF16":
            for chunk in chunks_iter:
                yield chunk
        elif source_dtype == "F32":
            for chunk in chunks_iter:
                yield f32_to_bf16(chunk) if chunk else chunk
        else:
            raise ValueError(f"{name}: unsupported source dtype {source_dtype}")
        return
    if target_quant == "f16":
        if source_dtype == "BF16":
            for chunk in chunks_iter:
                yield bf16_to_f16(chunk) if chunk else chunk
        elif source_dtype == "F32":
            for chunk in chunks_iter:
                yield f32_to_f16(chunk) if chunk else chunk
        else:
            raise ValueError(f"{name}: unsupported source dtype {source_dtype}")
        return
    if target_quant == "f32":
        if source_dtype == "BF16":
            for chunk in chunks_iter:
                yield bf16_to_f32(chunk) if chunk else chunk
        elif source_dtype == "F32":
            for chunk in chunks_iter:
                yield chunk
        else:
            raise ValueError(f"{name}: unsupported source dtype {source_dtype}")
        return
    if target_quant in {"q8_0", "q4_0"}:
        if not _is_quantisable_2d_weight(name, shape):
            # Not eligible for GGML block quantisation; fall back to source
            # bytes (caller already picked ggml_type = source dtype).
            for chunk in chunks_iter:
                yield chunk
            return
        quant_fn = quantize_q8_0 if target_quant == "q8_0" else quantize_q4_0
        if source_dtype == "BF16":
            words = np.frombuffer(b"".join(chunks_iter), dtype="<u2")
            f32 = (words.astype(np.uint32) << np.uint32(16)).view(np.float32).reshape(shape)
        elif source_dtype == "F32":
            f32 = np.frombuffer(b"".join(chunks_iter), dtype="<f4").reshape(shape)
        else:
            raise ValueError(f"{name}: unsupported source dtype {source_dtype}")
        yield quant_fn(f32)
        return
    raise ValueError(f"unsupported target_quant {target_quant!r}")


def _resolve_gguf_dims(name: str, source_dtype: str, shape: tuple[int, ...], target_quant: str):
    """GGUF dims for a tensor.  Quantised learned 2D weights are reversed."""
    return gguf_dims(shape)


def _build(
    path: Path,
    architecture: str,
    config_key: str,
    config: str,
    tensors,
    target_quant: str,
    tokenizer: str | None = None,
):
    writer = GgufWriter(path)
    writer.add_meta("general.architecture", architecture)
    writer.add_meta(config_key, config)
    if tokenizer is not None:
        writer.add_meta("breeze.tokenizer_json", tokenizer)
    for name, source, entry in tensors:
        source_dtype, shape, start, end = entry
        nbytes = end - start
        ggml_type = _source_ggml_type(name, source_dtype, target_quant)
        gguf_d = _resolve_gguf_dims(name, source_dtype, shape, target_quant)
        if (
            target_quant in {"q8_0", "q4_0"}
            and _is_quantisable_2d_weight(name, shape)
            and not name.startswith("codec_model.")
        ):
            ggml_type = GGML_Q8_0 if target_quant == "q8_0" else GGML_Q4_0

        def chunks(source=source, entry=entry):
            _dtype, _shape, _start, _end = entry
            with source.open("rb") as source_file:
                header_len = struct.unpack("<Q", source_file.read(8))[0]
                source_file.seek(8 + header_len + _start)
                remaining = _end - _start
                while remaining:
                    chunk = source_file.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise ValueError(f"{source}: truncated tensor")
                    remaining -= len(chunk)
                    yield chunk

        # codec_model.* tensors and F32-only tensors (norm / eoi / codebook
        # head) keep their source bytes verbatim — they cannot be re-encoded
        # to F16/F32/Q8_0/Q4_0 because the Rust loader pins them to F32 or
        # BF16 via ``load_f32_tensor``.
        if name.startswith("codec_model.") or _must_keep_source(name):
            encoded = b"".join(chunks())
        else:
            payload_iter = _quantized_payload(name, source_dtype, shape, chunks(), target_quant)
            encoded = b"".join(payload_iter)
        writer.add_tensor(name, ggml_type, gguf_d, encoded)
    writer.write()


def convert(
    model_dir: Path,
    out_dir: Path,
    main_quant: str = "bf16",
    codec_quant: str = "f32",
) -> tuple[Path, Path]:
    if main_quant not in _MAIN_QUANTS:
        raise ValueError(f"unsupported --quant {main_quant!r}; choices: {sorted(_MAIN_QUANTS)}")
    if codec_quant not in _CODEC_QUANTS:
        raise ValueError(f"unsupported --codec-quant {codec_quant!r}; choices: {sorted(_CODEC_QUANTS)}")
    model_dir = model_dir.expanduser().resolve()
    out_dir = out_dir.expanduser().resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config, config_raw = _json(_input_path(model_dir, "config.json"))
    tokenizer, tokenizer_raw = _json(_input_path(model_dir, "tokenizer.json"))
    audio_config, audio_config_raw = _json(_input_path(model_dir, "audio_tokenizer/config.json"))
    if not isinstance(config, dict) or config.get("model_type") != "breeze":
        raise ValueError("config.json: model_type must be breeze")
    if not isinstance(tokenizer, dict):
        raise ValueError("tokenizer.json: expected object")
    if not isinstance(audio_config, dict) or audio_config.get("model_type") != "qwen3_tts_tokenizer_12hz":
        raise ValueError("audio_tokenizer/config.json: invalid model_type")
    main_suffix = _QUANT_TO_SUFFIX[main_quant]
    codec_suffix = _QUANT_TO_SUFFIX[codec_quant]
    main_path = out_dir / f"breeze-tts-2-{main_suffix}.gguf"
    codec_path = out_dir / f"breeze-tts-2-mmproj-{codec_suffix}.gguf"
    if main_path.exists() or codec_path.exists():
        raise FileExistsError("output already exists")
    main_tensors = _load_main(model_dir)
    codec_tensors = _load_audio(model_dir, codec_quant)
    tmpdir = Path(tempfile.mkdtemp(prefix=".breeze-", dir=out_dir))
    try:
        tmp_main, tmp_codec = tmpdir / main_path.name, tmpdir / codec_path.name
        _build(tmp_main, "breeze", "breeze.config", config_raw, main_tensors, main_quant, tokenizer_raw)
        _build(tmp_codec, "breeze_audio", "breeze_audio.config", audio_config_raw, codec_tensors, codec_quant)
        os.link(tmp_main, main_path)
        try:
            os.link(tmp_codec, codec_path)
        except Exception:
            main_path.unlink(missing_ok=True)
            raise
    finally:
        for child in tmpdir.iterdir():
            child.unlink(missing_ok=True)
        tmpdir.rmdir()
    return main_path, codec_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument(
        "--quant",
        choices=sorted(_MAIN_QUANTS),
        default="bf16",
        help="main model precision (default bf16)",
    )
    parser.add_argument(
        "--codec-quant",
        choices=sorted(_CODEC_QUANTS),
        default="f32",
        help="codec mmproj precision (default f32)",
    )
    args = parser.parse_args()
    out = args.out_dir or args.model_dir
    print("\n".join(str(p) for p in convert(args.model_dir, out, args.quant, args.codec_quant)))


if __name__ == "__main__":
    main()
