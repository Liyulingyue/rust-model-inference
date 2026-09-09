#!/usr/bin/env python3
"""Convert Breeze TTS 2 safetensors checkpoints to unquantized GGUF."""
from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dots"))
from convert_dots_tts import GGML_BF16, GGML_F32, GgufWriter, gguf_dims  # noqa: E402

ORACLE_COMMIT = "e2c5ac2f54fe15daa94237a7dbf31e446660a4c9"
_ELEMENT_BYTES = {"BF16": 2, "F32": 4}


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


def _load_audio(model_dir: Path):
    path = _input_path(model_dir, "audio_tokenizer/model.safetensors")
    header = _read_shard(path)
    for name, entry in header.items():
        if entry[0] != "F32":
            raise ValueError(f"codec tensor {name}: dtype expected F32, got {entry[0]}")
    return [(name, path, entry) for name, entry in sorted(header.items())]


def _build(path: Path, architecture: str, config_key: str, config: str, tensors, tokenizer: str | None = None):
    writer = GgufWriter(path)
    writer.add_meta("general.architecture", architecture)
    writer.add_meta(config_key, config)
    if tokenizer is not None:
        writer.add_meta("breeze.tokenizer_json", tokenizer)
        writer.add_meta("breeze.oracle.commit", ORACLE_COMMIT)
    for name, source, entry in tensors:
        dtype, shape, _start, _end = entry
        nbytes = _end - _start
        def chunks(source=source, entry=entry):
            _dtype, _shape, start, end = entry
            with source.open("rb") as source_file:
                header_len = struct.unpack("<Q", source_file.read(8))[0]
                source_file.seek(8 + header_len + start)
                remaining = end - start
                while remaining:
                    chunk = source_file.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise ValueError(f"{source}: truncated tensor")
                    remaining -= len(chunk)
                    yield chunk
        writer.add_tensor_chunks(name, {"BF16": GGML_BF16, "F32": GGML_F32}[dtype], gguf_dims(shape), nbytes, chunks)
    writer.write()


def convert(model_dir: Path, out_dir: Path) -> tuple[Path, Path]:
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
    main_path = out_dir / "breeze-tts-2-BF16.gguf"
    codec_path = out_dir / "breeze-tts-2-codec-F32.gguf"
    if main_path.exists() or codec_path.exists():
        raise FileExistsError("output already exists")
    main_tensors, codec_tensors = _load_main(model_dir), _load_audio(model_dir)
    tmpdir = Path(tempfile.mkdtemp(prefix=".breeze-", dir=out_dir))
    try:
        tmp_main, tmp_codec = tmpdir / main_path.name, tmpdir / codec_path.name
        _build(tmp_main, "breeze", "breeze.config", config_raw, main_tensors, tokenizer_raw)
        _build(tmp_codec, "breeze_audio", "breeze_audio.config", audio_config_raw, codec_tensors)
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
    args = parser.parse_args()
    out = args.out_dir or args.model_dir
    print("\n".join(str(p) for p in convert(args.model_dir, out)))


if __name__ == "__main__":
    main()
