from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path

from tools.converter.utils.gguf import (
    GGML_BF16,
    GGML_F32,
    GgufWriter,
    gguf_dims,
    open_safetensors,
    read_gguf_directory,
)


EOD = 151643
ABC_START, ABC_END = 151847, 151848
MUSIC_START, MUSIC_END = 151851, 151852
CODEC_OFFSET, CODEC_SIZE = 151853, 32768
LATENT_START, LATENT_END, LATENT_PAD = 184621, 184622, 184623
VOCAB_SIZE, CONTEXT = 184704, 24576
PROTOCOL_VERSION = "yue2-native-v1"

_CHUNK_BYTES = 8 * 1024 * 1024


def _byte_to_gpt2_table() -> dict[int, str]:
    visible = list(range(ord("!"), ord("~") + 1))
    visible += list(range(ord("¡"), ord("¬") + 1))
    visible += list(range(ord("®"), ord("ÿ") + 1))
    extra = [byte for byte in range(256) if byte not in visible]
    chars = visible + [256 + index for index in range(len(extra))]
    return dict(zip(visible + extra, map(chr, chars)))


_BYTE_TO_GPT2 = _byte_to_gpt2_table()


def bytes_to_gpt2(raw: bytes) -> str:
    return "".join(_BYTE_TO_GPT2[byte] for byte in raw)


def split_before_rank(token: bytes, ranks: dict[bytes, int], rank: int) -> tuple[bytes, bytes]:
    parts = [bytes([byte]) for byte in token]
    while len(parts) > 2:
        candidates = [(ranks.get(parts[i] + parts[i + 1], rank), i) for i in range(len(parts) - 1)]
        merge_rank, index = min(candidates)
        if merge_rank >= rank:
            break
        parts[index : index + 2] = [parts[index] + parts[index + 1]]
    if len(parts) != 2:
        raise ValueError(f"cannot reconstruct merge rank {rank}")
    return parts[0], parts[1]


def parse_qwen_tiktoken(path: Path) -> dict[bytes, int]:
    ranks: dict[bytes, int] = {}
    for line_number, line in enumerate(path.read_bytes().splitlines(), 1):
        if not line:
            continue
        try:
            encoded, raw_rank = line.split()
            token = base64.b64decode(encoded, validate=True)
            rank = int(raw_rank)
        except (ValueError, TypeError) as error:
            raise ValueError(f"qwen.tiktoken:{line_number}: invalid entry") from error
        if rank != len(ranks):
            raise ValueError(f"qwen.tiktoken:{line_number}: expected rank {len(ranks)}, got {rank}")
        if token in ranks:
            raise ValueError(f"qwen.tiktoken:{line_number}: duplicate token")
        ranks[token] = rank
    return ranks


def tokenizer_metadata(merge_file: Path) -> dict[str, object]:
    ranks = parse_qwen_tiktoken(merge_file)
    if len(ranks) != EOD:
        raise ValueError(f"qwen.tiktoken: expected {EOD} ordinary tokens, got {len(ranks)}")
    ordinary = [raw for raw, _rank in sorted(ranks.items(), key=lambda item: item[1])]
    merges = []
    for rank, raw in enumerate(ordinary):
        if len(raw) > 1:
            left, right = split_before_rank(raw, ranks, rank)
            merges.append(f"{bytes_to_gpt2(left)} {bytes_to_gpt2(right)}")
    specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<R>", "<S>", "<X>", "<mask>", "<sep>"]
    specials += [f"<extra_{i}>" for i in range(200)]
    specials[204:206] = ["<abc>", "</abc>"]
    tokens = [bytes_to_gpt2(raw) for raw in ordinary] + specials
    tokens.extend(f"<yue2_unused_{token_id}>" for token_id in range(len(tokens), VOCAB_SIZE))
    for token_id, token in (
        (MUSIC_START, "<music>"),
        (MUSIC_END, "</music>"),
        (LATENT_START, "<latent>"),
        (LATENT_END, "</latent>"),
        (LATENT_PAD, "<latent_pad>"),
    ):
        tokens[token_id] = token
    types = [1] * EOD + [4] * len(specials) + [5] * (VOCAB_SIZE - EOD - len(specials))
    return {
        "tokenizer.ggml.model": "gpt2",
        "tokenizer.ggml.pre": "qwen2",
        "tokenizer.ggml.tokens": tokens,
        "tokenizer.ggml.token_type": types,
        "tokenizer.ggml.merges": merges,
        "tokenizer.ggml.add_bos_token": False,
        "tokenizer.ggml.add_eos_token": False,
        "tokenizer.ggml.normalizer.nfc": True,
    }


def _load_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise ValueError(f"{path}: invalid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _expect(config: dict[str, object], path: str, expected: object) -> None:
    value: object = config
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"{path}: expected {expected!r}, got missing")
        value = value[key]
    if value != expected:
        raise ValueError(f"{path}: expected {expected!r}, got {value!r}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_entries(source, *, dtype: str, prefix: str | None = None) -> list[tuple[str, tuple[int, ...], int, int]]:
    entries = []
    element_bytes = 2 if dtype == "BF16" else 4
    for name, info in source.header.items():
        if name == "__metadata__" or (prefix is not None and not name.startswith(prefix)):
            continue
        actual_dtype = info.get("dtype")
        if actual_dtype != dtype:
            raise ValueError(f"{name}: expected {dtype}, got {actual_dtype}")
        shape = tuple(info.get("shape", ()))
        if not shape or any(type(dim) is not int or dim <= 0 for dim in shape):
            raise ValueError(f"{name}: invalid shape {shape}")
        offsets = info.get("data_offsets")
        if not isinstance(offsets, list) or len(offsets) != 2 or any(type(offset) is not int for offset in offsets):
            raise ValueError(f"{name}: invalid data_offsets {offsets!r}")
        start, end = offsets
        expected = math.prod(shape) * element_bytes
        if start < 0 or end < start or end - start != expected or source.data_offset + end > source.file_size:
            raise ValueError(f"{name}: payload {end - start} does not match shape {shape} and dtype {dtype}")
        entries.append((name, shape, start, end))
    if not entries:
        raise ValueError(f"model.safetensors: no {prefix or ''}{dtype} tensors")
    return entries


def _require_shape(source, name: str, expected: tuple[int, ...]) -> None:
    info = source.header.get(name)
    if not isinstance(info, dict):
        raise ValueError(f"{name}: missing tensor")
    shape = tuple(info.get("shape", ()))
    if shape != expected:
        raise ValueError(f"{name}: expected shape {expected}, got {shape}")


def _chunks(path: Path, absolute_start: int, length: int) -> Iterable[bytes]:
    with path.open("rb") as source:
        source.seek(absolute_start)
        remaining = length
        while remaining:
            chunk = source.read(min(remaining, _CHUNK_BYTES))
            if not chunk:
                raise ValueError(f"{path}: truncated tensor payload")
            remaining -= len(chunk)
            yield chunk


def _write_atomic(
    output: Path,
    overwrite: bool,
    metadata: dict[str, object],
    source,
    entries: list[tuple[str, tuple[int, ...], int, int]],
    ggml_type: int,
) -> None:
    if output.exists() and not overwrite:
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    expected_tensors = {}
    try:
        writer = GgufWriter(temporary)
        for key, value in metadata.items():
            writer.add_meta(key, value)
        for name, shape, start, end in entries:
            dims = gguf_dims(shape)
            nbytes = end - start
            writer.add_tensor_chunks(
                name,
                ggml_type,
                dims,
                nbytes,
                lambda start=start, nbytes=nbytes: _chunks(source.path, source.data_offset + start, nbytes),
            )
            expected_tensors[name] = (ggml_type, dims)
        writer.write()
        actual_metadata, actual_tensors = read_gguf_directory(temporary)
        if actual_metadata != metadata:
            raise ValueError("GGUF metadata read-back mismatch")
        actual_shapes = {name: (entry[0], entry[1]) for name, entry in actual_tensors.items()}
        if actual_shapes != expected_tensors:
            raise ValueError("GGUF tensor directory read-back mismatch")
        if overwrite:
            os.replace(temporary, output)
        else:
            os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def convert_main(model_dir: Path, output: Path, overwrite: bool = False) -> None:
    model_dir, output = Path(model_dir), Path(output)
    if output.exists() and not overwrite:
        raise FileExistsError(output)
    config = _load_json(model_dir / "config.json")
    for key, expected in {
        "architectures": ["YuE2ForCausalLM"],
        "dtype": "bfloat16",
        "model_type": "yue2",
        "hidden_size": 2048,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "intermediate_size": 6144,
        "vocab_size": VOCAB_SIZE,
        "rms_norm_eps": 0.000001,
        "rope_theta": 1000000,
        "max_position_embeddings": CONTEXT,
        "latent_type": "vae",
        "latent_dim": 64,
        "max_latent_frames": CONTEXT,
        "timestep_shift": 1.0,
    }.items():
        _expect(config, key, expected)
    source = open_safetensors(model_dir / "model.safetensors")
    entries = _tensor_entries(source, dtype="BF16")
    _require_shape(source, "model.layers.0.self_attn.q_proj.weight", (2048, 2048))
    _require_shape(source, "model.layers.0.nar_self_attn.q_proj.weight", (2048, 2048))
    metadata = {
        "general.architecture": "yue2",
        "general.name": "YuE2-3B",
        "general.source_sha256": _sha256(source.path),
        "yue2.protocol_version": PROTOCOL_VERSION,
        "yue2.context_length": CONTEXT,
        "yue2.embedding_length": 2048,
        "yue2.block_count": 28,
        "yue2.attention.head_count": 16,
        "yue2.attention.head_count_kv": 8,
        "yue2.attention.head_dim": 128,
        "yue2.feed_forward_length": 6144,
        "yue2.vocab_size": VOCAB_SIZE,
        "yue2.rms_norm_eps": 0.000001,
        "yue2.rope.freq_base": 1000000,
        "yue2.latent_channels": 64,
        "yue2.timestep_shift": 1.0,
        "yue2.eod_token_id": EOD,
        "yue2.abc_start_token_id": ABC_START,
        "yue2.abc_end_token_id": ABC_END,
        "yue2.music_start_token_id": MUSIC_START,
        "yue2.music_end_token_id": MUSIC_END,
        "yue2.codec_offset": CODEC_OFFSET,
        "yue2.codec_size": CODEC_SIZE,
        "yue2.latent_start_token_id": LATENT_START,
        "yue2.latent_end_token_id": LATENT_END,
        "yue2.latent_pad_token_id": LATENT_PAD,
        "yue2.abc.temperature": 0.7,
        "yue2.abc.top_p": 0.9,
        "yue2.abc.top_k": 30,
        "yue2.abc.repetition_penalty": 1.005,
        "yue2.abc.penalty_window": 100,
        "yue2.abc.min_tokens": 32,
        "yue2.abc.max_tokens": 4096,
        "yue2.semantic.temperature": 1.0,
        "yue2.semantic.top_p": 0.95,
        "yue2.semantic.top_k": 100,
        "yue2.semantic.repetition_penalty": 1.2,
        "yue2.semantic.penalty_window": 50,
        "yue2.semantic.min_tokens": 200,
        "yue2.semantic.max_tokens": 9000,
        "yue2.tensor_count": len(entries),
        **tokenizer_metadata(model_dir / "qwen.tiktoken"),
    }
    _write_atomic(output, overwrite, metadata, source, entries, GGML_BF16)


def convert_vae(vae_dir: Path, output: Path, overwrite: bool = False) -> None:
    vae_dir, output = Path(vae_dir), Path(output)
    if output.exists() and not overwrite:
        raise FileExistsError(output)
    config = _load_json(vae_dir / "config.json")
    for key, expected in {
        "architectures": ["YuE2VAE"],
        "dtype": "float32",
        "model_type": "yue2_vae",
        "decoder_config.channels": 64,
        "decoder_config.latent_dim": 64,
        "decoder_config.out_channels": 2,
        "decoder_config.strides": [2, 2, 4, 4, 5, 6],
        "sample_rate": 48000,
        "latent_dim": 64,
        "downsampling_ratio": 1920,
        "audio_channels": 2,
        "release_variant": "standard",
        "decode_core_frames": 1024,
        "decode_halo_frames": 16,
    }.items():
        _expect(config, key, expected)
    source = open_safetensors(vae_dir / "model.safetensors")
    entries = _tensor_entries(source, dtype="F32", prefix="decoder.")
    _require_shape(source, "decoder.layers.0.bias", (2048,))
    metadata = {
        "general.architecture": "yue2_vae",
        "general.name": "YuE2-Vae",
        "general.source_sha256": _sha256(source.path),
        "yue2_vae.release_variant": "standard",
        "yue2_vae.strides": [2, 2, 4, 4, 5, 6],
        "yue2_vae.latent_channels": 64,
        "yue2_vae.output_channels": 2,
        "yue2_vae.sample_rate": 48000,
        "yue2_vae.downsampling_ratio": 1920,
        "yue2_vae.decode_core_frames": 1024,
        "yue2_vae.decode_halo_frames": 16,
        "yue2_vae.tensor_count": len(entries),
    }
    _write_atomic(output, overwrite, metadata, source, entries, GGML_F32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert the fixed YuE2 main and VAE artifacts to GGUF")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--vae-dir", type=Path, required=True)
    parser.add_argument("--main-out", type=Path, required=True)
    parser.add_argument("--vae-out", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    convert_main(args.model_dir, args.main_out, args.overwrite)
    convert_vae(args.vae_dir, args.vae_out, args.overwrite)
    for label, path in (("main", args.main_out), ("vae", args.vae_out)):
        _metadata, tensors = read_gguf_directory(path)
        print(f"{label}: {len(tensors)} tensors, {path.stat().st_size} bytes, sha256={_sha256(path)}")


if __name__ == "__main__":
    main()
