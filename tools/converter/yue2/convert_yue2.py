from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import tempfile
import time
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from tools.converter.utils.gguf import (
    GGML_BF16,
    GGML_F32,
    GGML_Q4K,
    GGML_Q6K,
    GGML_Q4_0,
    GGML_Q8_0,
    GgufWriter,
    gguf_dims,
    open_safetensors,
    quantize_q4_0,
    quantize_q8_0,
    read_gguf_directory,
)
from tools.converter.utils.kquants import quantize_q4_k, quantize_q6_k


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


# --- Quantization modes -----------------------------------------------------
#
# ``bf16`` is the pass-through default: the payload bytes are sliced straight out
# of the safetensors file, so the conversion is a copy with no math.  Every other
# mode has to decode BF16 -> f32, re-quantize, and therefore materializes the
# matrix in memory.  The block-quantized encoders are ported from
# ``ggml-quants.c`` (see ``tools/converter/utils/kquants.py``).
#
# Only the 2-D projections listed in ``MATRIX_WIDTHS`` are ever quantized.  The
# 1-D norms, biases and the two vocab-sized embedding matrices stay BF16: they
# are either tiny (norms) or extremely sensitive to quantization error relative
# to their role as a lookup table (embeddings), and the Rust loader reads them
# through ``load_f32_tensor`` anyway.
# `ar_q8_0` quantizes only the autoregressive half and leaves the NAR half in
# BF16. The NAR acoustic stream is a flow-matching diffusion solve that
# amplifies weight noise every step: with every projection at Q8_0 (SNR 45 dB,
# 0.54% relative error) the render is still musical at 1 step but turns to noise
# at 4 and 32 steps, while the same weights in BF16 are fine. Keeping NAR in
# BF16 avoids that, at the cost of the AR tokens changing -- a different take
# rather than a corrupted one.
QUANT_MODES = (
    "bf16",
    "f32",
    "q8_0",
    "ar_q8_0",
    "q4_0",
    "q4_k_m",
    "ar_q4_k_m",
    "q6_k",
)

# Per-mode scope: which per-layer projection stream the mode is allowed to touch.
_QUANT_STREAMS = {
    "bf16": (),
    "f32": ("", "nar_"),
    "q8_0": ("", "nar_"),
    "ar_q8_0": ("",),
    "q4_0": ("", "nar_"),
    "q4_k_m": ("", "nar_"),
    "ar_q4_k_m": ("",),
    "q6_k": ("", "nar_"),
}

# Per-layer 2-D projections, mirrored from the authoritative shape table in
# `src/models/yue2/ar.rs` (`YuE2Model::validate_shapes`).  The 1-D norms and
# biases are deliberately absent: the loader reads them through
# `load_f32_tensor`, so they stay BF16.
# Suffixes are relative to the layer base, so the AR stream (no prefix) and the
# NAR stream (`nar_` prefix) read different safetensors tensors and can be
# quantized independently. See `QUANT_MODES` for why that matters.
_AR_LAYER_MATRIX_SUFFIXES: tuple[str, ...] = (
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)

_NAR_LAYER_MATRIX_SUFFIXES: tuple[str, ...] = tuple(
    f"nar_{suffix}" for suffix in _AR_LAYER_MATRIX_SUFFIXES
)

# Global matrices that are ordinary matmuls and worth quantizing.  The two
# vocab-sized matrices (`embed_tokens`, `lm_head`) and the position/bridge
# lookup tables are left in BF16: they are read as tables rather than as dense
# projections, and the transformer projections already account for essentially
# all of the decoder FLOPs.
#
# These belong to the NAR stream despite having no `nar_` in their name: the
# time embedding is only ever consumed by `velocity()` (see
# `time_embedding` in `src/models/yue2/nar.rs`), never by the AR decode.  They
# are listed under the NAR scope so `ar_q8_0` really does leave the whole
# non-autoregressive half in BF16.
_NAR_GLOBAL_MATRIX_NAMES: tuple[str, ...] = (
    "time_embedder.mlp.0.weight",
    "time_embedder.mlp.2.weight",
)

# Smallest block any supported encoder needs.
_MIN_BLOCK_ELEMENTS = {
    "q8_0": 32,
    "q4_0": 32,
    "q4_k_m": 256,
    "ar_q4_k_m": 256,
    "q6_k": 256,
}


def _quantizable_names(layers: int, streams: tuple[str, ...] = ("", "nar_")) -> set[str]:
    """Every tensor name the requested mode is allowed to quantize.

    `streams` selects which of the two per-layer projections are in scope:
    `""` is the autoregressive half (`self_attn` / `mlp`) and `"nar_"` is the
    non-autoregressive half (`nar_self_attn` / `nar_mlp`).
    """
    names: set[str] = set()
    suffixes = _AR_LAYER_MATRIX_SUFFIXES if "" in streams else ()
    nar_suffixes = _NAR_LAYER_MATRIX_SUFFIXES if "nar_" in streams else ()
    if "nar_" in streams:
        names.update(_NAR_GLOBAL_MATRIX_NAMES)
    for layer in range(layers):
        for suffix in suffixes:
            names.add(f"model.layers.{layer}.{suffix}")
        for suffix in nar_suffixes:
            names.add(f"model.layers.{layer}.{suffix}")
    return names


def _read_payload(source, start: int, length: int) -> bytes:
    """Read one tensor payload out of the safetensors file."""
    with source.path.open("rb") as handle:
        handle.seek(source.data_offset + start)
        payload = handle.read(length)
    if len(payload) != length:
        raise ValueError(f"{source.path}: truncated payload at offset {start}")
    return payload


def _bf16_bytes_to_f32(raw: bytes) -> "np.ndarray":
    """Decode a little-endian BF16 payload into float32.

    BF16 is the top half of an IEEE-754 binary32, so widening is a shift rather
    than a conversion.
    """
    values = np.frombuffer(raw, dtype="<u2")
    return (values.astype(np.uint32) << 16).view(np.float32)


def _f32_bytes_to_bf16(values: "np.ndarray") -> bytes:
    """Round float32 to BF16 with round-to-nearest-even.

    ``(x + 0x7fff + lsb) >> 16`` is the standard RNE rounding shift; the explicit
    NaN guard keeps quiet NaNs from being flushed to infinity.
    """
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32).astype(np.uint64)
    nan = (bits & 0x7FFFFFFF) > 0x7F800000
    lsb = (bits >> 16) & 1
    rounded = (bits + 0x7FFF + lsb) >> 16
    rounded = np.where(nan, (bits >> 16) | 0x0040, rounded)
    return rounded.astype("<u2").tobytes()


def _quantize_matrix(values: "np.ndarray", mode: str) -> tuple[int, bytes]:
    """Return ``(ggml_type, payload)`` for one 2-D matrix in ``mode``."""
    if mode == "f32":
        return GGML_F32, np.ascontiguousarray(values, dtype=np.float32).tobytes()
    if mode == "q8_0":
        return GGML_Q8_0, quantize_q8_0(values)
    if mode == "q4_0":
        return GGML_Q4_0, quantize_q4_0(values)
    if mode == "q4_k_m":
        return GGML_Q4K, quantize_q4_k(values)
    if mode == "q6_k":
        return GGML_Q6K, quantize_q6_k(values)
    raise ValueError(f"unknown quantization mode {mode!r}")


def _k_m_type_for(name: str) -> int:
    """Per-tensor type for the `q4_k_m` mixed mode.

    The attention value projections and the FFN down projections carry the most
    visible error, so they get the 6-bit blocks; everything else takes 4-bit.
    This mirrors the Edge0 `--quant q4_k_m` policy.
    """
    if name.endswith("v_proj.weight") or name.endswith("down_proj.weight"):
        return GGML_Q6K
    return GGML_Q4K


def _write_atomic(
    output: Path,
    overwrite: bool,
    metadata: dict[str, object],
    source,
    entries: list[tuple[str, tuple[int, ...], int, int]],
    ggml_type: int,
    *,
    quant: str = "bf16",
    quantizable: set[str] | None = None,
    progress=None,
) -> None:
    if output.exists() and not overwrite:
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    expected_tensors = {}
    block = _MIN_BLOCK_ELEMENTS.get(quant.replace("ar_", ""), 32)
    try:
        writer = GgufWriter(temporary)
        for key, value in metadata.items():
            writer.add_meta(key, value)
        total = len(entries)
        for index, (name, shape, start, end) in enumerate(entries):
            dims = gguf_dims(shape)
            nbytes = end - start
            if quant == "bf16" or quantizable is None or name not in quantizable:
                # Pass-through: the payload already has the on-disk layout, so
                # slice it out of the safetensors file instead of decoding it.
                writer.add_tensor_chunks(
                    name,
                    ggml_type,
                    dims,
                    nbytes,
                    lambda start=start, nbytes=nbytes: _chunks(
                        source.path, source.data_offset + start, nbytes
                    ),
                )
                expected_tensors[name] = (ggml_type, dims)
            else:
                if len(shape) != 2:
                    raise ValueError(f"{name}: only 2-D tensors can be quantized, got {shape}")
                n_in = shape[1]
                if n_in % block != 0:
                    raise ValueError(
                        f"{name}: n_in={n_in} is not a multiple of the {quant} block size "
                        f"{block}; refusing to quantize"
                    )
                values = _bf16_bytes_to_f32(_read_payload(source, start, nbytes))
                if quant == "ar_q8_0":
                    # Same encoder as `q8_0`; only the set of tensors differs,
                    # which is decided by `_quantizable_names`.
                    tensor_type, payload = _quantize_matrix(values, "q8_0")
                elif quant in ("q4_k_m", "ar_q4_k_m"):
                    # The mixed mode picks 6-bit for some tensors and 4-bit for
                    # others, so the single-type encoder cannot be used.
                    # `ar_q4_k_m` is the same encoder with a narrower scope: the
                    # NAR half stays BF16, because the flow-matching solve
                    # amplifies weight noise every step and a 4-bit NAR is noise
                    # by 4 steps. See `QUANT_MODES`.
                    tensor_type = _k_m_type_for(name)
                    payload = (
                        quantize_q6_k(values)
                        if tensor_type == GGML_Q6K
                        else quantize_q4_k(values)
                    )
                else:
                    tensor_type, payload = _quantize_matrix(values, quant)
                del values
                writer.add_tensor(name, tensor_type, dims, payload)
                expected_tensors[name] = (tensor_type, dims)
            if progress is not None and (index + 1) % 8 == 0:
                progress(index + 1, total, name)
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


def convert_main(
    model_dir: Path,
    output: Path,
    overwrite: bool = False,
    quant: str = "bf16",
) -> None:
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
        "yue2.quant.mode": quant,
        **tokenizer_metadata(model_dir / "qwen.tiktoken"),
    }
    layers = int(config["num_hidden_layers"])
    streams = _QUANT_STREAMS[quant]
    quantizable = None if not streams else _quantizable_names(layers, streams)

    started = time.monotonic()

    def report(done: int, total: int, name: str) -> None:
        print(
            f"  [{quant}] {done}/{total} tensors  {time.monotonic() - started:6.1f}s  {name}",
            flush=True,
        )

    _write_atomic(
        output,
        overwrite,
        metadata,
        source,
        entries,
        GGML_BF16,
        quant=quant,
        quantizable=quantizable,
        progress=report if quant != "bf16" else None,
    )


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
    parser.add_argument(
        "--quant",
        choices=QUANT_MODES,
        default="bf16",
        help=(
            "main-model weight format. bf16 is a zero-copy pass-through of the "
            "source payload; the other modes decode BF16 to f32 and re-quantize the "
            "transformer projections (1-D norms, the vocab embeddings and the "
            "position/bridge tables always stay BF16). The VAE is always F32."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    convert_main(args.model_dir, args.main_out, args.overwrite, args.quant)
    convert_vae(args.vae_dir, args.vae_out, args.overwrite)
    for label, path in (("main", args.main_out), ("vae", args.vae_out)):
        _metadata, tensors = read_gguf_directory(path)
        print(f"{label}: {len(tensors)} tensors, {path.stat().st_size} bytes, sha256={_sha256(path)}")


if __name__ == "__main__":
    main()
