"""Shared GGUF / safetensors utilities for tool converters.

This module hosts the cross-model building blocks that previously lived
inside ``tools/dots/convert_dots_tts.py``.  Other converters import from
here instead of duplicating GGML constants, dtype conversion, quantisation,
or the GGUF writer/reader.

Conventions:

* All public symbols are re-exported from this module so callers can
  write ``from converter.utils.gguf import GgufWriter, open_safetensors``.
* The module is intentionally pure-stdlib plus ``numpy`` (already used by
  every converter) so it works in the project's existing ``.venv``.
* ``converter/__init__.py`` is empty so ``converter.utils.gguf`` is the
  canonical import path.
"""
from __future__ import annotations

import io
import json
import math
import struct
from array import array
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

GGML_F32 = 0
GGML_F16 = 1
GGML_Q4_0 = 2
GGML_Q8_0 = 8
GGML_I64 = 27
GGML_BF16 = 30

GGUF_ALIGNMENT = 32

_ELEMENT_BYTES = {GGML_F32: 4, GGML_F16: 2, GGML_Q8_0: 0, GGML_Q4_0: 0, GGML_I64: 8, GGML_BF16: 2}


def validated_dir(raw: str, *, must_exist: bool) -> Path:
    path = Path(raw).expanduser()
    if must_exist and not path.is_dir():
        raise FileNotFoundError(path)
    return path


@dataclass
class Tensor:
    name: str
    dtype: str
    shape: tuple[int, ...]
    raw: bytes


@dataclass
class Safetensors:
    path: Path
    header: dict[str, dict]
    data_offset: int
    file_size: int

    def get(self, name: str) -> Tensor:
        info = self.header.get(name)
        if info is None:
            raise KeyError(f"{self.path}: missing tensor {name}")
        dtype = info["dtype"]
        shape = tuple(info["shape"])
        start, end = info["data_offsets"]
        with self.path.open("rb") as source:
            source.seek(self.data_offset + start)
            raw = source.read(end - start)
        return Tensor(name=name, dtype=dtype, shape=shape, raw=raw)


def open_safetensors(path: Path) -> Safetensors:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"invalid safetensors path: {path}")
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
        try:
            header = json.loads(header_raw)
        except ValueError as exc:
            raise ValueError(f"{path}: invalid JSON header") from exc
    if not isinstance(header, dict):
        raise ValueError(f"{path}: safetensors header must be object")
    return Safetensors(path=path, header=header, data_offset=8 + header_len, file_size=file_size)


def _require_tensor(tensor: Tensor, dtype: str, shape: tuple[int, ...]) -> Tensor:
    if tensor.dtype != dtype:
        raise ValueError(f"{tensor.name}: dtype expected {dtype}, got {tensor.dtype}")
    if tensor.shape != shape:
        raise ValueError(f"{tensor.name}: shape expected {shape}, got {tensor.shape}")
    return tensor


def _bf16_to_f32_bits(bits: int) -> int:
    return bits << 16


def _f16_to_f32_bits(bits: int) -> int:
    sign = (bits >> 15) & 0x1
    exponent = (bits >> 10) & 0x1F
    mantissa = bits & 0x3FF
    if exponent == 0:
        if mantissa == 0:
            return sign << 31
        while not (mantissa & 0x400):
            mantissa <<= 1
            exponent -= 1
        exponent += 1
        mantissa &= 0x3FF
    elif exponent == 0x1F:
        return (sign << 31) | 0x7F800000 | (mantissa << 13)
    exponent += 127 - 15
    return (sign << 31) | (exponent << 23) | (mantissa << 13)


def bf16_to_f32(data: bytes) -> bytes:
    if len(data) == 0:
        return b""
    if len(data) % 2 != 0:
        raise ValueError(f"bf16_to_f32: data length {len(data)} is not a multiple of 2")
    arr = np.frombuffer(data, dtype="<u2")
    return (arr.astype(np.uint32) << np.uint32(16)).view(np.float32).tobytes()


def f16_to_f32(data: bytes) -> bytes:
    count = len(data) // 2
    out = array("I")
    for i in range(count):
        bits = struct.unpack_from("<H", data, i * 2)[0]
        out.append(_f16_to_f32_bits(bits))
    return out.tobytes()


def bf16_to_f16(data: bytes) -> bytes:
    if len(data) == 0:
        return b""
    arr = np.frombuffer(data, dtype="<u2")
    f32 = (arr.astype(np.uint32) << np.uint32(16)).view(np.float32)
    return f32.astype("<f2").tobytes()


def f32_to_bf16(data: bytes) -> bytes:
    """Round-to-nearest-even F32 -> BF16."""
    arr = np.frombuffer(data, dtype="<f4")
    f32_u32 = arr.view(np.uint32).copy()
    # add 0x7FFF + ((mantissa >> 16) & 1) for round-to-nearest-even, then
    # mask off the low 16 mantissa bits
    rounding_bias = np.uint32(0x00007FFF) + ((f32_u32 >> np.uint32(16)) & np.uint32(1))
    rounded = (f32_u32 + rounding_bias) & np.uint32(0xFFFF0000)
    # preserve NaNs (exponent all-ones, mantissa != 0): just keep the high bits
    is_nan = ((f32_u32 & np.uint32(0x7F800000)) == np.uint32(0x7F800000)) & (
        (f32_u32 & np.uint32(0x007FFFFF)) != np.uint32(0)
    )
    rounded = np.where(is_nan, f32_u32 & np.uint32(0xFFFF0000), rounded)
    out = (rounded >> np.uint32(16)).astype("<u2")
    return out.tobytes()


def f32_to_f16(data: bytes) -> bytes:
    if len(data) == 0:
        return b""
    if len(data) % 4 != 0:
        raise ValueError(f"f32_to_f16: data length {len(data)} not a multiple of 4")
    arr = np.frombuffer(data, dtype="<f4")
    return arr.astype("<f2").tobytes()


def f32_values(data: bytes) -> list[float]:
    count = len(data) // 4
    return list(struct.unpack_from(f"<{count}f", data))


Q8_0_BLOCK = 32
Q8_0_BLOCK_BYTES = 34  # f16 scale + 32 x int8


def quantize_q8_0(values: np.ndarray) -> bytes:
    """GGML Q8_0: per 32-element block, f16 scale = amax/127, int8 payload.

    Rounding is round-half-away-from-zero to match ggml's roundf, and a zero
    block encodes a zero scale.
    """
    flat = np.ascontiguousarray(values, dtype=np.float32).reshape(-1)
    if flat.size % Q8_0_BLOCK:
        raise ValueError(
            f"q8_0 payload {flat.size} elements is not a multiple of block size {Q8_0_BLOCK}"
        )
    blocks = flat.reshape(-1, Q8_0_BLOCK)
    amax = np.max(np.abs(blocks), axis=1)
    scale = (amax / 127.0).astype(np.float16)
    scale_f32 = scale.astype(np.float32)
    safe = np.where(scale_f32 == 0.0, np.float32(1.0), scale_f32)
    scaled = blocks / safe[:, None]
    q = (np.floor(np.abs(scaled) + 0.5) * np.sign(scaled)).clip(-127, 127).astype(np.int8)
    out = np.empty((blocks.shape[0], Q8_0_BLOCK_BYTES), dtype=np.uint8)
    out[:, 0:2] = scale.view(np.uint8).reshape(-1, 2)
    out[:, 2:] = q.view(np.uint8).reshape(-1, Q8_0_BLOCK)
    return out.tobytes()


def quantize_q4_0(values: np.ndarray) -> bytes:
    """GGML Q4_0: per 32-element block, f16 scale = amax/7, 4-bit packed payload.

    Layout per block (18 bytes):
      * f16 scale (amax / 7, 0 if block is all-zero)
      * 16 bytes of int4 nibbles (low nibble = element 2*i, high nibble = element 2*i+1)
        with bias +8 so each nibble is unsigned in [0, 15].
    """
    Q4_BLOCK = 32
    Q4_BLOCK_BYTES = 18  # f16 scale + 16 nibble-bytes
    flat = np.ascontiguousarray(values, dtype=np.float32).reshape(-1)
    if flat.size % Q4_BLOCK:
        raise ValueError(
            f"q4_0 payload {flat.size} elements is not a multiple of block size {Q4_BLOCK}"
        )
    blocks = flat.reshape(-1, Q4_BLOCK)
    amax = np.max(np.abs(blocks), axis=1)
    scale = (amax / 7.0).astype(np.float16)
    scale_f32 = scale.astype(np.float32)
    safe = np.where(scale_f32 == 0.0, np.float32(1.0), scale_f32)
    scaled = blocks / safe[:, None]
    q = (np.floor(np.abs(scaled) + 0.5) * np.sign(scaled)).clip(-8.0, 7.0)
    q_int = (q + 8.0).astype(np.uint8)  # unsigned in [0, 15]
    # Pack low nibble first: low = element 2*i, high = element 2*i+1
    low = q_int[:, 0::2]
    high = q_int[:, 1::2]
    packed = (high << 4) | low
    out = np.empty((blocks.shape[0], Q4_BLOCK_BYTES), dtype=np.uint8)
    out[:, 0:2] = scale.view(np.uint8).reshape(-1, 2)
    out[:, 2:] = packed
    return out.tobytes()


def bf16_bytes_to_q8_0(raw_bf16: bytes) -> bytes:
    words = np.frombuffer(raw_bf16, dtype="<u2")
    f32 = np.empty(words.size, dtype=np.float32)
    for i, bits in enumerate(words):
        f32[i] = struct.unpack("<f", struct.pack("<I", bits << 16))[0]
    return quantize_q8_0(f32)


def _gguf_str(value: str) -> bytes:
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _gguf_meta_value(value) -> bytes:
    if isinstance(value, str):
        return struct.pack("<I", 8) + _gguf_str(value)
    if isinstance(value, bool):
        return struct.pack("<I", 27) + struct.pack("<I", int(value))
    if isinstance(value, int):
        return struct.pack("<I", 27) + struct.pack("<q", value)
    if isinstance(value, float):
        return struct.pack("<I", 6) + struct.pack("<d", value)
    if isinstance(value, list):
        payload = b"".join(_gguf_meta_value(v) for v in value)
        return struct.pack("<I", 9) + struct.pack("<Q", len(value)) + payload
    raise TypeError(f"unsupported metadata value: {value!r}")


def _gguf_array(values: list) -> bytes:
    payload = b"".join(_gguf_meta_value(v) for v in values)
    return struct.pack("<Q", len(values)) + payload


def _tensor_nbytes(ggml_type: int, dims: tuple[int, ...]) -> int:
    if ggml_type == GGML_Q8_0:
        n = math.prod(dims)
        if n % Q8_0_BLOCK != 0:
            raise ValueError(f"Q8_0 tensor with {n} elements not divisible by {Q8_0_BLOCK}")
        return (n // Q8_0_BLOCK) * Q8_0_BLOCK_BYTES
    if ggml_type == GGML_Q4_0:
        n = math.prod(dims)
        if n % 32 != 0:
            raise ValueError(f"Q4_0 tensor with {n} elements not divisible by 32")
        return (n // 32) * 18
    element_bytes = _ELEMENT_BYTES.get(ggml_type)
    if element_bytes is None:
        raise ValueError(f"unsupported ggml type {ggml_type}")
    return math.prod(dims) * element_bytes


@dataclass
class TensorPayload:
    nbytes: int
    chunks: Iterable[bytes]


class GgufWriter:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.metadata: list[tuple[str, object]] = []
        self.tensors: list[tuple[str, int, tuple[int, ...], TensorPayload | bytes]] = []

    def add_meta(self, key: str, value: object) -> None:
        self.metadata.append((key, value))

    def add_tensor_chunks(
        self,
        name: str,
        ggml_type: int,
        gguf_dims: tuple,
        nbytes: int,
        chunks: Callable[[], Iterable[bytes]],
    ) -> None:
        expected = _tensor_nbytes(ggml_type, gguf_dims)
        if nbytes != expected:
            raise ValueError(f"{name}: payload {nbytes} != expected {expected}")
        self.tensors.append((name, ggml_type, gguf_dims, TensorPayload(nbytes, chunks())))

    def add_tensor(self, name: str, ggml_type: int, gguf_dims: tuple, raw: bytes) -> None:
        expected = _tensor_nbytes(ggml_type, gguf_dims)
        if len(raw) != expected:
            raise ValueError(f"{name}: payload {len(raw)} != expected {expected}")
        self.tensors.append((name, ggml_type, gguf_dims, raw))

    def write(self) -> None:
        # The GGUF dialect used by this engine stores per-tensor offsets (not
        # byte sizes) in the tensor directory and aligns each tensor's data
        # block to ``GGUF_ALIGNMENT`` bytes.  The reader derives the data
        # region as the aligned offset right after the header.
        placeholder = io.BytesIO()
        placeholder.write(b"GGUF")
        placeholder.write(struct.pack("<I", 3))
        placeholder.write(struct.pack("<Q", len(self.tensors)))
        placeholder.write(struct.pack("<Q", len(self.metadata)))
        for key, value in self.metadata:
            placeholder.write(_gguf_str(key))
            placeholder.write(_gguf_meta_value(value))
        for name, ggml_type, dims, _payload in self.tensors:
            placeholder.write(_gguf_str(name))
            placeholder.write(struct.pack("<I", len(dims)))
            for d in dims:
                placeholder.write(struct.pack("<Q", d))
            placeholder.write(struct.pack("<I", ggml_type))
            placeholder.write(struct.pack("<Q", 0))
        header_template_len = placeholder.tell()
        data_start = (header_template_len + GGUF_ALIGNMENT - 1) // GGUF_ALIGNMENT * GGUF_ALIGNMENT
        rel_offsets: list[int] = []
        pos = data_start
        for _name, _t, _dims, payload in self.tensors:
            rel_offsets.append(pos - data_start)
            nbytes = payload.nbytes if isinstance(payload, TensorPayload) else len(payload)
            pos += (nbytes + GGUF_ALIGNMENT - 1) // GGUF_ALIGNMENT * GGUF_ALIGNMENT
        with self.path.open("wb") as sink:
            sink.write(b"GGUF")
            sink.write(struct.pack("<I", 3))
            sink.write(struct.pack("<Q", len(self.tensors)))
            sink.write(struct.pack("<Q", len(self.metadata)))
            for key, value in self.metadata:
                sink.write(_gguf_str(key))
                sink.write(_gguf_meta_value(value))
            for (name, ggml_type, dims, _payload), offset in zip(self.tensors, rel_offsets):
                sink.write(_gguf_str(name))
                sink.write(struct.pack("<I", len(dims)))
                for d in dims:
                    sink.write(struct.pack("<Q", d))
                sink.write(struct.pack("<I", ggml_type))
                sink.write(struct.pack("<Q", offset))
            if sink.tell() != header_template_len:
                raise AssertionError("GGUF header size mismatch")
            if header_template_len < data_start:
                sink.write(b"\x00" * (data_start - header_template_len))
            for (name, _t, _dims, payload), rel in zip(self.tensors, rel_offsets):
                if sink.tell() != data_start + rel:
                    raise AssertionError(f"{name}: alignment drift")
                if isinstance(payload, TensorPayload):
                    written = 0
                    for chunk in payload.chunks:
                        sink.write(chunk)
                        written += len(chunk)
                    expected = payload.nbytes
                else:
                    sink.write(payload)
                    written = len(payload)
                    expected = len(payload)
                if written != expected:
                    raise ValueError(f"{name}: streamed {written} != {expected}")
                pad = (GGUF_ALIGNMENT - (expected % GGUF_ALIGNMENT)) % GGUF_ALIGNMENT
                if pad:
                    sink.write(b"\x00" * pad)


def gguf_dims(torch_dims: tuple) -> tuple:
    """GGUF stores dims reversed vs torch (dims[0] = contiguous/last torch dim)."""
    if not torch_dims:
        return (1,)
    out = []
    for d in torch_dims:
        if not isinstance(d, int) or d <= 0:
            raise ValueError(f"invalid dim {d!r}")
        out.append(int(d))
    return tuple(reversed(out))


def _read_gguf(path: Path) -> tuple[dict[str, object], dict[str, tuple[int, tuple[int, ...], int, int]]]:
    with path.open("rb") as source:
        magic = source.read(4)
        if magic != b"GGUF":
            raise ValueError(f"{path}: not a GGUF file")
        version = struct.unpack("<I", source.read(4))[0]
        if version != 3:
            raise ValueError(f"{path}: unsupported GGUF version {version}")
        n_tensors = struct.unpack("<Q", source.read(8))[0]
        n_metadata = struct.unpack("<Q", source.read(8))[0]
        metadata: dict[str, object] = {}
        for _ in range(n_metadata):
            key_len = struct.unpack("<Q", source.read(8))[0]
            key = source.read(key_len).decode("utf-8")
            metadata[key] = _read_meta(source)
        tensors: dict[str, tuple[int, tuple[int, ...], int, int]] = {}
        for _ in range(n_tensors):
            name_len = struct.unpack("<Q", source.read(8))[0]
            name = source.read(name_len).decode("utf-8")
            n_dims = struct.unpack("<I", source.read(4))[0]
            dims = tuple(struct.unpack(f"<{n_dims}Q", source.read(8 * n_dims)))
            ggml_type = struct.unpack("<I", source.read(4))[0]
            nbytes = struct.unpack("<Q", source.read(8))[0]
            start = source.tell()
            source.seek(start + nbytes)
            tensors[name] = (ggml_type, dims, start, start + nbytes)
    return metadata, tensors


def _read_meta(source: io.BufferedReader) -> object:
    tag = struct.unpack("<I", source.read(4))[0]
    if tag == 8:
        length = struct.unpack("<Q", source.read(8))[0]
        return source.read(length).decode("utf-8")
    if tag == 6:
        return struct.unpack("<d", source.read(8))[0]
    if tag == 27:
        return struct.unpack("<q", source.read(8))[0]
    if tag == 9:
        length = struct.unpack("<Q", source.read(8))[0]
        return [_read_meta(source) for _ in range(length)]
    raise ValueError(f"unsupported metadata tag {tag}")


def read_gguf_directory(path: Path) -> tuple[dict[str, object], dict[str, tuple[int, tuple[int, ...], int]]]:
    metadata, tensors = _read_gguf(path)
    return metadata, {name: (t[0], t[1], t[2]) for name, t in tensors.items()}


def read_gguf_tensor_bytes(path: Path, name: str) -> bytes:
    metadata, tensors = _read_gguf(path)
    if name not in tensors:
        raise KeyError(f"{path}: missing tensor {name}")
    _, _, start, end = tensors[name]
    with path.open("rb") as source:
        source.seek(start)
        return source.read(end - start)


def emit_tensor(gguf: GgufWriter, name: str, tensor: Tensor, quant: str) -> None:
    if quant == "BF16":
        ggml_type = GGML_BF16
        raw = tensor.raw
    elif quant == "F32":
        ggml_type = GGML_F32
        raw = tensor.raw
    elif quant == "Q8_0":
        ggml_type = GGML_Q8_0
        if tensor.dtype == "BF16":
            raw = bf16_bytes_to_q8_0(tensor.raw)
        elif tensor.dtype == "F32":
            arr = np.frombuffer(tensor.raw, dtype="<f4")
            raw = quantize_q8_0(arr)
        else:
            raise ValueError(f"{name}: Q8_0 quant from dtype {tensor.dtype} not supported")
    else:
        raise ValueError(f"unsupported quant {quant!r}")
    gguf.add_tensor(name, ggml_type, gguf_dims(tensor.shape), raw)


def load_latent_stats(pt_path: Path) -> dict:
    import pickle
    with pt_path.open("rb") as source:
        data = pickle.load(source)
    if not isinstance(data, dict):
        raise ValueError(f"{pt_path}: expected dict payload")
    return data


__all__ = [
    "GGML_F32",
    "GGML_F16",
    "GGML_Q4_0",
    "GGML_Q8_0",
    "GGML_I64",
    "GGML_BF16",
    "GGUF_ALIGNMENT",
    "Q8_0_BLOCK",
    "Q8_0_BLOCK_BYTES",
    "Tensor",
    "Safetensors",
    "open_safetensors",
    "validated_dir",
    "bf16_to_f32",
    "f16_to_f32",
    "bf16_to_f16",
    "f32_to_bf16",
    "f32_to_f16",
    "f32_values",
    "quantize_q8_0",
    "quantize_q4_0",
    "bf16_bytes_to_q8_0",
    "GgufWriter",
    "TensorPayload",
    "gguf_dims",
    "emit_tensor",
    "read_gguf_directory",
    "read_gguf_tensor_bytes",
    "load_latent_stats",
]
