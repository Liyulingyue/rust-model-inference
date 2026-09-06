#!/usr/bin/env python3
"""Export dots.tts-base / dots.tts.edit (ModelScope dots-studio) to GGUF + mmproj.

Produces, per variant:
  dots-tts-<variant>.gguf          — Qwen2 LLM (arch "qwen2", standard llama.cpp names)
  dots-tts-<variant>-mmproj.gguf   — everything else (arch "dotstts", dotstts.* rules)

Torch-free: safetensors read via mmap, latent_stats.pt via a tiny pickle unstub,
and a GGUF v3 writer (F16 for BF16 sources, F32 kept as F32; convs are stored
with weight-norm already folded, and the fixed kaiser filters are emitted too).
Tensor naming rules follow docs/superpowers/specs/2026-09-01-dots-tts-gguf-rust-design.md.

Usage:
  python3 tools/dots/convert_dots_tts.py models/dots.tts-base [--variant base] [--out-dir DIR]
  python3 tools/dots/convert_dots_tts.py models/dots.tts.edit [--variant edit] [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import pickle
import struct
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

ALIGNMENT = 32

GGML_F32 = 0
GGML_F16 = 1
GGML_I64 = 27
GGML_BF16 = 30

# GGUF metadata value types
_T_UINT8, _T_INT8, _T_UINT16, _T_INT16 = 0, 1, 2, 3
_T_UINT32, _T_INT32, _T_FLOAT32, _T_BOOL = 4, 5, 6, 7
_T_STRING, _T_ARRAY, _T_UINT64, _T_INT64, _T_FLOAT64 = 8, 9, 10, 11, 12


# --------------------------------------------------------------------------- #
# CLI path validation (all file access goes through validated absolute paths)
# --------------------------------------------------------------------------- #


def validated_dir(raw: str, *, must_exist: bool) -> Path:
    """Expand and canonically resolve a CLI path; reject anything that does
    not resolve to a plain directory (or, for outputs, a creatable path)."""
    path = Path(raw).expanduser().resolve()
    if must_exist and not path.is_dir():
        raise SystemExit(f"not a directory: {path}")
    if not must_exist:
        # forbid escaping to an unexpected location via leftover components
        resolved_parent = path.parent.resolve()
        if not resolved_parent.is_dir():
            raise SystemExit(f"output parent is not a directory: {resolved_parent}")
    return path


# --------------------------------------------------------------------------- #
# safetensors reader (no torch dependency)
# --------------------------------------------------------------------------- #

_SAFE_DTYPE_ELEM = {
    "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I8": 1, "U8": 1,
}


@dataclass
class Tensor:
    name: str
    dtype: str
    shape: tuple  # torch order: contiguous dim last
    raw: bytes


class Safetensors:
    """mmap view over a safetensors file."""

    def __init__(self, path: Path):
        with open(path, "rb") as fh:
            header_len = struct.unpack("<Q", fh.read(8))[0]
            self.header = json.loads(fh.read(header_len))
            self.data_offset = 8 + header_len
        self._file = open(path, "rb")

    def tensor(self, name: str) -> Tensor:
        info = self.header.get(name)
        if info is None:
            raise KeyError(f"{self.path_name}: missing tensor {name}")
        dtype = info["dtype"]
        elem = _SAFE_DTYPE_ELEM.get(dtype)
        if elem is None:
            raise ValueError(f"{self.path_name}: unsupported dtype {dtype} for {name}")
        shape = tuple(info["shape"])
        count = 1
        for dim in shape:
            count *= dim
        self._file.seek(self.data_offset + info["data_offsets"][0])
        return Tensor(name, dtype, shape, self._file.read(count * elem))

    @property
    def path_name(self) -> str:
        return getattr(self, "_path_name", "?")

    def close(self):
        self._file.close()


def open_safetensors(path: Path) -> Safetensors:
    sf = Safetensors(path)
    sf._path_name = str(path)
    return sf


def _require_tensor(tensor: Tensor, dtype: str, shape: tuple[int, ...]) -> Tensor:
    if tensor.dtype != dtype:
        raise ValueError(f"{tensor.name}: expected {dtype}, got {tensor.dtype}")
    if tensor.shape != shape:
        raise ValueError(f"{tensor.name}: expected shape {shape}, got {tensor.shape}")
    return tensor


# --------------------------------------------------------------------------- #
# dtype conversions
# --------------------------------------------------------------------------- #


def bf16_to_f32(data: bytes) -> bytes:
    out = bytearray(len(data) * 2)
    for i in range(0, len(data), 2):
        bits = struct.unpack_from("<H", data, i)[0]
        struct.pack_into("<f", out, i * 2, struct.unpack("<f", struct.pack("<I", bits << 16))[0])
    return bytes(out)


def f16_to_f32(data: bytes) -> bytes:
    out = bytearray(len(data) * 2)
    for i in range(0, len(data), 2):
        value = struct.unpack_from("<e", data, i)[0]
        struct.pack_into("<f", out, i * 2, value)
    return bytes(out)


def bf16_to_f16(data: bytes) -> bytes:
    out = bytearray(len(data))
    for i in range(0, len(data), 2):
        bits = struct.unpack_from("<H", data, i)[0]
        value = struct.unpack("<f", struct.pack("<I", bits << 16))[0]
        struct.pack_into("<e", out, i, value)
    return bytes(out)


def f32_to_f16(data: bytes) -> bytes:
    out = bytearray(len(data) // 2)
    for i in range(0, len(data), 4):
        value = struct.unpack_from("<f", data, i)[0]
        struct.pack_into("<e", out, i // 2, value)
    return bytes(out)


def f32_values(data: bytes) -> list:
    return list(struct.unpack(f"<{len(data) // 4}f", data))


def _fold_weight_norm_dim0_f32(
    g_raw: bytes, v_raw: bytes, out_channels: int
) -> tuple[bytes, bytes]:
    """Materialize legacy dim-0 weight norm like pinned Torch 2.8 ARM F32."""
    from array import array

    if out_channels < 1 or len(g_raw) != out_channels * 4:
        raise ValueError("weight_g must contain one F32 value per output channel")
    if len(v_raw) % (out_channels * 4):
        raise ValueError("weight_v rows must have a non-zero uniform width")
    row_width = len(v_raw) // (out_channels * 4)
    if row_width < 1:
        raise ValueError("weight_v rows must have a non-zero uniform width")

    g = array("f")
    g.frombytes(g_raw)
    v = array("f")
    v.frombytes(v_raw)
    norms = array("f", [0.0]) * out_channels
    weight = array("f", [0.0]) * len(v)
    rounded = array("f", [0.0])

    for row in range(out_channels):
        start = row * row_width
        if row_width < 4:
            rounded[0] = v[start] * v[start]
            sum_sq = rounded[0]
            for column in range(1, row_width):
                rounded[0] = v[start + column] * v[start + column]
                rounded[0] = sum_sq + rounded[0]
                sum_sq = rounded[0]
        else:
            lanes = array("f", [0.0, 0.0, 0.0, 0.0])
            full_width = row_width - row_width % 4
            for column in range(full_width):
                lane = column % 4
                rounded[0] = v[start + column] * v[start + column]
                lanes[lane] = lanes[lane] + rounded[0]
            for lane in range(row_width % 4):
                rounded[0] = v[start + full_width + lane] * v[start + full_width + lane]
                lanes[lane] = lanes[lane] + rounded[0]
            rounded[0] = lanes[0] + lanes[2]
            left = rounded[0]
            rounded[0] = lanes[1] + lanes[3]
            rounded[0] = left + rounded[0]
            sum_sq = rounded[0]

        norms[row] = math.sqrt(sum_sq)
        rounded[0] = g[row] / norms[row]
        scale = rounded[0]
        for column in range(row_width):
            weight[start + column] = scale * v[start + column]

    return norms.tobytes(), weight.tobytes()


# --------------------------------------------------------------------------- #
# latent_stats.pt — torch.save(zip) that here holds plain numpy arrays
# --------------------------------------------------------------------------- #


def load_latent_stats(pt_path: Path) -> dict:
    """Parse torch.save(zip) holding a dict of f4 tensors, supporting both the
    plain-numpy pickle (dots.tts-base) and the torch-storage format (edit)."""
    import numpy as np

    with zipfile.ZipFile(pt_path) as archive:
        pkl_name = next(n for n in archive.namelist() if n.endswith("/data.pkl"))
        prefix = pkl_name[: -len("data.pkl")]
        payload = archive.read(pkl_name)

    class _NumpyUnpickler(pickle.Unpickler):
        def find_class(self, module, name):  # noqa: N802
            if module == "_codecs" and name == "encode":
                return lambda text, enc: text.encode(enc)
            if module == "numpy.core.multiarray" and name == "_reconstruct":
                return np.core.multiarray._reconstruct
            if module == "numpy" and name == "ndarray":
                return np.ndarray
            if module == "numpy" and name == "dtype":
                return np.dtype
            return super().find_class(module, name)

    class _StorageMarker:
        """Placeholder for torch.*Storage classes so pickle resolves the GLOBAL
        before the storage is rehydrated via persistent_load."""

    class _TorchUnpickler(pickle.Unpickler):
        def __init__(self, *args, archive=None, prefix="", **kwargs):
            super().__init__(*args, **kwargs)
            self._archive = archive
            self._prefix = prefix

        def persistent_load(self, pid):
            # pid = ("storage", FloatStorage-class, key, location, numel)
            kind = pid[0]
            if kind == "storage":
                key = pid[2]
                numel = pid[4]
                raw = self._archive.read(f"{self._prefix}data/{key}")
                arr = np.frombuffer(raw, dtype="<f4")
                if arr.size < numel:
                    raise ValueError(f"storage {key} truncated: {arr.size} < {numel}")
                return np.ascontiguousarray(arr[:numel])
            raise ValueError(f"unsupported persistent id: {pid!r}")

        def find_class(self, module, name):  # noqa: N802
            if module == "torch" and name.endswith("Storage"):
                return _StorageMarker
            if module == "torch._utils" and name == "_rebuild_tensor_v2":
                def rebuild(storage, offset, size, stride, requires_grad, hooks):
                    del stride, requires_grad, hooks
                    return np.ascontiguousarray(
                        storage[offset : offset + int(np.prod(size))].reshape(size)
                    )
                return rebuild
            return super().find_class(module, name)

    for unpickler_cls in (_NumpyUnpickler, _TorchUnpickler):
        try:
            obj = unpickler_cls(io.BytesIO(payload), archive=zipfile.ZipFile(pt_path), prefix=prefix).load()
            if isinstance(obj, dict):
                return {
                    key: np.asarray(value, dtype=np.float32).reshape(-1).tolist()
                    for key, value in obj.items()
                }
        except Exception:
            continue
    raise ValueError(f"{pt_path}: could not parse latent stats")


# --------------------------------------------------------------------------- #
# GGUF writer
# --------------------------------------------------------------------------- #


def _gguf_str(value: str) -> bytes:
    # this engine's GGUF dialect stores string lengths as u64 (its reader
    # reads lengths with read_u64), matching the repo's existing GGUFs
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _gguf_meta_value(value) -> bytes:
    if isinstance(value, str):
        return struct.pack("<I", _T_STRING) + _gguf_str(value)
    if isinstance(value, bool):
        return struct.pack("<I", _T_BOOL) + struct.pack("<B", 1 if value else 0)
    if isinstance(value, int):
        return struct.pack("<I", _T_UINT64) + struct.pack("<Q", value)
    if isinstance(value, float):
        return struct.pack("<I", _T_FLOAT64) + struct.pack("<d", value)
    if isinstance(value, list):
        return struct.pack("<I", _T_ARRAY) + _gguf_array(value)
    raise TypeError(f"unsupported metadata value {value!r}")


def _gguf_array(values: list) -> bytes:
    # element type is i32, count follows the dialect's u64 lengths
    if all(isinstance(v, str) for v in values):
        items = b"".join(_gguf_str(v) for v in values)
        return struct.pack("<I", _T_STRING) + struct.pack("<Q", len(values)) + items
    if all(isinstance(v, bool) for v in values):
        return struct.pack("<I", _T_BOOL) + struct.pack("<Q", len(values)) + bytes(values)
    if all(isinstance(v, int) for v in values):
        items = b"".join(struct.pack("<I", v) for v in values)
        return struct.pack("<I", _T_UINT32) + struct.pack("<Q", len(values)) + items
    if all(isinstance(v, float) for v in values):
        items = b"".join(struct.pack("<d", v) for v in values)
        return struct.pack("<I", _T_FLOAT64) + struct.pack("<Q", len(values)) + items
    raise ValueError(f"mixed array {values!r}")


def _tensor_nbytes(ggml_type: int, dims: tuple[int, ...]) -> int:
    if any(dim <= 0 for dim in dims):
        raise ValueError(f"invalid tensor dimensions: {dims}")
    size = {GGML_F32: 4, GGML_F16: 2, GGML_BF16: 2, GGML_I64: 8}.get(ggml_type)
    if size is None:
        raise ValueError(f"unsupported GGML type {ggml_type}")
    return math.prod(dims) * size


def _read_gguf(path: Path) -> tuple[dict[str, object], dict[str, tuple[int, tuple[int, ...], int, int]]]:
    file_size = path.stat().st_size
    pos = 0
    fh = path.open("rb")

    def take(fmt: str):
        nonlocal pos
        size = struct.calcsize(fmt)
        raw = fh.read(size)
        if len(raw) != size:
            raise ValueError(f"{path}: truncated GGUF")
        result = struct.unpack(fmt, raw)
        pos += size
        return result[0] if len(result) == 1 else result

    def string() -> str:
        nonlocal pos
        length = take("<Q")
        raw = fh.read(length)
        if len(raw) != length:
            raise ValueError(f"{path}: truncated GGUF string")
        pos += length
        return raw.decode("utf-8")

    def value() -> object:
        value_type = take("<I")
        if value_type == _T_STRING:
            return string()
        if value_type == _T_BOOL:
            return bool(take("<B"))
        if value_type == _T_ARRAY:
            item_type, count = take("<I"), take("<Q")
            return [value_for_type(item_type) for _ in range(count)]
        return value_for_type(value_type)

    def value_for_type(value_type: int) -> object:
        scalar_formats = {
            _T_UINT8: "<B", _T_INT8: "<b", _T_UINT16: "<H", _T_INT16: "<h",
            _T_UINT32: "<I", _T_INT32: "<i", _T_FLOAT32: "<f", _T_UINT64: "<Q",
            _T_INT64: "<q", _T_FLOAT64: "<d",
        }
        if value_type == _T_STRING:
            return string()
        if value_type == _T_BOOL:
            return bool(take("<B"))
        fmt = scalar_formats.get(value_type)
        if fmt is None:
            raise ValueError(f"{path}: unsupported GGUF metadata type {value_type}")
        return take(fmt)

    try:
        if take("<4s") != b"GGUF" or take("<I") != 3:
            raise ValueError(f"{path}: unsupported GGUF header")
        tensor_count, metadata_count = take("<Q"), take("<Q")
        metadata: dict[str, object] = {}
        for _ in range(metadata_count):
            key = string()
            if key in metadata:
                raise ValueError(f"{path}: duplicate metadata key {key}")
            metadata[key] = value()
        directory: dict[str, tuple[int, tuple[int, ...], int, int]] = {}
        pending: list[tuple[str, int, tuple[int, ...], int]] = []
        tensor_names: set[str] = set()
        for _ in range(tensor_count):
            name = string()
            ndim = take("<I")
            dims = tuple(take("<Q") for _ in range(ndim))
            ggml_type, relative_offset = take("<I"), take("<Q")
            if name in tensor_names:
                raise ValueError(f"{path}: duplicate tensor {name}")
            tensor_names.add(name)
            pending.append((name, ggml_type, dims, relative_offset))
        data_start = (pos + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT
        for name, ggml_type, dims, relative_offset in pending:
            length = _tensor_nbytes(ggml_type, dims)
            absolute_offset = data_start + relative_offset
            if absolute_offset < data_start or absolute_offset + length > file_size:
                raise ValueError(f"{path}: tensor {name} lies outside file")
            directory[name] = (ggml_type, dims, length, absolute_offset)
        return metadata, directory
    finally:
        fh.close()


def read_gguf_directory(path: Path) -> tuple[dict[str, object], dict[str, tuple[int, tuple[int, ...], int]]]:
    metadata, directory = _read_gguf(path)
    return metadata, {name: entry[:3] for name, entry in directory.items()}


def read_gguf_tensor_bytes(path: Path, name: str) -> bytes:
    _metadata, directory = _read_gguf(path)
    try:
        _ggml_type, _dims, length, offset = directory[name]
    except KeyError as error:
        raise KeyError(f"{path}: missing tensor {name}") from error
    with path.open("rb") as fh:
        fh.seek(offset)
        return fh.read(length)


class GgufWriter:
    """Two-pass GGUF v3 writer: header (metadata + tensor info) then aligned data.

    File layout follows the GGUF spec; tensor data is alignment-padded and
    offsets are patched into the header before the data is written.
    """

    def __init__(self, path: Path):
        self.path = path
        self.metadata: list[tuple[str, object]] = []
        self.tensors: list[tuple[str, int, tuple, bytes]] = []  # name, ggml_type, gguf_dims, raw
        self._metadata_keys: set[str] = set()
        self._tensor_names: set[str] = set()

    def add_meta(self, key: str, value) -> None:
        if key in self._metadata_keys:
            raise ValueError(f"duplicate metadata key {key}")
        self._metadata_keys.add(key)
        self.metadata.append((key, value))

    def add_tensor(self, name: str, ggml_type: int, gguf_dims: tuple, raw: bytes) -> None:
        if name in self._tensor_names:
            raise ValueError(f"duplicate output tensor {name}")
        expected = _tensor_nbytes(ggml_type, gguf_dims)
        if len(raw) != expected:
            raise ValueError(f"{name}: {len(raw)} bytes, expected {expected}")
        self._tensor_names.add(name)
        self.tensors.append((name, ggml_type, gguf_dims, raw))

    def _build_header(self, offsets: list) -> bytes:
        buf = io.BytesIO()
        buf.write(b"GGUF")
        buf.write(struct.pack("<I", 3))
        buf.write(struct.pack("<Q", len(self.tensors)))
        buf.write(struct.pack("<Q", len(self.metadata)))
        for key, value in self.metadata:
            buf.write(_gguf_str(key))
            buf.write(_gguf_meta_value(value))
        # GGUF spec: metadata section is followed directly by tensor infos
        # (the total tensor count was already written up front).
        for (name, ggml_type, dims, _raw), offset in zip(self.tensors, offsets):
            buf.write(_gguf_str(name))
            buf.write(struct.pack("<I", len(dims)))
            for dim in dims:
                buf.write(struct.pack("<Q", dim))
            buf.write(struct.pack("<I", ggml_type))
            buf.write(struct.pack("<Q", offset))
        # note: this engine's dialect stores tensor offsets relative to the
        # padded data start and has no trailing alignment field; the reader
        # derives the data region as align_up(end-of-header).
        return buf.getvalue()

    def _write_file(self, path: Path) -> None:
        placeholder = self._build_header([0] * len(self.tensors))
        data_start = (len(placeholder) + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT
        # relative offsets the reader adds to its own padded data offset
        rel_offsets = []
        pos = data_start
        for _name, _t, _dims, raw in self.tensors:
            rel_offsets.append(pos - data_start)
            size = (len(raw) + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT
            pos += size
        header = self._build_header(rel_offsets)
        if len(header) != len(placeholder):
            raise AssertionError("header size instability")
        with open(path, "wb") as fh:
            fh.write(header)
            fh.write(b"\x00" * (data_start - len(header)))
            for rel, (_name, _t, _dims, raw) in zip(rel_offsets, self.tensors):
                assert fh.tell() == data_start + rel
                fh.write(raw)
                pad = (ALIGNMENT - (len(raw) % ALIGNMENT)) % ALIGNMENT
                if pad:
                    fh.write(b"\x00" * pad)

    def _validate_readback(self, metadata: dict[str, object], tensors: dict[str, tuple[int, tuple[int, ...], int]]) -> None:
        if metadata != dict(self.metadata):
            raise ValueError("GGUF metadata readback mismatch")
        expected = {
            name: (ggml_type, dims, len(raw))
            for name, ggml_type, dims, raw in self.tensors
        }
        if tensors != expected:
            raise ValueError("GGUF tensor directory readback mismatch")

    def write(self, *, overwrite: bool = False) -> None:
        if self.path.exists() and not overwrite:
            raise FileExistsError(f"output already exists: {self.path}")
        fd, raw_tmp = tempfile.mkstemp(prefix=f".{self.path.name}.", dir=self.path.parent)
        os.close(fd)
        tmp = Path(raw_tmp)
        try:
            self._write_file(tmp)
            metadata, tensors = read_gguf_directory(tmp)
            self._validate_readback(metadata, tensors)
            for name, _ggml_type, _dims, raw in self.tensors:
                if read_gguf_tensor_bytes(tmp, name) != raw:
                    raise ValueError(f"GGUF tensor payload readback mismatch: {name}")
            if overwrite:
                os.replace(tmp, self.path)
            else:
                os.link(tmp, self.path)
                tmp.unlink()
        finally:
            tmp.unlink(missing_ok=True)


def gguf_dims(torch_dims: tuple) -> tuple:
    """GGUF stores dims reversed vs torch (dims[0] = contiguous/last torch dim)."""
    return tuple(reversed(torch_dims))


# --------------------------------------------------------------------------- #
# main conversion
# --------------------------------------------------------------------------- #


def validate_variant(value: str) -> str:
    if value not in {"base", "edit"}:
        raise ValueError(f"unsupported dots.tts variant: {value}")
    return value


def export_model(model_dir: Path, variant: str, out_dir: Path, overwrite: bool) -> tuple[Path, Path]:
    variant = validate_variant(variant)
    model_dir = Path(model_dir).resolve()
    out_dir = Path(out_dir).resolve()
    required = (
        "model.safetensors", "speaker_encoder.safetensors", "vocoder.safetensors",
        "llm_config.json", "config.json", "tokenizer_config.json", "vocab.json",
        "added_tokens.json", "merges.txt", "latent_stats.pt",
    )
    missing = [str(model_dir / name) for name in required if not (model_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing required input paths: {', '.join(missing)}")
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"dots-tts-{variant}"
    llm_path = out_dir / f"{prefix}.gguf"
    mmproj_path = out_dir / f"{prefix}-mmproj.gguf"
    if not overwrite and (llm_path.exists() or mmproj_path.exists()):
        existing = llm_path if llm_path.exists() else mmproj_path
        raise FileExistsError(f"output already exists: {existing}")

    print(f"exporting variant={variant} from {model_dir}")
    core = speaker = vocoder = None
    try:
        core = open_safetensors(model_dir / "model.safetensors")
        speaker = open_safetensors(model_dir / "speaker_encoder.safetensors")
        vocoder = open_safetensors(model_dir / "vocoder.safetensors")
        return _export_open_model(model_dir, variant, out_dir, overwrite, core, speaker, vocoder)
    finally:
        for source in (core, speaker, vocoder):
            if source is not None:
                source.close()


def _export_open_model(
    model_dir: Path,
    variant: str,
    out_dir: Path,
    overwrite: bool,
    core: Safetensors,
    speaker: Safetensors,
    vocoder: Safetensors,
) -> tuple[Path, Path]:
    prefix = f"dots-tts-{variant}"
    llm_path = out_dir / f"{prefix}.gguf"
    mmproj_path = out_dir / f"{prefix}-mmproj.gguf"
    llm_cfg = json.loads((model_dir / "llm_config.json").read_text())
    cfg = json.loads((model_dir / "config.json").read_text())
    tok_cfg = json.loads((model_dir / "tokenizer_config.json").read_text())
    vocab = json.loads((model_dir / "vocab.json").read_text())
    added = json.loads((model_dir / "added_tokens.json").read_text())
    merges = (model_dir / "merges.txt").read_text().splitlines()
    latent_stats = load_latent_stats(model_dir / "latent_stats.pt")

    n_layer = llm_cfg["num_hidden_layers"]
    n_embd = llm_cfg["hidden_size"]
    n_head = llm_cfg["num_attention_heads"]
    n_kv = llm_cfg["num_key_value_heads"]
    if n_embd % n_head:
        raise ValueError(f"hidden_size {n_embd} is not divisible by attention heads {n_head}")
    n_kv_embd = n_kv * (n_embd // n_head)
    # ---------------- LLM gguf (arch qwen2) ---------------- #
    gguf = GgufWriter(llm_path)
    gguf.add_meta("general.architecture", "qwen2")
    gguf.add_meta("general.name", f"dots.tts-{variant}")
    gguf.add_meta("general.file_type", 32)
    gguf.add_meta("general.quantization_version", 2)
    gguf.add_meta("qwen2.block_count", n_layer)
    gguf.add_meta("qwen2.context_length", llm_cfg["max_position_embeddings"])
    gguf.add_meta("qwen2.embedding_length", n_embd)
    gguf.add_meta("qwen2.feed_forward_length", llm_cfg["intermediate_size"])
    gguf.add_meta("qwen2.attention.head_count", n_head)
    gguf.add_meta("qwen2.attention.head_count_kv", llm_cfg["num_key_value_heads"])
    gguf.add_meta("qwen2.attention.layer_norm_rms_epsilon", llm_cfg["rms_norm_eps"])
    gguf.add_meta("qwen2.rope.dimension_count", n_embd // n_head)
    gguf.add_meta("qwen2.rope.freq_base", llm_cfg.get("rope_theta", 1_000_000.0))
    gguf.add_meta("qwen2.vocab_size", len(vocab) + len(added))

    all_tokens: dict[int, str] = {}
    for token, tid in vocab.items():
        all_tokens[int(tid)] = token
    added_entries = [{"id": tid, "content": token} for token, tid in added.items()]
    for entry in sorted(added_entries, key=lambda e: e["id"]):
        if entry["id"] not in all_tokens:
            all_tokens[entry["id"]] = entry["content"]
    n_vocab = max(all_tokens) + 1
    tokens = [all_tokens.get(i, f"<|reserved_{i}|>") for i in range(n_vocab)]
    token_types = [1] * n_vocab  # NORMAL
    for entry in added_entries:
        if entry["id"] < n_vocab:
            token_types[entry["id"]] = 3  # CONTROL
    gguf.add_meta("tokenizer.ggml.model", "gpt2")
    gguf.add_meta("tokenizer.ggml.pre", "qwen2")
    gguf.add_meta("tokenizer.ggml.tokens", tokens)
    gguf.add_meta("tokenizer.ggml.token_type", token_types)
    gguf.add_meta("tokenizer.ggml.merges", merges)
    gguf.add_meta("tokenizer.ggml.bos_token_id", tok_cfg.get("bos_token_id", 151643))
    gguf.add_meta("tokenizer.ggml.eos_token_id", tok_cfg.get("eos_token_id", 151643))
    gguf.add_meta("tokenizer.ggml.add_bos_token", False)
    gguf.add_meta("tokenizer.ggml.add_eos_token", False)

    def emit_llm(src_name: str, dst_name: str, shape: tuple[int, ...]) -> None:
        t = _require_tensor(core.tensor(src_name), "BF16", shape)
        if dst_name.endswith("norm.weight"):
            gguf.add_tensor(dst_name, GGML_F32, gguf_dims(t.shape), bf16_to_f32(t.raw))
        else:
            gguf.add_tensor(dst_name, GGML_BF16, gguf_dims(t.shape), t.raw)

    embed = _require_tensor(core.tensor("llm.model.embed_tokens.weight"), "BF16", (llm_cfg["vocab_size"], n_embd))
    gguf.add_tensor("token_embd.weight", GGML_BF16, gguf_dims(embed.shape), embed.raw)
    gguf.add_tensor("output.weight", GGML_BF16, gguf_dims(embed.shape), embed.raw)  # tied
    emit_llm("llm.model.norm.weight", "output_norm.weight", (n_embd,))
    layer_map = {
        "input_layernorm.weight": "attn_norm.weight",
        "post_attention_layernorm.weight": "ffn_norm.weight",
        "self_attn.q_proj.weight": "attn_q.weight",
        "self_attn.k_proj.weight": "attn_k.weight",
        "self_attn.v_proj.weight": "attn_v.weight",
        "self_attn.q_proj.bias": "attn_q.bias",
        "self_attn.k_proj.bias": "attn_k.bias",
        "self_attn.v_proj.bias": "attn_v.bias",
        "self_attn.o_proj.weight": "attn_output.weight",
        "mlp.gate_proj.weight": "ffn_gate.weight",
        "mlp.up_proj.weight": "ffn_up.weight",
        "mlp.down_proj.weight": "ffn_down.weight",
    }
    llm_shapes = {
        "input_layernorm.weight": (n_embd,),
        "post_attention_layernorm.weight": (n_embd,),
        "self_attn.q_proj.weight": (n_embd, n_embd),
        "self_attn.k_proj.weight": (n_kv_embd, n_embd),
        "self_attn.v_proj.weight": (n_kv_embd, n_embd),
        "self_attn.q_proj.bias": (n_embd,),
        "self_attn.k_proj.bias": (n_kv_embd,),
        "self_attn.v_proj.bias": (n_kv_embd,),
        "self_attn.o_proj.weight": (n_embd, n_embd),
        "mlp.gate_proj.weight": (llm_cfg["intermediate_size"], n_embd),
        "mlp.up_proj.weight": (llm_cfg["intermediate_size"], n_embd),
        "mlp.down_proj.weight": (n_embd, llm_cfg["intermediate_size"]),
    }
    for layer in range(n_layer):
        for src_key, dst_key in layer_map.items():
            emit_llm(f"llm.model.layers.{layer}.{src_key}", f"blk.{layer}.{dst_key}", llm_shapes[src_key])
    gguf.write(overwrite=overwrite)
    print(f"wrote {llm_path} ({len(gguf.tensors)} tensors)")

    # ---------------- mmproj gguf (arch clip) ---------------- #
    gguf = GgufWriter(mmproj_path)
    gguf.add_meta("general.architecture", "clip")
    gguf.add_meta("general.name", f"dots.tts-{variant}-mmproj")
    gguf.add_meta("general.file_type", 32)
    gguf.add_meta("clip.has_vision_encoder", False)
    gguf.add_meta("clip.has_audio_encoder", True)
    gguf.add_meta("clip.has_gen_audio_encoder", True)
    gguf.add_meta("clip.audio.projector_type", "dotstts_spkenc")
    gguf.add_meta("clip.gen.audio.projector_type", "dotstts_gen")
    gguf.add_meta("dotstts.patch_size", cfg["patch_size"])
    gguf.add_meta("dotstts.latent_dim", cfg["latent_dim"])
    gguf.add_meta("dotstts.hop_size", math.prod(cfg["vocoder"]["downsample_rates"]))
    gguf.add_meta("dotstts.sample_rate", cfg["vocoder"]["sample_rate"])
    gguf.add_meta("dotstts.fm_hidden_size", cfg["DiT"]["hidden_size"])
    gguf.add_meta("dotstts.llm_hidden_size", n_embd)
    gguf.add_meta("dotstts.xvec_dim", cfg.get("campplus_embedding_size", 512))
    gguf.add_meta("dotstts.sampling.nfe", 10)
    gguf.add_meta("dotstts.sampling.guidance", 1.2)
    gguf.add_meta("dotstts.sampling.speaker_scale", 1.5)
    gguf.add_meta("dotstts.sampling.eos_threshold", 0.8)

    gguf.add_tensor("dotstts.latent_stats.mean", GGML_F32, (128,), struct.pack("<128f", *latent_stats["mean"][:128]))
    gguf.add_tensor("dotstts.latent_stats.var", GGML_F32, (128,), struct.pack("<128f", *latent_stats["var"][:128]))

    def emit(source: Safetensors, src_name: str, dst_name: str, ggml_type: int, shape: tuple[int, ...] | None = None) -> None:
        expected_dtype = "BF16" if ggml_type == GGML_BF16 else "F32"
        t = source.tensor(src_name)
        if shape is None:
            if t.dtype != expected_dtype:
                raise ValueError(f"{src_name}: expected {expected_dtype}, got {t.dtype}")
        else:
            _require_tensor(t, expected_dtype, shape)
        if ggml_type == GGML_BF16:
            raw = t.raw
        elif ggml_type == GGML_F16:
            raw = bf16_to_f16(t.raw) if t.dtype == "BF16" else (t.raw if t.dtype == "F16" else f32_to_f16(t.raw))
        elif ggml_type == GGML_F32:
            raw = t.raw
        else:
            raw = t.raw
        gguf.add_tensor(dst_name, ggml_type, gguf_dims(t.shape), raw)

    patch_cfg = cfg["PatchEncoder"]
    dit_cfg = cfg["DiT"]
    patch_hidden = patch_cfg["hidden_size"]
    dit_hidden = dit_cfg["hidden_size"]
    latent_dim = cfg["latent_dim"]
    xvec_dim = cfg.get("campplus_embedding_size", 512)

    # heads
    head_shapes = {
        "hidden_proj": ((dit_hidden, n_embd), (dit_hidden,)),
        "latent_proj": ((dit_hidden, latent_dim), (dit_hidden,)),
        "coordinate_proj": ((dit_hidden, latent_dim), (dit_hidden,)),
        "xvec_proj.0": ((dit_hidden, xvec_dim), (dit_hidden,)),
        "xvec_proj.1": ((dit_hidden,), (dit_hidden,)),
        "eos_proj.0": ((n_embd, n_embd), (n_embd,)),
        "eos_proj.2": ((2, n_embd), (2,)),
    }
    for name, (weight_shape, bias_shape) in head_shapes.items():
        emit(core, f"{name}.weight", f"dotstts.{name}.weight", GGML_BF16, weight_shape)
        emit(core, f"{name}.bias", f"dotstts.{name}.bias", GGML_BF16, bias_shape)

    # patch encoder
    patch_shapes = {
        "ds_proj": ((latent_dim, latent_dim, 2), (latent_dim,)),
        "in_proj": ((patch_hidden, latent_dim), (patch_hidden,)),
        "out_proj": ((n_embd, 2 * patch_hidden), (n_embd,)),
    }
    for part, (weight_shape, bias_shape) in patch_shapes.items():
        emit(core, f"patch_encoder.{part}.weight", f"dotstts.patch_encoder.{part}.weight", GGML_BF16, weight_shape)
        emit(core, f"patch_encoder.{part}.bias", f"dotstts.patch_encoder.{part}.bias", GGML_BF16, bias_shape)
    enc_map = {
        "attn_norm.weight": "attn_norm.weight",
        "attn.q_proj.weight": "attn_q.weight",
        "attn.k_proj.weight": "attn_k.weight",
        "attn.v_proj.weight": "attn_v.weight",
        "attn.o_proj.weight": "attn_output.weight",
        "attn.o_proj.bias": "attn_output.bias",
        "ffn_norm.weight": "ffn_norm.weight",
        "ffn.fc1.weight": "ffn_fc1.weight",
        "ffn.fc1.bias": "ffn_fc1.bias",
        "ffn.fc2.weight": "ffn_fc2.weight",
        "ffn.fc2.bias": "ffn_fc2.bias",
    }
    encoder_shapes = {
        "attn_norm.weight": (patch_hidden,),
        "attn.q_proj.weight": (patch_hidden, patch_hidden),
        "attn.k_proj.weight": (patch_hidden, patch_hidden),
        "attn.v_proj.weight": (patch_hidden, patch_hidden),
        "attn.o_proj.weight": (patch_hidden, patch_hidden),
        "attn.o_proj.bias": (patch_hidden,),
        "ffn_norm.weight": (patch_hidden,),
        "ffn.fc1.weight": (patch_cfg["ffn_hidden_size"], patch_hidden),
        "ffn.fc1.bias": (patch_cfg["ffn_hidden_size"],),
        "ffn.fc2.weight": (patch_hidden, patch_cfg["ffn_hidden_size"]),
        "ffn.fc2.bias": (patch_hidden,),
    }
    for layer in range(cfg["PatchEncoder"]["num_layers"]):
        for src_key, dst_key in enc_map.items():
            emit(core, f"patch_encoder.encoder.layers.{layer}.{src_key}",
                 f"dotstts.patch_encoder.encoder.layers.{layer}.{dst_key}", GGML_BF16,
                 encoder_shapes[src_key])

    # DiT
    for idx_name in ("input_layer",):
        emit(core, f"velocity_field_predictor.{idx_name}.weight", f"dotstts.dit.{idx_name}.weight", GGML_BF16, (dit_hidden, dit_hidden))
        emit(core, f"velocity_field_predictor.{idx_name}.bias", f"dotstts.dit.{idx_name}.bias", GGML_BF16, (dit_hidden,))
    for sub in ("mlp.0", "mlp.2"):
        emit(core, f"velocity_field_predictor.time_embedder.{sub}.weight",
             f"dotstts.dit.time_embedder.{sub}.weight", GGML_BF16,
             (dit_hidden, 256 if sub == "mlp.0" else dit_hidden))
        emit(core, f"velocity_field_predictor.time_embedder.{sub}.bias",
             f"dotstts.dit.time_embedder.{sub}.bias", GGML_BF16, (dit_hidden,))
    dit_block_map = {
        "attn.q_proj.weight": "attn.q.weight",
        "attn.k_proj.weight": "attn.k.weight",
        "attn.v_proj.weight": "attn.v.weight",
        "attn.o_proj.weight": "attn.o.weight",
        "attn.o_proj.bias": "attn.o.bias",
        "attn.q_norm.weight": "attn.q_norm.weight",
        "attn.k_norm.weight": "attn.k_norm.weight",
        "ffn.fc1.weight": "ffn.fc1.weight",
        "ffn.fc1.bias": "ffn.fc1.bias",
        "ffn.fc2.weight": "ffn.fc2.weight",
        "ffn.fc2.bias": "ffn.fc2.bias",
        "adaLN_modulation.1.weight": "adaLN_modulation.1.weight",
        "adaLN_modulation.1.bias": "adaLN_modulation.1.bias",
    }
    dit_shapes = {
        "attn.q_proj.weight": (dit_hidden, dit_hidden),
        "attn.k_proj.weight": (dit_hidden, dit_hidden),
        "attn.v_proj.weight": (dit_hidden, dit_hidden),
        "attn.o_proj.weight": (dit_hidden, dit_hidden),
        "attn.o_proj.bias": (dit_hidden,),
        "attn.q_norm.weight": (dit_hidden // dit_cfg["num_heads"],),
        "attn.k_norm.weight": (dit_hidden // dit_cfg["num_heads"],),
        "ffn.fc1.weight": (dit_cfg["ffn_hidden_size"], dit_hidden),
        "ffn.fc1.bias": (dit_cfg["ffn_hidden_size"],),
        "ffn.fc2.weight": (dit_hidden, dit_cfg["ffn_hidden_size"]),
        "ffn.fc2.bias": (dit_hidden,),
        "adaLN_modulation.1.weight": (6 * dit_hidden, dit_hidden),
        "adaLN_modulation.1.bias": (6 * dit_hidden,),
    }
    for layer in range(cfg["DiT"]["num_layers"]):
        for src_key, dst_key in dit_block_map.items():
            emit(core, f"velocity_field_predictor.blocks.{layer}.{src_key}",
                 f"dotstts.dit.blocks.{layer}.{dst_key}", GGML_BF16, dit_shapes[src_key])
    for sub in ("adaLN_modulation.1", "linear"):
        emit(core, f"velocity_field_predictor.output_layer.{sub}.weight",
             f"dotstts.dit.output_layer.{sub}.weight", GGML_BF16,
             (2 * dit_hidden, dit_hidden) if sub == "adaLN_modulation.1" else (latent_dim, dit_hidden))
        emit(core, f"velocity_field_predictor.output_layer.{sub}.bias",
             f"dotstts.dit.output_layer.{sub}.bias", GGML_BF16,
             (2 * dit_hidden,) if sub == "adaLN_modulation.1" else (latent_dim,))

    # speaker, strip the "model." prefix; I64 scalars (BN counters) pass through
    for name in sorted(speaker.header):
        if name.startswith("model."):
            dst = f"dotstts.speaker.{name[len('model.'):]}"
        elif name == "resample.kernel":
            dst = "dotstts.speaker.resample_kernel"
        else:
            raise ValueError(f"unexpected speaker tensor {name}")
        t = speaker.tensor(name)
        if t.dtype == "I64":
            shape = t.shape if t.shape else (1,)
            gguf.add_tensor(dst, GGML_I64, gguf_dims(shape), t.raw)
        else:
            emit(speaker, name, dst, GGML_F32)

    # vocoder (F32) — fold weight_norm pairs into plain weights first
    remaining = {}
    for name in sorted(vocoder.header):
        if name.endswith(".weight_g") or name.endswith(".weight_v"):
            base = name[:-len(".weight_g")] if name.endswith(".weight_g") else name[:-len(".weight_v")]
            remaining.setdefault(base, {})["g" if name.endswith(".weight_g") else "v"] = name
        else:
            remaining[name] = name
    folded_bases = set()
    for base in sorted(remaining):
        if base.endswith(".weight") and base[:-len(".weight")] in remaining:
            continue  # plain conv weight
    for base, pair in sorted(remaining.items()):
        if isinstance(pair, dict):
            g_t = vocoder.tensor(pair["g"])
            v_t = vocoder.tensor(pair["v"])
            out_c = g_t.shape[0]
            _, weight = _fold_weight_norm_dim0_f32(g_t.raw, v_t.raw, out_c)
            gguf.add_tensor(f"dotstts.vocoder.{base}.weight", GGML_F32, gguf_dims(v_t.shape),
                            weight)
            folded_bases.add(base)
    for name, dst in remaining.items():
        if name in folded_bases or dst in folded_bases:
            continue
        if isinstance(dst, str):
            t = vocoder.tensor(dst)
            if t.dtype == "I64":
                shape = t.shape if t.shape else (1,)
                gguf.add_tensor(f"dotstts.vocoder.{dst}", GGML_I64, gguf_dims(shape), t.raw)
            else:
                emit(vocoder, dst, f"dotstts.vocoder.{dst}", GGML_F32)

    gguf.write(overwrite=overwrite)
    print(f"wrote {mmproj_path} ({len(gguf.tensors)} tensors)")

    return llm_path, mmproj_path


def main() -> None:
    parser = argparse.ArgumentParser(description="export dots.tts to GGUF + mmproj")
    parser.add_argument("model_dir", type=str, help="models/dots.tts-base or models/dots.tts.edit")
    parser.add_argument("--variant", default=None, help="base|edit (default: from dir name)")
    parser.add_argument("--out-dir", default=None, help="output directory (default: model dir parent)")
    parser.add_argument("--overwrite", action="store_true", help="replace existing output files")
    args = parser.parse_args()
    model_dir = validated_dir(args.model_dir, must_exist=True)
    variant = args.variant or ("edit" if "edit" in model_dir.name else "base")
    out_dir = validated_dir(args.out_dir or str(model_dir.parent), must_exist=False)
    export_model(model_dir, variant, out_dir, args.overwrite)


if __name__ == "__main__":
    main()
