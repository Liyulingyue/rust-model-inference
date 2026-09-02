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
import pickle
import struct
import zipfile
from dataclasses import dataclass
from pathlib import Path

ALIGNMENT = 32

GGML_F32 = 0
GGML_F16 = 1
GGML_I64 = 27

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


class GgufWriter:
    """Two-pass GGUF v3 writer: header (metadata + tensor info) then aligned data.

    File layout follows the GGUF spec; tensor data is alignment-padded and
    offsets are patched into the header before the data is written.
    """

    def __init__(self, path: Path):
        self.path = path
        self.metadata: list[tuple[str, object]] = []
        self.tensors: list[tuple[str, int, tuple, bytes]] = []  # name, ggml_type, gguf_dims, raw

    def add_meta(self, key: str, value) -> None:
        self.metadata.append((key, value))

    def add_tensor(self, name: str, ggml_type: int, gguf_dims: tuple, raw: bytes) -> None:
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

    def write(self) -> None:
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
        with open(self.path, "wb") as fh:
            fh.write(header)
            fh.write(b"\x00" * (data_start - len(header)))
            for rel, (_name, _t, _dims, raw) in zip(rel_offsets, self.tensors):
                assert fh.tell() == data_start + rel
                fh.write(raw)
                pad = (ALIGNMENT - (len(raw) % ALIGNMENT)) % ALIGNMENT
                if pad:
                    fh.write(b"\x00" * pad)


def gguf_dims(torch_dims: tuple) -> tuple:
    """GGUF stores dims reversed vs torch (dims[0] = contiguous/last torch dim)."""
    return tuple(reversed(torch_dims))


# --------------------------------------------------------------------------- #
# kaiser-sinc filter (alias-free activations, fixed_filter=True paths)
# --------------------------------------------------------------------------- #


def kaiser_sinc_filter1d(cutoff: float, half_width: float, kernel_size: int) -> list:
    """Port of alias_free_filter.py kaiser_sinc_filter1d; normalizes to sum 1."""
    even = kernel_size % 2 == 0
    half_size = kernel_size // 2
    delta_f = 4 * half_width
    a = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    if a > 50.0:
        beta = 0.1102 * (a - 8.7)
    elif a >= 21.0:
        beta = 0.5842 * (a - 21.0) ** 0.4 + 0.07886 * (a - 21.0)
    else:
        beta = 0.0
    i0_beta = _i0(beta)
    window = []
    for n in range(kernel_size):
        t = 2 * n / (kernel_size - 1) - 1
        window.append(_i0(beta * math.sqrt(max(0.0, 1 - t * t))) / i0_beta)
    if even:
        times = [i + 0.5 for i in range(-half_size, half_size)]
    else:
        times = [i - half_size for i in range(kernel_size)]
    filt = []
    for w, t in zip(window, times):
        arg = 2 * cutoff * t
        sinc = 1.0 if t == 0 else math.sin(math.pi * arg) / (math.pi * arg)
        filt.append(2 * cutoff * w * sinc)
    total = sum(filt)
    return [x / total for x in filt]


def _i0(x: float) -> float:
    """Modified Bessel I0 via series (matches torch.kaiser_window numerics)."""
    if x == 0.0:
        return 1.0
    sum_, term, k = 1.0, 1.0, 0
    while True:
        k += 1
        term *= (x / 2) ** 2 / (k * k)
        sum_ += term
        if term < 1e-18 * sum_ or k > 200:
            break
    return sum_


# --------------------------------------------------------------------------- #
# main conversion
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(description="export dots.tts to GGUF + mmproj")
    parser.add_argument("model_dir", type=str, help="models/dots.tts-base or models/dots.tts.edit")
    parser.add_argument("--variant", default=None, help="base|edit (default: from dir name)")
    parser.add_argument("--out-dir", default=None, help="output directory (default: model dir parent)")
    args = parser.parse_args()

    model_dir = validated_dir(args.model_dir, must_exist=True)
    variant = args.variant or ("edit" if "edit" in model_dir.name else "base")
    out_dir = validated_dir(args.out_dir or str(model_dir.parent), must_exist=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"dots-tts-{variant}"
    llm_path = out_dir / f"{prefix}.gguf"
    mmproj_path = out_dir / f"{prefix}-mmproj.gguf"

    print(f"exporting variant={variant} from {model_dir}")
    core = open_safetensors(model_dir / "model.safetensors")
    speaker = open_safetensors(model_dir / "speaker_encoder.safetensors")
    vocoder = open_safetensors(model_dir / "vocoder.safetensors")
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

    # ---------------- LLM gguf (arch qwen2) ---------------- #
    gguf = GgufWriter(llm_path)
    gguf.add_meta("general.architecture", "qwen2")
    gguf.add_meta("general.name", f"dots.tts-{variant}")
    gguf.add_meta("general.file_type", 1)  # F16
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

    def emit_llm(src_name: str, dst_name: str) -> None:
        t = core.tensor(src_name)
        # norm weights must be F32/BF16 for the engine's load_f32_tensor
        if dst_name.endswith("norm.weight"):
            raw = (
                bf16_to_f32(t.raw)
                if t.dtype == "BF16"
                else (f16_to_f32(t.raw) if t.dtype == "F16" else t.raw)
            )
            gguf.add_tensor(dst_name, GGML_F32, gguf_dims(t.shape), raw)
        else:
            raw = (
                bf16_to_f16(t.raw)
                if t.dtype == "BF16"
                else (t.raw if t.dtype == "F16" else f32_to_f16(t.raw))
            )
            gguf.add_tensor(dst_name, GGML_F16, gguf_dims(t.shape), raw)

    embed = core.tensor("llm.model.embed_tokens.weight")
    raw_embed = bf16_to_f16(embed.raw) if embed.dtype == "BF16" else embed.raw
    gguf.add_tensor("token_embd.weight", GGML_F16, gguf_dims(embed.shape), raw_embed)
    gguf.add_tensor("output.weight", GGML_F16, gguf_dims(embed.shape), raw_embed)  # tied
    emit_llm("llm.model.norm.weight", "output_norm.weight")
    layer_map = {
        "input_layernorm.weight": "attn_norm.weight",
        "post_attention_layernorm.weight": "ffn_norm.weight",
        "self_attn.q_proj.weight": "attn_q.weight",
        "self_attn.k_proj.weight": "attn_k.weight",
        "self_attn.v_proj.weight": "attn_v.weight",
        "self_attn.o_proj.weight": "attn_output.weight",
        "mlp.gate_proj.weight": "ffn_gate.weight",
        "mlp.up_proj.weight": "ffn_up.weight",
        "mlp.down_proj.weight": "ffn_down.weight",
    }
    for layer in range(n_layer):
        for src_key, dst_key in layer_map.items():
            emit_llm(f"llm.model.layers.{layer}.{src_key}", f"blk.{layer}.{dst_key}")
    gguf.write()
    print(f"wrote {llm_path} ({len(gguf.tensors)} tensors)")

    # ---------------- mmproj gguf (arch dotstts) ---------------- #
    gguf = GgufWriter(mmproj_path)
    gguf.add_meta("general.architecture", "dotstts")
    gguf.add_meta("general.name", f"dots.tts-{variant}-mmproj")
    gguf.add_meta("general.file_type", 1)
    gguf.add_meta("dotstts.patch_size", cfg["patch_size"])
    gguf.add_meta("dotstts.latent_dim", cfg["latent_dim"])
    gguf.add_meta("dotstts.hop_size", math.prod(cfg["vocoder"]["downsample_rates"]))
    gguf.add_meta("dotstts.sample_rate", cfg["vocoder"]["sample_rate"])
    gguf.add_meta("dotstts.fm_hidden_size", cfg["DiT"]["hidden_size"])
    gguf.add_meta("dotstts.llm_hidden_size", n_embd)
    gguf.add_meta("dotstts.xvec_dim", cfg.get("campplus_embedding_size", 512))

    gguf.add_tensor("dotstts.latent_stats.mean", GGML_F32, (128,), struct.pack("<128f", *latent_stats["mean"][:128]))
    gguf.add_tensor("dotstts.latent_stats.var", GGML_F32, (128,), struct.pack("<128f", *latent_stats["var"][:128]))

    def emit(source: Safetensors, src_name: str, dst_name: str, ggml_type: int) -> None:
        t = source.tensor(src_name)
        if ggml_type == GGML_F16:
            raw = bf16_to_f16(t.raw) if t.dtype == "BF16" else (t.raw if t.dtype == "F16" else f32_to_f16(t.raw))
        elif ggml_type == GGML_F32:
            if t.dtype != "F32":
                raise ValueError(f"{src_name}: expected F32, got {t.dtype}")
            raw = t.raw
        else:
            raw = t.raw
        gguf.add_tensor(dst_name, ggml_type, gguf_dims(t.shape), raw)

    # heads
    for name in ("hidden_proj", "latent_proj", "coordinate_proj"):
        emit(core, f"{name}.weight", f"dotstts.{name}.weight", GGML_F16)
        emit(core, f"{name}.bias", f"dotstts.{name}.bias", GGML_F16)
    for idx in (0, 1):
        emit(core, f"xvec_proj.{idx}.weight", f"dotstts.xvec_proj.{idx}.weight", GGML_F16)
        emit(core, f"xvec_proj.{idx}.bias", f"dotstts.xvec_proj.{idx}.bias", GGML_F16)
    for idx in (0, 2):
        emit(core, f"eos_proj.{idx}.weight", f"dotstts.eos_proj.{idx}.weight", GGML_F16)
        emit(core, f"eos_proj.{idx}.bias", f"dotstts.eos_proj.{idx}.bias", GGML_F16)

    # patch encoder
    for part in ("ds_proj", "in_proj", "out_proj"):
        emit(core, f"patch_encoder.{part}.weight", f"dotstts.patch_encoder.{part}.weight", GGML_F16)
        emit(core, f"patch_encoder.{part}.bias", f"dotstts.patch_encoder.{part}.bias", GGML_F16)
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
    for layer in range(cfg["PatchEncoder"]["num_layers"]):
        for src_key, dst_key in enc_map.items():
            emit(core, f"patch_encoder.encoder.layers.{layer}.{src_key}",
                 f"dotstts.patch_encoder.encoder.layers.{layer}.{dst_key}", GGML_F16)

    # DiT
    for idx_name in ("input_layer",):
        emit(core, f"velocity_field_predictor.{idx_name}.weight", f"dotstts.dit.{idx_name}.weight", GGML_F16)
        emit(core, f"velocity_field_predictor.{idx_name}.bias", f"dotstts.dit.{idx_name}.bias", GGML_F16)
    for sub in ("mlp.0", "mlp.2"):
        emit(core, f"velocity_field_predictor.time_embedder.{sub}.weight",
             f"dotstts.dit.time_embedder.{sub}.weight", GGML_F16)
        emit(core, f"velocity_field_predictor.time_embedder.{sub}.bias",
             f"dotstts.dit.time_embedder.{sub}.bias", GGML_F16)
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
    for layer in range(cfg["DiT"]["num_layers"]):
        for src_key, dst_key in dit_block_map.items():
            emit(core, f"velocity_field_predictor.blocks.{layer}.{src_key}",
                 f"dotstts.dit.blocks.{layer}.{dst_key}", GGML_F16)
    for sub in ("adaLN_modulation.1", "linear"):
        emit(core, f"velocity_field_predictor.output_layer.{sub}.weight",
             f"dotstts.dit.output_layer.{sub}.weight", GGML_F16)
        emit(core, f"velocity_field_predictor.output_layer.{sub}.bias",
             f"dotstts.dit.output_layer.{sub}.bias", GGML_F16)

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
            g = f32_values(g_t.raw)
            v = f32_values(v_t.raw)
            out_c = g_t.shape[0]
            per_c = len(v) // out_c
            w = []
            for c in range(out_c):
                vv = v[c * per_c:(c + 1) * per_c]
                norm = math.sqrt(sum(x * x for x in vv)) + 1e-12
                w.extend(g[c] * x / norm for x in vv)
            gguf.add_tensor(f"dotstts.vocoder.{base}.weight", GGML_F32, gguf_dims(v_t.shape),
                            struct.pack(f"<{len(w)}f", *w))
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

    # fixed kaiser filters for the AMP-block activations (fixed_filter=True)
    n_resblocks = 6 * len(cfg["vocoder"]["resblock_kernel_sizes"])
    for b in range(n_resblocks):
        for a in range(6):
            for tag in ("up_filter", "down_filter"):
                filt = kaiser_sinc_filter1d(0.25, 0.3, 12)
                gguf.add_tensor(f"dotstts.vocoder.decoder.resblocks.{b}.activations.{a}.{tag}",
                                GGML_F32, (12,), struct.pack("<12f", *filt))

    gguf.write()
    print(f"wrote {mmproj_path} ({len(gguf.tensors)} tensors)")

    core.close()
    speaker.close()
    vocoder.close()


if __name__ == "__main__":
    main()