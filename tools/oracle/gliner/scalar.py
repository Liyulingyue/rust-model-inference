"""Replace numerical PyTorch kernels with independent single-threaded C F32 kernels."""

from __future__ import annotations

import ctypes
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional


def patch_model(model, trace_prefix=None):
    source = Path(__file__).with_name("scalar.c")
    library = Path(__file__).resolve().parents[3] / ".venv/gliner_scalar"
    library = library.with_suffix(".dylib" if sys.platform == "darwin" else ".so")
    if not library.exists() or library.stat().st_mtime < source.stat().st_mtime:
        flags = ["-dynamiclib"] if sys.platform == "darwin" else ["-shared", "-fPIC"]
        subprocess.run(["cc", "-O2", "-ffp-contract=off", "-fno-vectorize", "-fno-slp-vectorize", *flags, str(source), "-o", str(library)], check=True)
    native = ctypes.CDLL(str(library))
    pointer = ctypes.POINTER(ctypes.c_float)

    def address(array):
        return array.ctypes.data_as(pointer)

    def array(tensor):
        if tensor.dtype != torch.float32 or tensor.device.type != "cpu":
            raise ValueError("scalar GLiNER Oracle requires CPU F32 tensors")
        return np.ascontiguousarray(tensor.detach().numpy())

    def linear(input, weight, bias=None):
        x, w = array(input), array(weight)
        b = array(bias) if bias is not None else None
        rows, width, out = x.size // x.shape[-1], x.shape[-1], w.shape[0]
        result = np.empty((*x.shape[:-1], out), dtype=np.float32)
        native.linear_f32(address(x), address(w), address(b) if b is not None else None, address(result), ctypes.c_size_t(rows), ctypes.c_size_t(width), ctypes.c_size_t(out))
        return torch.from_numpy(result)

    def bmm(left, right):
        a, b = array(left), array(right)
        if a.ndim != 3 or b.ndim != 3 or a.shape[0] != b.shape[0] or a.shape[2] != b.shape[1]:
            raise ValueError("invalid scalar bmm shape")
        result = np.empty((a.shape[0], a.shape[1], b.shape[2]), dtype=np.float32)
        native.bmm_f32(address(a), address(b), address(result), ctypes.c_size_t(a.shape[0]), ctypes.c_size_t(a.shape[1]), ctypes.c_size_t(a.shape[2]), ctypes.c_size_t(b.shape[2]))
        return torch.from_numpy(result)

    def layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
        x = array(input)
        width = normalized_shape[-1] if isinstance(normalized_shape, (list, tuple)) else normalized_shape
        if x.shape[-1] != width or weight is None or bias is None:
            raise ValueError("unsupported scalar layer norm")
        result = np.empty_like(x)
        native.norm_f32(address(x), address(array(weight)), address(array(bias)), address(result), ctypes.c_size_t(x.size // width), ctypes.c_size_t(width), ctypes.c_float(eps))
        return torch.from_numpy(result)

    def softmax(input, dim=None, _stacklevel=3, dtype=None):
        x = array(input)
        if dim not in (-1, x.ndim - 1) or dtype is not None:
            raise ValueError("unsupported scalar softmax")
        result = np.empty_like(x)
        native.softmax_f32(address(x), address(result), ctypes.c_size_t(x.size // x.shape[-1]), ctypes.c_size_t(x.shape[-1]))
        if trace_prefix is not None and x.ndim == 4 and not getattr(softmax, "captured", False):
            Path(f"{trace_prefix}.gliner.scores.f32").write_bytes(x[0, 0].tobytes())
            Path(f"{trace_prefix}.gliner.probabilities.f32").write_bytes(result[0, 0].tobytes())
            softmax.captured = True
        return torch.from_numpy(result)

    def gelu(input):
        x = array(input)
        result = np.empty_like(x)
        native.gelu_f32(address(x), address(result), ctypes.c_size_t(x.size))
        return torch.from_numpy(result)

    functional.linear = linear
    functional.layer_norm = layer_norm
    functional.softmax = softmax
    torch.softmax = softmax
    torch.bmm = bmm
    class ScalarGelu(torch.nn.Module):
        def forward(self, input):
            return gelu(input)

    for layer in model.encoder.encoder.layer:
        layer.intermediate.intermediate_act_fn = ScalarGelu()
