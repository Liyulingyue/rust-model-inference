"""Independent gguf-py decoding and NumPy activation QDQ for the official Oracle."""
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from gguf import GGMLQuantizationType
from gguf.quants import dequantize

from tools.converter.utils.gguf import _read_gguf
from tools.oracle.thx02.oracle import activation_roundtrip


def load(module, path, architecture):
    path = Path(path)
    meta, tensors = _read_gguf(path)
    if meta.get("general.architecture") != architecture:
        raise ValueError(f"expected {architecture} GGUF")
    # Official codec exposes both old and parametrizations-style weight norm.
    for child in module.modules():
        if torch.nn.utils.parametrize.is_parametrized(child, "weight"):
            torch.nn.utils.parametrize.remove_parametrizations(child, "weight", leave_parametrized=True)
        elif any(hook.__class__.__name__ == "WeightNorm" for hook in child._forward_pre_hooks.values()):
            torch.nn.utils.remove_weight_norm(child)
    parameters = dict(module.named_parameters())
    if set(parameters) != set(tensors):
        raise ValueError(f"official/GGUF parameter inventories differ: {set(parameters) ^ set(tensors)}")
    contracts = {}
    with torch.no_grad(), path.open("rb") as source:
        for name, parameter in parameters.items():
            kind, dims, length, offset = tensors[name]
            if tuple(parameter.shape) != tuple(reversed(dims)):
                raise ValueError(f"official/GGUF shape differs: {name}")
            source.seek(offset)
            raw = source.read(length)
            if len(raw) != length:
                raise ValueError(f"truncated GGUF tensor: {name}")
            values = dequantize(np.frombuffer(raw, dtype=np.uint8), GGMLQuantizationType(kind))
            parameter.copy_(torch.from_numpy(values.reshape(tuple(parameter.shape)).copy()))
            minimum = None
            if kind == 3:
                blocks = np.frombuffer(raw, dtype=np.uint8).reshape(parameter.shape[0], -1, 20)
                minimum = torch.from_numpy(np.ascontiguousarray(blocks[:, :, 2:4]).view("<f2")[:, :, 0].astype(np.float32))
            contracts[parameter.data_ptr()] = (kind, minimum)
    return contracts, meta["arktts.bundle_id"]


@contextmanager
def prepared_linears(contracts):
    """Model input quantization independently, retaining PyTorch's dense accumulation.

    F16 native half accumulators remain a tolerance comparison; this Oracle
    rounds the activation to F16 and accumulates widely. Q4_1 also rounds the
    affine input-sum correction to F16, as the shared Q8_1 contract specifies.
    """
    original = functional.linear

    def linear(value, weight, bias=None):
        kind, minimum = contracts.get(weight.data_ptr(), (0, None))
        correction = None
        if kind == 1:
            value = value.half().float()
        elif kind in (2, 3, 8, 12, 13, 14):
            raw = value.detach().float().numpy()
            quantized = activation_roundtrip(raw, 8 if kind == 3 else kind)
            if kind == 3:
                blocks = raw.reshape(-1, raw.shape[-1] // 32, 32)
                scale = np.max(np.abs(blocks), axis=-1) / np.float32(127)
                inverse = np.divide(np.float32(1), scale, out=np.zeros_like(scale), where=scale != 0)
                codes = np.rint(np.clip(blocks * inverse[..., None], -127, 127))
                affine_sum = (codes.sum(axis=-1, dtype=np.float32) * scale).astype(np.float16).astype(np.float32)
                rounded_sum = quantized.reshape(blocks.shape).sum(axis=-1)
                correction = torch.from_numpy(affine_sum - rounded_sum) @ minimum.T
            value = torch.from_numpy(quantized.copy())
        result = original(value, weight, bias)
        return result if correction is None else result + correction.reshape(result.shape)

    functional.linear = linear
    try:
        yield
    finally:
        functional.linear = original
