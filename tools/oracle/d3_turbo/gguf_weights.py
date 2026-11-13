"""Independent llama.cpp/gguf dequantization, streamed into the official graph.

The checkpoint remains mmap-backed. Only the active layer is replaced, avoiding
a second full F32 model; no Rust converter or decoder is used by this reference.
"""
import contextlib
import functools

import numpy as np
import torch
from gguf import GGUFReader
from gguf.quants import dequantize

from stream_f32 import _float_tensors


class Weights:
    def __init__(self, path):
        self.reader = GGUFReader(str(path))
        self.tensors = {t.name: t for t in self.reader.tensors}
        field = self.reader.get_field("general.architecture")
        if field is None or field.contents() != "d3_turbo":
            raise ValueError("expected standalone d3_turbo GGUF")

    def value(self, name, indices=None):
        tensor = self.tensors[name]
        data = tensor.data if indices is None else tensor.data[indices]
        values = dequantize(data, tensor.tensor_type)
        shape = tuple(reversed(tensor.shape.tolist()))
        if indices is not None:
            shape = (len(indices), *shape[1:])
        return torch.from_numpy(np.array(values.reshape(shape), dtype=np.float32, copy=True))


def activation_roundtrip(value, kind):
    """The CPU block activation contract; float reductions remain PyTorch's own."""
    if kind == 1:
        # PreparedRows uses the shared F16 kernel's F16 activation contract.
        return value.half().float()
    width = 32 if kind in (2, 3, 8) else 256
    blocks = value.float().reshape(*value.shape[:-1], -1, width)
    maximum = blocks.abs().amax(-1, keepdim=True)
    if width == 32:
        # Scalar / Tensor uses reciprocal multiplication, which changes Q8 rounding boundaries.
        inverse = torch.where(maximum > 0, torch.div(torch.full_like(maximum, 127.), maximum), 0.)
        scale = torch.div(maximum, torch.full_like(maximum, 127.)).half().float()
    else:
        signed = blocks.gather(-1, blocks.abs().argmax(-1, keepdim=True))
        inverse = torch.where(maximum > 0, torch.div(torch.full_like(signed, -127.), signed), 0.)
        scale = torch.where(maximum > 0, torch.div(torch.ones_like(inverse), inverse), 0.)
    return (torch.round(blocks * inverse).clamp(-127, 127) * scale).reshape(value.shape)


def q4_1_min_correction(value, tensor):
    """Q8_1 stores a separately F16-rounded block sum for the Q4_1 minimum."""
    blocks = value.float().reshape(*value.shape[:-1], -1, 32)
    maximum = blocks.abs().amax(-1)
    inverse = torch.where(maximum > 0, torch.div(torch.full_like(maximum, 127.), maximum), 0.)
    integers = torch.round(blocks * inverse.unsqueeze(-1)).clamp(-127, 127).sum(-1)
    scale = torch.div(maximum, torch.full_like(maximum, 127.))
    difference = (integers * scale).half().float() - integers * scale.half().float()
    packed = tensor.data.reshape(-1, value.shape[-1] // 32, 20)
    minimum = np.array(packed[..., 2:4], copy=True).view("<f2").reshape(packed.shape[:2])
    return difference @ torch.from_numpy(minimum.astype(np.float32)).T


@contextlib.contextmanager
def stream_gguf(model, weights, activation_quantization=False):
    saved = []

    def wrap(module, prefix):
        original = module.forward
        saved.append((module, "forward" in vars(module), original))

        @functools.wraps(original)
        def forward(*args, **kwargs):
            replaced = []
            try:
                for name, parameter in module.named_parameters():
                    child_name, _, key = name.rpartition(".")
                    child = module.get_submodule(child_name) if child_name else module
                    value = weights.value(f"language_model.{prefix}.{name}")
                    if value.shape != parameter.shape:
                        raise ValueError(f"GGUF shape mismatch: {prefix}.{name}")
                    replaced.append((child, key, parameter))
                    child._parameters[key] = torch.nn.Parameter(value, requires_grad=False)
                return original(*args, **kwargs)
            finally:
                for child, key, parameter in reversed(replaced):
                    child._parameters[key] = parameter

        module.forward = forward

    def wrap_linear(module, name, kind):
        original = module.forward
        saved.append((module, "forward" in vars(module), original))

        @functools.wraps(original)
        def forward(value):
            output = original(activation_roundtrip(value, kind))
            if kind == 3:
                output = output + q4_1_min_correction(value, weights.tensors[name])
            return output

        module.forward = forward

    original = model.embed_tokens.forward
    saved.append((model.embed_tokens, "forward" in vars(model.embed_tokens), original))
    def embedding(ids):
        shape = (*ids.shape, model.config.hidden_size)
        return weights.value("language_model.embed_tokens.weight", ids.reshape(-1).tolist()).reshape(shape)
    model.embed_tokens.forward = embedding
    try:
        if activation_quantization:
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    kind = int(weights.tensors[f"language_model.{name}.weight"].tensor_type)
                    if kind in (1, 2, 3, 8, 12, 13, 14):
                        wrap_linear(module, f"language_model.{name}.weight", kind)
        for i, layer in enumerate(model.layers):
            wrap(layer, f"layers.{i}")
        wrap(model.norm, "norm")
        with _float_tensors(model.rotary_emb):
            yield
    finally:
        for module, had_override, original in reversed(saved):
            if had_override:
                module.forward = original
            else:
                del module.forward
