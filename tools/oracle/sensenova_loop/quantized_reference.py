"""Load independently dequantized GGUF projections into the official F32 model."""
import hashlib
from pathlib import Path

import gguf
import numpy as np
import torch

from tools.converter.sensenova_loop.convert import LINEAR_PROJECTIONS, tensor_name
from tools.oracle.thx02.oracle import activation_roundtrip


def install_activation_quantization(model, path):
    """Keep official Linear math, with the independently defined native input QDQ."""
    reader = gguf.GGUFReader(path)
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    kinds = (2, 3, 8, 12, 13, 14)
    expected = {name for name, tensor in tensors.items()
                if int(tensor.tensor_type) in kinds and name.endswith(LINEAR_PROJECTIONS)
                and name != "token_embd.weight"}
    description = []

    def wrap(module, tensor):
        original = module.forward
        kind = int(tensor.tensor_type)

        def forward(input):
            values = input.detach().float().cpu().numpy()
            quantized = activation_roundtrip(values, 8 if kind == 3 else kind)
            output = original(torch.from_numpy(quantized).to(input.device))
            if kind == 3:
                # Q8_1 rounds the block sum separately from its Q8_0 scale.
                blocks = values.reshape(*values.shape[:-1], -1, 32)
                maximum = np.max(np.abs(blocks), axis=-1)
                inverse = np.divide(np.float32(127), maximum, out=np.zeros_like(maximum), where=maximum != 0)
                codes = np.rint(blocks * inverse[..., None]).clip(-127, 127).sum(-1, dtype=np.float32)
                scale = maximum / np.float32(127)
                correction = (codes * scale).astype(np.float16).astype(np.float32) - codes * scale.astype(np.float16).astype(np.float32)
                packed = tensor.data.reshape(-1, values.shape[-1] // 32, 20)
                minimum = np.ascontiguousarray(packed[..., 2:4]).view("<f2").reshape(packed.shape[:2]).astype(np.float32)
                output = output + (torch.from_numpy(correction).to(output.device)
                                   @ torch.from_numpy(minimum).to(output.device).T)
            return output

        module.forward = forward

    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        try:
            target = tensor_name(name + ".weight")
        except ValueError:
            continue
        if target not in expected:
            continue
        tensor = tensors[target]
        wrap(module, tensor)
        description.append({"module": name, "tensor": target, "weight_type": int(tensor.tensor_type),
                            "activation": "q8_0" if int(tensor.tensor_type) in (2, 3, 8) else "q8_k"})
    missing = expected - {item["tensor"] for item in description}
    if missing:
        raise ValueError(f"unmapped activation projections: {sorted(missing)}")
    return description


def host_device_map(device):
    # An ancestor GPU entry makes Accelerate move CPU children there during initialization.
    return {
        "model.visual": "cpu",
        "model.language_model.embed_tokens": "cpu",
        "model.language_model.layers": device,
        "model.language_model.norm": device,
        "model.language_model.rotary_emb": device,
        "lm_head": "cpu",
    }


def materialize_cpu(module):
    from accelerate.hooks import remove_hook_from_module

    remove_hook_from_module(module, recurse=True)
    if any(parameter.is_meta for parameter in module.parameters()):
        raise ValueError("host input modules must have materialized parameters")
    return module.to("cpu")


def replace_projections(model, path: Path):
    reader = gguf.GGUFReader(path)
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    expected = {name for name in tensors if name.endswith(LINEAR_PROJECTIONS)}
    loaded = set()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            try:
                target = tensor_name(name)
            except ValueError:
                continue
            if not any(target.endswith(suffix) for suffix in LINEAR_PROJECTIONS):
                continue
            if parameter.is_meta:
                raise ValueError(f"unmaterialized reference projection {target}")
            tensor = tensors[target]
            if tuple(reversed(tensor.shape)) != tuple(parameter.shape):
                raise ValueError(f"invalid reference projection {target}")
            for row in range(0, parameter.shape[0], 512):
                values = gguf.quants.dequantize(tensor.data[row:row + 512], tensor.tensor_type)
                if not np.isfinite(values).all():
                    raise ValueError(f"nonfinite reference projection {target}")
                parameter[row:row + 512].copy_(torch.from_numpy(values.reshape(-1, parameter.shape[1]).copy()))
            loaded.add(target)
    if loaded != expected:
        raise ValueError(f"unmapped GGUF projections: {sorted(expected - loaded)}")
    if not loaded:
        raise ValueError("no GGUF projections loaded")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"gguf": str(path), "sha256": digest, "projections": len(loaded),
            "reference": "official F32 forward with gguf-py independent dequantization"}
