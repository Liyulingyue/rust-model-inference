"""Pinned official THX-02 CPU/F32 reference and independent same-GGUF reference.

No backbone math is reimplemented: execution uses the downloaded official
model.py/reference_logits and runtime.py. GGUF block decoding below is a
separate NumPy reader, followed by inverse checkpoint layout transforms.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import platform
import sys
import time
import types
from pathlib import Path

import numpy as np

from tools.converter.utils.gguf import _read_gguf

REVISION = "f17dca262dcd1d115e591424ce9cd012b51fc0db"
MODEL_SHA256 = "2e6585fe35a7c00afde5407b813f60f13cb7dbe0262a7a40e93bff12fb78976a"
READOUT_SHA256 = "e8b24b4b9124c075fe939cb8cbbeb2da5ab15d0228c96b423dbbb9cf050f5dee"
OFFICIAL_SHA256 = {
    "model.py": "3ba76dd06dc2c92d06cd9408f9d4d3aedff3d07de4b3e019df0138f4a1416fa5",
    "protocol.py": "b29d7a50d7214bd42397f8d3ddd5b5706cb8da2b257e175bf89b5a8da455d6a0",
    "runtime.py": "500a546816e233500dda54cb5d695a494e8197337e90ff2cda520806aa089c50",
    "engine.py": "9c2244134b8cc3b528bf06c19ca74881c5a93aace2151407a7b328211df5baca",
    "types.py": "3de3e545d81692346f65c0f4ff9843952696a12f7b357cf6a5d0e8545badff8a",
}
# GGML storage contract: (elements per block, bytes per block).
STORAGE = {0: (1, 4), 1: (1, 2), 30: (1, 2), 8: (32, 34),
           2: (32, 18), 12: (256, 144), 13: (256, 176), 14: (256, 210)}


def dequantize(raw, kind, count):
    if kind not in STORAGE:
        raise ValueError(f"unsupported independent GGUF decoder type {kind}")
    block, nbytes = STORAGE[kind]
    if count < 0 or count % block or len(raw) != count // block * nbytes:
        raise ValueError(f"invalid GGUF payload length/alignment for type {kind}")
    if kind == 0:
        values = np.frombuffer(raw, dtype="<f4")
    elif kind == 1:
        values = np.frombuffer(raw, dtype="<f2").astype(np.float32)
    elif kind == 30:
        values = (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
    else:
        packed = np.frombuffer(raw, dtype=np.uint8).reshape(-1, nbytes)
        if kind in (2, 8):
            scale = np.ascontiguousarray(packed[:, :2]).view("<f2").astype(np.float32)
            if kind == 8:
                codes = packed[:, 2:].view(np.int8).astype(np.float32)
            else:
                qs = packed[:, 2:]
                codes = np.concatenate((qs & 15, qs >> 4), axis=1).astype(np.int16) - 8
            values = scale * codes
        elif kind in (12, 13):
            scale = np.ascontiguousarray(packed[:, :2]).view("<f2").astype(np.float32)
            minimum = np.ascontiguousarray(packed[:, 2:4]).view("<f2").astype(np.float32)
            table = packed[:, 4:16]
            scales = np.concatenate((table[:, :4] & 63,
                                     (table[:, 8:12] & 15) | ((table[:, :4] >> 6) << 4)), axis=1)
            minima = np.concatenate((table[:, 4:8] & 63,
                                     (table[:, 8:12] >> 4) | ((table[:, 4:8] >> 6) << 4)), axis=1)
            qs = packed[:, 16 if kind == 12 else 48:].reshape(-1, 4, 32)
            codes = np.stack((qs & 15, qs >> 4), axis=2).reshape(-1, 8, 32)
            if kind == 13:
                qh = packed[:, 16:48]
                high = (qh[:, None, :] >> np.arange(8)[None, :, None]) & 1
                codes = codes.astype(np.int16) + high.astype(np.int16) * 16
            values = (scale[:, :, None] * scales[:, :, None] * codes
                      - minimum[:, :, None] * minima[:, :, None])
        else:
            ql = packed[:, :128].reshape(-1, 2, 64)
            qh = packed[:, 128:192].reshape(-1, 2, 32)
            codes = np.stack((
                (ql[:, :, :32] & 15) | ((qh & 3) << 4),
                (ql[:, :, 32:] & 15) | (((qh >> 2) & 3) << 4),
                (ql[:, :, :32] >> 4) | (((qh >> 4) & 3) << 4),
                (ql[:, :, 32:] >> 4) | (((qh >> 6) & 3) << 4),
            ), axis=2).reshape(-1, 256).astype(np.int16) - 32
            scales = packed[:, 192:208].view(np.int8).astype(np.float32)
            scale = np.ascontiguousarray(packed[:, 208:210]).view("<f2").astype(np.float32)
            values = scale * np.repeat(scales, 16, axis=1) * codes
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError("non-finite independently decoded GGUF values")
    return values


class GgufWeights:
    def __init__(self, path):
        self.path = Path(path)
        self.metadata, self.tensors = _read_gguf(self.path)
        for name, (kind, dims, length, offset) in self.tensors.items():
            if kind not in STORAGE or not dims or min(dims) <= 0:
                raise ValueError(f"unsupported GGUF tensor {name}: type={kind}, dims={dims}")
            block, size = STORAGE[kind]
            if math.prod(dims) % block or length != math.prod(dims) // block * size:
                raise ValueError(f"invalid GGUF tensor payload length: {name}")
            if offset + length > self.path.stat().st_size:
                raise ValueError(f"truncated GGUF tensor: {name}")

    def shape(self, name):
        if name not in self.tensors:
            raise ValueError(f"missing GGUF tensor: {name}")
        return tuple(reversed(self.tensors[name][1]))

    def rows(self, name, indices):
        shape = self.shape(name)
        kind, _, length, offset = self.tensors[name]
        row_width, row_bytes = math.prod(shape[1:]), length // shape[0]
        if len(indices) == 0 or np.min(indices) < 0 or np.max(indices) >= shape[0]:
            raise ValueError(f"invalid row indices for {name}")
        block, size = STORAGE[kind]
        if row_width % block or row_bytes != row_width // block * size:
            raise ValueError(f"unaligned GGUF row in {name}")
        payload = bytearray()
        with self.path.open("rb") as source:
            for run in np.split(indices, np.flatnonzero(np.diff(indices) != 1) + 1):
                source.seek(offset + int(run[0]) * row_bytes)
                raw = source.read(len(run) * row_bytes)
                if len(raw) != len(run) * row_bytes:
                    raise ValueError(f"truncated GGUF row in {name}")
                payload.extend(raw)
        return dequantize(payload, kind, len(indices) * row_width).reshape(len(indices), *shape[1:])


TEXT_NAMES = {
    "input_layernorm.weight": "attn_norm.weight",
    "post_attention_layernorm.weight": "post_attention_norm.weight",
    "mlp.gate_proj.weight": "ffn_gate.weight", "mlp.up_proj.weight": "ffn_up.weight",
    "mlp.down_proj.weight": "ffn_down.weight",
    "self_attn.q_proj.weight": "attn_q.weight", "self_attn.k_proj.weight": "attn_k.weight",
    "self_attn.v_proj.weight": "attn_v.weight", "self_attn.o_proj.weight": "attn_output.weight",
    "self_attn.q_norm.weight": "attn_q_norm.weight", "self_attn.k_norm.weight": "attn_k_norm.weight",
    "linear_attn.in_proj_qkv.weight": "attn_qkv.weight", "linear_attn.in_proj_z.weight": "attn_gate.weight",
    "linear_attn.in_proj_a.weight": "ssm_alpha.weight", "linear_attn.in_proj_b.weight": "ssm_beta.weight",
    "linear_attn.out_proj.weight": "ssm_out.weight", "linear_attn.norm.weight": "ssm_norm.weight",
    "linear_attn.conv1d.weight": "ssm_conv1d.weight", "linear_attn.A_log": "ssm_a",
    "linear_attn.dt_bias": "ssm_dt.bias",
}
VISION_NAMES = {"norm1": "ln1", "norm2": "ln2", "attn.qkv": "attn_qkv",
                "attn.proj": "attn_out", "mlp.linear_fc1": "ffn_up", "mlp.linear_fc2": "ffn_down"}


def source_mapping(name):
    direct = {
        "language_model.embed_tokens.weight": ("token_embd.weight",),
        "language_model.norm.weight": ("output_norm.weight",), "readout.weight": ("cls.output.weight",),
        "visual.patch_embed.proj.weight": ("v.patch_embd.weight", "v.patch_embd.weight.1"),
        "visual.patch_embed.proj.bias": ("v.patch_embd.bias",),
        "visual.pos_embed.weight": ("v.position_embd.weight",),
    }
    if name in direct:
        return direct[name]
    if name.startswith("language_model.layers."):
        layer, _, suffix = name[len("language_model.layers."):].partition(".")
        if layer.isdigit() and suffix in TEXT_NAMES:
            return (f"blk.{layer}.{TEXT_NAMES[suffix]}",)
    if name.startswith("visual.blocks."):
        layer, _, suffix = name[len("visual.blocks."):].partition(".")
        stem, _, kind = suffix.rpartition(".")
        if layer.isdigit() and stem in VISION_NAMES and kind in ("weight", "bias"):
            return (f"v.blk.{layer}.{VISION_NAMES[stem]}.{kind}",)
    if name.startswith("visual.merger."):
        stem, _, kind = name[len("visual.merger."):].rpartition(".")
        merger = {"norm": "v.post_ln", "linear_fc1": "mm.0", "linear_fc2": "mm.2"}
        if stem in merger and kind in ("weight", "bias"):
            return (f"{merger[stem]}.{kind}",)
    raise ValueError(f"unmapped official source parameter: {name}")


def activation_roundtrip(values, weight_kind):
    """Independent NumPy QDQ for the host's shared quantized matmul inputs.

    Q8_0/Q4_0 use 32-wide Q8 blocks with F16 scales. K types use
    256-wide signed-scale Q8_K blocks with F32 scales. NEON and AVX2
    round ties to even; the scalar Q8_0 path rounds away from zero.
    """
    if weight_kind not in (2, 8, 12, 13, 14):
        raise ValueError(f"no activation quantization for GGUF type {weight_kind}")
    values = np.asarray(values, dtype=np.float32)
    width = 32 if weight_kind in (2, 8) else 256
    if values.ndim == 0 or values.shape[-1] % width or not np.isfinite(values).all():
        raise ValueError(f"activation input must be finite with last dimension divisible by {width}")
    blocks = values.reshape(-1, width)
    if weight_kind in (2, 8):
        maximum = np.max(np.abs(blocks), axis=1, keepdims=True)
        scale = maximum / np.float32(127)
        machine = platform.machine().lower()
        if machine in ("x86_64", "amd64"):
            inverse = np.divide(np.float32(127), maximum, out=np.zeros_like(scale), where=maximum != 0)
        else:
            inverse = np.divide(np.float32(1), scale, out=np.zeros_like(scale), where=scale != 0)
        codes = blocks * inverse
        if machine in ("arm64", "aarch64", "x86_64", "amd64"):
            codes = np.rint(np.clip(codes, -127, 127))
        else:
            codes = np.copysign(np.floor(np.abs(codes) + np.float32(0.5)), codes)
        scale = scale.astype(np.float16).astype(np.float32)
    else:
        maximum = np.take_along_axis(blocks, np.argmax(np.abs(blocks), axis=1)[:, None], axis=1)
        inverse = np.divide(np.float32(-127), maximum, out=np.zeros_like(maximum), where=maximum != 0)
        codes = np.minimum(np.rint(blocks * inverse), np.float32(127))
        scale = np.divide(np.float32(1), inverse, out=np.zeros_like(inverse), where=inverse != 0)
    result = (codes.astype(np.int8).astype(np.float32) * scale).reshape(values.shape)
    if not np.isfinite(result).all():
        raise ValueError("activation quantization produced non-finite values")
    return result


class ActivationQuantization:
    """Hook native Linear inputs only when their GGUF weight is quantized."""

    def __init__(self, backbone, reader, readout=None):
        import torch
        self.hooks, self.description = [], []
        modules = list(backbone.named_modules())
        if readout is not None:
            modules.append(("readout", readout))
        for name, module in modules:
            if not isinstance(module, torch.nn.Linear):
                continue
            target, = source_mapping(name + ".weight")
            kind = reader.tensors[target][0]
            if kind not in (2, 8, 12, 13, 14):
                continue
            reorder = name.endswith(".linear_attn.out_proj")
            self.hooks.append(module.register_forward_pre_hook(self.quantize(kind, reorder), with_kwargs=True))
            self.description.append({"module": name, "tensor": target, "weight_type": kind,
                                     "activation": "q8_0" if kind in (2, 8) else "q8_k",
                                     "runtime_head_order": reorder})

    @staticmethod
    def quantize(kind, reorder):
        def hook(module, args, kwargs):
            import torch
            value = args[0] if args else kwargs["input"]
            if value.dtype != torch.float32 or value.device.type != "cpu":
                raise ValueError("activation-QDQ Oracle requires native CPU/F32 inputs")
            values = value.detach().numpy()
            if reorder:
                if values.shape[-1] != 4096:
                    raise ValueError("unexpected THX-02 SSM output activation width")
                permutation = np.r_[np.arange(0, 32, 2), np.arange(1, 32, 2)]
                heads = values.reshape(*values.shape[:-1], 32, 128)
                values = heads[..., permutation, :].reshape(values.shape)
            result = activation_roundtrip(values, kind)
            if reorder:
                result = result.reshape(*result.shape[:-1], 32, 128)[..., np.argsort(permutation), :].reshape(value.shape)
            quantized = torch.from_numpy(np.ascontiguousarray(result))
            if args:
                return (quantized, *args[1:]), kwargs
            return args, {**kwargs, "input": quantized}
        return hook

    def close(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


def _original_heads(indices, width):
    head, lane = indices // width, indices % width
    return ((head % 2) * 16 + head // 2) * width + lane


def restored_chunks(reader, name, shape, *, rows_per_chunk=64):
    targets = source_mapping(name)
    source_shape = tuple(shape)
    expected = source_shape
    if name == "visual.patch_embed.proj.weight":
        expected = (shape[0], shape[1], shape[3], shape[4])
    elif name.endswith(".linear_attn.conv1d.weight"):
        expected = (shape[0], shape[2])
    for target in targets:
        if reader.shape(target) != expected:
            raise ValueError(f"{target}: shape {reader.shape(target)} != official {expected}")
    suffix = name.partition(".linear_attn.")[2]
    for start in range(0, shape[0], rows_per_chunk):
        end = min(start + rows_per_chunk, shape[0])
        indices = np.arange(start, end)
        if suffix in ("in_proj_qkv.weight", "conv1d.weight"):
            mask = indices >= 4096
            indices[mask] = 4096 + _original_heads(indices[mask] - 4096, 128)
        elif suffix == "in_proj_z.weight":
            indices = _original_heads(indices, 128)
        elif suffix in ("in_proj_a.weight", "in_proj_b.weight", "A_log", "dt_bias"):
            indices = _original_heads(indices, 1)
        if len(targets) == 2:
            value = np.stack([reader.rows(target, indices) for target in targets], axis=2)
        else:
            value = reader.rows(targets[0], indices)
        if suffix == "out_proj.weight":
            value = value[:, _original_heads(np.arange(shape[1]), 128)]
        if name == "language_model.norm.weight" or name.endswith((
                ".input_layernorm.weight", ".post_attention_layernorm.weight",
                ".self_attn.q_norm.weight", ".self_attn.k_norm.weight")):
            value = value - np.float32(1)
        elif suffix == "A_log":
            if not (value < 0).all():
                raise ValueError("SSM A must be negative before inverse -exp")
            value = np.log(-value)
        yield start, end, np.ascontiguousarray(value.reshape(end - start, *shape[1:]), dtype=np.float32)


def checked_sha256(path, expected=None):
    with Path(path).open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    if expected is not None and digest != expected:
        raise ValueError(f"{path}: SHA256 differs from pinned revision: {digest}")
    return digest


def load_reference_modules(model_dir):
    model_dir = Path(model_dir).resolve()
    name = "thx02_reference_" + hashlib.sha256(str(model_dir).encode()).hexdigest()[:12]
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(model_dir / "thx02")]
        package.__package__ = name
        sys.modules[name] = package
    return tuple(importlib.import_module(f"{name}.{part}") for part in ("protocol", "model", "runtime"))


def official_modules(model_dir):
    model_dir = Path(model_dir).resolve()
    provenance = {name: checked_sha256(model_dir / "thx02" / name, digest)
                  for name, digest in OFFICIAL_SHA256.items()}
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["THX02_CPU_DTYPE"] = "float32"
    sys.modules.setdefault("fla", None)
    # The package initializer imports the optional SDK client; the reference needs only these local modules.
    protocol, model, runtime = load_reference_modules(model_dir)
    if Path(model.__file__).resolve() != model_dir / "thx02" / "model.py":
        raise ValueError("loaded a different thx02 package instead of the pinned local reference")
    return protocol, model, runtime, provenance


def read_requests(path):
    raw = sys.stdin.read() if str(path) == "-" else Path(path).read_text()
    request = json.loads(raw)
    single = isinstance(request, dict)
    requests = [request] if single else request
    if not isinstance(requests, list) or not requests or any(not isinstance(r, dict) for r in requests):
        raise ValueError("request must be a JSON object or a nonempty array of request objects")
    return requests, single


def load_gguf_into_decider(decider, path):
    import torch
    reader = GgufWeights(path)
    for key, value in {
        "general.architecture": "thx02", "thx02.component": "decision",
        "thx02.source_revision": REVISION, "thx02.model_sha256": MODEL_SHA256,
        "thx02.attention_mode": "noncausal_full_attention",
        "thx02.recurrent_head_layout": "qwen35-modulo-v1",
    }.items():
        if reader.metadata.get(key) != value:
            raise ValueError(f"same-weight Oracle expected {key}={value!r}")
    parameters = list(decider.backbone.named_parameters()) + [("readout.weight", decider.readout.weight)]
    expected = {target for name, _ in parameters for target in source_mapping(name)}
    if set(reader.tensors) != expected:
        raise ValueError("GGUF tensor set differs from the official backbone + readout")
    with torch.inference_mode():
        for name, parameter in parameters:
            for start, end, values in restored_chunks(reader, name, tuple(parameter.shape)):
                parameter[start:end].copy_(torch.from_numpy(values))
    return {"gguf_sha256": checked_sha256(path), "gguf_format": reader.metadata.get("general.file_type"),
            "gguf_path": str(Path(path).resolve()), "independent_dequantization": True}


def load_gguf_into_causal(model, path):
    import torch
    reader = GgufWeights(path)
    for key, value in {
        "general.architecture": "qwen35", "thx02.component": "thinker",
        "thx02.source_revision": REVISION, "thx02.model_sha256": MODEL_SHA256,
        "thx02.attention_mode": "causal", "thx02.recurrent_head_layout": "qwen35-modulo-v1",
    }.items():
        if reader.metadata.get(key) != value:
            raise ValueError(f"same-weight thinker Oracle expected {key}={value!r}")
    if model.get_input_embeddings().weight.data_ptr() != model.get_output_embeddings().weight.data_ptr():
        raise ValueError("THX-02 thinker requires tied input/output embeddings")
    parameters = list(model.model.named_parameters())
    if set(reader.tensors) != {target for name, _ in parameters for target in source_mapping(name)}:
        raise ValueError("thinker GGUF tensor set differs from the native causal backbone")
    with torch.inference_mode():
        for name, parameter in parameters:
            for start, end, values in restored_chunks(reader, name, tuple(parameter.shape)):
                parameter[start:end].copy_(torch.from_numpy(values))
    return {"gguf_sha256": checked_sha256(path), "gguf_format": reader.metadata.get("general.file_type"),
            "gguf_path": str(Path(path).resolve()), "independent_dequantization": True,
            "adapter_sha256": reader.metadata.get("thx02.thinker_adapter_sha256")}


def generate_thinker(model, inputs, directory, *, steps, stop_token_ids, pad_token_id):
    """Native cached greedy generation; retain every full-vocabulary pre-softmax logit row."""
    import torch
    if steps <= 0 or inputs["input_ids"].shape[0] != 1:
        raise ValueError("thinker Oracle needs positive steps and a single sequence")
    directory.mkdir(parents=True, exist_ok=False)
    starts, seconds, positions = [], [], []

    def start(module, args, kwargs):
        starts.append(time.perf_counter())

    def finish(module, args, output):
        seconds.append(time.perf_counter() - starts[-1])

    def rotary(module, args, kwargs):
        value = kwargs.get("position_ids")
        if value is None and len(args) > 1:
            value = args[1]
        if not positions and value is not None:
            positions.append(value.detach().cpu().numpy())

    hooks = [model.register_forward_pre_hook(start, with_kwargs=True),
             model.register_forward_hook(finish),
             model.model.language_model.rotary_emb.register_forward_pre_hook(rotary, with_kwargs=True)]
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            generated = model.generate(**inputs, do_sample=False, max_new_tokens=steps,
                                       eos_token_id=stop_token_ids, pad_token_id=pad_token_id,
                                       return_dict_in_generate=True, output_logits=True, logits_to_keep=1)
        generation_seconds = time.perf_counter() - started
    finally:
        for hook in hooks:
            hook.remove()
    ids = generated.sequences[0, inputs["input_ids"].shape[1]:].tolist()
    if len(ids) != len(generated.logits) or len(ids) != len(seconds):
        raise ValueError("native generation token/logit/forward counts differ")
    if len(positions) != 1 or positions[0].shape[:2] != (3, 1):
        raise ValueError("native thinker did not expose expected mRoPE positions")
    axes = positions[0][:, 0, :].T
    result = {"input_ids": inputs["input_ids"][0].tolist(), "generated_ids": ids,
              "positions": np.column_stack((axes, np.zeros(len(axes), dtype=int))).tolist(),
              "stop_token_ids": list(stop_token_ids), "steps": [], "seconds": sum(seconds),
              "generation_seconds": generation_seconds, "checkpoint_io_seconds": 0.0}
    for index, (token, logits, elapsed) in enumerate(zip(ids, generated.logits, seconds, strict=True)):
        values = logits[0].detach().float().cpu().numpy()
        if values.ndim != 1 or not np.isfinite(values).all():
            raise ValueError("native thinker returned invalid full-vocabulary logits")
        path = directory / f"step.{index:03}.logits.npy"
        started = time.perf_counter()
        np.save(path, values, allow_pickle=False)
        result["checkpoint_io_seconds"] += time.perf_counter() - started
        result["steps"].append({"token_id": token, "logits_path": str(path.resolve()),
                                "logits_shape": list(values.shape), "seconds": elapsed})
    return result


def causal_model(config):
    import torch
    from transformers.initialization import no_init_weights
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
    config.dtype = torch.float32
    config.text_config.dtype = torch.float32
    config.vision_config.dtype = torch.float32
    # Keep CPU RoPE buffers initialized while skipping random parameter fills.
    with no_init_weights():
        model = Qwen3_5ForConditionalGeneration(config)
    model.tie_weights()
    return model.requires_grad_(False).eval()


def run_thinker(args, requests, protocol, runtime, provenance, max_context):
    import torch
    from transformers import AutoProcessor
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
    model_dir = args.model_dir.resolve()
    provenance["model_sha256"] = checked_sha256(model_dir / "model.safetensors", MODEL_SHA256)
    provenance["adapter_source_sha256"] = checked_sha256(model_dir / "thinker" / "adapter_model.safetensors")
    config = Qwen3_5Config.from_pretrained(str(model_dir / "thinker" / "lm_config"), local_files_only=True)
    config._attn_implementation = "sdpa"
    started = time.perf_counter()
    model = causal_model(config)
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise ValueError("native thinker was not initialized as CPU/F32")
    provenance.update(load_gguf_into_causal(model, args.thinker_gguf))
    if provenance["adapter_source_sha256"] != provenance["adapter_sha256"]:
        raise ValueError("thinker adapter provenance differs from the local pinned source")
    provenance["load_seconds"] = time.perf_counter() - started
    provenance["activation_quantization"] = "float32"
    decision = json.loads((model_dir / "decision_config.json").read_text())
    decision_processor = AutoProcessor.from_pretrained(str(model_dir), local_files_only=True)
    encoder = protocol.Encoder(decision_processor, decision["codes"])
    processor = AutoProcessor.from_pretrained(str(model_dir / "thinker" / "lm_config"), local_files_only=True)
    processor.image_processor.size = {"shortest_edge": protocol.MIN_PIXELS, "longest_edge": decision["max_pixels"]}
    close = processor.tokenizer.encode("</think>", add_special_tokens=False)
    if len(close) != 1:
        raise ValueError("thinker stop string must be one token in the pinned tokenizer")
    stops = list(dict.fromkeys([config.text_config.eos_token_id, processor.tokenizer.eos_token_id, close[0]]))
    results = []
    for source_index, request in enumerate(requests):
        for key, question in request["questions"].items():
            single = {**request, "questions": {key: question}}
            payload = runtime.thinker_payload(encoder, single, args.steps)
            template = {"add_generation_prompt": True, **payload["chat_template_kwargs"]}
            prompt = processor.apply_chat_template(payload["messages"], tokenize=False, **template)
            inputs = processor.apply_chat_template(payload["messages"], tokenize=True, return_dict=True,
                                                   return_tensors="pt", **template)
            if inputs["input_ids"].shape[1] + args.steps > max_context:
                raise ValueError("thinker prompt plus requested steps exceeds max context")
            row = generate_thinker(model, inputs, args.output.with_suffix(".checkpoints") / f"case.{len(results):03}",
                                   steps=args.steps, stop_token_ids=stops, pad_token_id=processor.tokenizer.pad_token_id)
            row.update(request=single, source_request_index=source_index, question_id=key, prompt=prompt,
                       generated_text=processor.tokenizer.decode(row["generated_ids"], skip_special_tokens=False),
                       compute="official-transformers-causal-cpu-f32", weight_source="merged-thinker-gguf")
            if "image_grid_thw" in inputs:
                row["image_grid_thw"] = inputs["image_grid_thw"].tolist()
            results.append(row)
            print(f"thinker job={len(results) - 1} tokens={len(row['input_ids'])} steps={len(row['steps'])} "
                  f"seconds={row['seconds']:.3f}", flush=True)
    return results


def encoded_record(encoder, request, chunk):
    ids = chunk.input_ids.tolist()
    return {"request": request, "prompt": encoder.text(request), "input_ids": ids,
            "markers": chunk.markers.tolist(), "question_ids": list(chunk.keys),
            "option_counts": list(chunk.counts),
            "positions": [[i, i, i, 0] for i in range(len(ids))] if not request.get("images") else None,
            "image_grid_thw": None if chunk.image_grid_thw is None else chunk.image_grid_thw.tolist()}


def planned_record(request, chunks, trace, encoding_start):
    # Encoder.plan may encode and reject an oversized candidate before splitting.
    selected = [trace.by_identity[id(chunk)] for chunk in chunks]
    row = dict(selected[0]) if len(selected) == 1 else {"request": request, "chunks": selected}
    row["planning_encodings"] = trace.encodings[encoding_start:]
    return row


class Trace:
    def __init__(self, decider, protocol, directory):
        self.decider, self.protocol, self.directory = decider, protocol, directory
        directory.mkdir(parents=True, exist_ok=False)
        self.encodings, self.by_identity, self.passes, self.active = [], {}, [], None
        self.phase = "joint"
        self.original_encode = decider.encoder.encode
        self.original_reference = decider.reference_logits
        decider.encoder.encode = self.encode
        decider.reference_logits = self.reference
        text = decider.backbone.language_model
        self.hooks = [text.register_forward_pre_hook(self.positions, with_kwargs=True),
                      text.rotary_emb.register_forward_pre_hook(self.positions, with_kwargs=True),
                      decider.readout.register_forward_hook(self.logits)]
        self.hooks.append(text.embed_tokens.register_forward_hook(self.checkpoint("embedding")))
        for index, layer in enumerate(text.layers):
            self.hooks.append(layer.register_forward_hook(self.checkpoint(f"layer.{index:02}")))
        self.hooks.append(text.norm.register_forward_hook(self.checkpoint("output_norm")))
        visual = decider.backbone.visual
        self.hooks.append(visual.patch_embed.register_forward_hook(self.checkpoint("vision.patch_embed")))
        for index, layer in enumerate(visual.blocks):
            self.hooks.append(layer.register_forward_hook(self.checkpoint(f"vision.layer.{index:02}")))
        self.hooks.append(visual.merger.register_forward_hook(self.checkpoint("vision.merger")))

    def encode(self, request):
        chunk = self.original_encode(request)
        record = encoded_record(self.decider.encoder, request, chunk)
        record["encoding_id"] = len(self.encodings)
        self.encodings.append(record)
        self.by_identity[id(chunk)] = record
        return chunk

    def positions(self, module, args, kwargs):
        positions = kwargs.get("position_ids")
        if positions is None and len(args) > 1:
            positions = args[1]
        if self.active is not None and positions is not None:
            self.active["position_ids"] = positions.detach().cpu().tolist()

    def logits(self, module, args, output):
        if self.active is not None:
            self.active["logits"] = output.detach().float().cpu().tolist()

    def checkpoint(self, name):
        def capture(module, args, output):
            if self.active is None:
                return
            started = time.perf_counter()
            value = output[0] if isinstance(output, tuple) else output
            values = value.detach().float().cpu().numpy()
            path = self.directory / f"pass.{self.active['pass_id']:04}.{name}.npy"
            np.save(path, values, allow_pickle=False)
            self.active["checkpoints"].append({"name": name, "path": str(path.resolve()), "shape": list(values.shape)})
            self.active["checkpoint_io_seconds"] += time.perf_counter() - started
        return capture

    def reference(self, chunks):
        inputs = self.protocol.left_pad(chunks, self.decider.encoder.pad_id)
        record = {"pass_id": len(self.passes), "phase": self.phase,
                  "encoding_ids": [self.by_identity[id(c)]["encoding_id"] for c in chunks],
                  "input_ids": inputs["input_ids"].tolist(), "attention_mask": inputs["attention_mask"].tolist(),
                  "checkpoints": [], "checkpoint_io_seconds": 0.0}
        self.active = record
        started = time.perf_counter()
        try:
            result = self.original_reference(chunks)
            record["seconds"] = time.perf_counter() - started
            record["valid_logits"] = result
            self.passes.append(record)
            print(f"reference pass={record['pass_id']} phase={record['phase']} "
                  f"tokens={[chunk.length for chunk in chunks]} seconds={record['seconds']:.3f}", flush=True)
            positions = np.asarray(record["position_ids"])
            if positions.shape[0] != 3:
                raise ValueError(f"unexpected official mRoPE position shape {positions.shape}")
            for row, chunk in enumerate(chunks):
                axes = positions[:, row, -chunk.length:].T
                self.by_identity[id(chunk)]["positions"] = np.column_stack((axes, np.zeros(chunk.length, dtype=int))).tolist()
            return result
        finally:
            self.active = None

    def close(self):
        for hook in self.hooks:
            hook.remove()
        self.decider.encoder.encode = self.original_encode
        self.decider.reference_logits = self.original_reference


def context_limit(maximum, requested):
    limit = maximum if requested is None else requested
    if not 0 < limit <= maximum:
        raise ValueError(f"max context must be between 1 and backbone maximum {maximum}")
    return limit


def tokenizer_only(model_dir, requests, protocol, max_context=None):
    from transformers import AutoProcessor
    decision = json.loads((model_dir / "decision_config.json").read_text())
    processor = AutoProcessor.from_pretrained(str(model_dir), local_files_only=True)
    processor.tokenizer.padding_side = "left"
    processor.image_processor.size = {"shortest_edge": protocol.MIN_PIXELS, "longest_edge": decision["max_pixels"]}
    if [processor.tokenizer.encode(code, add_special_tokens=False) for code in decision["codes"]] != [
            [token_id] for token_id in decision["token_ids"]]:
        raise ValueError("official processor codebook differs from the pinned token IDs")
    encoder = protocol.Encoder(processor, decision["codes"])
    results = []
    for request in requests:
        started = time.perf_counter()
        chunks = encoder.plan(request, context_limit(262144, max_context))
        records = [encoded_record(encoder, {**request, "questions": {
            key: request["questions"][key] for key in chunk.keys}}, chunk) for chunk in chunks]
        row = dict(records[0]) if len(records) == 1 else {"request": request, "chunks": records}
        row.update(logits=None, answers=None, seconds=time.perf_counter() - started,
                   compute="official-tokenizer-only; model weights not loaded")
        results.append(row)
    return results


def run_runtime(runtime, request, *, skip=False):
    if skip:
        return None, 0.0
    started = time.perf_counter()
    response = runtime.decide(**{key: request[key] for key in (
        "state", "questions", "images", "think", "tta", "return_thoughts") if key in request})
    return response, time.perf_counter() - started


def run(args):
    import torch
    import transformers
    model_dir, output = args.model_dir.resolve(), args.output.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if args.threads <= 0:
        raise ValueError("threads must be positive")
    torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)
    requests, single = read_requests(args.request)
    protocol, model_module, runtime_module, official = official_modules(model_dir)
    provenance = {"revision": REVISION, "official_sources_sha256": official,
                  "torch": torch.__version__, "transformers": transformers.__version__,
                  "numpy": np.__version__, "platform": platform.platform(),
                  "machine": platform.machine(), "threads": args.threads}
    provenance["timing_note"] = args.timing_note or "Isolation not established by this tool."
    maximum = json.loads((model_dir / "config.json").read_text())["backbone_config"]["text_config"]["max_position_embeddings"]
    max_context = context_limit(maximum, args.max_context)
    provenance["max_context"] = max_context
    if args.thinker_gguf:
        results = run_thinker(args, requests, protocol, runtime_module, provenance, max_context)
        single = single and len(results) == 1
    elif args.tokenizer_only:
        results = tokenizer_only(model_dir, requests, protocol, max_context)
    else:
        if any(request.get("think") not in (None, False) for request in requests):
            raise ValueError("this CPU decision Oracle has no thinker server; use decision-only requests")
        provenance["model_sha256"] = checked_sha256(model_dir / "model.safetensors", MODEL_SHA256)
        provenance["readout_sha256"] = checked_sha256(model_dir / "readout.safetensors", READOUT_SHA256)
        started = time.perf_counter()
        decider = model_module.THXDecisionModel(model_dir, device="cpu", fast=False,
                                              cpu_threads=args.threads, max_length=max_context)
        if any(parameter.dtype != torch.float32 for parameter in decider.backbone.parameters()):
            raise ValueError("official CPU Oracle did not load the backbone as F32")
        if args.gguf:
            provenance.update(load_gguf_into_decider(decider, args.gguf))
        activation = ActivationQuantization(decider.backbone, GgufWeights(args.gguf), decider.readout) if args.activation_quantization else None
        provenance["activation_quantization"] = activation.description if activation else "float32"
        provenance["load_seconds"] = time.perf_counter() - started
        trace = Trace(decider, protocol, output.with_suffix(".checkpoints"))
        runtime = runtime_module.THX02(decider)
        results = []
        try:
            for request in requests:
                trace.phase = "joint"
                encoding_start, pass_start = len(trace.encodings), len(trace.passes)
                chunks = decider.encoder.plan(request, decider.max_length)
                valid, full = {}, {}
                seconds = 0.0
                for group in decider._batches(list(enumerate(chunks))):
                    selected = [chunk for _, chunk in group]
                    rows = decider.reference_logits(selected)
                    forward = trace.passes[-1]
                    seconds += forward["seconds"]
                    cursor = 0
                    for chunk, chunk_logits in zip(selected, rows, strict=True):
                        for key, values in zip(chunk.keys, chunk_logits, strict=True):
                            valid[key], full[key] = values, forward["logits"][cursor]
                            cursor += 1
                response = decider._respond(request, valid, chunks, decider.temperature)
                joint_encoding_end = len(trace.encodings)
                row = planned_record(request, chunks, trace, encoding_start)
                row.update(logits=[full[key] for key in request["questions"]], answers=response["answers"],
                           usage=response["usage"], seconds=seconds, joint_seconds=seconds,
                           joint_checkpoint_io_seconds=sum(p["checkpoint_io_seconds"] for p in trace.passes[pass_start:]),
                           threads=args.threads,
                           compute="official-pytorch-cpu-f32", weight_source="gguf" if args.gguf else "original-bf16-widened-to-f32")
                joint_output = trace.directory / f"case.{len(results):03}.joint.json"
                joint_output.write_text(json.dumps({**row, "provenance": provenance}, ensure_ascii=False, allow_nan=False) + "\n")
                print(f"joint case={len(results)} output={joint_output}", flush=True)
                trace.phase = "runtime"
                row["runtime_response"], row["runtime_seconds"] = run_runtime(runtime, request, skip=args.skip_runtime)
                row["forward_passes"] = trace.passes[pass_start:]
                row["runtime_encodings"] = trace.encodings[joint_encoding_end:]
                results.append(row)
        finally:
            trace.close()
            if activation:
                activation.close()
    for result in results:
        result["threads"], result["provenance"] = args.threads, provenance
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = results[0] if single else results
    with output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(f"{output}: {len(results)} request(s), {'tokenizer only' if args.tokenizer_only else 'official CPU F32'}")


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True, help="JSON request object/array, or - for stdin")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gguf", type=Path, help="independently decode this GGUF into official parameters")
    parser.add_argument("--thinker-gguf", type=Path, help="independently decode merged thinker GGUF for native causal generation")
    parser.add_argument("--steps", type=int, default=8, help="maximum greedy thinker generation steps")
    parser.add_argument("--activation-quantization", action="store_true", help="independent Q8_0/Q8_K input QDQ for quantized GGUF Linear weights")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-context", type=int, help="per-sequence planning limit; defaults to backbone maximum")
    parser.add_argument("--timing-note", help="record concurrency/isolation conditions with numerical evidence")
    parser.add_argument("--tokenizer-only", action="store_true", help="validate official protocol/tokenizer without loading model weights")
    parser.add_argument("--skip-runtime", action="store_true", help="run only joint readouts, without repeated runtime/TTA forwards")
    return parser


def main():
    parser = argument_parser()
    args = parser.parse_args()
    if args.gguf and args.thinker_gguf:
        parser.error("select only one of --gguf and --thinker-gguf")
    if args.tokenizer_only and (args.gguf or args.thinker_gguf):
        parser.error("GGUF references require model execution; omit them for --tokenizer-only")
    if args.thinker_gguf and args.steps <= 0:
        parser.error("--steps must be positive")
    if args.activation_quantization and (not args.gguf or args.tokenizer_only):
        parser.error("--activation-quantization requires --gguf model execution")
    run(args)


if __name__ == "__main__":
    main()
