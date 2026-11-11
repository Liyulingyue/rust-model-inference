"""Stream pinned HAL-X/THX-02 weights into a bundled decision or merged thinker GGUF.

Run from the repository root with ``python -m tools.converter.thx02.convert_thx02``.
The thinker is a separate causal qwen35 file; decisions retain explicit thx02
architecture and noncausal full attention metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from tools.converter.utils.gguf import (
    GGML_BF16, GGML_F16, GGML_F32, GGML_Q4_0, GGML_Q4_1, GGML_Q4K, GGML_Q5K,
    GGML_Q6K, GGML_Q8_0, GgufWriter, Safetensors, _tensor_nbytes,
    bf16_to_f32, f32_to_bf16, gguf_dims, model_metadata_value,
    open_safetensors, quantize_q4_0, quantize_q4_1, quantize_q8_0,
)
from tools.converter.utils.kquants import quantize_k
from tools.converter.utils.quantize_gguf import BLOCK_FORMATS

SOURCE_REVISION = "f17dca262dcd1d115e591424ce9cd012b51fc0db"
SOURCE_SHA256 = {
    "model": "2e6585fe35a7c00afde5407b813f60f13cb7dbe0262a7a40e93bff12fb78976a",
    "readout": "e8b24b4b9124c075fe939cb8cbbeb2da5ab15d0228c96b423dbbb9cf050f5dee",
}
CODEBOOK_SHA256 = "933e4842d6e318d24ade7a4e49df566aee3f8b6aa5e12b216c2711fc28be7005"
CHUNK_BYTES = 1 << 20
FORMATS = {"f32": (1, GGML_F32), **BLOCK_FORMATS, "q4_0": (32, GGML_Q4_0)}
# GGUF FTYPE describes the mixed file, independently of tensor GGML types.
FILE_TYPES = {"f32": 0, "f16": 1, "bf16": 32, "q8_0": 7,
              "q4_0": 2, "q4_1": 3, "q4_k": 14, "q5_k": 16, "q6_k": 18}
LORA_TARGETS = {
    "in_proj_qkv", "gate_proj", "q_proj", "o_proj", "v_proj", "down_proj",
    "out_proj", "in_proj_z", "k_proj", "up_proj",
}
SPECIAL_TOKENS = {
    "<|endoftext|>": 248044, "<|im_start|>": 248045, "<|im_end|>": 248046,
    "<|vision_start|>": 248053, "<|vision_end|>": 248054,
    "<|image_pad|>": 248056, "<|fim_suffix|>": 248062,
}


@dataclass(frozen=True)
class TensorSpec:
    source: str
    name: str
    shape: tuple[int, ...]
    kind: str = "matrix"
    transform: str = "none"
    vision: bool = False
    frame: int | None = None
    component: str = "model"

    @property
    def output_shape(self):
        if self.frame is not None:
            out, channels, _, height, width = self.shape
            return out, channels, height, width
        if self.name.endswith("ssm_conv1d.weight"):
            return self.shape[0], self.shape[2]
        return self.shape


@dataclass
class Contract:
    config: dict
    decision: dict
    joint: dict
    backbone: dict
    tokenizer: str
    tokenizer_config: dict
    protocol_tokenizer_config: dict
    adapter_config: dict | None = None


def _json(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _expect(config, expected, prefix):
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"unsupported {prefix}.{key}: {config.get(key)!r}; expected {value!r}")


def _validate_backbone(backbone):
    _expect(backbone, {
        "model_type": "qwen3_5", "tie_word_embeddings": True,
        "image_token_id": 248056, "video_token_id": 248057,
        "vision_start_token_id": 248053, "vision_end_token_id": 248054,
    }, "backbone_config")
    text = backbone.get("text_config", {})
    _expect(text, {
        "model_type": "qwen3_5_text", "hidden_size": 2560,
        "num_hidden_layers": 32, "intermediate_size": 9216,
        "num_attention_heads": 16, "num_key_value_heads": 4, "head_dim": 256,
        "linear_num_key_heads": 16, "linear_key_head_dim": 128,
        "linear_num_value_heads": 32, "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4, "full_attention_interval": 4,
        "max_position_embeddings": 262144, "vocab_size": 248320,
        "rms_norm_eps": 1e-6, "attention_bias": False, "attention_dropout": 0.0,
        "attn_output_gate": True, "hidden_act": "silu", "tie_word_embeddings": True,
        "mlp_only_layers": [], "mtp_num_hidden_layers": 1,
        "mtp_use_dedicated_embeddings": False, "mamba_ssm_dtype": "float32",
        "layer_types": ["full_attention" if (i + 1) % 4 == 0 else "linear_attention"
                        for i in range(32)],
    }, "text_config")
    _expect(text.get("rope_parameters", {}), {
        "rope_type": "default", "rope_theta": 10000000,
        "partial_rotary_factor": 0.25, "mrope_interleaved": True,
        "mrope_section": [11, 11, 10],
    }, "rope_parameters")
    _expect(backbone.get("vision_config", {}), {
        "depth": 24, "hidden_size": 1024, "intermediate_size": 4096,
        "num_heads": 16, "patch_size": 16, "temporal_patch_size": 2,
        "in_channels": 3, "spatial_merge_size": 2,
        "num_position_embeddings": 2304, "out_hidden_size": 2560,
        "deepstack_visual_indexes": [], "hidden_act": "gelu_pytorch_tanh",
    }, "vision_config")


def read_contract(model_dir: Path, *, thinker=False):
    model_dir = Path(model_dir)
    config = _json(model_dir / "config.json")
    _expect(config, {
        "model_type": "thx02", "architectures": ["THXDecisionModel"],
        "backbone": "qwen3_5", "max_options": 255,
        "answer_types": ["choice", "noul", "score"],
    }, "config")
    backbone = config.get("backbone_config", {})
    _validate_backbone(backbone)
    decision = _json(model_dir / "decision_config.json")
    _expect(decision, {
        "format_version": 2, "base_model": "Qwen/Qwen3.5-4B",
        "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "attention_mode": "noncausal_full_attention", "pooling": "last",
        "max_pixels": 1638400,
    }, "decision_config")
    temperature = decision.get("temperature")
    if not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("decision_config.temperature must be finite and positive")
    codebook = [decision.get("codes"), decision.get("token_ids")]
    encoded = json.dumps(codebook, separators=(",", ":")).encode()
    if hashlib.sha256(encoded).hexdigest() != CODEBOOK_SHA256:
        raise ValueError("decision_config codes/token_ids differ from the pinned 255-way codebook")
    joint = _json(model_dir / "joint_config.json")
    _expect(joint, {
        "version": 3, "template": "joint-question-inline-readout-v1",
        "marker": "<|fim_suffix|>", "readout_offset": 0,
        "max_questions": None, "max_length": None,
        "usage": "nonpadding_joint_sequence_including_template_and_image_positions",
    }, "joint_config")
    protocol_tokenizer_config = _json(model_dir / "tokenizer_config.json")
    tokenizer_dir, adapter_config = model_dir, None
    if thinker:
        adapter_config = _json(model_dir / "thinker" / "adapter_config.json")
        _expect(adapter_config, {
            "peft_type": "LORA", "r": 32, "lora_alpha": 32, "bias": "none",
            "fan_in_fan_out": False, "use_dora": False, "use_rslora": False,
            "use_qalora": False, "rank_pattern": {}, "alpha_pattern": {},
            "modules_to_save": None,
        }, "thinker.adapter_config")
        if set(adapter_config.get("target_modules", [])) != LORA_TARGETS:
            raise ValueError("unsupported thinker.adapter_config.target_modules")
        tokenizer_dir = model_dir / "thinker" / "lm_config"
        backbone = _json(tokenizer_dir / "config.json")
        _validate_backbone(backbone)
    raw = (tokenizer_dir / "tokenizer.json").read_text(encoding="utf-8")
    tokenizer_json = json.loads(raw)
    if tokenizer_json.get("model", {}).get("type") != "BPE":
        raise ValueError("expected a BPE tokenizer.json")
    if tokenizer_json.get("normalizer") != {"type": "NFC"}:
        raise ValueError("expected the tokenizer's NFC normalizer")
    vocab = tokenizer_json["model"].get("vocab", {})
    if not vocab or not tokenizer_json["model"].get("merges"):
        raise ValueError("tokenizer.json is missing vocabulary or merges")
    added = {entry["content"]: entry["id"] for entry in tokenizer_json.get("added_tokens", [])}
    for token, token_id in SPECIAL_TOKENS.items():
        if added.get(token) != token_id:
            raise ValueError(f"tokenizer special token {token} must have id {token_id}")
    all_ids = list(vocab.values()) + list(added.values())
    if any(type(i) is not int or not 0 <= i < 248320 for i in all_ids):
        raise ValueError("tokenizer token id is outside the embedding table")
    tokenizer = Tokenizer.from_str(raw)
    actual = tokenizer.encode_batch(decision["codes"], add_special_tokens=False)
    if [entry.ids for entry in actual] != [[i] for i in decision["token_ids"]]:
        raise ValueError("tokenizer does not encode the pinned decision codes as single tokens")
    tokenizer_config = (_json(tokenizer_dir / "tokenizer_config.json") if thinker
                        else protocol_tokenizer_config)
    return Contract(config, decision, joint, backbone, raw, tokenizer_config,
                    protocol_tokenizer_config, adapter_config)


def tensor_specs(backbone, *, include_readout=True):
    text, vision = backbone["text_config"], backbone["vision_config"]
    d, f, vocab = text["hidden_size"], text["intermediate_size"], text["vocab_size"]
    specs = []
    def add(source, name, shape, kind="matrix", transform="none", visual=False, frame=None,
            component="model"):
        specs.append(TensorSpec(source, name, tuple(shape), kind, transform, visual, frame, component))
    add("language_model.embed_tokens.weight", "token_embd.weight", (vocab, d))
    add("language_model.norm.weight", "output_norm.weight", (d,), "float", "norm")
    key = text["linear_num_key_heads"] * text["linear_key_head_dim"]
    inner = text["linear_num_value_heads"] * text["linear_value_head_dim"]
    rank, state = text["linear_num_value_heads"], text["linear_key_head_dim"]
    for i, layer_type in enumerate(text["layer_types"]):
        src, dst = f"language_model.layers.{i}.", f"blk.{i}."
        for source, target in (("input_layernorm", "attn_norm"),
                               ("post_attention_layernorm", "post_attention_norm")):
            add(src + source + ".weight", dst + target + ".weight", (d,), "float", "norm")
        for source, target, shape in (
            ("mlp.gate_proj", "ffn_gate", (f, d)), ("mlp.up_proj", "ffn_up", (f, d)),
            ("mlp.down_proj", "ffn_down", (d, f)),
        ):
            add(src + source + ".weight", dst + target + ".weight", shape)
        if layer_type == "full_attention":
            hd, heads, kv = text["head_dim"], text["num_attention_heads"], text["num_key_value_heads"]
            for source, target, shape in (
                ("q_proj", "attn_q", (2 * heads * hd, d)),
                ("k_proj", "attn_k", (kv * hd, d)), ("v_proj", "attn_v", (kv * hd, d)),
                ("o_proj", "attn_output", (d, heads * hd)),
            ):
                add(src + "self_attn." + source + ".weight", dst + target + ".weight", shape)
            for qk in ("q", "k"):
                add(src + f"self_attn.{qk}_norm.weight", dst + f"attn_{qk}_norm.weight",
                    (hd,), "float", "norm")
        else:
            for source, target, shape in (
                ("in_proj_qkv", "attn_qkv", (2 * key + inner, d)),
                ("in_proj_z", "attn_gate", (inner, d)), ("in_proj_a", "ssm_alpha", (rank, d)),
                ("in_proj_b", "ssm_beta", (rank, d)), ("out_proj", "ssm_out", (d, inner)),
            ):
                add(src + "linear_attn." + source + ".weight", dst + target + ".weight", shape)
            for source, target, shape, transform in (
                ("conv1d.weight", "ssm_conv1d.weight", (2 * key + inner, 1, text["linear_conv_kernel_dim"]), "none"),
                ("norm.weight", "ssm_norm.weight", (state,), "none"),
                ("A_log", "ssm_a", (rank,), "a_log"),
                ("dt_bias", "ssm_dt.bias", (rank,), "none"),
            ):
                add(src + "linear_attn." + source, dst + target, shape, "float", transform)
    vd, vf = vision["hidden_size"], vision["intermediate_size"]
    for i in range(vision["depth"]):
        src, dst = f"visual.blocks.{i}.", f"v.blk.{i}."
        for source, target, shape in (
            ("norm1", "ln1", (vd,)), ("norm2", "ln2", (vd,)),
            ("attn.qkv", "attn_qkv", (3 * vd, vd)),
            ("attn.proj", "attn_out", (vd, vd)),
            ("mlp.linear_fc1", "ffn_up", (vf, vd)),
            ("mlp.linear_fc2", "ffn_down", (vd, vf)),
        ):
            add(src + source + ".weight", dst + target + ".weight", shape,
                "matrix" if len(shape) == 2 else "float", visual=True)
            add(src + source + ".bias", dst + target + ".bias", (shape[0],), "float", visual=True)
    patch = vision["patch_size"]
    for frame in (0, 1):
        add("visual.patch_embed.proj.weight", "v.patch_embd.weight" + (".1" if frame else ""),
            (vd, 3, 2, patch, patch), "patch", visual=True, frame=frame)
    add("visual.patch_embed.proj.bias", "v.patch_embd.bias", (vd,), "float", visual=True)
    add("visual.pos_embed.weight", "v.position_embd.weight",
        (vision["num_position_embeddings"], vd), "float", visual=True)
    merged = vd * vision["spatial_merge_size"] ** 2
    for source, target, shape in (
        ("norm", "v.post_ln", (vd,)),
        ("linear_fc1", "mm.0", (merged, merged)),
        ("linear_fc2", "mm.2", (vision["out_hidden_size"], merged)),
    ):
        add("visual.merger." + source + ".weight", target + ".weight", shape,
            "matrix" if len(shape) == 2 else "float", visual=True)
        add("visual.merger." + source + ".bias", target + ".bias", (shape[0],), "float", visual=True)
    if include_readout:
        add("weight", "cls.output.weight", (255, d), component="readout")
    if len({spec.name for spec in specs}) != len(specs):
        raise ValueError("GGUF tensor names collide")
    return specs


def validate_source(source, expected, *, allowed_dtypes=("BF16",)):
    present = set(source.header) - {"__metadata__"}
    if missing := set(expected) - present:
        raise ValueError(f"{source.path}: missing tensors: {sorted(missing)[:4]}")
    if unexpected := present - set(expected):
        raise ValueError(f"{source.path}: unexpected tensors: {sorted(unexpected)[:4]}")
    spans = []
    for name, shape in expected.items():
        info = source.header[name]
        if info.get("shape") != list(shape):
            raise ValueError(f"{name}: shape {info.get('shape')} != {list(shape)}")
        dtype = info.get("dtype")
        if dtype not in allowed_dtypes:
            raise ValueError(f"{name}: unsupported dtype {dtype}")
        span = info.get("data_offsets")
        if not isinstance(span, list) or len(span) != 2 or any(type(i) is not int for i in span):
            raise ValueError(f"{name}: invalid data_offsets")
        start, end = span
        size = 4 if dtype == "F32" else 2
        if start < 0 or end - start != math.prod(shape) * size:
            raise ValueError(f"{name}: invalid data_offsets or payload length")
        if source.data_offset + end > source.file_size:
            raise ValueError(f"{name}: truncated payload")
        spans.append((start, end))
    last = 0
    for start, end in sorted(spans):
        if start != last:
            raise ValueError(f"{source.path}: overlapping tensors or gaps in payload")
        last = end
    if source.data_offset + last != source.file_size:
        raise ValueError(f"{source.path}: unexpected trailing payload")


def lora_contract(adapter, specs, config):
    rank, scale = config["r"], config["lora_alpha"] / config["r"]
    expected, pairs = {}, {}
    for spec in specs:
        if spec.vision or spec.source.split(".")[-2] not in config["target_modules"]:
            continue
        stem = "base_model.model." + spec.source.removesuffix(".weight")
        a, b = stem + ".lora_A.weight", stem + ".lora_B.weight"
        expected[a], expected[b] = (rank, spec.shape[1]), (spec.shape[0], rank)
        pairs[spec.source] = (a, b, scale)
    if not pairs:
        raise ValueError("thinker adapter has no matching text projections")
    validate_source(adapter, expected, allowed_dtypes=("BF16", "F16", "F32"))
    return pairs


def _format(target):
    target = target.lower()
    if target not in FORMATS:
        raise ValueError(f"unknown format {target!r}; expected {sorted(FORMATS)}")
    return target


def tensor_type(spec, target):
    target = _format(target)
    if spec.kind == "float":
        return GGML_F32
    if spec.vision and (target in ("q4_0", "q4_1", "q4_k", "q5_k", "q6_k")
                        or (spec.kind == "patch" and target == "q8_0")):
        return GGML_F32
    block, kind = FORMATS[target]
    width = math.prod(spec.output_shape[1:])
    return kind if width % block == 0 else GGML_F32


def _values(raw, dtype):
    if dtype == "BF16":
        return np.frombuffer(bf16_to_f32(raw), dtype="<f4")
    if dtype == "F16":
        return np.frombuffer(raw, dtype="<f2").astype("<f4")
    if dtype == "F32":
        return np.frombuffer(raw, dtype="<f4")
    raise ValueError(f"unsupported source dtype {dtype}")


def _encode(values, kind):
    values = np.ascontiguousarray(values, dtype="<f4")
    if not np.isfinite(values).all():
        raise ValueError("non-finite tensor values")
    if kind == GGML_F32:
        return values.tobytes()
    if kind == GGML_BF16:
        return f32_to_bf16(values.tobytes())
    if kind == GGML_F16:
        with np.errstate(over="ignore"):
            result = values.astype("<f2")
        if not np.isfinite(result).all():
            raise ValueError("tensor values overflow F16 storage")
        return result.tobytes()
    if kind == GGML_Q8_0:
        return quantize_q8_0(values)
    if kind == GGML_Q4_0:
        return quantize_q4_0(values)
    if kind == GGML_Q4_1:
        return quantize_q4_1(values)
    return quantize_k({GGML_Q4K: "q4_k", GGML_Q5K: "q5_k", GGML_Q6K: "q6_k"}[kind], values)


def recurrent_order(key_heads, value_heads, head_dim):
    heads = np.arange(value_heads)
    original = (heads % key_heads) * (value_heads // key_heads) + heads // key_heads
    return (original[:, None] * head_dim + np.arange(head_dim)).reshape(-1)


def _recurrent_rows(spec):
    order = np.arange(spec.shape[0])
    suffix = spec.source.partition(".linear_attn.")[2]
    if suffix in ("in_proj_qkv.weight", "conv1d.weight"):
        order[4096:] = 4096 + recurrent_order(16, 32, 128)
    elif suffix == "in_proj_z.weight":
        order = recurrent_order(16, 32, 128)
    elif suffix in ("in_proj_a.weight", "in_proj_b.weight", "A_log", "dt_bias"):
        order = recurrent_order(16, 32, 1)
    return order


def _read_rows(stream, offset, indices, row_bytes, name):
    runs = np.split(indices, np.flatnonzero(np.diff(indices) != 1) + 1)
    payload = bytearray()
    for run in runs:
        stream.seek(offset + int(run[0]) * row_bytes)
        length = len(run) * row_bytes
        raw = stream.read(length)
        if len(raw) != length:
            raise ValueError(f"{name}: truncated tensor")
        payload.extend(raw)
    return payload


def tensor_chunks(source, spec, kind, *, adapter=None, pair=None):
    info = source.header[spec.source]
    dtype = info["dtype"]
    element_bytes = 4 if dtype == "F32" else 2
    row_elements = math.prod(spec.shape[1:])
    rows_per_chunk = max(1, CHUNK_BYTES // (row_elements * element_bytes))
    row_order = _recurrent_rows(spec)
    a = b = None
    if pair is not None:
        import torch
        a_name, b_name, scale = pair
        at, bt = adapter.get(a_name), adapter.get(b_name)
        a = torch.from_numpy(_values(at.raw, at.dtype).reshape(at.shape).copy())
        b = torch.from_numpy(_values(bt.raw, bt.dtype).reshape(bt.shape).copy())
    with source.path.open("rb") as stream:
        for start in range(0, spec.shape[0], rows_per_chunk):
            rows = min(rows_per_chunk, spec.shape[0] - start)
            indices = row_order[start:start + rows]
            raw = _read_rows(stream, source.data_offset + info["data_offsets"][0], indices,
                             row_elements * element_bytes, spec.source)
            values = _values(raw, dtype).reshape((rows, *spec.shape[1:]))
            if spec.frame is not None:
                values = values[:, :, spec.frame, :, :]
            if pair is not None:
                # Match build_thinker.py: merge in FP32, then round to the source dtype.
                base = torch.from_numpy(values.copy())
                values = (base + scale * (b[indices] @ a)).numpy()
                if dtype == "BF16":
                    values = _values(f32_to_bf16(values.tobytes()), "BF16").reshape(values.shape)
                elif dtype == "F16":
                    values = values.astype("<f2").astype("<f4")
            if spec.source.endswith(".linear_attn.out_proj.weight"):
                values = values[:, recurrent_order(16, 32, 128)]
            if spec.transform == "norm":
                values = values + np.float32(1)
            elif spec.transform == "a_log":
                values = -np.exp(values)
            yield _encode(values.reshape(-1), kind)


def add_tensor(writer, source, spec, target, *, adapter=None, pair=None):
    kind = tensor_type(spec, target)
    dims = gguf_dims(spec.output_shape)
    writer.add_tensor_chunks(spec.name, kind, dims, _tensor_nbytes(kind, dims),
                             lambda: tensor_chunks(source, spec, kind, adapter=adapter, pair=pair))


def escape_tokens(tokenizer_config):
    tokens = {"<|fim_suffix|>"}
    def collect(value):
        if isinstance(value, str):
            tokens.add(value)
        elif isinstance(value, dict) and isinstance(value.get("content"), str):
            tokens.add(value["content"])
        elif isinstance(value, list):
            for entry in value:
                collect(entry)
        elif value is not None:
            raise ValueError("invalid tokenizer_config special token")
    for key in ("bos_token", "eos_token", "unk_token", "sep_token", "pad_token",
                "cls_token", "mask_token", "additional_special_tokens"):
        collect(tokenizer_config.get(key))
    for key in ("model_specific_special_tokens", "extra_special_tokens"):
        for value in tokenizer_config.get(key, {}).values():
            collect(value)
    return sorted(tokens, key=lambda token: (-len(token), token))


def write_metadata(writer, contract, target, *, thinker=False):
    target = _format(target)
    arch = "qwen35" if thinker else "thx02"
    text, vision = contract.backbone["text_config"], contract.backbone["vision_config"]
    metadata = {
        "general.architecture": arch, "general.name": "THX-02-Thinker" if thinker else "THX-02",
        "general.file_type": FILE_TYPES[target], "general.quantization_version": 2,
        "thx02.source_revision": SOURCE_REVISION,
        "thx02.component": "thinker" if thinker else "decision",
        "thx02.codes": contract.decision["codes"],
        "thx02.code_token_ids": contract.decision["token_ids"],
        "thx02.temperature": contract.decision["temperature"],
        "thx02.protocol_version": contract.joint["version"],
        "thx02.attention_mode": "causal" if thinker else contract.decision["attention_mode"],
        "thx02.recurrent_head_layout": "qwen35-modulo-v1",
        # The official thinker_payload builds its body with the decision Encoder.
        "thx02.escape_tokens": escape_tokens(contract.protocol_tokenizer_config),
        "thx02.config": json.dumps(contract.config, separators=(",", ":")),
        "thx02.decision_config": json.dumps(contract.decision, separators=(",", ":")),
        "thx02.joint_config": json.dumps(contract.joint, separators=(",", ":")),
        "thx02.tokenizer_json": contract.tokenizer,
        "thx02.text_storage": target,
        "thx02.vision_storage": "f32" if target in ("q4_0", "q4_k", "q5_k", "q6_k") else target,
        "thx02.float_storage": "f32",
        "thx02.thinker": thinker,
    }
    for key, value in {
        "embedding_length": text["hidden_size"], "block_count": text["num_hidden_layers"],
        "context_length": text["max_position_embeddings"], "feed_forward_length": text["intermediate_size"],
        "attention.head_count": text["num_attention_heads"], "attention.head_count_kv": text["num_key_value_heads"],
        "attention.key_length": text["head_dim"], "attention.value_length": text["head_dim"],
        "attention.layer_norm_rms_epsilon": text["rms_norm_eps"],
        "attention.recurrent_layers": [kind == "linear_attention" for kind in text["layer_types"]],
        "rope.freq_base": float(text["rope_parameters"]["rope_theta"]),
        "rope.dimension_count": int(text["head_dim"] * text["rope_parameters"]["partial_rotary_factor"]),
        "rope.dimension_sections": [*text["rope_parameters"]["mrope_section"], 0],
        "ssm.conv_kernel": text["linear_conv_kernel_dim"],
        "ssm.inner_size": text["linear_num_value_heads"] * text["linear_value_head_dim"],
        "ssm.state_size": text["linear_key_head_dim"],
        "ssm.time_step_rank": text["linear_num_value_heads"], "ssm.group_count": text["linear_num_key_heads"],
        "full_attention_interval": text["full_attention_interval"], "vocab_size": text["vocab_size"],
        "nextn_predict_layers": 0,
        "image_token_id": contract.backbone["image_token_id"],
        "vision_start_token_id": contract.backbone["vision_start_token_id"],
        "vision_end_token_id": contract.backbone["vision_end_token_id"],
    }.items():
        metadata[f"{arch}.{key}"] = value
    metadata.update({
        "clip.has_vision_encoder": True, "clip.projector_type": "qwen3vl_merger",
        "clip.vision.embedding_length": vision["hidden_size"],
        "clip.vision.feed_forward_length": vision["intermediate_size"],
        "clip.vision.block_count": vision["depth"], "clip.vision.attention.head_count": vision["num_heads"],
        "clip.vision.patch_size": vision["patch_size"], "clip.vision.image_size": 768,
        "clip.vision.image_min_pixels": 65536,
        "clip.vision.image_max_pixels": contract.decision["max_pixels"],
        "clip.vision.projection_dim": vision["out_hidden_size"],
        "clip.vision.spatial_merge_size": vision["spatial_merge_size"],
        "clip.vision.attention.layer_norm_epsilon": 1e-6,
        "clip.vision.image_mean": [0.5, 0.5, 0.5], "clip.vision.image_std": [0.5, 0.5, 0.5],
        "clip.vision.is_deepstack_layers": [False] * vision["depth"], "clip.use_gelu": True,
    })
    tokenizer = json.loads(contract.tokenizer)
    vocab = {int(token_id): token for token, token_id in tokenizer["model"]["vocab"].items()}
    types = [1] * text["vocab_size"]
    for entry in tokenizer["added_tokens"]:
        vocab[entry["id"]] = entry["content"]
        types[entry["id"]] = 3 if entry["special"] else 4
    merges = [pair if isinstance(pair, str) else " ".join(pair) for pair in tokenizer["model"]["merges"]]
    metadata.update({
        "tokenizer.ggml.model": "gpt2", "tokenizer.ggml.pre": "qwen2",
        "tokenizer.ggml.tokens": [vocab.get(i, f"<|reserved_{i}|>") for i in range(text["vocab_size"])],
        "tokenizer.ggml.token_type": types, "tokenizer.ggml.merges": merges,
        "tokenizer.ggml.bos_token_id": 248044,
        "tokenizer.ggml.eos_token_id": 248046 if thinker else 248044,
        "tokenizer.ggml.add_bos_token": False, "tokenizer.ggml.add_eos_token": False,
        "tokenizer.ggml.normalizer.nfc": True,
    })
    if template := contract.tokenizer_config.get("chat_template"):
        if not isinstance(template, str):
            raise ValueError("expected a single chat_template string")
        metadata["tokenizer.chat_template"] = template
    if thinker:
        metadata["qwen35.tokenizer_json"] = contract.tokenizer
        metadata["thx02.thinker_config"] = json.dumps(contract.backbone, separators=(",", ":"))
        metadata["thx02.thinker_adapter_config"] = json.dumps(contract.adapter_config, separators=(",", ":"))
    for key, value in metadata.items():
        writer.add_meta(key, value)


def checked_sha256(path, expected=None):
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if expected is not None and digest != expected:
        raise ValueError(f"{path}: SHA256 {digest} != pinned {expected}")
    return digest


def write_atomic(writer, output):
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=f".{output.name}.", suffix=".partial",
                                     dir=output.parent, delete=False) as temporary:
        writer.path = Path(temporary.name)
    try:
        writer.write()
        # A link publishes the complete file atomically and refuses to replace an existing output.
        os.link(writer.path, output)
    finally:
        writer.path.unlink(missing_ok=True)


def convert(model_dir: Path, output: Path, *, target_format="f32", thinker=False):
    target_format = _format(target_format)
    model_dir, output = Path(model_dir), Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    contract = read_contract(model_dir, thinker=thinker)
    specs = tensor_specs(contract.backbone, include_readout=not thinker)
    sources = {"model": open_safetensors(model_dir / "model.safetensors")}
    if not thinker:
        sources["readout"] = open_safetensors(model_dir / "readout.safetensors")
    for component, source in sources.items():
        expected = {s.source: s.shape for s in specs if s.component == component}
        validate_source(source, expected)
    adapter, pairs = None, {}
    if thinker:
        adapter = open_safetensors(model_dir / "thinker" / "adapter_model.safetensors")
        pairs = lora_contract(adapter, specs, contract.adapter_config)
    writer = GgufWriter(output)
    write_metadata(writer, contract, target_format, thinker=thinker)
    for component, source in sources.items():
        digest = checked_sha256(source.path, SOURCE_SHA256[component])
        writer.add_meta(f"thx02.{component}_sha256", digest)
    if adapter is not None:
        writer.add_meta("thx02.thinker_adapter_sha256", checked_sha256(adapter.path))
    for spec in specs:
        add_tensor(writer, sources[spec.component], spec, target_format,
                   adapter=adapter, pair=pairs.get(spec.source))
    writer.metadata = [(key, model_metadata_value(value)) for key, value in writer.metadata]
    write_atomic(writer, output)
    print(f"{output}: {len(specs)} tensors; text={target_format}, "
          f"vision={dict(writer.metadata)['thx02.vision_storage']}; norms/biases/positions/SSM=F32")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--format", default="f32", type=str.lower, choices=[*sorted(FORMATS), "q4_1"])
    parser.add_argument("--thinker", action="store_true", help="export the separate merged causal qwen35 thinker")
    args = parser.parse_args()
    convert(args.model_dir, args.output, target_format=args.format, thinker=args.thinker)


if __name__ == "__main__":
    main()
