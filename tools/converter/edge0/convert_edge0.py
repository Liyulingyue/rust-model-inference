"""Losslessly wrap Edge0-35B MLX-affine weights and LoRA in a GGUF container.

The packed U32 words are stored as GGUF I32 with identical bytes. The scales,
biases, and LoRA remain BF16/F16. This is an Edge0 GGUF, not a llama.cpp model.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path

from tools.converter.utils.gguf import (
    GGML_BF16,
    GGML_F16,
    GGML_I32,
    GgufWriter,
    gguf_dims,
    open_safetensors,
)


LAYER = re.compile(r"language_model\.model\.layers\.(\d+)\.(.+)")
STEMS = {
    "input_layernorm": "attn_norm",
    "post_attention_layernorm": "post_attention_norm",
    "self_attn.q_proj": "attn_q",
    "self_attn.k_proj": "attn_k",
    "self_attn.v_proj": "attn_v",
    "self_attn.o_proj": "attn_output",
    "self_attn.q_norm": "attn_q_norm",
    "self_attn.k_norm": "attn_k_norm",
    "linear_attn.in_proj_qkv": "attn_qkv",
    "linear_attn.in_proj_z": "attn_gate",
    "linear_attn.in_proj_a": "ssm_alpha",
    "linear_attn.in_proj_b": "ssm_beta",
    "linear_attn.conv1d": "ssm_conv1d",
    "linear_attn.norm": "ssm_norm",
    "linear_attn.out_proj": "ssm_out",
    "mlp.gate": "ffn_gate_inp",
    "mlp.shared_expert_gate": "ffn_shared_gate",
    "mlp.shared_expert.gate_proj": "ffn_gate",
    "mlp.shared_expert.up_proj": "ffn_up",
    "mlp.shared_expert.down_proj": "ffn_down",
    "mlp.switch_mlp.gate_proj": "ffn_gate_exps",
    "mlp.switch_mlp.up_proj": "ffn_up_exps",
    "mlp.switch_mlp.down_proj": "ffn_down_exps",
}
SPECIAL = {
    "linear_attn.A_log": "ssm_a",
    "linear_attn.dt_bias": "ssm_dt.bias",
}
TOP = {
    "language_model.model.embed_tokens": "token_embd",
    "language_model.model.norm": "output_norm",
    "language_model.lm_head": "output",
}
DTYPES = {"U32": GGML_I32, "BF16": GGML_BF16, "F16": GGML_F16}


def gguf_name(name: str) -> str:
    for source, target in TOP.items():
        if name.startswith(source + "."):
            return target + name[len(source):]
    match = LAYER.fullmatch(name)
    if match is None:
        raise ValueError(f"unexpected Edge0 tensor: {name}")
    layer, tail = match.groups()
    if tail in SPECIAL:
        return f"blk.{layer}.{SPECIAL[tail]}"
    stem, sep, suffix = tail.rpartition(".")
    if not sep or stem not in STEMS or suffix not in {"weight", "scales", "biases", "lora_A", "lora_B"}:
        raise ValueError(f"unexpected Edge0 tensor: {name}")
    return f"blk.{layer}.{STEMS[stem]}.{suffix}"


def chunks(path: Path, offset: int, length: int):
    with path.open("rb") as source:
        source.seek(offset)
        while length:
            data = source.read(min(length, 8 << 20))
            if not data:
                raise ValueError(f"truncated tensor in {path}")
            yield data
            length -= len(data)


def tensors(model_dir: Path):
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    shard_names = sorted(set(index["weight_map"].values()))
    if len(shard_names) != 4:
        raise ValueError(f"expected four Edge0 shards, got {shard_names}")
    seen = set()
    for shard_name in shard_names + ["lora_edge0_35b.safetensors"]:
        source = open_safetensors(model_dir / shard_name)
        for name, info in source.header.items():
            if name == "__metadata__":
                continue
            if shard_name in shard_names and index["weight_map"].get(name) != shard_name:
                raise ValueError(f"index mismatch for {name}")
            mapped = gguf_name(name)
            if mapped in seen:
                raise ValueError(f"duplicate tensor {mapped}")
            seen.add(mapped)
            shape = tuple(info["shape"])
            start, end = info["data_offsets"]
            dtype = info["dtype"]
            if dtype not in DTYPES or start < 0 or end < start or source.data_offset + end > source.file_size:
                raise ValueError(f"invalid tensor {name} in {shard_name}")
            nbytes = end - start
            if nbytes != (4 if dtype == "U32" else 2) * __import__("math").prod(shape):
                raise ValueError(f"invalid tensor size for {name}")
            yield mapped, DTYPES[dtype], gguf_dims(shape), source.path, source.data_offset + start, nbytes
    if len(seen) != len(index["weight_map"]) + 620:
        raise ValueError("Edge0 tensor or LoRA inventory is incomplete")


def add_metadata(writer: GgufWriter, model_dir: Path) -> None:
    config = json.loads((model_dir / "config.json").read_text())
    text = config["text_config"]
    quant = config["quantization"]
    if (config["model_type"], text["num_hidden_layers"], text["num_experts"],
        quant["group_size"], quant["bits"], quant["mode"]) != (
        "qwen3_5_moe", 40, 256, 64, 4, "affine"
    ):
        raise ValueError("unsupported Edge0-35B model contract")
    writer.add_meta("general.architecture", "edge0")
    writer.add_meta("general.name", "Edge0-35B-A3B-preview")
    writer.add_meta("edge0.quant.group_size", 64)
    writer.add_meta("edge0.expert_count", 256)
    writer.add_meta("edge0.expert_used_count", 4)
    writer.add_meta("edge0.expert_feed_forward_length", text["moe_intermediate_size"])
    writer.add_meta("edge0.shared_expert_feed_forward_length", text["shared_expert_intermediate_size"])
    writer.add_meta("edge0.lora.scale", 2.0)
    values = {
        "block_count": text["num_hidden_layers"],
        "context_length": text["max_position_embeddings"],
        "embedding_length": text["hidden_size"],
        "feed_forward_length": text["shared_expert_intermediate_size"],
        "attention.head_count": text["num_attention_heads"],
        "attention.head_count_kv": text["num_key_value_heads"],
        "attention.key_length": text["head_dim"],
        "attention.value_length": text["head_dim"],
        "attention.layer_norm_rms_epsilon": text["rms_norm_eps"],
        "rope.dimension_count": int(text["head_dim"] * text["partial_rotary_factor"]),
        "rope.dimension_sections": [*text["rope_parameters"]["mrope_section"], 0],
        "rope.freq_base": text["rope_parameters"]["rope_theta"],
        "ssm.conv_kernel": text["linear_conv_kernel_dim"],
        "ssm.state_size": text["linear_key_head_dim"],
        "ssm.group_count": text["linear_num_key_heads"],
        "ssm.time_step_rank": text["linear_num_value_heads"],
        "ssm.inner_size": text["linear_num_value_heads"] * text["linear_value_head_dim"],
        "full_attention_interval": text["full_attention_interval"],
        "vocab_size": text["vocab_size"],
    }
    for key, value in values.items():
        writer.add_meta(f"edge0.{key}", value)
    tokenizer = json.loads((model_dir / "tokenizer.json").read_text())
    vocab = {int(id): token for token, id in tokenizer["model"]["vocab"].items()}
    added = tokenizer["added_tokens"]
    vocab.update({entry["id"]: entry["content"] for entry in added})
    tokens = [vocab.get(i, f"<|reserved_{i}|>") for i in range(text["vocab_size"])]
    types = [1] * len(tokens)
    for entry in added:
        types[entry["id"]] = 3
    writer.add_meta("tokenizer.ggml.model", "gpt2")
    writer.add_meta("tokenizer.ggml.pre", "qwen2")
    writer.add_meta("tokenizer.ggml.tokens", tokens)
    writer.add_meta("tokenizer.ggml.token_type", types)
    writer.add_meta("tokenizer.ggml.merges", [" ".join(pair) for pair in tokenizer["model"]["merges"]])
    writer.add_meta("tokenizer.ggml.eos_token_id", 248046)
    writer.add_meta("tokenizer.ggml.bos_token_id", 248044)
    writer.add_meta("tokenizer.ggml.add_bos_token", False)
    writer.add_meta("tokenizer.ggml.add_eos_token", False)
    writer.add_meta("tokenizer.chat_template", (model_dir / "chat_template.jinja").read_text())


def convert(model_dir: Path, output: Path, check_only: bool) -> None:
    writer = GgufWriter(output.with_suffix(output.suffix + ".part"))
    add_metadata(writer, model_dir)
    count = 0
    total = 0
    for name, dtype, shape, source, offset, length in tensors(model_dir):
        writer.add_tensor_chunks(name, dtype, shape, length,
                                 lambda source=source, offset=offset, length=length: chunks(source, offset, length))
        count += 1
        total += length
    print(f"validated {count} lossless tensors; payload {total:,} bytes", flush=True)
    if check_only:
        return
    if output.exists() or writer.path.exists():
        raise FileExistsError(output)
    if shutil.disk_usage(output.parent).free < total + (64 << 20):
        raise OSError("not enough free space for lossless GGUF")
    try:
        writer.write()
        os.replace(writer.path, output)
    except BaseException:
        writer.path.unlink(missing_ok=True)
        raise
    print(output, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if not args.check and args.out is None:
        parser.error("--out is required unless --check is set")
    convert(args.model_dir, args.out or args.model_dir / "Edge0.gguf", args.check)
