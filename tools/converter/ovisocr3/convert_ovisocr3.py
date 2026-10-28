#!/usr/bin/env python3
"""Convert the ModelScope OvisOCR3 checkpoint with the pinned llama.cpp exporter.

OvisOCR3 is a Qwen3.5 text/vision checkpoint.  The adapter keeps the upstream
converter as the read-only oracle and validates the complete source contract
before asking it to emit the qwen35 GGUF and qwen3vl mmproj files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


LLAMA_CPP_COMMIT = "b96806d96061049a5b574269b049bf6241d63d46"
ARCHITECTURE = "Qwen3_5ForConditionalGeneration"

# llama.cpp's generic BPE exporter does not carry the Hugging Face NFC
# normalizer into GGUF.  Keep the source tokenizer contract in the text file,
# matching the existing NeoHorse/Qwen-Drive adapters.
_TEXT_CONVERTER = r'''
import sys
from pathlib import Path

llama_cpp, model, outfile = map(Path, sys.argv[1:])
sys.path.insert(0, str(llama_cpp))
sys.path.insert(0, str(llama_cpp / "gguf-py"))
from conversion.qwen import Qwen3_5TextModel

original_set_vocab = Qwen3_5TextModel.set_vocab

def set_vocab(self):
    original_set_vocab(self)
    self.gguf_writer.add_bool("tokenizer.ggml.normalizer.nfc", True)

Qwen3_5TextModel.set_vocab = set_vocab

import convert_hf_to_gguf
sys.argv = [
    "convert_hf_to_gguf.py",
    str(model),
    "--outfile", str(outfile),
    "--outtype", "bf16",
    "--no-nextn",
]
convert_hf_to_gguf.main()
'''


def _shape(config: dict, name: str) -> tuple[int, ...]:
    text = config["text_config"]
    vision = config["vision_config"]
    h = int(text["hidden_size"])
    ff = int(text["intermediate_size"])
    # Qwen3.5 full-attention stores a per-head attention gate beside Q, so the
    # HF q_proj width is twice the logical query width used by the kernel.
    q = int(text["num_attention_heads"]) * int(text["head_dim"]) * 2
    kv = int(text["num_key_value_heads"]) * int(text["head_dim"])
    linear_qkv = 2 * int(text["linear_num_key_heads"]) * int(text["linear_key_head_dim"])
    linear_qkv += int(text["linear_num_value_heads"]) * int(text["linear_value_head_dim"])
    linear_z = int(text["linear_num_value_heads"]) * int(text["linear_value_head_dim"])
    if name == "model.language_model.embed_tokens.weight":
        return (int(text["vocab_size"]), h)
    if name == "model.language_model.norm.weight":
        return (h,)
    if name.startswith("model.language_model.layers."):
        parts = name.split(".")
        layer = int(parts[3])
        suffix = ".".join(parts[4:])
        full = text["layer_types"][layer] == "full_attention"
        common = {
            "input_layernorm.weight": (h,),
            "post_attention_layernorm.weight": (h,),
            "mlp.down_proj.weight": (h, ff),
            "mlp.gate_proj.weight": (ff, h),
            "mlp.up_proj.weight": (ff, h),
        }
        if suffix in common:
            return common[suffix]
        if full:
            return {
                "self_attn.k_norm.weight": (int(text["head_dim"]),),
                "self_attn.k_proj.weight": (kv, h),
                "self_attn.o_proj.weight": (h, q // 2),
                "self_attn.q_norm.weight": (int(text["head_dim"]),),
                "self_attn.q_proj.weight": (q, h),
                "self_attn.v_proj.weight": (kv, h),
            }[suffix]
        return {
            "linear_attn.A_log": (int(text["linear_num_value_heads"]),),
            "linear_attn.conv1d.weight": (
                linear_qkv,
                1,
                int(text["linear_conv_kernel_dim"]),
            ),
            "linear_attn.dt_bias": (int(text["linear_num_value_heads"]),),
            "linear_attn.in_proj_a.weight": (
                int(text["linear_num_value_heads"]),
                h,
            ),
            "linear_attn.in_proj_b.weight": (
                int(text["linear_num_value_heads"]),
                h,
            ),
            "linear_attn.in_proj_qkv.weight": (linear_qkv, h),
            "linear_attn.in_proj_z.weight": (linear_z, h),
            "linear_attn.norm.weight": (
                int(text["linear_value_head_dim"]),
            ),
            "linear_attn.out_proj.weight": (h, linear_z),
        }[suffix]
    if name == "model.visual.patch_embed.proj.weight":
        return (
            int(vision["hidden_size"]),
            int(vision["in_channels"]),
            int(vision["temporal_patch_size"]),
            int(vision["patch_size"]),
            int(vision["patch_size"]),
        )
    if name == "model.visual.patch_embed.proj.bias":
        return (int(vision["hidden_size"]),)
    if name == "model.visual.pos_embed.weight":
        return (int(vision["num_position_embeddings"]), int(vision["hidden_size"]))
    if name.startswith("model.visual.blocks."):
        suffix = ".".join(name.split(".")[4:])
        vh = int(vision["hidden_size"])
        vf = int(vision["intermediate_size"])
        return {
            "attn.proj.bias": (vh,),
            "attn.proj.weight": (vh, vh),
            "attn.qkv.bias": (3 * vh,),
            "attn.qkv.weight": (3 * vh, vh),
            "mlp.linear_fc1.bias": (vf,),
            "mlp.linear_fc1.weight": (vf, vh),
            "mlp.linear_fc2.bias": (vh,),
            "mlp.linear_fc2.weight": (vh, vf),
            "norm1.bias": (vh,),
            "norm1.weight": (vh,),
            "norm2.bias": (vh,),
            "norm2.weight": (vh,),
        }[suffix]
    if name.startswith("model.visual.merger."):
        suffix = name.removeprefix("model.visual.merger.")
        vh = int(vision["hidden_size"])
        merged = vh * int(vision["spatial_merge_size"]) ** 2
        out = int(vision["out_hidden_size"])
        return {
            "norm.bias": (vh,),
            "norm.weight": (vh,),
            "linear_fc1.bias": (merged,),
            "linear_fc1.weight": (merged, merged),
            "linear_fc2.bias": (out,),
            "linear_fc2.weight": (out, merged),
        }[suffix]
    raise KeyError(name)


def expected_tensor_shapes(config: dict) -> dict[str, tuple[int, ...]]:
    text = config["text_config"]
    vision = config["vision_config"]
    names = [
        "model.language_model.embed_tokens.weight",
        "model.language_model.norm.weight",
    ]
    for layer in range(int(text["num_hidden_layers"])):
        prefix = f"model.language_model.layers.{layer}."
        names.extend(
            prefix + suffix
            for suffix in [
                "input_layernorm.weight",
                "mlp.down_proj.weight",
                "mlp.gate_proj.weight",
                "mlp.up_proj.weight",
                "post_attention_layernorm.weight",
            ]
        )
        if text["layer_types"][layer] == "full_attention":
            names.extend(
                prefix + suffix
                for suffix in [
                    "self_attn.k_norm.weight",
                    "self_attn.k_proj.weight",
                    "self_attn.o_proj.weight",
                    "self_attn.q_norm.weight",
                    "self_attn.q_proj.weight",
                    "self_attn.v_proj.weight",
                ]
            )
        else:
            names.extend(
                prefix + suffix
                for suffix in [
                    "linear_attn.A_log",
                    "linear_attn.conv1d.weight",
                    "linear_attn.dt_bias",
                    "linear_attn.in_proj_a.weight",
                    "linear_attn.in_proj_b.weight",
                    "linear_attn.in_proj_qkv.weight",
                    "linear_attn.in_proj_z.weight",
                    "linear_attn.norm.weight",
                    "linear_attn.out_proj.weight",
                ]
            )
    visual = [
        "model.visual.patch_embed.proj.weight",
        "model.visual.patch_embed.proj.bias",
        "model.visual.pos_embed.weight",
    ]
    for layer in range(int(vision["depth"])):
        visual.extend(
            f"model.visual.blocks.{layer}.{suffix}"
            for suffix in [
                "attn.proj.bias",
                "attn.proj.weight",
                "attn.qkv.bias",
                "attn.qkv.weight",
                "mlp.linear_fc1.bias",
                "mlp.linear_fc1.weight",
                "mlp.linear_fc2.bias",
                "mlp.linear_fc2.weight",
                "norm1.bias",
                "norm1.weight",
                "norm2.bias",
                "norm2.weight",
            ]
        )
    visual.extend(
        f"model.visual.merger.{suffix}"
        for suffix in [
            "norm.bias",
            "norm.weight",
            "linear_fc1.bias",
            "linear_fc1.weight",
            "linear_fc2.bias",
            "linear_fc2.weight",
        ]
    )
    names.extend(visual)
    return {name: _shape(config, name) for name in names}


def validate_config(config: dict) -> None:
    if config.get("architectures") != [ARCHITECTURE]:
        raise ValueError(f"unsupported architecture: {config.get('architectures')!r}")
    if config.get("model_type") != "qwen3_5":
        raise ValueError("expected model_type=qwen3_5")
    required_top = {
        "tie_word_embeddings": True,
        "image_token_id": 248056,
        "video_token_id": 248057,
        "vision_start_token_id": 248053,
        "vision_end_token_id": 248054,
    }
    for key, expected in required_top.items():
        if config.get(key) != expected:
            raise ValueError(f"{key}: expected {expected!r}, got {config.get(key)!r}")
    text = config.get("text_config")
    vision = config.get("vision_config")
    if not isinstance(text, dict) or not isinstance(vision, dict):
        raise ValueError("OvisOCR3 requires text_config and vision_config")
    required_text = {
        "attention_bias": False,
        "attention_dropout": 0.0,
        "attn_output_gate": True,
        "dtype": "bfloat16",
        "hidden_size": 1024,
        "intermediate_size": 3584,
        "hidden_act": "silu",
        "num_hidden_layers": 24,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "full_attention_interval": 4,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 16,
        "linear_value_head_dim": 128,
        "rms_norm_eps": 1e-6,
        "vocab_size": 248320,
        "max_position_embeddings": 262144,
        "mamba_ssm_dtype": "float32",
        "model_type": "qwen3_5_text",
        "tie_word_embeddings": True,
        "use_cache": True,
    }
    for key, expected in required_text.items():
        if text.get(key) != expected:
            raise ValueError(f"text_config.{key}: expected {expected!r}, got {text.get(key)!r}")
    if text.get("layer_types") != [
        "linear_attention" if i % 4 != 3 else "full_attention" for i in range(24)
    ]:
        raise ValueError("text_config.layer_types does not match Qwen3.5 0.8B")
    rope = text.get("rope_parameters", {})
    required_rope = {
        "mrope_interleaved": True,
        "mrope_section": [11, 11, 10],
        "partial_rotary_factor": 0.25,
        "rope_theta": 10000000,
        "rope_type": "default",
    }
    for key, expected in required_rope.items():
        if rope.get(key) != expected:
            raise ValueError(f"rope_parameters.{key}: expected {expected!r}, got {rope.get(key)!r}")
    required_vision = {
        "depth": 12,
        "hidden_size": 768,
        "intermediate_size": 3072,
        "model_type": "qwen3_5",
        "num_heads": 12,
        "patch_size": 16,
        "temporal_patch_size": 2,
        "spatial_merge_size": 2,
        "num_position_embeddings": 2304,
        "out_hidden_size": 1024,
        "in_channels": 3,
        "hidden_act": "gelu_pytorch_tanh",
    }
    for key, expected in required_vision.items():
        if vision.get(key) != expected:
            raise ValueError(f"vision_config.{key}: expected {expected!r}, got {vision.get(key)!r}")


def validate_tensor_inventory(
    inventory: dict[str, tuple[str, tuple[int, ...]]],
    expected: dict[str, tuple[int, ...]],
) -> None:
    actual = set(inventory)
    wanted = set(expected)
    if extra := sorted(actual - wanted):
        raise ValueError(f"unexpected tensor {extra[0]!r}")
    if missing := sorted(wanted - actual):
        raise ValueError(f"missing tensor {missing[0]!r}")
    for name, shape in expected.items():
        dtype, actual_shape = inventory[name]
        if dtype != "BF16":
            raise ValueError(f"{name}: dtype {dtype}, expected BF16")
        if tuple(actual_shape) != tuple(shape):
            raise ValueError(f"{name}: shape {actual_shape}, expected {shape}")


def _read_inventory(path: Path) -> dict[str, tuple[str, tuple[int, ...]]]:
    with path.open("rb") as source:
        header_len = int.from_bytes(source.read(8), "little")
        header = json.loads(source.read(header_len))
    return {
        name: (info["dtype"], tuple(info["shape"]))
        for name, info in header.items()
        if name != "__metadata__"
    }


def validate_source(model: Path) -> dict:
    config = json.loads((model / "config.json").read_text())
    validate_config(config)
    tokenizer = json.loads((model / "tokenizer.json").read_text())
    if tokenizer.get("model", {}).get("type") != "BPE":
        raise ValueError("OvisOCR3 requires a BPE tokenizer")
    if tokenizer.get("normalizer") != {"type": "NFC"}:
        raise ValueError("OvisOCR3 requires the NFC tokenizer normalizer")
    tokenizer_config = json.loads((model / "tokenizer_config.json").read_text())
    if tokenizer_config.get("add_bos_token") is not False:
        raise ValueError("OvisOCR3 requires add_bos_token=false")
    if tokenizer_config.get("tokenizer_class") != "Qwen2Tokenizer":
        raise ValueError("OvisOCR3 requires Qwen2Tokenizer")
    if tokenizer_config.get("bos_token") is not None:
        raise ValueError("OvisOCR3 requires no BOS token")
    if tokenizer_config.get("eos_token") != "<|im_end|>" or tokenizer_config.get("pad_token") != "<|endoftext|>":
        raise ValueError("OvisOCR3 requires the Qwen3.5 EOS/PAD tokens")
    special_tokens = tokenizer_config.get("added_tokens_decoder", {})
    for token_id, token in {
        248053: "<|vision_start|>",
        248054: "<|vision_end|>",
        248055: "<|vision_pad|>",
        248056: "<|image_pad|>",
        248057: "<|video_pad|>",
    }.items():
        if special_tokens.get(str(token_id), {}).get("content") != token:
            raise ValueError(f"tokenizer special token {token_id} is not {token!r}")
    processor = json.loads((model / "preprocessor_config.json").read_text())
    if processor.get("image_mean") != [0.5, 0.5, 0.5] or processor.get("image_std") != [0.5, 0.5, 0.5]:
        raise ValueError("unsupported OvisOCR3 image normalization")
    weights = model / "model.safetensors"
    if not weights.is_file():
        raise FileNotFoundError(weights)
    validate_tensor_inventory(_read_inventory(weights), expected_tensor_shapes(config))
    return config


def build_commands(
    llama_cpp: Path,
    model: Path,
    outfile: Path,
    mmproj_outfile: Path,
    *,
    python: str = sys.executable,
) -> list[list[str]]:
    converter = llama_cpp / "convert_hf_to_gguf.py"
    base = [python, str(converter), str(model), "--outtype", "bf16"]
    return [
        [python, "-c", _TEXT_CONVERTER, str(llama_cpp), str(model), str(outfile)],
        base + ["--outfile", str(mmproj_outfile), "--mmproj"],
    ]


def convert(
    model: Path,
    llama_cpp: Path,
    outfile: Path,
    mmproj_outfile: Path,
    *,
    python: str = sys.executable,
) -> None:
    validate_source(model)
    if outfile.exists() or mmproj_outfile.exists():
        raise FileExistsError("refusing to overwrite an existing GGUF")
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(llama_cpp), "rev-parse", "HEAD"], text=True
        ).strip()
    except subprocess.CalledProcessError as exc:
        raise ValueError(f"cannot inspect llama.cpp checkout: {llama_cpp}") from exc
    if commit != LLAMA_CPP_COMMIT:
        raise ValueError(f"llama.cpp must be {LLAMA_CPP_COMMIT}, got {commit}")
    for command in build_commands(llama_cpp, model, outfile, mmproj_outfile, python=python):
        subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--llama-cpp", type=Path, required=True)
    parser.add_argument("--outfile", type=Path, required=True)
    parser.add_argument("--mmproj-outfile", type=Path)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    mmproj = args.mmproj_outfile or args.outfile.with_name(f"mmproj-{args.outfile.name}")
    args.outfile.parent.mkdir(parents=True, exist_ok=True)
    convert(args.model, args.llama_cpp, args.outfile, mmproj, python=args.python)
    print(args.outfile)
    print(mmproj)


if __name__ == "__main__":
    main()
