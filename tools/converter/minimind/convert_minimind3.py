#!/usr/bin/env python3
"""Convert official MiniMind-3 dense/MoE safetensors into Qwen3 GGUF files.

MiniMind-3 publishes Hugging Face checkpoints (`Qwen3ForCausalLM`,
`Qwen3MoeForCausalLM`). The dense trunk maps to the `qwen3` GGUF architecture,
and the routed-expert variant maps to `qwen3moe`. Both checkpoints have Q/K
RMSNorm and tied input/output embeddings. The MoE checkpoint has four routed
experts, top-1 routing and no shared expert.

Quantization: `f16`, `f32`, `bf16`, `q8_0`, `q4_0`. Norms, Q/K norms and the
router stay F32. K-quants are rejected because the 2432-wide expert rows are not
divisible by the 256-value K-quant block.
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from tools.converter.utils.gguf import (
    GGML_BF16,
    GGML_F32,
    GGML_F16,
    GGML_Q4_0,
    GGML_Q8_0,
    GgufWriter,
    f32_to_bf16,
    f32_to_f16,
    gguf_dims,
    model_metadata_value,
    quantize_q4_0,
    quantize_q8_0,
)

QUANT_MODES = ("f16", "f32", "bf16", "q8_0", "q4_0")
QUANT_FILE_TYPE = {
    "f32": 0,
    "f16": 1,
    "q4_0": 2,
    "q8_0": 7,
    "bf16": 32,
}
QUANT_SUFFIX = {
    "f32": "F32",
    "f16": "F16",
    "bf16": "BF16",
    "q8_0": "Q8_0",
    "q4_0": "Q4_0",
}
QUANT_BLOCK = {
    "f32": 1,
    "f16": 1,
    "bf16": 1,
    "q8_0": 32,
    "q4_0": 32,
}
QUANT_GGML_TYPE = {
    "f32": GGML_F32,
    "f16": GGML_F16,
    "bf16": GGML_BF16,
    "q8_0": GGML_Q8_0,
    "q4_0": GGML_Q4_0,
}

ARCH_CONFIG = {
    "qwen3": {
        "model_type": "qwen3",
        "architectures": ("Qwen3ForCausalLM",),
        "name": "Minimind 3",
        "size_label": "64M",
    },
    "qwen3moe": {
        "model_type": "qwen3_moe",
        "architectures": ("Qwen3MoeForCausalLM",),
        "name": "Minimind 3 Moe",
        "size_label": "4x64M",
    },
}


def _require(state: Mapping[str, torch.Tensor], name: str) -> torch.Tensor:
    value = state.get(name)
    if value is None:
        raise KeyError(f"missing checkpoint tensor: {name}")
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name}: expected torch.Tensor, got {type(value).__name__}")
    return value


def _f32(tensor: torch.Tensor) -> np.ndarray:
    return np.ascontiguousarray(tensor.detach().cpu().to(torch.float32).numpy())


def _same_tensor(left: torch.Tensor, right: torch.Tensor) -> bool:
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    return bool(torch.equal(left, right))


def _config_int(config: Mapping[str, object], key: str) -> int:
    value = config.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"config.json requires positive integer {key}, got {value!r}")
    return value


def _read_config(model_dir: Path) -> dict:
    path = model_dir / "config.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing config.json at {path}")
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"{path}: config must be an object")
    return config


def _architecture(config: Mapping[str, object]) -> str:
    model_type = config.get("model_type")
    architectures = config.get("architectures")
    for arch, expected in ARCH_CONFIG.items():
        if model_type == expected["model_type"]:
            if architectures != list(expected["architectures"]):
                raise ValueError(
                    f"config.json architectures {architectures!r} do not match "
                    f"{list(expected['architectures'])!r}"
                )
            return arch
    raise ValueError(f"unsupported MiniMind-3 model_type {model_type!r}")


def _validate_config(config: Mapping[str, object], arch: str) -> dict:
    layout = {
        "arch": arch,
        "vocab_size": _config_int(config, "vocab_size"),
        "hidden_size": _config_int(config, "hidden_size"),
        "num_hidden_layers": _config_int(config, "num_hidden_layers"),
        "num_attention_heads": _config_int(config, "num_attention_heads"),
        "num_key_value_heads": _config_int(config, "num_key_value_heads"),
        "head_dim": _config_int(config, "head_dim"),
        "intermediate_size": _config_int(config, "intermediate_size"),
    }
    if config.get("hidden_act") != "silu":
        raise ValueError(f"config.json hidden_act must be 'silu', got {config.get('hidden_act')!r}")
    if config.get("attention_bias") is not False:
        raise ValueError("config.json attention_bias must be false")
    if config.get("tie_word_embeddings") is not True:
        raise ValueError("config.json tie_word_embeddings must be true")
    if config.get("rope_scaling") is not None:
        raise ValueError("config.json rope_scaling must be null")
    if config.get("use_sliding_window") is not False:
        raise ValueError("config.json use_sliding_window must be false")
    layer_types = config.get("layer_types")
    if layer_types is not None and (
        not isinstance(layer_types, list)
        or len(layer_types) != layout["num_hidden_layers"]
        or any(value != "full_attention" for value in layer_types)
    ):
        raise ValueError("config.json layer_types must contain only 'full_attention'")
    if layout["num_attention_heads"] % layout["num_key_value_heads"]:
        raise ValueError(
            f"{layout['num_attention_heads']} heads is not divisible by "
            f"{layout['num_key_value_heads']} KV heads"
        )
    if arch == "qwen3moe":
        layout["expert_count"] = _config_int(config, "num_experts")
        layout["expert_used_count"] = _config_int(config, "num_experts_per_tok")
        layout["expert_ffn"] = _config_int(config, "moe_intermediate_size")
        if layout["expert_used_count"] > layout["expert_count"]:
            raise ValueError("num_experts_per_tok cannot exceed num_experts")
        if config.get("mlp_only_layers") != []:
            raise ValueError("config.json mlp_only_layers must be empty")
        if config.get("shared_expert_intermediate_size") not in (None, 0):
            raise ValueError("MiniMind-3-MoE must not declare a shared expert")
    return layout


def _validate_shapes(
    state: Mapping[str, torch.Tensor], layout: Mapping[str, object],
) -> None:
    hidden = layout["hidden_size"]
    heads = layout["num_attention_heads"]
    kv_heads = layout["num_key_value_heads"]
    head_dim = layout["head_dim"]
    q_dim = heads * head_dim
    kv_dim = kv_heads * head_dim

    shape = tuple(_require(state, "model.embed_tokens.weight").shape)
    expected = (layout["vocab_size"], hidden)
    if shape != expected:
        raise ValueError(f"model.embed_tokens.weight shape {shape} != {expected}")
    if tuple(_require(state, "model.norm.weight").shape) != (hidden,):
        raise ValueError(f"model.norm.weight does not have {hidden} values")

    for layer in range(layout["num_hidden_layers"]):
        prefix = f"model.layers.{layer}"
        expected_shapes = {
            "self_attn.q_proj.weight": (q_dim, hidden),
            "self_attn.k_proj.weight": (kv_dim, hidden),
            "self_attn.v_proj.weight": (kv_dim, hidden),
            "self_attn.o_proj.weight": (hidden, q_dim),
            "self_attn.q_norm.weight": (head_dim,),
            "self_attn.k_norm.weight": (head_dim,),
            "input_layernorm.weight": (hidden,),
            "post_attention_layernorm.weight": (hidden,),
        }
        if layout["arch"] == "qwen3moe":
            expected_shapes["mlp.gate.weight"] = (layout["expert_count"], hidden)
            for expert in range(layout["expert_count"]):
                expected_shapes[f"mlp.experts.{expert}.gate_proj.weight"] = (
                    layout["expert_ffn"], hidden,
                )
                expected_shapes[f"mlp.experts.{expert}.up_proj.weight"] = (
                    layout["expert_ffn"], hidden,
                )
                expected_shapes[f"mlp.experts.{expert}.down_proj.weight"] = (
                    hidden, layout["expert_ffn"],
                )
        else:
            expected_shapes["mlp.gate_proj.weight"] = (layout["intermediate_size"], hidden)
            expected_shapes["mlp.up_proj.weight"] = (layout["intermediate_size"], hidden)
            expected_shapes["mlp.down_proj.weight"] = (hidden, layout["intermediate_size"])
        for suffix, expected in expected_shapes.items():
            name = f"{prefix}.{suffix}"
            shape = tuple(_require(state, name).shape)
            if shape != expected:
                raise ValueError(f"{name} shape {shape} != {expected}")


def _validate_state_keys(state: Mapping[str, torch.Tensor], layout: Mapping[str, object]) -> None:
    expected = {"model.embed_tokens.weight", "model.norm.weight"}
    for layer in range(layout["num_hidden_layers"]):
        prefix = f"model.layers.{layer}"
        expected.update(
            {
                f"{prefix}.self_attn.q_proj.weight",
                f"{prefix}.self_attn.k_proj.weight",
                f"{prefix}.self_attn.v_proj.weight",
                f"{prefix}.self_attn.o_proj.weight",
                f"{prefix}.self_attn.q_norm.weight",
                f"{prefix}.self_attn.k_norm.weight",
                f"{prefix}.input_layernorm.weight",
                f"{prefix}.post_attention_layernorm.weight",
            }
        )
        if layout["arch"] == "qwen3moe":
            expected.add(f"{prefix}.mlp.gate.weight")
            for expert in range(layout["expert_count"]):
                expected.update(
                    {
                        f"{prefix}.mlp.experts.{expert}.gate_proj.weight",
                        f"{prefix}.mlp.experts.{expert}.up_proj.weight",
                        f"{prefix}.mlp.experts.{expert}.down_proj.weight",
                    }
                )
        else:
            expected.update(
                {
                    f"{prefix}.mlp.gate_proj.weight",
                    f"{prefix}.mlp.up_proj.weight",
                    f"{prefix}.mlp.down_proj.weight",
                }
            )
    keys = set(state)
    missing = expected - keys
    extra = keys - expected - {"lm_head.weight"}
    if missing:
        raise ValueError(f"missing checkpoint tensors: {sorted(missing)}")
    if extra:
        raise ValueError(f"unexpected checkpoint tensors: {sorted(extra)}")


def _validate_tied_lm_head(state: Mapping[str, torch.Tensor]) -> None:
    if "lm_head.weight" not in state:
        return
    embed = _require(state, "model.embed_tokens.weight")
    lm_head = _require(state, "lm_head.weight")
    if not _same_tensor(embed, lm_head):
        raise ValueError("lm_head.weight differs from embed_tokens; tied embedding expected")


def _emit_matrix(
    writer: GgufWriter, name: str, values: np.ndarray, quant: str,
) -> None:
    if values.ndim < 1:
        raise ValueError(f"{name}: expected a matrix, got shape {values.shape}")
    block = QUANT_BLOCK[quant]
    if block > 1 and values.shape[-1] % block:
        raise ValueError(
            f"{name}: row width {values.shape[-1]} is not a multiple of the "
            f"{block}-value {quant} block"
        )
    if quant == "f32":
        payload = values.astype("<f4", copy=False).tobytes()
    elif quant == "f16":
        payload = f32_to_f16(values.tobytes())
    elif quant == "bf16":
        payload = f32_to_bf16(values.tobytes())
    elif quant == "q8_0":
        payload = quantize_q8_0(values)
    elif quant == "q4_0":
        payload = quantize_q4_0(values)
    else:
        raise ValueError(f"unknown quantization mode {quant!r}")
    writer.add_tensor(name, QUANT_GGML_TYPE[quant], gguf_dims(values.shape), payload)


def _emit_f32(writer: GgufWriter, name: str, values: np.ndarray) -> None:
    writer.add_tensor(
        name, GGML_F32, gguf_dims(values.shape), values.astype("<f4", copy=False).tobytes(),
    )


def _read_tokenizer(tokenizer_dir: Path, vocab_size: int) -> tuple[dict, list[str], list[int], list[str]]:
    tok_path = tokenizer_dir / "tokenizer.json"
    if not tok_path.is_file():
        raise FileNotFoundError(f"missing tokenizer.json at {tok_path}")
    tok_json = json.loads(tok_path.read_text(encoding="utf-8"))
    model = tok_json.get("model", {})
    if model.get("type") != "BPE":
        raise ValueError(f"expected BPE tokenizer, got {model.get('type')!r}")
    vocab = {int(index): token for token, index in model["vocab"].items()}
    added = {int(entry["id"]): entry for entry in tok_json.get("added_tokens", [])}
    tokens: list[str] = []
    for index in range(vocab_size):
        token = added[index]["content"] if index in added else vocab.get(index)
        if token is None:
            raise ValueError(f"tokenizer has no token at id {index}")
        tokens.append(token)
    types = [1] * vocab_size
    for index, entry in added.items():
        if index < vocab_size:
            content = entry["content"]
            # The official MiniMind-3 GGUF marks `<|...|>` buffer tokens as
            # control tokens even though tokenizer.json reports special=false.
            wrapped = content.startswith("<|") and content.endswith("|>")
            types[index] = 3 if entry.get("special") or wrapped else 4
    merges = [" ".join(pair) for pair in model.get("merges", [])]

    config_path = tokenizer_dir / "tokenizer_config.json"
    config = (
        json.loads(config_path.read_text(encoding="utf-8"))
        if config_path.is_file()
        else {}
    )
    return config, tokens, types, merges


def _resolve_token_id(config: Mapping[str, object], tokens: list[str], key: str) -> int:
    value = config.get(key)
    if not isinstance(value, str):
        raise ValueError(f"tokenizer_config.json is missing string {key}")
    try:
        return tokens.index(value)
    except ValueError as error:
        raise ValueError(f"{key}={value!r} is not in tokenizer.ggml.tokens") from error


def _default_chat_template(model_dir: Path, tokenizer_config: Mapping[str, object]) -> str:
    jinja = model_dir / "chat_template.jinja"
    if jinja.is_file():
        return jinja.read_text(encoding="utf-8")
    template = tokenizer_config.get("chat_template")
    if isinstance(template, str) and template:
        return template
    raise ValueError(f"{model_dir}: no chat_template.jinja or tokenizer_config chat_template")


def write_gguf(
    state: Mapping[str, torch.Tensor],
    model_dir: Path,
    out_path: Path,
    *,
    quant: str = "f16",
    name: str | None = None,
    context_length: int | None = None,
    rope_freq_base: float | None = None,
    norm_eps: float | None = None,
    chat_template: str | None = None,
) -> None:
    if quant not in QUANT_MODES:
        raise ValueError(f"unknown quantization mode {quant!r}; expected one of {QUANT_MODES}")
    config = _read_config(model_dir)
    arch = _architecture(config)
    layout = _validate_config(config, arch)
    _validate_state_keys(state, layout)
    _validate_shapes(state, layout)
    _validate_tied_lm_head(state)

    tokenizer_config, tokens, types, merges = _read_tokenizer(
        model_dir, layout["vocab_size"],
    )
    if chat_template is None:
        chat_template = _default_chat_template(model_dir, tokenizer_config)
    bos_id = _resolve_token_id(tokenizer_config, tokens, "bos_token")
    eos_id = _resolve_token_id(tokenizer_config, tokens, "eos_token")
    unknown_id = _resolve_token_id(tokenizer_config, tokens, "unk_token")
    padding_id = _resolve_token_id(tokenizer_config, tokens, "pad_token")
    add_bos = bool(tokenizer_config.get("add_bos_token", False))
    add_eos = bool(tokenizer_config.get("add_eos_token", False))

    if context_length is None:
        context_length = _config_int(config, "max_position_embeddings")
    rope_freq_base = (
        float(config["rope_theta"]) if rope_freq_base is None else rope_freq_base
    )
    norm_eps = float(config["rms_norm_eps"]) if norm_eps is None else norm_eps
    name = ARCH_CONFIG[arch]["name"] if name is None else name

    embed = _require(state, "model.embed_tokens.weight")
    writer = GgufWriter(out_path)
    writer.add_meta("general.architecture", arch)
    writer.add_meta("general.type", "model")
    writer.add_meta("general.name", name)
    writer.add_meta("general.version", "3")
    writer.add_meta("general.basename", "minimind")
    writer.add_meta("general.size_label", ARCH_CONFIG[arch]["size_label"])
    writer.add_meta("general.file_type", QUANT_FILE_TYPE[quant])
    writer.add_meta("general.quantization_version", 2)

    writer.add_meta(f"{arch}.block_count", layout["num_hidden_layers"])
    writer.add_meta(f"{arch}.context_length", context_length)
    writer.add_meta(f"{arch}.embedding_length", layout["hidden_size"])
    writer.add_meta(f"{arch}.feed_forward_length", layout["intermediate_size"])
    writer.add_meta(f"{arch}.attention.head_count", layout["num_attention_heads"])
    writer.add_meta(f"{arch}.attention.head_count_kv", layout["num_key_value_heads"])
    writer.add_meta(f"{arch}.rope.freq_base", rope_freq_base)
    writer.add_meta(f"{arch}.attention.layer_norm_rms_epsilon", norm_eps)
    writer.add_meta(f"{arch}.attention.key_length", layout["head_dim"])
    writer.add_meta(f"{arch}.attention.value_length", layout["head_dim"])
    if arch == "qwen3moe":
        writer.add_meta(f"{arch}.expert_count", layout["expert_count"])
        writer.add_meta(f"{arch}.expert_used_count", layout["expert_used_count"])
        writer.add_meta(f"{arch}.expert_feed_forward_length", layout["expert_ffn"])

    writer.add_meta("tokenizer.ggml.model", "gpt2")
    writer.add_meta("tokenizer.ggml.pre", "qwen2")
    writer.add_meta("tokenizer.ggml.tokens", tokens)
    writer.add_meta("tokenizer.ggml.token_type", types)
    writer.add_meta("tokenizer.ggml.merges", merges)
    writer.add_meta("tokenizer.ggml.bos_token_id", bos_id)
    writer.add_meta("tokenizer.ggml.eos_token_id", eos_id)
    writer.add_meta("tokenizer.ggml.unknown_token_id", unknown_id)
    writer.add_meta("tokenizer.ggml.padding_token_id", padding_id)
    writer.add_meta("tokenizer.ggml.add_bos_token", add_bos)
    writer.add_meta("tokenizer.ggml.add_eos_token", add_eos)
    writer.add_meta("tokenizer.chat_template", chat_template)

    _emit_matrix(writer, "token_embd.weight", _f32(embed), quant)

    for layer in range(layout["num_hidden_layers"]):
        prefix = f"model.layers.{layer}"
        block = f"blk.{layer}"
        _emit_f32(
            writer, f"{block}.attn_norm.weight",
            _f32(_require(state, f"{prefix}.input_layernorm.weight")),
        )
        _emit_f32(
            writer, f"{block}.ffn_norm.weight",
            _f32(_require(state, f"{prefix}.post_attention_layernorm.weight")),
        )
        _emit_f32(
            writer, f"{block}.attn_q_norm.weight",
            _f32(_require(state, f"{prefix}.self_attn.q_norm.weight")),
        )
        _emit_f32(
            writer, f"{block}.attn_k_norm.weight",
            _f32(_require(state, f"{prefix}.self_attn.k_norm.weight")),
        )
        for source, target in (
            ("self_attn.q_proj", "attn_q"),
            ("self_attn.k_proj", "attn_k"),
            ("self_attn.v_proj", "attn_v"),
            ("self_attn.o_proj", "attn_output"),
        ):
            _emit_matrix(
                writer, f"{block}.{target}.weight",
                _f32(_require(state, f"{prefix}.{source}.weight")), quant,
            )

        if arch == "qwen3moe":
            router = _require(state, f"{prefix}.mlp.gate.weight")
            _emit_f32(writer, f"{block}.ffn_gate_inp.weight", _f32(router))
            for projection in ("gate", "up", "down"):
                experts = [
                    _f32(_require(
                        state, f"{prefix}.mlp.experts.{expert}.{projection}_proj.weight",
                    ))
                    for expert in range(layout["expert_count"])
                ]
                _emit_matrix(
                    writer, f"{block}.ffn_{projection}_exps.weight",
                    np.stack(experts, axis=0), quant,
                )
        else:
            for projection in ("gate", "up", "down"):
                _emit_matrix(
                    writer, f"{block}.ffn_{projection}.weight",
                    _f32(_require(state, f"{prefix}.mlp.{projection}_proj.weight")), quant,
                )

    _emit_f32(writer, "output_norm.weight", _f32(_require(state, "model.norm.weight")))

    writer.metadata = [
        (key, model_metadata_value(value)) for key, value in writer.metadata
    ]
    writer.write()


def load_checkpoint(model_dir: Path) -> Mapping[str, torch.Tensor]:
    path = model_dir / "model.safetensors"
    if not path.is_file():
        raise FileNotFoundError(f"missing model.safetensors at {path}")
    loaded = load_file(str(path), device="cpu")
    if not isinstance(loaded, Mapping):
        raise ValueError(f"{path}: checkpoint is not a tensor mapping")
    return loaded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--outfile", type=Path, required=True)
    parser.add_argument("--quant", choices=QUANT_MODES, default="f16")
    parser.add_argument("--name")
    parser.add_argument("--context-length", type=int)
    parser.add_argument("--rope-freq-base", type=float)
    parser.add_argument("--norm-eps", type=float)
    parser.add_argument("--chat-template-file", type=Path)
    args = parser.parse_args()

    if not args.model_dir.is_dir():
        parser.error(f"model directory does not exist: {args.model_dir}")
    if args.outfile.exists():
        parser.error(f"output already exists: {args.outfile}")
    if args.chat_template_file is not None and not args.chat_template_file.is_file():
        parser.error(f"chat template does not exist: {args.chat_template_file}")

    chat_template = None
    if args.chat_template_file is not None:
        chat_template = args.chat_template_file.read_text(encoding="utf-8")

    state = load_checkpoint(args.model_dir)
    write_gguf(
        state,
        args.model_dir,
        args.outfile,
        quant=args.quant.lower(),
        name=args.name,
        context_length=args.context_length,
        rope_freq_base=args.rope_freq_base,
        norm_eps=args.norm_eps,
        chat_template=chat_template,
    )
    print(f"wrote {args.outfile} ({args.outfile.stat().st_size // 1024 // 1024} MiB)")


if __name__ == "__main__":
    main()