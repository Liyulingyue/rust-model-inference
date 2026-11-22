#!/usr/bin/env python3
"""Convert a MiniMind2-MoE PyTorch checkpoint into the `minimind2moe` GGUF.

The checkpoint is the plain `torch.save` state dict produced by the pinned
reference repository (`561979c:model/model_minimind.py`). It stores the routed
experts as separate `experts.{0..3}` tensors and the always-on shared expert as
`shared_experts.0`; the runtime expects the routed experts stacked into one
tensor per projection and the shared expert under the `*_shexp` names.

The trunk matches Qwen3 (GQA, RMSNorm, NeoX RoPE, SiLU) but has no Q/K norm and
no attention/FFN bias. The embedding is tied to the LM head, so the duplicate
`lm_head.weight` is validated and then omitted.

Quantization: `f16`, `f32`, `bf16`, `q8_0`, `q4_0`. Norms and the router stay
F32. K-quants are intentionally rejected: every matrix has a row width of 640
or 1728 values, which is not divisible by the 256-value K-quant block.
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch

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

ARCH = "minimind2moe"
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

DEFAULT_CONTEXT_LENGTH = 32768
DEFAULT_ROPE_FREQ_BASE = 1_000_000.0
DEFAULT_NORM_EPS = 1e-5
DEFAULT_NUM_HEADS = 8
DEFAULT_NUM_KV_HEADS = 2

# Pinned from `561979c:model/tokenizer_config.json`. The tokenizer repository's
# current `chat_template.jinja` also supports tools; the converter keeps the
# reference training/inference template by default so token ids stay aligned
# with the oracle at the pinned commit. `--chat-template-file` overrides it.
DEFAULT_CHAT_TEMPLATE = """{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{{ '<|im_start|>system\\n' + system_message + '<|im_end|>\\n' }}{% else %}{{ '<|im_start|>system\\nYou are a helpful assistant<|im_end|>\\n' }}{% endif %}{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<|im_start|>user\\n' + content + '<|im_end|>\\n<|im_start|>assistant\\n' }}{% elif message['role'] == 'assistant' %}{{ content + '<|im_end|>' + '\\n' }}{% endif %}{% endfor %}"""


def _require(state: Mapping[str, torch.Tensor], name: str) -> torch.Tensor:
    value = state.get(name)
    if value is None:
        raise KeyError(f"missing checkpoint tensor: {name}")
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name}: expected torch.Tensor, got {type(value).__name__}")
    return value


def _f32(tensor: torch.Tensor) -> np.ndarray:
    return np.ascontiguousarray(tensor.detach().cpu().to(torch.float32).numpy())


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


def _read_tokenizer(
    tokenizer_dir: Path, vocab_size: int,
) -> tuple[dict, list[str], list[int], list[str]]:
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
            types[index] = 3 if entry.get("special") else 4
    merges = [" ".join(pair) for pair in model.get("merges", [])]

    config_path = tokenizer_dir / "tokenizer_config.json"
    config = (
        json.loads(config_path.read_text(encoding="utf-8"))
        if config_path.is_file()
        else {}
    )
    return config, tokens, types, merges


def _resolve_token_id(config: dict, tokens: list[str], key: str) -> int:
    value = config.get(key)
    if not isinstance(value, str):
        raise ValueError(f"tokenizer_config.json is missing string {key}")
    try:
        return tokens.index(value)
    except ValueError as error:
        raise ValueError(f"{key}={value!r} is not in tokenizer.ggml.tokens") from error


def _same_tensor(left: torch.Tensor, right: torch.Tensor) -> bool:
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    return bool(torch.equal(left, right))


def _infer_layout(
    state: Mapping[str, torch.Tensor], num_heads: int, num_kv_heads: int,
) -> dict:
    embed = _require(state, "model.embed_tokens.weight")
    if embed.ndim != 2:
        raise ValueError(f"embed_tokens shape {tuple(embed.shape)} is not a matrix")
    vocab_size, n_embd = embed.shape
    layer_indexes = sorted(
        {
            int(key.split(".")[2])
            for key in state
            if key.startswith("model.layers.") and ".self_attn." in key
        }
    )
    if not layer_indexes or layer_indexes != list(range(len(layer_indexes))):
        raise ValueError(f"non-contiguous or missing layer indexes: {layer_indexes}")
    n_layer = len(layer_indexes)
    if n_embd % num_heads:
        raise ValueError(f"embedding_length {n_embd} is not divisible by {num_heads} heads")
    if num_heads % num_kv_heads:
        raise ValueError(f"{num_heads} heads is not divisible by {num_kv_heads} KV heads")
    head_dim = n_embd // num_heads
    kv_dim = head_dim * num_kv_heads

    expert_count = 0
    expert_ffn = 0
    shared_ffn = 0
    for layer in range(n_layer):
        prefix = f"model.layers.{layer}"
        q = _require(state, f"{prefix}.self_attn.q_proj.weight")
        k = _require(state, f"{prefix}.self_attn.k_proj.weight")
        v = _require(state, f"{prefix}.self_attn.v_proj.weight")
        o = _require(state, f"{prefix}.self_attn.o_proj.weight")
        expected = {
            "q_proj": (n_embd, n_embd),
            "k_proj": (kv_dim, n_embd),
            "v_proj": (kv_dim, n_embd),
            "o_proj": (n_embd, n_embd),
        }
        for name, tensor in (("q_proj", q), ("k_proj", k), ("v_proj", v), ("o_proj", o)):
            if tuple(tensor.shape) != expected[name]:
                raise ValueError(
                    f"{prefix}.self_attn.{name}.weight shape {tuple(tensor.shape)} "
                    f"!= {expected[name]}"
                )
        for norm in ("input_layernorm", "post_attention_layernorm"):
            if tuple(_require(state, f"{prefix}.{norm}.weight").shape) != (n_embd,):
                raise ValueError(f"{prefix}.{norm}.weight does not have {n_embd} values")

        router = _require(state, f"{prefix}.mlp.gate.weight")
        if router.ndim != 2 or router.shape[1] != n_embd:
            raise ValueError(f"{prefix}.mlp.gate.weight shape {tuple(router.shape)} is invalid")
        layer_experts = int(router.shape[0])
        if expert_count == 0:
            expert_count = layer_experts
        elif layer_experts != expert_count:
            raise ValueError(f"{prefix}: expert count {layer_experts} != {expert_count}")
        for expert in range(expert_count):
            gate = _require(state, f"{prefix}.mlp.experts.{expert}.gate_proj.weight")
            up = _require(state, f"{prefix}.mlp.experts.{expert}.up_proj.weight")
            down = _require(state, f"{prefix}.mlp.experts.{expert}.down_proj.weight")
            if layer == 0 and expert == 0:
                expert_ffn = int(gate.shape[0])
            if tuple(gate.shape) != (expert_ffn, n_embd):
                raise ValueError(f"{prefix}.mlp.experts.{expert}.gate_proj has shape {tuple(gate.shape)}")
            if tuple(up.shape) != (expert_ffn, n_embd):
                raise ValueError(f"{prefix}.mlp.experts.{expert}.up_proj has shape {tuple(up.shape)}")
            if tuple(down.shape) != (n_embd, expert_ffn):
                raise ValueError(f"{prefix}.mlp.experts.{expert}.down_proj has shape {tuple(down.shape)}")
        gate = _require(state, f"{prefix}.mlp.shared_experts.0.gate_proj.weight")
        up = _require(state, f"{prefix}.mlp.shared_experts.0.up_proj.weight")
        down = _require(state, f"{prefix}.mlp.shared_experts.0.down_proj.weight")
        if layer == 0:
            shared_ffn = int(gate.shape[0])
        if tuple(gate.shape) != (shared_ffn, n_embd):
            raise ValueError(f"{prefix}.mlp.shared_experts.0.gate_proj has shape {tuple(gate.shape)}")
        if tuple(up.shape) != (shared_ffn, n_embd):
            raise ValueError(f"{prefix}.mlp.shared_experts.0.up_proj has shape {tuple(up.shape)}")
        if tuple(down.shape) != (n_embd, shared_ffn):
            raise ValueError(f"{prefix}.mlp.shared_experts.0.down_proj has shape {tuple(down.shape)}")

    if tuple(_require(state, "model.norm.weight").shape) != (n_embd,):
        raise ValueError(f"model.norm.weight does not have {n_embd} values")
    if "lm_head.weight" in state:
        lm_head = _require(state, "lm_head.weight")
        if not _same_tensor(embed, lm_head):
            raise ValueError("lm_head.weight differs from embed_tokens; tied embedding expected")

    return {
        "vocab_size": int(vocab_size),
        "n_embd": int(n_embd),
        "n_layer": n_layer,
        "n_head": int(num_heads),
        "n_head_kv": int(num_kv_heads),
        "n_embd_head": int(head_dim),
        "expert_count": int(expert_count),
        "expert_ffn": int(expert_ffn),
        "shared_expert_ffn": int(shared_ffn),
    }


def _validate_state_keys(state: Mapping[str, torch.Tensor], layout: dict) -> None:
    expected = {
        "model.embed_tokens.weight",
        "model.norm.weight",
    }
    allowed_extra = {"lm_head.weight"}
    for layer in range(layout["n_layer"]):
        prefix = f"model.layers.{layer}"
        expected.update(
            {
                f"{prefix}.self_attn.q_proj.weight",
                f"{prefix}.self_attn.k_proj.weight",
                f"{prefix}.self_attn.v_proj.weight",
                f"{prefix}.self_attn.o_proj.weight",
                f"{prefix}.input_layernorm.weight",
                f"{prefix}.post_attention_layernorm.weight",
                f"{prefix}.mlp.gate.weight",
                f"{prefix}.mlp.shared_experts.0.gate_proj.weight",
                f"{prefix}.mlp.shared_experts.0.up_proj.weight",
                f"{prefix}.mlp.shared_experts.0.down_proj.weight",
            }
        )
        for expert in range(layout["expert_count"]):
            expected.update(
                {
                    f"{prefix}.mlp.experts.{expert}.gate_proj.weight",
                    f"{prefix}.mlp.experts.{expert}.up_proj.weight",
                    f"{prefix}.mlp.experts.{expert}.down_proj.weight",
                }
            )
    keys = set(state)
    missing = expected - keys
    extra = keys - expected - allowed_extra
    if missing:
        raise ValueError(f"missing checkpoint tensors: {sorted(missing)}")
    if extra:
        raise ValueError(f"unexpected checkpoint tensors: {sorted(extra)}")


def write_gguf(
    state: Mapping[str, torch.Tensor],
    tokenizer_dir: Path,
    out_path: Path,
    *,
    quant: str = "f16",
    num_heads: int = DEFAULT_NUM_HEADS,
    num_kv_heads: int = DEFAULT_NUM_KV_HEADS,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    rope_freq_base: float = DEFAULT_ROPE_FREQ_BASE,
    norm_eps: float = DEFAULT_NORM_EPS,
    name: str = "MiniMind2-MoE",
    chat_template: str | None = None,
) -> None:
    if quant not in QUANT_MODES:
        raise ValueError(f"unknown quantization mode {quant!r}; expected one of {QUANT_MODES}")
    layout = _infer_layout(state, num_heads, num_kv_heads)
    _validate_state_keys(state, layout)
    config, tokens, types, merges = _read_tokenizer(tokenizer_dir, layout["vocab_size"])
    if chat_template is None:
        chat_template = DEFAULT_CHAT_TEMPLATE
    bos_id = _resolve_token_id(config, tokens, "bos_token")
    eos_id = _resolve_token_id(config, tokens, "eos_token")
    add_bos = bool(config.get("add_bos_token", False))
    add_eos = bool(config.get("add_eos_token", False))

    embed = _require(state, "model.embed_tokens.weight")
    writer = GgufWriter(out_path)
    writer.add_meta("general.architecture", ARCH)
    writer.add_meta("general.name", name)
    writer.add_meta("general.basename", name)
    writer.add_meta("general.file_type", QUANT_FILE_TYPE[quant])
    writer.add_meta("general.quantization_version", 2)

    writer.add_meta(f"{ARCH}.embedding_length", layout["n_embd"])
    writer.add_meta(f"{ARCH}.block_count", layout["n_layer"])
    writer.add_meta(f"{ARCH}.context_length", context_length)
    writer.add_meta(f"{ARCH}.feed_forward_length", layout["expert_ffn"])
    writer.add_meta(f"{ARCH}.attention.head_count", layout["n_head"])
    writer.add_meta(f"{ARCH}.attention.head_count_kv", layout["n_head_kv"])
    writer.add_meta(f"{ARCH}.attention.key_length", layout["n_embd_head"])
    writer.add_meta(f"{ARCH}.attention.value_length", layout["n_embd_head"])
    writer.add_meta(f"{ARCH}.attention.layer_norm_rms_epsilon", norm_eps)
    writer.add_meta(f"{ARCH}.rope.freq_base", rope_freq_base)
    writer.add_meta(f"{ARCH}.expert_count", layout["expert_count"])
    writer.add_meta(f"{ARCH}.expert_used_count", 2)
    writer.add_meta(f"{ARCH}.expert_feed_forward_length", layout["expert_ffn"])
    writer.add_meta(
        f"{ARCH}.expert_shared_feed_forward_length", layout["shared_expert_ffn"],
    )
    writer.add_meta(f"{ARCH}.vocab_size", layout["vocab_size"])

    writer.add_meta("tokenizer.ggml.model", "gpt2")
    writer.add_meta("tokenizer.ggml.pre", "qwen2")
    writer.add_meta("tokenizer.ggml.tokens", tokens)
    writer.add_meta("tokenizer.ggml.token_type", types)
    writer.add_meta("tokenizer.ggml.merges", merges)
    writer.add_meta("tokenizer.ggml.bos_token_id", bos_id)
    writer.add_meta("tokenizer.ggml.eos_token_id", eos_id)
    writer.add_meta("tokenizer.ggml.add_bos_token", add_bos)
    writer.add_meta("tokenizer.ggml.add_eos_token", add_eos)
    writer.add_meta("tokenizer.chat_template", chat_template)

    _emit_matrix(writer, "token_embd.weight", _f32(embed), quant)

    for layer in range(layout["n_layer"]):
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

        router = _require(state, f"{prefix}.mlp.gate.weight")
        _emit_f32(writer, f"{block}.ffn_gate_inp.weight", _f32(router))

        for projection in ("gate", "up", "down"):
            experts = [
                _f32(_require(
                    state, f"{prefix}.mlp.experts.{expert}.{projection}_proj.weight",
                ))
                for expert in range(layout["expert_count"])
            ]
            stacked = np.stack(experts, axis=0)
            _emit_matrix(writer, f"{block}.ffn_{projection}_exps.weight", stacked, quant)
            shared = _f32(_require(
                state, f"{prefix}.mlp.shared_experts.0.{projection}_proj.weight",
            ))
            _emit_matrix(writer, f"{block}.ffn_{projection}_shexp.weight", shared, quant)

    _emit_f32(writer, "output_norm.weight", _f32(_require(state, "model.norm.weight")))

    writer.metadata = [
        (key, model_metadata_value(value)) for key, value in writer.metadata
    ]
    writer.write()


def load_checkpoint(path: Path) -> Mapping[str, torch.Tensor]:
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(loaded, Mapping):
        raise ValueError(f"{path}: checkpoint is not a state dict")
    return loaded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--outfile", type=Path, required=True)
    parser.add_argument("--quant", choices=QUANT_MODES, default="f16")
    parser.add_argument("--num-attention-heads", type=int, default=DEFAULT_NUM_HEADS)
    parser.add_argument("--num-key-value-heads", type=int, default=DEFAULT_NUM_KV_HEADS)
    parser.add_argument("--context-length", type=int, default=DEFAULT_CONTEXT_LENGTH)
    parser.add_argument("--rope-freq-base", type=float, default=DEFAULT_ROPE_FREQ_BASE)
    parser.add_argument("--norm-eps", type=float, default=DEFAULT_NORM_EPS)
    parser.add_argument("--name", default="MiniMind2-MoE")
    parser.add_argument("--chat-template-file", type=Path)
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")
    if not args.tokenizer_dir.is_dir():
        parser.error(f"tokenizer directory does not exist: {args.tokenizer_dir}")
    if args.outfile.exists():
        parser.error(f"output already exists: {args.outfile}")
    chat_template = None
    if args.chat_template_file is not None:
        chat_template = args.chat_template_file.read_text(encoding="utf-8")

    state = load_checkpoint(args.checkpoint)
    write_gguf(
        state,
        args.tokenizer_dir,
        args.outfile,
        quant=args.quant.lower(),
        num_heads=args.num_attention_heads,
        num_kv_heads=args.num_key_value_heads,
        context_length=args.context_length,
        rope_freq_base=args.rope_freq_base,
        norm_eps=args.norm_eps,
        name=args.name,
        chat_template=chat_template,
    )
    print(f"wrote {args.outfile} ({args.outfile.stat().st_size // 1024 // 1024} MiB)")


if __name__ == "__main__":
    main()