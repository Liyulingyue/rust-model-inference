#!/usr/bin/env python3
"""NumPy oracle for the VibeVoice ASR Qwen2.5 decoder.

The source may be the original sharded BF16 safetensors directory or the
exported Q8_0 GGUF. It prefills the exact prompt/audio rows and can dump each
layer's last-row hidden state plus the final normalized hidden and logits.

Usage:
  python3 tools/vibevoice/vibevoice_llm_oracle.py models/VibeVoice-ASR-Streaming-7B.gguf \
      --prompt-ids 2610,525,264,... [--dump /tmp/vv-oracle/llm_hidden.f32]
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dots"))
import convert_dots_tts as _dots  # noqa: E402
from convert_dots_tts import read_gguf_directory, read_gguf_tensor_bytes  # noqa: E402
from tools.vibevoice.convert_vibevoice_asr import ShardedSafetensors  # noqa: E402

# the shared reader predates Q8_0; extend its byte-size table
_orig_tensor_nbytes = _dots._tensor_nbytes


def _tensor_nbytes_with_q8_0(ggml_type: int, dims) -> int:
    if ggml_type == 8:
        import math

        elements = math.prod(dims)
        assert elements % 32 == 0
        return elements // 32 * 34
    return _orig_tensor_nbytes(ggml_type, dims)


_dots._tensor_nbytes = _tensor_nbytes_with_q8_0


def dequant_q8_0(raw: bytes, count: int) -> np.ndarray:
    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 34)
    scale = blocks[:, 0:2].copy().view(np.float16).astype(np.float32).reshape(-1)
    q = blocks[:, 2:].copy().view(np.int8).astype(np.float32).reshape(-1)
    values = q * np.repeat(scale, 32)
    return values[:count]


def load_gguf_tensor(model_path: Path, directory: dict, name: str) -> np.ndarray:
    ggml_type, dims, length = directory[name]
    raw = read_gguf_tensor_bytes(model_path, name)
    count = int(np.prod(dims))
    if ggml_type == 8:  # Q8_0
        values = dequant_q8_0(raw, count)
    elif ggml_type == 0:  # F32
        values = np.frombuffer(raw, dtype="<f4").astype(np.float32)
    else:
        raise ValueError(f"{name}: unsupported ggml type {ggml_type}")
    # gguf dims [n_in, n_out]; byte rows are n_out × n_in (1-D stays flat)
    if len(dims) == 1:
        return values
    return values.reshape(dims[1], dims[0])


_TOP_LEVEL_SAFETENSORS = {
    "token_embd.weight": "model.language_model.embed_tokens.weight",
    "output_norm.weight": "model.language_model.norm.weight",
    "output.weight": "lm_head.weight",
}

_LAYER_SAFETENSORS = {
    "attn_norm.weight": "input_layernorm.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_q.bias": "self_attn.q_proj.bias",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_k.bias": "self_attn.k_proj.bias",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_v.bias": "self_attn.v_proj.bias",
    "attn_output.weight": "self_attn.o_proj.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
}


def safetensors_name(name: str) -> str:
    if name in _TOP_LEVEL_SAFETENSORS:
        return _TOP_LEVEL_SAFETENSORS[name]
    parts = name.split(".", 2)
    if len(parts) == 3 and parts[0] == "blk" and parts[1].isdigit():
        suffix = _LAYER_SAFETENSORS.get(parts[2])
        if suffix is not None:
            return f"model.language_model.layers.{parts[1]}.{suffix}"
    raise KeyError(f"unsupported canonical tensor: {name}")


def tensor_to_f32(tensor) -> np.ndarray:
    if tensor.dtype != "BF16":
        raise ValueError(f"{tensor.name}: expected BF16, got {tensor.dtype}")
    raw = np.frombuffer(tensor.raw, dtype="<u2").astype(np.uint32)
    return (raw << np.uint32(16)).view(np.float32).reshape(tensor.shape)


class GgufSource:
    def __init__(self, path: Path):
        self.path = path
        metadata, self.directory = read_gguf_directory(path)
        self.config = {
            "hidden_size": int(metadata["qwen2.embedding_length"]),
            "num_attention_heads": int(metadata["qwen2.attention.head_count"]),
            "num_key_value_heads": int(metadata["qwen2.attention.head_count_kv"]),
            "intermediate_size": int(metadata["qwen2.feed_forward_length"]),
            "rms_norm_eps": float(metadata["qwen2.attention.layer_norm_rms_epsilon"]),
            "rope_theta": float(metadata["qwen2.rope.freq_base"]),
            "num_hidden_layers": int(metadata["qwen2.block_count"]),
        }

    def tensor(self, name: str) -> np.ndarray:
        return load_gguf_tensor(self.path, self.directory, name)

    def close(self) -> None:
        pass


class SafetensorsSource:
    def __init__(self, model_dir: Path):
        self.path = model_dir
        self.config = json.loads((model_dir / "config.json").read_text())["decoder_config"]
        self.reader = ShardedSafetensors(model_dir)

    def tensor(self, name: str) -> np.ndarray:
        return tensor_to_f32(self.reader.tensor(safetensors_name(name)))

    def close(self) -> None:
        self.reader.close()


def open_source(path: Path):
    if path.is_dir():
        return SafetensorsSource(path)
    return GgufSource(path)


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    mean_sq = np.mean(x * x, axis=-1, keepdims=True)
    return (x / np.sqrt(mean_sq + eps)) * weight


def rope_neox(x: np.ndarray, positions: np.ndarray, head_dim: int, theta: float) -> np.ndarray:
    # x: [T, heads, head_dim]
    half = head_dim // 2
    inv_freq = theta ** (-np.arange(0, half, dtype=np.float64) / half)
    angles = positions[:, None].astype(np.float64) * inv_freq[None, :]
    cos = np.cos(angles)[:, None, :]
    sin = np.sin(angles)[:, None, :]
    x1 = x[..., :half]
    x2 = x[..., half:]
    out = np.empty_like(x)
    out[..., :half] = x1 * cos - x2 * sin
    out[..., half:] = x2 * cos + x1 * sin
    return out


def assemble_input_rows(
    token_embeddings: np.ndarray,
    prompt_ids: list[int],
    audio_embeddings: np.ndarray | None = None,
    speech_start_id: int | None = None,
    speech_end_id: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if audio_embeddings is None:
        rows = token_embeddings[prompt_ids]
    else:
        if speech_start_id is None or speech_end_id is None:
            raise ValueError("speech token ids are required with audio embeddings")
        if audio_embeddings.ndim != 2 or audio_embeddings.shape[1] != token_embeddings.shape[1]:
            raise ValueError(
                f"audio embeddings must have shape [rows, {token_embeddings.shape[1]}]"
            )
        prefix = token_embeddings[prompt_ids + [speech_start_id]]
        suffix = token_embeddings[[speech_end_id]]
        rows = np.vstack([prefix, audio_embeddings, suffix])
    rows = np.ascontiguousarray(rows, dtype=np.float32)
    return rows, np.arange(rows.shape[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str)
    parser.add_argument("--prompt-ids", type=str, required=True)
    parser.add_argument("--dump", type=str, default=None)
    parser.add_argument("--layers", type=int, default=28)
    parser.add_argument("--embedding-dump", type=str, default=None)
    parser.add_argument("--speech-start-id", type=int, default=None)
    parser.add_argument("--speech-end-id", type=int, default=None)
    parser.add_argument("--dump-dir", type=str, default=None)
    args = parser.parse_args()
    model_path = Path(args.model)
    source = open_source(model_path)
    ids = [int(v) for v in args.prompt_ids.split(",")]
    config = source.config
    n_embd = int(config["hidden_size"])
    n_head = int(config["num_attention_heads"])
    n_kv = int(config["num_key_value_heads"])
    n_ff = int(config["intermediate_size"])
    eps = float(config["rms_norm_eps"])
    theta = float(config.get("rope_theta", 1_000_000.0))
    n_layer = int(config["num_hidden_layers"])
    head_dim = n_embd // n_head
    group = n_head // n_kv
    print(f"n_embd={n_embd} n_head={n_head} n_kv={n_kv} n_ff={n_ff} eps={eps} theta={theta}")

    # embeddings only for the ids we need
    dump_dir = Path(args.dump_dir) if args.dump_dir else None
    if dump_dir is not None:
        dump_dir.mkdir(parents=True, exist_ok=True)
    embed_all = source.tensor("token_embd.weight")
    audio_embeddings = None
    if args.embedding_dump:
        values = np.fromfile(args.embedding_dump, dtype="<f4")
        if values.size % n_embd:
            raise ValueError(
                f"{args.embedding_dump}: {values.size} values not divisible by {n_embd}"
            )
        audio_embeddings = values.reshape(-1, n_embd)
    x, positions = assemble_input_rows(
        embed_all,
        ids,
        audio_embeddings,
        args.speech_start_id,
        args.speech_end_id,
    )
    t = x.shape[0]
    print(f"sequence_rows={t} audio_rows={0 if audio_embeddings is None else len(audio_embeddings)}")
    del embed_all

    for layer in range(min(n_layer, args.layers)):
        prefix = f"blk.{layer}."
        attn_norm = source.tensor(prefix + "attn_norm.weight").reshape(-1)
        ffn_norm = source.tensor(prefix + "ffn_norm.weight").reshape(-1)
        wq = source.tensor(prefix + "attn_q.weight")
        wk = source.tensor(prefix + "attn_k.weight")
        wv = source.tensor(prefix + "attn_v.weight")
        wo = source.tensor(prefix + "attn_output.weight")
        q_bias = source.tensor(prefix + "attn_q.bias").reshape(-1)
        k_bias = source.tensor(prefix + "attn_k.bias").reshape(-1)
        v_bias = source.tensor(prefix + "attn_v.bias").reshape(-1)

        normed = rms_norm(x, attn_norm, eps)
        q = normed @ wq.T + q_bias
        k = normed @ wk.T + k_bias
        v = normed @ wv.T + v_bias
        del wq, wk, wv, normed
        q = q.reshape(t, n_head, head_dim)
        k = k.reshape(t, n_kv, head_dim)
        v = v.reshape(t, n_kv, head_dim)
        q = rope_neox(q, positions, head_dim, theta)
        k = rope_neox(k, positions, head_dim, theta)

        # broadcast kv heads across their q-head group: [h, T, S]
        k_expanded = np.repeat(k, group, axis=1)  # [S, n_head, head_dim]
        scores = np.einsum("thd,shd->hts", q, k_expanded) / np.sqrt(head_dim)
        mask = np.triu(np.ones((t, t), dtype=bool), 1)
        scores[:, mask] = -np.inf
        scores -= scores.max(axis=-1, keepdims=True)
        weights = np.exp(scores)
        weights /= weights.sum(axis=-1, keepdims=True)
        # per q-head: use its kv head's values
        v_expanded = np.repeat(v, group, axis=1)  # [S, n_head, head_dim]
        attn = np.einsum("hts,shd->thd", weights, v_expanded)
        attn = attn.reshape(t, n_head * head_dim)
        x = x + attn @ wo.T

        normed = rms_norm(x, ffn_norm, eps)
        w_gate = source.tensor(prefix + "ffn_gate.weight")
        w_up = source.tensor(prefix + "ffn_up.weight")
        gate = normed @ w_gate.T
        up = normed @ w_up.T
        del w_gate, w_up, normed
        ff = gate / (1.0 + np.exp(-gate)) * up
        del gate, up
        w_down = source.tensor(prefix + "ffn_down.weight")
        x = x + ff @ w_down.T
        del w_down, ff
        if dump_dir is not None:
            x[-1].astype(np.float32).tofile(dump_dir / f"vibevoice_llm_layer_{layer:02d}.f32")
        print(f"layer {layer}: hidden_norm {np.linalg.norm(x[-1]):.4f} "
              f"head {np.array2string(x[-1][:4], precision=4)}")

    output_norm = source.tensor("output_norm.weight").reshape(-1)
    hidden = rms_norm(x, output_norm, eps)
    if args.dump:
        Path(args.dump).parent.mkdir(parents=True, exist_ok=True)
        hidden.astype(np.float32).tofile(args.dump)
        print(f"dumped final hidden {hidden.shape} to {args.dump}")
    if dump_dir is not None:
        hidden[-1].astype(np.float32).tofile(dump_dir / "vibevoice_llm_normed.f32")
    lm_head = source.tensor("output.weight")
    logits = hidden[-1] @ lm_head.T
    if dump_dir is not None:
        logits.astype(np.float32).tofile(dump_dir / "vibevoice_llm_logits.f32")
    top = np.argsort(logits)[::-1][:8]
    print("top8:", [(int(i), round(float(logits[i]), 3)) for i in top])
    if dump_dir is not None:
        manifest = {
            "source": str(model_path.resolve()),
            "source_kind": "safetensors-bf16" if model_path.is_dir() else "gguf-q8_0",
            "layers": min(n_layer, args.layers),
            "hidden_size": n_embd,
            "vocab_size": int(logits.size),
            "sequence_rows": t,
            "top8": [int(index) for index in top],
        }
        (dump_dir / "vibevoice_llm_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    source.close()


if __name__ == "__main__":
    main()
