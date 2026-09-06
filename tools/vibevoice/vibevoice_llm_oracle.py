#!/usr/bin/env python3
"""Numpy oracle for the VibeVoice ASR LLM half (Qwen2.5-7B from the exported
gguf, Q8_0 dequantized). Prefills the exact prompt tokens and prints the
last-position logits so the Rust session can be compared.

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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dots"))
import convert_dots_tts as _dots  # noqa: E402
from convert_dots_tts import read_gguf_directory, read_gguf_tensor_bytes  # noqa: E402

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


def load_tensor(model_path: Path, name: str) -> np.ndarray:
    _metadata, directory = read_gguf_directory(model_path)
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
    parser.add_argument("gguf", type=str)
    parser.add_argument("--prompt-ids", type=str, required=True)
    parser.add_argument("--dump", type=str, default=None)
    parser.add_argument("--layers", type=int, default=28)
    parser.add_argument("--embedding-dump", type=str, default=None)
    parser.add_argument("--speech-start-id", type=int, default=None)
    parser.add_argument("--speech-end-id", type=int, default=None)
    args = parser.parse_args()
    model_path = Path(args.gguf)
    ids = [int(v) for v in args.prompt_ids.split(",")]
    metadata, directory = read_gguf_directory(model_path)
    n_embd = int(metadata["qwen2.embedding_length"])
    n_head = int(metadata["qwen2.attention.head_count"])
    n_kv = int(metadata["qwen2.attention.head_count_kv"])
    n_ff = int(metadata["qwen2.feed_forward_length"])
    eps = float(metadata["qwen2.attention.layer_norm_rms_epsilon"])
    theta = float(metadata["qwen2.rope.freq_base"])
    n_layer = int(metadata["qwen2.block_count"])
    head_dim = n_embd // n_head
    group = n_head // n_kv
    print(f"n_embd={n_embd} n_head={n_head} n_kv={n_kv} n_ff={n_ff} eps={eps} theta={theta}")

    # embeddings only for the ids we need
    embed_all = load_tensor(model_path, "token_embd.weight")
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
        attn_norm = load_tensor(model_path, prefix + "attn_norm.weight").reshape(-1)
        ffn_norm = load_tensor(model_path, prefix + "ffn_norm.weight").reshape(-1)
        wq = load_tensor(model_path, prefix + "attn_q.weight")
        wk = load_tensor(model_path, prefix + "attn_k.weight")
        wv = load_tensor(model_path, prefix + "attn_v.weight")
        wo = load_tensor(model_path, prefix + "attn_output.weight")
        q_bias = load_tensor(model_path, prefix + "attn_q.bias").reshape(-1)
        k_bias = load_tensor(model_path, prefix + "attn_k.bias").reshape(-1)
        v_bias = load_tensor(model_path, prefix + "attn_v.bias").reshape(-1)

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
        w_gate = load_tensor(model_path, prefix + "ffn_gate.weight")
        w_up = load_tensor(model_path, prefix + "ffn_up.weight")
        gate = normed @ w_gate.T
        up = normed @ w_up.T
        del w_gate, w_up, normed
        ff = gate / (1.0 + np.exp(-gate)) * up
        del gate, up
        w_down = load_tensor(model_path, prefix + "ffn_down.weight")
        x = x + ff @ w_down.T
        del w_down, ff
        print(f"layer {layer}: hidden_norm {np.linalg.norm(x[-1]):.4f} "
              f"head {np.array2string(x[-1][:4], precision=4)}")

    output_norm = load_tensor(model_path, "output_norm.weight").reshape(-1)
    hidden = rms_norm(x, output_norm, eps)
    if args.dump:
        Path(args.dump).parent.mkdir(parents=True, exist_ok=True)
        hidden.astype(np.float32).tofile(args.dump)
        print(f"dumped final hidden {hidden.shape} to {args.dump}")
    lm_head = load_tensor(model_path, "output.weight")
    logits = hidden[-1] @ lm_head.T
    top = np.argsort(logits)[::-1][:8]
    print("top8:", [(int(i), round(float(logits[i]), 3)) for i in top])


if __name__ == "__main__":
    main()
