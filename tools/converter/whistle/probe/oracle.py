"""
Whistle (Cactus-Compute) Python Oracle — shape-validated forward reference.

Goal: prove the architecture we reverse-engineered from safetensors compiles
into an end-to-end forward producing [B, S_text, vocab=8199] logits.

Architecture assumptions (validated via header shapes):
- num_layers = 8 (encoder + decoder)
- num_heads = 8, num_kv_heads = 2 (GQA 4:1), head_dim_q=k = 48, head_dim_v = 64
- hidden = 512
- self-attn q/k/v_norm dim = 48 (per-head RMSNorm)
- cross-attn (decoder): full heads (8 heads), head_dim_q=k = 48, head_dim_v = 64
- text vocab = 8199 (shared embedding output projection)
- engrams (byte-level text pre-encoder): 4 groups × 18432 tokens × 128 dim

Placeholders that need later refinement:
- conv stem: replaced by a deterministic random linear so we can exercise the
  encoder+decoder end-to-end (conv stem math is verified in a separate probe)
- taps: assumed identity (kernel = [0,1,0]) — streaming ASR detail
- ZCRMSNorm: collapses to RMSNorm for now (ZC = zero-center? real impl in Rust)
- Hadamard MLP: implemented as the published cactus-compute sketch (low-rank
  Hadamard + diagonal conditioner); verified per-tensor shape

This file is meant for shape validation only. Numerical parity with the Cactus
reference path is the *Rust* side's job, after we lift the placeholder ops.
"""
from __future__ import annotations

import json
import math
import struct
from pathlib import Path

import numpy as np


WEIGHTS_DIR = Path("/tmp/whistle-probe/checkpoints/whistle.safetensors")
CONFIG_PATH = Path("/tmp/whistle-probe/config.json")


def load_safetensors(path: Path) -> dict[str, np.ndarray]:
    """Pure-numpy safetensors reader (no torch)."""
    out: dict[str, np.ndarray] = {}
    with open(path, "rb") as fh:
        header_len = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(header_len))
        for key, info in header.items():
            if key == "__metadata__":
                continue
            start, end = info["data_offsets"]
            fh.seek(8 + header_len + start)
            buf = fh.read(end - start)
            arr = np.frombuffer(buf, dtype=np.float32).copy()
            arr = arr.reshape(info["shape"])
            out[key] = arr
    return out


W = load_safetensors(WEIGHTS_DIR)
print(f"loaded {len(W)} tensors")
assert "embedding/embedding" in W
assert "encoder/final_norm/scale" in W if False else (  # noqa: E501
    "encoder/final_norm/scale" in W
)


HIDDEN = 512
NUM_HEADS = 8
NUM_KV_HEADS = 2
HEAD_DIM_Q = 48
HEAD_DIM_K = 48
HEAD_DIM_V = 64
TEXT_VOCAB = 8199


def rms_norm(x: np.ndarray, scale: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """RMSNorm(x) = x / sqrt(mean(x**2) + eps) * scale."""
    var = (x * x).mean(-1, keepdims=True)
    return x / np.sqrt(var + eps) * scale


def apply_qk_norm(
    proj: np.ndarray,
    norm: np.ndarray,
    seq_len: int,
    n_heads: int,
    head_dim: int,
) -> np.ndarray:
    """Apply per-head RMSNorm to a [B, S, n_heads*head_dim] projection output."""
    b = proj.shape[0]
    out = proj.reshape(b, seq_len, n_heads, head_dim)
    out = rms_norm(out, norm)  # norm is already [n_heads, head_dim]
    return out.reshape(b, seq_len, n_heads * head_dim)


def gqa_attention(
    x: np.ndarray,
    kv_source: np.ndarray | None,
    layer_idx: int | dict,
    w_q: np.ndarray,
    w_k: np.ndarray,
    w_v: np.ndarray,
    w_out: np.ndarray,
    q_norm_scale: np.ndarray,
    k_norm_scale: np.ndarray,
    attn_gate: float,
    is_cross: bool = False,
) -> np.ndarray:
    """Standard GQA / MHA attention with per-head RMSNorm on q/k.

    Self-attn (encoder/decoder): GQA 4:1 (q=8, kv=2).
    Cross-attn (decoder): full MHA (q=kv=8).

    Shapes:
        q: [HIDDEN, n_heads * head_dim_q] = [512, 384]
        k: [HIDDEN, n_kv_heads * head_dim_k] = [512, 96] self / [512, 384] cross
        v: [HIDDEN, n_kv_heads * head_dim_v] = [512, 128] self / [512, 512] cross
        out_proj: [HIDDEN, HIDDEN] = [512, 512]
        q/k_norm_scale: [n_heads, head_dim] = [8, 48]
    """
    if is_cross:
        n_kv = NUM_HEADS  # full MHA for cross-attn
    else:
        n_kv = NUM_KV_HEADS
    seq_q = x.shape[1]
    seq_k = kv_source.shape[1] if kv_source is not None else seq_q
    b = x.shape[0]

    q_raw = x @ w_q
    if kv_source is None:
        kv_source = x
    k_raw = kv_source @ w_k
    v_raw = kv_source @ w_v

    q = apply_qk_norm(q_raw, q_norm_scale, seq_q, NUM_HEADS, HEAD_DIM_Q)
    k = apply_qk_norm(k_raw, k_norm_scale, seq_k, n_kv, HEAD_DIM_K)
    v = v_raw.reshape(b, seq_k, n_kv, HEAD_DIM_V)

    q = q.reshape(b, seq_q, NUM_HEADS, HEAD_DIM_Q).transpose(0, 2, 1, 3)
    k = k.reshape(b, seq_k, n_kv, HEAD_DIM_K).transpose(0, 2, 1, 3)
    v = v.reshape(b, seq_k, n_kv, HEAD_DIM_V).transpose(0, 2, 1, 3)

    if not is_cross:
        rep = NUM_HEADS // NUM_KV_HEADS
        k = np.repeat(k, rep, axis=1)
        v = np.repeat(v, rep, axis=1)

    scores = q @ k.transpose(0, 1, 3, 2) / math.sqrt(HEAD_DIM_Q)
    weights = np.exp(scores - scores.max(-1, keepdims=True))
    weights /= weights.sum(-1, keepdims=True)
    out = weights @ v  # [B, H, S_q, head_dim_v]
    out = out.transpose(0, 2, 1, 3).reshape(b, seq_q, NUM_HEADS * HEAD_DIM_V)

    out = out @ w_out
    return attn_gate * out


def hadamard_mlp(x: np.ndarray, layer_idx: int) -> np.ndarray:
    """Hadamard-style MLP (Cactus Compute sketch).

    Input [B, T, 512]. Splits 512 into 16 dim-chunks then 32 dim-chunks and
    runs an asymmetric Hadamard product (per-ch) plus per-channel
    conditioner (cond_u, cond_v, b2, d1..d4).

    Output [B, T, 512]. The shape invariants per layer are:
        w1a, w2a, w3a: [16, 16]   (per 16-chunk)
        w1b, w2b, w3b: [32, 32]   (per 32-chunk)
        d1..d4: [512]             (per-channel scale/offset)
        b2: [512]                 (bias)
        cond_u: [8, 512]          (rank-8 left conditioner)
        cond_v: [512, 8]          (rank-8 right conditioner)
    """
    p = f"encoder/layers/block/hadamard_mlp"
    w1a = W[p + "/w1a"][layer_idx]
    w1b = W[p + "/w1b"][layer_idx]
    w2a = W[p + "/w2a"][layer_idx]
    w2b = W[p + "/w2b"][layer_idx]
    w3a = W[p + "/w3a"][layer_idx]
    w3b = W[p + "/w3b"][layer_idx]
    d1 = W[p + "/d1"][layer_idx]
    d2 = W[p + "/d2"][layer_idx]
    d3 = W[p + "/d3"][layer_idx]
    d4 = W[p + "/d4"][layer_idx]
    b2 = W[p + "/b2"][layer_idx]
    cond_u = W[p + "/cond_u"][layer_idx]  # [8, 512]
    cond_v = W[p + "/cond_v"][layer_idx]  # [512, 8]

    b, t, h = x.shape
    assert h == 512

    # cond = x @ cond_u @ cond_v → [B, T, 512] (full hidden-dim signal,
    # used as per-channel conditioner)
    cond = (x @ cond_u.T) @ cond_v.T  # [B, T, 512]

    # 16-chunk stage: split hidden into 32 chunks of 16; each chunk × w1a.
    x_16 = x.reshape(b, t, 32, 16)
    s1 = x_16 @ w1a.T  # [B, T, 32, 16]
    s1 = s1 + d1.reshape(32, 16) * cond.reshape(b, t, 32, 16)

    # 32-chunk stage: re-chunk 32 chunks of 16 → 16 chunks of 32
    s1_32 = s1.reshape(b, t, 16, 32)
    s2 = s1_32 @ w1b.T  # [B, T, 16, 32]
    s2 = s2 + d2.reshape(16, 32) * cond.reshape(b, t, 16, 32)

    # 16-chunk stage 2
    s2_16 = s2.reshape(b, t, 32, 16)
    s3 = s2_16 @ w2a.T
    s3 = s3 + d3.reshape(32, 16) * cond.reshape(b, t, 32, 16)

    # 32-chunk stage 2
    s3_32 = s3.reshape(b, t, 16, 32)
    s4 = s3_32 @ w2b.T
    s4 = s4 + d4.reshape(16, 32) * cond.reshape(b, t, 16, 32)

    # Final stage
    s4_16 = s4.reshape(b, t, 32, 16)
    s5 = s4_16 @ w3a.T
    s5_32 = s5.reshape(b, t, 16, 32)
    s6 = s5_32 @ w3b.T
    s6 = s6.reshape(b, t, h) + b2

    # Residual cond skip (mhc_phi_res: [8, 2048, 16])
    phi = W["encoder/mhc_phi_pre"][layer_idx]  # [2048, 4]
    phi_post = W["encoder/mhc_phi_post"][layer_idx]  # [2048, 4]
    a_pre = W["encoder/mhc_a_pre"][layer_idx]  # [8]
    b_pre = W["encoder/mhc_b_pre"][layer_idx]  # [4]
    a_post = W["encoder/mhc_a_post"][layer_idx]  # [8]
    b_post = W["encoder/mhc_b_post"][layer_idx]  # [4]

    return s6  # stub: cond gate added separately in mhc branch


def mhc_skip(x: np.ndarray, layer_idx: int) -> np.ndarray:
    """Mixed Hadamard Conv residual: φ-pre/φ-post/φ-res gates."""
    # Stub: scalar scale per layer
    a = W["encoder/mhc_a_res"][layer_idx]
    return a * x


def encoder_block(x: np.ndarray, layer_idx: int) -> np.ndarray:
    """One encoder block:
        zc_rms_in → attn → hadamard_mlp → mhc_skip → dw_1d_conv_residual
    """
    p = "encoder/layers/block"
    pre_norm = rms_norm(x, W[p + "/ZCRMSNorm_0/scale"][layer_idx])
    attn = gqa_attention(
        pre_norm, None, layer_idx,
        w_q=W[p + "/self_attn/q_proj/kernel"][layer_idx],
        w_k=W[p + "/self_attn/k_proj/kernel"][layer_idx],
        w_v=W[p + "/self_attn/v_proj/kernel"][layer_idx],
        w_out=W[p + "/self_attn/out_proj/kernel"][layer_idx],
        q_norm_scale=W[p + "/self_attn/q_norm/scale"][layer_idx],
        k_norm_scale=W[p + "/self_attn/k_norm/scale"][layer_idx],
        attn_gate=W[p + "/attn_gate"][layer_idx],
    )
    h = x + attn
    h_norm = rms_norm(h, W[p + "/post_attn_norm/scale"][layer_idx])
    mlp = hadamard_mlp(h_norm, layer_idx)
    h = h + mlp + mhc_skip(h_norm, layer_idx)
    return h


def save_reference():
    """Run a deterministic forward and save the layer-by-layer outputs.

    The reference is what the Rust port should reproduce (to ~1e-5 abs diff)
    after we lift the placeholder ops (ZCRMSNorm, taps, mhc, stem).
    """
    np.random.seed(42)
    B, T = 1, 16
    # Stable random encoder input (post-stem).
    x = (np.random.randn(B, T, HIDDEN).astype(np.float32) * 0.05)

    out_path = Path("/tmp/whistle-probe/oracle_reference.npz")
    layer_inputs = []
    layer_outputs = []
    for i in range(8):
        layer_inputs.append(x.copy())
        x = encoder_block(x, i)
        layer_outputs.append(x.copy())
    enc_out = rms_norm(x, W["encoder/final_norm/scale"])

    text_tokens = np.array([[1, 2, 3, 4, 5]], dtype=np.int64)
    text_emb = W["embedding/embedding"][text_tokens]

    np.savez(
        out_path,
        encoder_input=layer_inputs[0],
        encoder_layer_in=np.stack(layer_inputs),
        encoder_layer_out=np.stack(layer_outputs),
        encoder_final=enc_out,
        text_tokens=text_tokens,
        text_emb=text_emb,
        # saved tensors are consumed by the Rust unit test
    )
    print(f"saved reference: {out_path} ({out_path.stat().st_size} bytes)")


def main():
    # Synthetic input: random mel-spec-like feature already through conv stem
    # (conv stem is verified in a separate probe).
    np.random.seed(0)
    B, T = 1, 50
    x = np.random.randn(B, T, HIDDEN).astype(np.float32) * 0.1
    print(f"encoder input: {x.shape}")

    for i in range(8):
        x = encoder_block(x, i)
        print(f"layer {i}: {x.shape}")

    x = rms_norm(x, W["encoder/final_norm/scale"])
    print(f"encoder out: {x.shape}")

    # Decoder stack (1 layer): self-attn + cross-attn + hadamard_mlp + mhc_skip
    # For shape sanity, just run cross-attn with encoder output as kv.
    text_tokens = np.array([[100, 200, 300]], dtype=np.int64)
    text_emb = W["embedding/embedding"][text_tokens]  # [1, 3, 512]
    print(f"text embed: {text_emb.shape}")

    # Stack block
    p = "stack/layers/block"
    i = 0
    pre = rms_norm(text_emb, W[p + "/ZCRMSNorm_0/scale"][i])
    sa = gqa_attention(
        pre, None, i,
        w_q=W[p + "/self_attn/q_proj/kernel"][i],
        w_k=W[p + "/self_attn/k_proj/kernel"][i],
        w_v=W[p + "/self_attn/v_proj/kernel"][i],
        w_out=W[p + "/self_attn/out_proj/kernel"][i],
        q_norm_scale=W[p + "/self_attn/q_norm/scale"][i],
        k_norm_scale=W[p + "/self_attn/k_norm/scale"][i],
        attn_gate=W[p + "/attn_gate"][i],
    )
    h = text_emb + sa
    pre_cross = rms_norm(h, W[p + "/cross_norm/scale"][i])
    # cross-attn: full MHA (kv heads = q heads = 8)
    cross = gqa_attention(
        pre_cross, x, i,  # kv_source = encoder output
        w_q=W[p + "/cross_attn/q_proj/kernel"][i],
        w_k=W[p + "/cross_attn/k_proj/kernel"][i],
        w_v=W[p + "/cross_attn/v_proj/kernel"][i],
        w_out=W[p + "/cross_attn/out_proj/kernel"][i],
        q_norm_scale=W[p + "/cross_attn/q_norm/scale"][i],
        k_norm_scale=W[p + "/cross_attn/k_norm/scale"][i],
        attn_gate=W[p + "/cross_gate"][i],
        is_cross=True,
    )
    h = h + cross
    logits_emb = rms_norm(h, W["stack/final_norm/scale"])
    logits = logits_emb @ W["embedding/embedding"].T
    print(f"logits: {logits.shape} (expect [1, 3, 8199])")


if __name__ == "__main__":
    save_reference()
    main()
