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
    """Real ZCRMSNorm: out = (1 + scale) * x / sqrt(mean(x²) + eps)."""
    rms = np.sqrt((x * x).mean(-1, keepdims=True) + eps)
    return (1.0 + scale) * x / rms


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


def _hada_blocks(n: int) -> tuple[int, int]:
    b = 1 << ((n - 1).bit_length() // 2)
    return b, n // b


def _hada_perms(n: int, seed: int) -> np.ndarray:
    """Reproduce cactus-compute/needle `_hada_perms`. The random perms are
    seeded by `_HADA_PERM_SEEDS = (11, 13)` in architecture.py."""
    return np.random.RandomState(seed).permutation(n)


def _kron_apply(z: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """`z` shape `[..., a_dim, b_dim]`. Return `a.T @ z @ b` reshaped to
    `[..., a_dim * b_dim]`. Equivalent to (a ⊗ b) @ vec(z)."""
    lead = z.shape[:-2]
    z = z.reshape(*lead, a.shape[0], b.shape[0])
    out = np.einsum("...ij,ik,jl->...kl", z, a, b)
    return out.reshape(*lead, a.shape[0] * b.shape[0])


def hadamard_mlp(x: np.ndarray, layer_idx: int) -> np.ndarray:
    """Real HadamardMLP from cactus-compute/needle/model/architecture.py.

    Stages:
      z = pad(x)                                  # n - d_model, usually 0
      z = kron_apply(d1 * z, *factors[0])[..., p1]
      z = kron_apply(silu(d2 * cond * z + b2), *factors[1])[..., p2]
      z = kron_apply(d3 * z, *factors[2])
      return (d4 * z)[..., :d_model]

    `cond = 1 + softmax(x @ cond_v) @ cond_u`.
    """
    p = "encoder/layers/block/hadamard_mlp"
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

    ba, bb = _hada_blocks(x.shape[-1])  # (16, 32) for d=512
    p1 = _hada_perms(x.shape[-1], 11)
    p2 = _hada_perms(x.shape[-1], 13)

    b, t, h = x.shape

    # cond = 1 + softmax(x @ cond_v) @ cond_u
    cond_logits = x @ cond_v  # [B, T, 8]
    cond_softmax = np.exp(cond_logits - cond_logits.max(-1, keepdims=True))
    cond_softmax /= cond_softmax.sum(-1, keepdims=True)
    cond = 1.0 + cond_softmax @ cond_u  # [B, T, 512]

    z = x  # n == d_model, no pad

    # Stage 1: kron_apply(d1 * z, w1a, w1b) then permute by p1
    z1 = d1 * z
    z1 = z1.reshape(b, t, ba, bb)
    z1 = _kron_apply(z1, w1a, w1b)
    z1 = z1[..., p1]

    # Stage 2: silu(d2 * cond * z1 + b2) -> kron_apply(w2a, w2b) -> permute p2
    z2_raw = d2 * cond * z1 + b2
    z2 = z2_raw * (1.0 / (1.0 + np.exp(-z2_raw)))  # silu
    z2 = z2.reshape(b, t, ba, bb)
    z2 = _kron_apply(z2, w2a, w2b)
    z2 = z2[..., p2]

    # Stage 3: kron_apply(d3 * z2, w3a, w3b) — no permute
    z3 = d3 * z2
    z3 = z3.reshape(b, t, ba, bb)
    z3 = _kron_apply(z3, w3a, w3b)

    return d4 * z3


def _lane_offsets(layer_idx: int, n: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """`pre_off = 8 * lane - 4`, `post_off = -4 * (1 - lane)`, cycling per layer.

    Reproduces cactus-compute/needle `_lane_offsets` for the active layer.
    """
    lane = np.zeros(n, dtype=np.float32)
    lane[layer_idx % n] = 1.0
    return 8 * lane - 4, -4 * (1 - lane)


def _sinkhorn(logits, iters: int = 20) -> np.ndarray:
    """Sinkhorn-Knopp: alternating log-row/col normalisation."""
    log_K = logits.copy()
    for _ in range(iters):
        log_K = log_K - np.log(np.exp(log_K).sum(axis=-1, keepdims=True) + 1e-30)
        log_K = log_K - np.log(np.exp(log_K).sum(axis=-2, keepdims=True) + 1e-30)
    return np.exp(log_K)


def _rms_unit(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Per-token RMS unit norm (no affine): x / sqrt(mean(x²) + eps)."""
    return x / np.sqrt((x * x).mean(axis=-1, keepdims=True) + eps)


def encoder_advance(stream: np.ndarray, layer_idx: int) -> np.ndarray:
    """Real multi-lane hyper-connection (`mhc_lanes=4`) + encoder block.

    Reproduces `Stack._advance` from cactus-compute/needle/model/architecture.py,
    applied to a single encoder layer. `stream` shape `[B, T, 4, 512]`.
    Returns new stream of the same shape (cumulative across layers).
    """
    B, T, n, C = stream.shape
    assert n == 4 and C == HIDDEN
    dt = stream.dtype
    xf = stream.astype(np.float32)
    nx = _rms_unit(stream.reshape(B, T, n * C)).astype(np.float32)

    pre_off, post_off = _lane_offsets(layer_idx, n)
    # Pre-gate: hpre = sigmoid(a_pre * (nx @ phi_pre) + b_pre + pre_off)
    a_pre = W["encoder/mhc_a_pre"][layer_idx]
    phi_pre = W["encoder/mhc_phi_pre"][layer_idx]
    b_pre = W["encoder/mhc_b_pre"][layer_idx]
    hpre = 1.0 / (
        1.0 + np.exp(-(a_pre * (nx @ phi_pre) + b_pre + pre_off))
    )  # [B, T, 4]

    # u = sum(hpre[..., None] * xf, axis=2)  (weighted sum of 4 lanes)
    u = (hpre[..., None] * xf).sum(axis=2)  # [B, T, 512]

    # Inner encoder block (ZCRMSNorm → attention → ZCRMSNorm → attn_gate →
    # ZCRMSNorm → HadamardMLP → residual)
    p2 = "encoder/layers/block"
    skip = u
    pre_norm = rms_norm(u, W[p2 + "/ZCRMSNorm_0/scale"][layer_idx])
    attn = gqa_attention(
        pre_norm, None, layer_idx,
        w_q=W[p2 + "/self_attn/q_proj/kernel"][layer_idx],
        w_k=W[p2 + "/self_attn/k_proj/kernel"][layer_idx],
        w_v=W[p2 + "/self_attn/v_proj/kernel"][layer_idx],
        w_out=W[p2 + "/self_attn/out_proj/kernel"][layer_idx],
        q_norm_scale=W[p2 + "/self_attn/q_norm/scale"][layer_idx],
        k_norm_scale=W[p2 + "/self_attn/k_norm/scale"][layer_idx],
        attn_gate=W[p2 + "/attn_gate"][layer_idx],
    )
    h = skip + attn
    skip = h
    h_norm = rms_norm(h, W[p2 + "/post_attn_norm/scale"][layer_idx])
    mlp = hadamard_mlp(h_norm, layer_idx)
    h = skip + mlp

    # y = block output - input (skip-style residual)
    y = h - u

    # Post-gate: hpost = 2 * sigmoid(a_post * (nx @ phi_post) + b_post + post_off)
    a_post = W["encoder/mhc_a_post"][layer_idx]
    phi_post = W["encoder/mhc_phi_post"][layer_idx]
    b_post = W["encoder/mhc_b_post"][layer_idx]
    hpost = 2.0 / (
        1.0 + np.exp(-(a_post * (nx @ phi_post) + b_post + post_off))
    )  # [B, T, 4]

    # Res gate: sinkhorn(a_res * (nx @ phi_res).reshape(n, n) + b_res)
    a_res = W["encoder/mhc_a_res"][layer_idx]
    phi_res = W["encoder/mhc_phi_res"][layer_idx]
    b_res = W["encoder/mhc_b_res"][layer_idx]
    res_logits = a_res * (nx @ phi_res).reshape(B, T, n, n) + b_res
    hres = _sinkhorn(res_logits)  # [B, T, 4, 4]

    # new_stream = einsum("btij,btjc->btic", hres, xf) + hpost[..., None] * y[..., None, :]
    new_xf = np.einsum("btij,btjc->btic", hres, xf) + hpost[..., None] * y[..., None, :]
    return new_xf.astype(dt)


def mhc_skip(x: np.ndarray, layer_idx: int) -> np.ndarray:
    """Legacy placeholder; use `encoder_advance` for the real algorithm."""
    a = W["encoder/mhc_a_res"][layer_idx]
    return a * x


def encoder_block(x: np.ndarray, layer_idx: int) -> np.ndarray:
    """Real encoder block on a fresh `[B, T, 512]` input.

    For parity with the multi-lane hyper-connection each call broadcasts
    to `[B, T, 4, 512]`, runs `_advance`, and returns `mean(stream, axis=2)`.
    """
    B, T, H = x.shape
    stream = np.broadcast_to(x[:, :, None, :], (B, T, 4, H)).astype(np.float32)
    return encoder_advance(stream, layer_idx).mean(axis=2)


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
    # Cumulative multi-lane stream: broadcast initial x to [B,T,4,512]
    stream = np.broadcast_to(
        x[:, :, None, :], (B, T, 4, HIDDEN)
    ).astype(np.float32)
    for i in range(8):
        # What Rust sees as `layer_in` is the previous stream's mean.
        layer_inputs.append(stream.mean(axis=2).copy())
        stream = encoder_advance(stream, i)
        layer_outputs.append(stream.mean(axis=2).copy())
    x_final = stream.mean(axis=2)
    enc_out = rms_norm(x_final, W["encoder/final_norm/scale"])

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
