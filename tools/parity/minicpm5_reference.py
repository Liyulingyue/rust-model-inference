#!/usr/bin/env python
"""MiniCPM5-1B step-by-step numpy reference forward (f64 ground truth).

Replays the exact token-by-token forward the Rust `llama` trunk performs
(chat-template prompt, one token per step), then compares every dumped
intermediate against the Rust `RUST_LLAMA_DEBUG_OUTFILE` trace.

Usage:
  models/.venv/bin/python tools/parity/minicpm5_reference.py \
      models/MiniCPM5-1B-GGUF/MiniCPM5-1B-Q8_0.gguf /tmp/minicpm_parity/rust_trace.txt
"""
import sys
import numpy as np
import gguf

EPS = np.float32(1e-6)
FREQ_BASE = np.float32(5000000.0)


def dequant(data: np.ndarray, ggml_type) -> np.ndarray:
    """Dequantize a GGUF tensor to float32."""
    raw = np.ascontiguousarray(data.view(np.uint8)) if data.dtype != np.uint8 else data
    if int(ggml_type) == 0:  # F32
        return raw.view(np.float32).astype(np.float64)
    if int(ggml_type) == 8:  # Q8_0
        n = raw.size // 34
        blocks = raw.reshape(n, 34)
        scales = blocks[:, :2].copy().view(np.float16).astype(np.float64)
        qs = blocks[:, 2:].copy().view(np.int8).astype(np.float64)
        return (scales * qs).reshape(n * 32)
    raise ValueError(f"unsupported ggml type {ggml_type}")


def load_tensors(path):
    reader = gguf.GGUFReader(path)
    out = {}
    for t in reader.tensors:
        data, shape = dequant(t.data, t.tensor_type), tuple(reversed(t.shape))
        out[t.name] = (data.reshape(shape), shape)
    return out


def rope_norm(x, pos, head_dim, heads):
    """Interleaved-pair (GGML ROPE_TYPE_NORM) rotation — the style used by
    the classic `llama` GGUF arch (incl. MiniCPM5) and the Rust `rope_norm`."""
    half = head_dim // 2
    theta_scale = float(np.float32(FREQ_BASE) ** (-2.0 / head_dim))
    theta = float(np.float32(pos))
    x = x.reshape(heads, head_dim)
    for i in range(half):
        t32 = np.float32(theta)
        cos_a, sin_a = float(np.cos(t32)), float(np.sin(t32))
        x0 = x[:, 2 * i].copy()
        x1 = x[:, 2 * i + 1].copy()
        x[:, 2 * i] = x0 * cos_a - x1 * sin_a
        x[:, 2 * i + 1] = x0 * sin_a + x1 * cos_a
        theta *= theta_scale
    return x.reshape(-1)


def rms_norm(x, w):
    mean_sq = np.mean(x * x)
    scale = 1.0 / np.sqrt(np.float64(np.float32(mean_sq)) + np.float64(EPS))
    return x * scale * w


def silu(x):
    return x / (1.0 + np.exp(-x))


def attend(K, V, q, n_head, group_size, head_dim_k, head_dim_v, n_embd_q):
    """Causal attention over the full KV history. K/V: list-per-step of (n_kv_head, dim)."""
    attn = np.empty(n_embd_q)
    for h in range(n_head):
        kv_h = h // group_size
        Km = np.stack([kt[kv_h] for kt in K])
        Vm = np.stack([vt[kv_h] for vt in V])
        qh = q.reshape(n_head, head_dim_k)[h]
        scores = Km @ qh / np.sqrt(float(head_dim_k))
        p = np.exp(scores - scores.max())
        p /= p.sum()
        attn[h * head_dim_v:(h + 1) * head_dim_v] = p @ Vm
    return attn


def main():
    model_path, trace_path = sys.argv[1], sys.argv[2]
    tensors = load_tensors(model_path)

    n_vocab, n_embd = tensors["token_embd.weight"][1]
    n_layer = 24
    n_head, n_kv_head = 16, 2
    head_dim_k = 128
    head_dim_v = 128
    n_embd_q = n_head * head_dim_k
    n_embd_gqa = n_kv_head * head_dim_v
    group_size = n_head // n_kv_head

    ids = [0, 130072, 8448, 220, 36417, 33, 813, 457, 447, 52, 130073, 220,
           130072, 130071, 220, 8, 220]

    # Parse Rust trace: "[step=S il=L label] v1 v2 ..."
    rust = {}
    with open(trace_path) as f:
        for line in f:
            if not line.startswith("[step="):
                continue
            head, rest = line.split("]", 1)
            parts = head[1:].split()
            step, il = int(parts[0][5:]), int(parts[1][3:])
            label = parts[2]
            rust[(step, il, label)] = np.array([float(v) for v in rest.split()])
    steps = sorted({k[0] for k in rust})
    final_step = max(steps)
    print(f"rust trace: steps={steps}, comparing at step={final_step}")

    kv = {}  # layer -> (K list, V list)
    x = None
    for step, tok in enumerate(ids):
        x = tensors["token_embd.weight"][0][tok].copy()
        for l in range(n_layer):
            W = {n: tensors[f"blk.{l}.{n}.weight"][0] for n in
                 ["attn_q", "attn_k", "attn_v", "attn_output",
                  "ffn_gate", "ffn_up", "ffn_down"]}
            attn_norm_w = tensors[f"blk.{l}.attn_norm.weight"][0]
            ffn_norm_w = tensors[f"blk.{l}.ffn_norm.weight"][0]

            normed = rms_norm(x, attn_norm_w)
            q = normed @ W["attn_q"].T
            k = normed @ W["attn_k"].T
            v = normed @ W["attn_v"].T

            q = rope_norm(q, step, head_dim_k, n_head)
            k = rope_norm(k, step, head_dim_k, n_kv_head)

            # mimic F16 KV cache rounding (Rust KvFormat::F16)
            k = k.astype(np.float16).astype(np.float64)
            v = v.astype(np.float16).astype(np.float64)
            K, V = kv.setdefault(l, ([], []))
            K.append(k.reshape(n_kv_head, head_dim_k))
            V.append(v.reshape(n_kv_head, head_dim_v))

            attn = attend(K, V, q, n_head, group_size, head_dim_k, head_dim_v, n_embd_q)

            attn_proj = attn @ W["attn_output"].T
            x = x + attn_proj

            normed2 = rms_norm(x, ffn_norm_w)
            up_out = normed2 @ W["ffn_up"].T     # note: Rust label swap
            gate_out = normed2 @ W["ffn_gate"].T
            h = silu(gate_out) * up_out
            down = h @ W["ffn_down"].T
            x = x + down

        if step == final_step:
            break

    norm_w = tensors["output_norm.weight"][0]
    normed = rms_norm(x, norm_w)
    logits = normed @ tensors["output.weight"][0].T

    # ---- compare ----
    def cmp(label, ref, il=None, tok_step=None):
        key = (tok_step, il, label) if il is not None else None
        if key is None or key not in rust:
            # ffn label swap fix for gate/up raw buffers
            return
        r = rust[key]
        d = np.abs(r - ref)
        rel = d.max() / max(np.abs(ref).max(), 1e-9)
        flag = "  <<<" if d.max() > 1e-2 else ""
        print(f"  L{il:02d} {label:20s} max_abs_diff={d.max():.6f} "
              f"rel={rel:.2e} argmax_rust={int(np.argmax(r))} argmax_ref={int(np.argmax(ref))}{flag}")

    print(f"\n== comparing step {final_step} (last prompt token) ==")
    tok = ids[final_step]
    x = tensors["token_embd.weight"][0][tok].copy()
    for l in range(n_layer):
        W = {n: tensors[f"blk.{l}.{n}.weight"][0] for n in
             ["attn_q", "attn_k", "attn_v", "attn_output",
              "ffn_gate", "ffn_up", "ffn_down"]}
        attn_norm_w = tensors[f"blk.{l}.attn_norm.weight"][0]
        ffn_norm_w = tensors[f"blk.{l}.ffn_norm.weight"][0]

        normed = rms_norm(x, attn_norm_w)
        cmp("attn_norm", normed, l, final_step)
        q = normed @ W["attn_q"].T
        k = normed @ W["attn_k"].T
        v = normed @ W["attn_v"].T
        q = rope_norm(q, final_step, head_dim_k, n_head)
        cmp("Qcur", q, l, final_step)
        k = rope_norm(k, final_step, head_dim_k, n_kv_head)
        cmp("Kcur", k, l, final_step)
        cmp("Vcur", v, l, final_step)

        K, V = kv[l]
        attn = attend(K, V, q, n_head, group_size, head_dim_k, head_dim_v, n_embd_q)
        cmp("attn_out", attn, l, final_step)

        attn_proj = attn @ W["attn_output"].T
        cmp("attn_proj", attn_proj, l, final_step)
        x = x + attn_proj
        cmp("ffn_inp", x, l, final_step)

        normed2 = rms_norm(x, ffn_norm_w)
        cmp("ffn_norm", normed2, l, final_step)
        # Rust: w_gate -> up_buf (dumped as ffn_up_buf_raw), w_up -> gate_buf
        # (dumped as ffn_gate_buf_raw)
        gate_out = normed2 @ W["ffn_gate"].T
        up_out = normed2 @ W["ffn_up"].T
        # Rust forward.rs: w_gate writes to up_buf (dumped as ffn_up_buf_raw),
        # w_up writes to gate_buf; silu_mul then turns gate_buf into the
        # FFN hidden h = silu(W_gate@x) * (W_up@x), dumped as ffn_gate_buf_raw.
        cmp("ffn_up_buf_raw", gate_out, l, final_step)
        h = silu(gate_out) * up_out
        cmp("ffn_gate_buf_raw", h, l, final_step)
        down = h @ W["ffn_down"].T
        cmp("down_buf", down, l, final_step)
        x = x + down
        cmp("ffn_out", x, l, final_step)
        cmp("l_out", x, l, final_step)

    print("\n== logits ==")
    top = np.argsort(logits)[::-1][:10]
    print("ref top10:", [(int(i), round(float(logits[i]), 3)) for i in top])


if __name__ == "__main__":
    main()
