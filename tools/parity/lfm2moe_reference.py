#!/usr/bin/env python
"""LFM2-8B-A1B (lfm2moe) step-by-step numpy reference forward (f64 ground truth).

Memory-safe: tensors are dequantized lazily (per layer / per expert use),
never the whole model at once.

Usage:
  models/.venv/bin/python tools/parity/lfm2moe_reference.py \
      models/LFM2-8B-A1B-GGUF/LFM2-8B-A1B-Q8_0.gguf <rust_top10_stderr> [llama_logits.bin]
"""
import sys
import numpy as np
import gguf

EPS = 9.999999747378752e-06
FREQ_BASE = 1_000_000.0
D_CONV = 2
L_CACHE = 3
N_EXP = 32
N_USED = 4
DENSE_LEAD = 2
ATTN_LAYERS = {2, 6, 10, 14, 18, 21}
N_KV = 8
HDK = 64


class Loader:
    def __init__(self, path):
        self.reader = gguf.GGUFReader(path)
        self.info = {t.name: t for t in self.reader.tensors}

    def dequant_bytes(self, raw: np.ndarray, ggml_type) -> np.ndarray:
        raw = np.ascontiguousarray(raw if raw.dtype == np.uint8 else raw.view(np.uint8))
        if int(ggml_type) == 0:  # F32
            return raw.view(np.float32).astype(np.float64)
        if int(ggml_type) == 8:  # Q8_0
            n = raw.size // 34
            blocks = raw.reshape(n, 34)
            scales = blocks[:, :2].copy().view(np.float16).astype(np.float64)
            qs = blocks[:, 2:].copy().view(np.int8).astype(np.float64)
            return (scales * qs).reshape(n * 32)
        raise ValueError(f"unsupported ggml type {ggml_type}")

    def tensor(self, name):
        t = self.info[name]
        shape = tuple(int(x) for x in reversed(t.shape))
        return self.dequant_bytes(t.data, t.tensor_type).reshape(shape)

    def row(self, name, row):
        """Dequantize one outer row (first ne[0..n-1] product elements)."""
        t = self.info[name]
        shape = tuple(int(x) for x in reversed(t.shape))
        row_elems = int(np.prod(shape[1:]))
        assert int(t.tensor_type) in (0, 8), name
        if int(t.tensor_type) == 0:
            nbytes = row_elems * 4
        else:
            nbytes = row_elems // 32 * 34
        data = (t.data if t.data.dtype == np.uint8 else t.data.view(np.uint8)).reshape(-1)
        return self.dequant_bytes(data[row * nbytes:(row + 1) * nbytes], t.tensor_type)

    def expert(self, name, expert, n_in, n_ff):
        """Dequantize one expert slab of a 3-D [n_in, n_ff, n_expert] tensor."""
        t = self.info[name]
        assert int(t.tensor_type) == 8
        per_bytes = n_in * n_ff // 32 * 34
        data = (t.data if t.data.dtype == np.uint8 else t.data.view(np.uint8)).reshape(-1)
        return self.dequant_bytes(
            data[expert * per_bytes:(expert + 1) * per_bytes], 8
        ).reshape(n_ff, n_in)


def rms_norm(x, w):
    scale = 1.0 / np.sqrt(np.float64(np.float32(np.mean(x * x))) + np.float64(np.float32(EPS)))
    return x * scale * w


def silu(x):
    return x / (1.0 + np.exp(-x))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def rope_neox(x, pos, head_dim, heads):
    half = head_dim // 2
    theta_scale = float(np.float32(FREQ_BASE) ** (-2.0 / head_dim))
    theta = float(np.float32(pos))
    x = x.reshape(heads, head_dim)
    for i in range(half):
        t32 = np.float32(theta)
        c, s = float(np.cos(t32)), float(np.sin(t32))
        x0 = x[:, i].copy()
        x1 = x[:, i + half].copy()
        x[:, i] = x0 * c - x1 * s
        x[:, i + half] = x0 * s + x1 * c
        theta *= theta_scale
    return x.reshape(-1)


def main():
    model_path = sys.argv[1]
    rust_stderr = sys.argv[2]
    llama_bin = sys.argv[3] if len(sys.argv) > 3 else None
    L = Loader(model_path)

    n_vocab, n_embd = 65536, 2048

    ids = [1, 6423, 708, 3493, 856, 779, 5706, 803, 4481, 540, 708, 64015, 708]

    rust = {}
    for line in open(rust_stderr):
        m = line.strip()
        if m.startswith("RUST_LOGITS step="):
            head, rest = m.split(" top10:")
            step = int(head.split("=")[1])
            rust[step] = {int(a): float(b) for a, b in (t.split(":") for t in rest.split())}

    llama = None
    if llama_bin:
        data = open(llama_bin, "rb").read()
        llama, off = {}, 0
        while off < len(data):
            step, tok, nv = np.frombuffer(data, np.int32, 3, off); off += 12
            llama[int(step)] = np.frombuffer(data, np.float32, nv, off).astype(np.float64)
            off += nv * 4

    kv = {}       # layer -> ([K per step], [V per step])
    bx_hist = {}  # layer -> list of bx vectors (prefill conv history)
    final_step = len(ids) - 1
    x = None
    for step, tok in enumerate(ids):
        x = L.row("token_embd.weight", tok).copy()
        for l in range(24):
            is_attn = l in ATTN_LAYERS
            normed = rms_norm(x, L.tensor(f"blk.{l}.attn_norm.weight"))
            if is_attn:
                q = (normed @ L.tensor(f"blk.{l}.attn_q.weight").T).reshape(n_head := 32, HDK)
                k = (normed @ L.tensor(f"blk.{l}.attn_k.weight").T).reshape(N_KV, HDK)
                v = (normed @ L.tensor(f"blk.{l}.attn_v.weight").T).reshape(N_KV, HDK)
                q = rms_norm(q, L.tensor(f"blk.{l}.attn_q_norm.weight"))
                k = rms_norm(k, L.tensor(f"blk.{l}.attn_k_norm.weight"))
                q = rope_neox(q, step, HDK, 32).reshape(32, HDK)
                k = rope_neox(k, step, HDK, N_KV).reshape(N_KV, HDK)
                k = k.astype(np.float16).astype(np.float64)
                v = v.astype(np.float16).astype(np.float64)
                K, V = kv.setdefault(l, ([], []))
                K.append(k); V.append(v)
                attn = np.empty(32 * HDK)
                group = 32 // N_KV
                Km = np.stack(K); Vm = np.stack(V)
                for h in range(32):
                    kv_h = h // group
                    scores = Km[:, kv_h, :] @ q[h] / np.sqrt(float(HDK))
                    p = np.exp(scores - scores.max()); p /= p.sum()
                    attn[h * HDK:(h + 1) * HDK] = p @ Vm[:, kv_h, :]
                x = x + attn @ L.tensor(f"blk.{l}.attn_output.weight").T
            else:
                bcx = normed @ L.tensor(f"blk.{l}.shortconv.in_proj.weight").T
                b, c, xc = bcx[:n_embd], bcx[n_embd:2*n_embd], bcx[2*n_embd:]
                bx = b * xc
                hist = bx_hist.setdefault(l, [])
                window = hist[-D_CONV:] + [bx]
                Kt = L.tensor(f"blk.{l}.shortconv.conv.weight").reshape(n_embd, L_CACHE)
                # left-pad the window to l_cache taps: tap 0 = oldest
                buf = np.zeros((L_CACHE, n_embd))
                buf[L_CACHE - len(window):] = np.stack(window)
                conv = np.sum(buf * Kt.T, axis=0)
                hist.append(bx)
                if len(hist) > D_CONV:
                    hist.pop(0)
                y = c * conv
                x = x + y @ L.tensor(f"blk.{l}.shortconv.out_proj.weight").T

            normed2 = rms_norm(x, L.tensor(f"blk.{l}.ffn_norm.weight"))
            if l < DENSE_LEAD:
                g = normed2 @ L.tensor(f"blk.{l}.ffn_gate.weight").T
                u = normed2 @ L.tensor(f"blk.{l}.ffn_up.weight").T
                x = x + (silu(g) * u) @ L.tensor(f"blk.{l}.ffn_down.weight").T
            else:
                router = L.tensor(f"blk.{l}.ffn_gate_inp.weight")  # (n_expert, n_embd)
                probs = sigmoid(router @ normed2)
                bias = L.tensor(f"blk.{l}.exp_probs_b.bias")
                sel = np.argsort(-(probs + bias))[:N_USED]
                w = probs[sel]
                w = w / max(w.sum(), 6.103515625e-5)
                acc = np.zeros(n_embd)
                for e, we in zip(sel, w):
                    g = normed2 @ L.expert(f"blk.{l}.ffn_gate_exps.weight", e, n_embd, 1792).T
                    u = normed2 @ L.expert(f"blk.{l}.ffn_up_exps.weight", e, n_embd, 1792).T
                    d = L.expert(f"blk.{l}.ffn_down_exps.weight", e, 1792, n_embd)
                    acc += we * ((silu(g) * u) @ d.T)
                x = x + acc
        if step in rust or (llama and step in llama):
            normed_f = rms_norm(x, L.tensor("token_embd_norm.weight")[0])
            t = L.info["token_embd.weight"]
            data = (t.data if t.data.dtype == np.uint8 else t.data.view(np.uint8)).reshape(-1)
            step_logits = np.empty(n_vocab, dtype=np.float64)
            chunk_rows = 4096
            row_elems = n_embd
            for r0 in range(0, n_vocab, chunk_rows):
                rows = min(chunk_rows, n_vocab - r0)
                raw = data[r0 * row_elems // 32 * 34:(r0 + rows) * row_elems // 32 * 34]
                W = L.dequant_bytes(raw, 8).reshape(rows, n_embd)
                step_logits[r0:r0 + rows] = W @ normed_f
            ref = step_logits
            line = f"step {step:2d}:"
            if step in rust:
                diffs = [abs(ref[t2] - v) for t2, v in rust[step].items()]
                rt1 = max(rust[step], key=rust[step].get)
                line += f"  vs_rust max={max(diffs):7.4f} rust_top1={rt1} np_top1={int(np.argmax(ref))}"
            if llama and step in llama:
                d = np.abs(ref - llama[step])
                line += f"  vs_llama max={d.max():7.4f} mean={d.mean():.4f} llama_top1={int(np.argmax(llama[step]))}"
            print(line, flush=True)
        if step == final_step:
            break

    # Logits: tied output = token_embd (dequantize in f32 chunks to bound memory)
    normed_f = rms_norm(x, L.tensor("token_embd_norm.weight"))
    t = L.info["token_embd.weight"]
    data = (t.data if t.data.dtype == np.uint8 else t.data.view(np.uint8)).reshape(-1)
    logits = np.empty(n_vocab, dtype=np.float64)
    chunk_rows = 4096
    row_elems = n_embd
    for r0 in range(0, n_vocab, chunk_rows):
        rows = min(chunk_rows, n_vocab - r0)
        raw = data[r0 * row_elems // 32 * 34:(r0 + rows) * row_elems // 32 * 34]
        W = L.dequant_bytes(raw, 8).reshape(rows, n_embd)
        logits[r0:r0 + rows] = W @ normed_f

    ref = logits
    if final_step in rust:
        diffs = [abs(ref[t] - v) for t, v in rust[final_step].items()]
        rt1 = max(rust[final_step], key=rust[final_step].get)
        print(f"numpy vs rust top10: max_diff={max(diffs):.4f}  rust_top1={rt1} numpy_top1={int(np.argmax(ref))}")
    if llama and final_step in llama:
        d = np.abs(ref - llama[final_step])
        lt1 = int(np.argmax(llama[final_step]))
        print(f"numpy vs llama.cpp full logits: max_diff={d.max():.4f} mean={d.mean():.4f} llama_top1={lt1} numpy_top1={int(np.argmax(ref))}")


if __name__ == "__main__":
    main()
