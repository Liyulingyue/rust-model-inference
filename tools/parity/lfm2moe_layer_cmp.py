#!/usr/bin/env python
"""Compare llama.cpp layer-oracle dumps (step 0) with a numpy replay.

Usage: models/.venv/bin/python tools/parity/lfm2moe_layer_cmp.py <model> <oracle.bin>
"""
import sys
import numpy as np
from lfm2moe_reference import (Loader, rms_norm, silu, sigmoid, rope_neox,
                               D_CONV, L_CACHE, DENSE_LEAD, ATTN_LAYERS, N_KV, HDK)

def load_oracle(path):
    data = open(path, 'rb').read()
    off, recs = 0, {}
    while off < len(data):
        nl, = np.frombuffer(data, np.int32, 1, off); off += 4
        name = data[off:off+nl].decode(); off += nl
        ne0, = np.frombuffer(data, np.int32, 1, off); off += 4
        vals = np.frombuffer(data, np.float32, ne0, off).astype(np.float64); off += ne0 * 4
        recs.setdefault(name, []).append(vals)
    return recs

def get(recs, name):
    lst = recs.get(name)
    return lst[-1] if lst else None

def cmp(label, oracle_name, ref, recs, n=None):
    o = get(recs, oracle_name)
    if o is None:
        print(f"  {label:34s} oracle tensor MISSING: {oracle_name}")
        return
    if n:
        o, ref = o[:n], ref[:n]
    d = np.abs(o - ref)
    mx = d.max()
    flag = "  <<< DIVERGE" if mx > 2e-3 else ""
    print(f"  {label:34s} max_diff={mx:.6f} first8 o={[round(float(v),4) for v in o[:4]]} r={[round(float(v),4) for v in ref[:4]]}{flag}")

def main():
    model_path, oracle_path = sys.argv[1], sys.argv[2]
    L = Loader(model_path)
    recs = load_oracle(oracle_path)
    n_vocab, n_embd = 65536, 2048

    ids = [1]
    x = L.row("token_embd.weight", ids[0]).copy()
    cmp("embed", "model.embed_tokens", x, recs)

    for l in range(24):
        is_attn = l in ATTN_LAYERS
        print(f"Layer {l} ({'attn' if is_attn else 'conv'})")
        normed = rms_norm(x, L.tensor(f"blk.{l}.attn_norm.weight"))
        cmp("attn_norm out", f"model.layers.{{}}.operator_norm-{l}", normed, recs)
        if is_attn:
            q_raw = normed @ L.tensor(f"blk.{l}.attn_q.weight").T
            k_raw = normed @ L.tensor(f"blk.{l}.attn_k.weight").T
            v = (normed @ L.tensor(f"blk.{l}.attn_v.weight").T).reshape(N_KV, HDK)
            # llama.cpp cb names Qcur/Kcur BEFORE q_norm/k_norm+rope
            cmp("q_raw", f"Qcur-{l}", q_raw, recs)
            cmp("k_raw", f"Kcur-{l}", k_raw, recs)
            cmp("v_raw", f"Vcur-{l}", v.reshape(-1), recs)
            q = rms_norm(q_raw.reshape(32, HDK), L.tensor(f"blk.{l}.attn_q_norm.weight"))
            k = rms_norm(k_raw.reshape(N_KV, HDK), L.tensor(f"blk.{l}.attn_k_norm.weight"))
            q = rope_neox(q, 0, HDK, 32).reshape(32, HDK)
            k = rope_neox(k, 0, HDK, N_KV).reshape(N_KV, HDK)
            # pos 0 attention: output per q head = its kv head's v
            attn = np.repeat(v, 32 // N_KV, axis=0).reshape(-1)
            cmp("kqv_out", f"kqv_out-{l}", attn, recs)
            attn_proj = attn @ L.tensor(f"blk.{l}.attn_output.weight").T
            cmp("attn out_proj", f"model.layers.{{}}.self_attn.out_proj-{l}", attn_proj, recs)
            x = x + attn_proj
        else:
            bcx = normed @ L.tensor(f"blk.{l}.shortconv.in_proj.weight").T
            cmp("conv in_proj", f"model.layers.{{}}.conv.in_proj-{l}", bcx, recs)
            b, c, xc = bcx[:n_embd], bcx[n_embd:2*n_embd], bcx[2*n_embd:]
            bx = b * xc
            window = [bx]  # step 0: state zeros
            Kt = L.tensor(f"blk.{l}.shortconv.conv.weight").reshape(n_embd, L_CACHE)
            buf = np.zeros((L_CACHE, n_embd))
            buf[L_CACHE - len(window):] = np.stack(window)
            conv = np.sum(buf * Kt.T, axis=0)
            cmp("conv out (raw)", f"model.layers.{{}}.conv.conv-{l}", conv, recs)
            y = c * conv
            out = y @ L.tensor(f"blk.{l}.shortconv.out_proj.weight").T
            cmp("conv out_proj", f"model.layers.{{}}.conv.out_proj-{l}", out, recs)
            x = x + out

        normed2 = rms_norm(x, L.tensor(f"blk.{l}.ffn_norm.weight"))
        cmp("ffn_norm out", f"model.layers.{{}}.ffn_out-{l}", normed2, recs)
        if l < DENSE_LEAD:
            g = normed2 @ L.tensor(f"blk.{l}.ffn_gate.weight").T
            u = normed2 @ L.tensor(f"blk.{l}.ffn_up.weight").T
            cmp("ffn_gate", f"ffn_gate-{l}", g, recs)
            cmp("ffn_up", f"ffn_up-{l}", u, recs)
            h = silu(g) * u
            cmp("ffn_swiglu", f"ffn_swiglu-{l}", h, recs)
            ffn = h @ L.tensor(f"blk.{l}.ffn_down.weight").T
            x = x + ffn
        else:
            router = L.tensor(f"blk.{l}.ffn_gate_inp.weight")
            logits_r = router @ normed2
            cmp("moe logits", f"ffn_moe_logits-{l}", logits_r, recs)
            probs = sigmoid(logits_r)
            cmp("moe probs", f"ffn_moe_probs-{l}", probs, recs)
            bias = L.tensor(f"blk.{l}.exp_probs_b.bias")
            cmp("moe probs biased", f"ffn_moe_probs_biased-{l}", probs + bias, recs)
            sel = np.argsort(-(probs + bias))[:4]
            print(f"    selected (numpy): {sel}")
            w = probs[sel]
            w = w / max(w.sum(), 6.103515625e-5)
            cmp("moe weights norm", f"ffn_moe_weights_norm-{l}", w, recs)
            acc = np.zeros(n_embd)
            for e, we in zip(sel, w):
                g = normed2 @ L.expert(f"blk.{l}.ffn_gate_exps.weight", e, n_embd, 1792).T
                u = normed2 @ L.expert(f"blk.{l}.ffn_up_exps.weight", e, n_embd, 1792).T
                d = L.expert(f"blk.{l}.ffn_down_exps.weight", e, 1792, n_embd)
                acc += we * ((silu(g) * u) @ d.T)
            cmp("moe out", f"ffn_moe_out-{l}", acc, recs)
            x = x + acc
        cmp("l_out", f"l_out-{l}", x, recs)

    normed_f = rms_norm(x, L.tensor("token_embd_norm.weight"))
    cmp("result_norm", "result_norm", normed_f, recs)

if __name__ == "__main__":
    main()
