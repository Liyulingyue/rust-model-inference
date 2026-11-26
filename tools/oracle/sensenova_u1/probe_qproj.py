import numpy as np
from gguf import GGUFReader

G = "/root/models/SenseNova-U1.6-8B-MoT-GGUF/SenseNova-U1.6-8B-MoT-Q8_0.gguf"
NAME = "language_model.model.layers.0.self_attn.q_proj.weight"

r = GGUFReader(G)
t = next(x for x in r.tensors if x.name == NAME)
print("gguf dims:", list(t.shape), "type:", t.tensor_type)
raw = np.array(t.data)
print("t.data shape:", raw.shape, "dtype:", raw.dtype)

# gguf hands back the raw quantised blocks; reshape to (n_out, blocks, 34)
n_in = int(t.shape[0]); n_out = int(t.shape[1])
blocks = n_in // 32
flat = raw.reshape(-1).view(np.uint8) if raw.dtype != np.uint8 else raw.reshape(-1)
print("flat bytes:", flat.size, "expected:", n_out * blocks * 34)
blk = flat.reshape(n_out, blocks, 34)
d = blk[:, :, 0:2].copy().view(np.float16).astype(np.float32).reshape(n_out, blocks)
qs = blk[:, :, 2:34].copy().view(np.int8).astype(np.float32)          # (n_out, blocks, 32)
W = (d[:, :, None] * qs).reshape(n_out, n_in)                        # (n_out, n_in)
print("dequantised W:", W.shape, "absmean %.6f" % np.abs(W).mean())

normed = np.fromfile("/root/oracle_out/pl0/prefix_l0_normed.f32", dtype="<f4").reshape(-1, n_in)
print("normed:", normed.shape)
ref = np.fromfile("/root/oracle_out/pl0/prefix_l0_q.f32", dtype="<f4").reshape(-1, n_out)

for label, acc in (("f32", np.float32), ("f64", np.float64)):
    x = normed.astype(acc)
    w = W.astype(acc)
    y = x @ w.T
    y = y.astype(np.float32).reshape(ref.shape)
    diff = np.abs(y.reshape(-1) - ref.reshape(-1))
    print(f"{label}: corr={np.corrcoef(y.reshape(-1), ref.reshape(-1))[0,1]:+.8f} max_abs={diff.max():.6e} rel={diff.max()/max(np.abs(ref).max(),1e-12):.3e}")
