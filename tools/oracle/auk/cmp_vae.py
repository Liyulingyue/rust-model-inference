"""Compare the Rust AuK VAE decode against the official PyTorch reference."""
import os

import numpy as np
import sys

# Repository root, derived from this file's location (tools/oracle/auk/).
REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
)

RUST = os.environ.get("AUK_VAE_RUST_DIR", os.path.join(REPO_ROOT, "target", "auk-reference", "rust-vae"))
REF = os.environ.get("AUK_REF_DIR", os.path.join(REPO_ROOT, "target", "auk-reference", "dumps"))


def load_rust(name):
    return np.fromfile(f"{RUST}/{name}.f32", dtype="<f4")


def load_ref(name):
    a = np.load(f"{REF}/ref_{name}.npy")
    # PyTorch tensors are [B, C, T]; Rust buffers are [C, T].
    if a.ndim == 3:
        a = a[0]
    return a.astype(np.float32)


def cmp(name, rust_name=None):
    rust_name = rust_name or name
    r = load_rust(rust_name)
    try:
        p = load_ref(name)
    except FileNotFoundError:
        print(f"{name}: (no reference)")
        return
    if r.size != p.size:
        print(f"{name}: SIZE MISMATCH rust={r.size} ref={p.size}")
        return
    r = r.reshape(p.shape)
    diff = np.abs(r - p)
    rms_r = np.sqrt((r**2).mean())
    rms_p = np.sqrt((p**2).mean())
    rel = diff.max() / max(rms_p, 1e-9)
    print(
        f"{name}: maxdiff={diff.max():.6g} rms(rust)={rms_r:.6g} rms(ref)={rms_p:.6g} "
        f"rel_max={rel:.4g} cos={float((r*p).sum()/max(np.sqrt((r*r).sum()*(p*p).sum()),1e-9)):.6f}"
    )


cmp("denorm", "denorm")
cmp("conv_pre", "conv_pre")
for i in range(6):
    cmp(f"stage{i}", f"stage{i}")
cmp("act_post", "act_post")
cmp("conv_post", "conv_post")
cmp("out", "out")
