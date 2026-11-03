"""Compare the Rust AuK DiT against the official PyTorch reference."""
import os

import numpy as np

# Repository root, derived from this file's location (tools/oracle/auk/).
REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
)

RUST = os.environ.get("AUK_RUST_DIR", os.path.join(REPO_ROOT, "target", "auk-reference", "rust-dit"))
REF = os.environ.get("AUK_REF_DIR", os.path.join(REPO_ROOT, "target", "auk-reference", "dumps"))


def rload(name):
    return np.fromfile(f"{RUST}/{name}", dtype="<f4")


def report(label, r, p):
    if r is None or p is None:
        return
    r = np.asarray(r, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    if r.shape != p.shape:
        print(f"{label}: SHAPE rust={r.shape} ref={p.shape}")
        return
    d = np.abs(r - p)
    rms_r = np.sqrt((r**2).mean())
    rms_p = np.sqrt((p**2).mean())
    denom = max(float(np.sqrt((r * r).sum() * (p * p).sum())), 1e-12)
    cos = float((r * p).sum()) / denom
    print(
        f"{label}: maxdiff={d.max():.6g} rms_rust={rms_r:.6g} rms_ref={rms_p:.6g} "
        f"rel={d.max() / max(rms_p, 1e-9):.4g} cos={cos:.8f}"
    )


text = rload("text_conditioning.f32")
nt = text.size // 2048
print(f"text_conditioning: {nt} tokens")

for step in range(4):
    # audio_embed: Rust [T][1536]; the reference .npy already is [T, 1536].
    audio_ref = np.load(f"{REF}/dit_audio_embed_{step}.npy")
    report(f"[{step}] audio_embed", rload(f"audio_embed_{step}.f32").reshape(audio_ref.shape),
           audio_ref)
    # text_embed: Rust [nt][1536]; the reference .npy already is [nt, 1536].
    text_ref = np.load(f"{REF}/dit_text_embed_{step}.npy")
    report(f"[{step}] text_embed", rload(f"text_embed_{step}.f32").reshape(text_ref.shape),
           text_ref)
    # velocity: Rust [D][T]; the reference .npy already is [T, D].
    velocity_ref = np.load(f"{REF}/dit_velocity_{step}.npy")
    report(f"[{step}] velocity",
           rload(f"velocity_cond_{step}.f32").reshape(velocity_ref.T.shape).T,
           velocity_ref)
    print()
