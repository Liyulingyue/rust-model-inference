"""PyTorch reference for the AuK Flux2Edit DiT (Tencent-Hunyuan/AuK).

Loads `auk_base.safetensors` into the official `Flux2Edit` module and runs
the same forward the Rust port runs, so intermediate buffers can be compared
numerically.

The Rust port dumps (into --dump):
  text_conditioning.f32   -- [cond_tokens][2048] fused LLM hidden states
  latent_in_<step>.f32    -- [64][T] channel-major latent at step start
  audio_embed_<step>.f32  -- [T][1536] audio_embed output
  text_embed_<step>.f32   -- [cond_tokens][1536] txt_proj + txt_norm
"""
import sys, os, argparse
import numpy as np
import torch

# Directory holding the official `Tencent-Hunyuan/AuK` checkout (the `src/`
# layout). Override with AUK_SRC.
# Repository root, derived from this file's location (tools/oracle/auk/).
REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
)

AUK_SRC = os.environ.get("AUK_SRC", os.path.join(REPO_ROOT, "target", "auk-reference", "AuK-main", "src"))
# Scratch directory for reference dumps. Override with AUK_REF_DIR.
REF_DIR = os.environ.get("AUK_REF_DIR", os.path.join(REPO_ROOT, "target", "auk-reference", "dumps"))
# Directory holding the Rust `RMI_AUK_*_DUMP` output. Override with AUK_RUST_DIR.
RUST_DIR = os.environ.get("AUK_RUST_DIR", os.path.join(REPO_ROOT, "target", "auk-reference", "rust-dit"))

# Directory holding the published AuK weights.
MODEL_DIR = os.path.join(REPO_ROOT, "models", "AuK-Base-and-Flash-GGUF")

sys.path.insert(0, AUK_SRC)

from auk.model.flux2_edit import Flux2Edit, Flux2EditConfig

torch.set_grad_enabled(False)

MODEL = os.path.join(MODEL_DIR, "auk_base.safetensors")
REF = REF_DIR

def build():
    model = Flux2Edit(
        dim=1536,
        heads=24,
        dim_head=64,
        ff_mult=2,
        text_hidden_dim=2048,
        num_layers=10,
        num_single_layers=20,
        attn_backend="torch",
        attn_mask_enabled=True,
        latent_dim=64,
    )
    from safetensors.torch import load_file
    sd = load_file(MODEL)
    tsd = {k[len("transformer."):]: v for k, v in sd.items() if k.startswith("transformer.")}
    missing, unexpected = model.load_state_dict(tsd, strict=False)
    print("missing", len(missing), missing[:8])
    print("unexpected", len(unexpected), unexpected[:8])
    model.eval()
    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default=RUST_DIR)
    args = ap.parse_args()
    os.makedirs(REF, exist_ok=True)

    model = build()
    d = args.dump

    def rload(name):
        return np.fromfile(f"{d}/{name}", dtype="<f4")

    text = rload("text_conditioning.f32")          # [nt][2048]
    nt = text.size // 2048
    D = 64

    text_t = torch.from_numpy(text.reshape(nt, 2048)).unsqueeze(0)   # [1,nt,2048]

    # The Rust latent is [D, T] channel-major; the official model wants
    # [B, T, D].
    for step in range(8):
        try:
            latent_cm = rload(f"latent_in_{step}.f32")
        except FileNotFoundError:
            break
        T = latent_cm.size // D
        latent_t = torch.from_numpy(latent_cm.reshape(D, T).T.copy()).unsqueeze(0)

        audio = model.audio_embed(latent_t)
        np.save(f"{REF}/dit_audio_embed_{step}.npy", audio[0].numpy())
        c = model.project_text(text_t)
        np.save(f"{REF}/dit_text_embed_{step}.npy", c[0].numpy())

        # The Rust scheduler uses the sway-warped grid; reproduce it exactly.
        u = step / 4.0
        tval = u + (-1.0) * (np.cos(np.pi / 2 * u) - 1 + u)
        t = torch.full((1,), tval)
        out = model(x=latent_t, text=text_t, time=t, mask=None, c_mask=None,
                    ref=None, ref_mask=None)
        arr = out[0].float().numpy()   # [T, D]
        np.save(f"{REF}/dit_velocity_{step}.npy", arr)
        print(f"step {step} t={tval}: velocity {arr.shape} rms {np.sqrt((arr**2).mean()):.4f}")

if __name__ == "__main__":
    main()
