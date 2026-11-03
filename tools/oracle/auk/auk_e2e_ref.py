"""End-to-end PyTorch reference for AuK: Flux2Edit DiT + BigVGANFlowVAE.

Runs the official flow-matching sampler (sway-warped Euler grid, CFG) and the
official VAE decoder on the same text embedding the Rust port produced, then
compares the final waveform against the Rust WAV.

The Rust port dumps (into --dump):
  text_conditioning.f32   -- [cond_tokens][2048] fused LLM hidden states
  latent_in_0.f32         -- [64][T] channel-major initial noise
"""
import argparse
import os
import sys

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

from auk.model.flux2_edit import Flux2Edit
from auk.model.vae.bigvgan_flow_vae import BigVGANFlowVAE, BigVGANFlowVAEConfig

torch.set_grad_enabled(False)

BASE_DIR = MODEL_DIR
REF = REF_DIR

def build_dit():
    model = Flux2Edit(
        dim=1536, heads=24, dim_head=64, ff_mult=2, text_hidden_dim=2048,
        num_layers=10, num_single_layers=20, attn_backend="torch",
        attn_mask_enabled=True, latent_dim=64,
    )
    from safetensors.torch import load_file
    sd = load_file(f"{BASE_DIR}/auk_base.safetensors")
    tsd = {k[len("transformer."):]: v for k, v in sd.items() if k.startswith("transformer.")}
    model.load_state_dict(tsd, strict=False)
    model.eval()
    return model

def build_vae():
    cfg = BigVGANFlowVAEConfig()
    model = BigVGANFlowVAE(cfg)
    from safetensors.torch import load_file
    model.load_state_dict(load_file(f"{BASE_DIR}/vae.safetensors"), strict=False)
    model.eval()
    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default=RUST_DIR)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--cfg", type=float, default=1.0)
    ap.add_argument("--sway", type=float, default=-1.0)
    args = ap.parse_args()
    os.makedirs(REF, exist_ok=True)

    dit = build_dit()
    vae = build_vae()

    d = args.dump
    text = np.fromfile(f"{d}/text_conditioning.f32", dtype="<f4")
    nt = text.size // 2048
    text_t = torch.from_numpy(text.reshape(nt, 2048)).unsqueeze(0)
    latent_cm = np.fromfile(f"{d}/latent_in_0.f32", dtype="<f4")
    D, T = 64, latent_cm.size // 64
    y = torch.from_numpy(latent_cm.reshape(D, T).T.copy()).unsqueeze(0)  # [1,T,D]

    # --- official time grid ---
    t = torch.linspace(0, 1, args.steps + 1)
    if args.sway is not None:
        t = t + args.sway * (torch.cos(torch.pi / 2 * t) - 1 + t)
    print("time grid:", [round(float(v), 6) for v in t])

    def fn(tt, x):
        if args.cfg < 1e-5:
            # Official: CFG is skipped only when `cfg_strength < 1e-5`.
            return dit(x=x, text=text_t, time=tt.reshape(1), mask=None,
                       c_mask=None, ref=None, ref_mask=None)
        # Official `cfg_infer=True` path: the transformer runs the
        # conditional and unconditional branches as a single batch of 2.
        # The unconditional branch zeroes the text **after** projection
        # (`project_text(drop_text=True)`).
        pred_cfg = dit(x=x, text=text_t, time=tt.reshape(1), mask=None,
                       c_mask=None, ref=None, ref_mask=None, cfg_infer=True)
        v_cond, v_uncond = torch.chunk(pred_cfg, 2, dim=0)
        return v_cond + (v_cond - v_uncond) * args.cfg

    # --- Euler integration over the grid ---
    for k in range(args.steps):
        tt = t[k].reshape(1)
        v = fn(tt, y)
        np.save(f"{REF}/e2e_latent_{k}.npy", y[0].numpy())
        y = y + v * (t[k + 1] - t[k])
        np.save(f"{REF}/e2e_latent_{k + 1}.npy", y[0].numpy())
        print(f"  step {k}: t={float(t[k]):.6f} -> {float(t[k+1]):.6f} "
              f"latent rms {float(y.pow(2).mean().sqrt()):.4f}")

    np.save(f"{REF}/e2e_latent.npy", y[0].numpy())

    # --- official VAE decode ---
    # `denormalize` operates on [B, T, D]; `inference_from_latents` on [B, D, T].
    lat = vae.denormalize(y)                # [1, T, D]
    lat = lat.permute(0, 2, 1)              # [1, D, T]
    audio = vae.inference_from_latents(lat)[0, 0]
    print("decoded", audio.numel(), "samples, rms", float(audio.pow(2).mean().sqrt()))
    np.save(f"{REF}/e2e_audio.npy", audio.numpy())

if __name__ == "__main__":
    main()
