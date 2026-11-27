"""Time the official PyTorch AuK VAE decoder against the Rust ports.

The official decoder is the reference for the Rust implementation, so this
probe answers "how much faster is the Rust port?". It decodes the same
latent the Rust ports use (`target/auk-reference/dumps/auk_latent.f32`) and
reports the wall time of `inference_from_latents`.

Two PyTorch paths are measured:

- CPU with oneDNN disabled. The official alias-free resampler calls
  `F.conv1d(..., groups=C)`, and oneDNN's grouped convolution aborts with
  "illegal immediate parameter (range error)" once the output exceeds ~1M
  elements -- i.e. any audio longer than ~1.5 s. That is a oneDNN bug in
  this build, so the reference has to run with oneDNN off, which leaves the
  grouped convolution on the slow ATen path.
- CUDA, which does not have that limitation.

Usage:
    .venv/bin/python3 tools/oracle/auk/probe_vae_timing.py
    AUK_VAE_LATENT=<file> .venv/bin/python3 tools/oracle/auk/probe_vae_timing.py
"""
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.environ.get(
    "AUK_SRC",
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
        "target/auk-reference/AuK-main/src",
    ),
))
from auk.model.vae.bigvgan_flow_vae import BigVGANFlowVAE, BigVGANFlowVAEConfig

torch.set_grad_enabled(False)

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
MODEL = os.path.join(REPO, "models/AuK-Base-and-Flash-GGUF/vae.safetensors")
LATENT = os.environ.get(
    "AUK_VAE_LATENT",
    os.path.join(REPO, "target/auk-reference/dumps/auk_latent.f32"),
)


def load_model(device):
    model = BigVGANFlowVAE(BigVGANFlowVAEConfig())
    from safetensors.torch import load_file
    model.load_state_dict(load_file(MODEL), strict=False)
    model.eval()
    return model.to(device)


def decode(model, latent, device):
    lat = model.denormalize(latent.to(device)).permute(0, 2, 1)
    with torch.no_grad():
        audio = model.inference_from_latents(lat)[0, 0]
    return audio


def time_decode(model, latent, device, repeats=3):
    audio = decode(model, latent, device)
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        audio = decode(model, latent, device)
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / repeats, audio


def main():
    latent_np = np.fromfile(LATENT, dtype="<f4")
    frames = latent_np.size // 64
    # The dump is channel-major `[64][frames]`, the decoder wants `[frames][64]`.
    latent = torch.from_numpy(latent_np.reshape(64, frames).T.copy()).unsqueeze(0)
    print(f"latent {frames} frames = {frames * 480 / 24000:.1f} s of audio")

    # oneDNN off: the only way the official decoder runs at all past 1.5 s.
    torch.backends.mkldnn.enabled = False
    model = load_model("cpu")
    elapsed, audio = time_decode(model, latent, "cpu")
    print(f"torch cpu (oneDNN off): {elapsed * 1000:.0f} ms  "
          f"({audio.numel()} samples, rms {float(audio.pow(2).mean().sqrt()):.5f})")

    # oneDNN on: the fast path, but it aborts past ~1.5 s of audio.
    torch.backends.mkldnn.enabled = True
    try:
        elapsed, audio = time_decode(model, latent, "cpu")
        print(f"torch cpu (oneDNN on):  {elapsed * 1000:.0f} ms  "
              f"({audio.numel()} samples, rms {float(audio.pow(2).mean().sqrt()):.5f})")
    except RuntimeError as error:
        print(f"torch cpu (oneDNN on):  FAILED -> {error}")

    if torch.cuda.is_available():
        torch.backends.mkldnn.enabled = True
        model = load_model("cuda")
        elapsed, audio = time_decode(model, latent, "cuda")
        print(f"torch cuda:            {elapsed * 1000:.0f} ms  "
              f"({audio.numel()} samples, rms {float(audio.pow(2).mean().sqrt()):.5f})")


if __name__ == "__main__":
    main()
