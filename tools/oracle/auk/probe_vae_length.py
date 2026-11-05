"""Find the length at which the official PyTorch AuK VAE decoder fails.

The official `Activation1d` -> `DownSample1d` -> `LowPassFilter1d` path does
`F.conv1d(x, filter.expand(C, -1, -1), stride=2, groups=C)`. The Rust port
implements the same math directly and does not hit this.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.environ.get(
    "AUK_SRC",
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
        "target", "auk-reference", "AuK-main", "src",
    ),
))
from auk.model.vae.bigvgan_flow_vae import BigVGANFlowVAE, BigVGANFlowVAEConfig

torch.set_grad_enabled(False)

# The official alias-free resampler calls
#   F.conv1d(x, filter.expand(C, -1, -1), stride=2, groups=C)
# PyTorch 2.15 on aarch64 routes grouped convolutions through oneDNN, whose
# OpenMP chunking aborts with "illegal immediate parameter (range error)" once
# the output exceeds ~1M elements -- i.e. any audio longer than ~1.5 s. That
# is a oneDNN bug in this build, not an AuK issue (the same call succeeds
# with oneDNN disabled, and the Rust port implements the same math directly
# with no length limit). Disable oneDNN so the reference can decode the
# lengths we actually ship.
torch.backends.mkldnn.enabled = False


REPO = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
MODEL = os.path.join(REPO, "models", "AuK-Base-and-Flash-GGUF", "vae.safetensors")

model = BigVGANFlowVAE(BigVGANFlowVAEConfig())
from safetensors.torch import load_file
model.load_state_dict(load_file(MODEL), strict=False)
model.eval()

for seconds in (1, 2, 3, 4, 5, 6):
    frames = seconds * 50  # 24 kHz / 480 downsample
    latent = torch.randn(1, frames, 64)
    lat = model.denormalize(latent).permute(0, 2, 1)
    try:
        audio = model.inference_from_latents(lat)
        print(f"{seconds}s ({frames} frames): OK, {audio.shape[-1]} samples, "
              f"peak {float(audio.abs().max()):.4f}")
    except Exception as error:  # noqa: BLE001
        print(f"{seconds}s ({frames} frames): FAILED -> {type(error).__name__}: {error}")
        break
