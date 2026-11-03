"""Isolate the AuK VAE decoder stages to find where Rust and PyTorch diverge."""
import os, sys, types, numpy as np, torch

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

from auk.model.vae.bigvgan_flow_vae import BigVGANFlowVAE, BigVGANFlowVAEConfig

torch.set_grad_enabled(False)
MODEL = os.path.join(MODEL_DIR, "vae.safetensors")

cfg = BigVGANFlowVAEConfig()
model = BigVGANFlowVAE(cfg)
from safetensors.torch import load_file
sd = load_file(MODEL)
model.load_state_dict(sd, strict=False)
model.eval()

latent_np = np.load(os.path.join(REF_DIR, "auk_latent.npy"))
latent = torch.from_numpy(latent_np).float()

x = model.denormalize(latent.transpose(1, 2)).transpose(1, 2)
x = model.conv_pre(x)
np.save(f"{REF_DIR}/ref_conv_pre.npy", x.numpy())

# Stage 0: upsample only (no resblocks)
up = model.ups[0][0](x)
np.save(f"{REF_DIR}/ref_stage0_up.npy", up.numpy())
print("stage0 up:", tuple(up.shape))

# Stage 0 with resblocks
xs = model.resblocks[0](up) + model.resblocks[1](up) + model.resblocks[2](up)
x0 = xs / model.num_kernels
np.save(f"{REF_DIR}/ref_stage0_rb.npy", x0.numpy())
print("stage0 rb:", tuple(x0.shape))

# One resblock in isolation
rb0 = model.resblocks[0](up)
np.save(f"{REF_DIR}/ref_stage0_rb0.npy", rb0.numpy())
print("rb0:", tuple(rb0.shape))

# Inside rb0: step through the 3 conv layers
h = up
for i, (c1, c2, a1, a2) in enumerate(zip(
        model.resblocks[0].convs1, model.resblocks[0].convs2,
        model.resblocks[0].activations[::2], model.resblocks[0].activations[1::2])):
    h = a1(h)
    np.save(f"{REF_DIR}/ref_rb0_l{i}_a1.npy", h.numpy())
    h = c1(h)
    np.save(f"{REF_DIR}/ref_rb0_l{i}_c1.npy", h.numpy())
    h = a2(h)
    np.save(f"{REF_DIR}/ref_rb0_l{i}_a2.npy", h.numpy())
    h = c2(h)
    h = h + (up if i == 0 else prev)
    np.save(f"{REF_DIR}/ref_rb0_l{i}_out.npy", h.numpy())
    prev = h
