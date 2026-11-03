"""Reference decode of AuK's BigVGANFlowVAE for numeric comparison with the Rust port."""
import os, sys, types, numpy as np, torch

class _Dummy(types.ModuleType):
    __path__ = []

    class _Any:
        def __init__(self, *a, **k):
            pass

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return self._Any

class _Finder:
    PREFIXES = ("torchdiffeq", "x_transformers", "flash_attn", "einops", "torchaudio")

    def find_spec(self, fullname, path=None, target=None):
        import importlib.util
        if any(fullname == p or fullname.startswith(p + ".") for p in self.PREFIXES):
            return importlib.util.spec_from_loader(fullname, self)
        return None

    def create_module(self, spec):
        return _Dummy(spec.name)

    def exec_module(self, module):
        sys.modules[module.__name__] = module

sys.meta_path.insert(0, _Finder())

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
missing, unexpected = model.load_state_dict(sd, strict=False)
print("missing", len(missing), missing[:5])
print("unexpected", len(unexpected), unexpected[:5])
model.eval()

# Fixed decoder input so the Rust and PyTorch sides decode exactly the same
# latent. Generated deterministically on first use; delete the file to change
# the seed.
LATENT_PATH = os.path.join(REF_DIR, "auk_latent.npy")
if not os.path.exists(LATENT_PATH):
    rng = np.random.default_rng(0)
    latent_np = rng.standard_normal((1, 64, 50)).astype(np.float32)
    np.save(LATENT_PATH, latent_np)
    print("generated fixed latent", latent_np.shape, "->", LATENT_PATH)
latent_np = np.load(LATENT_PATH)
latent = torch.from_numpy(latent_np).float()
print("latent", latent.shape, "rms", latent.pow(2).mean().sqrt().item())
x = model.denormalize(latent.transpose(1, 2)).transpose(1, 2)
print("denorm rms", x.pow(2).mean().sqrt().item(), "max", x.abs().max().item())
np.save(f"{REF_DIR}/ref_denorm.npy", x.numpy())

x = model.conv_pre(x)
print("conv_pre rms", x.pow(2).mean().sqrt().item(), "max", x.abs().max().item())
np.save(f"{REF_DIR}/ref_conv_pre.npy", x.numpy())

for i in range(model.num_upsamples):
    for i_up in range(len(model.ups[i])):
        x = model.ups[i][i_up](x)
    xs = None
    for j in range(model.num_kernels):
        if xs is None:
            xs = model.resblocks[i * model.num_kernels + j](x)
        else:
            xs += model.resblocks[i * model.num_kernels + j](x)
    x = xs / model.num_kernels
    print(f"stage {i}: shape {tuple(x.shape)} rms {x.pow(2).mean().sqrt().item():.4f} max {x.abs().max().item():.4f}")
    np.save(f"{REF_DIR}/ref_stage{i}.npy", x.numpy())

x = model.activation_post(x)
print("activation_post rms", x.pow(2).mean().sqrt().item())
np.save(f"{REF_DIR}/ref_act_post.npy", x.numpy())
x = model.conv_post(x)
np.save(f"{REF_DIR}/ref_conv_post.npy", x.numpy())
x = torch.clamp(x, -1, 1)
np.save(f"{REF_DIR}/ref_out.npy", x.numpy())
print("out rms", x.pow(2).mean().sqrt().item(), "max", x.abs().max().item())
