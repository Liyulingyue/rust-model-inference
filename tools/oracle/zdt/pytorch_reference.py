"""PyTorch reference for the ZDTaichu5.0-9B vision tower.

Runs the upstream C-RADIO tower (the `TaichuAI/ZDTaichu5.0-9B` checkpoint's
`cradio_model.py`) plus the `mlp1` projector head, and dumps the projected
embeddings as raw little-endian F32 so the Rust encoder can be compared
against it.

Two modes:

`conditioner-identity`
    The upstream `InputConditioner` is replaced by the identity, so the tower
    sees exactly the pixels the Rust encoder sees. This isolates the tower,
    the patch merge and the projector head from the input normalization.

`upstream`
    The upstream path unchanged: the image processor normalizes with the
    ImageNet statistics from `preprocessor_config.json` and the conditioner
    normalizes again with the CLIP statistics stored in the checkpoint. The
    Rust encoder and llama.cpp apply one normalization (the GGUF metadata's
    `clip.vision.image_mean` / `image_std`), so this mode measures the gap
    between the upstream pipeline and the llama.cpp pipeline.

The input is the same fixture the oracle uses: a solid blue 512x512 image, so
the pixels are `[0, 0, 1]` before any normalization.
"""
import argparse
import json
import shutil
import struct
import sys
import tempfile
from pathlib import Path

import torch
from safetensors.torch import load_file


def load_upstream_package(hf_dir: Path):
    """Import the upstream `cradio_model` / `cradio_config` as a package.

    They use a relative import, so they need a package name; the checkpoint
    directory itself is not importable (its name has dashes).
    """
    package = Path(tempfile.mkdtemp(prefix="zdt_ref_"))
    (package / "__init__.py").write_text("")
    for name in ("cradio_model.py", "cradio_config.py"):
        shutil.copy(hf_dir / name, package / name)
    sys.path.insert(0, str(package.parent))
    cradio_model = __import__(f"{package.name}.cradio_model", fromlist=["RADIOModel"])
    cradio_config = __import__(f"{package.name}.cradio_config", fromlist=["RADIOConfig"])
    return cradio_model.RADIOModel, cradio_config.RADIOConfig


class SquaredReLU(torch.nn.Module):
    def forward(self, x):
        return torch.pow(torch.nn.functional.relu(x), 2)


class RMSNorm(torch.nn.Module):
    """The projector's RMSNorm: ones-initialized weight, `output * weight`."""

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return (self.weight.to(torch.float32) * hidden_states).to(input_dtype)


def build_radio_model(hf_dir: Path, config):
    radiomodel_cls, radio_config_cls = load_upstream_package(hf_dir)
    vision_config = config["vision_config"]
    radio_config = radio_config_cls(args=vision_config["args"])
    model = radiomodel_cls(radio_config)
    weights = load_file(hf_dir / "model-5-of-5.safetensors")
    prefix = "vision_model."
    tower_weights = {
        key[len(prefix) :]: value for key, value in weights.items() if key.startswith(prefix)
    }
    missing, unexpected = model.load_state_dict(tower_weights, strict=False)
    missing = [key for key in missing if "summary_idxs" not in key]
    if missing or unexpected:
        raise SystemExit(f"tower weights did not load: missing={missing} unexpected={unexpected}")
    model.eval()
    return model


def build_projector(config, weights):
    vit_hidden = config["vit_hidden_size"]
    proj_hidden = config["projector_hidden_size"]
    llm_hidden = config["hidden_size"]
    pixel_shuffle_dim = vit_hidden * int(1 / config["downsample_ratio"]) ** 2
    head = torch.nn.Sequential(
        RMSNorm(pixel_shuffle_dim, eps=1e-5),
        torch.nn.Linear(pixel_shuffle_dim, proj_hidden, bias=False),
        SquaredReLU(),
        torch.nn.Linear(proj_hidden, llm_hidden, bias=False),
    )
    state = {
        "0.weight": weights["mlp1.0.weight"],
        "1.weight": weights["mlp1.1.weight"],
        "3.weight": weights["mlp1.3.weight"],
    }
    head.load_state_dict(state, strict=True)
    head.eval()
    return head


def pixel_shuffle(x: torch.Tensor, scale_factor: float) -> torch.Tensor:
    """The upstream `pixel_shuffle` (ps_version='v2')."""
    n, w, h, c = x.size()
    x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
    x = x.permute(0, 2, 1, 3).contiguous()
    x = x.view(
        n,
        int(h * scale_factor),
        int(w * scale_factor),
        int(c / (scale_factor * scale_factor)),
    )
    return x.permute(0, 2, 1, 3).contiguous()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--mode",
        choices=["conditioner-identity", "upstream"],
        default="conditioner-identity",
    )
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument(
        "--positions",
        type=Path,
        default=None,
        help="a GGUF mmproj whose pre-downsampled position embeddings replace "
        "the upstream's 128x128 table, so both sides use the same positions",
    )
    parser.add_argument(
        "--precision",
        choices=["f32", "bf16"],
        default="f32",
        help="f32 keeps the checkpoint's bf16 weights in f32 arithmetic and skips "
        "the upstream's bf16 feature cast, which isolates the tower's math; "
        "bf16 reproduces the upstream inference dtype exactly",
    )
    args = parser.parse_args()

    config = json.loads((args.hf_dir / "config.json").read_text(encoding="utf-8"))
    weights = load_file(args.hf_dir / "model-5-of-5.safetensors")
    tower = build_radio_model(args.hf_dir, config)
    head = build_projector(config, weights)

    # The solid blue fixture: [0, 0, 1] per pixel before normalization.
    pixels = torch.zeros(1, 3, args.size, args.size)
    pixels[:, 2] = 1.0

    if args.mode == "upstream":
        # The image processor normalizes with the ImageNet statistics.
        mean = torch.tensor(config.get("norm_mean") or [0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor(config.get("norm_std") or [0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        pixels = (pixels - mean) / std
    else:
        # The Rust encoder receives the pixels already normalized with the
        # GGUF metadata's statistics, so the conditioner must not touch them.
        conditioner = tower.radio_model.input_conditioner
        conditioner.norm_mean = torch.zeros_like(conditioner.norm_mean)
        conditioner.norm_std = torch.ones_like(conditioner.norm_std)

    if args.positions is not None:
        # The GGUF ships the position embeddings pre-downsampled to the fixed
        # 512x512 grid; the upstream keeps a 128x128 table and interpolates.
        # Replacing the table with the GGUF's puts both sides on the same
        # positions, which isolates the tower's arithmetic from the
        # conversion's downsampling.
        sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
        from tools.converter.utils.gguf import read_gguf_tensor_bytes

        raw = read_gguf_tensor_bytes(args.positions, "v.position_embd.weight")
        values = struct.unpack(f"<{len(raw) // 4}f", raw)
        positions = torch.tensor(values).reshape(1024, -1).contiguous()
        generator = tower.radio_model.model.patch_generator
        generator.pos_embed = torch.nn.Parameter(positions.unsqueeze(0))
        generator.num_patches = 1024
        generator.num_rows = 32
        generator.num_cols = 32

    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32
    tower = tower.to(dtype)
    head = head.to(dtype)
    pixels = pixels.to(dtype)

    with torch.inference_mode():
        features = tower(pixels).features
        if args.precision == "bf16":
            features = features.to(torch.bfloat16)
        h = w = int(features.shape[1] ** 0.5)
        features = features.reshape(features.shape[0], h, w, -1)
        features = pixel_shuffle(features, scale_factor=config["downsample_ratio"])
        features = features.reshape(features.shape[0], -1, features.shape[-1])
        projected = head(features)

    values = projected.reshape(-1).to(torch.float32).tolist()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(struct.pack(f"<{len(values)}f", *values))
    print(f"{args.mode}: wrote {args.out} ({len(values)} floats, {projected.shape})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
