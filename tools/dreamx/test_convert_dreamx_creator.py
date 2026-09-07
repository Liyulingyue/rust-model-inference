import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from tools.dreamx.convert_dreamx_creator import (
    JOINT_LAYERS,
    build_inventory,
    load_pt_state_dict,
    output_paths,
    pair_id,
)


VIDEO_CONFIG = {
    "dim": 3072,
    "ffn_dim": 14336,
    "num_heads": 24,
    "num_layers": 30,
    "in_dim": 48,
    "out_dim": 48,
    "text_dim": 4096,
    "text_len": 512,
}

AUDIO_CONFIG = {
    "dim": 1536,
    "ffn_dim": 8960,
    "num_heads": 12,
    "num_layers": 30,
    "in_dim": 128,
    "out_dim": 128,
    "text_dim": 4096,
    "text_len": 512,
}


def write_safetensors(path: Path, names=("weight",)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = {}
    offset = 0
    for name in names:
        header[name] = {
            "dtype": "F32",
            "shape": [1],
            "data_offsets": [offset, offset + 4],
        }
        offset += 4
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * ((8 - len(raw) % 8) % 8)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(offset))


def make_minimal_layout(root: Path, omit: str | None = None) -> None:
    files = (
        "creator/audio_model/diffusion_pytorch_model.safetensors",
        "creator/cross_attn_weights.safetensors",
        "wan2.2_ti2v_5b/models_t5_umt5-xxl-enc-bf16.pth",
        "wan2.2_ti2v_5b/Wan2.2_VAE.pth",
        "audio_vae/diffusion_pytorch_model.safetensors",
        "refiner/sr_dit_5b.pt",
        "refiner/latent_upsampler_flash.pt",
        "refiner/latent_upsampler_2d_causal.pt",
        "refiner/lightvae_nu_scheme3.pt",
        "wan2.2_ti2v_5b/google/umt5-xxl/tokenizer.json",
    )
    for relative in files:
        if relative == omit:
            continue
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".safetensors":
            names = (
                tuple(f"joint_blocks.{layer}.weight" for layer in JOINT_LAYERS)
                if path.name == "cross_attn_weights.safetensors"
                else ("weight",)
            )
            write_safetensors(path, names)
        else:
            path.write_bytes(b"{}" if path.suffix == ".json" else b"")

    video = root / "creator/video_model"
    audio = root / "creator/audio_model"
    video.mkdir(parents=True, exist_ok=True)
    audio.mkdir(parents=True, exist_ok=True)
    (video / "config.json").write_text(json.dumps(VIDEO_CONFIG))
    (audio / "config.json").write_text(json.dumps(AUDIO_CONFIG))
    shard = "diffusion_pytorch_model-00001-of-00001.safetensors"
    write_safetensors(video / shard)
    (video / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": shard}})
    )


class DreamXInventoryTest(unittest.TestCase):
    def test_output_paths_are_precision_explicit(self):
        root = Path("out")
        self.assertEqual(
            output_paths(root, "q8_0"),
            (
                root / "DreamX-Creator-Q8_0.gguf",
                root / "mmproj-DreamX-Creator-BF16.gguf",
            ),
        )
        self.assertEqual(
            output_paths(root, "bf16"),
            (
                root / "DreamX-Creator-BF16.gguf",
                root / "mmproj-DreamX-Creator-BF16.gguf",
            ),
        )

    def test_inventory_requires_every_released_component(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            make_minimal_layout(root, omit="refiner/lightvae_nu_scheme3.pt")
            with self.assertRaisesRegex(FileNotFoundError, "lightvae_nu_scheme3.pt"):
                build_inventory(root)

    def test_inventory_accepts_only_released_dimensions_and_joint_layers(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            make_minimal_layout(root)
            inventory = build_inventory(root)
            self.assertEqual(inventory.video_config["dim"], 3072)
            self.assertEqual(inventory.audio_config["dim"], 1536)
            self.assertEqual(JOINT_LAYERS, tuple(range(15, 30)))

            config_path = root / "creator/audio_model/config.json"
            bad = dict(AUDIO_CONFIG, ffn_dim=8192)
            config_path.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, "audio ffn_dim"):
                build_inventory(root)

    def test_pair_id_is_canonical(self):
        left = pair_id({"video": {"count": 825, "bytes": 20}})
        right = pair_id({"video": {"bytes": 20, "count": 825}})
        self.assertEqual(left, right)
        self.assertEqual(len(left), 64)

    def test_pt_loader_uses_bounded_cpu_options_and_unwraps_ema(self):
        tensor = object()
        torch = Mock()
        torch.load.return_value = {"ema": {"weight": tensor}}
        torch.is_tensor.side_effect = lambda value: value is tensor

        state = load_pt_state_dict(Path("weights.pt"), torch)

        self.assertEqual(state, {"weight": tensor})
        torch.load.assert_called_once_with(
            Path("weights.pt"),
            map_location="cpu",
            mmap=True,
            weights_only=True,
        )

    def test_pt_loader_rejects_non_tensor_entries(self):
        torch = Mock()
        torch.load.return_value = {"weight": object(), "epoch": 4}
        torch.is_tensor.side_effect = lambda value: not isinstance(value, int)
        with self.assertRaisesRegex(ValueError, "non-tensor.*epoch"):
            load_pt_state_dict(Path("weights.pt"), torch)


if __name__ == "__main__":
    unittest.main()
