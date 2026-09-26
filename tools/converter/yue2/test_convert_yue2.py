from __future__ import annotations

import base64
import json
import struct
import tempfile
import unittest
from pathlib import Path

from tools.converter.utils.gguf import read_gguf_directory
from tools.converter.yue2.convert_yue2 import convert_main, convert_vae


EOD = 151643


def write_safetensors(path: Path, tensors: dict[str, tuple[str, list[int], bytes]]) -> None:
    header, offset = {}, 0
    for name, (dtype, shape, raw) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(raw)]}
        offset += len(raw)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(v[2] for v in tensors.values()))


def write_qwen_tiktoken(path: Path) -> None:
    rank = 0
    with path.open("w", encoding="ascii") as output:
        for byte in range(256):
            output.write(f"{base64.b64encode(bytes([byte])).decode()} {rank}\n")
            rank += 1
        for left in range(256):
            for right in range(256):
                output.write(f"{base64.b64encode(bytes([left, right])).decode()} {rank}\n")
                rank += 1
        for first in range(256):
            for second in range(256):
                for third in range(256):
                    if rank == EOD:
                        return
                    raw = bytes([first, second, third])
                    output.write(f"{base64.b64encode(raw).decode()} {rank}\n")
                    rank += 1
    raise AssertionError(f"generated {rank} tokenizer entries, expected {EOD}")


MAIN_CONFIG = {
    "architectures": ["YuE2ForCausalLM"],
    "dtype": "bfloat16",
    "model_type": "yue2",
    "hidden_size": 2048,
    "num_hidden_layers": 28,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "intermediate_size": 6144,
    "vocab_size": 184704,
    "rms_norm_eps": 0.000001,
    "rope_theta": 1000000,
    "max_position_embeddings": 24576,
    "latent_type": "vae",
    "latent_dim": 64,
    "max_latent_frames": 24576,
    "timestep_shift": 1.0,
}

VAE_CONFIG = {
    "architectures": ["YuE2VAE"],
    "dtype": "float32",
    "model_type": "yue2_vae",
    "decoder_config": {
        "channels": 64,
        "latent_dim": 64,
        "out_channels": 2,
        "strides": [2, 2, 4, 4, 5, 6],
    },
    "sample_rate": 48000,
    "latent_dim": 64,
    "downsampling_ratio": 1920,
    "audio_channels": 2,
    "release_variant": "standard",
    "decode_core_frames": 1024,
    "decode_halo_frames": 16,
}


class ConvertYuE2Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sources = tempfile.TemporaryDirectory()
        root = Path(cls.sources.name)
        cls.main_dir = root / "main"
        cls.bad_main_dir = root / "bad-main"
        cls.bad_shape_main_dir = root / "bad-shape-main"
        cls.vae_dir = root / "vae"
        cls.legacy_vae_dir = root / "legacy-vae"
        for directory in (
            cls.main_dir,
            cls.bad_main_dir,
            cls.bad_shape_main_dir,
            cls.vae_dir,
            cls.legacy_vae_dir,
        ):
            directory.mkdir()

        q_weight = bytes(2048 * 2048 * 2)
        write_safetensors(
            cls.main_dir / "model.safetensors",
            {
                "model.layers.0.self_attn.q_proj.weight": ("BF16", [2048, 2048], q_weight),
                "model.layers.0.nar_self_attn.q_proj.weight": ("BF16", [2048, 2048], q_weight),
            },
        )
        (cls.main_dir / "config.json").write_text(json.dumps(MAIN_CONFIG))
        write_qwen_tiktoken(cls.main_dir / "qwen.tiktoken")

        write_safetensors(
            cls.bad_main_dir / "model.safetensors",
            {"model.layers.0.self_attn.q_proj.weight": ("F32", [2048, 2048], bytes(4))},
        )
        (cls.bad_main_dir / "config.json").write_text(json.dumps(MAIN_CONFIG))

        write_safetensors(
            cls.bad_shape_main_dir / "model.safetensors",
            {"model.layers.0.self_attn.q_proj.weight": ("BF16", [2, 2], bytes(8))},
        )
        (cls.bad_shape_main_dir / "config.json").write_text(json.dumps(MAIN_CONFIG))

        write_safetensors(
            cls.vae_dir / "model.safetensors",
            {
                "decoder.layers.0.bias": ("F32", [2048], bytes(2048 * 4)),
                "encoder.layers.0.bias": ("F32", [64], bytes(64 * 4)),
            },
        )
        (cls.vae_dir / "config.json").write_text(json.dumps(VAE_CONFIG))

        write_safetensors(
            cls.legacy_vae_dir / "model.safetensors",
            {"decoder.layers.0.bias": ("F32", [2048], bytes(2048 * 4))},
        )
        legacy = {**VAE_CONFIG, "release_variant": "legacy"}
        (cls.legacy_vae_dir / "config.json").write_text(json.dumps(legacy))

    @classmethod
    def tearDownClass(cls) -> None:
        cls.sources.cleanup()

    def setUp(self) -> None:
        self.outputs = tempfile.TemporaryDirectory()
        output_dir = Path(self.outputs.name)
        self.main_out = output_dir / "yue2.gguf"
        self.vae_out = output_dir / "yue2-vae.gguf"

    def tearDown(self) -> None:
        self.outputs.cleanup()

    def test_main_metadata_tokenizer_and_checkpoint_native_names(self) -> None:
        convert_main(self.main_dir, self.main_out)
        meta, tensors = read_gguf_directory(self.main_out)
        self.assertEqual(meta["general.architecture"], "yue2")
        self.assertEqual(meta["yue2.protocol_version"], "yue2-native-v1")
        self.assertEqual(meta["yue2.vocab_size"], 184704)
        self.assertEqual(meta["yue2.codec_offset"], 151853)
        self.assertEqual(meta["yue2.codec_size"], 32768)
        self.assertEqual(len(meta["tokenizer.ggml.tokens"]), 184704)
        self.assertEqual(meta["tokenizer.ggml.pre"], "qwen2")
        self.assertIs(meta["tokenizer.ggml.normalizer.nfc"], True)
        self.assertIn("model.layers.0.nar_self_attn.q_proj.weight", tensors)

    def test_vae_keeps_only_f32_decoder_tensors(self) -> None:
        convert_vae(self.vae_dir, self.vae_out)
        meta, tensors = read_gguf_directory(self.vae_out)
        self.assertEqual(meta["general.architecture"], "yue2_vae")
        self.assertEqual(meta["yue2_vae.strides"], [2, 2, 4, 4, 5, 6])
        self.assertEqual(meta["yue2_vae.latent_channels"], 64)
        self.assertEqual(meta["yue2_vae.output_channels"], 2)
        self.assertTrue(tensors)
        self.assertTrue(all(name.startswith("decoder.") for name in tensors))

    def test_converter_rejects_wrong_dtype_shape_variant_and_overwrite(self) -> None:
        with self.assertRaisesRegex(ValueError, "model.layers.0.self_attn.q_proj.weight.*BF16"):
            convert_main(self.bad_main_dir, self.main_out)
        with self.assertRaisesRegex(ValueError, "model.layers.0.self_attn.q_proj.weight.*shape"):
            convert_main(self.bad_shape_main_dir, self.main_out)
        with self.assertRaisesRegex(ValueError, "release_variant.*standard"):
            convert_vae(self.legacy_vae_dir, self.vae_out)
        self.main_out.write_bytes(b"keep")
        with self.assertRaises(FileExistsError):
            convert_main(self.main_dir, self.main_out)
        self.assertEqual(self.main_out.read_bytes(), b"keep")


if __name__ == "__main__":
    unittest.main()
