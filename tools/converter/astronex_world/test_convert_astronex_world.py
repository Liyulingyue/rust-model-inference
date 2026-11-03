import json
import tempfile
import unittest
from pathlib import Path

from tools.converter.astronex_world.convert_astronex_world import (
    SourceTensor,
    _write_pair_files,
    map_main_name,
    map_text_name,
    map_vae_name,
    read_safetensor_sources,
    _contract_shapes,
)
from tools.converter.dots.convert_dots_tts import (
    GGML_BF16,
    GGML_F32,
    read_gguf_directory,
    read_gguf_tensor_bytes,
)


def write_safetensors(path: Path, entries: dict[str, tuple[str, tuple[int, ...], bytes]]) -> None:
    header = {}
    offset = 0
    payload = bytearray()
    for name, (dtype, shape, raw) in entries.items():
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + len(raw)]}
        payload.extend(raw)
        offset += len(raw)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + payload)


class AstronexMappingTest(unittest.TestCase):
    def test_maps_native_component_names(self):
        self.assertEqual(map_main_name("model.blocks.0.self_attn.prope_o.weight"), "astronex.blocks.0.self_attn.prope_o.weight")
        self.assertEqual(map_text_name("shared.weight"), "astronex.text.token_embedding.weight")
        self.assertEqual(map_text_name("encoder.block.3.layer.1.DenseReluDense.wi_0.weight"), "astronex.text.blocks.3.ffn.gate.0.weight")
        self.assertEqual(map_vae_name("decoder.conv_in.weight"), "astronex.vae.decoder.conv1.weight")
        self.assertEqual(map_vae_name("encoder.down_blocks.1.resnets.0.norm1.gamma"), "astronex.vae.encoder.downsamples.1.downsamples.0.residual.0.gamma")

    def test_rejects_overlap_and_wrong_dtype(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "bad.safetensors"
            header = {
                "a": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]},
                "b": {"dtype": "BF16", "shape": [2], "data_offsets": [2, 6]},
            }
            encoded = json.dumps(header).encode()
            path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + b"\0" * 6)
            with self.assertRaisesRegex(ValueError, "overlap"):
                read_safetensor_sources("denoiser", (path,))

    def test_contract_catalog_has_exact_released_counts(self):
        self.assertEqual(len(_contract_shapes("denoiser")), 898)
        self.assertEqual(len(_contract_shapes("text")), 242)
        self.assertEqual(len(_contract_shapes("vae")), 196)
        vae_names = [map_vae_name(name) for name in _contract_shapes("vae")]
        self.assertEqual(len(vae_names), len(set(vae_names)))

    def test_lossless_tiny_pair_export_preserves_bf16_and_f32_payloads(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            bf16 = b"\x01\x80\x7f\x00"
            f32 = b"\x00\x00\x80\xbf"
            source = root / "source.safetensors"
            write_safetensors(source, {
                "model.patch_embedding.bias": ("BF16", (2,), bf16),
                "decoder.conv_in.bias": ("F32", (1,), f32),
            })
            sources = read_safetensor_sources("mixed", (source,))
            main = root / "main.gguf"
            components = root / "components.gguf"
            main_source = next(x for x in sources if x.name.startswith("model."))
            vae_source = next(x for x in sources if x.name.startswith("decoder."))
            _write_pair_files(
                main,
                components,
                [("main", map_main_name(main_source.name), GGML_BF16, main_source), ("components", map_vae_name(vae_source.name), GGML_F32, vae_source)],
                {"main": [("general.architecture", "astronex_world")], "components": [("general.architecture", "clip")]},
            )
            _, main_dir = read_gguf_directory(main)
            _, component_dir = read_gguf_directory(components)
            self.assertEqual(read_gguf_tensor_bytes(main, next(iter(main_dir))), bf16)
            self.assertEqual(read_gguf_tensor_bytes(components, next(iter(component_dir))), f32)

    def test_missing_real_model_is_rejected(self):
        from tools.converter.astronex_world.convert_astronex_world import export_model

        with tempfile.TemporaryDirectory() as raw:
            with self.assertRaises(FileNotFoundError):
                export_model(Path(raw), Path(raw) / "out")


if __name__ == "__main__":
    unittest.main()
