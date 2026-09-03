import tempfile
import unittest
from pathlib import Path

from convert_dots_tts import (
    GGML_BF16,
    GGML_F32,
    GgufWriter,
    read_gguf_directory,
    read_gguf_tensor_bytes,
    validate_variant,
)


class ExportContractTest(unittest.TestCase):
    def test_bf16_bits_and_clip_metadata_survive_readback(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "tiny.gguf"
            writer = GgufWriter(path)
            writer.add_meta("general.architecture", "clip")
            writer.add_meta("clip.has_audio_encoder", True)
            writer.add_meta("clip.has_gen_audio_encoder", True)
            writer.add_meta("clip.audio.projector_type", "dotstts_spkenc")
            writer.add_meta("clip.gen.audio.projector_type", "dotstts_gen")
            bf16 = bytes.fromhex("803f20c0")
            writer.add_tensor("bf16.weight", GGML_BF16, (2,), bf16)
            writer.add_tensor("f32.bias", GGML_F32, (1,), bytes.fromhex("0000803f"))
            writer.write()

            metadata, tensors = read_gguf_directory(path)
            self.assertEqual(metadata["general.architecture"], "clip")
            self.assertEqual(metadata["clip.gen.audio.projector_type"], "dotstts_gen")
            self.assertEqual(tensors["bf16.weight"], (GGML_BF16, (2,), 4))
            self.assertEqual(read_gguf_tensor_bytes(path, "bf16.weight"), bf16)

    def test_duplicate_names_and_implicit_overwrite_are_rejected(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "tiny.gguf"
            writer = GgufWriter(path)
            writer.add_meta("general.architecture", "clip")
            with self.assertRaises(ValueError):
                writer.add_meta("general.architecture", "clip")
            writer.add_tensor("x", GGML_F32, (1,), bytes(4))
            with self.assertRaises(ValueError):
                writer.add_tensor("x", GGML_F32, (1,), bytes(4))
            writer.write()
            with self.assertRaises(FileExistsError):
                writer.write()
            writer.write(overwrite=True)

    def test_variant_is_explicitly_bounded(self):
        self.assertEqual(validate_variant("base"), "base")
        self.assertEqual(validate_variant("edit"), "edit")
        with self.assertRaises(ValueError):
            validate_variant("experimental")


if __name__ == "__main__":
    unittest.main()
