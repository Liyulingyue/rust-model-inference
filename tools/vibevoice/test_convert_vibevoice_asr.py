import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.dots.convert_dots_tts import Tensor
from tools.vibevoice import convert_vibevoice_asr as converter
from tools.vibevoice.convert_vibevoice_asr import (
    ENCODER_DEPTHS,
    GGML_Q8_0,
    LLM_FILENAME,
    MMPROJ_FILENAME,
    encoder_tensor_rules,
    encoder_tensor_shapes,
    output_paths,
    quantize_q8_0,
    require_tensor,
)
from tools.vibevoice.vibevoice_llm_oracle import (
    assemble_input_rows,
    safetensors_name,
    tensor_to_f32,
)

shared_dots = sys.modules["convert_dots_tts"]


class FakeReader:
    def __init__(self, tensor):
        self.value = tensor

    def tensor(self, name):
        if name != self.value.name:
            raise KeyError(name)
        return self.value


class ConverterContractTests(unittest.TestCase):
    def test_import_keeps_shared_q8_row_width_validation(self):
        with self.assertRaisesRegex(ValueError, "row width"):
            shared_dots._tensor_nbytes(GGML_Q8_0, (16, 2))

    def test_output_paths_include_storage_types(self):
        llm, mmproj = output_paths(Path("/tmp/out"))
        self.assertEqual(llm.name, LLM_FILENAME)
        self.assertEqual(mmproj.name, MMPROJ_FILENAME)
        self.assertEqual(LLM_FILENAME, "VibeVoice-ASR-Streaming-7B-Q8_0.gguf")
        self.assertEqual(MMPROJ_FILENAME, "mmproj-VibeVoice-ASR-Streaming-7B-BF16.gguf")

    def test_q8_0_rounds_half_away_from_zero_and_encodes_zero_block(self):
        values = np.zeros(64, dtype=np.float32)
        values[:5] = [-127.0, -0.5, 0.5, 1.5, 127.0]
        raw = quantize_q8_0(values)
        blocks = np.frombuffer(raw, dtype=np.uint8).reshape(2, 34)
        self.assertEqual(blocks[0, :2].copy().view(np.float16)[0], np.float16(1.0))
        self.assertEqual(blocks[0, 2:7].view(np.int8).tolist(), [-127, -1, 1, 2, 127])
        self.assertEqual(blocks[1, :2].copy().view(np.float16)[0], np.float16(0.0))
        self.assertEqual(blocks[1, 2:].view(np.int8).tolist(), [0] * 32)

    def test_require_tensor_rejects_wrong_shape_and_dtype(self):
        tensor = Tensor("weight", "BF16", (2, 32), bytes(128))
        reader = FakeReader(tensor)
        self.assertIs(require_tensor(reader, "weight", (2, 32)), tensor)
        with self.assertRaisesRegex(ValueError, "expected shape"):
            require_tensor(reader, "weight", (32, 2))
        with self.assertRaisesRegex(ValueError, "expected dtype"):
            require_tensor(reader, "weight", (2, 32), ("F32",))

    def test_encoder_inventory_has_a_shape_for_every_rule(self):
        config = {"encoder_n_filters": 32, "encoder_ratios": [8, 5, 5, 4, 2, 2]}
        shapes = encoder_tensor_shapes(config, 64)
        rules = encoder_tensor_rules("acoustic")
        self.assertEqual(set(shapes), {source for source, _ in rules})
        self.assertEqual(len(rules), 2 + sum(2 + depth * 10 for depth in ENCODER_DEPTHS))
        self.assertEqual(shapes["downsample_layers.6.0.conv.conv.weight"], (2048, 1024, 16))
        self.assertEqual(shapes["head.conv.conv.weight"], (64, 2048, 7))

    def test_q8_payload_survives_atomic_gguf_readback(self):
        with tempfile.TemporaryDirectory() as raw_dir:
            path = Path(raw_dir) / "tiny.gguf"
            payload = quantize_q8_0(np.arange(32, dtype=np.float32))
            writer = converter.GgufWriter(path)
            writer.add_meta("general.architecture", "qwen2")
            writer.add_tensor("weight", GGML_Q8_0, (32, 1), payload)
            writer.write()

            metadata, tensors = converter._dots.read_gguf_directory(path)
            self.assertEqual(metadata["general.architecture"], "qwen2")
            self.assertEqual(tensors, {"weight": (GGML_Q8_0, (32, 1), 34)})
            self.assertEqual(converter._dots.read_gguf_tensor_bytes(path, "weight"), payload)

    def test_llm_oracle_inserts_audio_between_speech_tokens(self):
        embeddings = np.arange(12, dtype=np.float32).reshape(3, 4)
        token_embeddings = np.arange(40, dtype=np.float32).reshape(10, 4)
        rows, positions = assemble_input_rows(
            token_embeddings, [2, 3], embeddings, 7, 8
        )
        np.testing.assert_array_equal(
            rows,
            np.vstack([token_embeddings[[2, 3, 7]], embeddings, token_embeddings[[8]]]),
        )
        np.testing.assert_array_equal(positions, np.arange(7))

    def test_llm_safetensors_source_maps_names_and_preserves_matrix_layout(self):
        self.assertEqual(
            safetensors_name("token_embd.weight"),
            "model.language_model.embed_tokens.weight",
        )
        self.assertEqual(safetensors_name("output.weight"), "lm_head.weight")
        self.assertEqual(
            safetensors_name("blk.12.attn_output.weight"),
            "model.language_model.layers.12.self_attn.o_proj.weight",
        )
        self.assertEqual(
            safetensors_name("blk.27.ffn_norm.weight"),
            "model.language_model.layers.27.post_attention_layernorm.weight",
        )
        with self.assertRaisesRegex(KeyError, "unsupported canonical tensor"):
            safetensors_name("blk.0.unknown.weight")

        values = np.array([[1.0, -2.0], [3.5, 0.25]], dtype=np.float32)
        raw = (values.view(np.uint32) >> np.uint32(16)).astype("<u2").tobytes()
        tensor = Tensor("matrix", "BF16", values.shape, raw)
        np.testing.assert_array_equal(tensor_to_f32(tensor), values)


if __name__ == "__main__":
    unittest.main()
