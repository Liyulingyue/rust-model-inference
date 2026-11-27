import unittest

import numpy as np

from tools.converter.physbrain.convert_physbrain import (
    QUANT_BLOCK,
    QUANT_FILE_TYPE,
    QUANT_MODES,
    QUANT_SUFFIX,
    _emit_bf16_quantized,
    _quant_target,
)
from tools.converter.utils.gguf import (
    GGML_BF16,
    GGML_Q4_0,
    GGML_Q4K,
    GGML_Q6K,
    GGML_Q8_0,
    GgufWriter,
    f32_to_bf16,
    read_gguf_directory,
    read_gguf_tensor_bytes,
)


class _RecordingWriter(GgufWriter):
    """A GgufWriter that keeps the tensors instead of writing a file."""

    def __init__(self) -> None:  # noqa: D107 - deliberately skips the parent init
        self.metadata: list = []
        self.tensors: list = []

    def write(self) -> None:
        pass


class QuantPolicyTests(unittest.TestCase):
    def test_every_mode_has_a_file_type_and_a_filename_suffix(self):
        for mode in QUANT_MODES:
            self.assertIn(mode, QUANT_FILE_TYPE)
            self.assertIn(mode, QUANT_SUFFIX)
            self.assertIn(mode, QUANT_BLOCK)
        # llama.cpp's llama_ftype values, so other tools read the same label.
        self.assertEqual(QUANT_FILE_TYPE["q8_0"], 7)
        self.assertEqual(QUANT_FILE_TYPE["q4_k_m"], 15)
        self.assertEqual(QUANT_FILE_TYPE["q6_k"], 18)

    def test_q4_k_m_promotes_the_same_two_tensors_as_llama_cpp(self):
        # llama.cpp's use_more_bits promotes the attention value projection and
        # the FFN down projection; their error lands in the residual stream
        # that every later layer reads.
        self.assertEqual(_quant_target("blk.0.attn_v.weight", "q4_k_m"), GGML_Q6K)
        self.assertEqual(_quant_target("blk.0.ffn_down.weight", "q4_k_m"), GGML_Q6K)
        for name in [
            "blk.0.attn_q.weight",
            "blk.0.attn_k.weight",
            "blk.0.attn_output.weight",
            "blk.0.ffn_gate.weight",
            "blk.0.ffn_up.weight",
            "token_embd.weight",
        ]:
            self.assertEqual(_quant_target(name, "q4_k_m"), GGML_Q4K)

    def test_uniform_modes_ignore_the_tensor_name(self):
        for mode, expected in [
            ("bf16", GGML_BF16),
            ("q8_0", GGML_Q8_0),
            ("q4_0", GGML_Q4_0),
            ("q6_k", GGML_Q6K),
        ]:
            for name in ["blk.3.attn_v.weight", "token_embd.weight"]:
                self.assertEqual(_quant_target(name, mode), expected)

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            _quant_target("blk.0.attn_q.weight", "q3_k_m")

    def test_block_alignment_is_enforced(self):
        # n_in = 100 is not a multiple of the 256-value K-quant block, and a
        # 32-value Q4_0 block either. Both must refuse rather than silently
        # produce a payload the loader cannot read.
        values = np.linspace(-1.0, 1.0, 100 * 4, dtype=np.float32)
        raw = f32_to_bf16(values.tobytes())
        for mode in ["q4_0", "q4_k_m", "q6_k"]:
            writer = _RecordingWriter()
            with self.assertRaises(ValueError) as caught:
                _emit_bf16_quantized(writer, "blk.0.ffn_up.weight", raw, (4, 100), mode)
            self.assertIn("block size", str(caught.exception))

    def test_payload_size_matches_the_declared_ggml_type(self):
        # 512 values: two K-quant blocks, four Q4_0 blocks, sixteen Q8_0 blocks.
        n = 512
        values = np.linspace(-1.0, 1.0, n, dtype=np.float32)
        raw = f32_to_bf16(values.tobytes())
        expected = {
            "bf16": n * 2,
            "q8_0": (n // 32) * 34,
            "q4_0": (n // 32) * 18,
            "q4_k_m": (n // 256) * 144,
            "q6_k": (n // 256) * 210,
        }
        for mode, size in expected.items():
            writer = _RecordingWriter()
            _emit_bf16_quantized(writer, "blk.0.ffn_up.weight", raw, (1, n), mode)
            self.assertEqual(len(writer.tensors), 1)
            _name, ggml_type, _dims, payload = writer.tensors[0]
            self.assertEqual(ggml_type, _quant_target("blk.0.ffn_up.weight", mode))
            self.assertEqual(len(payload), size, f"{mode} payload size")

    def test_q4_k_m_round_trips_through_the_writer(self):
        # A full writer rejects a payload whose length disagrees with the
        # declared type, so a wrong size must fail there rather than at load.
        import tempfile
        from pathlib import Path

        n = 512
        raw = f32_to_bf16(np.linspace(-1.0, 1.0, n, dtype=np.float32).tobytes())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.gguf"
            writer = GgufWriter(path)
            _emit_bf16_quantized(
                writer, "blk.0.ffn_up.weight", raw, (1, n), "q4_k_m",
            )
            writer.write()
            _metadata, tensors = read_gguf_directory(path)
            self.assertIn("blk.0.ffn_up.weight", tensors)
            _ggml_type, dims, length = tensors["blk.0.ffn_up.weight"]
            self.assertEqual(list(dims), [n, 1])
            self.assertEqual(length, (n // 256) * 144)
            self.assertEqual(
                len(read_gguf_tensor_bytes(path, "blk.0.ffn_up.weight")),
                (n // 256) * 144,
            )


if __name__ == '__main__':
    unittest.main()
