import tempfile
import unittest
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tools.converter.d3_turbo.convert import CONFIG_FILES, convert, encode_matrix
from tools.converter.utils.gguf import GGML_BF16, GGML_F16, GGML_F32, GGML_Q8_0, GgufWriter
from tools.converter.utils.gguf import f32_to_bf16, read_gguf_directory, read_gguf_tensor_bytes


class MatrixFormats(unittest.TestCase):
    def test_float_formats_and_quantized_row_alignment(self):
        values = np.arange(64, dtype=np.float32).reshape(2, 32) / 64
        for kind, expected, size in [("f32", GGML_F32, 256), ("f16", GGML_F16, 128),
                                     ("bf16", GGML_BF16, 128), ("q8_0", GGML_Q8_0, 68)]:
            dtype, payload = encode_matrix(values, kind)
            self.assertEqual(dtype, expected)
            self.assertEqual(len(payload), size)
        dtype, payload = encode_matrix(values[:, :3], "q8_0")
        self.assertEqual(dtype, GGML_F32)
        np.testing.assert_array_equal(np.frombuffer(payload, dtype="<f4"), values[:, :3].ravel())

    def test_unknown_formats_fail(self):
        with self.assertRaises(ValueError):
            encode_matrix(np.ones((1, 32), dtype=np.float32), "invented")

    def test_streamed_bf16_checkpoint_round_trips_and_rejects_overwrite(self):
        def save(path, tensors):
            header, data = {}, bytearray()
            for name, shape, values, dtype in tensors:
                payload = values.astype("<f4").tobytes()
                if dtype == "BF16":
                    payload = f32_to_bf16(payload)
                start = len(data)
                data.extend(payload)
                header[name] = dict(dtype=dtype, shape=shape, data_offsets=[start, len(data)])
            encoded = json.dumps(header).encode()
            path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + data)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in CONFIG_FILES:
                (root / name).write_text("{}")
            (root / "config.json").write_text(json.dumps(dict(model_type="qwen3_5", text_config=dict(hidden_size=256))))
            (root / "decision_config.json").write_text(json.dumps(dict(format_id="d3-code-readout-v1", attention_mode="prefix_bidir")))
            matrix = np.linspace(-1, 1, 512, dtype=np.float32).reshape(2, 256)
            save(root / "model.safetensors", [("language_model.test.weight", [2, 256], matrix, "BF16")])
            head = np.zeros((255, 256), dtype=np.float32)
            save(root / "readout.safetensors", [("weight", list(head.shape), head, "F32")])
            (root / "model.safetensors.index.json").write_text(json.dumps(dict(weight_map={"language_model.test.weight":"model.safetensors"})))
            for kind in ["f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "q4_k", "q5_k", "q6_k"]:
                output = root / f"{kind}.gguf"
                convert(root, output, kind)
                metadata, tensors = read_gguf_directory(output)
                self.assertEqual(metadata["general.architecture"], "d3_turbo")
                self.assertNotIn("d3_turbo.source_revision", metadata)
                self.assertEqual(tensors["weight"][0], GGML_F32)
                self.assertEqual(read_gguf_tensor_bytes(output, "weight"), head.tobytes())
                self.assertFalse(output.with_suffix(".gguf.partial").exists())
                with self.assertRaises(FileExistsError):
                    convert(root, output, kind)
                if kind == "q4_k":
                    parallel = root / "parallel.gguf"
                    convert(root, parallel, kind, workers=2)
                    self.assertEqual(output.read_bytes(), parallel.read_bytes())
            output = root / "late-output.gguf"
            write = GgufWriter.write

            def publish_another_output(writer):
                write(writer)
                output.write_bytes(b"keep other output")

            with patch.object(GgufWriter, "write", publish_another_output):
                with self.assertRaises(FileExistsError):
                    convert(root, output, "bf16")
            self.assertEqual(output.read_bytes(), b"keep other output")
            self.assertFalse(output.with_suffix(".gguf.partial").exists())

            output = root / "late-partial.gguf"
            partial = output.with_suffix(".gguf.partial")
            add_tensor = GgufWriter.add_tensor_chunks

            def reserve_another_partial(writer, *args):
                partial.write_bytes(b"keep other partial")
                add_tensor(writer, *args)

            with patch.object(GgufWriter, "add_tensor_chunks", reserve_another_partial):
                with self.assertRaises(FileExistsError):
                    convert(root, output, "bf16")
            self.assertEqual(partial.read_bytes(), b"keep other partial")
            self.assertFalse(output.exists())
            save(root / "readout.safetensors", [("weight", list(head.shape), head, "BF16")])
            with self.assertRaisesRegex(ValueError, "readout must use F32"):
                convert(root, root / "bad-readout.gguf", "bf16")

    def test_nonfinite_and_f16_overflow_are_rejected(self):
        for value in [np.inf, np.nan, 1e10]:
            with self.assertRaises(ValueError):
                encode_matrix(np.full((1, 32), value, np.float32), "f16")


if __name__ == "__main__":
    unittest.main()
