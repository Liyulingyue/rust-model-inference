import hashlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tools.converter.underdog_woof.convert_underdog_woof import (
    CONTRACT, conv_taps, conv_transpose_matrix, convert, dequantize, encode_tensor,
    linear_weights, verify_sources,
)
from tools.converter.utils.gguf import (
    GGML_BF16, GGML_F16, GGML_F32, GGML_Q4_0, GGML_Q4_1,
    GGML_Q4K, GGML_Q5K, GGML_Q6K, GGML_Q8_0,
)


class ConversionTests(unittest.TestCase):
    def test_grouped_int8_nonzero_axis_and_partial_group(self):
        entry = {
            "q8": np.array([[1, -2, 3, -4, 5, -6], [-7, 8, -9, 10, -11, 12]], dtype=np.int8),
            "s": np.array([[0.5, 2], [0.25, 3]], dtype=np.float32),
            "shape": [3, 2, 2], "axis": 1, "g": 4,
        }
        expected_rows = np.array([[0.5, -1, 1.5, -2, 10, -12],
                                  [-1.75, 2, -2.25, 2.5, -33, 36]], dtype=np.float32)
        expected = np.moveaxis(expected_rows.reshape(2, 3, 2), 0, 1)
        np.testing.assert_array_equal(dequantize(entry), expected)

    def test_grouped_int8_rejects_wrong_scale_shape(self):
        with self.assertRaisesRegex(ValueError, "grouped-int8"):
            dequantize({"q8": np.ones((2, 5), np.int8), "s": np.ones((2, 1), np.float32),
                        "shape": [2, 5], "axis": 0, "g": 4})

    def test_linear_row_scale_and_native_orientation(self):
        weights, bias = linear_weights({"q": np.array([[1, 2, 3], [4, 5, 6]], np.int8),
                                       "s": np.array([0.5, 2], np.float32)})
        np.testing.assert_array_equal(weights, [[0.5, 1, 1.5], [8, 10, 12]])
        self.assertIsNone(bias)

    def test_conv_taps_match_causal_projection(self):
        weight = np.arange(24, dtype=np.float32).reshape(2, 4, 3)
        taps = conv_taps(weight)
        x = np.arange(20, dtype=np.float32).reshape(5, 4)
        expected = sum(x[j:j + 3] @ weight[:, :, j].T for j in range(3))
        actual = sum(x[j:j + 3] @ tap.T for j, tap in enumerate(taps))
        np.testing.assert_array_equal(actual, expected)

    def test_conv_transpose_keeps_time_before_output_channel(self):
        weight = np.arange(48, dtype=np.float32).reshape(3, 2, 8)
        matrix = conv_transpose_matrix(weight)
        x = np.array([[1, 2, 3], [-2, 4, 1]], dtype=np.float32)
        expected = np.stack([np.einsum("i,ioj->jo", row, weight).ravel() for row in x])
        np.testing.assert_array_equal(x @ matrix.T, expected)
        self.assertEqual(matrix.shape, (16, 3))

    def test_format_decisions_preserve_lookups_vectors_and_unaligned_rows(self):
        aligned = np.arange(64, dtype=np.float32).reshape(2, 32)
        self.assertEqual(encode_tensor("speech.ce.weight", aligned, "q8_0")[0], GGML_Q8_0)
        self.assertEqual(encode_tensor("speech.ce.weight", aligned[:, :31], "q8_0")[0], GGML_F32)
        self.assertEqual(encode_tensor("speech.ce.bias", aligned[0], "f16")[0], GGML_F32)
        self.assertEqual(encode_tensor("voices.juno", aligned, "q8_0")[0], GGML_F32)
        self.assertEqual(encode_tensor("speech.text_emb.weight", aligned, "bf16")[0], GGML_F32)
        self.assertEqual(encode_tensor("speech.ce.weight", aligned, "f16")[0], GGML_F16)
        self.assertEqual(encode_tensor("speech.ce.weight", aligned, "bf16")[0], GGML_BF16)
        self.assertEqual(encode_tensor("speech.ce.weight", aligned, "q4_k")[0], GGML_F32)

    def test_mismatched_source_hash_fails_before_loading_weights(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            name = next(iter(CONTRACT["source"]["hashes"]))
            path = directory / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"not the pinned source")
            with self.assertRaisesRegex(ValueError, "source hash mismatch"):
                verify_sources(directory)

    def test_all_block_formats_use_the_shared_payload_types(self):
        value = np.linspace(-2, 3, 256, dtype=np.float32).reshape(1, 256)
        for fmt, expected in (("q4_0", GGML_Q4_0), ("q4_1", GGML_Q4_1),
                              ("q4_k", GGML_Q4K), ("q5_k", GGML_Q5K), ("q6_k", GGML_Q6K)):
            with self.subTest(format=fmt):
                self.assertEqual(encode_tensor("projection.weight", value, fmt)[0], expected)

    def test_nonfinite_weights_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-finite"):
            encode_tensor("projection.weight", np.array([[np.nan]], np.float32), "f32")

    def test_output_guard_preserves_an_existing_file(self):
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw) / "existing.gguf"
            output.write_bytes(b"existing data")
            with self.assertRaises(FileExistsError):
                convert(Path(raw) / "missing-source", output)
            self.assertEqual(output.read_bytes(), b"existing data")

    def test_failed_write_cleans_temporary_output(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            (directory / "LICENSE").write_text("license")
            (directory / "NOTICE").write_text("notice")
            (directory / "underdog_woof").mkdir()
            (directory / "underdog_woof/respell.tsv").write_text("word\tword\n")
            output = directory / "output.gguf"
            module = "tools.converter.underdog_woof.convert_underdog_woof."
            with (patch(module + "verify_sources"), patch(module + "np.load"),
                  patch(module + "unflatten"), patch(module + "export_tensors", return_value={}),
                  patch(module + "GgufWriter.write", side_effect=OSError("disk full"))):
                with self.assertRaisesRegex(OSError, "disk full"):
                    convert(directory, output)
            self.assertFalse(output.exists())
            self.assertEqual(list(directory.glob(".output.gguf.*.tmp")), [])

    def test_download_streams_verifies_and_reuses_pinned_files(self):
        from tools.oracle.underdog_woof.download import download
        payload = b"pinned weights"
        contract = {"source": {"repository": "test/model", "revision": "pinned",
                    "weight_size": len(payload),
                    "hashes": {"woof_int8.npz": hashlib.sha256(payload).hexdigest()}}}
        with tempfile.TemporaryDirectory() as raw:
            with (patch("tools.oracle.underdog_woof.download.CONTRACT", contract),
                  patch("urllib.request.urlopen", return_value=io.BytesIO(payload)) as request):
                download(raw)
                download(raw)
                self.assertEqual(request.call_count, 1)
            self.assertEqual((Path(raw) / "woof_int8.npz").read_bytes(), payload)
            self.assertEqual(len(list(Path(raw).iterdir())), 1)

    def test_download_hash_failure_preserves_destination_and_removes_partial(self):
        from tools.oracle.underdog_woof.download import download
        contract = {"source": {"repository": "test/model", "revision": "pinned",
                    "hashes": {"LICENSE": hashlib.sha256(b"expected").hexdigest()}}}
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "LICENSE"
            path.write_bytes(b"existing")
            with (patch("tools.oracle.underdog_woof.download.CONTRACT", contract),
                  patch("urllib.request.urlopen", return_value=io.BytesIO(b"bad"))):
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    download(raw)
            self.assertEqual(path.read_bytes(), b"existing")
            self.assertEqual(list(Path(raw).iterdir()), [path])


if __name__ == "__main__":
    unittest.main()
