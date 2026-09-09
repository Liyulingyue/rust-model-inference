"""Runnable with python -m unittest discover -s tools/breeze -p 'test_compare*.py'."""
import json
import struct
import tempfile
import unittest
from pathlib import Path

from compare_breeze_trace import compare


class ExactTraceTest(unittest.TestCase):
    def test_comparison_rejects_missing_evidence_and_first_divergence(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            oracle, native = base / "oracle.jsonl", base / "native.jsonl"
            payload = base / "a.f32"
            payload.write_bytes(struct.pack("<ff", 1.0, 0.0))
            row = {"name": "breeze.text.norm", "layer": None, "step": None,
                   "shape": [1, 2], "len": 2, "occurrence": 0,
                   "binary_path": "a.f32"}
            token = {"name": "breeze.frame", "shape": [2], "len": 2, "token_ids": [7, 9]}

            def write(path, rows):
                path.write_text("".join(json.dumps(value) + "\n" for value in rows))

            write(oracle, [row, token])
            write(native, [row, token])
            self.assertEqual(compare(oracle, native)["status"], "identical")
            for key, value in [("layer", 1), ("step", 3), ("name", "different")]:
                write(native, [{**row, key: value}, token])
                result = compare(oracle, native)
                self.assertEqual((result["status"], result["field"]), ("metadata_mismatch", key))
            write(native, [{**row, "occurrence": 1}, token])
            with self.assertRaisesRegex(ValueError, "occurrence"):
                compare(oracle, native)
            write(native, [token, row])
            self.assertEqual(compare(oracle, native)["status"], "metadata_mismatch")
            write(native, [row])
            self.assertEqual(compare(oracle, native)["status"], "record_count_mismatch")
            write(native, [row, {**token, "token_ids": [7, 10]}])
            self.assertEqual(compare(oracle, native)["element"], 1)
            changed = base / "changed.f32"
            changed.write_bytes(struct.pack("<ff", 1.0, -0.0))
            write(native, [{**row, "binary_path": "changed.f32"}, token])
            result = compare(oracle, native)
            self.assertEqual((result["status"], result["element"], result["coordinate"]),
                             ("bit_mismatch", 1, [0, 1]))
            self.assertEqual((result["oracle_u32"], result["native_u32"]), (0, 0x80000000))
            changed.write_bytes(b"\0" * 4)
            with self.assertRaisesRegex(ValueError, "expected 8 bytes"):
                compare(oracle, native)
            changed.unlink()
            with self.assertRaises(FileNotFoundError):
                compare(oracle, native)
            for bad in [{key: value for key, value in row.items() if key != "binary_path"},
                        {**row, "shape": [3]}, {**row, "token_ids": [1, 2]}]:
                write(native, [bad])
                with self.assertRaises(ValueError):
                    compare(oracle, native)
            write(native, [])
            with self.assertRaisesRegex(ValueError, "empty trace"):
                compare(native, native)
            empty = {"name": "breeze.frame", "shape": [0], "len": 0, "token_ids": []}
            write(native, [empty])
            with self.assertRaisesRegex(ValueError, "no values"):
                compare(native, native)


if __name__ == "__main__":
    unittest.main()
