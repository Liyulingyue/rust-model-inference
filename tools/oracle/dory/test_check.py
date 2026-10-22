import json
import struct
import tempfile
import unittest
from pathlib import Path

import check


class TraceChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def manifest(self, side, *, value=1.0, name="l_out", generated=()):
        binary = self.root / f"{side}.f32"
        binary.write_bytes(struct.pack("<f", value))
        path = self.root / side
        baseline = self.root / f"{side}.baseline.f32"
        baseline.write_bytes(struct.pack("<f", 1.0))
        records = [dict(name="prompt_ids", token_ids=[1]),
                   dict(name="generated_ids", token_ids=list(generated))]
        for checkpoint, layer in [("embd", None)] + [("l_out", i) for i in range(192)] + [
            ("result_norm", None), ("result_output", None)
        ]:
            records.append(dict(name=checkpoint, layer=layer, step=0, shape=[1],
                                binary_path=str(binary if checkpoint == name else baseline)))
        path.write_text("\n".join(map(json.dumps, records)))
        return path

    def test_lossy_tolerance_does_not_weaken_embedding_gate(self):
        oracle = self.manifest("oracle")
        rust = self.manifest("rust", value=1.000001)
        self.assertTrue(check.compare(oracle, rust)["passed"])
        oracle = self.manifest("oracle", name="embd")
        rust = self.manifest("rust", name="embd", value=1.000001)
        self.assertFalse(check.compare(oracle, rust)["passed"])

    def test_real_numeric_and_token_divergences_fail(self):
        oracle = self.manifest("oracle")
        self.assertFalse(check.compare(oracle, self.manifest("rust", value=1.1))["passed"])
        self.assertFalse(check.compare(oracle, self.manifest("rust", generated=(3,)))["passed"])

    def test_empty_missing_and_duplicate_records_fail(self):
        oracle, rust = self.manifest("oracle"), self.manifest("rust")
        rust.write_text("")
        self.assertFalse(check.compare(oracle, rust)["passed"])
        rust = self.manifest("rust")
        oracle_rows, rust_rows = oracle.read_text().splitlines(), rust.read_text().splitlines()
        oracle.write_text("\n".join(oracle_rows[:-1]))
        rust.write_text("\n".join(rust_rows[:-1]))
        self.assertFalse(check.compare(oracle, rust)["passed"])
        oracle, rust = self.manifest("oracle"), self.manifest("rust")
        rust.write_text(rust.read_text() + "\n" + rust.read_text().splitlines()[-1])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            check.compare(oracle, rust)

    def test_nonfinite_and_truncated_payloads_fail(self):
        oracle = self.manifest("oracle")
        rust = self.manifest("rust", value=float("nan"))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            check.compare(oracle, rust)
        (self.root / "rust.f32").write_bytes(b"\0")
        with self.assertRaisesRegex(ValueError, "shape"):
            check.compare(oracle, rust)


if __name__ == "__main__":
    unittest.main()
