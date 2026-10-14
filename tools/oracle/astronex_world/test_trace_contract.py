"""Trace contract tests: a bit mismatch is reported at its first checkpoint."""
import tempfile
from pathlib import Path
import unittest

from trace_official import Trace, compare


class TraceContract(unittest.TestCase):
    def test_first_divergence_and_duplicate_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            left, right = root / "left", root / "right"
            for out in (left, right):
                trace = Trace(out)
                import torch
                trace.emit("fixture.value", torch.tensor([[1.0, 2.0]], dtype=torch.float32))
                trace.emit("fixture.token_ids", [4, 7], kind="token_ids")
                trace.finish()
            compare(left, right)
            raw = bytearray((right / "000000.fixture.value.0.f32").read_bytes())
            raw[0] ^= 1
            (right / "000000.fixture.value.0.f32").write_bytes(raw)
            with self.assertRaisesRegex(AssertionError, "first divergence fixture.value"):
                compare(left, right)


if __name__ == "__main__":
    unittest.main()
