import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from compare import compare


def trace(root, values, name="hy.logits"):
    root.mkdir(exist_ok=True)
    binary = root / "data.f32"
    np.asarray(values, dtype="<f4").tofile(binary)
    record = dict(name=name, occurrence=0, shape=[len(values)], binary_path=str(binary))
    (root / "trace.jsonl").write_text(json.dumps(record) + "\n")
    (root / "turns.json").write_text("[[1]]")


class ComparisonTests(unittest.TestCase):
    def test_lossy_budget_requires_an_independent_reference_and_equal_context(self):
        with tempfile.TemporaryDirectory() as d:
            ref, got, baseline = [Path(d) / p for p in ("ref", "got", "baseline")]
            for root, x in ((ref, [1, 2, 3]), (got, [1.05, 2.05, 3.05]), (baseline, [1.1, 2.1, 3.1])):
                trace(root, x)
            with self.assertRaises(AssertionError):
                compare(ref, got)
            self.assertIn("lossy_calibration", compare(ref, got, baseline))
            for alias in (ref, got):
                with self.assertRaises(AssertionError):
                    compare(ref, got, alias)
            (baseline / "turns.json").write_text("[[2]]")
            with self.assertRaises(AssertionError):
                compare(ref, got, baseline)
            (ref / "turns.json").unlink()
            with self.assertRaises(AssertionError):
                compare(ref, got, baseline)

    def test_nonfinite_shapes_and_checkpoint_names_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            ref, got = Path(d) / "ref", Path(d) / "got"
            trace(ref, [1, 2, 3])
            for x, name in (([1, float("nan"), 3], "hy.logits"),
                            ([1, 2], "hy.logits"), ([1, 2, 3], "hy.projected")):
                trace(got, x, name)
                with self.assertRaises(AssertionError):
                    compare(ref, got)

    def test_strict_projection_check_accepts_zero_and_rejects_material_error(self):
        with tempfile.TemporaryDirectory() as d:
            ref, got = Path(d) / "ref", Path(d) / "got"
            trace(ref, [0, 0], "hy.projection.test")
            trace(got, [0, 0], "hy.projection.test")
            self.assertEqual(compare(ref, got)["checkpoints"][0]["cosine"], 1)
            trace(got, [0.01, 0], "hy.projection.test")
            with self.assertRaises(AssertionError):
                compare(ref, got)


if __name__ == "__main__":
    unittest.main()
