import unittest
import json
import tempfile
from pathlib import Path
import numpy as np
from tools.oracle.sensenova_loop.compare import metrics, compare


class ComparisonTests(unittest.TestCase):
    def test_accepts_roundoff_but_rejects_real_error(self):
        self.assertTrue(metrics([1.000001, 2.0], [1.0, 2.0])["pass"])
        self.assertFalse(metrics([1.1, 2.0], [1.0, 2.0])["pass"])

    def test_nonfinite_or_shape_changes_are_errors(self):
        for actual in ([float("nan")], [[1.0]]):
            with self.assertRaises(ValueError):
                metrics(actual, [1.0])

    def test_prefill_chunks_compare_only_the_final_position_logits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixtures = [
                {"generated": [1], "steps": [{"rows": 3, "visits": [[3, 2], [3, 2]]}]}
            ]
            (root / "fixtures.json").write_text(json.dumps(fixtures))
            (root / "results.json").write_text(json.dumps([{"generated": [1]}]))
            records = []

            def record(name, values):
                path = root / f"{len(records)}.f32"
                np.asarray(values, dtype="<f4").tofile(path)
                records.append(
                    {"name": name, "shape": [1, 2], "binary_path": str(path)}
                )

            for rows in (2, 1):
                for visit in range(2):
                    for _ in range(rows):
                        record(f"loop.visit.{visit}", [1.0, 2.0])
                record("loop.logits", [99.0, 99.0] if rows == 2 else [1.0, 2.0])
            for visit in range(2):
                np.save(root / f"case0.step0.visit{visit}.npy", [[1.0, 2.0]] * 3)
            np.save(root / "case0.step0.logits.npy", [1.0, 2.0])
            (root / "case0.jsonl").write_text("\n".join(json.dumps(r) for r in records))
            self.assertTrue(compare(root, root)["pass"])
            (root / "case0.jsonl").write_text(
                "\n".join(json.dumps(r) for r in records[:-1])
            )
            with self.assertRaisesRegex(ValueError, "missing final-position logits"):
                compare(root, root)


if __name__ == "__main__":
    unittest.main()
