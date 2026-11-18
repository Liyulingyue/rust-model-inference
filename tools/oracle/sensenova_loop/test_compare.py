import unittest
import json
import tempfile
from pathlib import Path
import numpy as np
from tools.oracle.sensenova_loop.compare import metrics, compare


class ComparisonTests(unittest.TestCase):
    def test_lossy_gates_accept_small_error_and_reject_wrong_direction(self):
        self.assertFalse(metrics([1.01, 2.0], [1.0, 2.0])["pass"])
        self.assertTrue(metrics([1.01, 2.0], [1.0, 2.0], lossy=True)["pass"])
        self.assertFalse(metrics([-1.0, -2.0], [1.0, 2.0], lossy=True)["pass"])

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

    def test_greedy_divergence_marks_later_inputs_without_hiding_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = {"generated": [1, 2], "steps": [
                {"rows": 1, "visits": [[1, 2]]},
                {"rows": 1, "visits": [[1, 2]]},
            ]}
            (root / "fixtures.json").write_text(json.dumps([fixture]))
            (root / "results.json").write_text(json.dumps([{"generated": [9, 2]}]))
            records = []
            for step in range(2):
                for name, suffix in [("loop.visit.0", "visit0"), ("loop.logits", "logits")]:
                    path = root / f"{len(records)}.f32"
                    np.asarray([1.0 + step, 2.0], dtype="<f4").tofile(path)
                    records.append({"name": name, "shape": [1, 2], "binary_path": str(path)})
                    values = [[1.0, 2.0]] if suffix == "visit0" else [1.0, 2.0]
                    np.save(root / f"case0.step{step}.{suffix}.npy", values)
            (root / "case0.jsonl").write_text("\n".join(json.dumps(r) for r in records))
            result = compare(root, root)
            self.assertFalse(result["pass"])
            self.assertFalse(result["generated_exact"])
            self.assertEqual([r["same_input_prefix"] for r in result["checks"]], [True, True, False, False])
            self.assertEqual(sum(not r["pass"] for r in result["checks"]), 2)
            self.assertEqual(result["same_input_prefix_checkpoints"], 2)
            self.assertEqual(result["same_input_prefix_failed"], 0)


if __name__ == "__main__":
    unittest.main()
