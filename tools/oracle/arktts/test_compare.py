import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools.oracle.arktts.compare import compare


class ComparisonTests(unittest.TestCase):
    def check(self, expected, actual):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = []
            for name, values in [("a", expected), ("b", actual)]:
                path = root / f"{name}.f32"
                np.array(values, dtype="<f4").tofile(path)
                rows.append({"name": "layers", "layer": 0, "step": 0,
                             "shape": [len(values)], "binary_path": str(path)})
            (root / "oracle.json").write_text(json.dumps(rows[:1]))
            (root / "rust.jsonl").write_text(json.dumps(rows[1]))
            with contextlib.redirect_stdout(io.StringIO()):
                return compare(root / "oracle.json", root / "rust.jsonl")

    def test_large_activation_uses_scale_aware_error(self):
        result = self.check([10000.0, 2.0], [10000.01, 2.0])
        self.assertLess(result[0]["relative_rmse"], 2e-6)

    def test_substantial_numerical_error_is_rejected(self):
        with self.assertRaisesRegex(AssertionError, "out-of-tolerance"):
            self.check([1.0, 2.0], [1.1, 2.1])
        with self.assertRaisesRegex(AssertionError, "invalid checkpoint"):
            self.check([1.0], [float("nan")])

    def test_precision_profiles_are_explicit_and_unknown_profiles_are_rejected(self):
        from tools.oracle.arktts.compare import TOLERANCES
        self.assertEqual(set(TOLERANCES), {"float", "quantized", "f16"})
        with self.assertRaisesRegex(ValueError, "profile"):
            compare("unused", "unused", profile="typo")


if __name__ == "__main__":
    unittest.main()
