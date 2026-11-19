import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools.oracle.d3_turbo.compare import compare


class TraceAcceptance(unittest.TestCase):
    def test_lossy_embeddings_use_numerical_tolerance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / "oracle.jsonl", root / "native.jsonl"]
            for path, delta in zip(paths, [0., 1e-6]):
                records = []
                for name in ["embedding", "logits"]:
                    data = path.with_name(path.stem + name + ".f32")
                    np.array([1. + delta, 2.], dtype="<f4").tofile(data)
                    records.append(dict(name="vela3.d3." + name, shape=[2], len=2,
                        binary_path=str(data)))
                path.write_text("\n".join(json.dumps(row) for row in records))
            self.assertTrue(compare(*paths, lossy=True)["passed"])

    def test_incomplete_and_empty_runs_never_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            for text in ["", json.dumps(dict(name="vela3.d3.hidden", shape=[1], len=1))]:
                path.write_text(text)
                with self.assertRaisesRegex(ValueError, "incomplete"):
                    compare(path, path)

    def test_lossy_logits_still_require_the_same_decision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / "oracle.jsonl", root / "native.jsonl"]
            for path, values in zip(paths, [[1., 1.001], [1.001, 1.]]):
                data = path.with_suffix(".f32")
                np.array(values, dtype="<f4").tofile(data)
                path.write_text(json.dumps(dict(name="vela3.d3.logits", shape=[2], len=2,
                    binary_path=str(data))))
            self.assertFalse(compare(*paths, lossy=True)["passed"])
            paths[1].with_suffix(".f32").write_bytes(paths[0].with_suffix(".f32").read_bytes())
            self.assertTrue(compare(*paths, lossy=True)["passed"])


if __name__ == "__main__":
    unittest.main()
