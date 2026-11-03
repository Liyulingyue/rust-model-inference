"""Checks that the comparator fails on numeric, exact-token and shape regressions."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from verify import Trace, compare, validate_trace


def stages():
    return ([("uocr.token_ids", None), ("uocr.pixels", 0), ("uocr.sam_embed", None)]
            + [("uocr.sam_layer", i) for i in range(12)]
            + [("uocr.sam", None), ("uocr.clip_embed", None)]
            + [("uocr.clip_layer", i) for i in range(24)]
            + [("uocr.visual", None), ("uocr.embeddings", None)]
            + [(name, i) for i in range(12) for name in ["uocr.attention", "uocr.layer"]]
            + [("uocr.norm", None), ("uocr.logits", None), ("uocr.generated", None)])


class ComparatorTest(unittest.TestCase):
    def check_pair(self, name, left, right, passed):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            a, b = Trace(root / "a"), Trace(root / "b")
            for checkpoint, layer in stages():
                values = [3.] if checkpoint == "uocr.generated" else [0.]
                selected = checkpoint == name and layer in [None, 0]
                a.add(checkpoint, layer, left if selected else values)
                b.add(checkpoint, layer, right if selected else values)
            if passed:
                compare(a.root / "trace.jsonl", b.root / "trace.jsonl", root / "report.json")
                self.assertTrue(json.loads((root / "report.json").read_text())["passed"])
            else:
                with self.assertRaises(AssertionError):
                    compare(a.root / "trace.jsonl", b.root / "trace.jsonl", root / "report.json")

    def test_float_tolerance_and_exact_structure(self):
        self.check_pair("uocr.layer", [1., -2., .3], [1.000001, -2., .3], True)
        self.check_pair("uocr.layer", [1., -2., .3], [1.1, -2., .3], False)
        self.check_pair("uocr.layer", [0., 0.], [0., 0.], True)
        self.check_pair("uocr.token_ids", [1., 2.], [1., 2.0000002], False)
        self.check_pair("uocr.layer", [1., 2.], [[1., 2.]], False)
        self.check_pair("uocr.layer", [1., 2.], [float("nan"), 2.], False)

    def test_empty_and_truncated_traces_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = root / "trace.jsonl"
            for rows in [[], [dict(name="uocr.token_ids", layer=None)],
                         [dict(name="uocr.token_ids", layer=None), dict(name="uocr.logits", layer=None)]]:
                trace.write_text("\n".join(json.dumps(row) for row in rows))
                with self.assertRaisesRegex(AssertionError, "Incomplete trace"):
                    compare(trace, trace, root / "report.json")

    def test_missing_stage_is_rejected_in_each_scope(self):
        counts = {}
        rows = []
        for name, layer in stages():
            occurrence = counts.get(name, 0)
            rows.append(dict(name=name, layer=layer, shape=[1], occurrence=occurrence))
            counts[name] = occurrence + 1
        for scope in ["all", "vision", "decoder"]:
            validate_trace(rows, scope)
            for missing in ["uocr.pixels", "uocr.sam_layer", "uocr.clip_layer", "uocr.attention", "uocr.layer", "uocr.norm"]:
                broken = rows.copy()
                broken.pop(next(i for i, row in enumerate(rows) if row["name"] == missing))
                with self.assertRaisesRegex(AssertionError, "Incomplete trace"):
                    validate_trace(broken, scope)


if __name__ == "__main__":
    unittest.main()
