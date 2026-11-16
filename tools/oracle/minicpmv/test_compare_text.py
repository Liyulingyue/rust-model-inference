"""Contract checks for the text comparator; no model dependencies."""
import json
import struct
import tempfile
import unittest
from pathlib import Path

import compare_text


def write_trace(directory, name, records):
    """Write one side of a comparison and return its path.

    `records` is a list of `(checkpoint, layer, step, values)` tuples.
    """
    path = directory / "trace.jsonl"
    with path.open("w", encoding="utf-8") as out:
        for checkpoint, layer, step, values in records:
            binary = directory / ("%s.%s.%s.f32" % (checkpoint, layer, step))
            binary.write_bytes(struct.pack("<%df" % len(values), *values))
            out.write(
                json.dumps(
                    {
                        "name": checkpoint,
                        "layer": layer,
                        "step": step,
                        "shape": [len(values)],
                        "len": len(values),
                        "occurrence": 0,
                        "binary_path": str(binary),
                    }
                )
                + "\n"
            )
    return path


class ComparatorTests(unittest.TestCase):
    def test_the_final_norm_and_the_logits_share_a_group(self):
        # The Rust side labels them with layer `None`, the oracle with `-1`.
        self.assertEqual(compare_text.step_of({"occurrence": 3}), 3)
        self.assertEqual(compare_text.step_of({"step": 3, "occurrence": 9}), 3)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = write_trace(root, "result_output", [("result_output", None, None, [1.0])])
            groups = compare_text.by_name(path)
            self.assertEqual(list(groups), [("result_output", -1, 0)])

    def test_different_experts_are_not_a_match(self):
        # The comparator has to notice a changed expert id, not just a
        # changed float.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left = write_trace(root, "ffn_moe_topk", [("ffn_moe_topk", 0, 0, [3.0, 7.0])])
            (root / "right").mkdir()
            right = write_trace(
                root / "right", "ffn_moe_topk", [("ffn_moe_topk", 0, 0, [3.0, 9.0])]
            )
            a = compare_text.by_name(left)
            b = compare_text.by_name(right)
            key = next(iter(set(a) & set(b)))
            x, y = compare_text.values(a[key]), compare_text.values(b[key])
            self.assertNotEqual(x, y)

    def test_identical_checkpoints_match(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left = write_trace(root, "ffn_moe_topk", [("ffn_moe_topk", 0, 0, [3.0, 7.0])])
            (root / "right").mkdir()
            right = write_trace(
                root / "right", "ffn_moe_topk", [("ffn_moe_topk", 0, 0, [3.0, 7.0])]
            )
            a = compare_text.by_name(left)
            b = compare_text.by_name(right)
            key = next(iter(set(a) & set(b)))
            self.assertEqual(
                compare_text.values(a[key]), compare_text.values(b[key])
            )


if __name__ == "__main__":
    unittest.main()
