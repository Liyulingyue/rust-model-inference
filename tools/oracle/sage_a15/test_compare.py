import copy
import struct
import unittest
from compare import compare, key


class ContractTests(unittest.TestCase):
    def fixture(self):
        names = [("model.input_embed", -1), ("result_norm", -1), ("result_output", -1)]
        names += [("l_out", layer) for layer in range(40)]
        names += [(name, layer) for name in ["ffn_moe_logits", "ffn_moe_probs", "ffn_moe_topk", "ffn_moe_weights_norm"] for layer in [0, 3, 39]]
        return {(name, layer, 0): ({"shape": [2], "len": 2}, struct.pack("<2f", 1, 2)) for name, layer in names}

    def test_lossy_values_use_bounds_and_lossless_values_use_bits(self):
        oracle = self.fixture()
        rust = copy.deepcopy(oracle)
        rust[("result_output", -1, 0)] = ({"shape": [1, 2], "len": 2}, struct.pack("<2f", 1.00001, 2))
        self.assertEqual(len(compare(oracle, rust, 1)), 55)
        rust[("model.input_embed", -1, 0)] = rust[("result_output", -1, 0)]
        with self.assertRaisesRegex(AssertionError, "lossless"):
            compare(oracle, rust, 1)

    def test_missing_checkpoint_or_large_error_fails(self):
        oracle, rust = self.fixture(), self.fixture()
        del rust[("l_out", 39, 0)]
        with self.assertRaisesRegex(AssertionError, "coverage"):
            compare(oracle, rust, 1)
        rust = self.fixture()
        rust[("result_output", -1, 0)] = ({"shape": [2], "len": 2}, struct.pack("<2f", 10, 20))
        with self.assertRaises(AssertionError):
            compare(oracle, rust, 1)

    def test_incremental_boundary_uses_occurrence_and_moe_uses_absolute_step(self):
        self.assertEqual(key({"name": "l_out-39", "layer": 39, "step": None, "occurrence": 4}), ("l_out", 39, 4))
        self.assertEqual(key({"name": "ffn_moe_logits", "layer": 39, "step": 4, "occurrence": 199}), ("ffn_moe_logits", 39, 4))

    def test_identical_nonfinite_float_checkpoints_fail(self):
        for name in ["model.input_embed", "result_output"]:
            oracle = self.fixture()
            oracle[(name, -1, 0)] = ({"shape": [2], "len": 2}, struct.pack("<2f", float("nan"), 2))
            with self.assertRaisesRegex(AssertionError, "non-finite"):
                compare(oracle, copy.deepcopy(oracle), 1)


if __name__ == "__main__":
    unittest.main()
