import unittest

import torch

from tools.converter.arktts.convert_arktts import materialize_codec, validate_main


class ConverterTests(unittest.TestCase):
    def test_weight_norm_folding_keeps_non_normalized_parameters(self):
        state = {
            "a.parametrizations.weight.original0": torch.tensor([[[2.0]]]),
            "a.parametrizations.weight.original1": torch.tensor([[[3.0, 4.0]]]),
            "a.bias": torch.tensor([0.25]),
            "b.weight_g": torch.tensor([[[5.0]]]),
            "b.weight_v": torch.tensor([[[0.0, 2.0]]]),
        }
        out = materialize_codec(state)
        self.assertEqual(set(out), {"a.weight", "a.bias", "b.weight"})
        torch.testing.assert_close(out["a.weight"], torch.tensor([[[1.2, 1.6]]]))
        torch.testing.assert_close(out["b.weight"], torch.tensor([[[0.0, 5.0]]]))
        self.assertEqual(out["a.bias"].item(), 0.25)

    def test_wrong_architecture_and_missing_components_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "model_type"):
            validate_main({"model_type": "qwen2"}, {})
        with self.assertRaisesRegex(ValueError, "dim|config"):
            validate_main({"model_type": "arktts", "dim": 1}, {})

    def test_invalid_weight_norm_is_rejected(self):
        with self.assertRaises(ValueError):
            materialize_codec({"a.weight_g": torch.ones(1, 1, 1)})
        with self.assertRaisesRegex(ValueError, "finite"):
            materialize_codec({"a.weight": torch.tensor([float("nan")])})


if __name__ == "__main__":
    unittest.main()
