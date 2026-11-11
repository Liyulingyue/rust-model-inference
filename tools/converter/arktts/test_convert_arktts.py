import unittest

import torch

from tools.converter.arktts.convert_arktts import materialize_codec, validate_main, encode_tensor


class ConverterTests(unittest.TestCase):
    def test_precision_contract_preserves_vectors_convolutions_and_codebooks(self):
        import numpy as np
        for precision in ("f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "q4_k", "q5_k", "q6_k"):
            raw = np.ones((2, 256), dtype="<f4").tobytes()
            kind, payload = encode_tensor("proj.weight", raw, 0, (256, 2), precision)
            self.assertNotEqual(kind, 0, precision) if precision != "f32" else self.assertEqual(kind, 0)
            self.assertTrue(payload)
            for name, dims in [("norm.weight", (512,)), ("conv.weight", (2, 256, 1)),
                               ("q.codebook.weight", (8, 64)),
                               ("quantizer.upsample.0.1.pwconv1.weight", (256, 2))]:
                self.assertEqual(encode_tensor(name, raw, 0, dims, precision), (0, raw))
        # Main width 896 cannot hold a K block; down-projection width 4864 can.
        raw = np.ones((2, 896), dtype="<f4").tobytes()
        self.assertEqual(encode_tensor("proj.weight", raw, 0, (896, 2), "q4_k"), (0, raw))

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
