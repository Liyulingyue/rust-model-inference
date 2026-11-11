import unittest
import tempfile
from pathlib import Path

import numpy as np

from .audio_oracle import ActivationQuantization, checkpoint_values, compare_projection_checkpoints, runtime_activation_roundtrip


class ActivationTests(unittest.TestCase):
    def test_f16_prepared_inputs_round_but_f32_and_bf16_stay_f32(self):
        values = np.array([[1.0001, -0.3333, 0.00001]], dtype=np.float32)
        expected = values.astype(np.float16).astype(np.float32)
        np.testing.assert_array_equal(runtime_activation_roundtrip(values, 1), expected)
        for kind in (0, 30):
            np.testing.assert_array_equal(runtime_activation_roundtrip(values, kind), values)

    def test_q8_and_q4_1_inputs_use_32_wide_f16_scale_blocks(self):
        values = np.zeros((3, 32), dtype=np.float32)
        values[0, :5] = [127, 0.5, 1.5, -0.5, -1.5]
        values[2, :2] = [1.0, 0.01]
        expected = values.copy()
        expected[0, :5] = [127, 0, 2, 0, -2]
        expected[2, :2] = [0.99993896484375, 0.00787353515625]
        for kind in (2, 3, 8):
            np.testing.assert_array_equal(runtime_activation_roundtrip(values, kind), expected)

    def test_k_inputs_use_256_wide_signed_scale_and_handle_zero_blocks(self):
        values = np.zeros((2, 256), dtype=np.float32)
        values[0, :3] = [2, -2, 0.5]
        for kind in (12, 13, 14):
            actual = runtime_activation_roundtrip(values, kind)
            self.assertAlmostEqual(float(actual[0, 2]), 64 / 127, places=7)
            np.testing.assert_array_equal(actual[1], np.zeros(256, dtype=np.float32))

    def test_hooks_quantize_native_linear_and_functional_projector_inputs(self):
        import torch
        encoder = torch.nn.Sequential(torch.nn.Linear(32, 1, bias=False))
        with torch.no_grad():
            encoder[0].weight.zero_()
            encoder[0].weight[0, 1] = 1
        values = torch.zeros(1, 32)
        values[0, :2] = torch.tensor([127, 0.5])
        kinds = {"audio_tower.0.weight": 8, "multi_modal_projector.linear_1.weight": 1}
        activation = ActivationQuantization(encoder, kinds)
        try:
            self.assertEqual(encoder(values).item(), 0)
            rounded = activation.apply(torch.tensor([[1.0001]]), "multi_modal_projector.linear_1.weight")
            self.assertEqual(rounded.item(), 1)
            self.assertEqual(len(activation.description), 2)
        finally:
            activation.close()
        self.assertEqual(encoder(values).item(), 0.5)

    def test_invalid_or_overflowing_activation_inputs_are_rejected(self):
        for value, kind in [(np.array([np.nan]), 0), (np.ones(31), 8), (np.ones(255), 12),
                            (np.array([65536.0]), 1), (np.ones(32), 9)]:
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                runtime_activation_roundtrip(value, kind)


class CheckpointTests(unittest.TestCase):
    def test_projection_isolation_uses_exact_rust_input_and_declared_activation(self):
        import torch
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            x = np.zeros((1, 32), dtype=np.float32)
            x[0, :2] = [127, 0.5]
            x.tofile(base / "input.f32")
            np.array([[0]], dtype=np.float32).tofile(base / "output.f32")
            trace = [{"name": f"sophon.audio.linear.audio_tower.test.{stage}",
                      "shape": shape, "binary_path": str(base / f"{stage}.f32")}
                     for stage, shape in [("input", [1, 32]), ("output", [1, 1])]]
            weight = torch.zeros(1, 32)
            weight[0, 1] = 1
            tensors = {"audio_tower.test.weight": weight}
            kinds = {"audio_tower.test.weight": 8}
            result = compare_projection_checkpoints(trace, tensors, kinds, "runtime")
            self.assertEqual(len(result), 1)
            self.assertTrue(next(iter(result.values()))["pass"])
            f32 = compare_projection_checkpoints(trace, tensors, kinds, "f32")
            self.assertFalse(next(iter(f32.values()))["pass"])
            with self.assertRaises(ValueError):
                compare_projection_checkpoints(trace[:1], tensors, kinds, "runtime")

    def test_multichunk_checkpoints_drop_only_per_chunk_padding(self):
        mel = np.arange(2 * 2 * 8).reshape(2, 2, 8)
        encoder = np.arange(2 * 2 * 3).reshape(2, 2, 3)
        projected = np.arange(6).reshape(2, 3)
        actual = checkpoint_values(mel, encoder, projected, [8, 1], [2, 1])
        np.testing.assert_array_equal(actual["mel"], np.r_[mel[0].ravel(), mel[1, :, :1].ravel()])
        np.testing.assert_array_equal(actual["encoder"], np.r_[encoder[0].ravel(), encoder[1, :1].ravel()])
        np.testing.assert_array_equal(actual["projected"], projected.ravel())

    def test_lengths_must_match_the_padded_batch(self):
        for mel_lengths, encoder_lengths in [([1], [1, 1]), ([1, 9], [1, 1]), ([1, 1], [1, 3]), ([0, 1], [1, 1])]:
            with self.subTest(mel_lengths=mel_lengths, encoder_lengths=encoder_lengths), self.assertRaises(ValueError):
                checkpoint_values(np.ones((2, 2, 8)), np.ones((2, 2, 3)), np.ones((2, 3)), mel_lengths, encoder_lengths)


if __name__ == "__main__":
    unittest.main()
