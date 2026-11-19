"""Small input/loader/math-boundary checks for the independent code2wav Oracle."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from tools.oracle.sophon_realtime import code2wav_oracle as oracle

ROOT = Path(__file__).resolve().parents[3]
REFERENCE = ROOT / "target/sophon/reference/sophon_expressive"
ROTARY = ROOT / "target/sophon/tasks/x-transformers-2.31.7-reference.py"


class Code2wavOracleTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def reference(self):
        if not REFERENCE.is_dir() or not ROTARY.is_file():
            self.skipTest("fetch pinned official sources before reference checks")
        return oracle.load_reference(REFERENCE, ROTARY)

    def test_fixture_writes_exact_channel_and_sample_layouts(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            inputs = oracle.write_fixture(output, 17)
            stored = json.loads((output / "input.json").read_text())
            for key in [
                "tokens",
                "prompt_tokens",
                "prompt_mel",
                "speaker",
                "initial_noise",
                "mel",
                "excitation_noise",
            ]:
                self.assertIn(key, stored)
            self.assertEqual(len(inputs["tokens"]), 4)
            self.assertEqual(len(inputs["prompt_tokens"]), 2)
            self.assertEqual(np.shape(inputs["prompt_mel"]), (80, 4))
            self.assertEqual(np.shape(inputs["initial_noise"]), (80, 12))
            self.assertEqual(np.shape(inputs["mel"]), (80, 8))
            self.assertEqual(np.shape(inputs["excitation_noise"]), (8 * 480, 9))
            for key in ["initial_noise", "excitation_noise"]:
                raw = np.fromfile(output / f"{key}.f32", dtype="<f4")
                np.testing.assert_array_equal(
                    raw, np.asarray(stored[key], dtype=np.float32)
                )

    def test_component_gates_reject_scale_errors_and_handle_zero_reference(self):
        for name in [
            "flow.mu",
            "flow.velocity.0",
            "flow.mel",
            "hift.f0",
            "hift.source",
            "hift.spectrum",
            "hift.waveform",
        ]:
            self.assertTrue(oracle.comparison(name, np.ones(4), np.ones(4))["pass"])
            self.assertFalse(
                oracle.comparison(name, np.ones(4) * 1.1, np.ones(4))["pass"]
            )
            self.assertTrue(oracle.comparison(name, np.zeros(4), np.zeros(4))["pass"])
            self.assertFalse(oracle.comparison(name, np.ones(4), np.zeros(4))["pass"])

    def test_nonfinal_fixture_accounts_for_flow_lookahead_and_f0_crop(self):
        with tempfile.TemporaryDirectory() as directory:
            inputs = oracle.write_fixture(
                Path(directory), 0, token_count=8, mel_frames=12, finalize=False
            )
            self.assertFalse(inputs["finalize"])
            self.assertEqual(inputs["initial_noise"].shape, (80, 14))
            self.assertEqual(inputs["excitation_noise"].shape, (9 * 480, 9))
            self.assertEqual(inputs["mel"].shape, (80, 12))

    def test_longer_fixture_crosses_fifty_frame_flow_chunk(self):
        with tempfile.TemporaryDirectory() as directory:
            inputs = oracle.write_fixture(
                Path(directory), 0, token_count=30, mel_frames=20
            )
            self.assertEqual(inputs["initial_noise"].shape, (80, 64))
            self.assertEqual(inputs["excitation_noise"].shape, (20 * 480, 9))

    def test_runtime_activation_hooks_round_linears_and_close(self):
        layer = torch.nn.Sequential(torch.nn.Linear(32, 1, bias=False))
        layer[0].weight.data.fill_(1.0)
        values = torch.linspace(-0.9731, 1.1337, 32)[None]
        dense = layer(values).detach().clone()
        expected = values.half().float().sum()
        activation = oracle.RuntimeActivations(layer, {"0.weight": 1})
        with torch.no_grad():
            actual = layer(values)
        torch.testing.assert_close(actual[0, 0], expected, rtol=0, atol=1e-6)
        self.assertEqual(activation.description[0]["weight_type"], 1)
        activation.close()
        torch.testing.assert_close(layer(values), dense)

    def test_rotary_uses_adjacent_pairs_and_leaves_unrotated_tail(self):
        reference = self.reference()
        rotary = reference["RotaryEmbedding"](4)
        frequencies, scale = rotary.forward_from_seq_len(2)
        x = torch.tensor(
            [[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]]
        )
        actual = reference["apply_rotary_pos_emb"](x, frequencies, scale)
        angle = torch.tensor([1.0, 1.0, 0.01, 0.01])
        expected = (
            torch.tensor([1.0, 2.0, 3.0, 4.0]) * angle.cos()
            + torch.tensor([-2.0, 1.0, -4.0, 3.0]) * angle.sin()
        )
        torch.testing.assert_close(actual[0, 1, :4], expected)
        torch.testing.assert_close(actual[..., 4:], x[..., 4:])

    def test_fused_weight_norm_load_uses_materialized_weight_exactly(self):
        reference = self.reference()
        layer = torch.nn.utils.parametrizations.weight_norm(
            reference["convolution"].CausalConv1d(2, 3, 3, causal_type="left")
        )
        expected = torch.arange(18, dtype=torch.float32).reshape(3, 2, 3) / 10
        oracle.remove_weight_norm(layer)
        layer.load_state_dict({"weight": expected, "bias": torch.zeros(3)}, strict=True)
        values = torch.arange(8, dtype=torch.float32).reshape(1, 2, 4)
        actual = layer(values)
        wanted = torch.nn.functional.conv1d(
            torch.nn.functional.pad(values, (2, 0)), expected
        )
        torch.testing.assert_close(actual, wanted, rtol=0, atol=0)

    def test_f32_hift_excitation_preserves_explicit_noise_without_f64_inference(self):
        reference = self.reference()
        # Construct only the official sine/source path, avoiding 300-second random buffers.
        with torch.device("meta"):
            source = reference["generator"].SourceModuleHnNSF(
                24000,
                480,
                harmonic_num=8,
                voiced_threshod=10,
                sinegen_type="2",
                causal=True,
            )
        source.to_empty(device="cpu")
        source.l_linear.weight.data.fill_(1.0)
        source.l_linear.bias.data.zero_()
        f0 = torch.tensor([[0.0, 120.0]])
        noise = torch.full((1, 960, 9), 0.25)
        oracle.set_excitation(source, noise)
        source.eval()
        with torch.no_grad():
            sines, _, _ = source.l_sin_gen(f0[:, :, None].repeat_interleave(480, 1))
        self.assertEqual(sines.dtype, torch.float32)
        np.testing.assert_allclose(
            sines[0, :480].numpy(), np.full((480, 9), 0.25 / 30), rtol=1e-6
        )
        phase = np.arange(1, 10, dtype=np.float64) * (120 / 24000) * (2 * np.pi) * 480
        expected = np.sin(phase) * 0.1 + 0.25 * 0.003
        # The analytical reference uses F64; the official F32 phase rounds near 0.
        np.testing.assert_allclose(
            sines[0, 480].numpy(), expected, rtol=1e-4, atol=1e-6
        )


if __name__ == "__main__":
    unittest.main()
