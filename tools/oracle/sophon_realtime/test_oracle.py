import unittest
import tempfile
from types import SimpleNamespace

import numpy as np

from tools.oracle.sophon_realtime.oracle import metrics, tensor_name, meets_gate
from tools.oracle.sophon_realtime import oracle


class MetricsTests(unittest.TestCase):
    def test_full_prefix_matches_eight_cached_causal_rows(self):
        full_prefix = oracle.full_prefix_logits
        import torch
        from transformers.cache_utils import DynamicCache
        from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

        torch.manual_seed(97)
        config = Qwen3VLTextConfig(
            vocab_size=64, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            head_dim=8, rope_theta=5_000_000,
            rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 2], "mrope_interleaved": True},
        )
        config._attn_implementation = "eager"
        model = Qwen3VLTextModel(config).eval()
        prompt, teacher = [1, 4, 7], [2, 11, 5, 13, 3, 17, 9, 18]
        head = torch.randn(64, 32)
        with torch.no_grad():
            actual, _seconds = full_prefix(model, head, prompt, teacher, 8)
            cache, position, rows = DynamicCache(config=config), 0, []
            for step in range(8):
                current = torch.tensor([prompt if step == 0 else [teacher[step - 1]]])
                positions = torch.arange(position, position + current.shape[1])[None, None].expand(3, 1, -1)
                output = model(input_ids=current, position_ids=positions, past_key_values=cache, use_cache=True)
                rows.append(torch.nn.functional.linear(output.last_hidden_state[0, -1], head).numpy())
                position += current.shape[1]
        self.assertEqual(actual.shape, (8, 64))
        np.testing.assert_allclose(actual, rows, rtol=3e-6, atol=3e-6)

    def test_thinker_runtime_qdq_maps_linears_and_functional_head(self):
        import torch
        from tools.oracle.sophon_realtime.audio_oracle import runtime_activation_roundtrip

        model = torch.nn.Module()
        model.layers = torch.nn.ModuleList([torch.nn.Module()])
        model.layers[0].self_attn = torch.nn.Module()
        projection = torch.nn.Linear(32, 4, bias=False)
        model.layers[0].self_attn.q_proj = projection
        kinds = {"blk.0.attn_q.weight": 1, "output.weight": 8}
        value = torch.linspace(-1.2345, 1.2345, 64).reshape(2, 32)
        baseline = projection(value).detach()
        activation = oracle.thinker_activations(model, kinds)
        try:
            expected = torch.nn.functional.linear(value.half().float(), projection.weight)
            torch.testing.assert_close(projection(value), expected, rtol=0, atol=0)
            np.testing.assert_array_equal(
                activation.apply(value, "output.weight").numpy(),
                runtime_activation_roundtrip(value.numpy(), 8),
            )
            self.assertEqual(
                {row["tensor"] for row in activation.description}, set(kinds)
            )
        finally:
            activation.close()
        torch.testing.assert_close(projection(value), baseline, rtol=0, atol=0)

    def test_reference_weights_borrow_f32_and_map_decoded_weights(self):
        values = np.arange(12, dtype=np.float32).reshape(3, 4)
        def decode(data, kind):
            return data.astype(np.float32, copy=False)

        with tempfile.TemporaryFile() as scratch:
            f32 = SimpleNamespace(data=values, tensor_type=0, shape=[4, 3])
            borrowed = oracle.reference_weight(f32, decode, scratch)
            self.assertTrue(np.shares_memory(values, borrowed))
            self.assertEqual(scratch.tell(), 0)
            half = SimpleNamespace(data=values.astype(np.float16), tensor_type=1, shape=[4, 3])
            mapped = oracle.reference_weight(half, decode, scratch)
            self.assertIsInstance(mapped, np.memmap)
            np.testing.assert_array_equal(mapped, values)
            self.assertEqual(scratch.tell(), values.nbytes)
            half.data.fill(0)
            again = oracle.reference_weight(half, decode, scratch)
            self.assertEqual(scratch.tell(), 2 * values.nbytes)
            np.testing.assert_array_equal(mapped, values)
            np.testing.assert_array_equal(again, np.zeros_like(values))
        # Mappings keep the anonymous scratch file alive after its writer closes.
        np.testing.assert_array_equal(mapped, values)

    def test_precision_gates_reject_bad_or_missing_metrics(self):
        for precision in ["f32", "f16", "bf16", "q8_0", "q4_k"]:
            self.assertTrue(meets_gate(metrics([1, 2], [1, 2]), precision))
            self.assertFalse(meets_gate(metrics([2, 4], [1, 2]), precision))
            self.assertFalse(meets_gate(metrics([1, 0], [0, 0]), precision))
        self.assertFalse(meets_gate({"relative_rmse": 0.002, "cosine": 0.9999999}, "f32"))
        self.assertTrue(meets_gate({"relative_rmse": 0.002, "cosine": 0.9999999}, "f16"))
        with self.assertRaises(ValueError):
            meets_gate(metrics([1], [1]), "unknown")

    def test_decoder_names_and_unknown_components(self):
        self.assertEqual(tensor_name("blk.63.attn_q.weight"), "layers.63.self_attn.q_proj.weight")
        self.assertEqual(tensor_name("output.weight"), "lm_head.weight")
        for name in ["audio_tower.conv.weight", "blk.0.moe.weight", "blk.x.attn_q.weight"]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                tensor_name(name)

    def test_identical_logits_and_scaled_logits(self):
        identical = metrics([1, 2, 3], [1, 2, 3])
        self.assertEqual(identical["max_abs"], 0)
        self.assertAlmostEqual(identical["cosine"], 1)
        scaled = metrics([2, 4, 6], [1, 2, 3])
        self.assertEqual(scaled["max_abs"], 3)
        self.assertAlmostEqual(scaled["relative_rmse"], 1)
        self.assertAlmostEqual(scaled["cosine"], 1)

    def test_nonfinite_or_mismatched_outputs_cannot_pass(self):
        for actual, expected in [([1], [1, 2]), ([float("nan")], [1]), ([1], [float("inf")]), ([], [])]:
            with self.subTest(actual=actual, expected=expected), self.assertRaises(ValueError):
                metrics(actual, expected)

    def test_zero_reference_has_defined_error(self):
        same = metrics([0, 0], [0, 0])
        self.assertEqual(same["relative_rmse"], 0)
        self.assertEqual(same["cosine"], 1)
        different = metrics([1, 0], [0, 0])
        self.assertIsNone(different["relative_rmse"])
        self.assertEqual(different["cosine"], 0)


if __name__ == "__main__":
    unittest.main()
