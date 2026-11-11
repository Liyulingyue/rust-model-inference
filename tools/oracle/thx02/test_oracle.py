"""Independent GGUF decoding and THX-02 inverse-layout checks with tiny weights."""

from __future__ import annotations

import importlib
import json
import math
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from tools.converter.thx02.convert_thx02 import TensorSpec, add_tensor
from tools.converter.thx02.test_convert_thx02 import safetensors
from tools.converter.utils.gguf import (
    GGML_BF16, GGML_F16, GGML_F32, GGML_Q4_0, GGML_Q4K, GGML_Q5K,
    GGML_Q6K, GGML_Q8_0, GgufWriter, bf16_to_f32, f32_to_bf16,
)

try:
    oracle = importlib.import_module("tools.oracle.thx02.oracle")
except ModuleNotFoundError:
    oracle = None


def scale_table(scales, minima):
    table = np.zeros(12, dtype=np.uint8)
    table[:4] = np.array(scales[:4]) & 63
    table[4:8] = np.array(minima[:4]) & 63
    for i in range(4):
        table[i] |= (scales[4 + i] >> 4) << 6
        table[4 + i] |= (minima[4 + i] >> 4) << 6
        table[8 + i] = (scales[4 + i] & 15) | ((minima[4 + i] & 15) << 4)
    return table.tobytes()


class OracleExists(unittest.TestCase):
    def test_oracle_is_implemented(self):
        self.assertIsNotNone(oracle, "THX-02 Oracle is not implemented")


@unittest.skipIf(oracle is None, "Oracle not implemented yet")
class OracleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_float_payloads_decode_to_f32(self):
        values = np.array([-2, -0.125, 0, 0.25, 3], dtype="<f4")
        for kind, raw in (
            (GGML_F32, values.tobytes()), (GGML_F16, values.astype("<f2").tobytes()),
            (GGML_BF16, f32_to_bf16(values.tobytes())),
        ):
            np.testing.assert_array_equal(oracle.dequantize(raw, kind, len(values)), values)

    def test_q8_0_layout(self):
        q = np.arange(-16, 16, dtype=np.int8)
        raw = struct.pack("<e", 0.5) + q.tobytes()
        np.testing.assert_array_equal(oracle.dequantize(raw, GGML_Q8_0, 32), q * 0.5)

    def test_q4_0_low_and_high_halves(self):
        codes = np.arange(32, dtype=np.uint8) % 16
        raw = struct.pack("<e", -0.5) + ((codes[:16] & 15) | (codes[16:] << 4)).tobytes()
        expected = (codes.astype(np.int16) - 8) * -0.5
        np.testing.assert_array_equal(oracle.dequantize(raw, GGML_Q4_0, 32), expected)

    def test_q4_k_six_bit_scales_minima_and_nibble_groups(self):
        scales, minima = [1, 2, 3, 4, 17, 34, 51, 63], [5, 6, 7, 8, 31, 47, 62, 63]
        codes = np.arange(256, dtype=np.uint8).reshape(4, 2, 32) % 16
        packed = (codes[:, 0] | (codes[:, 1] << 4)).reshape(-1)
        raw = struct.pack("<ee", 0.5, 0.25) + scale_table(scales, minima) + packed.tobytes()
        expected = 0.5 * np.repeat(scales, 32) * codes.reshape(-1) - 0.25 * np.repeat(minima, 32)
        np.testing.assert_array_equal(oracle.dequantize(raw, GGML_Q4K, 256), expected)

    def test_q5_k_fifth_bits_follow_each_32_value_subblock(self):
        scales, minima = [1, 2, 3, 4, 17, 34, 51, 63], [5, 6, 7, 8, 31, 47, 62, 63]
        codes = (np.arange(256, dtype=np.uint16) * 7 % 32).astype(np.uint8).reshape(8, 32)
        qh = np.zeros(32, dtype=np.uint8)
        for sub in range(8):
            qh |= ((codes[sub] >> 4) & 1) << sub
        paired = codes.reshape(4, 2, 32)
        qs = (paired[:, 0] & 15) | ((paired[:, 1] & 15) << 4)
        raw = struct.pack("<ee", 0.5, 0.25) + scale_table(scales, minima) + qh.tobytes() + qs.tobytes()
        expected = 0.5 * np.repeat(scales, 32) * codes.reshape(-1) - 0.25 * np.repeat(minima, 32)
        np.testing.assert_array_equal(oracle.dequantize(raw, GGML_Q5K, 256), expected)

    def test_q6_k_high_planes_and_signed_subscales(self):
        codes = (np.arange(256, dtype=np.uint16) * 11 % 64).astype(np.uint8).reshape(2, 4, 32)
        ql, qh = np.zeros((2, 64), dtype=np.uint8), np.zeros((2, 32), dtype=np.uint8)
        ql[:, :32] = (codes[:, 0] & 15) | ((codes[:, 2] & 15) << 4)
        ql[:, 32:] = (codes[:, 1] & 15) | ((codes[:, 3] & 15) << 4)
        for plane in range(4):
            qh |= ((codes[:, plane] >> 4) & 3) << (2 * plane)
        scales = np.arange(-8, 8, dtype=np.int8)
        raw = ql.tobytes() + qh.tobytes() + scales.tobytes() + struct.pack("<e", 0.125)
        expected = 0.125 * np.repeat(scales, 16) * (codes.reshape(-1).astype(np.int16) - 32)
        np.testing.assert_array_equal(oracle.dequantize(raw, GGML_Q6K, 256), expected)

    def test_invalid_payloads_types_and_nonfinite_values_are_rejected(self):
        for kind, length, count in ((GGML_F32, 4, 1), (GGML_F16, 2, 1),
                                    (GGML_BF16, 2, 1), (GGML_Q8_0, 34, 32),
                                    (GGML_Q4_0, 18, 32), (GGML_Q4K, 144, 256),
                                    (GGML_Q5K, 176, 256), (GGML_Q6K, 210, 256)):
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                oracle.dequantize(bytes(length - 1), kind, count)
        with self.assertRaises(ValueError):
            oracle.dequantize(bytes(20), 3, 32)
        with self.assertRaises(ValueError):
            oracle.dequantize(struct.pack("<f", float("nan")), GGML_F32, 1)

    def restore(self, source, values, target="f32", kind="matrix", transform="none", frame=False):
        source_file = safetensors(self.root / "source.safetensors", {source: (values, "BF16")})
        writer = GgufWriter(self.root / "tiny.gguf")
        names = oracle.source_mapping(source)
        if frame:
            for index, name in enumerate(names):
                add_tensor(writer, source_file, TensorSpec(source, name, values.shape, "patch", "none", True, index), target)
        else:
            add_tensor(writer, source_file, TensorSpec(source, names[0], values.shape, kind, transform), target)
        writer.write()
        reader = oracle.GgufWeights(writer.path)
        chunks = list(oracle.restored_chunks(reader, source, values.shape, rows_per_chunk=3))
        return np.concatenate([value for _, _, value in chunks], axis=0)

    def test_inverse_norm_a_log_conv_shape_and_readout(self):
        for source, values, kind, transform, tolerance in (
            ("language_model.norm.weight", np.array([-0.125, 0, 0.25], dtype="<f4"), "float", "norm", 0),
            ("language_model.layers.0.linear_attn.A_log", np.arange(32, dtype="<f4") / 32, "float", "a_log", 1e-6),
            ("language_model.layers.0.linear_attn.conv1d.weight", np.arange(8192 * 4, dtype="<f4").reshape(8192, 1, 4) / 128, "float", "none", 0),
            ("readout.weight", np.arange(512, dtype="<f4").reshape(2, 256) / 128, "matrix", "none", 0),
        ):
            with self.subTest(source=source):
                actual = self.restore(source, values, kind=kind, transform=transform)
                expected = np.frombuffer(bf16_to_f32(f32_to_bf16(values.tobytes())), dtype="<f4").reshape(values.shape)
                np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=0)

    def test_inverse_all_recurrent_value_head_axes(self):
        for suffix, shape in (
            ("in_proj_qkv.weight", (8192, 1)), ("in_proj_z.weight", (4096, 1)),
            ("in_proj_a.weight", (32, 1)), ("in_proj_b.weight", (32, 1)),
            ("dt_bias", (32,)), ("out_proj.weight", (2, 4096)),
        ):
            with self.subTest(suffix=suffix):
                source = "language_model.layers.0.linear_attn." + suffix
                values = np.arange(math.prod(shape), dtype="<f4").reshape(shape) / 128
                actual = self.restore(source, values)
                expected = np.frombuffer(bf16_to_f32(f32_to_bf16(values.tobytes())), dtype="<f4").reshape(shape)
                np.testing.assert_array_equal(actual, expected)

    def test_temporal_conv3d_frames_rejoin_in_original_axis_order(self):
        values = np.arange(48, dtype="<f4").reshape(2, 3, 2, 2, 2)
        actual = self.restore("visual.patch_embed.proj.weight", values, frame=True)
        np.testing.assert_array_equal(actual, values)

    def test_all_exported_formats_restore_an_independent_projection(self):
        values = np.linspace(-1, 1, 512, dtype="<f4").reshape(2, 256)
        bounds = {"f32": 0.004, "f16": 0.004, "bf16": 0.004, "q8_0": 0.01,
                  "q4_0": 0.16, "q4_k": 0.12, "q5_k": 0.06, "q6_k": 0.04}
        for target, bound in bounds.items():
            with self.subTest(target=target):
                actual = self.restore("language_model.layers.3.mlp.gate_proj.weight", values, target)
                self.assertLessEqual(float(np.max(np.abs(actual - values))), bound)

    def test_unmapped_or_missing_tensors_fail_explicitly(self):
        with self.assertRaises(ValueError):
            oracle.source_mapping("other_model.weight")
        writer = GgufWriter(self.root / "missing.gguf")
        writer.add_tensor("other.weight", GGML_F32, (2,), bytes(8))
        writer.write()
        reader = oracle.GgufWeights(writer.path)
        with self.assertRaises(ValueError):
            list(oracle.restored_chunks(reader, "language_model.norm.weight", (2,)))

    def test_request_array_and_single_request_keep_order(self):
        requests = [{"state": "a", "questions": {"z": {"type": "noul"}, "a": {"type": "noul"}}}]
        path = self.root / "requests.json"
        path.write_text(json.dumps(requests))
        loaded, single = oracle.read_requests(path)
        self.assertFalse(single)
        self.assertEqual(list(loaded[0]["questions"]), ["z", "a"])
        path.write_text(json.dumps(requests[0]))
        loaded, single = oracle.read_requests(path)
        self.assertTrue(single)
        self.assertEqual(loaded, requests)

    def test_direct_module_loading_skips_unrelated_sdk_initializer(self):
        package = self.root / "thx02"
        package.mkdir()
        (package / "__init__.py").write_text("raise RuntimeError('SDK client is not present')")
        (package / "protocol.py").write_text("VALUE = 42")
        (package / "model.py").write_text("from .protocol import VALUE")
        (package / "runtime.py").write_text("from .model import VALUE")
        protocol, model, runtime = oracle.load_reference_modules(self.root)
        self.assertEqual((protocol.VALUE, model.VALUE, runtime.VALUE), (42, 42, 42))
        self.assertEqual(Path(model.__file__).resolve(), (package / "model.py").resolve())

    def test_positions_capture_actual_rotary_positional_arguments(self):
        import torch
        trace = object.__new__(oracle.Trace)
        trace.active = {}
        positions = torch.arange(4).reshape(1, 1, 4).expand(3, 1, 4)
        trace.positions(None, (torch.zeros(1, 4, 2), positions), {})
        self.assertEqual(trace.active["position_ids"], positions.tolist())

    def test_split_record_keeps_only_selected_chunks_and_retains_rejected_candidate(self):
        first, second = object(), object()
        request = {"state": "Alice is happy.", "questions": {key: {} for key in ("q0", "q1", "q2")}}
        rejected = {"input_ids": [0] * 143, "question_ids": ["q0", "q1", "q2"]}
        selected = [{"input_ids": [0] * 91, "question_ids": ["q0"]},
                    {"input_ids": [0] * 117, "question_ids": ["q1", "q2"]}]
        trace = SimpleNamespace(encodings=[rejected, *selected],
                                by_identity={id(first): selected[0], id(second): selected[1]})
        row = oracle.planned_record(request, [first, second], trace, 0)
        self.assertEqual([len(chunk["input_ids"]) for chunk in row["chunks"]], [91, 117])
        self.assertEqual([chunk["question_ids"] for chunk in row["chunks"]], [["q0"], ["q1", "q2"]])
        self.assertEqual(row["planning_encodings"], [rejected, *selected])
        self.assertEqual(row["request"], request)

    def test_performance_cli_defaults_to_four_threads_and_can_skip_runtime(self):
        args = oracle.argument_parser().parse_args([
            "--model-dir", str(self.root), "--request", "requests.json",
            "--output", "output.json", "--skip-runtime"])
        self.assertEqual(args.threads, 4)
        self.assertTrue(args.skip_runtime)

    def test_skip_runtime_returns_without_an_extra_forward(self):
        class UnavailableRuntime:
            def decide(self, **kwargs):
                raise AssertionError("runtime must not be called")
        self.assertEqual(oracle.run_runtime(UnavailableRuntime(), {}, skip=True), (None, 0.0))

    def test_context_limit_defaults_to_backbone_limit_and_rejects_invalid_limits(self):
        self.assertEqual(oracle.context_limit(262144, None), 262144)
        self.assertEqual(oracle.context_limit(262144, 117), 117)
        for invalid in (0, -1, 262145):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                oracle.context_limit(262144, invalid)

    def test_runtime_boundary_cli_can_limit_joint_and_runtime_planning(self):
        args = oracle.argument_parser().parse_args([
            "--model-dir", str(self.root), "--request", "requests.json",
            "--output", "output.json", "--max-context", "117"])
        self.assertEqual(args.max_context, 117)

    def test_thinker_cli_selects_same_weight_causal_oracle(self):
        args = oracle.argument_parser().parse_args([
            "--model-dir", str(self.root), "--request", "requests.json",
            "--output", "output.json", "--thinker-gguf", "thinker.gguf", "--steps", "8"])
        self.assertEqual(args.thinker_gguf, Path("thinker.gguf"))
        self.assertEqual(args.steps, 8)

    def test_q8_0_activation_roundtrip_uses_fp16_scale_and_even_ties_per_32(self):
        values = np.zeros((2, 64), dtype=np.float32)
        values[0, :5] = [127, 0.5, 1.5, -0.5, -1.5]
        values[0, 32:37] = [-254, 1, 3, -1, -3]
        values[1, 0] = 1.01
        actual = oracle.activation_roundtrip(values, GGML_Q8_0)
        np.testing.assert_array_equal(actual[0, :5], [127, 0, 2, 0, -2])
        np.testing.assert_array_equal(actual[0, 32:37], [-254, 0, 4, 0, -4])
        self.assertEqual(actual[1, 0], np.float32(np.float16(np.float32(1.01) / np.float32(127))) * 127)
        self.assertFalse(np.shares_memory(actual, values))
        np.testing.assert_array_equal(oracle.activation_roundtrip(values, GGML_Q4_0), actual)

    def test_q8_k_activations_keep_signed_scale_even_ties_and_256_boundaries(self):
        values = np.zeros((2, 512), dtype=np.float32)
        values[0, :5] = [-127, 0.5, 1.5, -0.5, -1.5]
        values[0, 256:261] = [254, 1, 3, -1, -3]
        expected = np.zeros_like(values)
        expected[0, :5] = [-127, 0, 2, 0, -2]
        expected[0, 256:261] = [254, 0, 4, 0, -4]
        for kind in (GGML_Q4K, GGML_Q5K, GGML_Q6K):
            np.testing.assert_array_equal(oracle.activation_roundtrip(values, kind), expected)
        with self.assertRaises(ValueError):
            oracle.activation_roundtrip(values[:, :255], GGML_Q4K)
        with self.assertRaises(ValueError):
            oracle.activation_roundtrip(np.array([[float("nan")] * 32]), GGML_Q8_0)

    def test_activation_hooks_select_quantized_linear_weights_and_remove_cleanly(self):
        import torch
        modules = torch.nn.Module()
        modules.embed_tokens = torch.nn.Embedding(128, 32)
        modules.layers = torch.nn.ModuleList([torch.nn.Module()])
        modules.layers[0].mlp = torch.nn.Module()
        modules.layers[0].mlp.gate_proj = torch.nn.Linear(32, 2, bias=False)
        modules.layers[0].mlp.up_proj = torch.nn.Linear(32, 2, bias=False)
        backbone = torch.nn.Module()
        backbone.language_model = modules
        with torch.no_grad():
            modules.layers[0].mlp.gate_proj.weight.fill_(1)
            modules.layers[0].mlp.up_proj.weight.fill_(1)
        weights = SimpleNamespace(tensors={
            "token_embd.weight": (GGML_Q8_0,), "blk.0.ffn_gate.weight": (GGML_Q8_0,),
            "blk.0.ffn_up.weight": (GGML_F32,)})
        values = torch.zeros(1, 32)
        values[0, :2] = torch.tensor([127, 0.5])
        hooks = oracle.ActivationQuantization(backbone, weights)
        self.assertEqual([entry["module"] for entry in hooks.description], ["language_model.layers.0.mlp.gate_proj"])
        with torch.no_grad():
            np.testing.assert_array_equal(modules.layers[0].mlp.gate_proj(values).numpy(), [[127, 127]])
            np.testing.assert_array_equal(modules.layers[0].mlp.up_proj(values).numpy(), [[127.5, 127.5]])
        hooks.close()
        with torch.no_grad():
            np.testing.assert_array_equal(modules.layers[0].mlp.gate_proj(values).numpy(), [[127.5, 127.5]])

    def test_k_activation_hook_quantizes_ssm_output_in_runtime_head_order(self):
        import torch
        layer = torch.nn.Linear(4096, 1, bias=False)
        with torch.no_grad():
            layer.weight.zero_()
            layer.weight[0, 256] = 1
        values = torch.zeros(1, 4096)
        values[0, 0] = 127
        values[0, 128] = 0.25
        values[0, 256] = 0.5
        backbone = torch.nn.Module()
        backbone.language_model = torch.nn.Module()
        backbone.language_model.layers = torch.nn.ModuleList([torch.nn.Module()])
        backbone.language_model.layers[0].linear_attn = torch.nn.Module()
        backbone.language_model.layers[0].linear_attn.out_proj = layer
        hooks = oracle.ActivationQuantization(backbone, SimpleNamespace(tensors={"blk.0.ssm_out.weight": (GGML_Q4K,)}))
        self.addCleanup(hooks.close)
        with torch.no_grad():
            self.assertEqual(layer(values).item(), 0)

    def test_thinker_same_gguf_restores_tied_embeddings_without_an_lm_head_tensor(self):
        import torch
        model = torch.nn.Module()
        model.model = torch.nn.Module()
        model.model.language_model = torch.nn.Module()
        model.model.language_model.embed_tokens = torch.nn.Embedding(3, 2)
        model.lm_head = torch.nn.Linear(2, 3, bias=False)
        model.lm_head.weight = model.model.language_model.embed_tokens.weight
        model.get_input_embeddings = lambda: model.model.language_model.embed_tokens
        model.get_output_embeddings = lambda: model.lm_head
        writer = GgufWriter(self.root / "thinker.gguf")
        for key, value in {
            "general.architecture": "qwen35", "thx02.component": "thinker",
            "thx02.source_revision": oracle.REVISION, "thx02.model_sha256": oracle.MODEL_SHA256,
            "thx02.attention_mode": "causal", "thx02.recurrent_head_layout": "qwen35-modulo-v1",
        }.items():
            writer.add_meta(key, value)
        values = np.arange(6, dtype="<f4").reshape(3, 2)
        writer.add_tensor("token_embd.weight", GGML_F32, (2, 3), values.tobytes())
        writer.write()
        oracle.load_gguf_into_causal(model, writer.path)
        np.testing.assert_array_equal(model.lm_head.weight.detach().numpy(), values)
        self.assertEqual(model.get_input_embeddings().weight.data_ptr(), model.get_output_embeddings().weight.data_ptr())
        model.lm_head.weight = torch.nn.Parameter(torch.zeros(3, 2))
        with self.assertRaises(ValueError):
            oracle.load_gguf_into_causal(model, writer.path)

    def test_native_thinker_generation_keeps_full_logits_and_includes_stop_step(self):
        import torch
        import sys
        sys.modules.setdefault("fla", None)
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
        config = Qwen3_5Config(text_config={
            "dtype": "bfloat16",
            "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 2,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
            "linear_num_key_heads": 2, "linear_num_value_heads": 4,
            "linear_key_head_dim": 4, "linear_value_head_dim": 4,
            "layer_types": ["linear_attention", "full_attention"], "full_attention_interval": 2,
            "vocab_size": 64, "eos_token_id": 63, "mtp_num_hidden_layers": 0,
            "rope_parameters": {"rope_theta": 10000, "rope_type": "default", "partial_rotary_factor": 0.5,
                                "mrope_interleaved": True, "mrope_section": [1, 1, 0]}},
            vision_config={"dtype": "bfloat16", "hidden_size": 32, "intermediate_size": 64, "depth": 1,
                           "num_heads": 4, "out_hidden_size": 32, "num_position_embeddings": 4},
            tie_word_embeddings=True, dtype="bfloat16")
        model = oracle.causal_model(config)
        self.assertTrue(all(parameter.dtype == torch.float32 for parameter in model.parameters()))
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        np.testing.assert_allclose(model.model.language_model.rotary_emb.inv_freq.numpy(), [1, 0.01], atol=1e-8)
        inputs = {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.ones(1, 3, dtype=torch.long)}
        row = oracle.generate_thinker(model, inputs, self.root / "generate", steps=8, stop_token_ids=[0], pad_token_id=0)
        self.assertEqual(row["generated_ids"], [0])
        self.assertEqual(row["positions"], [[0, 0, 0, 0], [1, 1, 1, 0], [2, 2, 2, 0]])
        self.assertEqual(len(row["steps"]), 1)
        logits = np.load(row["steps"][0]["logits_path"], allow_pickle=False)
        self.assertEqual(logits.shape, (64,))
        np.testing.assert_array_equal(logits, np.zeros(64, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
