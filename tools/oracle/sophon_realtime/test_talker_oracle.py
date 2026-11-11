"""Bounded independent talker Oracle checks; no source model weights required."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from transformers import Qwen2Config, Qwen2Model

from tools.oracle.sophon_realtime import talker_oracle as oracle

ROOT = Path(__file__).resolve().parents[3]
TAG = ROOT / "target/sophon/reference/tag/plugin/sophon_realtime"


class TalkerOracleTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.text = torch.arange(32, dtype=torch.float32).reshape(16, 2)
        self.speech = torch.arange(6761 * 2, dtype=torch.float32).reshape(6761, 2)

    @property
    def reference(self):
        if not TAG.is_dir():
            self.skipTest(
                "fetch the pinned official tag reference before running schedule checks"
            )
        return oracle.load_tag_reference(TAG)

    def test_official_fill_consumes_five_then_duplicates_last_input_for_final(self):
        schedule = oracle.Schedule(
            self.reference, list(range(7)), self.text, self.speech, self.text[12:13]
        )
        np.testing.assert_array_equal(schedule.accept(6564), self.text[:5])
        for token in range(15):
            schedule.accept(token)
        self.assertEqual(schedule.forced(), 6564)
        np.testing.assert_array_equal(
            schedule.accept(6564),
            torch.cat([self.speech[14:15], self.text[5:7], self.speech[6563:6564]]),
        )
        self.assertEqual(schedule.phase, "final")
        self.assertIsNone(schedule.accept(6562))
        with self.assertRaises(ValueError):
            schedule.accept(1)

    def test_forced_fill_and_phase_controls_cannot_be_overridden_by_teacher(self):
        schedule = oracle.Schedule(
            self.reference, [0], self.text, self.speech, self.text[12:13]
        )
        for token in range(15):
            schedule.accept(token)
        with self.assertRaises(ValueError):
            schedule.accept(0)
        with self.assertRaises(ValueError):
            schedule.accept(6562)
        schedule.accept(6564)
        for invalid in [-1, 6561, 6563, 6564, 6761]:
            with self.subTest(token=invalid), self.assertRaises(ValueError):
                schedule.accept(invalid)

    def test_exact_five_text_tokens_require_another_fill_before_task(self):
        schedule = oracle.Schedule(
            self.reference, list(range(5)), self.text, self.speech, self.text[12:13]
        )
        schedule.accept(6564)
        self.assertEqual(schedule.phase, "bi")
        np.testing.assert_array_equal(
            schedule.accept(6564), torch.cat([self.text[4:5], self.speech[6563:6564]])
        )
        self.assertEqual(schedule.phase, "final")

    def test_tag_clean_calls_pinned_official_behavior(self):
        self.assertEqual(
            self.reference["tag_clean"](
                "# <strong>你好，世界</strong>[cough](ignored)→yes", True
            ),
            " <strong>你好</strong>，世界[cough], so yes",
        )
        self.assertEqual(
            self.reference["tag_clean"]("<laughter>hi", True), "<laughter>hi</laughter>"
        )

    def test_name_mapping_includes_qkv_bias_and_rejects_unknown(self):
        self.assertEqual(
            oracle.tensor_name("blk.0.attn_k.bias"), "layers.0.self_attn.k_proj.bias"
        )
        self.assertEqual(oracle.tensor_name("token_embd.weight"), "embed_tokens.weight")
        for name in [
            "blk.0.attn_output.bias",
            "blk.-1.attn_q.weight",
            "blk.0.attn_q_norm.weight",
            "bad",
        ]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                oracle.tensor_name(name)

    def test_trace_requires_complete_tokens_exact_ids_and_contiguous_steps(self):
        trace = [
            {"name": "sophon.talker.prompt_ids", "token_ids": [4, 5]},
            {"name": "sophon.talker.text_ids", "token_ids": [6]},
            {"name": "sophon.talker.sampled_tokens", "token_ids": [6564, 1]},
            {
                "name": "sophon.talker.logits",
                "step": 0,
                "shape": [6761],
                "binary_path": "one",
            },
            {
                "name": "sophon.talker.logits",
                "step": 1,
                "shape": [6761],
                "binary_path": "two",
            },
        ]
        teacher, checkpoints = oracle.trace_inputs(trace, [4, 5], [6])
        self.assertEqual(teacher, [6564, 1])
        self.assertEqual(len(checkpoints), 2)
        with self.assertRaises(ValueError):
            oracle.trace_inputs(trace, [4, 7], [6])
        with self.assertRaises(ValueError):
            oracle.trace_inputs(trace[:-1], [4, 5], [6])
        trace[-1]["step"] = 3
        with self.assertRaises(ValueError):
            oracle.trace_inputs(trace, [4, 5], [6])

    def test_independent_gguf_decode_and_qwen2_mapping_match_original_forward(self):
        import gguf
        from gguf.quants import quantize, dequantize

        torch.manual_seed(8)
        config = Qwen2Config(
            vocab_size=16,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=128,
            rms_norm_eps=1e-6,
            rope_theta=1e6,
        )
        config._attn_implementation = "eager"
        original = Qwen2Model(config).eval()
        speech = torch.randn(6761, 32) * 0.02
        head = torch.randn(6761, 32) * 0.02
        source_names = {
            "embed_tokens.weight": "token_embd.weight",
            "norm.weight": "output_norm.weight",
            "layers.0.input_layernorm.weight": "blk.0.attn_norm.weight",
            "layers.0.post_attention_layernorm.weight": "blk.0.ffn_norm.weight",
        }
        for source, target in [
            ("self_attn.q_proj", "attn_q"),
            ("self_attn.k_proj", "attn_k"),
            ("self_attn.v_proj", "attn_v"),
            ("self_attn.o_proj", "attn_output"),
            ("mlp.gate_proj", "ffn_gate"),
            ("mlp.up_proj", "ffn_up"),
            ("mlp.down_proj", "ffn_down"),
        ]:
            source_names[f"layers.0.{source}.weight"] = f"blk.0.{target}.weight"
            if source in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"):
                source_names[f"layers.0.{source}.bias"] = f"blk.0.{target}.bias"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.gguf"
            writer = gguf.GGUFWriter(str(path), "sophon_realtime_talker")
            expected_state = {}
            for name, tensor in original.state_dict().items():
                values = tensor.numpy(force=True)
                kind = (
                    gguf.GGMLQuantizationType.Q8_0
                    if values.ndim == 2
                    else gguf.GGMLQuantizationType.F32
                )
                encoded = quantize(values, kind)
                writer.add_tensor(source_names[name], encoded, raw_dtype=kind)
                expected_state[name] = torch.from_numpy(
                    dequantize(encoded, kind).copy()
                )
            writer.add_tensor("speech_embd.weight", speech.numpy())
            writer.add_tensor("output.weight", head.numpy())
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()
            loaded, loaded_speech, loaded_head = oracle.load_model(
                gguf.GGUFReader(str(path), mode="c"), config, Path(directory)
            )
            original.load_state_dict(expected_state)
            initial = torch.cat(
                [speech[6561:6562], original.embed_tokens.weight[[2, 3]]]
            )
            with torch.no_grad():
                expected = torch.nn.functional.linear(
                    original(
                        inputs_embeds=initial[None], use_cache=False
                    ).last_hidden_state[0, -1],
                    head,
                )
                logits, record = next(
                    oracle.reference_steps(
                        loaded,
                        loaded_speech,
                        loaded_head,
                        self.reference,
                        [1, 2],
                        [2, 3],
                        [6564, 1, 6562],
                        3,
                    )
                )
            np.testing.assert_allclose(logits, expected.numpy(), rtol=1e-6, atol=1e-6)
            self.assertEqual(record["selected_token"], 6564)
            with torch.no_grad():
                records = list(
                    oracle.reference_steps(
                        loaded,
                        loaded_speech,
                        loaded_head,
                        self.reference,
                        [1, 2],
                        [2, 3],
                        [6564, 1, 6562],
                        3,
                    )
                )
                # Full causal prefixes independently verify repeated last input, TASK and cached decode.
                prefix = initial
                for index, following in enumerate(
                    [
                        torch.cat(
                            [
                                original.embed_tokens.weight[3:4],
                                original.embed_tokens.weight[[1, 2]],
                                speech[6563:6564],
                            ]
                        ),
                        speech[1:2],
                    ],
                    start=1,
                ):
                    prefix = torch.cat([prefix, following])
                    expected = torch.nn.functional.linear(
                        original(
                            inputs_embeds=prefix[None], use_cache=False
                        ).last_hidden_state[0, -1],
                        head,
                    )
                    np.testing.assert_allclose(
                        records[index][0], expected.numpy(), rtol=1e-5, atol=1e-6
                    )
            self.assertEqual([row[1]["input_rows"] for row in records], [3, 4, 1])
            self.assertEqual(
                [row[1]["phase"] for row in records], ["bi", "final", "final"]
            )


if __name__ == "__main__":
    unittest.main()
