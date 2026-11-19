"""Synthetic CosyVoice3 talker conversion; no real source weights are required."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from tools.converter.sophon_realtime import convert as c
from tools.converter.sophon_realtime import talker as t
from tools.converter.utils.gguf import GGML_F32, _read_gguf


def fixture(root, *, hidden=256, heads=2, kv_heads=1, intermediate=256):
    root.mkdir(exist_ok=True)
    text = root / "CosyVoice-BlankEN"
    text.mkdir(exist_ok=True)
    config = dict(
        model_type="qwen2",
        hidden_size=hidden,
        intermediate_size=intermediate,
        num_hidden_layers=1,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        vocab_size=320,
        max_position_embeddings=32,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        rope_theta=1000000.0,
        use_sliding_window=False,
        attention_dropout=0.0,
        tie_word_embeddings=True,
    )
    (text / "config.json").write_text(json.dumps(config))
    (root / "cosyvoice3.yaml").write_text(
        f"sample_rate: 24000\nllm_input_size: {hidden}\nllm_output_size: {hidden}\n"
        "token_frame_rate: 25\ntoken_mel_ratio: 2\n"
        "llm: !new:cosyvoice.llm.llm.CosyVoice3LM\n"
        "    speech_token_size: 5\n    mix_ratio: [5, 15]\n"
    )
    (text / "vocab.json").write_text(json.dumps({"a": 0, "b": 1, "ab": 2}))
    (text / "merges.txt").write_text("#version: 0.2\na b\n")
    tokenizer = dict(
        tokenizer_class="Qwen2Tokenizer",
        add_prefix_space=False,
        split_special_tokens=False,
        chat_template="{{ messages }}\n",
        eos_token="<|im_end|>",
        pad_token="<|endoftext|>",
        added_tokens_decoder={
            str(i): dict(
                content=piece,
                special=True,
                normalized=False,
                single_word=False,
                lstrip=False,
                rstrip=False,
            )
            for i, piece in enumerate(
                ("<|endoftext|>", "<|im_start|>", "<|im_end|>"), 3
            )
        },
    )
    (text / "tokenizer_config.json").write_text(json.dumps(tokenizer))
    head_dim = hidden // heads
    kv = kv_heads * head_dim
    shapes = {
        "llm.model.model.embed_tokens.weight": (320, hidden),
        "llm.model.lm_head.weight": (320, hidden),
        "llm.model.model.norm.weight": (hidden,),
        "llm_decoder.weight": (205, hidden),
        "speech_embedding.weight": (205, hidden),
    }
    shapes.update(
        {
            "llm.model.model.layers.0." + name: shape
            for name, shape in (
                ("input_layernorm.weight", (hidden,)),
                ("post_attention_layernorm.weight", (hidden,)),
                ("self_attn.q_proj.weight", (hidden, hidden)),
                ("self_attn.q_proj.bias", (hidden,)),
                ("self_attn.k_proj.weight", (kv, hidden)),
                ("self_attn.k_proj.bias", (kv,)),
                ("self_attn.v_proj.weight", (kv, hidden)),
                ("self_attn.v_proj.bias", (kv,)),
                ("self_attn.o_proj.weight", (hidden, hidden)),
                ("mlp.gate_proj.weight", (intermediate, hidden)),
                ("mlp.up_proj.weight", (intermediate, hidden)),
                ("mlp.down_proj.weight", (hidden, intermediate)),
            )
        }
    )
    state = {
        name: torch.linspace(-0.2, 0.3, int(np.prod(shape))).reshape(shape)
        for name, shape in shapes.items()
    }
    torch.save(state, root / "llm.pt")
    return config, state


def convert(root, output, quant="f32"):
    with patch.object(t, "K_WORKERS", 1):
        return t.convert_talker(root, output, quant=quant, validate_pin=False)


class TalkerTests(unittest.TestCase):
    def test_pinned_inventory_and_native_mapping(self):
        specs = list(t.talker_specs(t.PINNED_TEXT, 6561))
        self.assertEqual(len(specs), 293)
        by_source = {name: (target, shape) for name, target, shape in specs}
        self.assertEqual(
            by_source["llm.model.model.embed_tokens.weight"],
            ("token_embd.weight", (151936, 896)),
        )
        self.assertEqual(by_source["llm.model.lm_head.weight"], (None, (151936, 896)))
        self.assertEqual(
            by_source["llm_decoder.weight"], ("output.weight", (6761, 896))
        )
        self.assertEqual(
            by_source["speech_embedding.weight"], ("speech_embd.weight", (6761, 896))
        )
        self.assertEqual(
            by_source["llm.model.model.layers.23.self_attn.k_proj.bias"],
            ("blk.23.attn_k.bias", (128,)),
        )
        self.assertFalse(any("q_norm" in name for name in by_source))

    def test_all_formats_have_independently_decodable_payloads(self):
        from gguf import GGMLQuantizationType
        from gguf.quants import dequantize

        budgets = dict(
            f32=0,
            f16=0.0005,
            bf16=0.005,
            q8_0=0.01,
            q4_0=0.15,
            q4_1=0.15,
            q4_k=0.12,
            q5_k=0.08,
            q6_k=0.04,
        )
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            config, state = fixture(root)
            for quant, budget in budgets.items():
                out = root / (quant + ".gguf")
                with patch.object(c, "STREAM_CHUNK_BYTES", 32768):
                    convert(root, out, quant)
                metadata, directory = _read_gguf(out)
                self.assertEqual(
                    metadata["general.architecture"], "sophon_realtime_talker"
                )
                self.assertEqual(metadata["sophon_realtime.component"], "talker")
                self.assertEqual(
                    metadata["sophon_realtime.source_revision"], c.REVISION
                )
                self.assertEqual(metadata["qwen2.attention.head_count"], 2)
                self.assertEqual(metadata["qwen2.attention.head_count_kv"], 1)
                self.assertTrue(metadata["sophon_realtime.talker.has_qkv_bias"])
                self.assertEqual(len(directory), 16)
                self.assertEqual(directory["output.weight"][1], (256, 205))
                types = json.loads(metadata["sophon_realtime.tensor_types_json"])
                self.assertEqual(
                    types, {name: info[0] for name, info in directory.items()}
                )
                controls = json.loads(metadata["sophon_realtime.talker_config_json"])
                self.assertEqual(
                    {
                        key: controls[key]
                        for key in (
                            "speech_token_size",
                            "speech_vocab_size",
                            "sos_token_id",
                            "eos_token_id",
                            "task_token_id",
                            "fill_token_id",
                        )
                    },
                    dict(
                        speech_token_size=5,
                        speech_vocab_size=205,
                        sos_token_id=5,
                        eos_token_id=6,
                        task_token_id=7,
                        fill_token_id=8,
                    ),
                )
                self.assertIn(
                    "llm.model.lm_head.weight",
                    json.loads(metadata["sophon_realtime.omitted_tensors_json"]),
                )
                with out.open("rb") as stream:
                    for source, target, _ in t.talker_specs(config, 5):
                        if target is None:
                            continue
                        kind, _, length, offset = directory[target]
                        stream.seek(offset)
                        actual = dequantize(
                            np.frombuffer(stream.read(length), dtype=np.uint8),
                            GGMLQuantizationType(kind),
                        ).reshape(-1)
                        expected = state[source].numpy().reshape(-1)
                        if kind == GGML_F32:
                            np.testing.assert_array_equal(actual, expected)
                        elif quant in ("f16", "bf16"):
                            dtype = torch.float16 if quant == "f16" else torch.bfloat16
                            np.testing.assert_array_equal(
                                actual,
                                state[source].to(dtype).float().numpy().reshape(-1),
                            )
                        relative = np.linalg.norm(
                            actual.astype(np.float64) - expected
                        ) / np.linalg.norm(expected)
                        self.assertLessEqual(relative, budget)

    def test_tokenizer_preserves_originals_and_cosyvoice_special_order(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            fixture(root)
            out = root / "out.gguf"
            convert(root, out)
            metadata, _ = _read_gguf(out)
            tokens = metadata["tokenizer.ggml.tokens"]
            self.assertEqual(len(tokens), 320)
            self.assertEqual(
                tokens[0:7],
                [
                    "a",
                    "b",
                    "ab",
                    "<|endoftext|>",
                    "<|im_start|>",
                    "<|im_end|>",
                    "<|endofprompt|>",
                ],
            )
            self.assertEqual(tokens[7], "[breath]")
            self.assertEqual(tokens[23], "<|endofsystem|>")
            self.assertEqual(tokens[283], "[ǜ]")
            self.assertEqual(metadata["tokenizer.ggml.eos_token_id"], 3)
            self.assertEqual(metadata["tokenizer.ggml.padding_token_id"], 3)
            self.assertFalse(metadata["tokenizer.ggml.normalizer.nfc"])
            self.assertEqual(metadata["tokenizer.ggml.merges"], ["a b"])
            self.assertEqual(metadata["tokenizer.chat_template"], "{{ messages }}\n")
            for name, suffix in (
                ("vocab.json", "vocab_json"),
                ("tokenizer_config.json", "tokenizer_config_json"),
                ("merges.txt", "merges_txt"),
            ):
                self.assertEqual(
                    metadata["sophon_realtime.talker_" + suffix],
                    (root / "CosyVoice-BlankEN" / name).read_text(),
                )

    def test_k_unaligned_matrices_and_vectors_stay_f32(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            fixture(root, hidden=288, heads=9, intermediate=512)
            out = root / "q4_k.gguf"
            convert(root, out, "q4_k")
            _, tensors = _read_gguf(out)
            for name in (
                "token_embd.weight",
                "output.weight",
                "speech_embd.weight",
                "blk.0.attn_q.weight",
                "blk.0.attn_q.bias",
                "output_norm.weight",
            ):
                self.assertEqual(tensors[name][0], GGML_F32)
            self.assertEqual(tensors["blk.0.ffn_down.weight"][0], 12)

    def test_bad_checkpoint_inventory_shape_dtype_and_values_fail_atomically(self):
        for problem in (
            "missing",
            "unknown",
            "shape",
            "dtype",
            "nonfinite",
            "not_tensor",
            "nested",
        ):
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                _, state = fixture(root)
                name = "speech_embedding.weight"
                if problem == "missing":
                    del state[name]
                elif problem == "unknown":
                    state["unknown"] = torch.ones(1)
                elif problem == "shape":
                    state[name] = state[name][:-1]
                elif problem == "dtype":
                    state[name] = state[name].half()
                elif problem == "nonfinite":
                    state[name][0, 0] = float("nan")
                elif problem == "not_tensor":
                    state[name] = "not a tensor"
                elif problem == "nested":
                    state = {"state_dict": state}
                torch.save(state, root / "llm.pt")
                out = root / "out.gguf"
                with self.assertRaises(ValueError):
                    convert(root, out)
                self.assertFalse(out.exists())
                self.assertEqual(list(root.glob(".out.gguf.*")), [])

    def test_config_tokenizer_and_pin_rejection(self):
        for problem in ("heads", "activation", "yaml", "vocab", "flags", "pin"):
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                config, _ = fixture(root)
                text = root / "CosyVoice-BlankEN"
                if problem in ("heads", "activation"):
                    config[
                        "num_attention_heads" if problem == "heads" else "hidden_act"
                    ] = (3 if problem == "heads" else "relu")
                    (text / "config.json").write_text(json.dumps(config))
                elif problem == "yaml":
                    with (root / "cosyvoice3.yaml").open("a") as stream:
                        stream.write("    speech_token_size: 7\n")
                elif problem == "vocab":
                    (text / "vocab.json").write_text(json.dumps({"a": 0, "b": 0}))
                elif problem == "flags":
                    path = text / "tokenizer_config.json"
                    value = json.loads(path.read_text())
                    value["added_tokens_decoder"]["3"]["normalized"] = True
                    path.write_text(json.dumps(value))
                out = root / "out.gguf"
                with self.assertRaises(ValueError):
                    t.convert_talker(root, out, validate_pin=problem == "pin")
                self.assertFalse(out.exists())
                self.assertEqual(list(root.glob(".out.gguf.*")), [])

    def test_no_overwrite_or_leftovers_after_writer_failure(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            fixture(root)
            out = root / "out.gguf"
            out.write_bytes(b"existing")
            with self.assertRaises(FileExistsError):
                convert(root, out)
            self.assertEqual(out.read_bytes(), b"existing")
            out.unlink()
            out.symlink_to(root / "absent")
            with self.assertRaises(FileExistsError):
                convert(root, out)
            out.unlink()
            with patch.object(
                t.GgufWriter, "write", side_effect=RuntimeError("failed")
            ):
                with self.assertRaisesRegex(RuntimeError, "failed"):
                    convert(root, out)
            self.assertFalse(out.exists())
            self.assertEqual(list(root.glob(".out.gguf.*")), [])

    def test_release_root_and_unknown_format(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            fixture(root / "talker")
            convert(root, root / "out.gguf")
            self.assertTrue((root / "out.gguf").is_file())
            with self.assertRaisesRegex(ValueError, "unknown format"):
                convert(root, root / "bad.gguf", "unknown")


if __name__ == "__main__":
    unittest.main()
