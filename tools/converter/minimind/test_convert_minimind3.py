import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

from tools.converter.minimind.convert_minimind3 import (
    ARCH_CONFIG,
    QUANT_BLOCK,
    QUANT_FILE_TYPE,
    QUANT_MODES,
    QUANT_SUFFIX,
    _architecture,
    _validate_config,
    load_checkpoint,
    write_gguf,
)
from tools.converter.utils.gguf import (
    GGML_F16,
    GGML_F32,
    GGML_Q4_0,
    GGML_Q8_0,
    read_gguf_directory,
    read_gguf_tensor_bytes,
)

VOCAB = 32
HIDDEN = 64
LAYERS = 1
EXPERTS = 4
EXPERTS_USED = 1
FFN = 128
HEADS = 4
KV_HEADS = 2
HEAD_DIM = 16
Q_DIM = HEADS * HEAD_DIM
KV_DIM = KV_HEADS * HEAD_DIM


def _tensor(shape: tuple[int, ...], offset: int = 0) -> torch.Tensor:
    count = int(np.prod(shape))
    values = ((np.arange(count, dtype=np.float32) + offset) % 31 - 15) / 17
    return torch.from_numpy(values.reshape(shape)).to(torch.float16)


def _config(arch: str) -> dict:
    config = {
        "architectures": [ARCH_CONFIG[arch]["architectures"][0]],
        "model_type": ARCH_CONFIG[arch]["model_type"],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "dtype": "float16",
        "head_dim": HEAD_DIM,
        "hidden_act": "silu",
        "hidden_size": HIDDEN,
        "intermediate_size": FFN,
        "layer_types": ["full_attention"] * LAYERS,
        "max_position_embeddings": 32768,
        "num_attention_heads": HEADS,
        "num_hidden_layers": LAYERS,
        "num_key_value_heads": KV_HEADS,
        "rms_norm_eps": 1e-6,
        "rope_scaling": None,
        "rope_theta": 1_000_000.0,
        "tie_word_embeddings": True,
        "use_sliding_window": False,
        "vocab_size": VOCAB,
    }
    if arch == "qwen3moe":
        config.update(
            {
                "decoder_sparse_step": 1,
                "mlp_only_layers": [],
                "moe_intermediate_size": FFN,
                "norm_topk_prob": True,
                "num_experts": EXPERTS,
                "num_experts_per_tok": EXPERTS_USED,
            }
        )
    return config


def _state(arch: str) -> dict[str, torch.Tensor]:
    state = {
        "model.embed_tokens.weight": _tensor((VOCAB, HIDDEN)),
        "model.norm.weight": _tensor((HIDDEN,), 3),
    }
    for layer in range(LAYERS):
        prefix = f"model.layers.{layer}"
        state.update(
            {
                f"{prefix}.self_attn.q_proj.weight": _tensor((Q_DIM, HIDDEN), 1),
                f"{prefix}.self_attn.k_proj.weight": _tensor((KV_DIM, HIDDEN), 2),
                f"{prefix}.self_attn.v_proj.weight": _tensor((KV_DIM, HIDDEN), 3),
                f"{prefix}.self_attn.o_proj.weight": _tensor((HIDDEN, Q_DIM), 4),
                f"{prefix}.self_attn.q_norm.weight": _tensor((HEAD_DIM,), 5),
                f"{prefix}.self_attn.k_norm.weight": _tensor((HEAD_DIM,), 6),
                f"{prefix}.input_layernorm.weight": _tensor((HIDDEN,), 7),
                f"{prefix}.post_attention_layernorm.weight": _tensor((HIDDEN,), 8),
            }
        )
        if arch == "qwen3moe":
            state[f"{prefix}.mlp.gate.weight"] = _tensor((EXPERTS, HIDDEN), 9)
            for expert in range(EXPERTS):
                state.update(
                    {
                        f"{prefix}.mlp.experts.{expert}.gate_proj.weight": _tensor(
                            (FFN, HIDDEN), 10 + expert * 3,
                        ),
                        f"{prefix}.mlp.experts.{expert}.up_proj.weight": _tensor(
                            (FFN, HIDDEN), 11 + expert * 3,
                        ),
                        f"{prefix}.mlp.experts.{expert}.down_proj.weight": _tensor(
                            (HIDDEN, FFN), 12 + expert * 3,
                        ),
                    }
                )
        else:
            state.update(
                {
                    f"{prefix}.mlp.gate_proj.weight": _tensor((FFN, HIDDEN), 10),
                    f"{prefix}.mlp.up_proj.weight": _tensor((FFN, HIDDEN), 11),
                    f"{prefix}.mlp.down_proj.weight": _tensor((HIDDEN, FFN), 12),
                }
            )
    return state


def _model_dir(root: Path, arch: str) -> Path:
    tokens = [f"t{i}" for i in range(VOCAB)]
    tokens[0] = "<|endoftext|>"
    tokens[1] = "<|im_start|>"
    tokens[2] = "<|im_end|>"
    tokens[3] = "<|buffer1|>"
    tokens[4] = "<tool_call>"
    tokenizer = {
        "model": {
            "type": "BPE",
            "vocab": {token: index for index, token in enumerate(tokens)},
            "merges": [["t", "3"], ["t", "4"]],
        },
        "added_tokens": [
            {
                "id": index,
                "content": tokens[index],
                "special": index < 3,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
            }
            for index in range(5)
        ],
    }
    tokenizer_config = {
        "add_bos_token": False,
        "add_eos_token": False,
        "bos_token": "<|im_start|>",
        "eos_token": "<|im_end|>",
        "pad_token": "<|endoftext|>",
        "unk_token": "<|endoftext|>",
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps(_config(arch)), encoding="utf-8")
    (root / "tokenizer.json").write_text(json.dumps(tokenizer), encoding="utf-8")
    (root / "tokenizer_config.json").write_text(
        json.dumps(tokenizer_config), encoding="utf-8",
    )
    (root / "chat_template.jinja").write_text("test-template", encoding="utf-8")
    return root


class QuantPolicyTests(unittest.TestCase):
    def test_every_mode_has_a_file_type_suffix_and_block(self):
        for mode in QUANT_MODES:
            self.assertIn(mode, QUANT_FILE_TYPE)
            self.assertIn(mode, QUANT_SUFFIX)
            self.assertIn(mode, QUANT_BLOCK)
        self.assertEqual(QUANT_FILE_TYPE["f16"], 1)
        self.assertEqual(QUANT_FILE_TYPE["q8_0"], 7)
        self.assertEqual(QUANT_FILE_TYPE["q4_0"], 2)

    def test_k_quants_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(ValueError):
                write_gguf(
                    _state("qwen3"),
                    _model_dir(root / "model", "qwen3"),
                    root / "out.gguf",
                    quant="q4_k",
                )


class ConfigTests(unittest.TestCase):
    def test_architectures_are_detected_from_the_config(self):
        self.assertEqual(_architecture(_config("qwen3")), "qwen3")
        self.assertEqual(_architecture(_config("qwen3moe")), "qwen3moe")

    def test_unknown_model_type_is_rejected(self):
        config = _config("qwen3")
        config["model_type"] = "llama"
        with self.assertRaisesRegex(ValueError, "unsupported"):
            _architecture(config)

    def test_config_rejects_untied_or_shared_expert_checkpoints(self):
        dense = _config("qwen3")
        dense["tie_word_embeddings"] = False
        with self.assertRaisesRegex(ValueError, "tie_word_embeddings"):
            _validate_config(dense, "qwen3")

        moe = _config("qwen3moe")
        moe["shared_expert_intermediate_size"] = FFN
        with self.assertRaisesRegex(ValueError, "shared expert"):
            _validate_config(moe, "qwen3moe")

    def test_config_must_match_checkpoint_shapes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = _model_dir(root / "model", "qwen3")
            config = _config("qwen3")
            config["vocab_size"] = VOCAB + 1
            (model / "config.json").write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "embed_tokens"):
                write_gguf(_state("qwen3"), model, root / "out.gguf")


class CheckpointTests(unittest.TestCase):
    def test_safetensors_checkpoint_is_loaded(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = _state("qwen3")
            save_file(state, str(root / "model.safetensors"))
            loaded = load_checkpoint(root)
            self.assertEqual(set(loaded), set(state))
            self.assertTrue(torch.equal(loaded["model.norm.weight"], state["model.norm.weight"]))


class DenseContractTests(unittest.TestCase):
    def test_f16_contract_maps_qk_norm_and_tied_embedding(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "model.gguf"
            write_gguf(
                _state("qwen3"),
                _model_dir(root / "model", "qwen3"),
                output,
                quant="f16",
            )
            metadata, tensors = read_gguf_directory(output)
            embed = read_gguf_tensor_bytes(output, "token_embd.weight")

        self.assertEqual(metadata["general.architecture"], "qwen3")
        self.assertEqual(metadata["general.file_type"], 1)
        self.assertEqual(metadata["qwen3.embedding_length"], HIDDEN)
        self.assertEqual(metadata["qwen3.block_count"], LAYERS)
        self.assertEqual(metadata["qwen3.feed_forward_length"], FFN)
        self.assertEqual(metadata["qwen3.attention.head_count"], HEADS)
        self.assertEqual(metadata["qwen3.attention.head_count_kv"], KV_HEADS)
        self.assertEqual(metadata["qwen3.attention.key_length"], HEAD_DIM)
        self.assertNotIn("qwen3.expert_count", metadata)
        self.assertEqual(metadata["tokenizer.ggml.pre"], "qwen2")
        self.assertEqual(metadata["tokenizer.ggml.bos_token_id"], 1)
        self.assertEqual(metadata["tokenizer.ggml.eos_token_id"], 2)
        self.assertEqual(metadata["tokenizer.ggml.unknown_token_id"], 0)
        self.assertEqual(metadata["tokenizer.ggml.padding_token_id"], 0)
        self.assertEqual(metadata["tokenizer.chat_template"], "test-template")
        self.assertEqual(metadata["tokenizer.ggml.token_type"][:5], [3, 3, 3, 3, 4])

        expected = {
            "token_embd.weight": (GGML_F16, (HIDDEN, VOCAB)),
            "output_norm.weight": (GGML_F32, (HIDDEN,)),
            "blk.0.attn_norm.weight": (GGML_F32, (HIDDEN,)),
            "blk.0.ffn_norm.weight": (GGML_F32, (HIDDEN,)),
            "blk.0.attn_q_norm.weight": (GGML_F32, (HEAD_DIM,)),
            "blk.0.attn_k_norm.weight": (GGML_F32, (HEAD_DIM,)),
            "blk.0.attn_q.weight": (GGML_F16, (HIDDEN, Q_DIM)),
            "blk.0.attn_k.weight": (GGML_F16, (HIDDEN, KV_DIM)),
            "blk.0.attn_v.weight": (GGML_F16, (HIDDEN, KV_DIM)),
            "blk.0.attn_output.weight": (GGML_F16, (Q_DIM, HIDDEN)),
            "blk.0.ffn_gate.weight": (GGML_F16, (HIDDEN, FFN)),
            "blk.0.ffn_up.weight": (GGML_F16, (HIDDEN, FFN)),
            "blk.0.ffn_down.weight": (GGML_F16, (FFN, HIDDEN)),
        }
        self.assertEqual(set(tensors), set(expected))
        for name, (ggml_type, dims) in expected.items():
            self.assertEqual(tensors[name][0], ggml_type, name)
            self.assertEqual(tensors[name][1], dims, name)
        self.assertNotIn("output.weight", tensors)
        self.assertEqual(embed, _state("qwen3")["model.embed_tokens.weight"].numpy().tobytes())

    def test_tied_lm_head_is_validated_and_omitted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = _model_dir(root / "model", "qwen3")
            state = _state("qwen3")
            state["lm_head.weight"] = state["model.embed_tokens.weight"].clone()
            output = root / "model.gguf"
            write_gguf(state, model, output, quant="f16")
            _metadata, tensors = read_gguf_directory(output)
            self.assertNotIn("output.weight", tensors)

            state["lm_head.weight"][0, 0] += 1
            with self.assertRaisesRegex(ValueError, "tied embedding"):
                write_gguf(state, model, root / "bad.gguf", quant="f16")


class MoeContractTests(unittest.TestCase):
    def test_f16_contract_stacks_experts_and_keeps_router_f32(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "model.gguf"
            write_gguf(
                _state("qwen3moe"),
                _model_dir(root / "model", "qwen3moe"),
                output,
                quant="f16",
            )
            metadata, tensors = read_gguf_directory(output)

        self.assertEqual(metadata["general.architecture"], "qwen3moe")
        self.assertEqual(metadata["qwen3moe.expert_count"], EXPERTS)
        self.assertEqual(metadata["qwen3moe.expert_used_count"], EXPERTS_USED)
        self.assertEqual(metadata["qwen3moe.expert_feed_forward_length"], FFN)
        self.assertNotIn("qwen3moe.expert_shared_feed_forward_length", metadata)
        self.assertNotIn("qwen3moe.vocab_size", metadata)
        self.assertEqual(
            tensors["blk.0.ffn_gate_exps.weight"],
            (GGML_F16, (HIDDEN, FFN, EXPERTS), EXPERTS * FFN * HIDDEN * 2),
        )
        self.assertEqual(
            tensors["blk.0.ffn_up_exps.weight"],
            (GGML_F16, (HIDDEN, FFN, EXPERTS), EXPERTS * FFN * HIDDEN * 2),
        )
        self.assertEqual(
            tensors["blk.0.ffn_down_exps.weight"],
            (GGML_F16, (FFN, HIDDEN, EXPERTS), EXPERTS * FFN * HIDDEN * 2),
        )
        self.assertEqual(tensors["blk.0.ffn_gate_inp.weight"][0], GGML_F32)
        self.assertEqual(tensors["blk.0.ffn_gate_inp.weight"][1], (HIDDEN, EXPERTS))
        self.assertNotIn("blk.0.ffn_gate.weight", tensors)
        self.assertNotIn("blk.0.ffn_gate_shexp.weight", tensors)

    def test_moe_config_requires_expert_tensors(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "missing checkpoint"):
                write_gguf(
                    _state("qwen3"),
                    _model_dir(root / "model", "qwen3moe"),
                    root / "out.gguf",
                )


class ValidationTests(unittest.TestCase):
    def test_missing_and_extra_tensors_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = _model_dir(root / "model", "qwen3")

            missing = _state("qwen3")
            del missing["model.layers.0.self_attn.q_norm.weight"]
            with self.assertRaisesRegex(ValueError, "missing checkpoint"):
                write_gguf(missing, model, root / "missing.gguf")

            extra = _state("qwen3")
            extra["model.layers.0.mlp.bias"] = _tensor((FFN,))
            with self.assertRaisesRegex(ValueError, "unexpected checkpoint"):
                write_gguf(extra, model, root / "extra.gguf")

    def test_wrong_qk_norm_shape_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = _state("qwen3")
            state["model.layers.0.self_attn.q_norm.weight"] = _tensor((HEAD_DIM + 1,))
            with self.assertRaisesRegex(ValueError, "q_norm"):
                write_gguf(
                    state,
                    _model_dir(root / "model", "qwen3"),
                    root / "out.gguf",
                )


class QuantPayloadTests(unittest.TestCase):
    def test_q8_0_and_q4_0_payload_sizes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = _model_dir(root / "model", "qwen3moe")
            for mode, ggml_type, block_bytes in (
                ("q8_0", GGML_Q8_0, 34),
                ("q4_0", GGML_Q4_0, 18),
            ):
                with self.subTest(mode=mode):
                    output = root / f"model-{mode}.gguf"
                    write_gguf(_state("qwen3moe"), model, output, quant=mode)
                    _metadata, tensors = read_gguf_directory(output)
                    self.assertEqual(tensors["token_embd.weight"][0], ggml_type)
                    self.assertEqual(
                        tensors["token_embd.weight"][2],
                        VOCAB * HIDDEN // 32 * block_bytes,
                    )
                    self.assertEqual(
                        tensors["blk.0.ffn_gate_exps.weight"][0], ggml_type,
                    )
                    self.assertEqual(
                        tensors["blk.0.ffn_gate_exps.weight"][2],
                        EXPERTS * FFN * HIDDEN // 32 * block_bytes,
                    )
                    self.assertEqual(tensors["blk.0.ffn_gate_inp.weight"][0], GGML_F32)
                    self.assertEqual(tensors["blk.0.attn_norm.weight"][0], GGML_F32)
                    self.assertEqual(tensors["blk.0.attn_q_norm.weight"][0], GGML_F32)


if __name__ == "__main__":
    unittest.main()