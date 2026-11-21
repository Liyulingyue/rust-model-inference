import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from tools.converter.minimind.convert_minimind2moe import (
    DEFAULT_CHAT_TEMPLATE,
    QUANT_BLOCK,
    QUANT_FILE_TYPE,
    QUANT_MODES,
    QUANT_SUFFIX,
    _infer_layout,
    write_gguf,
)
from tools.converter.utils.gguf import (
    GGML_F32,
    GGML_F16,
    GGML_Q4_0,
    GGML_Q8_0,
    read_gguf_directory,
    read_gguf_tensor_bytes,
)

VOCAB = 16
HIDDEN = 64
LAYERS = 1
EXPERTS = 2
EXPERT_FFN = 96
SHARED_FFN = 32
HEADS = 2
KV_HEADS = 1
HEAD_DIM = HIDDEN // HEADS


def _tensor(shape: tuple[int, ...], offset: int = 0) -> torch.Tensor:
    count = int(np.prod(shape))
    values = ((np.arange(count, dtype=np.float32) + offset) % 31 - 15) / 17
    return torch.from_numpy(values.reshape(shape)).to(torch.float16)


def _state() -> dict[str, torch.Tensor]:
    state = {
        "model.embed_tokens.weight": _tensor((VOCAB, HIDDEN)),
        "model.norm.weight": _tensor((HIDDEN,), 3),
        "lm_head.weight": _tensor((VOCAB, HIDDEN)),
    }
    for layer in range(LAYERS):
        prefix = f"model.layers.{layer}"
        state.update(
            {
                f"{prefix}.self_attn.q_proj.weight": _tensor((HIDDEN, HIDDEN), 1),
                f"{prefix}.self_attn.k_proj.weight": _tensor((KV_HEADS * HEAD_DIM, HIDDEN), 2),
                f"{prefix}.self_attn.v_proj.weight": _tensor((KV_HEADS * HEAD_DIM, HIDDEN), 3),
                f"{prefix}.self_attn.o_proj.weight": _tensor((HIDDEN, HIDDEN), 4),
                f"{prefix}.input_layernorm.weight": _tensor((HIDDEN,), 5),
                f"{prefix}.post_attention_layernorm.weight": _tensor((HIDDEN,), 6),
                f"{prefix}.mlp.gate.weight": _tensor((EXPERTS, HIDDEN), 7),
                f"{prefix}.mlp.shared_experts.0.gate_proj.weight": _tensor(
                    (SHARED_FFN, HIDDEN), 8,
                ),
                f"{prefix}.mlp.shared_experts.0.up_proj.weight": _tensor(
                    (SHARED_FFN, HIDDEN), 9,
                ),
                f"{prefix}.mlp.shared_experts.0.down_proj.weight": _tensor(
                    (HIDDEN, SHARED_FFN), 10,
                ),
            }
        )
        for expert in range(EXPERTS):
            state.update(
                {
                    f"{prefix}.mlp.experts.{expert}.gate_proj.weight": _tensor(
                        (EXPERT_FFN, HIDDEN), 11 + expert * 3,
                    ),
                    f"{prefix}.mlp.experts.{expert}.up_proj.weight": _tensor(
                        (EXPERT_FFN, HIDDEN), 12 + expert * 3,
                    ),
                    f"{prefix}.mlp.experts.{expert}.down_proj.weight": _tensor(
                        (HIDDEN, EXPERT_FFN), 13 + expert * 3,
                    ),
                }
            )
    return state


def _tokenizer_dir(root: Path) -> Path:
    tokens = [f"t{i}" for i in range(VOCAB)]
    tokens[0] = "<|endoftext|>"
    tokens[1] = "<|im_start|>"
    tokens[2] = "<|im_end|>"
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
                "special": True,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
            }
            for index in range(3)
        ],
    }
    config = {
        "add_bos_token": False,
        "add_eos_token": False,
        "bos_token": "<|im_start|>",
        "eos_token": "<|im_end|>",
        "chat_template": "unused",
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "tokenizer.json").write_text(json.dumps(tokenizer), encoding="utf-8")
    (root / "tokenizer_config.json").write_text(json.dumps(config), encoding="utf-8")
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
                    _state(),
                    _tokenizer_dir(root / "tokenizer"),
                    root / "out.gguf",
                    quant="q4_k",
                    num_heads=HEADS,
                    num_kv_heads=KV_HEADS,
                )


class LayoutTests(unittest.TestCase):
    def test_layout_is_derived_from_the_checkpoint(self):
        layout = _infer_layout(_state(), HEADS, KV_HEADS)
        self.assertEqual(layout["vocab_size"], VOCAB)
        self.assertEqual(layout["n_embd"], HIDDEN)
        self.assertEqual(layout["n_layer"], LAYERS)
        self.assertEqual(layout["n_head"], HEADS)
        self.assertEqual(layout["n_head_kv"], KV_HEADS)
        self.assertEqual(layout["n_embd_head"], HEAD_DIM)
        self.assertEqual(layout["expert_count"], EXPERTS)
        self.assertEqual(layout["expert_ffn"], EXPERT_FFN)
        self.assertEqual(layout["shared_expert_ffn"], SHARED_FFN)

    def test_lm_head_must_match_the_tied_embedding(self):
        state = _state()
        state["lm_head.weight"] = state["lm_head.weight"].clone()
        state["lm_head.weight"][0, 0] += 1
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "tied embedding"):
                write_gguf(
                    state,
                    _tokenizer_dir(root / "tokenizer"),
                    root / "out.gguf",
                    num_heads=HEADS,
                    num_kv_heads=KV_HEADS,
                )

    def test_qk_norm_is_rejected(self):
        state = _state()
        state["model.layers.0.self_attn.q_norm.weight"] = _tensor((HEAD_DIM,))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "unexpected checkpoint tensors"):
                write_gguf(
                    state,
                    _tokenizer_dir(root / "tokenizer"),
                    root / "out.gguf",
                    num_heads=HEADS,
                    num_kv_heads=KV_HEADS,
                )


class GgufContractTests(unittest.TestCase):
    def test_f16_contract_maps_shared_and_stacked_experts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "model.gguf"
            write_gguf(
                _state(),
                _tokenizer_dir(root / "tokenizer"),
                output,
                quant="f16",
                num_heads=HEADS,
                num_kv_heads=KV_HEADS,
            )
            metadata, tensors = read_gguf_directory(output)
            embed = read_gguf_tensor_bytes(output, "token_embd.weight")

        self.assertEqual(metadata["general.architecture"], "minimind2moe")
        self.assertEqual(metadata["general.file_type"], 1)
        self.assertEqual(metadata["minimind2moe.embedding_length"], HIDDEN)
        self.assertEqual(metadata["minimind2moe.block_count"], LAYERS)
        self.assertEqual(metadata["minimind2moe.attention.head_count"], HEADS)
        self.assertEqual(metadata["minimind2moe.attention.head_count_kv"], KV_HEADS)
        self.assertEqual(metadata["minimind2moe.attention.key_length"], HEAD_DIM)
        self.assertEqual(metadata["minimind2moe.expert_count"], EXPERTS)
        self.assertEqual(metadata["minimind2moe.expert_used_count"], 2)
        self.assertEqual(metadata["minimind2moe.expert_feed_forward_length"], EXPERT_FFN)
        self.assertEqual(
            metadata["minimind2moe.expert_shared_feed_forward_length"], SHARED_FFN,
        )
        self.assertEqual(metadata["tokenizer.ggml.pre"], "qwen2")
        self.assertEqual(metadata["tokenizer.ggml.bos_token_id"], 1)
        self.assertEqual(metadata["tokenizer.ggml.eos_token_id"], 2)
        self.assertEqual(metadata["tokenizer.chat_template"], DEFAULT_CHAT_TEMPLATE)
        self.assertEqual(metadata["tokenizer.ggml.token_type"][:3], [3, 3, 3])
        self.assertEqual(metadata["tokenizer.ggml.token_type"][3], 1)

        expected = {
            "token_embd.weight": (GGML_F16, (HIDDEN, VOCAB)),
            "output_norm.weight": (GGML_F32, (HIDDEN,)),
            "blk.0.attn_norm.weight": (GGML_F32, (HIDDEN,)),
            "blk.0.ffn_norm.weight": (GGML_F32, (HIDDEN,)),
            "blk.0.attn_q.weight": (GGML_F16, (HIDDEN, HIDDEN)),
            "blk.0.attn_k.weight": (GGML_F16, (HIDDEN, KV_HEADS * HEAD_DIM)),
            "blk.0.attn_v.weight": (GGML_F16, (HIDDEN, KV_HEADS * HEAD_DIM)),
            "blk.0.attn_output.weight": (GGML_F16, (HIDDEN, HIDDEN)),
            "blk.0.ffn_gate_inp.weight": (GGML_F32, (HIDDEN, EXPERTS)),
            "blk.0.ffn_gate_exps.weight": (GGML_F16, (HIDDEN, EXPERT_FFN, EXPERTS)),
            "blk.0.ffn_up_exps.weight": (GGML_F16, (HIDDEN, EXPERT_FFN, EXPERTS)),
            "blk.0.ffn_down_exps.weight": (GGML_F16, (EXPERT_FFN, HIDDEN, EXPERTS)),
            "blk.0.ffn_gate_shexp.weight": (GGML_F16, (HIDDEN, SHARED_FFN)),
            "blk.0.ffn_up_shexp.weight": (GGML_F16, (HIDDEN, SHARED_FFN)),
            "blk.0.ffn_down_shexp.weight": (GGML_F16, (SHARED_FFN, HIDDEN)),
        }
        self.assertEqual(set(tensors), set(expected))
        for name, (ggml_type, dims) in expected.items():
            self.assertEqual(tensors[name][0], ggml_type, name)
            self.assertEqual(tensors[name][1], dims, name)
        self.assertNotIn("output.weight", tensors)
        self.assertNotIn("blk.0.attn_q_norm.weight", tensors)
        self.assertNotIn("blk.0.attn_k_norm.weight", tensors)

        self.assertEqual(embed, _state()["model.embed_tokens.weight"].numpy().tobytes())

    def test_q8_0_and_q4_0_payload_sizes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tokenizer = _tokenizer_dir(root / "tokenizer")
            for mode, llvm_type, block_bytes in (
                ("q8_0", GGML_Q8_0, 34),
                ("q4_0", GGML_Q4_0, 18),
            ):
                with self.subTest(mode=mode):
                    output = root / f"model-{mode}.gguf"
                    write_gguf(
                        _state(), tokenizer, output, quant=mode,
                        num_heads=HEADS, num_kv_heads=KV_HEADS,
                    )
                    _metadata, tensors = read_gguf_directory(output)
                    self.assertEqual(tensors["token_embd.weight"][0], llvm_type)
                    self.assertEqual(
                        tensors["token_embd.weight"][2],
                        VOCAB * HIDDEN // 32 * block_bytes,
                    )
                    self.assertEqual(tensors["blk.0.ffn_gate_exps.weight"][0], llvm_type)
                    self.assertEqual(
                        tensors["blk.0.ffn_gate_exps.weight"][2],
                        EXPERTS * EXPERT_FFN * HIDDEN // 32 * block_bytes,
                    )
                    self.assertEqual(tensors["blk.0.ffn_gate_inp.weight"][0], GGML_F32)
                    self.assertEqual(tensors["blk.0.attn_norm.weight"][0], GGML_F32)


if __name__ == "__main__":
    unittest.main()