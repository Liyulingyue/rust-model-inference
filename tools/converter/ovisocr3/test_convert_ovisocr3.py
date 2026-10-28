import copy
import unittest
from pathlib import Path

from tools.converter.ovisocr3.convert_ovisocr3 import (
    LLAMA_CPP_COMMIT,
    build_commands,
    expected_tensor_shapes,
    validate_config,
    validate_tensor_inventory,
)


def model_config() -> dict:
    return {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "image_token_id": 248056,
        "model_type": "qwen3_5",
        "tie_word_embeddings": True,
        "video_token_id": 248057,
        "vision_end_token_id": 248054,
        "vision_start_token_id": 248053,
        "text_config": {
            "attention_bias": False,
            "attention_dropout": 0.0,
            "attn_output_gate": True,
            "dtype": "bfloat16",
            "hidden_size": 1024,
            "intermediate_size": 3584,
            "hidden_act": "silu",
            "num_hidden_layers": 24,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "full_attention_interval": 4,
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 16,
            "linear_value_head_dim": 128,
            "rms_norm_eps": 1e-6,
            "vocab_size": 248320,
            "max_position_embeddings": 262144,
            "mamba_ssm_dtype": "float32",
            "model_type": "qwen3_5_text",
            "tie_word_embeddings": True,
            "use_cache": True,
            "rope_parameters": {
                "mrope_interleaved": True,
                "rope_theta": 10000000,
                "mrope_section": [11, 11, 10],
                "partial_rotary_factor": 0.25,
                "rope_type": "default",
            },
            "layer_types": [
                "linear_attention" if i % 4 != 3 else "full_attention"
                for i in range(24)
            ],
        },
        "vision_config": {
            "depth": 12,
            "hidden_size": 768,
            "intermediate_size": 3072,
            "model_type": "qwen3_5",
            "num_heads": 12,
            "patch_size": 16,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
            "num_position_embeddings": 2304,
            "out_hidden_size": 1024,
            "hidden_act": "gelu_pytorch_tanh",
            "in_channels": 3,
        },
    }


class OvisOCR3ContractTest(unittest.TestCase):
    def test_accepts_ovis_qwen35_contract_and_catalogues_all_tensors(self):
        validate_config(model_config())
        shapes = expected_tensor_shapes(model_config())
        self.assertEqual(len(shapes), 473)
        self.assertEqual(shapes["model.visual.patch_embed.proj.weight"], (768, 3, 2, 16, 16))

    def test_rejects_nearby_architecture_instead_of_falling_back(self):
        invalid = copy.deepcopy(model_config())
        invalid["architectures"] = ["Qwen3VLForConditionalGeneration"]
        with self.assertRaisesRegex(ValueError, "architecture"):
            validate_config(invalid)

    def test_rejects_unknown_or_wrong_dtype_tensor(self):
        config = model_config()
        shapes = expected_tensor_shapes(config)
        inventory = {name: ("BF16", shape) for name, shape in shapes.items()}
        inventory["extra.weight"] = ("BF16", (1,))
        with self.assertRaisesRegex(ValueError, "unexpected tensor"):
            validate_tensor_inventory(inventory, shapes)
        inventory.pop("extra.weight")
        inventory["model.language_model.norm.weight"] = ("F16", (1024,))
        with self.assertRaisesRegex(ValueError, "dtype"):
            validate_tensor_inventory(inventory, shapes)

    def test_commands_pin_oracle_and_preserve_separate_mmproj(self):
        commands = build_commands(
            Path("/opt/llama.cpp"),
            Path("/models/ovis"),
            Path("/out/ovis.gguf"),
            Path("/out/mmproj-ovis.gguf"),
            python="python3",
        )
        self.assertEqual(commands[0][:2], ["python3", "-c"])
        self.assertIn("tokenizer.ggml.normalizer.nfc", commands[0][2])
        self.assertIn("--no-nextn", commands[0][2])
        self.assertEqual(commands[0][-3:], ["/opt/llama.cpp", "/models/ovis", "/out/ovis.gguf"])
        self.assertIn("--mmproj", commands[1])
        self.assertEqual(LLAMA_CPP_COMMIT, "b96806d96061049a5b574269b049bf6241d63d46")


if __name__ == "__main__":
    unittest.main()
