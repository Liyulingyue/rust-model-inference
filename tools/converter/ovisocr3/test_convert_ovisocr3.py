import copy
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from tools.converter.ovisocr3.convert_ovisocr3 import (
    LLAMA_CPP_COMMIT,
    build_commands,
    expected_tensor_shapes,
    validate_config,
    validate_source,
    validate_tensor_inventory,
)


def processor_config() -> dict:
    return {
        "size": {"shortest_edge": 65536, "longest_edge": 16777216},
        "patch_size": 16,
        "temporal_patch_size": 2,
        "merge_size": 2,
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
        "processor_class": "Qwen3VLProcessor",
        "image_processor_type": "Qwen2VLImageProcessorFast",
    }


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
    def test_mmproj_preserves_source_pixel_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "preprocessor_config.json").write_text(json.dumps(processor_config()))
            (root / "conversion").mkdir()
            (root / "conversion/__init__.py").touch()
            (root / "conversion/qwen3vl.py").write_text('''
class Writer:
    def __init__(self):
        self.values = {}
    def add_vision_min_pixels(self, value):
        self.values["clip.vision.image_min_pixels"] = value
    def add_vision_max_pixels(self, value):
        self.values["clip.vision.image_max_pixels"] = value
class Qwen3VLVisionModel:
    def __init__(self):
        self.gguf_writer = Writer()
    def set_gguf_parameters(self):
        self.gguf_writer.values["base"] = True
''')
            (root / "convert_hf_to_gguf.py").write_text('''
import json, sys
from pathlib import Path
from conversion.qwen3vl import Qwen3VLVisionModel
def main():
    model = Qwen3VLVisionModel()
    model.set_gguf_parameters()
    Path(sys.argv[sys.argv.index("--outfile") + 1]).write_text(json.dumps(model.gguf_writer.values))
if __name__ == "__main__":
    main()
''')
            output = root / "mmproj.gguf"
            command = build_commands(root, root, root / "text.gguf", output)[1]
            subprocess.run(command, check=True)
            self.assertEqual(json.loads(output.read_text()), {
                "base": True,
                "clip.vision.image_min_pixels": 65536,
                "clip.vision.image_max_pixels": 16777216,
            })

    def test_rejects_image_processor_drift(self):
        config = model_config()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text(json.dumps(config))
            (root / "tokenizer.json").write_text(json.dumps({
                "model": {"type": "BPE"}, "normalizer": {"type": "NFC"},
            }))
            (root / "tokenizer_config.json").write_text(json.dumps({
                "add_bos_token": False, "tokenizer_class": "Qwen2Tokenizer",
                "bos_token": None, "eos_token": "<|im_end|>", "pad_token": "<|endoftext|>",
                "added_tokens_decoder": {str(i): {"content": token} for i, token in enumerate([
                    "<|vision_start|>", "<|vision_end|>", "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>",
                ], 248053)},
            }))
            header = json.dumps({name: {"dtype": "BF16", "shape": shape}
                                 for name, shape in expected_tensor_shapes(config).items()}).encode()
            (root / "model.safetensors").write_bytes(len(header).to_bytes(8, "little") + header)
            processor_path = root / "preprocessor_config.json"
            processor_path.write_text(json.dumps(processor_config()))
            validate_source(root)
            for key, value in {
                "size": {"shortest_edge": 8192, "longest_edge": 4194304},
                "patch_size": 14, "temporal_patch_size": 1, "merge_size": 1,
                "image_processor_type": "CLIPImageProcessor",
            }.items():
                with self.subTest(key=key):
                    invalid = processor_config()
                    invalid[key] = value
                    processor_path.write_text(json.dumps(invalid))
                    with self.assertRaisesRegex(ValueError, "preprocessor"):
                        validate_source(root)

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
        self.assertIn("--mmproj", commands[1][2])
        self.assertEqual(LLAMA_CPP_COMMIT, "b96806d96061049a5b574269b049bf6241d63d46")


if __name__ == "__main__":
    unittest.main()
