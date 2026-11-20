import unittest
import copy
import json
import struct
import tempfile
from pathlib import Path
import numpy as np
from tools.converter.sensenova_loop.convert import (
    convert,
    execution_plan,
    tensor_name,
    validate_config,
)
from tools.converter.ovisocr3.convert_ovisocr3 import expected_tensor_shapes
from tools.converter.utils.gguf import _read_gguf, f32_to_bf16, GGML_F32, GGML_Q8_0


class ContractTests(unittest.TestCase):
    def test_bf16_release_exports_exact_f32_and_quantized_projections(self):
        config = json.loads((Path(__file__).parents[2] / "oracle/sensenova_loop/release-config.json").read_text(encoding="utf-8"))
        config["text_config"].update(hidden_size=32, intermediate_size=64, num_hidden_layers=4,
            num_attention_heads=2, num_key_value_heads=1, head_dim=32, vocab_size=32,
            layer_types=["linear_attention"] * 3 + ["full_attention"],
            linear_key_head_dim=4, linear_value_head_dim=4, linear_num_key_heads=2, linear_num_value_heads=4)
        config["text_config"]["rope_parameters"]["mrope_section"] = [2, 1, 1]
        config["loop_settings"].update(start=1, end=3)
        config["mhc"].update(start=1, end=3)
        config["vision_config"].update(depth=1, hidden_size=8, intermediate_size=16,
            num_heads=2, num_position_embeddings=16, out_hidden_size=32, patch_size=2)
        shapes = expected_tensor_shapes(config)
        shapes["lm_head.weight"] = (32, 32)
        for layer in (1, 2):
            for kind in ("mhc_attn", "mhc_ffn"):
                shapes[f"model.language_model.layers.{layer}.{kind}.weight"] = (8, 64)
                shapes[f"model.language_model.layers.{layer}.{kind}.bias"] = (8,)
        header, payload = {}, bytearray()
        for name, shape in shapes.items():
            raw = f32_to_bf16(np.full(shape, 0.25, dtype=np.float32))
            header[name] = {"dtype": "BF16", "shape": shape, "data_offsets": [len(payload), len(payload) + len(raw)]}
            payload.extend(raw)
        gate = "model.language_model.loop_gate.logits"
        raw = np.zeros(32, dtype="<f4").tobytes()
        header[gate] = {"dtype": "F32", "shape": [32], "data_offsets": [len(payload), len(payload) + len(raw)]}
        payload.extend(raw)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text(json.dumps(config))
            (root / "preprocessor_config.json").write_text(json.dumps({"image_mean": [0.5] * 3, "image_std": [0.5] * 3, "size": {"shortest_edge": 16, "longest_edge": 1024}}))
            (root / "tokenizer.json").write_text(json.dumps({"model": {"type": "BPE", "vocab": {"a": 0}, "merges": []}, "normalizer": {"type": "NFC"}, "added_tokens": []}))
            (root / "chat_template.jinja").write_text("{{ messages }}")
            encoded = json.dumps(header).encode()
            encoded += b" " * (-len(encoded) % 8)
            (root / "model.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
            for quant in ("f32", "q8_0"):
                output, mmproj = root / f"{quant}.gguf", root / f"{quant}-mmproj.gguf"
                convert(root, output, mmproj, quant)
                _, tensors = _read_gguf(output)
                self.assertNotIn(gate, tensors)
                self.assertEqual(tensors["token_embd.weight"][0], GGML_F32 if quant == "f32" else GGML_Q8_0)
                self.assertEqual(tensors["blk.1.mhc_attn.weight"][0], GGML_F32)
                dtype, _, length, offset = tensors["output_norm.weight"]
                self.assertEqual(dtype, GGML_F32)
                with output.open("rb") as stream:
                    stream.seek(offset)
                    np.testing.assert_array_equal(np.frombuffer(stream.read(length), dtype="<f4"), np.full(32, 1.25))

    def config(self):
        return {
            "text_config": {
                "num_hidden_layers": 8,
                "layer_types": ["linear_attention"] * 3
                + ["full_attention"]
                + ["linear_attention"] * 3
                + ["full_attention"],
            },
            "loop_settings": {"start": 2, "end": 6, "eval_count": 2, "counts": [2]},
            "mhc": {
                "start": 2,
                "end": 6,
                "streams": 2,
                "iterations": 20,
                "fixed_identity": False,
                "pre_scale": 1.0,
            },
        }

    def test_plan_preserves_physical_weights_and_independent_visits(self):
        self.assertEqual(
            execution_plan(self.config()), [0, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 7]
        )

    def test_rejects_mhc_crossing_loop_boundary(self):
        config = self.config()
        config["mhc"]["end"] = 7
        with self.assertRaises(ValueError):
            execution_plan(config)

    def test_rejects_non_integer_and_unbounded_plan(self):
        for count in (True, 0, 10**20):
            config = self.config()
            config["loop_settings"]["eval_count"] = count
            config["loop_settings"]["counts"] = [count]
            with self.assertRaises(ValueError):
                execution_plan(config)

    def test_distinct_router_names_and_canonical_backbone(self):
        self.assertEqual(
            tensor_name("model.language_model.layers.4.mhc_attn.weight"),
            "blk.4.mhc_attn.weight",
        )
        self.assertEqual(
            tensor_name("model.language_model.layers.4.linear_attn.A_log"),
            "blk.4.ssm_a",
        )
        self.assertEqual(tensor_name("lm_head.weight"), "output.weight")
        with self.assertRaises(ValueError):
            tensor_name("unexpected.weight")

    def test_rejects_unsupported_rope_and_vision_semantics(self):
        config = json.loads(
            (
                Path(__file__).parents[2] / "oracle/sensenova_loop/config.json"
            ).read_text(encoding="utf-8")
        )
        validate_config(config)
        for section, key, value in (
            ("text_config", "hidden_act", "gelu"),
            ("vision_config", "hidden_act", "silu"),
            ("vision_config", "out_hidden_size", 1),
        ):
            broken = copy.deepcopy(config)
            broken[section][key] = value
            with self.assertRaises(ValueError):
                validate_config(broken)

        for key, value in (
            ("mrope_interleaved", False),
            ("rope_type", "yarn"),
            ("mrope_section", [1, 1, 1]),
        ):
            broken = copy.deepcopy(config)
            broken["text_config"]["rope_parameters"][key] = value
            with self.assertRaises(ValueError):
                validate_config(broken)

    def test_release_yarn_contract_and_invalid_scaling(self):
        config = json.loads(
            (Path(__file__).parents[2] / "oracle/sensenova_loop/release-config.json")
            .read_text(encoding="utf-8")
        )
        validate_config(config)
        self.assertEqual(len(execution_plan(config)), 56)
        for key, value in (
            ("factor", 0),
            ("factor", float("nan")),
            ("original_max_position_embeddings", 0),
            ("original_max_position_embeddings", True),
            ("rope_type", "dynamic"),
        ):
            broken = copy.deepcopy(config)
            broken["text_config"]["rope_parameters"][key] = value
            with self.assertRaises(ValueError):
                validate_config(broken)


if __name__ == "__main__":
    unittest.main()
