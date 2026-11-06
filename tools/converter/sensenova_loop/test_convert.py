import unittest
import copy
import json
from pathlib import Path
from tools.converter.sensenova_loop.convert import (
    execution_plan,
    tensor_name,
    validate_config,
)


class ContractTests(unittest.TestCase):
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
            ).read_text()
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


if __name__ == "__main__":
    unittest.main()
