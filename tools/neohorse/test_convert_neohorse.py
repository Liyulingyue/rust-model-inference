import json
from pathlib import Path
import tempfile
import unittest

from convert_neohorse import validate_source


class SourceValidationTest(unittest.TestCase):
    def test_requires_text_architecture_nfc_and_no_mtp_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            inputs = {
                "config.json": {"architectures": ["Qwen3_5ForCausalLM"]},
                "tokenizer.json": {"normalizer": {"type": "NFC"}},
                "model.safetensors.index.json": {"weight_map": {"lm_head.weight": "part.safetensors"}},
            }
            for name, value in inputs.items():
                (model / name).write_text(json.dumps(value))
            validate_source(model)
            for name, invalid, error in [
                ("config.json", {"architectures": ["Qwen3ForCausalLM"]}, "text-only"),
                ("tokenizer.json", {"normalizer": None}, "NFC"),
                ("model.safetensors.index.json", {"weight_map": {"mtp.layers.0.weight": "part.safetensors"}}, "MTP"),
            ]:
                with self.subTest(name=name):
                    (model / name).write_text(json.dumps(invalid))
                    with self.assertRaisesRegex(ValueError, error):
                        validate_source(model)
                    (model / name).write_text(json.dumps(inputs[name]))


if __name__ == "__main__":
    unittest.main()
