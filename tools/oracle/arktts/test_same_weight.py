import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch
import torch.nn.functional as functional

from tools.converter.arktts.convert_arktts import FORMATS, encode_tensor
from tools.converter.utils.gguf import GgufWriter
from tools.oracle.arktts.same_weight import load, prepared_linears


class SameWeightTests(unittest.TestCase):
    def test_independent_decode_and_prepared_contracts_cover_every_export_format(self):
        weights = np.random.default_rng(611).normal(size=(2, 256)).astype("<f4")
        inputs = torch.from_numpy(np.random.default_rng(612).normal(size=(3, 256)).astype("<f4"))
        for precision in FORMATS:
            with self.subTest(precision=precision), tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "model.gguf"
                writer = GgufWriter(path)
                writer.add_meta("general.architecture", "arktts")
                writer.add_meta("arktts.bundle_id", "a" * 64)
                kind, payload = encode_tensor("weight", weights.tobytes(), 0, (256, 2), precision)
                writer.add_tensor("weight", kind, (256, 2), payload)
                writer.write()
                module = torch.nn.Linear(256, 2, bias=False)
                contracts, bundle = load(module, path, "arktts")
                self.assertEqual(bundle, "a" * 64)
                relative = np.linalg.norm(module.weight.detach().numpy() - weights) / np.linalg.norm(weights)
                self.assertLess(relative, .16)
                with prepared_linears(contracts):
                    actual = module(inputs)
                self.assertTrue(torch.isfinite(actual).all())
                if precision == "f32":
                    torch.testing.assert_close(actual, functional.linear(inputs, torch.from_numpy(weights)))
                with self.assertRaisesRegex(ValueError, "expected"):
                    load(module, path, "arktts_codec")

    def test_linear_hook_restores_on_exception(self):
        original = functional.linear
        with self.assertRaisesRegex(RuntimeError, "injected"):
            with prepared_linears({}):
                raise RuntimeError("injected")
        self.assertIs(functional.linear, original)


if __name__ == "__main__":
    unittest.main()
