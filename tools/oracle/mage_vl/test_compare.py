import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools.oracle.mage_vl.compare import compare_quantized_logits


class QuantizedComparisonTests(unittest.TestCase):
    def test_divergent_prefix_limits_comparison_but_requires_complete_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            reference, native = Path(directory)/'ref', Path(directory)/'native'
            reference.mkdir()
            (reference/'weights.json').write_text(json.dumps({'decoder': 'upstream gguf.quants.dequantize'}))
            for case in ['text', 'square', 'wide', 'resize']:
                for root in [reference, native]:
                    folder = root/case
                    folder.mkdir(parents=True)
                    ids = [1]*8 if root == reference else [1, 2]+[1]*6
                    (folder/'greedy.json').write_text(json.dumps(ids))
                    for step in range(8):
                        # Later logits belong to different prefixes and must not be compared.
                        values = [1, 2] if root == reference or step < 2 else [100, 200]
                        np.asarray(values, dtype='<f4').tofile(folder/f'logits{step}.f32')
            report = compare_quantized_logits(reference, native)
            self.assertTrue(all(row['same_prefix_logit_steps'] == 2 for row in report.values()))
            self.assertFalse(report['text']['generated_equal'])
            (native/'text/logits7.f32').unlink()
            with self.assertRaises(AssertionError):
                compare_quantized_logits(reference, native)

    def test_same_prefix_error_cannot_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'weights.json').write_text(json.dumps({'decoder': 'upstream gguf.quants.dequantize'}))
            for base in [root, root/'native']:
                folder = base/'text'
                folder.mkdir(parents=True)
                (folder/'greedy.json').write_text(json.dumps([1]*8))
                for step in range(8):
                    np.asarray([1, 2] if base == root else [10, 20], dtype='<f4').tofile(folder/f'logits{step}.f32')
            with self.assertRaises(AssertionError):
                compare_quantized_logits(root, root/'native')
