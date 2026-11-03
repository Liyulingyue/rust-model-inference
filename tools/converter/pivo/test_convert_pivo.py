import math
from pathlib import Path
import tempfile
import unittest

from tools.converter.pivo.convert_pivo import CONTRACT, convert


class ConversionTest(unittest.TestCase):
    def test_checkpoint_contract_covers_both_networks_and_feature_buffers(self):
        self.assertEqual(sum(math.prod(shape) for shape in CONTRACT['tensors'].values()), 1401562 + 40)
        self.assertEqual(CONTRACT['tensors']['vocoder.step.gru1.weight_ih'], [480, 272])
        self.assertEqual(CONTRACT['tensors']['acoustic.phase_projection.weight'], [96, 4])

    def test_invalid_sources_and_existing_output_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'checkpoints').mkdir()
            (root / 'checkpoints/acoustic.pt').write_bytes(b'invalid')
            output = root / 'result.gguf'
            with self.assertRaisesRegex(ValueError, 'source hash mismatch'):
                convert(root, output)
            self.assertFalse(output.exists())
            output.write_bytes(b'preserve me')
            with self.assertRaises(FileExistsError):
                convert(root, output)
            self.assertEqual(output.read_bytes(), b'preserve me')


if __name__ == '__main__':
    unittest.main()
