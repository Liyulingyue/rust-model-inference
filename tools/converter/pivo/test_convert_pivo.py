import math
from pathlib import Path
import tempfile
import unittest

from tools.converter.pivo.convert_pivo import CONTRACT, convert, quantizable


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

    def test_q8_0_covers_the_matrices_whose_rows_are_block_aligned(self):
        # GGML quantizes a row at a time, so the row length — the fastest
        # varying dimension — has to be a whole number of 32-value blocks.
        self.assertTrue(quantizable((96, 96), None))
        self.assertTrue(quantizable((480, 160), None))
        self.assertFalse(quantizable((96, 4), None))
        self.assertFalse(quantizable((1, 80), None))
        self.assertFalse(quantizable((128, 688), None))
        # The one-dimensional norms, biases and statistics stay F32.
        self.assertFalse(quantizable((96,), None))
        self.assertFalse(quantizable((20,), None))
        # The convolution kernels keep a kernel axis of 3 or 5.
        self.assertFalse(quantizable((96, 96, 5), None))
        self.assertFalse(quantizable((128, 64, 3), None))


if __name__ == '__main__':
    unittest.main()
