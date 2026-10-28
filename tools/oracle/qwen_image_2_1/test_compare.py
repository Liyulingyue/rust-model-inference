import struct
import tempfile
import unittest
from pathlib import Path

from compare import compare


class NumericalComparisonTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def row(self, filename, values, name='qwen.block.0'):
        path = Path(self.directory.name)/filename
        path.write_bytes(struct.pack(f'<{len(values)}f', *values))
        return dict(name=name, shape=[len(values)], len=len(values), binary_path=str(path))

    def test_accepts_and_reports_small_rounding_error(self):
        a = self.row('a', [1.0001, 2.0001])
        b = self.row('b', [1.0, 2.0])
        result = compare([a], [b], 0.001)
        self.assertEqual(result['f32_values'], 2)
        self.assertGreater(result['max_abs'], 0)
        self.assertLess(result['max_relative_l2'], 0.001)
        with self.assertRaises(AssertionError):
            compare([a], [b])

    def test_rejects_large_or_nonfinite_errors(self):
        b = self.row('b', [1.0])
        for value in [1.1, float('nan'), float('inf')]:
            with self.assertRaises(AssertionError):
                compare([self.row('a', [value])], [b], 0.001)

    def test_token_ids_and_schedule_stay_exact(self):
        for name in ['qwen.text.tokens', 'qwen.sample.sigmas', 'qwen.text.image_slots']:
            a = self.row('a', [1.0000001], name)
            b = self.row('b', [1.0], name)
            with self.assertRaises(AssertionError):
                compare([a], [b], 0.01)

    def test_derived_sampler_inputs_use_numerical_bounds(self):
        for name in ['qwen.input.x', 'qwen.input.context']:
            a = self.row('a', [1.0001], name)
            b = self.row('b', [1.0], name)
            compare([a], [b], 0.01)
            with self.assertRaises(AssertionError):
                compare([a], [b], 0.01, exact_inputs=True)

    def test_rejects_missing_checkpoint_or_shape(self):
        a = self.row('a', [1.0, 2.0])
        b = self.row('b', [1.0, 2.0])
        with self.assertRaises(AssertionError):
            compare([a], [], 0.01)
        b['shape'] = [3]
        with self.assertRaises(AssertionError):
            compare([a], [b], 0.01)

    def test_rejects_invalid_tolerance(self):
        a = self.row('a', [1.0])
        for tolerance in [0, -1, float('nan'), float('inf')]:
            with self.assertRaises(AssertionError):
                compare([a], [a], tolerance)


if __name__ == '__main__':
    unittest.main()
