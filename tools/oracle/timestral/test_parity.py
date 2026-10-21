import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools.oracle.timestral.parity import compare


class ComparatorTest(unittest.TestCase):
    def test_rejects_wrong_layout_bits_numbers_and_missing_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            data = directory/'values.f32'
            trace = directory/'trace.jsonl'
            expected = np.array([0., 1.], dtype='f4')

            def write(name, values, shape=[2]):
                np.array(values, dtype='<f4').tofile(data)
                trace.write_text(json.dumps(dict(name=name, shape=shape, occurrence=0, binary_path=str(data)))+'\n')

            write('timestral.observed', expected)
            self.assertEqual(compare([('timestral.observed', [2], expected)], trace)[0]['max_abs'], 0)
            for name, values, shape in [('timestral.observed', [-0., 1.], [2]),
                                         ('timestral.observed', [0., 1.], [1, 2]),
                                         ('timestral.positions', [0., 1.], [2])]:
                write(name, values, shape)
                with self.assertRaises(AssertionError):
                    compare([('timestral.observed', [2], expected)], trace)
            write('timestral.forecast', [0., 2.])
            with self.assertRaises(AssertionError):
                compare([('timestral.forecast', [2], expected)], trace)
            near_zero = np.array([1e-4, -1e-4], dtype='f4')
            write('timestral.head_monotone', near_zero + 8e-8)
            compare([('timestral.head_monotone', [2], near_zero)], trace)
            trace.write_text('')
            with self.assertRaises(AssertionError):
                compare([('timestral.forecast', [2], expected)], trace)


if __name__ == '__main__':
    unittest.main()
