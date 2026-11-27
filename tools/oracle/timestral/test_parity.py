import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools.oracle.timestral.parity import compare, verify_gpu_execution


class ComparatorTest(unittest.TestCase):
    def test_gpu_requires_real_dispatches_and_rejects_fallback(self):
        good = 'compute scope=timestral.transformer backend=Vulkan\ncompute scope=timestral.transformer gpu_dispatches=252'
        self.assertEqual(verify_gpu_execution(good)['blocks'], 1)
        for bad in ['', good.replace('Vulkan', 'Cpu'), good.replace('252', '0'), good+'\ncompute scope=timestral.transformer backend=Cpu']:
            with self.assertRaises(AssertionError):
                verify_gpu_execution(bad)

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
