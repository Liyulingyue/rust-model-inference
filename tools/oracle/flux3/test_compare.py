import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from tools.oracle.flux3.compare import compare, validate_reference


class ComparisonTests(unittest.TestCase):
    def test_raw_bits_numeric_bounds_and_missing_occurrences(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            def trace(filename,name,values):
                data=root/(filename+'.f32')
                np.asarray(values,dtype='<f4').tofile(data)
                path=root/filename
                path.write_text(json.dumps(dict(name=name,occurrence=0,shape=[len(values)],binary_path=str(data)))+'\n')
                return path
            oracle=trace('reference','flux3.noise',[0.0,1.0])
            actual=trace('actual','flux3.noise',[-0.0,1.0])
            self.assertFalse(compare(oracle,actual)[0]['passed'])
            oracle=trace('reference','flux3.velocity',[1.0,2.0])
            actual=trace('actual','flux3.velocity',[1.0001,2.0001])
            self.assertTrue(compare(oracle,actual)[0]['passed'])
            actual=trace('actual','flux3.velocity',[1.1,2.1])
            self.assertFalse(compare(oracle,actual)[0]['passed'])
            actual.write_text('')
            with self.assertRaises(AssertionError): compare(oracle,actual)
            oracle.write_text('')
            with self.assertRaises(AssertionError): compare(oracle,actual)

    def test_both_traces_cannot_omit_the_same_sampling_step(self):
        incomplete=[dict(name='flux3.token_ids',occurrence=0,shape=[80]),dict(name='flux3.text',occurrence=0,shape=[80,20480])]
        with self.assertRaises(AssertionError):
            validate_reference(incomplete,dict(side=64,steps=2,reference=[]))
        validate_reference(incomplete,dict(side=64,text_only=True))

    def test_threshold_comes_from_oracle_dtype(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for name, shape in [('flux3.token_ids', [80]), ('flux3.text', [80, 20480])]:
                data = root / (name + '.f32')
                np.ones(shape, dtype='<f4').tofile(data)
                rows.append(dict(name=name, occurrence=0, shape=shape, binary_path=str(data)))
            oracle = root / 'trace.jsonl'
            oracle.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            actual_data = root / 'actual.f32'
            np.full((80, 20480), 1.04, dtype='<f4').tofile(actual_data)
            rows[-1]['binary_path'] = str(actual_data)
            actual = root / 'actual.jsonl'
            actual.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            for dtype, passed in [('f32', False), ('bf16', True)]:
                (root/'run.json').write_text(json.dumps(dict(arguments=dict(side=64, text_only=True, dtype=dtype))))
                self.assertEqual(compare(oracle, actual)[-1]['passed'], passed)


if __name__=='__main__': unittest.main()
