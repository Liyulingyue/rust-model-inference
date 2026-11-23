import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from tools.oracle.fastframes_l.compare import compare,metrics

class ComparisonTests(unittest.TestCase):
    def test_shared_noise_must_match_exactly_even_within_float_tolerance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'tokens.json').write_text('[1]')
            np.ones(32,dtype='<f4').tofile(root/'noise.f32')
            np.full(32,1.001,dtype='<f4').tofile(root/'actual.f32')
            (root/'manifest.json').write_text(json.dumps({'recipe':'controlled','checkpoints':[{'name':'fastframes.noise','file':'noise.f32'}]}))
            trace=root/'trace.jsonl'
            trace.write_text('\n'.join(json.dumps(x) for x in [
                {'name':'fastframes.token_ids','token_ids':[1]},
                {'name':'fastframes.noise','binary_path':str(root/'actual.f32')},
            ]))
            with self.assertRaisesRegex(ValueError,'shared noise'):
                compare(trace,root,root/'result.json')

    def test_known_metric_and_invalid_shape(self):
        values=metrics([1.,2.],[1.,3.])
        self.assertAlmostEqual(values['rmse'],np.sqrt(.5))
        self.assertEqual(values['max_abs'],1.)
        with self.assertRaises(ValueError):metrics([1.],[1.,2.])
        with self.assertRaises(ValueError):metrics([float('nan')],[1.])

    def test_compares_repeated_checkpoints_in_execution_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);records=[];actual=[]
            (root/'tokens.json').write_text('[1]')
            for i,name in enumerate(['a','b','a']):
                path=root/f'reference-{i}.f32';np.ones(4,dtype='<f4').tofile(path)
                records.append({'name':name,'file':path.name})
                path=root/f'actual-{i}.f32';np.full(4,1. if i==0 else 2.,dtype='<f4').tofile(path)
                actual.append({'name':name,'binary_path':str(path)})
            (root/'manifest.json').write_text(json.dumps({'recipe':'controlled','checkpoints':records}))
            trace=root/'trace.jsonl';trace.write_text('\n'.join(json.dumps(x) for x in [{'name':'fastframes.token_ids','token_ids':[1]},*actual]))
            report=root/'result.json'
            self.assertFalse(compare(trace,root,report))
            self.assertEqual(json.loads(report.read_text())['first_failure']['name'],'b')
            trace.write_text(json.dumps({'name':'fastframes.token_ids','token_ids':[2]}))
            with self.assertRaises(ValueError):compare(trace,root,report)

if __name__=='__main__':unittest.main()
