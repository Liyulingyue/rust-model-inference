"""Compare reference.py and mage_vl_oracle artifacts; fail outside declared gates."""
import argparse
import json
from pathlib import Path
import numpy as np


def metrics(a, b):
    assert a.shape == b.shape and a.size and np.isfinite(a).all() and np.isfinite(b).all()
    a,b = a.astype('float64'),b.astype('float64')
    d = a-b
    return {'max_abs':float(abs(d).max()), 'rmse':float(np.sqrt(np.mean(d*d))),
            'relative_rmse':float(np.linalg.norm(d)/np.linalg.norm(b)),
            'cosine':float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)))}


def compare_quantized_logits(reference, native):
    provenance = json.loads((reference/'weights.json').read_text())
    assert provenance['decoder'] == 'upstream gguf.quants.dequantize'
    report = {}
    for case in ['text', 'square', 'wide', 'resize']:
        ref, actual = reference/case, native/case
        expected_ids = json.loads((ref/'greedy.json').read_text())
        actual_ids = json.loads((actual/'greedy.json').read_text())
        assert len(expected_ids) == len(actual_ids) == 8, case
        # Require complete artifacts even when a different greedy prefix limits comparison.
        for root in [ref, actual]:
            assert all((root/f'logits{i}.f32').is_file() for i in range(8)), str(root)
        checks = []
        for step in range(8):
            result = metrics(np.fromfile(actual/f'logits{step}.f32', dtype='<f4'),
                             np.fromfile(ref/f'logits{step}.f32', dtype='<f4'))
            # Native quantized kernels also quantize activations; HF keeps them F32.
            assert result['max_abs'] < 0.5 and result['rmse'] < 0.05 and result['cosine'] > 0.9999, (case, step, result)
            checks.append(result)
            if expected_ids[step] != actual_ids[step]:
                break
        report[case] = {'generated_equal': expected_ids == actual_ids,
                        'same_prefix_logit_steps': len(checks), 'total_steps': 8,
                        'checks': checks}
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('reference', type=Path)
    p.add_argument('native', type=Path)
    p.add_argument('--kv-format', choices=['f32','f16'], default='f32')
    p.add_argument('--quantized-logits', action='store_true',
                   help='Compare against the independent same-weight F32-activation Oracle')
    a = p.parse_args()
    if a.quantized_logits:
        assert a.kv_format == 'f32', 'quantized logit gates require F32 KV'
        print(json.dumps(compare_quantized_logits(a.reference, a.native), indent=2))
        return
    report = {}
    for case in ['text','square','wide','resize']:
        ref,native = a.reference/case,a.native/case
        assert json.loads((ref/'greedy.json').read_text()) == json.loads((native/'greedy.json').read_text()), case+' greedy mismatch'
        results = {}
        for path in sorted(native.glob('*.f32')):
            expected = ref/path.name
            if not expected.exists():
                continue
            x,y = np.fromfile(path,dtype='<f4'),np.fromfile(expected,dtype='<f4')
            result = metrics(x,y)
            results[path.stem] = result
            if path.stem == 'pixels':
                # Resizing uint8 is lossy: Pillow and torchvision Fast use different integer coefficients.
                bound = 2.01/(255*0.26130258) if case == 'resize' else 1e-6
                assert result['max_abs'] < bound, (case,path.stem,result)
            elif a.kv_format == 'f16' and path.stem.startswith('logits'):
                # Cache rounding accumulates across decoder layers and continuation steps.
                assert result['max_abs'] < 0.03 and result['rmse'] < 0.005 and result['relative_rmse'] < 0.005 and result['cosine'] > 0.99999, (case,path.stem,result)
            elif case == 'resize':
                assert result['relative_rmse'] < 0.005 and result['cosine'] > 0.99998, (case,path.stem,result)
                if path.stem.startswith('logits'):
                    assert result['max_abs'] < 0.005 and result['rmse'] < 0.002, (case,path.stem,result)
            else:
                assert result['max_abs'] < 0.005 and result['rmse'] < 0.0005 and result['cosine'] > 0.999999, (case,path.stem,result)
        assert all(f'logits{i}' in results for i in range(8)), case+' missing logits'
        assert all(k in results for k in ['lm0','lm17','lm35','norm']), case+' missing language checkpoints'
        if case != 'text':
            assert all(k in results for k in ['pixels','patch','pre','projected',*[f'vit{i}' for i in range(24)]]), case+' missing vision checkpoints'
        report[case] = results
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
