"""Compare every selected checkpoint, preserving name/order/shape/occurrence.

Lossless IDs/noise use raw bits; mixed precision math uses explicit relative-L2
bounds. Thresholds are fixed here before running the model, never fitted to output.
"""
import argparse
import json
from pathlib import Path

import numpy as np

if not __debug__:
    raise RuntimeError('FLUX3 verification requires assertions (no -O/PYTHONOPTIMIZE)')


def validate_reference(rows, arguments):
    """Do not let a truncated Oracle trace certify a truncated Rust trace."""
    side = arguments['side']
    assert 64 <= side <= 4096 and side % 16 == 0, 'invalid image size'
    assert 1 <= arguments.get('steps', 2) <= 1000, 'invalid sampling steps'
    assert len(arguments.get('reference', [])) <= 10, 'too many references'
    image_tokens = (side // 16) ** 2
    expected = []
    counts = {}
    def add(name, shape):
        occurrence = counts.get(name, 0)
        counts[name] = occurrence + 1
        expected.append((name, occurrence, shape))
    if arguments.get('vae_only'):
        add('flux3.reference.0', [image_tokens, 128])
        add('flux3.noise', [image_tokens, 128])
    else:
        text_tokens = rows[0]['shape'][0]
        assert 0 < text_tokens <= 4096 and (text_tokens % 80 == 0 or text_tokens == 4096), 'invalid text length'
        add('flux3.token_ids', [text_tokens])
        add('flux3.text', [text_tokens, 20480])
        if arguments.get('text_only'):
            assert [(r['name'],r['occurrence'],r['shape']) for r in rows] == expected
            return
        reference_tokens = 0
        for index, _ in enumerate(arguments.get('reference', [])):
            shape = rows[len(expected)]['shape']
            assert len(shape) == 2 and shape[1] == 128 and shape[0] > 0
            add(f'flux3.reference.{index}', shape)
            reference_tokens += shape[0]
        add('flux3.noise', [image_tokens, 128])
        for step in range(arguments['steps']):
            streams = [('image', 'content_mode_blocks.image', image_tokens)]
            if reference_tokens:
                streams.append(('image_cond', 'content_mode_blocks.image_cond', reference_tokens))
            streams.append(('txt', 'txt_mode_blocks', text_tokens))
            for name, _, tokens in streams: add(f'flux3.{name}.input', [tokens, 3072])
            for layer in range(5):
                for _, prefix, tokens in streams: add(f'{prefix}.{layer}', [tokens, 3072])
            for layer in range(28): add(f'single_blocks.{layer}', [image_tokens+reference_tokens+text_tokens, 3072])
            add('flux3.velocity', [image_tokens, 128])
            add(f'flux3.latent.{step}', [image_tokens, 128])
    add('flux3.vae_input', [32, side//8, side//8])
    add('z_image.vae.rgb_channels', [side, side, 3])
    assert [(r['name'],r['occurrence'],r['shape']) for r in rows] == expected, 'incomplete or invalid Oracle trace'


def compare(reference, actual):
    expected = [json.loads(line) for line in reference.read_text().splitlines()]
    observed = [json.loads(line) for line in actual.read_text().splitlines()]
    assert expected, 'empty Oracle trace'
    provenance = reference.parent / 'run.json'
    dtype = 'f32'
    if provenance.exists():
        arguments = json.loads(provenance.read_text())['arguments']
        dtype = arguments['dtype']
        assert dtype in ('f32', 'bf16'), 'invalid Oracle dtype'
        validate_reference(expected, arguments)
    # Existing VAE debug checkpoints are outside this adapter's canonical trace.
    observed = [r for r in observed if r['name'].startswith(('flux3.','content_mode_blocks.','txt_mode_blocks.','single_blocks.')) or r['name']=='z_image.vae.rgb_channels']
    contract = lambda rows: [(r['name'],r['occurrence'],r['shape']) for r in rows]
    if contract(expected) != contract(observed):
        for index, (a,b) in enumerate(zip(contract(expected),contract(observed))):
            if a != b:
                raise AssertionError(f'first trace contract mismatch at {index}: {a} != {b}')
        raise AssertionError(f'trace length {len(expected)} != {len(observed)}')
    metrics = []
    for a,b in zip(expected,observed):
        x = np.fromfile(a['binary_path'],dtype='<f4')
        y = np.fromfile(b['binary_path'],dtype='<f4')
        assert x.shape==y.shape and len(x)==np.prod(a['shape']), a['name']
        assert np.isfinite(x).all() and np.isfinite(y).all(), a['name']
        name = a['name']
        exact = name in ('flux3.token_ids','flux3.noise')
        delta = y.astype(np.float64)-x.astype(np.float64)
        rel = float(np.linalg.norm(delta)/max(np.linalg.norm(x.astype(np.float64)),1e-30))
        bound = 0.06 if dtype == 'bf16' else (0.03 if name=='z_image.vae.rgb_channels' or name.startswith('flux3.reference.') else 0.005)
        passed = np.array_equal(x.view('<u4'),y.view('<u4')) if exact else rel<=bound
        metric = dict(name=name,occurrence=a['occurrence'],relative_l2=rel,max_absolute=float(np.max(np.abs(delta))),criterion='raw_bits' if exact else f'relative_l2 <= {bound}',passed=bool(passed))
        metrics.append(metric)
    return metrics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference',type=Path)
    parser.add_argument('actual',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if not (args.reference.parent/'run.json').is_file():
        parser.error('the Oracle trace must have its run.json provenance file')
    metrics=compare(args.reference,args.actual)
    args.output.write_text(json.dumps(metrics,indent=2)+'\n')
    failures=[m for m in metrics if not m['passed']]
    if failures:
        raise SystemExit('first failed checkpoint: '+json.dumps(failures[0]))
    print(f'PASS: {len(metrics)} checkpoint invocations')
    for name in ['flux3.text','flux3.velocity','z_image.vae.rgb_channels']:
        selected=[m for m in metrics if m['name']==name]
        if selected:print(name,'max relative-L2',max(m['relative_l2'] for m in selected))


if __name__=='__main__': main()
