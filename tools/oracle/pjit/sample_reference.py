"""Multi-step official F32 DiT/APG replay from identical Rust text/noise traces."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from tools.oracle.pjit.compare import metrics
from tools.oracle.pjit.reference import PIN


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--oracle', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--guidance', type=float, default=15.)
    args = parser.parse_args()
    if subprocess.check_output(['git', '-C', str(args.oracle), 'rev-parse', 'HEAD'], text=True).strip() != PIN:
        raise ValueError('wrong official Oracle revision')
    sys.path.insert(0, str(args.oracle.resolve()))
    from pjit_21k import PyramidJiT, set_attention_backend
    from pjit_21k.sampler import MomentumBuffer, adaptive_projected_guidance
    set_attention_backend('sdpa')
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    records = list(map(json.loads, args.trace.read_text().splitlines()))
    def read(record):
        return np.fromfile(record['binary_path'], dtype='<f4').reshape(record['shape'])
    captions = [read(r) for r in records if r['name'] == 'pjit.text']
    if len(captions) != 2:
        raise ValueError('expected conditional and negative text traces')
    noise = next(read(r) for r in records if r['name'] == 'pjit.initial_noise')
    actual = [read(r) for r in records if r['name'] == 'pjit.euler']
    if not actual:
        raise ValueError('no sampler states found')
    text = torch.zeros((2, 256, 7680), device='cuda')
    for index, caption in enumerate(captions):
        text[index, :len(caption)] = torch.from_numpy(caption).to('cuda')
    lengths = torch.tensor([len(c) for c in captions], dtype=torch.int32, device='cuda')
    model = PyramidJiT.from_pretrained(str(args.weights), device='cuda')
    x = torch.from_numpy(noise).to('cuda')[None, :, None]
    momentum = torch.zeros_like(x)
    times = torch.linspace(1., 0., len(actual) + 1, device='cuda')
    results = []
    with torch.inference_mode():
        for index, expected_actual in enumerate(actual):
            t, next_t = times[index:index + 2]
            state = x.to(torch.bfloat16).float()
            prediction = model(torch.cat([state, state]), text, lengths,
                               (t * 1000).long().expand(2))
            cond, other = ((state - p) / t for p in prediction.chunk(2))
            buffer = MomentumBuffer(-.75)
            buffer.running_average = momentum.to(torch.bfloat16).float()
            guided = adaptive_projected_guidance(cond, other, args.guidance, buffer, 0., 10.)
            momentum = buffer.running_average
            predicted = state - t * guided
            x = x + (next_t - t) * ((x - predicted) / t)
            result = dict(step=index + 1, **metrics(expected_actual, x[0, :, 0].cpu().numpy()))
            result['passed'] = result['nrmse'] <= .02 and result['cosine'] >= .999
            results.append(result)
    args.output.write_text(json.dumps(results, indent=2))
    print(json.dumps(results))


if __name__ == '__main__':
    main()
