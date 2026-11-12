"""Multi-step official DiT/APG replay from identical Rust text/noise traces."""
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
    parser.add_argument('--gguf', type=Path, help='independently decode the same stored weights')
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--guidance', type=float, default=15.)
    parser.add_argument('--batch-one-baseline', action='store_true')
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
    predictions = [read(r) for r in records if r['name'] == 'pjit.prediction']
    if not actual:
        raise ValueError('no sampler states found')
    if predictions and len(predictions) != 2 * len(actual):
        raise ValueError('expected conditional and negative prediction for each step')
    text = torch.zeros((2, 256, 7680), device='cuda')
    for index, caption in enumerate(captions):
        text[index, :len(caption)] = torch.from_numpy(caption).to('cuda')
    lengths = torch.tensor([len(c) for c in captions], dtype=torch.int32, device='cuda')
    model = PyramidJiT.from_pretrained(str(args.weights), device='cuda')
    if args.gguf:
        import gguf
        state = model.state_dict()
        reader = gguf.GGUFReader(str(args.gguf))
        if len(reader.tensors) != len(state) or {t.name for t in reader.tensors} != set(state):
            raise ValueError('GGUF must contain the exact released DiT parameter inventory')
        with torch.no_grad():
            for tensor in reader.tensors:
                values = gguf.quants.dequantize(tensor.data, tensor.tensor_type).copy()
                state[tensor.name].copy_(torch.from_numpy(values).reshape(state[tensor.name].shape))
    x = torch.from_numpy(noise).to('cuda')[None, :, None]
    momentum = torch.zeros_like(x)
    rust_x, rust_momentum = x.clone(), momentum.clone()
    baseline_x, baseline_momentum = x.clone(), momentum.clone()
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
            if predictions:
                rust_state = rust_x.to(torch.bfloat16).float()
                different = torch.count_nonzero(rust_state != state).item()
                local_prediction = prediction if not different else model(
                    torch.cat([rust_state, rust_state]), text, lengths,
                    (t * 1000).long().expand(2))
                rust_predictions = [torch.from_numpy(p).to('cuda')[None, :, None]
                                    for p in predictions[2 * index:2 * index + 2]]
                result['same_input_dit'] = [metrics(r[0, :, 0].cpu().numpy(), p[0, :, 0].cpu().numpy())
                                            for r, p in zip(rust_predictions, local_prediction.chunk(2))]
                velocities = [(rust_state - p) / t for p in rust_predictions]
                local_buffer = MomentumBuffer(-.75)
                local_buffer.running_average = rust_momentum.to(torch.bfloat16).float()
                local_guided = adaptive_projected_guidance(*velocities, args.guidance, local_buffer, 0., 10.)
                rust_momentum = local_buffer.running_average
                local_x = rust_x + (next_t - t) * ((rust_x - (rust_state - t * local_guided)) / t)
                result['same_input_sampler'] = metrics(expected_actual, local_x[0, :, 0].cpu().numpy())
                result['bf16_input_different_elements'] = different
                rust_x = torch.from_numpy(expected_actual).to('cuda')[None, :, None]
            if args.batch_one_baseline:
                baseline_state = baseline_x.to(torch.bfloat16).float()
                baseline_prediction = [model(baseline_state, text[i:i + 1], lengths[i:i + 1],
                                             (t * 1000).long().expand(1)) for i in range(2)]
                baseline_velocities = [(baseline_state - p) / t for p in baseline_prediction]
                baseline_buffer = MomentumBuffer(-.75)
                baseline_buffer.running_average = baseline_momentum.to(torch.bfloat16).float()
                baseline_guided = adaptive_projected_guidance(*baseline_velocities, args.guidance,
                                                              baseline_buffer, 0., 10.)
                baseline_momentum = baseline_buffer.running_average
                baseline_x = baseline_x + (next_t - t) * ((baseline_x - (baseline_state - t * baseline_guided)) / t)
                result['oracle_batch_one_vs_batch_two'] = metrics(baseline_x[0, :, 0].cpu().numpy(), x[0, :, 0].cpu().numpy())
            results.append(result)
            args.output.write_text(json.dumps(results, indent=2))
            print(f'Oracle sampling step {index + 1}/{len(actual)}', flush=True)
    from PIL import Image
    pixels = ((x[0, :, 0].clamp(-1., 1.) + 1.) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()
    Image.fromarray(pixels).save(args.output.with_suffix('.png'))
    print(json.dumps(results))


if __name__ == '__main__':
    main()
