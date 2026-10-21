"""Compare native Timestral with an unmodified, pinned official CPU/F32 Oracle.

Exact: weight payloads, masks, positions, checkpoint order/shapes/counts.
Numerical: F32 reductions, transcendental functions, attention and forecasts.
"""
import argparse
import gc
import hashlib
import json
import os
import platform
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
import torch

from tools.converter.utils.gguf import open_safetensors, read_gguf_directory, read_gguf_tensor_bytes

ORACLE = 'a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff'
WEIGHTS = '1456effb9927e3daa8c7234f10497e3424233869245a9c72b8ecebaf7ad2b7fa'
ATOL, RTOL, RMS_RTOL = 3e-5, 3e-4, 5e-5
EXACT = {'observed', 'patch_input_mask', 'time_mask', 'variate_mask', 'positions'}


class Trace:
    def __init__(self):
        self.records = []

    def emit(self, name, value):
        a = value.detach().float().cpu().contiguous().numpy().copy()
        self.records.append((f'timestral.{name}', list(a.shape), a))


def instrument(model, forecaster, trace):
    import timestral.layers as layers
    stats, current = {}, {}
    original_scale = model.scale

    def scale(self, values, observed):
        result = original_scale(values, observed)
        stats['loc'], stats['scale'] = result[1:]
        return result

    model.scale = types.MethodType(scale, model)
    original_known = model._known_cov_scaling

    def known(self, *args):
        result = original_known(*args)
        stats['loc'], stats['scale'] = result[1:]
        return result

    model._known_cov_scaling = types.MethodType(known, model)

    def representation_pre(module, args):
        scaled, observed = args
        trace.emit('observed', observed)
        trace.emit('scaled', scaled)
        trace.emit('loc', stats['loc'])
        trace.emit('scale', stats['scale'])
        r, t = scaled.shape
        ps = model.config.patch_size
        patch = torch.cat([scaled.reshape(r, t//ps, ps), (~observed).reshape(r, t//ps, ps).float()], -1)
        trace.emit('patch_input', patch)

    model.representation.register_forward_pre_hook(representation_pre)
    original_run = model._run_two_stream

    def run(self, enc, dec, time_mask, variate_mask, positions):
        x = torch.cat([enc, dec], 1)
        r, s, _ = x.shape
        trace.emit('time_mask', time_mask)
        trace.emit('variate_mask', variate_mask if variate_mask is not None else torch.ones(s, 1, r, r))
        trace.emit('positions', positions)
        trace.emit('embedding', x)
        return original_run(enc, dec, time_mask, variate_mask, positions)

    model._run_two_stream = types.MethodType(run, model)
    original_rotary = layers._apply_rotary

    def emit_qkv(q, k, v, time):
        prefix = current['prefix']
        if not time:
            q, k, v = (a.permute(1, 0, 2, 3) for a in (q, k, v))
        for name, a in [('q', q), ('k', k), ('v', v)]:
            trace.emit(prefix+name, a)

    def rotary(q, k, cos, sin):
        result = original_rotary(q, k, cos, sin)
        emit_qkv(*result, current['v'], True)
        return result

    layers._apply_rotary = rotary
    for i, block in enumerate(model.transformer.layers):
        for j, attn in enumerate(block.attns):
            prefix = f'layer.{i}.attn.{j}.'
            block.attn_norms[j].register_forward_hook(lambda m, a, out, p=prefix: trace.emit(p+'norm', out))
            original_qkv = attn._qkv

            def qkv(self, x, original=original_qkv, p=prefix):
                q, k, v = original(x)
                current.update(prefix=p, v=v)
                if not self.is_time:
                    emit_qkv(q, k, v, False)
                return q, k, v

            attn._qkv = types.MethodType(qkv, attn)
            attn.register_forward_hook(lambda m, a, out, p=prefix: trace.emit(p+'output', out))
        block.register_forward_hook(lambda m, a, out, i=i: trace.emit(f'layer.{i}.output', out))
    model.transformer.out_norm.register_forward_hook(lambda m, a, out: trace.emit('out_norm', out))
    model.head.projection.register_forward_hook(lambda m, a, out: trace.emit('head_raw', out.reshape(out.shape[0], -1, 21)))
    model.head.register_forward_hook(lambda m, a, out: trace.emit('head_monotone', out))
    original_predict = model.predict_block

    def predict(self, *args, **kwargs):
        out = original_predict(*args, **kwargs)
        trace.emit('block_quantiles', out)
        return out

    model.predict_block = types.MethodType(predict, model)
    original_refine = forecaster._refine_block

    def refine(self, *args):
        out = original_refine(*args)
        trace.emit('refined', out)
        return out

    forecaster._refine_block = types.MethodType(refine, forecaster)
    return lambda: setattr(layers, '_apply_rotary', original_rotary)


def cases():
    wave = lambda n, shift=0: [float(np.float32(np.sin(i/7)+i/97+shift)) for i in range(n)]
    short = wave(35, 2)
    short[0:3] = [None]*3
    short[14] = None
    multi = [wave(67), [None]*33+wave(34, 3), [None]*67]
    length, horizon = 47, 65
    return {
        'missing_default': dict(context=short, prediction_length=7),
        'multivariate': dict(context=multi, prediction_length=65, symmetric_averaging=False, clip_nonnegative=False),
        'numeric_covariates': dict(context=wave(length), prediction_length=horizon, symmetric_averaging=False,
            clip_nonnegative=False, past_covariates=[wave(length, 5)],
            known_covariates=[wave(length+horizon, 1), [i%2 for i in range(length+horizon)]]),
        'constant': dict(context=[4.]*32, prediction_length=33),
        'all_missing': dict(context=[None]*33, prediction_length=7),
        'feedback_window': dict(context=[wave(75), wave(75, 1)], prediction_length=65,
            max_context=64, per_shot_horizon_patches=1, symmetric_averaging=False, clip_nonnegative=False),
        'covariate_feedback': dict(context=wave(47), prediction_length=35, past_covariates=[wave(47, 2)],
            known_covariates=[wave(82, 4)], per_shot_horizon_patches=1, symmetric_averaging=False),
        'long_horizon': dict(context=wave(32), prediction_length=1025, symmetric_averaging=False, clip_nonnegative=False),
        'explicit_mask': dict(context=[wave(41)], observation_mask=[[i%9!=0 for i in range(41)]],
            prediction_length=9, symmetric_averaging=False),
        'clipping_window': dict(context=[-1.]+[0.]*64, prediction_length=7, max_context=32),
    }


def compare(records, native_path):
    native = [json.loads(line) for line in native_path.read_text().splitlines()]
    assert len(native) == len(records), (len(native), len(records))
    metrics = []
    occurrences = {}
    for (name, shape, expected), actual in zip(records, native):
        assert (actual['name'], actual['shape']) == (name, shape), (actual['name'], name, actual['shape'], shape)
        assert actual['occurrence'] == occurrences.get(name, 0), name
        occurrences[name] = actual['occurrence'] + 1
        got = np.fromfile(actual['binary_path'], dtype='<f4').reshape(shape)
        assert np.isfinite(got).all() and np.isfinite(expected).all(), name
        short = name.removeprefix('timestral.')
        if short in EXACT:
            assert np.array_equal(got.view('u4'), expected.view('u4')), f'{name}: exact mismatch'
        if short == 'patch_input':
            assert np.array_equal(got[..., 32:].view('u4'), expected[..., 32:].view('u4')), 'missing channels differ'
        delta = got.astype('f8')-expected.astype('f8')
        max_abs = float(np.max(np.abs(delta), initial=0))
        amplitude = float(np.max(np.abs(expected), initial=0))
        rms = float(np.sqrt(np.mean(expected.astype('f8')**2)))
        rmse = float(np.sqrt(np.mean(delta**2)))
        nrmse = rmse / max(rms, 1e-6)
        # Absolute allowance applies to RMS too: relative error alone is unstable near zero.
        assert max_abs <= ATOL+RTOL*amplitude and rmse <= ATOL+RMS_RTOL*rms, (name, max_abs, amplitude, nrmse)
        metrics.append(dict(name=name, max_abs=max_abs, rmse=rmse, nrmse=nrmse))
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--oracle', required=True, type=Path)
    parser.add_argument('--model-dir', required=True, type=Path)
    parser.add_argument('--gguf', required=True, type=Path)
    parser.add_argument('--binary', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--case', action='append', choices=list(cases()))
    args = parser.parse_args()
    assert subprocess.check_output(['git', '-C', str(args.oracle), 'rev-parse', 'HEAD'], text=True).strip() == ORACLE
    assert not subprocess.check_output(['git', '-C', str(args.oracle), 'status', '--porcelain', '--untracked-files=no'], text=True).strip()
    sys.path.insert(0, str(args.oracle.resolve()/'src'))
    from timestral import TimestralForecaster
    from timestral.loading import load_model
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    with (args.model_dir/'model.safetensors').open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == WEIGHTS
    source = open_safetensors(args.model_dir/'model.safetensors')
    metadata, tensors = read_gguf_directory(args.gguf)
    assert metadata['general.architecture'] == 'timestral'
    assert set(tensors) == set(source.header)-{'__metadata__'}
    for name, (kind, dims, size) in tensors.items():
        assert kind == 0 and dims == tuple(reversed(source.header[name]['shape']))
        assert read_gguf_tensor_bytes(args.gguf, name) == source.get(name).raw, name
    model = load_model(args.model_dir, device='cpu')
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report = dict(oracle=ORACLE, source_sha256=WEIGHTS, tensor_count=len(tensors),
        platform=platform.platform(), binary_sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest(),
        torch=torch.__version__, atol=ATOL, rtol=RTOL, rmse_rtol=RMS_RTOL, cases={})
    for name, request in cases().items():
        if args.case and name not in args.case:
            continue
        trace = Trace()
        forecaster = TimestralForecaster(model, max_context=request.get('max_context'),
            per_shot_horizon_patches=request.get('per_shot_horizon_patches', 32))
        restore = instrument(model, forecaster, trace)
        kwargs = {k: np.array(v, dtype=bool if k=='observation_mask' else 'f4') for k,v in request.items()
                  if k in ('observation_mask', 'past_covariates', 'known_covariates')}
        for k in ('past_covariates', 'known_covariates'):
            if k in kwargs:
                kwargs[k] = kwargs[k][None]
        context = np.array(request['context'], dtype='f4')
        if context.ndim == 2:
            context = context[None]  # One series, jointly forecast variates; official 2D means a batch.
        with torch.inference_mode():
            expected = forecaster.forecast(context,
                request['prediction_length'], symmetric_averaging=request.get('symmetric_averaging', True),
                clip_nonnegative=request.get('clip_nonnegative', True), **kwargs).quantiles[0]
        trace.emit('forecast', expected)
        restore()
        # Remove hooks before the next request, without modifying the Oracle's source.
        for module in model.modules():
            module._forward_hooks.clear()
            module._forward_pre_hooks.clear()
        # Wrappers capture original methods; reload so later cases cannot nest wrappers.
        del forecaster
        del module
        del model
        gc.collect()
        model = load_model(args.model_dir, device='cpu')
        request_path = output_dir/f'{name}.json'
        request_path.write_text(json.dumps(request))
        report['cases'][name] = {}
        for mode in ('scalar', 'simd'):
            trace_path = output_dir/f'{name}.{mode}.jsonl'
            trace_path.unlink(missing_ok=True)
            env = dict(os.environ, RMI_PARITY_TRACE=str(trace_path))
            env.pop('RMI_PARITY_FILTER', None)
            env.pop('RMI_SCALAR', None)
            if mode=='scalar':
                env['RMI_SCALAR']='1'
            run = subprocess.run([str(args.binary.resolve()), '--model', str(args.gguf.resolve()),
                '--timestral-request', str(request_path), '--threads', '1' if mode=='scalar' else '4'],
                env=env, text=True, capture_output=True, check=True)
            (output_dir/f'{name}.{mode}.stdout').write_text(run.stdout)
            (output_dir/f'{name}.{mode}.stderr').write_text(run.stderr)
            actual = json.loads(run.stdout.strip().splitlines()[-1])
            got = np.asarray(actual['quantiles'], dtype='f4')
            assert got.shape == tuple(expected.shape)
            assert np.all(got[..., 1:] >= got[..., :-1]) and np.isfinite(got).all()
            metrics = compare(trace.records, trace_path)
            last = json.loads(trace_path.read_text().splitlines()[-1])
            assert np.array_equal(got.ravel().view('u4'), np.fromfile(last['binary_path'], dtype='<u4'))
            assert np.array_equal(np.array(actual['quantile_levels'], dtype='f4'), np.array(model.config.quantile_knots, dtype='f4'))
            worst = max(metrics, key=lambda m:m['nrmse'])
            result = dict(checkpoints=len(metrics), worst_nrmse=worst,
                forecast=metrics[-1], max_abs=max(m['max_abs'] for m in metrics))
            report['cases'][name][mode]=result
            print(name, mode, json.dumps(result), flush=True)
        (output_dir/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(output_dir/'report.json')


if __name__=='__main__':
    main()
