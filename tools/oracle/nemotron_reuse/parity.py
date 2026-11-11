#!/usr/bin/env python3
"""Check real CLI traces, the official network on identical features, and FFT drift."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import wave

import numpy as np
import torch

from reference import load_oracle


def metrics(actual, expected, circular=False):
    actual, expected = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    assert actual.shape == expected.shape
    assert np.isfinite(actual).all() and np.isfinite(expected).all()
    diff = actual-expected
    if circular:
        diff = (diff+np.pi) % (2*np.pi)-np.pi
    rms = float(np.sqrt(np.mean(expected*expected)))
    return dict(max_abs=float(np.abs(diff).max()), rmse=float(np.sqrt(np.mean(diff*diff))),
                reference_peak=float(np.abs(expected).max()), reference_rms=rms)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', required=True, type=Path)
    parser.add_argument('--oracle', required=True, type=Path)
    parser.add_argument('--binary', type=Path, default=Path('target/release-fast/rust-model-inference'))
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--case', action='append')
    parser.add_argument('--mode', action='append', choices=['cpu', 'auto', 'scalar'])
    parser.add_argument('--waveform-only', action='store_true', help='measure a normal build without parity-trace checkpoints')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.oracle/'manifest.json').read_text())
    torch.set_num_threads(1)
    model, _, istft = load_oracle(args.model_dir)
    expected_trace = []
    def hook(name):
        def capture(_module, _inputs, output):
            expected_trace.append((name, output[0].permute(1,2,0).contiguous().numpy()))
        return capture
    model.dense_encoder.register_forward_hook(hook('reuse.encoder'))
    for i, block in enumerate(model.TSMamba):
        block.register_forward_hook(hook(f'reuse.block.{i}'))
    records = []
    with torch.inference_mode():
        for case in manifest['cases']:
            name, rate, channels = case['name'], case['sample_rate'], case['channels']
            if args.case and name not in args.case:
                continue
            samples = np.fromfile(args.oracle/case['input'], dtype='<f4')
            input_wav = args.output/f'{name}.wav'
            with wave.open(str(input_wav), 'wb') as stream:
                stream.setparams((channels,2,rate,len(samples)//channels,'NONE','not compressed'))
                stream.writeframes((samples*32768).astype('<i2').tobytes())
            cpu_wav = None
            for mode in args.mode or ['cpu','auto','scalar']:
                trace = args.output/f'{name}.{mode}.jsonl'
                trace.unlink(missing_ok=True)
                output_wav = args.output/f'{name}.{mode}.wav'
                env = os.environ.copy()
                env.pop('RMI_SCALAR', None)
                env['RMI_PARITY_TRACE'] = str(trace)
                env['RMI_COMPUTE_TRACE'] = '1'
                if mode == 'scalar':
                    env['RMI_SCALAR'] = '1'
                command = [str(args.binary.resolve()), '--enhance', '--model', str(args.model_dir),
                           '--audio', str(input_wav), '--out', str(output_wav), '--overwrite',
                           '--threads', '4' if mode != 'scalar' else '1', '--compute', 'auto' if mode == 'auto' else 'cpu']
                start = time.monotonic()
                run = subprocess.run(command, env=env, capture_output=True, text=True, timeout=300)
                (args.output/f'{name}.{mode}.log').write_text(run.stdout+run.stderr)
                assert run.returncode == 0, run.stderr
                assert 'backend=Cpu' in run.stderr and 'backend=Vulkan' not in run.stderr, run.stderr
                if args.waveform_only:
                    assert not trace.exists(), 'waveform-only requires a build without parity-trace'
                rows = [] if args.waveform_only else [json.loads(line) for line in trace.read_text().splitlines()]
                if not args.waveform_only:
                    assert len(rows) == len(case['trace']), (len(rows),len(case['trace']))
                def read(row):
                    return np.fromfile(row['binary_path'],dtype='<f4').reshape(row['shape'])
                checkpoint_results = []
                for offset in range(0,len(rows),36):
                    part, reference = rows[offset:offset+36], case['trace'][offset:offset+36]
                    assert [(v['name'],v['shape']) for v in part] == [(v['name'],v['shape']) for v in reference]
                    mag, phase = read(part[0]), read(part[1])
                    ref_mag = np.fromfile(args.oracle/reference[0]['file'],dtype='<f4').reshape(mag.shape)
                    ref_phase = np.fromfile(args.oracle/reference[1]['file'],dtype='<f4').reshape(phase.shape)
                    frontend = metrics(mag*np.cos(phase), ref_mag*np.cos(ref_phase))
                    frontend_im = metrics(mag*np.sin(phase), ref_mag*np.sin(ref_phase))
                    assert max(frontend['max_abs'],frontend_im['max_abs']) <= 1e-3
                    checkpoint_results.append(dict(name='reuse.stft', real=frontend, imaginary=frontend_im,
                        circular_phase=metrics(phase,ref_phase,True),
                        phase_branch_changes=int(np.count_nonzero(np.abs(phase-ref_phase)>np.pi))))
                    # Phase has a branch cut: feed exactly the native features to the
                    # unmodified official network, isolating its arithmetic from FFT drift.
                    expected_trace.clear()
                    amp, pha, _ = model(torch.from_numpy(mag.T.copy())[None], torch.from_numpy(phase.T.copy())[None])
                    expected_trace.extend([('reuse.magnitude',amp[0].T.contiguous().numpy()),
                                           ('reuse.phase',pha[0].T.contiguous().numpy())])
                    assert len(expected_trace) == len(part[2:35]), 'official checkpoint count differs'
                    for row, (checkpoint, expected) in zip(part[2:35],expected_trace):
                        assert row['name'] == checkpoint
                        result = metrics(read(row),expected,checkpoint=='reuse.phase')
                        if checkpoint != 'reuse.phase':
                            # Low-variance silence amplifies normalization rounding; keep
                            # both an absolute floor and a scale-relative error bound.
                            assert result['max_abs'] <= 5e-4 + 3e-4*result['reference_peak'], (name,checkpoint,result)
                            assert result['rmse'] <= 1e-4 + 1e-4*result['reference_rms'], (name,checkpoint,result)
                        checkpoint_results.append(dict(name=checkpoint,**result))
                    native_amplitude = np.expm1(np.maximum(read(part[33]),0))
                    ref_amplitude = torch.expm1(torch.relu(amp))[0].T.numpy()
                    for component, trig in [('real',np.cos),('imaginary',np.sin)]:
                        result = metrics(native_amplitude*trig(read(part[34])),ref_amplitude*trig(pha[0].T.numpy()))
                        assert result['max_abs'] <= 5e-4 and result['rmse'] <= 1e-4, (name,component,result)
                        checkpoint_results.append(dict(name=f'reuse.spectrum.{component}',**result))
                    amplitude = torch.expm1(torch.relu(amp))
                    amp[:,:,((amplitude==0).sum(1)/amplitude.shape[1]>.5)[0]] = 0
                    even = lambda base: (base*rate//8000)+(base*rate//8000)%2
                    output = istft(amp,pha,even(320),even(40),even(320),'relu_log1p')[0]
                    length = part[35]['shape'][0]
                    output = torch.nn.functional.pad(output[:length],(0,max(0,length-len(output))),value=1e-8)
                    result = metrics(read(part[35]),output.numpy())
                    assert result['max_abs'] <= 5e-4 and result['rmse'] <= 5e-5, (name,result)
                    checkpoint_results.append(dict(name='reuse.waveform',**result))
                with wave.open(str(output_wav),'rb') as stream:
                    assert stream.getframerate() == rate and stream.getnchannels() == channels
                    actual = np.frombuffer(stream.readframes(stream.getnframes()),dtype='<i2').astype(np.float64)/32768
                expected = np.fromfile(args.oracle/case['output'],dtype='<f4').clip(-1,1).reshape(-1)
                assert actual.shape == samples.shape
                end_to_end = metrics(actual,expected)
                if mode == 'cpu':
                    cpu_wav = output_wav.read_bytes()
                if mode == 'auto' and cpu_wav is not None:
                    assert output_wav.read_bytes() == cpu_wav, f'{name}: CPU/Auto differ'
                record = dict(name=name,mode=mode,seconds=time.monotonic()-start,
                              checkpoint_validation=not args.waveform_only,
                              end_to_end=end_to_end,checkpoints=checkpoint_results)
                records.append(record)
                (args.output/'results.json').write_text(json.dumps(records,indent=2)+'\n')
                print(name,mode,f'{record["seconds"]:.2f}s','independent_frontend_waveform',end_to_end,flush=True)
    assert records, 'no matching cases'


if __name__ == '__main__':
    main()
