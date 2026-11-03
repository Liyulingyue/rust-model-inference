#!/usr/bin/env python3
"""Run unmodified, pinned SEMamba with Mamba's official CPU selective_scan_ref."""
import argparse
import ast
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import types
import urllib.request
import wave

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from safetensors.torch import load_file

REVISION = 'd66af5dfc20db53a49f885eb32e7e7f1fcb70af2'
MAMBA = 'd7b1ceb3c367ec9022925e812f507bcf706937c6'
HASHES = {
    'config.json': 'e69986483774f01893d2727503addc49ef201d19e6e1a266b8d4e971e28c478d',
    'model.safetensors': '06badaf4029b297f97f7fe48d62990b48d5482608210df577354e0d4e0ed70e1',
    'models/generator_SEMamba_time_d4.py': 'beebaf2d9ab7b83c101f46246ae7a7a8a2d2f57b6683025d3712c64c1b7d6817',
    'models/codec_module_time_d4.py': '02bd1d478f36a9b38dce7d44368805298f42798ea239304580f1ccffafe766a0',
    'models/mamba_block2_SEMamba.py': '97397af2d7e0f5e6530aa8c4a45b4f0f0030b4b56d8c9922a9b0bc25f8b5962e',
    'models/stfts.py': 'b82383c1ec7d7e9fe9bca1f93b3c0afe9609e191c38efc16f705e7145198d835',
    'noisy_audio/mic_test2.wav': 'e6433420a00163f30a339fc83a191517a82597cf2141cbd310233470e83ce0d7',
}
MAMBA_FILES = {
    'modules/mamba_simple.py': 'a17e4c51b582dc0d4d690a649eba521cd0c1ee3dc8f0473a0967cdc9ec0874e3',
    'ops/selective_scan_interface.py': '543edd077f11c80cccd5c5e2f926420d9105fd238d9e8960e53660cfa234dce8',
}


def verify(path, expected):
    with path.open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    if actual != expected:
        raise ValueError(f'{path}: SHA256 {actual} != {expected}')


def load_oracle(root):
    for name, sha in HASHES.items():
        verify(root / name, sha)
    reference = root / 'mamba-reference'
    reference.mkdir(exist_ok=True)
    for name, sha in MAMBA_FILES.items():
        target = reference / Path(name).name
        if not target.exists():
            url = f'https://api.github.com/repos/state-spaces/mamba/contents/mamba_ssm/{name}?ref={MAMBA}'
            req = urllib.request.Request(url, headers={'User-Agent': 'rmi-oracle'})
            with urllib.request.urlopen(req, timeout=60) as response:
                target.write_bytes(base64.b64decode(json.load(response)['content']))
        verify(target, sha)

    # Compile the official function verbatim, avoiding imports of CUDA extensions.
    scan_path = reference / 'selective_scan_interface.py'
    tree = ast.parse(scan_path.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'selective_scan_ref')
    namespace = dict(torch=torch, F=F, rearrange=rearrange, repeat=repeat)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(scan_path), 'exec'), namespace)
    scan = types.ModuleType('mamba_ssm.ops.selective_scan_interface')
    scan.selective_scan_fn = namespace['selective_scan_ref']
    def no_fused(*args, **kwargs):
        raise AssertionError('CPU oracle must not enter a fused CUDA kernel')
    scan.mamba_inner_fn = no_fused
    for name in ['mamba_ssm', 'mamba_ssm.ops']:
        package = types.ModuleType(name)
        package.__path__ = []
        sys.modules[name] = package
    sys.modules[scan.__name__] = scan
    spec = importlib.util.spec_from_file_location('official_mamba', reference / 'mamba_simple.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.causal_conv1d_fn is None
    sys.modules['mamba_ssm'].Mamba = module.Mamba
    sys.path.insert(0, str(root.resolve()))
    from models.generator_SEMamba_time_d4 import SEMamba
    from models.stfts import mag_phase_stft, mag_phase_istft
    model = SEMamba(json.loads((root / 'config.json').read_text())).eval()
    model.load_state_dict(load_file(root / 'model.safetensors'), strict=True)
    return model, mag_phase_stft, mag_phase_istft


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--case', action='append', help='case names; default: all')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(0)
    model, stft, istft = load_oracle(args.model_dir)
    args.output.mkdir(parents=True, exist_ok=True)
    current_trace = []

    def save_trace(name, array):
        array = np.asarray(array, dtype='<f4')
        index = len(current_trace)
        path = f'{case_name}.trace.{index}.f32'
        array.tofile(args.output / path)
        current_trace.append(dict(name=name, shape=list(array.shape), file=path))

    def hook(name):
        def record(_module, _input, output):
            save_trace(name, output[0].permute(1, 2, 0).contiguous().numpy())
        return record
    model.dense_encoder.register_forward_hook(hook('reuse.encoder'))
    for i, block in enumerate(model.TSMamba):
        block.register_forward_hook(hook(f'reuse.block.{i}'))

    def chunk(samples, rate):
        even = lambda base: (base * rate // 8000) + (base * rate // 8000) % 2
        n, hop = even(320), even(40)
        x = torch.from_numpy(samples.copy())[None]
        mag, phase, _ = stft(x, n, hop, n, 'relu_log1p', addeps=False)
        save_trace('reuse.stft.mag', mag[0].T.contiguous().numpy())
        save_trace('reuse.stft.phase', phase[0].T.contiguous().numpy())
        amp, phase, _ = model(mag, phase)
        save_trace('reuse.magnitude', amp[0].T.contiguous().numpy())
        save_trace('reuse.phase', phase[0].T.contiguous().numpy())
        magnitude = torch.expm1(torch.relu(amp))
        amp[:, :, ((magnitude == 0).sum(1) / magnitude.shape[1] > 0.5)[0]] = 0
        output = istft(amp, phase, n, hop, n, 'relu_log1p')[0]
        output = F.pad(output[:len(samples)], (0, max(0, len(samples)-len(output))), value=1e-8)
        save_trace('reuse.waveform', output.numpy())
        return output.numpy()

    def enhance(samples, rate):
        size = 2 * rate
        if len(samples) <= size:
            return chunk(samples, rate)
        hop = size // 2
        count = (len(samples)-size+hop-1)//hop+1
        window = torch.hann_window(size).clamp_min(1e-7).numpy()
        output, norm = np.zeros_like(samples), np.zeros_like(samples)
        for i in range(count):
            start = len(samples)-size if i+1 == count else i*hop
            output[start:start+size] += chunk(samples[start:start+size], rate)*window
            norm[start:start+size] += window
        return output/norm

    rng = np.random.default_rng(17)
    cases = []
    for rate in [8000, 16000, 22050, 24000, 32000, 44100, 48000]:
        t = np.arange(int(rate*0.065)+7)/rate
        samples = (0.15*np.sin(2*np.pi*317*t)+0.04*np.sin(2*np.pi*1379*t)+rng.normal(0, 0.02, len(t))).astype('<f4')
        cases.append((f'noisy_{rate}', rate, samples[:, None]))
    cases.append(('silence_8000', 8000, np.zeros((647, 1), dtype='<f4')))
    with wave.open(str(args.model_dir / 'noisy_audio/mic_test2.wav')) as stream:
        rate = stream.getframerate()
        speech = np.frombuffer(stream.readframes(stream.getnframes()), dtype='<i2').astype(np.float32)/32768
    cases.append(('speech_44100', rate, speech[11025:11025+4417, None]))
    cases.append(('stereo_16000', 16000, np.stack([cases[1][2][:, 0], cases[1][2][::-1, 0]], axis=1)))
    t = np.arange(16813)/8000
    cases.append(('chunked_8000', 8000, (0.15*np.sin(2*np.pi*317*t)+rng.normal(0,0.02,len(t))).astype('<f4')[:,None]))
    manifest = dict(model_revision=REVISION, mamba_commit=MAMBA, torch=torch.__version__, numpy=np.__version__, cases=[])
    with torch.inference_mode():
        for case_name, rate, samples in cases:
            if args.case and case_name not in args.case:
                continue
            samples = (samples.clip(-1, 1)*32768).astype('<i2').astype('<f4')/32768
            current_trace.clear()
            start = time.monotonic()
            output = np.stack([enhance(samples[:, c], rate) for c in range(samples.shape[1])], axis=1)
            assert output.shape == samples.shape and np.isfinite(output).all()
            samples.tofile(args.output / f'{case_name}.input.f32')
            output.astype('<f4').tofile(args.output / f'{case_name}.output.f32')
            manifest['cases'].append(dict(name=case_name, sample_rate=rate, channels=samples.shape[1],
                                         input=f'{case_name}.input.f32', output=f'{case_name}.output.f32',
                                         trace=list(current_trace)))
            print(case_name, f'{time.monotonic()-start:.2f}s', flush=True)
            (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    assert manifest['cases'], 'no matching cases'


if __name__ == '__main__':
    main()
