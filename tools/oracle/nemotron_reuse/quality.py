#!/usr/bin/env python3
"""Paired clean/noisy speech quality, independent CPU/Auto output, sweep ablation."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import urllib.request
import wave

import numpy as np
from pesq import pesq
from pystoi import stoi
from scipy.signal import resample_poly
import soundfile as sf
import torch
import torch.nn.functional as F

from reference import HASHES, MAMBA, REVISION, load_oracle, verify

# LibriSpeech clips and attribution files, pinned by librosa 0.11.0's registry.
SOURCES = {
    '5703-47212-0000': ('a284612b46af0535f7e1873758c4387bb8369f6dbbe192ffdec1f171108f98dd',
                      '1c5bd5eb792ff0d2fbe895e3a3dd0acf3beb48ee364e64e12cde3a646304a9f4'),
    '3436-172162-0000': ('780e482b52f2c8a500356babf60178ce005449cc171a8537eb3ef2961eb3e855',
                       '8cf4b20e148c4366205bc607f447c169256e945d62c64ead43d0a4dfa7d3ab98'),
    '198-209-0000': ('0473ba9613befa1026db5e254546f105aeffba8366b5468fc5e8d474d844ca1d',
                    'c5bf58b2a4aa0d6bf1e6051690312f3238960d455776a6d37e469d80db1da167'),
}
DELTA_LIMITS = dict(pesq=0.05, stoi=0.005)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paired_speech(speech, source_rate, rate, snr, seed):
    # Three seconds from [1, 4), including the native two-chunk overlap path.
    assert speech.ndim == 1 and len(speech) >= 4 * source_rate
    speech = resample_poly(speech[source_rate:4 * source_rate], rate, source_rate)
    speech = speech / np.sqrt(np.mean(speech ** 2)) * 0.1
    noise = np.random.default_rng(seed).standard_normal(len(speech))
    noise *= np.sqrt(np.mean(speech ** 2) / np.mean(noise ** 2) / 10 ** (snr / 10))
    gain = min(1.0, 0.9 / max(np.abs(speech).max(), np.abs(speech + noise).max()))
    quantize = lambda x: (x * gain * 32768).astype('<i2').astype('<f4') / 32768
    return quantize(speech), quantize(speech + noise)


def oracle_pair(model, stft, istft, samples, rate):
    size, hop = 2 * rate, rate
    starts = list(range(0, len(samples) - size, hop)) + [len(samples) - size]
    window = torch.hann_window(size).clamp_min(1e-7).numpy()
    output = np.zeros((2, len(samples)), dtype=np.float32)
    norm = np.zeros(len(samples), dtype=np.float32)
    suppressed, frames = 0, 0
    even = lambda base: (base * rate // 8000) + (base * rate // 8000) % 2
    n, step = even(320), even(40)
    for start in starts:
        x = torch.from_numpy(samples[start:start + size].copy())[None]
        mag, phase, _ = stft(x, n, step, n, 'relu_log1p', addeps=False)
        amp, phase, _ = model(mag, phase)
        # Exact published inference.py / inference_chunk.py postprocessing.
        magnitude = torch.expm1(torch.relu(amp))
        mask = ((magnitude == 0).sum(1) / magnitude.shape[1] > 0.5)[0]
        suppressed += int(mask.sum())
        frames += mask.numel()
        filtered = amp.clone()
        filtered[:, :, mask] = 0
        for index, spectrum in enumerate([filtered, amp]):
            audio = istft(spectrum, phase, n, step, n, 'relu_log1p')[0]
            audio = F.pad(audio[:size], (0, max(0, size - len(audio))), value=1e-8)
            output[index, start:start + size] += audio.numpy() * window
        norm[start:start + size] += window
    assert np.all(norm > 0)
    return output / norm, dict(suppressed=suppressed, total=frames)


def scores(clean, actual, rate):
    assert clean.shape == actual.shape and np.isfinite(actual).all()
    return dict(pesq=float(pesq(rate, clean, actual, 'nb' if rate == 8000 else 'wb')),
                stoi=float(stoi(clean, actual, rate, extended=False)))


def waveform_error(actual, expected):
    assert actual.shape == expected.shape and np.isfinite(actual).all()
    rmse = float(np.sqrt(np.mean((actual.astype(np.float64) - expected) ** 2)))
    return dict(max_abs=float(np.max(np.abs(actual - expected))), rmse=rmse,
                relative_rmse=rmse / float(np.sqrt(np.mean(expected.astype(np.float64) ** 2))))


def write_wav(path, samples, rate):
    assert np.isfinite(samples).all() and np.max(np.abs(samples)) < 1
    with wave.open(str(path), 'wb') as stream:
        stream.setparams((1, 2, rate, len(samples), 'NONE', 'not compressed'))
        stream.writeframes((samples * 32768).astype('<i2').tobytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--binary', type=Path, default=Path('target/release-fast/rust-model-inference'))
    parser.add_argument('--case', action='append')
    args = parser.parse_args()
    args.corpus.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(0)
    model, stft, istft = load_oracle(args.model_dir)
    report = dict(model_revision=REVISION, mamba_commit=MAMBA, model_sha256=HASHES['model.safetensors'],
                  binary_sha256=digest(args.binary), script_sha256=digest(Path(__file__)),
                  host=platform.platform(), machine=platform.machine(), python=platform.python_version(),
                  packages={p: importlib.metadata.version(p) for p in
                            ['torch', 'numpy', 'scipy', 'pesq', 'pystoi', 'soundfile']},
                  quality_delta_limits=DELTA_LIMITS, cases=[])
    for speaker, (ogg_hash, license_hash) in SOURCES.items():
        for suffix, sha in [('ogg', ogg_hash), ('txt', license_hash)]:
            path = args.corpus / f'{speaker}.{suffix}'
            if not path.exists():
                with urllib.request.urlopen('https://librosa.org/data/audio/' + path.name, timeout=60) as response:
                    path.write_bytes(response.read())
            verify(path, sha)
        speech, source_rate = sf.read(args.corpus / f'{speaker}.ogg')
        for rate in [8000, 16000]:
            for snr in [0, 10]:
                name = f'{speaker}_{rate}_snr{snr}'
                if args.case and name not in args.case:
                    continue
                clean, noisy = paired_speech(speech, source_rate, rate, snr, 17)
                input_wav = args.output / f'{name}.input.wav'
                clean_wav = args.output / f'{name}.clean.wav'
                write_wav(input_wav, noisy, rate)
                write_wav(clean_wav, clean, rate)
                with torch.inference_mode():
                    start = time.monotonic()
                    oracle, frames = oracle_pair(model, stft, istft, noisy, rate)
                    oracle_seconds = time.monotonic() - start
                assert oracle.shape == (2, len(clean)) and np.isfinite(oracle).all()
                oracle.astype('<f4').tofile(args.output / f'{name}.oracle.f32')
                # Apply the native PCM16 writer to both oracle variants, so quality
                # deltas compare equal output encoding. Raw waveform drift is separate.
                encoded = (oracle.clip(-1, 1) * 32767).astype('<i2').astype('<f4') / 32768
                row = dict(name=name, sample_rate=rate, frames=len(clean), source_sha256=ogg_hash,
                           clean_sha256=digest(clean_wav), input_sha256=digest(input_wav),
                           requested_snr_db=snr, measured_snr_db=float(10 * np.log10(
                               np.mean(clean.astype(np.float64) ** 2) /
                               np.mean((noisy.astype(np.float64) - clean) ** 2))),
                           noisy=scores(clean, noisy, rate), official=scores(clean, encoded[0], rate),
                           without_suppress=scores(clean, encoded[1], rate), suppress_frames=frames,
                           suppress_waveform=waveform_error(oracle[0], oracle[1]),
                           official_seconds=oracle_seconds, native={})
                for mode in ['cpu', 'auto']:
                    target = args.output / f'{name}.{mode}.wav'
                    trace = args.output / f'{name}.{mode}.trace.jsonl'
                    trace.unlink(missing_ok=True)
                    env = os.environ.copy()
                    env.pop('RMI_SCALAR', None)
                    env['RMI_PARITY_TRACE'], env['RMI_COMPUTE_TRACE'] = str(trace.resolve()), '1'
                    start = time.monotonic()
                    run = subprocess.run([str(args.binary.resolve()), '--enhance', '--model', str(args.model_dir),
                                          '--audio', str(input_wav), '--out', str(target), '--overwrite',
                                          '--threads', '4', '--compute', mode], env=env,
                                         capture_output=True, text=True, timeout=600)
                    (args.output / f'{name}.{mode}.log').write_text(run.stdout + run.stderr)
                    assert run.returncode == 0, run.stderr
                    assert not trace.exists(), 'quality requires an ordinary build without parity-trace'
                    assert 'backend=Cpu' in run.stderr and 'backend=Vulkan' not in run.stderr
                    with wave.open(str(target), 'rb') as stream:
                        assert (stream.getnchannels(), stream.getsampwidth(), stream.getframerate()) == (1, 2, rate)
                        actual = np.frombuffer(stream.readframes(stream.getnframes()), dtype='<i2').astype('<f4') / 32768
                    quality = scores(clean, actual, rate)
                    delta = {k: quality[k] - row['official'][k] for k in DELTA_LIMITS}
                    row['native'][mode] = dict(seconds=time.monotonic() - start, quality=quality,
                                              delta_from_official=delta, waveform=waveform_error(actual, oracle[0]),
                                              within_quality_limits=all(abs(delta[k]) <= v for k, v in DELTA_LIMITS.items()))
                assert (args.output / f'{name}.cpu.wav').read_bytes() == (args.output / f'{name}.auto.wav').read_bytes()
                report['cases'].append(row)
                (args.output / 'quality_results.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
                print(name, row['native']['cpu']['delta_from_official'], frames, flush=True)
    assert report['cases'], 'no matching cases'


if __name__ == '__main__':
    main()
