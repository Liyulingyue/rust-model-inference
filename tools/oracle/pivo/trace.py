"""Capture unmodified, hash-pinned PiVo ONNX CPU inference for Rust regression checks."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


class Capture:
    def __init__(self, session, name):
        self.session, self.name = session, name
        self.calls, self.records = 0, []

    def run(self, outputs, inputs):
        result = self.session.run(outputs, inputs)
        if self.name in ('encoder', 'decoder') or self.calls in (0, 1, 16, 128, 256, 512, 1024):
            self.records.append(dict(inputs={k: v.tolist() for k, v in inputs.items()},
                                     outputs=[v.tolist() for v in result]))
        self.calls += 1
        return result


class NextFloatDecoder:
    """Sensitivity probe only: leave the graph untouched, perturb its output by one ULP."""
    def __init__(self, session):
        self.session = session

    def run(self, outputs, inputs):
        result = self.session.run(outputs, inputs)
        result[0] = np.nextafter(result[0], np.float32(np.inf))
        return result


def capture(model, request):
    from pivo.text import SPECIAL, phonemize
    front = phonemize(request['input'], overrides=request.get('overrides'))
    for session in model.runtime.sessions.values():
        session.calls, session.records = 0, []
    pcm = model.synthesize(request['input'], voice=request['voice'], speed=request['speed'],
                           overrides=request.get('overrides'))
    sessions = model.runtime.sessions
    encoded, logs = [np.asarray(x, dtype=np.float32) for x in sessions['encoder'].records[0]['outputs']]
    tokens = np.array([front.token_ids])
    durations = np.rint(np.expm1(logs.clip(0, np.log1p(200))) / request['speed']).clip(0, 200).astype(np.int64)
    durations = np.maximum(durations, (tokens >= len(SPECIAL)).astype(np.int64))
    durations[tokens <= 2] = 0
    features = np.concatenate([np.asarray(r['outputs'][0], dtype=np.float32)
                               for r in sessions['decoder'].records], axis=1)
    expanded = np.concatenate([np.asarray(r['inputs']['expanded'], dtype=np.float32)
                               for r in sessions['decoder'].records], axis=1)
    return dict(request=request, normalized=front.normalized_text, tokens=front.token_ids,
                durations=durations.ravel().tolist(), encoded=encoded.ravel().tolist(),
                log_durations=logs.ravel().tolist(), expanded=expanded.ravel().tolist(),
                features=features.ravel().tolist(), pcm=pcm.tolist(),
                periods=model.runtime.periods(features).ravel().tolist(),
                conditioning=sessions['conditioning'].records,
                steps=sessions['vocoder_step'].records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model_dir', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    pins = json.loads(Path(__file__).with_name('sources.json').read_text())
    for row in pins['files']:
        path = args.model_dir / row['Path']
        if path.stat().st_size != row['Size'] or hashlib.sha256(path.read_bytes()).hexdigest() != row['Sha256']:
            raise ValueError(f'Oracle source mismatch: {path}')
    sys.path.insert(0, str(args.model_dir.resolve()))
    from pivo import PiVo
    model = PiVo(args.model_dir, threads=1)
    model.runtime.sessions = {name: Capture(session, name) for name, session in model.runtime.sessions.items()}
    cases = [
        ('NASA is ready for the next mission.', 'Pi', 1.),
        ('Hello, my name is Vo.', 'Vo', 1.),
        ('CPU and GPU.', 'Pi', 0.5),
        ('Dr. Smith paid $2.50 on 2024-02-29.', 'Vo', 2.),
        ('Hello, Zzvo.', 'Vo', 1.),
        ('“Ｈｅｌｌｏ”—world…', 'Pi', 1.25),
        ('a', 'Pi', 2.),
        ('The quick brown fox jumps over the lazy dog.', 'Vo', 0.5),
    ]
    records = []
    for text, voice, speed in cases:
        request = dict(input=text, voice=voice, speed=speed)
        if 'Zzvo' in text:
            request['overrides'] = {'zzvo': ['Z', 'IY1', 'V', 'OW0']}
        record = capture(model, request)
        records.append(record)
        print(voice, speed, len(record['tokens']), len(record['pcm']), text, flush=True)
    probe = PiVo(args.model_dir, threads=1)
    probe.runtime.sessions['decoder'] = NextFloatDecoder(probe.runtime.sessions['decoder'])
    request = records[1]['request']
    perturbed = probe.synthesize(request['input'], voice=request['voice'], speed=request['speed'])
    delta = perturbed.astype(np.float64) - np.asarray(records[1]['pcm'], dtype=np.float64)
    assert delta.shape == (len(records[1]['pcm']),) and np.isfinite(delta).all()
    sensitivity = dict(max_abs=float(np.abs(delta).max()), rmse=float(np.sqrt(np.mean(delta ** 2))))
    records[1]['one_ulp_sensitivity'] = sensitivity
    print('Official decoder output +1 ULP, official vocoder:', sensitivity)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records) + '\n')


if __name__ == '__main__':
    main()
