"""Exercise native PiVo CLI/HTTP against pinned fixtures; requires built binaries."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
import wave

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path)
    parser.add_argument('fixtures', type=Path)
    parser.add_argument('--bin-dir', type=Path, default=Path('target/release-fast'))
    args = parser.parse_args()
    cases = json.loads(args.fixtures.read_text())
    cli = str(args.bin_dir.resolve() / 'rust-model-inference')
    server = str(args.bin_dir.resolve() / 'rust-model-server')
    env = {**os.environ, 'RMI_COMPUTE_TRACE': '1'}
    common = ['--tts', '--model', str(args.model.resolve()), '--threads', '2']

    with tempfile.TemporaryDirectory(prefix='pivo-check-') as directory:
        root = Path(directory)

        def run(request, compute='cpu'):
            output = root / 'speech.wav'
            command = [cli, *common, '--prompt', request['input'], '--voice', request['voice'],
                       '--speed', str(request['speed']), '--compute', compute, '--out', str(output),
                       '--pronunciations', json.dumps(request.get('overrides', {}))]
            result = subprocess.run(command, env=env, capture_output=True, text=True, check=True)
            assert 'scope=pivo backend=Cpu' in result.stderr, result.stderr
            return output.read_bytes()

        wavs = []
        for case in cases:
            data = run(case['request'])
            with wave.open(io.BytesIO(data)) as wav:
                assert (wav.getnchannels(), wav.getframerate(), wav.getsampwidth()) == (1, 16000, 2)
                pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype='<i2') / 32767.
            expected = np.clip(case['pcm'], -1., 1.)
            assert pcm.shape == expected.shape
            delta = pcm - expected
            assert np.max(np.abs(delta)) <= 5.1e-3 and np.sqrt(np.mean(delta ** 2)) <= 7.6e-4
            assert data == run(case['request'], 'auto'), 'CPU/Auto WAV differs'
            wavs.append(data)
        print(f'CLI: {len(cases)} oracle cases; CPU/Auto byte-identical WAV', flush=True)

        for flags in [
            ['--voice', 'Other'], ['--speed', 'NaN'], ['--speed', '2.1'], ['--language', 'Chinese'],
            ['--mmproj', 'unused.gguf'], ['--temp', '0.1'], ['--max-tokens', '10'], ['--seed', '7'],
            ['--repetition-penalty', '1.1'], ['--kv-cache', 'f16'], ['--jinja'],
            ['--resolution', '48000'], ['--size', '48000'], ['--duration', '10'], ['--dry-run'],
            ['--tts-model', 'unused.gguf'], ['--pronunciations', '{invalid'], ['--jev'],
        ]:
            result = subprocess.run([cli, *common, '--prompt', 'hello', '--out', str(root / 'bad.wav'),
                                     *flags], env=env, capture_output=True)
            assert result.returncode != 0, flags
            assert not (root / 'bad.wav').exists(), flags
        result = subprocess.run([cli, '--model', str(args.model), '--prompt', 'hello'], capture_output=True)
        assert result.returncode != 0 and b'pivo' in result.stderr.lower(), result.stderr
        result = subprocess.run([cli, *common, '--prompt', 'hello', '--out', str(root / 'bad.wav'),
                                 '--compute', 'vulkan'], capture_output=True)
        assert result.returncode != 0 and b'vulkan' in result.stderr.lower(), result.stderr
        print('CLI: incompatible controls, task routes and strict Vulkan rejected', flush=True)

        with socket.socket() as probe:
            probe.bind(('127.0.0.1', 0))
            port = probe.getsockname()[1]
        address = f'http://127.0.0.1:{port}'
        override = {'zzvo': ['Z', 'IY1', 'V', 'OW0']}
        default = dict(input='Hello, Zzvo.', voice='Vo', speed=1.5, overrides=override)
        default_wav = run(default)

        def http(path, body=None):
            request = urllib.request.Request(address + path,
                data=None if body is None else json.dumps(body).encode(),
                headers={'Content-Type': 'application/json'})
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    return response.status, response.read(), response.headers.get_content_type()
            except urllib.error.HTTPError as error:
                return error.code, error.read(), error.headers.get_content_type()

        with (root / 'server.log').open('w+') as log:
            process = subprocess.Popen([server, *common, '--compute', 'auto', '--host', '127.0.0.1',
                '--port', str(port), '--voice', 'Vo', '--speed', '1.5', '--pronunciations', json.dumps(override)],
                env=env, stdout=log, stderr=log)
            try:
                deadline = time.monotonic() + 30
                while True:
                    try:
                        status, body, _ = http('/v1/models')
                        assert status == 200
                        model_id = json.loads(body)['data'][0]['id']
                        break
                    except (urllib.error.URLError, ConnectionError):
                        if process.poll() is not None or time.monotonic() >= deadline:
                            log.seek(0)
                            raise RuntimeError(log.read())
                        time.sleep(0.1)
                for _ in range(2):
                    status, data, content_type = http('/v1/audio/speech', {'input': default['input']})
                    assert (status, content_type, data) == (200, 'audio/wav', default_wav)
                request = {**cases[1]['request'], 'overrides': {}, 'model': model_id}
                status, data, content_type = http('/v1/audio/speech', request)
                assert (status, content_type, data) == (200, 'audio/wav', wavs[1])
                status, data, content_type = http('/v1/audio/speech', {**request, 'response_format': 'pcm'})
                assert (status, content_type, data) == (200, 'audio/pcm', wavs[1][44:])
                for patch, expected in [
                    ({'voice': 'Other'}, 400), ({'speed': 0.1}, 400), ({'model': 'wrong'}, 400),
                    ({'response_format': 'mp3'}, 400), ({'input': 'zzunknownword'}, 400),
                    ({'input': 'hello' * 401}, 400), ({'input': '?!'}, 400),
                    ({'overrides': {'hello': ['UNKNOWN']}, 'input': 'hello'}, 400),
                    ({'overrides': {'hello': ['B'], 'HELLO': ['HH']}}, 400),
                    ({'extra': True}, 422),
                ]:
                    status, data, _ = http('/v1/audio/speech', {**request, **patch})
                    assert status == expected, (patch, status, data)
                for path, payload in [
                    ('/v1/chat/completions', {}), ('/v1/embeddings', {}),
                    ('/v1/jev', {'context': 'hello'}), ('/v1/jev', {'image': 'unused.png'}),
                ]:
                    assert http(path, payload)[0] == 404, path
                with ThreadPoolExecutor(max_workers=4) as pool:
                    statuses = list(pool.map(lambda _: http('/v1/audio/speech',
                        {**request, 'input': 'hello ' * 30})[0], range(4)))
                assert 200 in statuses and 429 in statuses and set(statuses) <= {200, 429}, statuses
                assert http('/v1/audio/speech', request)[1] == wavs[1], 'state leaked after concurrency'
                print('HTTP: defaults/overrides, PCM/WAV, reset, 400/404/422/429 passed', statuses)
            finally:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


if __name__ == '__main__':
    main()
