#!/usr/bin/env python3
"""Exercise the real enhancement server, including invalid requests and its slot."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import http.client
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import threading
import time
import wave


def wav(rate=8000, channels=1, frames=520):
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as stream:
        stream.setparams((channels, 2, rate, frames, 'NONE', 'not compressed'))
        stream.writeframes(bytes(frames*channels*2))
    return buffer.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', required=True, type=Path)
    parser.add_argument('--oracle', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--server', type=Path, default=Path('target/release-fast/rust-model-server'))
    parser.add_argument('--binary', type=Path, default=Path('target/release-fast/rust-model-inference'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    input_wav = args.oracle/'speech_44100.wav'
    input_bytes = input_wav.read_bytes()
    env = os.environ.copy()
    env.pop('RMI_SCALAR', None)
    env.pop('RMI_PARITY_TRACE', None)
    results = []
    for mode in ['cpu', 'auto']:
        with socket.socket() as free:
            free.bind(('127.0.0.1', 0))
            port = free.getsockname()[1]

        def request(path='/v1/audio/enhancements', body=input_bytes, media='Audio/WAV; charset=binary'):
            connection = http.client.HTTPConnection('127.0.0.1', port, timeout=180)
            try:
                connection.request('POST', path, body, {'Content-Type': media})
                response = connection.getresponse()
                return response.status, response.getheader('Content-Type'), response.read()
            finally:
                connection.close()

        log = (args.output/f'http.{mode}.log').open('w')
        process = subprocess.Popen([str(args.server.resolve()), '--enhance', '--model', str(args.model_dir),
                                    '--host', '127.0.0.1', '--port', str(port), '--threads', '4', '--compute', mode],
                                   stdout=log, stderr=log, env=env)
        checks = []
        try:
            deadline = time.monotonic()+30
            while True:
                assert process.poll() is None, f'server exited: {args.output}/http.{mode}.log'
                connection = http.client.HTTPConnection('127.0.0.1', port, timeout=1)
                try:
                    connection.request('GET', '/health')
                    response = connection.getresponse()
                    if response.status == 200:
                        response.read()
                        break
                except OSError:
                    pass
                finally:
                    connection.close()
                assert time.monotonic() < deadline, 'server readiness timeout'
                time.sleep(0.1)
            status, media, output = request()
            assert status == 200 and media == 'audio/wav', (status, output)
            cli_output = args.output/f'http.{mode}.cli.wav'
            subprocess.run([str(args.binary.resolve()), '--enhance', '--model', str(args.model_dir),
                            '--audio', str(input_wav), '--out', str(cli_output), '--overwrite',
                            '--threads', '4', '--compute', mode], check=True, capture_output=True, timeout=180, env=env)
            assert output == cli_output.read_bytes(), 'HTTP and CLI WAV differ'
            checks.append('mixed-case MIME; HTTP equals CLI')
            for label, body, media in [
                ('malformed WAV', b'not a WAV', 'audio/wav'),
                ('sample rate', wav(rate=96000, frames=3000), 'audio/wav'),
                ('channels', wav(channels=9), 'audio/wav'),
                ('short audio', wav(frames=160), 'audio/wav'),
                ('media type', input_bytes, 'application/json'),
            ]:
                status, _, error = request(body=body, media=media)
                assert status == 400 and json.loads(error)['error'], (label, status, error)
                checks.append(f'{label}: 400')
            for path in ['/v1/chat/completions', '/v1/jev/score', '/v1/jev/grouped',
                         '/v1/jev/image', '/v1/jev/image_grouped']:
                status, _, body = request(path=path, body=b'{}', media='application/json')
                assert status == 404, (path, status, body)
            checks.append('text/image JEV and chat: 404')
            status, _, body = request(body=bytes(64*1024*1024+1))
            assert status == 413, (status, body)
            checks.append('request cap: 413')
            barrier = threading.Barrier(2)
            def simultaneous(_):
                barrier.wait(timeout=5)
                return request(body=wav(frames=4000))[0]
            with ThreadPoolExecutor(max_workers=2) as workers:
                statuses = sorted(workers.map(simultaneous, range(2)))
            assert statuses == [200, 429], statuses
            checks.append('concurrent requests: 200/429')
            results.append(dict(mode=mode, checks=checks))
            print(mode, checks, flush=True)
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            log.close()
    (args.output/'http-results.json').write_text(json.dumps(results, indent=2)+'\n')


if __name__ == '__main__':
    main()
