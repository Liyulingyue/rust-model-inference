#!/usr/bin/env python3
"""Real thinker CLI/HTTP checks; retain commands, thoughts, and device traces."""
import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import urllib.request

from tools.oracle.thx02.verify_entries import comparable, compare


def check(assets, binaries, compute, format_name='F16'):
    output = assets / 'thinking-validation' / (format_name + '-' + compute + '-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    output.mkdir(parents=True)
    common = ['--model', str(assets / f'THX-02-{format_name}.gguf'), '--decision-thinker',
              str(assets / f'THX-02-Thinker-{format_name}.gguf'), '--compute', compute,
              '--threads', '4', '--max-context', '4096']
    env = dict(os.environ, RMI_COMPUTE_TRACE='1')
    records = []
    x = 7
    for _ in range(17):
        x = (13 * x + 19) % 97
    text = {'state': 'A program starts with x=7 and repeats x=(13*x+19) mod 97 exactly 17 times.',
            'questions': {'result': {'type': 'choice', 'instructions': 'What is the final value of x?',
                                    'criteria': {f'v{i}': str((x + i - 7) % 97) for i in range(16)}}},
            'think': True, 'return_thoughts': True}
    image = copy.deepcopy(text)
    image['images'] = json.loads((assets / 'request-image.json').read_text())['images']
    auto = dict(text, think='auto')

    def cli(name, body):
        path = output / (name + '.request.json')
        path.write_text(json.dumps(body, ensure_ascii=False, indent=2))
        command = [str(binaries / 'rust-model-inference'), *common, '--decision-request', str(path)]
        started = time.monotonic()
        result = subprocess.run(command, capture_output=True, text=True, env=env, timeout=1800)
        (output / (name + '.stderr.log')).write_text(result.stderr)
        (output / (name + '.stdout.json')).write_text(result.stdout)
        records.append(dict(entry='CLI', name=name, command=command, exit=result.returncode,
                            seconds=time.monotonic()-started))
        assert result.returncode == 0, (name, result.stderr[-2000:])
        if compute != 'cpu':
            assert 'device_submissions=' in result.stderr and 'device_submissions=0 ' not in result.stderr
        return json.loads(result.stdout)

    baseline = cli('baseline', dict(text, think=False))
    direct = [cli('true-text', text), cli('true-image', image)]
    automatic = cli('auto-text', auto)
    for result in direct:
        assert result['usage']['sequences'] >= 3 and result['usage']['output_tokens'] > 0, result
        assert result['answers']['result'].get('thought'), result
    auto_thought = bool(automatic['answers']['result'].get('thought'))
    max_probability = max(baseline['answers']['result']['probabilities'].values())
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    command = [str(binaries / 'rust-model-server'), *common, '--host', '127.0.0.1', '--port', str(port)]
    with (output / 'server.log').open('w') as log:
        server = subprocess.Popen(command, stdout=log, stderr=log, env=env)
        records.append(dict(entry='HTTP server', command=command))
        try:
            deadline = time.monotonic()+300
            while True:
                assert server.poll() is None, 'server exited before listening'
                try:
                    with socket.create_connection(('127.0.0.1', port), timeout=.2):
                        break
                except OSError:
                    assert time.monotonic() < deadline
                    time.sleep(.1)

            def http(name, route, body):
                started = time.monotonic()
                request = urllib.request.Request(f'http://127.0.0.1:{port}{route}',
                    data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
                with urllib.request.urlopen(request, timeout=1800) as response:
                    assert response.status == 200
                    result = json.loads(response.read())
                (output / (name + '.http.json')).write_text(json.dumps(result, ensure_ascii=False, indent=2))
                records.append(dict(entry='HTTP', name=name, route=route, request=body, status=200,
                                    seconds=time.monotonic()-started))
                return result

            compare(comparable(direct), comparable(http('true-batch', '/v1/decide/batch', {'requests': [text, image]})['results']))
            compare(comparable(automatic), comparable(http('auto-text', '/v1/decide', auto)))
        finally:
            server.terminate()
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
            (output / 'commands.json').write_text(json.dumps(records, ensure_ascii=False, indent=2))
    result = dict(format=format_name, compute=compute, passed=True, evidence=str(output),
                  direct_thinking_text_and_image=True, auto_generated_thought=auto_thought,
                  baseline_max_probability=max_probability, auto_usage=automatic['usage'])
    (assets / 'thinking-validation' / f'summary-{format_name}-{compute}.json').write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--binaries', type=Path, default=Path('target/release-fast'))
    parser.add_argument('--formats', nargs='+', default=['F16'],
                        choices=['F32', 'F16', 'BF16', 'Q8_0', 'Q4_0', 'Q4_K', 'Q5_K', 'Q6_K'])
    parser.add_argument('--computes', nargs='+', default=['cpu', 'auto', 'vulkan'])
    args = parser.parse_args()
    for format_name in args.formats:
        for compute in args.computes:
            print(json.dumps(check(args.model_dir.resolve(), args.binaries.resolve(), compute, format_name)), flush=True)


if __name__ == '__main__':
    main()
