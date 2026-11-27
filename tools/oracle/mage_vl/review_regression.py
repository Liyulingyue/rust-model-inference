"""Real-weight regressions for JEV context limits and expanded HTTP usage."""
import argparse
import base64
from contextlib import contextmanager
import json
from pathlib import Path
import re
import subprocess
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('model')
    p.add_argument('vision')
    p.add_argument('square', type=Path, help='Pinned reference/square directory')
    p.add_argument('out', type=Path)
    p.add_argument('--compute', choices=['cpu', 'auto', 'vulkan'], default='cpu')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    base = ['target/release-fast/rust-model-inference', '--model', a.model,
            '--threads', '8', '--prefill-batch-size', '32', '--compute', a.compute]
    url = 'http://127.0.0.1:18084'
    report = {}

    def check(name, actual, expected):
        report[name] = {'actual': actual, 'expected': expected, 'pass': actual == expected}
        print(name, report[name], flush=True)

    def check_probabilities(name, actual, expected):
        def values(result):
            if 'groups' in result:
                return [group['probabilities'][k] for group in result['groups']
                        for k in sorted(group['probabilities'])]
            return [result['probabilities'][k] for k in sorted(result['probabilities'])]
        left, right = values(actual), values(expected)
        assert len(left) == len(right)
        error = max(abs(x - y) for x, y in zip(left, right))
        report[name] = {'max_abs': error, 'limit': 1e-6, 'pass': error < 1e-6}
        print(name, report[name], flush=True)

    def cli(args, cap):
        return subprocess.run(base + ['--max-context', str(cap)] + args,
                              capture_output=True, text=True, timeout=900)

    @contextmanager
    def server(cap):
        with (a.out / f'server-{cap}.log').open('w') as log:
            process = subprocess.Popen(base + ['--max-context', str(cap), '--mmproj', a.vision,
                                       '--serve', '--host', '127.0.0.1', '--port', '18084'],
                                       stdout=log, stderr=log)
            try:
                for _ in range(120):
                    assert process.poll() is None, 'server exited'
                    try:
                        with urlopen(url + '/health', timeout=1):
                            break
                    except (URLError, TimeoutError):
                        time.sleep(0.5)
                else:
                    raise RuntimeError('server startup timeout')
                yield
            finally:
                process.terminate()
                process.wait(timeout=30)

    def post(path, body):
        request = Request(url + path, data=json.dumps(body).encode(),
                          headers={'Content-Type': 'application/json'})
        try:
            with urlopen(request, timeout=900) as response:
                if body.get('stream'):
                    events = [json.loads(line[6:]) for line in response.read().decode().splitlines()
                              if line.startswith('data: ') and line[6:] != '[DONE]']
                    return response.status, events
                return response.status, json.load(response)
        except HTTPError as error:
            return error.code, json.load(error)

    jev = ['--jev', '--jev-context', 'Colors.', '--jev-question', 'Are there colors?',
           '--jev-option', 'yes', '--jev-option', 'no', '--jev-output', 'json']
    questions = [{'text': 'Are there colors?', 'options': ['yes', 'no']}]
    grouped = [{'text': 'Are there colors?', 'groups': [{'label': 'color', 'options': ['yes', 'no']}]}]
    normal_results = {}
    for mode, args in [('single', jev), ('grouped', jev + ['--jev-multi']),
                       ('block', jev[:5] + ['--jev-block', 'color'] + jev[5:])]:
        rejected = cli(args, 2)
        (a.out / f'cli-{mode}-limit.log').write_text(rejected.stdout + rejected.stderr)
        check(f'cli-{mode}-over-limit', rejected.returncode != 0, True)
        # A valid limit must leave scoring unchanged, including the exact boundary.
        match = re.search(r'Session capacity (\d+) exceeds model context 2', rejected.stderr)
        check(f'cli-{mode}-limit-diagnostic', match is not None, True)
        mismatch = cli(args + ['--mmproj', a.model], 2048)
        check(f'cli-{mode}-mismatched-projector', mismatch.returncode != 0, True)
        if match:
            length = int(match[1])
            normal, boundary, below = cli(args, 2048), cli(args, length), cli(args, length - 1)
            check(f'cli-{mode}-exact-boundary', boundary.returncode, 0)
            check(f'cli-{mode}-below-boundary', below.returncode != 0, True)
            if normal.returncode == boundary.returncode == 0:
                normal_results[mode] = json.loads(normal.stdout)
                check_probabilities(f'cli-{mode}-probabilities', json.loads(boundary.stdout),
                                    normal_results[mode])

    (a.out / 'jev-normal.json').write_text(json.dumps(normal_results, indent=2))
    with server(2):
        for path, items in [('/v1/jev/score', questions), ('/v1/jev/grouped', grouped)]:
            status, value = post(path, {'context': 'Colors.', 'questions': items})
            check(path + '-over-limit', status, 400)
            (a.out / (path.rsplit('/', 1)[-1] + '-limit.json')).write_text(json.dumps(value))

    case = json.loads((a.square / 'case.json').read_text())
    # Oracle fixture has one placeholder expanded to four visual tokens: 23 -> 26.
    assert len(case['raw_ids']) == 23 and len(case['input_ids']) == 26
    data = base64.b64encode((a.square / 'image.png').read_bytes()).decode()
    text = 'Describe the colors in this image briefly.'
    content = {
        '/v1/chat/completions': [{'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + data}},
                                 {'type': 'text', 'text': text}],
        '/v1/messages': [{'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': data}},
                         {'type': 'text', 'text': text}],
        '/v1/responses': [{'type': 'input_image', 'image_url': 'data:image/png;base64,' + data},
                          {'type': 'input_text', 'text': text}],
    }
    with server(2048):
        for mode, path, items in [('single', '/v1/jev/score', questions),
                                  ('grouped', '/v1/jev/grouped', grouped),
                                  ('block', '/v1/jev/grouped', grouped)]:
            body = {'context': 'Colors.', 'questions': items}
            if mode != 'single':
                body['mode'] = 'block_choice' if mode == 'block' else 'multi_select'
                body['questions'] = [{'text': 'Are there colors?', 'groups': [
                    {'label': 'color' if mode == 'block' else 'pair_1', 'options': ['yes', 'no']}]}]
            status, value = post(path, body)
            check(f'http-{mode}-normal-status', status, 200)
            check_probabilities(f'http-{mode}-probabilities', value['results'][0], normal_results[mode])
        for path, parts in content.items():
            body = {'messages': [{'role': 'user', 'content': parts}], 'max_tokens': 1,
                    'temperature': 0, 'enable_thinking': False}
            if path.endswith('responses'):
                body['input'] = body.pop('messages')
                body['max_output_tokens'] = body.pop('max_tokens')
                body['store'] = False
            for stream in [False, True]:
                body['stream'] = stream
                if path.endswith('completions'):
                    body['stream_options'] = {'include_usage': True}
                status, value = post(path, body)
                check(path + f'-stream-{stream}-status', status, 200)
                (a.out / (path.rsplit('/', 1)[-1] + f'-{stream}.json')).write_text(json.dumps(value))
                if stream:
                    if path.endswith('completions'):
                        count = next(e['usage']['prompt_tokens'] for e in value if e.get('usage'))
                    elif path.endswith('messages'):
                        count = next(e['message']['usage']['input_tokens'] for e in value if e['type'] == 'message_start')
                    else:
                        count = next(e['response']['usage']['input_tokens'] for e in value if e['type'] in ('response.completed', 'response.incomplete'))
                else:
                    count = value['usage']['prompt_tokens' if path.endswith('completions') else 'input_tokens']
                check(path + f'-stream-{stream}-input', count, 26)
            if path.endswith('messages'):
                status, value = post('/v1/messages/count_tokens', dict(body, stream=False))
                check('count-tokens-status', status, 200)
                check('count-tokens-input', value.get('input_tokens'), 26)
    (a.out / 'report.json').write_text(json.dumps(report, indent=2))
    assert all(r['pass'] for r in report.values()), 'review regressions failed; see report.json'


if __name__ == '__main__':
    main()
