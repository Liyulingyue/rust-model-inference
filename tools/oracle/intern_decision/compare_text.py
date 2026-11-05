"""Check prompt IDs, complete logits and multi-step greedy output against llama.cpp."""
import argparse
import json
import math
from pathlib import Path
import struct


def floats(path, count):
    raw = Path(path).read_bytes()
    if len(raw) != count * 4:
        raise AssertionError(f"invalid F32 payload: {path}")
    values = [v[0] for v in struct.iter_unpack('<f', raw)]
    if not all(math.isfinite(v) for v in values):
        raise AssertionError(f"nonfinite F32 payload: {path}")
    return values


def compare(trace, oracle, log, bound=0.03, teacher_forced=False):
    rows = [json.loads(line) for line in Path(trace).read_text().splitlines()]
    prompts = [r['token_ids'] for r in rows if r['name'].endswith('prompt_ids')]
    lines = Path(log).read_text().splitlines()
    expected_prompts = [json.loads(line.removeprefix('RMI_PROMPT '))
                        for line in lines if line.startswith('RMI_PROMPT ')]
    if len(prompts) != 1 or not prompts[0] or prompts != expected_prompts:
        raise AssertionError('prompt token IDs/count differ')
    greedy = [int(line.split()[1]) for line in lines if line.startswith('RMI_GREEDY ')]
    expected = [line.split('\t') for line in (Path(oracle) / 'index.tsv').read_text().splitlines()]
    actual = [r for r in rows if r['name'] == 'result_output']
    if len(greedy) < 2 or len(greedy) != len(expected):
        raise AssertionError('missing multi-step Oracle logits/greedy records')
    if teacher_forced:
        forced = [r['token_ids'] for r in rows if r['name'] == 'teacher_forced_ids']
        if forced != [prompts[0] + greedy[:-1]]:
            raise AssertionError('teacher-forced history differs from Oracle')
    # Rust emits every prefill token; llama.cpp only materializes its last row.
    if len(actual) != len(prompts[0]) + len(expected) - 1:
        raise AssertionError('Rust prefill/decode checkpoint count differs')
    actual = actual[len(prompts[0]) - 1:]
    metrics, actual_greedy = [], []
    for step, (row, (index, name, dimensions), token) in enumerate(zip(actual, expected, greedy)):
        shape = [int(v.strip()) for v in dimensions.split(',')]
        count = math.prod(shape)
        if name != 'result_output' or row['shape'] != [count] or row['len'] != count:
            raise AssertionError(f'logits checkpoint name/shape differs at step {step}')
        a = floats(row['binary_path'], count)
        b = floats(Path(oracle) / f'{index}.f32', count)
        selected = max(range(count), key=a.__getitem__)
        actual_greedy.append(selected)
        if max(range(count), key=b.__getitem__) != token or (not teacher_forced and selected != token):
            raise AssertionError(f'greedy token differs at step {step}')
        norm = sum(v * v for v in b)
        diff = sum((x - y) ** 2 for x, y in zip(a, b))
        relative_l2 = math.sqrt(diff / norm) if norm else (0.0 if not diff else math.inf)
        if not math.isfinite(bound) or bound <= 0 or not relative_l2 <= bound:
            raise AssertionError(f'logits relative L2 {relative_l2} > {bound} at step {step}')
        metrics.append(dict(relative_l2=relative_l2, max_abs=max(abs(x-y) for x,y in zip(a,b))))
    return dict(prompt_tokens=len(prompts[0]), generated_ids=greedy, rust_argmax_ids=actual_greedy,
                teacher_forced=teacher_forced, logits=metrics,
                relative_l2_bound=bound, comparison='numerical (quantized/approximate operators)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('rust_trace')
    parser.add_argument('oracle_directory')
    parser.add_argument('oracle_log')
    parser.add_argument('--relative-l2', type=float, default=0.03)
    parser.add_argument('--teacher-forced', action='store_true')
    args = parser.parse_args()
    print(json.dumps(compare(args.rust_trace, args.oracle_directory, args.oracle_log,
                             args.relative_l2, args.teacher_forced), indent=2))
