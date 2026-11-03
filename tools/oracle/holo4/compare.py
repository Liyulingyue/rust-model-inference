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


def compare(trace, oracle, log, bound=0.03, teacher_forced=False, graph=False):
    rows = [json.loads(line) for line in Path(trace).read_text().splitlines()]
    prompts = [r['token_ids'] for r in rows if r['name'].endswith('prompt_ids')]
    lines = Path(log).read_text().splitlines()
    expected_prompts = [json.loads(line.removeprefix('RMI_PROMPT '))
                        for line in lines if line.startswith('RMI_PROMPT ')]
    if len(prompts) != 1 or not prompts[0] or prompts != expected_prompts:
        raise AssertionError('prompt token IDs/count differ')
    greedy = [int(line.split()[1]) for line in lines if line.startswith('RMI_GREEDY ')]
    index = [line.split('\t') for line in (Path(oracle) / 'index.tsv').read_text().splitlines()]
    expected = [entry for entry in index if entry[1] == 'result_output']
    actual = [r for r in rows if r['name'] == 'result_output']
    checkpoints = len(prompts[0]) + len(greedy) - 1
    if len(greedy) < 2 or len(expected) != checkpoints:
        raise AssertionError('missing multi-step Oracle logits/greedy records')
    if teacher_forced:
        forced = [r['token_ids'] for r in rows if r['name'] == 'teacher_forced_ids']
        if forced != [prompts[0] + greedy[:-1]]:
            raise AssertionError('teacher-forced history differs from Oracle')
    # Both runners use sequential prefill (-b 1 -ub 1 in llama.cpp).
    if len(actual) != checkpoints:
        raise AssertionError('Rust prefill/decode checkpoint count differs')
    actual = actual[-len(greedy):]
    expected = expected[-len(greedy):]
    metrics, actual_greedy = [], []
    for step, (row, (checkpoint_index, name, dimensions), token) in enumerate(zip(actual, expected, greedy)):
        shape = [int(v.strip()) for v in dimensions.split(',')]
        count = math.prod(shape)
        if (name != 'result_output' or shape != [count, 1, 1, 1]
                or row['shape'] != [count] or row['len'] != count):
            raise AssertionError(f'logits checkpoint name/shape differs at step {step}')
        a = floats(row['binary_path'], count)
        b = floats(Path(oracle) / f'{checkpoint_index}.f32', count)
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
    result = dict(prompt_tokens=len(prompts[0]), generated_ids=greedy, rust_argmax_ids=actual_greedy,
                teacher_forced=teacher_forced, logits=metrics,
                relative_l2_bound=bound, comparison='numerical (quantized/approximate operators)')
    if graph:
        generated = [r['token_ids'] for r in rows if r['name'] == 'qwen35.greedy_token_ids']
        if not teacher_forced and generated != [greedy]:
            raise AssertionError('CLI generated token IDs differ')
        result['checkpoints'] = compare_graph(rows, index, oracle, len(prompts[0]), len(greedy), bound)
    return result


def compare_graph(rows, index, oracle, prompt_tokens, steps, bound):
    metrics = {}
    names = ['model.input_embed', 'l_out-0', 'l_out-3', 'l_out-39', 'result_norm', 'result_output']
    checkpoints = prompt_tokens + steps - 1
    if [entry[1] for entry in index] != names * checkpoints:
        raise AssertionError('Oracle checkpoint order/count differs (use -b 1 -ub 1)')
    rust_names = names[1:]
    if [row['name'] for row in rows if row['name'] in rust_names] != rust_names * checkpoints:
        raise AssertionError('Rust checkpoint order/count differs')
    for name in names[:-1]:
        reference = [entry for entry in index if entry[1] == name]
        rust_name = 'qwen35.embedding' if name == 'model.input_embed' else name
        actual = [row for row in rows if row['name'] == rust_name]
        required = 1 if name == 'model.input_embed' else checkpoints
        if len(actual) != required:
            raise AssertionError(f'{name}: checkpoint count differs')
        errors = []
        if name == 'model.input_embed':
            hidden = int(reference[0][2].split(',')[0])
            row = actual[0]
            if row['shape'] != [prompt_tokens, hidden] or row['len'] != prompt_tokens * hidden:
                raise AssertionError(f'{name}: Rust checkpoint shape differs')
            a = floats(row['binary_path'], row['len'])
        for step, (number, _, dimensions) in enumerate(reference):
            shape = [int(v.strip()) for v in dimensions.split(',')]
            if shape != [shape[0], 1, 1, 1]:
                raise AssertionError(f'{name}: Oracle checkpoint shape differs')
            count = shape[0]
            b = floats(Path(oracle) / f'{number}.f32', count)
            if name == 'model.input_embed':
                if step >= prompt_tokens:
                    continue  # The CLI emits its complete input embedding only once.
                values = a[step * count:(step + 1) * count]
                if struct.pack(f'<{count}f', *values) != struct.pack(f'<{count}f', *b):
                    raise AssertionError('input embedding bits differ')
                errors.append(dict(bitwise_equal=True, values=count))
            else:
                row = actual[step]
                rust_shape = [count] if name == 'result_norm' else [1, count]
                if row['shape'] != rust_shape or row['len'] != count:
                    raise AssertionError(f'{name}: Rust checkpoint shape differs')
                values = floats(row['binary_path'], count)
                norm = sum(v*v for v in b)
                diff = sum((x-y)**2 for x,y in zip(values,b))
                error = math.sqrt(diff/norm) if norm else (0.0 if not diff else math.inf)
                if not error <= bound:
                    raise AssertionError(f'{name}: relative L2 {error} > {bound} at step {step}')
                errors.append(dict(relative_l2=error, max_abs=max(abs(x-y) for x,y in zip(values,b)), values=count))
        metrics[name] = errors
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('rust_trace')
    parser.add_argument('oracle_directory')
    parser.add_argument('oracle_log')
    parser.add_argument('--relative-l2', type=float, default=0.03)
    parser.add_argument('--teacher-forced', action='store_true')
    parser.add_argument('--graph', action='store_true')
    args = parser.parse_args()
    print(json.dumps(compare(args.rust_trace, args.oracle_directory, args.oracle_log,
                             args.relative_l2, args.teacher_forced, args.graph), indent=2))
