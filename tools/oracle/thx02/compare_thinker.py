"""Compare cached thinker steps with native full-vocabulary logits."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from .compare import items, metrics


def compare(actual, expected, fmt='F16'):
    limits = (.02, .005) if fmt == 'F32' else (.04, .01) if fmt == 'F16' else (.10, .025)
    exact = {key: actual[key] == expected[key]
             for key in ('prompt', 'input_ids', 'positions', 'generated_ids')}
    pairs = itertools.zip_longest(actual['generated_ids'], expected['generated_ids'])
    divergence = next((i for i, (a, b) in enumerate(pairs) if a != b), None)
    rows = []
    for i, (a, b) in enumerate(zip(actual['steps'], expected['steps'])):
        reference = np.load(b['logits_path'], allow_pickle=False)
        error = metrics(a['logits'], reference)
        rows.append(dict(step=i, token_exact=a['token_id'] == b['token_id'],
                         same_prefix=divergence is None or i <= divergence,
                         vocabulary=reference.size, logits=error,
                         passed=reference.shape == (248320,) and error['max_abs'] <= limits[0] and error['rmse'] <= limits[1]))
    return dict(format=fmt, limits=dict(logits_max_abs=limits[0], logits_rmse=limits[1]), exact=exact,
                first_greedy_divergence=divergence, steps=rows,
                passed=all(exact.values()) and len(rows) == len(actual['steps']) == len(expected['steps'])
                       and bool(rows) and all(row['passed'] and row['token_exact'] for row in rows),
                actual_generated_ids=actual['generated_ids'], reference_generated_ids=expected['generated_ids'],
                device_stats=actual.get('device_stats'), actual_compute=actual.get('compute'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rust', type=Path, required=True)
    parser.add_argument('--oracle', type=Path, required=True)
    parser.add_argument('--format', default='F16',
                        choices=['F32', 'F16', 'BF16', 'Q8_0', 'Q4_0', 'Q4_K', 'Q5_K', 'Q6_K'])
    parser.add_argument('--expected-case', type=int, default=0)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = compare(items(args.rust)[0], items(args.oracle)[args.expected_case], args.format)
    result.update(actual=str(args.rust.resolve()), oracle=str(args.oracle.resolve()), expected_case=args.expected_case)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(passed=result['passed'], first_greedy_divergence=result['first_greedy_divergence'],
                          output=str(args.output))))
    assert result['passed'], 'THX-02 thinker numerical/discrete gate failed; see report'


if __name__ == '__main__':
    main()
