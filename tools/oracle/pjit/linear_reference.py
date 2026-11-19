"""Replay traced Rust projection inputs with independent GGUF decoding and QDQ.

This separates local kernel error from propagation across quantization boundaries.
It does not replace the end-to-end acceptance gate.
"""
import argparse
import json
from pathlib import Path

import gguf
import numpy as np

from tools.oracle.pjit.compare import metrics
from tools.oracle.thx02.oracle import activation_roundtrip


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('gguf', type=Path)
    parser.add_argument('trace', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    reader = gguf.GGUFReader(str(args.gguf))
    tensors = {t.name: t for t in reader.tensors}
    records = {r['name']: r for r in map(json.loads, args.trace.read_text().splitlines())}
    results = []
    for key, record in records.items():
        if not key.startswith('pjit.linear.') or not key.endswith('.input'):
            continue
        name = key[len('pjit.linear.'):-len('.input')]
        weight = tensors[name + '.weight']
        values = np.fromfile(record['binary_path'], dtype='<f4').reshape(record['shape'])
        kind = int(weight.tensor_type)
        if kind in (2, 3, 8, 12, 13, 14):
            values = activation_roundtrip(values, 2 if kind == 3 else kind)
        expected = values @ gguf.quants.dequantize(weight.data, weight.tensor_type).T
        if name + '.bias' in tensors:
            bias = tensors[name + '.bias']
            expected += gguf.quants.dequantize(bias.data, bias.tensor_type)
        actual = records['pjit.linear.' + name + '.output']
        actual = np.fromfile(actual['binary_path'], dtype='<f4').reshape(actual['shape'])
        result = dict(name=name, kind=kind, **metrics(actual, expected))
        result['passed'] = result['nrmse'] <= .02 and result['cosine'] >= .999
        results.append(result)
    if not results:
        raise ValueError('no projection traces found')
    args.output.write_text(json.dumps(results, indent=2))
    print(json.dumps(dict(count=len(results), failed=sum(not r['passed'] for r in results),
                          worst=sorted(results, key=lambda r: r['nrmse'])[-3:])))


if __name__ == '__main__':
    main()
