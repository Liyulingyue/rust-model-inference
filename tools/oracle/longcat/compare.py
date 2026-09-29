"""Compare full F32 bit patterns and ordered checkpoint contracts; no tolerances."""
import argparse
import json
import pathlib
import struct


def compare(oracle_path, rust_path):
    oracle_path, rust_path = pathlib.Path(oracle_path), pathlib.Path(rust_path)
    oracle = [json.loads(line) for line in oracle_path.read_text().splitlines()]
    rust = [json.loads(line) for line in rust_path.read_text().splitlines()]
    if len(oracle) != len(rust):
        raise AssertionError(f'checkpoint count: Oracle {len(oracle)}, Rust {len(rust)}')
    names = ['longcat.prelude.img', 'longcat.prelude.txt', 'longcat.prelude.vec']
    names += [f'longcat.double.{i}.{stream}' for i in range(10) for stream in ('img', 'txt')]
    names += [f'longcat.single.{i}' for i in range(20)] + ['longcat.output']
    if [item['name'] for item in oracle] != names:
        raise AssertionError('Oracle must contain all 44 LongCat checkpoints in execution order')
    ni, nt = oracle[0]['shape'][0], oracle[1]['shape'][0]
    if ni <= 0 or nt <= 0:
        raise AssertionError('empty input checkpoint')
    shapes = [[ni, 3072], [nt, 3072], [1, 3072]]
    shapes += [[ni if stream == 'img' else nt, 3072] for i in range(10) for stream in ('img', 'txt')]
    shapes += [[ni + nt, 3072] for i in range(20)] + [[ni, 64]]
    if [item['shape'] for item in oracle] != shapes:
        raise AssertionError('Oracle checkpoint shapes do not match the LongCat graph')
    total = 0
    for a, b in zip(oracle, rust):
        if (a['name'], a['shape']) != (b['name'], b['shape']):
            raise AssertionError(f'checkpoint order/shape: {a} / {b}')
        if b.get('occurrence', 0) != 0 or not b.get('finite', False):
            raise AssertionError(f'invalid Rust checkpoint: {b}')
        x = (oracle_path.parent / a['file']).read_bytes()
        y = pathlib.Path(b['binary_path']).read_bytes()
        expected = 4
        for dimension in a['shape']:
            expected *= dimension
        if len(x) != expected or len(y) != expected:
            raise AssertionError(f'{a["name"]}: invalid raw buffer length')
        if x != y:
            for index, (u, v) in enumerate(zip(struct.iter_unpack('<I', x), struct.iter_unpack('<I', y))):
                if u != v:
                    fx = struct.unpack('<f', struct.pack('<I', u[0]))[0]
                    fy = struct.unpack('<f', struct.pack('<I', v[0]))[0]
                    raise AssertionError(f'{a["name"]}[{index}]: Oracle 0x{u[0]:08x} ({fx}), Rust 0x{v[0]:08x} ({fy})')
        total += expected // 4
    print(f'PASS: {len(oracle)} ordered checkpoints, {total} F32 bit patterns identical')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('oracle_manifest')
    parser.add_argument('rust_manifest')
    args = parser.parse_args()
    compare(args.oracle_manifest, args.rust_manifest)
