"""Check a converted OCRia GGUF against the pinned ONNX graphs.

Reads the file back with the shared GGUF reader and asserts, tensor by tensor,
that it holds exactly the contract's names and shapes and the same numbers the
ONNX graphs do:

* an F32 tensor must equal the ONNX array **bit for bit**;
* a Q8_0 tensor must equal `quantize_q8_0` of that array **byte for byte** -- the
  stored blocks are compared, not the dequantized values, so a change in the
  block layout or the rounding cannot pass;
* the storage type must follow `quantizable` uniformly: in a `q8_0` file every
  2-D tensor whose row length is a multiple of 32 is Q8_0 and everything else is
  F32, never "silently kept F32";
* the metadata must carry the documented format string, the exact `ocria.config`
  string, the four charsets (parsing to one fewer entry than the head's classes)
  and the source hashes of `sources.json`.

Usage:

    python -m tools.converter.ocria.check_ocria /workspace/models/OCRia-nano-1 \\
        out.gguf --quant q8_0
"""
import argparse
import json
from pathlib import Path

import numpy as np

from tools.converter.ocria.convert_ocria import (
    CONTRACT,
    FORMATS,
    GROUPS,
    SOURCES,
    quantizable,
    source_files,
    tensor_map,
)
from tools.converter.utils.gguf import (
    GGML_F32,
    GGML_Q8_0,
    read_gguf_directory,
    read_gguf_tensor_bytes,
    quantize_q8_0,
)


def check(model_dir, gguf, quant):
    tensors, classes = tensor_map(model_dir)
    metadata, directory = read_gguf_directory(gguf)
    if set(directory) != set(CONTRACT['tensors']):
        missing = sorted(set(CONTRACT['tensors']) - set(directory))
        extra = sorted(set(directory) - set(CONTRACT['tensors']))
        raise ValueError(f'tensor set differs: missing {missing}, extra {extra}')
    stored = {GGML_F32: 0, GGML_Q8_0: 0}
    for name, shape in CONTRACT['tensors'].items():
        ggml_type, dims, _nbytes = directory[name]
        if dims != tuple(reversed(shape)):
            raise ValueError(f'{name}: dims {dims} != {tuple(reversed(shape))}')
        expected = np.ascontiguousarray(tensors[name], dtype=np.float32)
        raw = read_gguf_tensor_bytes(gguf, name)
        if ggml_type == GGML_Q8_0:
            if quant != 'q8_0' or not quantizable(shape):
                raise ValueError(f'{name}: unexpected Q8_0 storage')
            if raw != quantize_q8_0(expected):
                raise ValueError(f'{name}: Q8_0 blocks differ from quantize_q8_0')
        elif ggml_type == GGML_F32:
            if quant == 'q8_0' and quantizable(shape):
                raise ValueError(f'{name}: quantizable tensor kept in F32')
            if raw != expected.astype('<f4').tobytes():
                raise ValueError(f'{name}: F32 payload differs from the ONNX array')
        else:
            raise ValueError(f'{name}: unsupported storage {ggml_type}')
        stored[ggml_type] += 1
    expected_metadata = {
        'general.architecture': 'ocria',
        'ocria.format': FORMATS[quant],
        'ocria.config': json.dumps(CONTRACT['config'], separators=(',', ':')),
    }
    for key, value in expected_metadata.items():
        if metadata.get(key) != value:
            raise ValueError(f'metadata {key}: {metadata.get(key)!r} != {value!r}')
    for group in GROUPS:
        charset = json.loads(metadata[f'ocria.rec.{group}.charset'])
        if len(charset) + 1 != classes[group]:
            raise ValueError(f'ocria.rec.{group}.charset has {len(charset)} entries '
                             f'for {classes[group]} classes')
    hashes = json.loads(metadata['ocria.source.files'])
    pins = {row['Path']: row['Sha256'] for row in SOURCES['files']}
    for name in source_files():
        if hashes.get(name) != pins[name]:
            raise ValueError(f'ocria.source.files lacks the pin for {name}')
    return stored, len(directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model_dir', type=Path)
    parser.add_argument('gguf', type=Path)
    parser.add_argument('--quant', choices=sorted(FORMATS), default='f32')
    args = parser.parse_args()
    stored, total = check(args.model_dir, args.gguf, args.quant)
    print(f'{args.gguf}: {total} tensors, F32 {stored[GGML_F32]}, Q8_0 {stored[GGML_Q8_0]}, '
          f'{args.gguf.stat().st_size} bytes')


if __name__ == '__main__':
    main()
