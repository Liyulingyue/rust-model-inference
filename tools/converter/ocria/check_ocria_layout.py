"""Check a converted LayoutNet GGUF against the pinned ONNX graph and `.bin`.

Reads the file back with the shared GGUF reader and asserts, tensor by tensor,
that it holds exactly the contract's names and shapes and the same numbers the
ONNX graph does:

* an F32 tensor must equal the ONNX array **bit for bit**;
* a Q8_0 tensor must equal `quantize_q8_0` of that array **byte for byte** -- the
  stored blocks are compared, not the dequantized values, so a change in the
  block layout or the rounding cannot pass;
* the storage type must follow `quantizable` uniformly: in a `q8_0` file every
  2-D tensor whose row length is a multiple of 32 is Q8_0 and everything else is
  F32, never "silently kept F32";
* the metadata must carry the architecture, the documented format string, the
  exact `ocria-layout.config` string (the one the loader compares against) and
  the source hashes of `sources.json`.

It then re-runs the second opinion the converter's naming rests on: every tensor
must also be **bit-identical to `tablenet.bin`**, the reference runtime's own
weight file, under the mapping documented in `layout_graph_notes.md` §2 --
identical for the 106 tensors the `.bin` carries, and equal to the float32
BatchNorm fold for the four `img.cnn.*.bias` tensors it does not. That catches a
wrong orientation or a wrong name assignment that the ONNX comparison alone
would happily accept, because the GGUF would still match the file it came from.

Usage:

    python -m tools.converter.ocria.check_ocria_layout /workspace/models/OCRia-nano-1 \\
        out.gguf --quant q8_0
"""
import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np

from tools.converter.ocria.convert_ocria_layout import (
    ARCHITECTURE,
    CONTRACT,
    FORMATS,
    SOURCES,
    quantizable,
    source_pins,
    tensor_map,
)
from tools.converter.utils.gguf import (
    GGML_F32,
    GGML_Q8_0,
    read_gguf_directory,
    read_gguf_tensor_bytes,
    quantize_q8_0,
)

#: conv module -> the BatchNorm module the ONNX export folded into it. The `.bin`
#: carries the raw kernel and the BN arrays; the GGUF carries the folded pair.
CONV_FOLD_MODULES = {
    'img.cnn.0': 'img.cnn.1',
    'img.cnn.3.0': 'img.cnn.3.1',
    'img.cnn.4.0': 'img.cnn.4.1',
    'img.cnn.5.0': 'img.cnn.5.1',
}


def read_bin(path):
    """`tablenet.bin`: `b"TNET"`, u32 LE header length, JSON header, raw LE f32."""
    data = path.read_bytes()
    if data[:4] != b'TNET':
        raise ValueError(f'{path.name}: not a LayoutNet weight file')
    length = struct.unpack('<I', data[4:8])[0]
    header = json.loads(data[8:8 + length])
    body = data[8 + length:]
    tensors = {}
    for entry in header['tensors']:
        count = int(np.prod(entry['shape'], dtype=np.int64))
        raw = body[entry['offset'] * 4:entry['offset'] * 4 + count * 4]
        tensors[entry['name']] = np.frombuffer(raw, dtype='<f4').reshape(entry['shape'])
    return tensors, header


def fold(kernel, bn):
    """The float32 BatchNorm fold the ONNX export applied to one conv."""
    scale = (bn['weight'] / np.sqrt(bn['running_var'] + np.float32(1e-5))).astype(np.float32)
    return (kernel * scale[:, None, None, None]).astype(np.float32), \
        (bn['bias'] - bn['running_mean'] * scale).astype(np.float32)


def check(model_dir, gguf, quant):
    tensors, graph = tensor_map(model_dir)
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
        'general.architecture': ARCHITECTURE,
        'ocria-layout.format': FORMATS[quant],
    }
    for key, value in expected_metadata.items():
        if metadata.get(key) != value:
            raise ValueError(f'metadata {key}: {metadata.get(key)!r} != {value!r}')
    # The loader compares the GGUF's config to `contract.json` by value, so key order
    # and whitespace are free. What this pins is the converter's own invariant: the
    # file carries the graph's `liteparse.tablenet` string verbatim, and it parses back
    # to the contract's `config`.
    graph_config = graph.metadata['liteparse.tablenet']
    if metadata['ocria-layout.config'] != graph_config:
        raise ValueError('ocria-layout.config is not the graph metadata verbatim')
    if json.loads(metadata['ocria-layout.config']) != CONTRACT['config']:
        raise ValueError('ocria-layout.config disagrees with the contract config')
    hashes = json.loads(metadata['ocria-layout.source.files'])
    for name, sha in source_pins().items():
        if hashes.get(name) != sha:
            raise ValueError(f'ocria-layout.source.files lacks the pin for {name}')
    # second opinion: the reference runtime's own weight file
    reference, header = read_bin(model_dir / 'tablenet.bin')
    if header['cfg'] != CONTRACT['config']['cfg'] or \
            header['feat_dim'] != CONTRACT['config']['feat_dim']:
        raise ValueError('LayoutNet .bin header disagrees with the contract config')
    for name in CONTRACT['tensors']:
        expected = np.ascontiguousarray(tensors[name], dtype=np.float32)
        module, _, kind = name.rpartition('.')
        if module in CONV_FOLD_MODULES:
            prefix = CONV_FOLD_MODULES[module] + '.'
            arrays = {key[len(prefix):]: reference[key] for key in reference
                      if key.startswith(prefix)}
            folded_kernel, folded_bias = fold(reference[module + '.weight'], arrays)
            reference_array = folded_kernel if kind == 'weight' else folded_bias
        else:
            if name not in reference:
                raise ValueError(f'{name}: missing from tablenet.bin')
            reference_array = reference[name]
        if expected.shape != reference_array.shape:
            raise ValueError(f'{name}: .bin shape {reference_array.shape} != {expected.shape}')
        if not np.array_equal(expected, reference_array):
            raise ValueError(f'{name}: differs from tablenet.bin')
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
