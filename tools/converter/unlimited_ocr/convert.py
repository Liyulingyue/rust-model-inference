"""Export original Unlimited-OCR-2 shards to a native, self-contained GGUF.

Run with python -m tools.converter.unlimited_ocr.convert MODEL_DIR OUTPUT --type q8_0.
Convolutions, positions and vectors stay floating point in quantized exports.
"""
import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np
from tools.converter.utils import gguf
from tools.converter.utils.kquants import quantize_k

TYPES = {'bf16':30, 'f16':1, 'f32':0, 'q8_0':8, 'q4_k':12, 'q6_k':14}
DTYPES = {'BF16':(30,2), 'F16':(1,2), 'F32':(0,4), 'I64':(27,8)}


def tensor_type(name, info, kind):
    code, _ = DTYPES[info['dtype']]
    if code == gguf.GGML_I64:
        return code
    if kind in ('bf16', 'f16', 'f32'):
        return TYPES[kind]
    shape = info['shape']
    block = 32 if kind == 'q8_0' else 256
    if len(shape) == 2 and name.endswith('.weight') and 'pos' not in name and shape[-1] % block == 0:
        return TYPES[kind]
    return code


def sources(model_dir):
    index = model_dir/'model.safetensors.index.json'
    if index.exists():
        names = sorted(set(json.loads(index.read_text())['weight_map'].values()))
        if any(Path(n).name != n for n in names):
            raise ValueError('Shard names must be local filenames')
        paths = [model_dir/n for n in names]
    else:
        paths = sorted(model_dir.glob('*.safetensors'))
    if not paths:
        raise ValueError('No safetensors found')
    seen = set()
    for path in paths:
        source = gguf.open_safetensors(path)
        spans = []
        for name, info in source.header.items():
            if name == '__metadata__':
                continue
            if name in seen or info['dtype'] not in DTYPES:
                raise ValueError(f'Duplicate or unsupported tensor {name}')
            seen.add(name)
            dims = gguf.gguf_dims(tuple(info['shape']))
            start, end = info['data_offsets']
            if start < 0 or end-start != math.prod(dims)*DTYPES[info['dtype']][1] or source.data_offset+end > source.file_size:
                raise ValueError(f'Invalid tensor span {name}')
            spans.append((start,end))
        spans.sort()
        if not spans or spans[0][0] != 0 or any(a[1] != b[0] for a,b in zip(spans,spans[1:])) or source.data_offset+spans[-1][1] != source.file_size:
            raise ValueError(f'Non-contiguous or overlapping shard {path}')
        yield source


def chunks(source, name, code):
    info = source.header[name]
    start, end = info['data_offsets']
    dtype = info['dtype']
    element_bytes = DTYPES[dtype][1]
    width = info['shape'][-1]
    chunk_bytes = max(1, 1024*1024//width)*width*element_bytes
    with source.path.open('rb') as stream:
        stream.seek(source.data_offset+start)
        remaining = end-start
        while remaining:
            raw = stream.read(min(remaining, chunk_bytes))
            if not raw:
                raise ValueError(f'Truncated tensor {name}')
            remaining -= len(raw)
            if dtype == 'I64':
                yield raw
                continue
            values = {'BF16':gguf.bf16_to_f32, 'F16':gguf.f16_to_f32, 'F32':lambda x:x}[dtype](raw)
            array = np.frombuffer(values, dtype='<f4')
            if not np.isfinite(array).all():
                raise ValueError(f'Non-finite tensor {name}')
            if code == DTYPES[dtype][0]:
                yield raw
            elif code == gguf.GGML_F32:
                yield values
            elif code == gguf.GGML_BF16:
                yield gguf.f32_to_bf16(values)
            elif code == gguf.GGML_F16:
                encoded = gguf.f32_to_f16(values)
                if not np.isfinite(np.frombuffer(encoded, dtype='<f2')).all():
                    raise ValueError(f'F16 overflow in {name}')
                yield encoded
            elif code == gguf.GGML_Q8_0:
                yield gguf.quantize_q8_0(array)
            else:
                yield quantize_k('q4_k' if code == gguf.GGML_Q4K else 'q6_k', array)


def convert(model_dir, output, kind='bf16'):
    model_dir, output = Path(model_dir), Path(output)
    if output.exists():
        raise FileExistsError(output)
    if kind not in TYPES:
        raise ValueError(f'Unknown type {kind}')
    config = json.loads((model_dir/'config.json').read_text())
    if config.get('model_type') != 'unlimited-ocr' or config.get('architectures') != ['UnlimitedOCRForCausalLM']:
        raise ValueError('Expected original Unlimited-OCR-2 config')
    tokenizer = (model_dir/'tokenizer.json').read_text()
    json.loads(tokenizer)
    shards = list(sources(model_dir))
    writer = gguf.GgufWriter(output)
    writer.add_meta('general.architecture', 'unlimited-ocr')
    writer.add_meta('general.name', 'Unlimited-OCR-2')
    writer.add_meta('unlimited_ocr.config', json.dumps(config))
    writer.add_meta('unlimited_ocr.tokenizer_json', tokenizer)
    writer.add_meta('unlimited_ocr.weight_type', kind)
    for shard in shards:
        with shard.path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        writer.add_meta('unlimited_ocr.source_sha256.'+shard.path.name, digest)
        for name, info in shard.header.items():
            if name == '__metadata__':
                continue
            code = tensor_type(name, info, kind)
            dims = gguf.gguf_dims(tuple(info['shape']))
            writer.add_tensor_chunks(name, code, dims, gguf._tensor_nbytes(code,dims), lambda shard=shard,name=name,code=code:chunks(shard,name,code))
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=output.name+'.', suffix='.tmp', dir=output.parent)
    os.close(fd)
    writer.path = Path(temporary)
    try:
        writer.write()
        # Atomic no-clobber publication, including concurrent converter runs.
        os.link(temporary, output)
    finally:
        Path(temporary).unlink(missing_ok=True)
    print(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model_dir', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--type', choices=TYPES, default='bf16')
    args = parser.parse_args()
    convert(args.model_dir, args.output, args.type)
