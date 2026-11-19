"""Convert the pinned FF-L + Wan umT5 checkpoints to one native GGUF bundle.

Run on the model host. Tensor payloads are encoded in row chunks, so neither
an uncompressed model copy nor all quantized weights are retained in RAM.
"""
import argparse
import hashlib
import json
import math
import multiprocessing
import os
import tempfile
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from tools.converter.utils.gguf import GgufWriter, gguf_dims
from tools.converter.utils.quantize_gguf import BLOCK_FORMATS, FILE_TYPES, quantize_tensor

CONTRACT_PATH = Path(__file__).resolve().parents[3] / 'src/models/diffusion/fastframes_l/contract.json'
CONTRACT = json.loads(CONTRACT_PATH.read_text())
FORMATS = ('f32','f16','bf16','q8_0','q4_0','q4_1','q4_k','q5_k','q6_k')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def mapped_tensors(component, name, value):
    prefix = {'dit':'dit','text':'text','kv_refiner':'refiner','pixel_decoder':'decoder'}[component]
    grouped = 'heads_grouped.gemms.0.'
    if component == 'kv_refiner' and name.startswith(grouped):
        field = name[len(grouped):]
        for layer in range(value.shape[0]):
            tensor = value[layer]
            if field.endswith('_w'):
                field_name, tensor = field[:-2] + '.weight', tensor.T
            elif field.endswith('_b'):
                field_name, tensor = field[:-2] + '.bias', tensor.reshape(-1)
            else:
                field_name, tensor = field, tensor.reshape(-1)
            yield f'{prefix}.heads.{layer}.{field_name}', tensor
        return
    if name == 'patch_embedding.weight' or (len(value.shape) == 5 and value.shape[2:] == (1,1,1)):
        value = value.reshape(value.shape[0], -1)
    if name.endswith('modulation'):
        value = value.reshape(-1)
    yield f'{prefix}.{name}', value


def encode_chunks(name, value, fmt, pool=None):
    shape = tuple(value.shape)
    dims = gguf_dims(shape)
    target = 0
    block, block_bytes = 1, 4
    if len(shape) == 2 and 'pos_embedding' not in name and fmt != 'f32':
        alignment, kind = BLOCK_FORMATS[fmt]
        if shape[1] % alignment == 0:
            target, block = kind, alignment
            block_bytes = {'f16':2,'bf16':2,'q8_0':34,'q4_0':18,'q4_1':20,'q4_k':144,'q5_k':176,'q6_k':210}[fmt]
    nbytes = math.prod(shape) // block * block_bytes
    rows_per_chunk = max(1, 16 * 1024 * 1024 // (4 * math.prod(shape[1:])))

    def chunks():
        pending = deque()
        for start in range(0,shape[0],rows_per_chunk):
            chunk = value[start:start + rows_per_chunk]
            if isinstance(chunk, torch.Tensor):
                chunk = chunk.detach().float().numpy()
            chunk = np.ascontiguousarray(chunk,dtype='<f4')
            if not np.isfinite(chunk).all():
                raise ValueError(f'non-finite tensor: {name}')
            raw = chunk.tobytes()
            if pool is not None and target in {12,13,14}:
                pending.append(pool.submit(quantize_tensor,name,raw,0,(dims[0],chunk.shape[0]),fmt))
                if len(pending)<16:
                    continue
                actual,raw=pending.popleft().result()
                if actual!=target:
                    raise ValueError(f'inconsistent parallel quantizer for {name}')
            elif target:
                actual, raw = quantize_tensor(name,raw,0,(dims[0],chunk.shape[0]),fmt)
                if actual != target:
                    raise ValueError(f'inconsistent quantizer for {name}')
            yield raw
        for future in pending:
            actual,raw=future.result()
            if actual!=target:
                raise ValueError(f'inconsistent parallel quantizer for {name}')
            yield raw
    return target, nbytes, chunks()


def verify_sources(directory, section):
    for name, expected in CONTRACT[section]['hashes'].items():
        if sha256(directory / name) != expected:
            raise ValueError(f'source SHA256 mismatch: {name}')


def check_inventory(component, state):
    schema = CONTRACT['schemas'][component]
    expected = dict(schema['tensors'])
    for layer in range(schema['block_count']):
        expected.update({f"{schema['block_prefix']}{layer}.{key}": value for key,value in schema['block'].items()})
    if set(state) != set(expected):
        raise ValueError(f'{component} tensor inventory mismatch: missing={set(expected)-set(state)}, extra={set(state)-set(expected)}')
    for name, tensor in state.items():
        actual_type = str(tensor.dtype).replace('torch.bfloat16','BF16').replace('torch.float32','F32')
        if list(tensor.shape) != expected[name]['shape'] or actual_type != expected[name]['dtype']:
            raise ValueError(f'{component}.{name}: source shape/type mismatch')


def convert(model_dir, text_dir, output, fmt):
    if fmt not in FORMATS:
        raise ValueError(f'unknown format: {fmt}')
    if output.exists():
        raise FileExistsError(output)
    verify_sources(model_dir,'source')
    verify_sources(text_dir,'text_source')
    output.parent.mkdir(parents=True,exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f'.{output.name}.',dir=output.parent)
    os.close(fd)
    try:
        with ExitStack() as stack:
            pool = stack.enter_context(ProcessPoolExecutor(
                max_workers=min(16,os.cpu_count() or 1),
                mp_context=multiprocessing.get_context('spawn'),
            )) if fmt in {'q4_k','q5_k','q6_k'} else None
            writer = GgufWriter(Path(temporary))
            metadata = {
                'general.architecture':'fastframes_l','general.name':'FastFrames-L',
                'general.file_type':FILE_TYPES[fmt], 'fastframes_l.contract_version':CONTRACT['version'],
                'fastframes_l.contract_sha256':hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest(),
                'fastframes_l.source_revision':CONTRACT['source']['revision'],
                'fastframes_l.text_revision':CONTRACT['text_source']['revision'],
                'fastframes_l.oracle_commit':CONTRACT['oracle']['commit'],
                'fastframes_l.storage_format':fmt,
                'fastframes_l.tokenizer_json':(text_dir/'google/umt5-xxl/tokenizer.json').read_text(),
            }
            for key,value in metadata.items(): writer.add_meta(key,value)
            for component,filename in [('dit','dit'),('kv_refiner','kv_refiner'),('pixel_decoder','pixel_decoder'),('text',None)]:
                if filename:
                    handle = stack.enter_context(safe_open(model_dir/f'fastframes-l/{filename}.safetensors',framework='pt',device='cpu'))
                    state = {name:handle.get_tensor(name) for name in handle.keys()}
                else:
                    state = torch.load(text_dir/'models_t5_umt5-xxl-enc-bf16.pth',map_location='cpu',mmap=True,weights_only=True)
                check_inventory(component,state)
                for name,value in state.items():
                    for target,tensor in mapped_tensors(component,name,value):
                        kind,nbytes,chunks = encode_chunks(target,tensor,fmt,pool)
                        writer.add_tensor_chunks(target,kind,gguf_dims(tensor.shape),nbytes,lambda chunks=chunks:chunks)
            writer.write()
        os.link(temporary,output)
    finally:
        Path(temporary).unlink(missing_ok=True)
    manifest = {'format':fmt,'size':output.stat().st_size,'sha256':sha256(output),'contract_sha256':metadata['fastframes_l.contract_sha256']}
    output.with_suffix(output.suffix+'.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir',type=Path,required=True)
    parser.add_argument('--text-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--format',choices=FORMATS,default='bf16')
    args = parser.parse_args()
    torch.set_num_threads(4)
    convert(args.model_dir,args.text_dir,args.output,args.format)

if __name__ == '__main__': main()
