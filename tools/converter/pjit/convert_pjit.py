"""Export the pinned P-JiT 21K DiT to GGUF using existing quantizers.

Run on the remote host: python -m tools.converter.pjit.convert_pjit INPUT OUTPUT --format f32
The Qwen3.5 text component uses the existing pinned llama.cpp text converter.
"""
import argparse
import json
import math
import os
from pathlib import Path
import tempfile

ARCHITECTURE = dict(dim=2944, num_heads=23, num_layers=22, ffn_dim=7936,
    refiner_blocks=2, refiner_ffn_dim=4096, adaln_rank=256, text_dim=7680,
    text_len=256, in_channels=3, out_channels=3, freq_dim=256, eps=1e-6,
    bottleneck_dim=1024, patch=[1,32,32], rope_max_positions=1024,
    rope_spatial_scale=0.5, text_layers=[7,15,27])
FORMATS = ('f32','f16','bf16','q8_0','q4_0','q4_1','q4_k','q5_k','q6_k')


def validate_config(config):
    if config.get('architecture') != ARCHITECTURE:
        raise ValueError('unsupported P-JiT architecture')
    if config.get('text_encoder', {}).get('model') != 'Qwen3.5-4B':
        raise ValueError('P-JiT requires Qwen3.5-4B')


def shapes():
    result = {}
    def linear(name, ni, no, bias=True):
        result[name+'.weight'] = (no,ni)
        if bias: result[name+'.bias'] = (no,)
    result['patch_embed.proj1.weight'] = (1024,3,1,32,32)
    result['patch_embed.proj2.weight'] = (2944,1024,1,1,1)
    result['patch_embed.proj2.bias'] = (2944,)
    linear('time_embed.0',256,2944)
    linear('time_embed.2',2944,2944)
    for stream in ('image','text'): linear(stream+'_adaln.1',2944,256)
    linear('text_proj.0',7680,2944,False)
    linear('text_proj.2',2944,2944,False)
    for stack,count,ff in [('image_refiner',2,2816),('text_refiner',2,2816),('blocks',22,5376)]:
        for i in range(count):
            p = f'{stack}.{i}'
            for norm in ('norm1','norm2','post_norm1','post_norm2','attn.norm_q','attn.norm_k'):
                result[f'{p}.{norm}.weight'] = (2944,)
            for proj in ('q','k','v','o'): linear(f'{p}.attn.{proj}',2944,2944)
            linear(f'{p}.attn.gate_proj',2944,23)
            linear(f'{p}.ffn.w13',2944,2*ff,False)
            linear(f'{p}.ffn.w2',ff,2944,False)
            linear(f'{p}.adaln_up',256,4*2944)
    result['head.norm.weight'] = result['head.norm.bias'] = (2944,)
    result['head.modulation'] = (1,2,2944)
    linear('head.proj',2944,3072)
    return result


def tensor_layout(name, shape):
    if name == 'head.modulation': return (math.prod(shape),)
    if name.startswith('patch_embed.') and name.endswith('.weight'):
        return (math.prod(shape[1:]),shape[0])
    return tuple(reversed(shape))


def convert(root, output, fmt):
    import numpy as np
    from tools.converter.utils.gguf import open_safetensors, GgufWriter, GGML_F32
    from tools.converter.utils.quantize_gguf import quantize_tensor, FILE_TYPES
    root, output = Path(root), Path(output)
    validate_config(json.loads((root/'config.json').read_text()))
    tensors = open_safetensors(root/'model.safetensors')
    expected = shapes()
    actual = {k:tuple(v['shape']) for k,v in tensors.header.items() if k != '__metadata__'}
    if actual != expected or any(tensors.header[k]['dtype'] != 'F32' for k in expected):
        raise ValueError('expected the exact released F32 P-JiT tensor inventory')
    if output.exists(): raise FileExistsError(output)
    writer = GgufWriter(output)
    writer.add_meta('general.architecture','pjit')
    writer.add_meta('general.name','P-JiT 21K')
    writer.add_meta('general.file_type',FILE_TYPES[fmt])
    writer.add_meta('pjit.contract',1)
    for name in sorted(expected):
        tensor = tensors.get(name)
        if len(tensor.raw) != math.prod(expected[name])*4 or not np.isfinite(np.frombuffer(tensor.raw,dtype='<f4')).all():
            raise ValueError(f'invalid or non-finite tensor {name}')
        dims = tensor_layout(name,tensor.shape)
        kind,raw = (GGML_F32,tensor.raw) if fmt == 'f32' else quantize_tensor(name,tensor.raw,GGML_F32,dims,fmt)
        writer.add_tensor(name,kind,dims,raw)
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent,prefix=output.name+'.',suffix='.partial',delete=False) as handle:
        temporary=Path(handle.name)
    writer.path = temporary
    try:
        writer.write()
        os.link(temporary,output)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--format',choices=FORMATS,default='f32')
    args=parser.parse_args()
    convert(args.input,args.output,args.format)
