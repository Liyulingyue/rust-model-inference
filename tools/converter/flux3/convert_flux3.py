"""Convert the released FLUX 3 Image Klein BF16 package to three GGUFs.

python -m tools.converter.flux3.convert_flux3 MODEL_DIRECTORY OUTPUT_DIRECTORY
DiT/text matrices retain BF16 bits; the shared VAE uses F16 convolutions.
"""
import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from tools.converter.utils.gguf import (
    GGML_BF16, GGML_F16, GGML_F32, GgufWriter, bf16_to_f32, open_safetensors,
)


def dit_shapes():
    shapes = {
        'emb_in.image.weight': (3072, 128),
        'emb_in.image_cond.weight': (3072, 128),
        'txt_in.weight': (3072, 20480),
        'final_layer.image.linear.weight': (128, 3072),
        'final_layer.image.adaLN_modulation.1.weight': (6144, 3072),
    }
    for name in ('time_in', 'ref_time_in'):
        shapes[f'{name}.in_layer.weight'] = (3072, 256)
        shapes[f'{name}.out_layer.weight'] = (3072, 3072)
    for stage in ('early_stream_modulations', 'single_stream_modulations'):
        for mode in ('image', 'image_cond', 'txt'):
            shapes[f'{stage}.{mode}.lin.weight'] = (9216, 3072)
    for stack, count in [('content_mode_blocks.image', 5), ('content_mode_blocks.image_cond', 5),
                         ('txt_mode_blocks', 5), ('single_blocks', 28)]:
        for i in range(count):
            for suffix, shape in {
                'q_proj.weight': (3072, 3072), 'k_proj.weight': (3072, 3072),
                'v_proj.weight': (3072, 3072), 'attn_out.weight': (3072, 3072),
                'mlp_in.weight': (18432, 3072), 'mlp_out.weight': (3072, 9216),
                'norm.query_norm.scale': (128,), 'norm.key_norm.scale': (128,),
            }.items():
                shapes[f'{stack}.{i}.{suffix}'] = shape
    return shapes


def validate_config(config):
    if config.get('model') != 'flux-3-image-klein' or config.get('quantization') is not None:
        raise ValueError('expected the distilled BF16 flux-3-image-klein package')
    expected = dict(hidden=3072, heads=24, depth=5, depth_single=28, mlp_ratio=3.0,
                    axes_dim=[32]*4, theta=10000, in_channels=dict(image=128, image_cond=128, text=20480),
                    flow_map=True, image_rope_scale=0.5)
    if any(config.get('dit', {}).get(k) != v for k, v in expected.items()):
        raise ValueError('unsupported FLUX3 DiT configuration')


def stream_tensor(tensors, name):
    info = tensors.header[name]
    start, end = info['data_offsets']
    with tensors.path.open('rb') as source:
        source.seek(tensors.data_offset + start)
        remaining = end - start
        while remaining:
            block = source.read(min(remaining, 4 << 20))
            if not block:
                raise ValueError(f'truncated tensor {name}')
            remaining -= len(block)
            yield block


def copy_tensor(writer, tensors, source_name, name):
    info = tensors.header[source_name]
    if info['dtype'] not in ('BF16', 'F32'):
        raise ValueError(f'{source_name}: unsupported dtype {info["dtype"]}')
    size = info['data_offsets'][1] - info['data_offsets'][0]
    kind = GGML_BF16 if info['dtype'] == 'BF16' else GGML_F32
    writer.add_tensor_chunks(name, kind, tuple(info['shape'][::-1]), size,
                             lambda: stream_tensor(tensors, source_name))


def convert_dit(root, out):
    validate_config(json.loads((root / 'config.json').read_text()))
    tensors = open_safetensors(root / 'model.safetensors')
    expected = dit_shapes()
    actual = {k: tuple(v['shape']) for k, v in tensors.header.items() if k != '__metadata__'}
    if actual != expected or any(tensors.header[k]['dtype'] != 'BF16' for k in expected):
        raise ValueError('FLUX3 requires the exact 359-tensor distilled BF16 inventory')
    writer = GgufWriter(out)
    writer.add_meta('general.architecture', 'flux3')
    writer.add_meta('general.name', 'FLUX 3 Image Klein distilled')
    writer.add_meta('flux3.flow_map', True)
    for name in sorted(expected):
        copy_tensor(writer, tensors, name, name)
    writer.write()


def text_tensor_name(name):
    prefix = 'model.language_model.'
    if not name.startswith(prefix):
        return None
    name = name[len(prefix):]
    if name == 'embed_tokens.weight':
        return 'token_embd.weight'
    if name == 'norm.weight':
        return 'output_norm.weight'
    _, layer, suffix = name.split('.', 2)
    mapped = {
        'input_layernorm.weight': 'attn_norm.weight',
        'post_attention_layernorm.weight': 'ffn_norm.weight',
        'self_attn.q_norm.weight': 'attn_q_norm.weight',
        'self_attn.k_norm.weight': 'attn_k_norm.weight',
        'self_attn.q_proj.weight': 'attn_q.weight',
        'self_attn.k_proj.weight': 'attn_k.weight',
        'self_attn.v_proj.weight': 'attn_v.weight',
        'self_attn.o_proj.weight': 'attn_output.weight',
        'mlp.gate_proj.weight': 'ffn_gate.weight', 'mlp.up_proj.weight': 'ffn_up.weight',
        'mlp.down_proj.weight': 'ffn_down.weight',
    }
    return f'blk.{int(layer)}.{mapped[suffix]}'


def text_shapes():
    shapes = {'token_embd.weight': (151936,2560), 'output_norm.weight': (2560,)}
    for i in range(36):
        for suffix, shape in {
            'attn_norm.weight': (2560,), 'ffn_norm.weight': (2560,),
            'attn_q_norm.weight': (128,), 'attn_k_norm.weight': (128,),
            'attn_q.weight': (4096,2560), 'attn_k.weight': (1024,2560),
            'attn_v.weight': (1024,2560), 'attn_output.weight': (2560,4096),
            'ffn_gate.weight': (9728,2560), 'ffn_up.weight': (9728,2560),
            'ffn_down.weight': (2560,9728),
        }.items():
            shapes[f'blk.{i}.{suffix}'] = shape
    return shapes


def vae_shapes():
    shapes = {'bn.running_mean': (128,), 'bn.running_var': (128,)}
    def conv(name, input, output, kernel=3):
        shapes[name+'.weight'] = (output,input,kernel,kernel)
        shapes[name+'.bias'] = (output,)
    def norm(name, channels):
        shapes[name+'.weight'] = (channels,)
        shapes[name+'.bias'] = (channels,)
    def block(name, input, output):
        for i in (1,2):
            norm(name+f'.norm{i}',input if i==1 else output)
            conv(name+f'.conv{i}',input if i==1 else output,output)
        if input != output: conv(name+'.nin_shortcut',input,output,1)
    for mode in ('encoder','decoder'):
        conv(mode+'.conv_in',3 if mode=='encoder' else 32,128 if mode=='encoder' else 512)
        for i in (1,2): block(mode+f'.mid.block_{i}',512,512)
        norm(mode+'.mid.attn_1.norm',512)
        for name in ('q','k','v','proj_out'): conv(mode+'.mid.attn_1.'+name,512,512,1)
        norm(mode+'.norm_out',512 if mode=='encoder' else 128)
        conv(mode+'.conv_out',512 if mode=='encoder' else 128,64 if mode=='encoder' else 3)
    conv('encoder.quant_conv',64,64,1)
    conv('decoder.post_quant_conv',32,32,1)
    for stage, input, output in [(0,128,128),(1,128,256),(2,256,512),(3,512,512)]:
        for i in range(2): block(f'encoder.down.{stage}.block.{i}',input if i==0 else output,output)
        if stage!=3: conv(f'encoder.down.{stage}.downsample.conv',output,output)
    for stage, input, output in [(3,512,512),(2,512,512),(1,512,256),(0,256,128)]:
        for i in range(3): block(f'decoder.up.{stage}.block.{i}',input if i==0 else output,output)
        if stage!=0: conv(f'decoder.up.{stage}.upsample.conv',output,output)
    return shapes


def add_tokenizer(writer, root):
    tok = json.loads((root / 'tokenizer.json').read_text())
    if tok['model']['type'] != 'BPE' or tok['normalizer'] != {'type': 'NFC'}:
        raise ValueError('expected Qwen BPE with NFC normalization')
    if any(not isinstance(rule, str) or len(rule.split(' ')) != 2 for rule in tok['model']['merges']):
        raise ValueError('expected the released string BPE merge rules')
    tokens, types = [f'<|reserved_{i}|>' for i in range(151936)], [5] * 151936
    for piece, i in tok['model']['vocab'].items():
        if not isinstance(i, int) or not 0 <= i < len(tokens):
            raise ValueError('tokenizer id out of range')
        tokens[i], types[i] = piece, 1
    for entry in tok['added_tokens']:
        if not isinstance(entry['id'], int) or not 0 <= entry['id'] < len(tokens):
            raise ValueError('added token id out of range')
        tokens[entry['id']], types[entry['id']] = entry['content'], 3 if entry['special'] else 4
    for k, v in {
        'model': 'gpt2', 'pre': 'qwen2', 'tokens': tokens, 'token_type': types,
        'merges': tok['model']['merges'],
        'bos_token_id': 151643, 'eos_token_id': 151645, 'padding_token_id': 151643,
        'add_bos_token': False, 'add_eos_token': False, 'normalizer.nfc': True,
    }.items():
        writer.add_meta('tokenizer.ggml.' + k, v)

def convert_text(root, out):
    root = root / 'text_encoder'
    cfg = json.loads((root / 'config.json').read_text())['text_config']
    expected = dict(hidden_size=2560, num_hidden_layers=36, num_attention_heads=32,
                    num_key_value_heads=8, head_dim=128, intermediate_size=9728,
                    rope_theta=5000000, vocab_size=151936, rms_norm_eps=1e-6, hidden_act='silu', attention_bias=False)
    if any(cfg.get(k) != v for k, v in expected.items()):
        raise ValueError('expected the released Qwen3-VL-4B text encoder')
    if cfg.get('rope_scaling') != {'mrope_interleaved': True, 'mrope_section': [24,20,20], 'rope_type': 'default'}:
        raise ValueError('unsupported Qwen3-VL rotary configuration')
    writer = GgufWriter(out)
    writer.add_meta('general.architecture', 'qwen3vl')
    for k, v in {
        'embedding_length': 2560, 'block_count': 36, 'context_length': 262144,
        'attention.head_count': 32, 'attention.head_count_kv': 8,
        'attention.key_length': 128, 'attention.value_length': 128,
        'feed_forward_length': 9728, 'attention.layer_norm_rms_epsilon': 1e-6,
        'rope.freq_base': 5000000.0, 'rope.dimension_count': 128,
        'rope.dimension_sections': [24, 20, 20, 0],
    }.items():
        writer.add_meta('qwen3vl.' + k, v)
    add_tokenizer(writer, root)
    actual = {}
    for shard in sorted(root.glob('model-*.safetensors')):
        tensors = open_safetensors(shard)
        for source_name in sorted(tensors.header):
            name = text_tensor_name(source_name)
            if name:
                if name in actual or tensors.header[source_name]['dtype'] != 'BF16':
                    raise ValueError(f'duplicate or non-BF16 text tensor {source_name}')
                actual[name] = tuple(tensors.header[source_name]['shape'])
                copy_tensor(writer, tensors, source_name, name)
    if actual != text_shapes():
        raise ValueError('expected the exact 398-tensor Qwen3-VL-4B BF16 text inventory')
    writer.write()


def convert_vae(root, out):
    tensors = open_safetensors(root / 'ae.safetensors')
    actual = {k:tuple(v['shape']) for k,v in tensors.header.items() if k not in ('__metadata__', 'bn.num_batches_tracked')}
    if actual != vae_shapes():
        raise ValueError('expected the 32-channel FLUX3 autoencoder')
    writer = GgufWriter(out)
    writer.add_meta('general.architecture', 'flux3_vae')
    for name in sorted(tensors.header):
        if name == '__metadata__' or name == 'bn.num_batches_tracked':
            continue
        tensor = tensors.get(name)
        if tensor.dtype != 'F32':
            raise ValueError(f'{name}: expected F32')
        values = np.frombuffer(tensor.raw, dtype='<f4')
        if not np.isfinite(values).all():
            raise ValueError(f'{name}: non-finite weight')
        kind = GGML_F16 if len(tensor.shape) == 4 else GGML_F32
        mapped = name.removeprefix('decoder.') if name.startswith('decoder.post_quant_conv.') else name
        writer.add_tensor(mapped, kind, tuple(tensor.shape[::-1]),
                          values.astype('<f2' if kind == GGML_F16 else '<f4').tobytes())
    writer.write()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model_dir', type=Path)
    parser.add_argument('output_dir', type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, convert in [('dit', convert_dit), ('text', convert_text), ('vae', convert_vae)]:
        path = args.output_dir / f'flux3-{name}.gguf'
        if path.exists():
            raise FileExistsError(path)
        print(f'Converting {name} -> {path}', flush=True)
        fd, temporary = tempfile.mkstemp(prefix=f'.flux3-{name}.', dir=args.output_dir)
        os.close(fd)
        temporary = Path(temporary)
        try:
            convert(args.model_dir, temporary)
            os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
