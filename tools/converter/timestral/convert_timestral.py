"""Lossless F32 Timestral-209M export; preserves official tensor names and QKV layout."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from tools.converter.utils.gguf import GGML_F32, GgufWriter, gguf_dims, open_safetensors

CONFIG = Path(__file__).resolve().parents[3] / 'src/models/timestral/config.json'


def validate_config(config):
    if config != json.loads(CONFIG.read_text()):
        raise ValueError('Only the published Timestral-209M configuration is supported')


def tensor_contracts(config):
    validate_config(config)
    d, f, ps = config['d_model'], config['mlp_hidden_dim'], config['patch_size']
    shapes = {'representation.forecast_query': (d,), 'transformer.out_norm.scale': (d,)}
    for prefix, inp, out, bias in [('representation.patch_proj', 2*ps, d, True),
                                   ('head.projection', d, ps*len(config['quantile_knots']), False)]:
        for name, n_in, n_out in [('hidden_layer', inp, 4*d), ('output_layer', 4*d, out),
                                  ('residual_layer', inp, out)]:
            shapes[f'{prefix}.{name}.weight'] = (n_out, n_in)
            if bias:
                shapes[f'{prefix}.{name}.bias'] = (n_out,)
    for i, kind in enumerate(config['block_schedule']):
        p = f'transformer.layers.{i}'
        shapes.update({f'{p}.mlp_norm.scale': (d,), f'{p}.mlp.w12.weight': (2*f, d),
                       f'{p}.mlp.w3.weight': (d, f)})
        for j in range(2 if kind == 'time_variate' else 1):
            shapes[f'{p}.attn_norms.{j}.scale'] = (d,)
            for name, n_out in [('wQKV', 3*d), ('wO', d), ('_gate', d)]:
                shapes[f'{p}.attns.{j}.{name}.weight'] = (n_out, d)
    return shapes


def convert(model_dir, output):
    if output.exists():
        raise FileExistsError(output)
    config = json.loads((model_dir / 'config.json').read_text())
    shapes = tensor_contracts(config)
    source = open_safetensors(model_dir / 'model.safetensors')
    if set(source.header) - {'__metadata__'} != set(shapes):
        raise ValueError('Timestral tensor names do not match the contract')
    spans = []
    for name, shape in shapes.items():
        info = source.header[name]
        start, end = info['data_offsets']
        if (tuple(info['shape']) != shape or info['dtype'] != 'F32' or start < 0
                or end-start != math.prod(shape)*4 or source.data_offset+end > source.file_size):
            raise ValueError(f'Invalid tensor {name}: {info}')
        spans.append((start, end))
    spans.sort()
    if spans[0][0] != 0 or any(a[1] != b[0] for a, b in zip(spans, spans[1:])) or source.data_offset+spans[-1][1] != source.file_size:
        raise ValueError('Non-contiguous or overlapping safetensors data')
    writer = GgufWriter(output)
    writer.add_meta('general.architecture', 'timestral')
    writer.add_meta('general.name', 'Timestral-209M')
    writer.add_meta('timestral.config', json.dumps(config))
    with source.path.open('rb') as stream:
        writer.add_meta('timestral.source_sha256', hashlib.file_digest(stream, 'sha256').hexdigest())

    def chunks(name):
        start, end = source.header[name]['data_offsets']
        with source.path.open('rb') as stream:
            stream.seek(source.data_offset+start)
            remaining = end-start
            while remaining:
                raw = stream.read(min(remaining, 4*1024*1024))
                if not raw or not np.isfinite(np.frombuffer(raw, dtype='<f4')).all():
                    raise ValueError(f'Truncated or non-finite tensor {name}')
                yield raw
                remaining -= len(raw)

    for name, shape in shapes.items():
        writer.add_tensor_chunks(name, GGML_F32, gguf_dims(shape), math.prod(shape)*4,
                                 lambda name=name: chunks(name))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix+'.tmp')
    if temporary.exists():
        raise FileExistsError(temporary)
    writer.path = temporary
    try:
        writer.write()
        temporary.rename(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    print(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model_dir', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    convert(args.model_dir, args.output)
