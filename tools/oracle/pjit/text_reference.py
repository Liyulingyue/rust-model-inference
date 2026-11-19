"""Same-GGUF caption reference: independently decode, undo pinned export layouts,
then run the unmodified official encoder with F32 arithmetic and BF16 features.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from tools.oracle.pjit.reference import PIN


def grouped(values, axis, head_dim):
    """Undo the pinned converter's grouped-to-tiled V-head permutation."""
    shape = list(values.shape)
    expanded = shape[:axis] + [2, 16, head_dim] + shape[axis + 1:]
    return values.reshape(expanded).swapaxes(axis, axis + 1).reshape(shape)


def restore(values, name, shape):
    if 'linear_attn.' in name:
        if '.in_proj_qkv.' in name or '.conv1d.' in name:
            values = values.reshape(values.shape[0], -1)
            values = np.concatenate([values[:4096], grouped(values[4096:], 0, 128)])
        elif '.in_proj_z.' in name:
            values = grouped(values, 0, 128)
        elif '.in_proj_a.' in name or '.in_proj_b.' in name or name.endswith(('.A_log', '.dt_bias')):
            values = grouped(values.reshape(-1, *values.shape[1:]), 0, 1)
        elif '.out_proj.' in name:
            values = grouped(values, 1, 128)
    if name.endswith('.A_log'):
        if not (values < 0).all():
            raise ValueError('invalid exported SSM A')
        values = np.log(-values)
    elif name.endswith('norm.weight') and not name.endswith('linear_attn.norm.weight'):
        values = values - 1.
    return values.reshape(shape).copy()


def main():
    import gguf
    import torch
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--oracle', type=Path, required=True)
    p.add_argument('--qwen', type=Path, required=True)
    p.add_argument('--gguf', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--activation-quant', action='store_true')
    p.add_argument('--batch-size', type=int, choices=(1, 2), default=1)
    p.add_argument('--prompt', default='A red cube on a white table.')
    a = p.parse_args()
    if subprocess.check_output(['git', '-C', str(a.oracle), 'rev-parse', 'HEAD'], text=True).strip() != PIN:
        raise ValueError('wrong official Oracle revision')
    sys.path.insert(0, str(a.oracle.resolve()))
    from pjit_21k import QwenTextEncoder
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    encoder = QwenTextEncoder(str(a.qwen), device=torch.device('cpu'))
    encoder.model.float()
    reader = gguf.GGUFReader(str(a.gguf))
    tensors = {t.name: t for t in reader.tensors}
    mapping = gguf.get_tensor_name_map(gguf.MODEL_ARCH.QWEN35, 32)
    state = encoder.model.state_dict()
    loaded = []
    for name, original in state.items():
        if name == 'lm_head.weight':
            continue
        key = name.replace('model.language_model.', 'model.').replace('.dt_bias', '.dt_proj.bias')
        mapped = mapping.get_name(key, try_suffixes=('.weight', '.bias'))
        if mapped not in tensors:
            raise ValueError(f'unmapped parameter {name}: {mapped}')
        tensor = tensors[mapped]
        values = gguf.quants.dequantize(tensor.data, tensor.tensor_type)
        with torch.no_grad():
            original.copy_(torch.from_numpy(restore(values, name, tuple(original.shape))))
        loaded.append(mapped)
    del state, original, values
    encoder.model.to('cuda')
    encoder.device = torch.device('cuda')
    if a.activation_quant:
        from tools.oracle.thx02.oracle import activation_roundtrip
        def quantize(kind, reorder):
            def hook(module,args):
                value=args[0];values=value.detach().float().cpu().numpy()
                if reorder:
                    values=values.reshape(*values.shape[:-1],16,2,128).swapaxes(-3,-2).reshape(values.shape)
                values=activation_roundtrip(values,2 if kind==3 else kind)
                if reorder:values=grouped(values,values.ndim-1,128)
                return (torch.from_numpy(np.ascontiguousarray(values)).to(value.device),*args[1:])
            return hook
        for name,module in encoder.model.named_modules():
            key=(name+'.weight').replace('model.language_model.', 'model.')
            mapped=mapping.get_name(key,try_suffixes=('.weight','.bias'))
            tensor=tensors.get(mapped)
            if isinstance(module,torch.nn.Linear) and tensor is not None and int(tensor.tensor_type) in (2,3,8,12,13,14):
                module.register_forward_pre_hook(quantize(int(tensor.tensor_type),name.endswith('.linear_attn.out_proj')))
    output = encoder([a.prompt] * a.batch_size)[0].float().cpu().numpy()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    output.astype('<f4').tofile(a.output)
    a.output.with_suffix('.json').write_text(json.dumps(dict(gguf=str(a.gguf), tensors=len(loaded), shape=list(output.shape), device='cuda', batch_size=a.batch_size, activation_quant=a.activation_quant, oracle_commit=PIN), indent=2))


if __name__ == '__main__':
    main()
