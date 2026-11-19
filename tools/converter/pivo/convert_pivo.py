"""Export the pinned PiVo-1.4M acoustic/vocoder checkpoints to a GGUF.

The ``f32`` export is lossless. The ``q8_0`` export stores the weight
matrices and convolution kernels in Q8_0 and keeps the one-dimensional
norms, biases and per-channel statistics in F32: PiVo still computes in F32,
so the quantized export trades a little weight precision for a smaller file.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from tools.converter.utils.gguf import (
    GGML_F32,
    GGML_Q8_0,
    GgufWriter,
    gguf_dims,
    quantize_q8_0,
)

CONTRACT = json.loads((Path(__file__).resolve().parents[3] /
                      'src/models/pivo/contract.json').read_text())
HASHES = {
    'checkpoints/acoustic.pt': '0b4a2dc993e7e500aac1825465886ff23c11f2e1a2662577f2f5cd27f4efb4eb',
    'checkpoints/vocoder.pt': '355d470826817afaa4e729d1b19c8383e985f365af5d2b913f1b9acdf8f7eca4',
    'pivo/resources/cmudict.dict': '81917843c7f44ce2b094ac63873c2c7a4cf802040792c455ba3ca406891c3d22',
    'LICENSE': 'c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4',
    'NOTICE': 'd9dadef1822718ffa7b97d2706198f5b2f9e50ff1c4631a1b0bd254cb9c9d6de',
    'THIRD_PARTY.md': '74fcfc6843f555ddb971128803ea9063a62b7cde5f2cae1c2035576bb44ae86b',
    'pivo/resources/CMUDICT_LICENSE': 'bd4ce8e44170a5f9f481310ca85c51de3c4f851a65e679b40e603b143bd3542a',
    'LICENSES/Opus-BSD-3-Clause.txt': '01e1167d54a096d123cf6dfbbeb19587278845c6481d2d66d545669846079551',
}

FORMATS = {'f32': 'pivo-f32-v1', 'q8_0': 'pivo-q8_0-v1'}
Q8_0_BLOCK = 32


def quantizable(shape, values):
    """Whether a tensor may be stored in Q8_0.

    GGML quantizes a tensor row by row: the row length is the first
    dimension, and a Q8_0 block holds 32 values, so the row length has to be
    a whole number of blocks. Only the weight matrices qualify: the
    one-dimensional norms, biases and per-channel statistics stay F32
    because they are small and precision sensitive.

    The loader enforces the same rule in ``load_tensor`` in
    ``src/models/pivo/mod.rs``; the two have to stay in step, because a
    file this function accepts but the loader rejects would fail to load.
    """
    if len(shape) < 2:
        return False
    return shape[-1] % Q8_0_BLOCK == 0


def convert(directory, output, quant='f32'):
    if output.exists():
        raise FileExistsError(output)
    for name, expected in HASHES.items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'PiVo source hash mismatch: {name}')
    weights = {}
    for component in ('acoustic', 'vocoder'):
        checkpoint = torch.load(directory / f'checkpoints/{component}.pt',
                                map_location='cpu', weights_only=True)
        if component == 'acoustic' and checkpoint['config'] != CONTRACT['config']:
            raise ValueError('Unsupported PiVo acoustic config')
        weights.update({f'{component}.{name}': tensor
                        for name, tensor in checkpoint['state_dict'].items()})
    if weights.keys() != CONTRACT['tensors'].keys():
        raise ValueError('PiVo tensor inventory mismatch')
    writer = GgufWriter(output)
    writer.add_meta('general.architecture', 'pivo')
    writer.add_meta('general.name', 'PiVo-1.4M')
    writer.add_meta('general.license', 'Apache-2.0 (acoustic); BSD-3-Clause (vocoder); CMU (dictionary)')
    writer.add_meta('pivo.licenses', '\n\n'.join((directory / name).read_text()
                    for name in HASHES if name not in ('checkpoints/acoustic.pt',
                    'checkpoints/vocoder.pt', 'pivo/resources/cmudict.dict')))
    writer.add_meta('pivo.format', FORMATS[quant])
    writer.add_meta('pivo.config', json.dumps(CONTRACT['config']))
    writer.add_meta('pivo.source_hashes', json.dumps(HASHES))
    writer.add_meta('pivo.lexicon', (directory / 'pivo/resources/cmudict.dict').read_text())
    quantized = 0
    payloads = []
    for name, tensor in weights.items():
        if list(tensor.shape) != CONTRACT['tensors'][name] or tensor.dtype != torch.float32:
            raise ValueError(f'Invalid PiVo tensor: {name}')
        array = tensor.detach().contiguous().numpy()
        if not np.isfinite(array).all():
            raise ValueError(f'Non-finite PiVo tensor: {name}')
        if quant == 'q8_0' and quantizable(tensor.shape, array):
            payloads.append((name, GGML_Q8_0, gguf_dims(array.shape), quantize_q8_0(array)))
            quantized += 1
        else:
            payloads.append((name, GGML_F32, gguf_dims(array.shape),
                             array.astype('<f4').tobytes()))
    if quantized:
        writer.add_meta('pivo.quantized_tensors', quantized)
    for name, ggml_type, dims, raw in payloads:
        writer.add_tensor(name, ggml_type, dims, raw)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.tmp')
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
    parser.add_argument('directory', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--quant', choices=sorted(FORMATS), default='f32',
                        help='weight storage: f32 (lossless) or q8_0 (smaller file)')
    args = parser.parse_args()
    convert(args.directory, args.output, args.quant)
