#!/usr/bin/env python3
"""Run the pinned official graph with independently decoded weights from the same GGUF."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tools.converter.utils.gguf import _read_gguf
from tools.oracle.idol import oracle

FILES = {"denoiser": "idol/denoiser.safetensors", "octree": "idol/decoder.safetensors",
         "gaussian": "idol/decoder_gs.safetensors", "dino": "dinov3-vith16plus/model.safetensors",
         "vae": "FLUX.2-dev/vae/diffusion_pytorch_model.safetensors",
         "background": "idol/birefnet.safetensors"}


class Weights:
    def __init__(self, path, models):
        self.path, self.models = Path(path).resolve(), Path(models).resolve()
        self.metadata, self.tensors = _read_gguf(self.path)
        if self.metadata.get('general.architecture') != 'idol' or self.metadata.get('idol.format_version') != 1:
            raise ValueError('expected Idol GGUF format version 1')
        if self.metadata.get('idol.oracle_commit') != oracle.COMMIT:
            raise ValueError('GGUF does not match the pinned official graph')
        scalars = json.loads(self.metadata['idol.scalar_tensors_json'])
        if (not isinstance(scalars, list) or any(not isinstance(name, str) for name in scalars)
                or len(scalars) != len(set(scalars))
                or any(name not in self.tensors or self.tensors[name][1] != (1,) for name in scalars)):
            raise ValueError('invalid GGUF scalar list')
        self.scalars = set(scalars)
        self.roles = {str((self.models / relative).resolve()): role for role, relative in FILES.items()}

    def load(self, filename, device='cpu'):
        import torch
        from gguf import GGMLQuantizationType
        from gguf.quants import dequantize
        if device != 'cpu' or str(Path(filename).resolve()) not in self.roles:
            raise ValueError(f'unexpected same-weight oracle load: {filename} device={device}')
        role = self.roles[str(Path(filename).resolve())]
        prefix = role + '.'
        state = {}
        with self.path.open('rb') as source:
            for name, (kind, dims, length, offset) in self.tensors.items():
                if not name.startswith(prefix):
                    continue
                source.seek(offset)
                raw = source.read(length)
                if len(raw) != length:
                    raise ValueError(f'truncated GGUF payload: {name}')
                shape = () if name in self.scalars else tuple(reversed(dims))
                if kind == 27:
                    values = np.frombuffer(raw, dtype='<i8').copy()
                else:
                    # llama.cpp's Python decoder is independent of the Rust reader and export quantizers.
                    values = dequantize(np.frombuffer(raw, dtype=np.uint8), GGMLQuantizationType(kind))
                    if not np.isfinite(values).all():
                        raise ValueError(f'non-finite GGUF reference weight: {name}')
                state[name[len(prefix):]] = torch.from_numpy(np.array(values.reshape(shape), copy=True))
        if not state:
            raise ValueError(f'missing GGUF component {role}')
        return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gguf', type=Path, required=True)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args, rest = parser.parse_known_args()
    weights = Weights(args.gguf, args.models)
    with patch('safetensors.torch.load_file', weights.load), patch.object(sys, 'argv', [
            str(Path(oracle.__file__)), '--models', str(args.models), '--output', str(args.output), *rest]):
        oracle.main()
    metadata_path = args.output / 'fixtures.json'
    metadata = json.loads(metadata_path.read_text())
    metadata['gguf'] = dict(sha256=oracle.sha256(args.gguf), path=str(args.gguf.resolve()),
                            weight_format=weights.metadata['idol.weight_format'],
                            independent_decoder=f"gguf.quants {importlib.metadata.version('gguf')}",
                            integer_storage='I64 preserved exactly; no float conversion')
    metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()
