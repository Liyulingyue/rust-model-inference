"""Run both real LongCat GGUFs against the pinned scalar C++ Oracle."""
import argparse
import hashlib
import pathlib
import struct
import subprocess
import tempfile
import os
from compare import compare

MODELS = {
    'edit': ('LongCat-Image-Edit-GGUF/LongCat-Image-Edit-Q8_0.gguf', 'a998d17d51943ab15bf3ba3ca1a509be3792987395c3bcb8d6bcbfafa7030a42'),
    'turbo': ('LongCat-Image-Edit-Turbo-GGUF/LongCat-Image-Edit-Turbo-Q8_0.gguf', 'ac4ae6172eea6dc892ba8c1cb63ec5b1232a294d0c430299ec26006f956e437b'),
}


def run(argv, log, env=None):
    with log.open('w') as output:
        subprocess.run([str(v) for v in argv], env=env, stdout=output, stderr=subprocess.STDOUT, check=True)


def fixture(directory, ni, nt, offset):
    directory.mkdir()
    def write(name, values):
        directory.joinpath(name).write_bytes(b''.join(struct.pack('<f', v) for v in values))
    write('img.f32', (((i + offset) % 29 - 14) / 32 for i in range(ni * 64)))
    write('txt.f32', (((i + offset + 4) % 31 - 15) / 64 for i in range(nt * 3584)))
    positions = [(0, i, i) for i in range(nt)] + [(i + 1, nt + i % 2, nt + i // 2) for i in range(ni)]
    write('positions.f32', (v for row in positions for v in row))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('oracle', type=pathlib.Path)
    parser.add_argument('model_root', type=pathlib.Path)
    args = parser.parse_args()
    repo = pathlib.Path(__file__).resolve().parents[3]
    root = pathlib.Path(tempfile.mkdtemp(prefix='longcat-parity-', dir=repo / 'target')).resolve()
    print(f'Artifacts: {root}', flush=True)
    for kind, (name, expected) in MODELS.items():
        model = args.model_root / name
        with model.open('rb') as f:
            digest = hashlib.file_digest(f, 'sha256').hexdigest()
        if digest != expected:
            raise AssertionError(f'{model}: unexpected SHA256 {digest}')
        for case, (ni, nt, timestep, offset) in enumerate([(2, 2, 0.375, 3), (3, 1, 0.875, 11)]):
            work = root / f'{kind}-{case}'
            work.mkdir()
            inputs, oracle = work / 'input', work / 'oracle'
            fixture(inputs, ni, nt, offset)
            oracle.mkdir()
            run([args.oracle, model, inputs, oracle, ni, nt, timestep, kind], work / 'oracle.log')
            trace = work / 'rust.jsonl'
            env = {
                **os.environ,
                'RUSTFLAGS': '-C no-vectorize-loops -C no-vectorize-slp',
                'RMI_SCALAR': '1',
                'RAYON_NUM_THREADS': '1',
                'RMI_PARITY_TRACE': str(trace),
                'RMI_LONGCAT_KIND': kind,
                'RMI_LONGCAT_MODEL': str(model.resolve()),
                'RMI_LONGCAT_INPUT': str(inputs.resolve()),
                'RMI_LONGCAT_OUTPUT': str(work / 'output.f32'),
                'RMI_LONGCAT_TIMESTEP': str(timestep),
            }
            run([
                'cargo', 'test', '--profile', 'release-fast', '--features', 'parity-trace',
                '--manifest-path', repo / 'Cargo.toml', '--test', 'longcat_reference',
                '--', '--ignored', '--exact', 'longcat_transformer_case', '--nocapture',
            ], work / 'rust.log', env)
            print(f'{kind}, case {case}, timestep {timestep}: ', end='', flush=True)
            compare(oracle / 'trace.jsonl', trace)
    print('Both real checkpoints passed both input/position/timestep fixtures.')


if __name__ == '__main__':
    main()
