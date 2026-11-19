"""Export Qwen3.5-4B with the repository's pinned, read-only llama.cpp converter."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from tools.converter.ovisocr3.convert_ovisocr3 import LLAMA_CPP_COMMIT, _TEXT_CONVERTER


def convert(root, output, llama_cpp, fmt='bf16'):
    root, output, llama_cpp = map(Path, (root, output, llama_cpp))
    if fmt not in ('f32', 'bf16', 'f16'):
        raise ValueError('unsupported text export format')
    revision = subprocess.check_output(['git', '-C', str(llama_cpp), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != LLAMA_CPP_COMMIT:
        raise ValueError(f'expected llama.cpp {LLAMA_CPP_COMMIT}, got {revision}')
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    code = _TEXT_CONVERTER.replace('"bf16"', repr(fmt))
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=output.name + '.') as temporary:
        staging = Path(temporary) / 'text.gguf'
        subprocess.run([sys.executable, '-c', code, str(llama_cpp.resolve()), str(root.resolve()), str(staging.resolve())], check=True)
        os.link(staging, output)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('input',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--llama-cpp',type=Path,required=True)
    p.add_argument('--format',choices=('f32','bf16','f16'),default='bf16')
    args=p.parse_args()
    convert(args.input, args.output, args.llama_cpp, args.format)
