"""Report the GGUF type mix of a model file.

Usage: ./.venv/bin/python -m tools.converter.utils.inspect_gguf <file.gguf>
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

from tools.converter.utils.gguf import _read_gguf

NAMES = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    8: "Q8_0",
    12: "Q4_K",
    13: "Q5_K",
    14: "Q6_K",
    30: "BF16",
}


def main() -> None:
    path = Path(sys.argv[1])
    metadata, tensors = _read_gguf(path)
    counts = Counter(value[0] for value in tensors.values())
    total = 0
    for ggml_type, count in sorted(counts.items()):
        print(f"{NAMES.get(ggml_type, ggml_type):>5} x {count:>4}")
    for name, (ggml_type, dims, length, _offset) in sorted(tensors.items()):
        total += length
        if ggml_type == 30 and len(dims) == 2:
            print(f"  kept BF16 2-D: {name} {dims}")
    print(f"{path.name}: {len(tensors)} tensors, {total} payload bytes")
    print(f"file_type metadata: {metadata.get('general.file_type')}")


if __name__ == "__main__":
    main()
