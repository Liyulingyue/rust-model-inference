"""Independent scalar CLM-head Oracle. Torch only deserializes the checkpoint."""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import struct
from pathlib import Path


LIBC = ctypes.CDLL(None)
LIBC.erff.argtypes = [ctypes.c_float]
LIBC.erff.restype = ctypes.c_float


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def save(path: Path, values: list[float]) -> None:
    path.write_bytes(struct.pack(f"<{len(values)}f", *values))


def load(path: Path) -> list[float]:
    data = path.read_bytes()
    return list(struct.unpack(f"<{len(data) // 4}f", data))


def dot(x: list[float], weight, start: int, length: int) -> float:
    total = 0.0
    for index in range(length):
        total += f32(x[index] * float(weight[start + index]))
    return f32(total)


def linear(x: list[float], head: dict, name: str) -> list[float]:
    weight = memoryview(head[name + ".weight"].numpy().ravel())
    bias = memoryview(head[name + ".bias"].numpy().ravel())
    rows = len(bias)
    return [f32(dot(x, weight, row * len(x), len(x)) + float(bias[row])) for row in range(rows)]


def gelu(x: list[float]) -> list[float]:
    factor = f32(1.0 / math.sqrt(2.0))
    return [f32(f32(0.5 * value) * f32(1.0 + LIBC.erff(f32(value * factor)))) for value in x]


def norm(x: list[float], head: dict) -> list[float]:
    weight = memoryview(head["norms.0.weight"].numpy().ravel())
    bias = memoryview(head["norms.0.bias"].numpy().ravel())
    mean = sum(x) / len(x)
    variance = sum((value - mean) * (value - mean) for value in x) / len(x)
    scale = f32(1.0 / f32(math.sqrt(f32(f32(variance) + f32(1e-5)))))
    return [f32(f32(f32(f32(value - f32(mean)) * scale) * float(weight[index])) + float(bias[index]))
            for index, value in enumerate(x)]


def unit(x: list[float]) -> list[float]:
    total = sum(f32(value * value) for value in x)
    scale = f32(1.0 / math.sqrt(total)) if total > 0.0 else 0.0
    return [f32(value * scale) for value in x]


def project(input_values: list[float], head: dict, label: str, trace: Path) -> list[float]:
    def checkpoint(name: str, values: list[float]) -> list[float]:
        save(Path(f"{trace}.clm.{label}.{name}.f32"), values)
        return values

    x = checkpoint("inp", linear(input_values, head, "inp"))
    x = checkpoint("gelu1", gelu(x))
    x = checkpoint("hidden", linear(x, head, "hidden.0"))
    x = checkpoint("norm", norm(x, head))
    x = checkpoint("gelu2", gelu(x))
    x = checkpoint("out", linear(x, head, "out"))
    return checkpoint("unit", unit(x))


def run(checkpoint: Path, state: Path, action: Path, trace: Path) -> None:
    import torch

    ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state_vec = project(load(state), ck["state_head"], "state", trace)
    action_vec = project(load(action), ck["action_head"], "action", trace)
    score = f32(f32(100.0 * dot(state_vec, action_vec, 0, len(state_vec))) / 1.0)
    save(Path(f"{trace}.clm.logits.f32"), [score])
    print(json.dumps({"logit_bits": f"{struct.unpack('<I', struct.pack('<f', score))[0]:08x}", "logit": score}))


def fixture(path: Path, offset: int) -> None:
    values = [f32(math.sin((index + offset) * 0.17)) for index in range(4096)]
    save(path, unit(values))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("state", type=Path)
    parser.add_argument("action", type=Path)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--create-fixture", action="store_true")
    args = parser.parse_args()
    if args.create_fixture:
        fixture(args.state, 0)
        fixture(args.action, 29)
    run(args.checkpoint, args.state, args.action, args.trace)
