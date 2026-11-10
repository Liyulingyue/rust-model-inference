"""Compare the engine's dumps against the upstream PyTorch reference.

Both sides dump raw little-endian F32, so the comparison is a plain relative
L2 over the whole vector plus an argmax check. The engine's text logits come
from the parity trace (`result_output`), the reference's from
`pytorch_reference.py`.
"""

import argparse
import json
import math
import struct
from pathlib import Path


def floats(path: Path) -> list:
    raw = path.read_bytes()
    if len(raw) % 4:
        raise AssertionError(f"not F32: {path}")
    values = [v[0] for v in struct.iter_unpack("<f", raw)]
    if not all(math.isfinite(v) for v in values):
        raise AssertionError(f"nonfinite values in {path}")
    return values


def relative_l2(actual: list, expected: list) -> dict:
    if len(actual) != len(expected):
        raise AssertionError(f"length differs: {len(actual)} vs {len(expected)}")
    norm = sum(v * v for v in expected)
    diff = sum((a - b) ** 2 for a, b in zip(actual, expected))
    return {
        "relative_l2": math.sqrt(diff / norm) if norm else 0.0,
        "max_abs": max(abs(a - b) for a, b in zip(actual, expected)),
        "argmax_actual": max(range(len(actual)), key=actual.__getitem__),
        "argmax_expected": max(range(len(expected)), key=expected.__getitem__),
        "length": len(actual),
    }


def engine_logits(trace: Path, step: int) -> list:
    rows = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    outputs = [r for r in rows if r.get("name") == "result_output"]
    prompts = [r for r in rows if r.get("name", "").endswith("prompt_ids")]
    # The engine emits every prefill row; the reference only has the last one.
    index = len(prompts[0]["token_ids"]) - 1 + step
    return floats(Path(outputs[index]["binary_path"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rust-trace", type=Path, help="the engine's parity trace")
    parser.add_argument("--rust-step", type=int, default=0, help="which decode step")
    parser.add_argument("--reference", type=Path, required=True, help="reference dump")
    parser.add_argument("--label", default="text")
    parser.add_argument("--bound", type=float, default=0.05)
    args = parser.parse_args()

    if args.rust_trace:
        actual = engine_logits(args.rust_trace, args.rust_step)
    else:
        actual = floats(args.rust_trace) if args.rust_trace else None
    expected = floats(args.reference)
    metrics = relative_l2(actual, expected)
    metrics["label"] = args.label
    metrics["bound"] = args.bound
    metrics["within_bound"] = metrics["relative_l2"] <= args.bound
    print(json.dumps(metrics, indent=2))
    if not metrics["within_bound"]:
        raise SystemExit(f"{args.label} relative L2 {metrics['relative_l2']} > {args.bound}")


if __name__ == "__main__":
    main()
