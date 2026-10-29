#!/usr/bin/env python3
"""Compare System One responses: exact structure/decisions/offsets, tolerant floats."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def compare(reference, actual, atol, rtol):
    errors, floating = [], []

    def walk(a, b, path):
        if isinstance(a, dict) and isinstance(b, dict):
            if a.keys() != b.keys():
                errors.append(dict(path=path, reason="object keys differ", oracle=list(a), native=list(b)))
                return
            for key in a:
                # Error codes are the contract; implementation-specific diagnostics may differ.
                if key != "message" or "error" not in a:
                    walk(a[key], b[key], f"{path}.{key}")
        elif isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                errors.append(dict(path=path, reason="array length differs", oracle=len(a), native=len(b)))
                return
            for i, (x, y) in enumerate(zip(a, b)):
                walk(x, y, f"{path}[{i}]")
        elif isinstance(a, float) and type(b) in (float, int):
            delta = abs(a - b)
            floating.append(dict(path=path, absolute_error=delta))
            if not math.isfinite(a) or not math.isfinite(b) or delta > atol + rtol * abs(a):
                errors.append(dict(path=path, reason="float tolerance exceeded", oracle=a, native=b, absolute_error=delta))
        elif type(a) is not type(b) or a != b:
            errors.append(dict(path=path, reason="exact value differs", oracle=a, native=b))

    walk(reference, actual, "response")
    return dict(passed=not errors, atol=atol, rtol=rtol,
                floats_compared=len(floating), max_absolute_error=max((x["absolute_error"] for x in floating), default=0),
                largest_errors=sorted(floating, key=lambda x: x["absolute_error"], reverse=True)[:10], errors=errors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("oracle", type=Path)
    parser.add_argument("native", type=Path)
    parser.add_argument("--atol", type=float, default=0.01)
    parser.add_argument("--rtol", type=float, default=0.01)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if not all(math.isfinite(v) and v >= 0 for v in (args.atol, args.rtol)):
        parser.error("tolerances must be finite and nonnegative")
    reference, actual = [json.loads(p.read_text()) for p in (args.oracle, args.native)]
    result = compare(reference.get("response", reference), actual.get("response", actual), args.atol, args.rtol)
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.out:
        args.out.write_text(text)
    print(text, end="")
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
