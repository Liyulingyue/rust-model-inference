"""Compare consecutive full-vocabulary little-endian f32 logits dumps."""

import argparse
import json
import math
import struct
from pathlib import Path


def compare(reference, actual, width=248_320):
    row = struct.Struct(f"<{width}f")
    steps = []
    with Path(reference).open("rb") as ref, Path(actual).open("rb") as got:
        while True:
            a, b = ref.read(row.size), got.read(row.size)
            if not a and not b:
                break
            if len(a) != row.size or len(b) != row.size:
                raise ValueError("mismatched or truncated logits dump")
            x, y = row.unpack(a), row.unpack(b)
            if not all(map(math.isfinite, x)) or not all(map(math.isfinite, y)):
                raise ValueError("non-finite logits")
            norm_x = sum(v * v for v in x)
            norm_y = sum(v * v for v in y)
            errors = [abs(v - w) for v, w in zip(x, y)]
            squared_error = sum(e * e for e in errors)
            steps.append({
                "oracle_id": max(range(width), key=x.__getitem__),
                "rust_id": max(range(width), key=y.__getitem__),
                "cosine": sum(v * w for v, w in zip(x, y)) / math.sqrt(norm_x * norm_y),
                "relative_l2": math.sqrt(squared_error / norm_x),
                "rmse": math.sqrt(squared_error / width),
                "max_abs": max(errors),
            })
    if not steps:
        raise ValueError("empty logits dump")
    return steps


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference")
    parser.add_argument("actual")
    args = parser.parse_args()
    result = compare(args.reference, args.actual)
    print(json.dumps(result, indent=2))
    raise SystemExit(int(any(s["cosine"] < 0.995 or s["relative_l2"] > 0.10
                             or s["oracle_id"] != s["rust_id"] for s in result)))
