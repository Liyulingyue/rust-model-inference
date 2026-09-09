#!/usr/bin/env python3
"""Compare every Breeze checkpoint, in order, using exact little-endian F32 bits."""
from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path

PAYLOADS = ("binary_path", "token_ids", "usize_values", "bool_values")


def records(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"{path}: empty trace")
    occurrences: dict[str, int] = {}
    for index, row in enumerate(rows):
        location = f"{path}: record {index}"
        if not isinstance(row, dict) or not isinstance(row.get("name"), str) or not row.get("name"):
            raise ValueError(f"{location}: missing checkpoint name")
        kinds = [key for key in PAYLOADS if key in row]
        if len(kinds) != 1:
            raise ValueError(f"{location}: expected exactly one payload, got {kinds}")
        kind = kinds[0]
        shape = row.get("shape")
        if kind != "bool_values" or shape is not None:
            if not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape):
                raise ValueError(f"{location}: invalid shape")
            length = math.prod(shape)
        else:
            length = len(row[kind])
        if kind in ("binary_path", "token_ids") and (type(row.get("len")) is not int or row["len"] != length):
            raise ValueError(f"{location}: len does not match shape")
        if kind == "binary_path":
            for key in ("layer", "step", "occurrence"):
                if key not in row:
                    raise ValueError(f"{location}: missing {key}")
                value = row[key]
                if value is not None and (type(value) is not int or value < 0):
                    raise ValueError(f"{location}: invalid {key}")
            expected = occurrences.get(row["name"], 0)
            if row["occurrence"] != expected:
                raise ValueError(f"{location}: occurrence must be {expected}")
            occurrences[row["name"]] = expected + 1
            if not isinstance(row[kind], str) or not row[kind]:
                raise ValueError(f"{location}: missing binary path")
        else:
            values = row[kind]
            if not isinstance(values, list) or len(values) != length:
                raise ValueError(f"{location}: payload length does not match shape")
            if kind == "bool_values":
                valid = all(type(value) is bool for value in values)
            else:
                valid = all(type(value) is int and value >= 0 for value in values)
            if not valid:
                raise ValueError(f"{location}: invalid {kind}")
    return rows


def binary(trace: Path, row: dict) -> bytes:
    path = Path(row["binary_path"])
    if not path.is_absolute():
        path = trace.parent / path
    data = path.read_bytes()
    if len(data) != row["len"] * 4:
        raise ValueError(f"{path}: expected {row['len'] * 4} bytes, got {len(data)}")
    return data


def compare(oracle: Path, native: Path) -> dict:
    left, right = records(oracle), records(native)
    compared_values = 0
    for index, (a, b) in enumerate(zip(left, right)):
        context = {"record": index, "name": a["name"], "shape": a.get("shape")}
        # Missing vs null is intentional: emit the same checkpoint_at/token_ids schema.
        for key in ("name", "shape", "len", "layer", "step", "occurrence"):
            if (key in a) != (key in b) or a.get(key) != b.get(key):
                return {"status": "metadata_mismatch", **context, "field": key,
                        "oracle": a.get(key), "native": b.get(key)}
        kind_a = next(key for key in PAYLOADS if key in a)
        kind_b = next(key for key in PAYLOADS if key in b)
        if kind_a != kind_b:
            return {"status": "payload_mismatch", **context,
                    "oracle": kind_a, "native": kind_b}
        if kind_a == "binary_path":
            a_bytes, b_bytes = binary(oracle, a), binary(native, b)
            compared_values += a["len"]
            if a_bytes != b_bytes:
                byte = next(i for i, pair in enumerate(zip(a_bytes, b_bytes)) if pair[0] != pair[1])
                element = byte // 4
                coordinate, remainder = [], element
                for dimension in reversed(a["shape"]):
                    coordinate.append(remainder % dimension)
                    remainder //= dimension
                return {"status": "bit_mismatch", **context, "element": element,
                        "coordinate": list(reversed(coordinate)),
                        "oracle_u32": struct.unpack_from("<I", a_bytes, element * 4)[0],
                        "native_u32": struct.unpack_from("<I", b_bytes, element * 4)[0]}
        else:
            compared_values += len(a[kind_a])
            if a[kind_a] != b[kind_b]:
                element = next(i for i, pair in enumerate(zip(a[kind_a], b[kind_b])) if pair[0] != pair[1])
                return {"status": "value_mismatch", **context, "element": element,
                        "oracle": a[kind_a][element], "native": b[kind_b][element]}
    if len(left) != len(right):
        return {"status": "record_count_mismatch", "oracle": len(left), "native": len(right)}
    if compared_values == 0:
        raise ValueError("Trace contains no values to compare")
    return {"status": "identical", "records": len(left), "values": compared_values}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("oracle", type=Path)
    parser.add_argument("native", type=Path)
    args = parser.parse_args()
    try:
        result = compare(args.oracle, args.native)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(json.dumps({"status": "invalid_trace", "error": str(exc)}))
        return 2
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "identical" else 1


if __name__ == "__main__":
    raise SystemExit(main())
