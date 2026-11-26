#!/usr/bin/env python3
"""Minimal dependency-free GGUF header dumper (metadata + tensor inventory).

Used to pin the SenseNova-U1.6-8B-MoT contract without loading 20 GB of weights:
it only reads the header region (magic .. end of tensor-info table).

Usage: gguf_dump.py <file.gguf> [--json out.json] [--md out.md]
"""

from __future__ import annotations

import argparse
import json
import struct
import sys

GGUF_MAGIC = 0x46554747  # "GGUF" little-endian

# value type -> (struct fmt, size)
SCALAR = {
    0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2),
    4: ("<I", 4), 5: ("<i", 4), 6: ("<f", 4), 7: ("<?", 1),
    10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8),
}
STRING = 8
ARRAY = 9

GGML_TYPE_NAMES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K",
    14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS",
    19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS",
    24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M",
    30: "BF16", 34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4",
}


class Reader:
    def __init__(self, f):
        self.f = f

    def raw(self, n: int) -> bytes:
        b = self.f.read(n)
        if len(b) != n:
            raise EOFError(f"wanted {n} bytes, got {len(b)}")
        return b

    def scalar(self, vtype: int):
        fmt, size = SCALAR[vtype]
        return struct.unpack(fmt, self.raw(size))[0]

    def string(self) -> str:
        n = struct.unpack("<Q", self.raw(8))[0]
        return self.raw(n).decode("utf-8", errors="replace")

    def value(self, vtype: int):
        if vtype in SCALAR:
            return self.scalar(vtype)
        if vtype == STRING:
            return self.string()
        if vtype == ARRAY:
            etype = struct.unpack("<I", self.raw(4))[0]
            count = struct.unpack("<Q", self.raw(8))[0]
            return [self.value(etype) for _ in range(count)]
        raise ValueError(f"unknown GGUF value type {vtype}")


def dump(path: str) -> dict:
    with open(path, "rb") as f:
        r = Reader(f)
        magic = struct.unpack("<I", r.raw(4))[0]
        if magic != GGUF_MAGIC:
            raise SystemExit(f"not a GGUF file (magic={magic:#x})")
        version = struct.unpack("<I", r.raw(4))[0]
        n_tensors = struct.unpack("<Q", r.raw(8))[0]
        n_kv = struct.unpack("<Q", r.raw(8))[0]

        meta = {}
        for _ in range(n_kv):
            key = r.string()
            vtype = struct.unpack("<I", r.raw(4))[0]
            meta[key] = r.value(vtype)

        tensors = []
        for _ in range(n_tensors):
            name = r.string()
            n_dims = struct.unpack("<I", r.raw(4))[0]
            dims = [struct.unpack("<Q", r.raw(8))[0] for _ in range(n_dims)]
            ttype = struct.unpack("<I", r.raw(4))[0]
            offset = struct.unpack("<Q", r.raw(8))[0]
            tensors.append({
                "name": name,
                "dims": dims,
                "type": GGML_TYPE_NAMES.get(ttype, f"UNKNOWN_{ttype}"),
                "offset": offset,
            })
        data_start = f.tell()

    return {
        "path": path,
        "version": version,
        "tensor_count": n_tensors,
        "kv_count": n_kv,
        "metadata": meta,
        "data_start": data_start,
        "tensors": tensors,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("--json")
    ap.add_argument("--md")
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args()

    d = dump(args.gguf)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(d, f, indent=1, default=str)
        print(f"[json] {args.json}")

    if args.md:
        with open(args.md, "w") as f:
            f.write(f"# GGUF contract: `{d['path']}`\n\n")
            f.write(f"- gguf version: {d['version']}\n")
            f.write(f"- tensors: {d['tensor_count']}\n\n")
            f.write("## Metadata\n\n| key | value |\n| --- | --- |\n")
            for k in sorted(d["metadata"]):
                v = d["metadata"][k]
                s = json.dumps(v, default=str)
                if len(s) > 300:
                    s = s[:300] + " …"
                f.write(f"| `{k}` | `{s}` |\n")
            f.write("\n## Tensors\n\n| name | dims | type |\n| --- | --- | --- |\n")
            for t in d["tensors"]:
                f.write(f"| `{t['name']}` | {t['dims']} | {t['type']} |\n")
        print(f"[md] {args.md}")

    if args.stats or not (args.json or args.md):
        from collections import Counter
        print(f"gguf_version={d['version']} tensors={d['tensor_count']} kv={d['kv_count']}")
        print("types:", dict(Counter(t["type"] for t in d["tensors"])))
        print("\n-- metadata --")
        for k in sorted(d["metadata"]):
            v = d["metadata"][k]
            s = json.dumps(v, default=str)
            print(f"  {k} = {s[:200]}")
        print("\n-- tensor name prefixes --")
        pref = Counter()
        for t in d["tensors"]:
            parts = t["name"].split(".")
            pref[".".join(parts[:2]) if len(parts) > 2 else t["name"]] += 1
        for k, v in sorted(pref.items()):
            print(f"  {k}: {v}")


if __name__ == "__main__":
    sys.exit(main())
