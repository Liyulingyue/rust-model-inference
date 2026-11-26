#!/usr/bin/env python3
"""Compare the oracle's ``state_dict()`` naming against the GGUF tensor inventory.

The Rust port must prove it covers every tensor the official loader touches, so
this reports the three-way diff:

  * ``missing_from_gguf``  - state_dict keys with no GGUF tensor of that name
  * ``extra_in_gguf``      - GGUF tensors the model never asks for
  * ``shape_mismatch``     - same name, different element count

GGUF names come from ``gguf_dump.py --json``; state_dict names from
``make_reference.py``'s ``state_dict_keys.json``.

GGUF stores the *quantized* payload, so a Q8_0 weight of logical shape
``[out, in]`` appears as ``[in, out]`` in the tensor info and its element count
is ``out * in`` only after dequantization; therefore element counts are compared
against the product of the GGUF dims, which is what ``gguf`` reports.

Usage:
  python3 compare_naming.py state_dict_keys.json gguf.json [--md out.md]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def load_state_dict(path: Path) -> dict[str, list[int]]:
    doc = json.loads(path.read_text())
    out = {}
    for row in doc["state_dict"]:
        shape = row.get("shape")
        out[row["name"]] = shape
    return out


def load_gguf(path: Path) -> dict[str, list[int]]:
    doc = json.loads(path.read_text())
    return {t["name"]: list(t["dims"]) for t in doc["tensors"]}


def prefix_of(name: str, depth: int = 2) -> str:
    parts = name.split(".")
    return ".".join(parts[:depth]) if len(parts) > depth else name


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("state_dict")
    ap.add_argument("gguf")
    ap.add_argument("--md")
    args = ap.parse_args()

    sd = load_state_dict(Path(args.state_dict))
    gg = load_gguf(Path(args.gguf))

    sd_names, gg_names = set(sd), set(gg)
    missing = sorted(sd_names - gg_names)
    extra = sorted(gg_names - sd_names)
    common = sorted(sd_names & gg_names)

    # GGUF dims are stored reversed relative to torch (ggml is column-major-ish),
    # so compare the product, which is order independent.
    def prod(dims):
        n = 1
        for d in dims:
            n *= int(d)
        return n

    shape_mismatch = []
    for name in common:
        a, b = sd[name], gg[name]
        if a is None:
            continue
        if prod(a) != prod(b):
            shape_mismatch.append({"name": name, "state_dict": a, "gguf": b,
                                   "numel_state_dict": prod(a), "numel_gguf": prod(b)})

    sd_pref = Counter(prefix_of(n) for n in sd_names)
    gg_pref = Counter(prefix_of(n) for n in gg_names)

    lines = []
    w = lines.append
    w(f"# GGUF vs oracle state_dict naming\n")
    w(f"- state_dict keys: **{len(sd_names)}**")
    w(f"- GGUF tensors: **{len(gg_names)}**")
    w(f"- common: {len(common)}  missing from GGUF: **{len(missing)}**  extra in GGUF: **{len(extra)}**")
    w(f"- element-count mismatches on common names: {len(shape_mismatch)}\n")

    w("## Top-level / 2-level prefixes\n")
    w("| prefix | state_dict | gguf |")
    w("| --- | --- | --- |")
    for p in sorted(set(sd_pref) | set(gg_pref)):
        w(f"| `{p}` | {sd_pref.get(p, 0)} | {gg_pref.get(p, 0)} |")

    w("\n## In state_dict but NOT in GGUF\n")
    if missing:
        for n in missing:
            w(f"- `{n}` {sd[n]}")
    else:
        w("_none_")

    w("\n## In GGUF but NOT in state_dict\n")
    if extra:
        for n in extra:
            w(f"- `{n}` {gg[n]}")
    else:
        w("_none_")

    w("\n## Element-count mismatches\n")
    if shape_mismatch:
        w("| name | state_dict | gguf |")
        w("| --- | --- | --- |")
        for m in shape_mismatch:
            w(f"| `{m['name']}` | {m['state_dict']} ({m['numel_state_dict']}) | {m['gguf']} ({m['numel_gguf']}) |")
    else:
        w("_none_")

    text = "\n".join(lines) + "\n"
    if args.md:
        Path(args.md).write_text(text)
        print(f"[md] {args.md}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
