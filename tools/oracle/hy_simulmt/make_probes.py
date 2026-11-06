#!/usr/bin/env python3
"""Generate identical F32 vectors for real Q4_K/Q6_K/F32 projection checks."""
import json
import math
from pathlib import Path
import struct
import sys

root = Path(sys.argv[1])
root.mkdir(parents=True, exist_ok=True)
for width in (768, 2048, 4096):
    values = [0.7 * math.sin(i * 0.173) + 0.3 * math.cos(i * 0.037) for i in range(width)]
    (root / f"input-{width}.f32").write_bytes(struct.pack(f"<{width}f", *values))
cases = []
for layer in (1, 47):
    for role in ("gate", "up", "down"):
        for expert in (0, 127, None):
            cases.append(dict(
                tensor=f"blk.{layer}.ffn_{role}_{'shexp' if expert is None else 'exps'}.weight",
                expert=0 if expert is None else expert,
                input=f"input-{768 if role == 'down' else 2048}.f32",
            ))
for name, width in (("blk.0.attn_q.weight", 2048),
                    ("blk.47.attn_output.weight", 4096),
                    ("blk.1.ffn_gate_inp.weight", 2048)):
    cases.append(dict(tensor=name, expert=0, input=f"input-{width}.f32"))
assert len(cases) == 21 and len({c["tensor"] + str(c["expert"]) for c in cases}) == 21
(root / "projections.json").write_text(json.dumps(cases, indent=2) + "\n")
