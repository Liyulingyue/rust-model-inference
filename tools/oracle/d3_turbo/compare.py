#!/usr/bin/env python3
"""Compare checkpoints with fixed floating/lossy gates, never a bitwise lossy gate.

Float: atol=5e-4, rtol=1e-4 (the existing D3 F32 gate).
CPU block activations: relative RMSE<=2%, cosine>=0.9998; logits additionally
atol=0.05, rtol=0.01 and identical argmax. The envelope accounts for 8-bit
activation rounding discontinuities over 64 layers, not weight quality loss.
Tokens, positions and checkpoint order/shapes always remain exact contracts.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np


def compare(oracle, native, lossy=False):
    def records(path):
        return [r for line in path.read_text().splitlines() if (r := json.loads(line)).get("name", "").startswith("vela3.d3.")]
    expected, actual = records(oracle), records(native)
    if not expected or not actual or expected[-1]["name"] != "vela3.d3.logits" or actual[-1]["name"] != "vela3.d3.logits":
        raise ValueError("empty or incomplete decision trace")
    if len(expected) != len(actual):
        raise ValueError(f"checkpoint count differs: {len(expected)} vs {len(actual)}")
    checks = []
    for a, b in zip(expected, actual):
        if (a["name"], a.get("layer"), a["shape"]) != (b["name"], b.get("layer"), b["shape"]):
            raise ValueError("checkpoint order/shape differs")
        if "token_ids" in a:
            if a["token_ids"] != b.get("token_ids"):
                raise ValueError("token IDs differ")
            continue
        x, y = (np.fromfile(r["binary_path"], dtype="<f4").astype(np.float64) for r in (a, b))
        if x.size != a["len"] or y.size != b["len"] or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("invalid checkpoint payload")
        rmse = float(np.sqrt(np.mean((x-y)**2)))
        relative = rmse / max(float(np.sqrt(np.mean(x*x))), 1e-30)
        cosine = float(np.sum(x*y) / max(np.sqrt(np.sum(x*x)*np.sum(y*y)), 1e-30))
        exact = a["name"].endswith("positions")
        passed = bool(np.array_equal(x,y)) if exact else bool(np.allclose(x,y,atol=5e-4,rtol=1e-4))
        if lossy and not exact:
            passed = relative <= 0.02 and (cosine >= 0.9998 or np.array_equal(x,y))
            if a["name"].endswith("logits"):
                passed = passed and bool(np.allclose(x,y,atol=0.05,rtol=0.01)) and x.argmax() == y.argmax()
        checks.append(dict(name=a["name"],layer=a.get("layer"),passed=bool(passed),max_abs=float(np.max(np.abs(x-y),initial=0)),rmse=rmse,relative_rmse=relative,cosine=cosine))
    return dict(passed=all(r["passed"] for r in checks),profile="cpu-block-activations" if lossy else "float",checks=checks)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("oracle",type=Path)
    p.add_argument("native",type=Path)
    p.add_argument("--lossy",action="store_true")
    a=p.parse_args()
    result=compare(a.oracle,a.native,a.lossy)
    print(json.dumps(result,indent=2))
    sys.exit(0 if result["passed"] else 1)
