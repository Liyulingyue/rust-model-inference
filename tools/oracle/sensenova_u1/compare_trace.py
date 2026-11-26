#!/usr/bin/env python3
"""Diff the Rust port's checkpoint trace against the oracle's `.npy` dumps.

The Rust binary writes raw little-endian f32 under `RMI_SENSENOVA_TRACE=<dir>`
(see `trace_dump` in `src/models/diffusion/sensenova_u1/sampler.rs`); the oracle
writes `.npy` via `make_reference.py`. This script pairs them by name, reshapes
both to the oracle's shape, and reports the first divergence in pipeline order.

Pipeline order matters: the first checkpoint that already differs localises the
bug, so the table is printed in that order rather than sorted.

Usage: compare_trace.py <trace_dir> <ref_dir>
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Checkpoints in pipeline order. `(trace_name, ref_name)`; the ref name defaults
# to the trace name with `.npy`.
ORDER = [
    "noise_init",
    "timesteps",
    "step0_t",
    "step0_t_next",
    "prefix_last_hidden_state_cond",
    "prefix_kv_layer00_k",
    "prefix_kv_layer00_v",
    "step0_z",
    "step0_image_embeds",
    "step0_v_pred_cond",
    "step0_v_pred",
    "step0_z_next",
    "final_image_prediction",
]


def load_trace(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype="<f4")


def compare(name: str, trace_dir: Path, ref_dir: Path) -> tuple[str, str] | None:
    tpath = trace_dir / f"{name}.f32"
    rpath = ref_dir / f"{name}.npy"
    if not tpath.exists():
        return (name, "MISSING in trace")
    if not rpath.exists():
        return (name, "MISSING in reference")
    got = load_trace(tpath).astype(np.float64)
    ref = np.load(rpath).astype(np.float64).reshape(-1)
    if got.size != ref.size:
        return (name, f"LENGTH {got.size} vs {ref.size}")
    diff = np.abs(got - ref)
    scale = max(float(np.abs(ref).max()), 1e-12)
    max_abs = float(diff.max())
    rel = max_abs / scale
    rmse = float(np.sqrt(np.mean((got - ref) ** 2)))
    # A correct port with a different reduction order lands well under 1e-2
    # relative; a structural bug is O(1) relative.
    verdict = "ok" if rel < 1e-2 else "DIVERGED"
    return (name, f"max_abs={max_abs:.6g} rel={rel:.3e} rmse={rmse:.6g} [{verdict}]")


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    trace_dir = Path(sys.argv[1])
    ref_dir = Path(sys.argv[2])

    if not trace_dir.is_dir():
        print(f"trace dir missing: {trace_dir}")
        return 1

    first_bad = None
    for name in ORDER:
        result = compare(name, trace_dir, ref_dir)
        if result is None:
            print(f"{name:<34} (identical names, nothing to do)")
            continue
        label, message = result
        print(f"{label:<34} {message}")
        if first_bad is None and ("DIVERGED" in message or "MISMATCH" in message
                                  or "MISSING" in message or "LENGTH" in message):
            first_bad = label

    print()
    if first_bad is None:
        print("FIRST DIVERGENCE: none — every checkpoint matches within tolerance")
        return 0
    print(f"FIRST DIVERGENCE: {first_bad}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
