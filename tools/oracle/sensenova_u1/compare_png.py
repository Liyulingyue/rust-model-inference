#!/usr/bin/env python3
"""Compare two PNGs produced by the Rust port and the Python oracle.

Reports the metrics the contract asks for (max abs diff, RMSE, mean abs diff,
PSNR, and the share of pixels whose 8-bit channels differ at all), and exits
non-zero when the image is not within the acceptance thresholds.

Thresholds are deliberately stated here rather than chosen after seeing a
result: both sides run the same weights and the same f32 math, but the linear
kernels accumulate in a different order (the oracle uses torch's BLAS-backed
GEMV, the Rust side uses the repo's quantized kernels), so the two are not
expected to be bit-identical. A correct port lands far inside these bounds; a
structural bug (wrong RoPE pairing, wrong patch order, wrong timestep schedule)
blows past them by orders of magnitude.

Usage:
  compare_png.py <rust.png> <oracle.png> [--max-abs 16] [--psnr 30]
"""

from __future__ import annotations

import argparse
import math
import sys

import numpy as np
from PIL import Image

# 8-bit RGB. A per-channel max deviation of 16 levels (~6% of range) on a
# 4-step, CFG 4.0 sample is roughly what f32 reduction-order noise produces
# through 42 dual-branch layers; anything larger means a real divergence.
DEFAULT_MAX_ABS = 16
# 30 dB is a loose floor: it passes a port with correct structure and normal
# f32 noise, and fails one with a wrong patch layout or RoPE pairing.
DEFAULT_PSNR = 30.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("rust_png")
    ap.add_argument("oracle_png")
    ap.add_argument("--max-abs", type=float, default=DEFAULT_MAX_ABS)
    ap.add_argument("--psnr", type=float, default=DEFAULT_PSNR)
    ap.add_argument("--save-diff", default=None)
    args = ap.parse_args()

    a = np.asarray(Image.open(args.rust_png).convert("RGB"), dtype=np.float64)
    b = np.asarray(Image.open(args.oracle_png).convert("RGB"), dtype=np.float64)
    if a.shape != b.shape:
        print(f"SHAPE MISMATCH: {a.shape} vs {b.shape}")
        return 1

    diff = np.abs(a - b)
    mse = float(np.mean((a - b) ** 2))
    rmse = math.sqrt(mse)
    psnr = float("inf") if mse == 0.0 else 10.0 * math.log10((255.0**2) / mse)
    differing = float(np.mean(diff > 0.0)) * 100.0
    max_abs = float(diff.max())

    print(f"shape        {a.shape}")
    print(f"max_abs      {max_abs:.1f}   (threshold {args.max_abs})")
    print(f"mean_abs     {diff.mean():.4f}")
    print(f"rmse         {rmse:.4f}")
    print(f"psnr         {psnr:.2f} dB (threshold {args.psnr})")
    print(f"pixels_differing {differing:.2f} %")

    if args.save_diff:
        Image.fromarray(diff.clip(0, 255).astype(np.uint8)).save(args.save_diff)
        print(f"diff -> {args.save_diff}")

    ok = max_abs <= args.max_abs and psnr >= args.psnr
    print("RESULT", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
