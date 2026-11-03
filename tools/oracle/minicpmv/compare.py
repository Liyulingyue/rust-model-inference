#!/usr/bin/env python3
"""Compare the Rust `minicpmv` projector dumps against the pinned oracle's.

Usage:
    # 1. oracle (after applying minicpmv-parity-dump.patch)
    RMI_MTMD_DUMP_DIR=$DUMP \
      llama-mtmd-cli -m MiniCPM-V-4_7-1B-Q8_0.gguf \
                     --mmproj mmproj-model-f16.gguf \
                     --image img.png -p "x" -n 1 --temp 0 --jinja
    # 2. rust
    RMI_MINICPMV_MMPROJ=mmproj-model-f16.gguf \
    RMI_MINICPMV_IMAGE=img.png \
    RMI_MINICPMV_DUMP=$DUMP_RUST \
      cargo test --profile release-fast --test minicpmv_reference \
                 dump_tile -- --ignored --nocapture
    # 3. compare
    ./compare.py $DUMP $DUMP_RUST

Exit code 0 means every compared element is bit-identical.
"""

import struct
import sys


def read_f32(path):
    with open(path, "rb") as handle:
        data = handle.read()
    return list(struct.unpack("<%df" % (len(data) // 4), data))


def bits(values):
    return [struct.unpack("<I", struct.pack("<f", v))[0] for v in values]


def compare_pixels(oracle_dir, rust_dir):
    a = read_f32(f"{oracle_dir}/pixels_0.bin")
    b = read_f32(f"{rust_dir}/pixels_0.bin")
    if len(a) != len(b):
        print(f"pixels: length {len(a)} vs {len(b)}")
        return False
    bad = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    print(f"pixels: {len(a)} values, {len(bad)} differ")
    if bad:
        i = bad[0]
        print(f"  first at {i}: oracle={a[i]!r} rust={b[i]!r}")
    return not bad


def compare_tile(oracle_dir, rust_dir):
    # The oracle's file carries the trailing separator row(s) on top of the
    # tile's own tokens; compare the tile part only.
    a = read_f32(f"{oracle_dir}/tile_0.bin")
    b = read_f32(f"{rust_dir}/tile_0_nosuffix.bin")
    n = min(len(a), len(b))
    # Convert once: recomputing inside the loop would be quadratic.
    ba, bb = bits(a[:n]), bits(b[:n])
    bad = [i for i in range(n) if ba[i] != bb[i]]
    print(f"projector: {n} values compared, {len(bad)} differ")
    if bad:
        i = bad[0]
        print(f"  first at {i}: oracle={a[i]!r} rust={b[i]!r}")
    return not bad


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    oracle_dir, rust_dir = sys.argv[1], sys.argv[2]
    ok_pixels = compare_pixels(oracle_dir, rust_dir)
    ok_tile = compare_tile(oracle_dir, rust_dir)
    print("PIXELS   " + ("OK" if ok_pixels else "DIFFER"))
    print("PROJECTOR " + ("OK" if ok_tile else "DIFFER"))
    return 0 if (ok_pixels and ok_tile) else 1


if __name__ == "__main__":
    sys.exit(main())