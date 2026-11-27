#!/usr/bin/env python3
"""Generate the transposed twin of a tiled Q8_0 kernel.

The tiled kernels read the weight in ggml's own layout: one output row
after the other, every block 34 bytes long. A lane owns one output
column, so the 64 lanes of a workgroup touch 64 rows that are
`blocks_per_row * 34` bytes apart, and every 4-byte payload load
straddles a block boundary.

The transposed layout groups the rows of a tile of 64 output rows
instead, so the payload of one block is one contiguous run of 64 rows
and the loads are aligned. The arithmetic does not change: the same
scale and the same 32 payload bytes are read, in the same order.

This script writes the twin of a tiled kernel so the two stay in step.
Run it from the repository root:

    python3 scripts/transposed-tiled-shader.py

It rewrites the generated files in place and is idempotent.
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
GLSL = ROOT / "shaders" / "glsl"

SOURCES = [
    "q8_matmul_tiled_dp4a",
    "q8_matmul_tiled_dp4a_wide",
    "q8_matmul_tiled_dp4a_wide_add",
]

# The addressing of the native layout, and its transposed counterpart.
#
# Native: row `r` holds `blocks_per_row` blocks of 34 bytes, so the
# block `b` of row `r` starts at `r * blocks_per_row * 34 + b * 34`,
# the scale is the first two bytes of the block and the payload follows.
#
# Transposed: a tile of 64 rows holds, for every block, the 32 payload
# bytes of each of the 64 rows (64 * 32 bytes) and then the 64 scales
# (64 * 2 bytes), so the block `b` of row `r` has its payload at
# `tile_base + b * 64 * 34 + (r % 64) * 32` and its scale at
# `tile_base + b * 64 * 34 + 64 * 32 + (r % 64) * 2`.
NATIVE_ROW = """    uint weight_row = read_column * pc.blocks_per_row * 34u;"""

TRANSPOSED_ROW = """    // The transposed layout groups the rows of a tile of 64 output rows,
    // so the payload of one block is one contiguous run and the 4-byte
    // loads are aligned. The values are the ones ggml stored, only moved.
    uint tile = read_column / TILE_ROWS;
    uint tile_row = read_column % TILE_ROWS;
    uint tile_bytes = pc.blocks_per_row * TILE_ROWS * BLOCK_BYTES;
    uint tile_base = tile * tile_bytes;
    uint payload_base = tile_base + tile_row * PAYLOAD_BYTES;
    uint scale_base = tile_base + TILE_ROWS * PAYLOAD_BYTES + tile_row * 2u;"""

NATIVE_BLOCK = """            uint byte_offset = weight_row + block * 34u;
            uint scale_word = weight_word(slot, byte_offset >> 2u);
            uint scale_bits = (byte_offset & 3u) == 0u
                ? scale_word & 0xffffu
                : (scale_word >> 16u) & 0xffffu;
            float weight_scale = f16_decode(scale_bits);
            uint body = byte_offset + 2u;"""

TRANSPOSED_BLOCK = """            uint payload_byte = payload_base + block * TILE_ROWS * BLOCK_BYTES;
            uint scale_byte = scale_base + block * TILE_ROWS * BLOCK_BYTES;
            uint scale_word = weight_word(slot, scale_byte >> 2u);
            uint scale_shift = (scale_byte & 3u) * 8u;
            uint scale_bits = scale_shift == 0u
                ? scale_word & 0xffffu
                : ((scale_word >> scale_shift) & 0xffffu);
            float weight_scale = f16_decode(scale_bits);"""

NATIVE_PACKED = """                uint group_offset = body + g * 4u;
                uint first = weight_word(slot, group_offset >> 2u);
                uint shift = (group_offset & 3u) * 8u;
                if (shift == 0u) {
                    packed[g] = first;
                } else {
                    uint second = weight_word(slot, (group_offset >> 2u) + 1u);
                    packed[g] = (first >> shift) | (second << (32u - shift));
                }"""

TRANSPOSED_PACKED = """                // The payload of a block starts on a 4-byte boundary, so
                // the eight words are read one by one.
                packed[g] = weight_word(slot, (payload_byte >> 2u) + g);"""

CONSTANTS = """const uint TILE_ROWS = 64u;
const uint PAYLOAD_BYTES = 32u;
const uint BLOCK_BYTES = 34u;
"""


def transform(source: str) -> str:
    text = source
    for old, new in (
        (NATIVE_ROW, TRANSPOSED_ROW),
        (NATIVE_BLOCK, TRANSPOSED_BLOCK),
        (NATIVE_PACKED, TRANSPOSED_PACKED),
    ):
        if old not in text:
            raise SystemExit(f"pattern not found:\n{old}")
        text = text.replace(old, new, 1)
    # The layout constants go next to the other compile-time ones.
    anchor = "const uint CHUNK_BLOCKS ="
    if anchor not in text:
        raise SystemExit("CHUNK_BLOCKS anchor not found")
    text = text.replace(anchor, CONSTANTS + anchor, 1)
    return text


def main() -> int:
    for name in SOURCES:
        source = (GLSL / f"{name}.comp").read_text()
        target = GLSL / f"{name}_transposed.comp"
        header = (
            "// Generated from "
            f"{name}.comp by scripts/transposed-tiled-shader.py.\n"
            "// Do not edit this file; edit the source and re-run the script.\n"
            "//\n"
            "// This is the same kernel as the source, reading the *transposed*\n"
            "// Q8_0 layout: the rows of a tile of 64 output rows are neighbours,\n"
            "// so a warp reads one contiguous run instead of 64 sectors one row\n"
            "// apart. The arithmetic is unchanged, so the result is bit for bit\n"
            "// the one the native layout produces.\n"
        )
        target.write_text(header + transform(source))
        print(f"wrote {target.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
