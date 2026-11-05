//! The transposed Q8_0 weight layout.
//!
//! # The problem
//!
//! ggml stores a Q8_0 matrix as one row per output, and each row holds
//! `n_in / 32` blocks of 34 bytes: an f16 scale followed by 32 int8 values.
//!
//! The device kernel that reads such a matrix gives every lane its own
//! output row and walks the blocks of that row. A warp therefore reads 32
//! sectors that are one output row apart -- 32 separate 32-byte pieces
//! spread over 78 KB -- and every 4-byte payload load straddles the
//! 34-byte block boundary, so it has to be assembled from two words. On
//! the GB10 box that reaches 60 GB/s, while the device itself streams
//! 171 GB/s over the same bytes (a plain element-wise read of one big
//! buffer) and PyTorch CUDA reaches 130 GB/s on the same model. The
//! kernel was losing to its own access pattern, not to the memory system.
//!
//! # The layout
//!
//! The transposed layout groups the rows of a *tile* of 64 output rows
//! instead of the blocks of one row:
//!
//! ```text
//! for every tile of 64 output rows:
//!     for every block b of the row:
//!         payload: the 32 int8 of every row of the tile   (64 * 32 bytes)
//!         scales:  the f16 scale of every row of the tile  (64 *  2 bytes)
//! ```
//!
//! The tile width is 64 because that is how wide a workgroup is: the
//! kernels give one lane per output row, so a workgroup owns 64
//! consecutive rows, and in this layout those rows are neighbours.
//!
//! Two details matter for the bandwidth:
//!
//! * The payload of a block is one contiguous run of 64 rows, so the 32
//!   lanes of a warp read one contiguous 1 KB run instead of 32 sectors
//!   one row apart.
//! * The payload starts at a 32-byte boundary, so the eight 4-byte loads
//!   of a block need no shifting and no second word.
//!
//! The scales follow the payload of the same block rather than sitting in
//! one region of their own: that keeps a workgroup on a single memory
//! stream per block instead of two far apart ones. (Grouping the scales
//! per tile instead was measured too and reads the same bytes, but the
//! block-local form is what the shader addresses in one add.)
//!
//! # What it does not change
//!
//! The transpose moves bytes; it computes nothing. Every block keeps its
//! scale and its 32 payload bytes, so the layout is a permutation of the
//! input and can be undone -- `transposed_layout_round_trips_to_the_same_bytes`
//! checks that. The kernel that reads the layout keeps the native
//! reduction order, so the device result is bit-for-bit the native one:
//! `vulkan_q8_reduction_modes_match_golden_bits` runs it against the same
//! hard-coded IEEE-754 bits as the native kernels, in all three reduction
//! modes.
//!
//! # Where it applies
//!
//! Only to the one-row kernel. The tiled kernel reads a whole tile of
//! rows as one run in either layout and is limited by the shared-memory
//! staging of the activations, not by the weight access -- transposing
//! there measured 0.60x, i.e. slower. So the tiled path keeps ggml's
//! layout and the one-row path can use this one.
//!
//! # The cost
//!
//! Rewriting a weight takes about 26 ms per 240 MB (9 GB/s, a gather and
//! a scatter, not a copy) and happens once, when the weight is uploaded.
//! The transposed bytes are written straight into the device buffer, so
//! the upload does not need a second staging buffer. The device buffer is
//! exactly as large as the native one, up to the padding of the last tile.
//!
//! # Why it is opt-in
//!
//! `RMI_GPU_Q8_TRANSPOSED=1` turns it on. The layout cannot change a
//! result, but a mistake in its addressing would read past the weight and
//! fault the device, and no Mellum projection runs on the device yet --
//! so there is nothing to gain from turning it on before the trunk can
//! hand the device whole projections. See
//! [`crate::vulkan::ops::q8_transposed_enabled`].

/// Bytes of one Q8_0 block: an f16 scale and 32 int8 values.
pub(crate) const BLOCK_BYTES: usize = 34;

/// Bytes of the int8 payload of one Q8_0 block.
pub(crate) const PAYLOAD_BYTES: usize = 32;

/// Output rows one tile of the transposed layout holds.
///
/// The tile width is the width of a workgroup: the kernels give one lane
/// per output row, and a workgroup covers 64 of them.
pub(crate) const TILE_ROWS: usize = 64;

/// Blocks one staging tile of the copy covers.
const BLOCK_TILE: usize = 8;

/// Rewrite a Q8_0 weight matrix into the transposed device layout.
///
/// `weight` holds `n_out` rows of `n_in / 32` blocks in ggml's own layout.
/// The result holds the same bytes regrouped by tile, so it is as long as
/// the input plus the padding of the last tile.
///
/// The copy is tiled: a staging tile is `BLOCK_TILE` blocks of `TILE_ROWS`
/// output rows. Gathering a tile reads one contiguous run per row and
/// scattering it writes one contiguous run per block, so both sides of the
/// copy walk memory in long runs instead of 32-byte steps. That is what
/// makes the copy run at 9 GB/s instead of the 0.9 GB/s of a plain
/// 34-byte-at-a-time loop, which is dominated by the per-copy overhead.
///
/// The result is a permutation of the input: the scale and the payload of
/// every block are moved, never rewritten.
pub(crate) fn transpose_q8_0_weight(weight: &[u8], n_in: usize, n_out: usize) -> Vec<u8> {
    let mut packed = vec![0u8; transposed_size(n_in, n_out)];
    transpose_q8_0_weight_into(&mut packed, weight, n_in, n_out);
    packed
}

/// The length of the transposed layout of an `n_in` by `n_out` matrix.
///
/// The layout pads the last tile to a whole 64 rows, so the length is the
/// input length rounded up to a tile, and never shorter.
pub(crate) fn transposed_size(n_in: usize, n_out: usize) -> usize {
    let blocks = n_in / PAYLOAD_BYTES;
    n_out.div_ceil(TILE_ROWS) * blocks * TILE_ROWS * BLOCK_BYTES
}

/// Rewrite a Q8_0 weight matrix into the transposed device layout, in place.
///
/// `packed` has to be [`transposed_size`] bytes long. Writing the layout
/// straight into the device buffer saves a second copy of the weight: the
/// upload would otherwise stage the transposed bytes in a temporary
/// buffer and copy them over, which for a 240 MB weight is another
/// allocation and another 240 MB of traffic.
///
/// The addressing mirrors what the shaders do:
///
/// ```text
/// tile      = row / 64
/// tile_row  = row % 64
/// tile_base = tile * blocks * 64 * 34
/// payload   = tile_base + block * 64 * 34 + tile_row * 32
/// scale     = tile_base + block * 64 * 34 + 64 * 32 + tile_row * 2
/// ```
pub(crate) fn transpose_q8_0_weight_into(
    packed: &mut [u8],
    weight: &[u8],
    n_in: usize,
    n_out: usize,
) {
    let blocks = n_in / PAYLOAD_BYTES;
    let tiles = n_out.div_ceil(TILE_ROWS);
    let tile_bytes = blocks * TILE_ROWS * BLOCK_BYTES;
    if blocks == 0 || n_out == 0 {
        return;
    }
    // The staging tile holds the gathered blocks of one output row each.
    let mut tile = vec![0u8; TILE_ROWS * BLOCK_TILE * BLOCK_BYTES];
    for tile_index in 0..tiles {
        let first = tile_index * TILE_ROWS;
        let rows = (n_out - first).min(TILE_ROWS);
        let tile_base = tile_index * tile_bytes;
        for block_base in (0..blocks).step_by(BLOCK_TILE) {
            let count = (blocks - block_base).min(BLOCK_TILE);
            let stride = count * BLOCK_BYTES;
            // Gather: one contiguous run per output row.
            for index in 0..rows {
                let source = (first + index) * blocks * BLOCK_BYTES + block_base * BLOCK_BYTES;
                tile[index * stride..index * stride + stride]
                    .copy_from_slice(&weight[source..source + stride]);
            }
            // Scatter: the payload of one block is one run, and so are its
            // scales.
            for block in 0..count {
                // The scales of a block follow its payload, so a workgroup
                // reads one contiguous run per block.
                let block_base_byte = tile_base + (block_base + block) * TILE_ROWS * BLOCK_BYTES;
                let block_payload = block_base_byte;
                let block_scale = block_base_byte + TILE_ROWS * PAYLOAD_BYTES;
                for index in 0..rows {
                    let source = index * stride + block * BLOCK_BYTES;
                    let target = block_payload + index * PAYLOAD_BYTES;
                    packed[target..target + PAYLOAD_BYTES]
                        .copy_from_slice(&tile[source + 2..source + BLOCK_BYTES]);
                    let target = block_scale + index * 2;
                    packed[target..target + 2].copy_from_slice(&tile[source..source + 2]);
                }
            }
        }
    }
}

/// Read one weight element out of the transposed layout.
///
/// The device kernels walk the layout the same way, so this is the host
/// mirror of their addressing.
#[cfg(test)]
fn transposed_element(
    packed: &[u8],
    n_in: usize,
    n_out: usize,
    out: usize,
    index: usize,
) -> (u16, u8) {
    let blocks = n_in / PAYLOAD_BYTES;
    let tile = out / TILE_ROWS;
    let row = out % TILE_ROWS;
    let tile_base = tile * blocks * TILE_ROWS * BLOCK_BYTES;
    let block = index / PAYLOAD_BYTES;
    let lane = index % PAYLOAD_BYTES;
    let block_base = tile_base + block * TILE_ROWS * BLOCK_BYTES;
    let payload = block_base + row * PAYLOAD_BYTES + lane;
    let scale = block_base + TILE_ROWS * PAYLOAD_BYTES + row * 2;
    let scale_bits = u16::from_le_bytes([packed[scale], packed[scale + 1]]);
    (scale_bits, packed[payload])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The straight-line transpose, kept as the baseline of the tiled one.
    fn transpose_q8_0_weight_naive(weight: &[u8], n_in: usize, n_out: usize) -> Vec<u8> {
        let blocks = n_in / PAYLOAD_BYTES;
        let payload_bytes = blocks * n_out * PAYLOAD_BYTES;
        let mut packed = vec![0u8; payload_bytes + blocks * n_out * 2];
        for block in 0..blocks {
            let payload_base = block * n_out * PAYLOAD_BYTES;
            let scale_base = payload_bytes + block * n_out * 2;
            for out in 0..n_out {
                let source = out * blocks * BLOCK_BYTES + block * BLOCK_BYTES;
                let target = payload_base + out * PAYLOAD_BYTES;
                packed[target..target + PAYLOAD_BYTES]
                    .copy_from_slice(&weight[source + 2..source + BLOCK_BYTES]);
                packed[scale_base + out * 2..scale_base + out * 2 + 2]
                    .copy_from_slice(&weight[source..source + 2]);
            }
        }
        packed
    }

    /// One Q8_0 row: `blocks` blocks of an f16 scale and 32 payload bytes.
    fn native_row(row: usize, blocks: usize) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(blocks * BLOCK_BYTES);
        for block in 0..blocks {
            let scale = ((row * 37 + block * 11) & 0xffff) as u16;
            bytes.extend_from_slice(&scale.to_le_bytes());
            for lane in 0..PAYLOAD_BYTES {
                bytes.push(((row * 13 + block * 7 + lane) & 0xff) as u8);
            }
        }
        bytes
    }

    #[test]
    fn transposed_layout_holds_the_same_values() {
        let n_in = 32 * 5;
        let n_out = 130;
        let blocks = n_in / PAYLOAD_BYTES;
        let mut native = Vec::new();
        for out in 0..n_out {
            native.extend_from_slice(&native_row(out, blocks));
        }
        let packed = transpose_q8_0_weight(&native, n_in, n_out);
        // Two full tiles and a half of a third.
        assert_eq!(packed.len(), 3 * blocks * TILE_ROWS * BLOCK_BYTES);
        for out in 0..n_out {
            for index in 0..n_in {
                let (scale, payload) = transposed_element(&packed, n_in, n_out, out, index);
                let block = index / PAYLOAD_BYTES;
                let lane = index % PAYLOAD_BYTES;
                let source = out * blocks * BLOCK_BYTES + block * BLOCK_BYTES;
                let expected_scale = u16::from_le_bytes([native[source], native[source + 1]]);
                assert_eq!(scale, expected_scale, "scale of ({out}, {index})");
                assert_eq!(
                    payload,
                    native[source + 2 + lane],
                    "payload of ({out}, {index})"
                );
            }
        }
    }

    /// The transpose is a permutation of the bytes, not a rewrite of them.
    ///
    /// This is the structural half of the correctness argument: if the
    /// layout can be undone, the device kernel reads exactly the values
    /// ggml stored, so the device result cannot differ from the CPU one
    /// because of the layout.
    #[test]
    fn transposed_layout_round_trips_to_the_same_bytes() {
        for (n_in, n_out) in [
            (32, 1),
            (32, 63),
            (32, 64),
            (32, 65),
            (64, 127),
            (32 * 5, 130),
        ] {
            let blocks = n_in / PAYLOAD_BYTES;
            let mut native = Vec::new();
            for out in 0..n_out {
                native.extend_from_slice(&native_row(out, blocks));
            }
            let packed = transpose_q8_0_weight(&native, n_in, n_out);
            // Undo the layout tile by tile.
            let mut restored = vec![0u8; n_out * blocks * BLOCK_BYTES];
            for tile in 0..n_out.div_ceil(TILE_ROWS) {
                let first = tile * TILE_ROWS;
                let rows = (n_out - first).min(TILE_ROWS);
                let tile_base = tile * blocks * TILE_ROWS * BLOCK_BYTES;
                for block in 0..blocks {
                    let block_base = tile_base + block * TILE_ROWS * BLOCK_BYTES;
                    for row in 0..rows {
                        let out = first + row;
                        let source = out * blocks * BLOCK_BYTES + block * BLOCK_BYTES;
                        restored[source] = packed[block_base + TILE_ROWS * PAYLOAD_BYTES + row * 2];
                        restored[source + 1] =
                            packed[block_base + TILE_ROWS * PAYLOAD_BYTES + row * 2 + 1];
                        restored[source + 2..source + BLOCK_BYTES].copy_from_slice(
                            &packed[block_base + row * PAYLOAD_BYTES
                                ..block_base + row * PAYLOAD_BYTES + PAYLOAD_BYTES],
                        );
                    }
                }
            }
            assert_eq!(restored, native, "n_in={n_in} n_out={n_out}");
        }
    }

    #[test]
    fn transposed_layout_survives_a_partial_tile() {
        // 70 output rows and 3 blocks do not fill a tile, so the edges of
        // the tile loop carry the layout across.
        let n_in = 32 * 3;
        let n_out = 70;
        let blocks = n_in / PAYLOAD_BYTES;
        let mut native = Vec::new();
        for out in 0..n_out {
            native.extend_from_slice(&native_row(out, blocks));
        }
        let packed = transpose_q8_0_weight(&native, n_in, n_out);
        assert_eq!(packed.len(), 2 * blocks * TILE_ROWS * BLOCK_BYTES);
        for out in 0..n_out {
            let (scale, payload) = transposed_element(&packed, n_in, n_out, out, 0);
            let source = out * blocks * BLOCK_BYTES;
            assert_eq!(
                scale,
                u16::from_le_bytes([native[source], native[source + 1]])
            );
            assert_eq!(payload, native[source + 2]);
        }
    }

    /// How long the transpose takes on the size that matters: the 240 MB
    /// output head of the 12B Mellum checkpoint.
    ///
    /// The transpose runs once per weight, when the weight is uploaded, so
    /// it has to stay well below the time the kernel saves.
    #[test]
    #[ignore = "timing harness, not a correctness gate"]
    fn transpose_cost() {
        use std::time::Instant;
        let n_in = 2304;
        let n_out = 98_304;
        let blocks = n_in / PAYLOAD_BYTES;
        let weight = vec![7u8; n_out * blocks * BLOCK_BYTES];
        let bytes = weight.len() as f64;
        // The first run pays for the page faults of a fresh 240 MB
        // allocation, so the steady-state cost is what a warm cache shows.
        let naive = transpose_q8_0_weight_naive(&weight, n_in, n_out);
        let tiled = transpose_q8_0_weight(&weight, n_in, n_out);
        assert_eq!(naive.len(), tiled.len());
        let mut naive_time = f64::MAX;
        let mut tiled_time = f64::MAX;
        for _ in 0..3 {
            let start = Instant::now();
            transpose_q8_0_weight_naive(&weight, n_in, n_out);
            naive_time = naive_time.min(start.elapsed().as_secs_f64());
            let start = Instant::now();
            transpose_q8_0_weight(&weight, n_in, n_out);
            tiled_time = tiled_time.min(start.elapsed().as_secs_f64());
        }
        eprintln!(
            "[transpose] {:.0} MB: naive {:.1} ms ({:.1} GB/s), tiled {:.1} ms ({:.1} GB/s)",
            bytes / 1e6,
            naive_time * 1e3,
            bytes / naive_time / 1e9,
            tiled_time * 1e3,
            bytes / tiled_time / 1e9,
        );
    }
}
