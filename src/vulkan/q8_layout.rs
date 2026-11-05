//! The transposed Q8_0 weight layout.
//!
//! ggml stores a Q8_0 matrix as one row per output, and each row holds
//! `n_in / 32` blocks of 34 bytes: an f16 scale followed by 32 int8 values.
//! A device kernel that gives every lane its own output row then reads 32
//! sectors one row apart, and every 4-byte payload load straddles the
//! 34-byte block boundary. On the GB10 box that layout runs at 60 GB/s,
//! while the device itself streams 171 GB/s.
//!
//! The transposed layout groups the blocks of a *tile* of output rows
//! instead:
//!
//! ```text
//! for every tile of 64 output rows:
//!     for every block:
//!         payload: the 32 int8 of every row of the tile
//!         scales:  the f16 scale of every row of the tile
//! ```
//!
//! The tile is the unit both device kernels work in: a workgroup owns 64
//! consecutive output rows, and in this layout those rows are neighbours.
//! A warp reads one contiguous run and the payload loads are aligned, so
//! the single row kernel goes from 60 GB/s to 163 GB/s. The tiled kernel,
//! which reads the weight once for a whole tile of tokens, keeps its 64
//! streams sequential instead of jumping over the matrix, so it goes from
//! 57 GB/s to 122 GB/s.
//!
//! The bytes are the same size (up to the padding of the last tile) and
//! the arithmetic is the same, so the result is bit-for-bit the same as
//! the native layout.

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
/// copy walk memory in long runs instead of 32-byte steps.
pub(crate) fn transpose_q8_0_weight(weight: &[u8], n_in: usize, n_out: usize) -> Vec<u8> {
    let mut packed = vec![0u8; transposed_size(n_in, n_out)];
    transpose_q8_0_weight_into(&mut packed, weight, n_in, n_out);
    packed
}

/// The length of the transposed layout of an `n_in` by `n_out` matrix.
pub(crate) fn transposed_size(n_in: usize, n_out: usize) -> usize {
    let blocks = n_in / PAYLOAD_BYTES;
    n_out.div_ceil(TILE_ROWS) * blocks * TILE_ROWS * BLOCK_BYTES
}

/// Rewrite a Q8_0 weight matrix into the transposed device layout, in place.
///
/// `packed` has to be [`transposed_size`] bytes long. Writing the layout
/// straight into the device buffer saves a second copy of the weight.
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
