//! SenseNova-U1.6 "vision tower" — `NEOVisionEmbeddings`.
//!
//! Despite the name there is no transformer stack: the encoder is three
//! operations on the pixel grid (see
//! `references/src/sensenova_u1/models/neo_unify/modeling_neo_vit.py` and
//! `docs/develop/SENSENOVA_U1.md` §4):
//!
//! ```text
//! pixels [N, 3*16*16]                      (patch rows, channel-major per patch)
//!   -> conv2d(3 -> 1024, k=16, s=16)       patch_embedding   + bias
//!   -> GELU                                nn.GELU() = exact erf form
//!   -> 2D RoPE                             first 512 lanes by abs_x,
//!                                          last  512 lanes by abs_y
//!   -> conv2d(1024 -> 4096, k=2, s=2)      dense_embedding   + bias
//!   -> [N / 4, 4096]                       token-major over the merged grid
//! ```
//!
//! The two towers (`vision_model` for understanding,
//! `fm_modules.vision_model_mot_gen` for generation) are structurally identical
//! and differ only in weights, so [`VisionEmbeddings::load`] takes the GGUF
//! prefix instead of hard-coding a name.
//!
//! ## 2D RoPE layout
//!
//! Upstream (`apply_2d_rotary_pos_emb` + `apply_rotary_emb_1d`,
//! `modeling_neo_vit.py:55-109`) splits the 1024-dim patch embedding into two
//! 512-lane halves. Half 0 rotates by the patch's `abs_x`, half 1 by `abs_y`,
//! and inside each half the rotated pairs are the **interleaved** `(2i, 2i+1)`
//! lanes — `x1 = x[..., 0::2]`, `x2 = x[..., 1::2]` — with
//! `inv_freq[i] = theta^(-i/256)`, `theta = rope_theta_vision = 10000`.
//!
//! That is GGML's "norm" pairing (`crate::ops::rope::rope_norm`), *not* the
//! NeoX half-split `(i, i + 256)` pairing. The two share the same frequency
//! table but pair different lanes, so the distinction is visible from position
//! 1 onwards. `docs/develop/SENSENOVA_U1.md` §4 calls this a half-split
//! rotation; the reference implementation at the pinned oracle commit
//! (`d2d9bd1`) is unambiguous, so this module follows the reference and
//! `vision_rope_pairing_is_interleaved` pins the decision.
//!
//! The cos/sin table is built once per load: positions are patch coordinates
//! that repeat across the grid, so computing them per patch would be
//! `O(patches * 512)` transcendental calls instead of `O(max_pos * 256)`.

use crate::core::tensor::TensorSource;
use crate::core::thread_pool::ComputePool;
use crate::ops::gelu_erf_inplace;
use crate::ops::nn::{conv2d, load_float_values};
use crate::ops::rope::rope_sin_cos;

use super::config::{HIDDEN, MERGE_SIZE, PATCH_SIZE, VISION_HIDDEN, VISION_THETA};

/// Values in one `pixels` row: `3 * 16 * 16`.
const PATCH_VALUES: usize = 3 * PATCH_SIZE * PATCH_SIZE;
/// Lanes of the patch embedding rotated by a single axis (1024 / 2).
const ROPE_HALF: usize = VISION_HIDDEN / 2;
/// Frequency pairs inside one rotated half (512 / 2).
const ROPE_PAIRS: usize = ROPE_HALF / 2;
/// Positions covered by the RoPE table.
///
/// Upstream precomputes `max_position_embeddings_vision = 10000` rows; a patch
/// coordinate is a grid index, so this cap only has to cover the longest image
/// side the sampler can produce. 1024 covers a 16384-pixel side at
/// `patch_size = 16`, far beyond `max_pixels = 16777216` for any usable aspect
/// ratio, and costs 2 MiB per tower instead of 20 MiB. Grids that exceed it are
/// rejected rather than silently wrapping.
const ROPE_MAX_POS: usize = 1024;

/// `NEOVisionEmbeddings`: patch conv + GELU + 2D RoPE + downsample conv.
pub(crate) struct VisionEmbeddings {
    /// `patch_embedding.weight`, `[1024, 3, 16, 16]` (out, in, kh, kw).
    patch_weight: Vec<f32>,
    /// `patch_embedding.bias`, `[1024]`.
    patch_bias: Vec<f32>,
    /// `dense_embedding.weight`, `[4096, 1024, 2, 2]` (out, in, kh, kw).
    dense_weight: Vec<f32>,
    /// `dense_embedding.bias`, `[4096]`.
    dense_bias: Vec<f32>,
    /// `[ROPE_MAX_POS, ROPE_PAIRS]` cosines, indexed `pos * ROPE_PAIRS + pair`.
    rope_cos: Vec<f32>,
    /// `[ROPE_MAX_POS, ROPE_PAIRS]` sines, same indexing.
    rope_sin: Vec<f32>,
}

impl VisionEmbeddings {
    /// Load one tower. `prefix` is `"vision_model"` or
    /// `"fm_modules.vision_model_mot_gen"`.
    pub(crate) fn load(source: &dyn TensorSource, prefix: &str) -> Result<Self, String> {
        let patch_weight = load_conv_weight(
            source,
            &format!("{prefix}.embeddings.patch_embedding.weight"),
            VISION_HIDDEN,
            3,
            PATCH_SIZE,
        )?;
        let patch_bias = load_float_values(
            source,
            &format!("{prefix}.embeddings.patch_embedding.bias"),
            &[VISION_HIDDEN as u64],
        )?;
        let dense_weight = load_conv_weight(
            source,
            &format!("{prefix}.embeddings.dense_embedding.weight"),
            HIDDEN,
            VISION_HIDDEN,
            MERGE_SIZE,
        )?;
        let dense_bias = load_float_values(
            source,
            &format!("{prefix}.embeddings.dense_embedding.bias"),
            &[HIDDEN as u64],
        )?;

        let (rope_cos, rope_sin) = build_rope_table();
        Ok(Self {
            patch_weight,
            patch_bias,
            dense_weight,
            dense_bias,
            rope_cos,
            rope_sin,
        })
    }

    /// `pixels` is `[grid_h * grid_w, 3 * 16 * 16]` (one row per 16x16 patch,
    /// channel-major inside the patch, row-major over the patch grid). Returns
    /// `[grid_h * grid_w / 4, 4096]`, token-major over the merged grid.
    pub(crate) fn forward(
        &self,
        pixels: &[f32],
        grid_h: usize,
        grid_w: usize,
        pool: &ComputePool,
    ) -> Result<Vec<f32>, String> {
        if grid_h == 0 || grid_w == 0 {
            return Err(format!(
                "VisionEmbeddings: empty patch grid {grid_h}x{grid_w}"
            ));
        }
        if grid_h % MERGE_SIZE != 0 || grid_w % MERGE_SIZE != 0 {
            return Err(format!(
                "VisionEmbeddings: patch grid {grid_h}x{grid_w} is not a multiple of \
                 merge_size {MERGE_SIZE}"
            ));
        }
        let patches = grid_h
            .checked_mul(grid_w)
            .ok_or_else(|| "VisionEmbeddings: patch grid overflows".to_string())?;
        let expected = patches
            .checked_mul(PATCH_VALUES)
            .ok_or_else(|| "VisionEmbeddings: pixel buffer length overflows".to_string())?;
        if pixels.len() != expected {
            return Err(format!(
                "VisionEmbeddings: pixels has {} values, expected {expected} \
                 ({patches} patches x {PATCH_VALUES})",
                pixels.len()
            ));
        }
        let longest_side = grid_h.max(grid_w);
        if longest_side > ROPE_MAX_POS {
            return Err(format!(
                "VisionEmbeddings: patch grid {grid_h}x{grid_w} needs RoPE position \
                 {longest_side} but the table covers {ROPE_MAX_POS}"
            ));
        }

        let height = grid_h * PATCH_SIZE;
        let width = grid_w * PATCH_SIZE;

        // The patch embedding is a 16x16/16 convolution, so reassemble the
        // patch rows into the pixel image and let `conv2d` do the reduction.
        let image = assemble_pixels(pixels, grid_h, grid_w);
        let (mut hidden, _) = conv2d(
            pool,
            &image,
            [3, height, width],
            &self.patch_weight,
            [VISION_HIDDEN, 3, PATCH_SIZE, PATCH_SIZE],
            Some(&self.patch_bias),
            [PATCH_SIZE; 2],
            [0; 2],
            [1; 2],
            1,
        )?;
        // `[VISION_HIDDEN, grid_h, grid_w]`, channel-major.

        // `nn.GELU()` is the exact erf form; `gelu_inplace` is the tanh
        // approximation and would drift ~1e-3 on the patch embeddings.
        gelu_erf_inplace(&mut hidden);

        rotate_2d(
            &mut hidden,
            patches,
            0,
            |patch| patch % grid_w,
            &self.rope_cos,
            &self.rope_sin,
        );
        rotate_2d(
            &mut hidden,
            patches,
            ROPE_HALF,
            |patch| patch / grid_w,
            &self.rope_cos,
            &self.rope_sin,
        );

        let (down, shape) = conv2d(
            pool,
            &hidden,
            [VISION_HIDDEN, grid_h, grid_w],
            &self.dense_weight,
            [HIDDEN, VISION_HIDDEN, MERGE_SIZE, MERGE_SIZE],
            Some(&self.dense_bias),
            [MERGE_SIZE; 2],
            [0; 2],
            [1; 2],
            1,
        )?;
        // `[HIDDEN, grid_h / 2, grid_w / 2]`, channel-major.

        let tokens = shape[1]
            .checked_mul(shape[2])
            .ok_or_else(|| "VisionEmbeddings: merged grid overflows".to_string())?;
        if tokens * MERGE_SIZE * MERGE_SIZE != patches {
            return Err(format!(
                "VisionEmbeddings: dense_embedding produced {tokens} tokens for {patches} patches"
            ));
        }
        Ok(to_token_major(&down, shape[0], tokens))
    }
}

/// Build the `[ROPE_MAX_POS, ROPE_PAIRS]` cos/sin table.
///
/// Mirrors `precompute_rope_freqs_sincos(512, 10000)` +
/// `apply_rotary_emb_1d`: `inv_freq[i] = 1 / theta^(i / 256)` in f32, and the
/// angle of pair `i` at position `pos` is `pos * inv_freq[i]`.
fn build_rope_table() -> (Vec<f32>, Vec<f32>) {
    let mut inv_freq = vec![0.0f32; ROPE_PAIRS];
    for (pair, freq) in inv_freq.iter_mut().enumerate() {
        *freq = 1.0 / VISION_THETA.powf(pair as f32 / ROPE_PAIRS as f32);
    }
    let mut cos = vec![0.0f32; ROPE_MAX_POS * ROPE_PAIRS];
    let mut sin = vec![0.0f32; ROPE_MAX_POS * ROPE_PAIRS];
    for pos in 0..ROPE_MAX_POS {
        let row = pos * ROPE_PAIRS;
        for (pair, &freq) in inv_freq.iter().enumerate() {
            // `rope_sin_cos` is the repo's f32 sin/cos entry point (it honours
            // the scalar/oracle mode); it returns `(cos, sin)`.
            let (c, s) = rope_sin_cos(pos as f32 * freq);
            cos[row + pair] = c;
            sin[row + pair] = s;
        }
    }
    (cos, sin)
}

/// Rotate the interleaved `(2i, 2i + 1)` lane pairs of one 512-lane half of
/// every patch, in place, on a channel-major `[VISION_HIDDEN, patches]` map.
///
/// `lane_base` selects the half (0 for `abs_x`, [`ROPE_HALF`] for `abs_y`) and
/// `position_of` maps a patch index to the coordinate that half rotates by.
/// Because the map is channel-major, the two lanes of a pair are two whole
/// rows, which keeps both streams contiguous across patches.
fn rotate_2d(
    map: &mut [f32],
    patches: usize,
    lane_base: usize,
    position_of: impl Fn(usize) -> usize,
    cos: &[f32],
    sin: &[f32],
) {
    for pair in 0..ROPE_PAIRS {
        let low = (lane_base + 2 * pair) * patches;
        let high = low + patches;
        for patch in 0..patches {
            let entry = position_of(patch) * ROPE_PAIRS + pair;
            let (c, s) = (cos[entry], sin[entry]);
            let x0 = map[low + patch];
            let x1 = map[high + patch];
            // Same operation order as `apply_rotary_emb_1d`; Rust does not
            // contract float multiply-adds by default, so this stays
            // `x0 * c - x1 * s` rather than an FMA.
            map[low + patch] = x0 * c - x1 * s;
            map[high + patch] = x0 * s + x1 * c;
        }
    }
}

/// Rebuild the `[3, grid_h * 16, grid_w * 16]` pixel image from the patch rows
/// of `patchify(..., channel_first=True)` (`nchpwq -> nhwcpq` upstream).
fn assemble_pixels(pixels: &[f32], grid_h: usize, grid_w: usize) -> Vec<f32> {
    let height = grid_h * PATCH_SIZE;
    let width = grid_w * PATCH_SIZE;
    let plane = height * width;
    let mut image = vec![0.0f32; 3 * plane];
    for patch in 0..grid_h * grid_w {
        let patch_y = patch / grid_w;
        let patch_x = patch % grid_w;
        for channel in 0..3 {
            for row in 0..PATCH_SIZE {
                let source =
                    patch * PATCH_VALUES + channel * PATCH_SIZE * PATCH_SIZE + row * PATCH_SIZE;
                let target =
                    channel * plane + (patch_y * PATCH_SIZE + row) * width + patch_x * PATCH_SIZE;
                image[target..target + PATCH_SIZE]
                    .copy_from_slice(&pixels[source..source + PATCH_SIZE]);
            }
        }
    }
    image
}

/// Transpose a channel-major `[channels, tokens]` map into token-major
/// `[tokens, channels]`.
fn to_token_major(channel_major: &[f32], channels: usize, tokens: usize) -> Vec<f32> {
    let mut tokens_major = vec![0.0f32; channels * tokens];
    for channel in 0..channels {
        let row = &channel_major[channel * tokens..(channel + 1) * tokens];
        for (token, &value) in row.iter().enumerate() {
            tokens_major[token * channels + channel] = value;
        }
    }
    tokens_major
}

/// Load a GGUF convolution weight into the `[out, in, kh, kw]` row-major buffer
/// [`conv2d`] indexes.
///
/// GGUF stores the *reversed* PyTorch shape — a `nn.Conv2d` weight of shape
/// `[out, in, kh, kw]` carries header dims `[kw, kh, in, out]` — while the
/// payload keeps PyTorch's contiguous `[out][in][kh][kw]` order, which is
/// exactly the order `conv2d` reads (`w[((o * in + i) * kh + y) * kw + x]`).
/// Reordering the shape is therefore a header-level reinterpretation and the
/// element buffer is taken verbatim: permuting the elements here would scramble
/// the kernel. (The upstream loader hands `GGUFReader`'s `tensor.data`, whose
/// shape is `dims[::-1]`, straight to `nn.Conv2d` for the same reason.)
pub(super) fn load_conv_weight(
    source: &dyn TensorSource,
    name: &str,
    out_channels: usize,
    in_channels: usize,
    kernel: usize,
) -> Result<Vec<f32>, String> {
    let values = load_float_values(
        source,
        name,
        &[
            kernel as u64,
            kernel as u64,
            in_channels as u64,
            out_channels as u64,
        ],
    )?;
    debug_assert_eq!(
        values.len(),
        out_channels * in_channels * kernel * kernel,
        "{name}: unexpected element count"
    );
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::tensor::{GGMLType, MetaValue, TensorInfo};
    use std::collections::HashMap;

    /// In-memory `TensorSource` holding F32 tensors (see
    /// `src/models/qwen35/trunk/tests.rs` for the same double).
    struct TestSource {
        infos: Vec<TensorInfo>,
        data: HashMap<String, Vec<u8>>,
    }

    impl TestSource {
        fn new() -> Self {
            Self {
                infos: Vec::new(),
                data: HashMap::new(),
            }
        }

        fn add(&mut self, name: &str, dims: &[u64], values: &[f32]) {
            self.infos.push(TensorInfo {
                name: name.to_string(),
                dims: dims.to_vec(),
                ggml_type: GGMLType::F32,
                offset: 0,
            });
            self.data.insert(
                name.to_string(),
                values.iter().flat_map(|v| v.to_le_bytes()).collect(),
            );
        }
    }

    impl TensorSource for TestSource {
        fn metadata(&self, _: &str) -> Option<&MetaValue> {
            None
        }

        fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
            self.infos.iter().find(|info| info.name == name)
        }

        fn tensor_slice(&self, name: &str) -> Option<&[u8]> {
            self.data.get(name).map(Vec::as_slice)
        }
    }

    /// Deterministic pseudo-random values in `[-scale/2, scale/2)`.
    fn pattern(len: usize, scale: f32, seed: u64) -> Vec<f32> {
        let mut state = seed | 1;
        (0..len)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let unit = (state >> 40) as f32 / (1u64 << 24) as f32;
                (unit - 0.5) * scale
            })
            .collect()
    }

    #[test]
    fn rope_table_matches_upstream_frequencies() {
        let (cos, sin) = build_rope_table();
        assert_eq!(cos.len(), ROPE_MAX_POS * ROPE_PAIRS);
        // `inv_freq[0] == 1`, so position 0 is the identity rotation.
        assert_eq!(cos[0], 1.0);
        assert_eq!(sin[0], 0.0);
        // Position 1, pair i: angle = 10000^(-i/256).
        for pair in [0usize, 1, 7, 255] {
            let angle = 10000.0f32.powf(-(pair as f32) / 256.0);
            let entry = ROPE_PAIRS + pair;
            assert!(
                (cos[entry] - angle.cos()).abs() < 1e-6,
                "pair {pair}: cos {} vs {}",
                cos[entry],
                angle.cos()
            );
            assert!((sin[entry] - angle.sin()).abs() < 1e-6, "pair {pair}");
        }
    }

    /// Required check: a 512-dim half at position 1 equals an explicitly
    /// computed cos/sin rotation of the same lanes.
    #[test]
    fn vision_rope_position_one_matches_hand_computed_rotation() {
        let (cos, sin) = build_rope_table();
        // One patch, so the channel-major map is the 512-dim vector itself.
        let original: Vec<f32> = (0..ROPE_HALF)
            .map(|lane| 0.25 * lane as f32 / ROPE_HALF as f32 - 0.125)
            .collect();
        let mut rotated = original.clone();
        rotate_2d(&mut rotated, 1, 0, |_| 1, &cos, &sin);

        for pair in 0..ROPE_PAIRS {
            let angle = 10000.0f32.powf(-(pair as f32) / 256.0);
            let (c, s) = (angle.cos(), angle.sin());
            let x0 = original[2 * pair];
            let x1 = original[2 * pair + 1];
            let expected_low = x0 * c - x1 * s;
            let expected_high = x0 * s + x1 * c;
            assert!(
                (rotated[2 * pair] - expected_low).abs() < 1e-6,
                "pair {pair}: low {} vs {expected_low}",
                rotated[2 * pair]
            );
            assert!(
                (rotated[2 * pair + 1] - expected_high).abs() < 1e-6,
                "pair {pair}: high"
            );
        }
        // Position 0 must be the identity rotation.
        let mut identity = original.clone();
        rotate_2d(&mut identity, 1, 0, |_| 0, &cos, &sin);
        for (a, b) in identity.iter().zip(&original) {
            assert!((a - b).abs() < 1e-7);
        }
    }

    /// Pins the pairing: the reference pairs `(2i, 2i + 1)` inside each half, so
    /// the NeoX half-split rotation `(i, i + 256)` must produce a *different*
    /// vector. If someone re-reads `docs/develop/SENSENOVA_U1.md` §4 ("half-split
    /// rope") and switches the pairing, this test fails and points at
    /// `modeling_neo_vit.py:69-77`.
    #[test]
    fn vision_rope_pairing_is_interleaved() {
        let (cos, sin) = build_rope_table();
        let original: Vec<f32> = (0..ROPE_HALF)
            .map(|lane| 0.25 * lane as f32 / ROPE_HALF as f32 - 0.125)
            .collect();
        let mut rotated = original.clone();
        rotate_2d(&mut rotated, 1, 0, |_| 1, &cos, &sin);

        let mut half_split = original.clone();
        for pair in 0..ROPE_PAIRS {
            let angle = 10000.0f32.powf(-(pair as f32) / 256.0);
            let (c, s) = (angle.cos(), angle.sin());
            let x0 = original[pair];
            let x1 = original[pair + ROPE_PAIRS];
            half_split[pair] = x0 * c - x1 * s;
            half_split[pair + ROPE_PAIRS] = x0 * s + x1 * c;
        }
        let difference: f32 = rotated
            .iter()
            .zip(&half_split)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(
            difference > 1e-3,
            "interleaved and half-split pairings must differ (max delta {difference})"
        );
    }

    /// The in-place channel-major rotation must agree with a patch-major
    /// transcription of the reference: patch `p` rotates lanes `[0, 512)` by
    /// `p % grid_w` (its `abs_x`) and lanes `[512, 1024)` by `p / grid_w`
    /// (`abs_y`).
    #[test]
    fn vision_rope_matches_patch_major_reference() {
        let (grid_h, grid_w) = (2usize, 3usize);
        let patches = grid_h * grid_w;
        let (cos, sin) = build_rope_table();
        let channel_major = pattern(VISION_HIDDEN * patches, 1.0, 31);

        let mut rotated = channel_major.clone();
        rotate_2d(&mut rotated, patches, 0, |patch| patch % grid_w, &cos, &sin);
        rotate_2d(
            &mut rotated,
            patches,
            ROPE_HALF,
            |patch| patch / grid_w,
            &cos,
            &sin,
        );

        let mut expected = channel_major;
        for patch in 0..patches {
            let positions = [patch % grid_w, patch / grid_w];
            for (half, &pos) in positions.iter().enumerate() {
                for pair in 0..ROPE_PAIRS {
                    let lane = half * ROPE_HALF + 2 * pair;
                    let x0 = expected[lane * patches + patch];
                    let x1 = expected[(lane + 1) * patches + patch];
                    let (c, s) = (cos[pos * ROPE_PAIRS + pair], sin[pos * ROPE_PAIRS + pair]);
                    expected[lane * patches + patch] = x0 * c - x1 * s;
                    expected[(lane + 1) * patches + patch] = x0 * s + x1 * c;
                }
            }
        }
        assert_eq!(rotated, expected);
    }

    #[test]
    fn assemble_pixels_places_patches_row_major() {
        // Two patches stacked vertically -> a 32x16 image.
        let mut pixels = vec![0.0f32; 2 * PATCH_VALUES];
        pixels[0] = 7.0; // patch 0, channel 0, row 0, column 0
        pixels[PATCH_SIZE] = 8.0; // patch 0, channel 0, row 1, column 0
        pixels[PATCH_VALUES] = 9.0; // patch 1, channel 0, row 0, column 0
        let image = assemble_pixels(&pixels, 2, 1);
        assert_eq!(image.len(), 3 * 32 * 16);
        let plane = 32 * 16;
        assert_eq!(image[0], 7.0);
        assert_eq!(image[16], 8.0);
        assert_eq!(image[16 * 16], 9.0); // patch 1 starts at image row 16
        assert_eq!(image[plane], 0.0); // channel 1 untouched
    }

    #[test]
    fn to_token_major_is_row_major_over_tokens() {
        // [3 channels, 2 tokens] -> [2 tokens, 3 channels]
        let channel_major = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        assert_eq!(
            to_token_major(&channel_major, 3, 2),
            vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]
        );
    }

    /// Full tower on a 2x2 patch grid (one merged token) with real tensor
    /// shapes: checks the conv plumbing, the RoPE pass and the final transpose
    /// all agree on the output contract.
    #[test]
    fn vision_forward_produces_one_token_per_merge_block() {
        let prefix = "fm_modules.vision_model_mot_gen";
        let mut source = TestSource::new();
        source.add(
            &format!("{prefix}.embeddings.patch_embedding.weight"),
            &[
                PATCH_SIZE as u64,
                PATCH_SIZE as u64,
                3,
                VISION_HIDDEN as u64,
            ],
            &pattern(VISION_HIDDEN * 3 * PATCH_SIZE * PATCH_SIZE, 0.1, 11),
        );
        source.add(
            &format!("{prefix}.embeddings.patch_embedding.bias"),
            &[VISION_HIDDEN as u64],
            &pattern(VISION_HIDDEN, 0.01, 12),
        );
        source.add(
            &format!("{prefix}.embeddings.dense_embedding.weight"),
            &[
                MERGE_SIZE as u64,
                MERGE_SIZE as u64,
                VISION_HIDDEN as u64,
                HIDDEN as u64,
            ],
            &pattern(HIDDEN * VISION_HIDDEN * MERGE_SIZE * MERGE_SIZE, 0.02, 13),
        );
        source.add(
            &format!("{prefix}.embeddings.dense_embedding.bias"),
            &[HIDDEN as u64],
            &pattern(HIDDEN, 0.01, 14),
        );

        let embeddings = VisionEmbeddings::load(&source, prefix).expect("load");
        let (grid_h, grid_w) = (2, 2);
        let pixels = pattern(grid_h * grid_w * PATCH_VALUES, 2.0, 15);
        let pool = ComputePool::new(1);
        let output = embeddings
            .forward(&pixels, grid_h, grid_w, &pool)
            .expect("forward");
        assert_eq!(output.len(), HIDDEN);
        assert!(output.iter().all(|value| value.is_finite()));
        assert!(output.iter().any(|value| *value != 0.0));
    }

    #[test]
    fn vision_forward_rejects_bad_shapes() {
        // The shape checks run before the first convolution, so an empty-weight
        // tower is enough here and keeps the test from allocating the 67 MiB
        // dense projection just to throw it away.
        let (rope_cos, rope_sin) = build_rope_table();
        let embeddings = VisionEmbeddings {
            patch_weight: Vec::new(),
            patch_bias: Vec::new(),
            dense_weight: Vec::new(),
            dense_bias: Vec::new(),
            rope_cos,
            rope_sin,
        };
        let pool = ComputePool::new(1);
        // 3x3 patches: not a multiple of merge_size.
        let pixels = vec![0.0f32; 9 * PATCH_VALUES];
        assert!(embeddings.forward(&pixels, 3, 3, &pool).is_err());
        // Right grid, wrong buffer length.
        assert!(embeddings.forward(&pixels, 2, 2, &pool).is_err());
        // Empty grid.
        assert!(embeddings.forward(&[], 0, 0, &pool).is_err());
    }
}
