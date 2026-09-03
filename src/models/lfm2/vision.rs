//! LFM2.5-VL vision: SigLIP encoder + pixel-shuffle projector + LFM2-VL
//! tiling preprocessor, mirroring llama.cpp `tools/mtmd` (PROJECTOR_TYPE_LFM2).
//!
//! Pipeline (per image):
//! 1. tile preprocessing — overview (smart-resized whole image) + optional
//!    512×512 tiles (llava-uhd style, grid from closest aspect ratio);
//! 2. per entry: SigLIP ViT — patch conv 16×16 → +bilinearly-resized learned
//!    position embedding → 27 pre-LN blocks (full bidirectional attention,
//!    GELU MLP) → final LayerNorm;
//! 3. pixel unshuffle 2×2 (1152·4 = 4608 dims per merged token) →
//!    `mm.1` linear → GELU → `mm.2` linear → 2048-dim LFM2 tokens;
//! 4. token assembly — `<|image_start|>`, per-tile `<|img_row_R_col_C|>`,
//!    `<|img_thumbnail|>`, `<|image_end|>` markers interleaved with the
//!    embedding rows of the LFM2 prefill stream.

use crate::core::tensor::TensorSource;
use crate::ops::kernel::{QuantizedTensor, Weight};
use crate::ops::{gelu, softmax_inplace};

use super::trunk::forward::{run_inference_stream, Lfm2StreamItem};
use crate::core::scratchpad::KvFormat;

pub struct VisionConfig {
    pub image_size: usize,
    pub patch_size: usize,
    pub n_merge: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_layer: usize,
    pub n_ff: usize,
    pub eps: f32,
    pub projection_dim: usize,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
}

impl VisionConfig {
    fn from_source(source: &dyn TensorSource) -> Result<Self, String> {
        let get_u32 = |key: &str| -> Result<u32, String> {
            let v = source
                .metadata(key)
                .ok_or_else(|| format!("Missing metadata: {key}"))?;
            v.to_u64()
                .map(|x| x as u32)
                .ok_or_else(|| format!("{key} is not an integer"))
        };
        let get_f32 = |key: &str| -> Result<f32, String> {
            let v = source
                .metadata(key)
                .ok_or_else(|| format!("Missing metadata: {key}"))?;
            v.to_f64()
                .map(|x| x as f32)
                .ok_or_else(|| format!("{key} is not a float"))
        };
        let get_f32x3 = |key: &str| -> Result<[f32; 3], String> {
            let v = source
                .metadata(key)
                .ok_or_else(|| format!("Missing metadata: {key}"))?
                .clone();
            if let crate::core::tensor::MetaValue::Array(_, items) = v {
                let mut out = [0.0f32; 3];
                for (i, item) in items.iter().enumerate().take(3) {
                    out[i] = crate::core::tensor::MetaValue::to_f64(item)
                        .ok_or_else(|| format!("{key} non-float"))?
                        as f32;
                }
                Ok(out)
            } else {
                Err(format!("{key} is not an array"))
            }
        };
        Ok(Self {
            image_size: get_u32("clip.vision.image_size")? as usize,
            patch_size: get_u32("clip.vision.patch_size")? as usize,
            n_merge: get_u32("clip.vision.projector.scale_factor")? as usize,
            n_embd: get_u32("clip.vision.embedding_length")? as usize,
            n_head: get_u32("clip.vision.attention.head_count")? as usize,
            n_layer: get_u32("clip.vision.block_count")? as usize,
            n_ff: get_u32("clip.vision.feed_forward_length")? as usize,
            eps: get_f32("clip.vision.attention.layer_norm_epsilon")?,
            projection_dim: get_u32("clip.vision.projection_dim")? as usize,
            image_mean: get_f32x3("clip.vision.image_mean")?,
            image_std: get_f32x3("clip.vision.image_std")?,
        })
    }
}

// std430-free helper: metadata array reference without importing the variant
// type twice.
use crate::core::tensor::MetaValue as MetaValueRef;

pub struct VisionLayerWeights<'a> {
    pub ln1: (Vec<f32>, Vec<f32>),
    pub ln2: (Vec<f32>, Vec<f32>),
    pub wq: Weight<'a>,
    pub bq: Vec<f32>,
    pub wk: Weight<'a>,
    pub bk: Vec<f32>,
    pub wv: Weight<'a>,
    pub bv: Vec<f32>,
    pub wo: Weight<'a>,
    pub bo: Vec<f32>,
    pub ffn_up: Weight<'a>,
    pub ffn_up_b: Vec<f32>,
    pub ffn_down: Weight<'a>,
    pub ffn_down_b: Vec<f32>,
}

pub struct VisionModel<'a> {
    pub config: VisionConfig,
    pub patch_embd: Vec<f32>, // [oc][ic][kh][kw] f32, conv weight
    pub patch_bias: Vec<f32>,
    pub position_embd: Vec<f32>, // [n_pos][n_embd] f32, base grid
    pub post_ln: (Vec<f32>, Vec<f32>),
    pub blocks: Vec<VisionLayerWeights<'a>>,
    pub mm1: Weight<'a>,
    pub mm1_b: Vec<f32>,
    pub mm2: Weight<'a>,
    pub mm2_b: Vec<f32>,
}

fn quant_weight<'a>(
    source: &'a dyn TensorSource,
    name: &str,
    n_in: usize,
    n_out: usize,
) -> Result<Weight<'a>, String> {
    let bytes = source
        .tensor_slice(name)
        .ok_or_else(|| format!("tensor {name} not found"))?;
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("tensor info {name} not found"))?;
    Ok(Weight::from_quantized(QuantizedTensor::from_bytes(
        bytes,
        info.ggml_type,
        n_in,
        n_out,
    )))
}

fn f32_vec(source: &dyn TensorSource, name: &str, expected: usize) -> Result<Vec<f32>, String> {
    crate::core::tensor::load_f32_tensor(source, name, &[expected as u64])
}

impl<'a> VisionModel<'a> {
    pub fn from_source(source: &'a dyn TensorSource) -> Result<Self, String> {
        let config = VisionConfig::from_source(source)?;
        let ne = config.n_embd;
        let ps = config.patch_size;
        // ggml conv weight layout: [kw][kh][ic][oc] (ne[0] fastest).
        let mut patch_embd = crate::core::tensor::load_f32_tensor(
            source,
            "v.patch_embd.weight",
            &[ps as u64, ps as u64, 3, ne as u64],
        )?;
        // ggml stores conv weight as [kw][kh][ic][oc]; flatten to oc-major
        // [oc][ic][kh][kw] for the direct conv below.
        {
            let (kw, kh, ic, oc) = (config.patch_size, config.patch_size, 3usize, ne);
            let mut reordered = vec![0.0f32; patch_embd.len()];
            for o in 0..oc {
                for i in 0..ic {
                    for h in 0..kh {
                        for w in 0..kw {
                            let src = w + h * kw + i * kw * kh + o * kw * kh * ic;
                            let dst = o * ic * kh * kw + i * kh * kw + h * kw + w;
                            reordered[dst] = patch_embd[src];
                        }
                    }
                }
            }
            patch_embd = reordered;
            let _ = (kw, kh, ic, oc);
        }
        let patch_bias = f32_vec(source, "v.patch_embd.bias", ne)?;
        // [n_embd, 256] in ggml order — flattened row-major ne[0]-fastest,
        // i.e. position-major rows of n_embd, matching our [pos][ne] layout.
        let position_embd = crate::core::tensor::load_f32_tensor(
            source,
            "v.position_embd.weight",
            &[ne as u64, 256],
        )?;
        let post_ln = (
            f32_vec(source, "v.post_ln.weight", ne)?,
            f32_vec(source, "v.post_ln.bias", ne)?,
        );

        let mut blocks = Vec::with_capacity(config.n_layer);
        for l in 0..config.n_layer {
            let ln = |n: &str| -> Result<(Vec<f32>, Vec<f32>), String> {
                Ok((
                    f32_vec(source, &format!("v.blk.{l}.{n}.weight"), ne)?,
                    f32_vec(source, &format!("v.blk.{l}.{n}.bias"), ne)?,
                ))
            };
            blocks.push(VisionLayerWeights {
                ln1: ln("ln1")?,
                ln2: ln("ln2")?,
                wq: quant_weight(source, &format!("v.blk.{l}.attn_q.weight"), ne, ne)?,
                bq: f32_vec(source, &format!("v.blk.{l}.attn_q.bias"), ne)?,
                wk: quant_weight(source, &format!("v.blk.{l}.attn_k.weight"), ne, ne)?,
                bk: f32_vec(source, &format!("v.blk.{l}.attn_k.bias"), ne)?,
                wv: quant_weight(source, &format!("v.blk.{l}.attn_v.weight"), ne, ne)?,
                bv: f32_vec(source, &format!("v.blk.{l}.attn_v.bias"), ne)?,
                wo: quant_weight(source, &format!("v.blk.{l}.attn_out.weight"), ne, ne)?,
                bo: f32_vec(source, &format!("v.blk.{l}.attn_out.bias"), ne)?,
                ffn_up: quant_weight(source, &format!("v.blk.{l}.ffn_up.weight"), ne, config.n_ff)?,
                ffn_up_b: f32_vec(source, &format!("v.blk.{l}.ffn_up.bias"), config.n_ff)?,
                ffn_down: quant_weight(
                    source,
                    &format!("v.blk.{l}.ffn_down.weight"),
                    config.n_ff,
                    ne,
                )?,
                ffn_down_b: f32_vec(source, &format!("v.blk.{l}.ffn_down.bias"), ne)?,
            });
        }

        let mm1_info = source
            .tensor_info("mm.1.weight")
            .ok_or_else(|| "tensor mm.1.weight not found".to_string())?;
        let mm1_out_dim = mm1_info
            .dims
            .get(1)
            .copied()
            .unwrap_or(config.projection_dim as u64) as usize;

        let mm2_info = source
            .tensor_info("mm.2.weight")
            .ok_or_else(|| "tensor mm.2.weight not found".to_string())?;
        let mm2_out_dim = mm2_info
            .dims
            .get(1)
            .copied()
            .unwrap_or(config.projection_dim as u64) as usize;

        let mm1_b = f32_vec(source, "mm.1.bias", mm1_out_dim)?;
        let mm2_b = f32_vec(source, "mm.2.bias", mm2_out_dim)?;

        Ok(Self {
            mm1: quant_weight(source, "mm.1.weight", 4 * ne, mm1_out_dim)?,
            mm1_b,
            mm2: quant_weight(source, "mm.2.weight", mm1_out_dim, mm2_out_dim)?,
            mm2_b,
            config,
            patch_embd,
            patch_bias,
            position_embd,
            post_ln,
            blocks,
        })
    }
}

// ---------------------------------------------------------------------------
// image preprocessing (LFM2 tiling, mirrors mtmd_image_preprocessor_lfm2)
// ---------------------------------------------------------------------------

const TILE_SIZE: usize = 512;
const MIN_TILES: usize = 2;
const MAX_TILES: usize = 10;
const MAX_PIXELS_TOLERANCE: f32 = 2.0;
const ALIGN_SIZE: usize = 32; // patch_size(16) * n_merge(2)
const MIN_PIXELS: usize = 64 * 16 * 16 * 2 * 2; // 64 merged tokens
const MAX_PIXELS: usize = 256 * 16 * 16 * 2 * 2; // 256 merged tokens

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Size {
    pub width: usize,
    pub height: usize,
}

fn round_by_factor(x: f64, f: usize) -> usize {
    (x / f as f64).round() as usize * f
}
fn ceil_by_factor(x: f64, f: usize) -> usize {
    (x / f as f64).ceil() as usize * f
}
fn floor_by_factor(x: f64, f: usize) -> usize {
    (x / f as f64).floor() as usize * f
}

/// transformers-style smart_resize (align + min/max pixel clamp).
pub fn smart_resize(width: usize, height: usize) -> Size {
    let mut w_bar = round_by_factor(width as f64, ALIGN_SIZE).max(ALIGN_SIZE);
    let mut h_bar = round_by_factor(height as f64, ALIGN_SIZE).max(ALIGN_SIZE);
    if h_bar * w_bar > MAX_PIXELS {
        let beta = ((height * width) as f64 / MAX_PIXELS as f64).sqrt();
        h_bar = floor_by_factor(height as f64 / beta, ALIGN_SIZE).max(ALIGN_SIZE);
        w_bar = floor_by_factor(width as f64 / beta, ALIGN_SIZE).max(ALIGN_SIZE);
    } else if h_bar * w_bar < MIN_PIXELS {
        let beta = (MIN_PIXELS as f64 / ((height * width) as f64)).sqrt();
        h_bar = ceil_by_factor(height as f64 * beta, ALIGN_SIZE);
        w_bar = ceil_by_factor(width as f64 * beta, ALIGN_SIZE);
    }
    Size {
        width: w_bar,
        height: h_bar,
    }
}

fn target_ratios() -> Vec<Size> {
    let mut ratios = Vec::new();
    for n in MIN_TILES..=MAX_TILES {
        for w in 1..=n {
            for h in 1..=n {
                if (w * h) >= MIN_TILES
                    && (w * h) <= MAX_TILES
                    && !ratios.iter().any(|r: &Size| r.width == w && r.height == h)
                {
                    ratios.push(Size {
                        width: w,
                        height: h,
                    });
                }
            }
        }
    }
    ratios.sort_by_key(|r| r.width * r.height);
    ratios
}

fn grid_layout(width: usize, height: usize) -> Size {
    let aspect = width as f32 / height as f32;
    let ratios = target_ratios();
    let mut best_diff = f32::MAX;
    let mut best = Size {
        width: 1,
        height: 1,
    };
    let area = (width * height) as f32;
    for ratio in &ratios {
        let target_aspect = ratio.width as f32 / ratio.height as f32;
        let diff = (aspect - target_aspect).abs();
        if diff < best_diff {
            best_diff = diff;
            best = *ratio;
        } else if diff == best_diff {
            let target_area = (TILE_SIZE * TILE_SIZE * ratio.width * ratio.height) as f32;
            if area > 0.5 * target_area {
                best = *ratio;
            }
        }
    }
    best
}

/// One image entry to encode: pixel data (RGB, CHW, normalized) + size.
pub struct ImageEntry {
    pub data: Vec<f32>, // CHW
    pub width: usize,
    pub height: usize,
    /// `false` for a single-tile image: no `<|img_thumbnail|>` marker.
    pub is_overview: bool,
}

/// LFM2-VL tiling: returns (overview, tiles, grid) — grid is (0,0) when the
/// image fits in one tile (overview only, no thumbnail marker).
pub fn preprocess_image(
    rgb: &[u8],
    width: usize,
    height: usize,
    mean: [f32; 3],
    std: [f32; 3],
) -> (ImageEntry, Vec<ImageEntry>, Size) {
    let overview_size = smart_resize(width, height);
    let needs_tiling = {
        let h_bar = round_by_factor(height as f64, ALIGN_SIZE).max(ALIGN_SIZE);
        let w_bar = round_by_factor(width as f64, ALIGN_SIZE).max(ALIGN_SIZE);
        (h_bar * w_bar) as f32 > MAX_PIXELS as f32 * MAX_PIXELS_TOLERANCE
    };

    let to_f32 = |img: &image::DynamicImage| -> Vec<f32> {
        let rgb8 = img.to_rgb8();
        let (w, h) = (rgb8.width() as usize, rgb8.height() as usize);
        let mut chw = vec![0.0f32; 3 * w * h];
        for y in 0..h {
            for x in 0..w {
                let px = rgb8.get_pixel(x as u32, y as u32);
                for c in 0..3 {
                    let v = px[c] as f32 / 255.0;
                    chw[c * w * h + y * w + x] = (v - mean[c]) / std[c];
                }
            }
        }
        chw
    };

    if !needs_tiling {
        let img = image::DynamicImage::ImageRgb8(to_rgb8(rgb, width, height)).resize_exact(
            overview_size.width as u32,
            overview_size.height as u32,
            image::imageops::FilterType::Triangle,
        );
        let data = to_f32(&img);
        return (
            ImageEntry {
                data,
                width: overview_size.width,
                height: overview_size.height,
                is_overview: false,
            },
            Vec::new(),
            Size {
                width: 0,
                height: 0,
            },
        );
    }

    let grid = grid_layout(width, height);
    let refined = Size {
        width: TILE_SIZE * grid.width,
        height: TILE_SIZE * grid.height,
    };
    let resized = image::DynamicImage::ImageRgb8(to_rgb8(rgb, width, height)).resize_exact(
        refined.width as u32,
        refined.height as u32,
        image::imageops::FilterType::Triangle,
    );

    let overview_img = image::DynamicImage::ImageRgb8(to_rgb8(rgb, width, height)).resize_exact(
        overview_size.width as u32,
        overview_size.height as u32,
        image::imageops::FilterType::Triangle,
    );
    let overview = ImageEntry {
        data: to_f32(&overview_img),
        width: overview_size.width,
        height: overview_size.height,
        is_overview: true,
    };

    let mut tiles = Vec::new();
    for row in 0..grid.height {
        for col in 0..grid.width {
            let mut tile_img = resized.clone();
            let cropped = tile_img.crop(
                (col * TILE_SIZE) as u32,
                (row * TILE_SIZE) as u32,
                TILE_SIZE as u32,
                TILE_SIZE as u32,
            );
            let data = to_f32(&image::DynamicImage::ImageRgb8(cropped.to_rgb8()));
            tiles.push(ImageEntry {
                data,
                width: TILE_SIZE,
                height: TILE_SIZE,
                is_overview: false,
            });
        }
    }
    (overview, tiles, grid)
}

fn to_rgb8(rgb: &[u8], width: usize, height: usize) -> image::RgbImage {
    let mut img = image::RgbImage::new(width as u32, height as u32);
    for y in 0..height {
        for x in 0..width {
            let i = (y * width + x) * 3;
            img.put_pixel(
                x as u32,
                y as u32,
                image::Rgb([rgb[i], rgb[i + 1], rgb[i + 2]]),
            );
        }
    }
    img
}

// ---------------------------------------------------------------------------
// SigLIP encoder + projector
// ---------------------------------------------------------------------------

fn layer_norm(x: &mut [f32], w: &[f32], b: &[f32], eps: f32) {
    let n = x.len();
    let mean = x.iter().sum::<f32>() / n as f32;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
    let scale = 1.0 / (var + eps).sqrt();
    for (v, (wi, bi)) in x.iter_mut().zip(w.iter().zip(b.iter())) {
        *v = (*v - mean) * scale * wi + bi;
    }
}

impl<'a> VisionModel<'a> {
    /// Bilinearly interpolate the (16, 16) learned position grid to
    /// (w_patches, h_patches); returns [n_pos][n_embd].
    fn resized_pos_embd(&self, w_patches: usize, h_patches: usize) -> Vec<f32> {
        let ne = self.config.n_embd;
        let src_side = 16usize; // sqrt(256 positions)
        if w_patches == src_side && h_patches == src_side {
            return self.position_embd.clone();
        }
        let mut out = vec![0.0f32; w_patches * h_patches * ne];
        let sx = src_side as f64 / w_patches as f64;
        let sy = src_side as f64 / h_patches as f64;
        for hp in 0..h_patches {
            for wp in 0..w_patches {
                // sample at output center (clamped bilinear)
                let fx = ((wp as f64 + 0.5) * sx - 0.5).clamp(0.0, (src_side - 1) as f64);
                let fy = ((hp as f64 + 0.5) * sy - 0.5).clamp(0.0, (src_side - 1) as f64);
                let x0 = fx.floor() as usize;
                let y0 = fy.floor() as usize;
                let x1 = (x0 + 1).min(src_side - 1);
                let y1 = (y0 + 1).min(src_side - 1);
                let dx = (fx - x0 as f64) as f32;
                let dy = (fy - y0 as f64) as f32;
                for c in 0..ne {
                    let p00 = self.position_embd[(y0 * src_side + x0) * ne + c];
                    let p01 = self.position_embd[(y0 * src_side + x1) * ne + c];
                    let p10 = self.position_embd[(y1 * src_side + x0) * ne + c];
                    let p11 = self.position_embd[(y1 * src_side + x1) * ne + c];
                    let top = p00 * (1.0 - dx) + p01 * dx;
                    let bot = p10 * (1.0 - dx) + p11 * dx;
                    out[(hp * w_patches + wp) * ne + c] = top * (1.0 - dy) + bot * dy;
                }
            }
        }
        out
    }

    /// Encode one entry: returns merged-token rows of `projection_dim`.
    pub fn encode_entry(
        &self,
        entry: &ImageEntry,
        pool: &std::sync::Arc<ComputePool>,
    ) -> Result<Vec<f32>, String> {
        let cfg = &self.config;
        let ne = cfg.n_embd;
        let (width, height) = (entry.width, entry.height);
        let w_patches = width / cfg.patch_size;
        let h_patches = height / cfg.patch_size;
        let n_patches = w_patches * h_patches;
        let d_head = ne / cfg.n_head;
        let kq_scale = 1.0 / (d_head as f32).sqrt();

        // ---- patch conv 16x16 stride 16 ----
        let mut inp = vec![0.0f32; n_patches * ne];
        let p = cfg.patch_size;
        let inp_chw = &entry.data;
        for ph in 0..h_patches {
            for pw in 0..w_patches {
                let tok = ph * w_patches + pw;
                for oc in 0..ne {
                    let mut sum = self.patch_bias[oc];
                    let w_base = oc * 3 * p * p;
                    for ic in 0..3 {
                        let i_base = ic * width * height;
                        let w_off = w_base + ic * p * p;
                        for kh in 0..p {
                            let row = (ph * p + kh) * width + pw * p;
                            for kw in 0..p {
                                sum += inp_chw[i_base + row + kw]
                                    * self.patch_embd[w_off + kh * p + kw];
                            }
                        }
                    }
                    inp[tok * ne + oc] = sum;
                }
            }
        }

        // ---- position embedding ----
        let pos = self.resized_pos_embd(w_patches, h_patches);
        for i in 0..n_patches {
            for c in 0..ne {
                inp[i * ne + c] += pos[i * ne + c];
            }
        }

        // ---- ViT blocks ----
        let mut hidden = inp;
        for layer in &self.blocks {
            // LN1
            let mut normed = hidden.clone();
            for t in 0..n_patches {
                layer_norm(
                    &mut normed[t * ne..(t + 1) * ne],
                    &layer.ln1.0,
                    &layer.ln1.1,
                    cfg.eps,
                );
            }
            // q/k/v
            let mut q = vec![0.0f32; n_patches * ne];
            let mut k = vec![0.0f32; n_patches * ne];
            let mut v = vec![0.0f32; n_patches * ne];
            for t in 0..n_patches {
                let i = &normed[t * ne..(t + 1) * ne];
                layer
                    .wq
                    .kernel
                    .forward(i, &mut q[t * ne..(t + 1) * ne], ne, ne);
                layer
                    .wk
                    .kernel
                    .forward(i, &mut k[t * ne..(t + 1) * ne], ne, ne);
                layer
                    .wv
                    .kernel
                    .forward(i, &mut v[t * ne..(t + 1) * ne], ne, ne);
                for c in 0..ne {
                    q[t * ne + c] += layer.bq[c];
                    k[t * ne + c] += layer.bk[c];
                    v[t * ne + c] += layer.bv[c];
                }
            }
            // attention per head (bidirectional, no mask)
            let mut attn_out = vec![0.0f32; n_patches * ne];
            for h in 0..cfg.n_head {
                let dh = h * d_head;
                for t in 0..n_patches {
                    let q_off = t * ne + dh;
                    let mut scores = vec![0.0f32; n_patches];
                    for s in 0..n_patches {
                        let k_off = s * ne + dh;
                        let mut dot = 0.0f32;
                        for d in 0..d_head {
                            dot += q[q_off + d] * k[k_off + d];
                        }
                        scores[s] = dot * kq_scale;
                    }
                    softmax_inplace(&mut scores);
                    for d in 0..d_head {
                        let mut acc = 0.0f32;
                        for s in 0..n_patches {
                            acc += scores[s] * v[s * ne + dh + d];
                        }
                        attn_out[t * ne + dh + d] = acc;
                    }
                }
            }
            // out proj + residual
            let mut attn_proj = vec![0.0f32; n_patches * ne];
            for t in 0..n_patches {
                layer.wo.kernel.forward(
                    &attn_out[t * ne..(t + 1) * ne],
                    &mut attn_proj[t * ne..(t + 1) * ne],
                    ne,
                    ne,
                );
                for c in 0..ne {
                    attn_proj[t * ne + c] += layer.bo[c];
                }
            }
            for i in 0..n_patches * ne {
                hidden[i] += attn_proj[i];
            }

            // LN2 + MLP(gelu)
            let mut normed2 = hidden.clone();
            for t in 0..n_patches {
                layer_norm(
                    &mut normed2[t * ne..(t + 1) * ne],
                    &layer.ln2.0,
                    &layer.ln2.1,
                    cfg.eps,
                );
            }
            let mut up = vec![0.0f32; n_patches * cfg.n_ff];
            for t in 0..n_patches {
                layer.ffn_up.kernel.forward(
                    &normed2[t * ne..(t + 1) * ne],
                    &mut up[t * cfg.n_ff..(t + 1) * cfg.n_ff],
                    ne,
                    cfg.n_ff,
                );
                for c in 0..cfg.n_ff {
                    up[t * cfg.n_ff + c] += layer.ffn_up_b[c];
                }
            }
            for v in up.iter_mut() {
                *v = gelu(*v);
            }
            let mut down = vec![0.0f32; n_patches * ne];
            for t in 0..n_patches {
                layer.ffn_down.kernel.forward(
                    &up[t * cfg.n_ff..(t + 1) * cfg.n_ff],
                    &mut down[t * ne..(t + 1) * ne],
                    cfg.n_ff,
                    ne,
                );
                for c in 0..ne {
                    down[t * ne + c] += layer.ffn_down_b[c];
                }
            }
            for i in 0..n_patches * ne {
                hidden[i] += down[i];
            }
        }

        // ---- post LN ----
        for t in 0..n_patches {
            layer_norm(
                &mut hidden[t * ne..(t + 1) * ne],
                &self.post_ln.0,
                &self.post_ln.1,
                cfg.eps,
            );
        }

        // ---- pixel unshuffle 2×2 ----
        // hidden is [n_patches][n_embd] with patches (y, x) x-fastest.
        let (h_p, w_p) = (h_patches, w_patches);
        let merged_h = h_p / cfg.n_merge;
        let merged_w = w_p / cfg.n_merge;
        let merged_dim = ne * cfg.n_merge * cfg.n_merge;
        let mut merged = vec![0.0f32; merged_h * merged_w * merged_dim];
        for mh in 0..merged_h {
            for mw in 0..merged_w {
                let mut out_off = (mh * merged_w + mw) * merged_dim;
                for sh in 0..cfg.n_merge {
                    for sw in 0..cfg.n_merge {
                        let patch = (mh * cfg.n_merge + sh) * w_p + mw * cfg.n_merge + sw;
                        for c in 0..ne {
                            merged[out_off] = hidden[patch * ne + c];
                            out_off += 1;
                        }
                    }
                }
                let _ = out_off;
            }
        }

        // ---- projector: mm.1 → gelu → mm.2 ----
        let n_tok = merged_h * merged_w;
        let mut mid = vec![0.0f32; n_tok * cfg.projection_dim];
        for t in 0..n_tok {
            self.mm1.kernel.forward(
                &merged[t * merged_dim..(t + 1) * merged_dim],
                &mut mid[t * cfg.projection_dim..(t + 1) * cfg.projection_dim],
                merged_dim,
                cfg.projection_dim,
            );
            for c in 0..cfg.projection_dim {
                mid[t * cfg.projection_dim + c] += self.mm1_b[c];
            }
        }
        for v in mid.iter_mut() {
            *v = gelu(*v);
        }
        let mut out = vec![0.0f32; n_tok * cfg.projection_dim];
        for t in 0..n_tok {
            self.mm2.kernel.forward(
                &mid[t * cfg.projection_dim..(t + 1) * cfg.projection_dim],
                &mut out[t * cfg.projection_dim..(t + 1) * cfg.projection_dim],
                cfg.projection_dim,
                cfg.projection_dim,
            );
            for c in 0..cfg.projection_dim {
                out[t * cfg.projection_dim + c] += self.mm2_b[c];
            }
        }
        Ok(out)
    }
}

use crate::core::thread_pool::ComputePool;

/// Multimodal entry: image + prompt → LFM2 prefill stream → generation.
#[allow(clippy::too_many_arguments)]
pub fn run_multimodal(
    llm_source: &dyn TensorSource,
    mmproj_source: &dyn TensorSource,
    image_path: &std::path::Path,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    n_threads_arg: usize,
    kv_format: KvFormat,
) -> Result<(), String> {
    let img = image::open(image_path)
        .map_err(|e| format!("Failed to open image {}: {e}", image_path.display()))?
        .to_rgb8();
    let (width, height) = (img.width() as usize, img.height() as usize);
    let rgb = img.into_raw();

    let model = VisionModel::from_source(mmproj_source)?;
    let (overview, tiles, grid) = preprocess_image(
        &rgb,
        width,
        height,
        model.config.image_mean,
        model.config.image_std,
    );

    println!(
        "Vision: grid {}x{} ({} tiles), overview {}x{}",
        grid.width,
        grid.height,
        tiles.len(),
        overview.width,
        overview.height
    );

    let pool = std::sync::Arc::new(ComputePool::new(if n_threads_arg > 0 {
        n_threads_arg
    } else {
        8
    }));

    // encode tiles (row-major) then overview (thumbnail last)
    let mut tile_rows: Vec<Vec<f32>> = Vec::new();
    for (i, tile) in tiles.iter().enumerate() {
        let rows = model.encode_entry(tile, &pool)?;
        println!(
            "tile {i}: {} tokens",
            rows.len() / model.config.projection_dim
        );
        tile_rows.push(rows);
    }
    let overview_rows = model.encode_entry(&overview, &pool)?;
    println!(
        "overview: {} tokens",
        overview_rows.len() / model.config.projection_dim
    );

    // ---- token + embedding stream ----
    let tokenizer = crate::core::tokenizer::BPETokenizer::from_gguf_metadata(|k| {
        llm_source.metadata(k).cloned()
    })
    .map_err(|e| format!("Failed to initialize tokenizer: {e}"))?;
    let tok = |marker: &str| -> Result<Lfm2StreamItem, String> {
        let id = tokenizer
            .token_id(marker)
            .or_else(|| tokenizer.special_token_id(marker))
            .ok_or_else(|| format!("image marker {marker} missing from vocab"))?;
        Ok(Lfm2StreamItem::Token(id))
    };

    let mut stream: Vec<Lfm2StreamItem> = Vec::new();
    stream.push(tok("<|startoftext|>")?);
    stream.extend(
        tokenizer
            .encode(
                "user\n",
                crate::core::tokenizer::EncodeOptions {
                    add_special: false,
                    parse_special: true,
                },
            )
            .into_iter()
            .map(Lfm2StreamItem::Token),
    );
    stream.push(tok("<|image_start|>")?);
    if !tiles.is_empty() {
        for (i, rows) in tile_rows.iter().enumerate() {
            let y = i / grid.width + 1;
            let x = i % grid.width + 1;
            stream.push(tok(&format!("<|img_row_{y}_col_{x}|>"))?);
            stream.extend(
                rows.chunks(model.config.projection_dim)
                    .map(|r| Lfm2StreamItem::Embedding(r.to_vec())),
            );
        }
        stream.push(tok("<|img_thumbnail|>")?);
        stream.extend(
            overview_rows
                .chunks(model.config.projection_dim)
                .map(|r| Lfm2StreamItem::Embedding(r.to_vec())),
        );
    } else {
        stream.extend(
            overview_rows
                .chunks(model.config.projection_dim)
                .map(|r| Lfm2StreamItem::Embedding(r.to_vec())),
        );
    }
    stream.push(tok("<|image_end|>")?);
    stream.extend(
        tokenizer
            .encode(
                &format!("\n{prompt}\n"),
                crate::core::tokenizer::EncodeOptions {
                    add_special: false,
                    parse_special: true,
                },
            )
            .into_iter()
            .map(Lfm2StreamItem::Token),
    );
    stream.extend(
        tokenizer
            .encode(
                "assistant\n",
                crate::core::tokenizer::EncodeOptions {
                    add_special: false,
                    parse_special: true,
                },
            )
            .into_iter()
            .map(Lfm2StreamItem::Token),
    );

    run_inference_stream(
        llm_source,
        stream,
        max_tokens,
        temperature,
        n_threads_arg,
        false,
        kv_format,
    )
}
