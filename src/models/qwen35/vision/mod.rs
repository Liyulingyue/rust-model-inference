pub mod clip_config;

use crate::core::tensor::TensorSource;
use crate::ops::{
    dot_f16_f32, dot_f32, gelu_inplace, rope_vision, softmax_inplace, sum_f32, sum_sq_centered_f32,
    sum_sq_f32, vec_add, vec_add_into, vec_mad_f32,
};
use clip_config::ClipVisionConfig;
use rayon::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VisionGrid {
    pub grid_t: usize,
    pub grid_h: usize,
    pub grid_w: usize,
    pub patch_size: usize,
    pub merge_size: usize,
}

impl VisionGrid {
    pub fn from_image_size(
        grid_t: usize,
        image_width: usize,
        image_height: usize,
        patch_size: usize,
        merge_size: usize,
    ) -> Result<Self, String> {
        if grid_t != 1 || patch_size == 0 || merge_size == 0 {
            return Err("Qwen image v1 requires grid_t=1 and nonzero patch/merge sizes".into());
        }
        let factor = patch_size
            .checked_mul(merge_size)
            .ok_or("Vision grid factor overflow")?;
        if image_width == 0
            || image_height == 0
            || image_width % factor != 0
            || image_height % factor != 0
        {
            return Err(format!(
                "Image {image_width}x{image_height} is not aligned to patch*merge={factor}"
            ));
        }
        let grid = Self {
            grid_t,
            grid_h: image_height / factor,
            grid_w: image_width / factor,
            patch_size,
            merge_size,
        };
        grid.checked_token_count()?;
        Ok(grid)
    }

    pub(crate) fn checked_token_count(self) -> Result<usize, String> {
        if self.grid_t == 0 || self.grid_h == 0 || self.grid_w == 0 {
            return Err("Vision grid dimensions must be nonzero".into());
        }
        self.grid_t
            .checked_mul(self.grid_h)
            .and_then(|count| count.checked_mul(self.grid_w))
            .ok_or_else(|| "Vision grid token count overflow".into())
    }

    pub fn token_count(self) -> usize {
        self.checked_token_count()
            .expect("Vision grid token count must be validated")
    }

    pub fn image_width(self) -> usize {
        self.grid_w * self.patch_size * self.merge_size
    }

    pub fn image_height(self) -> usize {
        self.grid_h * self.patch_size * self.merge_size
    }

    pub fn position_span(self) -> usize {
        self.grid_h.max(self.grid_w)
    }
}

fn aligned_round(value: f64, factor: usize) -> usize {
    ((value / factor as f64).round().max(1.0) as usize) * factor
}

pub fn qwen_smart_resize(
    original_width: usize,
    original_height: usize,
    config: &ClipVisionConfig,
) -> Result<VisionGrid, String> {
    if original_width == 0 || original_height == 0 {
        return Err("Image dimensions must be nonzero".into());
    }
    let factor = config
        .patch_size
        .checked_mul(config.spatial_merge_size)
        .ok_or("Vision resize factor overflow")?;
    let ratio =
        original_width.max(original_height) as f64 / original_width.min(original_height) as f64;
    if ratio > 200.0 {
        return Err(format!("Image aspect ratio {ratio:.2} exceeds 200"));
    }

    let original_pixels = original_width
        .checked_mul(original_height)
        .ok_or("Original image pixel count overflow")?;
    let mut width = aligned_round(original_width as f64, factor);
    let mut height = aligned_round(original_height as f64, factor);
    let aligned_pixels = width
        .checked_mul(height)
        .ok_or("Aligned image pixel count overflow")?;
    if aligned_pixels > config.image_max_pixels {
        // Pinned llama.cpp chooses the branch from aligned dimensions but
        // computes beta from the original dimensions.
        let scale = (original_pixels as f64 / config.image_max_pixels as f64).sqrt();
        width = (((original_width as f64 / scale) / factor as f64)
            .floor()
            .max(1.0) as usize)
            * factor;
        height = (((original_height as f64 / scale) / factor as f64)
            .floor()
            .max(1.0) as usize)
            * factor;
    } else if aligned_pixels < config.image_min_pixels {
        let scale = (config.image_min_pixels as f64 / original_pixels as f64).sqrt();
        width = (((original_width as f64 * scale) / factor as f64).ceil() as usize) * factor;
        height = (((original_height as f64 * scale) / factor as f64).ceil() as usize) * factor;
    }
    VisionGrid::from_image_size(
        1,
        width,
        height,
        config.patch_size,
        config.spatial_merge_size,
    )
}

fn checked_len(label: &str, factors: &[usize]) -> Result<usize, String> {
    factors.iter().try_fold(1usize, |value, factor| {
        value
            .checked_mul(*factor)
            .ok_or_else(|| format!("{label} length overflow"))
    })
}

struct Q8Weight {
    data: Vec<u8>,
    f32_data: Option<Vec<f32>>,
    n_in: usize,
    n_out: usize,
}

impl Q8Weight {
    fn from_f32(weight: &[f32], n_in: usize, n_out: usize) -> Self {
        if n_in % 32 != 0 {
            return Self {
                data: Vec::new(),
                f32_data: Some(weight.to_vec()),
                n_in,
                n_out,
            };
        }
        let blocks_per_row = n_in / 32;
        let row_stride = blocks_per_row * 34;
        let mut data = vec![0u8; n_out * row_stride];
        let mut q8_row = vec![0u8; n_in];
        let mut scales = vec![0.0f32; blocks_per_row];
        for o in 0..n_out {
            let row = &weight[o * n_in..(o + 1) * n_in];
            crate::ops::quantize_q8_0_into(row, n_in, &mut q8_row, &mut scales);
            let off = o * row_stride;
            for b in 0..blocks_per_row {
                let block_off = off + b * 34;
                let scale_f16 = crate::ops::f32_to_f16(scales[b]);
                data[block_off] = (scale_f16 & 0xFF) as u8;
                data[block_off + 1] = (scale_f16 >> 8) as u8;
                data[block_off + 2..block_off + 34].copy_from_slice(&q8_row[b * 32..(b + 1) * 32]);
            }
        }
        Q8Weight {
            data,
            f32_data: None,
            n_in,
            n_out,
        }
    }

    fn matmul_batch(
        &self,
        input: &[f32],
        output: &mut [f32],
        n_tokens: usize,
        q8_buf: &mut [u8],
        scale_buf: &mut [f32],
    ) {
        if let Some(weight) = &self.f32_data {
            for t in 0..n_tokens {
                let input = &input[t * self.n_in..(t + 1) * self.n_in];
                let output = &mut output[t * self.n_out..(t + 1) * self.n_out];
                for o in 0..self.n_out {
                    output[o] = input
                        .iter()
                        .zip(&weight[o * self.n_in..(o + 1) * self.n_in])
                        .map(|(x, w)| x * w)
                        .sum();
                }
            }
            return;
        }
        let n_in = self.n_in;
        let n_out = self.n_out;
        let blocks = n_in / 32;
        let weight = &self.data;
        for t in 0..n_tokens {
            crate::ops::quantize_q8_0_into(
                &input[t * n_in..(t + 1) * n_in],
                n_in,
                &mut q8_buf[t * n_in..(t + 1) * n_in],
                &mut scale_buf[t * blocks..(t + 1) * blocks],
            );
        }
        let total_rows = n_tokens * n_out;
        if total_rows >= 256 {
            output
                .par_chunks_mut(n_out)
                .enumerate()
                .for_each(|(t, out_chunk)| {
                    let q8_off = t * n_in;
                    let scale_off = t * blocks;
                    crate::ops::matmul_q8_0_quantized_parallel(
                        weight,
                        &q8_buf[q8_off..q8_off + n_in],
                        &scale_buf[scale_off..scale_off + blocks],
                        out_chunk,
                        n_in,
                        n_out,
                    );
                });
        } else {
            for t in 0..n_tokens {
                let q8_off = t * n_in;
                let scale_off = t * blocks;
                crate::ops::matmul_q8_0_quantized_parallel(
                    weight,
                    &q8_buf[q8_off..q8_off + n_in],
                    &scale_buf[scale_off..scale_off + blocks],
                    &mut output[t * n_out..(t + 1) * n_out],
                    n_in,
                    n_out,
                );
            }
        }
    }

    fn matmul_single(
        &self,
        input: &[f32],
        output: &mut [f32],
        q8_buf: &mut [u8],
        scale_buf: &mut [f32],
    ) {
        if let Some(weight) = &self.f32_data {
            for o in 0..self.n_out {
                output[o] = input
                    .iter()
                    .zip(&weight[o * self.n_in..(o + 1) * self.n_in])
                    .map(|(x, w)| x * w)
                    .sum();
            }
            return;
        }
        let n_in = self.n_in;
        let n_out = self.n_out;
        let blocks = n_in / 32;
        crate::ops::quantize_q8_0_into(input, n_in, &mut q8_buf[..n_in], &mut scale_buf[..blocks]);
        crate::ops::matmul_q8_0_quantized_parallel(
            &self.data,
            &q8_buf[..n_in],
            &scale_buf[..blocks],
            output,
            n_in,
            n_out,
        );
    }
}

pub struct VisionEncoder<'a> {
    pub config: ClipVisionConfig,
    pub patch_embd_weight: &'a [u8],
    pub patch_embd_weight_1: Option<&'a [u8]>,
    pub position_embd: Option<&'a [u8]>,
    pub post_ln_weight: Option<&'a [u8]>,
    pub post_ln_bias: Option<&'a [u8]>,
    pub patch_bias: Option<&'a [u8]>,
    pub layers: Vec<VisionLayer<'a>>,
    pub mm_0_weight: &'a [u8],
    pub mm_0_bias: Option<&'a [u8]>,
    pub mm_2_weight: &'a [u8],
    pub mm_2_bias: Option<&'a [u8]>,
    pub precomputed: Option<VisionPrecomputed>,
}

pub struct VisionPrecomputed {
    pub qkv_weights: Vec<Option<Q8Weight>>,
    pub qkv_biases: Vec<Option<Vec<f32>>>,
    pub q_weights: Vec<Option<Q8Weight>>,
    pub q_biases: Vec<Option<Vec<f32>>>,
    pub k_weights: Vec<Option<Q8Weight>>,
    pub k_biases: Vec<Option<Vec<f32>>>,
    pub v_weights: Vec<Option<Q8Weight>>,
    pub v_biases: Vec<Option<Vec<f32>>>,
    pub out_weights: Vec<Q8Weight>,
    pub out_biases: Vec<Option<Vec<f32>>>,
    pub ffn_up_weights: Vec<Option<Q8Weight>>,
    pub ffn_up_biases: Vec<Option<Vec<f32>>>,
    pub ffn_gate_weights: Vec<Option<Q8Weight>>,
    pub ffn_gate_biases: Vec<Option<Vec<f32>>>,
    pub ffn_down_weights: Vec<Option<Q8Weight>>,
    pub ffn_down_biases: Vec<Option<Vec<f32>>>,
    pub ln1_weights: Vec<Vec<f32>>,
    pub ln1_biases: Vec<Option<Vec<f32>>>,
    pub ln2_weights: Vec<Vec<f32>>,
    pub ln2_biases: Vec<Option<Vec<f32>>>,
    pub post_ln_weight: Vec<f32>,
    pub post_ln_bias: Option<Vec<f32>>,
    pub patch_bias: Option<Vec<f32>>,
    pub mm_0_weight: Q8Weight,
    pub mm_0_bias: Option<Vec<f32>>,
    pub mm_2_weight: Q8Weight,
    pub mm_2_bias: Option<Vec<f32>>,
}

pub struct VisionLayer<'a> {
    pub ln1_weight: &'a [u8],
    pub ln1_bias: Option<&'a [u8]>,
    pub ln2_weight: &'a [u8],
    pub ln2_bias: Option<&'a [u8]>,
    pub qkv_weight: Option<&'a [u8]>,
    pub qkv_bias: Option<&'a [u8]>,
    pub q_weight: Option<&'a [u8]>,
    pub q_bias: Option<&'a [u8]>,
    pub k_weight: Option<&'a [u8]>,
    pub k_bias: Option<&'a [u8]>,
    pub v_weight: Option<&'a [u8]>,
    pub v_bias: Option<&'a [u8]>,
    pub out_weight: &'a [u8],
    pub out_bias: Option<&'a [u8]>,
    pub ffn_up_weight: &'a [u8],
    pub ffn_up_bias: Option<&'a [u8]>,
    pub ffn_gate_weight: Option<&'a [u8]>,
    pub ffn_gate_bias: Option<&'a [u8]>,
    pub ffn_down_weight: &'a [u8],
    pub ffn_down_bias: Option<&'a [u8]>,
}

impl<'a> VisionEncoder<'a> {
    pub fn from_source<S: TensorSource + ?Sized>(source: &'a S) -> Result<Self, String> {
        let mut config = ClipVisionConfig::from_source(source)?;

        if let Some(info) = source.tensor_info("v.blk.0.ffn_up.weight") {
            if info.dims.len() == 2 {
                if let Ok(width) = usize::try_from(info.dims[1]) {
                    config.n_ff = width;
                }
            }
        }

        let patch_embd_weight = source
            .tensor_slice("v.patch_embd.weight")
            .ok_or("Missing v.patch_embd.weight")?;
        let patch_embd_weight_1 = source.tensor_slice("v.patch_embd.weight.1");
        let position_embd = source.tensor_slice("v.position_embd.weight");
        let post_ln_weight = source.tensor_slice("v.post_ln.weight");
        let post_ln_bias = source.tensor_slice("v.post_ln.bias");
        let patch_bias = source.tensor_slice("v.patch_embd.bias");

        let mut layers = Vec::with_capacity(config.n_layer);
        for i in 0..config.n_layer {
            let ln1_weight = source
                .tensor_slice(&format!("v.blk.{}.ln1.weight", i))
                .ok_or_else(|| format!("Missing v.blk.{}.ln1.weight", i))?;
            let ln1_bias = source.tensor_slice(&format!("v.blk.{}.ln1.bias", i));
            let ln2_weight = source
                .tensor_slice(&format!("v.blk.{}.ln2.weight", i))
                .ok_or_else(|| format!("Missing v.blk.{}.ln2.weight", i))?;
            let ln2_bias = source.tensor_slice(&format!("v.blk.{}.ln2.bias", i));
            let qkv_weight = source.tensor_slice(&format!("v.blk.{}.attn_qkv.weight", i));
            let qkv_bias = source.tensor_slice(&format!("v.blk.{}.attn_qkv.bias", i));
            let (q_weight, q_bias, k_weight, k_bias, v_weight, v_bias) = if qkv_weight.is_none() {
                (
                    Some(
                        source
                            .tensor_slice(&format!("v.blk.{}.attn_q.weight", i))
                            .ok_or_else(|| format!("Missing v.blk.{}.attn_q.weight", i))?,
                    ),
                    source.tensor_slice(&format!("v.blk.{}.attn_q.bias", i)),
                    Some(
                        source
                            .tensor_slice(&format!("v.blk.{}.attn_k.weight", i))
                            .ok_or_else(|| format!("Missing v.blk.{}.attn_k.weight", i))?,
                    ),
                    source.tensor_slice(&format!("v.blk.{}.attn_k.bias", i)),
                    Some(
                        source
                            .tensor_slice(&format!("v.blk.{}.attn_v.weight", i))
                            .ok_or_else(|| format!("Missing v.blk.{}.attn_v.weight", i))?,
                    ),
                    source.tensor_slice(&format!("v.blk.{}.attn_v.bias", i)),
                )
            } else {
                (None, None, None, None, None, None)
            };
            let out_weight = source
                .tensor_slice(&format!("v.blk.{}.attn_out.weight", i))
                .ok_or_else(|| format!("Missing v.blk.{}.attn_out.weight", i))?;
            let out_bias = source.tensor_slice(&format!("v.blk.{}.attn_out.bias", i));
            let ffn_up_weight = source
                .tensor_slice(&format!("v.blk.{}.ffn_up.weight", i))
                .ok_or_else(|| format!("Missing v.blk.{}.ffn_up.weight", i))?;
            let ffn_up_bias = source.tensor_slice(&format!("v.blk.{}.ffn_up.bias", i));
            let ffn_gate_weight = source.tensor_slice(&format!("v.blk.{}.ffn_gate.weight", i));
            let ffn_gate_bias = source.tensor_slice(&format!("v.blk.{}.ffn_gate.bias", i));
            let ffn_down_weight = source
                .tensor_slice(&format!("v.blk.{}.ffn_down.weight", i))
                .ok_or_else(|| format!("Missing v.blk.{}.ffn_down.weight", i))?;
            let ffn_down_bias = source.tensor_slice(&format!("v.blk.{}.ffn_down.bias", i));

            layers.push(VisionLayer {
                ln1_weight,
                ln1_bias,
                ln2_weight,
                ln2_bias,
                qkv_weight,
                qkv_bias,
                q_weight,
                q_bias,
                k_weight,
                k_bias,
                v_weight,
                v_bias,
                out_weight,
                out_bias,
                ffn_up_weight,
                ffn_up_bias,
                ffn_gate_weight,
                ffn_gate_bias,
                ffn_down_weight,
                ffn_down_bias,
            });
        }

        let mm_0_weight = source
            .tensor_slice("mm.0.weight")
            .ok_or("Missing mm.0.weight")?;
        let mm_0_bias = source.tensor_slice("mm.0.bias");
        let mm_2_info = source
            .tensor_info("mm.2.weight")
            .ok_or("Missing tensor info: mm.2.weight")?;
        let expected_mm_2_input = config
            .n_embd
            .checked_mul(config.spatial_merge_size)
            .and_then(|value| value.checked_mul(config.spatial_merge_size))
            .ok_or("Vision projector input width overflow")?;
        if mm_2_info.dims.len() != 2
            || usize::try_from(mm_2_info.dims[0]).ok() != Some(expected_mm_2_input)
            || mm_2_info.dims[1] == 0
        {
            return Err(format!(
                "Invalid mm.2.weight shape {:?}; expected [{expected_mm_2_input}, output]",
                mm_2_info.dims
            ));
        }
        config.projection_dim = usize::try_from(mm_2_info.dims[1])
            .map_err(|_| "mm.2.weight output width does not fit usize")?;
        let mm_2_weight = source
            .tensor_slice("mm.2.weight")
            .ok_or("Missing mm.2.weight")?;
        let mm_2_bias = source.tensor_slice("mm.2.bias");

        Ok(Self {
            config,
            patch_embd_weight,
            patch_embd_weight_1,
            position_embd,
            post_ln_weight,
            post_ln_bias,
            patch_bias,
            layers,
            mm_0_weight,
            mm_0_bias,
            mm_2_weight,
            mm_2_bias,
            precomputed: None,
        })
    }

    pub fn precompute(&mut self) {
        let n_layer = self.config.n_layer;
        let n_embd = self.config.n_embd;
        let n_ff = self.config.n_ff;
        let mut qkv_weights = Vec::with_capacity(n_layer);
        let mut qkv_biases = Vec::with_capacity(n_layer);
        let mut q_weights = Vec::with_capacity(n_layer);
        let mut q_biases = Vec::with_capacity(n_layer);
        let mut k_weights = Vec::with_capacity(n_layer);
        let mut k_biases = Vec::with_capacity(n_layer);
        let mut v_weights = Vec::with_capacity(n_layer);
        let mut v_biases = Vec::with_capacity(n_layer);
        let mut out_weights = Vec::with_capacity(n_layer);
        let mut out_biases = Vec::with_capacity(n_layer);
        let mut ffn_up_weights = Vec::with_capacity(n_layer);
        let mut ffn_up_biases = Vec::with_capacity(n_layer);
        let mut ffn_gate_weights = Vec::with_capacity(n_layer);
        let mut ffn_gate_biases = Vec::with_capacity(n_layer);
        let mut ffn_down_weights = Vec::with_capacity(n_layer);
        let mut ffn_down_biases = Vec::with_capacity(n_layer);
        let mut ln1_weights = Vec::with_capacity(n_layer);
        let mut ln1_biases = Vec::with_capacity(n_layer);
        let mut ln2_weights = Vec::with_capacity(n_layer);
        let mut ln2_biases = Vec::with_capacity(n_layer);

        for layer in &self.layers {
            if let Some(weight) = layer.qkv_weight {
                let qkv_f32 = decode_linear_weight(weight, n_embd, n_embd * 3);
                qkv_weights.push(Some(Q8Weight::from_f32(&qkv_f32, n_embd, n_embd * 3)));
                q_weights.push(None);
                k_weights.push(None);
                v_weights.push(None);
            } else {
                qkv_weights.push(None);
                let q = decode_linear_weight(layer.q_weight.expect("missing q weight"), n_embd, n_embd);
                let k = decode_linear_weight(layer.k_weight.expect("missing k weight"), n_embd, n_embd);
                let v = decode_linear_weight(layer.v_weight.expect("missing v weight"), n_embd, n_embd);
                q_weights.push(Some(Q8Weight::from_f32(&q, n_embd, n_embd)));
                k_weights.push(Some(Q8Weight::from_f32(&k, n_embd, n_embd)));
                v_weights.push(Some(Q8Weight::from_f32(&v, n_embd, n_embd)));
            }
            qkv_biases.push(layer.qkv_bias.map(decode_f32_slice));
            q_biases.push(layer.q_bias.map(decode_f32_slice));
            k_biases.push(layer.k_bias.map(decode_f32_slice));
            v_biases.push(layer.v_bias.map(decode_f32_slice));
            let out_f32 = decode_linear_weight(layer.out_weight, n_embd, n_embd);
            out_weights.push(Q8Weight::from_f32(&out_f32, n_embd, n_embd));
            out_biases.push(layer.out_bias.map(decode_f32_slice));
            let ffn_up_f32 = decode_linear_weight(layer.ffn_up_weight, n_embd, n_ff);
            ffn_up_weights.push(Some(Q8Weight::from_f32(&ffn_up_f32, n_embd, n_ff)));
            ffn_up_biases.push(layer.ffn_up_bias.map(decode_f32_slice));
            if let Some(weight) = layer.ffn_gate_weight {
                let gate_f32 = decode_linear_weight(weight, n_embd, n_ff);
                ffn_gate_weights.push(Some(Q8Weight::from_f32(&gate_f32, n_embd, n_ff)));
            } else {
                ffn_gate_weights.push(None);
            }
            ffn_gate_biases.push(layer.ffn_gate_bias.map(decode_f32_slice));
            let ffn_down_f32 = decode_linear_weight(layer.ffn_down_weight, n_ff, n_embd);
            ffn_down_weights.push(Some(Q8Weight::from_f32(&ffn_down_f32, n_ff, n_embd)));
            ffn_down_biases.push(layer.ffn_down_bias.map(decode_f32_slice));
            ln1_weights.push(decode_f32_slice(layer.ln1_weight));
            ln1_biases.push(layer.ln1_bias.map(decode_f32_slice));
            ln2_weights.push(decode_f32_slice(layer.ln2_weight));
            ln2_biases.push(layer.ln2_bias.map(decode_f32_slice));
        }

        let mm_merged_embd =
            n_embd * self.config.spatial_merge_size * self.config.spatial_merge_size;
        let mm0_f32 = decode_linear_weight(self.mm_0_weight, mm_merged_embd, mm_merged_embd);
        let mm2_f32 =
            decode_linear_weight(self.mm_2_weight, mm_merged_embd, self.config.projection_dim);

        self.precomputed = Some(VisionPrecomputed {
            qkv_weights,
            qkv_biases,
            q_weights,
            q_biases,
            k_weights,
            k_biases,
            v_weights,
            v_biases,
            out_weights,
            out_biases,
            ffn_up_weights,
            ffn_up_biases,
            ffn_gate_weights,
            ffn_gate_biases,
            ffn_down_weights,
            ffn_down_biases,
            ln1_weights,
            ln1_biases,
            ln2_weights,
            ln2_biases,
            post_ln_weight: self.post_ln_weight.map_or_else(Vec::new, decode_f32_slice),
            post_ln_bias: self.post_ln_bias.map(decode_f32_slice),
            patch_bias: self.patch_bias.map(decode_f32_slice),
            mm_0_weight: Q8Weight::from_f32(&mm0_f32, mm_merged_embd, mm_merged_embd),
            mm_0_bias: self.mm_0_bias.map(decode_f32_slice),
            mm_2_weight: Q8Weight::from_f32(&mm2_f32, mm_merged_embd, self.config.projection_dim),
            mm_2_bias: self.mm_2_bias.map(decode_f32_slice),
        });
    }

    pub fn encode_image(
        &self,
        image_pixels: &[f32],
        img_w: usize,
        img_h: usize,
        scratch: &mut VisionScratchpad,
    ) -> Result<VisionGrid, String> {
        self.encode_pair(image_pixels, image_pixels, img_w, img_h, scratch)
    }

    pub fn encode_pair(
        &self,
        frame_a: &[f32],
        frame_b: &[f32],
        img_w: usize,
        img_h: usize,
        scratch: &mut VisionScratchpad,
    ) -> Result<VisionGrid, String> {
        let cfg = &self.config;
        let n_embd = cfg.n_embd;
        let merge = cfg.spatial_merge_size;
        let grid = VisionGrid::from_image_size(1, img_w, img_h, cfg.patch_size, merge)?;
        let expected_pixels = checked_len(
            "normalized image",
            &[grid.image_width(), grid.image_height(), 3],
        )?;
        if frame_a.len() != expected_pixels || frame_b.len() != expected_pixels {
            return Err(format!(
                "Normalized frame length mismatch: expected {expected_pixels}, got {} and {}",
                frame_a.len(),
                frame_b.len()
            ));
        }

        let n_patches_x = grid
            .grid_w
            .checked_mul(grid.merge_size)
            .ok_or("Vision patch columns overflow")?;
        let n_patches_y = grid
            .grid_h
            .checked_mul(grid.merge_size)
            .ok_or("Vision patch rows overflow")?;
        let n_patches = checked_len(
            "vision patch grid",
            &[grid.grid_t, n_patches_x, n_patches_y],
        )?;
        let n_tokens = n_patches;
        let n_projected = checked_len(
            "projected token grid",
            &[grid.grid_t, grid.grid_h, grid.grid_w],
        )?;
        let merge_area = checked_len("vision merge area", &[merge, merge])?;
        if n_tokens / merge_area != n_projected {
            return Err(format!(
                "Vision token mismatch: patches={n_tokens}, merge_area={merge_area}, projected={n_projected}"
            ));
        }

        scratch.ensure_capacity(cfg, grid)?;
        let t_embed = std::time::Instant::now();
        self.patch_embed(
            frame_a,
            frame_b,
            grid.image_width(),
            grid.image_height(),
            scratch,
        );
        let t_embed = t_embed.elapsed();
        let t_merge = std::time::Instant::now();
        spatial_merge(
            &mut scratch.patch_embd[..n_patches * n_embd],
            n_patches_x,
            n_patches_y,
            n_embd,
            merge,
            &mut scratch.merged[..n_patches * n_embd],
        );
        let t_merge = t_merge.elapsed();

        let t_bias = std::time::Instant::now();
        if let Some(ref precomputed) = self.precomputed {
            if let Some(ref bias) = precomputed.patch_bias {
                for token in 0..n_tokens {
                    let offset = token * n_embd;
                    for embd in 0..n_embd {
                        scratch.merged[offset + embd] += bias[embd];
                    }
                }
            }
        } else if let Some(patch_bias_data) = self.patch_bias {
            let bias = decode_f32_slice(patch_bias_data);
            for token in 0..n_tokens {
                let offset = token * n_embd;
                for embd in 0..n_embd {
                    scratch.merged[offset + embd] += bias[embd];
                }
            }
        }
        let t_bias = t_bias.elapsed();
        #[cfg(feature = "parity-trace")]
        crate::parity_trace::report(crate::parity_trace::checkpoint(
            "omni.vision.patch_bias",
            None,
            &[n_tokens, n_embd],
            &scratch.merged[..n_tokens * n_embd],
        ));

        let t_pos = std::time::Instant::now();
        if let Some(position_data) = self.position_embd {
            self.apply_position_embedding_merged(
                &mut scratch.merged[..n_tokens * n_embd],
                n_patches_x,
                n_patches_y,
                n_embd,
                merge,
                position_data,
                &mut scratch.pos_embd_buf,
            );
        }
        let t_pos = t_pos.elapsed();
        #[cfg(feature = "parity-trace")]
        crate::parity_trace::report(crate::parity_trace::checkpoint(
            "omni.vision.inp_pos_emb",
            None,
            &[n_tokens, n_embd],
            &scratch.merged[..n_tokens * n_embd],
        ));

        let t_layers = std::time::Instant::now();
        let mrope_positions = build_vit_mrope_positions(n_patches_x, n_patches_y, merge);
        for layer in 0..cfg.n_layer {
            self.forward_vit_layer(layer, scratch, n_tokens, &mrope_positions);
            #[cfg(feature = "parity-trace")]
            if layer == 0 || layer + 1 == cfg.n_layer {
                crate::parity_trace::report(crate::parity_trace::checkpoint(
                    "omni.vision.layer_out",
                    Some(layer),
                    &[n_tokens, n_embd],
                    &scratch.merged[..n_tokens * n_embd],
                ));
            }
        }
        let t_layers = t_layers.elapsed();

        let t_postln = std::time::Instant::now();
        if let Some(ref precomputed) = self.precomputed {
            if !precomputed.post_ln_weight.is_empty() {
                for token in 0..n_tokens {
                    let offset = token * n_embd;
                    if let Some(ref bias) = precomputed.post_ln_bias {
                        layer_norm_with_bias(
                            &mut scratch.merged[offset..offset + n_embd],
                            &precomputed.post_ln_weight,
                            bias,
                            cfg.eps,
                        );
                    } else {
                        layer_norm_without_bias(
                            &mut scratch.merged[offset..offset + n_embd],
                            &precomputed.post_ln_weight,
                            cfg.eps,
                        );
                    }
                }
            }
        } else if let (Some(weight_data), Some(bias_data)) =
            (self.post_ln_weight, self.post_ln_bias)
        {
            let weight = decode_f32_slice(weight_data);
            let bias = decode_f32_slice(bias_data);
            for token in 0..n_tokens {
                let offset = token * n_embd;
                layer_norm_with_bias(
                    &mut scratch.merged[offset..offset + n_embd],
                    &weight,
                    &bias,
                    cfg.eps,
                );
            }
        }
        let t_postln = t_postln.elapsed();

        let t_proj = std::time::Instant::now();
        self.project(n_patches_x, n_patches_y, n_embd, merge, scratch);
        let t_proj = t_proj.elapsed();

        let total = t_embed + t_merge + t_bias + t_pos + t_layers + t_postln + t_proj;
        eprintln!(
            "[vision-timing] total={:.3}s  patch_embed={:.3}s ({:.1}%)  spatial_merge={:.3}s ({:.1}%)  patch_bias={:.3}s ({:.1}%)  pos_emb={:.3}s ({:.1}%)  vit_layers={:.3}s ({:.1}%)  post_ln={:.3}s ({:.1}%)  projection={:.3}s ({:.1}%)",
            total.as_secs_f64(),
            t_embed.as_secs_f64(), t_embed.as_secs_f64()/total.as_secs_f64()*100.0,
            t_merge.as_secs_f64(), t_merge.as_secs_f64()/total.as_secs_f64()*100.0,
            t_bias.as_secs_f64(), t_bias.as_secs_f64()/total.as_secs_f64()*100.0,
            t_pos.as_secs_f64(), t_pos.as_secs_f64()/total.as_secs_f64()*100.0,
            t_layers.as_secs_f64(), t_layers.as_secs_f64()/total.as_secs_f64()*100.0,
            t_postln.as_secs_f64(), t_postln.as_secs_f64()/total.as_secs_f64()*100.0,
            t_proj.as_secs_f64(), t_proj.as_secs_f64()/total.as_secs_f64()*100.0,
        );
        let expected_projected = checked_len(
            "projected vision output",
            &[n_projected, cfg.projection_dim],
        )?;
        if scratch.projected.len() != expected_projected {
            return Err(format!(
                "Projected vision length mismatch: expected {expected_projected}, got {}",
                scratch.projected.len()
            ));
        }
        Ok(grid)
    }

    fn patch_embed(
        &self,
        frame_a: &[f32],
        frame_b: &[f32],
        img_w: usize,
        img_h: usize,
        scratch: &mut VisionScratchpad,
    ) {
        let cfg = &self.config;
        let ps = cfg.patch_size;
        let n_embd = cfg.n_embd;

        if scratch.patch_weight_buf.is_empty() {
            let patch_dim = 3 * ps * ps;
            scratch.patch_weight_buf =
                decode_linear_weight(self.patch_embd_weight, patch_dim, n_embd);
            scratch.patch_weight_1_buf = self
                .patch_embd_weight_1
                .map(|data| decode_linear_weight(data, patch_dim, n_embd));
        }
        let w0 = &scratch.patch_weight_buf;
        let w1 = scratch.patch_weight_1_buf.as_deref();

        #[cfg(target_arch = "x86_64")]
        if crate::ops::has_avx2_fma() && ps == 16 && n_embd % 8 == 0 {
            if scratch.patch_weight_packed.is_empty() {
                let patch_dim = 3 * ps * ps;
                scratch.patch_weight_packed = repack_patch_weights(w0, n_embd, patch_dim);
                scratch.patch_weight_1_packed =
                    w1.map(|d| repack_patch_weights(d, n_embd, patch_dim));
            }
            unsafe {
                patch_embed_simd_avx2(
                    frame_a,
                    frame_b,
                    img_w,
                    img_h,
                    &scratch.patch_weight_packed,
                    scratch.patch_weight_1_packed.as_deref(),
                    &mut scratch.patch_embd,
                    ps,
                    n_embd,
                );
            }
            return;
        }

        patch_embed_scalar(
            frame_a,
            frame_b,
            img_w,
            img_h,
            w0,
            w1,
            &mut scratch.patch_embd,
            ps,
            n_embd,
        );
    }

    fn apply_position_embedding_merged(
        &self,
        merged: &mut [f32],
        n_patches_x: usize,
        n_patches_y: usize,
        n_embd: usize,
        merge: usize,
        pos_data: &[u8],
        pos_merged_buf: &mut [f32],
    ) {
        let pos_len = pos_data.len() / 4;
        let pos_per_side = (pos_len / n_embd) as usize;
        let pos_side = (pos_per_side as f64).sqrt() as usize;

        let decoded_pos: Vec<f32>;
        if pos_side == n_patches_x && pos_side == n_patches_y {
            decoded_pos = decode_f32_slice(pos_data);
        } else {
            let raw = decode_f32_slice(pos_data);
            decoded_pos =
                bilinear_resize_2d(&raw, pos_side, pos_side, n_embd, n_patches_y, n_patches_x);
        }

        let total = n_patches_x * n_patches_y * n_embd;
        spatial_merge(
            &decoded_pos[..total],
            n_patches_x,
            n_patches_y,
            n_embd,
            merge,
            &mut pos_merged_buf[..total],
        );

        for i in 0..total {
            if i < merged.len() {
                merged[i] += pos_merged_buf[i];
            }
        }
    }

    fn forward_vit_layer(
        &self,
        il: usize,
        scratch: &mut VisionScratchpad,
        n_tokens: usize,
        mrope_positions: &[[usize; 4]],
    ) {
        let do_profile = std::env::var("PROFILE_VIT_LAYER").is_ok();
        let cfg = &self.config;
        let n_embd = cfg.n_embd;
        let n_head = cfg.n_head;
        let d_head = cfg.d_head();
        let eps = cfg.eps;

        let t0_res = std::time::Instant::now();
        scratch.residual[..n_tokens * n_embd].copy_from_slice(&scratch.merged[..n_tokens * n_embd]);

        let (
            mut t_ln1,
            mut t_qkv,
            mut t_rope,
            mut t_attn,
            mut t_attn_out,
            mut t_ln2_ffn,
            mut t_act,
            mut t_ffn_down,
        ) = (
            0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64,
        );

        let t_ln1_start = std::time::Instant::now();
        if let Some(ref pc) = self.precomputed {
            if let Some(ref b) = pc.ln1_biases[il] {
                for t in 0..n_tokens {
                    let off = t * n_embd;
                    layer_norm_with_bias(
                        &mut scratch.merged[off..off + n_embd],
                        &pc.ln1_weights[il],
                        b,
                        eps,
                    );
                }
            } else {
                for t in 0..n_tokens {
                    let off = t * n_embd;
                    layer_norm_without_bias(
                        &mut scratch.merged[off..off + n_embd],
                        &pc.ln1_weights[il],
                        eps,
                    );
                }
            }

            if let Some(ref qkv) = pc.qkv_weights[il] {
                qkv.matmul_batch(
                    &scratch.merged[..n_tokens * n_embd],
                    &mut scratch.qkv_buf[..n_tokens * n_embd * 3],
                    n_tokens,
                    &mut scratch.q8_buf,
                    &mut scratch.q8_scale_buf,
                );
                if let Some(ref bias) = pc.qkv_biases[il] {
                    for t in 0..n_tokens {
                        let off = t * n_embd * 3;
                        vec_add_into(bias.as_slice(), &mut scratch.qkv_buf[off..off + n_embd * 3]);
                    }
                }
            } else {
                let input = &scratch.merged[..n_tokens * n_embd];
                for (weight, bias, output_offset) in [
                    (&pc.q_weights[il], &pc.q_biases[il], 0usize),
                    (&pc.k_weights[il], &pc.k_biases[il], n_tokens * n_embd),
                    (&pc.v_weights[il], &pc.v_biases[il], 2 * n_tokens * n_embd),
                ] {
                    weight
                        .as_ref()
                        .expect("missing separate vision attention weight")
                        .matmul_batch(
                            input,
                            &mut scratch.separate_qkv_buf
                                [output_offset..output_offset + n_tokens * n_embd],
                            n_tokens,
                            &mut scratch.q8_buf,
                            &mut scratch.q8_scale_buf,
                        );
                    if let Some(bias) = bias {
                        let output = &mut scratch.separate_qkv_buf
                            [output_offset..output_offset + n_tokens * n_embd];
                        for row in output.chunks_exact_mut(n_embd) {
                            vec_add_into(bias.as_slice(), row);
                        }
                    }
                }
                for t in 0..n_tokens {
                    let dst = &mut scratch.qkv_buf[t * n_embd * 3..(t + 1) * n_embd * 3];
                    for part in 0..3 {
                        let src = &scratch.separate_qkv_buf
                            [part * n_tokens * n_embd + t * n_embd..part * n_tokens * n_embd + (t + 1) * n_embd];
                        dst[part * n_embd..(part + 1) * n_embd].copy_from_slice(src);
                    }
                }
            }
        } else {
            let layer = &self.layers[il];

            if let Some(bias_data) = layer.ln1_bias {
                let w = decode_f32_slice(layer.ln1_weight);
                let b = decode_f32_slice(bias_data);
                for t in 0..n_tokens {
                    let off = t * n_embd;
                    layer_norm_with_bias(&mut scratch.merged[off..off + n_embd], &w, &b, eps);
                }
            } else {
                let w = decode_f32_slice(layer.ln1_weight);
                for t in 0..n_tokens {
                    let off = t * n_embd;
                    layer_norm_without_bias(&mut scratch.merged[off..off + n_embd], &w, eps);
                }
            }

            if let Some(qkv_weight) = layer.qkv_weight {
                for t in 0..n_tokens {
                    let inp_off = t * n_embd;
                    let out_off = t * n_embd * 3;
                    matmul_f16_f32_single(
                        qkv_weight,
                        &scratch.merged[inp_off..inp_off + n_embd],
                        &mut scratch.qkv_buf[out_off..out_off + n_embd * 3],
                        n_embd,
                        n_embd * 3,
                    );
                }
                if let Some(bias_data) = layer.qkv_bias {
                    let bias = decode_f32_slice(bias_data);
                    for t in 0..n_tokens {
                        for j in 0..n_embd * 3 {
                            scratch.qkv_buf[t * n_embd * 3 + j] += bias[j];
                        }
                    }
                }
            } else {
                let input = &scratch.merged[..n_tokens * n_embd];
                for (weight, bias, output_offset) in [
                    (layer.q_weight, layer.q_bias, 0usize),
                    (layer.k_weight, layer.k_bias, n_tokens * n_embd),
                    (layer.v_weight, layer.v_bias, 2 * n_tokens * n_embd),
                ] {
                    for t in 0..n_tokens {
                        let src = &input[t * n_embd..(t + 1) * n_embd];
                        let dst = &mut scratch.separate_qkv_buf
                            [output_offset + t * n_embd..output_offset + (t + 1) * n_embd];
                        matmul_f16_f32_single(
                            weight.expect("missing separate vision attention weight"),
                            src,
                            dst,
                            n_embd,
                            n_embd,
                        );
                        if let Some(bias_data) = bias {
                            vec_add_into(&decode_f32_slice(bias_data), dst);
                        }
                    }
                }
                for t in 0..n_tokens {
                    let dst = &mut scratch.qkv_buf[t * n_embd * 3..(t + 1) * n_embd * 3];
                    for part in 0..3 {
                        let src = &scratch.separate_qkv_buf
                            [part * n_tokens * n_embd + t * n_embd..part * n_tokens * n_embd + (t + 1) * n_embd];
                        dst[part * n_embd..(part + 1) * n_embd].copy_from_slice(src);
                    }
                }
            }
        }
        t_ln1 = t_ln1_start.elapsed().as_secs_f64();

        let t_rope_start = std::time::Instant::now();
        for t in 0..n_tokens {
            let src_off = t * n_embd * 3;
            for h in 0..n_head {
                for d in 0..d_head {
                    scratch.attn_buf[h * n_tokens * d_head + t * d_head + d] =
                        scratch.qkv_buf[src_off + h * d_head + d];
                    scratch.attn_buf
                        [n_head * n_tokens * d_head + h * n_tokens * d_head + t * d_head + d] =
                        scratch.qkv_buf[src_off + n_embd + h * d_head + d];
                    scratch.attn_buf
                        [2 * n_head * n_tokens * d_head + h * n_tokens * d_head + t * d_head + d] =
                        scratch.qkv_buf[src_off + 2 * n_embd + h * d_head + d];
                }
            }
        }

        let mrope_sections: [i32; 4] = [
            (d_head / 4) as i32,
            (d_head / 4) as i32,
            (d_head / 4) as i32,
            (d_head / 4) as i32,
        ];
        let freq_base = 10000.0f32;
        for h in 0..n_head {
            let q_base = h * n_tokens * d_head;
            let k_base = n_head * n_tokens * d_head + h * n_tokens * d_head;
            for t in 0..n_tokens {
                rope_vision(
                    &mut scratch.attn_buf[q_base + t * d_head..q_base + t * d_head + d_head],
                    mrope_positions[t],
                    mrope_sections,
                    d_head,
                    freq_base,
                    d_head / 2,
                );
                rope_vision(
                    &mut scratch.attn_buf[k_base + t * d_head..k_base + t * d_head + d_head],
                    mrope_positions[t],
                    mrope_sections,
                    d_head,
                    freq_base,
                    d_head / 2,
                );
            }
        }
        t_rope = t_rope_start.elapsed().as_secs_f64();

        let t_attn_start = std::time::Instant::now();
        let scale = 1.0 / (d_head as f32).sqrt();
        let attn_buf = &scratch.attn_buf[..3 * n_head * n_tokens * d_head];

        let scores = scratch.score_buf.as_mut_ptr();
        let out_buf = scratch.attn_out_buf.as_mut_ptr();

        struct PtrWrap(*mut f32);
        unsafe impl Sync for PtrWrap {}
        unsafe impl Send for PtrWrap {}
        impl PtrWrap {
            unsafe fn slice(&self, offset: usize, len: usize) -> &mut [f32] {
                std::slice::from_raw_parts_mut(self.0.add(offset), len)
            }
        }

        let sw = PtrWrap(scores);
        let ow = PtrWrap(out_buf);
        #[cfg(target_arch = "x86_64")]
        let use_avx2 = crate::ops::has_avx2_fma();

        (0..n_head).into_par_iter().for_each(move |h| {
            let q_base = h * n_tokens * d_head;
            let k_base = n_head * n_tokens * d_head + h * n_tokens * d_head;
            let v_base = 2 * n_head * n_tokens * d_head + h * n_tokens * d_head;
            let score_off = h * n_tokens * n_tokens;
            unsafe {
                let score_slice = sw.slice(score_off, n_tokens * n_tokens);
                let out_slice = ow.slice(h * n_tokens * d_head, n_tokens * d_head);
                for t in 0..n_tokens {
                    let q_ptr = attn_buf.as_ptr().add(q_base + t * d_head);
                    #[cfg(target_arch = "x86_64")]
                    if use_avx2 {
                        unsafe {
                            attention_qk_avx2(
                                q_ptr,
                                attn_buf.as_ptr().add(k_base),
                                &mut score_slice[t * n_tokens..t * n_tokens + n_tokens],
                                n_tokens,
                                d_head,
                                scale,
                            );
                        }
                    } else {
                        for s in 0..n_tokens {
                            let k_ptr = attn_buf.as_ptr().add(k_base + s * d_head);
                            let mut sum = 0.0f32;
                            for i in 0..d_head {
                                sum += *q_ptr.add(i) * *k_ptr.add(i);
                            }
                            score_slice[t * n_tokens + s] = sum * scale;
                        }
                    }
                    #[cfg(not(target_arch = "x86_64"))]
                    {
                        let q_slice = std::slice::from_raw_parts(q_ptr, d_head);
                        for s in 0..n_tokens {
                            let k_ptr = attn_buf.as_ptr().add(k_base + s * d_head);
                            let k_slice = std::slice::from_raw_parts(k_ptr, d_head);
                            score_slice[t * n_tokens + s] =
                                dot_f32(q_slice, k_slice, d_head) * scale;
                        }
                    }
                    softmax_inplace(&mut score_slice[t * n_tokens..t * n_tokens + n_tokens]);

                    let out_base = t * d_head;
                    for d in 0..d_head {
                        out_slice[out_base + d] = 0.0;
                    }
                    #[cfg(target_arch = "x86_64")]
                    if use_avx2 {
                        for s in 0..n_tokens {
                            let sc = score_slice[t * n_tokens + s];
                            unsafe {
                                attn_scaled_add_avx2(
                                    &mut out_slice[out_base..out_base + d_head],
                                    attn_buf.as_ptr().add(v_base + s * d_head),
                                    sc,
                                    d_head,
                                );
                            }
                        }
                    } else {
                        for s in 0..n_tokens {
                            let sc = score_slice[t * n_tokens + s];
                            let v_ptr = attn_buf.as_ptr().add(v_base + s * d_head);
                            for d in 0..d_head {
                                out_slice[out_base + d] += sc * *v_ptr.add(d);
                            }
                        }
                    }
                    #[cfg(not(target_arch = "x86_64"))]
                    for s in 0..n_tokens {
                        let v_ptr = attn_buf.as_ptr().add(v_base + s * d_head);
                        let value = std::slice::from_raw_parts(v_ptr, d_head);
                        vec_mad_f32(
                            &mut out_slice[out_base..out_base + d_head],
                            value,
                            score_slice[t * n_tokens + s],
                        );
                    }
                }
            }
        });
        t_attn = t_attn_start.elapsed().as_secs_f64();

        let t_attn_out_start = std::time::Instant::now();
        for t in 0..n_tokens {
            for h in 0..n_head {
                for d in 0..d_head {
                    scratch.attn_concat[t * n_embd + h * d_head + d] =
                        scratch.attn_out_buf[h * n_tokens * d_head + t * d_head + d];
                }
            }
        }

        if let Some(ref pc) = self.precomputed {
            pc.out_weights[il].matmul_batch(
                &scratch.attn_concat[..n_tokens * n_embd],
                &mut scratch.proj_buf[..n_tokens * n_embd],
                n_tokens,
                &mut scratch.q8_buf,
                &mut scratch.q8_scale_buf,
            );
            if let Some(ref bias) = pc.out_biases[il] {
                for t in 0..n_tokens {
                    let off = t * n_embd;
                    vec_add_into(bias.as_slice(), &mut scratch.proj_buf[off..off + n_embd]);
                }
            }
        } else {
            let layer = &self.layers[il];
            for t in 0..n_tokens {
                let inp_off = t * n_embd;
                let out_off = t * n_embd;
                matmul_f16_f32_single(
                    layer.out_weight,
                    &scratch.attn_concat[inp_off..inp_off + n_embd],
                    &mut scratch.proj_buf[out_off..out_off + n_embd],
                    n_embd,
                    n_embd,
                );
            }
            if let Some(bias_data) = layer.out_bias {
                let bias = decode_f32_slice(bias_data);
                for t in 0..n_tokens {
                    let off = t * n_embd;
                    vec_add_into(&bias, &mut scratch.proj_buf[off..off + n_embd]);
                }
            }
        }
        for t in 0..n_tokens {
            let off = t * n_embd;
            vec_add(
                &scratch.residual[off..off + n_embd],
                &scratch.proj_buf[off..off + n_embd],
                &mut scratch.merged[off..off + n_embd],
            );
        }
        t_attn_out = t_attn_out_start.elapsed().as_secs_f64();

        scratch.residual[..n_tokens * n_embd].copy_from_slice(&scratch.merged[..n_tokens * n_embd]);

        let t_ln2_start = std::time::Instant::now();
        if let Some(ref pc) = self.precomputed {
            if let Some(ref b) = pc.ln2_biases[il] {
                for t in 0..n_tokens {
                    let off = t * n_embd;
                    layer_norm_with_bias(
                        &mut scratch.merged[off..off + n_embd],
                        &pc.ln2_weights[il],
                        b,
                        eps,
                    );
                }
            } else {
                for t in 0..n_tokens {
                    let off = t * n_embd;
                    layer_norm_without_bias(
                        &mut scratch.merged[off..off + n_embd],
                        &pc.ln2_weights[il],
                        eps,
                    );
                }
            }

            pc.ffn_up_weights[il].as_ref().expect("missing vision ffn up weight").matmul_batch(
                &scratch.merged[..n_tokens * n_embd],
                &mut scratch.ffn_buf[..n_tokens * cfg.n_ff],
                n_tokens,
                &mut scratch.q8_buf,
                &mut scratch.q8_scale_buf,
            );
            if let Some(ref bias) = pc.ffn_up_biases[il] {
                for t in 0..n_tokens {
                    let off = t * cfg.n_ff;
                    vec_add_into(bias.as_slice(), &mut scratch.ffn_buf[off..off + cfg.n_ff]);
                }
            }
            if let Some(ref gate) = pc.ffn_gate_weights[il] {
                gate.matmul_batch(
                    &scratch.merged[..n_tokens * n_embd],
                    &mut scratch.ffn_gate_buf[..n_tokens * cfg.n_ff],
                    n_tokens,
                    &mut scratch.q8_buf,
                    &mut scratch.q8_scale_buf,
                );
                if let Some(ref bias) = pc.ffn_gate_biases[il] {
                    for t in 0..n_tokens {
                        let off = t * cfg.n_ff;
                        vec_add_into(bias.as_slice(), &mut scratch.ffn_gate_buf[off..off + cfg.n_ff]);
                    }
                }
            }
        } else {
            let layer = &self.layers[il];
            if let Some(bias_data) = layer.ln2_bias {
                let w = decode_f32_slice(layer.ln2_weight);
                let b = decode_f32_slice(bias_data);
                for t in 0..n_tokens {
                    let off = t * n_embd;
                    layer_norm_with_bias(&mut scratch.merged[off..off + n_embd], &w, &b, eps);
                }
            } else {
                let w = decode_f32_slice(layer.ln2_weight);
                for t in 0..n_tokens {
                    let off = t * n_embd;
                    layer_norm_without_bias(&mut scratch.merged[off..off + n_embd], &w, eps);
                }
            }

            let gate_weight = layer.ffn_gate_weight;
            for t in 0..n_tokens {
                let inp_off = t * n_embd;
                let out_off = t * cfg.n_ff;
                matmul_f16_f32_single(
                    layer.ffn_up_weight,
                    &scratch.merged[inp_off..inp_off + n_embd],
                    &mut scratch.ffn_buf[out_off..out_off + cfg.n_ff],
                    n_embd,
                    cfg.n_ff,
                );
                if let Some(gate_weight) = gate_weight {
                    matmul_f16_f32_single(
                        gate_weight,
                        &scratch.merged[inp_off..inp_off + n_embd],
                        &mut scratch.ffn_gate_buf[out_off..out_off + cfg.n_ff],
                        n_embd,
                        cfg.n_ff,
                    );
                }
            }

            if let Some(bias_data) = layer.ffn_up_bias {
                let bias = decode_f32_slice(bias_data);
                for t in 0..n_tokens {
                    let off = t * cfg.n_ff;
                    vec_add_into(&bias, &mut scratch.ffn_buf[off..off + cfg.n_ff]);
                }
            }
            if let Some(bias_data) = layer.ffn_gate_bias {
                let bias = decode_f32_slice(bias_data);
                for t in 0..n_tokens {
                    let off = t * cfg.n_ff;
                    vec_add_into(&bias, &mut scratch.ffn_gate_buf[off..off + cfg.n_ff]);
                }
            }
        }
        t_ln2_ffn = t_ln2_start.elapsed().as_secs_f64();

        let t_act_start = std::time::Instant::now();
        if self.layers[il].ffn_gate_weight.is_some() {
            crate::ops::silu_mul_inplace(
                &scratch.ffn_gate_buf[..n_tokens * cfg.n_ff],
                &mut scratch.ffn_buf[..n_tokens * cfg.n_ff],
            );
        } else {
            gelu_inplace(&mut scratch.ffn_buf[..n_tokens * cfg.n_ff]);
        }
        t_act = t_act_start.elapsed().as_secs_f64();

        let t_ffn_down_start = std::time::Instant::now();
        if let Some(ref pc) = self.precomputed {
            pc.ffn_down_weights[il].as_ref().expect("missing vision ffn down weight").matmul_batch(
                &scratch.ffn_buf[..n_tokens * cfg.n_ff],
                &mut scratch.proj_buf[..n_tokens * n_embd],
                n_tokens,
                &mut scratch.q8_buf,
                &mut scratch.q8_scale_buf,
            );
            if let Some(ref bias) = pc.ffn_down_biases[il] {
                for t in 0..n_tokens {
                    let off = t * n_embd;
                    vec_add_into(bias.as_slice(), &mut scratch.proj_buf[off..off + n_embd]);
                }
            }
        } else {
            let layer = &self.layers[il];
            for t in 0..n_tokens {
                let inp_off = t * cfg.n_ff;
                let out_off = t * n_embd;
                matmul_f16_f32_single(
                    layer.ffn_down_weight,
                    &scratch.ffn_buf[inp_off..inp_off + cfg.n_ff],
                    &mut scratch.proj_buf[out_off..out_off + n_embd],
                    cfg.n_ff,
                    n_embd,
                );
            }

            if let Some(bias_data) = layer.ffn_down_bias {
                let bias = decode_f32_slice(bias_data);
                for t in 0..n_tokens {
                    let off = t * n_embd;
                    vec_add_into(&bias, &mut scratch.proj_buf[off..off + n_embd]);
                }
            }
        }

        for t in 0..n_tokens {
            let off = t * n_embd;
            vec_add(
                &scratch.residual[off..off + n_embd],
                &scratch.proj_buf[off..off + n_embd],
                &mut scratch.merged[off..off + n_embd],
            );
        }
        t_ffn_down = t_ffn_down_start.elapsed().as_secs_f64();

        if do_profile {
            let total_layer = t_ln1 + t_rope + t_attn + t_attn_out + t_ln2_ffn + t_act + t_ffn_down;
            eprintln!(
                "  vit_layer[{}]: ln1+qkv={:.3}s rope={:.3}s attn={:.3}s attn_out={:.3}s ln2+ffn_up={:.3}s act={:.3}s ffn_down={:.3}s | layer={:.3}s",
                il, t_ln1, t_rope, t_attn, t_attn_out, t_ln2_ffn, t_act, t_ffn_down, total_layer
            );
        }
    }

    fn project(
        &self,
        n_patches_x: usize,
        n_patches_y: usize,
        n_embd: usize,
        merge: usize,
        scratch: &mut VisionScratchpad,
    ) {
        let cfg = &self.config;
        let n_merged_x = n_patches_x / merge;
        let n_merged_y = n_patches_y / merge;
        let n_projected = n_merged_x * n_merged_y;
        let proj_dim = cfg.projection_dim;
        let merged_embd = n_embd * merge * merge;
        let hidden = &scratch.merged;
        let out = &mut scratch.projected;

        let concat_size = n_projected * merged_embd;
        if scratch.project_concat_buf.len() < concat_size {
            scratch.project_concat_buf.resize(concat_size, 0.0);
        }
        if scratch.project_mm0_out.len() < concat_size {
            scratch.project_mm0_out.resize(concat_size, 0.0);
        }
        let concat_buf = &mut scratch.project_concat_buf[..concat_size];
        let mm0_out = &mut scratch.project_mm0_out[..concat_size];

        concat_buf.copy_from_slice(&hidden[..concat_size]);

        if let Some(ref pc) = self.precomputed {
            for t in 0..n_projected {
                let src_off = t * merged_embd;
                let dst_off = t * merged_embd;
                pc.mm_0_weight.matmul_single(
                    &concat_buf[src_off..src_off + merged_embd],
                    &mut mm0_out[dst_off..dst_off + merged_embd],
                    &mut scratch.q8_buf,
                    &mut scratch.q8_scale_buf,
                );
            }
            if let Some(ref bias) = pc.mm_0_bias {
                for t in 0..n_projected {
                    for j in 0..bias.len().min(merged_embd) {
                        mm0_out[t * merged_embd + j] += bias[j];
                    }
                }
            }
        } else {
            for t in 0..n_projected {
                let src_off = t * merged_embd;
                let dst_off = t * merged_embd;
                matmul_f16_f32_single(
                    self.mm_0_weight,
                    &concat_buf[src_off..src_off + merged_embd],
                    &mut mm0_out[dst_off..dst_off + merged_embd],
                    merged_embd,
                    merged_embd,
                );
            }
            if let Some(bias_data) = self.mm_0_bias {
                let bias = decode_f32_slice(bias_data);
                for t in 0..n_projected {
                    for j in 0..bias.len().min(merged_embd) {
                        mm0_out[t * merged_embd + j] += bias[j];
                    }
                }
            }
        }

        crate::ops::gelu_inplace(&mut mm0_out[..n_projected * merged_embd]);

        if let Some(ref pc) = self.precomputed {
            for t in 0..n_projected {
                let src_off = t * merged_embd;
                let dst_off = t * proj_dim;
                pc.mm_2_weight.matmul_single(
                    &mm0_out[src_off..src_off + merged_embd],
                    &mut out[dst_off..dst_off + proj_dim],
                    &mut scratch.q8_buf,
                    &mut scratch.q8_scale_buf,
                );
            }
            if let Some(ref bias) = pc.mm_2_bias {
                for t in 0..n_projected {
                    for j in 0..proj_dim {
                        out[t * proj_dim + j] += bias[j];
                    }
                }
            }
        } else {
            for t in 0..n_projected {
                let src_off = t * merged_embd;
                let dst_off = t * proj_dim;
                matmul_f16_f32_single(
                    self.mm_2_weight,
                    &mm0_out[src_off..src_off + merged_embd],
                    &mut out[dst_off..dst_off + proj_dim],
                    merged_embd,
                    proj_dim,
                );
            }
            if let Some(bias_data) = self.mm_2_bias {
                let bias = decode_f32_slice(bias_data);
                for t in 0..n_projected {
                    for j in 0..proj_dim {
                        out[t * proj_dim + j] += bias[j];
                    }
                }
            }
        }
    }
}

fn spatial_merge(
    input: &[f32],
    n_patches_x: usize,
    n_patches_y: usize,
    n_embd: usize,
    merge: usize,
    output: &mut [f32],
) {
    let n_merged_x = n_patches_x / merge;
    let n_merged_y = n_patches_y / merge;

    let mut ptr = 0usize;
    for my in 0..n_merged_y {
        for mx in 0..n_merged_x {
            for dy in 0..merge {
                for dx in 0..merge {
                    let src_py = my * merge + dy;
                    let src_px = mx * merge + dx;
                    let src_idx = src_py * n_patches_x + src_px;
                    let src_off = src_idx * n_embd;
                    let dst_off = ptr * n_embd;
                    for e in 0..n_embd {
                        if dst_off + e < output.len() && src_off + e < input.len() {
                            output[dst_off + e] = input[src_off + e];
                        }
                    }
                    ptr += 1;
                }
            }
        }
    }
}

fn build_vit_mrope_positions(
    n_patches_x: usize,
    n_patches_y: usize,
    merge: usize,
) -> Vec<[usize; 4]> {
    let pw = n_patches_x;
    let ph = n_patches_y;
    let n_tokens = pw * ph;
    let mut positions = vec![[0usize; 4]; n_tokens];

    let mut ptr = 0usize;
    for y in 0..ph {
        for x in 0..pw {
            positions[ptr] = [y, x, y, x];
            ptr += 1;
        }
    }

    let mut merged = vec![[0usize; 4]; n_tokens];
    spatial_merge_positions(&positions, n_patches_x, n_patches_y, merge, &mut merged);
    merged
}

fn spatial_merge_positions(
    input: &[[usize; 4]],
    n_patches_x: usize,
    n_patches_y: usize,
    merge: usize,
    output: &mut [[usize; 4]],
) {
    let n_merged_x = n_patches_x / merge;
    let n_merged_y = n_patches_y / merge;

    let mut ptr = 0usize;
    for my in 0..n_merged_y {
        for mx in 0..n_merged_x {
            for dy in 0..merge {
                for dx in 0..merge {
                    let src_py = my * merge + dy;
                    let src_px = mx * merge + dx;
                    let src_idx = src_py * n_patches_x + src_px;
                    output[ptr] = input[src_idx];
                    ptr += 1;
                }
            }
        }
    }
}

fn bilinear_resize_2d(
    input: &[f32],
    src_h: usize,
    src_w: usize,
    n_embd: usize,
    dst_h: usize,
    dst_w: usize,
) -> Vec<f32> {
    let dst_size = dst_h * dst_w;
    let mut output = vec![0.0f32; dst_size * n_embd];

    for dy in 0..dst_h {
        for dx in 0..dst_w {
            let src_y = if dst_h > 1 {
                dy as f32 * (src_h as f32 - 1.0) / (dst_h as f32 - 1.0)
            } else {
                0.0
            };
            let src_x = if dst_w > 1 {
                dx as f32 * (src_w as f32 - 1.0) / (dst_w as f32 - 1.0)
            } else {
                0.0
            };

            let y0 = src_y.floor() as usize;
            let x0 = src_x.floor() as usize;
            let y1 = (y0 + 1).min(src_h - 1);
            let x1 = (x0 + 1).min(src_w - 1);

            let fy = src_y - y0 as f32;
            let fx = src_x - x0 as f32;

            let dst_idx = dy * dst_w + dx;
            let dst_off = dst_idx * n_embd;

            let i00 = y0 * src_w + x0;
            let i01 = y0 * src_w + x1;
            let i10 = y1 * src_w + x0;
            let i11 = y1 * src_w + x1;

            for e in 0..n_embd {
                let v00 = input[i00 * n_embd + e];
                let v01 = input[i01 * n_embd + e];
                let v10 = input[i10 * n_embd + e];
                let v11 = input[i11 * n_embd + e];

                let v = v00 * (1.0 - fx) * (1.0 - fy)
                    + v01 * fx * (1.0 - fy)
                    + v10 * (1.0 - fx) * fy
                    + v11 * fx * fy;

                output[dst_off + e] = v;
            }
        }
    }

    output
}

pub struct VisionScratchpad {
    pub patch_embd: Vec<f32>,
    pub merged: Vec<f32>,
    pub pos_embd_buf: Vec<f32>,
    pub qkv_buf: Vec<f32>,
    separate_qkv_buf: Vec<f32>,
    pub attn_buf: Vec<f32>,
    pub attn_out_buf: Vec<f32>,
    pub score_buf: Vec<f32>,
    pub proj_buf: Vec<f32>,
    pub ffn_buf: Vec<f32>,
    ffn_gate_buf: Vec<f32>,
    pub projected: Vec<f32>,
    pub attn_concat: Vec<f32>,
    pub residual: Vec<f32>,
    pub patch_weight_buf: Vec<f32>,
    pub patch_weight_1_buf: Option<Vec<f32>>,
    pub patch_weight_packed: Vec<f32>,
    pub patch_weight_1_packed: Option<Vec<f32>>,
    pub project_concat_buf: Vec<f32>,
    pub project_mm0_out: Vec<f32>,
    pub q8_buf: Vec<u8>,
    pub q8_scale_buf: Vec<f32>,
}

impl VisionScratchpad {
    pub fn new(_config: &ClipVisionConfig) -> Self {
        Self {
            patch_embd: Vec::new(),
            merged: Vec::new(),
            pos_embd_buf: Vec::new(),
            qkv_buf: Vec::new(),
            separate_qkv_buf: Vec::new(),
            attn_buf: Vec::new(),
            attn_out_buf: Vec::new(),
            score_buf: Vec::new(),
            proj_buf: Vec::new(),
            ffn_buf: Vec::new(),
            ffn_gate_buf: Vec::new(),
            projected: Vec::new(),
            attn_concat: Vec::new(),
            residual: Vec::new(),
            patch_weight_buf: Vec::new(),
            patch_weight_1_buf: None,
            patch_weight_packed: Vec::new(),
            patch_weight_1_packed: None,
            project_concat_buf: Vec::new(),
            project_mm0_out: Vec::new(),
            q8_buf: Vec::new(),
            q8_scale_buf: Vec::new(),
        }
    }

    fn ensure_capacity(
        &mut self,
        config: &ClipVisionConfig,
        grid: VisionGrid,
    ) -> Result<(), String> {
        let n_patches = checked_len(
            "vision patch",
            &[grid.token_count(), grid.merge_size, grid.merge_size],
        )?;
        let n_tokens = n_patches;
        let n_embd = config.n_embd;
        let n_head = config.n_head;
        let d_head = config.d_head();
        let n_projected = grid.token_count();

        self.patch_embd
            .resize(checked_len("patch_embd", &[n_patches, n_embd])?, 0.0);
        self.merged
            .resize(checked_len("merged", &[n_tokens, n_embd])?, 0.0);
        self.pos_embd_buf
            .resize(checked_len("pos_embd_buf", &[n_tokens, n_embd])?, 0.0);
        self.qkv_buf
            .resize(checked_len("qkv_buf", &[n_tokens, n_embd, 3])?, 0.0);
        self.separate_qkv_buf
            .resize(checked_len("separate_qkv_buf", &[n_tokens, n_embd, 3])?, 0.0);
        self.attn_buf.resize(
            checked_len("attn_buf", &[3, n_head, n_tokens, d_head])?,
            0.0,
        );
        self.attn_out_buf.resize(
            checked_len("attn_out_buf", &[n_head, n_tokens, d_head])?,
            0.0,
        );
        self.score_buf.resize(
            checked_len("score_buf", &[n_head, n_tokens, n_tokens])?,
            0.0,
        );
        self.proj_buf
            .resize(checked_len("proj_buf", &[n_tokens, n_embd])?, 0.0);
        self.ffn_buf
            .resize(checked_len("ffn_buf", &[n_tokens, config.n_ff])?, 0.0);
        self.ffn_gate_buf
            .resize(checked_len("ffn_gate_buf", &[n_tokens, config.n_ff])?, 0.0);
        self.projected.resize(
            checked_len("projected", &[n_projected, config.projection_dim])?,
            0.0,
        );
        self.attn_concat
            .resize(checked_len("attn_concat", &[n_tokens, n_embd])?, 0.0);
        self.residual
            .resize(checked_len("residual", &[n_tokens, n_embd])?, 0.0);
        self.project_concat_buf.resize(
            checked_len(
                "project_concat_buf",
                &[n_projected, n_embd, grid.merge_size, grid.merge_size],
            )?,
            0.0,
        );
        self.project_mm0_out.resize(
            checked_len(
                "project_mm0_out",
                &[n_projected, n_embd, grid.merge_size, grid.merge_size],
            )?,
            0.0,
        );
        let q8_width = config.n_ff.max(n_embd * grid.merge_size * grid.merge_size);
        let q8_width = q8_width.div_ceil(32) * 32;
        self.q8_buf
            .resize(checked_len("q8_buf", &[n_tokens, q8_width])?, 0);
        let q8_values = checked_len("q8_scale_buf", &[n_tokens, q8_width])?;
        if q8_values % 32 != 0 {
            return Err("q8_scale_buf length is not Q8_0 block aligned".into());
        }
        self.q8_scale_buf.resize(q8_values / 32, 0.0);
        Ok(())
    }
}

fn decode_f32_slice(data: &[u8]) -> Vec<f32> {
    let n = data.len() / 4;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(f32_from_le_bytes(&data[i * 4..i * 4 + 4]));
    }
    out
}

fn decode_f16_slice_to_f32(data: &[u8]) -> Vec<f32> {
    let n = data.len() / 2;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let bits = u16::from_le_bytes([data[i * 2], data[i * 2 + 1]]);
        out.push(crate::ops::f16_to_f32(bits));
    }
    out
}

fn decode_linear_weight(data: &[u8], n_in: usize, n_out: usize) -> Vec<f32> {
    let elements = n_in
        .checked_mul(n_out)
        .expect("vision linear weight element count overflow");
    if data.len() == elements * 4 {
        decode_f32_slice(data)
    } else if data.len() == elements * 2 {
        decode_f16_slice_to_f32(data)
    } else if n_in % 32 == 0 && data.len() == n_out * (n_in / 32) * 34 {
        crate::ops::quant::dequant_q80_weight(data, n_in, n_out)
    } else {
        panic!(
            "Unsupported vision linear weight byte length {} for [{n_in}, {n_out}]",
            data.len()
        );
    }
}

fn f32_from_le_bytes(b: &[u8]) -> f32 {
    f32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

fn layer_norm_with_bias(x: &mut [f32], w: &[f32], b: &[f32], eps: f32) {
    let n = x.len().min(w.len()).min(b.len());
    // 用 `sum_sq_centered_f32` 替代代数恒等式 `Σx² - n·mean²`：
    // per-element `(x-mean)²` 保留 bounded 误差，避免 `mean² >> Var` 时的
    // 灾难性 cancellation（实测 mean=1000 时代数方案会差 80 倍以上）。
    // AVX2 单 pass：broadcast mean → sub → square → 累加到 f64（bit-exact）。
    let n_f64 = n as f64;
    let mean = (sum_f32(&x[..n]) / n_f64) as f32;
    let var = (sum_sq_centered_f32(&x[..n], mean) / n_f64) as f32;
    let inv = 1.0 / (var + eps).sqrt();
    layer_norm_scale_bias(&mut x[..n], &w[..n], &b[..n], mean, inv);
}

fn layer_norm_without_bias(x: &mut [f32], w: &[f32], eps: f32) {
    let n = x.len().min(w.len());
    let n_f64 = n as f64;
    let mean = (sum_f32(&x[..n]) / n_f64) as f32;
    let var = (sum_sq_centered_f32(&x[..n], mean) / n_f64) as f32;
    let inv = 1.0 / (var + eps).sqrt();
    layer_norm_scale(&mut x[..n], &w[..n], mean, inv);
}

fn layer_norm_scale_bias(x: &mut [f32], w: &[f32], b: &[f32], mean: f32, inv: f32) {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::ops::has_avx2_fma() {
            unsafe { layer_norm_scale_bias_avx2(x, w, b, mean, inv) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if crate::ops::has_neon() {
            unsafe {
                layer_norm_scale_bias_neon(x, w, b, mean, inv);
            }
            return;
        }
    }
    for i in 0..x.len() {
        x[i] = (x[i] - mean) * inv * w[i] + b[i];
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn layer_norm_scale_bias_neon(
    x: &mut [f32],
    weight: &[f32],
    bias: &[f32],
    mean: f32,
    inv: f32,
) {
    use std::arch::aarch64::*;
    let mean_v = vdupq_n_f32(mean);
    let inv_v = vdupq_n_f32(inv);
    let mut i = 0;
    while i + 4 <= x.len() {
        let centered = vsubq_f32(vld1q_f32(x.as_ptr().add(i)), mean_v);
        let normalized = vmulq_f32(
            vmulq_f32(centered, inv_v),
            vld1q_f32(weight.as_ptr().add(i)),
        );
        let value = vaddq_f32(normalized, vld1q_f32(bias.as_ptr().add(i)));
        vst1q_f32(x.as_mut_ptr().add(i), value);
        i += 4;
    }
    while i < x.len() {
        x[i] = (x[i] - mean) * inv * weight[i] + bias[i];
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn layer_norm_scale_bias_avx2(x: &mut [f32], w: &[f32], b: &[f32], mean: f32, inv: f32) {
    use std::arch::x86_64::*;
    let n = x.len();
    let n8 = n / 8 * 8;
    let vmean = _mm256_set1_ps(mean);
    let vinv = _mm256_set1_ps(inv);
    let mut i = 0;
    while i < n8 {
        let vx = _mm256_loadu_ps(x.as_ptr().add(i));
        let vw = _mm256_loadu_ps(w.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        let d = _mm256_sub_ps(vx, vmean);
        let scaled = _mm256_mul_ps(_mm256_mul_ps(d, vinv), vw);
        _mm256_storeu_ps(x.as_mut_ptr().add(i), _mm256_add_ps(scaled, vb));
        i += 8;
    }
    while i < n {
        x[i] = (x[i] - mean) * inv * w[i] + b[i];
        i += 1;
    }
}

fn layer_norm_scale(x: &mut [f32], w: &[f32], mean: f32, inv: f32) {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::ops::has_avx2_fma() {
            unsafe { layer_norm_scale_avx2(x, w, mean, inv) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if crate::ops::has_neon() {
            unsafe {
                layer_norm_scale_neon(x, w, mean, inv);
            }
            return;
        }
    }
    for i in 0..x.len() {
        x[i] = (x[i] - mean) * inv * w[i];
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn layer_norm_scale_neon(x: &mut [f32], weight: &[f32], mean: f32, inv: f32) {
    use std::arch::aarch64::*;
    let mean_v = vdupq_n_f32(mean);
    let inv_v = vdupq_n_f32(inv);
    let mut i = 0;
    while i + 4 <= x.len() {
        let centered = vsubq_f32(vld1q_f32(x.as_ptr().add(i)), mean_v);
        let value = vmulq_f32(
            vmulq_f32(centered, inv_v),
            vld1q_f32(weight.as_ptr().add(i)),
        );
        vst1q_f32(x.as_mut_ptr().add(i), value);
        i += 4;
    }
    while i < x.len() {
        x[i] = (x[i] - mean) * inv * weight[i];
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn layer_norm_scale_avx2(x: &mut [f32], w: &[f32], mean: f32, inv: f32) {
    use std::arch::x86_64::*;
    let n = x.len();
    let n8 = n / 8 * 8;
    let vmean = _mm256_set1_ps(mean);
    let vinv = _mm256_set1_ps(inv);
    let mut i = 0;
    while i < n8 {
        let vx = _mm256_loadu_ps(x.as_ptr().add(i));
        let vw = _mm256_loadu_ps(w.as_ptr().add(i));
        let d = _mm256_sub_ps(vx, vmean);
        _mm256_storeu_ps(
            x.as_mut_ptr().add(i),
            _mm256_mul_ps(_mm256_mul_ps(d, vinv), vw),
        );
        i += 8;
    }
    while i < n {
        x[i] = (x[i] - mean) * inv * w[i];
        i += 1;
    }
}

fn matmul_f32_single(
    weight: &[f32],
    input: &[f32],
    output: &mut [f32],
    in_dim: usize,
    out_dim: usize,
) {
    if out_dim >= 512 {
        output
            .par_chunks_mut(64)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let row_start = chunk_idx * 64;
                for (local, o) in (row_start..row_start + chunk.len()).enumerate() {
                    chunk[local] =
                        dot_f32(&weight[o * in_dim..][..in_dim], &input[..in_dim], in_dim);
                }
            });
    } else {
        for o in 0..out_dim {
            output[o] = dot_f32(&weight[o * in_dim..][..in_dim], &input[..in_dim], in_dim);
        }
    }
}

fn patch_embed_scalar(
    frame_a: &[f32],
    frame_b: &[f32],
    img_w: usize,
    img_h: usize,
    w0: &[f32],
    w1: Option<&[f32]>,
    output: &mut [f32],
    ps: usize,
    n_embd: usize,
) {
    let n_patches_x = img_w / ps;
    let n_patches_y = img_h / ps;
    let patch_dim = 3 * ps * ps;
    for py in 0..n_patches_y {
        for px in 0..n_patches_x {
            let patch_idx = py * n_patches_x + px;
            let out_off = patch_idx * n_embd;
            for e in 0..n_embd {
                let mut sum0 = 0.0f32;
                let mut sum1 = 0.0f32;
                let w0_row = &w0[e * patch_dim..e * patch_dim + patch_dim];
                let w1_row = w1.map(|w| &w[e * patch_dim..e * patch_dim + patch_dim]);
                for c in 0..3usize {
                    for ky in 0..ps {
                        for kx in 0..ps {
                            let pix_x = px * ps + kx;
                            let pix_y = py * ps + ky;
                            let pixel = (pix_y * img_w + pix_x) * 3 + c;
                            let w_idx = c * ps * ps + ky * ps + kx;
                            sum0 += w0_row[w_idx] * frame_a[pixel];
                            if let Some(w1d) = w1_row {
                                sum1 += w1d[w_idx] * frame_b[pixel];
                            }
                        }
                    }
                }
                output[out_off + e] = sum0 + sum1;
            }
        }
    }
}

fn repack_patch_weights(weights: &[f32], n_embd: usize, patch_dim: usize) -> Vec<f32> {
    let mut packed = vec![0.0f32; patch_dim * n_embd];
    for e in 0..n_embd {
        for k in 0..patch_dim {
            packed[k * n_embd + e] = weights[e * patch_dim + k];
        }
    }
    packed
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn patch_embed_simd_avx2(
    frame_a: &[f32],
    frame_b: &[f32],
    img_w: usize,
    img_h: usize,
    w0_packed: &[f32],
    w1_packed: Option<&[f32]>,
    output: &mut [f32],
    ps: usize,
    n_embd: usize,
) {
    use std::arch::x86_64::*;
    let n_patches_x = img_w / ps;
    let n_patches_y = img_h / ps;
    let patch_dim = 3 * ps * ps;
    let n_embd_8 = n_embd / 8;

    for py in 0..n_patches_y {
        for px in 0..n_patches_x {
            let patch_idx = py * n_patches_x + px;
            let out_off = patch_idx * n_embd;

            let mut acc0: Vec<__m256> = vec![_mm256_setzero_ps(); n_embd_8];
            let mut acc1: Vec<__m256> = if w1_packed.is_some() {
                vec![_mm256_setzero_ps(); n_embd_8]
            } else {
                Vec::new()
            };
            let acc1_ptr = if acc1.is_empty() {
                std::ptr::null_mut()
            } else {
                acc1.as_mut_ptr()
            };

            for c in 0..3usize {
                for ky in 0..ps {
                    for kx in 0..ps {
                        let pix_x = px * ps + kx;
                        let pix_y = py * ps + ky;
                        let pixel = (pix_y * img_w + pix_x) * 3 + c;
                        let frame_a_vec = _mm256_set1_ps(frame_a[pixel]);
                        let frame_b_vec = _mm256_set1_ps(frame_b[pixel]);
                        let w_idx = c * ps * ps + ky * ps + kx;
                        let w_row_base = w_idx * n_embd;
                        let acc0_ptr_inner = acc0.as_mut_ptr();
                        let w0_ptr = w0_packed.as_ptr().add(w_row_base);
                        let w1_ptr = if w1_packed.is_some() {
                            w1_packed.unwrap().as_ptr().add(w_row_base)
                        } else {
                            std::ptr::null()
                        };
                        let mut e = 0;
                        while e < n_embd_8 {
                            let w_vec = _mm256_loadu_ps(w0_ptr.add(e * 8));
                            *acc0_ptr_inner.add(e) =
                                _mm256_fmadd_ps(w_vec, frame_a_vec, *acc0_ptr_inner.add(e));
                            if !w1_ptr.is_null() {
                                let w1_vec = _mm256_loadu_ps(w1_ptr.add(e * 8));
                                *acc1_ptr.add(e) =
                                    _mm256_fmadd_ps(w1_vec, frame_b_vec, *acc1_ptr.add(e));
                            }
                            e += 1;
                        }
                    }
                }
            }

            for e in 0..n_embd_8 {
                _mm256_storeu_ps(output.as_mut_ptr().add(out_off + e * 8), acc0[e]);
            }
            if !acc1_ptr.is_null() {
                for e in 0..n_embd_8 {
                    let existing = _mm256_loadu_ps(output.as_ptr().add(out_off + e * 8));
                    let sum = _mm256_add_ps(existing, acc1[e]);
                    _mm256_storeu_ps(output.as_mut_ptr().add(out_off + e * 8), sum);
                }
            }
        }
    }
}

fn matmul_f32_batch(
    weight: &[f32],
    input: &[f32],
    output: &mut [f32],
    in_dim: usize,
    out_dim: usize,
    n_tokens: usize,
) {
    let total_rows = n_tokens * out_dim;
    if total_rows >= 512 {
        output
            .par_chunks_mut(64)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let global_row = chunk_idx * 64;
                for (local, row) in (global_row..global_row + chunk.len()).enumerate() {
                    let t = row / out_dim;
                    let o = row % out_dim;
                    let inp_off = t * in_dim;
                    chunk[local] = dot_f32(
                        &weight[o * in_dim..][..in_dim],
                        &input[inp_off..inp_off + in_dim],
                        in_dim,
                    );
                }
            });
    } else {
        for t in 0..n_tokens {
            let inp_off = t * in_dim;
            let out_off = t * out_dim;
            for o in 0..out_dim {
                output[out_off + o] = dot_f32(
                    &weight[o * in_dim..][..in_dim],
                    &input[inp_off..inp_off + in_dim],
                    in_dim,
                );
            }
        }
    }
}

fn matmul_f16_f32_single(
    weight_f16: &[u8],
    input: &[f32],
    output: &mut [f32],
    in_dim: usize,
    out_dim: usize,
) {
    let n_half = in_dim / 2;
    let u16_ptr = weight_f16.as_ptr() as *const u16;
    let w_u16: &[u16] = unsafe { std::slice::from_raw_parts(u16_ptr, weight_f16.len() / 2) };
    for o in 0..out_dim {
        let row_off = o * in_dim;
        let row_u16 = &w_u16[row_off..row_off + n_half];
        output[o] = dot_f16_f32(&input[..in_dim], row_u16, in_dim);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn attention_dot_avx2(q: *const f32, k: *const f32, d: usize) -> f32 {
    use std::arch::x86_64::*;
    let n8 = d / 8 * 8;
    let mut acc = _mm256_setzero_ps();
    let mut i = 0;
    while i < n8 {
        let vq = _mm256_loadu_ps(q.add(i));
        let vk = _mm256_loadu_ps(k.add(i));
        acc = _mm256_fmadd_ps(vq, vk, acc);
        i += 8;
    }
    let mut sum = crate::ops::hsum_ps(acc);
    while i < d {
        sum += *q.add(i) * *k.add(i);
        i += 1;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn attention_qk_avx2(
    q: *const f32,
    k_base: *const f32,
    scores: &mut [f32],
    n_tokens: usize,
    d_head: usize,
    scale: f32,
) {
    use std::arch::x86_64::*;
    let n8 = d_head / 8 * 8;
    let d_stride = d_head as isize;
    let tile4 = n_tokens / 4 * 4;
    let mut s = 0;
    while s < tile4 {
        let k0 = k_base.offset((s as isize) * d_stride);
        let k1 = k_base.offset(((s + 1) as isize) * d_stride);
        let k2 = k_base.offset(((s + 2) as isize) * d_stride);
        let k3 = k_base.offset(((s + 3) as isize) * d_stride);
        let mut cv0 = _mm256_setzero_ps();
        let mut cv1 = _mm256_setzero_ps();
        let mut cv2 = _mm256_setzero_ps();
        let mut cv3 = _mm256_setzero_ps();
        let mut i = 0;
        while i < n8 {
            let vq = _mm256_loadu_ps(q.add(i));
            cv0 = _mm256_fmadd_ps(vq, _mm256_loadu_ps(k0.add(i)), cv0);
            cv1 = _mm256_fmadd_ps(vq, _mm256_loadu_ps(k1.add(i)), cv1);
            cv2 = _mm256_fmadd_ps(vq, _mm256_loadu_ps(k2.add(i)), cv2);
            cv3 = _mm256_fmadd_ps(vq, _mm256_loadu_ps(k3.add(i)), cv3);
            i += 8;
        }
        let mut r0 = crate::ops::hsum_ps(cv0);
        let mut r1 = crate::ops::hsum_ps(cv1);
        let mut r2 = crate::ops::hsum_ps(cv2);
        let mut r3 = crate::ops::hsum_ps(cv3);
        i = n8;
        while i < d_head {
            let qv = *q.add(i);
            r0 += qv * *k0.add(i);
            r1 += qv * *k1.add(i);
            r2 += qv * *k2.add(i);
            r3 += qv * *k3.add(i);
            i += 1;
        }
        scores[s] = r0 * scale;
        scores[s + 1] = r1 * scale;
        scores[s + 2] = r2 * scale;
        scores[s + 3] = r3 * scale;
        s += 4;
    }
    while s < n_tokens {
        let k_ptr = k_base.offset((s as isize) * d_stride);
        let dot = attention_dot_avx2(q, k_ptr, d_head);
        scores[s] = dot * scale;
        s += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn attn_scaled_add_avx2(out: &mut [f32], v: *const f32, scale: f32, d: usize) {
    use std::arch::x86_64::*;
    let vs = _mm256_set1_ps(scale);
    let n8 = d / 8 * 8;
    let out_ptr = out.as_mut_ptr();
    let mut i = 0;
    while i < n8 {
        let vv = _mm256_loadu_ps(v.add(i));
        let ov = _mm256_loadu_ps(out_ptr.add(i));
        _mm256_storeu_ps(out_ptr.add(i), _mm256_fmadd_ps(vs, vv, ov));
        i += 8;
    }
    while i < d {
        *out_ptr.add(i) += scale * *v.add(i);
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_weight_decoder_accepts_f32_patch_weights() {
        let bytes = [1.0f32, -2.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();

        assert_eq!(decode_linear_weight(&bytes, 2, 1), [1.0, -2.0]);
    }
    use crate::core::tensor::{GGMLType, MetaValue, TensorInfo};
    use std::collections::HashMap;

    #[derive(Default)]
    struct MapTensorSource {
        metadata: HashMap<String, MetaValue>,
        tensors: HashMap<String, TensorInfo>,
        data: HashMap<String, Vec<u8>>,
    }

    impl TensorSource for MapTensorSource {
        fn metadata(&self, key: &str) -> Option<&MetaValue> {
            self.metadata.get(key)
        }

        fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
            self.tensors.get(name)
        }

        fn tensor_slice(&self, name: &str) -> Option<&[u8]> {
            self.data.get(name).map(Vec::as_slice)
        }
    }

    #[test]
    fn projection_width_comes_from_mm2_tensor() {
        let mut source = MapTensorSource {
            metadata: HashMap::from([
                ("clip.vision.projection_dim".into(), MetaValue::Uint32(4096)),
                ("clip.vision.image_size".into(), MetaValue::Uint32(224)),
                ("clip.vision.patch_size".into(), MetaValue::Uint32(16)),
                ("clip.vision.embedding_length".into(), MetaValue::Uint32(8)),
                (
                    "clip.vision.feed_forward_length".into(),
                    MetaValue::Uint32(32),
                ),
                ("clip.vision.block_count".into(), MetaValue::Uint32(0)),
                (
                    "clip.vision.attention.head_count".into(),
                    MetaValue::Uint32(1),
                ),
                (
                    "clip.vision.attention.layer_norm_epsilon".into(),
                    MetaValue::Float32(1e-6),
                ),
            ]),
            ..MapTensorSource::default()
        };
        for name in ["v.patch_embd.weight", "mm.0.weight", "mm.2.weight"] {
            source.data.insert(name.into(), Vec::new());
        }
        source.tensors.insert(
            "mm.2.weight".into(),
            TensorInfo {
                name: "mm.2.weight".into(),
                dims: vec![32, 1024],
                ggml_type: GGMLType::F16,
                offset: 0,
            },
        );

        let encoder = VisionEncoder::from_source(&source).unwrap();

        assert_eq!(encoder.config.projection_dim, 1024);
    }

    #[test]
    fn temporal_patch_uses_the_second_frame_for_the_second_kernel() {
        let frame_a = [1.0, 2.0, 3.0];
        let frame_b = [10.0, 20.0, 30.0];
        let mut output = [0.0];

        patch_embed_scalar(
            &frame_a,
            &frame_b,
            1,
            1,
            &[1.0, 0.0, 0.0],
            Some(&[0.0, 1.0, 0.0]),
            &mut output,
            1,
            1,
        );

        assert_eq!(output, [21.0]);
    }

    #[test]
    fn projector_keeps_the_existing_spatial_block_order() {
        let mut config = test_clip_config(1, 2, 1, 64);
        config.n_embd = 1;
        config.n_layer = 0;
        config.projection_dim = 1;
        let mm0 = vec![0u8; 4 * 4 * 2];
        let mm2 = vec![0u8; 4 * 2];
        let encoder = VisionEncoder {
            config: config.clone(),
            patch_embd_weight: &[],
            patch_embd_weight_1: None,
            position_embd: None,
            post_ln_weight: None,
            post_ln_bias: None,
            patch_bias: None,
            layers: Vec::new(),
            mm_0_weight: &mm0,
            mm_0_bias: None,
            mm_2_weight: &mm2,
            mm_2_bias: None,
            precomputed: None,
        };
        let block_order = [0.0, 1.0, 4.0, 5.0, 2.0, 3.0, 6.0, 7.0];
        let mut scratch = VisionScratchpad::new(&config);
        scratch.merged.resize(block_order.len(), 0.0);
        scratch.projected.resize(2, 0.0);
        scratch.merged[..block_order.len()].copy_from_slice(&block_order);

        encoder.project(4, 2, 1, 2, &mut scratch);

        assert_eq!(&scratch.project_concat_buf[..8], &block_order);
    }

    fn test_clip_config(
        patch_size: usize,
        spatial_merge_size: usize,
        min_grid_tokens: usize,
        max_grid_tokens: usize,
    ) -> ClipVisionConfig {
        let factor = patch_size * spatial_merge_size;
        let factor_pixels = factor * factor;
        ClipVisionConfig {
            projection_dim: 8,
            image_size: 224,
            patch_size,
            n_embd: 8,
            n_ff: 32,
            n_layer: 1,
            n_head: 1,
            spatial_merge_size,
            image_min_pixels: factor_pixels * min_grid_tokens,
            image_max_pixels: factor_pixels * max_grid_tokens,
            video_min_pixels: factor_pixels * min_grid_tokens,
            video_max_pixels: factor_pixels * max_grid_tokens,
            eps: 1e-6,
            use_gelu: true,
            use_silu: false,
            n_wa_pattern: 0,
            image_mean: [0.0; 3],
            image_std: [1.0; 3],
            has_deepstack_layers: vec![false],
        }
    }

    #[test]
    fn smart_resize_preserves_ratio_and_alignment() {
        let config = test_clip_config(16, 2, 8, 4096);
        let grid = qwen_smart_resize(100, 50, &config).unwrap();
        assert_eq!(grid.image_width(), 128);
        assert_eq!(grid.image_height(), 64);
        assert_eq!(grid.grid_w, 4);
        assert_eq!(grid.grid_h, 2);
        assert_eq!(grid.token_count(), 8);
    }

    #[test]
    fn smart_resize_uses_original_pixels_for_the_scale() {
        let config = test_clip_config(16, 2, 8, 4096);
        let grid = qwen_smart_resize(30, 46, &config).unwrap();
        assert_eq!((grid.image_width(), grid.image_height()), (96, 128));
    }

    #[test]
    fn vision_grid_rejects_unaligned_dimensions() {
        assert!(VisionGrid::from_image_size(1, 100, 64, 16, 2).is_err());
    }

    #[test]
    fn vision_grid_rejects_token_count_overflow() {
        assert!(VisionGrid::from_image_size(1, usize::MAX, 2, 1, 1).is_err());
    }

    #[test]
    fn rectangular_position_embedding_keeps_width_and_height_orientation() {
        let mut config = test_clip_config(1, 1, 1, 64);
        config.n_embd = 1;
        let encoder = VisionEncoder {
            config,
            patch_embd_weight: &[],
            patch_embd_weight_1: None,
            position_embd: None,
            post_ln_weight: None,
            post_ln_bias: None,
            patch_bias: None,
            layers: Vec::new(),
            mm_0_weight: &[],
            mm_0_bias: None,
            mm_2_weight: &[],
            mm_2_bias: None,
            precomputed: None,
        };
        let position_data: Vec<u8> = [0.0f32, 10.0, 20.0, 30.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        let mut merged = vec![0.0; 6];
        let mut position_buffer = vec![0.0; 6];

        encoder.apply_position_embedding_merged(
            &mut merged,
            3,
            2,
            1,
            1,
            &position_data,
            &mut position_buffer,
        );

        assert_eq!(merged, [0.0, 5.0, 10.0, 20.0, 25.0, 30.0]);
    }

    fn close(a: f32, b: f32) {
        assert!((a - b).abs() <= 1e-4 + 1e-4 * b.abs(), "a={a} b={b}");
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_layer_norm_helpers_match_scalar() {
        let input: Vec<f32> = (0..13).map(|i| i as f32 * 0.1 - 0.7).collect();
        let weight: Vec<f32> = (0..13).map(|i| 0.9 + i as f32 * 0.01).collect();
        let bias: Vec<f32> = (0..13).map(|i| i as f32 * -0.005).collect();
        let mean = input.iter().sum::<f32>() / input.len() as f32;
        let inv = 1.0
            / (input.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / input.len() as f32
                + 1e-6)
                .sqrt();
        let mut expected = input.clone();
        for i in 0..expected.len() {
            expected[i] = (expected[i] - mean) * inv * weight[i] + bias[i];
        }
        let mut actual = input.clone();
        unsafe { layer_norm_scale_bias_neon(&mut actual, &weight, &bias, mean, inv) };
        for i in 0..actual.len() {
            close(actual[i], expected[i]);
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn shared_neon_attention_ops_match_scalar() {
        let q: Vec<f32> = (0..13).map(|i| i as f32 * 0.07 - 0.3).collect();
        let k: Vec<f32> = (0..13).map(|i| 0.4 - i as f32 * 0.02).collect();
        let expected: f32 = q.iter().zip(&k).map(|(x, y)| x * y).sum();
        close(dot_f32(&q, &k, q.len()), expected);
        let mut out = vec![0.25f32; 13];
        vec_mad_f32(&mut out, &k, 0.5);
        for i in 0..13 {
            close(out[i], 0.25 + 0.5 * k[i]);
        }
    }

    fn make_patch_embed_inputs(
        img_w: usize,
        img_h: usize,
        ps: usize,
        n_embd: usize,
        seed: u64,
    ) -> (Vec<f32>, Vec<f32>) {
        let patch_dim = 3 * ps * ps;
        let mut pixels: Vec<f32> = vec![0.0; img_w * img_h * 3];
        let mut weights: Vec<f32> = vec![0.0; n_embd * patch_dim];
        let mut state = seed.wrapping_add(1);
        for v in pixels.iter_mut() {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *v = ((state >> 33) as i32 as f32) / (1u32 << 31) as f32 * 0.5;
        }
        for v in weights.iter_mut() {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *v = ((state >> 33) as i32 as f32) / (1u32 << 31) as f32 * 0.1;
        }
        (pixels, weights)
    }

    #[test]
    fn patch_embed_scalar_matches_naive_loop() {
        let (img_w, img_h) = (32, 32);
        let ps = 16;
        let n_embd = 64;
        let n_patches = (img_w / ps) * (img_h / ps);
        let (pixels, weights) = make_patch_embed_inputs(img_w, img_h, ps, n_embd, 7);
        let mut naive = vec![0.0f32; n_patches * n_embd];
        let patch_dim = 3 * ps * ps;
        for py in 0..img_h / ps {
            for px in 0..img_w / ps {
                let out_off = (py * img_w / ps + px) * n_embd;
                for e in 0..n_embd {
                    let mut sum = 0.0f32;
                    for c in 0..3usize {
                        for ky in 0..ps {
                            for kx in 0..ps {
                                let pix_val =
                                    pixels[((py * ps + ky) * img_w + (px * ps + kx)) * 3 + c];
                                let w_idx = e * patch_dim + c * ps * ps + ky * ps + kx;
                                sum += weights[w_idx] * pix_val;
                            }
                        }
                    }
                    naive[out_off + e] = sum;
                }
            }
        }
        let mut out = vec![0.0f32; n_patches * n_embd];
        patch_embed_scalar(
            &pixels, &pixels, img_w, img_h, &weights, None, &mut out, ps, n_embd,
        );
        for (a, b) in naive.iter().zip(out.iter()) {
            assert_eq!(*a, *b, "scalar mismatch: naive={a} out={b}");
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn patch_embed_simd_matches_scalar() {
        for &(img_dim, ps, n_embd) in &[(32usize, 16usize, 64usize), (64, 16, 128), (128, 16, 256)]
        {
            let (img_w, img_h) = (img_dim, img_dim);
            let n_patches = (img_w / ps) * (img_h / ps);
            let (pixels, weights) = make_patch_embed_inputs(img_w, img_h, ps, n_embd, 11);
            let mut scalar_out = vec![0.0f32; n_patches * n_embd];
            patch_embed_scalar(
                &pixels,
                &pixels,
                img_w,
                img_h,
                &weights,
                None,
                &mut scalar_out,
                ps,
                n_embd,
            );
            let patch_dim = 3 * ps * ps;
            let packed = repack_patch_weights(&weights, n_embd, patch_dim);
            let mut simd_out = vec![0.0f32; n_patches * n_embd];
            unsafe {
                patch_embed_simd_avx2(
                    &pixels,
                    &pixels,
                    img_w,
                    img_h,
                    &packed,
                    None,
                    &mut simd_out,
                    ps,
                    n_embd,
                );
            }
            let mut max_abs = 0.0f32;
            let mut max_rel = 0.0f32;
            for (a, b) in scalar_out.iter().zip(simd_out.iter()) {
                max_abs = max_abs.max((*a - *b).abs());
                let denom = a.abs().max(*b).max(1e-6);
                max_rel = max_rel.max(((*a - *b).abs()) / denom);
            }
            println!("img={img_dim} ps={ps} n_embd={n_embd}: max_abs_err={max_abs:.3e} max_rel_err={max_rel:.3e}");
            assert!(
                max_abs < 1e-2,
                "img={img_dim} ps={ps} n_embd={n_embd}: max_abs_err={max_abs}"
            );
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn patch_embed_simd_with_w1_matches_scalar() {
        let (img_w, img_h) = (32, 32);
        let ps = 16;
        let n_embd = 64;
        let patch_dim = 3 * ps * ps;
        let n_patches = (img_w / ps) * (img_h / ps);
        let (pixels, w0) = make_patch_embed_inputs(img_w, img_h, ps, n_embd, 23);
        let w1 = make_patch_embed_inputs(img_w, img_h, ps, n_embd, 29).1;
        let mut scalar_out = vec![0.0f32; n_patches * n_embd];
        patch_embed_scalar(
            &pixels,
            &pixels,
            img_w,
            img_h,
            &w0,
            Some(&w1),
            &mut scalar_out,
            ps,
            n_embd,
        );
        let p0 = repack_patch_weights(&w0, n_embd, patch_dim);
        let p1 = repack_patch_weights(&w1, n_embd, patch_dim);
        let mut simd_out = vec![0.0f32; n_patches * n_embd];
        unsafe {
            patch_embed_simd_avx2(
                &pixels,
                &pixels,
                img_w,
                img_h,
                &p0,
                Some(&p1),
                &mut simd_out,
                ps,
                n_embd,
            );
        }
        let mut max_abs = 0.0f32;
        for (a, b) in scalar_out.iter().zip(simd_out.iter()) {
            max_abs = max_abs.max((*a - *b).abs());
        }
        assert!(max_abs < 1e-2, "max_abs_err with w1={max_abs}");
    }

    #[test]
    fn repack_patch_weights_round_trip() {
        let patch_dim = 3 * 16 * 16;
        let n_embd = 64;
        let mut w = vec![0.0f32; n_embd * patch_dim];
        for (i, v) in w.iter_mut().enumerate() {
            *v = i as f32 * 0.01;
        }
        let packed = repack_patch_weights(&w, n_embd, patch_dim);
        for e in 0..n_embd {
            for k in 0..patch_dim {
                assert_eq!(packed[k * n_embd + e], w[e * patch_dim + k]);
            }
        }
    }
}
