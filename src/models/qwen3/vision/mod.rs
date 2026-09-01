pub mod clip_config;

use crate::core::tensor::TensorSource;
use crate::ops::{
    gelu_inplace, rope_mrope_interleaved, softmax_inplace, sum_f32, sum_sq_centered_f32, vec_add,
    vec_add_into,
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
        let n_in = self.n_in;
        let n_out = self.n_out;
        if let Some(weight) = &self.f32_data {
            matmul_f32_weight(weight, input, output, n_tokens, n_in, n_out);
            return;
        }
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
        let n_in = self.n_in;
        let n_out = self.n_out;
        if let Some(weight) = &self.f32_data {
            matmul_f32_weight(weight, input, output, 1, n_in, n_out);
            return;
        }
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

fn matmul_f32_weight(
    weight: &[f32],
    input: &[f32],
    output: &mut [f32],
    n_tokens: usize,
    n_in: usize,
    n_out: usize,
) {
    output
        .par_chunks_mut(n_out)
        .take(n_tokens)
        .enumerate()
        .for_each(|(token, output_row)| {
            let input_row = &input[token * n_in..(token + 1) * n_in];
            for (out, value) in output_row.iter_mut().enumerate() {
                *value =
                    crate::ops::dot_f32(&weight[out * n_in..(out + 1) * n_in], input_row, n_in);
            }
        });
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
    qkv_weights: Vec<Q8Weight>,
    qkv_biases: Vec<Option<Vec<f32>>>,
    out_weights: Vec<Q8Weight>,
    out_biases: Vec<Option<Vec<f32>>>,
    ffn_up_weights: Vec<Q8Weight>,
    ffn_up_biases: Vec<Option<Vec<f32>>>,
    ffn_down_weights: Vec<Q8Weight>,
    ffn_down_biases: Vec<Option<Vec<f32>>>,
    ln1_weights: Vec<Vec<f32>>,
    ln1_biases: Vec<Option<Vec<f32>>>,
    ln2_weights: Vec<Vec<f32>>,
    ln2_biases: Vec<Option<Vec<f32>>>,
    post_ln_weight: Vec<f32>,
    post_ln_bias: Option<Vec<f32>>,
    patch_bias: Option<Vec<f32>>,
    mm_0_weight: Q8Weight,
    mm_0_bias: Option<Vec<f32>>,
    mm_2_weight: Q8Weight,
    mm_2_bias: Option<Vec<f32>>,
    deepstack: Vec<Option<DeepstackPrecomputed>>,
}

struct DeepstackPrecomputed {
    norm_weight: Vec<f32>,
    norm_bias: Vec<f32>,
    fc1_weight: Q8Weight,
    fc1_bias: Vec<f32>,
    fc2_weight: Q8Weight,
    fc2_bias: Vec<f32>,
}

pub struct VisionLayer<'a> {
    pub ln1_weight: &'a [u8],
    pub ln1_bias: Option<&'a [u8]>,
    pub ln2_weight: &'a [u8],
    pub ln2_bias: Option<&'a [u8]>,
    pub qkv_weight: &'a [u8],
    pub qkv_bias: Option<&'a [u8]>,
    pub out_weight: &'a [u8],
    pub out_bias: Option<&'a [u8]>,
    pub ffn_up_weight: &'a [u8],
    pub ffn_up_bias: Option<&'a [u8]>,
    pub ffn_down_weight: &'a [u8],
    pub ffn_down_bias: Option<&'a [u8]>,
    pub deepstack: Option<DeepstackWeights<'a>>,
}

pub struct DeepstackWeights<'a> {
    pub norm_weight: &'a [u8],
    pub norm_bias: &'a [u8],
    pub fc1_weight: &'a [u8],
    pub fc1_bias: &'a [u8],
    pub fc2_weight: &'a [u8],
    pub fc2_bias: &'a [u8],
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

fn decode_f32_slice(data: &[u8]) -> Vec<f32> {
    assert_eq!(data.len() % 4, 0);
    let n = data.len() / 4;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let bits = u32::from_le_bytes([
            data[i * 4],
            data[i * 4 + 1],
            data[i * 4 + 2],
            data[i * 4 + 3],
        ]);
        out.push(f32::from_bits(bits));
    }
    out
}

fn dequant_q8_0_tensor(data: &[u8], n_cols: usize, n_rows: usize) -> Vec<f32> {
    let blocks_per_row = n_cols / 32;
    let mut out = vec![0.0f32; n_rows * n_cols];
    for row in 0..n_rows {
        for bi in 0..blocks_per_row {
            let boff = row * blocks_per_row * 34 + bi * 34;
            if boff + 34 > data.len() {
                continue;
            }
            let d = crate::ops::f16_to_f32(u16::from_le_bytes([data[boff], data[boff + 1]]));
            let out_base = row * n_cols + bi * 32;
            for j in 0..32 {
                let q = data[boff + 2 + j] as i8 as i32;
                out[out_base + j] = d * q as f32;
            }
        }
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
        dequant_q8_0_tensor(data, n_in, n_out)
    } else {
        panic!(
            "Unsupported vision linear weight byte length {} for [{n_in}, {n_out}]",
            data.len()
        );
    }
}

fn disable_missing_deepstack_layers<S: TensorSource + ?Sized>(source: &S, layers: &mut [bool]) {
    for (index, enabled) in layers.iter_mut().enumerate() {
        if *enabled
            && source
                .tensor_info(&format!("v.deepstack.{index}.norm.weight"))
                .is_none()
        {
            *enabled = false;
        }
    }
}

impl<'a> VisionEncoder<'a> {
    pub fn from_source<S: TensorSource + ?Sized>(source: &'a S) -> Result<Self, String> {
        let mut config = ClipVisionConfig::from_source(source)?;
        disable_missing_deepstack_layers(source, &mut config.has_deepstack_layers);

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
            let qkv_weight = source
                .tensor_slice(&format!("v.blk.{}.attn_qkv.weight", i))
                .ok_or_else(|| format!("Missing v.blk.{}.attn_qkv.weight", i))?;
            let qkv_bias = source.tensor_slice(&format!("v.blk.{}.attn_qkv.bias", i));
            let out_weight = source
                .tensor_slice(&format!("v.blk.{}.attn_out.weight", i))
                .ok_or_else(|| format!("Missing v.blk.{}.attn_out.weight", i))?;
            let out_bias = source.tensor_slice(&format!("v.blk.{}.attn_out.bias", i));
            let ffn_up_weight = source
                .tensor_slice(&format!("v.blk.{}.ffn_up.weight", i))
                .ok_or_else(|| format!("Missing v.blk.{}.ffn_up.weight", i))?;
            let ffn_up_bias = source.tensor_slice(&format!("v.blk.{}.ffn_up.bias", i));
            let ffn_down_weight = source
                .tensor_slice(&format!("v.blk.{}.ffn_down.weight", i))
                .ok_or_else(|| format!("Missing v.blk.{}.ffn_down.weight", i))?;
            let ffn_down_bias = source.tensor_slice(&format!("v.blk.{}.ffn_down.bias", i));
            let deepstack = if config.has_deepstack_layers.get(i).copied().unwrap_or(false) {
                let tensor = |suffix: &str| {
                    let name = format!("v.deepstack.{i}.{suffix}");
                    source
                        .tensor_slice(&name)
                        .ok_or_else(|| format!("Missing {name}"))
                };
                Some(DeepstackWeights {
                    norm_weight: tensor("norm.weight")?,
                    norm_bias: tensor("norm.bias")?,
                    fc1_weight: tensor("fc1.weight")?,
                    fc1_bias: tensor("fc1.bias")?,
                    fc2_weight: tensor("fc2.weight")?,
                    fc2_bias: tensor("fc2.bias")?,
                })
            } else {
                None
            };

            layers.push(VisionLayer {
                ln1_weight,
                ln1_bias,
                ln2_weight,
                ln2_bias,
                qkv_weight,
                qkv_bias,
                out_weight,
                out_bias,
                ffn_up_weight,
                ffn_up_bias,
                ffn_down_weight,
                ffn_down_bias,
                deepstack,
            });
        }

        let mm_0_weight = source
            .tensor_slice("mm.0.weight")
            .ok_or("Missing mm.0.weight")?;
        let mm_0_bias = source.tensor_slice("mm.0.bias");
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
        let mut out_weights = Vec::with_capacity(n_layer);
        let mut out_biases = Vec::with_capacity(n_layer);
        let mut ffn_up_weights = Vec::with_capacity(n_layer);
        let mut ffn_up_biases = Vec::with_capacity(n_layer);
        let mut ffn_down_weights = Vec::with_capacity(n_layer);
        let mut ffn_down_biases = Vec::with_capacity(n_layer);
        let mut ln1_weights = Vec::with_capacity(n_layer);
        let mut ln1_biases = Vec::with_capacity(n_layer);
        let mut ln2_weights = Vec::with_capacity(n_layer);
        let mut ln2_biases = Vec::with_capacity(n_layer);
        let mut deepstack = Vec::with_capacity(n_layer);

        let qkv_n_in = n_embd;
        let qkv_n_out = n_embd * 3;
        let out_n_in = n_embd;
        let out_n_out = n_embd;
        let ffn_up_n_in = n_embd;
        let ffn_up_n_out = n_ff;
        let ffn_down_n_in = n_ff;
        let ffn_down_n_out = n_embd;
        let mm_merged_embd =
            n_embd * self.config.spatial_merge_size * self.config.spatial_merge_size;

        for layer in &self.layers {
            let qkv_f32 = decode_linear_weight(layer.qkv_weight, qkv_n_in, qkv_n_out);
            qkv_weights.push(Q8Weight::from_f32(&qkv_f32, qkv_n_in, qkv_n_out));
            qkv_biases.push(layer.qkv_bias.map(decode_f32_slice));
            let out_f32 = decode_linear_weight(layer.out_weight, out_n_in, out_n_out);
            out_weights.push(Q8Weight::from_f32(&out_f32, out_n_in, out_n_out));
            out_biases.push(layer.out_bias.map(decode_f32_slice));
            let ffn_up_f32 = decode_linear_weight(layer.ffn_up_weight, ffn_up_n_in, ffn_up_n_out);
            ffn_up_weights.push(Q8Weight::from_f32(&ffn_up_f32, ffn_up_n_in, ffn_up_n_out));
            ffn_up_biases.push(layer.ffn_up_bias.map(decode_f32_slice));
            let ffn_down_f32 =
                decode_linear_weight(layer.ffn_down_weight, ffn_down_n_in, ffn_down_n_out);
            ffn_down_weights.push(Q8Weight::from_f32(
                &ffn_down_f32,
                ffn_down_n_in,
                ffn_down_n_out,
            ));
            ffn_down_biases.push(layer.ffn_down_bias.map(decode_f32_slice));
            ln1_weights.push(decode_f32_slice(layer.ln1_weight));
            ln1_biases.push(layer.ln1_bias.map(decode_f32_slice));
            ln2_weights.push(decode_f32_slice(layer.ln2_weight));
            ln2_biases.push(layer.ln2_bias.map(decode_f32_slice));
            deepstack.push(layer.deepstack.as_ref().map(|weights| {
                let fc1 = decode_linear_weight(weights.fc1_weight, mm_merged_embd, mm_merged_embd);
                let fc2 = decode_linear_weight(
                    weights.fc2_weight,
                    mm_merged_embd,
                    self.config.projection_dim,
                );
                DeepstackPrecomputed {
                    norm_weight: decode_f32_slice(weights.norm_weight),
                    norm_bias: decode_f32_slice(weights.norm_bias),
                    fc1_weight: Q8Weight::from_f32(&fc1, mm_merged_embd, mm_merged_embd),
                    fc1_bias: decode_f32_slice(weights.fc1_bias),
                    fc2_weight: Q8Weight::from_f32(
                        &fc2,
                        mm_merged_embd,
                        self.config.projection_dim,
                    ),
                    fc2_bias: decode_f32_slice(weights.fc2_bias),
                }
            }));
        }

        let mm0_f32 = decode_linear_weight(self.mm_0_weight, mm_merged_embd, mm_merged_embd);
        let mm2_f32 =
            decode_linear_weight(self.mm_2_weight, mm_merged_embd, self.config.projection_dim);

        self.precomputed = Some(VisionPrecomputed {
            qkv_weights,
            qkv_biases,
            out_weights,
            out_biases,
            ffn_up_weights,
            ffn_up_biases,
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
            deepstack,
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
        frame_a_pixels: &[f32],
        frame_b_pixels: &[f32],
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
        if frame_a_pixels.len() != expected_pixels || frame_b_pixels.len() != expected_pixels {
            return Err(format!(
                "Normalized image length mismatch: expected {expected_pixels}, got {} and {}",
                frame_a_pixels.len(),
                frame_b_pixels.len()
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
            frame_a_pixels,
            frame_b_pixels,
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

        let t_layers = std::time::Instant::now();
        let mrope_positions = build_vit_mrope_positions(n_patches_x, n_patches_y, merge);
        let mut deepstack_index = 0;
        for layer in 0..cfg.n_layer {
            self.forward_vit_layer(layer, scratch, n_tokens, &mrope_positions);
            if self.layers[layer].deepstack.is_some() {
                self.project_deepstack(
                    layer,
                    deepstack_index,
                    n_patches_x,
                    n_patches_y,
                    n_embd,
                    merge,
                    scratch,
                )?;
                deepstack_index += 1;
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

            pc.qkv_weights[il].matmul_batch(
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

            for t in 0..n_tokens {
                let inp_off = t * n_embd;
                let out_off = t * n_embd * 3;
                matmul_f16_f32_single(
                    layer.qkv_weight,
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
                rope_mrope_interleaved(
                    &mut scratch.attn_buf[q_base + t * d_head..q_base + t * d_head + d_head],
                    mrope_positions[t],
                    mrope_sections,
                    d_head,
                    freq_base,
                    d_head / 2,
                );
                rope_mrope_interleaved(
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
                    for s in 0..n_tokens {
                        let k_ptr = attn_buf.as_ptr().add(k_base + s * d_head);
                        let mut sum = 0.0f32;
                        for i in 0..d_head {
                            sum += *q_ptr.add(i) * *k_ptr.add(i);
                        }
                        score_slice[t * n_tokens + s] = sum * scale;
                    }
                    softmax_inplace(&mut score_slice[t * n_tokens..t * n_tokens + n_tokens]);

                    for d in 0..d_head {
                        out_slice[t * d_head + d] = 0.0;
                    }
                    for s in 0..n_tokens {
                        let sc = score_slice[t * n_tokens + s];
                        let v_ptr = attn_buf.as_ptr().add(v_base + s * d_head);
                        for d in 0..d_head {
                            out_slice[t * d_head + d] += sc * *v_ptr.add(d);
                        }
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

            pc.ffn_up_weights[il].matmul_batch(
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
            }

            if let Some(bias_data) = layer.ffn_up_bias {
                let bias = decode_f32_slice(bias_data);
                for t in 0..n_tokens {
                    let off = t * cfg.n_ff;
                    vec_add_into(&bias, &mut scratch.ffn_buf[off..off + cfg.n_ff]);
                }
            }
        }
        t_ln2_ffn = t_ln2_start.elapsed().as_secs_f64();

        let t_act_start = std::time::Instant::now();
        gelu_inplace(&mut scratch.ffn_buf[..n_tokens * cfg.n_ff]);
        t_act = t_act_start.elapsed().as_secs_f64();

        let t_ffn_down_start = std::time::Instant::now();
        if let Some(ref pc) = self.precomputed {
            pc.ffn_down_weights[il].matmul_batch(
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

        spatial_blocks(hidden, n_patches_x, n_patches_y, n_embd, merge, concat_buf);

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

    #[allow(clippy::too_many_arguments)]
    fn project_deepstack(
        &self,
        vision_layer: usize,
        deepstack_index: usize,
        n_patches_x: usize,
        n_patches_y: usize,
        n_embd: usize,
        merge: usize,
        scratch: &mut VisionScratchpad,
    ) -> Result<(), String> {
        let precomputed = self
            .precomputed
            .as_ref()
            .ok_or("Vision deepstack requires precompute()")?;
        let weights = precomputed.deepstack[vision_layer]
            .as_ref()
            .ok_or("Missing precomputed vision deepstack weights")?;
        let n_projected = (n_patches_x / merge) * (n_patches_y / merge);
        let merged_embd = n_embd * merge * merge;
        let projection_dim = self.config.projection_dim;
        let concat_len = n_projected * merged_embd;
        spatial_blocks(
            &scratch.merged,
            n_patches_x,
            n_patches_y,
            n_embd,
            merge,
            &mut scratch.project_concat_buf[..concat_len],
        );
        for row in scratch.project_concat_buf[..concat_len].chunks_exact_mut(merged_embd) {
            layer_norm_with_bias(
                row,
                &weights.norm_weight,
                &weights.norm_bias,
                self.config.eps,
            );
        }
        weights.fc1_weight.matmul_batch(
            &scratch.project_concat_buf[..concat_len],
            &mut scratch.project_mm0_out[..concat_len],
            n_projected,
            &mut scratch.q8_buf,
            &mut scratch.q8_scale_buf,
        );
        for row in scratch.project_mm0_out[..concat_len].chunks_exact_mut(merged_embd) {
            vec_add_into(&weights.fc1_bias, row);
        }
        gelu_inplace(&mut scratch.project_mm0_out[..concat_len]);

        let output_start = deepstack_index * n_projected * projection_dim;
        let output =
            &mut scratch.deepstack[output_start..output_start + n_projected * projection_dim];
        weights.fc2_weight.matmul_batch(
            &scratch.project_mm0_out[..concat_len],
            output,
            n_projected,
            &mut scratch.q8_buf,
            &mut scratch.q8_scale_buf,
        );
        for row in output.chunks_exact_mut(projection_dim) {
            vec_add_into(&weights.fc2_bias, row);
        }
        Ok(())
    }
}

fn spatial_blocks(
    input: &[f32],
    n_patches_x: usize,
    n_patches_y: usize,
    n_embd: usize,
    merge: usize,
    output: &mut [f32],
) {
    let len = (n_patches_x / merge) * (n_patches_y / merge) * n_embd * merge * merge;
    output[..len].copy_from_slice(&input[..len]);
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
    pub attn_buf: Vec<f32>,
    pub attn_out_buf: Vec<f32>,
    pub score_buf: Vec<f32>,
    pub proj_buf: Vec<f32>,
    pub ffn_buf: Vec<f32>,
    pub projected: Vec<f32>,
    pub deepstack: Vec<f32>,
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
            attn_buf: Vec::new(),
            attn_out_buf: Vec::new(),
            score_buf: Vec::new(),
            proj_buf: Vec::new(),
            ffn_buf: Vec::new(),
            projected: Vec::new(),
            deepstack: Vec::new(),
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
        let n_ff = config.n_ff;
        let n_head = config.n_head;
        let d_head = config.d_head();
        let proj_dim = config.projection_dim;
        let n_merged_x = grid.grid_w;
        let n_merged_y = grid.grid_h;
        let merge_area = checked_len("merge area", &[grid.merge_size, grid.merge_size])?;
        let n_projected = checked_len("projected", &[n_merged_x, n_merged_y])?;
        let merged_embd = n_embd * merge_area;

        self.patch_embd.resize(n_tokens * n_embd, 0.0);
        self.merged.resize(n_tokens * n_embd, 0.0);
        self.pos_embd_buf.resize(n_tokens * n_embd, 0.0);
        self.residual.resize(n_tokens * n_embd, 0.0);
        self.qkv_buf.resize(n_tokens * n_embd * 3, 0.0);
        self.attn_buf.resize(3 * n_head * n_tokens * d_head, 0.0);
        self.attn_out_buf.resize(n_head * n_tokens * d_head, 0.0);
        self.score_buf.resize(n_head * n_tokens * n_tokens, 0.0);
        self.proj_buf.resize(n_tokens * n_embd, 0.0);
        self.ffn_buf.resize(n_tokens * n_ff, 0.0);
        self.projected.resize(n_projected * proj_dim, 0.0);
        self.deepstack.resize(
            config
                .has_deepstack_layers
                .iter()
                .filter(|enabled| **enabled)
                .count()
                * n_projected
                * proj_dim,
            0.0,
        );
        self.attn_concat.resize(n_tokens * n_embd, 0.0);
        self.project_concat_buf
            .resize(n_projected * merged_embd, 0.0);
        self.project_mm0_out.resize(n_projected * merged_embd, 0.0);
        self.patch_weight_buf.resize(0, 0.0);
        self.patch_weight_1_buf = None;
        self.patch_weight_packed.resize(0, 0.0);
        self.patch_weight_1_packed = None;
        let max_linear_input = n_embd.max(n_ff).max(merged_embd);
        self.q8_buf.resize(n_tokens * max_linear_input, 0u8);
        self.q8_scale_buf
            .resize(n_tokens * max_linear_input / 32, 0.0);
        Ok(())
    }
}

fn layer_norm_with_bias(x: &mut [f32], w: &[f32], b: &[f32], eps: f32) {
    let n = x.len().min(w.len()).min(b.len());
    let n_f64 = n as f64;
    let mean = (sum_f32(&x[..n]) / n_f64) as f32;
    let var = (sum_sq_centered_f32(&x[..n], mean) / n_f64) as f32;
    let inv = 1.0 / (var + eps).sqrt();
    let offset = inv * mean;
    for i in 0..n {
        x[i] = w[i] * (inv * x[i] - offset) + b[i];
    }
}

fn layer_norm_without_bias(x: &mut [f32], w: &[f32], eps: f32) {
    let n = x.len().min(w.len());
    let n_f64 = n as f64;
    let mean = (sum_f32(&x[..n]) / n_f64) as f32;
    let var = (sum_sq_centered_f32(&x[..n], mean) / n_f64) as f32;
    let inv = 1.0 / (var + eps).sqrt();
    for i in 0..n {
        x[i] = w[i] * inv * (x[i] - mean);
    }
}

fn patch_embed_scalar(
    frame_a: &[f32],
    frame_b: &[f32],
    img_w: usize,
    img_h: usize,
    w0: &[f32],
    w1: Option<&[f32]>,
    out: &mut [f32],
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
                            let pix_val_a = frame_a[(pix_y * img_w + pix_x) * 3 + c];
                            let pix_val_b = frame_b[(pix_y * img_w + pix_x) * 3 + c];
                            let w_idx = c * ps * ps + ky * ps + kx;
                            sum0 += w0_row[w_idx] * pix_val_a;
                            if let Some(w1d) = w1_row {
                                sum1 += w1d[w_idx] * pix_val_b;
                            }
                        }
                    }
                }
                out[out_off + e] = sum0 + sum1;
            }
        }
    }
}

fn matmul_f16_f32_single(
    weight: &[u8],
    input: &[f32],
    output: &mut [f32],
    n_in: usize,
    n_out: usize,
) {
    let blocks = n_in / 32;
    let mut scales = vec![0.0f32; blocks];
    let mut q8_buf = vec![0u8; n_in];

    for row in 0..n_out {
        crate::ops::quantize_q8_0_into(&input[..n_in], n_in, &mut q8_buf, &mut scales);
        let weight_row_off = row * blocks * 34;
        let mut sum = 0.0f32;
        for (bi, &scale) in scales.iter().enumerate() {
            let off = weight_row_off + bi * 34;
            let d = crate::ops::f16_to_f32(u16::from_le_bytes([weight[off], weight[off + 1]]));
            for j in 0..32 {
                let q = weight[off + 2 + j] as i8 as i32;
                sum += d * q as f32 * input[bi * 32 + j];
            }
        }
        output[row] = sum;
    }
}

#[cfg(test)]
mod tests {
    use super::{decode_linear_weight, disable_missing_deepstack_layers, spatial_blocks, Q8Weight};
    use crate::core::tensor::{GGMLType, MetaValue, TensorInfo, TensorSource};

    struct TensorNames(Vec<String>);

    impl TensorSource for TensorNames {
        fn metadata(&self, _key: &str) -> Option<&MetaValue> {
            None
        }

        fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
            self.0.iter().any(|value| value == name).then(|| {
                &*Box::leak(Box::new(TensorInfo {
                    name: name.into(),
                    dims: vec![1],
                    ggml_type: GGMLType::F32,
                    offset: 0,
                }))
            })
        }

        fn tensor_slice(&self, _name: &str) -> Option<&[u8]> {
            None
        }
    }

    #[test]
    fn linear_weight_decoder_accepts_f32_f16_and_q8_0() {
        let f32_bytes = [1.0f32, -2.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        assert_eq!(decode_linear_weight(&f32_bytes, 2, 1), [1.0, -2.0]);

        let f16_bytes = [1.0f32, -2.0]
            .into_iter()
            .flat_map(|value| crate::ops::f32_to_f16(value).to_le_bytes())
            .collect::<Vec<_>>();
        assert_eq!(decode_linear_weight(&f16_bytes, 2, 1), [1.0, -2.0]);

        let mut q8_bytes = crate::ops::f32_to_f16(0.5).to_le_bytes().to_vec();
        q8_bytes.extend([2u8; 32]);
        assert_eq!(decode_linear_weight(&q8_bytes, 32, 1), [1.0; 32]);
    }

    #[test]
    fn spatial_blocks_keep_projector_and_deepstack_order_identical() {
        let input = [0.0, 1.0, 4.0, 5.0, 2.0, 3.0, 6.0, 7.0];
        let mut output = [0.0; 8];

        spatial_blocks(&input, 4, 2, 1, 2, &mut output);

        assert_eq!(output, input);
    }

    #[test]
    fn deepstack_flags_ignore_layers_missing_from_projector() {
        let source = TensorNames(vec!["v.deepstack.1.norm.weight".into()]);
        let mut flags = vec![false, true, false, true];

        disable_missing_deepstack_layers(&source, &mut flags);

        assert_eq!(flags, vec![false, true, false, false]);
    }

    #[test]
    fn q8_weight_supports_non_aligned_input_dimensions() {
        let weight = Q8Weight::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);
        let mut output = [0.0; 2];

        weight.matmul_single(&[1.0, 1.0, 1.0], &mut output, &mut [], &mut []);

        assert_eq!(output, [6.0, 15.0]);

        let mut batch_output = [0.0; 4];
        weight.matmul_batch(
            &[1.0, 1.0, 1.0, 2.0, 0.0, -1.0],
            &mut batch_output,
            2,
            &mut [],
            &mut [],
        );

        assert_eq!(batch_output, [6.0, 15.0, -1.0, 2.0]);
    }
}
