use super::super::contract::{require_tensor, require_tensor_any};
use super::uv_config::Gemma4UvConfig;
use crate::core::tensor::{GGMLType, TensorSource};
use crate::core::thread_pool::ComputePool;
use crate::ops::{dot_f16_f16_bytes, dot_f32, f32_slice_to_f16, sum_sq_f32};
use std::path::Path;

const POS_TABLE_STRIDE: usize = 1120;

pub struct Gemma4UvVisionModel<'a> {
    pub config: Gemma4UvConfig,
    pool: ComputePool,
    patch_weight: Vec<f32>,
    patch_bias: Vec<f32>,
    patch_norm_1_w: Vec<f32>,
    patch_norm_1_b: Vec<f32>,
    patch_norm_2_w: Vec<f32>,
    patch_norm_2_b: Vec<f32>,
    patch_norm_3_w: Vec<f32>,
    patch_norm_3_b: Vec<f32>,
    positions: Vec<f32>,
    projection: F16Linear<'a>,
}

#[derive(Clone, Copy)]
struct SharedMut<T>(*mut T);
unsafe impl<T> Send for SharedMut<T> {}
unsafe impl<T> Sync for SharedMut<T> {}

impl<T> SharedMut<T> {
    unsafe fn write(&self, index: usize, value: T) {
        self.0.add(index).write(value);
    }
}

struct F16Linear<'a> {
    weight: &'a [u8],
    input: usize,
    output: usize,
}

impl<'a> Gemma4UvVisionModel<'a> {
    pub fn from_source(source: &'a dyn TensorSource, threads: usize) -> Result<Self, String> {
        let config = Gemma4UvConfig::from_source(source)?;
        let patch_dim = (config.patch_size * config.patch_size * config.in_channels) as u64;
        let patch_weight = f32_tensor(
            source,
            "v.patch_embd.weight",
            &[patch_dim, config.embd as u64],
        )?;
        let patch_bias = f32_tensor(source, "v.patch_embd.bias", &[config.embd as u64])?;
        let patch_norm_1_w = f32_tensor(source, "v.patch_norm.1.weight", &[patch_dim])?;
        let patch_norm_1_b = f32_tensor(source, "v.patch_norm.1.bias", &[patch_dim])?;
        let patch_norm_2_w = f32_tensor(source, "v.patch_norm.2.weight", &[config.embd as u64])?;
        let patch_norm_2_b = f32_tensor(source, "v.patch_norm.2.bias", &[config.embd as u64])?;
        let patch_norm_3_w = f32_tensor(source, "v.patch_norm.3.weight", &[config.embd as u64])?;
        let patch_norm_3_b = f32_tensor(source, "v.patch_norm.3.bias", &[config.embd as u64])?;
        let positions = f32_tensor(
            source,
            "v.position_embd.weight",
            &[config.embd as u64, config.position_size as u64, 2],
        )?;
        let projection = F16Linear {
            weight: f16_tensor(
                source,
                "mm.input_projection.weight",
                &[config.embd as u64, config.projection as u64],
            )?,
            input: config.embd,
            output: config.projection,
        };
        Ok(Self {
            config,
            pool: ComputePool::new(threads.max(1)),
            patch_weight,
            patch_bias,
            patch_norm_1_w,
            patch_norm_1_b,
            patch_norm_2_w,
            patch_norm_2_b,
            patch_norm_3_w,
            patch_norm_3_b,
            positions,
            projection,
        })
    }

    pub fn encode_path(&self, path: &Path) -> Result<Vec<f32>, String> {
        let rgb = image::open(path)
            .map_err(|error| {
                format!(
                    "Failed to decode Gemma4-uv image {}: {error}",
                    path.display()
                )
            })?
            .to_rgb8();
        let image = preprocess_rgb(
            rgb.as_raw(),
            rgb.width() as usize,
            rgb.height() as usize,
            self.config,
        )?;
        self.encode_preprocessed(&image)
    }

    fn encode_preprocessed(&self, image: &PreprocessedImage) -> Result<Vec<f32>, String> {
        if image.width % self.config.patch_size != 0 || image.height % self.config.patch_size != 0 {
            return Err("Gemma4-uv image is not patch-aligned".into());
        }
        let patches_x = image.width / self.config.patch_size;
        let patches_y = image.height / self.config.patch_size;
        let n_patches = checked_len("Gemma4-uv patch count", &[patches_x, patches_y])?;
        if n_patches == 0 {
            return Err("Gemma4-uv image produced no patches".into());
        }
        if patches_x > self.config.position_size || patches_y > self.config.position_size {
            return Err("Gemma4-uv image exceeds position embedding table".into());
        }
        let c = self.config.in_channels;
        let patch_dim = self.config.patch_size * self.config.patch_size * c;
        let mut patches = zeroed_f32(
            "Gemma4-uv patches",
            checked_len("Gemma4-uv patches", &[n_patches, patch_dim])?,
        )?;
        im2col(
            &image.values,
            image.width,
            image.height,
            self.config.patch_size,
            c,
            &mut patches,
        )?;

        layer_norm_rows_inplace(
            &mut patches,
            &self.patch_norm_1_w,
            &self.patch_norm_1_b,
            self.config.norm_eps,
        )?;

        let mut embedded = zeroed_f32(
            "Gemma4-uv embedded patches",
            checked_len("Gemma4-uv embedded patches", &[n_patches, self.config.embd])?,
        )?;
        f32_matmul(
            &self.pool,
            &self.patch_weight,
            &patches,
            patch_dim,
            self.config.embd,
            n_patches,
            &mut embedded,
        )?;
        for (target, bias) in embedded.iter_mut().zip(self.patch_bias.iter().cycle()) {
            *target += *bias;
        }

        layer_norm_rows_inplace(
            &mut embedded,
            &self.patch_norm_2_w,
            &self.patch_norm_2_b,
            self.config.norm_eps,
        )?;
        let mut pos_x = Vec::with_capacity(n_patches);
        let mut pos_y = Vec::with_capacity(n_patches);
        for py in 0..patches_y {
            for px in 0..patches_x {
                pos_x.push(px as u32);
                pos_y.push(py as u32);
            }
        }
        add_positions(
            &self.positions,
            &mut embedded,
            &pos_x,
            &pos_y,
            self.config.embd,
        )?;

        layer_norm_rows_inplace(
            &mut embedded,
            &self.patch_norm_3_w,
            &self.patch_norm_3_b,
            self.config.norm_eps,
        )?;

        rms_norm_rows_inplace(&mut embedded, self.config.embd, self.config.rms_eps)?;

        let output_len = checked_len(
            "Gemma4-uv projected vision",
            &[n_patches, self.projection.output],
        )?;
        let mut output = zeroed_f32("Gemma4-uv projected vision", output_len)?;
        // ggml converts F32 activations to F16 before the matmul and
        // accumulates in `ggml_float` (double on x86_64) using SIMD FMA.
        // We reproduce the activation quantization, then do F32 multiply +
        // F64 scalar accumulation for reproducibility.
        f16_matmul(
            &self.pool,
            self.projection.weight,
            &embedded,
            self.projection.input,
            self.projection.output,
            n_patches,
            &mut output,
        )?;
        validate_finite("Gemma4-uv projected vision", &output)?;
        Ok(output)
    }
}

fn im2col(
    image: &[f32],
    width: usize,
    height: usize,
    patch: usize,
    channels: usize,
    output: &mut [f32],
) -> Result<(), String> {
    let n_patches_x = width / patch;
    let n_patches_y = height / patch;
    let plane = checked_len("Gemma4-uv plane", &[width, height])?;
    let expected = plane.checked_mul(channels).ok_or("Image length overflow")?;
    if image.len() != expected
        || output.len() != n_patches_x * n_patches_y * patch * patch * channels
    {
        return Err("Invalid Gemma4-uv im2col shape".into());
    }
    for py in 0..n_patches_y {
        for px in 0..n_patches_x {
            for channel in 0..channels {
                for y in 0..patch {
                    for x in 0..patch {
                        let source = channel * plane + (py * patch + y) * width + px * patch + x;
                        let target = ((py * n_patches_x + px) * channels * patch * patch)
                            + channel * patch * patch
                            + y * patch
                            + x;
                        output[target] = image[source];
                    }
                }
            }
        }
    }
    Ok(())
}

fn layer_norm_rows_inplace(
    values: &mut [f32],
    weight: &[f32],
    bias: &[f32],
    eps: f32,
) -> Result<(), String> {
    if values.is_empty() || weight.len() != bias.len() {
        return Err("Invalid Gemma4-uv LayerNorm parameter shape".into());
    }
    if values.len() % weight.len() != 0 {
        return Err("Invalid Gemma4-uv LayerNorm row shape".into());
    }
    // ggml_compute_forward_norm_f32 uses **f32** accumulation across two passes
    // (center into y, then y *= scale). We replicate that by computing mean/variance
    // in f32 and applying the weight/bias step in a second pass so the
    // rounding error pattern matches ggml bit-for-bit.
    for row in values.chunks_exact_mut(weight.len()) {
        let n = row.len() as f32;
        let mut sum = 0.0f32;
        for &value in row.iter() {
            sum += value;
        }
        let mean = sum / n;
        let mut variance = 0.0f32;
        for slot in row.iter_mut() {
            let centered = *slot - mean;
            *slot = centered;
            variance += centered * centered;
        }
        variance /= n;
        let scale = 1.0f32 / (variance + eps).sqrt();
        for (slot, (&w, &b)) in row.iter_mut().zip(weight.iter().zip(bias.iter())) {
            *slot = *slot * scale * w + b;
        }
    }
    Ok(())
}

fn rms_norm_rows_inplace(values: &mut [f32], width: usize, eps: f32) -> Result<(), String> {
    if width == 0 || values.is_empty() || values.len() % width != 0 {
        return Err("Invalid Gemma4-uv RMSNorm shape".into());
    }
    for row in values.chunks_exact_mut(width) {
        let mean_sq = sum_sq_f32(row) / row.len() as f64;
        let scale = 1.0 / (mean_sq + f64::from(eps)).sqrt();
        for slot in row.iter_mut() {
            *slot = (*slot as f64 * scale) as f32;
        }
    }
    Ok(())
}

fn add_positions(
    positions: &[f32],
    hidden: &mut [f32],
    pos_x: &[u32],
    pos_y: &[u32],
    embd: usize,
) -> Result<(), String> {
    if hidden.len() != pos_x.len() * embd || hidden.len() != pos_y.len() * embd {
        return Err("Invalid Gemma4-uv position embedding shape".into());
    }
    let y_offset = POS_TABLE_STRIDE * embd;
    for (index, (&x, &y)) in pos_x.iter().zip(pos_y.iter()).enumerate() {
        for feature in 0..embd {
            let px = positions[x as usize * embd + feature];
            let py = positions[y_offset + y as usize * embd + feature];
            let slot = &mut hidden[index * embd + feature];
            *slot = (*slot + px) + py;
        }
    }
    Ok(())
}

fn f32_matmul(
    pool: &ComputePool,
    weight: &[f32],
    input: &[f32],
    in_dim: usize,
    out_dim: usize,
    rows: usize,
    output: &mut [f32],
) -> Result<(), String> {
    let input_len = checked_len("Gemma4-uv matmul input", &[rows, in_dim])?;
    let output_len = checked_len("Gemma4-uv matmul output", &[rows, out_dim])?;
    if input.len() != input_len || output.len() != output_len {
        return Err("Invalid Gemma4-uv matmul shape".into());
    }
    let output_ptr = SharedMut(output.as_mut_ptr());
    let total = output_len;
    pool.compute(|thread, threads| {
        for index in (thread..total).step_by(threads) {
            let row = index / out_dim;
            let column = index % out_dim;
            let value = dot_f32(
                &input[row * in_dim..(row + 1) * in_dim],
                &weight[column * in_dim..(column + 1) * in_dim],
                in_dim,
            );
            unsafe { output_ptr.write(index, value) };
        }
    });
    Ok(())
}

fn f16_matmul(
    pool: &ComputePool,
    weight: &[u8],
    input: &[f32],
    in_dim: usize,
    out_dim: usize,
    rows: usize,
    output: &mut [f32],
) -> Result<(), String> {
    let input_len = checked_len("Gemma4-uv f16 matmul input", &[rows, in_dim])?;
    let output_len = checked_len("Gemma4-uv f16 matmul output", &[rows, out_dim])?;
    if input.len() != input_len || output.len() != output_len {
        return Err("Invalid Gemma4-uv f16 matmul shape".into());
    }
    let mut activation = zeroed_u16("Gemma4-uv activation", input_len)?;
    f32_slice_to_f16(input, &mut activation);
    let output_ptr = SharedMut(output.as_mut_ptr());
    let total = output_len;
    pool.compute(|thread, threads| {
        for index in (thread..total).step_by(threads) {
            let row = index / out_dim;
            let column = index % out_dim;
            let value = dot_f16_f16_bytes(
                &activation[row * in_dim..(row + 1) * in_dim],
                &weight[column * in_dim * 2..(column + 1) * in_dim * 2],
                in_dim,
            );
            unsafe { output_ptr.write(index, value) };
        }
    });
    Ok(())
}

fn checked_len(label: &str, factors: &[usize]) -> Result<usize, String> {
    factors.iter().try_fold(1usize, |length, factor| {
        length
            .checked_mul(*factor)
            .ok_or_else(|| format!("{label} length overflow"))
    })
}

fn zeroed_f32(label: &str, len: usize) -> Result<Vec<f32>, String> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|_| format!("{label} allocation failed"))?;
    values.resize(len, 0.0);
    Ok(values)
}

fn zeroed_u16(label: &str, len: usize) -> Result<Vec<u16>, String> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|_| format!("{label} allocation failed"))?;
    values.resize(len, 0);
    Ok(values)
}

fn validate_finite(label: &str, values: &[f32]) -> Result<(), String> {
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return Err(format!("{label} contains no values or non-finite values"));
    }
    Ok(())
}

fn f32_tensor(source: &dyn TensorSource, name: &str, dims: &[u64]) -> Result<Vec<f32>, String> {
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("Missing tensor: {name}"))?;
    if info.dims != dims || info.ggml_type != GGMLType::F32 {
        return Err(format!(
            "Invalid tensor {name}: shape {:?} type {:?}; expected {:?} F32",
            info.dims, info.ggml_type, dims
        ));
    }
    let expected = usize::try_from(
        info.checked_nbytes()
            .ok_or_else(|| format!("Invalid tensor byte size: {name}"))?,
    )
    .map_err(|_| format!("Tensor byte size does not fit usize: {name}"))?;
    let bytes = source
        .tensor_slice(name)
        .ok_or_else(|| format!("Missing tensor data: {name}"))?;
    if bytes.len() != expected {
        return Err(format!(
            "Invalid tensor data length for {name}: {}; expected {expected}",
            bytes.len()
        ));
    }
    if bytes
        .chunks_exact(4)
        .any(|chunk| !f32::from_le_bytes(chunk.try_into().unwrap()).is_finite())
    {
        return Err(format!("Non-finite F32 tensor: {name}"));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect())
}

fn f16_tensor<'a>(
    source: &'a dyn TensorSource,
    name: &str,
    dims: &[u64],
) -> Result<&'a [u8], String> {
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("Missing tensor: {name}"))?;
    if info.dims != dims || info.ggml_type != GGMLType::F16 {
        return Err(format!(
            "Invalid tensor {name}: shape {:?} type {:?}; expected {:?} F16",
            info.dims, info.ggml_type, dims
        ));
    }
    let expected = usize::try_from(
        info.checked_nbytes()
            .ok_or_else(|| format!("Invalid tensor byte size: {name}"))?,
    )
    .map_err(|_| format!("Tensor byte size does not fit usize: {name}"))?;
    let bytes = source
        .tensor_slice(name)
        .ok_or_else(|| format!("Missing tensor data: {name}"))?;
    if bytes.len() != expected {
        return Err(format!(
            "Invalid tensor data length for {name}: {}; expected {expected}",
            bytes.len()
        ));
    }
    if bytes.chunks_exact(2).any(|chunk| {
        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
        bits & 0x7c00 == 0x7c00
    }) {
        return Err(format!("Non-finite F16 tensor: {name}"));
    }
    Ok(bytes)
}

/// Image preprocessing for Gemma4-uv.
///
/// `in_channels = 3` (RGB), effective `patch_size = 48` (base 16 × n_merge=3,
/// then n_merge collapses to 1 per llama.cpp `clip.cpp` line 1636). The image
/// must be resized, preserving aspect ratio, to the closest multiple of
/// `patch_size × n_merge = 48` while satisfying the `image_min_pixels` and
/// `image_max_pixels` bounds, using the `mtmd_image_preprocessor_dyn_size`
/// algorithm. For a 256×256 input this yields 432×432 (9×9 patches).
struct PreprocessedImage {
    values: Vec<f32>,
    width: usize,
    height: usize,
}

fn calc_resize_dyn_size(
    width: usize,
    height: usize,
    align_size: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> (usize, usize) {
    let round_by_factor =
        |x: f32| -> usize { ((x / align_size as f32).round() as usize) * align_size };
    let floor_by_factor =
        |x: f32| -> usize { ((x / align_size as f32).floor() as usize) * align_size };
    let ceil_by_factor =
        |x: f32| -> usize { ((x / align_size as f32).ceil() as usize) * align_size };
    let mut w_bar = align_size.max(round_by_factor(width as f32));
    let mut h_bar = align_size.max(round_by_factor(height as f32));
    if max_pixels > 0 && h_bar * w_bar > max_pixels {
        let beta = ((height as f32 * width as f32) / max_pixels as f32).sqrt();
        h_bar = align_size.max(floor_by_factor(height as f32 / beta));
        w_bar = align_size.max(floor_by_factor(width as f32 / beta));
    } else if min_pixels > 0 && h_bar * w_bar < min_pixels {
        let beta = (min_pixels as f32 / (height as f32 * width as f32)).sqrt();
        h_bar = ceil_by_factor(height as f32 * beta);
        w_bar = ceil_by_factor(width as f32 * beta);
    }
    (w_bar, h_bar)
}

fn preprocess_rgb(
    rgb: &[u8],
    width: usize,
    height: usize,
    config: Gemma4UvConfig,
) -> Result<PreprocessedImage, String> {
    if width == 0 || height == 0 || rgb.len() != width * height * 3 {
        return Err("Invalid Gemma4-uv RGB image".into());
    }
    // After llama.cpp's `n_merge = 1` collapse, the dyn_size preprocessor
    // aligns the resized image to `patch_size` (multiples of 48 for 12B).
    let (target_w, target_h) = calc_resize_dyn_size(
        width,
        height,
        config.patch_size,
        config.image_min_pixels,
        config.image_max_pixels,
    );
    let resized = super::resize_bicubic_pillow(rgb, width, height, target_w, target_h)?;
    let plane = target_w * target_h;
    let channels = config.in_channels;
    let mut values = zeroed_f32("Gemma4-uv image", plane * channels)?;
    for pixel in 0..plane {
        for channel in 0..channels {
            // RGB layout: [H, W, C]; plane layout: [C, H*W].
            let src = resized[pixel * 3 + channel] as f32 / 255.0;
            values[channel * plane + pixel] = src;
        }
    }
    validate_finite("Gemma4-uv preprocessed image", &values)?;
    Ok(PreprocessedImage {
        values,
        width: target_w,
        height: target_h,
    })
}

#[allow(dead_code)]
fn _unused() {
    let _ = require_tensor;
    let _ = require_tensor_any;
}

#[cfg(test)]
mod tests {
    use super::{layer_norm_rows_inplace, rms_norm_rows_inplace};

    #[test]
    fn layernorm_matches_ggml_scalar_accumulation() {
        // ggml_compute_forward_norm_f32 computes mean and variance using
        // scalar f32 accumulation (matches ggml's reference behavior).
        // This test pins that behavior so we can detect any drift.
        let mut row = vec![1.5f32, -2.25, 0.0, 4.5, -1.0, 0.75, -3.0, 2.0];
        let weight = vec![1.0f32; 8];
        let bias = vec![0.0f32; 8];
        let eps = 1e-5f32;
        let mut expected = row.clone();
        let n = expected.len() as f32;
        let mean = expected.iter().sum::<f32>() / n;
        let mut variance = 0.0f32;
        for value in expected.iter_mut() {
            *value -= mean;
            variance += *value * *value;
        }
        let scale = 1.0f32 / (variance / n + eps).sqrt();
        for value in expected.iter_mut() {
            *value = *value * scale * weight[0] + bias[0];
        }
        layer_norm_rows_inplace(&mut row, &weight, &bias, eps).unwrap();
        let row_bits: Vec<u32> = row.iter().map(|v| v.to_bits()).collect();
        let expected_bits: Vec<u32> = expected.iter().map(|v| v.to_bits()).collect();
        assert_eq!(row_bits, expected_bits);
    }

    #[test]
    fn rms_norm_with_ggml_eps_matches_scalar_accumulation() {
        let mut row = vec![1.5f32, -2.25, 0.0, 4.5, -1.0];
        let saved = row.clone();
        let n = row.len() as f64;
        let mean_sq: f64 = saved
            .iter()
            .map(|v| f64::from(*v) * f64::from(*v))
            .sum::<f64>()
            / n;
        let eps = 1e-6f32;
        let scale = 1.0f64 / (mean_sq + f64::from(eps)).sqrt();
        let expected: Vec<f32> = saved
            .iter()
            .map(|v| (f64::from(*v) * scale) as f32)
            .collect();
        let width = row.len();
        rms_norm_rows_inplace(&mut row, width, eps).unwrap();
        let row_bits: Vec<u32> = row.iter().map(|v| v.to_bits()).collect();
        let expected_bits: Vec<u32> = expected.iter().map(|v| v.to_bits()).collect();
        assert_eq!(row_bits, expected_bits);
    }
}
