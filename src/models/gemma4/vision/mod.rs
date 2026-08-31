pub mod config;
use crate::core::tensor::{GGMLType, TensorSource};
use crate::core::thread_pool::ComputePool;
use crate::ops::{dot_f16_f16_bytes, dot_f32, f16_to_f32, f32_to_f16, rope_neox, softmax_inplace};
pub use config::Gemma4VisionConfig;
use std::path::Path;

const EMBED: usize = 768;
const FFN: usize = 3072;
const HEADS: usize = 12;
const HEAD_DIM: usize = 64;
const PATCH: usize = 16;
const MERGE: usize = 3;
const PROJECTION: usize = 1536;
const POSITION_ROWS: usize = 10_240;
const EPS: f32 = 1e-6;
const ROPE_BASE: f32 = 100.0;
const ALIGN: usize = PATCH * MERGE;
const MIN_PIXELS: usize = 40 * ALIGN * ALIGN;
const MAX_PIXELS: usize = 280 * ALIGN * ALIGN;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ResizeGrid {
    width: usize,
    height: usize,
}

struct PreprocessedImage {
    values: Vec<f32>,
    width: usize,
    height: usize,
}

#[derive(Clone, Copy)]
struct Clamp {
    input_min: f32,
    input_max: f32,
    output_min: f32,
    output_max: f32,
}

struct F16Linear<'a> {
    weight: &'a [u8],
    input: usize,
    output: usize,
    clamp: Option<Clamp>,
}

struct VisionLayer<'a> {
    ln1: Vec<f32>,
    q: F16Linear<'a>,
    k: F16Linear<'a>,
    v: F16Linear<'a>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    output: F16Linear<'a>,
    attn_post_norm: Vec<f32>,
    ln2: Vec<f32>,
    gate: F16Linear<'a>,
    up: F16Linear<'a>,
    down: F16Linear<'a>,
    ffn_post_norm: Vec<f32>,
}

pub struct Gemma4VisionModel<'a> {
    pub config: Gemma4VisionConfig,
    pool: ComputePool,
    patch_weight: &'a [u8],
    positions: &'a [u8],
    layers: Vec<VisionLayer<'a>>,
    projection: F16Linear<'a>,
}

struct VisionScratch {
    activation_f16: Vec<u16>,
    x: Vec<f32>,
    normed: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    v_transposed: Vec<f32>,
    scores: Vec<f32>,
    attention: Vec<f32>,
    projected: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
}

#[derive(Clone, Copy)]
struct SharedMut<T>(*mut T);

unsafe impl<T> Send for SharedMut<T> {}
unsafe impl<T> Sync for SharedMut<T> {}

impl<T> SharedMut<T> {
    unsafe fn write(&self, index: usize, value: T) {
        self.0.add(index).write(value);
    }

    unsafe fn slice<'b>(&self, offset: usize, len: usize) -> &'b mut [T] {
        std::slice::from_raw_parts_mut(self.0.add(offset), len)
    }
}

impl<'a> Gemma4VisionModel<'a> {
    pub fn from_source(source: &'a dyn TensorSource, threads: usize) -> Result<Self, String> {
        let config = Gemma4VisionConfig::from_source(source)?;
        let patch_weight = f16_tensor(
            source,
            "v.patch_embd.weight",
            &[PATCH as u64, PATCH as u64, 3, EMBED as u64],
        )?;
        let positions = f32_tensor_bytes(
            source,
            "v.position_embd.weight",
            &[EMBED as u64, POSITION_ROWS as u64, 2],
        )?;
        let mut layers = Vec::new();
        layers
            .try_reserve_exact(config.layers)
            .map_err(|_| "Gemma4 vision layer allocation failed")?;
        for index in 0..config.layers {
            let prefix = format!("v.blk.{index}");
            layers.push(VisionLayer {
                ln1: f32_tensor(source, &format!("{prefix}.ln1.weight"), &[EMBED as u64])?,
                q: F16Linear::clippable(source, &format!("{prefix}.attn_q"), EMBED, EMBED)?,
                k: F16Linear::clippable(source, &format!("{prefix}.attn_k"), EMBED, EMBED)?,
                v: F16Linear::clippable(source, &format!("{prefix}.attn_v"), EMBED, EMBED)?,
                q_norm: f32_tensor(
                    source,
                    &format!("{prefix}.attn_q_norm.weight"),
                    &[HEAD_DIM as u64],
                )?,
                k_norm: f32_tensor(
                    source,
                    &format!("{prefix}.attn_k_norm.weight"),
                    &[HEAD_DIM as u64],
                )?,
                output: F16Linear::clippable(source, &format!("{prefix}.attn_out"), EMBED, EMBED)?,
                attn_post_norm: f32_tensor(
                    source,
                    &format!("{prefix}.attn_post_norm.weight"),
                    &[EMBED as u64],
                )?,
                ln2: f32_tensor(source, &format!("{prefix}.ln2.weight"), &[EMBED as u64])?,
                gate: F16Linear::clippable(source, &format!("{prefix}.ffn_gate"), EMBED, FFN)?,
                up: F16Linear::clippable(source, &format!("{prefix}.ffn_up"), EMBED, FFN)?,
                down: F16Linear::clippable(source, &format!("{prefix}.ffn_down"), FFN, EMBED)?,
                ffn_post_norm: f32_tensor(
                    source,
                    &format!("{prefix}.ffn_post_norm.weight"),
                    &[EMBED as u64],
                )?,
            });
        }
        let projection = F16Linear::plain(source, "mm.input_projection.weight", EMBED, PROJECTION)?;
        Ok(Self {
            config,
            pool: ComputePool::new(threads.max(1)),
            patch_weight,
            positions,
            layers,
            projection,
        })
    }

    pub fn encode_path(&self, path: &Path) -> Result<Vec<f32>, String> {
        let rgb = image::open(path)
            .map_err(|error| format!("Failed to decode Gemma4 image {}: {error}", path.display()))?
            .to_rgb8();
        let image = preprocess_rgb(rgb.as_raw(), rgb.width() as usize, rgb.height() as usize)?;
        #[cfg(feature = "parity-trace")]
        crate::parity_trace::report(crate::parity_trace::checkpoint(
            "gemma4.vision.preprocessed",
            None,
            &[image.width, image.height, 3, 1],
            &image.values,
        ));
        self.encode_preprocessed(&image)
    }

    fn encode_preprocessed(&self, image: &PreprocessedImage) -> Result<Vec<f32>, String> {
        if image.width % ALIGN != 0 || image.height % ALIGN != 0 {
            return Err("Gemma4 image is not patch-and-merge aligned".into());
        }
        let patches_x = image.width / PATCH;
        let patches_y = image.height / PATCH;
        let tokens = checked_len("Gemma4 vision token", &[patches_x, patches_y])?;
        let output_rows = checked_len(
            "Gemma4 projected row",
            &[patches_x / MERGE, patches_y / MERGE],
        )?;
        if tokens == 0 || output_rows == 0 {
            return Err("Gemma4 image produced no visual tokens".into());
        }

        let patches = patchify(&image.values, image.width, image.height)?;
        let token_embed = checked_len("Gemma4 token embedding", &[tokens, EMBED])?;
        let token_ffn = checked_len("Gemma4 FFN", &[tokens, FFN])?;
        let score_len = checked_len("Gemma4 attention score", &[HEADS, tokens, tokens])?;
        let mut scratch = VisionScratch {
            activation_f16: Vec::new(),
            x: zeroed_f32("Gemma4 hidden", token_embed)?,
            normed: zeroed_f32("Gemma4 normalized hidden", token_embed)?,
            q: zeroed_f32("Gemma4 queries", token_embed)?,
            k: zeroed_f32("Gemma4 keys", token_embed)?,
            v: zeroed_f32("Gemma4 values", token_embed)?,
            v_transposed: zeroed_f32("Gemma4 transposed values", token_embed)?,
            scores: zeroed_f32("Gemma4 attention scores", score_len)?,
            attention: zeroed_f32("Gemma4 attention output", token_embed)?,
            projected: zeroed_f32("Gemma4 layer projection", token_embed)?,
            gate: zeroed_f32("Gemma4 FFN gate", token_ffn)?,
            up: zeroed_f32("Gemma4 FFN up", token_ffn)?,
        };
        patch_embed(
            &self.pool,
            self.patch_weight,
            &patches,
            tokens,
            &mut scratch.x,
        )?;
        add_positions(self.positions, &mut scratch.x, patches_x, patches_y)?;

        for layer in &self.layers {
            scratch.normed.copy_from_slice(&scratch.x);
            weighted_rms_rows(&mut scratch.normed, &layer.ln1, EMBED)?;
            layer.q.forward(
                &self.pool,
                &scratch.normed,
                tokens,
                &mut scratch.q,
                &mut scratch.activation_f16,
            )?;
            layer.k.forward(
                &self.pool,
                &scratch.normed,
                tokens,
                &mut scratch.k,
                &mut scratch.activation_f16,
            )?;
            layer.v.forward(
                &self.pool,
                &scratch.normed,
                tokens,
                &mut scratch.v,
                &mut scratch.activation_f16,
            )?;
            weighted_rms_rows(&mut scratch.q, &layer.q_norm, HEAD_DIM)?;
            weighted_rms_rows(&mut scratch.k, &layer.k_norm, HEAD_DIM)?;
            unweighted_rms_rows(&mut scratch.v, HEAD_DIM)?;
            apply_2d_rope(&mut scratch.q, patches_x, patches_y)?;
            apply_2d_rope(&mut scratch.k, patches_x, patches_y)?;
            attention(
                &self.pool,
                &scratch.q,
                &scratch.k,
                &scratch.v,
                tokens,
                &mut scratch.v_transposed,
                &mut scratch.scores,
                &mut scratch.attention,
            )?;
            layer.output.forward(
                &self.pool,
                &scratch.attention,
                tokens,
                &mut scratch.projected,
                &mut scratch.activation_f16,
            )?;
            weighted_rms_rows(&mut scratch.projected, &layer.attn_post_norm, EMBED)?;
            add_inplace(&mut scratch.x, &scratch.projected)?;

            scratch.normed.copy_from_slice(&scratch.x);
            weighted_rms_rows(&mut scratch.normed, &layer.ln2, EMBED)?;
            layer.gate.forward(
                &self.pool,
                &scratch.normed,
                tokens,
                &mut scratch.gate,
                &mut scratch.activation_f16,
            )?;
            layer.up.forward(
                &self.pool,
                &scratch.normed,
                tokens,
                &mut scratch.up,
                &mut scratch.activation_f16,
            )?;
            for (gate, up) in scratch.gate.iter_mut().zip(&scratch.up) {
                *gate = gelu_quick(*gate) * *up;
            }
            layer.down.forward(
                &self.pool,
                &scratch.gate,
                tokens,
                &mut scratch.projected,
                &mut scratch.activation_f16,
            )?;
            weighted_rms_rows(&mut scratch.projected, &layer.ffn_post_norm, EMBED)?;
            add_inplace(&mut scratch.x, &scratch.projected)?;
        }

        let mut pooled = average_pool_3x3(&scratch.x, patches_x, patches_y)?;
        let scale = (EMBED as f32).sqrt();
        for row in pooled.chunks_exact_mut(EMBED) {
            for value in row.iter_mut() {
                *value *= scale;
            }
            rms_row_inplace(row, None)?;
        }
        let projected_len = checked_len("Gemma4 projected vision", &[output_rows, PROJECTION])?;
        let mut output = zeroed_f32("Gemma4 projected vision", projected_len)?;
        self.projection.forward(
            &self.pool,
            &pooled,
            output_rows,
            &mut output,
            &mut scratch.activation_f16,
        )?;
        validate_finite("Gemma4 projected vision", &output)?;
        #[cfg(feature = "parity-trace")]
        crate::parity_trace::report(crate::parity_trace::checkpoint(
            "gemma4.vision.projected",
            None,
            &[PROJECTION, output_rows, 1, 1],
            &output,
        ));
        Ok(output)
    }
}

impl<'a> F16Linear<'a> {
    fn plain(
        source: &'a dyn TensorSource,
        name: &str,
        input: usize,
        output: usize,
    ) -> Result<Self, String> {
        Ok(Self {
            weight: f16_tensor(source, name, &[input as u64, output as u64])?,
            input,
            output,
            clamp: None,
        })
    }

    fn clippable(
        source: &'a dyn TensorSource,
        prefix: &str,
        input: usize,
        output: usize,
    ) -> Result<Self, String> {
        let clamp = Clamp {
            input_max: f32_scalar(source, &format!("{prefix}.input_max"))?,
            input_min: f32_scalar(source, &format!("{prefix}.input_min"))?,
            output_max: f32_scalar(source, &format!("{prefix}.output_max"))?,
            output_min: f32_scalar(source, &format!("{prefix}.output_min"))?,
        };
        if clamp.input_min > clamp.input_max || clamp.output_min > clamp.output_max {
            return Err(format!("Invalid clamp bounds: {prefix}"));
        }
        Ok(Self {
            weight: f16_tensor(
                source,
                &format!("{prefix}.weight"),
                &[input as u64, output as u64],
            )?,
            input,
            output,
            clamp: Some(clamp),
        })
    }

    fn forward(
        &self,
        pool: &ComputePool,
        input: &[f32],
        rows: usize,
        output: &mut [f32],
        activation: &mut Vec<u16>,
    ) -> Result<(), String> {
        let input_len = checked_len("F16 linear input", &[rows, self.input])?;
        let output_len = checked_len("F16 linear output", &[rows, self.output])?;
        if input.len() != input_len || output.len() != output_len || rows == 0 {
            return Err(format!(
                "Invalid F16 linear shape: input {}, output {}, rows {rows}, widths {} -> {}",
                input.len(),
                output.len(),
                self.input,
                self.output
            ));
        }
        resize_u16(activation, input_len, "F16 activation")?;
        match self.clamp {
            Some(clamp) => {
                for (source, target) in input.iter().zip(activation.iter_mut()) {
                    *target = f32_to_f16(source.clamp(clamp.input_min, clamp.input_max));
                }
            }
            None => {
                for (source, target) in input.iter().zip(activation.iter_mut()) {
                    *target = f32_to_f16(*source);
                }
            }
        }
        let output_ptr = SharedMut(output.as_mut_ptr());
        let total = output_len;
        pool.compute(|thread, threads| {
            for index in (thread..total).step_by(threads) {
                let row = index / self.output;
                let column = index % self.output;
                let value = dot_f16_f16_bytes(
                    &activation[row * self.input..(row + 1) * self.input],
                    &self.weight[column * self.input * 2..(column + 1) * self.input * 2],
                    self.input,
                );
                let value = self
                    .clamp
                    .map(|clamp| value.clamp(clamp.output_min, clamp.output_max))
                    .unwrap_or(value);
                unsafe { output_ptr.write(index, value) };
            }
        });
        validate_finite("F16 linear output", output)
    }
}

fn gemma4v_resize_grid(width: usize, height: usize) -> Result<ResizeGrid, String> {
    if width == 0 || height == 0 {
        return Err("Gemma4 image dimensions must be nonzero".into());
    }
    let pixels = width
        .checked_mul(height)
        .ok_or("Gemma4 image dimensions overflow")?;
    let round_aligned =
        |value: usize| (((value as f32 / ALIGN as f32).round() as usize).max(1)) * ALIGN;
    let mut aligned_width = round_aligned(width);
    let mut aligned_height = round_aligned(height);
    let aligned_pixels = aligned_width
        .checked_mul(aligned_height)
        .ok_or("Gemma4 aligned image dimensions overflow")?;
    if aligned_pixels > MAX_PIXELS {
        let beta = (pixels as f32 / MAX_PIXELS as f32).sqrt();
        aligned_width = ((width as f32 / beta / ALIGN as f32).floor() as usize).max(1) * ALIGN;
        aligned_height = ((height as f32 / beta / ALIGN as f32).floor() as usize).max(1) * ALIGN;
    } else if aligned_pixels < MIN_PIXELS {
        let beta = (MIN_PIXELS as f32 / pixels as f32).sqrt();
        aligned_width = ((width as f32 * beta / ALIGN as f32).ceil() as usize).max(1) * ALIGN;
        aligned_height = ((height as f32 * beta / ALIGN as f32).ceil() as usize).max(1) * ALIGN;
    }
    aligned_width
        .checked_mul(aligned_height)
        .ok_or("Gemma4 resized image dimensions overflow")?;
    Ok(ResizeGrid {
        width: aligned_width,
        height: aligned_height,
    })
}

fn preprocess_rgb(rgb: &[u8], width: usize, height: usize) -> Result<PreprocessedImage, String> {
    let source_len = checked_len("Gemma4 RGB image", &[width, height, 3])?;
    if width == 0 || height == 0 || rgb.len() != source_len {
        return Err("Invalid Gemma4 RGB image".into());
    }
    let grid = gemma4v_resize_grid(width, height)?;
    let scale = (grid.width as f32 / width as f32).min(grid.height as f32 / height as f32);
    let resized_width = ((width as f32 * scale).ceil() as usize).min(grid.width);
    let resized_height = ((height as f32 * scale).ceil() as usize).min(grid.height);
    if resized_width == 0 || resized_height == 0 {
        return Err("Gemma4 image resize produced an empty image".into());
    }
    let resized = resize_bicubic_pillow(rgb, width, height, resized_width, resized_height)?;
    let padded_len = checked_len("Gemma4 padded RGB image", &[grid.width, grid.height, 3])?;
    let mut padded = zeroed_u8("Gemma4 padded RGB image", padded_len)?;
    let offset_x = (grid.width - resized_width) / 2;
    let offset_y = (grid.height - resized_height) / 2;
    for y in 0..resized_height {
        let source = &resized[y * resized_width * 3..(y + 1) * resized_width * 3];
        let target_start = ((y + offset_y) * grid.width + offset_x) * 3;
        padded[target_start..target_start + source.len()].copy_from_slice(source);
    }
    let plane = checked_len("Gemma4 image plane", &[grid.width, grid.height])?;
    let mut values = zeroed_f32("Gemma4 preprocessed image", plane * 3)?;
    for index in 0..plane {
        values[index] = padded[index * 3] as f32 / 255.0;
        values[plane + index] = padded[index * 3 + 1] as f32 / 255.0;
        values[2 * plane + index] = padded[index * 3 + 2] as f32 / 255.0;
    }
    validate_finite("Gemma4 preprocessed image", &values)?;
    Ok(PreprocessedImage {
        values,
        width: grid.width,
        height: grid.height,
    })
}

fn patchify(pixels: &[f32], width: usize, height: usize) -> Result<Vec<u16>, String> {
    let plane = checked_len("Gemma4 patch image plane", &[width, height])?;
    let expected = plane
        .checked_mul(3)
        .ok_or("Gemma4 patch image length overflow")?;
    if width == 0
        || height == 0
        || width % PATCH != 0
        || height % PATCH != 0
        || pixels.len() != expected
        || pixels.iter().any(|value| !value.is_finite())
    {
        return Err("Invalid Gemma4 patch image".into());
    }
    let tokens = checked_len("Gemma4 patch count", &[width / PATCH, height / PATCH])?;
    let patch_len = 3 * PATCH * PATCH;
    let mut patches = zeroed_u16(
        "Gemma4 patches",
        checked_len("Gemma4 patches", &[tokens, patch_len])?,
    )?;
    let mut token = 0;
    for patch_y in 0..height / PATCH {
        for patch_x in 0..width / PATCH {
            let target = &mut patches[token * patch_len..(token + 1) * patch_len];
            let mut offset = 0;
            for channel in 0..3 {
                for y in 0..PATCH {
                    for x in 0..PATCH {
                        let source =
                            channel * plane + (patch_y * PATCH + y) * width + patch_x * PATCH + x;
                        target[offset] = f32_to_f16(pixels[source].mul_add(2.0, -1.0));
                        offset += 1;
                    }
                }
            }
            token += 1;
        }
    }
    Ok(patches)
}

fn patch_embed(
    pool: &ComputePool,
    weight: &[u8],
    patches: &[u16],
    tokens: usize,
    output: &mut [f32],
) -> Result<(), String> {
    let patch_len = 3 * PATCH * PATCH;
    if patches.len() != tokens * patch_len || output.len() != tokens * EMBED {
        return Err("Invalid Gemma4 patch embedding shape".into());
    }
    let output_ptr = SharedMut(output.as_mut_ptr());
    pool.compute(|thread, threads| {
        for index in (thread..output.len()).step_by(threads) {
            let token = index / EMBED;
            let column = index % EMBED;
            let value = dot_f16_f16_bytes(
                &patches[token * patch_len..(token + 1) * patch_len],
                &weight[column * patch_len * 2..(column + 1) * patch_len * 2],
                patch_len,
            );
            unsafe { output_ptr.write(index, value) };
        }
    });
    validate_finite("Gemma4 patch embeddings", output)
}

fn add_positions(
    positions: &[u8],
    hidden: &mut [f32],
    patches_x: usize,
    patches_y: usize,
) -> Result<(), String> {
    if patches_x > POSITION_ROWS
        || patches_y > POSITION_ROWS
        || hidden.len() != patches_x * patches_y * EMBED
    {
        return Err("Invalid Gemma4 position embedding shape".into());
    }
    let y_table_offset = POSITION_ROWS * EMBED;
    for y in 0..patches_y {
        for x in 0..patches_x {
            let token = y * patches_x + x;
            for feature in 0..EMBED {
                let index = token * EMBED + feature;
                hidden[index] = add_position_tables(
                    hidden[index],
                    read_f32(positions, x * EMBED + feature),
                    read_f32(positions, y_table_offset + y * EMBED + feature),
                );
            }
        }
    }
    validate_finite("Gemma4 position embeddings", hidden)
}

fn add_position_tables(hidden: f32, pos_x: f32, pos_y: f32) -> f32 {
    (hidden + pos_x) + pos_y
}

fn apply_2d_rope(values: &mut [f32], patches_x: usize, patches_y: usize) -> Result<(), String> {
    if values.len() != patches_x * patches_y * EMBED {
        return Err("Invalid Gemma4 RoPE shape".into());
    }
    for y in 0..patches_y {
        for x in 0..patches_x {
            let token = y * patches_x + x;
            for head in 0..HEADS {
                let offset = token * EMBED + head * HEAD_DIM;
                rope_neox(
                    &mut values[offset..offset + HEAD_DIM / 2],
                    x,
                    HEAD_DIM / 2,
                    ROPE_BASE,
                );
                rope_neox(
                    &mut values[offset + HEAD_DIM / 2..offset + HEAD_DIM],
                    y,
                    HEAD_DIM / 2,
                    ROPE_BASE,
                );
            }
        }
    }
    validate_finite("Gemma4 RoPE", values)
}

#[allow(clippy::too_many_arguments)]
fn attention(
    pool: &ComputePool,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    tokens: usize,
    v_transposed: &mut [f32],
    scores: &mut [f32],
    output: &mut [f32],
) -> Result<(), String> {
    let hidden_len = checked_len("Gemma4 attention hidden", &[tokens, EMBED])?;
    let score_len = checked_len("Gemma4 attention score", &[HEADS, tokens, tokens])?;
    if q.len() != hidden_len
        || k.len() != hidden_len
        || v.len() != hidden_len
        || v_transposed.len() != hidden_len
        || output.len() != hidden_len
        || scores.len() != score_len
    {
        return Err("Invalid Gemma4 attention shape".into());
    }
    for head in 0..HEADS {
        for dimension in 0..HEAD_DIM {
            for token in 0..tokens {
                v_transposed[(head * HEAD_DIM + dimension) * tokens + token] =
                    v[token * EMBED + head * HEAD_DIM + dimension];
            }
        }
    }
    let score_ptr = SharedMut(scores.as_mut_ptr());
    let output_ptr = SharedMut(output.as_mut_ptr());
    let tasks = HEADS * tokens;
    pool.compute(|thread, threads| {
        for task in (thread..tasks).step_by(threads) {
            let head = task / tokens;
            let query = task % tokens;
            let score = unsafe { score_ptr.slice(task * tokens, tokens) };
            let q_row = &q[query * EMBED + head * HEAD_DIM..query * EMBED + (head + 1) * HEAD_DIM];
            for key in 0..tokens {
                let k_row = &k[key * EMBED + head * HEAD_DIM..key * EMBED + (head + 1) * HEAD_DIM];
                score[key] = dot_f32(q_row, k_row, HEAD_DIM);
            }
            softmax_inplace(score);
            for dimension in 0..HEAD_DIM {
                let values = &v_transposed[(head * HEAD_DIM + dimension) * tokens
                    ..(head * HEAD_DIM + dimension + 1) * tokens];
                let value = dot_f32(values, score, tokens);
                unsafe { output_ptr.write(query * EMBED + head * HEAD_DIM + dimension, value) };
            }
        }
    });
    validate_finite("Gemma4 attention", output)
}

fn average_pool_3x3(input: &[f32], patches_x: usize, patches_y: usize) -> Result<Vec<f32>, String> {
    if patches_x % MERGE != 0
        || patches_y % MERGE != 0
        || input.len() != patches_x * patches_y * EMBED
    {
        return Err("Invalid Gemma4 average-pool shape".into());
    }
    let out_x = patches_x / MERGE;
    let out_y = patches_y / MERGE;
    let mut output = zeroed_f32(
        "Gemma4 pooled vision",
        checked_len("Gemma4 pooled vision", &[out_x, out_y, EMBED])?,
    )?;
    for y in 0..out_y {
        for x in 0..out_x {
            let output_row = &mut output[(y * out_x + x) * EMBED..(y * out_x + x + 1) * EMBED];
            for dy in 0..MERGE {
                for dx in 0..MERGE {
                    let source = &input[((y * MERGE + dy) * patches_x + x * MERGE + dx) * EMBED
                        ..((y * MERGE + dy) * patches_x + x * MERGE + dx + 1) * EMBED];
                    for (target, value) in output_row.iter_mut().zip(source) {
                        *target += *value;
                    }
                }
            }
            for value in output_row {
                *value /= (MERGE * MERGE) as f32;
            }
        }
    }
    validate_finite("Gemma4 pooled vision", &output)?;
    Ok(output)
}

fn weighted_rms_rows(values: &mut [f32], weight: &[f32], width: usize) -> Result<(), String> {
    if width == 0 || values.len() % width != 0 || weight.len() != width {
        return Err("Invalid weighted RMS shape".into());
    }
    for row in values.chunks_exact_mut(width) {
        rms_row_inplace(row, Some(weight))?;
    }
    Ok(())
}

fn unweighted_rms_rows(values: &mut [f32], width: usize) -> Result<(), String> {
    if width == 0 || values.len() % width != 0 {
        return Err("Invalid unweighted RMS shape".into());
    }
    for row in values.chunks_exact_mut(width) {
        rms_row_inplace(row, None)?;
    }
    Ok(())
}

fn rms_row_inplace(row: &mut [f32], weight: Option<&[f32]>) -> Result<(), String> {
    if row.is_empty() || weight.is_some_and(|weight| weight.len() != row.len()) {
        return Err("Invalid Gemma4 RMS row".into());
    }
    let sum = ggml_sequential_sum_sq(row);
    let mean = (sum / row.len() as f64) as f32;
    let scale = 1.0 / (mean + EPS).sqrt();
    match weight {
        Some(weight) => {
            for (value, weight) in row.iter_mut().zip(weight) {
                *value = *value * scale * *weight;
            }
        }
        None => {
            for value in row.iter_mut() {
                *value *= scale;
            }
        }
    }
    validate_finite("Gemma4 RMS row", row)
}

fn ggml_sequential_sum_sq(values: &[f32]) -> f64 {
    let mut sum = 0.0f64;
    for &value in values {
        sum += f64::from(value * value);
    }
    sum
}

fn gelu_quick(value: f32) -> f32 {
    let value = f16_to_f32(f32_to_f16(value));
    f16_to_f32(f32_to_f16(value / (1.0 + (-1.702 * value).exp())))
}

fn add_inplace(target: &mut [f32], source: &[f32]) -> Result<(), String> {
    if target.len() != source.len() {
        return Err("Invalid Gemma4 residual shape".into());
    }
    for (target, source) in target.iter_mut().zip(source) {
        *target += *source;
    }
    validate_finite("Gemma4 residual", target)
}

fn resize_bicubic_pillow(
    rgb: &[u8],
    width: usize,
    height: usize,
    target_width: usize,
    target_height: usize,
) -> Result<Vec<u8>, String> {
    if width == target_width && height == target_height {
        return Ok(rgb.to_vec());
    }
    let mut current = rgb.to_vec();
    let mut current_width = width;
    let mut current_height = height;
    if target_width != width {
        let (kernel, bounds, weights) = bicubic_weights(width, target_width)?;
        let mut horizontal = zeroed_u8(
            "Gemma4 horizontal resize",
            checked_len("Gemma4 horizontal resize", &[target_width, height, 3])?,
        )?;
        for y in 0..height {
            for x in 0..target_width {
                let start = bounds[x * 2];
                let count = bounds[x * 2 + 1];
                for channel in 0..3 {
                    let mut sum = 1i32 << 21;
                    for offset in 0..count {
                        sum += current[(y * width + start + offset) * 3 + channel] as i32
                            * weights[x * kernel + offset];
                    }
                    horizontal[(y * target_width + x) * 3 + channel] = clip_u8(sum >> 22);
                }
            }
        }
        current = horizontal;
        current_width = target_width;
    }
    if target_height != height {
        let (kernel, bounds, weights) = bicubic_weights(height, target_height)?;
        let mut vertical = zeroed_u8(
            "Gemma4 vertical resize",
            checked_len("Gemma4 vertical resize", &[current_width, target_height, 3])?,
        )?;
        for y in 0..target_height {
            let start = bounds[y * 2];
            let count = bounds[y * 2 + 1];
            for x in 0..current_width {
                for channel in 0..3 {
                    let mut sum = 1i32 << 21;
                    for offset in 0..count {
                        sum += current[((start + offset) * current_width + x) * 3 + channel] as i32
                            * weights[y * kernel + offset];
                    }
                    vertical[(y * current_width + x) * 3 + channel] = clip_u8(sum >> 22);
                }
            }
        }
        current = vertical;
        current_height = target_height;
    }
    if current_width != target_width || current_height != target_height {
        return Err("Gemma4 bicubic resize shape mismatch".into());
    }
    Ok(current)
}

fn bicubic_weights(input: usize, output: usize) -> Result<(usize, Vec<usize>, Vec<i32>), String> {
    if input == 0 || output == 0 {
        return Err("Invalid Gemma4 bicubic resize dimension".into());
    }
    let scale = input as f64 / output as f64;
    let filter_scale = scale.max(1.0);
    let support = 2.0 * filter_scale;
    let kernel = support.ceil() as usize * 2 + 1;
    let mut bounds = zeroed_usize("Gemma4 bicubic bounds", output * 2)?;
    let mut weights = zeroed_i32(
        "Gemma4 bicubic weights",
        output
            .checked_mul(kernel)
            .ok_or("Gemma4 bicubic weight length overflow")?,
    )?;
    for out in 0..output {
        let center = (out as f64 + 0.5) * scale;
        let start = ((center - support + 0.5) as isize).max(0) as usize;
        let end = ((center + support + 0.5) as usize).min(input);
        let count = end.saturating_sub(start);
        if count == 0 || count > kernel {
            return Err("Invalid Gemma4 bicubic filter support".into());
        }
        bounds[out * 2] = start;
        bounds[out * 2 + 1] = count;
        let mut floating = Vec::new();
        floating
            .try_reserve_exact(count)
            .map_err(|_| "Gemma4 bicubic coefficient allocation failed")?;
        let mut sum = 0.0;
        for index in 0..count {
            let value = bicubic_filter((index + start) as f64 - center + 0.5, filter_scale);
            floating.push(value);
            sum += value;
        }
        for (index, value) in floating.into_iter().enumerate() {
            let normalized = if sum == 0.0 { value } else { value / sum };
            let scaled = normalized * (1u64 << 22) as f64;
            weights[out * kernel + index] = (scaled + if scaled < 0.0 { -0.5 } else { 0.5 }) as i32;
        }
    }
    Ok((kernel, bounds, weights))
}

fn bicubic_filter(distance: f64, filter_scale: f64) -> f64 {
    let x = (distance / filter_scale).abs();
    if x < 1.0 {
        ((1.5 * x - 2.5) * x * x) + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * -0.5
    } else {
        0.0
    }
}

fn clip_u8(value: i32) -> u8 {
    value.clamp(0, 255) as u8
}

fn f16_tensor<'a>(
    source: &'a dyn TensorSource,
    name: &str,
    dims: &[u64],
) -> Result<&'a [u8], String> {
    let bytes = checked_tensor(source, name, dims, GGMLType::F16)?;
    if bytes.chunks_exact(2).any(|chunk| {
        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
        bits & 0x7c00 == 0x7c00
    }) {
        return Err(format!("Non-finite F16 tensor: {name}"));
    }
    Ok(bytes)
}

fn f32_tensor_bytes<'a>(
    source: &'a dyn TensorSource,
    name: &str,
    dims: &[u64],
) -> Result<&'a [u8], String> {
    let bytes = checked_tensor(source, name, dims, GGMLType::F32)?;
    if bytes
        .chunks_exact(4)
        .any(|chunk| !f32::from_le_bytes(chunk.try_into().unwrap()).is_finite())
    {
        return Err(format!("Non-finite F32 tensor: {name}"));
    }
    Ok(bytes)
}

fn f32_tensor(source: &dyn TensorSource, name: &str, dims: &[u64]) -> Result<Vec<f32>, String> {
    let bytes = f32_tensor_bytes(source, name, dims)?;
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect())
}

fn f32_scalar(source: &dyn TensorSource, name: &str) -> Result<f32, String> {
    let values = f32_tensor(source, name, &[1])?;
    Ok(values[0])
}

fn checked_tensor<'a>(
    source: &'a dyn TensorSource,
    name: &str,
    dims: &[u64],
    kind: GGMLType,
) -> Result<&'a [u8], String> {
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("Missing tensor: {name}"))?;
    if dims.is_empty() || dims.contains(&0) || info.dims != dims || info.ggml_type != kind {
        return Err(format!(
            "Invalid tensor {name}: shape {:?} type {:?}; expected {:?} {:?}",
            info.dims, info.ggml_type, dims, kind
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
    Ok(bytes)
}

fn read_f32(bytes: &[u8], index: usize) -> f32 {
    let offset = index * 4;
    f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
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

fn zeroed_u8(label: &str, len: usize) -> Result<Vec<u8>, String> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|_| format!("{label} allocation failed"))?;
    values.resize(len, 0);
    Ok(values)
}

fn zeroed_usize(label: &str, len: usize) -> Result<Vec<usize>, String> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|_| format!("{label} allocation failed"))?;
    values.resize(len, 0);
    Ok(values)
}

fn zeroed_i32(label: &str, len: usize) -> Result<Vec<i32>, String> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|_| format!("{label} allocation failed"))?;
    values.resize(len, 0);
    Ok(values)
}

fn resize_u16(values: &mut Vec<u16>, len: usize, label: &str) -> Result<(), String> {
    if values.len() < len {
        values
            .try_reserve_exact(len - values.len())
            .map_err(|_| format!("{label} allocation failed"))?;
    }
    values.resize(len, 0);
    Ok(())
}

fn validate_finite(label: &str, values: &[f32]) -> Result<(), String> {
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return Err(format!("{label} contains no values or non-finite values"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemma4v_resize_is_patch_and_merge_aligned() {
        let grid = gemma4v_resize_grid(640, 480).unwrap();
        assert_eq!(grid.width % 16, 0);
        assert_eq!(grid.height % 16, 0);
        assert_eq!((grid.width / 16) % 3, 0);
        assert_eq!((grid.height / 16) % 3, 0);
    }

    #[test]
    fn gemma4v_accepts_extreme_aspect_ratio_oracle_grid() {
        let grid = gemma4v_resize_grid(1, 100_000).unwrap();
        assert_eq!(grid.width, ALIGN);
        assert!(grid.height > MAX_PIXELS / ALIGN);
    }

    #[test]
    fn attention_uses_stable_scalar_softmax() {
        let tokens = 4;
        let hidden_len = tokens * EMBED;
        let mut q = vec![0.0; hidden_len];
        let mut k = vec![0.0; hidden_len];
        let mut v = vec![0.0; hidden_len];
        q[0] = 1.0;
        let logits = [0x4100_5f5f, 0x40e9_d754, 0x411f_b16f, 0x4110_0e1d].map(f32::from_bits);
        let values = [1.0, -2.0, 0.75, 4.0];
        for token in 0..tokens {
            k[token * EMBED] = logits[token];
            v[token * EMBED] = values[token];
        }
        let mut v_transposed = vec![0.0; hidden_len];
        let mut scores = vec![0.0; HEADS * tokens * tokens];
        let mut output = vec![0.0; hidden_len];

        attention(
            &ComputePool::new(1),
            &q,
            &k,
            &v,
            tokens,
            &mut v_transposed,
            &mut scores,
            &mut output,
        )
        .unwrap();

        assert_eq!(
            scores[..tokens]
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            [0x3db6_4746, 0x3d32_346f, 0x3f21_5bc5, 0x3e72_e02c]
        );
        assert_eq!(output[0].to_bits(), 0x3fb6_33ae);
    }

    #[test]
    fn rms_matches_llama_sequential_f64_accumulation() {
        let mut actual: Vec<f32> = (0..EMBED)
            .map(|index| {
                let index = index as f32;
                (index * 0.013).sin() * 2.5 + (index * 0.007).cos() * 0.125
            })
            .collect();
        let mut expected = actual.clone();
        let mut sum = 0.0f64;
        for &value in &expected {
            sum += f64::from(value * value);
        }
        assert_eq!(ggml_sequential_sum_sq(&expected).to_bits(), sum.to_bits());
        let mean = (sum / expected.len() as f64) as f32;
        let scale = 1.0f32 / (mean + EPS).sqrt();
        for value in &mut expected {
            *value *= scale;
        }

        rms_row_inplace(&mut actual, None).unwrap();

        assert_eq!(
            actual
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn position_embedding_adds_x_then_y_like_ggml() {
        let hidden = f32::from_bits(0xbe8d_e000);
        let pos_x = f32::from_bits(0xb202_0000);
        let pos_y = f32::from_bits(0xb211_0000);

        assert_eq!(
            add_position_tables(hidden, pos_x, pos_y).to_bits(),
            0xbe8d_e000
        );
    }

    #[test]
    fn gemma4v_rejects_empty_or_nonfinite_pixels() {
        assert!(preprocess_rgb(&[], 0, 0).is_err());
        assert!(patchify(&vec![f32::NAN; 224 * 224 * 3], 224, 224).is_err());
    }
}
