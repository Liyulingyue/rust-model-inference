//! Vision forward path for EmbeddingGemma-2 (SigLIP-style + axial RoPE).
//!
//! 16-layer ViT with:
//! - patch_size=16, spatial_merge_size=3
//! - hidden_size=768, intermediate_size=3072, head_count=12 (head_dim=64)
//! - axial RoPE with rope_theta=100.0 (rope_type=axial)
//! - position_embedding_size=10240 (max h, w grid positions)
//! - 4-norm sandwich: input_layernorm, post_attention_layernorm,
//!   pre_feedforward_layernorm, post_feedforward_layernorm
//! - Q/K RMSNorm (per-head)
//!
//! Forward: image (H, W, 3) → patch_embed → add 2D positional embeddings →
//! 16 ViT blocks → spatial merge (groups of 3×3 → 1) → 280 soft tokens →
//! `vision_embed.weight` projection to the text hidden space (512).
//!
//! Image preprocessing:
//! - resize to image_size × image_size (default 768)
//! - per-patch: pixel 16×16×3 → linear (3*16*16 → 768) + bias
//! - patches arranged in (n_patches_y, n_patches_x) = (image/patch) grid
//! - patch_pos: rows × cols (each (h, w) gets pos = position_embedding[h, w])
//! - spatial merge: every (merge × merge) block of patches becomes 1 token

use crate::core::tensor::TensorSource;
use crate::core::thread_pool::ComputePool;
use crate::ops::kernel::{QuantizedTensor, Weight};
use std::sync::Arc;

pub const RMS_NORM_EPS: f32 = 1e-6;

#[derive(Debug, Clone)]
pub struct EmbeddingGemmaVisionConfig {
    pub depth: usize,
    pub embd: usize,
    pub intermediate_size: usize,
    pub head_count: usize,
    pub head_dim: usize,
    pub patch_size: usize,
    pub spatial_merge_size: usize,
    pub projection_dim: usize,
    pub position_embedding_size: usize,
    pub default_output_length: usize,
    pub rope_theta: f32,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
}

impl EmbeddingGemmaVisionConfig {
    pub fn from_metadata<S: TensorSource + ?Sized>(source: &S) -> Result<Self, String> {
        let get = |key: &str| -> Result<usize, String> {
            source
                .metadata(key)
                .and_then(|v| v.to_u64())
                .and_then(|v| v.try_into().ok())
                .ok_or_else(|| format!("Missing metadata: {key}"))
        };
        let depth = get("gemma4-embedding.vision.depth")?;
        let embd = get("gemma4-embedding.vision.embedding_length")?;
        let intermediate_size = get("gemma4-embedding.vision.feed_forward_length")?;
        let head_count = get("gemma4-embedding.vision.attention.head_count")?;
        let patch_size = get("gemma4-embedding.vision.patch_size")?;
        let merge = get("gemma4-embedding.vision.spatial_merge_size")?;
        let projection_dim = get("gemma4-embedding.vision.projection_dim")?;
        let position_embedding_size = get("gemma4-embedding.vision.position_embedding_size")?;
        let default_output_length = get("gemma4-embedding.vision.default_output_length")?;
        let image_mean = [0.5f32, 0.5, 0.5];
        let image_std = [0.5f32, 0.5, 0.5];
        Ok(Self {
            depth,
            embd,
            intermediate_size,
            head_count,
            head_dim: embd / head_count,
            patch_size,
            spatial_merge_size: merge,
            projection_dim,
            position_embedding_size,
            default_output_length,
            rope_theta: 100.0,
            image_mean,
            image_std,
        })
    }
}

fn decode_f32(data: &[u8], expected: usize) -> Result<Vec<f32>, String> {
    let n = data.len() / 4;
    if n != expected {
        return Err(format!(
            "f32 payload length mismatch: {} != {expected}",
            data.len() / 4
        ));
    }
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
    Ok(out)
}

/// 1-D BF16 dequant (for RMSNorm/affine weights and 1-D scalars).
fn dequant_bf16_to_f32(raw: &[u8], n: usize) -> Vec<f32> {
    if raw.len() < n * 2 {
        panic!(
            "dequant_bf16_to_f32: payload {} bytes < {} expected",
            raw.len(),
            n * 2
        );
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let bits = u16::from_le_bytes([raw[i * 2], raw[i * 2 + 1]]);
        out.push(f32::from_bits((bits as u32) << 16));
    }
    out
}

/// Linear weight dequant. GGUF stores as `[n_in, n_out]` row-major;
/// transpose to PyTorch `[n_out, n_in]` for our Linear::forward loop.
fn dequant_bf16_to_f32_linear(raw: &[u8], n_in: usize, n_out: usize) -> Vec<f32> {
    let n = n_in * n_out;
    // The converter preserves the source PyTorch [out, in] payload. GGUF
    // reverses only the dimension metadata, so Linear can consume it directly.
    dequant_bf16_to_f32(raw, n)
}

#[cfg(test)]
mod tests {
    use super::dequant_bf16_to_f32_linear;

    #[test]
    fn linear_bf16_payload_keeps_pytorch_row_order() {
        let values = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let raw: Vec<u8> = values
            .iter()
            .flat_map(|value| half::bf16::from_f32(*value).to_bits().to_le_bytes())
            .collect();
        assert_eq!(dequant_bf16_to_f32_linear(&raw, 3, 2), values);
    }
}

/// One linear projection over the borrowed GGUF bytes.
///
/// The vision tower's tensors are stored with their GGUF dims reversed
/// (`[n_out, n_in]`, see the converter), so the loader takes the semantic
/// shape and lets `Weight` keep the bytes as they are.
struct Linear<'a> {
    weight: Weight<'a>,
}

impl Linear<'_> {
    /// A single row. The device's per-submission fence costs more than one
    /// row of arithmetic, so this stays on the shared kernel.
    fn forward(&self, input: &[f32], output: &mut [f32]) {
        debug_assert_eq!(input.len(), self.weight.n_in, "Linear::forward input shape");
        debug_assert_eq!(
            output.len(),
            self.weight.n_out,
            "Linear::forward output shape"
        );
        self.weight
            .kernel
            .forward_batched(input, output, self.weight.n_in, self.weight.n_out);
    }

    /// A whole row batch: the device first, then the shared kernel with the
    /// rows partitioned over the pool.
    fn forward_rows(&self, pool: &ComputePool, input: &[f32], output: &mut [f32], rows: usize) {
        debug_assert_eq!(input.len(), rows * self.weight.n_in);
        debug_assert_eq!(output.len(), rows * self.weight.n_out);
        #[cfg(feature = "vulkan")]
        if self.weight.try_vulkan_rows(input, output, rows) {
            return;
        }
        let n_in = self.weight.n_in;
        let n_out = self.weight.n_out;
        let input_ptr = input.as_ptr();
        let output_ptr = output.as_mut_ptr();
        pool.compute(move |ith, nth| {
            let (start, end) = crate::ops::kernel::f32::scalar::row_range(rows, ith, nth);
            if start >= end {
                return;
            }
            // SAFETY: the caller checked the lengths; each worker owns a
            // disjoint row range and the kernel writes only its own rows.
            unsafe {
                let input_slice =
                    std::slice::from_raw_parts(input_ptr.add(start * n_in), (end - start) * n_in);
                let output_slice = std::slice::from_raw_parts_mut(
                    output_ptr.add(start * n_out),
                    (end - start) * n_out,
                );
                self.weight
                    .kernel
                    .forward_batched(input_slice, output_slice, n_in, n_out);
            }
        });
    }
}

/// The device-side attention: an arena with the q/k/v and the output.
///
/// The vision tower attends in both directions over every patch, so the
/// runtime is driven with `causal: false` and the unit scale Gemma4 uses.
#[cfg(feature = "vulkan")]
struct DeviceAttention {
    runtime: crate::vulkan::ops::AttentionRuntime,
    rows: usize,
}

#[cfg(feature = "vulkan")]
impl DeviceAttention {
    fn new(config: &EmbeddingGemmaVisionConfig, rows: usize) -> Option<Self> {
        let context = crate::ops::get_vulkan_context()?;
        let heads = config.head_count;
        let width = heads * config.head_dim;
        let runtime =
            crate::vulkan::ops::AttentionRuntime::new(context, (rows * width, rows * width))
                .ok()?;
        Some(Self { runtime, rows })
    }
}

fn linear_from_gguf<'a>(
    source: &'a dyn TensorSource,
    name: &str,
    n_in: usize,
    n_out: usize,
) -> Result<Linear<'a>, String> {
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("Missing tensor: {name}"))?;
    if info.dims != [n_in as u64, n_out as u64] {
        return Err(format!(
            "{name}: shape {:?} != [n_in={n_in}, n_out={n_out}]",
            info.dims
        ));
    }
    let bytes = source
        .tensor_slice(name)
        .ok_or_else(|| format!("Missing tensor: {name}"))?;
    let expected = usize::try_from(info.checked_nbytes().unwrap_or(u64::MAX))
        .map_err(|_| format!("Invalid tensor byte size: {name}"))?;
    if bytes.len() != expected {
        return Err(format!("Invalid tensor data length for {name}"));
    }
    Ok(Linear {
        weight: Weight::from_quantized(QuantizedTensor::from_bytes(
            bytes,
            info.ggml_type,
            n_in,
            n_out,
        )),
    })
}

struct VisionLayer<'a> {
    input_layernorm: Vec<f32>,
    attn_q_w: Linear<'a>,
    attn_k_w: Linear<'a>,
    attn_v_w: Linear<'a>,
    attn_out: Linear<'a>,
    attn_q_norm: Vec<f32>,
    attn_k_norm: Vec<f32>,
    post_attention_layernorm: Vec<f32>,
    pre_ffn_norm: Vec<f32>,
    ffn_gate: Linear<'a>,
    ffn_up: Linear<'a>,
    ffn_down: Linear<'a>,
    post_ffn_norm: Vec<f32>,
}

pub struct EmbeddingGemmaVisionModel<'a> {
    pub config: EmbeddingGemmaVisionConfig,
    pub(super) source: &'a Arc<dyn TensorSource>,
    pub(super) pool: Arc<ComputePool>,
    /// The device attention, built on first use. `None` until then and after
    /// any failure, in which case the layer falls back to the CPU.
    #[cfg(feature = "vulkan")]
    device: std::cell::RefCell<Option<DeviceAttention>>,
    pub(super) patch_input_proj: Linear<'a>,
    /// `vision.position_embedding_table` shape `(2, position_embedding_size,
    /// embd)` — separate (h, w) tables. We read it as a flat F32 vector.
    pub(super) position_embedding_table: Vec<f32>,
    pub(super) layers: Vec<VisionLayer<'a>>,
    pub(super) vision_embed: Linear<'a>,
}

impl<'a> EmbeddingGemmaVisionModel<'a> {
    pub fn from_source(
        source: &'a Arc<dyn TensorSource>,
        n_threads: usize,
    ) -> Result<Self, String> {
        let config = EmbeddingGemmaVisionConfig::from_metadata(source.as_ref())?;
        let pool = Arc::new(ComputePool::new(n_threads.max(1)));

        // patch_input_proj: stored as `[embd, in_channels * patch * patch]`.
        // GGUF stores `[embd, 3*16*16]` (reversed: last dim first).
        let pe_info = source
            .tensor_info("vision.patch_input_proj.weight")
            .ok_or("Missing tensor: vision.patch_input_proj.weight")?;
        if pe_info.dims.len() != 2 {
            return Err(format!(
                "vision.patch_input_proj.weight dims {:?} (expected 2)",
                pe_info.dims
            ));
        }
        let pe_raw = source
            .tensor_slice("vision.patch_input_proj.weight")
            .ok_or("Missing tensor: vision.patch_input_proj.weight")?;
        // Convert GGUF storage [embd, in_features] → [out, in].
        let embd = config.embd;
        let in_features = (config.patch_size * config.patch_size * 3) as u64;
        if pe_info.dims[0] as u64 != embd as u64 || pe_info.dims[1] as u64 != in_features {
            return Err(format!(
                "patch_input_proj dims {:?} != [{}, {}]",
                pe_info.dims, embd, in_features
            ));
        }
        let in_features_usize = in_features as usize;
        let patch_input_proj = Linear {
            weight: Weight::from_quantized(QuantizedTensor::from_bytes(
                pe_raw,
                pe_info.ggml_type,
                in_features_usize,
                embd,
            )),
        };

        // position_embedding_table: shape `(2, position_embedding_size, embd)`.
        // Flatten to a 2D access pattern: rows 0..position_embedding_size are h,
        // rows position_embedding_size..2*position_embedding_size are w.
        let pos_raw = source
            .tensor_slice("vision.position_embedding_table")
            .ok_or("Missing tensor: vision.position_embedding_table")?;
        let expected_pos = 2 * config.position_embedding_size * embd;
        let position_embedding_table = dequant_bf16_to_f32(pos_raw, expected_pos);

        let mut layers = Vec::with_capacity(config.depth);
        for i in 0..config.depth {
            let blk =
                |suffix: &str| -> Result<String, String> { Ok(format!("vision.blk.{i}.{suffix}")) };
            let ln1 = decode_f32(
                source
                    .tensor_slice(&blk("ln1.weight")?)
                    .ok_or_else(|| "Missing ln1.weight".to_string())?,
                embd,
            )?;
            // attn_q/k/v are separate (not fused QKV).
            let attn_q_w = linear_from_gguf(source.as_ref(), &blk("attn_q.weight")?, embd, embd)?;
            let attn_k_w = linear_from_gguf(source.as_ref(), &blk("attn_k.weight")?, embd, embd)?;
            let attn_v_w = linear_from_gguf(source.as_ref(), &blk("attn_v.weight")?, embd, embd)?;
            let out = linear_from_gguf(source.as_ref(), &blk("attn_output.weight")?, embd, embd)?;
            let qn = decode_f32(
                source
                    .tensor_slice(&blk("attn_q_norm.weight")?)
                    .ok_or_else(|| "Missing attn_q_norm.weight".to_string())?,
                config.head_dim,
            )?;
            let kn = decode_f32(
                source
                    .tensor_slice(&blk("attn_k_norm.weight")?)
                    .ok_or_else(|| "Missing attn_k_norm.weight".to_string())?,
                config.head_dim,
            )?;
            let post_attn = decode_f32(
                source
                    .tensor_slice(&blk("post_attention_norm.weight")?)
                    .ok_or_else(|| "Missing post_attention_norm.weight".to_string())?,
                embd,
            )?;
            let pre_ffn = decode_f32(
                source
                    .tensor_slice(&blk("pre_ffn_norm.weight")?)
                    .ok_or_else(|| "Missing pre_ffn_norm.weight".to_string())?,
                embd,
            )?;
            let gate = linear_from_gguf(
                source.as_ref(),
                &blk("ffn_gate.weight")?,
                embd,
                config.intermediate_size,
            )?;
            let up = linear_from_gguf(
                source.as_ref(),
                &blk("ffn_up.weight")?,
                embd,
                config.intermediate_size,
            )?;
            let down = linear_from_gguf(
                source.as_ref(),
                &blk("ffn_down.weight")?,
                config.intermediate_size,
                embd,
            )?;
            let post_ffn = decode_f32(
                source
                    .tensor_slice(&blk("post_ffn_norm.weight")?)
                    .ok_or_else(|| "Missing post_ffn_norm.weight".to_string())?,
                embd,
            )?;

            layers.push(VisionLayer {
                input_layernorm: ln1,
                attn_q_w,
                attn_k_w,
                attn_v_w,
                attn_out: out,
                attn_q_norm: qn,
                attn_k_norm: kn,
                post_attention_layernorm: post_attn,
                pre_ffn_norm: pre_ffn,
                ffn_gate: gate,
                ffn_up: up,
                ffn_down: down,
                post_ffn_norm: post_ffn,
            });
        }

        let text_dim = source
            .metadata("gemma4-embedding.embedding_length")
            .and_then(|value| value.to_u64())
            .and_then(|value| usize::try_from(value).ok())
            .ok_or("Missing metadata: gemma4-embedding.embedding_length")?;
        let vision_embed = linear_from_gguf(
            source.as_ref(),
            "vision_embed.weight",
            config.embd,
            text_dim,
        )?;

        Ok(Self {
            config,
            source,
            pool,
            #[cfg(feature = "vulkan")]
            device: std::cell::RefCell::new(None),
            patch_input_proj,
            position_embedding_table,
            layers,
            vision_embed,
        })
    }

    /// Encode an image to pooled vision-tower tokens (one 768-dim row per
    /// `spatial_merge_size` square), before the multimodal text projector.
    ///
    /// `image_pixels` is `(image_h * patch_size, image_w * patch_size, 3)`
    /// in RGB order. For our test path we accept any shape that's a
    /// multiple of `patch_size` and equal in both dims (the default
    /// produces `default_output_length` merged tokens).
    pub fn encode_tower(
        &self,
        image_pixels: &[f32],
        image_h: usize,
        image_w: usize,
    ) -> Result<Vec<f32>, String> {
        let cfg = &self.config;
        let ps = cfg.patch_size;
        if image_h % ps != 0 || image_w % ps != 0 {
            return Err(format!(
                "image dims ({image_h}, {image_w}) not divisible by patch_size {ps}"
            ));
        }
        if image_pixels.len() != image_h * image_w * 3 {
            return Err(format!(
                "image_pixels len {} != {}",
                image_pixels.len(),
                image_h * image_w * 3
            ));
        }
        let n_patches_y = image_h / ps;
        let n_patches_x = image_w / ps;
        let n_patches = n_patches_y * n_patches_x;
        let in_features = ps * ps * 3;
        let embd = cfg.embd;
        let merge = cfg.spatial_merge_size;
        // After merge: n_tokens = (n_patches_y / merge) * (n_patches_x / merge).
        let merged_y = n_patches_y / merge;
        let merged_x = n_patches_x / merge;
        let n_tokens = merged_y * merged_x;

        // Step 1: patch embed. HF scales RGB values from [0, 1] to [-1, 1]
        // immediately before the learned patch projection.
        //   embed[p] = patch_input_proj @ pixels[p]  (linear in_features → embd)
        // stored as `[embd, in_features]` so it's a matvec.
        let mut patches = vec![0.0f32; n_patches * embd];
        let mut patch_pixels = vec![0.0f32; n_patches * in_features];
        for p in 0..n_patches {
            // Extract patch pixels: a 16x16x3 block starting at row=p/n_patches_x, col=p%n_patches_x.
            let py = p / n_patches_x;
            let px = p % n_patches_x;
            for dy in 0..ps {
                for dx in 0..ps {
                    let img_row = py * ps + dy;
                    let img_col = px * ps + dx;
                    for c in 0..3 {
                        let idx = p * in_features + (dy * ps + dx) * 3 + c;
                        let img_idx = (img_row * image_w + img_col) * 3 + c;
                        patch_pixels[idx] = 2.0 * (image_pixels[img_idx] - 0.5);
                    }
                }
            }
        }
        self.patch_input_proj
            .forward_rows(&self.pool, &patch_pixels, &mut patches, n_patches);

        // Step 2: add 2D position embeddings. `position_embedding_table`
        // stores `[2, position_embedding_size, embd]`. Index 0 is the
        // (h, *) table; index 1 is the (*, w) table. We extract
        // `(py_index, px_index)` for each patch.
        for p in 0..n_patches {
            let py = p / n_patches_x;
            let px = p % n_patches_x;
            if py >= cfg.position_embedding_size || px >= cfg.position_embedding_size {
                return Err("Image exceeds position embedding table".to_string());
            }
            // HF uses position ids in (x, y) order, matching the two table
            // slices in `position_embedding_table`.
            let x_off = px * embd;
            let y_off = cfg.position_embedding_size * embd + py * embd;
            for e in 0..embd {
                patches[p * embd + e] += self.position_embedding_table[x_off + e]
                    + self.position_embedding_table[y_off + e];
            }
        }

        // Step 3: 16 ViT blocks (axial RoPE).
        for il in 0..cfg.depth {
            self.forward_vision_layer(il, &mut patches, n_patches_x, n_patches_y)?;
        }

        // Step 4: spatial pool — HF averages each merge×merge patch block and
        // scales the resulting tower token by sqrt(hidden_size).
        if n_patches_y % merge != 0 || n_patches_x % merge != 0 {
            return Err(format!(
                "patch grid ({n_patches_y}, {n_patches_x}) is not divisible by merge {merge}"
            ));
        }
        let mut pooled = vec![0.0f32; n_tokens * embd];
        for t in 0..n_tokens {
            let my = t / merged_x;
            let mx = t % merged_x;
            for dy in 0..merge {
                for dx in 0..merge {
                    let p = (my * merge + dy) * n_patches_x + (mx * merge + dx);
                    for e in 0..embd {
                        pooled[t * embd + e] += patches[p * embd + e];
                    }
                }
            }
            let scale = (embd as f32).sqrt() / (merge * merge) as f32;
            for e in 0..embd {
                pooled[t * embd + e] *= scale;
            }
        }
        Ok(pooled)
    }

    /// Encode an image to the 512-dim language-space soft tokens consumed by
    /// the EmbeddingGemma multimodal embedder.
    pub fn encode(
        &self,
        image_pixels: &[f32],
        image_h: usize,
        image_w: usize,
    ) -> Result<Vec<f32>, String> {
        let tower = self.encode_tower(image_pixels, image_h, image_w)?;
        let n_tokens = tower.len() / self.config.embd;
        let mut output = vec![0.0f32; n_tokens * self.vision_embed.weight.n_out];
        let mut normalized = tower;
        for row in normalized.chunks_exact_mut(self.config.embd) {
            crate::ops::norm::rms_unit_inplace(row, RMS_NORM_EPS);
        }
        self.vision_embed
            .forward_rows(&self.pool, &normalized, &mut output, n_tokens);
        Ok(output)
    }

    /// Run the bidirectional attention on the device. `false` means the
    /// caller should use the CPU path.
    #[cfg(feature = "vulkan")]
    fn device_attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        rows: usize,
        attn_out: &mut [f32],
    ) -> bool {
        let spec = crate::ops::nn::AttentionSpec {
            query_tokens: rows,
            key_tokens: rows,
            query_heads: self.config.head_count,
            key_value_heads: self.config.head_count,
            head_dim: self.config.head_dim,
            causal: false,
            scale: 1.0,
        };
        let mut cell = self.device.borrow_mut();
        if cell.as_ref().is_none_or(|device| device.rows != rows) {
            *cell = DeviceAttention::new(&self.config, rows);
        }
        match cell.as_mut() {
            Some(device) => device.runtime.forward(q, k, v, attn_out, spec).is_ok(),
            None => false,
        }
    }

    /// The CPU attention: one task per (query patch, head), each worker with
    /// its own score buffer.
    fn cpu_attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        n_patches: usize,
        attn_out: &mut [f32],
    ) {
        let embd = self.config.embd;
        let n_head = self.config.head_count;
        let head_dim = self.config.head_dim;
        let tasks = n_patches * n_head;
        let q_ptr = q.as_ptr();
        let k_ptr = k.as_ptr();
        let v_ptr = v.as_ptr();
        let out_ptr = attn_out.as_mut_ptr();
        self.pool.compute(move |ith, nth| {
            let (start, end) = crate::ops::kernel::f32::scalar::row_range(tasks, ith, nth);
            if start >= end {
                return;
            }
            let mut scores = vec![0.0f32; n_patches];
            for task in start..end {
                let i = task / n_head;
                let head_off = (task % n_head) * head_dim;
                // SAFETY: the pointers come from slices the caller checked; each
                // task reads only its own head column and writes only its own
                // output rows.
                unsafe {
                    let q_i = std::slice::from_raw_parts(q_ptr.add(i * embd + head_off), head_dim);
                    for j in 0..n_patches {
                        let k_j =
                            std::slice::from_raw_parts(k_ptr.add(j * embd + head_off), head_dim);
                        scores[j] = crate::ops::dot_f32(q_i, k_j, head_dim);
                    }
                    crate::ops::softmax::softmax_inplace(&mut scores);
                    let out_i =
                        std::slice::from_raw_parts_mut(out_ptr.add(i * embd + head_off), head_dim);
                    // The reduction below accumulates, so the slice starts at
                    // zero: the device path may have written it before failing.
                    for value in out_i.iter_mut() {
                        *value = 0.0;
                    }
                    for j in 0..n_patches {
                        let v_j =
                            std::slice::from_raw_parts(v_ptr.add(j * embd + head_off), head_dim);
                        for d in 0..head_dim {
                            out_i[d] += scores[j] * v_j[d];
                        }
                    }
                }
            }
        });
    }

    fn forward_vision_layer(
        &self,
        il: usize,
        x: &mut [f32],
        n_patches_x: usize,
        n_patches_y: usize,
    ) -> Result<(), String> {
        let cfg = &self.config;
        let embd = cfg.embd;
        let n_patches = n_patches_x * n_patches_y;
        let layer = &self.layers[il];
        let head_dim = cfg.head_dim;
        let n_head = cfg.head_count;
        let rope_freq = cfg.rope_theta;

        // Step a: normed = rms_norm(input_layernorm, x).
        let mut normed = x.to_vec();
        for row in normed.chunks_exact_mut(embd) {
            crate::ops::norm::rms_norm_inplace(row, &layer.input_layernorm, RMS_NORM_EPS);
        }

        // Step b: separate Q, K, V projections.
        let mut q = vec![0.0f32; n_patches * embd];
        let mut k = vec![0.0f32; n_patches * embd];
        let mut v = vec![0.0f32; n_patches * embd];
        layer
            .attn_q_w
            .forward_rows(&self.pool, &normed, &mut q, n_patches);
        layer
            .attn_k_w
            .forward_rows(&self.pool, &normed, &mut k, n_patches);
        layer
            .attn_v_w
            .forward_rows(&self.pool, &normed, &mut v, n_patches);

        // Step c: per-head Q/K RMSNorm + axial RoPE.
        // HF computes head_dim/4 frequencies and repeats each axis' frequency
        // table twice into the [X, X, Y, Y] rotary layout.
        let half = head_dim / 2;
        let quarter = head_dim / 4;
        let spatial_dim = head_dim / 2;
        let mut cos_h = vec![0.0f32; n_patches * half];
        let mut sin_h = vec![0.0f32; n_patches * half];
        let mut cos_w = vec![0.0f32; n_patches * half];
        let mut sin_w = vec![0.0f32; n_patches * half];
        for p in 0..n_patches {
            let py = p / n_patches_x;
            let px = p % n_patches_x;
            for j in 0..half {
                let freq = 1.0 / rope_freq.powf(2.0 * (j % quarter) as f32 / spatial_dim as f32);
                let angle_h = px as f32 * freq;
                let angle_w = py as f32 * freq;
                cos_h[p * half + j] = angle_h.cos();
                sin_h[p * half + j] = angle_h.sin();
                cos_w[p * half + j] = angle_w.cos();
                sin_w[p * half + j] = angle_w.sin();
            }
        }
        // Apply RMSNorm + axial RoPE.
        for p in 0..n_patches {
            for h in 0..n_head {
                let off = p * embd + h * head_dim;
                // Q
                crate::ops::norm::rms_norm_inplace(
                    &mut q[off..off + head_dim],
                    &layer.attn_q_norm,
                    RMS_NORM_EPS,
                );
                apply_axial_rope(
                    &mut q[off..off + head_dim],
                    head_dim,
                    &cos_h[p * half..(p + 1) * half],
                    &sin_h[p * half..(p + 1) * half],
                    &cos_w[p * half..(p + 1) * half],
                    &sin_w[p * half..(p + 1) * half],
                );
                // K
                crate::ops::norm::rms_norm_inplace(
                    &mut k[off..off + head_dim],
                    &layer.attn_k_norm,
                    RMS_NORM_EPS,
                );
                apply_axial_rope(
                    &mut k[off..off + head_dim],
                    head_dim,
                    &cos_h[p * half..(p + 1) * half],
                    &sin_h[p * half..(p + 1) * half],
                    &cos_w[p * half..(p + 1) * half],
                    &sin_w[p * half..(p + 1) * half],
                );
                crate::ops::norm::rms_unit_inplace(&mut v[off..off + head_dim], RMS_NORM_EPS);
            }
        }

        // Step d: full bidirectional attention (no mask). Gemma4 vision uses
        // a unit attention scale; the head dimension is already normalized by
        // the trained projections.
        let mut attn_out = vec![0.0f32; n_patches * embd];
        #[cfg(feature = "vulkan")]
        if !self.device_attention(&q, &k, &v, n_patches, &mut attn_out) {
            self.cpu_attention(&q, &k, &v, n_patches, &mut attn_out);
        }
        #[cfg(not(feature = "vulkan"))]
        self.cpu_attention(&q, &k, &v, n_patches, &mut attn_out);

        // Step e: output projection.
        let mut proj_out = vec![0.0f32; n_patches * embd];
        layer
            .attn_out
            .forward_rows(&self.pool, &attn_out, &mut proj_out, n_patches);

        // Step f: residual + post_attention_norm.
        for p in 0..n_patches {
            let mut tmp = proj_out[p * embd..(p + 1) * embd].to_vec();
            crate::ops::norm::rms_norm_inplace(
                &mut tmp,
                &layer.post_attention_layernorm,
                RMS_NORM_EPS,
            );
            for e in 0..embd {
                x[p * embd + e] += tmp[e];
            }
        }

        // Step g: pre_ffn_norm.
        let mut ff_x = x.to_vec();
        for row in ff_x.chunks_exact_mut(embd) {
            crate::ops::norm::rms_norm_inplace(row, &layer.pre_ffn_norm, RMS_NORM_EPS);
        }

        // Step h: SwiGLU FFN.
        let mut gate_buf = vec![0.0f32; n_patches * cfg.intermediate_size];
        let mut up_buf = vec![0.0f32; n_patches * cfg.intermediate_size];
        let mut ffn_out = vec![0.0f32; n_patches * embd];
        layer
            .ffn_gate
            .forward_rows(&self.pool, &ff_x, &mut gate_buf, n_patches);
        layer
            .ffn_up
            .forward_rows(&self.pool, &ff_x, &mut up_buf, n_patches);
        for j in 0..gate_buf.len() {
            gate_buf[j] = crate::ops::activation::gelu(gate_buf[j]) * up_buf[j];
        }
        layer
            .ffn_down
            .forward_rows(&self.pool, &gate_buf, &mut ffn_out, n_patches);

        // Step i: post_ffn_norm + residual.
        for p in 0..n_patches {
            let mut tmp = ffn_out[p * embd..(p + 1) * embd].to_vec();
            crate::ops::norm::rms_norm_inplace(&mut tmp, &layer.post_ffn_norm, RMS_NORM_EPS);
            for e in 0..embd {
                x[p * embd + e] += tmp[e];
            }
        }
        Ok(())
    }
}

/// Apply axial RoPE in-place using a precomputed cos/sin table for the
/// height and width axes. Splits head_dim in half: first half rotated by
/// (cos_h, sin_h), second half by (cos_w, sin_w).
fn apply_axial_rope(
    x: &mut [f32],
    head_dim: usize,
    cos_h: &[f32],
    sin_h: &[f32],
    cos_w: &[f32],
    sin_w: &[f32],
) {
    let axis_dim = head_dim / 2;
    let pair_dim = axis_dim / 2;
    debug_assert_eq!(cos_h.len(), axis_dim);
    debug_assert_eq!(sin_h.len(), axis_dim);
    debug_assert_eq!(cos_w.len(), axis_dim);
    debug_assert_eq!(sin_w.len(), axis_dim);
    // HF splits each head into two 32-wide axis chunks, then applies the
    // usual rotate_half within each chunk.  The frequency table is repeated
    // twice in each chunk ([H,H,W,W]), so pairs are separated by 16 here.
    for i in 0..pair_dim {
        let h0 = x[i];
        let h1 = x[i + pair_dim];
        x[i] = h0 * cos_h[i] - h1 * sin_h[i];
        x[i + pair_dim] = h1 * cos_h[i + pair_dim] + h0 * sin_h[i + pair_dim];

        let w0 = x[axis_dim + i];
        let w1 = x[axis_dim + i + pair_dim];
        x[axis_dim + i] = w0 * cos_w[i] - w1 * sin_w[i];
        x[axis_dim + i + pair_dim] = w1 * cos_w[i + pair_dim] + w0 * sin_w[i + pair_dim];
    }
}
