//! ERNIE-Image / ERNIE-Image-Turbo single-stream DiT port.
//!
//! Reference: `references/stable-diffusion.cpp/src/model/diffusion/ernie_image.hpp`
//! (commit `de298c2`). The DiT layer reuses the [Z-Image](super::z_image)
//! numerical kernel layout (Q8_0 row matmul, F16 fallback, NEON/AVX2 vectorized
//! attention reduction) and replaces Z-Image's per-block AdaLN modulation with
//! the shared-AdaLN scheme that ERNIE-Image uses. The text encoder path
//! reuses the llama trunk `forward_to_block` to extract Ministral-3 hidden
//! states; the cross-modal `text_proj` lives inside this module.
//!
//! Public surface:
//! - [`ErnieImagePipeline`] — high-level `load(text, dit, vae) -> generate_rgb(prompt, opts)`
//! - [`ErnieImageOptions`] — steps / resolution / seed
//! - [`ErnieImageRgb`] — decoded output bytes
//! - [`ErnieImageDit`] — DiT model itself (loaded separately so tests can pin
//!   the contract without the full pipeline)

use crate::core::tensor::{GGMLType, TensorSource};
use crate::core::thread_pool::ComputePool;
use crate::ops::kernel::f16::F16Kernel;
use crate::ops::matmul_q8_0_quantized_parallel_rows;
use std::sync::Arc;

pub(crate) mod dit;
pub(crate) mod text;

pub struct ErnieImageRgb {
    pub width: u32,
    pub height: u32,
    pub bytes: Vec<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ErnieImageOptions {
    pub(crate) steps: usize,
    pub(crate) resolution: usize,
    pub(crate) seed: i64,
}

pub(crate) struct ErnieImagePipeline {
    dit: dit::ErnieImageDit,
    text: text::ErnieImageTextEncoder,
    vae: super::z_image::vae::FluxVae,
}

impl ErnieImagePipeline {
    pub(crate) fn load(
        diffusion: Arc<dyn TensorSource>,
        text: Arc<dyn TensorSource>,
        vae: Arc<dyn TensorSource>,
        n_threads: usize,
    ) -> Result<Self, String> {
        validate_component(diffusion.as_ref(), Component::Dit)?;
        validate_component(text.as_ref(), Component::Text)?;
        let pool = Arc::new(ComputePool::new(n_threads.max(1)));
        Ok(Self {
            dit: dit::ErnieImageDit::load(diffusion, Arc::clone(&pool))?,
            text: text::ErnieImageTextEncoder::load(text, Arc::clone(&pool))?,
            vae: super::z_image::vae::FluxVae::load(vae, pool)?,
        })
    }

    pub(crate) fn generate_rgb(
        &self,
        prompt: &str,
        options: &ErnieImageOptions,
    ) -> Result<ErnieImageRgb, String> {
        validate_generate_request(prompt, options)?;
        let total_start = std::time::Instant::now();
        let t = std::time::Instant::now();
        let context = self.text.encode(prompt)?;
        let context_tokens = context_token_count(&context)?;
        let t_text = t.elapsed();
        let t = std::time::Instant::now();
        let latent = self.dit.denoise(&context, context_tokens, options)?;
        let t_denoise = t.elapsed();
        drop(context);
        let t = std::time::Instant::now();
        let latent_side = validate_latent_shape(&latent, options.resolution)?;
        let rgb = self.vae.decode_rgb(&latent, latent_side)?;
        let t_vae = t.elapsed();
        let rgb = ErnieImageRgb {
            width: rgb.width,
            height: rgb.height,
            bytes: rgb.bytes,
        };
        validate_decoded_rgb(&rgb, options.resolution)?;
        eprintln!(
            "[ernie-image-stage-profile] text_encode={:.1}ms  denoise={:.1}ms  vae_decode={:.1}ms  total={:.1}ms",
            t_text.as_secs_f64() * 1000.0,
            t_denoise.as_secs_f64() * 1000.0,
            t_vae.as_secs_f64() * 1000.0,
            total_start.elapsed().as_secs_f64() * 1000.0,
        );
        Ok(rgb)
    }
}

fn context_token_count(context: &[f32]) -> Result<usize, String> {
    const WIDTH: usize = dit::TEXT_IN_DIM;
    if context.is_empty() || context.len() % WIDTH != 0 {
        return Err(format!(
            "Invalid ERNIE-Image context length: expected non-empty rows of {WIDTH}, got {}",
            context.len()
        ));
    }
    if !context.iter().all(|value| value.is_finite()) {
        return Err("Non-finite ERNIE-Image context".into());
    }
    Ok(context.len() / WIDTH)
}

fn validate_generate_request(prompt: &str, options: &ErnieImageOptions) -> Result<(), String> {
    if prompt.trim().is_empty() {
        return Err("ERNIE-Image prompt must not be empty".into());
    }
    if options.steps == 0 || options.resolution == 0 || options.resolution % 16 != 0 {
        return Err("ERNIE-Image requires positive steps and a resolution divisible by 16".into());
    }
    Ok(())
}

fn validate_latent_shape(latent: &[f32], resolution: usize) -> Result<usize, String> {
    let latent_side = resolution / 8;
    let expected = latent_side
        .checked_mul(latent_side)
        .and_then(|spatial| spatial.checked_mul(dit::LATENT_CHANNELS))
        .ok_or("ERNIE-Image latent shape overflow")?;
    if latent_side == 0 || latent.len() != expected {
        return Err(format!(
            "Invalid ERNIE-Image denoised latent length: expected {expected}, got {}",
            latent.len()
        ));
    }
    if !latent.iter().all(|value| value.is_finite()) {
        return Err("Non-finite ERNIE-Image denoised latent".into());
    }
    Ok(latent_side)
}

fn validate_decoded_rgb(rgb: &ErnieImageRgb, resolution: usize) -> Result<(), String> {
    let resolution = u32::try_from(resolution)
        .map_err(|_| "ERNIE-Image output resolution does not fit u32")?;
    if rgb.width != resolution || rgb.height != resolution {
        return Err("Invalid ERNIE-Image decoded RGB dimensions".into());
    }
    let expected = usize::try_from(rgb.width)
        .ok()
        .and_then(|width| {
            usize::try_from(rgb.height)
                .ok()
                .and_then(|height| width.checked_mul(height))
        })
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or("ERNIE-Image decoded RGB size overflow")?;
    if rgb.bytes.len() != expected {
        return Err(format!(
            "Invalid ERNIE-Image decoded RGB length: expected {expected}, got {}",
            rgb.bytes.len()
        ));
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Component {
    Text,
    Dit,
}

pub(crate) fn validate_component(
    source: &dyn TensorSource,
    component: Component,
) -> Result<(), String> {
    match component {
        Component::Text => validate_text(source),
        Component::Dit => validate_dit(source),
    }
}

fn require_tensor(
    source: &dyn TensorSource,
    name: &str,
    dims: &[u64],
    ggml_type: GGMLType,
) -> Result<(), String> {
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("Missing tensor: {name}"))?;
    if info.dims != dims {
        return Err(format!("Invalid {name} dimensions"));
    }
    if info.ggml_type != ggml_type {
        return Err(format!(
            "Invalid {name} type: expected {ggml_type:?}, got {:?}",
            info.ggml_type
        ));
    }
    let expected = usize::try_from(
        info.checked_nbytes()
            .ok_or_else(|| format!("Invalid {name} byte size"))?,
    )
    .map_err(|_| format!("Invalid {name} byte size"))?;
    let bytes = source
        .tensor_slice(name)
        .ok_or_else(|| format!("Missing tensor data: {name}"))?;
    if bytes.len() != expected {
        return Err(format!("Invalid {name} byte length"));
    }
    Ok(())
}

/// A 2-D projection the reader can execute: either the quantized kernel or the
/// F16 one. The dispatch happens in [`linear_into_scaled_impl`], so the loader
/// should not pin one dtype here -- F16 additionally gives an unquantized
/// model, which is the whole point of exporting one.
fn require_matrix(source: &dyn TensorSource, name: &str, dims: &[u64]) -> Result<(), String> {
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("Missing tensor: {name}"))?;
    if info.dims != dims {
        return Err(format!("Invalid {name} dimensions"));
    }
    if !matches!(info.ggml_type, GGMLType::F16 | GGMLType::Q8_0) {
        return Err(format!(
            "Invalid {name} type: expected F16 or Q8_0, got {:?}",
            info.ggml_type
        ));
    }
    Ok(())
}

fn validate_text(source: &dyn TensorSource) -> Result<(), String> {
    let hidden = dit::TEXT_IN_DIM as u64;
    let n_head = dit::TEXT_NUM_HEADS as u64;
    let n_head_kv = dit::TEXT_NUM_KV_HEADS as u64;
    let head_dim = dit::TEXT_HEAD_DIM as u64;
    let ffn = dit::TEXT_FFN as u64;
    let n_layer = dit::TEXT_NUM_LAYERS;
    require_matrix(source, "model.embed_tokens.weight", &[hidden, dit::TEXT_VOCAB as u64])?;
    for layer in 0..n_layer {
        let prefix = format!("model.layers.{layer}");
        for (suffix, dims) in [
            ("mlp.down_proj.weight", [ffn, hidden]),
            ("mlp.gate_proj.weight", [hidden, ffn]),
            ("mlp.up_proj.weight", [hidden, ffn]),
            ("self_attn.k_proj.weight", [hidden, n_head_kv * head_dim]),
            ("self_attn.o_proj.weight", [n_head * head_dim, hidden]),
            ("self_attn.q_proj.weight", [hidden, n_head * head_dim]),
            ("self_attn.v_proj.weight", [hidden, n_head_kv * head_dim]),
        ] {
            require_matrix(source, &format!("{prefix}.{suffix}"), &dims)?;
        }
        for (suffix, dims) in [
            ("input_layernorm.weight", hidden),
            ("post_attention_layernorm.weight", hidden),
            ("self_attn.k_norm.weight", head_dim),
            ("self_attn.q_norm.weight", head_dim),
        ] {
            require_tensor(
                source,
                &format!("{prefix}.{suffix}"),
                &[dims],
                GGMLType::F32,
            )?;
        }
    }
    require_tensor(source, "model.norm.weight", &[hidden], GGMLType::F32)?;
    Ok(())
}

fn validate_dit(source: &dyn TensorSource) -> Result<(), String> {
    let hidden = dit::HIDDEN as u64;
    let inner = dit::INNER_DIM as u64;
    let ffn = dit::FFN_WIDTH as u64;
    let patch = dit::PATCH_AREA as u64;
    let text_in = dit::TEXT_IN_DIM as u64;
    for (name, dims) in [
        ("adaLN_modulation.1.bias", hidden),
        ("final_norm.linear.bias", 2 * hidden),
        ("final_norm.norm.weight", hidden),
        ("final_linear.bias", dit::OUT_CHANNELS as u64 * patch),
        ("time_embedding.linear_1.bias", hidden),
        ("time_embedding.linear_2.bias", hidden),
        ("x_embedder.proj.bias", hidden),
    ] {
        require_tensor(source, name, &[dims], GGMLType::F32)?;
    }
    for (name, dims) in [
        ("adaLN_modulation.1.weight", [hidden, 6 * hidden]),
        ("final_norm.linear.weight", [hidden, 2 * hidden]),
        ("final_linear.weight", [hidden, dit::OUT_CHANNELS as u64 * patch]),
        ("time_embedding.linear_1.weight", [hidden, hidden]),
        ("time_embedding.linear_2.weight", [hidden, hidden]),
        ("x_embedder.proj.weight", [dit::IN_CHANNELS as u64 * patch, hidden]),
    ] {
        require_tensor(source, name, &dims, GGMLType::F16)?;
    }
    if source.tensor_info("text_proj.weight").is_some() {
        require_matrix(source, "text_proj.weight", &[text_in, hidden])?;
    }
    for layer in 0..dit::NUM_LAYERS {
        let prefix = format!("layers.{layer}");
        require_tensor(
            source,
            &format!("{prefix}.adaLN_sa_ln.weight"),
            &[hidden],
            GGMLType::F32,
        )?;
        require_tensor(
            source,
            &format!("{prefix}.adaLN_mlp_ln.weight"),
            &[hidden],
            GGMLType::F32,
        )?;
        require_tensor(
            source,
            &format!("{prefix}.self_attention.norm_q.weight"),
            &[dit::ROPE_HEAD_WIDTH as u64],
            GGMLType::F32,
        )?;
        require_tensor(
            source,
            &format!("{prefix}.self_attention.norm_k.weight"),
            &[dit::ROPE_HEAD_WIDTH as u64],
            GGMLType::F32,
        )?;
        for (suffix, dims) in [
            ("self_attention.to_q.weight", [hidden, inner]),
            ("self_attention.to_k.weight", [hidden, inner]),
            ("self_attention.to_v.weight", [hidden, inner]),
            ("self_attention.to_out.0.weight", [inner, hidden]),
            ("mlp.gate_proj.weight", [hidden, ffn]),
            ("mlp.up_proj.weight", [hidden, ffn]),
            ("mlp.linear_fc2.weight", [ffn, hidden]),
        ] {
            // The reader dispatches on the stored dtype, so both the quantized
            // kernel and the F16 kernel are valid here. Accept either.
            require_matrix(source, &format!("{prefix}.{suffix}"), &dims)?;
        }
    }
    Ok(())
}

pub(crate) struct Q8Scratch {
    scaled: Vec<f32>,
    force_f32_row: Vec<f32>,
    f16_input: Vec<u16>,
    /// Per-thread F16 staging buffers, indexed by `ith`. `forward_scaled_rows`
    /// converts the input row into f16 itself, and the pool runs the rows
    /// concurrently, so each worker needs its own buffer -- but they are
    /// allocated once here and reused across every call, the way `f16_input`
    /// is, rather than a fresh `Vec` per worker per matmul.
    f16_inputs: Vec<Vec<u16>>,
    values: Vec<u8>,
    scales: Vec<f32>,
}

impl Q8Scratch {
    pub(crate) fn new(n_in: usize) -> Self {
        Self {
            scaled: Vec::new(),
            force_f32_row: Vec::new(),
            f16_input: Vec::new(),
            f16_inputs: Vec::new(),
            values: vec![0; n_in],
            scales: vec![0.0; n_in.div_ceil(32)],
        }
    }

    fn prepare(&mut self, input: &[f32], n_in: usize) -> Result<(), String> {
        if input.len() != n_in {
            return Err("Invalid linear input length".into());
        }
        self.values.resize(n_in, 0);
        self.scales.resize(n_in.div_ceil(32), 0.0);
        crate::ops::quantize_q8_0_into(input, n_in, &mut self.values, &mut self.scales);
        Ok(())
    }

    fn prepare_scaled(&mut self, input: &[f32], n_in: usize, scale: f32) -> Result<(), String> {
        if input.len() != n_in {
            return Err("Invalid linear input length".into());
        }
        self.scaled.resize(n_in, 0.0);
        for (scaled, &value) in self.scaled.iter_mut().zip(input) {
            *scaled = value * scale;
        }
        self.values.resize(n_in, 0);
        self.scales.resize(n_in.div_ceil(32), 0.0);
        crate::ops::quantize_q8_0_into(&self.scaled, n_in, &mut self.values, &mut self.scales);
        Ok(())
    }
}

pub(crate) fn linear_into(
    source: &dyn TensorSource,
    name: &str,
    n_in: usize,
    n_out: usize,
    input: &[f32],
    output: &mut [f32],
    q8: &mut Q8Scratch,
    pool: &ComputePool,
) -> Result<(), String> {
    linear_into_scaled_impl(source, name, n_in, n_out, input, output, q8, pool, 1.0)
}

#[allow(clippy::too_many_arguments)]
fn linear_into_scaled_impl(
    source: &dyn TensorSource,
    name: &str,
    n_in: usize,
    n_out: usize,
    input: &[f32],
    output: &mut [f32],
    q8: &mut Q8Scratch,
    pool: &ComputePool,
    scale: f32,
) -> Result<(), String> {
    if input.len() != n_in {
        return Err("Invalid linear input length".into());
    }
    if output.len() != n_out {
        return Err("Invalid linear output length".into());
    }
    n_in.checked_mul(n_out)
        .ok_or_else(|| format!("Invalid {name} dimensions"))?;
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("Missing tensor: {name}"))?;
    if info.dims != [n_in as u64, n_out as u64] {
        return Err(format!("Invalid {name} dimensions"));
    }
    if !matches!(info.ggml_type, GGMLType::F16 | GGMLType::Q8_0) {
        return Err(format!(
            "Unsupported matrix type {:?} for {name}",
            info.ggml_type
        ));
    }
    let expected = usize::try_from(
        info.checked_nbytes()
            .ok_or_else(|| format!("Invalid {name} byte size"))?,
    )
    .map_err(|_| format!("Invalid {name} byte size"))?;
    let bytes = source
        .tensor_slice(name)
        .ok_or_else(|| format!("Missing tensor data: {name}"))?;
    if bytes.len() != expected {
        return Err(format!("Invalid {name} byte length"));
    }
    match info.ggml_type {
        GGMLType::F16 => {
            let weight_ptr = bytes.as_ptr() as usize;
            let weight_len = bytes.len();
            let input_ptr = input.as_ptr() as usize;
            let input_len = input.len();
            let output_ptr = output.as_mut_ptr() as usize;
            let output_len = output.len();
            let scale_copy = scale;
            let threads = pool.n_threads();
            if q8.f16_inputs.len() < threads {
                q8.f16_inputs.resize_with(threads, Vec::new);
            }
            let staging = &q8.f16_inputs[..threads];
            let staging_ptr = staging.as_ptr() as usize;
            pool.compute(move |ith, nth| {
                let weight =
                    unsafe { std::slice::from_raw_parts(weight_ptr as *const u8, weight_len) };
                let values =
                    unsafe { std::slice::from_raw_parts(input_ptr as *const f32, input_len) };
                let out =
                    unsafe { std::slice::from_raw_parts_mut(output_ptr as *mut f32, output_len) };
                let buffer = unsafe { &mut *((staging_ptr as *mut Vec<u16>).add(ith)) };
                F16Kernel::new(weight)
                    .forward_scaled_rows(values, out, n_in, n_out, scale_copy, buffer, ith, nth);
            });
        }
        GGMLType::Q8_0 => {
            if scale == 1.0 {
                q8.prepare(input, n_in)?;
            } else {
                q8.prepare_scaled(input, n_in, scale)?;
            }
            let weight_ptr = bytes.as_ptr() as usize;
            let weight_len = bytes.len();
            let input_ptr = q8.values.as_ptr() as usize;
            let input_len = q8.values.len();
            let scale_ptr = q8.scales.as_ptr() as usize;
            let scale_len = q8.scales.len();
            let output_ptr = output.as_mut_ptr() as usize;
            pool.compute(move |ith, nth| {
                let weight =
                    unsafe { std::slice::from_raw_parts(weight_ptr as *const u8, weight_len) };
                let input =
                    unsafe { std::slice::from_raw_parts(input_ptr as *const u8, input_len) };
                let scales =
                    unsafe { std::slice::from_raw_parts(scale_ptr as *const f32, scale_len) };
                let out =
                    unsafe { std::slice::from_raw_parts_mut(output_ptr as *mut f32, n_out) };
                matmul_q8_0_quantized_parallel_rows(
                    weight, input, scales, out, n_in, n_out, ith, nth,
                );
            });
        }
        _ => unreachable!(),
    }
    Ok(())
}
