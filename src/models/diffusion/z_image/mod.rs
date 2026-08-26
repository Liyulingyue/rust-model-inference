use crate::core::tensor::{GGMLType, TensorSource};
use crate::core::thread_pool::ComputePool;
use crate::ops::kernel::f16::F16Kernel;
use std::sync::Arc;

pub(crate) mod dit;
pub(crate) mod text;
pub(crate) mod vae;

pub struct ZImageRgb {
    pub width: u32,
    pub height: u32,
    pub bytes: Vec<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ZImageOptions {
    pub(crate) steps: usize,
    pub(crate) resolution: usize,
    pub(crate) seed: i64,
}

pub(crate) struct ZImagePipeline {
    dit: dit::ZImageDit,
    text: text::Qwen3TextEncoder,
    vae: vae::FluxVae,
}

impl ZImagePipeline {
    pub(crate) fn load(
        diffusion: Arc<dyn TensorSource>,
        text: Arc<dyn TensorSource>,
        vae: Arc<dyn TensorSource>,
        n_threads: usize,
    ) -> Result<Self, String> {
        validate_component(diffusion.as_ref(), Component::Dit)?;
        validate_component(text.as_ref(), Component::Text)?;
        validate_component(vae.as_ref(), Component::Vae)?;
        let pool = Arc::new(ComputePool::new(n_threads.max(1)));
        Ok(Self {
            dit: dit::ZImageDit::load(diffusion, Arc::clone(&pool))?,
            text: text::Qwen3TextEncoder::load(text, pool)?,
            vae: vae::FluxVae::load(vae)?,
        })
    }

    pub(crate) fn generate_rgb(
        &self,
        prompt: &str,
        options: &ZImageOptions,
    ) -> Result<ZImageRgb, String> {
        validate_generate_request(prompt, options)?;
        let context = self.text.encode_layer_35(prompt)?;
        let context_tokens = context_token_count(&context)?;
        let latent = self.dit.denoise(&context, context_tokens, options)?;
        drop(context);
        let latent_side = validate_latent_shape(&latent, options.resolution)?;
        let rgb = self.vae.decode_rgb(&latent, latent_side)?;
        validate_decoded_rgb(&rgb, options.resolution)?;
        Ok(rgb)
    }
}

fn validate_generate_request(prompt: &str, options: &ZImageOptions) -> Result<(), String> {
    if prompt.trim().is_empty() {
        return Err("Z-Image prompt must not be empty".into());
    }
    if options.steps == 0 || options.resolution == 0 || options.resolution % 16 != 0 {
        return Err("Z-Image requires positive steps and a resolution divisible by 16".into());
    }
    Ok(())
}

fn context_token_count(context: &[f32]) -> Result<usize, String> {
    const WIDTH: usize = 2_560;
    if context.is_empty() || context.len() % WIDTH != 0 {
        return Err(format!(
            "Invalid Z-Image context length: expected non-empty rows of {WIDTH}, got {}",
            context.len()
        ));
    }
    if !context.iter().all(|value| value.is_finite()) {
        return Err("Non-finite Z-Image context".into());
    }
    Ok(context.len() / WIDTH)
}

fn validate_latent_shape(latent: &[f32], resolution: usize) -> Result<usize, String> {
    let latent_side = resolution / 8;
    let expected = latent_side
        .checked_mul(latent_side)
        .and_then(|spatial| spatial.checked_mul(16))
        .ok_or("Z-Image latent shape overflow")?;
    if latent_side == 0 || latent.len() != expected {
        return Err(format!(
            "Invalid Z-Image denoised latent length: expected {expected}, got {}",
            latent.len()
        ));
    }
    if !latent.iter().all(|value| value.is_finite()) {
        return Err("Non-finite Z-Image denoised latent".into());
    }
    Ok(latent_side)
}

fn validate_decoded_rgb(rgb: &ZImageRgb, resolution: usize) -> Result<(), String> {
    let resolution =
        u32::try_from(resolution).map_err(|_| "Z-Image output resolution does not fit u32")?;
    if rgb.width != resolution || rgb.height != resolution {
        return Err("Invalid Z-Image decoded RGB dimensions".into());
    }
    let expected = usize::try_from(rgb.width)
        .ok()
        .and_then(|width| {
            usize::try_from(rgb.height)
                .ok()
                .and_then(|height| width.checked_mul(height))
        })
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or("Z-Image decoded RGB size overflow")?;
    if rgb.bytes.len() != expected {
        return Err(format!(
            "Invalid Z-Image decoded RGB length: expected {expected}, got {}",
            rgb.bytes.len()
        ));
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Component {
    Text,
    Dit,
    Vae,
}

pub(crate) fn validate_component(
    source: &dyn TensorSource,
    component: Component,
) -> Result<(), String> {
    match component {
        Component::Text => validate_text(source),
        Component::Dit => validate_dit(source),
        Component::Vae => validate_vae(source),
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

fn validate_text(source: &dyn TensorSource) -> Result<(), String> {
    require_tensor(
        source,
        "model.embed_tokens.weight",
        &[2560, 151936],
        GGMLType::Q8_0,
    )?;
    for layer in 0..36 {
        let prefix = format!("model.layers.{layer}");
        for (suffix, dims) in [
            ("mlp.down_proj.weight", [9728, 2560]),
            ("mlp.gate_proj.weight", [2560, 9728]),
            ("mlp.up_proj.weight", [2560, 9728]),
            ("self_attn.k_proj.weight", [2560, 1024]),
            ("self_attn.o_proj.weight", [4096, 2560]),
            ("self_attn.q_proj.weight", [2560, 4096]),
            ("self_attn.v_proj.weight", [2560, 1024]),
        ] {
            require_tensor(source, &format!("{prefix}.{suffix}"), &dims, GGMLType::Q8_0)?;
        }
        for (suffix, dims) in [
            ("input_layernorm.weight", 2560),
            ("post_attention_layernorm.weight", 2560),
            ("self_attn.k_norm.weight", 128),
            ("self_attn.q_norm.weight", 128),
        ] {
            require_tensor(
                source,
                &format!("{prefix}.{suffix}"),
                &[dims],
                GGMLType::F32,
            )?;
        }
    }
    require_tensor(source, "model.norm.weight", &[2560], GGMLType::F32)
}

fn validate_dit(source: &dyn TensorSource) -> Result<(), String> {
    for (name, dims) in [
        ("cap_embedder.0.weight", 2560),
        ("cap_embedder.1.bias", 3840),
        ("final_layer.adaLN_modulation.1.bias", 3840),
        ("final_layer.linear.bias", 64),
        ("t_embedder.mlp.0.bias", 1024),
        ("t_embedder.mlp.2.bias", 256),
        ("x_embedder.bias", 3840),
    ] {
        require_tensor(source, name, &[dims], GGMLType::F32)?;
    }
    for (name, dims) in [
        ("cap_embedder.1.weight", [2560, 3840]),
        ("cap_pad_token", [3840, 1]),
        ("final_layer.adaLN_modulation.1.weight", [256, 3840]),
        ("final_layer.linear.weight", [3840, 64]),
        ("t_embedder.mlp.0.weight", [256, 1024]),
        ("t_embedder.mlp.2.weight", [1024, 256]),
        ("x_embedder.weight", [64, 3840]),
        ("x_pad_token", [3840, 1]),
    ] {
        require_tensor(source, name, &dims, GGMLType::F16)?;
    }
    for layer in 0..2 {
        validate_refiner(source, &format!("context_refiner.{layer}"), false)?;
        validate_refiner(source, &format!("noise_refiner.{layer}"), true)?;
    }
    for layer in 0..30 {
        let prefix = format!("layers.{layer}");
        require_tensor(
            source,
            &format!("{prefix}.adaLN_modulation.0.bias"),
            &[15360],
            GGMLType::F32,
        )?;
        for (suffix, dims) in [
            ("adaLN_modulation.0.weight", [256, 15360]),
            ("attention.out.weight", [3840, 3840]),
            ("attention.qkv.weight", [3840, 11520]),
            ("feed_forward.w1.weight", [3840, 10240]),
            ("feed_forward.w2.weight", [10240, 3840]),
            ("feed_forward.w3.weight", [3840, 10240]),
        ] {
            require_tensor(source, &format!("{prefix}.{suffix}"), &dims, GGMLType::Q8_0)?;
        }
        validate_transformer_vectors(source, &prefix)?;
    }
    Ok(())
}

fn validate_refiner(
    source: &dyn TensorSource,
    prefix: &str,
    has_adaln: bool,
) -> Result<(), String> {
    if has_adaln {
        require_tensor(
            source,
            &format!("{prefix}.adaLN_modulation.0.weight"),
            &[256, 15360],
            GGMLType::F16,
        )?;
        require_tensor(
            source,
            &format!("{prefix}.adaLN_modulation.0.bias"),
            &[15360],
            GGMLType::F32,
        )?;
    }
    for (suffix, dims) in [
        ("attention.out.weight", [3840, 3840]),
        ("attention.qkv.weight", [3840, 11520]),
        ("feed_forward.w1.weight", [3840, 10240]),
        ("feed_forward.w2.weight", [10240, 3840]),
        ("feed_forward.w3.weight", [3840, 10240]),
    ] {
        require_tensor(source, &format!("{prefix}.{suffix}"), &dims, GGMLType::F16)?;
    }
    validate_transformer_vectors(source, prefix)
}

fn validate_transformer_vectors(source: &dyn TensorSource, prefix: &str) -> Result<(), String> {
    for (suffix, dims) in [
        ("attention_norm1.weight", 3840),
        ("attention_norm2.weight", 3840),
        ("attention.k_norm.weight", 128),
        ("attention.q_norm.weight", 128),
        ("ffn_norm1.weight", 3840),
        ("ffn_norm2.weight", 3840),
    ] {
        require_tensor(
            source,
            &format!("{prefix}.{suffix}"),
            &[dims],
            GGMLType::F32,
        )?;
    }
    Ok(())
}

fn validate_vae(source: &dyn TensorSource) -> Result<(), String> {
    for (name, dims, ggml_type) in [
        ("decoder.conv_in.bias", &[512][..], GGMLType::F32),
        (
            "decoder.conv_in.weight",
            &[3, 3, 16, 512][..],
            GGMLType::F16,
        ),
        ("decoder.conv_out.bias", &[3][..], GGMLType::F32),
        (
            "decoder.conv_out.weight",
            &[3, 3, 128, 3][..],
            GGMLType::F16,
        ),
        ("decoder.norm_out.bias", &[128][..], GGMLType::F32),
        ("decoder.norm_out.weight", &[128][..], GGMLType::F32),
    ] {
        require_tensor(source, name, dims, ggml_type)?;
    }
    validate_vae_attention(source, "decoder.mid.attn_1", 512)?;
    validate_vae_block(source, "decoder.mid.block_1", 512, 512, false)?;
    validate_vae_block(source, "decoder.mid.block_2", 512, 512, false)?;

    for (stage, input_channels, output_channels) in
        [(0, 256, 128), (1, 512, 256), (2, 512, 512), (3, 512, 512)]
    {
        for block in 0..3 {
            validate_vae_block(
                source,
                &format!("decoder.up.{stage}.block.{block}"),
                if block == 0 {
                    input_channels
                } else {
                    output_channels
                },
                output_channels,
                block == 0 && input_channels != output_channels,
            )?;
        }
        if stage != 0 {
            require_tensor(
                source,
                &format!("decoder.up.{stage}.upsample.conv.weight"),
                &[3, 3, output_channels, output_channels],
                GGMLType::F16,
            )?;
            require_tensor(
                source,
                &format!("decoder.up.{stage}.upsample.conv.bias"),
                &[output_channels],
                GGMLType::F32,
            )?;
        }
    }
    Ok(())
}

fn validate_vae_attention(
    source: &dyn TensorSource,
    prefix: &str,
    channels: u64,
) -> Result<(), String> {
    for name in ["k", "proj_out", "q", "v"] {
        require_tensor(
            source,
            &format!("{prefix}.{name}.weight"),
            &[1, 1, channels, channels],
            GGMLType::F16,
        )?;
        require_tensor(
            source,
            &format!("{prefix}.{name}.bias"),
            &[channels],
            GGMLType::F32,
        )?;
    }
    require_tensor(
        source,
        &format!("{prefix}.norm.weight"),
        &[channels],
        GGMLType::F32,
    )?;
    require_tensor(
        source,
        &format!("{prefix}.norm.bias"),
        &[channels],
        GGMLType::F32,
    )
}

fn validate_vae_block(
    source: &dyn TensorSource,
    prefix: &str,
    input_channels: u64,
    output_channels: u64,
    has_shortcut: bool,
) -> Result<(), String> {
    for (name, dims) in [
        ("conv1", [3, 3, input_channels, output_channels]),
        ("conv2", [3, 3, output_channels, output_channels]),
    ] {
        require_tensor(
            source,
            &format!("{prefix}.{name}.weight"),
            &dims,
            GGMLType::F16,
        )?;
        require_tensor(
            source,
            &format!("{prefix}.{name}.bias"),
            &[output_channels],
            GGMLType::F32,
        )?;
    }
    for (name, channels) in [("norm1", input_channels), ("norm2", output_channels)] {
        require_tensor(
            source,
            &format!("{prefix}.{name}.weight"),
            &[channels],
            GGMLType::F32,
        )?;
        require_tensor(
            source,
            &format!("{prefix}.{name}.bias"),
            &[channels],
            GGMLType::F32,
        )?;
    }
    if has_shortcut {
        require_tensor(
            source,
            &format!("{prefix}.nin_shortcut.weight"),
            &[1, 1, input_channels, output_channels],
            GGMLType::F16,
        )?;
        require_tensor(
            source,
            &format!("{prefix}.nin_shortcut.bias"),
            &[output_channels],
            GGMLType::F32,
        )?;
    }
    Ok(())
}

pub(crate) struct Q8Scratch {
    scaled: Vec<f32>,
    force_f32_row: Vec<f32>,
    f16_input: Vec<u16>,
    values: Vec<u8>,
    scales: Vec<f32>,
}

impl Q8Scratch {
    pub(crate) fn new(n_in: usize) -> Self {
        Self {
            scaled: Vec::new(),
            force_f32_row: Vec::new(),
            f16_input: Vec::new(),
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

#[allow(clippy::too_many_arguments)]
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
    linear_into_scaled_impl(
        source, name, n_in, n_out, input, output, q8, pool, 1.0, false,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn linear_into_ggml(
    source: &dyn TensorSource,
    name: &str,
    n_in: usize,
    n_out: usize,
    input: &[f32],
    output: &mut [f32],
    q8: &mut Q8Scratch,
    pool: &ComputePool,
) -> Result<(), String> {
    linear_into_scaled_impl(
        source, name, n_in, n_out, input, output, q8, pool, 1.0, true,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn linear_into_scaled(
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
    linear_into_scaled_impl(
        source, name, n_in, n_out, input, output, q8, pool, scale, false,
    )
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
    ggml_reduction: bool,
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
        GGMLType::F16 => F16Kernel::new(bytes).forward_scaled(
            input,
            output,
            n_in,
            n_out,
            scale,
            &mut q8.f16_input,
        ),
        GGMLType::Q8_0 => {
            if scale == 1.0 {
                q8.prepare(input, n_in)?;
            } else {
                q8.prepare_scaled(input, n_in, scale)?;
            }
            if ggml_reduction {
                matmul_q8_0_ggml(bytes, &q8.values, &q8.scales, output, n_in, n_out, pool);
            } else {
                crate::ops::matmul_q8_0_quantized_dynamic(
                    bytes, &q8.values, &q8.scales, output, n_in, n_out, pool,
                );
            }
            if scale != 1.0 {
                let inverse_scale = scale.recip();
                for value in output {
                    *value *= inverse_scale;
                }
            }
        }
        _ => unreachable!(),
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn matmul_q8_0_ggml(
    weight: &[u8],
    input_q8: &[u8],
    input_scales: &[f32],
    output: &mut [f32],
    n_in: usize,
    n_out: usize,
    pool: &ComputePool,
) {
    #[cfg(target_arch = "aarch64")]
    {
        let weight_ptr = weight.as_ptr() as usize;
        let weight_len = weight.len();
        let input_ptr = input_q8.as_ptr() as usize;
        let input_len = input_q8.len();
        let scale_ptr = input_scales.as_ptr() as usize;
        let scale_len = input_scales.len();
        let output_ptr = output.as_mut_ptr() as usize;
        pool.compute(move |thread, threads| {
            let rows_per_thread = n_out.div_ceil(threads);
            let row_start = thread * rows_per_thread;
            let row_end = (row_start + rows_per_thread).min(n_out);
            if row_start >= row_end {
                return;
            }
            let weight = unsafe { std::slice::from_raw_parts(weight_ptr as *const u8, weight_len) };
            let input_q8 = unsafe { std::slice::from_raw_parts(input_ptr as *const u8, input_len) };
            let input_scales =
                unsafe { std::slice::from_raw_parts(scale_ptr as *const f32, scale_len) };
            let output = unsafe {
                std::slice::from_raw_parts_mut(
                    (output_ptr as *mut f32).add(row_start),
                    row_end - row_start,
                )
            };
            unsafe {
                crate::ops::kernel::q8_0::neon::matmul_q8_0_vs_q8_0_neon_nrc1(
                    weight,
                    input_q8,
                    input_scales,
                    output,
                    n_in,
                    row_start,
                    row_end,
                );
            }
        });
    }
    #[cfg(not(target_arch = "aarch64"))]
    crate::ops::matmul_q8_0_quantized_dynamic(
        weight,
        input_q8,
        input_scales,
        output,
        n_in,
        n_out,
        pool,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::tensor::{GGMLType, MetaValue, TensorInfo, TensorSource};
    use crate::core::thread_pool::ComputePool;
    use half::f16;
    use std::collections::HashMap;

    #[derive(Default)]
    struct TestSource {
        metadata: HashMap<String, MetaValue>,
        tensors: HashMap<String, TensorInfo>,
        data: HashMap<String, Vec<u8>>,
    }

    struct MaskedSource<'a> {
        inner: &'a crate::core::loader::GGUFLoader,
        missing: &'a str,
    }

    impl TensorSource for MaskedSource<'_> {
        fn metadata(&self, key: &str) -> Option<&MetaValue> {
            self.inner.metadata(key)
        }

        fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
            (name != self.missing)
                .then(|| self.inner.tensor_info(name))
                .flatten()
        }

        fn tensor_slice(&self, name: &str) -> Option<&[u8]> {
            (name != self.missing)
                .then(|| self.inner.tensor_slice(name))
                .flatten()
        }
    }

    impl TestSource {
        fn with_metadata(mut self, key: &str, value: &str) -> Self {
            self.metadata
                .insert(key.into(), MetaValue::String(value.into()));
            self
        }

        fn with_tensor(mut self, name: &str, dims: &[u64], ggml_type: GGMLType) -> Self {
            let info = TensorInfo {
                name: name.into(),
                dims: dims.into(),
                ggml_type,
                offset: 0,
            };
            self.data.insert(name.into(), vec![0; info.nbytes()]);
            self.tensors.insert(name.into(), info);
            self
        }

        fn with_raw_tensor(
            mut self,
            name: &str,
            dims: &[u64],
            ggml_type: GGMLType,
            data: Vec<u8>,
        ) -> Self {
            self.tensors.insert(
                name.into(),
                TensorInfo {
                    name: name.into(),
                    dims: dims.into(),
                    ggml_type,
                    offset: 0,
                },
            );
            self.data.insert(name.into(), data);
            self
        }

        fn f16_matrix(name: &str, dims: &[u64], values: [f32; 4]) -> Self {
            let bytes = values
                .into_iter()
                .flat_map(|value| f16::from_f32(value).to_bits().to_le_bytes())
                .collect();
            Self::default().with_raw_tensor(name, dims, GGMLType::F16, bytes)
        }
    }

    impl TensorSource for TestSource {
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
    fn pig_metadata_cannot_identify_a_component() {
        let source = TestSource::default()
            .with_metadata("general.architecture", "pig")
            .with_tensor("x_embedder.weight", &[64, 3840], GGMLType::F16);
        assert!(validate_component(&source, Component::Dit).is_err());
        assert!(validate_component(&source, Component::Text).is_err());
        assert!(validate_component(&source, Component::Vae).is_err());
    }

    #[test]
    fn f16_linear_uses_little_endian_half_values() {
        let source = TestSource::f16_matrix("w", &[2, 2], [1.0, 2.0, 3.0, 4.0]);
        let mut out = [99.0, 99.0];
        linear_into(
            &source,
            "w",
            2,
            2,
            &[5.0, 6.0],
            &mut out,
            &mut Q8Scratch::new(2),
            &ComputePool::new(1),
        )
        .unwrap();
        assert_eq!(out, [17.0, 39.0]);
    }

    #[test]
    fn f16_linear_reuses_caller_owned_input_scratch() {
        let source = TestSource::f16_matrix("w", &[2, 2], [1.0, 2.0, 3.0, 4.0]);
        let mut scratch = Q8Scratch::new(2);
        let mut output = [0.0; 2];

        linear_into(
            &source,
            "w",
            2,
            2,
            &[5.0, 6.0],
            &mut output,
            &mut scratch,
            &ComputePool::new(1),
        )
        .unwrap();
        let input_ptr = scratch.f16_input.as_ptr();
        linear_into(
            &source,
            "w",
            2,
            2,
            &[5.0, 6.0],
            &mut output,
            &mut scratch,
            &ComputePool::new(1),
        )
        .unwrap();

        assert_eq!(output, [17.0, 39.0]);
        assert_eq!(scratch.f16_input.as_ptr(), input_ptr);
        assert_eq!(scratch.f16_input.len(), 2);
    }

    #[test]
    fn q8_linear_with_two_threads_overwrites_output() {
        let mut weight = f16::from_f32(1.0).to_bits().to_le_bytes().to_vec();
        weight.extend([1u8; 32]);
        let source = TestSource::default().with_raw_tensor("w", &[32, 1], GGMLType::Q8_0, weight);
        let mut out = [99.0];
        linear_into(
            &source,
            "w",
            32,
            1,
            &[2.0; 32],
            &mut out,
            &mut Q8Scratch::new(32),
            &ComputePool::new(2),
        )
        .unwrap();
        assert!((out[0] - 64.0).abs() < 0.01, "{}", out[0]);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn ggml_q8_linear_uses_even_odd_lane_reduction() {
        let mut state = 1u32;
        let mut next = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            state
        };
        let mut weight = Vec::with_capacity(8 * 34);
        for _ in 0..8 {
            let scale = f16::from_f32(((next() % 10_000) as f32 + 1.0) * 1e-5);
            weight.extend_from_slice(&scale.to_bits().to_le_bytes());
            for _ in 0..32 {
                weight.push(((next() % 255) as i16 - 127) as i8 as u8);
            }
        }
        let input = (0..256)
            .map(|_| ((next() % 20_001) as f32 - 10_000.0) * 1e-4)
            .collect::<Vec<_>>();
        let source = TestSource::default().with_raw_tensor("w", &[256, 1], GGMLType::Q8_0, weight);
        let mut output = [0.0f32];

        linear_into_ggml(
            &source,
            "w",
            256,
            1,
            &input,
            &mut output,
            &mut Q8Scratch::new(256),
            &ComputePool::new(1),
        )
        .unwrap();

        assert_eq!(output[0].to_bits(), 0xc24b_f6df);
    }

    #[test]
    fn linear_rejects_wrong_input_length() {
        let source = TestSource::f16_matrix("w", &[2, 2], [1.0; 4]);
        let error = linear_into(
            &source,
            "w",
            2,
            2,
            &[1.0],
            &mut [0.0; 2],
            &mut Q8Scratch::new(2),
            &ComputePool::new(1),
        )
        .unwrap_err();
        assert_eq!(error, "Invalid linear input length");
    }

    #[test]
    fn linear_rejects_wrong_output_length() {
        let source = TestSource::f16_matrix("w", &[2, 2], [1.0; 4]);
        let error = linear_into(
            &source,
            "w",
            2,
            2,
            &[1.0; 2],
            &mut [0.0; 1],
            &mut Q8Scratch::new(2),
            &ComputePool::new(1),
        )
        .unwrap_err();
        assert_eq!(error, "Invalid linear output length");
    }

    #[test]
    fn linear_rejects_wrong_dimensions() {
        let source = TestSource::f16_matrix("w", &[2, 2], [1.0; 4]);
        let error = linear_into(
            &source,
            "w",
            4,
            1,
            &[1.0; 4],
            &mut [0.0],
            &mut Q8Scratch::new(4),
            &ComputePool::new(1),
        )
        .unwrap_err();
        assert_eq!(error, "Invalid w dimensions");
    }

    #[test]
    fn linear_rejects_truncated_tensor_data() {
        let source = TestSource::default().with_raw_tensor("w", &[2, 2], GGMLType::F16, vec![0; 6]);
        let error = linear_into(
            &source,
            "w",
            2,
            2,
            &[1.0; 2],
            &mut [0.0; 2],
            &mut Q8Scratch::new(2),
            &ComputePool::new(1),
        )
        .unwrap_err();
        assert_eq!(error, "Invalid w byte length");
    }

    #[test]
    fn linear_rejects_unsupported_matrix_type() {
        let source =
            TestSource::default().with_raw_tensor("w", &[2, 2], GGMLType::F32, vec![0; 16]);
        let error = linear_into(
            &source,
            "w",
            2,
            2,
            &[1.0; 2],
            &mut [0.0; 2],
            &mut Q8Scratch::new(2),
            &ComputePool::new(1),
        )
        .unwrap_err();
        assert_eq!(error, "Unsupported matrix type F32 for w");
    }

    #[test]
    fn pipeline_requires_nonempty_complete_qwen_context_rows() {
        assert_eq!(context_token_count(&vec![0.0; 5_120]).unwrap(), 2);
        assert!(context_token_count(&[]).is_err());
        assert!(context_token_count(&vec![0.0; 2_561]).is_err());
    }

    #[test]
    fn pipeline_context_boundary_rejects_nan_and_infinity() {
        for value in [f32::NAN, f32::INFINITY] {
            let mut context = vec![0.0; 2_560];
            context[17] = value;
            assert!(context_token_count(&context).is_err());
        }
    }

    #[test]
    fn pipeline_latent_boundary_rejects_nan_and_infinity() {
        for value in [f32::NAN, f32::NEG_INFINITY] {
            let mut latent = vec![0.0; 64];
            latent[17] = value;
            assert!(validate_latent_shape(&latent, 16).is_err());
        }
    }

    #[test]
    fn pipeline_decoded_rgb_boundary_rejects_malformed_byte_length() {
        let malformed = ZImageRgb {
            width: 2,
            height: 2,
            bytes: vec![0; 11],
        };
        assert!(validate_decoded_rgb(&malformed, 2).is_err());
    }

    #[test]
    fn pipeline_decoded_rgb_boundary_rejects_length_overflow() {
        let overflow = ZImageRgb {
            width: u32::MAX,
            height: u32::MAX,
            bytes: Vec::new(),
        };
        assert!(validate_decoded_rgb(&overflow, u32::MAX as usize).is_err());
    }

    #[test]
    fn pipeline_revalidates_generation_options_and_latent_shape() {
        let valid = ZImageOptions {
            steps: 1,
            resolution: 16,
            seed: 42,
        };
        validate_generate_request("fox", &valid).unwrap();
        assert_eq!(validate_latent_shape(&vec![0.0; 64], 16).unwrap(), 2);
        assert!(validate_generate_request("  ", &valid).is_err());
        assert!(validate_generate_request("fox", &ZImageOptions { steps: 0, ..valid }).is_err());
        assert!(validate_generate_request(
            "fox",
            &ZImageOptions {
                resolution: 24,
                ..valid
            }
        )
        .is_err());
        assert!(validate_latent_shape(&vec![0.0; 63], 16).is_err());
    }

    #[test]
    #[ignore = "requires Z_IMAGE_TEXT_GGUF, Z_IMAGE_DIT_GGUF, and Z_IMAGE_VAE_GGUF"]
    fn supplied_ggufs_cover_every_component_signature() {
        let load = |name: &str| {
            crate::core::loader::GGUFLoader::from_file(
                std::env::var(name).unwrap_or_else(|_| panic!("missing {name}")),
            )
            .unwrap()
        };
        let cases = [
            (load("Z_IMAGE_TEXT_GGUF"), Component::Text, None, 398),
            (load("Z_IMAGE_DIT_GGUF"), Component::Dit, None, 453),
            (
                load("Z_IMAGE_VAE_GGUF"),
                Component::Vae,
                Some("decoder."),
                138,
            ),
        ];

        for (source, component, prefix, expected_count) in &cases {
            validate_component(source, *component).unwrap();
            for other in [Component::Text, Component::Dit, Component::Vae] {
                if other != *component {
                    assert!(validate_component(source, other).is_err());
                }
            }

            let required: Vec<_> = source
                .tensors()
                .iter()
                .filter(|info| prefix.is_none_or(|prefix| info.name.starts_with(prefix)))
                .collect();
            assert_eq!(required.len(), *expected_count);
            for info in required {
                let masked = MaskedSource {
                    inner: source,
                    missing: &info.name,
                };
                assert!(
                    validate_component(&masked, *component).is_err(),
                    "{component:?} accepted missing tensor {}",
                    info.name
                );
            }
        }
    }
}
