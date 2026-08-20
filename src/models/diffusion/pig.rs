use std::sync::Arc;
use crate::core::tensor::TensorSource;
use crate::ops::{matmul_q8_0_quantized_parallel_rows, quantize_q8_0_into, dot_f32, rms_norm, rms_norm_inplace, silu, f16_to_f32};
use crate::core::thread_pool::ComputePool;

#[derive(Debug)]
pub struct PigConfig {
    pub n_layer: usize,
    pub n_embed: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub n_ff: usize,
    pub latent_channels: usize,
    pub patch_size: usize,
    pub head_dim: usize,
    pub axes_dim_sum: usize,
    pub cap_feat_dim: usize,
    pub n_refiner_layers: usize,
    pub context_len: usize,
    pub theta: i32,
    pub axes_dim: Vec<usize>,
}

impl PigConfig {
    pub fn from_source<S: crate::core::tensor::TensorSource + ?Sized>(source: &S) -> Result<Self, String> {
        let n_layer = source.metadata("pig.block_count").and_then(|v| v.to_u64()).map(|v| v as usize).unwrap_or(30);
        let n_embed = source.metadata("pig.embedding_length").and_then(|v| v.to_u64()).map(|v| v as usize).unwrap_or(3840);
        let n_head = source.metadata("pig.attention.head_count").and_then(|v| v.to_u64()).map(|v| v as usize).unwrap_or(30);
        let n_head_kv = source.metadata("pig.attention.head_count_kv").and_then(|v| v.to_u64()).map(|v| v as usize).unwrap_or(30);
        let n_ff = source.metadata("pig.feed_forward_length").and_then(|v| v.to_u64()).map(|v| v as usize).unwrap_or(10240);
        Ok(Self {
            n_layer,
            n_embed,
            n_head,
            n_head_kv,
            n_ff,
            latent_channels: 16,
            patch_size: 2,
            head_dim: 128,
            axes_dim_sum: 256,
            cap_feat_dim: 2560,
            n_refiner_layers: 2,
            context_len: 256,
            theta: 256,
            axes_dim: vec![64, 96, 96],
        })
    }
}

pub struct PigModel {
    config: PigConfig,
    pool: Arc<ComputePool>,
    x_embedder_weight: Vec<f32>,
    x_embedder_bias: Vec<f32>,
    t_embedder_mlp_0_weight: Vec<f32>,
    t_embedder_mlp_0_bias: Vec<f32>,
    t_embedder_mlp_2_weight: Vec<f32>,
    t_embedder_mlp_2_bias: Vec<f32>,
    cap_embedder_0_weight: Vec<f32>,
    cap_embedder_1_weight: Vec<f32>,
    cap_embedder_1_bias: Vec<f32>,
    cap_pad_token: Vec<f32>,
    x_pad_token: Vec<f32>,
    context_refiner_layers: Vec<RefinerLayer>,
    noise_refiner_layers: Vec<RefinerLayer>,
    layers: Vec<PigLayer>,
    final_layer_adaln_weight: Vec<f32>,
    final_layer_adaln_bias: Vec<f32>,
    final_layer_linear_weight: Vec<f32>,
    final_layer_linear_bias: Vec<f32>,
}

struct RefinerLayer {
    qkv_weight: Vec<u8>,
    out_weight: Vec<u8>,
    q_norm_weight: Vec<f32>,
    k_norm_weight: Vec<f32>,
    attention_norm1_weight: Vec<f32>,
    attention_norm2_weight: Vec<f32>,
    ffn_norm1_weight: Vec<f32>,
    ffn_norm2_weight: Vec<f32>,
    w1_weight: Vec<u8>,
    w2_weight: Vec<u8>,
    w3_weight: Vec<u8>,
    adaln_weight: Option<Vec<f32>>,
    adaln_bias: Option<Vec<f32>>,
}

impl PigModel {
    pub fn from_source(source: Arc<dyn TensorSource>, pool: Arc<ComputePool>) -> Result<Self, String> {
        let config = PigConfig::from_source(source.as_ref())?;

        let x_embedder_weight = load_f16_as_f32(source.as_ref(), "x_embedder.weight")?;
        let x_embedder_bias = load_f32(source.as_ref(), "x_embedder.bias")?;
        let t_embedder_mlp_0_weight = load_f16_as_f32(source.as_ref(), "t_embedder.mlp.0.weight")?;
        let t_embedder_mlp_0_bias = load_f32(source.as_ref(), "t_embedder.mlp.0.bias")?;
        let t_embedder_mlp_2_weight = load_f16_as_f32(source.as_ref(), "t_embedder.mlp.2.weight")?;
        let t_embedder_mlp_2_bias = load_f32(source.as_ref(), "t_embedder.mlp.2.bias")?;
        let cap_embedder_0_weight = load_f32(source.as_ref(), "cap_embedder.0.weight")?;
        let cap_embedder_1_weight = load_f16_as_f32(source.as_ref(), "cap_embedder.1.weight")?;
        let cap_embedder_1_bias = load_f32(source.as_ref(), "cap_embedder.1.bias")?;
        let cap_pad_token = load_f16_as_f32(source.as_ref(), "cap_pad_token")?;
        let x_pad_token = load_f16_as_f32(source.as_ref(), "x_pad_token")?;

        let mut context_refiner_layers = Vec::new();
        for i in 0..config.n_refiner_layers {
            context_refiner_layers.push(load_refiner_layer(source.as_ref(), &format!("context_refiner.{}", i), &config)?);
        }

        let mut noise_refiner_layers = Vec::new();
        for i in 0..config.n_refiner_layers {
            noise_refiner_layers.push(load_refiner_layer(source.as_ref(), &format!("noise_refiner.{}", i), &config)?);
        }

        let mut layers = Vec::with_capacity(config.n_layer);
        for i in 0..config.n_layer {
            layers.push(load_pig_layer(source.as_ref(), i, &config)?);
        }

        let final_layer_adaln_weight = load_f16_as_f32(source.as_ref(), "final_layer.adaLN_modulation.1.weight")?;
        let final_layer_adaln_bias = load_f32(source.as_ref(), "final_layer.adaLN_modulation.1.bias")?;
        let final_layer_linear_weight = load_f16_as_f32(source.as_ref(), "final_layer.linear.weight")?;
        let final_layer_linear_bias = load_f32(source.as_ref(), "final_layer.linear.bias")?;

        Ok(Self {
            config,
            pool,
            x_embedder_weight,
            x_embedder_bias,
            t_embedder_mlp_0_weight,
            t_embedder_mlp_0_bias,
            t_embedder_mlp_2_weight,
            t_embedder_mlp_2_bias,
            cap_embedder_0_weight,
            cap_embedder_1_weight,
            cap_embedder_1_bias,
            cap_pad_token,
            x_pad_token,
            context_refiner_layers,
            noise_refiner_layers,
            layers,
            final_layer_adaln_weight,
            final_layer_adaln_bias,
            final_layer_linear_weight,
            final_layer_linear_bias,
        })
    }

    pub fn config(&self) -> &PigConfig {
        &self.config
    }

    pub fn pool(&self) -> &ComputePool {
        &self.pool
    }
}

fn load_refiner_layer(source: &dyn TensorSource, prefix: &str, config: &PigConfig) -> Result<RefinerLayer, String> {
    let qkv_weight = load_q8_0(source, &format!("{}.attention.qkv.weight", prefix))?;
    let out_weight = load_q8_0(source, &format!("{}.attention.out.weight", prefix))?;
    let q_norm_weight = load_f32(source, &format!("{}.attention.q_norm.weight", prefix))?;
    let k_norm_weight = load_f32(source, &format!("{}.attention.k_norm.weight", prefix))?;
    let attention_norm1_weight = load_f32(source, &format!("{}.attention_norm1.weight", prefix))?;
    let attention_norm2_weight = load_f32(source, &format!("{}.attention_norm2.weight", prefix))?;
    let ffn_norm1_weight = load_f32(source, &format!("{}.ffn_norm1.weight", prefix))?;
    let ffn_norm2_weight = load_f32(source, &format!("{}.ffn_norm2.weight", prefix))?;
    let w1_weight = load_q8_0(source, &format!("{}.feed_forward.w1.weight", prefix))?;
    let w2_weight = load_q8_0(source, &format!("{}.feed_forward.w2.weight", prefix))?;
    let w3_weight = load_q8_0(source, &format!("{}.feed_forward.w3.weight", prefix))?;
    let adaln_weight = if let Ok(w) = load_f16_as_f32(source, &format!("{}.adaLN_modulation.0.weight", prefix)) {
        Some(w)
    } else {
        None
    };
    let adaln_bias = if let Ok(b) = load_f32(source, &format!("{}.adaLN_modulation.0.bias", prefix)) {
        Some(b)
    } else {
        None
    };
    Ok(RefinerLayer {
        qkv_weight, out_weight, q_norm_weight, k_norm_weight,
        attention_norm1_weight, attention_norm2_weight,
        ffn_norm1_weight, ffn_norm2_weight,
        w1_weight, w2_weight, w3_weight,
        adaln_weight, adaln_bias,    })
}

fn load_pig_layer(source: &dyn TensorSource, i: usize, config: &PigConfig) -> Result<PigLayer, String> {
    let adaln_weight = load_q8_0(source, &format!("layers.{}.adaLN_modulation.0.weight", i))?;
    let adaln_bias = load_f32(source, &format!("layers.{}.adaLN_modulation.0.bias", i))?;
    let qkv_weight = load_q8_0(source, &format!("layers.{}.attention.qkv.weight", i))?;
    let out_weight = load_q8_0(source, &format!("layers.{}.attention.out.weight", i))?;
    let q_norm_weight = load_f32(source, &format!("layers.{}.attention.q_norm.weight", i))?;
    let k_norm_weight = load_f32(source, &format!("layers.{}.attention.k_norm.weight", i))?;
    let attention_norm1_weight = load_f32(source, &format!("layers.{}.attention_norm1.weight", i))?;
    let attention_norm2_weight = load_f32(source, &format!("layers.{}.attention_norm2.weight", i))?;
    let ffn_norm1_weight = load_f32(source, &format!("layers.{}.ffn_norm1.weight", i))?;
    let ffn_norm2_weight = load_f32(source, &format!("layers.{}.ffn_norm2.weight", i))?;
    let w1_weight = load_q8_0(source, &format!("layers.{}.feed_forward.w1.weight", i))?;
    let w2_weight = load_q8_0(source, &format!("layers.{}.feed_forward.w2.weight", i))?;
    let w3_weight = load_q8_0(source, &format!("layers.{}.feed_forward.w3.weight", i))?;
    Ok(PigLayer {
        adaln_weight, adaln_bias, qkv_weight, out_weight,
        q_norm_weight, k_norm_weight,
        attention_norm1_weight, attention_norm2_weight,
        ffn_norm1_weight, ffn_norm2_weight,
        w1_weight, w2_weight, w3_weight,
    })
}

#[derive(Clone)]
struct PigLayer {
    adaln_weight: Vec<u8>,
    adaln_bias: Vec<f32>,
    qkv_weight: Vec<u8>,
    out_weight: Vec<u8>,
    q_norm_weight: Vec<f32>,
    k_norm_weight: Vec<f32>,
    attention_norm1_weight: Vec<f32>,
    attention_norm2_weight: Vec<f32>,
    ffn_norm1_weight: Vec<f32>,
    ffn_norm2_weight: Vec<f32>,
    w1_weight: Vec<u8>,
    w2_weight: Vec<u8>,
    w3_weight: Vec<u8>,
}

pub struct PigVAE {
    conv_in_weight: Vec<f32>,
    conv_in_bias: Vec<f32>,
    conv_out_weight: Vec<f32>,
    conv_out_bias: Vec<f32>,
    norm_out_weight: Vec<f32>,
    norm_out_bias: Vec<f32>,
    hidden: usize,
    n_up: usize,
}

impl PigVAE {
    pub fn from_source(source: &dyn TensorSource) -> Result<Self, String> {
        let conv_in_weight = load_f32(source, "decoder.conv_in.weight")?;
        let conv_in_bias = load_f32(source, "decoder.conv_in.bias")?;
        let conv_out_weight = load_f32(source, "decoder.conv_out.weight")?;
        let conv_out_bias = load_f32(source, "decoder.conv_out.bias")?;
        let norm_out_weight = load_f32(source, "decoder.norm_out.weight")?;
        let norm_out_bias = load_f32(source, "decoder.norm_out.bias")?;
        let hidden = conv_in_weight.len() / (16 * 9);
        Ok(Self {
            conv_in_weight, conv_in_bias,
            conv_out_weight, conv_out_bias,
            norm_out_weight, norm_out_bias,
            hidden, n_up: 4,
        })
    }

    pub fn decode(&self, patches: &[f32], n_patches: usize, latent_size: usize) -> Result<Vec<u8>, String> {
        let patches_per_dim = (latent_size as f32).sqrt() as usize;
        let latent_channels = 16;
        let mut latents = vec![0.0f32; latent_channels * latent_size * latent_size];
        for p in 0..n_patches {
            let px = p % patches_per_dim;
            let py = p / patches_per_dim;
            for ky in 0..2 {
                for kx in 0..2 {
                    for c in 0..latent_channels {
                        let src_idx = ky * 2 * latent_channels + kx * latent_channels + c;
                        let dst_idx = (py * 2 + ky) * latent_size * latent_channels + (px * 2 + kx) * latent_channels + c;
                        if src_idx < patches.len() / n_patches && dst_idx < latents.len() {
                            latents[dst_idx] = patches[p * patches.len() / n_patches + src_idx];
                        }
                    }
                }
            }
        }
        let mut hidden = vec![0.0f32; self.hidden * latent_size * latent_size];
        self.conv2d(&latents, latent_channels, self.hidden, latent_size, &self.conv_in_weight, &self.conv_in_bias, &mut hidden, 3, 1);
        let mut normalized = hidden.clone();
        self.group_norm(&mut normalized, &self.norm_out_weight, &self.norm_out_bias, 32);
        for x in normalized.iter_mut() { *x = silu(*x); }
        let out_channels = 3;
        let mut rgb = vec![0.0f32; out_channels * latent_size * latent_size];
        self.conv2d(&normalized, self.hidden, out_channels, latent_size, &self.conv_out_weight, &self.conv_out_bias, &mut rgb, 3, 1);
        let final_size = latent_size * 8;
        let mut upsampled = vec![0.0f32; out_channels * final_size * final_size];
        self.upsample_nearest(&rgb, out_channels, latent_size, &mut upsampled, 8);
        let mut pixels = Vec::with_capacity(final_size * final_size * 4);
        for y in 0..final_size {
            for x in 0..final_size {
                let r = clamp01_to_u8(upsampled[0 * final_size * final_size + y * final_size + x]);
                let g = clamp01_to_u8(upsampled[1 * final_size * final_size + y * final_size + x]);
                let b = clamp01_to_u8(upsampled[2 * final_size * final_size + y * final_size + x]);
                pixels.push(r); pixels.push(g); pixels.push(b); pixels.push(255);
            }
        }
        Ok(pixels)
    }

    fn conv2d(&self, input: &[f32], in_c: usize, out_c: usize, h: usize, weight: &[f32], bias: &[f32], output: &mut [f32], kernel_size: usize, _stride: usize) {
        let pad = kernel_size / 2;
        for oc in 0..out_c {
            for y in 0..h {
                for x in 0..h {
                    let mut sum = 0.0f32;
                    for ic in 0..in_c {
                        for ky in 0..kernel_size {
                            for kx in 0..kernel_size {
                                let iy = y as i32 + ky as i32 - pad as i32;
                                let ix = x as i32 + kx as i32 - pad as i32;
                                if iy >= 0 && iy < h as i32 && ix >= 0 && ix < h as i32 {
                                    let input_idx = ic * h * h + iy as usize * h + ix as usize;
                                    let weight_idx = oc * in_c * kernel_size * kernel_size + ic * kernel_size * kernel_size + ky * kernel_size + kx;
                                    sum += input[input_idx] * weight[weight_idx];
                                }
                            }
                        }
                    }
                    sum += bias[oc];
                    output[oc * h * h + y * h + x] = sum;
                }
            }
        }
    }

    fn group_norm(&self, input: &mut [f32], weight: &[f32], bias: &[f32], num_groups: usize) {
        let ch_per_group = weight.len() / num_groups;
        for g in 0..num_groups {
            let start = g * ch_per_group;
            let end = start + ch_per_group;
            let group = &input[start..end];
            let mean = group.iter().sum::<f32>() / ch_per_group as f32;
            let var = group.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / ch_per_group as f32;
            let inv_std = 1.0 / (var.sqrt() + 1e-5);
            for i in 0..ch_per_group {
                input[start + i] = (input[start + i] - mean) * inv_std * weight[start + i] + bias[start + i];
            }
        }
    }

    fn upsample_nearest(&self, input: &[f32], channels: usize, in_size: usize, output: &mut [f32], factor: usize) {
        let out_size = in_size * factor;
        for c in 0..channels {
            for y in 0..out_size {
                for x in 0..out_size {
                    output[c * out_size * out_size + y * out_size + x] = input[c * in_size * in_size + (y / factor) * in_size + (x / factor)];
                }
            }
        }
    }
}

fn clamp01_to_u8(v: f32) -> u8 {
    ((v.clamp(-1.5, 1.5) + 1.5) * 255.0 / 3.0).max(0.0).min(255.0) as u8
}

pub struct PigSession<'a> {
    model: &'a PigModel,
    resolution: usize,
    vae: Option<&'a PigVAE>,
}

impl<'a> PigSession<'a> {
    pub fn new(model: &'a PigModel, resolution: usize) -> Result<Self, String> {
        if resolution % 8 != 0 || resolution == 0 {
            return Err(format!("resolution {resolution} must be a positive multiple of 8"));
        }
        Ok(Self { model, resolution, vae: None })
    }

    pub fn set_vae(&mut self, vae: &'a PigVAE) {
        self.vae = Some(vae);
    }

    pub fn generate_image(&mut self, text_context: &[f32], steps: usize) -> Result<Vec<u8>, String> {
        let cfg = &self.model.config;
        let latent_size = self.resolution / 8;
        let patches_per_dim = latent_size / cfg.patch_size;
        let n_patches = patches_per_dim * patches_per_dim;
        let n_embed = cfg.n_embed;
        let seq_len = cfg.context_len + n_patches;

        let sigma_min = 0.029f32;
        let sigma_max = 1.0f32;
        let sigmas = self.discrete_scheduler_sigmas(steps, sigma_min, sigma_max);

        let mut latents = vec![0.0f32; n_patches * n_embed];

        for (step_idx, &sigma) in sigmas.iter().enumerate() {
            if step_idx < sigmas.len() - 1 {
                eprintln!("Denoising step {}/{}", step_idx + 1, sigmas.len() - 1);
            }

            let x_noisy = if step_idx == 0 {
                use std::time::{SystemTime, UNIX_EPOCH};
                let seed = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() as u64;
                let mut rng = seed;
                for lat in latents.iter_mut() {
                    rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                    *lat = ((rng >> 33) as f32 / u32::MAX as f32) * sigma;
                }
                latents.clone()
            } else {
                latents.clone()
            };

            let denoised = self.denoise_step(&x_noisy, sigma, text_context, n_patches, patches_per_dim)?;

            let sigma_next = sigmas[step_idx + 1];
            let sigma_ratio = sigma_next / sigma;
            let updated: Vec<f32> = x_noisy.iter().zip(denoised.iter()).map(|(&x, &d)| {
                sigma_ratio * x + (1.0 - sigma_ratio) * d
            }).collect();

            latents = updated;
        }

        let pixels = self.decode_patches(&latents, n_patches, latent_size)?;
        Ok(pixels)
    }

    fn discrete_scheduler_sigmas(&self, n: usize, sigma_min: f32, sigma_max: f32) -> Vec<f32> {
        if n == 0 { return vec![]; }
        if n == 1 { return vec![sigma_max, 0.0]; }
        let step = (sigma_max - sigma_min) / (n as f32 - 1.0);
        let mut sigmas: Vec<f32> = (0..n).map(|i| sigma_max - step * i as f32).collect();
        sigmas.push(0.0);
        sigmas
    }

    fn denoise_step(&self, x: &[f32], sigma: f32, text_context: &[f32], n_patches: usize, patches_per_dim: usize) -> Result<Vec<f32>, String> {
        let cfg = &self.model.config;
        let n_embed = cfg.n_embed;
        let head_dim = cfg.head_dim;
        let n_head = cfg.n_head;
        let n_head_kv = cfg.n_head_kv;
        let context_len = cfg.context_len;

        let t_embed = self.compute_t_embed(sigma);

        let n_txt_pad = (32 - (context_len % 32)) % 32;
        let n_img_pad = (32 - (n_patches % 32)) % 32;
        let total_seq = context_len + n_txt_pad + n_patches + n_img_pad;

        let pe = self.gen_z_image_pe(total_seq, context_len, patches_per_dim, patches_per_dim);

        let mut txt = self.cap_embedder(text_context, context_len)?;

        let txt_pad = vec![0.0f32; n_embed * n_txt_pad];
        txt.extend(txt_pad);

        let img = self.x_embedder(x, n_patches)?;
        let img_pad = vec![0.0f32; n_embed * n_img_pad];
        let mut img = img;
        img.extend(img_pad);

        let txt_pe = &pe[..context_len * cfg.axes_dim_sum];
        let img_pe = &pe[context_len * cfg.axes_dim_sum..];

        let mut txt_pe_mat: Vec<f32> = Vec::with_capacity(context_len * cfg.axes_dim_sum);
        txt_pe_mat.extend_from_slice(txt_pe);
        let mut img_pe_mat: Vec<f32> = Vec::with_capacity(n_patches * cfg.axes_dim_sum);
        img_pe_mat.extend_from_slice(img_pe);

        for layer in &self.model.context_refiner_layers {
            txt = self.refiner_block(&txt, &txt_pe_mat, layer, None)?;
        }

        for layer in &self.model.noise_refiner_layers {
            img = self.refiner_block(&img, &img_pe_mat, layer, Some(&t_embed))?;
        }

        let mut txt_img: Vec<f32> = txt;
        txt_img.extend(img);

        for layer in &self.model.layers {
            txt_img = self.pig_layer_block(&txt_img, total_seq, &pe, &t_embed, layer)?;
        }

        let final_out = self.final_layer(&txt_img[..n_patches * n_embed], &t_embed)?;

        let mut result = vec![0.0f32; n_patches * 64];
        for p in 0..n_patches {
            let px = p % patches_per_dim;
            let py = p / patches_per_dim;
            for ky in 0..2 {
                for kx in 0..2 {
                    for c in 0..64 {
                        let src_idx = p * n_embed + (ky * 2 + kx) * 64 + c;
                        let dst_idx = (py * 2 + ky) * 2 * 64 + (px * 2 + kx) * 64 + c;
                        if src_idx < final_out.len() && dst_idx < result.len() {
                            result[dst_idx] = final_out[src_idx];
                        }
                    }
                }
            }
        }

        Ok(result)
    }

    fn compute_t_embed(&self, sigma: f32) -> Vec<f32> {
        let cfg = &self.model.config;
        let theta = 10000.0f32;

        let freqs = (0..128i32).map(|i| {
            let freq = 1.0 / theta.powf((2 * i) as f32 / 128.0);
            (sigma * freq as f32).sin_cos()
        }).collect::<Vec<_>>();

        let mut embed = vec![0.0f32; 256];
        for i in 0..128 {
            embed[i] = freqs[i].1;
            embed[i + 128] = freqs[i].0;
        }

        let mut h = vec![0.0f32; 1024];
        for i in 0..256 {
            for j in 0..1024 {
                h[j] += embed[i] * self.model.t_embedder_mlp_0_weight[i * 1024 + j];
            }
            h[i % 1024] += self.model.t_embedder_mlp_0_bias[i % 1024];
        }
        for v in h.iter_mut() { *v = silu(*v); }

        let mut out = vec![0.0f32; 256];
        for i in 0..1024 {
            for j in 0..256 {
                out[j] += h[i] * self.model.t_embedder_mlp_2_weight[i * 256 + j];
            }
            out[i % 256] += self.model.t_embedder_mlp_2_bias[i % 256];
        }

        out
    }

    fn gen_z_image_pe(&self, total_seq: usize, context_len: usize, _h_len: usize, _w_len: usize) -> Vec<f32> {
        let cfg = &self.model.config;
        let axes = &cfg.axes_dim;
        let theta = cfg.theta as f32;
        let seq_len = total_seq;

        let mut pe = vec![0.0f32; seq_len * cfg.axes_dim_sum];

        let mut all_freqs: Vec<Vec<Vec<(f32, f32)>>> = Vec::new();
        for axis_idx in 0..3 {
            let dim = axes[axis_idx];
            let freq_base = theta / dim as f32;
            let mut axis_freqs = Vec::with_capacity(seq_len);
            for pos in 0..seq_len {
                let p = if pos < context_len { pos as f32 + 1.0 } else { 0.0 };
                let mut pos_freqs = Vec::with_capacity(dim);
                for i in 0..dim {
                    let theta_freq = freq_base.powf(-(i as f32) / dim as f32);
                    let angle = p * theta_freq;
                    pos_freqs.push(angle.sin_cos());
                }
                axis_freqs.push(pos_freqs);
            }
            all_freqs.push(axis_freqs);
        }

        for pos in 0..seq_len {
            let mut offset = pos * cfg.axes_dim_sum;
            for axis_idx in 0..3 {
                let dim = axes[axis_idx];
                let half = dim / 2;
                for i in 0..half {
                    let (sin, cos) = all_freqs[axis_idx][pos][i];
                    pe[offset] = sin;
                    pe[offset + 1] = cos;
                    offset += 2;
                }
            }
        }

        pe
    }

    fn cap_embedder(&self, text_context: &[f32], context_len: usize) -> Result<Vec<f32>, String> {
        let cfg = &self.model.config;
        let cap_feat_dim = cfg.cap_feat_dim;
        let n_embed = cfg.n_embed;
        let n_tokens = text_context.len() / cap_feat_dim;

        let mut normed = vec![0.0f32; n_tokens * cap_feat_dim];
        for t in 0..n_tokens {
            let mean = text_context[t * cap_feat_dim..(t + 1) * cap_feat_dim].iter().sum::<f32>() / cap_feat_dim as f32;
            let var = text_context[t * cap_feat_dim..(t + 1) * cap_feat_dim].iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / cap_feat_dim as f32;
            let inv_std = 1.0 / (var.sqrt() + 1e-5);
            for i in 0..cap_feat_dim {
                normed[t * cap_feat_dim + i] = (text_context[t * cap_feat_dim + i] - mean) * inv_std * self.model.cap_embedder_0_weight[i];
            }
        }

        let mut out = vec![0.0f32; n_tokens * n_embed];
        let w = &self.model.cap_embedder_1_weight;
        for t in 0..n_tokens {
            for i in 0..n_embed {
                let mut sum = self.model.cap_embedder_1_bias[i];
                for j in 0..cap_feat_dim {
                    let f = w[i * cap_feat_dim + j];
                    sum += normed[t * cap_feat_dim + j] * f;
                }
                out[t * n_embed + i] = sum;
            }
        }

        Ok(out)
    }

    fn x_embedder(&self, patches: &[f32], n_patches: usize) -> Result<Vec<f32>, String> {
        let cfg = &self.model.config;
        let patch_area = cfg.patch_size * cfg.patch_size * cfg.latent_channels;
        let n_embed = cfg.n_embed;
        let mut out = vec![0.0f32; n_patches * n_embed];

        for p in 0..n_patches {
            let patch_data = &patches[p * patch_area..(p + 1) * patch_area];
            let mean = patch_data.iter().sum::<f32>() / patch_area as f32;
            let var = patch_data.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / patch_area as f32;
            let inv_std = 1.0 / (var.sqrt() + 1e-5);

            for i in 0..n_embed {
                let mut sum = self.model.x_embedder_bias[i];
                for j in 0..patch_area {
                    sum += (patch_data[j] - mean) * inv_std * self.model.x_embedder_weight[j + i * patch_area];
                }
                out[p * n_embed + i] = sum;
            }
        }

        Ok(out)
    }

    fn refiner_block(&self, input: &[f32], pe: &[f32], layer: &RefinerLayer, adaln_input: Option<&[f32]>) -> Result<Vec<f32>, String> {
        let cfg = &self.model.config;
        let n_embed = cfg.n_embed;
        let head_dim = cfg.head_dim;
        let n_head = cfg.n_head;
        let n_tokens = input.len() / n_embed;
        let n_head_kv = cfg.n_head_kv;
        let group_size = n_head / n_head_kv;
        let kq_scale = 1.0 / (head_dim as f32).sqrt();

        let mut h = input.to_vec();

        let adaln_mod = if let (Some(w), Some(b), Some(t_emb)) = (layer.adaln_weight.as_deref(), layer.adaln_bias.as_deref(), adaln_input) {
            self.compute_adaln_modulation_from_f32(w, b, t_emb)
        } else {
            vec![1.0f32; n_embed]
        };

        let normed: Vec<f32> = h.chunks_exact(n_embed)
            .flat_map(|chunk| {
                let mean = chunk.iter().sum::<f32>() / n_embed as f32;
                let var = chunk.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n_embed as f32;
                let inv_std = 1.0 / (var.sqrt() + 1e-5);
                chunk.iter().enumerate().map(|(i, &x)| (x - mean) * inv_std * layer.attention_norm1_weight[i]).collect::<Vec<_>>()
            }).collect();

        let mut qkv = vec![0.0f32; n_tokens * n_embed * 3];
        for t in 0..n_tokens {
            let token_input = &normed[t * n_embed..(t + 1) * n_embed];
            let token_qkv = self.matmul_q8_single(&layer.qkv_weight, token_input, n_embed, n_embed * 3);
            let base = t * n_embed * 3;
            qkv[base..base + n_embed * 3].copy_from_slice(&token_qkv);
        }

        let mut q_all = vec![0.0f32; n_tokens * n_head * head_dim];
        let mut k_all = vec![0.0f32; n_tokens * n_head_kv * head_dim];
        let mut v_all = vec![0.0f32; n_tokens * n_head_kv * head_dim];
        for t in 0..n_tokens {
            let row_base = t * n_embed * 3;
            for h_idx in 0..n_head {
                let src = row_base + h_idx * head_dim;
                let dst = t * n_head * head_dim + h_idx * head_dim;
                q_all[dst..dst + head_dim].copy_from_slice(&qkv[src..src + head_dim]);
            }
            for h_idx in 0..n_head_kv {
                let k_src = row_base + n_embed + h_idx * head_dim;
                let v_src = row_base + n_embed * 2 + h_idx * head_dim;
                k_all[t * n_head_kv * head_dim + h_idx * head_dim..t * n_head_kv * head_dim + (h_idx + 1) * head_dim]
                    .copy_from_slice(&qkv[k_src..k_src + head_dim]);
                v_all[t * n_head_kv * head_dim + h_idx * head_dim..t * n_head_kv * head_dim + (h_idx + 1) * head_dim]
                    .copy_from_slice(&qkv[v_src..v_src + head_dim]);
            }
        }

        for t in 0..n_tokens {
            for h_idx in 0..n_head {
                let off = h_idx * head_dim;
                let q_slice = &mut q_all[t * n_head * head_dim + off..t * n_head * head_dim + off + head_dim];
                let pe_slice = &pe[t * cfg.axes_dim_sum..t * cfg.axes_dim_sum + cfg.axes_dim_sum];
                for i in 0..head_dim / 2 {
                    let cos = pe_slice[i * 2];
                    let sin = pe_slice[i * 2 + 1];
                    let cos2 = pe_slice[head_dim + i * 2];
                    let sin2 = pe_slice[head_dim + i * 2 + 1];
                    let q0 = q_slice[i];
                    let q1 = q_slice[i + head_dim / 2];
                    q_slice[i] = q0 * cos + q1 * sin;
                    q_slice[i + head_dim / 2] = q0 * (-sin2) + q1 * cos2;
                }
                let kv_off = (h_idx / group_size) * head_dim;
                let k_slice = &mut k_all[t * n_head_kv * head_dim + kv_off..t * n_head_kv * head_dim + kv_off + head_dim];
                for i in 0..head_dim / 2 {
                    let cos = pe_slice[i * 2];
                    let sin = pe_slice[i * 2 + 1];
                    let cos2 = pe_slice[head_dim + i * 2];
                    let sin2 = pe_slice[head_dim + i * 2 + 1];
                    let k0 = k_slice[i];
                    let k1 = k_slice[i + head_dim / 2];
                    k_slice[i] = k0 * cos + k1 * sin;
                    k_slice[i + head_dim / 2] = k0 * (-sin2) + k1 * cos2;
                }
            }
        }

        for h_idx in 0..n_head {
            let kv_head = h_idx / group_size;
            let q_off_base = h_idx * head_dim;
            let k_off_base = kv_head * head_dim;
            for i in 0..n_tokens {
                let q_off = i * n_head * head_dim + q_off_base;
                let q_slice = &mut q_all[q_off..q_off + head_dim];
                rms_norm_inplace(q_slice, &layer.q_norm_weight, 1e-5);
            }
            for j in 0..n_tokens {
                let k_off = j * n_head_kv * head_dim + k_off_base;
                let k_slice = &mut k_all[k_off..k_off + head_dim];
                rms_norm_inplace(k_slice, &layer.k_norm_weight, 1e-5);
            }
            for i in 0..n_tokens {
                let q_off = i * n_head * head_dim + q_off_base;
                let q_slice = &q_all[q_off..q_off + head_dim];
                let mut max_val = f32::NEG_INFINITY;
                let mut scores = vec![0.0f32; n_tokens];
                for j in 0..n_tokens {
                    let k_off = j * n_head_kv * head_dim + k_off_base;
                    let k_slice = &k_all[k_off..k_off + head_dim];
                    scores[j] = dot_f32(q_slice, k_slice, head_dim) * kq_scale;
                    if scores[j] > max_val { max_val = scores[j]; }
                }
                let mut exp_sum = 0.0f32;
                for j in 0..n_tokens {
                    scores[j] = (scores[j] - max_val).exp();
                    exp_sum += scores[j];
                }
                for j in 0..n_tokens {
                    scores[j] /= exp_sum;
                }
                for d in 0..head_dim {
                    q_all[q_off + d] = 0.0;
                }
                for j in 0..n_tokens {
                    let v_off = j * n_head_kv * head_dim + k_off_base;
                    let v_slice = &v_all[v_off..v_off + head_dim];
                    for d in 0..head_dim {
                        q_all[q_off + d] += scores[j] * v_slice[d];
                    }
                }
            }
        }

        let mut attn_out = vec![0.0f32; n_tokens * n_embed];
        for t in 0..n_tokens {
            for h_idx in 0..n_head {
                let src = t * n_head * head_dim + h_idx * head_dim;
                let dst = t * n_embed + h_idx * head_dim;
                attn_out[dst..dst + head_dim].copy_from_slice(&q_all[src..src + head_dim]);
            }
        }

        let attn_proj = self.matmul_q8_single(&layer.out_weight, &attn_out, n_embed, n_embed);

        let modulated: Vec<f32> = h.iter().zip(adaln_mod.iter()).map(|(&x, &a)| x + x * a).collect();

        let ffn_out = self.apply_ffn_q8(&modulated, &layer.w1_weight, &layer.w2_weight, &layer.w3_weight, n_embed, cfg.n_ff)?;

        let result: Vec<f32> = modulated.iter().zip(ffn_out.iter()).map(|(m, f)| m + f).collect();
        Ok(result)
    }

    fn pig_layer_block(&self, input: &[f32], n_tokens: usize, pe: &[f32], t_embed: &[f32], layer: &PigLayer) -> Result<Vec<f32>, String> {
        let cfg = &self.model.config;
        let n_embed = cfg.n_embed;
        let head_dim = cfg.head_dim;
        let n_head = cfg.n_head;
        let n_head_kv = cfg.n_head_kv;
        let group_size = n_head / n_head_kv;
        let kq_scale = 1.0 / (head_dim as f32).sqrt();

        let adaln_mod = self.compute_adaln_modulation_q8(&layer.adaln_weight, &layer.adaln_bias, t_embed);

        let normed: Vec<f32> = input.chunks_exact(n_embed)
            .flat_map(|chunk| {
                let mean = chunk.iter().sum::<f32>() / n_embed as f32;
                let var = chunk.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n_embed as f32;
                let inv_std = 1.0 / (var.sqrt() + 1e-5);
                chunk.iter().enumerate().map(|(i, &x)| (x - mean) * inv_std * layer.attention_norm1_weight[i]).collect::<Vec<_>>()
            }).collect();

        let mut qkv = vec![0.0f32; n_tokens * n_embed * 3];
        for t in 0..n_tokens {
            let token_input = &normed[t * n_embed..(t + 1) * n_embed];
            let token_qkv = self.matmul_q8_single(&layer.qkv_weight, token_input, n_embed, n_embed * 3);
            let base = t * n_embed * 3;
            qkv[base..base + n_embed * 3].copy_from_slice(&token_qkv);
        }

        let mut q_all = vec![0.0f32; n_tokens * n_head * head_dim];
        let mut k_all = vec![0.0f32; n_tokens * n_head_kv * head_dim];
        let mut v_all = vec![0.0f32; n_tokens * n_head_kv * head_dim];
        for t in 0..n_tokens {
            let row_base = t * n_embed * 3;
            for h_idx in 0..n_head {
                let src = row_base + h_idx * head_dim;
                let dst = t * n_head * head_dim + h_idx * head_dim;
                q_all[dst..dst + head_dim].copy_from_slice(&qkv[src..src + head_dim]);
            }
            for h_idx in 0..n_head_kv {
                let k_src = row_base + n_embed + h_idx * head_dim;
                let v_src = row_base + n_embed * 2 + h_idx * head_dim;
                k_all[t * n_head_kv * head_dim + h_idx * head_dim..t * n_head_kv * head_dim + (h_idx + 1) * head_dim]
                    .copy_from_slice(&qkv[k_src..k_src + head_dim]);
                v_all[t * n_head_kv * head_dim + h_idx * head_dim..t * n_head_kv * head_dim + (h_idx + 1) * head_dim]
                    .copy_from_slice(&qkv[v_src..v_src + head_dim]);
            }
        }

        for t in 0..n_tokens {
            for h_idx in 0..n_head {
                let off = h_idx * head_dim;
                let q_slice = &mut q_all[t * n_head * head_dim + off..t * n_head * head_dim + off + head_dim];
                let pe_slice = &pe[t * cfg.axes_dim_sum..t * cfg.axes_dim_sum + cfg.axes_dim_sum];
                for i in 0..head_dim / 2 {
                    let cos = pe_slice[i * 2];
                    let sin = pe_slice[i * 2 + 1];
                    let cos2 = pe_slice[head_dim + i * 2];
                    let sin2 = pe_slice[head_dim + i * 2 + 1];
                    let q0 = q_slice[i];
                    let q1 = q_slice[i + head_dim / 2];
                    q_slice[i] = q0 * cos + q1 * sin;
                    q_slice[i + head_dim / 2] = q0 * (-sin2) + q1 * cos2;
                }
                let kv_off = (h_idx / group_size) * head_dim;
                let k_slice = &mut k_all[t * n_head_kv * head_dim + kv_off..t * n_head_kv * head_dim + kv_off + head_dim];
                for i in 0..head_dim / 2 {
                    let cos = pe_slice[i * 2];
                    let sin = pe_slice[i * 2 + 1];
                    let cos2 = pe_slice[head_dim + i * 2];
                    let sin2 = pe_slice[head_dim + i * 2 + 1];
                    let k0 = k_slice[i];
                    let k1 = k_slice[i + head_dim / 2];
                    k_slice[i] = k0 * cos + k1 * sin;
                    k_slice[i + head_dim / 2] = k0 * (-sin2) + k1 * cos2;
                }
            }
        }

        for h_idx in 0..n_head {
            let kv_head = h_idx / group_size;
            let q_off_base = h_idx * head_dim;
            let k_off_base = kv_head * head_dim;
            for i in 0..n_tokens {
                let q_off = i * n_head * head_dim + q_off_base;
                let q_slice = &mut q_all[q_off..q_off + head_dim];
                rms_norm_inplace(q_slice, &layer.q_norm_weight, 1e-5);
            }
            for j in 0..n_tokens {
                let k_off = j * n_head_kv * head_dim + k_off_base;
                let k_slice = &mut k_all[k_off..k_off + head_dim];
                rms_norm_inplace(k_slice, &layer.k_norm_weight, 1e-5);
            }
            for i in 0..n_tokens {
                let q_off = i * n_head * head_dim + q_off_base;
                let q_slice = &q_all[q_off..q_off + head_dim];
                let mut max_val = f32::NEG_INFINITY;
                let mut scores = vec![0.0f32; n_tokens];
                for j in 0..n_tokens {
                    let k_off = j * n_head_kv * head_dim + k_off_base;
                    let k_slice = &k_all[k_off..k_off + head_dim];
                    scores[j] = dot_f32(q_slice, k_slice, head_dim) * kq_scale;
                    if scores[j] > max_val { max_val = scores[j]; }
                }
                let mut exp_sum = 0.0f32;
                for j in 0..n_tokens {
                    scores[j] = (scores[j] - max_val).exp();
                    exp_sum += scores[j];
                }
                for j in 0..n_tokens {
                    scores[j] /= exp_sum;
                }
                let q_slice = &mut q_all[i * n_head * head_dim + q_off_base..i * n_head * head_dim + q_off_base + head_dim];
                for d in 0..head_dim {
                    q_slice[d] = 0.0;
                }
                for j in 0..n_tokens {
                    let v_off = j * n_head_kv * head_dim + k_off_base;
                    let v_slice = &v_all[v_off..v_off + head_dim];
                    for d in 0..head_dim {
                        q_slice[d] += scores[j] * v_slice[d];
                    }
                }
            }
        }

        let mut attn_out = vec![0.0f32; n_tokens * n_embed];
        for t in 0..n_tokens {
            for h_idx in 0..n_head {
                attn_out[t * n_embed + h_idx * head_dim..t * n_embed + (h_idx + 1) * head_dim]
                    .copy_from_slice(&q_all[t * n_head * head_dim + h_idx * head_dim..t * n_head * head_dim + (h_idx + 1) * head_dim]);
            }
        }

        let attn_proj = self.matmul_q8_single(&layer.out_weight, &attn_out, n_embed, n_embed);

        let scale_msa = &adaln_mod[..n_embed];
        let gate_msa = &adaln_mod[n_embed..2 * n_embed];
        let scale_mlp = &adaln_mod[2 * n_embed..3 * n_embed];
        let gate_mlp = &adaln_mod[3 * n_embed..];

        let modulated: Vec<f32> = input.iter().zip(scale_msa.iter()).map(|(&x, &s)| x * (1.0 + s)).collect();
        let gate_tanh: Vec<f32> = gate_msa.iter().map(|&g| g.tanh()).collect();
        let attn_gated: Vec<f32> = modulated.iter().zip(gate_tanh.iter()).map(|(m, g)| m * g).collect();
        let residual: Vec<f32> = input.iter().zip(attn_gated.iter()).map(|(&x, &a)| x + a).collect();

        let ffn_normed: Vec<f32> = residual.chunks_exact(n_embed)
            .flat_map(|chunk| {
                let mean = chunk.iter().sum::<f32>() / n_embed as f32;
                let var = chunk.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n_embed as f32;
                let inv_std = 1.0 / (var.sqrt() + 1e-5);
                chunk.iter().enumerate().map(|(i, &x)| (x - mean) * inv_std * layer.ffn_norm1_weight[i]).collect::<Vec<_>>()
            }).collect();

        let ffn_out = self.apply_ffn_q8(&ffn_normed, &layer.w1_weight, &layer.w2_weight, &layer.w3_weight, n_embed, cfg.n_ff)?;

        let scale_mlp_tanh: Vec<f32> = scale_mlp.iter().map(|&s| s.tanh()).collect();
        let ffn_gated: Vec<f32> = ffn_out.iter().zip(scale_mlp_tanh.iter()).map(|(f, s)| f * s).collect();
        let final_out: Vec<f32> = residual.iter().zip(ffn_gated.iter()).map(|(&r, &f)| r + f).collect();

        Ok(final_out)
    }

    fn final_layer(&self, input: &[f32], t_embed: &[f32]) -> Result<Vec<f32>, String> {
        let cfg = &self.model.config;
        let n_embed = cfg.n_embed;
        let n_patches = input.len() / n_embed;

        let silu_t: Vec<f32> = t_embed.iter().map(|&x| silu(x)).collect();
        let scale: Vec<f32> = silu_t.iter().enumerate().map(|(i, &s)| {
            let mut sum = self.model.final_layer_adaln_bias[i];
            for j in 0..256 {
                let f = self.model.final_layer_adaln_weight[i + j * 256];
                sum += s * f;
            }
            sum
        }).collect();

        let mean_all = input.iter().sum::<f32>() / (n_patches * n_embed) as f32;
        let var_all = input.iter().map(|v| (v - mean_all) * (v - mean_all)).sum::<f32>() / (n_patches * n_embed) as f32;
        let inv_std_all = 1.0 / (var_all.sqrt() + 1e-6);

        let normed: Vec<f32> = input.iter().map(|&x| (x - mean_all) * inv_std_all).collect();

        let modulated: Vec<f32> = normed.iter().enumerate().map(|(i, &x)| x * (1.0 + scale[i % n_embed])).collect();

        let w = &self.model.final_layer_linear_weight;
        let mut out = vec![0.0f32; n_patches * 64];
        for p in 0..n_patches {
            for i in 0..64 {
                let mut sum = self.model.final_layer_linear_bias[i];
                for j in 0..n_embed {
                    let f = w[j + i * n_embed];
                    sum += modulated[p * n_embed + j] * f;
                }
                out[p * 64 + i] = sum;
            }
        }

        Ok(out)
    }

    fn compute_adaln_modulation_q8(&self, weight: &[u8], bias: &[f32], t_embed: &[f32]) -> Vec<f32> {
        let cfg = &self.model.config;
        let n_embed = cfg.n_embed;
        let mut input = vec![0.0f32; 256];
        input[..256].copy_from_slice(&t_embed[..256]);
        let mut tmp = vec![0.0f32; n_embed * 4];
        let tmp2 = self.matmul_q8_single(weight, &input, 256, n_embed * 4);
        for i in 0..tmp2.len() { tmp[i] = tmp2[i]; }
        let mut mod_vals = vec![0.0f32; n_embed * 4];
        for i in 0..n_embed * 4 {
            mod_vals[i] = 1.0 + bias[i] + tmp[i];
        }
        mod_vals
    }

    fn compute_adaln_modulation_from_f32(&self, weight: &[f32], bias: &[f32], t_embed: &[f32]) -> Vec<f32> {
        let cfg = &self.model.config;
        let n_embed = cfg.n_embed;
        let mut input = vec![0.0f32; 256];
        input[..256].copy_from_slice(&t_embed[..256]);
        let mut tmp = vec![0.0f32; n_embed * 4];
        for i in 0..n_embed * 4 {
            tmp[i] = bias[i];
            for j in 0..256 {
                tmp[i] += input[j] * weight[i * 256 + j];
            }
        }
        let mut mod_vals = vec![0.0f32; n_embed * 4];
        for i in 0..n_embed * 4 {
            mod_vals[i] = 1.0 + tmp[i];
        }
        mod_vals
    }

    fn matmul_q8_single(&self, weight: &[u8], input: &[f32], n_in: usize, n_out: usize) -> Vec<f32> {
        let blocks = (n_in + 31) / 32;
        let mut q8_buf = vec![0u8; n_in];
        let mut scale_buf = vec![0.0f32; blocks];
        quantize_q8_0_into(input, n_in, &mut q8_buf, &mut scale_buf);
        let mut output = vec![0.0f32; n_out];
        let weight_len = weight.len();
        let pool = Arc::clone(&self.model.pool);
        let weight_usize = weight.as_ptr() as usize;
        let q8_usize = q8_buf.as_ptr() as usize;
        let scale_usize = scale_buf.as_ptr() as usize;
        let out_usize = output.as_mut_ptr() as usize;
        pool.compute(move |thread, threads| {
            let w = unsafe { std::slice::from_raw_parts(weight_usize as *const u8, weight_len) };
            let q = unsafe { std::slice::from_raw_parts(q8_usize as *const u8, n_in) };
            let s = unsafe { std::slice::from_raw_parts(scale_usize as *const f32, blocks) };
            let o = unsafe { std::slice::from_raw_parts_mut(out_usize as *mut f32, n_out) };
            matmul_q8_0_quantized_parallel_rows(w, q, s, o, n_in, n_out, thread, threads);
        });
        output
    }

    fn matmul_f16_f32(&self, weight: &[u8], input: &[f32], n_in: usize, n_out: usize) -> Vec<f32> {
        let mut output = vec![0.0f32; n_out];
        let w_ptr = weight.as_ptr() as usize;
        for out_idx in 0..n_out {
            let mut sum = 0.0f32;
            for in_idx in 0..n_in {
                let byte_offset = (in_idx + out_idx * n_in) * 2;
                let w_bits = unsafe {
                    let ptr = (w_ptr + byte_offset) as *const u8;
                    let bits = *ptr as u16 | ((*ptr.add(1) as u16) << 8);
                    f16_to_f32(bits)
                };
                sum += input[in_idx] * w_bits;
            }
            output[out_idx] = sum;
        }
        output
    }

    fn apply_ffn_q8(&self, input: &[f32], w1: &[u8], w2: &[u8], w3: &[u8], in_dim: usize, hidden_dim: usize) -> Result<Vec<f32>, String> {
        let gate = self.matmul_q8_single(w1, input, in_dim, hidden_dim);
        let up = self.matmul_q8_single(w3, input, in_dim, hidden_dim);
        let gated: Vec<f32> = gate.iter().zip(up.iter()).map(|(g, u)| g * silu(*g) * u).collect();
        let down = self.matmul_q8_single(w2, &gated, hidden_dim, in_dim);
        Ok(down)
    }

    fn decode_patches(&self, patches: &[f32], n_patches: usize, latent_size: usize) -> Result<Vec<u8>, String> {
        if let Some(vae) = self.vae {
            return vae.decode(patches, n_patches, latent_size);
        }
        let cfg = &self.model.config;
        let patches_per_dim = latent_size / cfg.patch_size;
        let mut latents = vec![0.0f32; latent_size * latent_size * cfg.latent_channels];
        for p in 0..n_patches {
            let patch_offset = p * cfg.n_embed;
            let px = p % patches_per_dim;
            let py = p / patches_per_dim;
            let start_x = px * cfg.patch_size;
            let start_y = py * cfg.patch_size;
            for ky in 0..cfg.patch_size {
                for kx in 0..cfg.patch_size {
                    for c in 0..cfg.latent_channels {
                        let latent_idx = (start_y + ky) * latent_size * cfg.latent_channels + (start_x + kx) * cfg.latent_channels + c;
                        let patch_elem = ky * cfg.patch_size * cfg.latent_channels + kx * cfg.latent_channels + c;
                        if patch_elem < cfg.n_embed && latent_idx < latents.len() {
                            latents[latent_idx] = patches[patch_offset + patch_elem];
                        }
                    }
                }
            }
        }
        let mut pixels = Vec::with_capacity(latent_size * latent_size * 4);
        for &latent in latents.iter().take(latent_size * latent_size) {
            let c = ((latent.clamp(-1.0, 1.0) * 127.5 + 128.0) as u8).min(255);
            pixels.push(c); pixels.push(c); pixels.push(c); pixels.push(255);
        }
        Ok(pixels)
    }
}

fn load_f32(source: &dyn TensorSource, name: &str) -> Result<Vec<f32>, String> {
    let data = source.tensor_slice(name).ok_or_else(|| format!("Missing tensor: {}", name))?;
    let n_el = data.len() / 4;
    let mut out = Vec::with_capacity(n_el);
    for i in 0..n_el {
        let off = i * 4;
        out.push(f32::from_le_bytes([data[off], data[off+1], data[off+2], data[off+3]]));
    }
    Ok(out)
}

fn load_f16_as_f32(source: &dyn TensorSource, name: &str) -> Result<Vec<f32>, String> {
    let data = source.tensor_slice(name).ok_or_else(|| format!("Missing tensor: {}", name))?;
    let n_el = data.len() / 2;
    let mut out = Vec::with_capacity(n_el);
    for i in 0..n_el {
        let off = i * 2;
        let bits = data[off] as u16 | ((data[off + 1] as u16) << 8);
        let f = f32::from_bits((bits as f32).to_bits());
        out.push(f);
    }
    Ok(out)
}

fn load_q8_0(source: &dyn TensorSource, name: &str) -> Result<Vec<u8>, String> {
    let data = source.tensor_slice(name).ok_or_else(|| format!("Missing tensor: {}", name))?;
    Ok(data.to_vec())
}
