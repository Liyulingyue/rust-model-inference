use std::sync::Arc;

use crate::model::{GGMLType, TensorSource};
use crate::ops::{matmul_q8_0_quantized_parallel_rows, quantize_q8_0_into};
use crate::thread_pool::ComputePool;

#[derive(Debug)]
pub struct PigConfig {
    pub n_layer: usize,
    pub n_embed: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub n_ff: usize,
    pub x_embed_dim: usize,
    pub image_size: usize,
    pub latent_channels: usize,
    pub patch_size: usize,
}

impl PigConfig {
    pub fn from_source<S: crate::model::TensorSource + ?Sized>(source: &S) -> Result<Self, String> {
        let n_layer = source
            .metadata("pig.block_count")
            .and_then(|v| v.to_u64())
            .map(|v| v as usize)
            .unwrap_or(30);
        let n_embed = source
            .metadata("pig.embedding_length")
            .and_then(|v| v.to_u64())
            .map(|v| v as usize)
            .unwrap_or(3840);
        let n_head = source
            .metadata("pig.attention.head_count")
            .and_then(|v| v.to_u64())
            .map(|v| v as usize)
            .unwrap_or(16);
        let n_head_kv = source
            .metadata("pig.attention.head_count_kv")
            .and_then(|v| v.to_u64())
            .map(|v| v as usize)
            .unwrap_or(16);
        let n_ff = source
            .metadata("pig.feed_forward_length")
            .and_then(|v| v.to_u64())
            .map(|v| v as usize)
            .unwrap_or(10240);
        Ok(Self {
            n_layer,
            n_embed,
            n_head,
            n_head_kv,
            n_ff,
            x_embed_dim: 256,
            image_size: 512,
            latent_channels: 16,
            patch_size: 2,
        })
    }
}

pub struct PigModel {
    config: PigConfig,
    pool: Arc<ComputePool>,
    x_embedder_weight: &'static [u8],
    t_embedder_mlp_0_weight: &'static [u8],
    t_embedder_mlp_0_bias: Vec<f32>,
    layers: Vec<PigLayer>,
    final_layer_adaln_weight: &'static [u8],
    final_layer_adaln_bias: Vec<f32>,
    final_layer_linear_weight: &'static [u8],
}

#[derive(Clone)]
struct PigLayer {
    adaln_weight: &'static [u8],
    adaln_bias: Vec<f32>,
    qkv_weight: &'static [u8],
    out_weight: &'static [u8],
    q_norm_weight: Vec<f32>,
    k_norm_weight: Vec<f32>,
    attention_norm1_weight: Vec<f32>,
    attention_norm2_weight: Vec<f32>,
    w1_weight: &'static [u8],
    w2_weight: &'static [u8],
    w3_weight: &'static [u8],
    ffn_norm1_weight: Vec<f32>,
    ffn_norm2_weight: Vec<f32>,
}

impl PigModel {
    pub fn from_source(
        source: Arc<dyn crate::model::TensorSource>,
        pool: Arc<ComputePool>,
    ) -> Result<Self, String> {
        let config = PigConfig::from_source(source.as_ref())?;

        let x_embedder_weight = static_tensor(source.as_ref(), "x_embedder.weight", &[256, 3840], GGMLType::Q8_0)?;
        let t_embedder_mlp_0_weight = static_tensor(source.as_ref(), "t_embedder.mlp.0.weight", &[256, 1024], GGMLType::Q8_0)?;
        let t_embedder_mlp_0_bias = load_f32(source.as_ref(), "t_embedder.mlp.0.bias")?;

        let mut layers = Vec::with_capacity(config.n_layer);
        for i in 0..config.n_layer {
            let n_embd_dim = [3840u64];
            let n_ff_dim = [10240u64];
            let n_adaln_dim = [256, 15360];
            layers.push(PigLayer {
                adaln_weight: static_tensor(source.as_ref(), &format!("layers.{}.adaLN_modulation.0.weight", i), &n_adaln_dim, GGMLType::Q8_0)?,
                adaln_bias: load_f32(source.as_ref(), &format!("layers.{}.adaLN_modulation.0.bias", i))?,
                qkv_weight: static_tensor(source.as_ref(), &format!("layers.{}.attention.qkv.weight", i), &[3840, 11520], GGMLType::Q8_0)?,
                out_weight: static_tensor(source.as_ref(), &format!("layers.{}.attention.out.weight", i), &[3840, 3840], GGMLType::Q8_0)?,
                q_norm_weight: load_f32(source.as_ref(), &format!("layers.{}.attention.q_norm.weight", i))?,
                k_norm_weight: load_f32(source.as_ref(), &format!("layers.{}.attention.k_norm.weight", i))?,
                attention_norm1_weight: load_f32(source.as_ref(), &format!("layers.{}.attention_norm1.weight", i))?,
                attention_norm2_weight: load_f32(source.as_ref(), &format!("layers.{}.attention_norm2.weight", i))?,
                w1_weight: static_tensor(source.as_ref(), &format!("layers.{}.feed_forward.w1.weight", i), &n_ff_dim, GGMLType::Q8_0)?,
                w2_weight: static_tensor(source.as_ref(), &format!("layers.{}.feed_forward.w2.weight", i), &[10240, 3840], GGMLType::Q8_0)?,
                w3_weight: static_tensor(source.as_ref(), &format!("layers.{}.feed_forward.w3.weight", i), &n_ff_dim, GGMLType::Q8_0)?,
                ffn_norm1_weight: load_f32(source.as_ref(), &format!("layers.{}.ffn_norm1.weight", i))?,
                ffn_norm2_weight: load_f32(source.as_ref(), &format!("layers.{}.ffn_norm2.weight", i))?,
            });
        }

        let final_layer_adaln_weight = static_tensor(source.as_ref(), "final_layer.adaLN_modulation.1.weight", &[256, 15360], GGMLType::Q8_0)?;
        let final_layer_adaln_bias = load_f32(source.as_ref(), "final_layer.adaLN_modulation.1.bias")?;
        let final_layer_linear_weight = static_tensor(source.as_ref(), "final_layer.linear.weight", &[3840, 256], GGMLType::Q8_0)?;

        Ok(Self {
            config,
            pool,
            x_embedder_weight,
            t_embedder_mlp_0_weight,
            t_embedder_mlp_0_bias,
            layers,
            final_layer_adaln_weight,
            final_layer_adaln_bias,
            final_layer_linear_weight,
        })
    }

    pub fn config(&self) -> &PigConfig {
        &self.config
    }

    pub fn pool(&self) -> &ComputePool {
        &self.pool
    }
}

fn static_tensor(
    source: &dyn TensorSource,
    name: &str,
    _dims: &[u64],
    _ggml_type: GGMLType,
) -> Result<&'static [u8], String> {
    let bytes = source.tensor_slice(name).ok_or_else(|| format!("Missing tensor: {}", name))?;
    let info = source.tensor_info(name).ok_or_else(|| format!("Missing tensor info: {}", name))?;
    eprintln!("Loading {}: dims={:?}, type={:?}", name, info.dims, info.ggml_type);
    // SAFETY: PigModel stores a strong Arc to this immutable TensorSource and never unloads.
    Ok(unsafe { std::mem::transmute::<&[u8], &'static [u8]>(bytes) })
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

pub struct PigVAE;

impl PigVAE {
    pub fn decode(&self, _latents: &[f32]) -> Result<Vec<u8>, String> {
        Ok(vec![0u8; 512 * 512 * 4])
    }
}

pub struct PigSession<'a> {
    model: &'a PigModel,
}

impl<'a> PigSession<'a> {
    pub fn new(model: &'a PigModel, _max_resolution: usize) -> Result<Self, String> {
        Ok(Self { model })
    }

    pub fn generate_image(&mut self, _prompt: &str, steps: usize) -> Result<Vec<u8>, String> {
        let config = &self.model.config;
        let latent_size = config.image_size / 8;
        let patches_per_dim = latent_size / config.patch_size;
        let n_patches = patches_per_dim * patches_per_dim;
        let n_embed = config.n_embed;

        let mut patches: Vec<f32> = vec![0.0; n_patches * n_embed];

        let timesteps: Vec<f32> = (0..steps).map(|t| 1.0 - t as f32 / steps as f32).collect();

        for (step_idx, &t) in timesteps.iter().enumerate() {
            eprintln!("Denoising step {}/{}", step_idx + 1, steps);
            let t_emb = self.compute_t_embed(t);

            for layer_idx in 0..config.n_layer {
                eprintln!("  Layer {}/{}", layer_idx + 1, config.n_layer);
                self.apply_transformer_block(&mut patches, n_patches, &t_emb, layer_idx);
            }

            self.apply_final_layer(&mut patches, &t_emb);
        }

        let pixels = self.decode_patches(&patches, n_patches, latent_size)?;

        Ok(pixels)
    }

    fn compute_t_embed(&self, t: f32) -> Vec<f32> {
        let n_embed = self.model.config.n_embed;
        let t_embed_out = 1024;

        let mut h = vec![t; 256];
        let mut output = vec![0.0f32; t_embed_out];

        self.matmul_q8(&self.model.t_embedder_mlp_0_weight, &h, 256, t_embed_out, &mut output);

        for i in 0..t_embed_out {
            output[i] += self.model.t_embedder_mlp_0_bias[i];
        }

        output
    }

    fn matmul_q8(&self, weight: &[u8], input: &[f32], n_in: usize, n_out: usize, output: &mut [f32]) {
        let blocks = (n_in + 31) / 32;
        let mut q8_buf = vec![0u8; n_in];
        let mut scale_buf = vec![0.0f32; blocks];
        quantize_q8_0_into(input, n_in, &mut q8_buf, &mut scale_buf);

        let weight_ptr = weight.as_ptr();
        let q8_ptr = q8_buf.as_ptr();
        let scale_ptr = scale_buf.as_ptr();
        let out_ptr = output.as_mut_ptr();

        let pool = Arc::clone(&self.model.pool);
        pool.compute(move |thread, threads| {
            let weight = unsafe { std::slice::from_raw_parts(weight_ptr, weight.len()) };
            let q8 = unsafe { std::slice::from_raw_parts(q8_ptr, n_in) };
            let scales = unsafe { std::slice::from_raw_parts(scale_ptr, blocks) };
            let output = unsafe { std::slice::from_raw_parts_mut(out_ptr, n_out) };
            matmul_q8_0_quantized_parallel_rows(
                weight,
                q8,
                scales,
                output,
                n_in,
                n_out,
                thread,
                threads,
            );
        });
    }

    fn apply_transformer_block(&self, patches: &mut [f32], n_patches: usize, t_emb: &[f32], layer_idx: usize) {
        let config = &self.model.config;
        let layer = &self.model.layers[layer_idx];
        let n_embed = config.n_embed;

        let adaln_mod = self.compute_adaln_modulation(&layer.adaln_weight, &layer.adaln_bias, t_emb);

        let mut h = patches.to_vec();

        for patch_idx in 0..n_patches {
            let patch_offset = patch_idx * n_embed;

            let residual: Vec<f32> = h[patch_offset..patch_offset + n_embed].to_vec();

            let x_normed = self.layer_norm(&residual, &layer.attention_norm1_weight);

            let mut qkv = vec![0.0f32; 3 * n_embed];
            self.matmul_q8(&layer.qkv_weight, &x_normed, n_embed, 3 * n_embed, &mut qkv);

            let head_dim = n_embed / config.n_head;
            let q = &qkv[..n_embed];
            let k = &qkv[n_embed..2*n_embed];
            let v = &qkv[2*n_embed..];

            let q_normed = q.to_vec();
            let k_normed = k.to_vec();

            let attn_out = self.compute_attention(&q_normed, &k_normed, v, config.n_head, head_dim);

            let mut attn_proj = vec![0.0f32; n_embed];
            self.matmul_q8(&layer.out_weight, &attn_out, n_embed, n_embed, &mut attn_proj);

            let attn_normed = self.layer_norm(&attn_proj, &layer.attention_norm2_weight);

            let ffn_input = self.layer_norm(&residual, &layer.ffn_norm1_weight);

            let ffn_out = self.apply_ffn(&ffn_input, &layer.w1_weight, &layer.w2_weight, &layer.w3_weight, n_embed, config.n_ff);

            let ffn_normed = self.layer_norm(&ffn_out, &layer.ffn_norm2_weight);

            for i in 0..n_embed {
                h[patch_offset + i] = residual[i] + adaln_mod[i] * (attn_normed[i] + ffn_normed[i]);
            }
        }

        patches.copy_from_slice(&h);
    }

    fn apply_final_layer(&self, patches: &mut [f32], t_emb: &[f32]) {
        let config = &self.model.config;
        let n_embed = config.n_embed;
        let n_patches = patches.len() / n_embed;
        let final_out_dim = 64;

        let adaln_mod = self.compute_adaln_modulation(
            &self.model.final_layer_adaln_weight,
            &self.model.final_layer_adaln_bias,
            t_emb,
        );

        let mut h = patches.to_vec();

        for patch_idx in 0..n_patches {
            let patch_offset = patch_idx * n_embed;

            let residual: Vec<f32> = h[patch_offset..patch_offset + n_embed].to_vec();

            let mut out = vec![0.0f32; final_out_dim];
            self.matmul_q8(&self.model.final_layer_linear_weight, &residual, n_embed, final_out_dim, &mut out);

            for i in 0..final_out_dim {
                h[patch_offset + i] = residual[i] + adaln_mod[i] * out[i];
            }
        }

        patches.copy_from_slice(&h);
    }

    fn compute_adaln_modulation(&self, weight: &[u8], bias: &[f32], t_emb: &[f32]) -> Vec<f32> {
        let n_embed = self.model.config.n_embed;

        let mut input = vec![0.0f32; 256];
        for i in 0..256 {
            input[i] = t_emb[i % t_emb.len()];
        }

        let mut tmp = vec![0.0f32; n_embed];
        self.matmul_q8(weight, &input, 256, n_embed, &mut tmp);

        let mut mod_vals = vec![0.0f32; n_embed];
        for i in 0..n_embed {
            mod_vals[i] = 1.0 + bias[i] + tmp[i];
        }
        mod_vals
    }

    fn layer_norm(&self, input: &[f32], weight: &[f32]) -> Vec<f32> {
        let n = input.len();
        let mean = input.iter().sum::<f32>() / n as f32;
        let var = input.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n as f32;
        let inv_std = 1.0 / (var.sqrt() + 1e-5);
        let mut output = vec![0.0f32; n];
        for i in 0..n {
            output[i] = (input[i] - mean) * inv_std * weight[i];
        }
        output
    }

    fn compute_attention(&self, q: &[f32], k: &[f32], v: &[f32], n_head: usize, head_dim: usize) -> Vec<f32> {
        use crate::ops::dot_f32;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut scores = vec![0.0f32; n_head * head_dim];

        for h in 0..n_head {
            for i in 0..head_dim {
                let q_row = &q[h * head_dim..h * head_dim + head_dim];
                let k_row = &k[h * head_dim..h * head_dim + head_dim];
                scores[h * head_dim + i] = dot_f32(q_row, k_row, head_dim) * scale;
            }
        }

        for h in 0..n_head {
            let h_offset = h * head_dim;
            let mut max_val = f32::MIN;
            for i in 0..head_dim {
                if scores[h_offset + i] > max_val {
                    max_val = scores[h_offset + i];
                }
            }
            let mut exp_sum = 0.0f32;
            for i in 0..head_dim {
                scores[h_offset + i] = (scores[h_offset + i] - max_val).exp();
                exp_sum += scores[h_offset + i];
            }
            for i in 0..head_dim {
                scores[h_offset + i] /= exp_sum;
            }
        }

        let mut final_out = vec![0.0f32; n_head * head_dim];
        for h in 0..n_head {
            for i in 0..head_dim {
                let mut sum = 0.0f32;
                for j in 0..head_dim {
                    sum += scores[h * head_dim + j] * v[h * head_dim + i];
                }
                final_out[h * head_dim + i] = sum;
            }
        }

        final_out
    }

    fn apply_ffn(&self, input: &[f32], w1: &[u8], w2: &[u8], w3: &[u8], in_dim: usize, hidden_dim: usize) -> Vec<f32> {
        let mut gate = vec![0.0f32; hidden_dim];
        self.matmul_q8(w1, input, in_dim, hidden_dim, &mut gate);

        let mut up = vec![0.0f32; hidden_dim];
        self.matmul_q8(w3, input, in_dim, hidden_dim, &mut up);

        for i in 0..hidden_dim {
            let g = gate[i];
            gate[i] = g * (1.0f32 / (1.0f32 + (-g).exp()));
            gate[i] *= up[i];
        }

        let mut down = vec![0.0f32; in_dim];
        self.matmul_q8(w2, &gate, hidden_dim, in_dim, &mut down);

        down
    }

    fn decode_patches(&self, patches: &[f32], n_patches: usize, latent_size: usize) -> Result<Vec<u8>, String> {
        let config = &self.model.config;
        let patch_size = config.patch_size;
        let latent_channels = config.latent_channels;

        let patches_per_dim = (latent_size / patch_size) as usize;
        let mut latents = vec![0.0f32; latent_size * latent_size * latent_channels];

        for patch_idx in 0..n_patches {
            let patch_offset = patch_idx * config.n_embed;
            let px = patch_idx % patches_per_dim;
            let py = patch_idx / patches_per_dim;

            let start_x = px * patch_size;
            let start_y = py * patch_size;

            for ky in 0..patch_size as usize {
                for kx in 0..patch_size as usize {
                    for c in 0..latent_channels as usize {
                        let latent_idx = (start_y + ky) * latent_size * latent_channels
                                        + (start_x + kx) * latent_channels
                                        + c;
                        let patch_elem = ky * patch_size as usize * latent_channels as usize
                                        + kx * latent_channels as usize + c;
                        if patch_elem < config.n_embed && latent_idx < latents.len() {
                            latents[latent_idx] = patches[patch_offset + patch_elem];
                        }
                    }
                }
            }
        }

        let mut pixels = Vec::with_capacity(latent_size * latent_size * 4);
        for &latent in latents.iter().take(latent_size * latent_size) {
            let c = ((latent.clamp(-1.0, 1.0) * 127.5 + 128.0) as u8).min(255);
            pixels.push(c);
            pixels.push(c);
            pixels.push(c);
            pixels.push(255);
        }

        Ok(pixels)
    }
}
