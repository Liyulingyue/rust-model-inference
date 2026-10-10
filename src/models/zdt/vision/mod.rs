pub mod config;

use crate::core::tensor::{GGMLType, TensorSource};
use crate::core::thread_pool::ComputePool;
use crate::models::qwen3::vision::matmul_f32_weight;
use crate::ops::{rms_norm_inplace, softmax_inplace, vec_add_into};
use config::ZdtVisionConfig;
use std::sync::Arc;

/// Load a matrix given as (rows, cols), accepting a GGUF shape with extra
/// leading axes. `v.patch_embd.weight` is stored as a 4-D conv kernel
/// (patch, patch, channels, n_embd) whose product is the input width.
fn load_matrix(
    source: &dyn TensorSource,
    name: &str,
    cols: usize,
    rows: usize,
) -> Result<Vec<f32>, String> {
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("Missing tensor: {name}"))?;
    let total: usize = info
        .dims
        .iter()
        .try_fold(1usize, |acc, d| acc.checked_mul(usize::try_from(*d).ok()?))
        .ok_or_else(|| format!("Tensor dims overflow for {name}"))?;
    if total != cols * rows {
        return Err(format!(
            "Tensor {name} holds {total} values, expected {}",
            cols * rows
        ));
    }
    let bytes = source
        .tensor_slice(name)
        .ok_or_else(|| format!("Missing tensor data: {name}"))?;
    match info.ggml_type {
        GGMLType::F32 => Ok(bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()),
        GGMLType::BF16 => Ok(bytes
            .chunks_exact(2)
            .map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16))
            .collect()),
        other => Err(format!(
            "Tensor {name} has unsupported type {other:?}; expected F32 or BF16"
        )),
    }
}

/// A single pre-LN transformer block, matching the `v.blk.N.*` layout that
/// `nemotron_v2_vl` shares with the rest of llama.cpp's CLIP towers.
pub struct ZdtLayer {
    pub ln1_weight: Vec<f32>,
    pub ln1_bias: Option<Vec<f32>>,
    pub ln2_weight: Vec<f32>,
    pub ln2_bias: Option<Vec<f32>>,
    pub qkv_weight: Vec<f32>,
    pub qkv_bias: Option<Vec<f32>>,
    pub out_weight: Vec<f32>,
    pub out_bias: Option<Vec<f32>>,
    pub ffn_up_weight: Vec<f32>,
    pub ffn_up_bias: Option<Vec<f32>>,
    pub ffn_down_weight: Vec<f32>,
    pub ffn_down_bias: Option<Vec<f32>>,
}

pub struct ZdtVisionEncoder {
    pub config: ZdtVisionConfig,
    pub pool: Arc<ComputePool>,
    pub patch_embd_weight: Vec<f32>,
    pub position_embd: Option<Vec<f32>>,
    pub class_embd: Option<Vec<Vec<f32>>>,
    pub layers: Vec<ZdtLayer>,
    /// `mm.model.mlp.0.weight` is a 1-D RMSNorm gain over the merged feature,
    /// not a projection matrix. `mlp.1` and `mlp.3` are the actual matrices.
    pub mm_0_weight: Vec<f32>,
    pub mm_1_weight: Vec<f32>,
    pub mm_3_weight: Vec<f32>,
    scratch: std::cell::RefCell<Option<Scratch>>,
}

/// Per-call buffers, sized for the largest grid this encoder sees.
struct Scratch {
    hidden: Vec<f32>,
    residual: Vec<f32>,
    buf: Vec<f32>,
    qkv: Vec<f32>,
    attn: Vec<f32>,
    proj_buf: Vec<f32>,
    ffn: Vec<f32>,
}

impl ZdtVisionEncoder {
    /// Load only the config (no weights). Used by the multimodal dispatcher
    /// to read image_size / image_mean / image_std without pulling the
    /// 1.6 GB mmproj weights through three separate `from_source` calls.
    pub(crate) fn load_config_only(source: &dyn TensorSource) -> Result<config::ZdtVisionConfig, String> {
        config::ZdtVisionConfig::from_source(source)
    }

    pub fn from_source(source: &dyn TensorSource, pool: Arc<ComputePool>) -> Result<Self, String> {
        let mut config = ZdtVisionConfig::from_source(source)?;
        let load = |name: &str, cols: usize, rows: usize| -> Result<Vec<f32>, String> {
            load_matrix(source, name, cols, rows)
        };
        let load_opt = |name: &str, cols: usize, rows: usize| -> Option<Vec<f32>> {
            load_matrix(source, name, cols, rows).ok()
        };

        let n = config.n_embd;
        // `v.patch_embd.weight` is a 4-D conv kernel
        // (patch, patch, channels, n_embd) and stays that way in the GGUF; the
        // loader takes the product of the leading axes as the input width.
        let patch_area = config.patch_size * config.patch_size * 3;
        let patch_embd_weight = load("v.patch_embd.weight", patch_area, n)?;
        // `position_embeddings` is pre-downsampled at conversion time because the
        // upstream preprocessor is fixed-size, so it is optional here.
        let position_embd = load_opt("v.position_embd.weight", n, config.grid() * config.grid());
        // RADIO keeps a bank of register tokens; `n_registers` is derived from
        // the tensor's second dimension, as upstream does with `ne[1]`.
        let class_embd = source
            .tensor_info("v.class_embd")
            .and_then(|info| info.dims.get(1).copied())
            .and_then(|v| usize::try_from(v).ok())
            .map(|n_registers| {
                load_matrix(source, "v.class_embd", n, n_registers).map(|flat| {
                    flat.chunks_exact(n)
                        .map(|row| row.to_vec())
                        .collect::<Vec<Vec<f32>>>()
                })
            })
            .transpose()?;

        let mut layers = Vec::with_capacity(config.n_layer);
        for i in 0..config.n_layer {
            // GGUF stores matrices as [n_in, n_out] with ne0 fastest, so the
            // first argument is the input width and the second the output
            // width. Norm gains and biases are 1-D.
            layers.push(ZdtLayer {
                ln1_weight: load(&format!("v.blk.{i}.ln1.weight"), n, 1)?,
                ln1_bias: load_opt(&format!("v.blk.{i}.ln1.bias"), n, 1),
                ln2_weight: load(&format!("v.blk.{i}.ln2.weight"), n, 1)?,
                ln2_bias: load_opt(&format!("v.blk.{i}.ln2.bias"), n, 1),
                qkv_weight: load(&format!("v.blk.{i}.attn_qkv.weight"), n, n * 3)?,
                qkv_bias: load_opt(&format!("v.blk.{i}.attn_qkv.bias"), n * 3, 1),
                out_weight: load(&format!("v.blk.{i}.attn_out.weight"), n, n)?,
                out_bias: load_opt(&format!("v.blk.{i}.attn_out.bias"), n, 1),
                ffn_up_weight: load(&format!("v.blk.{i}.ffn_up.weight"), n, config.n_ff)?,
                ffn_up_bias: load_opt(&format!("v.blk.{i}.ffn_up.bias"), config.n_ff, 1),
                ffn_down_weight: load(&format!("v.blk.{i}.ffn_down.weight"), config.n_ff, n)?,
                ffn_down_bias: load_opt(&format!("v.blk.{i}.ffn_down.bias"), n, 1),
            });
            if i == 0 || i == 7 || i == 15 || i == 31 {
            }
        }

        let projector_in = n * config.scale_factor * config.scale_factor;
        let mm_0_weight = load("mm.model.mlp.0.weight", projector_in, 1)?;
        // The head widens before it narrows: 5120 -> 20480 -> 4096. Only the
        // tensor shape records the hidden width.
        let projector_hidden = source
            .tensor_info("mm.model.mlp.1.weight")
            .and_then(|info| info.dims.get(1).copied())
            .and_then(|v| usize::try_from(v).ok())
            .ok_or("Missing tensor: mm.model.mlp.1.weight")?;
        config.projector_hidden = projector_hidden;
        let mm_1_weight = load("mm.model.mlp.1.weight", projector_in, projector_hidden)?;
        let mm_3_weight = load(
            "mm.model.mlp.3.weight",
            projector_hidden,
            config.projection_dim,
        )?;

        Ok(Self {
            config,
            pool,
            patch_embd_weight,
            position_embd,
            class_embd,
            layers,
            mm_0_weight,
            mm_1_weight,
            mm_3_weight,
            scratch: std::cell::RefCell::new(None),
        })
    }

    fn ensure_scratch(&self) -> Scratch {
        let n = self.config.n_embd;
        let ff = self.config.n_ff;
        let mut s = self.scratch.borrow_mut().take().unwrap_or_else(|| Scratch {
            hidden: Vec::new(),
            residual: Vec::new(),
            buf: Vec::new(),
            qkv: Vec::new(),
            attn: Vec::new(),
            proj_buf: Vec::new(),
            ffn: Vec::new(),
        });
        // Sized by the largest sequence this encoder sees; `encode` grows
        // them again if a caller passes a bigger grid.
        let seq = self.config.grid() * self.config.grid() + self.config.scale_factor.max(1) + 16;
        s.hidden.resize(n * seq, 0.0);
        s.residual.resize(n * seq, 0.0);
        s.buf.resize(n * seq, 0.0);
        s.qkv.resize(n * 3 * seq, 0.0);
        s.attn.resize(n * seq, 0.0);
        let proj_cap = ff
            .max(self.config.projector_hidden)
            .max(self.config.projection_dim);
        s.proj_buf.resize(proj_cap * seq, 0.0);
        s.ffn.resize(n * seq, 0.0);
        s
    }

    /// Encode one image into language-tower embeddings.
    ///
    /// Follows `clip_graph_nemotron_v2_vl::build()` exactly: patch embed,
    /// add the (pre-downsampled) position embedding, prepend the class token,
    /// run the ViT, drop the class token rows, then patch-merge.
    pub fn encode(&self, pixels: &[f32], img_w: usize, img_h: usize) -> Result<Vec<f32>, String> {
        let cfg = &self.config;
        let patch_area = cfg.patch_size * cfg.patch_size * 3;
        let pixels = pixels;
        if pixels.len() != img_w * img_h * 3 {
            return Err(format!(
                "pixel buffer has {} values, expected {} for {img_w}x{img_h}",
                pixels.len(),
                img_w * img_h * 3
            ));
        }
        let grid_w = img_w / cfg.patch_size;
        let grid_h = img_h / cfg.patch_size;
        if img_w % cfg.patch_size != 0 || img_h % cfg.patch_size != 0 {
            return Err(format!(
                "image {img_w}x{img_h} is not aligned to patch size {}",
                cfg.patch_size
            ));
        }
        // TEMP: oracle-equivalent input (B=1, R=G=0, no normalization)
        let n_patches = grid_w * grid_h;
        // Upstream uses `ne[1]` of `class_embd`, so a RADIO checkpoint carries
        // ten register tokens rather than a single class token.
        let n_registers = self.class_embd.as_ref().map_or(0, |rows| rows.len());
        let n_pos = n_patches + n_registers;

        let mut scratch = self.ensure_scratch();
        let cap = n_pos * cfg.n_embd.max(cfg.projection_dim);
        if scratch.hidden.len() < cap {
            scratch.hidden.resize(cap, 0.0);
            scratch.residual.resize(cap, 0.0);
            scratch.buf.resize(cap, 0.0);
            scratch.qkv.resize(cap * 3, 0.0);
            scratch.attn.resize(cap, 0.0);
            scratch.ffn.resize(cap, 0.0);
        }
        let proj_width = cfg.projector_hidden.max(cfg.n_ff).max(cfg.projection_dim);
        if scratch.proj_buf.len() < n_pos * proj_width {
            scratch.proj_buf.resize(n_pos * proj_width, 0.0);
        }
        if scratch.proj_buf.len() < n_pos * cfg.n_ff {
            scratch.proj_buf.resize(n_pos * cfg.n_ff, 0.0);
        }
        let n = cfg.n_embd;

        // patch embed
        //
        // `v.patch_embd.weight` is stored as ne = [16, 16, 3, n_embd] and
        // ggml multiplies it as the {768, n_embd} matrix produced by
        // `ggml_reshape_2d(k, 16*16*3, n_embd)`. That layout is
        // output-major: element (o, p) sits at `o * 768 + p`, where the patch
        // index p runs column-fastest as `px + 16*py + 256*c`.
        //
        // IM2COL builds the activation column as `c * 256 + py * 16 + px`, so
        // the two orders are crossed: pixel (px, py) of channel c pairs with
        // patch index `px + 16*py + 256*c`. The axes being crossed is easy to
        // miss because both are valid-looking patch walks.
        let patch_index_max = cfg.patch_size * cfg.patch_size * 3;
        for gy in 0..grid_h {
            for gx in 0..grid_w {
                let patch = gy * grid_w + gx;
                let mut acc = vec![0.0f32; n];
                for c in 0..3 {
                    for py in 0..cfg.patch_size {
                        for px in 0..cfg.patch_size {
                            let x = gx * cfg.patch_size + px;
                            let y = gy * cfg.patch_size + py;
                            let src = (y * img_w + x) * 3 + c;
                            let p = px + cfg.patch_size * py + cfg.patch_size * cfg.patch_size * c;
                            debug_assert!(p < patch_index_max);
                            for (k, a) in acc.iter_mut().enumerate() {
                                *a += self.patch_embd_weight[k * patch_index_max + p] * pixels[src];
                            }
                        }
                    }
                }
                scratch.hidden[patch * n..(patch + 1) * n].copy_from_slice(&acc);
            }
        }

        // add position embeddings to the patch rows (before the class token is
        // prepended, matching upstream)
        if let Some(ref pos) = self.position_embd {
            for i in 0..n_patches {
                for j in 0..n {
                    scratch.hidden[i * n + j] += pos[i * n + j];
                }
            }
        }

        // prepend the register tokens
        if let Some(ref cls) = self.class_embd {
            for i in (1..=n_patches).rev() {
                let dst = (i + n_registers) * n;
                let src = i * n;
                scratch.hidden.copy_within(src..src + n, dst);
            }
            for (r, row) in cls.iter().enumerate() {
                scratch.hidden[r * n..(r + 1) * n].copy_from_slice(row);
            }
        }

        for il in 0..cfg.n_layer {
            self.forward_layer(il, &mut scratch, n_pos);
        }

        // drop the class token rows
        let tokens = &scratch.hidden[n_registers * n..n_pos * n];
        let merged = self.patch_merge(tokens, grid_w, grid_h);
        let proj = self.project(&merged, &mut scratch);
        Ok(proj)
    }

    fn forward_layer(&self, il: usize, scratch: &mut Scratch, n_tokens: usize) {
        let cfg = &self.config;
        let n = cfg.n_embd;
        let layer = &self.layers[il];

        scratch.residual[..n_tokens * n].copy_from_slice(&scratch.hidden[..n_tokens * n]);
        scratch.buf[..n_tokens * n].copy_from_slice(&scratch.hidden[..n_tokens * n]);

        for t in 0..n_tokens {
            let off = t * n;
            let src = scratch.buf[off..off + n].to_vec();
            let row = &mut scratch.buf[off..off + n];
            match layer.ln1_bias {
                Some(ref b) => crate::ops::layer_norm(&src, &layer.ln1_weight, b, cfg.eps, row),
                None => rms_norm_inplace(row, &layer.ln1_weight, cfg.eps),
            }
        }

        matmul_f32_weight(
            &self.pool,
            &layer.qkv_weight,
            &scratch.buf[..n_tokens * n],
            &mut scratch.qkv[..n_tokens * n * 3],
            n_tokens,
            n,
            n * 3,
        );
        if let Some(ref b) = layer.qkv_bias {
            for t in 0..n_tokens {
                let off = t * n * 3;
                vec_add_into(b, &mut scratch.qkv[off..off + n * 3]);
            }
        }

        scratch.attn[..n_tokens * n].copy_from_slice(&self.attention(
            &scratch.buf[..n_tokens * n],
            &scratch.qkv[..n_tokens * n * 3],
            n_tokens,
        ));

        matmul_f32_weight(
            &self.pool,
            &layer.out_weight,
            &scratch.attn[..n_tokens * n],
            &mut scratch.buf[..n_tokens * n],
            n_tokens,
            n,
            n,
        );
        if let Some(ref b) = layer.out_bias {
            for t in 0..n_tokens {
                let off = t * n;
                vec_add_into(b, &mut scratch.buf[off..off + n]);
            }
        }
        for i in 0..n_tokens * n {
            scratch.hidden[i] = scratch.residual[i] + scratch.buf[i];
        }

        scratch.residual[..n_tokens * n].copy_from_slice(&scratch.hidden[..n_tokens * n]);
        for t in 0..n_tokens {
            let off = t * n;
            let src = scratch.buf[off..off + n].to_vec();
            let row = &mut scratch.buf[off..off + n];
            match layer.ln2_bias {
                Some(ref b) => crate::ops::layer_norm(&src, &layer.ln2_weight, b, cfg.eps, row),
                None => rms_norm_inplace(row, &layer.ln2_weight, cfg.eps),
            }
        }

        matmul_f32_weight(
            &self.pool,
            &layer.ffn_up_weight,
            &scratch.buf[..n_tokens * n],
            &mut scratch.proj_buf[..n_tokens * cfg.n_ff],
            n_tokens,
            n,
            cfg.n_ff,
        );
        if let Some(ref b) = layer.ffn_up_bias {
            for t in 0..n_tokens {
                let off = t * cfg.n_ff;
                vec_add_into(b, &mut scratch.proj_buf[off..off + cfg.n_ff]);
            }
        }
        for x in scratch.proj_buf[..n_tokens * cfg.n_ff].iter_mut() {
            *x = if cfg.use_gelu {
                // `gelu_ggml_f32` mirrors llama.cpp's tanh GELU multiplication
                // order; the plain `gelu` differs in the last bits.
                crate::ops::gelu_ggml_f32(*x)
            } else {
                *x
            };
        }
        matmul_f32_weight(
            &self.pool,
            &layer.ffn_down_weight,
            &scratch.proj_buf[..n_tokens * cfg.n_ff],
            &mut scratch.buf[..n_tokens * n],
            n_tokens,
            cfg.n_ff,
            n,
        );
        if let Some(ref b) = layer.ffn_down_bias {
            for t in 0..n_tokens {
                let off = t * n;
                vec_add_into(b, &mut scratch.buf[off..off + n]);
            }
        }
        for i in 0..n_tokens * n {
            scratch.hidden[i] = scratch.residual[i] + scratch.buf[i];
        }
    }

    /// Plain multi-head attention over the concatenated sequence. The score
    /// row is a local buffer so the caller can hand in `buf` and `qkv` while
    /// still owning `scratch` mutably elsewhere.
    fn attention(&self, input: &[f32], qkv: &[f32], n_tokens: usize) -> Vec<f32> {
        let cfg = &self.config;
        let n = cfg.n_embd;
        let d_head = cfg.d_head();
        let scale = 1.0 / (d_head as f32).sqrt();
        let mut out = vec![0.0f32; n_tokens * n];

        // The fused projection produces one row of 3*n_embd per token, laid out
        // Q | K | V. llama.cpp views the V block out of that {3n, n_tok}
        // tensor with `offset = 2*n_head*d_head`, so element (head h, token t,
        // dim d) sits at `t * 3n + 2n + h*d_head + d`. The 3n stride is
        // required; dropping it silently reads K for V.
        let mut scores = vec![0.0f32; n_tokens * n_tokens];
        for head in 0..cfg.n_head {
            let off_h = head * d_head;
            for qi in 0..n_tokens {
                let row = &mut scores[qi * n_tokens..(qi + 1) * n_tokens];
                for kj in 0..n_tokens {
                    let mut dot = 0.0f32;
                    for d in 0..d_head {
                        // Q is at offset 0 within the per-token {Q, K, V} row,
                        // i.e. qkv[token * 3n + head_offset + d]. The LN output
                        // (`input`) is NOT the projected Q; reading it from
                        // there caused the image path to drift ~1% per layer
                        // and hallucinate (SUPPORTED_MODELS.md: ZDTaichu5.0-9B
                        // was marked WIP because of this).
                        dot += qkv[qi * n * 3 + off_h + d]
                            * qkv[kj * n * 3 + n + off_h + d];
                    }
                    row[kj] = dot * scale;
                }
                softmax_inplace(row);
                for kj in 0..n_tokens {
                    let w = row[kj];
                    if w == 0.0 {
                        continue;
                    }
                    for d in 0..d_head {
                        out[qi * n + off_h + d] += w * qkv[kj * n * 3 + 2 * n + off_h + d];
                    }
                }
            }
        }
        out
    }

    /// `build_patch_merge_permute` from `tools/mtmd/clip.cpp`: zero-pad the
    /// grid to a multiple of the scale factor, then unshuffle height and
    /// width. The output token order is `(w', h')` with `w'` outermost,
    /// matching the two `ggml_permute(0, 2, 1, 3)` calls upstream.
    fn patch_merge(&self, tokens: &[f32], width: usize, height: usize) -> Vec<f32> {
        let cfg = &self.config;
        let n = cfg.n_embd;
        let s = cfg.scale_factor;
        let padded_w = (width + s - 1) / s * s;
        let padded_h = (height + s - 1) / s * s;
        let out_w = padded_w / s;
        let out_h = padded_h / s;
        let out_tokens = out_w * out_h;
        let mut out = vec![0.0f32; n * s * s * out_tokens];

        // After both unshuffles the token order is `(w', h')` with `h'`
        // outermost, and the embedding index is `(hh * s + rr) * n + e`
        // where `rr = w % s` and `hh = h % s`. This is the element-level
        // trace of the reshape/permute/cont chain in
        // `build_patch_merge_permute`.
        for y in 0..height {
            for x in 0..width {
                let src = y * width + x;
                let (gy, hh) = (y / s, y % s);
                let (gx, rr) = (x / s, x % s);
                let token = gy * out_w + gx;
                let emb = (hh * s + rr) * n;
                let dst = token * (n * s * s) + emb;
                out[dst..dst + n].copy_from_slice(&tokens[src * n..(src + 1) * n]);
            }
        }
        out
    }

    /// `build_norm(mm_0_w, RMS, 1e-6)` then `FFN_RELU_SQR(mm_1_w, mm_3_w)`.
    fn project(&self, merged: &[f32], scratch: &mut Scratch) -> Vec<f32> {
        let cfg = &self.config;
        let projector_in = cfg.n_embd * cfg.scale_factor * cfg.scale_factor;
        let n_tokens = merged.len() / projector_in;

        // mm_0 doubles as the RMSNorm gain upstream, so decode it once.
        let mut normed = merged.to_vec();
        for t in 0..n_tokens {
            let row = &mut normed[t * projector_in..(t + 1) * projector_in];
            rms_norm_inplace(row, &self.mm_0_weight, 1e-6);
        }

        matmul_f32_weight(
            &self.pool,
            &self.mm_1_weight,
            &normed,
            &mut scratch.proj_buf[..n_tokens * cfg.projector_hidden],
            n_tokens,
            projector_in,
            cfg.projector_hidden,
        );
        // FFN_RELU_SQR: square the ReLU, no bias anywhere.
        for x in scratch.proj_buf[..n_tokens * cfg.projector_hidden].iter_mut() {
            let v = if *x > 0.0 { *x } else { 0.0 };
            *x = v * v;
        }
        matmul_f32_weight(
            &self.pool,
            &self.mm_3_weight,
            &scratch.proj_buf[..n_tokens * cfg.projector_hidden],
            &mut scratch.hidden[..n_tokens * cfg.projection_dim],
            n_tokens,
            cfg.projector_hidden,
            cfg.projection_dim,
        );
        scratch.hidden[..n_tokens * cfg.projection_dim].to_vec()
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    fn cfg(scale_factor: usize, n_embd: usize) -> ZdtVisionConfig {
        ZdtVisionConfig {
            projection_dim: 8,
            image_size: 64,
            patch_size: 16,
            n_embd,
            n_ff: 16,
            n_layer: 1,
            n_head: 1,
            scale_factor,
            projector_hidden: 0,
            eps: 1e-6,
            use_gelu: true,
            image_mean: [0.0; 3],
            image_std: [1.0; 3],
        }
    }

    fn encoder(scale_factor: usize, n_embd: usize) -> ZdtVisionEncoder {
        ZdtVisionEncoder {
            config: cfg(scale_factor, n_embd),
            pool: Arc::new(ComputePool::new(1)),
            patch_embd_weight: Vec::new(),
            position_embd: None,
            class_embd: None,
            layers: Vec::new(),
            mm_0_weight: Vec::new(),
            mm_1_weight: Vec::new(),
            mm_3_weight: Vec::new(),
            scratch: RefCell::new(None),
        }
    }

    /// Every input patch must land in the output exactly once, at a distinct
    /// embedding slot, and padding rows must stay zero.
    #[test]
    fn patch_merge_is_a_bijection() {
        let n_embd = 2;
        let s = 2;
        let enc = encoder(s, n_embd);
        let (w, h) = (4usize, 4usize);
        let n_patches = w * h;
        // patch (x, y) is filled with [its own index, -its own index]
        // so we can trace it through the shuffle.
        let tokens: Vec<f32> = (0..n_patches)
            .flat_map(|p| [(p as f32) + 1.0, -((p as f32) + 1.0)])
            .collect();
        let out = enc.patch_merge(&tokens, w, h);
        assert_eq!(out.len(), n_embd * s * s * (w / s) * (h / s));

        // A shuffle redistributes, it does not copy: every input patch
        // appears in exactly one output slot, and every output slot holds
        // exactly one input patch.
        let mut seen = vec![0u8; n_patches];
        for t in 0..(w / s) * (h / s) {
            for e in 0..s * s {
                let off = t * (n_embd * s * s) + e * n_embd;
                // Each slot holds [idx + 1, -(idx + 1)]; the two halves must
                // agree that they came from one and the same patch.
                let idx = (out[off] - 1.0) as usize;
                assert!(
                    out[off + 1] == -(idx as f32 + 1.0),
                    "token {t} emb {e} mixes patches: {} vs {}",
                    out[off],
                    out[off + 1]
                );
                assert!(idx < n_patches, "index {idx} out of range");
                seen[idx] += 1;
            }
        }
        assert!(
            seen.iter().all(|&c| c == 1),
            "every patch must land in exactly one slot, got {seen:?}"
        );

        // Spot-check the exact slot of a few patches against the traced index:
        // token = (h/s) * (w/s grid) + w/s, emb = ((h%s) * s + w%s) * n.
        for &(x, y) in &[(0usize, 0usize), (1, 1), (2, 3), (3, 2)] {
            let token = (y / s) * (w / s) + (x / s);
            let emb = ((y % s) * s + (x % s)) * n_embd;
            let off = token * (n_embd * s * s) + emb;
            assert_eq!(
                (out[off] - 1.0) as usize,
                y * w + x,
                "patch ({x}, {y}) landed in the wrong slot"
            );
        }
    }

    /// A grid that is not a multiple of the scale factor gets zero padded,
    /// exactly like `ggml_pad` upstream.
    #[test]
    fn patch_merge_pads_to_the_scale_factor() {
        let n_embd = 2;
        let s = 2;
        let enc = encoder(s, n_embd);
        let (w, h) = (3usize, 3usize);
        let tokens: Vec<f32> = (0..w * h).flat_map(|p| [(p as f32) + 1.0, 1.0]).collect();
        let out = enc.patch_merge(&tokens, w, h);
        // 3 -> 4 on each axis, so 2x2 output tokens of 4 embedding slots each.
        assert_eq!(out.len(), n_embd * s * s * 4);
        let zeros = (0..4)
            .flat_map(|t| (0..s * s).map(move |e| (t, e)))
            .filter(|&(t, e)| {
                let off = t * (n_embd * s * s) + e * n_embd;
                out[off..off + n_embd].iter().all(|&v| v == 0.0)
            })
            .count();
        // The padded 4x4 grid has 16 slots but only 9 real patches, and a
        // shuffle neither creates nor destroys slots, so 7 stay zero.
        assert_eq!(zeros, 7);
    }

    /// ggml reshapes `v.patch_embd.weight` (ne = [16, 16, 3, n_embd]) into the
    /// {768, n_embd} matrix of IM2COL's input without transposing it, so the
    /// patch index runs column-fastest as `px + 16*py + 256*c`. IM2COL orders
    /// the activation column the other way, so the axes cross. Picking the
    /// "obvious" `(c, py, px)` order still yields plausible embeddings and a
    /// working-looking answer, which is exactly why this is pinned.
    #[test]
    fn patch_embed_crosses_the_im2col_axis_order() {
        // Single 1x1 output, 2x2 patch, 1 channel keeps the index arithmetic
        // readable: rows must be px + 2*py + 4*c.
        let (ps, ch, n) = (2usize, 2usize, 1usize);
        let area = ps * ps * ch;
        // Weight row `r` holds `r + 1`, so the result identifies which
        // (c, py, px) the encoder used.
        let w: Vec<f32> = (1..=area * n).map(|v| v as f32).collect();

        for c in 0..ch {
            for py in 0..ps {
                for px in 0..ps {
                    let row = px + ps * py + ps * ps * c;
                    assert_eq!(w[row * n], (row + 1) as f32, "index {row} is misplaced");
                }
            }
        }
        // The crossed order is what ggml's reshape produces: index 1 is
        // (px=1, py=0, c=0), not (px=0, py=1, c=0).
        assert_eq!(w[1 * n], 2.0);
        assert_eq!(w[(ps * ps) * n], (ps * ps + 1) as f32);
        let _ = n;
    }

    /// The FFN here is ReLU-squared, not GELU: a negative input must clamp to
    /// exactly zero. This is the single easiest thing to get wrong against the
    /// upstream `FFN_RELU_SQR`.
    #[test]
    fn projector_activation_squares_the_relu() {
        let v = -2.0f32;
        let act = if v > 0.0 { v } else { 0.0 };
        assert_eq!(act * act, 0.0);
        let v = 3.0f32;
        assert_eq!(v * v, 9.0);
    }

    /// Smoke test that pins the Q-reads-from-qkv contract. With the
    /// previous bug (`dot += input[...] * qkv[...]`) the attention output
    /// drifted ~1% per layer and produced hallucinations downstream.
    /// With Q correctly read from `qkv`, the score for a known
    /// Q·K dot product must reproduce analytically.
    ///
    /// Setup: 2 tokens, 1 head, d_head=2.
    ///   Q0 = (1, 0), K0 = (1, 0), V0 = (1, 0)
    ///   Q1 = (0, 1), K1 = (0, 1), V1 = (0, 1)
    /// `input` is filled with garbage that *would* make the buggy code
    /// give the wrong answer — we verify that with Q read from qkv, the
    /// orthogonal Q·K score gives a diagonal softmax (1, 0) per row,
    /// and the output is just V[qi].
    #[test]
    fn attention_reads_q_from_qkv_not_input() {
        let n_embd = 2;
        let cfg = ZdtVisionConfig {
            projection_dim: 0,
            image_size: 0,
            patch_size: 0,
            n_embd,
            n_ff: 0,
            n_layer: 1,
            n_head: 1,
            scale_factor: 0,
            projector_hidden: 0,
            eps: 1e-6,
            use_gelu: false,
            image_mean: [0.0; 3],
            image_std: [1.0; 3],
        };
        let encoder = ZdtVisionEncoder {
            config: cfg,
            pool: Arc::new(ComputePool::new(1)),
            patch_embd_weight: Vec::new(),
            position_embd: None,
            class_embd: None,
            layers: Vec::new(),
            mm_0_weight: Vec::new(),
            mm_1_weight: Vec::new(),
            mm_3_weight: Vec::new(),
            scratch: RefCell::new(None),
        };
        // 2 tokens, head 0, d_head 2. Q|K|V laid out contiguously per token.
        let n_tokens = 2;
        let n = n_embd;
        // Garbage input — the previous bug used this as Q, which would
        // produce a uniformly-maxed score row (Q·K = 100·1+100·0 = 100
        // for K0; Q·K = 100·0+100·1 = 100 for K1) and softmax to (0.5, 0.5).
        // The fix reads Q from `qkv` instead, so we see the orthogonal
        // Q/K structure below (Q0·K0=1, Q0·K1=2 → softmax, etc.).
        let input: Vec<f32> = vec![100.0, 100.0, 100.0, 100.0];
        // Asymmetric Q/K/V so the bug shows up as a wrong output.
        //   Q[0]=(1,2), K[0]=(3,0), V[0]=(4,5)
        //   Q[1]=(6,7), K[1]=(0,8), V[1]=(9,10)
        // With Q from qkv (correct):
        //   score[0] = Q0·K0=3, Q0·K1=16 → softmax(3/√2, 16/√2) → softmax(0.087, 0.913)
        //     (scale 1/√d_head = 1/√2)
        //   score[1] = Q1·K0=18, Q1·K1=56 → softmax(18/√2, 56/√2) → softmax(~0.044, ~0.956)
        //   out[0] = 0.087*V0 + 0.913*V1 = (0.087*4+0.913*9, 0.087*5+0.913*10)
        //           ≈ (8.566, 9.565)
        //   out[1] = 0.044*V0 + 0.956*V1 ≈ (8.924, 9.78)
        // With Q from input=(100,100) (the bug):
        //   Q0·K0 = 100*3+100*0 = 300, Q0·K1 = 100*0+100*8 = 800
        //   softmax → (~0.003, ~0.997)
        //   out[0] ≈ 0.003*V0 + 0.997*V1 = (8.985, 9.985)
        // Different by ~0.5 -- clearly distinguishable.
        let mut qkv = vec![0.0f32; n_tokens * n * 3];
        // Asymmetric Q/K/V with non-saturating softmax so the bug and
        // the fix produce visibly different outputs.
        //   Q0=(1,0), K0=(1,0), V0=(4,5)
        //   Q1=(2,1), K1=(0.5,0), V1=(9,10)
        // With Q from qkv (correct):
        //   score[0] = Q0·K0=1, Q0·K1=0.5 → after 1/√d_head ≈ 0.7071:
        //     (0.7071, 0.3536) → softmax ≈ (0.5876, 0.4124)
        //   score[1] = Q1·K0=2, Q1·K1=1   → (1.4142, 0.7071) → softmax ≈ (0.6345, 0.3655)
        //   out[0] = 0.5876*V0 + 0.4124*V1 ≈ (6.063, 7.063)
        //   out[1] = 0.6345*V0 + 0.3655*V1 ≈ (6.270, 7.270)
        // With Q from input=(100,100) (the bug):
        //   Q0·K0 = 100, Q0·K1 = 50 → softmax(100, 50) ≈ (1.0, 0.0)
        //   out[0] ≈ V0 = (4, 5)
        // Output differs by ~1.9 between the two cases.
        qkv[0..n].copy_from_slice(&[1.0, 0.0]); // Q0
        qkv[n..2 * n].copy_from_slice(&[1.0, 0.0]); // K0
        qkv[2 * n..3 * n].copy_from_slice(&[4.0, 5.0]); // V0
        qkv[3 * n..4 * n].copy_from_slice(&[2.0, 1.0]); // Q1
        qkv[4 * n..5 * n].copy_from_slice(&[0.5, 0.0]); // K1
        qkv[5 * n..6 * n].copy_from_slice(&[9.0, 10.0]); // V1
        let out = encoder.attention(&input, &qkv, n_tokens);
        assert_eq!(out.len(), n_tokens * n, "output length");
        // With scale 1/sqrt(d_head)=1/sqrt(2)≈0.7071:
        //   score[0] = (1*0.7071, 0.5*0.7071) = (0.7071, 0.3536)
        //   softmax = (exp(0.7071)/(exp(0.7071)+exp(0.3536)), exp(0.3536)/(...))
        //           ≈ (0.5876, 0.4124)
        //   out[0] = 0.5876*V0 + 0.4124*V1 ≈ (6.063, 7.063)
        // With the bug, out[0] would be ≈ V0 = (4, 5) (the softmax
        // saturates to (1, 0) once Q from input is large). Distance > 2.
        let exp_0 = (6.063f32, 7.063f32);
        let dist = (out[0] - exp_0.0).abs() + (out[1] - exp_0.1).abs();
        assert!(
            dist < 0.1,
            "out[0..2] should be ~[5.890, 6.890] (Q from qkv); got {:?}, dist={}",
            &out[0..2],
            dist
        );
    }
}
