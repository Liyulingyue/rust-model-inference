use crate::models::clip_config::Qwen35Config;
use crate::core::tensor::{GGMLType, TensorSource};
use crate::ops::kernel::{QuantizedTensor, Weight};
use crate::ops::{attention_value_f32, dot_f32, softmax_inplace, rope_neox, rope_mrope, vec_mad_f32};
#[cfg(feature = "parity-trace")]
use crate::parity_trace;
use crate::ops::quant::{self, BlockQ8K, QK_K};
use crate::core::thread_pool::ComputePool;
use crate::models::vision::VisionGrid;



pub fn build_qwen35_positions(
    token_ids: &[u32],
    image_token_id: Option<u32>,
    image_grids: &[VisionGrid],
) -> Result<(Vec<[usize; 4]>, usize), String> {
    let mut positions = Vec::with_capacity(token_ids.len());
    let mut next = 0usize;
    let mut token = 0usize;
    let mut grid_index = 0usize;

    while token < token_ids.len() {
        if image_token_id == Some(token_ids[token]) {
            let grid = *image_grids
                .get(grid_index)
                .ok_or("Image placeholder has no matching vision grid")?;
            let count = grid.checked_token_count()?;
            let end = token
                .checked_add(count)
                .ok_or("Image placeholder range overflow")?;
            if end > token_ids.len()
                || token_ids[token..end]
                    .iter()
                    .any(|id| Some(*id) != image_token_id)
            {
                return Err(format!(
                    "Image grid {grid_index} requires {count} contiguous placeholders"
                ));
            }
            let base = next;
            for image_index in 0..count {
                let row = image_index / grid.grid_w;
                let column = image_index % grid.grid_w;
                positions.push([base, base + row, base + column, 0]);
            }
            next = next
                .checked_add(grid.position_span())
                .ok_or("Qwen3.5 logical position overflow")?;
            token = end;
            grid_index += 1;
        } else {
            positions.push([next, next, next, 0]);
            next = next
                .checked_add(1)
                .ok_or("Qwen3.5 logical position overflow")?;
            token += 1;
        }
    }

    if grid_index != image_grids.len() {
        return Err(format!(
            "Unused vision grids: consumed {grid_index}, provided {}",
            image_grids.len()
        ));
    }
    Ok((positions, next))
}

pub struct Qwen35Model<'a> {
    pub config: Qwen35Config,
    pub tok_embd: Vec<f32>,
    pub output_norm: Vec<f32>,
    pub output_weight: Weight<'a>,
    pub layers: Vec<Qwen35LayerWeights<'a>>,
}

fn load_weight<'a, S: TensorSource + ?Sized>(source: &'a S, name: &str) -> Option<Weight<'a>> {
    let ti = source.tensor_info(name)?;
    let data = source.tensor_slice(name)?;
    let n_cols = ti.dims[0] as usize;
    let n_rows = if ti.dims.len() >= 2 { ti.dims[1] as usize } else { 1 };

    match ti.ggml_type {
        GGMLType::F32 | GGMLType::F16 | GGMLType::Q8_0 | GGMLType::Q4_0
        | GGMLType::Q4_1 | GGMLType::Q4K | GGMLType::Q5K | GGMLType::Q6K => {
            Some(Weight::from_quantized(QuantizedTensor::from_bytes(data, ti.ggml_type, n_cols, n_rows)))
        }
        _ => {
            eprintln!("WARNING: unsupported quant type {:?} for tensor {}", ti.ggml_type, name);
            None
        }
    }
}

fn load_weight_f32<S: TensorSource + ?Sized>(source: &S, name: &str) -> Option<Vec<f32>> {
    let ti = source.tensor_info(name)?;
    let data = source.tensor_slice(name)?;
    let n_el = ti.n_elements();
    match ti.ggml_type {
        GGMLType::F32 => {
            let mut out = Vec::with_capacity(n_el);
            for i in 0..n_el {
                let off = i * 4;
                if off + 4 <= data.len() {
                    out.push(f32::from_le_bytes([data[off], data[off+1], data[off+2], data[off+3]]));
                } else { out.push(0.0); }
            }
            Some(out)
        }
        GGMLType::F16 => {
            let mut out = Vec::with_capacity(n_el);
            for i in 0..n_el { out.push(f16_at(data, i)); }
            Some(out)
        }
        _ => None,
    }
}

pub struct Qwen35LayerWeights<'a> {
    pub attn_norm: Vec<f32>,
    pub attn_post_norm: Vec<f32>,
    pub wq: Option<Weight<'a>>,
    pub wk: Option<Weight<'a>>,
    pub wv: Option<Weight<'a>>,
    pub wo: Option<Weight<'a>>,
    pub attn_q_norm: Option<Vec<f32>>,
    pub attn_k_norm: Option<Vec<f32>>,
    pub wqkv: Option<Weight<'a>>,
    pub wqkv_gate: Option<Weight<'a>>,
    pub ssm_conv1d: Option<Vec<f32>>,
    pub ssm_dt: Option<Vec<f32>>,
    pub ssm_a: Option<Vec<f32>>,
    pub ssm_beta: Option<Weight<'a>>,
    pub ssm_alpha: Option<Weight<'a>>,
    pub ssm_norm: Option<Vec<f32>>,
    pub ssm_out: Option<Weight<'a>>,
    pub ffn_gate: Weight<'a>,
    pub ffn_up: Weight<'a>,
    pub ffn_down: Weight<'a>,
}

impl<'a> Qwen35Model<'a> {
    pub fn from_source(source: &'a dyn TensorSource) -> Result<Self, String> {
        let config = Qwen35Config::from_source(source)?;

        let tok_embd = {
            let ti = source.tensor_info("token_embd.weight").ok_or("Missing token_embd.weight")?;
            let data = source.tensor_slice("token_embd.weight").unwrap();
            let n_cols = ti.dims[0] as usize;
            let n_rows = ti.dims[1] as usize;
            match ti.ggml_type {
                GGMLType::F16 => (0..n_cols * n_rows).map(|i| f16_at(data, i)).collect(),
                GGMLType::Q8_0 => quant::dequant_q80_weight(data, n_cols, n_rows),
                GGMLType::Q6K => quant::dequant_q6k_weight(data, n_cols, n_rows),
                _ => return Err("Unsupported token_embd type".into()),
            }
        };

        let output_norm = load_weight_f32(source, "output_norm.weight").ok_or("Missing output_norm.weight")?;

        let output_weight = {
            let name = if source.tensor_info("output.weight").is_some() { "output.weight" } else { "token_embd.weight" };
            load_weight(source, name).ok_or("Missing output weight")?
        };

        let mut layers = Vec::with_capacity(config.n_layer);
        for i in 0..config.n_layer {
            let attn_norm = load_weight_f32(source, &format!("blk.{}.attn_norm.weight", i))
                .ok_or_else(|| format!("Missing blk.{}.attn_norm.weight", i))?;
            let attn_post_norm = load_weight_f32(source, &format!("blk.{}.post_attention_norm.weight", i))
                .ok_or_else(|| format!("Missing blk.{}.post_attention_norm.weight", i))?;
            let is_recr = config.is_recurrent[i];

            let (wq, wk, wv, wo, attn_q_norm, attn_k_norm) = if !is_recr {
                (
                    load_weight(source, &format!("blk.{}.attn_q.weight", i)),
                    load_weight(source, &format!("blk.{}.attn_k.weight", i)),
                    load_weight(source, &format!("blk.{}.attn_v.weight", i)),
                    load_weight(source, &format!("blk.{}.attn_output.weight", i)),
                    load_weight_f32(source, &format!("blk.{}.attn_q_norm.weight", i)),
                    load_weight_f32(source, &format!("blk.{}.attn_k_norm.weight", i)),
                )
            } else { (None, None, None, None, None, None) };

            let (wqkv, wqkv_gate, ssm_conv1d, ssm_dt, ssm_a, ssm_beta, ssm_alpha, ssm_norm, ssm_out) = if is_recr {
                (
                    load_weight(source, &format!("blk.{}.attn_qkv.weight", i)),
                    load_weight(source, &format!("blk.{}.attn_gate.weight", i)),
                    load_weight_f32(source, &format!("blk.{}.ssm_conv1d.weight", i)),
                    load_weight_f32(source, &format!("blk.{}.ssm_dt.bias", i)),
                    load_weight_f32(source, &format!("blk.{}.ssm_a", i)),
                    load_weight(source, &format!("blk.{}.ssm_beta.weight", i)),
                    load_weight(source, &format!("blk.{}.ssm_alpha.weight", i)),
                    load_weight_f32(source, &format!("blk.{}.ssm_norm.weight", i)),
                    load_weight(source, &format!("blk.{}.ssm_out.weight", i)),
                )
            } else { (None, None, None, None, None, None, None, None, None) };

            let ffn_gate = load_weight(source, &format!("blk.{}.ffn_gate.weight", i))
                .ok_or_else(|| format!("Missing blk.{}.ffn_gate.weight", i))?;
            let ffn_up = load_weight(source, &format!("blk.{}.ffn_up.weight", i))
                .ok_or_else(|| format!("Missing blk.{}.ffn_up.weight", i))?;
            let ffn_down = load_weight(source, &format!("blk.{}.ffn_down.weight", i))
                .ok_or_else(|| format!("Missing blk.{}.ffn_down.weight", i))?;
            layers.push(Qwen35LayerWeights {
                attn_norm, attn_post_norm, wq, wk, wv, wo,
                attn_q_norm, attn_k_norm,
                wqkv, wqkv_gate, ssm_conv1d, ssm_dt, ssm_a, ssm_beta, ssm_alpha, ssm_norm, ssm_out,
                ffn_gate, ffn_up, ffn_down,
            });
        }

        Ok(Self { config, tok_embd, output_norm, output_weight, layers })
    }

    pub fn forward(
        &self,
        n_tokens: usize,
        kv_cache: &mut crate::core::scratchpad::KvCache,
        scratch: &mut Qwen35Scratchpad,
        pool: &ComputePool,
        mrope_positions: &[[usize; 4]],
    ) -> Result<Vec<f32>, String> {
        if mrope_positions.len() != n_tokens {
            return Err(format!(
                "Qwen3.5 position count mismatch: tokens={n_tokens}, positions={}",
                mrope_positions.len()
            ));
        }
        let cfg = &self.config;
        let n_embd = cfg.n_embd;
        let n_layer = cfg.n_layer;
        let eps = cfg.norm_eps;
        let profile = std::env::var("PROFILE_QWEN35").is_ok();
        let mut t_attn: f64 = 0.0;
        let mut t_ffn: f64 = 0.0;
        #[cfg(feature = "parity-trace")]
        let first_dense_layer = self.config.is_recurrent.iter().position(|value| !*value);
        #[cfg(feature = "parity-trace")]
        let first_recurrent_layer = self.config.is_recurrent.iter().position(|value| *value);

        for il in 0..n_layer {
            let layer = &self.layers[il];
            let is_recr = cfg.is_recurrent[il];

            for t in 0..n_tokens {
                let off = t * n_embd;
                scratch.normed_buf[off..off + n_embd].copy_from_slice(&scratch.x[off..off + n_embd]);
                crate::ops::rms_norm_inplace(&mut scratch.normed_buf[off..off + n_embd], &layer.attn_norm, eps);
            }
            #[cfg(feature = "parity-trace")]
            if first_dense_layer == Some(il) {
                parity_trace::report(parity_trace::checkpoint(
                    &format!("attn_norm-{il}"),
                    Some(il),
                    &[n_tokens, n_embd],
                    &scratch.normed_buf[..n_tokens * n_embd],
                ));
            }

            let t0 = std::time::Instant::now();
            let normed_ptr = scratch.normed_buf.as_ptr();
            let normed_len = n_tokens * n_embd;
            let attn_out = if is_recr {
                let normed_input = unsafe { std::slice::from_raw_parts(normed_ptr, normed_len) };
                #[cfg(feature = "parity-trace")]
                {
                    self.forward_recurrent_layer(
                        il,
                        normed_input,
                        n_tokens,
                        scratch,
                        pool,
                        first_recurrent_layer == Some(il),
                    )
                }
                #[cfg(not(feature = "parity-trace"))]
                {
                    self.forward_recurrent_layer(il, normed_input, n_tokens, scratch, pool)
                }
            } else {
                let normed_input = unsafe { std::slice::from_raw_parts(normed_ptr, normed_len) };
                #[cfg(feature = "parity-trace")]
                {
                    self.forward_dense_attn_layer(
                        il,
                        normed_input,
                        n_tokens,
                        kv_cache,
                        scratch,
                        pool,
                        mrope_positions,
                        first_dense_layer == Some(il),
                    )
                }
                #[cfg(not(feature = "parity-trace"))]
                {
                    self.forward_dense_attn_layer(
                        il,
                        normed_input,
                        n_tokens,
                        kv_cache,
                        scratch,
                        pool,
                        mrope_positions,
                    )
                }
            };
            t_attn += t0.elapsed().as_secs_f64();

            for t in 0..n_tokens {
                let off = t * n_embd;
                crate::ops::vec_add_into(&attn_out[off..off + n_embd], &mut scratch.x[off..off + n_embd]);
            }

            for t in 0..n_tokens {
                let off = t * n_embd;
                scratch.buf[off..off + n_embd].copy_from_slice(&scratch.x[off..off + n_embd]);
                crate::ops::rms_norm_inplace(&mut scratch.buf[off..off + n_embd], &layer.attn_post_norm, eps);
            }

            let t0 = std::time::Instant::now();
            let buf_ptr = scratch.buf.as_ptr();
            let buf_len = n_tokens * n_embd;
            let ffn_input = unsafe { std::slice::from_raw_parts(buf_ptr, buf_len) };
            self.forward_ffn_parallel(layer, ffn_input, n_tokens, scratch, pool);
            t_ffn += t0.elapsed().as_secs_f64();

            for t in 0..n_tokens {
                let off = t * n_embd;
                crate::ops::vec_add_into(&scratch.buf[off..off + n_embd], &mut scratch.x[off..off + n_embd]);
            }
        }

        if profile {
            let total = t_attn + t_ffn;
            eprintln!("PROFILE: attn={:.1}% ({:.3}s) ffn={:.1}% ({:.3}s)", t_attn/total*100.0, t_attn, t_ffn/total*100.0, t_ffn);
        }

        let mut normed = vec![0.0f32; n_tokens * n_embd];
        for t in 0..n_tokens {
            let off = t * n_embd;
            normed[off..off + n_embd].copy_from_slice(&scratch.x[off..off + n_embd]);
            crate::ops::rms_norm_inplace(&mut normed[off..off + n_embd], &self.output_norm, eps);
        }

        let last_normed = &normed[(n_tokens - 1) * n_embd..n_tokens * n_embd];
        self.output_weight.quantize_and_matmul_with_scratch(
            last_normed,
            &mut scratch.q8k_buf,
            &mut scratch.q8_buf,
            &mut scratch.scale_buf,
            &mut scratch.matmul_out,
            cfg.vocab_size,
            pool,
        );
        let mut result = vec![0.0f32; cfg.vocab_size];
        let n = scratch.matmul_out.len().min(cfg.vocab_size);
        result[..n].copy_from_slice(&scratch.matmul_out[..n]);
        #[cfg(feature = "parity-trace")]
        parity_trace::report(parity_trace::checkpoint(
            "result_output",
            None,
            &[cfg.vocab_size],
            &result[..cfg.vocab_size],
        ));
        Ok(result)
    }

    fn forward_dense_attn_layer(
        &self, il: usize, input: &[f32], n_tokens: usize,
        kv_cache: &mut crate::core::scratchpad::KvCache, scratch: &mut Qwen35Scratchpad,
        pool: &ComputePool, mrope_positions: &[[usize; 4]],
        #[cfg(feature = "parity-trace")] trace_layer: bool,
    ) -> Vec<f32> {
        let profile = std::env::var("PROFILE_QWEN35").is_ok();
        let cfg = &self.config;
        let n_embd = cfg.n_embd;
        let n_head = cfg.n_head;
        let n_head_kv = cfg.n_head_kv;
        let n_embd_head = cfg.n_embd_head();
        let eps = cfg.norm_eps;
        let nth = pool.n_threads();
        let layer = &self.layers[il];
        let wq = layer.wq.as_ref().unwrap();
        let wk = layer.wk.as_ref().unwrap();
        let wv = layer.wv.as_ref().unwrap();
        let wo = layer.wo.as_ref().unwrap();
        let q_norm_w = layer.attn_q_norm.as_ref().unwrap();
        let k_norm_w = layer.attn_k_norm.as_ref().unwrap();
        let q_dim = n_embd_head * n_head * 2;
        let k_dim = n_embd_head * n_head_kv;
        let v_dim = n_embd_head * n_head_kv;
        let n_embd_heads_total = n_embd_head * n_head;

        let mut t_qkv: f64 = 0.0;
        let mut t_score: f64 = 0.0;
        let mut t_wo: f64 = 0.0;

        for t in 0..n_tokens {
            let inp_off = t * n_embd;
            let t0 = std::time::Instant::now();
            let inp_slice = &input[inp_off..inp_off + n_embd];
            wq.quantize_and_matmul_with_scratch(inp_slice, &mut scratch.q8k_buf, &mut scratch.q8_buf, &mut scratch.scale_buf, &mut scratch.matmul_out, q_dim, pool);
            scratch.q_buf[t * q_dim..t * q_dim + q_dim].copy_from_slice(&scratch.matmul_out[..q_dim]);
            wk.quantize_and_matmul_with_scratch(inp_slice, &mut scratch.q8k_buf, &mut scratch.q8_buf, &mut scratch.scale_buf, &mut scratch.matmul_out, k_dim, pool);
            scratch.k_buf[t * k_dim..t * k_dim + k_dim].copy_from_slice(&scratch.matmul_out[..k_dim]);
            wv.quantize_and_matmul_with_scratch(inp_slice, &mut scratch.q8k_buf, &mut scratch.q8_buf, &mut scratch.scale_buf, &mut scratch.matmul_out, v_dim, pool);
            scratch.v_buf[t * v_dim..t * v_dim + v_dim].copy_from_slice(&scratch.matmul_out[..v_dim]);
            t_qkv += t0.elapsed().as_secs_f64();
        }

        for t in 0..n_tokens {
            for h in 0..n_head {
                let q_off = t * q_dim + h * n_embd_head * 2;
                crate::ops::rms_norm_inplace(&mut scratch.q_buf[q_off..q_off + n_embd_head], q_norm_w, eps);
            }
            for h in 0..n_head_kv { crate::ops::rms_norm_inplace(&mut scratch.k_buf[t * k_dim + h * n_embd_head..][..n_embd_head], k_norm_w, eps); }
        }
        #[cfg(feature = "parity-trace")]
        let mut q_trace = Vec::with_capacity(n_tokens * n_head * n_embd_head);
        #[cfg(feature = "parity-trace")]
        if trace_layer {
            for token in 0..n_tokens {
                for head in 0..n_head {
                    let offset = token * q_dim + head * n_embd_head * 2;
                    q_trace.extend_from_slice(&scratch.q_buf[offset..offset + n_embd_head]);
                }
            }
            parity_trace::report(parity_trace::checkpoint(
                &format!("Qcur_normed-{il}"),
                Some(il),
                &[n_tokens, n_head, n_embd_head],
                &q_trace,
            ));
            parity_trace::report(parity_trace::checkpoint(
                &format!("Kcur_normed-{il}"),
                Some(il),
                &[n_tokens, n_head_kv, n_embd_head],
                &scratch.k_buf[..n_tokens * k_dim],
            ));
        }

        let kv_pos = kv_cache_pos(kv_cache, il, k_dim, cfg.n_layer);
        let sections = cfg.rope_dimension_sections;
        let use_mrope = sections[0] > 0 && sections[1] > 0;
        for t in 0..n_tokens {
            let positions = mrope_positions[t];
            for h in 0..n_head {
                let q_off = t * q_dim + h * n_embd_head * 2;
                if use_mrope {
                    rope_mrope(&mut scratch.q_buf[q_off..q_off + cfg.rope_dimension_count], positions, sections, cfg.rope_dimension_count, cfg.rope_freq_base);
                } else {
                    rope_neox(&mut scratch.q_buf[q_off..q_off + cfg.rope_dimension_count], positions[0], cfg.rope_dimension_count, cfg.rope_freq_base);
                }
            }
            for h in 0..n_head_kv {
                let k_off = t * k_dim + h * n_embd_head;
                if use_mrope {
                    rope_mrope(&mut scratch.k_buf[k_off..k_off + cfg.rope_dimension_count], positions, sections, cfg.rope_dimension_count, cfg.rope_freq_base);
                } else {
                    rope_neox(&mut scratch.k_buf[k_off..k_off + cfg.rope_dimension_count], positions[0], cfg.rope_dimension_count, cfg.rope_freq_base);
                }
            }
        }
        #[cfg(feature = "parity-trace")]
        let mut q_trace = Vec::with_capacity(n_tokens * n_head * n_embd_head);
        #[cfg(feature = "parity-trace")]
        if trace_layer {
            for token in 0..n_tokens {
                for head in 0..n_head {
                    let offset = token * q_dim + head * n_embd_head * 2;
                    q_trace.extend_from_slice(&scratch.q_buf[offset..offset + n_embd_head]);
                }
            }
            parity_trace::report(parity_trace::checkpoint(
                &format!("Qcur-{il}"),
                Some(il),
                &[n_tokens, n_head, n_embd_head],
                &q_trace,
            ));
            parity_trace::report(parity_trace::checkpoint(
                &format!("Kcur-{il}"),
                Some(il),
                &[n_tokens, n_head_kv, n_embd_head],
                &scratch.k_buf[..n_tokens * k_dim],
            ));
        }

        kv_cache_store(kv_cache, il, cfg.n_layer, &scratch.k_buf[..n_tokens * k_dim], &scratch.v_buf[..n_tokens * v_dim], k_dim, v_dim, kv_pos);
        let n_kv = kv_pos + n_tokens;
        let scale = 1.0 / (n_embd_head as f32).sqrt();

        let (k_cache, v_cache) = match kv_cache {
            crate::core::scratchpad::KvCache::F32(c) => (&c.k, &c.v),
            _ => return vec![0.0; n_tokens * n_embd],
        };
        let k_len = k_cache.len() / cfg.n_layer;
        let v_len = v_cache.len() / cfg.n_layer;

        let t0 = std::time::Instant::now();
        for t in 0..n_tokens {
            for h in 0..n_head {
                let q_off = t * q_dim + h * n_embd_head * 2;
                let kv_h = h / (n_head / n_head_kv);
                let n_attend = kv_pos + t + 1;
                let n_padded = n_attend.div_ceil(256) * 256;
                for s in 0..n_attend {
                    let k_off = il * k_len + s * k_dim + kv_h * n_embd_head;
                    let dot = dot_f32(&scratch.q_buf[q_off..q_off + n_embd_head], &k_cache[k_off..k_off + n_embd_head], n_embd_head);
                    scratch.score_buf[s] = dot * scale;
                }
                scratch.score_buf[n_attend..n_padded].fill(f32::NEG_INFINITY);
                softmax_inplace(&mut scratch.score_buf[..n_padded]);
                let out_base = t * n_embd_heads_total + h * n_embd_head;
                scratch.attn_out_buf[out_base..out_base + n_embd_head].fill(0.0);
                for s in 0..n_attend {
                    let v_row_base = il * v_len + s * v_dim + kv_h * n_embd_head;
                    let score = scratch.score_buf[s];
                    let v_row = &v_cache[v_row_base..v_row_base + n_embd_head];
                    vec_mad_f32(
                        &mut scratch.attn_out_buf[out_base..out_base + n_embd_head],
                        v_row,
                        score,
                    );
                }
            }
        }
        t_score += t0.elapsed().as_secs_f64();

        for t in 0..n_tokens {
            for h in 0..n_head {
                let gate_off = t * q_dim + h * n_embd_head * 2 + n_embd_head;
                let out_off = t * n_embd_heads_total + h * n_embd_head;
                for d in 0..n_embd_head {
                    scratch.attn_out_buf[out_off + d] *= sigmoid_f32(scratch.q_buf[gate_off + d]);
                }
            }
        }

        let mut result = vec![0.0f32; n_tokens * n_embd];
        let t0 = std::time::Instant::now();
        for t in 0..n_tokens {
            let wo_input = &scratch.attn_out_buf[t * n_embd_heads_total..t * n_embd_heads_total + n_embd_heads_total];
            wo.quantize_and_matmul_with_scratch(wo_input, &mut scratch.q8k_buf, &mut scratch.q8_buf, &mut scratch.scale_buf, &mut scratch.matmul_out, n_embd, pool);
            result[t * n_embd..t * n_embd + n_embd].copy_from_slice(&scratch.matmul_out[..n_embd]);
        }        t_wo += t0.elapsed().as_secs_f64();
        if profile {
            eprintln!("  dense_attn[{}]: qkv={:.3}s score={:.3}s wo={:.3}s", il, t_qkv, t_score, t_wo);
        }
        result
    }

    fn forward_recurrent_layer(
        &self,
        il: usize,
        input: &[f32],
        n_tokens: usize,
        scratch: &mut Qwen35Scratchpad,
        pool: &ComputePool,
        #[cfg(feature = "parity-trace")] trace_layer: bool,
    ) -> Vec<f32> {
        let profile = std::env::var("PROFILE_QWEN35").is_ok();
        let cfg = &self.config;
        let n_embd = cfg.n_embd;
        let d_inner = cfg.ssm_d_inner;
        let head_k_dim = cfg.ssm_d_state;
        let num_k_heads = cfg.ssm_n_group;
        let num_v_heads = cfg.ssm_dt_rank;
        let head_v_dim = d_inner / num_v_heads;
        let key_dim = cfg.key_dim();
        let value_dim = cfg.value_dim();
        let conv_dim = cfg.conv_dim();
        let d_conv = cfg.ssm_d_conv;
        let eps = cfg.norm_eps;

        let layer = &self.layers[il];
        let wqkv = layer.wqkv.as_ref().unwrap();
        let wqkv_gate = layer.wqkv_gate.as_ref().unwrap();
        let ssm_conv1d = layer.ssm_conv1d.as_ref().unwrap();
        let ssm_dt = layer.ssm_dt.as_ref().unwrap();
        let ssm_a = layer.ssm_a.as_ref().unwrap();
        let ssm_beta = layer.ssm_beta.as_ref().unwrap();
        let ssm_alpha = layer.ssm_alpha.as_ref().unwrap();
        let ssm_norm_w = layer.ssm_norm.as_ref().unwrap();
        let ssm_out = layer.ssm_out.as_ref().unwrap();

        let t0 = std::time::Instant::now();
        for t in 0..n_tokens {
            let inp_off = t * n_embd;
            let inp_slice = &input[inp_off..inp_off + n_embd];
            wqkv.quantize_and_matmul_with_scratch(inp_slice, &mut scratch.q8k_buf, &mut scratch.q8_buf, &mut scratch.scale_buf, &mut scratch.matmul_out, conv_dim, pool);
            scratch.qkv_buf[t * conv_dim..t * conv_dim + conv_dim].copy_from_slice(&scratch.matmul_out[..conv_dim]);
            wqkv_gate.quantize_and_matmul_with_scratch(inp_slice, &mut scratch.q8k_buf, &mut scratch.q8_buf, &mut scratch.scale_buf, &mut scratch.matmul_out, value_dim, pool);
            scratch.z_buf[t * value_dim..t * value_dim + value_dim].copy_from_slice(&scratch.matmul_out[..value_dim]);
            ssm_beta.quantize_and_matmul_with_scratch(inp_slice, &mut scratch.q8k_buf, &mut scratch.q8_buf, &mut scratch.scale_buf, &mut scratch.matmul_out, num_v_heads, pool);
            let n_beta = num_v_heads;
            scratch.beta_buf[t * num_v_heads..t * num_v_heads + n_beta].copy_from_slice(&scratch.matmul_out[..n_beta]);
            for v in 0..num_v_heads { scratch.beta_buf[t * num_v_heads + v] = sigmoid_f32(scratch.beta_buf[t * num_v_heads + v]); }
            ssm_alpha.quantize_and_matmul_with_scratch(inp_slice, &mut scratch.q8k_buf, &mut scratch.q8_buf, &mut scratch.scale_buf, &mut scratch.matmul_out, num_v_heads, pool);
            let n_alpha = num_v_heads;
            scratch.alpha_buf[t * num_v_heads..t * num_v_heads + n_alpha].copy_from_slice(&scratch.matmul_out[..n_alpha]);
            for v in 0..num_v_heads {
                let a_biased = scratch.alpha_buf[t * num_v_heads + v] + ssm_dt[v % ssm_dt.len()];
                scratch.alpha_buf[t * num_v_heads + v] = softplus_f32(a_biased) * ssm_a[v % ssm_a.len()];
            }
        }
        let t_matmul = t0.elapsed().as_secs_f64();

        let tc0 = std::time::Instant::now();
        let conv_state = &mut scratch.conv_states[il];
        #[cfg(feature = "parity-trace")]
        let mut conv_raw = if trace_layer {
            vec![0.0f32; n_tokens * conv_dim]
        } else {
            Vec::new()
        };
        for t in 0..n_tokens {
            let qkv_off = t * conv_dim;
            for c in 0..conv_dim {
                for k in 0..d_conv - 1 { conv_state[k * conv_dim + c] = conv_state[(k + 1) * conv_dim + c]; }
                conv_state[(d_conv - 1) * conv_dim + c] = scratch.qkv_buf[qkv_off + c];
            }
            for c in 0..conv_dim {
                let mut conv_val = 0.0f32;
                for k in 0..d_conv { conv_val += ssm_conv1d[c * d_conv + k] * conv_state[k * conv_dim + c]; }
                #[cfg(feature = "parity-trace")]
                if trace_layer {
                    conv_raw[t * conv_dim + c] = conv_val;
                }
                scratch.qkv_buf[qkv_off + c] = conv_val;
            }
            crate::ops::silu_approx_inplace(&mut scratch.qkv_buf[qkv_off..qkv_off + conv_dim]);
        }
        #[cfg(feature = "parity-trace")]
        if trace_layer {
            parity_trace::report(parity_trace::checkpoint(
                &format!("conv_output_raw-{il}"),
                Some(il),
                &[n_tokens, conv_dim],
                &conv_raw,
            ));
        }

        for t in 0..n_tokens {
            let qkv_off = t * conv_dim;
            for h in 0..num_k_heads {
                for d in 0..head_k_dim { scratch.q_buf[t * key_dim + h * head_k_dim + d] = scratch.qkv_buf[qkv_off + h * head_k_dim + d]; }
                for d in 0..head_k_dim { scratch.k_buf2[t * key_dim + h * head_k_dim + d] = scratch.qkv_buf[qkv_off + key_dim + h * head_k_dim + d]; }
            }
            for h in 0..num_v_heads {
                for d in 0..head_v_dim { scratch.v_buf2[t * value_dim + h * head_v_dim + d] = scratch.qkv_buf[qkv_off + 2 * key_dim + h * head_v_dim + d]; }
            }
            for h in 0..num_k_heads {
                l2_norm(&mut scratch.q_buf[t * key_dim + h * head_k_dim..][..head_k_dim], eps);
                l2_norm(&mut scratch.k_buf2[t * key_dim + h * head_k_dim..][..head_k_dim], eps);
            }
        }
        #[cfg(feature = "parity-trace")]
        if trace_layer {
            parity_trace::report(parity_trace::checkpoint(
                &format!("q_conv_predelta-{il}"),
                Some(il),
                &[n_tokens, num_k_heads, head_k_dim],
                &scratch.q_buf[..n_tokens * key_dim],
            ));
            parity_trace::report(parity_trace::checkpoint(
                &format!("k_conv_predelta-{il}"),
                Some(il),
                &[n_tokens, num_k_heads, head_k_dim],
                &scratch.k_buf2[..n_tokens * key_dim],
            ));
        }

        let tc = tc0.elapsed().as_secs_f64();

        let ts0 = std::time::Instant::now();
        let q_scale = 1.0 / (head_k_dim as f32).sqrt();
        #[cfg(feature = "parity-trace")]
        let state_before = if trace_layer {
            Some(scratch.ssm_states[il].clone())
        } else {
            None
        };
        #[cfg(feature = "parity-trace")]
        if let Some(state_before) = state_before.as_deref() {
            parity_trace::report(parity_trace::checkpoint(
                &format!("state_predelta-{il}"),
                Some(il),
                &[num_v_heads, head_v_dim, head_v_dim],
                state_before,
            ));
        }
        let ssm_state = &mut scratch.ssm_states[il];
        for t in 0..n_tokens {
            let q_off = t * key_dim;
            let k2_off = t * key_dim;
            let v2_off = t * value_dim;
            for v_h in 0..num_v_heads {
                let gate_val = scratch.alpha_buf[t * num_v_heads + v_h];
                let beta_val = scratch.beta_buf[t * num_v_heads + v_h];
                let state_off = v_h * head_v_dim * head_v_dim;
                let k_h = v_h % num_k_heads;
                let decay = gate_val.exp();
                crate::ops::ssm_state_decay(&mut ssm_state[state_off..state_off + head_v_dim * head_v_dim], decay);
                let k_slice = &scratch.k_buf2[k2_off + k_h * head_k_dim..][..head_v_dim];
                let mut sk = [0.0f32; 128];
                crate::ops::ssm_matvec(&ssm_state[state_off..][..head_v_dim * head_v_dim], k_slice, head_v_dim, head_v_dim, &mut sk[..head_v_dim]);
                let v_slice = &scratch.v_buf2[v2_off + v_h * head_v_dim..][..head_v_dim];
                let mut d_vec = [0.0f32; 128];
                for d in 0..head_v_dim { d_vec[d] = (v_slice[d] - sk[d]) * beta_val; }
                crate::ops::ssm_outer_product_update(&mut ssm_state[state_off..][..head_v_dim * head_v_dim], k_slice, &d_vec[..head_v_dim], head_v_dim);
                let q_slice = &scratch.q_buf[q_off + k_h * head_k_dim..][..head_v_dim];
                let out_off = t * value_dim + v_h * head_v_dim;
                crate::ops::ssm_matvec_scaled(&ssm_state[state_off..][..head_v_dim * head_v_dim], q_slice, head_v_dim, head_v_dim, &mut scratch.attn_out_buf[out_off..out_off + head_v_dim], q_scale);
            }
        }
        #[cfg(feature = "parity-trace")]
        if trace_layer {
            parity_trace::report(parity_trace::checkpoint(
                &format!("new_state-{il}"),
                Some(il),
                &[num_v_heads, head_v_dim, head_v_dim],
                ssm_state,
            ));
        }

        let tssm = ts0.elapsed().as_secs_f64();
        let tn0 = std::time::Instant::now();
        for t in 0..n_tokens {
            for h in 0..num_v_heads {
                let off = t * value_dim + h * head_v_dim;
                crate::ops::rms_norm_inplace(&mut scratch.attn_out_buf[off..off + head_v_dim], ssm_norm_w, eps);
            }
            let z_off = t * value_dim;
            crate::ops::silu_mul_approx_inplace(&scratch.z_buf[z_off..z_off + value_dim], &mut scratch.attn_out_buf[t * value_dim..t * value_dim + value_dim]);
        }
        #[cfg(feature = "parity-trace")]
        if trace_layer {
            parity_trace::report(parity_trace::checkpoint(
                &format!("final_output-{il}"),
                Some(il),
                &[n_tokens, num_v_heads, head_v_dim],
                &scratch.attn_out_buf[..n_tokens * value_dim],
            ));
        }

        let tnorm = tn0.elapsed().as_secs_f64();
        let mut result = vec![0.0f32; n_tokens * n_embd];
        let t0 = std::time::Instant::now();
        for t in 0..n_tokens {
            let inp = &scratch.attn_out_buf[t * value_dim..][..value_dim];
            ssm_out.quantize_and_matmul_with_scratch(inp, &mut scratch.q8k_buf, &mut scratch.q8_buf, &mut scratch.scale_buf, &mut scratch.matmul_out, n_embd, pool);
            result[t * n_embd..t * n_embd + n_embd].copy_from_slice(&scratch.matmul_out[..n_embd]);
        }
        let t_out_matmul = t0.elapsed().as_secs_f64();
        if profile {
            eprintln!("  recr[{}]: matmul={:.3}s conv={:.3}s ssm={:.3}s norm={:.3}s out={:.3}s", il, t_matmul, tc, tssm, tnorm, t_out_matmul);
        }
        result
    }

    fn forward_ffn_parallel(&self, layer: &Qwen35LayerWeights, hidden: &[f32], n_tokens: usize, scratch: &mut Qwen35Scratchpad, pool: &ComputePool) {
        let n_embd = self.config.n_embd;
        let n_ff = self.config.n_ff;

        for t in 0..n_tokens {
            let off = t * n_embd;
            let inp = &hidden[off..off + n_embd];
            layer.ffn_gate.quantize_and_matmul_with_scratch(inp, &mut scratch.q8k_buf, &mut scratch.q8_buf, &mut scratch.scale_buf, &mut scratch.matmul_out, n_ff, pool);
            scratch.ffn_gate_buf[t * n_ff..t * n_ff + n_ff].copy_from_slice(&scratch.matmul_out[..n_ff]);
            layer.ffn_up.quantize_and_matmul_with_scratch(inp, &mut scratch.q8k_buf, &mut scratch.q8_buf, &mut scratch.scale_buf, &mut scratch.matmul_out, n_ff, pool);
            scratch.ffn_up_buf[t * n_ff..t * n_ff + n_ff].copy_from_slice(&scratch.matmul_out[..n_ff]);
        }

        crate::ops::silu_mul_approx_inplace(&scratch.ffn_gate_buf[..n_tokens * n_ff], &mut scratch.ffn_up_buf[..n_tokens * n_ff]);

        for t in 0..n_tokens {
            let down_inp = &scratch.ffn_up_buf[t * n_ff..][..n_ff];
            layer.ffn_down.quantize_and_matmul_with_scratch(down_inp, &mut scratch.q8k_buf, &mut scratch.q8_buf, &mut scratch.scale_buf, &mut scratch.matmul_out, n_embd, pool);
            scratch.buf[t * n_embd..t * n_embd + n_embd].copy_from_slice(&scratch.matmul_out[..n_embd]);
        }
    }

    fn forward_ffn(&self, layer: &Qwen35LayerWeights, hidden: &[f32], n_tokens: usize, scratch: &mut Qwen35Scratchpad) {
        let n_embd = self.config.n_embd;
        let n_ff = self.config.n_ff;
        let profile = std::env::var("PROFILE_QWEN35").is_ok();
        let mut t_gate: f64 = 0.0;
        let mut t_up: f64 = 0.0;
        let mut t_down: f64 = 0.0;
        for t in 0..n_tokens {
            let off = t * n_embd;
            let t0 = std::time::Instant::now();
            let gate_out = layer.ffn_gate.matmul(&hidden[off..off + n_embd], n_ff);
            t_gate += t0.elapsed().as_secs_f64();
            scratch.ffn_gate_buf[t * n_ff..t * n_ff + gate_out.len().min(n_ff)].copy_from_slice(&gate_out[..gate_out.len().min(n_ff)]);
            let t0 = std::time::Instant::now();
            let up_out = layer.ffn_up.matmul(&hidden[off..off + n_embd], n_ff);
            t_up += t0.elapsed().as_secs_f64();
            scratch.ffn_up_buf[t * n_ff..t * n_ff + up_out.len().min(n_ff)].copy_from_slice(&up_out[..up_out.len().min(n_ff)]);
        }
        crate::ops::silu_mul_approx_inplace(&scratch.ffn_gate_buf[..n_tokens * n_ff], &mut scratch.ffn_up_buf[..n_tokens * n_ff]);
        for t in 0..n_tokens {
            let t0 = std::time::Instant::now();
            let down_out = layer.ffn_down.matmul(&scratch.ffn_up_buf[t * n_ff..][..n_ff], n_embd);
            t_down += t0.elapsed().as_secs_f64();
            scratch.buf[t * n_embd..t * n_embd + down_out.len().min(n_embd)].copy_from_slice(&down_out[..down_out.len().min(n_embd)]);
        }
        if profile {
            eprintln!("  ffn: gate={:.3}s up={:.3}s down={:.3}s", t_gate, t_up, t_down);
        }
    }
}

fn kv_cache_pos(cache: &crate::core::scratchpad::KvCache, il: usize, k_dim: usize, n_layer: usize) -> usize {
    if let crate::core::scratchpad::KvCache::F32(c) = cache {
        let k_len = c.k.len() / n_layer;
        let mut pos = 0;
        for p in 0..k_len / k_dim {
            if c.k[il * k_len + p * k_dim..il * k_len + (p + 1) * k_dim].iter().all(|v| *v == 0.0) { pos = p; break; }
            pos = p + 1;
        }
        pos
    } else { 0 }
}

fn kv_cache_store(cache: &mut crate::core::scratchpad::KvCache, il: usize, n_layer: usize, k_data: &[f32], v_data: &[f32], k_dim: usize, v_dim: usize, pos: usize) {
    if let crate::core::scratchpad::KvCache::F32(c) = cache {
        let k_len = c.k.len() / n_layer;
        let v_len = c.v.len() / n_layer;
        let n_tokens = k_data.len() / k_dim;
        for t in 0..n_tokens {
            let k_dst = il * k_len + (pos + t) * k_dim;
            let v_dst = il * v_len + (pos + t) * v_dim;
            c.k[k_dst..k_dst + k_dim].copy_from_slice(&k_data[t * k_dim..(t + 1) * k_dim]);
            c.v[v_dst..v_dst + v_dim].copy_from_slice(&v_data[t * v_dim..(t + 1) * v_dim]);
        }
    }
}

pub struct Qwen35Scratchpad {
    pub x: Vec<f32>,
    pub buf: Vec<f32>,
    pub normed_buf: Vec<f32>,
    pub q_buf: Vec<f32>,
    pub k_buf: Vec<f32>,
    pub v_buf: Vec<f32>,
    pub k_buf2: Vec<f32>,
    pub v_buf2: Vec<f32>,
    pub qkv_buf: Vec<f32>,
    pub z_buf: Vec<f32>,
    pub beta_buf: Vec<f32>,
    pub alpha_buf: Vec<f32>,
    pub score_buf: Vec<f32>,
    pub attention_value_buf: Vec<f32>,
    pub attn_out_buf: Vec<f32>,
    pub ffn_up_buf: Vec<f32>,
    pub ffn_gate_buf: Vec<f32>,
    pub conv_states: Vec<Vec<f32>>,
    pub ssm_states: Vec<Vec<f32>>,
    pub matmul_out: Vec<f32>,
    pub q8k_buf: Vec<quant::BlockQ8K>,
    pub q8_buf: Vec<u8>,
    pub scale_buf: Vec<f32>,
}

impl Qwen35Scratchpad {
    pub fn new(config: &Qwen35Config, max_tokens: usize) -> Self {
        let n_embd = config.n_embd;
        let n_head = config.n_head;
        let n_head_kv = config.n_head_kv;
        let n_embd_head = config.n_embd_head();
        let n_ff = config.n_ff;
        let n_layer = config.n_layer;
        let d_inner = config.ssm_d_inner;
        let key_dim = config.key_dim();
        let value_dim = config.value_dim();
        let conv_dim = config.conv_dim();
        let d_conv = config.ssm_d_conv;
        let num_v_heads = config.ssm_dt_rank;
        let head_v_dim = d_inner / num_v_heads;
        let q_dim = n_embd_head * n_head * 2;
        let dense_attn_out_dim = n_embd_head * n_head;
        let max_matmul_input = n_embd.max(n_ff).max(value_dim).max(dense_attn_out_dim);

        Self {
            x: vec![0.0; max_tokens * n_embd],
            buf: vec![0.0; max_tokens * n_embd],
            q_buf: vec![0.0; max_tokens * q_dim.max(key_dim)],
            k_buf: vec![0.0; max_tokens * n_embd_head * n_head_kv],
            v_buf: vec![0.0; max_tokens * n_embd_head * n_head_kv],
            k_buf2: vec![0.0; max_tokens * key_dim],
            v_buf2: vec![0.0; max_tokens * value_dim],
            qkv_buf: vec![0.0; max_tokens * conv_dim],
            z_buf: vec![0.0; max_tokens * value_dim],
            beta_buf: vec![0.0; max_tokens * num_v_heads],
            alpha_buf: vec![0.0; max_tokens * num_v_heads],
            score_buf: vec![0.0; config.n_ctx.div_ceil(256) * 256],
            attention_value_buf: vec![0.0; config.n_ctx.div_ceil(256) * 256],
            attn_out_buf: vec![0.0; max_tokens * dense_attn_out_dim.max(value_dim)],
            ffn_up_buf: vec![0.0; max_tokens * n_ff],
            ffn_gate_buf: vec![0.0; max_tokens * n_ff],
            conv_states: (0..n_layer).map(|_| vec![0.0; d_conv * conv_dim]).collect(),
            ssm_states: (0..n_layer).map(|_| vec![0.0; num_v_heads * head_v_dim * head_v_dim]).collect(),
            matmul_out: vec![0.0; (2 * n_ff).max(conv_dim).max(n_embd).max(config.vocab_size)],
            normed_buf: vec![0.0; max_tokens * n_embd],
            q8k_buf: vec![quant::BlockQ8K { d: 0.0, qs: [0i8; 256], bsums: [0i16; 16] }; (max_matmul_input + 255) / 256],
            q8_buf: vec![0u8; max_matmul_input],
            scale_buf: vec![0.0; (max_matmul_input + 31) / 32],
        }
    }
}

pub fn f16_at(data: &[u8], idx: usize) -> f32 {
    if idx * 2 + 2 > data.len() { return 0.0; }
    let bits = u16::from_le_bytes([data[idx * 2], data[idx * 2 + 1]]);
    f16_bits_to_f32(bits)
}

pub fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = if bits & 0x8000 != 0 { -1.0f32 } else { 1.0f32 };
    let exp = ((bits >> 10) & 0x1F) as i32;
    let frac = (bits & 0x3FF) as f32 / 1024.0;
    if exp == 0 { sign * frac * 2.0f32.powi(-14) }
    else if exp == 31 { if frac == 0.0 { sign * f32::INFINITY } else { sign * f32::NAN } }
    else { sign * (1.0 + frac) * 2.0f32.powi(exp - 15) }
}

fn l2_norm(x: &mut [f32], eps: f32) {
    let mut sum = 0.0f64;
    for &v in x.iter() {
        sum += f64::from(v * v);
    }
    let scale = 1.0f32 / (sum as f32).sqrt().max(eps);
    for v in x.iter_mut() { *v *= scale; }
}

fn sigmoid_f32(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }
fn softplus_f32(x: f32) -> f32 { if x > 20.0 { x } else { (1.0 + x.exp()).ln() } }

fn quantize_row_q8_k(input: &[f32]) -> Vec<BlockQ8K> {
    quant::quantize_row_q8_k(input)
}

fn vec_dot_q4k_q8k_fast(q4k_data: &[u8], q8k: &[BlockQ8K]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if crate::ops::has_avx2_fma() {
        return unsafe { quant::vec_dot_q4k_q8k_avx2_direct(q4k_data, q8k) };
    }
    quant::vec_dot_q4k_q8k_scalar(q4k_data, q8k)
}

fn vec_dot_q5k_q8k_fast(q5k_data: &[u8], q8k: &[BlockQ8K]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if crate::ops::has_avx2_fma() {
        return unsafe { quant::vec_dot_q5k_q8k_avx2_direct(q5k_data, q8k) };
    }
    quant::vec_dot_q5k_q8k_scalar(q5k_data, q8k)
}

fn vec_dot_q6k_q8k_fast(q6k_data: &[u8], q8k: &[BlockQ8K]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if crate::ops::has_avx2_fma() {
        return unsafe { quant::vec_dot_q6k_q8k_avx2_direct(q6k_data, q8k) };
    }
    quant::vec_dot_q6k_q8k_scalar(q6k_data, q8k)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dense_test_config(n_ctx: usize) -> Qwen35Config {
        Qwen35Config {
            n_embd: 32,
            n_layer: 1,
            n_head: 4,
            n_head_kv: 1,
            n_ff: 64,
            n_ctx,
            vocab_size: 32,
            rope_freq_base: 1_000_000.0,
            norm_eps: 1e-6,
            rope_dimension_count: 8,
            rope_dimension_sections: [2; 4],
            ssm_d_conv: 2,
            ssm_d_state: 8,
            ssm_n_group: 1,
            ssm_dt_rank: 1,
            ssm_d_inner: 8,
            full_attention_interval: 1,
            is_recurrent: vec![false],
            key_length: 32,
            value_length: 8,
        }
    }

    fn tiny_dense_model(k_weight: [f32; 4], v_weight: [f32; 4]) -> Qwen35Model<'static> {
        let config = Qwen35Config {
            n_embd: 2,
            n_layer: 1,
            n_head: 1,
            n_head_kv: 1,
            n_ff: 2,
            n_ctx: 256,
            vocab_size: 2,
            rope_freq_base: 1_000_000.0,
            norm_eps: 0.0,
            rope_dimension_count: 2,
            rope_dimension_sections: [0; 4],
            ssm_d_conv: 1,
            ssm_d_state: 2,
            ssm_n_group: 1,
            ssm_dt_rank: 1,
            ssm_d_inner: 2,
            full_attention_interval: 1,
            is_recurrent: vec![false],
            key_length: 2,
            value_length: 2,
        };
        let weight = |data: Vec<f32>, _n_rows| Weight::from_quantized(QuantizedTensor::F32(data));
        let identity = || weight(vec![1.0, 0.0, 0.0, 1.0], 2);
        let layer = Qwen35LayerWeights {
            attn_norm: vec![1.0; 2],
            attn_post_norm: vec![1.0; 2],
            wq: Some(weight(vec![1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 4)),
            wk: Some(weight(k_weight.to_vec(), 2)),
            wv: Some(weight(v_weight.to_vec(), 2)),
            wo: Some(identity()),
            attn_q_norm: Some(vec![1.0; 2]),
            attn_k_norm: Some(vec![4.0, 1.0]),
            wqkv: None,
            wqkv_gate: None,
            ssm_conv1d: None,
            ssm_dt: None,
            ssm_a: None,
            ssm_beta: None,
            ssm_alpha: None,
            ssm_norm: None,
            ssm_out: None,
            ffn_gate: identity(),
            ffn_up: identity(),
            ffn_down: identity(),
        };
        Qwen35Model {
            config,
            tok_embd: Vec::new(),
            output_norm: vec![1.0; 2],
            output_weight: identity(),
            layers: vec![layer],
        }
    }

    #[test]
    fn qwen35_l2_norm_matches_pinned_llama_cpp_bits() {
        const INPUT_BITS: [u32; 128] = [
            0x3b11b23c, 0x3abb616e, 0x3a74a52a, 0xbe36d3f9, 0x3c5be35f, 0xbb7eed3d, 0x3cf3b675, 0x3c352a95,
            0xbd12fa43, 0x3d0ed381, 0xbd019bdf, 0xbaa1fbf3, 0xbb00502c, 0x3b97baf3, 0x3c0a9eb1, 0x3b2a60d7,
            0x3c867f8c, 0x39269d95, 0xbc86036e, 0x3c9ca5fe, 0xbd2beb72, 0xba3c9646, 0x3c19ee2b, 0x3c6bebc5,
            0xba1078eb, 0x3cddb829, 0xb9cb7b05, 0x3c10df58, 0x38ad8928, 0x3a1bb101, 0xbc9dea12, 0xb830b929,
            0xbce68c07, 0x3c77b70a, 0x3a9a5020, 0x3aa1e526, 0xbc830d2f, 0xbccdf802, 0xbae28161, 0x3c13147a,
            0xbab844a1, 0xbd6a98ad, 0xb92c2313, 0x3b279123, 0x3cda6539, 0x398fc008, 0xb9883eab, 0x3c7a1914,
            0x3a9fd962, 0x3c024411, 0x3cfce9fa, 0x39fe7429, 0x39a78347, 0xbcd4f897, 0xb972edb6, 0x3cd07782,
            0xba48c562, 0x3d226f5a, 0x3a002199, 0x38995247, 0x3d24b1fe, 0x3c95409c, 0xb8b86102, 0xb9c4d2a9,
            0x3d1a4c1e, 0xb9679706, 0xbc94ecb1, 0xbb87b477, 0xbc306760, 0x3ad1617d, 0x3c85d8db, 0xb886458a,
            0x3baf244f, 0xbd5d49dc, 0xbb8260a4, 0xbc4a82db, 0x3aa20bbd, 0x3d2b8415, 0xba532f00, 0x39b7c6d9,
            0xbad2ca65, 0xbd023279, 0xbc77cee6, 0x393b6a88, 0x3c95ab9c, 0xba920ce0, 0xbb9881f0, 0xbaafd1fc,
            0x3b8edc22, 0x390e7b29, 0x3bbe234b, 0xbb803967, 0xb926490d, 0xbcf4b75a, 0x3c56bbe5, 0x3b016c3f,
            0xbc6cef21, 0x3b30b9c6, 0xb9ef1f3c, 0xb8aecebe, 0xba7b7fec, 0xb929766d, 0x3c9b5ced, 0xbc8ca6a4,
            0xbcd36384, 0x3d2261e8, 0x3ccda1b0, 0xbd298883, 0x3d40c6d6, 0x3969c035, 0x3c9c3466, 0xb991a558,
            0xb976ce00, 0xb9b01921, 0x398eef4c, 0xb5e9fa58, 0xbd416b03, 0x3be68c77, 0x39f3d603, 0x3d0243cb,
            0x3b3fc530, 0xbc016f46, 0x3bb1d80e, 0xba45c19f, 0xba89fe14, 0x3b26ebc8, 0xb9d28f76, 0xbbec5142,
        ];
        const EXPECTED_BITS: [u32; 128] = [
            0x3c05bcb8, 0x3bac0000, 0x3b60906f, 0xbf27d235, 0x3d49d6dc, 0xbc6a0077, 0x3ddfb552, 0x3d264bbc,
            0xbe06e9d2, 0x3e031a4c, 0xbdedf0cd, 0xbb94b02d, 0xbbeb8fdb, 0x3c8b46a4, 0x3cfe7bbe, 0x3c1c64ae,
            0x3d76eaac, 0x3a18f07d, 0xbd7606d0, 0x3d8fca57, 0xbe1dcee5, 0xbb2d1b7f, 0x3d0d4ba1, 0x3d588e5d,
            0xbb049d1f, 0x3dcb852c, 0xbabac748, 0x3d04fb24, 0x399f4aa6, 0x3b0ee976, 0xbd90f3d1, 0xb92237ac,
            0xbdd39f8b, 0x3d6361ce, 0x3b8da58c, 0x3b949b3f, 0xbd7096cc, 0xbdbd0ffc, 0xbbcfe9d2, 0x3d0701e2,
            0xbba9249b, 0xbe57571a, 0xba1e01f6, 0x3c19d00d, 0x3dc87814, 0x3a83f369, 0xba7a1f83, 0x3d6591c5,
            0x3b92ba79, 0x3cef2594, 0x3de8277f, 0x3ae99153, 0x3a99c355, 0xbdc37d6e, 0xba5efd0e, 0x3dbf5aff,
            0xbb384a95, 0x3e151a1b, 0x3aeb3a5a, 0x398cbc89, 0x3e172d40, 0x3d89005e, 0xb9a93ea7, 0xbab4aad2,
            0x3e0da1de, 0xba5494a0, 0xbd88b357, 0xbc7921cb, 0xbd21ec9a, 0x3bc031c5, 0x3d75b8a7, 0xb976802e,
            0x3ca0c40e, 0xbe4b1fec, 0xbc6f5a09, 0xbd39e37d, 0x3b94beab, 0x3e1d7004, 0xbb41d966, 0x3aa8b126,
            0xbbc17d0d, 0xbdef0548, 0xbd6377b4, 0x3a2c085b, 0x3d896296, 0xbb860fec, 0xbc8bfd4c, 0xbba16379,
            0x3c832238, 0x3a02c934, 0x3cae87ed, 0xbc6b660e, 0xba18a2e6, 0xbde0a121, 0x3d451bb1, 0x3bed995e,
            0xbd597c6f, 0x3c22383d, 0xbadb7e90, 0xb9a07583, 0xbb66db29, 0xba1b8d82, 0x3d8e9c48, 0xbd811b24,
            0xbdc2099b, 0x3e150dc4, 0x3dbcc0c0, 0xbe1b9e1c, 0x3e30f405, 0x3a569067, 0x3d8f6212, 0xba85b0e3,
            0xba628be5, 0xbaa1a4c7, 0x3a8333cf, 0xb6d6c5c4, 0xbe318ab8, 0x3cd39ff2, 0x3adfd249, 0x3def2514,
            0x3c300785, 0xbced9eed, 0x3ca33f05, 0xbb35862b, 0xbb7d54e2, 0x3c193845, 0xbac146f5, 0xbcd8eb85,
        ];

        let mut actual = INPUT_BITS.map(f32::from_bits);
        l2_norm(&mut actual, 1e-6);

        assert_eq!(actual.map(f32::to_bits), EXPECTED_BITS);
    }

    #[test]
    #[ignore = "requires RMI_QWEN35_MODEL"]
    fn qwen35_q8_0_model_loads() {
        let path = std::env::var("RMI_QWEN35_MODEL").expect("RMI_QWEN35_MODEL must be set");
        let source = crate::open_model_source(
            std::path::Path::new(&path),
            crate::ComponentRole::Llm,
        )
        .unwrap();

        assert_eq!(
            source.tensor_info("token_embd.weight").unwrap().ggml_type,
            GGMLType::Q8_0,
        );
        Qwen35Model::from_source(source.as_ref()).unwrap();
    }

    #[test]
    #[ignore = "requires an F16-token-embedding RMI_QWEN35_MODEL"]
    fn qwen35_f16_token_embedding_model_loads() {
        let path = std::env::var("RMI_QWEN35_MODEL").expect("RMI_QWEN35_MODEL must be set");
        let source = crate::open_model_source(
            std::path::Path::new(&path),
            crate::ComponentRole::Llm,
        )
        .unwrap();

        assert_eq!(
            source.tensor_info("token_embd.weight").unwrap().ggml_type,
            GGMLType::F16,
        );
        Qwen35Model::from_source(source.as_ref()).unwrap();
    }

    #[test]
    fn qwen35_q8_0_quantize_and_matmul_dispatches() {
        let mut data = Vec::with_capacity(2 * 8 * quant::BLOCK_Q80_SIZE);
        for quantized_value in [1u8, (-2i8) as u8] {
            for _ in 0..8 {
                data.extend_from_slice(&[0x00, 0x3c]);
                data.extend(std::iter::repeat_n(quantized_value, 32));
            }
        }
        let weight = QuantizedTensor::Q8_0 { data: &data, n_cols: 256, n_rows: 2 };
        let input = [1.0f32; 256];
        let mut q8k_buf = vec![BlockQ8K { d: 0.0, qs: [0; 256], bsums: [0; 16] }; 1];
        let mut output = [0.0f32; 2];

        weight.quantize_and_matmul(&input, &mut q8k_buf, &mut output);

        for (actual, expected) in output.into_iter().zip([256.0f32, -512.0]) {
            assert!((actual - expected).abs() < 0.05, "actual={actual}, expected={expected}");
        }
    }

    #[test]
    fn qwen35_scratch_covers_dense_attention_output_projection() {
        let config = dense_test_config(8);
        let dense_attn_out_dim = config.n_embd_head() * config.n_head;
        assert!(dense_attn_out_dim > config.n_embd.max(config.n_ff).max(config.value_dim()));

        let scratch = Qwen35Scratchpad::new(&config, 1);

        assert!(
            scratch.q8_buf.len() >= dense_attn_out_dim
                && scratch.scale_buf.len() >= (dense_attn_out_dim + 31) / 32,
            "dense_attn_out_dim={dense_attn_out_dim}, q8={}, scales={}",
            scratch.q8_buf.len(),
            scratch.scale_buf.len(),
        );
    }

    #[test]
    fn qwen35_scratch_pads_attention_buffers_to_ggml_row_size() {
        let scratch = Qwen35Scratchpad::new(&dense_test_config(257), 1);

        assert_eq!(scratch.score_buf.len(), 512);
        assert_eq!(scratch.attention_value_buf.len(), 512);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn qwen35_dense_attention_softmax_uses_ggml_padded_row() {
        let model = tiny_dense_model([1.0, 1.0, 0.0, 0.5], [1.0, 0.0, 0.0, 1.0]);
        let mut scratch = Qwen35Scratchpad::new(&model.config, 2);
        let mut kv_cache = crate::core::scratchpad::KvCache::new_f32(1, model.config.n_ctx, 2);
        let pool = ComputePool::new(1);

        model.forward_dense_attn_layer(
            0,
            &[1.0, 0.0, 0.0, 1.0],
            2,
            &mut kv_cache,
            &mut scratch,
            &pool,
            &[[0; 4]; 2],
            #[cfg(feature = "parity-trace")]
            false,
        );

        assert_eq!(
            [scratch.score_buf[0].to_bits(), scratch.score_buf[1].to_bits()],
            [0x3f25_1fe0, 0x3eb5_c03f],
        );
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn qwen35_dense_attention_value_uses_ggml_padded_reduction() {
        let model = tiny_dense_model([1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]);
        let n_tokens = 18;
        let mut scratch = Qwen35Scratchpad::new(&model.config, n_tokens);
        let mut kv_cache = crate::core::scratchpad::KvCache::new_f32(1, model.config.n_ctx, 2);
        let pool = ComputePool::new(1);
        let input: Vec<f32> = std::iter::repeat_n([1.0, 0.0], n_tokens).flatten().collect();

        let output = model.forward_dense_attn_layer(
            0,
            &input,
            n_tokens,
            &mut kv_cache,
            &mut scratch,
            &pool,
            &vec![[0; 4]; n_tokens],
            #[cfg(feature = "parity-trace")]
            false,
        );

        assert_eq!(output[(n_tokens - 1) * 2].to_bits(), 0x3f00_0000);
    }

    #[test]
    fn qwen35_positions_use_time_row_column_order() {
        let grid = VisionGrid {
            grid_t: 1,
            grid_h: 2,
            grid_w: 3,
            patch_size: 16,
            merge_size: 2,
        };
        let tokens = [10, 99, 99, 99, 99, 99, 99, 11];
        let (positions, next) = build_qwen35_positions(&tokens, Some(99), &[grid]).unwrap();
        assert_eq!(positions[1], [1, 1, 1, 0]);
        assert_eq!(positions[6], [1, 2, 3, 0]);
        assert_eq!(positions[7], [4, 4, 4, 0]);
        assert_eq!(next, 5);
    }

    #[test]
    fn qwen35_placeholder_count_must_equal_grid_tokens() {
        let grid = VisionGrid {
            grid_t: 1,
            grid_h: 2,
            grid_w: 3,
            patch_size: 16,
            merge_size: 2,
        };
        assert!(build_qwen35_positions(&[10, 99, 99, 11], Some(99), &[grid]).is_err());
    }

    #[test]
    fn qwen35_positions_reject_public_grid_token_overflow() {
        let grid = VisionGrid {
            grid_t: 1,
            grid_h: usize::MAX,
            grid_w: 2,
            patch_size: 1,
            merge_size: 1,
        };

        assert!(build_qwen35_positions(&[99], Some(99), &[grid]).is_err());
    }
}
