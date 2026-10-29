//! LFM3's gated attention / short convolution and sandwich-normalized dense FFN.

use super::config::Lfm3Config;
use super::session::{Lfm3Model, Lfm3Session};
use super::trace;
use super::weights::Lfm3LayerWeights;
use crate::compute::linear::LinearExecutor;
use crate::compute::ComputeError;
use crate::core::scratchpad::{ExecutionScratchpad, KvCache, KvFormat};
use crate::core::tensor::TensorSource;
use crate::core::tokenizer::BPETokenizer;
use crate::ops::kernel::Weight;
use crate::ops::{
    dot_f32, rms_norm, rms_norm_inplace, rope_neox_inplace_precise,
    rope_neox_inplace_with_freqs_precise, sigmoid_inplace, silu_mul_inplace, softmax_inplace,
    vec_add_into, vec_mul_inplace,
};
use std::io::{self, Write};
use std::time::{Duration, Instant};

#[allow(clippy::too_many_arguments)]
pub(crate) fn forward_layer<'model, 'weights>(
    linear: &mut LinearExecutor<'model, 'weights>,
    lw: &'model Lfm3LayerWeights<'weights>,
    layer: usize,
    cfg: &Lfm3Config,
    scratch: &mut ExecutionScratchpad,
    kv: &mut KvCache,
    pos: usize,
    conv_state: &mut [f32],
) -> Result<(), ComputeError> {
    let ExecutionScratchpad {
        x,
        normed,
        q,
        k_new,
        v_new,
        attn_out,
        attn_proj,
        gate_buf,
        up_buf,
        down_buf,
        scores,
        ..
    } = scratch;
    let width = cfg.n_embd;
    let eps = cfg.norm_eps;
    rms_norm(x, &lw.attn_norm, normed, eps);
    trace::layer_1d("attn_norm", layer, pos, normed);
    if lw.is_attn {
        let dim = cfg.n_embd_head_k;
        let q_width = cfg.n_head * dim;
        let kv_width = cfg.n_head_kv_per_layer[layer] * dim;
        let q = &mut q[..q_width];
        let k = &mut k_new[..kv_width];
        let v = &mut v_new[..kv_width];
        let gate = &mut gate_buf[..cfg.n_head];
        linear.run_group(
            [
                linear.id_for(lw.wq.as_ref().unwrap())?,
                linear.id_for(lw.wk.as_ref().unwrap())?,
                linear.id_for(lw.wv.as_ref().unwrap())?,
                linear.id_for(lw.attn_gate.as_ref().unwrap())?,
            ],
            normed,
            1,
            [q, k, v, gate],
        )?;
        trace::layer_2d("q_proj", layer, pos, dim, cfg.n_head, q);
        trace::layer_2d("k_proj", layer, pos, dim, cfg.n_head_kv_per_layer[layer], k);
        trace::layer_2d("v_proj", layer, pos, dim, cfg.n_head_kv_per_layer[layer], v);
        for head in q.chunks_exact_mut(dim) {
            rms_norm_inplace(head, lw.q_norm.as_ref().unwrap(), eps);
        }
        for head in k.chunks_exact_mut(dim) {
            rms_norm_inplace(head, lw.k_norm.as_ref().unwrap(), eps);
        }
        trace::layer_2d("q_norm", layer, pos, dim, cfg.n_head, q);
        trace::layer_2d("k_norm", layer, pos, dim, cfg.n_head_kv_per_layer[layer], k);
        for values in [&mut *q, &mut *k] {
            if cfg.sliding_window_pattern[layer] {
                rope_neox_inplace_precise(values, pos, dim, cfg.rope_freq_base_swa);
            } else {
                rope_neox_inplace_with_freqs_precise(
                    values,
                    pos,
                    dim,
                    cfg.rope_freq_base,
                    lw.rope_freqs.as_ref().unwrap(),
                );
            }
        }
        trace::layer_2d("q_rope", layer, pos, dim, cfg.n_head, q);
        trace::layer_2d("k_rope", layer, pos, dim, cfg.n_head_kv_per_layer[layer], k);
        let offset = pos * kv_width;
        match kv {
            KvCache::F32(cache) => {
                cache.k[offset..offset + kv_width].copy_from_slice(k);
                cache.v[offset..offset + kv_width].copy_from_slice(v);
            }
            KvCache::F16(cache) => {
                for i in 0..kv_width {
                    cache.k[offset + i] = crate::ops::f32_to_f16(k[i]);
                    cache.v[offset + i] = crate::ops::f32_to_f16(v[i]);
                }
            }
        }
        let start = attention_start(
            pos,
            cfg.sliding_window_pattern[layer].then_some(cfg.sliding_window),
        );
        let scores = &mut scores[..pos + 1 - start];
        let mut values = vec![0.0; scores.len()];
        let mut cached_key = vec![0.0; dim];
        for head in 0..cfg.n_head {
            let kv_head = head / (cfg.n_head / cfg.n_head_kv_per_layer[layer]);
            let query = &q[head * dim..(head + 1) * dim];
            for (i, score) in scores.iter_mut().enumerate() {
                let offset = (start + i) * kv_width + kv_head * dim;
                let key = match kv {
                    KvCache::F32(cache) => &cache.k[offset..offset + dim],
                    KvCache::F16(cache) => {
                        for (out, &v) in cached_key.iter_mut().zip(&cache.k[offset..offset + dim]) {
                            *out = crate::ops::f16_to_f32(v);
                        }
                        &cached_key
                    }
                };
                *score = dot_f32(query, key, dim) / (dim as f32).sqrt();
            }
            softmax_inplace(scores);
            for d in 0..dim {
                for (i, out) in values.iter_mut().enumerate() {
                    let offset = (start + i) * kv_width + kv_head * dim + d;
                    *out = match kv {
                        KvCache::F32(cache) => cache.v[offset],
                        KvCache::F16(cache) => crate::ops::f16_to_f32(cache.v[offset]),
                    };
                }
                attn_out[head * dim + d] = dot_f32(scores, &values, scores.len());
            }
        }
        trace::layer_1d("attn_values", layer, pos, &attn_out[..q_width]);
        let gate = &mut gate_buf[..cfg.n_head];
        sigmoid_inplace(gate);
        trace::layer_1d("attn_gate", layer, pos, gate);
        for (head, &gate) in attn_out[..q_width].chunks_exact_mut(dim).zip(gate.iter()) {
            crate::ops::vec_scale_f32(head, gate);
        }
        project(
            linear,
            lw.wo.as_ref().unwrap(),
            &attn_out[..q_width],
            attn_proj,
        )?;
    } else {
        let bcx = &mut gate_buf[..3 * width];
        project(linear, lw.shortconv_in.as_ref().unwrap(), normed, bcx)?;
        trace::layer_1d("conv_in_proj", layer, pos, bcx);
        let (b, cx) = bcx.split_at(width);
        let (c, input) = cx.split_at(width);
        let mut bx = b.to_vec();
        vec_mul_inplace(input, &mut bx);
        let mut conv = shortconv_step(
            conv_state,
            &bx,
            lw.shortconv_conv.as_ref().unwrap(),
            cfg.l_cache,
        );
        trace::layer_1d("conv_conv", layer, pos, &conv);
        vec_mul_inplace(c, &mut conv);
        project(linear, lw.shortconv_out.as_ref().unwrap(), &conv, attn_proj)?;
        trace::layer_1d("conv_out_proj", layer, pos, attn_proj);
    }
    trace::layer_1d("attn_proj", layer, pos, attn_proj);
    if let Some(norm) = &lw.post_attention_norm {
        rms_norm_inplace(attn_proj, norm, eps);
    }
    trace::layer_1d("post_operator_norm", layer, pos, attn_proj);
    vec_add_into(attn_proj, x);
    rms_norm(x, &lw.ffn_norm, normed, eps);
    trace::layer_1d("ffn_norm", layer, pos, normed);
    let gate = &mut gate_buf[..cfg.n_ff];
    let up = &mut up_buf[..cfg.n_ff];
    linear.run_group(
        [linear.id_for(&lw.w_gate)?, linear.id_for(&lw.w_up)?],
        normed,
        1,
        [gate, up],
    )?;
    trace::layer_1d("ffn_gate", layer, pos, gate);
    trace::layer_1d("ffn_up", layer, pos, up);
    silu_mul_inplace(gate, up);
    trace::layer_1d("ffn_swiglu", layer, pos, up);
    project(linear, &lw.w_down, up, down_buf)?;
    trace::layer_1d("ffn_down", layer, pos, down_buf);
    if let Some(norm) = &lw.post_ffw_norm {
        rms_norm_inplace(down_buf, norm, eps);
    }
    trace::layer_1d("post_ffn_norm", layer, pos, down_buf);
    vec_add_into(down_buf, x);
    trace::layer_1d("l_out", layer, pos, x);
    Ok(())
}

fn project(
    linear: &mut LinearExecutor<'_, '_>,
    weight: &Weight<'_>,
    input: &[f32],
    output: &mut [f32],
) -> Result<(), ComputeError> {
    linear
        .run(linear.id_for(weight)?, input, 1, output)
        .map(|_| ())
}

fn attention_start(pos: usize, window: Option<usize>) -> usize {
    window.map_or(0, |window| (pos + 1).saturating_sub(window))
}

fn shortconv_step(state: &mut [f32], bx: &[f32], kernel: &[f32], taps: usize) -> Vec<f32> {
    let width = bx.len();
    let mut out = vec![0.0; width];
    for channel in 0..width {
        for t in 0..taps - 1 {
            out[channel] =
                state[t * width + channel].mul_add(kernel[channel * taps + t], out[channel]);
        }
        // ggml's shortconv accumulates F32 products with FMA.
        out[channel] = bx[channel].mul_add(kernel[channel * taps + taps - 1], out[channel]);
    }
    state.copy_within(width.., 0);
    state[(taps - 2) * width..].copy_from_slice(bx);
    out
}

pub fn run_forward_logits_lfm3_with_batch(
    source: &dyn TensorSource,
    input: &[u32],
    n_threads: usize,
    kv_format: KvFormat,
    max_context: usize,
    batch_size: usize,
) -> Result<(Vec<f32>, Duration), String> {
    let model = Lfm3Model::from_source(source, n_threads)?;
    let mut session = Lfm3Session::new(&model, kv_format, max_context)?;
    let start = Instant::now();
    let logits = session.forward_logits_chunked(input, batch_size)?;
    Ok((logits, start.elapsed()))
}

#[allow(clippy::too_many_arguments)]
pub fn run_inference(
    source: &dyn TensorSource,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    n_threads: usize,
    profile: bool,
    kv_format: KvFormat,
    max_context: usize,
    repetition_penalty: f32,
    thinking: bool,
    jinja: &crate::prompt::jinja::Options,
) -> Result<(), String> {
    let tokenizer = BPETokenizer::from_gguf_metadata(|key| source.metadata(key).cloned())?;
    let template = crate::prompt::jinja::Options {
        jinja: true,
        file: jinja.file.clone(),
    };
    let input = crate::prompt::jinja::single_turn_tokens(
        &tokenizer,
        &template,
        &|key| source.metadata(key).cloned(),
        prompt,
        thinking,
    )?
    .ok_or("LFM3 requires its GGUF chat template or --chat-template-file")?;
    let model = Lfm3Model::from_source(source, n_threads)?;
    let mut session = Lfm3Session::new(&model, kv_format, max_context)?;
    if input.len().saturating_add(max_tokens.saturating_sub(1)) > session.max_context {
        return Err(format!(
            "LFM3 prompt and generation exceed context capacity {}",
            session.max_context
        ));
    }
    trace::tokens("prompt_ids", &input);
    let start = Instant::now();
    let mut logits = session.forward_logits_chunked(&input, 1)?;
    let prefill = start.elapsed();
    let mut sampler = crate::ops::sampling::Lfm2MoeSampler::new();
    sampler.prime(&input);
    let mut decoder = tokenizer.streaming_decoder(false);
    let mut generated = Vec::new();
    let decode_start = Instant::now();
    print!("Output: ");
    for step in 0..max_tokens {
        let token = sampler.sample(&mut logits, temperature, repetition_penalty);
        if Some(token) == tokenizer.eos_id() {
            break;
        }
        generated.push(token);
        print!("{}", decoder.push(token));
        io::stdout().flush().map_err(|e| e.to_string())?;
        if step + 1 < max_tokens {
            logits = session.forward_logits_chunked(&[token], 1)?;
        }
    }
    println!("{}", decoder.finish());
    trace::tokens("generated_ids", &generated);
    if profile {
        eprintln!(
            "LFM3: {} prompt tokens in {:?}, {} generated tokens in {:?}",
            input.len(),
            prefill,
            generated.len(),
            decode_start.elapsed()
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sliding_window_boundary() {
        assert_eq!(attention_start(255, Some(256)), 0);
        assert_eq!(attention_start(256, Some(256)), 1);
        assert_eq!(attention_start(400, None), 0);
    }
    #[test]
    fn shortconv_uses_fused_products() {
        let a = 1.0 + f32::EPSILON;
        let b = 1.0 - f32::EPSILON;
        let out = shortconv_step(&mut [-1.0], &[a], &[1.0, b], 2);
        assert_eq!(out[0].to_bits(), a.mul_add(b, -1.0).to_bits());
        assert_ne!(out[0], a * b - 1.0);
    }
    #[test]
    fn shortconv_retains_six_tokens_in_kernel_order() {
        let mut state = vec![0.0; 6];
        let kernel = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        for step in 1..=9 {
            let out = shortconv_step(&mut state, &[step as f32], &kernel, 7);
            let expected: f32 = (0..7)
                .map(|t| (step as i32 - 6 + t as i32).max(0) as f32 * kernel[t])
                .sum();
            assert_eq!(out, vec![expected]);
        }
        assert_eq!(state, vec![4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    }
}
