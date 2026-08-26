//! # Qwen3 纯文本推理
//!
//! 此模块包含 Qwen3 纯文本推理的完整实现，通过 `QuantizedTensor` 分派
//! 量化 kernel，支持 Q4_K、Q6_K、Q8_0 等所有 GGUF 量化格式。
//!
//! ## 架构说明
//!
//! Qwen3 推理存在两套代码路径：
//!
//! | 路径 | 模块 | 量化支持 | 用途 |
//! |------|------|----------|------|
//! | 纯文本推理 | `models::qwen3` | 所有格式 | CLI 文本生成 |
//! | VL/ASR/TTS | `models::qwen3::base_multimodal` | 多量化 | 多模态推理 |

use crate::app::cli::{per_second, inference_step_budget, resolve_thread_count, KvFormat};
use crate::core::loader::model_config_from_source;
use crate::core::scratchpad::{ExecutionScratchpad, KvCache};
use crate::core::tensor::{GGMLType, TensorSource};
use crate::core::thread_pool::ComputePool;
use crate::core::tokenizer::{BPETokenizer, EncodeOptions};
use crate::models::qwen3::skeleton::{load_layers, Qwen3LayerWeights, get_f32_tensor};
use crate::prompt::{build_qwen_chat_prompt, QwenMessage};
use crate::ops::embedding_lookup;
use crate::ops::kernel::{Kernel, QuantizedTensor, Weight};
use crate::ops::{
    attention_value_f32, dot_f32, dot_f16_f32, f32_slice_to_f16, quantize_q8_0_into,
    rms_norm, rms_norm_inplace, rope_neox, silu_mul_approx_inplace, softmax_inplace,
    vec_mad_f16_f32, vec_scale_f32,
};

use std::io::{self, Write};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[macro_export]
macro_rules! slice_from_mut {
    ($ptr:expr, $len:expr) => {
        unsafe { std::slice::from_raw_parts_mut($ptr, $len) }
    };
}

#[macro_export]
macro_rules! slice_from_ref {
    ($ptr:expr, $len:expr) => {
        unsafe { std::slice::from_raw_parts($ptr, $len) }
    };
}

#[macro_export]
macro_rules! raw_parts {
    ($ptr:expr, $len:expr) => {
        unsafe { std::slice::from_raw_parts($ptr, $len) }
    };
}

pub fn run_inference(
    source: &dyn TensorSource,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    n_threads_arg: usize,
    thinking: bool,
    bench: bool,
    profile: bool,
    kv_format: KvFormat,
) -> Result<(), String> {
    let input_tokens = {
        let tokenizer = BPETokenizer::from_gguf_metadata(|k| source.metadata(k).cloned())
            .map_err(|error| format!("Failed to initialize tokenizer: {error}"))?;

        if bench {
            tokenizer.encode(
                prompt,
                EncodeOptions {
                    add_special: true,
                    parse_special: true,
                },
            )
        } else {
            build_qwen_chat_prompt(
                &tokenizer,
                &[QwenMessage {
                    role: "user",
                    content: prompt,
                }],
                thinking,
            )?
        }
    };

    run_inference_tokens(
        source,
        input_tokens,
        max_tokens,
        temperature,
        n_threads_arg,
        bench,
        profile,
        kv_format,
    )
}

pub fn run_inference_tokens(
    source: &dyn TensorSource,
    input_tokens: Vec<u32>,
    max_tokens: usize,
    temperature: f32,
    n_threads_arg: usize,
    bench: bool,
    profile: bool,
    kv_format: KvFormat,
) -> Result<(), String> {
    let t0 = Instant::now();
    let config = model_config_from_source(source)
        .map_err(|error| format!("Failed to parse model config: {error}"))?;

    let arch = source
        .metadata("general.architecture")
        .and_then(|v| v.to_string_val())
        .unwrap_or_default();

    let tokenizer = BPETokenizer::from_gguf_metadata(|k| source.metadata(k).cloned())
        .map_err(|error| format!("Failed to initialize tokenizer: {error}"))?;

    let vocab = tokenizer.vocab_size();

    let max_ctx = 512usize.min(config.n_ctx);
    let n_embd = config.n_embd;
    let n_layer = config.n_layer;
    let n_head = config.n_head;
    let n_head_kv = config.n_head_kv;
    let n_embd_head = config.n_embd_head;
    let n_embd_head_k = if let Some(v) = source.metadata(&format!("{}.attention.key_length", arch)) {
        v.to_u64().unwrap_or(n_embd_head as u64) as usize
    } else {
        n_embd_head
    };
    let n_embd_head_v =
        if let Some(v) = source.metadata(&format!("{}.attention.value_length", arch)) {
            v.to_u64().unwrap_or(n_embd_head as u64) as usize
        } else {
            n_embd_head
        };
    let n_embd_q = n_head * n_embd_head_k;
    let n_embd_gqa = n_head_kv * n_embd_head_v;
    let n_ff = config.n_ff;
    let eps = config.norm_eps;
    let freq_base = config.rope_freq_base;

    let output_norm = get_f32_tensor(source, "output_norm.weight", n_embd);
    let embd_info = source.tensor_info("token_embd.weight").expect("no token_embd.weight");
    if !matches!(embd_info.ggml_type, GGMLType::F16 | GGMLType::Q8_0 | GGMLType::Q4_0 | GGMLType::Q6K) {
        panic!(
            "token_embd.weight has unsupported type {:?}; only F16, Q8_0, Q4_0, and Q6K are supported",
            embd_info.ggml_type
        );
    }
    let embd_weight = source.tensor_slice("token_embd.weight").expect("no embd");
    let embd_weight_static: &'static [u8] = unsafe { std::mem::transmute(embd_weight) };
    let token_embedding = Weight::from_quantized(QuantizedTensor::from_bytes(
        embd_weight_static,
        embd_info.ggml_type,
        n_embd,
        vocab,
    ));
    let output_weight = source.tensor_slice("output.weight").unwrap_or(embd_weight);
    let output_weight_static: &'static [u8] = unsafe { std::mem::transmute(output_weight) };
    let output_type = source.tensor_info("output.weight").unwrap_or(embd_info).ggml_type;
    let output_weight_quantized = Weight::from_quantized(QuantizedTensor::from_bytes(
        output_weight_static,
        output_type,
        n_embd,
        vocab,
    ));

    let layers: Vec<Qwen3LayerWeights> =
        load_layers(source, n_layer, n_embd, n_embd_q, n_embd_gqa, n_ff, n_embd_head_k, true);

    let load_ms = t0.elapsed().as_millis();
    println!(
        "Model: {} | n_embd={} n_layer={} n_head={} n_head_kv={} n_ff={} | loaded in {}ms",
        arch, n_embd, n_layer, n_head, n_head_kv, n_ff, load_ms
    );

    let kv_cache = match kv_format {
        KvFormat::F16 => KvCache::new_f16(n_layer, max_ctx, n_embd_gqa),
        KvFormat::F32 => KvCache::new_f32(n_layer, max_ctx, n_embd_gqa),
    };

    let available_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let n_threads = resolve_thread_count(n_threads_arg, available_threads);

    let mut scratch = ExecutionScratchpad::new(n_embd, n_embd_q, n_embd_gqa, n_ff, vocab, n_threads, max_ctx);
    let pool = Arc::new(ComputePool::new(n_threads));
    eprintln!("compute pool: {} threads", pool.n_threads());
    println!("Prompt: {} tokens", input_tokens.len());

    let eos_id = tokenizer.eos_id();
    let im_end_id = tokenizer.special_token_id("im_end");
    let mut generated_tokens: Vec<u32> = Vec::new();
    let mut all_tokens: Vec<u32> = input_tokens.clone();
    let mut decoder = tokenizer.streaming_decoder(false);

    let group_size = n_head / n_head_kv;
    let kq_scale = 1.0f32 / (n_embd_head_k as f32).sqrt();

    let mut t_norm: f64 = 0.0;
    let _t_quant: f64 = 0.0;
    let mut t_qkv: f64 = 0.0;
    let mut t_wo: f64 = 0.0;
    let mut t_ffn1: f64 = 0.0;
    let _t_silu: f64 = 0.0;
    let _t_down: f64 = 0.0;
    let mut t_logits: f64 = 0.0;

    print!("Output: ");
    io::stdout().flush().unwrap();

    let t_infer = Instant::now();
    let total_steps = inference_step_budget(input_tokens.len(), max_tokens, bench);
    let mut prefill_evals = 0usize;
    let mut prefill_time = Duration::ZERO;
    let mut decode_evals = 0usize;
    let mut decode_time = Duration::ZERO;

    for step in 0..total_steps {
        let eval_started = Instant::now();
        let token_id = if step < input_tokens.len() {
            input_tokens[step]
        } else {
            *generated_tokens.last().unwrap_or(&0)
        };

        let pos = step;

        token_embedding.embedding_lookup(token_id, &mut scratch.x);

        for layer in 0..n_layer {
            let lw = &layers[layer];

            let x_ptr = scratch.x.as_mut_ptr();
            let normed_ptr = scratch.normed.as_mut_ptr();
            let q_ptr = scratch.q.as_mut_ptr();
            let k_ptr = scratch.k_new.as_mut_ptr();
            let v_ptr = scratch.v_new.as_mut_ptr();
            let attn_out_ptr = scratch.attn_out.as_mut_ptr();
            let attn_proj_ptr = scratch.attn_proj.as_mut_ptr();
            let down_buf_ptr = scratch.down_buf.as_mut_ptr();
            let scores_ptr = scratch.scores.as_mut_ptr();
            let score_stride = scratch.score_stride;
            let gate_buf_ptr = scratch.gate_buf.as_mut_ptr();
            let up_buf_ptr = scratch.up_buf.as_mut_ptr();
            let q8_buf_ptr = scratch.q8_buf.as_mut_ptr();
            let scale_buf_ptr = scratch.scale_buf.as_mut_ptr();
            let q8k_buf_ptr = scratch.q8k_buf.as_mut_ptr();
            let kv_cache_size = n_layer * max_ctx * n_embd_gqa;
            let (k_cache_f16_ptr, v_cache_f16_ptr) = match &kv_cache {
                KvCache::F16(c) => (c.k.as_ptr() as *mut u16, c.v.as_ptr() as *mut u16),
                _ => (std::ptr::null_mut(), std::ptr::null_mut()),
            };
            let (k_cache_f32_ptr, v_cache_f32_ptr) = match &kv_cache {
                KvCache::F32(c) => (c.k.as_ptr() as *mut f32, c.v.as_ptr() as *mut f32),
                _ => (std::ptr::null_mut(), std::ptr::null_mut()),
            };

            let max_n_in = n_embd_q.max(n_ff);
            let x = slice_from_mut!(x_ptr, n_embd);
            let normed = slice_from_mut!(normed_ptr, n_embd);
            let q8_buf = slice_from_mut!(q8_buf_ptr, max_n_in);
            let scale_buf = slice_from_mut!(scale_buf_ptr, max_n_in / 32);
            let q8k_buf = slice_from_mut!(q8k_buf_ptr, max_n_in / 256);

            let t0 = Instant::now();
            rms_norm(x, &lw.attn_norm, normed, eps);
            quantize_q8_0_into(normed, n_embd, &mut q8_buf[..n_embd], &mut scale_buf[..n_embd / 32]);
            let q8 = q8_buf[..n_embd].as_ptr();
            let sc = scale_buf[..n_embd / 32].as_ptr();
            crate::ops::quantize_row_q8_k_into(normed, &mut q8k_buf[..n_embd / 256]);
            let q8k = q8k_buf[..n_embd / 256].as_ptr();

            pool.compute(move |ith: usize, nth: usize| {
                let input = raw_parts!(normed_ptr, n_embd);
                let q8 = raw_parts!(q8, n_embd);
                let sc = raw_parts!(sc, n_embd / 32);
                let q8k = raw_parts!(q8k, n_embd / 256);
                let q = slice_from_mut!(q_ptr, n_embd_q);
                let k_new = slice_from_mut!(k_ptr, n_embd_gqa);
                let v_new = slice_from_mut!(v_ptr, n_embd_gqa);

                lw.wq.kernel.forward_prepared(input, q8, sc, Some(q8k), q, n_embd, n_embd_q, ith, nth);
                lw.wk.kernel.forward_prepared(input, q8, sc, Some(q8k), k_new, n_embd, n_embd_gqa, ith, nth);
                lw.wv.kernel.forward_prepared(input, q8, sc, Some(q8k), v_new, n_embd, n_embd_gqa, ith, nth);
            });

            {
                let q = slice_from_mut!(q_ptr, n_embd_q);
                let k_new = slice_from_mut!(k_ptr, n_embd_gqa);
                let v_new = slice_from_mut!(v_ptr, n_embd_gqa);
                let q_norm = lw.q_norm.as_deref();
                let k_norm = lw.k_norm.as_deref();

                if let (Some(qn), Some(kn)) = (q_norm, k_norm) {
                    for h in 0..n_head {
                        rms_norm_inplace(&mut q[h * n_embd_head_k..(h + 1) * n_embd_head_k], qn, eps);
                    }
                    for h in 0..n_head_kv {
                        rms_norm_inplace(&mut k_new[h * n_embd_head_k..(h + 1) * n_embd_head_k], kn, eps);
                    }
                }

                for h in 0..n_head {
                    rope_neox(&mut q[h * n_embd_head_k..(h + 1) * n_embd_head_k], pos, n_embd_head_k, freq_base);
                }
                for h in 0..n_head_kv {
                    rope_neox(&mut k_new[h * n_embd_head_k..(h + 1) * n_embd_head_k], pos, n_embd_head_v, freq_base);
                }

                let kb = layer * max_ctx * n_embd_gqa;

                if kv_format == KvFormat::F16 {
                    let k_cache = slice_from_mut!(k_cache_f16_ptr, kv_cache_size);
                    let v_cache = slice_from_mut!(v_cache_f16_ptr, kv_cache_size);
                    for h in 0..n_head_kv {
                        let off = h * n_embd_head_k;
                        f32_slice_to_f16(
                            &k_new[off..off + n_embd_head_k],
                            &mut k_cache[kb + pos * n_embd_gqa + off..kb + pos * n_embd_gqa + off + n_embd_head_k],
                        );
                        f32_slice_to_f16(
                            &v_new[off..off + n_embd_head_v],
                            &mut v_cache[kb + pos * n_embd_gqa + off..kb + pos * n_embd_gqa + off + n_embd_head_v],
                        );
                    }
                } else {
                    let k_cache = slice_from_mut!(k_cache_f32_ptr, kv_cache_size);
                    let v_cache = slice_from_mut!(v_cache_f32_ptr, kv_cache_size);
                    for h in 0..n_head_kv {
                        let off = h * n_embd_head_k;
                        k_cache[kb + pos * n_embd_gqa + off..kb + pos * n_embd_gqa + off + n_embd_head_k]
                            .copy_from_slice(&k_new[off..off + n_embd_head_k]);
                        v_cache[kb + pos * n_embd_gqa + off..kb + pos * n_embd_gqa + off + n_embd_head_v]
                            .copy_from_slice(&v_new[off..off + n_embd_head_v]);
                    }
                }
            }

            pool.compute(move |ith: usize, nth: usize| {
                let q = slice_from_ref!(q_ptr, n_embd_q);
                let attn_out = slice_from_mut!(attn_out_ptr, n_embd_q);
                let h_start = ith * n_head / nth;
                let h_end = (ith + 1) * n_head / nth;

                let kb = layer * max_ctx * n_embd_gqa;

                if kv_format == KvFormat::F16 {
                    let k_cache = slice_from_ref!(k_cache_f16_ptr, kv_cache_size);
                    let v_cache = slice_from_ref!(v_cache_f16_ptr, kv_cache_size);
                    for h in h_start..h_end {
                        let kv_h = h / group_size;
                        let q_off = h * n_embd_head_k;
                        let n_cached = pos + 1;
                        let n_padded = (n_cached + 255) / 256 * 256;
                        let out_base = h * n_embd_head_v;
                        let mut ms = 0.0f32;
                        let mut s_sum = 0.0f32;
                        for d in 0..n_embd_head_v {
                            attn_out[out_base + d] = 0.0;
                        }
                        for t in 0..n_cached {
                            let score = dot_f16_f32(
                                &q[q_off..q_off + n_embd_head_k],
                                &k_cache[kb + t * n_embd_gqa + kv_h * n_embd_head_v
                                    ..kb + t * n_embd_gqa + kv_h * n_embd_head_v + n_embd_head_k],
                                n_embd_head_k,
                            ) * kq_scale;
                            if score > ms {
                                let rescale = (ms - score).exp();
                                vec_scale_f32(&mut attn_out[out_base..out_base + n_embd_head_v], rescale);
                                s_sum *= rescale;
                                ms = score;
                            }
                            let vs = (score - ms).exp();
                            let v_base = kb + t * n_embd_gqa + kv_h * n_embd_head_v;
                            vec_mad_f16_f32(&mut attn_out[out_base..out_base + n_embd_head_v], &v_cache[v_base..v_base + n_embd_head_v], vs);
                            s_sum += vs;
                        }
                        let inv_sum = 1.0 / s_sum;
                        vec_scale_f32(&mut attn_out[out_base..out_base + n_embd_head_v], inv_sum);
                    }
                } else {
                    let k_cache = slice_from_ref!(k_cache_f32_ptr, kv_cache_size);
                    let v_cache = slice_from_ref!(v_cache_f32_ptr, kv_cache_size);
                    let scores = slice_from_mut!(scores_ptr, n_threads * score_stride);
                    for h in h_start..h_end {
                        let kv_h = h / group_size;
                        let q_off = h * n_embd_head_k;
                        let n_cached = pos + 1;
                        let n_padded = (n_cached + 255) / 256 * 256;
                        let out_base = h * n_embd_head_v;
                        let s_off = ith * score_stride;
                        for t in 0..n_cached {
                            scores[s_off + t] = dot_f32(
                                &q[q_off..q_off + n_embd_head_k],
                                &k_cache[kb + t * n_embd_gqa + kv_h * n_embd_head_v
                                    ..kb + t * n_embd_gqa + kv_h * n_embd_head_v + n_embd_head_k],
                                n_embd_head_k,
                            ) * kq_scale;
                        }
                        scores[s_off + n_cached..s_off + n_padded].fill(f32::NEG_INFINITY);
                        softmax_inplace(&mut scores[s_off..s_off + n_padded]);
                        let mut values = [0.0f32; 512];
                        for d in 0..n_embd_head_v {
                            for t in 0..n_cached {
                                values[t] = v_cache[kb + t * n_embd_gqa + kv_h * n_embd_head_v + d];
                            }
                            attn_out[out_base + d] = attention_value_f32(
                                &values[..n_padded],
                                &scores[s_off..s_off + n_padded],
                                n_cached,
                                n_padded,
                            );
                        }
                    }
                }
            });
            t_qkv += t0.elapsed().as_secs_f64();

            let attn_out = slice_from_mut!(attn_out_ptr, n_embd_q);
            let q8_buf = slice_from_mut!(q8_buf_ptr, max_n_in);
            let scale_buf = slice_from_mut!(scale_buf_ptr, max_n_in / 32);
            let q8k_buf = slice_from_mut!(q8k_buf_ptr, max_n_in / 256);
            let t0 = Instant::now();
            quantize_q8_0_into(attn_out, n_embd_q, &mut q8_buf[..n_embd_q], &mut scale_buf[..n_embd_q / 32]);
            crate::ops::quantize_row_q8_k_into(attn_out, &mut q8k_buf[..n_embd_q / 256]);
            let q8 = q8_buf[..n_embd_q].as_ptr();
            let sc = scale_buf[..n_embd_q / 32].as_ptr();
            let q8k = q8k_buf[..n_embd_q / 256].as_ptr();
            pool.compute(move |ith: usize, nth: usize| {
                let input = raw_parts!(attn_out_ptr, n_embd_q);
                let q8 = raw_parts!(q8, n_embd_q);
                let sc = raw_parts!(sc, n_embd_q / 32);
                let q8k = raw_parts!(q8k, n_embd_q / 256);
                let attn_proj = slice_from_mut!(attn_proj_ptr, n_embd);
                lw.wo.kernel.forward_prepared(input, q8, sc, Some(q8k), attn_proj, n_embd_q, n_embd, ith, nth);
            });
            t_wo += t0.elapsed().as_secs_f64();

            let attn_proj = slice_from_mut!(attn_proj_ptr, n_embd);
            let x = slice_from_mut!(x_ptr, n_embd);
            let normed = slice_from_mut!(normed_ptr, n_embd);
            for i in 0..n_embd {
                x[i] += attn_proj[i];
            }

            // FFN — uses `kernel.forward_prepared` with `ExecutionScratchpad`-managed
            // buffers instead of `quantize_and_matmul_with_scratch`.  This enables
            // two optimizations:
            //
            // 1. **Fused gate+up**: both projections share the same quantized input
            //    and run inside a single `pool.compute` call, avoiding a second
            //    round of quantization.
            //
            // 2. **Buffer reuse**: `gate_buf / up_buf / down_buf` are allocated once
            //    in `ExecutionScratchpad` and reused across every token position.
            //
            // Contrast with embedding.rs which processes each token independently
            // and uses `quantize_and_matmul_with_scratch` — a natural fit when there
            // is no pre-allocated scratch context.
            let t0 = Instant::now();
            rms_norm(x, &lw.ffn_norm, normed, eps);
            quantize_q8_0_into(normed, n_embd, &mut q8_buf[..n_embd], &mut scale_buf[..n_embd / 32]);
            crate::ops::quantize_row_q8_k_into(normed, &mut q8k_buf[..n_embd / 256]);
            let q8 = q8_buf[..n_embd].as_ptr();
            let sc = scale_buf[..n_embd / 32].as_ptr();
            let q8k = q8k_buf[..n_embd / 256].as_ptr();

            pool.compute(move |ith: usize, nth: usize| {
                let input = raw_parts!(normed_ptr, n_embd);
                let q8 = raw_parts!(q8, n_embd);
                let sc = raw_parts!(sc, n_embd / 32);
                let q8k = raw_parts!(q8k, n_embd / 256);
                let gate_buf = slice_from_mut!(gate_buf_ptr, n_ff);
                let up_buf = slice_from_mut!(up_buf_ptr, n_ff);
                lw.w_gate.kernel.forward_prepared(input, q8, sc, Some(q8k), up_buf, n_embd, n_ff, ith, nth);
                lw.w_up.kernel.forward_prepared(input, q8, sc, Some(q8k), gate_buf, n_embd, n_ff, ith, nth);

                let rows_per = n_ff / nth;
                let r_start = ith * rows_per;
                let r_end = if ith == nth - 1 { n_ff } else { r_start + rows_per };
                silu_mul_approx_inplace(&up_buf[r_start..r_end], &mut gate_buf[r_start..r_end]);
            });

            {
                let gate_buf = slice_from_mut!(gate_buf_ptr, n_ff);
                let q8_buf = slice_from_mut!(q8_buf_ptr, max_n_in);
                let scale_buf = slice_from_mut!(scale_buf_ptr, max_n_in / 32);
                let q8k_buf = slice_from_mut!(q8k_buf_ptr, max_n_in / 256);
                quantize_q8_0_into(gate_buf, n_ff, &mut q8_buf[..n_ff], &mut scale_buf[..n_ff / 32]);
                crate::ops::quantize_row_q8_k_into(gate_buf, &mut q8k_buf[..n_ff / 256]);
            }

            let q8 = q8_buf[..n_ff].as_ptr();
            let sc = scale_buf[..n_ff / 32].as_ptr();
            let q8k = q8k_buf[..n_ff / 256].as_ptr();
            pool.compute(move |ith: usize, nth: usize| {
                let input = raw_parts!(gate_buf_ptr, n_ff);
                let q8 = raw_parts!(q8, n_ff);
                let sc = raw_parts!(sc, n_ff / 32);
                let q8k = raw_parts!(q8k, n_ff / 256);
                let down_buf = slice_from_mut!(down_buf_ptr, n_embd);
                lw.w_down.kernel.forward_prepared(input, q8, sc, Some(q8k), down_buf, n_ff, n_embd, ith, nth);
            });
            t_ffn1 += t0.elapsed().as_secs_f64();

            let down_buf = slice_from_mut!(down_buf_ptr, n_embd);
            let x = slice_from_mut!(x_ptr, n_embd);
            for i in 0..n_embd {
                x[i] += down_buf[i];
            }
        }

        {
            let x = &mut scratch.x;
            let normed = &mut scratch.normed;
            let logits_ptr = scratch.logits.as_mut_ptr();
            let q8_buf = &mut scratch.q8_buf;
            let scale_buf = &mut scratch.scale_buf;
            let q8k_buf = &mut scratch.q8k_buf;

            let t0 = Instant::now();
            rms_norm(x, &output_norm, normed, eps);
            t_norm += t0.elapsed().as_secs_f64();

            let t0 = Instant::now();
            quantize_q8_0_into(normed, n_embd, &mut q8_buf[..n_embd], &mut scale_buf[..n_embd / 32]);
            crate::ops::quantize_row_q8_k_into(normed, &mut q8k_buf[..n_embd / 256]);
            let q8 = q8_buf[..n_embd].as_ptr();
            let sc = scale_buf[..n_embd / 32].as_ptr();
            let q8k = q8k_buf[..n_embd / 256].as_ptr();
            let input = normed.as_ptr();
            let output_pw = Weight::from_quantized(crate::ops::kernel::QuantizedTensor::from_bytes(output_weight, output_type, n_embd, vocab));
            pool.compute(move |ith: usize, nth: usize| {
                let input = raw_parts!(input, n_embd);
                let q8 = raw_parts!(q8, n_embd);
                let sc = raw_parts!(sc, n_embd / 32);
                let q8k = raw_parts!(q8k, n_embd / 256);
                let logits = slice_from_mut!(logits_ptr, vocab);
                output_pw.kernel.forward_prepared(input, q8, sc, Some(q8k), logits, n_embd, vocab, ith, nth);
            });
            t_logits += t0.elapsed().as_secs_f64();
        }

        let eval_elapsed = eval_started.elapsed();
        if step < input_tokens.len() {
            prefill_evals += 1;
            prefill_time += eval_elapsed;
        } else {
            decode_evals += 1;
            decode_time += eval_elapsed;
        }

        if step < input_tokens.len() - 1 {
            continue;
        }

        let logits = &mut scratch.logits;
        let chosen = if temperature <= 0.0 {
            logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i).unwrap_or(0)
        } else {
            for l in logits.iter_mut() {
                *l /= temperature;
            }
            let top = crate::ops::sample_top_k(logits, 40);
            let mut rng = 0u64;
            for &t in &all_tokens {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(t as u64);
            }
            let r = ((rng >> 33) as f32) / (1u64 << 31) as f32;
            let mut cum = 0.0f32;
            let mut chosen = top[0].0;
            for &(idx, prob) in &top {
                cum += prob;
                if cum >= r {
                    chosen = idx;
                    break;
                }
            }
            chosen
        };

        let chosen_id = chosen as u32;
        if !bench && (eos_id == Some(chosen_id) || im_end_id == Some(chosen_id)) {
            break;
        }
        if generated_tokens.len() >= max_tokens {
            break;
        }

        generated_tokens.push(chosen_id);
        all_tokens.push(chosen_id);

        let text = decoder.push(chosen_id);
        print!("{}", text);
        io::stdout().flush().unwrap();

        if generated_tokens.len() == 1 {
            eprintln!();
        }
    }

    let tail = decoder.finish();
    if !tail.is_empty() {
        print!("{}", tail);
        io::stdout().flush().unwrap();
    }

    let infer_ms = t_infer.elapsed().as_millis();
    let tok_s = if infer_ms > 0 {
        generated_tokens.len() as f64 / infer_ms as f64 * 1000.0
    } else {
        0.0
    };
    let total = t_norm + _t_quant + t_qkv + t_wo + t_ffn1 + t_logits;
    if bench || profile {
        eprintln!();
    }
    if bench {
        eprintln!(
            "BENCH: pp {} evals in {:.3}s | {:.1} eval/s",
            prefill_evals,
            prefill_time.as_secs_f64(),
            per_second(prefill_evals, prefill_time),
        );
        eprintln!(
            "BENCH: tg {} evals in {:.3}s | {:.1} eval/s",
            decode_evals,
            decode_time.as_secs_f64(),
            per_second(decode_evals, decode_time),
        );
    }
    eprintln!(
        "Prompt: {:.1} t/s | Generation: {:.1} t/s | end-to-end: {:.1} tok/s",
        per_second(prefill_evals, prefill_time),
        per_second(decode_evals, decode_time),
        tok_s
    );
    if profile {
        eprintln!(
            "PROFILE: norm={:.1}% quant={:.1}% qkv+attn={:.1}% wo={:.1}% ffn={:.1}% logits={:.1}%",
            t_norm / total * 100.0,
            _t_quant / total * 100.0,
            t_qkv / total * 100.0,
            t_wo / total * 100.0,
            t_ffn1 / total * 100.0,
            t_logits / total * 100.0
        );
    }
    println!();
    println!("[{} output tokens in {}ms]", generated_tokens.len(), infer_ms);
    Ok(())
}
