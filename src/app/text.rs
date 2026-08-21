use crate::app::cli::{resolve_thread_count, per_second, inference_step_budget, KvFormat};
use crate::app::{LayerWeights, get_f32_tensor, slice_from_mut, slice_from_ref, raw_parts};
use crate::format::ggufrs::{open_model_source, ComponentRole};
use crate::core::loader::model_config_from_source;
use crate::core::tensor::{GGMLType, TensorSource};
use crate::ops::{dot_f32, dot_f16_f32, f32_slice_to_f16, quantize_q8_0_into, rms_norm, rms_norm_inplace, rope_neox, silu_mul_inplace, softmax, attention_value_f32, vec_mad_f16_f32, vec_scale_f32};
use crate::prompt::{append_qwen_assistant_prefix, append_qwen_message_tokens, build_hunyuan_chat_prompt, build_qwen_chat_prompt, HunyuanMessage, QwenMessage};
use crate::models::qwen35::{build_qwen35_positions, Qwen35Model};
use crate::models::qwen3::{qwen_text_positions, Qwen3GenerateOptions, Qwen3Input, Qwen3Model};
use crate::core::scratchpad::{ExecutionScratchpad, KvCache};
use crate::core::thread_pool::ComputePool;
use crate::core::tokenizer::{BPETokenizer, EncodeOptions};
use crate::models::vision::{qwen_smart_resize, VisionEncoder, VisionScratchpad};
use crate::ops::embedding_lookup;
use crate::ops::kernel::Kernel;
use std::io::{self, Write};
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

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
    let t0 = Instant::now();
    let config = model_config_from_source(source)
        .map_err(|error| format!("Failed to parse model config: {error}"))?;

    let arch = source
        .metadata("general.architecture")
        .and_then(|v| v.to_string_val())
        .unwrap_or_default();
    let is_qwen3 = arch == "qwen3" || arch == "hunyuan-dense";

    if arch == "pig" {
        return Err("pig model requires image generation flow, not text inference".into());
    }

    let tokenizer = BPETokenizer::from_gguf_metadata(|k| source.metadata(k).cloned())
        .map_err(|error| format!("Failed to initialize tokenizer: {error}"))?;

    let max_ctx = 512usize.min(config.n_ctx);
    let n_embd = config.n_embd;
    let n_layer = config.n_layer;
    let n_head = config.n_head;
    let n_head_kv = config.n_head_kv;
    let n_embd_head = config.n_embd_head;
    let n_embd_head_k = if let Some(v) = source.metadata(&format!("{}.attention.key_length", arch))
    {
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
    let output_weight = source.tensor_slice("output.weight").unwrap_or(embd_weight);
    let embd_type = embd_info.ggml_type;
    let output_type = source
        .tensor_info("output.weight")
        .unwrap_or(embd_info)
        .ggml_type;

    let layers: Vec<LayerWeights> = (0..n_layer)
        .map(|l| LayerWeights {
            attn_norm: get_f32_tensor(source, &format!("blk.{}.attn_norm.weight", l), n_embd),
            ffn_norm: get_f32_tensor(source, &format!("blk.{}.ffn_norm.weight", l), n_embd),
            q_norm: if is_qwen3 {
                Some(get_f32_tensor(
                    source,
                    &format!("blk.{}.attn_q_norm.weight", l),
                    n_embd_head_k,
                ))
            } else {
                None
            },
            k_norm: if is_qwen3 {
                Some(get_f32_tensor(
                    source,
                    &format!("blk.{}.attn_k_norm.weight", l),
                    n_embd_head_k,
                ))
            } else {
                None
            },
            wq: {
                let info = source.tensor_info(&format!("blk.{}.attn_q.weight", l)).unwrap();
                crate::ops::kernel::QuantizedTensor::from_bytes(
                    source.tensor_slice(&format!("blk.{}.attn_q.weight", l)).unwrap(),
                    info.ggml_type,
                    n_embd,
                    n_embd_q,
                ).into_kernel()
            },
            wk: {
                let info = source.tensor_info(&format!("blk.{}.attn_k.weight", l)).unwrap();
                crate::ops::kernel::QuantizedTensor::from_bytes(
                    source.tensor_slice(&format!("blk.{}.attn_k.weight", l)).unwrap(),
                    info.ggml_type,
                    n_embd,
                    n_embd_gqa,
                ).into_kernel()
            },
            wv: {
                let info = source.tensor_info(&format!("blk.{}.attn_v.weight", l)).unwrap();
                crate::ops::kernel::QuantizedTensor::from_bytes(
                    source.tensor_slice(&format!("blk.{}.attn_v.weight", l)).unwrap(),
                    info.ggml_type,
                    n_embd,
                    n_embd_gqa,
                ).into_kernel()
            },
            wo: {
                let info = source.tensor_info(&format!("blk.{}.attn_output.weight", l)).unwrap();
                crate::ops::kernel::QuantizedTensor::from_bytes(
                    source.tensor_slice(&format!("blk.{}.attn_output.weight", l)).unwrap(),
                    info.ggml_type,
                    n_embd_q,
                    n_embd,
                ).into_kernel()
            },
            w_gate: {
                let info = source.tensor_info(&format!("blk.{}.ffn_gate.weight", l)).unwrap();
                crate::ops::kernel::QuantizedTensor::from_bytes(
                    source.tensor_slice(&format!("blk.{}.ffn_gate.weight", l)).unwrap(),
                    info.ggml_type,
                    n_embd,
                    n_ff,
                ).into_kernel()
            },
            w_up: {
                let info = source.tensor_info(&format!("blk.{}.ffn_up.weight", l)).unwrap();
                crate::ops::kernel::QuantizedTensor::from_bytes(
                    source.tensor_slice(&format!("blk.{}.ffn_up.weight", l)).unwrap(),
                    info.ggml_type,
                    n_embd,
                    n_ff,
                ).into_kernel()
            },
            w_down: {
                let info = source.tensor_info(&format!("blk.{}.ffn_down.weight", l)).unwrap();
                crate::ops::kernel::QuantizedTensor::from_bytes(
                    source.tensor_slice(&format!("blk.{}.ffn_down.weight", l)).unwrap(),
                    info.ggml_type,
                    n_ff,
                    n_embd,
                ).into_kernel()
            },
        })
        .collect();

    let load_ms = t0.elapsed().as_millis();
    println!(
        "Model: {} | n_embd={} n_layer={} n_head={} n_head_kv={} n_ff={} | loaded in {}ms",
        arch, n_embd, n_layer, n_head, n_head_kv, n_ff, load_ms
    );

    let kv_cache = match kv_format {
        KvFormat::F16 => KvCache::new_f16(n_layer, max_ctx, n_embd_gqa),
        KvFormat::F32 => KvCache::new_f32(n_layer, max_ctx, n_embd_gqa),
    };

    let vocab = tokenizer.vocab_size();
    let input_tokens = if bench {
        tokenizer.encode(
            prompt,
            EncodeOptions {
                add_special: true,
                parse_special: true,
            },
        )
    } else if arch == "hunyuan-dense" {
        build_hunyuan_chat_prompt(
            &tokenizer,
            &[HunyuanMessage {
                role: "user",
                content: prompt,
            }],
            true,
        )?
    } else {
        build_qwen_chat_prompt(
            &tokenizer,
            &[QwenMessage {
                role: "user",
                content: prompt,
            }],
            thinking,
        )?
    };

    let available_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let n_threads = resolve_thread_count(n_threads_arg, available_threads);

    let mut scratch = ExecutionScratchpad::new(
        n_embd, n_embd_q, n_embd_gqa, n_ff, vocab, n_threads, max_ctx,
    );
    let pool = Arc::new(ComputePool::new(n_threads));
    eprintln!("compute pool: {} threads", pool.n_threads());
    println!("Prompt: {} ({} tokens)", prompt, input_tokens.len());

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

        embedding_lookup(embd_weight, token_id, n_embd, embd_type, &mut scratch.x);

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

            let t0 = Instant::now();
            rms_norm(x, &lw.attn_norm, normed, eps);
            quantize_q8_0_into(
                normed,
                n_embd,
                &mut q8_buf[..n_embd],
                &mut scale_buf[..n_embd / 32],
            );
            let q8 = q8_buf[..n_embd].as_ptr();
            let sc = scale_buf[..n_embd / 32].as_ptr();

            pool.compute(move |ith: usize, nth: usize| {
                let input = raw_parts!(normed_ptr, n_embd);
                let q8 = raw_parts!(q8, n_embd);
                let sc = raw_parts!(sc, n_embd / 32);
                let q = slice_from_mut!(q_ptr, n_embd_q);
                let k_new = slice_from_mut!(k_ptr, n_embd_gqa);
                let v_new = slice_from_mut!(v_ptr, n_embd_gqa);

                lw.wq.forward_prepared(input, q8, sc, q, n_embd, n_embd_q, ith, nth);
                lw.wk.forward_prepared(input, q8, sc, k_new, n_embd, n_embd_gqa, ith, nth);
                lw.wv.forward_prepared(input, q8, sc, v_new, n_embd, n_embd_gqa, ith, nth);
            });

            {
                let q = slice_from_mut!(q_ptr, n_embd_q);
                let k_new = slice_from_mut!(k_ptr, n_embd_gqa);
                let v_new = slice_from_mut!(v_ptr, n_embd_gqa);
                let q_norm = lw.q_norm.as_deref();
                let k_norm = lw.k_norm.as_deref();

                if let (Some(qn), Some(kn)) = (q_norm, k_norm) {
                    for h in 0..n_head {
                        rms_norm_inplace(
                            &mut q[h * n_embd_head_k..(h + 1) * n_embd_head_k],
                            qn,
                            eps,
                        );
                    }
                    for h in 0..n_head_kv {
                        rms_norm_inplace(
                            &mut k_new[h * n_embd_head_k..(h + 1) * n_embd_head_k],
                            kn,
                            eps,
                        );
                    }
                }

                for h in 0..n_head {
                    rope_neox(
                        &mut q[h * n_embd_head_k..(h + 1) * n_embd_head_k],
                        pos,
                        n_embd_head_k,
                        freq_base,
                    );
                }
                for h in 0..n_head_kv {
                    rope_neox(
                        &mut k_new[h * n_embd_head_k..(h + 1) * n_embd_head_k],
                        pos,
                        n_embd_head_v,
                        freq_base,
                    );
                }

                let kb = layer * max_ctx * n_embd_gqa;

                if kv_format == KvFormat::F16 {
                    let k_cache = slice_from_mut!(k_cache_f16_ptr, kv_cache_size);
                    let v_cache = slice_from_mut!(v_cache_f16_ptr, kv_cache_size);
                    for h in 0..n_head_kv {
                        let off = h * n_embd_head_k;
                        f32_slice_to_f16(
                            &k_new[off..off + n_embd_head_k],
                            &mut k_cache[kb + pos * n_embd_gqa + off
                                ..kb + pos * n_embd_gqa + off + n_embd_head_k],
                        );
                        f32_slice_to_f16(
                            &v_new[off..off + n_embd_head_v],
                            &mut v_cache[kb + pos * n_embd_gqa + off
                                ..kb + pos * n_embd_gqa + off + n_embd_head_v],
                        );
                    }
                } else {
                    let k_cache = slice_from_mut!(k_cache_f32_ptr, kv_cache_size);
                    let v_cache = slice_from_mut!(v_cache_f32_ptr, kv_cache_size);
                    for h in 0..n_head_kv {
                        let off = h * n_embd_head_k;
                        k_cache[kb + pos * n_embd_gqa + off
                            ..kb + pos * n_embd_gqa + off + n_embd_head_k]
                            .copy_from_slice(&k_new[off..off + n_embd_head_k]);
                        v_cache[kb + pos * n_embd_gqa + off
                            ..kb + pos * n_embd_gqa + off + n_embd_head_v]
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
                                vec_scale_f32(
                                    &mut attn_out[out_base..out_base + n_embd_head_v],
                                    rescale,
                                );
                                s_sum *= rescale;
                                ms = score;
                            }
                            let vs = (score - ms).exp();
                            let v_base = kb + t * n_embd_gqa + kv_h * n_embd_head_v;
                            vec_mad_f16_f32(
                                &mut attn_out[out_base..out_base + n_embd_head_v],
                                &v_cache[v_base..v_base + n_embd_head_v],
                                vs,
                            );
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
                        softmax(&mut scores[s_off..s_off + n_padded]);
                        let mut values = [0.0f32; 512];
                        for d in 0..n_embd_head_v {
                            for t in 0..n_cached {
                                values[t] = v_cache[
                                    kb + t * n_embd_gqa + kv_h * n_embd_head_v + d
                                ];
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
            let t0 = Instant::now();
            quantize_q8_0_into(
                attn_out,
                n_embd_q,
                &mut q8_buf[..n_embd_q],
                &mut scale_buf[..n_embd_q / 32],
            );
            let q8 = q8_buf[..n_embd_q].as_ptr();
            let sc = scale_buf[..n_embd_q / 32].as_ptr();
            pool.compute(move |ith: usize, nth: usize| {
                let input = raw_parts!(attn_out_ptr, n_embd_q);
                let q8 = raw_parts!(q8, n_embd_q);
                let sc = raw_parts!(sc, n_embd_q / 32);
                let attn_proj = slice_from_mut!(attn_proj_ptr, n_embd);
                lw.wo.forward_prepared(input, q8, sc, attn_proj, n_embd_q, n_embd, ith, nth);
            });
            t_wo += t0.elapsed().as_secs_f64();

            let attn_proj = slice_from_mut!(attn_proj_ptr, n_embd);
            let x = slice_from_mut!(x_ptr, n_embd);
            let normed = slice_from_mut!(normed_ptr, n_embd);
            for i in 0..n_embd {
                x[i] += attn_proj[i];
            }

            let t0 = Instant::now();
            rms_norm(x, &lw.ffn_norm, normed, eps);
            quantize_q8_0_into(
                normed,
                n_embd,
                &mut q8_buf[..n_embd],
                &mut scale_buf[..n_embd / 32],
            );
            let q8 = q8_buf[..n_embd].as_ptr();
            let sc = scale_buf[..n_embd / 32].as_ptr();

            pool.compute(move |ith: usize, nth: usize| {
                let input = raw_parts!(normed_ptr, n_embd);
                let q8 = raw_parts!(q8, n_embd);
                let sc = raw_parts!(sc, n_embd / 32);
                let gate_buf = slice_from_mut!(gate_buf_ptr, n_ff);
                let up_buf = slice_from_mut!(up_buf_ptr, n_ff);
                lw.w_gate.forward_prepared(input, q8, sc, up_buf, n_embd, n_ff, ith, nth);
                lw.w_up.forward_prepared(input, q8, sc, gate_buf, n_embd, n_ff, ith, nth);

                let rows_per = n_ff / nth;
                let r_start = ith * rows_per;
                let r_end = if ith == nth - 1 {
                    n_ff
                } else {
                    r_start + rows_per
                };
                silu_mul_inplace(&up_buf[r_start..r_end], &mut gate_buf[r_start..r_end]);
            });

            {
                let gate_buf = slice_from_mut!(gate_buf_ptr, n_ff);
                let q8_buf = slice_from_mut!(q8_buf_ptr, max_n_in);
                let scale_buf = slice_from_mut!(scale_buf_ptr, max_n_in / 32);
                quantize_q8_0_into(
                    gate_buf,
                    n_ff,
                    &mut q8_buf[..n_ff],
                    &mut scale_buf[..n_ff / 32],
                );
            }

            let q8 = q8_buf[..n_ff].as_ptr();
            let sc = scale_buf[..n_ff / 32].as_ptr();
            pool.compute(move |ith: usize, nth: usize| {
                let input = raw_parts!(gate_buf_ptr, n_ff);
                let q8 = raw_parts!(q8, n_ff);
                let sc = raw_parts!(sc, n_ff / 32);
                let down_buf = slice_from_mut!(down_buf_ptr, n_embd);
                lw.w_down.forward_prepared(input, q8, sc, down_buf, n_ff, n_embd, ith, nth);
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

            let t0 = Instant::now();
            rms_norm(x, &output_norm, normed, eps);
            t_norm += t0.elapsed().as_secs_f64();

            let t0 = Instant::now();
            quantize_q8_0_into(
                normed,
                n_embd,
                &mut q8_buf[..n_embd],
                &mut scale_buf[..n_embd / 32],
            );
            let q8 = q8_buf[..n_embd].as_ptr();
            let sc = scale_buf[..n_embd / 32].as_ptr();
            let input = normed.as_ptr();
            let output_pw = crate::ops::kernel::QuantizedTensor::from_bytes(
                output_weight,
                output_type,
                n_embd,
                vocab,
            );
            pool.compute(move |ith: usize, nth: usize| {
                let input = raw_parts!(input, n_embd);
                let q8 = raw_parts!(q8, n_embd);
                let sc = raw_parts!(sc, n_embd / 32);
                let logits = slice_from_mut!(logits_ptr, vocab);
                output_pw.forward_prepared(input, q8, sc, logits, n_embd, vocab, ith, nth);
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
            logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0)
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

pub fn run_interactive(
    source: &dyn TensorSource,
    max_tokens: usize,
    temperature: f32,
    n_threads_arg: usize,
) -> Result<(), String> {
    println!("=== RustModelInference Interactive Mode ===");
    println!("Type your prompt and press Enter. Ctrl+C to exit.\n");

    loop {
        print!("> ");
        io::stdout()
            .flush()
            .map_err(|error| format!("Failed to flush prompt: {error}"))?;
        let mut line = String::new();
        if io::stdin()
            .read_line(&mut line)
            .map_err(|error| format!("Failed to read prompt: {error}"))?
            == 0
        {
            break;
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        run_inference(
            source,
            line,
            max_tokens,
            temperature,
            n_threads_arg,
            false,
            false,
            false,
            KvFormat::F16,
        )?;
        println!();
    }
    Ok(())
}

pub fn run_shared_inference(
    source: Arc<dyn TensorSource>,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    n_threads_arg: usize,
    thinking: bool,
) -> Result<(), String> {
    let started = Instant::now();
    let tokenizer = std::sync::Arc::new(
        BPETokenizer::from_gguf_metadata(|key| source.metadata(key).cloned())
            .map_err(|error| format!("Failed to initialize tokenizer: {error}"))?,
    );
    let available_threads = std::thread::available_parallelism()
        .map(|threads| threads.get())
        .unwrap_or(4);
    let pool = std::sync::Arc::new(ComputePool::new(resolve_thread_count(
        n_threads_arg,
        available_threads,
    )));
    let model = Qwen3Model::from_source(source, std::sync::Arc::clone(&tokenizer), pool)?;
    let arch = model.config().architecture.clone();
    let input_tokens = if arch == "hunyuan-dense" {
        build_hunyuan_chat_prompt(
            &tokenizer,
            &[HunyuanMessage {
                role: "user",
                content: prompt,
            }],
            true,
        )?
    } else {
        build_qwen_chat_prompt(
            &tokenizer,
            &[QwenMessage {
                role: "user",
                content: prompt,
            }],
            thinking,
        )?
    };
    let positions = qwen_text_positions(input_tokens.len());
    println!(
        "Model: {} | n_embd={} n_layer={} n_head={} n_head_kv={} n_ff={} | loaded in {}ms",
        model.config().architecture,
        model.config().n_embd,
        model.config().n_layer,
        model.config().n_head,
        model.config().n_head_kv,
        model.config().n_ff,
        started.elapsed().as_millis(),
    );
    eprintln!("compute pool: {} threads", model.pool().n_threads());
    println!("Prompt: {} ({} tokens)", prompt, input_tokens.len());
    print!("Output: ");
    io::stdout().flush().map_err(|error| error.to_string())?;
    let inference_started = Instant::now();
    let generation = model.generate(
        Qwen3Input {
            token_ids: &input_tokens,
            positions: &positions,
            embeddings: None,
        },
        Qwen3GenerateOptions {
            max_new_tokens: max_tokens,
            temperature,
        },
    )?;
    print!("{}", generation.text);
    io::stdout().flush().map_err(|error| error.to_string())?;
    let elapsed_ms = inference_started.elapsed().as_millis();
    let tokens_per_second = if elapsed_ms > 0 {
        generation.token_ids.len() as f64 / elapsed_ms as f64 * 1000.0
    } else {
        0.0
    };
    println!();
    println!(
        "[end-to-end: {} output tokens in {}ms | {:.1} tok/s]",
        generation.token_ids.len(),
        elapsed_ms,
        tokens_per_second,
    );
    Ok(())
}

pub fn inject_vision_embeddings(
    llm: &Qwen35Model,
    tokens: &[i32],
    image_token_id: Option<i32>,
    vis_embd: &[f32],
    _n_vis_tokens: usize,
    proj_dim: usize,
) -> Vec<f32> {
    let n_embd = llm.config.n_embd;
    let n_tokens = tokens.len();
    let mut embeddings = vec![0.0f32; n_tokens * n_embd];

    let mut vis_idx = 0;

    for t in 0..n_tokens {
        if image_token_id == Some(tokens[t]) && vis_idx * proj_dim < vis_embd.len() {
            let embd_off = t * n_embd;
            let vis_off = vis_idx * proj_dim;
            if proj_dim == n_embd {
                embeddings[embd_off..embd_off + n_embd]
                    .copy_from_slice(&vis_embd[vis_off..vis_off + n_embd]);
            } else {
                for e in 0..n_embd.min(proj_dim) {
                    embeddings[embd_off + e] = vis_embd[vis_off + e];
                }
            }
            vis_idx += 1;
        } else {
            let tok = tokens[t] as usize;
            let tok_off = tok * n_embd;
            let embd_off = t * n_embd;
            for e in 0..n_embd {
                if tok_off + e < llm.tok_embd.len() {
                    embeddings[embd_off + e] = llm.tok_embd[tok_off + e];
                }
            }
        }
    }

    embeddings
}

pub fn sample_token(logits: &[f32], temperature: f32) -> i32 {
    if temperature <= 0.0 {
        return logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as i32)
            .unwrap_or(0);
    }
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    let mut probs = vec![0.0f32; logits.len()];
    for (i, l) in logits.iter().enumerate() {
        probs[i] = ((l - max_logit) / temperature).exp();
        sum += probs[i];
    }
    for p in probs.iter_mut() {
        *p /= sum;
    }

    let r = 0.5f32;
    let mut cumsum = 0.0f32;
    for (i, p) in probs.iter().enumerate() {
        cumsum += p;
        if cumsum >= r {
            return i as i32;
        }
    }
    (logits.len() - 1) as i32
}

pub fn decode_image(path: &Path) -> Result<image::DynamicImage, String> {
    let bytes = std::fs::read(path)
        .map_err(|error| format!("Failed to read image {}: {error}", path.display()))?;
    image::load_from_memory(&bytes)
        .map_err(|error| format!("Failed to decode image {}: {error}", path.display()))
}

pub fn normalize_resized_image(
    image: &image::DynamicImage,
    target_w: usize,
    target_h: usize,
    mean: &[f32; 3],
    std: &[f32; 3],
) -> Result<Vec<f32>, String> {
    if std.iter().any(|value| *value == 0.0) {
        return Err("Vision normalization std must be nonzero".into());
    }
    let width = u32::try_from(target_w).map_err(|_| "Vision width exceeds u32")?;
    let height = u32::try_from(target_h).map_err(|_| "Vision height exceeds u32")?;
    let resized = image
        .resize_exact(width, height, image::imageops::FilterType::Lanczos3)
        .to_rgb8();
    let output_len = target_w
        .checked_mul(target_h)
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or("Normalized image length overflow")?;
    let mut output = vec![0.0f32; output_len];
    for y in 0..target_h {
        for x in 0..target_w {
            let pixel = resized.get_pixel(x as u32, y as u32);
            let offset = (y * target_w + x) * 3;
            for channel in 0..3 {
                output[offset + channel] =
                    (f32::from(pixel[channel]) / 255.0 - mean[channel]) / std[channel];
            }
        }
    }
    Ok(output)
}

pub fn run_multimodal(
    llm_source: &dyn TensorSource,
    model_path: &Path,
    mmproj_path: Option<&Path>,
    image_path: Option<&Path>,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    n_threads_arg: usize,
) -> Result<(), String> {
    let arch = llm_source
        .metadata("general.architecture")
        .and_then(|v| v.to_string_val())
        .unwrap_or_default();
    println!("LLM arch: {}", arch);
    if arch != "qwen35" {
        return Err(format!(
            "Only qwen35 architecture is supported for multimodal, got: {arch}"
        ));
    }

    let (image_grid, vis_embeddings_vec) = if let Some(image_path) = image_path {
        let projector_path = mmproj_path.unwrap_or(model_path);
        println!("Loading mmproj {} ...", projector_path.display());
        let mmproj_source =
            open_model_source(projector_path, ComponentRole::Mmproj).map_err(|error| {
                if mmproj_path.is_none() {
                    format!(
                        "Model {} has no bundled mmproj; pass --mmproj: {error}",
                        model_path.display()
                    )
                } else {
                    format!(
                        "Failed to load mmproj {}: {error}",
                        projector_path.display()
                    )
                }
            })?;
        let mut encoder = VisionEncoder::from_source(mmproj_source.as_ref())
            .map_err(|error| format!("Failed to parse vision encoder: {error}"))?;
        encoder.precompute();
        println!(
            "Vision encoder loaded: {} layers, n_embd={}, image_size={}, patch_size={}, merge={}",
            encoder.config.n_layer,
            encoder.config.n_embd,
            encoder.config.image_size,
            encoder.config.patch_size,
            encoder.config.spatial_merge_size
        );
        let image = decode_image(image_path)?;
        let original_w = usize::try_from(image.width())
            .map_err(|_| "Original image width does not fit usize")?;
        let original_h = usize::try_from(image.height())
            .map_err(|_| "Original image height does not fit usize")?;
        let grid = qwen_smart_resize(original_w, original_h, &encoder.config)?;
        let pixels = normalize_resized_image(
            &image,
            grid.image_width(),
            grid.image_height(),
            &encoder.config.image_mean,
            &encoder.config.image_std,
        )?;
        println!(
            "Image resized to {}x{} ({} vision tokens)",
            grid.image_width(),
            grid.image_height(),
            grid.token_count()
        );
        let projection_dim = encoder.config.projection_dim;
        let mut scratch = VisionScratchpad::new(&encoder.config);
        println!("Encoding image...");
        let encoded_grid = encoder.encode_image(
            &pixels,
            grid.image_width(),
            grid.image_height(),
            &mut scratch,
        )?;
        if encoded_grid != grid {
            return Err(format!(
                "Vision grid mismatch: preprocess={grid:?}, encoder={encoded_grid:?}"
            ));
        }
        let projected_len = grid
            .token_count()
            .checked_mul(projection_dim)
            .ok_or("Projected vision length overflow")?;
        if scratch.projected.len() != projected_len {
            return Err(format!(
                "Projected vision length mismatch: expected {projected_len}, got {}",
                scratch.projected.len()
            ));
        }
        println!(
            "Vision tokens: {} (dim={})",
            grid.token_count(),
            projection_dim
        );
        (Some(grid), scratch.projected[..projected_len].to_vec())
    } else {
        (None, Vec::new())
    };
    let n_vis_tokens = image_grid.map(|g| g.token_count()).unwrap_or(0);
    let vis_embeddings = &vis_embeddings_vec[..];
    if image_grid.is_some() {
        println!(
            "First 5 vision embedding values: {:?}",
            &vis_embeddings[..5.min(vis_embeddings.len())]
        );
    }

    let llm = Qwen35Model::from_source(llm_source)
        .map_err(|error| format!("Failed to parse Qwen3.5 model: {error}"))?;
    println!("Qwen3.5 model loaded: {} layers, n_embd={}, n_head={}, n_ff={}, rope_freq_base={}, rope_sections={:?}, rope_dim_count={}", llm.config.n_layer, llm.config.n_embd, llm.config.n_head, llm.config.n_ff, llm.config.rope_freq_base, llm.config.rope_dimension_sections, llm.config.rope_dimension_count);

    let tokenizer = BPETokenizer::from_gguf_metadata(|k| llm_source.metadata(k).cloned())
        .map_err(|error| format!("Failed to initialize tokenizer: {error}"))?;
    let image_token_id = if image_grid.is_some() {
        Some(
            tokenizer
                .special_token_id("image_pad")
                .ok_or("Required token missing: <|image_pad|>")?,
        )
    } else {
        None
    };

    let mut content_tokens = Vec::new();
    if let Some(image_token_id) = image_token_id {
        content_tokens.push(
            tokenizer
                .special_token_id("vision_start")
                .ok_or("Required token missing: <|vision_start|>")?,
        );
        content_tokens.extend(std::iter::repeat(image_token_id).take(n_vis_tokens));
        content_tokens.push(
            tokenizer
                .special_token_id("vision_end")
                .ok_or("Required token missing: <|vision_end|>")?,
        );
    }
    content_tokens.extend(tokenizer.encode(
        prompt,
        EncodeOptions {
            add_special: false,
            parse_special: false,
        },
    ));

    let mut prompt_ids = Vec::new();
    append_qwen_message_tokens(&mut prompt_ids, &tokenizer, "user", &content_tokens)?;
    append_qwen_assistant_prefix(&mut prompt_ids, &tokenizer, false)?;
    let image_grids: Vec<crate::models::vision::VisionGrid> = image_grid.iter().copied().collect();
    let (prompt_positions, mut next_text_position) =
        build_qwen35_positions(&prompt_ids, image_token_id, &image_grids)?;
    let prompt_tokens: Vec<i32> = prompt_ids
        .iter()
        .copied()
        .map(|id| i32::try_from(id).map_err(|_| format!("Token ID {id} exceeds i32")))
        .collect::<Result<_, _>>()?;

    let projected_count = if vis_embeddings.is_empty() {
        0
    } else {
        let projection_dim = llm.config.n_embd;
        if vis_embeddings.len() % projection_dim != 0 {
            return Err("Projected vision embeddings are not row aligned".into());
        }
        vis_embeddings.len() / projection_dim
    };
    if projected_count != n_vis_tokens || prompt_positions.len() != prompt_tokens.len() {
        return Err(format!(
            "Vision/position count mismatch: placeholders={n_vis_tokens}, projected={projected_count}, positions={}, tokens={}",
            prompt_positions.len(),
            prompt_tokens.len()
        ));
    }
    let image_token_id = image_token_id
        .map(|id| i32::try_from(id).map_err(|_| format!("Token ID {id} exceeds i32")))
        .transpose()?;

    println!(
        "Prompt tokens: {} (including {} vision placeholders)",
        prompt_tokens.len(),
        n_vis_tokens
    );

    let max_seq = llm.config.n_ctx;
    let mut kv_cache = crate::core::scratchpad::KvCache::new_f32(
        llm.config.n_layer,
        max_seq,
        llm.config.n_embd_head() * llm.config.n_head_kv,
    );
    let mut llm_scratch =
        crate::models::qwen35::Qwen35Scratchpad::new(&llm.config, prompt_tokens.len().max(max_tokens));

    let prompt_embd = inject_vision_embeddings(
        &llm,
        &prompt_tokens,
        image_token_id,
        vis_embeddings,
        n_vis_tokens,
        llm.config.n_embd,
    );

    let n_prompt = prompt_tokens.len();
    let mut all_tokens = prompt_tokens.clone();

    let n_threads = if n_threads_arg > 0 { n_threads_arg } else { 8 };
    let pool = std::sync::Arc::new(ComputePool::new(n_threads));
    eprintln!("compute pool: {} threads", pool.n_threads());

    let mut generated = String::new();
    let mut decoder = tokenizer.streaming_decoder(false);
    println!("\n--- Generation ---");
    let t_gen_start = std::time::Instant::now();

    for step in 0..max_tokens {
        let tokens = if step == 0 {
            &prompt_tokens
        } else {
            &all_tokens[all_tokens.len() - 1..all_tokens.len() - 1 + 1]
        };
        let n_tok = tokens.len();

        if step == 0 {
            for t in 0..n_prompt {
                let embd_off = t * llm.config.n_embd;
                llm_scratch.x[embd_off..embd_off + llm.config.n_embd]
                    .copy_from_slice(&prompt_embd[embd_off..embd_off + llm.config.n_embd]);
            }
        } else {
            let tok = tokens[0] as usize;
            let tok_off = tok * llm.config.n_embd;
            for e in 0..llm.config.n_embd {
                if tok_off + e < llm.tok_embd.len() {
                    llm_scratch.x[e] = llm.tok_embd[tok_off + e];
                }
            }
        }

        let decode_position = [[
            next_text_position,
            next_text_position,
            next_text_position,
            0,
        ]];
        let positions = if step == 0 {
            &prompt_positions[..]
        } else {
            &decode_position[..]
        };
        let logits = llm.forward(n_tok, &mut kv_cache, &mut llm_scratch, &pool, positions)?;
        if step > 0 {
            next_text_position = next_text_position
                .checked_add(1)
                .ok_or("Qwen3.5 decode position overflow")?;
        }

        let next_token = if temperature <= 0.0 {
            logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i as i32)
                .unwrap_or(0)
        } else {
            sample_token(&logits, temperature)
        };

        if next_token >= 0
            && (tokenizer.eos_id() == Some(next_token as u32)
                || tokenizer.special_token_id("im_end") == Some(next_token as u32))
        {
            break;
        }

        let token_str = decoder.push(next_token as u32);
        generated.push_str(&token_str);
        print!("{}", token_str);
        std::io::Write::flush(&mut std::io::stdout()).ok();

        all_tokens.push(next_token);
    }

    let tail = decoder.finish();
    generated.push_str(&tail);
    if !tail.is_empty() {
        print!("{}", tail);
        std::io::Write::flush(&mut std::io::stdout()).ok();
    }

    let gen_ms = t_gen_start.elapsed().as_millis();
    let n_gen = all_tokens.len() - n_prompt;
    let tok_s = if gen_ms > 0 {
        n_gen as f64 / gen_ms as f64 * 1000.0
    } else {
        0.0
    };
    println!("\n--- End ---");
    eprintln!(
        "[{} gen tokens in {}ms | {:.1} tok/s]",
        n_gen, gen_ms, tok_s
    );
    Ok(())
}
