use crate::app::cli::KvFormat;
use crate::app::{get_f32_tensor, raw_parts, slice_from_mut, slice_from_ref, LayerWeights};
use crate::core::loader::model_config_from_source;
use crate::core::scratchpad::{ExecutionScratchpad, KvCache};
use crate::core::tensor::{GGMLType, TensorSource};
use crate::core::thread_pool::ComputePool;
use crate::core::tokenizer::BPETokenizer;
use crate::ops::embedding_lookup;
use crate::ops::kernel::Kernel;
use crate::ops::{
    dot_f16_f32, dot_f32, f32_slice_to_f16, quantize_q8_0_into, rms_norm, rms_norm_inplace,
    rope_neox, softmax_inplace,
};
use std::sync::Arc;

#[cfg(feature = "parity-trace")]
macro_rules! trace_checkpoint {
    ($name:expr, $layer:expr, $step:expr, $shape:expr, $values:expr) => {
        crate::parity_trace::report(
            crate::parity_trace::checkpoint_at($name, $layer, Some($step), $shape, $values)
                .map(|_| ()),
        );
    };
}

#[cfg(feature = "parity-trace")]
macro_rules! trace_tokens {
    ($name:expr, $values:expr) => {
        crate::parity_trace::report(crate::parity_trace::token_ids($name, $values));
    };
}

pub fn run_dump_logits(
    source: &dyn TensorSource,
    prompt: &str,
    max_tokens: usize,
    n_threads_arg: usize,
    kv_format: KvFormat,
) -> Result<(), String> {
    let config = model_config_from_source(source)
        .map_err(|error| format!("Failed to parse model config: {error}"))?;

    let mut bin_out = std::fs::File::create("/tmp/rust_logits.bin")
        .map_err(|error| format!("Failed to create /tmp/rust_logits.bin: {error}"))?;

    let arch = source
        .metadata("general.architecture")
        .and_then(|v| v.to_string_val())
        .unwrap_or_default();
    let is_qwen3 = arch == "qwen3";

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
    let embd_info = source
        .tensor_info("token_embd.weight")
        .expect("no token_embd.weight");
    if !matches!(
        embd_info.ggml_type,
        GGMLType::F16 | GGMLType::Q8_0 | GGMLType::Q4_0 | GGMLType::Q6K
    ) {
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
                let info = source
                    .tensor_info(&format!("blk.{}.attn_q.weight", l))
                    .unwrap();
                crate::ops::kernel::QuantizedTensor::from_bytes(
                    source
                        .tensor_slice(&format!("blk.{}.attn_q.weight", l))
                        .unwrap(),
                    info.ggml_type,
                    n_embd,
                    n_embd_q,
                )
                .into_kernel()
            },
            wk: {
                let info = source
                    .tensor_info(&format!("blk.{}.attn_k.weight", l))
                    .unwrap();
                crate::ops::kernel::QuantizedTensor::from_bytes(
                    source
                        .tensor_slice(&format!("blk.{}.attn_k.weight", l))
                        .unwrap(),
                    info.ggml_type,
                    n_embd,
                    n_embd_gqa,
                )
                .into_kernel()
            },
            wv: {
                let info = source
                    .tensor_info(&format!("blk.{}.attn_v.weight", l))
                    .unwrap();
                crate::ops::kernel::QuantizedTensor::from_bytes(
                    source
                        .tensor_slice(&format!("blk.{}.attn_v.weight", l))
                        .unwrap(),
                    info.ggml_type,
                    n_embd,
                    n_embd_gqa,
                )
                .into_kernel()
            },
            wo: {
                let info = source
                    .tensor_info(&format!("blk.{}.attn_output.weight", l))
                    .unwrap();
                crate::ops::kernel::QuantizedTensor::from_bytes(
                    source
                        .tensor_slice(&format!("blk.{}.attn_output.weight", l))
                        .unwrap(),
                    info.ggml_type,
                    n_embd_q,
                    n_embd,
                )
                .into_kernel()
            },
            w_gate: {
                let info = source
                    .tensor_info(&format!("blk.{}.ffn_gate.weight", l))
                    .unwrap();
                crate::ops::kernel::QuantizedTensor::from_bytes(
                    source
                        .tensor_slice(&format!("blk.{}.ffn_gate.weight", l))
                        .unwrap(),
                    info.ggml_type,
                    n_embd,
                    n_ff,
                )
                .into_kernel()
            },
            w_up: {
                let info = source
                    .tensor_info(&format!("blk.{}.ffn_up.weight", l))
                    .unwrap();
                crate::ops::kernel::QuantizedTensor::from_bytes(
                    source
                        .tensor_slice(&format!("blk.{}.ffn_up.weight", l))
                        .unwrap(),
                    info.ggml_type,
                    n_embd,
                    n_ff,
                )
                .into_kernel()
            },
            w_down: {
                let info = source
                    .tensor_info(&format!("blk.{}.ffn_down.weight", l))
                    .unwrap();
                crate::ops::kernel::QuantizedTensor::from_bytes(
                    source
                        .tensor_slice(&format!("blk.{}.ffn_down.weight", l))
                        .unwrap(),
                    info.ggml_type,
                    n_ff,
                    n_embd,
                )
                .into_kernel()
            },
        })
        .collect();

    eprintln!(
        "Model: {} | n_embd={} n_layer={} n_head={} n_head_kv={} n_ff={}",
        arch, n_embd, n_layer, n_head, n_head_kv, n_ff
    );

    let prompt_tokens = tokenizer.encode(
        prompt,
        crate::core::tokenizer::EncodeOptions {
            add_special: true,
            parse_special: true,
        },
    );
    eprintln!(
        "Tokenized to {} tokens: {:?}",
        prompt_tokens.len(),
        prompt_tokens
    );
    #[cfg(feature = "parity-trace")]
    trace_tokens!("prompt_ids", &prompt_tokens);

    let vocab = tokenizer.vocab_size();
    let n_threads = if n_threads_arg > 0 { n_threads_arg } else { 1 };

    {
        use std::io::Write as IoWrite;
        let header: [i32; 3] = [vocab as i32, prompt_tokens.len() as i32, max_tokens as i32];
        bin_out
            .write_all(unsafe { std::slice::from_raw_parts(header.as_ptr() as *const u8, 12) })
            .unwrap();
        let pt: Vec<i32> = prompt_tokens.iter().map(|&t| t as i32).collect();
        bin_out
            .write_all(unsafe {
                std::slice::from_raw_parts(pt.as_ptr() as *const u8, pt.len() * 4)
            })
            .unwrap();
    }

    let kv_cache = match kv_format {
        KvFormat::F16 => KvCache::new_f16(n_layer, max_ctx, n_embd_gqa),
        KvFormat::F32 => KvCache::new_f32(n_layer, max_ctx, n_embd_gqa),
    };

    let mut scratch = ExecutionScratchpad::new(
        n_embd, n_embd_q, n_embd_gqa, n_ff, vocab, n_threads, max_ctx,
    );

    let input_tokens = prompt_tokens.clone();
    let pool = Arc::new(ComputePool::new(n_threads));

    let group_size = n_head / n_head_kv;
    let kq_scale = 1.0f32 / (n_embd_head_k as f32).sqrt();

    let mut generated_tokens: Vec<u32> = Vec::new();
    let mut all_tokens: Vec<u32> = input_tokens.clone();

    for step in 0..(input_tokens.len() + max_tokens) {
        let token_id = if step < input_tokens.len() {
            input_tokens[step]
        } else {
            *generated_tokens.last().unwrap_or(&0)
        };

        let pos = step;
        println!(
            "[STEP] step={} token_id={} input_tokens.len()={}",
            step,
            token_id,
            input_tokens.len()
        );

        #[cfg(feature = "parity-trace")]
        trace_tokens!("input_token", &[token_id]);

        embedding_lookup(embd_weight, token_id, n_embd, embd_type, &mut scratch.x);
        #[cfg(feature = "parity-trace")]
        trace_checkpoint!("embedding", None, step, &[n_embd], &scratch.x);

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

            let x = slice_from_mut!(x_ptr, n_embd);
            let normed = slice_from_mut!(normed_ptr, n_embd);
            let q8_buf = slice_from_mut!(q8_buf_ptr, n_embd_q.max(n_ff));
            let scale_buf = slice_from_mut!(scale_buf_ptr, n_embd_q.max(n_ff) / 32);

            rms_norm(x, &lw.attn_norm, normed, eps);
            #[cfg(feature = "parity-trace")]
            trace_checkpoint!("attn_norm", Some(layer), step, &[n_embd], normed);
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

                lw.wq
                    .forward_prepared(input, q8, sc, None, q, n_embd, n_embd_q, ith, nth);
                lw.wk
                    .forward_prepared(input, q8, sc, None, k_new, n_embd, n_embd_gqa, ith, nth);
                lw.wv
                    .forward_prepared(input, q8, sc, None, v_new, n_embd, n_embd_gqa, ith, nth);
            });

            #[cfg(feature = "parity-trace")]
            {
                trace_checkpoint!("q_proj", Some(layer), step, &[n_embd_q], &scratch.q);
                trace_checkpoint!("k_proj", Some(layer), step, &[n_embd_gqa], &scratch.k_new);
                trace_checkpoint!("v_proj", Some(layer), step, &[n_embd_gqa], &scratch.v_new);
            }

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

                #[cfg(feature = "parity-trace")]
                {
                    trace_checkpoint!("q_norm", Some(layer), step, &[n_embd_q], q);
                    trace_checkpoint!("k_norm", Some(layer), step, &[n_embd_gqa], k_new);
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

                #[cfg(feature = "parity-trace")]
                {
                    trace_checkpoint!("q_rope", Some(layer), step, &[n_embd_q], q);
                    trace_checkpoint!("k_rope", Some(layer), step, &[n_embd_gqa], k_new);
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

            #[cfg(feature = "parity-trace")]
            let n_cached = pos + 1;
            #[cfg(feature = "parity-trace")]
            let mut trace_scores = vec![0.0f32; n_head * n_cached];
            #[cfg(feature = "parity-trace")]
            let mut trace_probs = vec![0.0f32; n_head * n_cached];
            #[cfg(feature = "parity-trace")]
            let trace_scores_ptr = trace_scores.as_mut_ptr();
            #[cfg(feature = "parity-trace")]
            let trace_probs_ptr = trace_probs.as_mut_ptr();

            pool.compute(move |ith: usize, nth: usize| {
                let q = slice_from_ref!(q_ptr, n_embd_q);
                let attn_out = slice_from_mut!(attn_out_ptr, n_embd_q);
                let scores = slice_from_mut!(scores_ptr, n_threads * score_stride);
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
                        let s_off = ith * score_stride;
                        for t in 0..n_cached {
                            scores[s_off + t] = dot_f16_f32(
                                &q[q_off..q_off + n_embd_head_k],
                                &k_cache[kb + t * n_embd_gqa + kv_h * n_embd_head_v
                                    ..kb + t * n_embd_gqa + kv_h * n_embd_head_v + n_embd_head_k],
                                n_embd_head_k,
                            ) * kq_scale;
                        }
                        #[cfg(feature = "parity-trace")]
                        slice_from_mut!(trace_scores_ptr, n_head * n_cached)
                            [h * n_cached..(h + 1) * n_cached]
                            .copy_from_slice(&scores[s_off..s_off + n_cached]);
                        softmax_inplace(&mut scores[s_off..s_off + n_cached]);
                        #[cfg(feature = "parity-trace")]
                        slice_from_mut!(trace_probs_ptr, n_head * n_cached)
                            [h * n_cached..(h + 1) * n_cached]
                            .copy_from_slice(&scores[s_off..s_off + n_cached]);
                        for d in 0..n_embd_head_v {
                            let mut val = 0.0f32;
                            for t in 0..n_cached {
                                val += scores[s_off + t]
                                    * crate::ops::f16_to_f32(
                                        v_cache[kb + t * n_embd_gqa + kv_h * n_embd_head_v + d],
                                    );
                            }
                            attn_out[h * n_embd_head_v + d] = val;
                        }
                    }
                } else {
                    let k_cache = slice_from_ref!(k_cache_f32_ptr, kv_cache_size);
                    let v_cache = slice_from_ref!(v_cache_f32_ptr, kv_cache_size);
                    for h in h_start..h_end {
                        let kv_h = h / group_size;
                        let q_off = h * n_embd_head_k;
                        let n_cached = pos + 1;
                        let s_off = ith * score_stride;
                        for t in 0..n_cached {
                            scores[s_off + t] = dot_f32(
                                &q[q_off..q_off + n_embd_head_k],
                                &k_cache[kb + t * n_embd_gqa + kv_h * n_embd_head_v
                                    ..kb + t * n_embd_gqa + kv_h * n_embd_head_v + n_embd_head_k],
                                n_embd_head_k,
                            ) * kq_scale;
                        }
                        #[cfg(feature = "parity-trace")]
                        slice_from_mut!(trace_scores_ptr, n_head * n_cached)
                            [h * n_cached..(h + 1) * n_cached]
                            .copy_from_slice(&scores[s_off..s_off + n_cached]);
                        softmax_inplace(&mut scores[s_off..s_off + n_cached]);
                        #[cfg(feature = "parity-trace")]
                        slice_from_mut!(trace_probs_ptr, n_head * n_cached)
                            [h * n_cached..(h + 1) * n_cached]
                            .copy_from_slice(&scores[s_off..s_off + n_cached]);
                        for d in 0..n_embd_head_v {
                            #[cfg(feature = "parity-trace")]
                            let mut val = 0.0f64;
                            #[cfg(not(feature = "parity-trace"))]
                            let mut val = 0.0f32;
                            for t in 0..n_cached {
                                #[cfg(feature = "parity-trace")]
                                {
                                    val += f64::from(
                                        scores[s_off + t]
                                            * v_cache
                                                [kb + t * n_embd_gqa + kv_h * n_embd_head_v + d],
                                    );
                                }
                                #[cfg(not(feature = "parity-trace"))]
                                {
                                    val += scores[s_off + t]
                                        * v_cache[kb + t * n_embd_gqa + kv_h * n_embd_head_v + d];
                                }
                            }
                            #[cfg(feature = "parity-trace")]
                            let val = val as f32;
                            attn_out[h * n_embd_head_v + d] = val;
                        }
                    }
                }
            });

            #[cfg(feature = "parity-trace")]
            {
                trace_checkpoint!(
                    "attn_scores",
                    Some(layer),
                    step,
                    &[n_head, n_cached],
                    &trace_scores
                );
                trace_checkpoint!(
                    "attn_probs",
                    Some(layer),
                    step,
                    &[n_head, n_cached],
                    &trace_probs
                );
                trace_checkpoint!(
                    "attn_values",
                    Some(layer),
                    step,
                    &[n_embd_q],
                    &scratch.attn_out
                );
            }

            let attn_out = slice_from_mut!(attn_out_ptr, n_embd_q);
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
                lw.wo
                    .forward_prepared(input, q8, sc, None, attn_proj, n_embd_q, n_embd, ith, nth);
            });

            #[cfg(feature = "parity-trace")]
            trace_checkpoint!(
                "attn_proj",
                Some(layer),
                step,
                &[n_embd],
                &scratch.attn_proj
            );

            let attn_proj = slice_from_mut!(attn_proj_ptr, n_embd);
            let x = slice_from_mut!(x_ptr, n_embd);
            let normed = slice_from_mut!(normed_ptr, n_embd);
            for i in 0..n_embd {
                x[i] += attn_proj[i];
            }

            #[cfg(feature = "parity-trace")]
            trace_checkpoint!("post_attn_residual", Some(layer), step, &[n_embd], x);

            rms_norm(x, &lw.ffn_norm, normed, eps);
            #[cfg(feature = "parity-trace")]
            trace_checkpoint!("ffn_norm", Some(layer), step, &[n_embd], normed);
            quantize_q8_0_into(
                normed,
                n_embd,
                &mut q8_buf[..n_embd],
                &mut scale_buf[..n_embd / 32],
            );
            let q8 = q8_buf[..n_embd].as_ptr();
            let sc = scale_buf[..n_embd / 32].as_ptr();

            #[cfg(feature = "parity-trace")]
            let mut trace_ffn_gate = vec![0.0f32; n_ff];
            #[cfg(feature = "parity-trace")]
            let mut trace_ffn_up = vec![0.0f32; n_ff];
            #[cfg(feature = "parity-trace")]
            let trace_ffn_gate_ptr = trace_ffn_gate.as_mut_ptr();
            #[cfg(feature = "parity-trace")]
            let trace_ffn_up_ptr = trace_ffn_up.as_mut_ptr();

            pool.compute(move |ith: usize, nth: usize| {
                let input = raw_parts!(normed_ptr, n_embd);
                let q8 = raw_parts!(q8, n_embd);
                let sc = raw_parts!(sc, n_embd / 32);
                let gate_buf = slice_from_mut!(gate_buf_ptr, n_ff);
                let up_buf = slice_from_mut!(up_buf_ptr, n_ff);
                lw.w_gate
                    .forward_prepared(input, q8, sc, None, up_buf, n_embd, n_ff, ith, nth);
                lw.w_up
                    .forward_prepared(input, q8, sc, None, gate_buf, n_embd, n_ff, ith, nth);

                let rows_per = n_ff / nth;
                let r_start = ith * rows_per;
                let r_end = if ith == nth - 1 {
                    n_ff
                } else {
                    r_start + rows_per
                };
                #[cfg(feature = "parity-trace")]
                {
                    slice_from_mut!(trace_ffn_gate_ptr, n_ff)[r_start..r_end]
                        .copy_from_slice(&up_buf[r_start..r_end]);
                    slice_from_mut!(trace_ffn_up_ptr, n_ff)[r_start..r_end]
                        .copy_from_slice(&gate_buf[r_start..r_end]);
                }
                crate::ops::silu_mul_approx_inplace(
                    &up_buf[r_start..r_end],
                    &mut gate_buf[r_start..r_end],
                );
            });

            #[cfg(feature = "parity-trace")]
            {
                trace_checkpoint!("ffn_gate", Some(layer), step, &[n_ff], &trace_ffn_gate);
                trace_checkpoint!("ffn_up", Some(layer), step, &[n_ff], &trace_ffn_up);
                trace_checkpoint!(
                    "ffn_silu_gate",
                    Some(layer),
                    step,
                    &[n_ff],
                    &scratch.gate_buf
                );
            }

            let gate_buf = slice_from_mut!(gate_buf_ptr, n_ff);
            quantize_q8_0_into(
                gate_buf,
                n_ff,
                &mut q8_buf[..n_ff],
                &mut scale_buf[..n_ff / 32],
            );

            let q8 = q8_buf[..n_ff].as_ptr();
            let sc = scale_buf[..n_ff / 32].as_ptr();
            pool.compute(move |ith: usize, nth: usize| {
                let input = raw_parts!(gate_buf_ptr, n_ff);
                let q8 = raw_parts!(q8, n_ff);
                let sc = raw_parts!(sc, n_ff / 32);
                let down_buf = slice_from_mut!(down_buf_ptr, n_embd);
                lw.w_down
                    .forward_prepared(input, q8, sc, None, down_buf, n_ff, n_embd, ith, nth);
            });

            #[cfg(feature = "parity-trace")]
            trace_checkpoint!("ffn_down", Some(layer), step, &[n_embd], &scratch.down_buf);

            let down_buf = slice_from_mut!(down_buf_ptr, n_embd);
            let x = slice_from_mut!(x_ptr, n_embd);
            for i in 0..n_embd {
                x[i] += down_buf[i];
            }
            #[cfg(feature = "parity-trace")]
            trace_checkpoint!("post_ffn_residual", Some(layer), step, &[n_embd], x);
        }

        {
            let x = &mut scratch.x;
            let normed = &mut scratch.normed;
            let logits_ptr = scratch.logits.as_mut_ptr();
            let q8_buf = &mut scratch.q8_buf;
            let scale_buf = &mut scratch.scale_buf;

            rms_norm(x, &output_norm, normed, eps);
            #[cfg(feature = "parity-trace")]
            trace_checkpoint!("result_norm", None, step, &[n_embd], normed);
            quantize_q8_0_into(
                normed,
                n_embd,
                &mut q8_buf[..n_embd],
                &mut scale_buf[..n_embd / 32],
            );
            let output_pw = crate::ops::kernel::QuantizedTensor::from_bytes(
                output_weight,
                output_type,
                n_embd,
                vocab,
            );
            let q8_ptr = q8_buf.as_ptr();
            let sc = scale_buf[..n_embd / 32].as_ptr();
            let input = normed.as_ptr();
            pool.compute(move |ith: usize, nth: usize| {
                let input = raw_parts!(input, n_embd);
                let q8 = raw_parts!(q8_ptr, n_embd);
                let sc_ptr = raw_parts!(sc, n_embd / 32);
                let logits = slice_from_mut!(logits_ptr, vocab);
                output_pw.forward_prepared(input, q8, sc_ptr, None, logits, n_embd, vocab, ith, nth);
            });
        }

        #[cfg(feature = "parity-trace")]
        trace_checkpoint!("result_output", None, step, &[vocab], &scratch.logits);

        if step < input_tokens.len() - 1 {
            continue;
        }

        let logits = &scratch.logits;

        let mut best_idx = 0usize;
        let mut best_val = logits[0];
        for (i, &v) in logits.iter().enumerate().skip(1) {
            if v > best_val {
                best_val = v;
                best_idx = i;
            }
        }

        println!("=== Step {} token={} ===", step, token_id);
        println!("  argmax={} logit={:.8}", best_idx, best_val);

        let mut indexed: Vec<(usize, f32)> =
            logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        for k in 0..5 {
            println!("  [{}] token={} logit={:.8}", k, indexed[k].0, indexed[k].1);
        }

        let sum: f32 = logits.iter().sum();
        let sq_sum: f32 = logits.iter().map(|&v| v * v).sum();
        let mn = logits.iter().cloned().fold(f32::INFINITY, f32::min);
        let mx = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mean = sum / vocab as f32;
        let std = (sq_sum / vocab as f32 - mean * mean).sqrt();
        println!(
            "  stats: sum={:.6} mean={:.6} std={:.6} min={:.6} max={:.6}",
            sum, mean, std, mn, mx
        );

        {
            use std::io::Write as IoWrite;
            bin_out
                .write_all(unsafe {
                    std::slice::from_raw_parts(logits.as_ptr() as *const u8, vocab * 4)
                })
                .unwrap();
        }

        let chosen = best_idx as u32;
        if generated_tokens.len() >= max_tokens {
            break;
        }
        generated_tokens.push(chosen);
        all_tokens.push(chosen);
    }
    #[cfg(feature = "parity-trace")]
    trace_tokens!("generated_ids", &generated_tokens);
    Ok(())
}
