//! # LLaMA Text Inference
//!
//! LLaMA-family text generation, aligning with llama.cpp's forward pass and
//! sample/decode path. Standard LLaMA has no Q/K per-head RMSNorm.

use super::weights::{get_f32_tensor, load_layers, LlamaLayerWeights};
use crate::app::cli::{inference_step_budget, resolve_thread_count, KvFormat};
use crate::core::loader::model_config_from_source;
use crate::core::scratchpad::{ExecutionScratchpad, KvCache};
use crate::core::tensor::TensorSource;
use crate::core::thread_pool::ComputePool;
use crate::core::tokenizer::{BPETokenizer, EncodeOptions};
use crate::ops::embedding_lookup;
use crate::ops::kernel::{Kernel, QuantizedTensor, Weight};
use crate::ops::{
    attention_value_f32, dot_f16_f32, dot_f32, f32_slice_to_f16, quantize_q8_0_into, rms_norm,
    rope_norm, silu_mul_approx_inplace, softmax_inplace, sum_sq_f32, vec_mad_f16_f32,
    vec_scale_f32,
};

use std::io::{self, Write};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Read sampling defaults from GGUF metadata (matching llama.cpp's
/// behaviour). MiniCPM5 ships `general.sampling.{top_k,top_p,temp}` keys
/// that override the C++ defaults (40, 0.95, 1.0).
fn sample_defaults(source: &dyn TensorSource) -> (usize, f32) {
    let top_k = source
        .metadata("general.sampling.top_k")
        .and_then(|v| v.to_u64())
        .map(|v| v as usize)
        .unwrap_or(40);
    let top_p = source
        .metadata("general.sampling.top_p")
        .and_then(|v| v.to_f64())
        .map(|v| v as f32)
        .unwrap_or(0.95);
    (top_k, top_p)
}

/// Print first 8 floats of `buf` to stderr, tagged by step/layer/label.
/// Triggered by the `RUST_LLAMA_DEBUG_TENSORS` env var (set to a layer
/// count, e.g. `RUST_LLAMA_DEBUG_TENSORS=1`).
fn dbg_tensor(step: usize, label: &'static str, il: usize, buf: &[f32]) {
    static ON: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    let limit = ON.get_or_init(|| {
        std::env::var("RUST_LLAMA_DEBUG_TENSORS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
    });
    if *limit == 0 || (il as u32) >= *limit {
        return;
    }
    let n = buf.len().min(8);
    let mut line = format!("RUST_TENSOR step={} il={} {} first8=", step, il, label);
    for v in &buf[..n] {
        line.push_str(&format!("{:.5} ", v));
    }
    line.push('\n');
    let _ = io::stderr().write_all(line.as_bytes());
    let _ = io::stderr().flush();
}

/// Print a single scalar (scale/mean) tagged by step/layer/label.
fn dbg_scalar(step: usize, label: &'static str, il: usize, value: f32) {
    static ON: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    let limit = ON.get_or_init(|| {
        std::env::var("RUST_LLAMA_DEBUG_TENSORS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
    });
    if *limit == 0 || (il as u32) >= *limit {
        return;
    }
    let line = format!(
        "RUST_SCALAR step={} il={} {}={:.10}\n",
        step, il, label, value
    );
    let _ = io::stderr().write_all(line.as_bytes());
    let _ = io::stderr().flush();
}

fn dbg_scalar_full(step: usize, label: &'static str, il: usize, value: f64) {
    static ON: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    let limit = ON.get_or_init(|| {
        std::env::var("RUST_LLAMA_DEBUG_TENSORS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
    });
    if *limit == 0 || (il as u32) >= *limit {
        return;
    }
    let line = format!(
        "RUST_SCALAR step={} il={} {}={:.10}\n",
        step, il, label, value
    );
    let _ = io::stderr().write_all(line.as_bytes());
    let _ = io::stderr().flush();
}

/// Dump full values of `buf` to a file specified by RUST_LLAMA_DEBUG_OUTFILE.
fn dbg_full(step: usize, label: &'static str, il: usize, buf: &[f32], n: usize) {
    static ON: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    let limit = ON.get_or_init(|| {
        std::env::var("RUST_LLAMA_DEBUG_TENSORS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
    });
    if *limit == 0 || (il as u32) >= *limit {
        return;
    }
    let path = match std::env::var("RUST_LLAMA_DEBUG_OUTFILE") {
        Ok(p) => p,
        Err(_) => return,
    };
    let mut line = format!("[step={} il={} {}]", step, il, label);
    for i in 0..n {
        line.push_str(&format!(" {:.5}", buf[i]));
    }
    line.push('\n');
    let _ = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .and_then(|mut f| std::io::Write::write_all(&mut f, line.as_bytes()));
}

pub fn run_inference(
    source: &dyn TensorSource,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    n_threads_arg: usize,
    bench: bool,
    profile: bool,
    kv_format: KvFormat,
) -> Result<(), String> {
    let input_tokens = {
        let tokenizer = BPETokenizer::from_gguf_metadata(|k| source.metadata(k).cloned())
            .map_err(|error| format!("Failed to initialize tokenizer: {error}"))?;

        // Build the prompt using MiniCPM5's chat template (rendered via
        // the Jinja template that ships in the GGUF; auto-detected by
        // llama.cpp with `enable_thinking=true`). The rendered format for
        // a single user message is:
        //
        //     <s><|im_start|>user\n{prompt}<|im_end|>\n
        //     <|im_start|>assistant\n<think>\n
        //
        // (BOS is added because the chat template explicitly emits `<s>`.)
        let im_start_str = "<|im_start|>";
        let im_end_str = "<|im_end|>";
        // Build the prompt text with the literal special tokens inline.
        let prompt_text =
            format!("{im_start_str}user\n{prompt}{im_end_str}\n{im_start_str}assistant\n<think>\n");
        // encode() with add_special=true prepends `<s>` and parse_special=true
        // emits the literal `<|im_start|>`/`<|im_end|>` strings as single token ids.
        let mut body = tokenizer.encode(
            &prompt_text,
            EncodeOptions {
                add_special: false,
                parse_special: true,
            },
        );
        // The chat template starts with `{{- bos_token }}`, but MiniCPM5
        // has `tokenizer.ggml.add_bos_token=false`, so encode() does not
        // emit BOS automatically. Prepend BOS manually to match llama.cpp.
        if let Some(bos) = tokenizer.bos_id() {
            body.insert(0, bos);
        }
        eprintln!("[RUST_TOKENS] n={} ids={:?}", body.len(), body);
        body
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
    crate::ops::embedding::expect_supported_embedding("token_embd.weight", embd_info.ggml_type);
    let embd_weight = source.tensor_slice("token_embd.weight").expect("no embd");
    let output_weight = source.tensor_slice("output.weight").unwrap_or(embd_weight);
    let embd_type = embd_info.ggml_type;
    let output_type = source
        .tensor_info("output.weight")
        .unwrap_or(embd_info)
        .ggml_type;

    let layers: Vec<LlamaLayerWeights> =
        load_layers(source, n_layer, n_embd, n_embd_q, n_embd_gqa, n_ff);

    // DEBUG: dump first N bytes of L23's w_gate, w_up, w_down for comparison with llama.cpp.
    if std::env::var("RUST_LLAMA_DEBUG_L23_WEIGHTS").is_ok() {
        let l = n_layer - 1; // L23 = last layer
        let layer = &layers[l];
        let dump_n = 256usize; // first 256 bytes (= ~7.5 Q8_0 blocks)
        for (name, qw) in [
            ("w_gate", &layer.w_gate),
            ("w_up", &layer.w_up),
            ("w_down", &layer.w_down),
        ] {
            // Get the raw bytes from the QuantizedTensor
            // The bytes are stored in the tensor slice from the source
            // We need to re-fetch them since QuantizedTensor doesn't expose raw bytes
            let source_bytes = source
                .tensor_slice(&format!("blk.{}.ffn_{}.weight", l, &name[2..]))
                .unwrap();
            eprintln!("[RUST_W_L23] {} first {} bytes:", name, dump_n);
            for chunk in source_bytes[..dump_n].chunks(16) {
                let hex: Vec<String> = chunk.iter().map(|b| format!("{:02x}", b)).collect();
                eprintln!("  {}", hex.join(" "));
            }
        }
    }

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

    let available_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let n_threads = resolve_thread_count(n_threads_arg, available_threads);

    let mut scratch = ExecutionScratchpad::new(
        n_embd, n_embd_q, n_embd_gqa, n_ff, vocab, n_threads, max_ctx,
    );
    let pool = Arc::new(ComputePool::new(n_threads));
    eprintln!("compute pool: {} threads", pool.n_threads());
    println!("Prompt: {} tokens", input_tokens.len());

    let eos_id = tokenizer.eos_id();
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
        dbg_tensor(step, "embed_out", 0, &scratch.x);

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
            let x = unsafe { std::slice::from_raw_parts_mut(x_ptr, n_embd) };
            let normed = unsafe { std::slice::from_raw_parts_mut(normed_ptr, n_embd) };
            let q8_buf = unsafe { std::slice::from_raw_parts_mut(q8_buf_ptr, max_n_in) };
            let scale_buf = unsafe { std::slice::from_raw_parts_mut(scale_buf_ptr, max_n_in / 32) };
            let q8k_buf = unsafe { std::slice::from_raw_parts_mut(q8k_buf_ptr, max_n_in / 256) };

            let t0 = Instant::now();
            rms_norm(x, &lw.attn_norm, normed, eps);
            dbg_tensor(step, "attn_norm", layer, normed);
            quantize_q8_0_into(
                normed,
                n_embd,
                &mut q8_buf[..n_embd],
                &mut scale_buf[..n_embd / 32],
            );
            let q8 = q8_buf[..n_embd].as_ptr();
            let sc = scale_buf[..n_embd / 32].as_ptr();
            crate::ops::quantize_row_q8_k_into(normed, &mut q8k_buf[..n_embd / 256]);
            let q8k = q8k_buf[..n_embd / 256].as_ptr();

            pool.compute(move |ith: usize, nth: usize| {
                let input = unsafe { std::slice::from_raw_parts(normed_ptr, n_embd) };
                let q8 = unsafe { std::slice::from_raw_parts(q8, n_embd) };
                let sc = unsafe { std::slice::from_raw_parts(sc, n_embd / 32) };
                let q8k = unsafe { std::slice::from_raw_parts(q8k, n_embd / 256) };
                let q = unsafe { std::slice::from_raw_parts_mut(q_ptr, n_embd_q) };
                let k_new = unsafe { std::slice::from_raw_parts_mut(k_ptr, n_embd_gqa) };
                let v_new = unsafe { std::slice::from_raw_parts_mut(v_ptr, n_embd_gqa) };

                lw.wq.kernel.forward_prepared(
                    input,
                    q8,
                    sc,
                    Some(q8k),
                    q,
                    n_embd,
                    n_embd_q,
                    ith,
                    nth,
                );
                lw.wk.kernel.forward_prepared(
                    input,
                    q8,
                    sc,
                    Some(q8k),
                    k_new,
                    n_embd,
                    n_embd_gqa,
                    ith,
                    nth,
                );
                lw.wv.kernel.forward_prepared(
                    input,
                    q8,
                    sc,
                    Some(q8k),
                    v_new,
                    n_embd,
                    n_embd_gqa,
                    ith,
                    nth,
                );
            });

            {
                let q = unsafe { std::slice::from_raw_parts_mut(q_ptr, n_embd_q) };
                let k_new = unsafe { std::slice::from_raw_parts_mut(k_ptr, n_embd_gqa) };
                let v_new = unsafe { std::slice::from_raw_parts_mut(v_ptr, n_embd_gqa) };

                // LLaMA does not have QK norm.
                // The `llama` GGUF arch uses interleaved ("normal"-style)
                // RoPE — the converter permutes HF rotate_half weights into
                // adjacent-pair layout (MiniCPM5 ships this arch too).
                for h in 0..n_head {
                    rope_norm(
                        &mut q[h * n_embd_head_k..(h + 1) * n_embd_head_k],
                        pos,
                        n_embd_head_k,
                        freq_base,
                    );
                }
                for h in 0..n_head_kv {
                    rope_norm(
                        &mut k_new[h * n_embd_head_k..(h + 1) * n_embd_head_k],
                        pos,
                        n_embd_head_k,
                        freq_base,
                    );
                }
                dbg_tensor(step, "Qcur", layer, q);
                dbg_tensor(step, "Kcur", layer, k_new);
                dbg_tensor(step, "Vcur", layer, v_new);

                let kb = layer * max_ctx * n_embd_gqa;

                if kv_format == KvFormat::F16 {
                    let k_cache =
                        unsafe { std::slice::from_raw_parts_mut(k_cache_f16_ptr, kv_cache_size) };
                    let v_cache =
                        unsafe { std::slice::from_raw_parts_mut(v_cache_f16_ptr, kv_cache_size) };
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
                    let k_cache =
                        unsafe { std::slice::from_raw_parts_mut(k_cache_f32_ptr, kv_cache_size) };
                    let v_cache =
                        unsafe { std::slice::from_raw_parts_mut(v_cache_f32_ptr, kv_cache_size) };
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
                let q = unsafe { std::slice::from_raw_parts(q_ptr, n_embd_q) };
                let attn_out = unsafe { std::slice::from_raw_parts_mut(attn_out_ptr, n_embd_q) };
                let h_start = ith * n_head / nth;
                let h_end = (ith + 1) * n_head / nth;

                let kb = layer * max_ctx * n_embd_gqa;

                if kv_format == KvFormat::F16 {
                    let k_cache =
                        unsafe { std::slice::from_raw_parts(k_cache_f16_ptr, kv_cache_size) };
                    let v_cache =
                        unsafe { std::slice::from_raw_parts(v_cache_f16_ptr, kv_cache_size) };
                    for h in h_start..h_end {
                        let kv_h = h / group_size;
                        let q_off = h * n_embd_head_k;
                        let n_cached = pos + 1;
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
                    let k_cache =
                        unsafe { std::slice::from_raw_parts(k_cache_f32_ptr, kv_cache_size) };
                    let v_cache =
                        unsafe { std::slice::from_raw_parts(v_cache_f32_ptr, kv_cache_size) };
                    let scores = unsafe {
                        std::slice::from_raw_parts_mut(scores_ptr, n_threads * score_stride)
                    };
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
                        // The values scratch is sized to the next multiple of
                        // 256 above max_ctx (n_padded_max). Heap-allocated so
                        // long contexts don't overflow.
                        let mut values = vec![0.0f32; n_padded];
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

            let attn_out = unsafe { std::slice::from_raw_parts_mut(attn_out_ptr, n_embd_q) };
            dbg_tensor(step, "attn_out", layer, attn_out);
            let q8_buf = unsafe { std::slice::from_raw_parts_mut(q8_buf_ptr, max_n_in) };
            let scale_buf = unsafe { std::slice::from_raw_parts_mut(scale_buf_ptr, max_n_in / 32) };
            let q8k_buf = unsafe { std::slice::from_raw_parts_mut(q8k_buf_ptr, max_n_in / 256) };
            let t0 = Instant::now();
            quantize_q8_0_into(
                attn_out,
                n_embd_q,
                &mut q8_buf[..n_embd_q],
                &mut scale_buf[..n_embd_q / 32],
            );
            crate::ops::quantize_row_q8_k_into(attn_out, &mut q8k_buf[..n_embd_q / 256]);
            let q8 = q8_buf[..n_embd_q].as_ptr();
            let sc = scale_buf[..n_embd_q / 32].as_ptr();
            let q8k = q8k_buf[..n_embd_q / 256].as_ptr();
            pool.compute(move |ith: usize, nth: usize| {
                let input = unsafe { std::slice::from_raw_parts(attn_out_ptr, n_embd_q) };
                let q8 = unsafe { std::slice::from_raw_parts(q8, n_embd_q) };
                let sc = unsafe { std::slice::from_raw_parts(sc, n_embd_q / 32) };
                let q8k = unsafe { std::slice::from_raw_parts(q8k, n_embd_q / 256) };
                let attn_proj = unsafe { std::slice::from_raw_parts_mut(attn_proj_ptr, n_embd) };
                lw.wo.kernel.forward_prepared(
                    input,
                    q8,
                    sc,
                    Some(q8k),
                    attn_proj,
                    n_embd_q,
                    n_embd,
                    ith,
                    nth,
                );
            });
            t_wo += t0.elapsed().as_secs_f64();

            let attn_proj = unsafe { std::slice::from_raw_parts_mut(attn_proj_ptr, n_embd) };
            dbg_tensor(step, "attn_proj", layer, attn_proj);
            let x = unsafe { std::slice::from_raw_parts_mut(x_ptr, n_embd) };
            let normed = unsafe { std::slice::from_raw_parts_mut(normed_ptr, n_embd) };
            for i in 0..n_embd {
                x[i] += attn_proj[i];
            }
            dbg_tensor(step, "ffn_inp", layer, x);
            dbg_full(step, "ffn_inp", layer, x, n_embd);

            let t0 = Instant::now();
            {
                let sum_sq = sum_sq_f32(&x[..n_embd]);
                let mean_sq = (sum_sq / n_embd as f64) as f32;
                let scale = 1.0f32 / (mean_sq + eps).sqrt();
                dbg_scalar(step, "ffn_norm_scale", layer, scale);
                dbg_scalar(step, "ffn_norm_mean", layer, mean_sq);
                dbg_scalar_full(step, "ffn_norm_sum_sq", layer, sum_sq);
            }
            rms_norm(x, &lw.ffn_norm, normed, eps);
            dbg_tensor(step, "ffn_norm", layer, normed);
            dbg_full(step, "ffn_norm", layer, normed, n_embd);
            dbg_tensor(
                step,
                "ffn_norm_weight",
                layer,
                &lw.ffn_norm[..n_embd.min(lw.ffn_norm.len())],
            );
            dbg_full(
                step,
                "ffn_norm_weight",
                layer,
                &lw.ffn_norm[..n_embd.min(lw.ffn_norm.len())],
                n_embd,
            );
            quantize_q8_0_into(
                normed,
                n_embd,
                &mut q8_buf[..n_embd],
                &mut scale_buf[..n_embd / 32],
            );
            crate::ops::quantize_row_q8_k_into(normed, &mut q8k_buf[..n_embd / 256]);
            let q8 = q8_buf[..n_embd].as_ptr();
            let sc = scale_buf[..n_embd / 32].as_ptr();
            let q8k = q8k_buf[..n_embd / 256].as_ptr();

            // DEBUG: dump ffn_norm Q8_0 quantization to verify against Python reference
            if std::env::var("RUST_LLAMA_DEBUG_FFN_Q8").is_ok() && layer == n_layer - 1 && step == 0
            {
                let q8_slice = unsafe { std::slice::from_raw_parts(q8, n_embd) };
                let sc_slice = unsafe { std::slice::from_raw_parts(sc, n_embd / 32) };
                eprintln!(
                    "[RUST_FFN_Q8_L23] input q8 first 32 bytes (block 0): {:?}",
                    &q8_slice[..32]
                );
                eprintln!(
                    "[RUST_FFN_Q8_L23] input q8 bytes 32..64 (block 1): {:?}",
                    &q8_slice[32..64]
                );
                eprintln!("[RUST_FFN_Q8_L23] input scale[0..4]: {:?}", &sc_slice[..4]);
                eprintln!(
                    "[RUST_FFN_Q8_L23] input q8 last 32 bytes: {:?}",
                    &q8_slice[n_embd - 32..]
                );
            }

            pool.compute(move |ith: usize, nth: usize| {
                let input = unsafe { std::slice::from_raw_parts(normed_ptr, n_embd) };
                let q8 = unsafe { std::slice::from_raw_parts(q8, n_embd) };
                let sc = unsafe { std::slice::from_raw_parts(sc, n_embd / 32) };
                let q8k = unsafe { std::slice::from_raw_parts(q8k, n_embd / 256) };
                let gate_buf = unsafe { std::slice::from_raw_parts_mut(gate_buf_ptr, n_ff) };
                let up_buf = unsafe { std::slice::from_raw_parts_mut(up_buf_ptr, n_ff) };
                lw.w_gate.kernel.forward_prepared(
                    input,
                    q8,
                    sc,
                    Some(q8k),
                    up_buf,
                    n_embd,
                    n_ff,
                    ith,
                    nth,
                );
                lw.w_up.kernel.forward_prepared(
                    input,
                    q8,
                    sc,
                    Some(q8k),
                    gate_buf,
                    n_embd,
                    n_ff,
                    ith,
                    nth,
                );

                // Must match the matmul kernel's ceil row partition exactly: a floor
                // split races with the kernel when n_ff % nth != 0 (silu would
                // read rows the matmul hasn't written yet).
                let per_thread = (n_ff + nth - 1) / nth;
                let r_start = ith * per_thread;
                let r_end = (r_start + per_thread).min(n_ff);
                silu_mul_approx_inplace(&up_buf[r_start..r_end], &mut gate_buf[r_start..r_end]);
            });

            {
                let gate_buf = unsafe { std::slice::from_raw_parts(gate_buf_ptr, n_ff) };
                let up_buf = unsafe { std::slice::from_raw_parts(up_buf_ptr, n_ff) };
                dbg_tensor(step, "ffn_gate_buf_raw", layer, gate_buf);
                dbg_tensor(step, "ffn_up_buf_raw", layer, up_buf);
                dbg_full(step, "ffn_gate_buf_raw", layer, gate_buf, n_ff);
                dbg_full(step, "ffn_up_buf_raw", layer, up_buf, n_ff);
            }

            {
                let gate_buf = unsafe { std::slice::from_raw_parts_mut(gate_buf_ptr, n_ff) };
                let q8_buf = unsafe { std::slice::from_raw_parts_mut(q8_buf_ptr, max_n_in) };
                let scale_buf =
                    unsafe { std::slice::from_raw_parts_mut(scale_buf_ptr, max_n_in / 32) };
                let q8k_buf =
                    unsafe { std::slice::from_raw_parts_mut(q8k_buf_ptr, max_n_in / 256) };
                quantize_q8_0_into(
                    gate_buf,
                    n_ff,
                    &mut q8_buf[..n_ff],
                    &mut scale_buf[..n_ff / 32],
                );
                crate::ops::quantize_row_q8_k_into(gate_buf, &mut q8k_buf[..n_ff / 256]);
            }

            let q8 = q8_buf[..n_ff].as_ptr();
            let sc = scale_buf[..n_ff / 32].as_ptr();
            let q8k = q8k_buf[..n_ff / 256].as_ptr();
            pool.compute(move |ith: usize, nth: usize| {
                let input = unsafe { std::slice::from_raw_parts(gate_buf_ptr, n_ff) };
                let q8 = unsafe { std::slice::from_raw_parts(q8, n_ff) };
                let sc = unsafe { std::slice::from_raw_parts(sc, n_ff / 32) };
                let q8k = unsafe { std::slice::from_raw_parts(q8k, n_ff / 256) };
                let down_buf = unsafe { std::slice::from_raw_parts_mut(down_buf_ptr, n_embd) };
                lw.w_down.kernel.forward_prepared(
                    input,
                    q8,
                    sc,
                    Some(q8k),
                    down_buf,
                    n_ff,
                    n_embd,
                    ith,
                    nth,
                );
            });
            t_ffn1 += t0.elapsed().as_secs_f64();

            // DEBUG: print raw down_buf (W_down matmul output) for L23 step 0.
            // Marker: [RUST_L23_DOWN_BUF_RAW]
            if std::env::var("RUST_LLAMA_DEBUG_L23_DOWN").is_ok()
                && layer == n_layer - 1
                && step == 0
            {
                let down_buf = unsafe { std::slice::from_raw_parts(down_buf_ptr, n_embd) };
                eprint!("[RUST_L23_DOWN_BUF_RAW] first16=");
                for v in down_buf.iter().take(16) {
                    eprint!(" {:.5}", v);
                }
                eprintln!();
            }

            let down_buf = unsafe { std::slice::from_raw_parts_mut(down_buf_ptr, n_embd) };
            dbg_tensor(step, "down_buf", layer, down_buf);
            dbg_full(step, "down_buf", layer, down_buf, n_embd);
            let x = unsafe { std::slice::from_raw_parts_mut(x_ptr, n_embd) };
            for i in 0..n_embd {
                x[i] += down_buf[i];
            }
            dbg_tensor(step, "ffn_out", layer, x);
            dbg_tensor(step, "l_out", layer, x);
            dbg_full(step, "ffn_out", layer, x, n_embd);
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
            dbg_tensor(step, "output_norm", 0, normed);
            t_norm += t0.elapsed().as_secs_f64();

            let t0 = Instant::now();
            quantize_q8_0_into(
                normed,
                n_embd,
                &mut q8_buf[..n_embd],
                &mut scale_buf[..n_embd / 32],
            );
            crate::ops::quantize_row_q8_k_into(normed, &mut q8k_buf[..n_embd / 256]);
            let q8 = q8_buf[..n_embd].as_ptr();
            let sc = scale_buf[..n_embd / 32].as_ptr();
            let q8k = q8k_buf[..n_embd / 256].as_ptr();
            let input = normed.as_ptr();
            let output_pw = Weight::from_quantized(QuantizedTensor::from_bytes(
                output_weight,
                output_type,
                n_embd,
                vocab,
            ));
            pool.compute(move |ith: usize, nth: usize| {
                let input = unsafe { std::slice::from_raw_parts(input, n_embd) };
                let q8 = unsafe { std::slice::from_raw_parts(q8, n_embd) };
                let sc = unsafe { std::slice::from_raw_parts(sc, n_embd / 32) };
                let q8k = unsafe { std::slice::from_raw_parts(q8k, n_embd / 256) };
                let logits = unsafe { std::slice::from_raw_parts_mut(logits_ptr, vocab) };
                output_pw.kernel.forward_prepared(
                    input,
                    q8,
                    sc,
                    Some(q8k),
                    logits,
                    n_embd,
                    vocab,
                    ith,
                    nth,
                );
            });
            t_logits += t0.elapsed().as_secs_f64();

            // DEBUG: print LOGITS for step 0 (first 16 values).
            // Marker: [RUST_LOGITS_RAW]
            if std::env::var("RUST_LLAMA_DEBUG_LOGITS").is_ok() && step == 0 {
                let logits = unsafe { std::slice::from_raw_parts(logits_ptr, vocab) };
                eprint!("[RUST_LOGITS_RAW] first16=");
                for v in logits.iter().take(16) {
                    eprint!(" {:.5}", v);
                }
                eprintln!();
            }
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
        // DEBUG: print top-10 logits so we can diff against llama.cpp.
        if std::env::var("RUST_LLAMA_DEBUG_LOGITS").is_ok() {
            let mut idxs: Vec<(usize, f32)> =
                logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
            idxs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let mut line = format!("RUST_LOGITS step={} top10:", step);
            for k in 0..10 {
                line.push_str(&format!(" {}:{:.5}", idxs[k].0, idxs[k].1));
            }
            line.push('\n');
            let _ = io::stderr().write_all(line.as_bytes());
            let _ = io::stderr().flush();
        }
        // Sample using llama.cpp's default sampler chain:
        //   top_k=20 -> top_p=0.95 -> temperature=0.6 -> dist sample.
        // The top_k / top_p values come from the GGUF metadata
        // (`general.sampling.top_k` / `.top_p`) when available.
        let (top_k, top_p) = sample_defaults(source);
        let rng_u64 = if temperature <= 0.0 {
            0
        } else {
            // Seed RNG from generated history (deterministic per prompt).
            let mut rng = 0u64.wrapping_add(0x9E3779B97F4A7C15);
            for &t in &all_tokens {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(t as u64);
            }
            rng
        };
        let chosen = crate::ops::sample_llama_cpp(logits, top_k, top_p, temperature, rng_u64);

        let chosen_id = chosen as u32;
        if !bench && eos_id == Some(chosen_id) {
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
    eprintln!(
        "Prompt: {:.1} t/s | Generation: {:.1} t/s | end-to-end: {:.1} tok/s",
        crate::app::cli::per_second(prefill_evals, prefill_time),
        crate::app::cli::per_second(decode_evals, decode_time),
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
    println!(
        "[{} output tokens in {}ms]",
        generated_tokens.len(),
        infer_ms
    );
    Ok(())
}
