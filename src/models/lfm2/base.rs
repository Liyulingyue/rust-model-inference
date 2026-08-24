//! # LFM2 Inference Loop
//!
//! Text autoregressive inference for LFM2 hybrid (attention + shortconv)
//! models. The attention path mirrors qwen3 (Q/K norm → RoPE → standard
//! attention with KV cache). The recurrent path uses a per-layer
//! `shortconv_state` tensor of shape `[d_conv, n_embd]` and the operator
//!
//! ```text
//!   in_proj(x) -> (b, c, x_chunked)
//!   bx = b * x              // gating
//!   bx = concat(state, bx)  // prepend conv state
//!   state_new = bx[-d_conv..]
//!   y = c * conv1d(bx, kernel)
//!   out = out_proj(y) + x   // residual
//! ```
//!
//! For decode (one new token per step), `n_tokens == 1` and `bx` is the
//! single new element; the conv state shifts left and a new column is
//! appended. For prefill (n_tokens > 1) we process them all at once and
//! write the trailing `d_conv - 1` columns back to the state.

use crate::core::scratchpad::{ExecutionScratchpad, KvCache};
use crate::core::tensor::{GGMLType, TensorSource};
use crate::core::thread_pool::ComputePool;
use crate::core::tokenizer::{BPETokenizer, EncodeOptions};
use crate::ops::kernel::Kernel;
use crate::ops::{
    attention_value_f32, dot_f32, dot_f16_f32, embedding_lookup, quantize_q8_0_into,
    quantize_row_q8_k_into, rms_norm, rms_norm_inplace, rope_neox, sample_top_k,
    silu_mul_approx_inplace, softmax_inplace, vec_mad_f16_f32, vec_scale_f32,
};
use crate::prompt::{build_lfm2_chat_prompt, Lfm2Message};

use std::io::{self, Write};
use std::sync::Arc;
use std::time::{Duration, Instant};

use super::skeleton::{get_f32_tensor, load_layers, Lfm2Config, Lfm2LayerWeights};

pub fn run_inference(
    source: &dyn TensorSource,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    n_threads_arg: usize,
    profile: bool,
    kv_format: KvCacheFmt,
) -> Result<(), String> {
    let t0 = Instant::now();
    let cfg = Lfm2Config::from_source(source)?;
    let n_embd = cfg.n_embd;
    let n_layer = cfg.n_layer;
    let n_head = cfg.n_head;
    let n_ff = cfg.n_ff;

    let arch = "lfm2";

    let tokenizer = BPETokenizer::from_gguf_metadata(|k| source.metadata(k).cloned())
        .map_err(|error| format!("Failed to initialize tokenizer: {error}"))?;

    let max_ctx = 512usize.min(cfg.n_ctx);
    let eps = cfg.norm_eps;
    let freq_base = cfg.rope_freq_base;

    let output_norm = get_f32_tensor(source, "token_embd_norm.weight", n_embd);
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

    let layers = load_layers(source, &cfg)?;

    let load_ms = t0.elapsed().as_millis();
    println!(
        "Model: {} | n_embd={} n_layer={} n_head={} n_ff={} d_conv={} | loaded in {}ms",
        arch, n_embd, n_layer, n_head, n_ff, cfg.d_conv, load_ms
    );

    let input_tokens = build_lfm2_chat_prompt(
        &tokenizer,
        &[Lfm2Message {
            role: "user",
            content: prompt,
        }],
    )?;

    // Determine max n_embd_q across attention layers for scratchpad sizing.
    let n_embd_q = n_head * cfg.n_embd_head_k;
    let n_embd_gqa = cfg
        .n_head_kv_per_layer
        .iter()
        .map(|&h| h * cfg.n_embd_head_k)
        .max()
        .unwrap_or(0)
        .max(n_embd_q);

    let vocab = tokenizer.vocab_size();
    let available_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let n_threads = if n_threads_arg > 0 { n_threads_arg } else { available_threads };

    let mut scratch =
        ExecutionScratchpad::new(n_embd, n_embd_q, n_embd_gqa, n_ff, vocab, n_threads, max_ctx);
    let pool = Arc::new(ComputePool::new(n_threads));
    eprintln!("compute pool: {} threads", pool.n_threads());
    println!("Prompt: {} tokens", input_tokens.len());

    let mut shortconv_states: Vec<Vec<f32>> = Vec::with_capacity(n_layer);
    for (l, lw) in layers.iter().enumerate() {
        if lw.is_attn {
            shortconv_states.push(Vec::new());
        } else {
            // [d_conv, n_embd], padded in time dimension (row major: time is fast).
            // The llama.cpp convention is [n_embd, d_conv] (channels x time). We use
            // the same: state[l][c * d_conv + t].
            shortconv_states.push(vec![0.0f32; n_embd * cfg.d_conv]);
            let _ = l;
        }
    }

    // KV cache: only used by attention layers. To keep allocations simple,
    // we allocate a single cache keyed by layer; non-attention layers just
    // never touch their slot.
    let kv_cache = match kv_format {
        KvCacheFmt::F16 => KvCache::new_f16(n_layer, max_ctx, n_embd_gqa),
        KvCacheFmt::F32 => KvCache::new_f32(n_layer, max_ctx, n_embd_gqa),
    };

    let eos_id = tokenizer.eos_id();
    let mut generated_tokens: Vec<u32> = Vec::new();
    let mut all_tokens: Vec<u32> = input_tokens.clone();
    let mut decoder = tokenizer.streaming_decoder(false);

    let total_steps = input_tokens.len() + max_tokens;
    let t_infer = Instant::now();
    let mut prefill_evals = 0usize;
    let mut prefill_time = Duration::ZERO;
    let mut decode_evals = 0usize;
    let mut decode_time = Duration::ZERO;

    print!("Output: ");
    io::stdout().flush().unwrap();

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
            forward_layer(
                &pool,
                lw,
                layer,
                n_layer,
                &cfg,
                &mut scratch,
                &kv_cache,
                max_ctx,
                pos,
                eps,
                freq_base,
                &mut shortconv_states[layer],
            );
        }

        // Output norm + LM head.
        let x_ptr = scratch.x.as_mut_ptr();
        let normed_ptr = scratch.normed.as_mut_ptr();
        let x = unsafe { std::slice::from_raw_parts_mut(x_ptr, n_embd) };
        let normed = unsafe { std::slice::from_raw_parts_mut(normed_ptr, n_embd) };
        rms_norm(x, &output_norm, normed, eps);

        let q8_buf = unsafe {
            std::slice::from_raw_parts_mut(
                scratch.q8_buf.as_mut_ptr() as *mut u8,
                scratch.q8_buf.len(),
            )
        };
        let scale_buf = unsafe { std::slice::from_raw_parts_mut(scratch.scale_buf.as_mut_ptr(), n_embd / 32) };
        let q8k_buf = unsafe {
            std::slice::from_raw_parts_mut(scratch.q8k_buf.as_mut_ptr(), scratch.q8k_buf.len())
        };
        quantize_q8_0_into(normed, n_embd, &mut q8_buf[..n_embd], &mut scale_buf[..n_embd / 32]);
        quantize_row_q8_k_into(normed, &mut q8k_buf[..n_embd / 256]);
        let q8 = &q8_buf[..n_embd];
        let sc = &scale_buf[..n_embd / 32];
        let q8k = &q8k_buf[..n_embd / 256];

        // Build the output weight lazily once.
        let output_pw = crate::ops::kernel::Weight::from_quantized(
            crate::ops::kernel::QuantizedTensor::from_bytes(
                output_weight,
                output_type,
                n_embd,
                vocab,
            ),
        );

        let logits_ptr = scratch.logits.as_mut_ptr();
        pool.compute(move |ith, nth| {
            let input = unsafe { std::slice::from_raw_parts(normed.as_ptr(), n_embd) };
            let q8 = unsafe { std::slice::from_raw_parts(q8.as_ptr(), n_embd) };
            let sc = unsafe { std::slice::from_raw_parts(sc.as_ptr(), n_embd / 32) };
            let q8k = unsafe { std::slice::from_raw_parts(q8k.as_ptr(), n_embd / 256) };
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
            let top = sample_top_k(logits, 40);
            let mut rng = 0u64;
            for &t in &all_tokens {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(t as u64);
            }
            let r = ((rng >> 33) as f32) / (1u32 << 31) as f32;
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
        if eos_id == Some(chosen_id) {
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
    let per_second = |count: usize, secs: Duration| -> f64 {
        let s = secs.as_secs_f64();
        if s > 0.0 { count as f64 / s } else { 0.0 }
    };
    eprintln!(
        "\nPrompt: {:.1} t/s | Generation: {:.1} t/s | end-to-end: {:.1} tok/s",
        per_second(prefill_evals, prefill_time),
        per_second(decode_evals, decode_time),
        tok_s
    );
    println!();
    println!("[{} output tokens in {}ms]", generated_tokens.len(), infer_ms);
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvCacheFmt {
    F16,
    F32,
}

fn forward_layer(
    pool: &Arc<ComputePool>,
    lw: &Lfm2LayerWeights<'_>,
    layer: usize,
    n_layer: usize,
    cfg: &Lfm2Config,
    mut scratch: &mut ExecutionScratchpad,
    kv_cache: &KvCache,
    max_ctx: usize,
    pos: usize,
    eps: f32,
    freq_base: f32,
    shortconv_state: &mut Vec<f32>,
) {
    let n_embd = cfg.n_embd;
    let n_embd_q = cfg.n_head * cfg.n_embd_head_k;

    let x_ptr = scratch.x.as_mut_ptr();
    let normed_ptr = scratch.normed.as_mut_ptr();
    let q_ptr = scratch.q.as_mut_ptr();
    let k_ptr = scratch.k_new.as_mut_ptr();
    let v_ptr = scratch.v_new.as_mut_ptr();
    let attn_out_ptr = scratch.attn_out.as_mut_ptr();
    let attn_proj_ptr = scratch.attn_proj.as_mut_ptr();
    let gate_buf_ptr = scratch.gate_buf.as_mut_ptr();
    let up_buf_ptr = scratch.up_buf.as_mut_ptr();
    let down_buf_ptr = scratch.down_buf.as_mut_ptr();
    let q8_buf_ptr = scratch.q8_buf.as_mut_ptr() as *mut u8;
    let scale_buf_ptr = scratch.scale_buf.as_mut_ptr();
    let q8k_buf_ptr = scratch.q8k_buf.as_mut_ptr();
    let max_n_in = n_embd_q.max(cfg.n_ff);
    let q8_buf = unsafe { std::slice::from_raw_parts_mut(q8_buf_ptr, max_n_in) };
    let scale_buf =
        unsafe { std::slice::from_raw_parts_mut(scale_buf_ptr, max_n_in / 32) };
    let q8k_buf = unsafe { std::slice::from_raw_parts_mut(q8k_buf_ptr, max_n_in / 256) };

    unsafe {
        // ---- Pre-attention / pre-conv norm ----
        let x = std::slice::from_raw_parts_mut(x_ptr, n_embd);
        let normed = std::slice::from_raw_parts_mut(normed_ptr, n_embd);
        rms_norm(x, &lw.attn_norm, normed, eps);
        quantize_q8_0_into(
            normed,
            n_embd,
            &mut q8_buf[..n_embd],
            &mut scale_buf[..n_embd / 32],
        );
        quantize_row_q8_k_into(normed, &mut q8k_buf[..n_embd / 256]);
    }

    let q8 = &q8_buf[..n_embd];
    let sc = &scale_buf[..n_embd / 32];
    let q8k = &q8k_buf[..n_embd / 256];

let cur_after_block = if lw.is_attn {
        forward_attention(
            &pool,
            lw,
            cfg,
            &mut scratch,
            &kv_cache,
            max_ctx,
            pos,
            freq_base,
            eps,
            layer,
            n_layer,
        );
        let attn_proj = unsafe { std::slice::from_raw_parts(attn_proj_ptr, n_embd) };
        let x = unsafe { std::slice::from_raw_parts_mut(x_ptr, n_embd) };
        for i in 0..n_embd {
            x[i] += attn_proj[i];
        }
        unsafe { std::slice::from_raw_parts(x_ptr, n_embd).to_vec() }
    } else {
        let cur = forward_shortconv(
            &pool,
            lw,
            cfg,
            &mut scratch,
            pos,
            shortconv_state,
        );
        let x = unsafe { std::slice::from_raw_parts_mut(x_ptr, n_embd) };
        for i in 0..n_embd {
            x[i] += cur[i];
        }
        unsafe { std::slice::from_raw_parts(x_ptr, n_embd).to_vec() }
    };
    let _ = cur_after_block;

    // ---- FFN (always present) ----
    unsafe {
        let x = std::slice::from_raw_parts_mut(x_ptr, n_embd);
        let normed = std::slice::from_raw_parts_mut(normed_ptr, n_embd);
        rms_norm(x, &lw.ffn_norm, normed, eps);
        quantize_q8_0_into(
            normed,
            n_embd,
            &mut q8_buf[..n_embd],
            &mut scale_buf[..n_embd / 32],
        );
        quantize_row_q8_k_into(normed, &mut q8k_buf[..n_embd / 256]);
    }

    let q8 = &q8_buf[..n_embd];
    let sc = &scale_buf[..n_embd / 32];
    let q8k = &q8k_buf[..n_embd / 256];
    let n_ff = cfg.n_ff;

    pool.compute({
        let input_ptr = normed_ptr;
        let q8_ptr = q8.as_ptr();
        let sc_ptr = sc.as_ptr();
        let q8k_ptr = q8k.as_ptr();
        let gate_buf_ptr = gate_buf_ptr;
        let up_buf_ptr = up_buf_ptr;
        move |ith, nth| {
            let input = unsafe { std::slice::from_raw_parts(input_ptr, n_embd) };
            let q8 = unsafe { std::slice::from_raw_parts(q8_ptr, n_embd) };
            let sc = unsafe { std::slice::from_raw_parts(sc_ptr, n_embd / 32) };
            let q8k = unsafe { std::slice::from_raw_parts(q8k_ptr, n_embd / 256) };
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
            let rows_per = n_ff / nth;
            let r_start = ith * rows_per;
            let r_end = if ith == nth - 1 { n_ff } else { r_start + rows_per };
            silu_mul_approx_inplace(&up_buf[r_start..r_end], &mut gate_buf[r_start..r_end]);
        }
    });

    let gate_buf =
        unsafe { std::slice::from_raw_parts_mut(gate_buf_ptr, n_ff) };
    quantize_q8_0_into(
        gate_buf,
        n_ff,
        &mut q8_buf[..n_ff],
        &mut scale_buf[..n_ff / 32],
    );
    quantize_row_q8_k_into(gate_buf, &mut q8k_buf[..n_ff / 256]);
    let q8 = &q8_buf[..n_ff];
    let sc = &scale_buf[..n_ff / 32];
    let q8k = &q8k_buf[..n_ff / 256];
    pool.compute({
        let gate_buf_ptr = gate_buf_ptr;
        let q8_ptr = q8.as_ptr();
        let sc_ptr = sc.as_ptr();
        let q8k_ptr = q8k.as_ptr();
        let down_buf_ptr = down_buf_ptr;
        move |ith, nth| {
            let input = unsafe { std::slice::from_raw_parts(gate_buf_ptr, n_ff) };
            let q8 = unsafe { std::slice::from_raw_parts(q8_ptr, n_ff) };
            let sc = unsafe { std::slice::from_raw_parts(sc_ptr, n_ff / 32) };
            let q8k = unsafe { std::slice::from_raw_parts(q8k_ptr, n_ff / 256) };
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
        }
    });

    let down_buf = unsafe { std::slice::from_raw_parts(down_buf_ptr, n_embd) };
    let x = unsafe { std::slice::from_raw_parts_mut(x_ptr, n_embd) };
    for i in 0..n_embd {
        x[i] += down_buf[i];
    }
    let _ = cur_after_block;
}

fn forward_attention(
    pool: &Arc<ComputePool>,
    lw: &Lfm2LayerWeights<'_>,
    cfg: &Lfm2Config,
    scratch: &mut ExecutionScratchpad,
    kv_cache: &KvCache,
    max_ctx: usize,
    pos: usize,
    freq_base: f32,
    eps: f32,
    layer: usize,
    n_layer: usize,
) {
    let n_embd = cfg.n_embd;
    let n_head = cfg.n_head;
    let n_head_kv = cfg.n_head_kv_per_layer[layer];
    let n_embd_head_k = cfg.n_embd_head_k;
    let n_embd_head_v = cfg.n_embd_head_v;
    let n_embd_q = n_head * n_embd_head_k;
    let n_embd_gqa = n_head_kv * n_embd_head_v;

    let q_ptr = scratch.q.as_mut_ptr();
    let k_ptr = scratch.k_new.as_mut_ptr();
    let v_ptr = scratch.v_new.as_mut_ptr();
    let attn_out_ptr = scratch.attn_out.as_mut_ptr();
    let attn_proj_ptr = scratch.attn_proj.as_mut_ptr();
    let normed_ptr = scratch.normed.as_mut_ptr();
    let max_n_in = n_embd_q.max(cfg.n_ff);
    let q8_buf_ptr = scratch.q8_buf.as_mut_ptr() as *mut u8;
    let scale_buf_ptr = scratch.scale_buf.as_mut_ptr();
    let q8k_buf_ptr = scratch.q8k_buf.as_mut_ptr();

    let q8_buf = unsafe { std::slice::from_raw_parts_mut(q8_buf_ptr, max_n_in) };
    let scale_buf =
        unsafe { std::slice::from_raw_parts_mut(scale_buf_ptr, max_n_in / 32) };
    let q8k_buf = unsafe { std::slice::from_raw_parts_mut(q8k_buf_ptr, max_n_in / 256) };

    // QKV matmul.
    let q8 = &q8_buf[..n_embd];
    let sc = &scale_buf[..n_embd / 32];
    let q8k = &q8k_buf[..n_embd / 256];

    pool.compute({
        let input_ptr = normed_ptr;
        let q8_ptr = q8.as_ptr();
        let sc_ptr = sc.as_ptr();
        let q8k_ptr = q8k.as_ptr();
        let q_ptr = q_ptr;
        let k_ptr = k_ptr;
        let v_ptr = v_ptr;
        move |ith, nth| {
            let input = unsafe { std::slice::from_raw_parts(input_ptr, n_embd) };
            let q8 = unsafe { std::slice::from_raw_parts(q8_ptr, n_embd) };
            let sc = unsafe { std::slice::from_raw_parts(sc_ptr, n_embd / 32) };
            let q8k = unsafe { std::slice::from_raw_parts(q8k_ptr, n_embd / 256) };
            let q = unsafe { std::slice::from_raw_parts_mut(q_ptr, n_embd_q) };
            let k_new = unsafe { std::slice::from_raw_parts_mut(k_ptr, n_embd_gqa) };
            let v_new = unsafe { std::slice::from_raw_parts_mut(v_ptr, n_embd_gqa) };
            lw.wq.as_ref().unwrap().kernel.forward_prepared(
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
            lw.wk.as_ref().unwrap().kernel.forward_prepared(
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
            lw.wv.as_ref().unwrap().kernel.forward_prepared(
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
        }
    });

    // Q/K norm + RoPE.
    unsafe {
        let q = std::slice::from_raw_parts_mut(q_ptr, n_embd_q);
        let k_new = std::slice::from_raw_parts_mut(k_ptr, n_embd_gqa);
        let qn = lw.q_norm.as_deref().unwrap();
        let kn = lw.k_norm.as_deref().unwrap();
        for h in 0..n_head {
            rms_norm_inplace(&mut q[h * n_embd_head_k..(h + 1) * n_embd_head_k], qn, eps);
        }
        for h in 0..n_head_kv {
            rms_norm_inplace(&mut k_new[h * n_embd_head_k..(h + 1) * n_embd_head_k], kn, eps);
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
                n_embd_head_k,
                freq_base,
            );
        }
    }

    // KV cache write.
    let kb = layer * max_ctx * n_embd_gqa;
    let (k_cache_f16_ptr, v_cache_f16_ptr) = match kv_cache {
        KvCache::F16(c) => (c.k.as_ptr() as *mut u16, c.v.as_ptr() as *mut u16),
        _ => (std::ptr::null_mut(), std::ptr::null_mut()),
    };
    let (k_cache_f32_ptr, v_cache_f32_ptr) = match kv_cache {
        KvCache::F32(c) => (c.k.as_ptr() as *mut f32, c.v.as_ptr() as *mut f32),
        _ => (std::ptr::null_mut(), std::ptr::null_mut()),
    };
    unsafe {
        let k_new = std::slice::from_raw_parts(k_ptr, n_embd_gqa);
        let v_new = std::slice::from_raw_parts(v_ptr, n_embd_gqa);
        let off = kb + pos * n_embd_gqa;
        if !k_cache_f16_ptr.is_null() {
            for i in 0..n_embd_gqa {
                *k_cache_f16_ptr.add(off + i) = crate::ops::f32_to_f16(k_new[i]);
                *v_cache_f16_ptr.add(off + i) = crate::ops::f32_to_f16(v_new[i]);
            }
        } else {
            for i in 0..n_embd_gqa {
                *k_cache_f32_ptr.add(off + i) = k_new[i];
                *v_cache_f32_ptr.add(off + i) = v_new[i];
            }
        }
    }

    // Attention compute (single-token decode path).
    let group_size = n_head / n_head_kv;
    let kq_scale = 1.0f32 / (n_embd_head_k as f32).sqrt();

    match kv_cache {
        KvCache::F16(_) => {
            let k_cache = unsafe { std::slice::from_raw_parts(k_cache_f16_ptr as *const u16, n_layer * max_ctx * n_embd_gqa) };
            let v_cache = unsafe { std::slice::from_raw_parts(v_cache_f16_ptr as *const u16, n_layer * max_ctx * n_embd_gqa) };
            pool.compute({
                let q_ptr = q_ptr;
                let attn_out_ptr = attn_out_ptr;
                let kb_local = kb;
                move |ith, nth| {
                    let q = unsafe { std::slice::from_raw_parts(q_ptr, n_embd_q) };
                    let attn_out = unsafe { std::slice::from_raw_parts_mut(attn_out_ptr, n_embd_q) };
                    let h_start = ith * n_head / nth;
                    let h_end = (ith + 1) * n_head / nth;
                    for h in h_start..h_end {
                        let kv_h = h / group_size;
                        let q_off = h * n_embd_head_k;
                        let out_base = h * n_embd_head_v;
                        let mut ms = 0.0f32;
                        let mut s_sum = 0.0f32;
                        for d in 0..n_embd_head_v {
                            attn_out[out_base + d] = 0.0;
                        }
                        for t in 0..pos + 1 {
                            let score = dot_f16_f32(
                                &q[q_off..q_off + n_embd_head_k],
                                &k_cache[kb_local + t * n_embd_gqa + kv_h * n_embd_head_v
                                    ..kb_local + t * n_embd_gqa + kv_h * n_embd_head_v + n_embd_head_k],
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
                            let v_base = kb_local + t * n_embd_gqa + kv_h * n_embd_head_v;
                            vec_mad_f16_f32(
                                &mut attn_out[out_base..out_base + n_embd_head_v],
                                &v_cache[v_base..v_base + n_embd_head_v],
                                vs,
                            );
                            s_sum += vs;
                        }
                        let inv_sum = 1.0 / s_sum;
                        vec_scale_f32(
                            &mut attn_out[out_base..out_base + n_embd_head_v],
                            inv_sum,
                        );
                    }
                }
            });
        }
        KvCache::F32(_) => {
            let k_cache = unsafe { std::slice::from_raw_parts(k_cache_f32_ptr, n_layer * max_ctx * n_embd_gqa) };
            let v_cache = unsafe { std::slice::from_raw_parts(v_cache_f32_ptr, n_layer * max_ctx * n_embd_gqa) };
            let n_cached = pos + 1;
            let n_padded = (n_cached + 255) / 256 * 256;
            let score_stride = scratch.score_stride;
            let scores_ptr = scratch.scores.as_mut_ptr();
            pool.compute({
                let q_ptr = q_ptr;
                let attn_out_ptr = attn_out_ptr;
                let kb_local = kb;
                move |ith, nth| {
                    let q = unsafe { std::slice::from_raw_parts(q_ptr, n_embd_q) };
                    let attn_out = unsafe { std::slice::from_raw_parts_mut(attn_out_ptr, n_embd_q) };
                    let h_start = ith * n_head / nth;
                    let h_end = (ith + 1) * n_head / nth;
                    for h in h_start..h_end {
                        let kv_h = h / group_size;
                        let q_off = h * n_embd_head_k;
                        let out_base = h * n_embd_head_v;
                        let s_off = ith * score_stride;
                        let scores = unsafe {
                            std::slice::from_raw_parts_mut(
                                scores_ptr.add(s_off),
                                score_stride,
                            )
                        };
                        for t in 0..n_cached {
                            scores[t] = dot_f32(
                                &q[q_off..q_off + n_embd_head_k],
                                &k_cache[kb_local + t * n_embd_gqa + kv_h * n_embd_head_v
                                    ..kb_local + t * n_embd_gqa + kv_h * n_embd_head_v + n_embd_head_k],
                                n_embd_head_k,
                            ) * kq_scale;
                        }
                        for v in &mut scores[n_cached..n_padded] {
                            *v = f32::NEG_INFINITY;
                        }
                        softmax_inplace(&mut scores[..n_padded]);
                        let mut values = [0.0f32; 512];
                        for d in 0..n_embd_head_v {
                            for t in 0..n_cached {
                                values[t] = v_cache[kb_local + t * n_embd_gqa + kv_h * n_embd_head_v + d];
                            }
                            attn_out[out_base + d] = attention_value_f32(
                                &values[..n_padded],
                                &scores[..n_padded],
                                n_cached,
                                n_padded,
                            );
                        }
                    }
                }
            });
        }
    }

    // Output projection.
    let attn_out = unsafe { std::slice::from_raw_parts(attn_out_ptr, n_embd_q) };
    quantize_q8_0_into(
        attn_out,
        n_embd_q,
        &mut q8_buf[..n_embd_q],
        &mut scale_buf[..n_embd_q / 32],
    );
    quantize_row_q8_k_into(attn_out, &mut q8k_buf[..n_embd_q / 256]);
    let q8 = &q8_buf[..n_embd_q];
    let sc = &scale_buf[..n_embd_q / 32];
    let q8k = &q8k_buf[..n_embd_q / 256];

    pool.compute({
        let attn_out_ptr = attn_out_ptr;
        let q8_ptr = q8.as_ptr();
        let sc_ptr = sc.as_ptr();
        let q8k_ptr = q8k.as_ptr();
        let attn_proj_ptr = attn_proj_ptr;
        move |ith, nth| {
            let input = unsafe { std::slice::from_raw_parts(attn_out_ptr, n_embd_q) };
            let q8 = unsafe { std::slice::from_raw_parts(q8_ptr, n_embd_q) };
            let sc = unsafe { std::slice::from_raw_parts(sc_ptr, n_embd_q / 32) };
            let q8k = unsafe { std::slice::from_raw_parts(q8k_ptr, n_embd_q / 256) };
            let attn_proj = unsafe { std::slice::from_raw_parts_mut(attn_proj_ptr, n_embd) };
            lw.wo.as_ref().unwrap().kernel.forward_prepared(
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
        }
    });
}

fn forward_shortconv(
    pool: &Arc<ComputePool>,
    lw: &Lfm2LayerWeights<'_>,
    cfg: &Lfm2Config,
    scratch: &mut ExecutionScratchpad,
    pos: usize,
    state: &mut Vec<f32>,
) -> Vec<f32> {
    let n_embd = cfg.n_embd;
    let d_conv = cfg.d_conv;
    let l_cache = cfg.l_cache;
    let n_embd_in = scratch.normed.len();
    debug_assert_eq!(n_embd_in, n_embd);

    let normed_ptr = scratch.normed.as_mut_ptr();
    let q8_buf_ptr = scratch.q8_buf.as_mut_ptr() as *mut u8;
    let scale_buf_ptr = scratch.scale_buf.as_mut_ptr();
    let q8k_buf_ptr = scratch.q8k_buf.as_mut_ptr();
    let gate_buf_ptr = scratch.gate_buf.as_mut_ptr();

    let max_n_in = (n_embd * 3).max(cfg.n_ff);
    let q8_buf = unsafe { std::slice::from_raw_parts_mut(q8_buf_ptr, max_n_in) };
    let scale_buf =
        unsafe { std::slice::from_raw_parts_mut(scale_buf_ptr, max_n_in / 32) };
    let q8k_buf = unsafe { std::slice::from_raw_parts_mut(q8k_buf_ptr, max_n_in / 256) };

    unsafe {
        let normed = std::slice::from_raw_parts(normed_ptr, n_embd);
        quantize_q8_0_into(
            normed,
            n_embd,
            &mut q8_buf[..n_embd],
            &mut scale_buf[..n_embd / 32],
        );
        quantize_row_q8_k_into(normed, &mut q8k_buf[..n_embd / 256]);
    }

    // Step 1: in_proj -> 3 chunks of size n_embd: b, c, x.
    let q8 = &q8_buf[..n_embd];
    let sc = &scale_buf[..n_embd / 32];
    let q8k = &q8k_buf[..n_embd / 256];
    let three_n = 3 * n_embd;

    pool.compute({
        let input_ptr = normed_ptr;
        let q8_ptr = q8.as_ptr();
        let sc_ptr = sc.as_ptr();
        let q8k_ptr = q8k.as_ptr();
        let gate_buf_ptr = gate_buf_ptr;
        move |ith, nth| {
            let input = unsafe { std::slice::from_raw_parts(input_ptr, n_embd) };
            let q8 = unsafe { std::slice::from_raw_parts(q8_ptr, n_embd) };
            let sc = unsafe { std::slice::from_raw_parts(sc_ptr, n_embd / 32) };
            let q8k = unsafe { std::slice::from_raw_parts(q8k_ptr, n_embd / 256) };
            // gate_buf holds `[b | c | x]` (concatenated, length 3*n_embd).
            let bcx = unsafe { std::slice::from_raw_parts_mut(gate_buf_ptr, three_n) };
            lw.shortconv_in.as_ref().unwrap().kernel.forward_prepared(
                input,
                q8,
                sc,
                Some(q8k),
                bcx,
                n_embd,
                three_n,
                ith,
                nth,
            );
        }
    });

    // DEBUG: dump bcx first 16 values for layer 0
    if pos == 0 && cfg.d_conv > 0 && false {
        unsafe {
            let bcx_debug = std::slice::from_raw_parts(gate_buf_ptr, three_n);
            eprintln!("[LFM2-DEBUG] layer 0 step 0 bcx shape={} d_conv={} pos={}", three_n, cfg.d_conv, pos);
            eprintln!("[LFM2-DEBUG] bcx[0]={} bcx[1]={} bcx[2]={} bcx[3]={}", bcx_debug[0], bcx_debug[1], bcx_debug[2], bcx_debug[3]);
            eprintln!("[LFM2-DEBUG] x[0..4] (4096..4100): bcx[4096]={} bcx[4097]={} bcx[4098]={} bcx[4099]={}", bcx_debug[4096], bcx_debug[4097], bcx_debug[4098], bcx_debug[4099]);
        }
    }

    // Step 2: bx = b * x (per-channel); conv1d over (state || bx); multiply by c; out_proj.
    let bcx = unsafe { std::slice::from_raw_parts(gate_buf_ptr, three_n) };
    let b = &bcx[..n_embd];
    let c = &bcx[n_embd..2 * n_embd];
    let x = &bcx[2 * n_embd..3 * n_embd];

    // bx = b * x
    let mut bx: Vec<f32> = vec![0.0; n_embd];
    for i in 0..n_embd {
        bx[i] = b[i] * x[i];
    }

    // bx_buf is full conv input (state || bx). State has d_conv rows, bx has 1 row.
    let mut bx_buf: Vec<f32> = Vec::with_capacity(d_conv * n_embd + n_embd);
    if state.len() != d_conv * n_embd {
        state.resize(d_conv * n_embd, 0.0);
    }
    bx_buf.extend_from_slice(state);
    bx_buf.extend_from_slice(&bx);

    // DEBUG: print bx for layer 0 BOS
    if pos == 0 && cfg.d_conv > 0 && false {
        eprintln!("[LFM2-DEBUG] layer 0 pos={} bx[0..8] = {:?}", pos, &bx[..8]);
    }

    // Update state: keep the last `d_conv` columns of bx_buf.
    // bx_buf[d_conv * n_embd..] = the new bx.
    // The new state is the last d_conv columns, i.e. bx_buf[n_embd..(d_conv + 1) * n_embd].
    let new_state_start = n_embd; // skip the first (oldest) row, take the remaining d_conv rows
    for (i, v) in bx_buf[new_state_start..new_state_start + d_conv * n_embd]
        .iter()
        .enumerate()
    {
        state[i] = *v;
    }

    // Per-channel conv1d with kernel of size `l_cache`. The conv slides over
    // bx_buf (which has `d_conv + 1` rows). Number of output positions:
    // `n_t = (d_conv + 1) - l_cache + 1 = (d_conv + 1) - (d_conv + 1) + 1 + 1 - 1`
    // Hmm wait that's confusing. Let me restate:
    //   d_conv = l_cache - 1
    //   bx_buf.len() (rows) = d_conv + 1 = l_cache
    //   For ssm_conv: n_t = (l_cache) - l_cache + 1 = 1
    // So we get exactly 1 output. Good.
    let kernel = lw.shortconv_conv.as_ref().unwrap();
    debug_assert_eq!(kernel.len(), l_cache * n_embd);
    // Per-channel conv1d: y[c] = sum_{k=0..l_cache-1} bx_buf[i2 + k][c] * kernel[k][c]
    // For n_t = 1, only i2 = 0:
    let mut conv_out: Vec<f32> = vec![0.0; n_embd];
    for c_idx in 0..n_embd {
        let mut acc = 0.0f32;
        for k in 0..l_cache {
            acc += bx_buf[k * n_embd + c_idx] * kernel[k * n_embd + c_idx];
        }
        conv_out[c_idx] = acc;
    }

    // y = c * conv_out (per-channel gating)
    for i in 0..n_embd {
        conv_out[i] *= c[i];
    }

    // Step 3: out_proj.
    quantize_q8_0_into(
        &conv_out,
        n_embd,
        &mut q8_buf[..n_embd],
        &mut scale_buf[..n_embd / 32],
    );
    quantize_row_q8_k_into(&conv_out, &mut q8k_buf[..n_embd / 256]);
    let q8 = &q8_buf[..n_embd];
    let sc = &scale_buf[..n_embd / 32];
    let q8k = &q8k_buf[..n_embd / 256];
    let mut out: Vec<f32> = vec![0.0; n_embd];
    pool.compute({
        let q8_ptr = q8.as_ptr();
        let sc_ptr = sc.as_ptr();
        let q8k_ptr = q8k.as_ptr();
        let out_ptr = out.as_mut_ptr();
        move |ith, nth| {
            let input = &conv_out[..];
            let q8 = unsafe { std::slice::from_raw_parts(q8_ptr, n_embd) };
            let sc = unsafe { std::slice::from_raw_parts(sc_ptr, n_embd / 32) };
            let q8k = unsafe { std::slice::from_raw_parts(q8k_ptr, n_embd / 256) };
            let o = unsafe { std::slice::from_raw_parts_mut(out_ptr, n_embd) };
            lw.shortconv_out.as_ref().unwrap().kernel.forward_prepared(
                input,
                q8,
                sc,
                Some(q8k),
                o,
                n_embd,
                n_embd,
                ith,
                nth,
            );
        }
    });

    let _ = pos;
    out
}