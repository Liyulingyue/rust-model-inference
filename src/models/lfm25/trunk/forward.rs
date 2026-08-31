use crate::core::scratchpad::{ExecutionScratchpad, KvCache, KvFormat};
use crate::core::tensor::TensorSource;
use crate::core::thread_pool::ComputePool;
use crate::core::tokenizer::BPETokenizer;
use crate::ops::kernel::Kernel;
use crate::ops::{
    attention_value_f32, dot_f32, dot_f16_f32, embedding_lookup, quantize_q8_0_into,
    quantize_row_q8_k_into, rms_norm, rms_norm_inplace, rope_neox, sample_top_k,
    silu_mul_inplace, softmax_inplace, vec_add_into, vec_mad_f16_f32, vec_mul_inplace,
    vec_scale_f32,
};
use crate::prompt::{build_lfm2_chat_prompt, Lfm2Message};

use std::io::{self, Write};
use std::sync::Arc;
use std::time::{Duration, Instant};

use super::config::Lfm25Config;
use super::weights::{get_f32_tensor, load_layers, Lfm25LayerWeights};

pub fn run_inference(
    source: &dyn TensorSource,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    n_threads_arg: usize,
    profile: bool,
    kv_format: KvFormat,
) -> Result<(), String> {
    let t0 = Instant::now();
    let cfg = Lfm25Config::from_source(source)?;
    let n_embd = cfg.n_embd;
    let n_layer = cfg.n_layer;
    let n_head = cfg.n_head;
    let n_ff = cfg.n_ff;

    let arch = "lfm2.5";

    let tokenizer = BPETokenizer::from_gguf_metadata(|k| source.metadata(k).cloned())
        .map_err(|error| format!("Failed to initialize tokenizer: {error}"))?;

    let max_ctx = 512usize.min(cfg.n_ctx);
    let eps = cfg.norm_eps;
    let freq_base = cfg.rope_freq_base;

    let output_norm = get_f32_tensor(source, "token_embd_norm.weight", n_embd);
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
    eprintln!("[RUST_TOKENS] n={} ids={:?}", input_tokens.len(), input_tokens);

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
    let mut accumulated_bx: Vec<Vec<Vec<f32>>> = Vec::with_capacity(n_layer);
    for (l, lw) in layers.iter().enumerate() {
        if lw.is_attn {
            shortconv_states.push(Vec::new());
            accumulated_bx.push(Vec::new());
        } else {
            shortconv_states.push(vec![0.0f32; n_embd * cfg.d_conv]);
            accumulated_bx.push(Vec::new());
            let _ = l;
        }
    }

    let kv_cache = match kv_format {
        KvFormat::F16 => KvCache::new_f16(n_layer, max_ctx, n_embd_gqa),
        KvFormat::F32 => KvCache::new_f32(n_layer, max_ctx, n_embd_gqa),
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

        let is_prefill = step < input_tokens.len();
        for layer in 0..n_layer {
            let lw = &layers[layer];
            if !lw.is_attn && is_prefill {
                let d_conv = cfg.d_conv;
                let n_embd = cfg.n_embd;
                let state = &mut shortconv_states[layer];
                state.resize(d_conv * n_embd, 0.0);
                let hist = &accumulated_bx[layer];
                for k_p in 0..d_conv {
                    // Right-align the history: zero pads the FRONT of the
                    // conv window (ggml_ssm_conv taps K[0] on the oldest entry).
                    let idx = k_p as isize - (d_conv - hist.len()) as isize;
                    if idx >= 0 {
                        let src = &hist[idx as usize];
                        for ci in 0..n_embd {
                            state[k_p * n_embd + ci] = src[ci];
                        }
                    }
                }
            }
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
                &mut accumulated_bx[layer],
                is_prefill,
            );
        }

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
        // Parity debugging: dump top-10 logits per step when
        // RUST_LFM25_DEBUG_LOGITS is set (mirrors the other trunks).
        if std::env::var("RUST_LFM25_DEBUG_LOGITS").is_ok() {
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
        let chosen = if temperature <= 0.0 {
            logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
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

fn forward_layer(
    pool: &Arc<ComputePool>,
    lw: &Lfm25LayerWeights<'_>,
    layer: usize,
    n_layer: usize,
    cfg: &Lfm25Config,
    mut scratch: &mut ExecutionScratchpad,
    kv_cache: &KvCache,
    max_ctx: usize,
    pos: usize,
    eps: f32,
    freq_base: f32,
    shortconv_state: &mut Vec<f32>,
    accumulated_bx: &mut Vec<Vec<f32>>,
    is_prefill: bool,
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

    let _cur_after_block = if lw.is_attn {
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
        vec_add_into(attn_proj, x);
        unsafe { std::slice::from_raw_parts(x_ptr, n_embd).to_vec() }
    } else {
        let (cur, bx) = forward_shortconv(
            &pool,
            lw,
            layer,
            cfg,
            &mut scratch,
            pos,
            shortconv_state,
            accumulated_bx,
            is_prefill,
        );
        let x = unsafe { std::slice::from_raw_parts_mut(x_ptr, n_embd) };
        vec_add_into(&cur, x);
        unsafe { std::slice::from_raw_parts(x_ptr, n_embd).to_vec() }
    };

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
                input, q8, sc, Some(q8k), up_buf, n_embd, n_ff, ith, nth,
            );
            lw.w_up.kernel.forward_prepared(
                input, q8, sc, Some(q8k), gate_buf, n_embd, n_ff, ith, nth,
            );
            if crate::ops::gpu_matmul_active() {
                // Matmul ran as one fenced GPU dispatch owned by thread 0.
                if ith == 0 {
                    silu_mul_inplace(&up_buf[..n_ff], &mut gate_buf[..n_ff]);
                }
            } else {
                // Must match the matmul kernel's ceil row partition exactly: a floor
                // split races with the kernel when n_ff % nth != 0 (silu would
                // read rows the matmul hasn't written yet).
                let per_thread = (n_ff + nth - 1) / nth;
                let r_start = ith * per_thread;
                let r_end = (r_start + per_thread).min(n_ff);
                silu_mul_inplace(&up_buf[r_start..r_end], &mut gate_buf[r_start..r_end]);
            }
        }
    });

    let gate_buf =
        unsafe { std::slice::from_raw_parts_mut(gate_buf_ptr, n_ff) };
    quantize_q8_0_into(
        gate_buf, n_ff, &mut q8_buf[..n_ff], &mut scale_buf[..n_ff / 32],
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
                input, q8, sc, Some(q8k), down_buf, n_ff, n_embd, ith, nth,
            );
        }
    });

    let down_buf = unsafe { std::slice::from_raw_parts(down_buf_ptr, n_embd) };
    let x = unsafe { std::slice::from_raw_parts_mut(x_ptr, n_embd) };
    vec_add_into(down_buf, x);
}

fn forward_attention(
    pool: &Arc<ComputePool>,
    lw: &Lfm25LayerWeights<'_>,
    cfg: &Lfm25Config,
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
                input, q8, sc, Some(q8k), q, n_embd, n_embd_q, ith, nth,
            );
            lw.wk.as_ref().unwrap().kernel.forward_prepared(
                input, q8, sc, Some(q8k), k_new, n_embd, n_embd_gqa, ith, nth,
            );
            lw.wv.as_ref().unwrap().kernel.forward_prepared(
                input, q8, sc, Some(q8k), v_new, n_embd, n_embd_gqa, ith, nth,
            );
        }
    });

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
            let k_dst =
                std::slice::from_raw_parts_mut(k_cache_f32_ptr.add(off), n_embd_gqa);
            let v_dst =
                std::slice::from_raw_parts_mut(v_cache_f32_ptr.add(off), n_embd_gqa);
            k_dst.copy_from_slice(k_new);
            v_dst.copy_from_slice(v_new);
        }
    }

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
                input, q8, sc, Some(q8k), attn_proj, n_embd_q, n_embd, ith, nth,
            );
        }
    });
}

fn forward_shortconv(
    pool: &Arc<ComputePool>,
    lw: &Lfm25LayerWeights<'_>,
    layer_idx: usize,
    cfg: &Lfm25Config,
    scratch: &mut ExecutionScratchpad,
    pos: usize,
    state: &mut Vec<f32>,
    accumulated_bx: &mut Vec<Vec<f32>>,
    is_prefill: bool,
) -> (Vec<f32>, Vec<f32>) {
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

    let three_n = 3 * n_embd;
    let q8 = &q8_buf[..n_embd];
    let sc = &scale_buf[..n_embd / 32];
    let q8k = &q8k_buf[..n_embd / 256];

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
            let bcx = unsafe { std::slice::from_raw_parts_mut(gate_buf_ptr, three_n) };
            lw.shortconv_in.as_ref().unwrap().kernel.forward_prepared(
                input, q8, sc, Some(q8k), bcx, n_embd, three_n, ith, nth,
            );
        }
    });

    let bcx = unsafe { std::slice::from_raw_parts(gate_buf_ptr, three_n) };
    let b = &bcx[..n_embd];
    let c = &bcx[n_embd..2 * n_embd];
    let x = &bcx[2 * n_embd..3 * n_embd];

    let mut bx: Vec<f32> = b.to_vec();
    vec_mul_inplace(x, bx.as_mut_slice());

    if is_prefill {
        accumulated_bx.push(bx.clone());
        let keep = cfg.d_conv;
        if accumulated_bx.len() > keep {
            let drop_n = accumulated_bx.len() - keep;
            accumulated_bx.drain(0..drop_n);
        }
    }

    let l_buf = d_conv + 1;
    let mut bx_buf: Vec<f32> = vec![0.0f32; n_embd * l_buf];
    if state.len() != d_conv * n_embd {
        state.resize(d_conv * n_embd, 0.0);
    }
    for k in 0..d_conv {
        let row = &state[k * n_embd..(k + 1) * n_embd];
        for c in 0..n_embd {
            bx_buf[c * l_buf + k] = row[c];
        }
    }

    for ci in 0..n_embd {
        bx_buf[ci * l_buf + d_conv] = bx[ci];
    }

    // Sliding window: shift left one row, append b*x as the newest entry
    // (llama.cpp's new_conv = last d_conv columns of (state ‖ bx)).
    if !is_prefill {
        state.copy_within(n_embd..d_conv * n_embd, 0);
        state[(d_conv - 1) * n_embd..d_conv * n_embd].copy_from_slice(&bx);
    }

    let kernel = lw.shortconv_conv.as_ref().unwrap();
    debug_assert_eq!(kernel.len(), l_cache * n_embd);
    let mut conv_out: Vec<f32> = vec![0.0; n_embd];
    for c_idx in 0..n_embd {
        let k_off = c_idx * l_cache;
        let b_off = c_idx * l_buf;
        let mut acc = 0.0f32;
        for k in 0..l_cache {
            acc += bx_buf[b_off + k] * kernel[k_off + k];
        }
        conv_out[c_idx] = acc;
    }

    vec_mul_inplace(c, conv_out.as_mut_slice());

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
                input, q8, sc, Some(q8k), o, n_embd, n_embd, ith, nth,
            );
        }
    });

    let _ = pos;
    (out, bx)
}