//! CLI inference entry for Xing4.0.
//!
//! Mirrors the llama trunk's loop shape: prefill through the runtime
//! one token at a time, then decode with the shared llama-family
//! sampler, so CLI output matches what an HTTP adapter on the same
//! prompt would produce.

use super::config::ARCH;
use super::forward::Xing4Runtime;
use crate::app::cli::{inference_step_budget, resolve_thread_count, KvFormat};
use crate::core::tensor::TensorSource;
use crate::core::tokenizer::{load_tokenizer, EncodeOptions};
use std::io::Write;

/// Build the token sequence for one user turn. Xing4.0's chat template
/// is TeleChat-style: `<_system>` prefix, `<_user>` user turns,
/// `<_bot>` assistant turns, `<_end>` terminators.
pub fn build_xing4_prompt(system: Option<&str>, user: &str, thinking: bool) -> String {
    let mut out = String::new();
    match system {
        Some(text) if !text.is_empty() => {
            out.push_str("<_system>");
            out.push_str(text);
        }
        _ => out.push_str("<_system>"),
    }
    out.push_str("<_user>");
    out.push_str(user);
    out.push('\n');
    out.push_str("<_bot>");
    if thinking {
        out.push_str("<think>\n");
    } else {
        out.push_str("’}");
    }
    out
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
    max_context: usize,
    repetition_penalty: f32,
    thinking: bool,
) -> Result<(), String> {
    let _ = kv_format;
    let cfg = super::config::Xing4Config::from_source(source)?;
    let available = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let n_threads = resolve_thread_count(n_threads_arg, available);
    let tokenizer = load_tokenizer(|k| source.metadata(k).cloned())
        .map_err(|error| format!("Failed to initialize tokenizer: {error}"))?;

    let max_ctx = max_context.min(cfg.n_ctx).max(1);
    let mut runtime = Xing4Runtime::new(source, cfg, n_threads, max_ctx)?;
    let cfg = runtime.cfg();
    let vocab = cfg.n_vocab;

    let prompt_text = build_xing4_prompt(None, prompt, thinking);
    eprintln!("[RUST_PROMPT_TEXT] {prompt_text}");
    let input_tokens = tokenizer.encode(
        &prompt_text,
        EncodeOptions {
            add_special: true,
            parse_special: true,
        },
    );
    let bos = tokenizer.bos_id();
    let mut ids = input_tokens;
    if ids.first().copied() != bos {
        if let Some(bos) = bos {
            ids.insert(0, bos);
        }
    }
    eprintln!("[RUST_TOKENS] n={} ids={:?}", ids.len(), ids);

    if ids.len() >= max_ctx {
        return Err(format!(
            "prompt is {} tokens but the context holds {max_ctx}",
            ids.len()
        ));
    }

    let mut sampler = crate::ops::sampling::LlamaSampler::new(
        sample_defaults(source).0,
        sample_defaults(source).1,
    );
    sampler.prime(&ids);
    let mut decoder = crate::core::tokenizer::StreamingDecoder::new(&*tokenizer, false);
    let mut scratch = runtime.scratch();
    let mut hc_state = vec![0.0f32; runtime.cfg().hc_count * runtime.cfg().n_embd];

    print!("Output: ");
    std::io::stdout().flush().map_err(|e| e.to_string())?;

    let t_infer = std::time::Instant::now();
    let total_steps = inference_step_budget(ids.len(), max_tokens, bench);
    let mut generated: Vec<u32> = Vec::new();
    let eos_id = tokenizer.eos_id();

    for step in 0..total_steps {
        let token_id = if step < ids.len() {
            ids[step]
        } else {
            *generated.last().unwrap_or(&0)
        };
        let pos = step;
        runtime.init_hc(token_id, &mut hc_state);
        runtime.forward_token(pos, &mut hc_state, &mut scratch);
        let logits = runtime.logits_from_hc(&hc_state, &mut scratch);
        if pos < ids.len() - 1 {
            continue;
        }
        let mut logits = logits;
        if std::env::var_os("RUST_XING4_DEBUG").is_some() {
            let mut idxs: Vec<(usize, f32)> =
                logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
            idxs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let top_ids: Vec<u32> = idxs[..8].iter().map(|&(i, _)| i as u32).collect();
            let decoded = tokenizer.decode(&top_ids, false);
            let top: Vec<String> = idxs[..8]
                .iter()
                .map(|&(i, v)| format!("{i}:{v:.3}"))
                .collect();
            eprintln!("[xing4] decoded top8 = {decoded:?}");
            let sum: f64 = logits.iter().map(|&v| v as f64).sum();
            let sqsum: f64 = logits.iter().map(|&v| (v as f64) * (v as f64)).sum();
            eprintln!(
                "[xing4] pos={pos} logits sum={sum:.3} sq={sqsum:.2} top8={}",
                top.join(" ")
            );
        }
        let chosen_id = sampler.sample(&mut logits, temperature, repetition_penalty);
        if stop_after(chosen_id, generated.len(), max_tokens, bench, eos_id) {
            break;
        }
        generated.push(chosen_id);
        let text = decoder.push(chosen_id);
        print!("{text}");
        std::io::stdout().flush().map_err(|e| e.to_string())?;
    }

    let tail = decoder.finish();
    if !tail.is_empty() {
        print!("{tail}");
    }
    let infer_ms = t_infer.elapsed().as_millis();
    let tok_s = if infer_ms > 0 {
        generated.len() as f64 / infer_ms as f64 * 1000.0
    } else {
        0.0
    };
    let _ = profile;
    eprintln!("\nPrompt: n/a t/s | Generation: {tok_s:.1} t/s | vocab={vocab} arch={ARCH}");
    Ok(())
}

fn sample_defaults(source: &dyn TensorSource) -> (usize, f32) {
    let top_k = source
        .metadata("general.sampling.top_k")
        .and_then(|v| v.to_u64())
        .unwrap_or(40) as usize;
    let top_p = source
        .metadata("general.sampling.top_p")
        .and_then(|v| v.to_f64())
        .unwrap_or(0.95) as f32;
    (top_k, top_p)
}

fn stop_after(
    chosen: u32,
    generated: usize,
    max_tokens: usize,
    bench: bool,
    eos_id: Option<u32>,
) -> bool {
    if !bench && Some(chosen) == eos_id {
        return true;
    }
    generated >= max_tokens
}
