// Standalone resident-forward benchmark; build/run commands are in README.md.
use rust_model_inference::compute::ComputePolicy;
use rust_model_inference::models::qwen35::Qwen35Session;
use rust_model_inference::{
    open_model_source, qwen_text_positions, BPETokenizer, ComponentRole, ComputePool,
    EncodeOptions, Qwen35Model, TensorSource,
};
use serde_json::json;
use std::{path::Path, sync::Arc, time::Instant};

const PROMPT_TOKENS: usize = 32;
const DECODE: usize = 64;
const THREADS: usize = 8;
const BATCH: usize = 16;

fn greedy(logits: &[f32]) -> u32 {
    assert!(!logits.is_empty() && logits.iter().all(|x| x.is_finite()));
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .unwrap()
        .0 as u32
}

fn measure(
    session: &mut Qwen35Session<'_, '_>,
    tokens: &[u32],
    history: &[u32],
) -> Result<(f64, f64, Vec<f32>), String> {
    assert_eq!(tokens.len(), PROMPT_TOKENS);
    assert_eq!(history.len(), DECODE);
    session.reset();
    let positions = qwen_text_positions(tokens.len());
    let start = Instant::now();
    let first = session.step_with_tokens(tokens, &positions)?;
    let pp = start.elapsed().as_secs_f64();
    let mut snapshot = Vec::new();
    let dump = std::env::var_os("RMI_SPEED_SNAPSHOT");
    if dump.is_some() {
        snapshot.extend(first.iter().flat_map(|v| v.to_le_bytes()));
    }
    let start = Instant::now();
    let mut logits = Vec::new();
    for &token in history {
        let p = session.next_position();
        logits = session.step_with_tokens(&[token], &[[p, p, p, 0]])?;
        if dump.is_some() {
            snapshot.extend(logits.iter().flat_map(|v| v.to_le_bytes()));
        }
    }
    let tg = start.elapsed().as_secs_f64();
    if let Some(path) = dump {
        std::fs::write(path, snapshot).map_err(|e| e.to_string())?;
    }
    assert_eq!(session.next_position(), PROMPT_TOKENS + DECODE);
    assert!(logits.iter().all(|x| x.is_finite()));
    Ok((pp, tg, logits))
}

fn run() -> Result<(), String> {
    assert!(std::env::var_os("RMI_PARITY_TRACE").is_none());
    let rounds: usize = std::env::var("RMI_SPEED_ROUNDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    let path = std::env::args().nth(1).ok_or("model path required")?;
    let source: Arc<dyn TensorSource> = Arc::from(
        open_model_source(Path::new(&path), ComponentRole::Llm).map_err(|e| e.to_string())?,
    );
    let tokenizer = BPETokenizer::from_gguf_metadata(|k| source.metadata(k).cloned())?;
    let seed = tokenizer.encode(
        "The quick brown fox jumps over the lazy dog. ",
        EncodeOptions {
            add_special: false,
            parse_special: false,
        },
    );
    assert!(!seed.is_empty());
    let tokens: Vec<_> = seed.iter().copied().cycle().take(PROMPT_TOKENS).collect();
    let mut cpu_model = Qwen35Model::from_source(source.as_ref())?;
    let mut gpu_model = Qwen35Model::from_source(source.as_ref())?;
    let pool = Arc::new(ComputePool::new(THREADS));
    let mut cpu = Qwen35Session::new_with_compute(
        &mut cpu_model,
        512,
        BATCH,
        pool.clone(),
        ComputePolicy::Cpu,
    )?;
    rust_model_inference::ops::enable_gpu();
    let context = rust_model_inference::ops::get_vulkan_context().ok_or("no Vulkan context")?;
    let before_init = context.compute_stats();
    let init = Instant::now();
    let mut gpu =
        Qwen35Session::new_with_compute(&mut gpu_model, 512, BATCH, pool, ComputePolicy::Vulkan)?;
    println!(
        "{}",
        json!({"kind":"config", "device": context.device_name(), "model":path,
        "prompt_tokens":tokens, "decode_evals":DECODE, "threads":THREADS, "batch":BATCH,
        "capacity":512, "kv":"f32", "rounds":rounds, "eos_id":tokenizer.eos_id(),
        "im_end_id":tokenizer.special_token_id("im_end"), "gpu_session_init_seconds":init.elapsed().as_secs_f64(),
        "gpu_static_upload_bytes":context.compute_stats().since(before_init).static_upload_bytes,
        "timing_scope":"resident forward only; reset, loading and upload excluded; teacher forced CPU history"})
    );
    let mut logits = cpu.step_with_tokens(&tokens, &qwen_text_positions(tokens.len()))?;
    let mut history = Vec::new();
    for _ in 0..DECODE {
        let id = greedy(&logits);
        history.push(id);
        let p = cpu.next_position();
        logits = cpu.step_with_tokens(&[id], &[[p, p, p, 0]])?;
    }
    let cpu_warm = logits;
    let before = context.compute_stats();
    let (_, _, gpu_warm) = measure(&mut gpu, &tokens, &history)?;
    assert!(context.compute_stats().since(before).submissions > 0);
    let l2 = gpu_warm
        .iter()
        .zip(&cpu_warm)
        .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
        .sum::<f64>()
        .sqrt()
        / cpu_warm
            .iter()
            .map(|x| (*x as f64).powi(2))
            .sum::<f64>()
            .sqrt();
    println!(
        "{}",
        json!({"kind":"warmup", "history":history, "warmup_final_logit_relative_l2":l2,
        "warmup_final_argmax_equal":greedy(&cpu_warm)==greedy(&gpu_warm), "precision_pass_claimed":false})
    );
    let mut cpu_baseline: Option<Vec<u32>> = None;
    let mut gpu_baseline: Option<Vec<u32>> = None;
    for round in 0..rounds {
        let order = if round % 2 == 0 {
            [false, true]
        } else {
            [true, false]
        };
        for (order_index, use_gpu) in order.into_iter().enumerate() {
            let before = context.compute_stats();
            let (pp, tg, logits) =
                measure(if use_gpu { &mut gpu } else { &mut cpu }, &tokens, &history)?;
            let stats = context.compute_stats().since(before);
            assert_eq!(
                stats.static_uploads, 0,
                "weights uploaded during measurement"
            );
            assert_eq!(stats.static_upload_bytes, 0);
            assert_eq!(stats.submissions > 0, use_gpu, "wrong backend/fallback");
            let bits: Vec<_> = logits.iter().map(|x| x.to_bits()).collect();
            let baseline = if use_gpu {
                &mut gpu_baseline
            } else {
                &mut cpu_baseline
            };
            if let Some(previous) = baseline {
                assert_eq!(*previous, bits, "reset/repeat changed logits");
            } else {
                *baseline = Some(bits);
            }
            println!(
                "{}",
                json!({"kind":"sample", "round":round+1, "order":order_index+1,
                "backend":if use_gpu {"vulkan"} else {"cpu"}, "prefill_seconds":pp,"decode_seconds":tg,
                "pp_tok_s": PROMPT_TOKENS as f64/pp, "tg_tok_s": DECODE as f64/tg,
                "resident_total_seconds":pp+tg,"first_token_seconds":pp,"submissions":stats.submissions,
                "transfer_submissions":stats.transfer_submissions,"static_uploads":stats.static_uploads,
                "static_upload_bytes":stats.static_upload_bytes,"host_write_bytes":stats.host_write_bytes,
                "host_read_bytes":stats.host_read_bytes,"finite":true,"repeat_exact":true})
            );
        }
    }
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
