//! Oracle companion for the Hy-MT3 models; see `tools/oracle/hy_mt3/README.md`.
//!
//! Dumps exactly the artifacts `tools/oracle/hy_mt3/oracle.cpp` produces, so
//! `tools/oracle/hy_mt3/compare.py` can diff the two directories without caring
//! which side wrote them:
//!
//! * `prompt_tokens.json` — the IDs our tokenizer produced
//! * `logits_prompt.f32` — `n_vocab` f32 logits at the last prompt position
//! * `logits_all.bin` — `LLOG` header + `n_gen * n_vocab` f32 logits per step
//! * `generated_tokens.json` — greedy argmax tokens
//! * `text.txt` — the decoded generated text
//!
//! Two model families share the name: `hy_v3` (30B-A3B, `HySession`) and
//! `hunyuan-dense` (1.8B / 7B, the qwen3 trunk). `--mode` picks the decoder; the
//! artifact format is identical so the same comparison runs on both.
//!
//! ```text
//! cargo run --release --example hy_mt3_parity -- \
//!   <model.gguf> <prompt-file> <out-dir> <v3|dense> [n_gen] [threads] [max_ctx]
//! ```
use rust_model_inference::compute::ComputePolicy;
use rust_model_inference::core::{
    loader::GGUFLoader,
    scratchpad::{KvFormat, KvLifecycle},
    tensor::TensorSource,
    thread_pool::ComputePool,
    tokenizer::{BPETokenizer, EncodeOptions},
};
use std::io::Write;
use std::sync::Arc;

fn write_f32(path: &str, values: &[f32]) -> Result<(), Box<dyn std::error::Error>> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    std::fs::write(path, bytes)?;
    Ok(())
}

fn argmax(values: &[f32]) -> usize {
    let mut best = 0usize;
    for (index, value) in values.iter().enumerate() {
        if *value > values[best] {
            best = index;
        }
    }
    best
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 {
        return Err(
            "usage: hy_mt3_parity <model.gguf> <prompt-file> <out-dir> <v3|dense> \
             [n_gen] [threads] [max_ctx]"
                .into(),
        );
    }
    let model_path = &args[1];
    let prompt_path = &args[2];
    let out_dir = &args[3];
    let mode = &args[4];
    let n_gen: usize = args.get(5).map(|v| v.parse()).transpose()?.unwrap_or(32);
    let threads: usize = args.get(6).map(|v| v.parse()).transpose()?.unwrap_or(8);
    let max_ctx: usize = args.get(7).map(|v| v.parse()).transpose()?.unwrap_or(2048);
    std::fs::create_dir_all(out_dir)?;

    let source: Arc<dyn TensorSource> = Arc::new(GGUFLoader::from_file(model_path)?);
    let tokenizer = Arc::new(BPETokenizer::from_gguf_metadata(|k| {
        source.metadata(k).cloned()
    })?);
    let text = std::fs::read_to_string(prompt_path)?;
    let prompt_ids = tokenizer.encode(
        &text,
        EncodeOptions {
            add_special: false,
            parse_special: true,
        },
    );
    std::fs::write(
        format!("{out_dir}/prompt_tokens.json"),
        serde_json::to_vec_pretty(&serde_json::json!({ "ids": prompt_ids }))?,
    )?;
    eprintln!(
        "hy_mt3_parity: mode={mode} prompt={} tokens n_gen={n_gen} threads={threads}",
        prompt_ids.len()
    );

    // Same knob as `hy_simulmt_parity`: the device comparison needs the two
    // runs to differ only in the compute policy.
    let compute = match std::env::var("HY_COMPUTE").as_deref().unwrap_or("cpu") {
        "cpu" => ComputePolicy::Cpu,
        "auto" => ComputePolicy::Auto,
        "vulkan" => ComputePolicy::Vulkan,
        other => {
            return Err(format!("HY_COMPUTE must be cpu, auto or vulkan (got {other:?})").into())
        }
    };
    let kv_format = match std::env::var("HY_KV").as_deref().unwrap_or("f32") {
        "f32" => KvFormat::F32,
        "f16" => KvFormat::F16,
        other => return Err(format!("HY_KV must be f32 or f16 (got {other:?})").into()),
    };
    let _scope = compute.enter_legacy_scope();
    let pool = Arc::new(ComputePool::new(threads));
    let mut all: Vec<f32> = Vec::new();
    let mut generated: Vec<u32> = Vec::new();
    let mut n_vocab = 0usize;

    // Greedy argmax loop shared by both decoders. `forward` maps a token slice
    // to the logits of its last position, so the loop never needs to know
    // which session type it is driving.
    macro_rules! run {
        ($forward:expr) => {{
            let mut forward = $forward;
            let mut logits = forward(&prompt_ids[..])?;
            n_vocab = logits.len();
            for step in 0..n_gen {
                all.extend_from_slice(&logits);
                let id = argmax(&logits) as u32;
                generated.push(id);
                if step + 1 == n_gen {
                    break;
                }
                logits = forward(&[id])?;
            }
        }};
    }

    if mode == "dense" {
        let model = rust_model_inference::models::qwen3::Qwen3Model::from_source(
            source.clone(),
            tokenizer.clone(),
            pool.clone(),
        )?;
        let mut session = rust_model_inference::models::qwen3::Qwen3Session::new_with_kv_state(
            &model,
            max_ctx,
            kv_format,
            KvLifecycle::Ephemeral,
        )?;
        let mut position = prompt_ids.len();
        run!(|ids: &[u32]| -> Result<Vec<f32>, String> {
            let positions: Vec<[usize; 4]> = (0..ids.len())
                .map(|i| {
                    let p = position + i;
                    [p, p, p, 0]
                })
                .collect();
            position += ids.len();
            session
                .forward_logits(
                    rust_model_inference::models::qwen3::Qwen3Input {
                        token_ids: ids,
                        positions: &positions,
                        embeddings: None,
                        deepstack_embeddings: None,
                    },
                    256,
                )
                .map(|(values, _)| values)
        });
    } else if mode == "v3" {
        let decoder = rust_model_inference::models::hy_simulmt::HyDecoder::from_source(
            source.clone(),
            tokenizer.clone(),
            pool.clone(),
        )?;
        let mut session = rust_model_inference::models::hy_simulmt::HySession::new(
            &decoder, max_ctx, kv_format, compute,
        )?;
        run!(|ids: &[u32]| -> Result<Vec<f32>, String> { session.forward(ids, None, 256) });
    } else {
        return Err(format!("unknown mode {mode:?}; expected v3 or dense").into());
    }

    write_f32(&format!("{out_dir}/logits_prompt.f32"), &all[..n_vocab])?;
    {
        let mut bytes = Vec::with_capacity(16 + all.len() * 4);
        bytes.extend_from_slice(b"LLOG");
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&(n_vocab as u32).to_le_bytes());
        bytes.extend_from_slice(&(n_gen as u32).to_le_bytes());
        for value in &all {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        std::fs::write(format!("{out_dir}/logits_all.bin"), bytes)?;
    }
    std::fs::write(
        format!("{out_dir}/generated_tokens.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "n_generated": generated.len(),
            "generated_tokens": generated,
        }))?,
    )?;
    let mut text_out = String::new();
    for &id in &generated {
        text_out.push_str(&String::from_utf8_lossy(
            &tokenizer.decode_bytes(&[id], false),
        ));
    }
    std::fs::write(format!("{out_dir}/text.txt"), &text_out)?;
    let mut stdout = std::io::stdout();
    writeln!(stdout, "generated_ids: {generated:?}")?;
    eprintln!("hy_mt3_parity: wrote {out_dir} (n_vocab={n_vocab})");
    Ok(())
}
