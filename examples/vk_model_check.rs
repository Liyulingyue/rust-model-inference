#[cfg(feature = "vulkan")]
use rust_model_inference::{
    build_simple_prompt, open_model_source, qwen_text_positions, BPETokenizer, ComponentRole,
    ComputePool, Qwen3GenerateOptions, Qwen3Input, Qwen3Model, Qwen3Session, TensorSource,
};
#[cfg(feature = "vulkan")]
use std::path::PathBuf;
#[cfg(feature = "vulkan")]
use std::sync::Arc;

#[cfg(feature = "vulkan")]
const PROMPT: &str = "法国的首都是";
#[cfg(feature = "vulkan")]
const GREEDY_TOKENS: usize = 32;
#[cfg(feature = "vulkan")]
const LOGIT_ABS: f32 = 2e-3;
#[cfg(feature = "vulkan")]
const LOGIT_REL: f32 = 2e-3;

#[cfg(feature = "vulkan")]
struct Arguments {
    model: PathBuf,
    benchmark: bool,
}

#[cfg(feature = "vulkan")]
fn arguments() -> Result<Arguments, String> {
    let mut args = std::env::args().skip(1);
    if args.next().as_deref() != Some("qwen3") {
        return Err("usage: vk_model_check qwen3 --model PATH [--benchmark]".into());
    }
    let mut model = None;
    let mut benchmark = false;
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--model" => model = Some(PathBuf::from(args.next().ok_or("--model needs a path")?)),
            "--benchmark" => benchmark = true,
            _ => return Err(format!("unknown argument: {argument}")),
        }
    }
    Ok(Arguments {
        model: model.ok_or("--model is required")?,
        benchmark,
    })
}

#[cfg(feature = "vulkan")]
fn generate(
    session: &mut Qwen3Session<'_>,
    token_ids: &[u32],
    positions: &[[usize; 4]],
    max_new_tokens: usize,
) -> Result<Vec<u32>, String> {
    Ok(session
        .generate(
            Qwen3Input {
                token_ids,
                positions,
                embeddings: None,
                deepstack_embeddings: None,
            },
            Qwen3GenerateOptions {
                max_new_tokens,
                temperature: 0.0,
            },
        )?
        .token_ids)
}

#[cfg(feature = "vulkan")]
fn assert_close(name: &str, gpu: &[f32], cpu: &[f32]) -> Result<(), String> {
    if gpu.len() != cpu.len() {
        return Err(format!(
            "{name} length mismatch: gpu={} cpu={}",
            gpu.len(),
            cpu.len()
        ));
    }
    let mut max_absolute = 0.0f32;
    let mut max_relative = 0.0f32;
    for (index, (&gpu, &cpu)) in gpu.iter().zip(cpu).enumerate() {
        let absolute = (gpu - cpu).abs();
        let relative = absolute / cpu.abs().max(f32::MIN_POSITIVE);
        max_absolute = max_absolute.max(absolute);
        max_relative = max_relative.max(relative);
        if !gpu.is_finite() || absolute > LOGIT_ABS + LOGIT_REL * cpu.abs() {
            return Err(format!(
                "{name} mismatch at {index}: gpu={gpu} cpu={cpu} abs={absolute} rel={relative}"
            ));
        }
    }
    println!("check={name} max_abs={max_absolute:.3e} max_rel={max_relative:.3e}");
    Ok(())
}

#[cfg(feature = "vulkan")]
fn run() -> Result<(), String> {
    let arguments = arguments()?;
    let source: Arc<dyn TensorSource> = Arc::from(
        open_model_source(&arguments.model, ComponentRole::Llm)
            .map_err(|error| error.to_string())?,
    );
    let tokenizer = Arc::new(BPETokenizer::from_gguf_metadata(|key| {
        source.metadata(key).cloned()
    })?);
    let model = Qwen3Model::from_source(
        Arc::clone(&source),
        Arc::clone(&tokenizer),
        Arc::new(ComputePool::new(4)),
    )?;
    let prompt_tokens = build_simple_prompt(&tokenizer, PROMPT);
    let positions = qwen_text_positions(prompt_tokens.len());
    let capacity = prompt_tokens
        .len()
        .checked_add(GREEDY_TOKENS + 1)
        .ok_or("session capacity overflow")?;

    let mut cpu = Qwen3Session::new(&model, capacity)?;
    generate(&mut cpu, &prompt_tokens, &positions, 1)?;
    let cpu_logits = cpu.last_logits().to_vec();
    cpu.reset_kv();
    let cpu_tokens = generate(&mut cpu, &prompt_tokens, &positions, GREEDY_TOKENS + 1)?;

    rust_model_inference::ops::enable_gpu();
    let context = rust_model_inference::ops::get_vulkan_context()
        .ok_or("Vulkan backend did not initialize")?;
    let mut gpu = Qwen3Session::new(&model, capacity)?;
    generate(&mut gpu, &prompt_tokens, &positions, 1)?;
    let gpu_logits = gpu.last_logits().to_vec();
    gpu.reset_kv();
    let before = context.submission_count();
    let gpu_tokens = generate(&mut gpu, &prompt_tokens, &positions, GREEDY_TOKENS + 1)?;
    let submissions = context.submission_count() - before;

    assert_close("prefill_logits", &gpu_logits, &cpu_logits)?;
    let cpu_tokens = cpu_tokens
        .get(..GREEDY_TOKENS)
        .ok_or("CPU stopped before 32 greedy tokens")?;
    let gpu_tokens = gpu_tokens
        .get(..GREEDY_TOKENS)
        .ok_or("Vulkan stopped before 32 greedy tokens")?;
    if gpu_tokens != cpu_tokens {
        return Err(format!(
            "greedy token mismatch: gpu={gpu_tokens:?} cpu={cpu_tokens:?}"
        ));
    }
    let expected_submissions = prompt_tokens.len() + GREEDY_TOKENS;
    if submissions != expected_submissions as u64 {
        return Err(format!(
            "expected one submission per token ({expected_submissions}), got {submissions}"
        ));
    }
    println!(
        "device={} prompt_tokens={} greedy_tokens={} submissions={submissions}",
        context.device_name(),
        prompt_tokens.len(),
        GREEDY_TOKENS
    );
    println!("tokens={gpu_tokens:?}");
    if arguments.benchmark {
        return Err("benchmark mode is not implemented yet".into());
    }
    Ok(())
}

#[cfg(feature = "vulkan")]
fn main() -> std::process::ExitCode {
    match run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("vk_model_check failed: {error}");
            std::process::ExitCode::FAILURE
        }
    }
}

#[cfg(not(feature = "vulkan"))]
fn main() {
    eprintln!("vk_model_check requires --features vulkan");
}
