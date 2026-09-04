#[cfg(feature = "vulkan")]
use rust_model_inference::{
    build_simple_prompt, open_model_source, qwen_text_positions, BPETokenizer, ComponentRole,
    ComputePool, Qwen3GenerateOptions, Qwen3Generation, Qwen3Input, Qwen3Model, Qwen3Session,
    TensorSource,
};
#[cfg(feature = "vulkan")]
use std::path::PathBuf;
#[cfg(feature = "vulkan")]
use std::sync::Arc;
#[cfg(feature = "vulkan")]
use std::time::Duration;

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
) -> Result<Qwen3Generation, String> {
    session.generate(
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
    )
}

#[cfg(feature = "vulkan")]
#[derive(Clone, Copy)]
struct BenchmarkSample {
    prompt: f64,
    decode: f64,
}

#[cfg(feature = "vulkan")]
fn per_second(tokens: usize, elapsed: Duration) -> f64 {
    tokens as f64 / elapsed.as_secs_f64()
}

#[cfg(feature = "vulkan")]
fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[sorted.len() / 2]
}

#[cfg(feature = "vulkan")]
fn benchmark_once(
    session: &mut Qwen3Session<'_>,
    token_ids: &[u32],
    positions: &[[usize; 4]],
) -> Result<BenchmarkSample, String> {
    session.reset_kv();
    let generation = generate(session, token_ids, positions, GREEDY_TOKENS + 1)?;
    generation
        .token_ids
        .get(..GREEDY_TOKENS)
        .ok_or("benchmark stopped before 32 greedy tokens")?;
    Ok(BenchmarkSample {
        prompt: per_second(token_ids.len(), generation.prompt_duration),
        decode: per_second(GREEDY_TOKENS, generation.decode_duration),
    })
}

#[cfg(feature = "vulkan")]
fn benchmark(
    cpu: &mut Qwen3Session<'_>,
    gpu: &mut Qwen3Session<'_>,
    token_ids: &[u32],
    positions: &[[usize; 4]],
) -> Result<(), String> {
    benchmark_once(cpu, token_ids, positions)?;
    benchmark_once(gpu, token_ids, positions)?;
    println!("benchmark warmup=complete backends=cpu,gpu");

    let mut cpu_samples = Vec::with_capacity(5);
    let mut gpu_samples = Vec::with_capacity(5);
    for sample in 1..=5 {
        let cpu_sample = benchmark_once(cpu, token_ids, positions)?;
        println!(
            "benchmark sample={sample} backend=cpu prompt_tps={:.3} decode_tps={:.3}",
            cpu_sample.prompt, cpu_sample.decode
        );
        cpu_samples.push(cpu_sample);

        let gpu_sample = benchmark_once(gpu, token_ids, positions)?;
        println!(
            "benchmark sample={sample} backend=gpu prompt_tps={:.3} decode_tps={:.3}",
            gpu_sample.prompt, gpu_sample.decode
        );
        gpu_samples.push(gpu_sample);
    }

    let cpu_prompt = median(
        &cpu_samples
            .iter()
            .map(|sample| sample.prompt)
            .collect::<Vec<_>>(),
    );
    let cpu_decode = median(
        &cpu_samples
            .iter()
            .map(|sample| sample.decode)
            .collect::<Vec<_>>(),
    );
    let gpu_prompt = median(
        &gpu_samples
            .iter()
            .map(|sample| sample.prompt)
            .collect::<Vec<_>>(),
    );
    let gpu_decode = median(
        &gpu_samples
            .iter()
            .map(|sample| sample.decode)
            .collect::<Vec<_>>(),
    );
    let prompt_speedup = gpu_prompt / cpu_prompt;
    let decode_speedup = gpu_decode / cpu_decode;
    println!("benchmark median backend=cpu prompt_tps={cpu_prompt:.3} decode_tps={cpu_decode:.3}");
    println!("benchmark median backend=gpu prompt_tps={gpu_prompt:.3} decode_tps={gpu_decode:.3}");
    println!(
        "benchmark prompt_speedup={prompt_speedup:.3} decode_speedup={decode_speedup:.3} acceleration={}",
        prompt_speedup > 1.0 && decode_speedup > 1.0
    );
    Ok(())
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
    let cpu_tokens = generate(&mut cpu, &prompt_tokens, &positions, GREEDY_TOKENS + 1)?.token_ids;

    rust_model_inference::ops::enable_gpu();
    let context = rust_model_inference::ops::get_vulkan_context()
        .ok_or("Vulkan backend did not initialize")?;
    let mut gpu = Qwen3Session::new(&model, capacity)?;
    generate(&mut gpu, &prompt_tokens, &positions, 1)?;
    let gpu_logits = gpu.last_logits().to_vec();
    gpu.reset_kv();
    let before = context.submission_count();
    let gpu_tokens = generate(&mut gpu, &prompt_tokens, &positions, GREEDY_TOKENS + 1)?.token_ids;
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
        benchmark(&mut cpu, &mut gpu, &prompt_tokens, &positions)?;
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

#[cfg(all(test, feature = "vulkan"))]
mod tests {
    use super::{median, per_second};
    use std::time::Duration;

    #[test]
    fn benchmark_statistics_use_sorted_middle_and_elapsed_seconds() {
        assert_eq!(median(&[9.0, 1.0, 5.0, 3.0, 7.0]), 5.0);
        assert_eq!(per_second(8, Duration::from_millis(500)), 16.0);
    }
}
