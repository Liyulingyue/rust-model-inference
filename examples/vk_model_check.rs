#[cfg(feature = "vulkan")]
use rust_model_inference::models::qwen35::Qwen35Session;
#[cfg(feature = "vulkan")]
use rust_model_inference::{
    build_simple_prompt, open_model_source, qwen_text_positions, BPETokenizer, ComponentRole,
    ComputePool, EncodeOptions, GGMLType, Qwen35Model, Qwen3GenerateOptions, Qwen3Generation,
    Qwen3Input, Qwen3Model, Qwen3Session, TensorSource,
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
    mode: Mode,
    model: PathBuf,
    benchmark: bool,
    cpu_benchmark: bool,
    compare_prefill_batches: Option<Vec<usize>>,
    prompt_repeats: usize,
}

#[cfg(feature = "vulkan")]
#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Qwen3,
    Llama,
    Qwen35,
    Embedding,
}

#[cfg(feature = "vulkan")]
fn arguments() -> Result<Arguments, String> {
    let mut args = std::env::args().skip(1);
    let mode = match args.next().as_deref() {
        Some("llama") => Mode::Llama,
        Some("qwen3") => Mode::Qwen3,
        Some("qwen35") => Mode::Qwen35,
        Some("embedding") => Mode::Embedding,
        _ => {
            return Err(
                "usage: vk_model_check <llama|qwen3|qwen35|embedding> --model PATH [--benchmark | --cpu-benchmark (qwen3)] [--compare-prefill-batches 1,64]".into(),
            )
        }
    };
    let mut model = None;
    let mut benchmark = false;
    let mut cpu_benchmark = false;
    let mut compare_prefill_batches = None;
    let mut prompt_repeats = 1;
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--cpu-benchmark" if mode == Mode::Qwen3 => cpu_benchmark = true,
            "--model" => model = Some(PathBuf::from(args.next().ok_or("--model needs a path")?)),
            "--benchmark" if matches!(mode, Mode::Qwen3 | Mode::Llama) => benchmark = true,
            "--prompt-repeats" if mode == Mode::Qwen3 => {
                prompt_repeats = args
                    .next()
                    .ok_or("--prompt-repeats needs a positive integer")?
                    .parse::<usize>()
                    .ok()
                    .filter(|&count| (1..=256).contains(&count))
                    .ok_or("--prompt-repeats must be in 1..=256")?;
            }
            "--compare-prefill-batches" if mode != Mode::Embedding => {
                compare_prefill_batches =
                    Some(parse_prefill_batches(&args.next().ok_or(
                        "--compare-prefill-batches needs comma-separated sizes",
                    )?)?);
            }
            _ => return Err(format!("unknown argument: {argument}")),
        }
    }
    Ok(Arguments {
        mode,
        model: model.ok_or("--model is required")?,
        benchmark,
        cpu_benchmark,
        compare_prefill_batches,
        prompt_repeats,
    })
}

#[cfg(feature = "vulkan")]
fn parse_prefill_batches(value: &str) -> Result<Vec<usize>, String> {
    let batches = value
        .split(',')
        .map(|value| {
            value
                .parse::<usize>()
                .map_err(|_| "invalid prefill batch size".to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    if batches.len() < 2 || batches.contains(&0) {
        return Err("prefill comparison requires at least two positive batch sizes".into());
    }
    Ok(batches)
}

#[cfg(feature = "vulkan")]
fn kv_bits(state: &rust_model_inference::core::scratchpad::KvState) -> (usize, Vec<u32>) {
    use rust_model_inference::core::scratchpad::KvCache;
    let stride = state.arch.n_head_kv * state.arch.n_embd_head_k.max(state.arch.n_embd_head_v);
    let mut words = Vec::new();
    for layer in 0..state.arch.n_layer {
        let start = layer * state.capacity * stride;
        let end = start + state.seq_len * stride;
        match &state.cache {
            KvCache::F16(cache) => words.extend(
                cache.k[start..end]
                    .iter()
                    .chain(&cache.v[start..end])
                    .map(|&word| word as u32),
            ),
            KvCache::F32(cache) => words.extend(
                cache.k[start..end]
                    .iter()
                    .chain(&cache.v[start..end])
                    .map(|word| word.to_bits()),
            ),
        }
    }
    (state.seq_len, words)
}

#[cfg(feature = "vulkan")]
fn compare_prefill_batches(
    model: &Qwen3Model,
    tokens: &[u32],
    positions: &[[usize; 4]],
    batches: &[usize],
    gpu: bool,
) -> Result<(), String> {
    let context = if gpu {
        rust_model_inference::ops::enable_gpu();
        Some(
            rust_model_inference::ops::get_vulkan_context()
                .ok_or("Vulkan backend did not initialize")?,
        )
    } else {
        None
    };
    let submissions = || {
        context
            .as_ref()
            .map_or(0, |context| context.submission_count())
    };
    let backend = if gpu { "vulkan" } else { "cpu" };
    let capacity = tokens
        .len()
        .checked_add(GREEDY_TOKENS + 1)
        .ok_or("session capacity overflow")?;
    let mut baseline = None;
    for &batch in batches {
        let mut session = Qwen3Session::new(model, capacity)?;
        let input = Qwen3Input {
            token_ids: tokens,
            positions,
            embeddings: None,
            deepstack_embeddings: None,
        };
        let before = submissions();
        session.generate(
            input.clone(),
            Qwen3GenerateOptions {
                max_new_tokens: 1,
                temperature: 0.0,
                prefill_batch_size: batch,
            },
        )?;
        let prefill_submissions = submissions() - before;
        let expected = if gpu {
            tokens.len().div_ceil(batch) as u64
        } else {
            0
        };
        if prefill_submissions != expected {
            return Err(format!(
                "batch={batch} expected {expected} prefill submissions, got {prefill_submissions}"
            ));
        }
        let logits: Vec<_> = session
            .last_logits()
            .iter()
            .map(|value| value.to_bits())
            .collect();
        let prompt_kv = kv_bits(session.kv_state());
        session.reset_kv();
        let before = submissions();
        let generation = session.generate(
            input,
            Qwen3GenerateOptions {
                max_new_tokens: GREEDY_TOKENS + 1,
                temperature: 0.0,
                prefill_batch_size: batch,
            },
        )?;
        let generated = generation
            .token_ids
            .get(..GREEDY_TOKENS)
            .ok_or("session stopped before 32 greedy tokens")?
            .to_vec();
        let total_submissions = submissions() - before;
        let expected_decode = if gpu { GREEDY_TOKENS as u64 } else { 0 };
        if generation.prompt_submissions != expected
            || generation.decode_submissions != expected_decode
        {
            return Err(format!(
                "batch={batch} phase submission counts changed: prompt={} decode={}",
                generation.prompt_submissions, generation.decode_submissions
            ));
        }
        if total_submissions != expected + expected_decode {
            return Err(format!(
                "batch={batch} decode submission count changed: {total_submissions}"
            ));
        }
        let result = (logits, prompt_kv, generated, kv_bits(session.kv_state()));
        if let Some(previous) = &baseline {
            if &result != previous {
                return Err(format!("same-{backend} prefill mismatch for batch={batch}: logits, KV or greedy tokens differ"));
            }
        } else {
            baseline = Some(result);
        }
        println!("backend={backend} batch={batch} prompt_tokens={} prefill_submissions={prefill_submissions} total_submissions={total_submissions} greedy_tokens={GREEDY_TOKENS}", tokens.len());
    }
    println!("check=same_{backend}_prefill exact_logits=true exact_prompt_kv=true exact_decode_kv=true exact_greedy_tokens=true");
    Ok(())
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
            prefill_batch_size: rust_model_inference::core::prefill::DEFAULT_PREFILL_BATCH_SIZE,
        },
    )
}

#[cfg(feature = "vulkan")]
#[derive(Clone, Copy)]
struct BenchmarkSample {
    prompt: f64,
    decode: f64,
    wall_ms: f64,
    stats: rust_model_inference::vulkan::ComputeStats,
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
    let context = rust_model_inference::ops::get_vulkan_context();
    let before = context.map(|c| c.compute_stats()).unwrap_or_default();
    let started = std::time::Instant::now();
    session.reset_kv();
    let generation = generate(session, token_ids, positions, GREEDY_TOKENS + 1)?;
    generation
        .token_ids
        .get(..GREEDY_TOKENS)
        .ok_or("benchmark stopped before 32 greedy tokens")?;
    Ok(BenchmarkSample {
        prompt: per_second(token_ids.len(), generation.prompt_duration),
        decode: per_second(GREEDY_TOKENS, generation.decode_duration),
        wall_ms: started.elapsed().as_secs_f64() * 1000.0,
        stats: context
            .map(|c| c.compute_stats().since(before))
            .unwrap_or_default(),
    })
}

#[cfg(feature = "vulkan")]
fn benchmark(
    cpu: &mut Qwen3Session<'_>,
    gpu: &mut Qwen3Session<'_>,
    token_ids: &[u32],
    positions: &[[usize; 4]],
    correctness: bool,
) -> Result<(), String> {
    paired_benchmark(correctness, |use_gpu| {
        if use_gpu {
            benchmark_once(gpu, token_ids, positions)
        } else {
            benchmark_once(cpu, token_ids, positions)
        }
    })
}

#[cfg(feature = "vulkan")]
fn benchmark_order(round: usize) -> [bool; 2] {
    if round % 2 == 0 {
        [false, true]
    } else {
        [true, false]
    }
}

#[cfg(feature = "vulkan")]
fn auto_gain_passes(cpu_wall: &[f64], gpu_wall: &[f64], correctness: bool) -> bool {
    correctness
        && cpu_wall.len() >= 5
        && gpu_wall.len() == cpu_wall.len()
        && cpu_wall
            .iter()
            .chain(gpu_wall)
            .all(|v| v.is_finite() && *v > 0.0)
        && median(gpu_wall) <= 0.9 * median(cpu_wall)
}

#[cfg(feature = "vulkan")]
fn paired_benchmark(
    correctness: bool,
    mut run: impl FnMut(bool) -> Result<BenchmarkSample, String>,
) -> Result<(), String> {
    run(false)?;
    run(true)?;
    println!("benchmark warmup=complete rounds=5 order=alternating correctness={correctness} scope=resident_decoder rss=external_process_peak");
    let mut samples = [Vec::new(), Vec::new()];
    for round in 0..5 {
        for gpu in benchmark_order(round) {
            let sample = run(gpu)?;
            let backend = if gpu { "vulkan" } else { "cpu" };
            println!("benchmark round={} backend={backend} prompt_tps={:.3} decode_tps={:.3} wall_ms={:.3} compute_submissions={} transfers={} uploads={} upload_bytes={} host_write_bytes={} host_read_bytes={} device_buffer_peak_bytes={}",
                round+1, sample.prompt, sample.decode, sample.wall_ms,
                sample.stats.submissions - sample.stats.transfer_submissions, sample.stats.transfer_submissions,
                sample.stats.static_uploads, sample.stats.static_upload_bytes, sample.stats.host_write_bytes,
                sample.stats.host_read_bytes, sample.stats.peak_allocation_bytes);
            if !gpu && sample.stats.submissions != 0 {
                return Err("CPU benchmark unexpectedly submitted Vulkan work".into());
            }
            samples[gpu as usize].push(sample);
        }
    }
    for (i, name) in ["cpu", "vulkan"].iter().enumerate() {
        let s = &samples[i];
        println!(
            "benchmark median backend={name} prompt_tps={:.3} decode_tps={:.3} wall_ms={:.3}",
            median(&s.iter().map(|s| s.prompt).collect::<Vec<_>>()),
            median(&s.iter().map(|s| s.decode).collect::<Vec<_>>()),
            median(&s.iter().map(|s| s.wall_ms).collect::<Vec<_>>())
        );
    }
    let cpu: Vec<_> = samples[0].iter().map(|s| s.wall_ms).collect();
    let gpu: Vec<_> = samples[1].iter().map(|s| s.wall_ms).collect();
    println!(
        "benchmark wall_improvement_percent={:.3} threshold_passed={} auto_rule_enabled=false",
        (1.0 - median(&gpu) / median(&cpu)) * 100.0,
        auto_gain_passes(&cpu, &gpu, correctness)
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
        if !gpu.is_finite() || !cpu.is_finite() || absolute > LOGIT_ABS + LOGIT_REL * cpu.abs() {
            return Err(format!(
                "{name} mismatch at {index}: gpu={gpu} cpu={cpu} abs={absolute} rel={relative}"
            ));
        }
    }
    println!("check={name} max_abs={max_absolute:.3e} max_rel={max_relative:.3e}");
    Ok(())
}

#[cfg(feature = "vulkan")]
fn load_model(
    arguments: &Arguments,
) -> Result<(Arc<dyn TensorSource>, Arc<BPETokenizer>, Qwen3Model), String> {
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
    Ok((source, tokenizer, model))
}

#[cfg(feature = "vulkan")]
fn format_summary(layer_formats: &[[GGMLType; 7]], output: GGMLType) -> Result<String, String> {
    if layer_formats.is_empty() {
        return Err("model has no layers".into());
    }
    let format_set = |operation: usize| {
        let mut formats = Vec::new();
        for layer in layer_formats {
            let format = layer[operation];
            if !formats.contains(&format) {
                formats.push(format);
            }
        }
        format!(
            "{{{}}}",
            formats
                .iter()
                .map(|format| format!("{format:?}"))
                .collect::<Vec<_>>()
                .join(",")
        )
    };
    Ok(format!(
        "formats=layers={};q={},k={},v={},o={},gate={},up={},down={};output={output:?}",
        layer_formats.len(),
        format_set(0),
        format_set(1),
        format_set(2),
        format_set(3),
        format_set(4),
        format_set(5),
        format_set(6),
    ))
}

#[cfg(feature = "vulkan")]
fn print_formats(model: &Qwen3Model, source: &dyn TensorSource) -> Result<(), String> {
    let layer_formats = model
        .layers()
        .iter()
        .map(|layer| {
            [
                layer.wq.ggml_type,
                layer.wk.ggml_type,
                layer.wv.ggml_type,
                layer.wo.ggml_type,
                layer.w_gate.ggml_type,
                layer.w_up.ggml_type,
                layer.w_down.ggml_type,
            ]
        })
        .collect::<Vec<_>>();
    let output = source
        .tensor_info("output.weight")
        .or_else(|| source.tensor_info("token_embd.weight"))
        .ok_or("missing output weight")?
        .ggml_type;
    println!("{}", format_summary(&layer_formats, output)?);
    Ok(())
}

#[cfg(feature = "vulkan")]
fn run_qwen3(arguments: &Arguments) -> Result<(), String> {
    let started = std::time::Instant::now();
    let (source, tokenizer, model) = load_model(arguments)?;
    println!(
        "cold model_load_ms={:.3}",
        started.elapsed().as_secs_f64() * 1000.0
    );
    print_formats(&model, source.as_ref())?;
    // Exercise a committed prefix and a short tail in the batch-64 comparison.
    let prompt = if arguments.compare_prefill_batches.is_some() {
        PROMPT.repeat(33)
    } else {
        PROMPT.repeat(arguments.prompt_repeats)
    };
    let prompt_tokens = build_simple_prompt(&tokenizer, &prompt);
    let positions = qwen_text_positions(prompt_tokens.len());
    if let Some(batches) = &arguments.compare_prefill_batches {
        compare_prefill_batches(&model, &prompt_tokens, &positions, batches, false)?;
        return compare_prefill_batches(&model, &prompt_tokens, &positions, batches, true);
    }
    let capacity = prompt_tokens
        .len()
        .checked_add(GREEDY_TOKENS + 1)
        .ok_or("session capacity overflow")?;

    let started = std::time::Instant::now();
    let mut cpu = Qwen3Session::new_with_compute(
        &model,
        capacity,
        rust_model_inference::core::scratchpad::KvFormat::F16,
        rust_model_inference::KvLifecycle::Ephemeral,
        rust_model_inference::compute::ComputePolicy::Cpu,
    )?;
    println!("cold backend=cpu session_init_ms={:.3} kv=f16 threads=4 batch=64 context={capacity} input_ids={prompt_tokens:?}", started.elapsed().as_secs_f64()*1000.0);
    if arguments.cpu_benchmark {
        benchmark_once(&mut cpu, &prompt_tokens, &positions)?;
        for round in 0..5 {
            let sample = benchmark_once(&mut cpu, &prompt_tokens, &positions)?;
            println!(
                "cpu_benchmark round={round} prompt_tps={:.3} decode_tps={:.3} wall_ms={:.3}",
                sample.prompt, sample.decode, sample.wall_ms
            );
        }
        return Ok(());
    }
    generate(&mut cpu, &prompt_tokens, &positions, 1)?;
    let cpu_logits = cpu.last_logits().to_vec();
    cpu.reset_kv();
    let cpu_tokens = generate(&mut cpu, &prompt_tokens, &positions, GREEDY_TOKENS + 1)?.token_ids;

    let started = std::time::Instant::now();
    rust_model_inference::ops::enable_gpu();
    let context = rust_model_inference::ops::get_vulkan_context()
        .ok_or("Vulkan backend did not initialize")?;
    println!(
        "cold context_init_ms={:.3}",
        started.elapsed().as_secs_f64() * 1000.0
    );
    let started = std::time::Instant::now();
    let before_upload = context.compute_stats();
    let mut gpu = Qwen3Session::new_with_compute(
        &model,
        capacity,
        rust_model_inference::core::scratchpad::KvFormat::F16,
        rust_model_inference::KvLifecycle::Ephemeral,
        rust_model_inference::compute::ComputePolicy::Vulkan,
    )?;
    println!(
        "cold backend=vulkan session_init_ms={:.3} stats={:?}",
        started.elapsed().as_secs_f64() * 1000.0,
        context.compute_stats().since(before_upload)
    );
    generate(&mut gpu, &prompt_tokens, &positions, 1)?;
    let gpu_logits = gpu.last_logits().to_vec();
    gpu.reset_kv();
    let before = context.submission_count();
    let gpu_tokens = generate(&mut gpu, &prompt_tokens, &positions, GREEDY_TOKENS + 1)?.token_ids;
    let submissions = context.submission_count() - before;

    let correctness = (|| {
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
        let expected_submissions = prompt_tokens
            .len()
            .div_ceil(rust_model_inference::core::prefill::DEFAULT_PREFILL_BATCH_SIZE)
            + GREEDY_TOKENS;
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
        Ok(())
    })();
    if let Err(error) = &correctness {
        eprintln!("correctness_failed: {error}");
    }
    if arguments.benchmark {
        benchmark(
            &mut cpu,
            &mut gpu,
            &prompt_tokens,
            &positions,
            correctness.is_ok(),
        )?;
    }
    correctness
}

#[cfg(feature = "vulkan")]
fn qwen35_generate(
    session: &mut Qwen35Session<'_, '_>,
    token_ids: &[u32],
    positions: &[[usize; 4]],
    max_new_tokens: usize,
) -> Result<Vec<u32>, String> {
    let mut logits = session.step_with_tokens(token_ids, positions)?;
    let mut generated = Vec::with_capacity(max_new_tokens);
    for index in 0..max_new_tokens {
        let token = logits
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.total_cmp(right))
            .map(|(index, _)| index as u32)
            .ok_or("Qwen3.5 produced empty logits")?;
        generated.push(token);
        if index + 1 < max_new_tokens {
            let position = session.next_position();
            logits = session.step_with_tokens(&[token], &[[position, position, position, 0]])?;
        }
    }
    Ok(generated)
}

#[cfg(feature = "vulkan")]
fn qwen35_state_bits(session: &Qwen35Session<'_, '_>) -> Vec<u32> {
    let rust_model_inference::core::scratchpad::KvCache::F32(cache) = session.kv_cache() else {
        unreachable!("Qwen3.5 requires F32 KV");
    };
    cache
        .k
        .iter()
        .chain(&cache.v)
        .chain(session.scratch().conv_states.iter().flatten())
        .chain(session.scratch().ssm_states.iter().flatten())
        .map(|value| value.to_bits())
        .collect()
}

#[cfg(feature = "vulkan")]
fn check_qwen35_submissions(actual: u64, chunks: usize, gpu: bool) -> Result<(), String> {
    // Qwen3.5 flushes long command buffers between layers for driver correctness.
    if (gpu && actual >= chunks as u64) || (!gpu && actual == 0) {
        Ok(())
    } else {
        Err(format!(
            "Qwen3.5 submissions={actual}, chunks={chunks}, gpu={gpu}"
        ))
    }
}

#[cfg(feature = "vulkan")]
fn compare_qwen35_prefill_batches(
    model: &mut Qwen35Model<'_>,
    tokens: &[u32],
    positions: &[[usize; 4]],
    batches: &[usize],
    gpu: bool,
) -> Result<(), String> {
    let context = if gpu {
        rust_model_inference::ops::enable_gpu();
        Some(
            rust_model_inference::ops::get_vulkan_context()
                .ok_or("Vulkan backend did not initialize")?,
        )
    } else {
        None
    };
    let submissions = || {
        context
            .as_ref()
            .map_or(0, |context| context.submission_count())
    };
    let backend = if gpu { "vulkan" } else { "cpu" };
    let capacity = tokens
        .len()
        .checked_add(GREEDY_TOKENS + 1)
        .ok_or("session capacity overflow")?;
    let pool = Arc::new(ComputePool::new(4));
    let mut baseline = None;
    for &batch in batches {
        let mut session = Qwen35Session::new_with_compute(
            model,
            capacity,
            batch,
            Arc::clone(&pool),
            if gpu {
                rust_model_inference::compute::ComputePolicy::Vulkan
            } else {
                rust_model_inference::compute::ComputePolicy::Cpu
            },
        )?;
        let before = submissions();
        let mut logits = session.step_with_tokens(tokens, positions)?;
        let prefill_submissions = submissions() - before;
        let chunks = tokens.len().div_ceil(batch);
        check_qwen35_submissions(prefill_submissions, chunks, gpu)?;
        let prompt_logits = logits
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>();
        let prompt_state = qwen35_state_bits(&session);
        let mut generated = Vec::with_capacity(GREEDY_TOKENS);
        for _ in 0..GREEDY_TOKENS {
            let token = logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .ok_or("empty logits")?
                .0 as u32;
            generated.push(token);
            let position = session.next_position();
            logits = session.step_with_tokens(&[token], &[[position, position, position, 0]])?;
        }
        let total_submissions = submissions() - before;
        check_qwen35_submissions(total_submissions - prefill_submissions, GREEDY_TOKENS, gpu)?;
        let result = (
            prompt_logits,
            prompt_state,
            generated,
            qwen35_state_bits(&session),
            logits
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
        );
        if let Some(previous) = &baseline {
            if &result != previous {
                return Err(format!("same-{backend} Qwen3.5 prefill mismatch for batch={batch}: logits, dense KV, conv/SSM or greedy tokens differ"));
            }
        } else {
            baseline = Some(result);
        }
        println!("backend={backend} batch={batch} prompt_tokens={} prefill_submissions={prefill_submissions} total_submissions={total_submissions} greedy_tokens={GREEDY_TOKENS}", tokens.len());
    }
    println!("check=same_{backend}_prefill exact_logits=true exact_dense_kv=true exact_conv_ssm=true exact_greedy_tokens=true exact_decode_state=true");
    Ok(())
}

#[cfg(feature = "vulkan")]
fn run_qwen35(arguments: &Arguments) -> Result<(), String> {
    let source: Arc<dyn TensorSource> = Arc::from(
        open_model_source(&arguments.model, ComponentRole::Llm)
            .map_err(|error| error.to_string())?,
    );
    let tokenizer = BPETokenizer::from_gguf_metadata(|key| source.metadata(key).cloned())?;
    let mut model = Qwen35Model::from_source(source.as_ref())?;
    let prompt = if arguments.compare_prefill_batches.is_some() {
        PROMPT.repeat(20)
    } else {
        PROMPT.to_string()
    };
    let prompt_tokens = build_simple_prompt(&tokenizer, &prompt);
    let positions = qwen_text_positions(prompt_tokens.len());
    if let Some(batches) = &arguments.compare_prefill_batches {
        compare_qwen35_prefill_batches(&mut model, &prompt_tokens, &positions, batches, false)?;
        return compare_qwen35_prefill_batches(
            &mut model,
            &prompt_tokens,
            &positions,
            batches,
            true,
        );
    }
    let capacity = prompt_tokens
        .len()
        .checked_add(GREEDY_TOKENS + 1)
        .ok_or("session capacity overflow")?;
    let pool = Arc::new(ComputePool::new(4));

    let mut cpu = Qwen35Session::new_with_compute(
        &mut model,
        capacity,
        64,
        Arc::clone(&pool),
        rust_model_inference::compute::ComputePolicy::Cpu,
    )?;
    let cpu_logits = cpu.step_with_tokens(&prompt_tokens, &positions)?;
    cpu.reset();
    let cpu_tokens = qwen35_generate(&mut cpu, &prompt_tokens, &positions, GREEDY_TOKENS + 1)?;
    drop(cpu);

    rust_model_inference::ops::enable_gpu();
    let context = rust_model_inference::ops::get_vulkan_context()
        .ok_or("Vulkan backend did not initialize")?;
    let mut gpu = Qwen35Session::new_with_compute(
        &mut model,
        capacity,
        64,
        pool,
        rust_model_inference::compute::ComputePolicy::Vulkan,
    )?;
    let gpu_logits = gpu.step_with_tokens(&prompt_tokens, &positions)?;
    assert_close("prefill_logits", &gpu_logits, &cpu_logits)?;
    gpu.reset();
    let before = context.submission_count();
    let gpu_tokens = qwen35_generate(&mut gpu, &prompt_tokens, &positions, GREEDY_TOKENS + 1)?;
    let submissions = context.submission_count() - before;

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
    let expected_submissions = prompt_tokens
        .len()
        .div_ceil(rust_model_inference::core::prefill::DEFAULT_PREFILL_BATCH_SIZE)
        + GREEDY_TOKENS;
    check_qwen35_submissions(submissions, expected_submissions, true)?;
    println!("formats=matmul={{BF16}};auxiliary={{F32}};backend=vulkan");
    println!(
        "device={} prompt_tokens={} greedy_tokens={} submissions={submissions}",
        context.device_name(),
        prompt_tokens.len(),
        GREEDY_TOKENS
    );
    println!("tokens={gpu_tokens:?}");
    Ok(())
}

#[cfg(feature = "vulkan")]
fn normalize(mut values: Vec<f32>) -> Result<Vec<f32>, String> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err("embedding contains a non-finite value".into());
    }
    let sum = values
        .iter()
        .map(|&value| f64::from(value * value))
        .sum::<f64>();
    let scale = if sum > 0.0 {
        (1.0 / sum.sqrt()) as f32
    } else {
        0.0
    };
    for value in &mut values {
        *value *= scale;
    }
    Ok(values)
}

#[cfg(feature = "vulkan")]
fn embed_text(
    model: &Qwen3Model,
    tokenizer: &BPETokenizer,
    text: &str,
) -> Result<(Vec<f32>, usize), String> {
    let tokens = tokenizer.encode(
        text,
        EncodeOptions {
            add_special: true,
            parse_special: true,
        },
    );
    if tokens.is_empty() {
        return Err("embedding fixture produced no tokens".into());
    }
    let hidden = model.text_encode(&tokens, &qwen_text_positions(tokens.len()))?;
    let width = model.config().n_embd;
    let last = hidden
        .get(
            hidden
                .len()
                .checked_sub(width)
                .ok_or("missing final hidden row")?..,
        )
        .ok_or("missing final hidden row")?
        .to_vec();
    Ok((normalize(last)?, tokens.len()))
}

#[cfg(feature = "vulkan")]
fn cosine(left: &[f32], right: &[f32]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(&left, &right)| f64::from(left * right))
        .sum()
}

#[cfg(feature = "vulkan")]
fn ranking(scores: [f64; 2]) -> [usize; 2] {
    let mut order = [0, 1];
    order.sort_by(|&left, &right| scores[right].total_cmp(&scores[left]));
    order
}

#[cfg(feature = "vulkan")]
fn run_embedding(arguments: &Arguments) -> Result<(), String> {
    const TEXTS: [&str; 3] = [
        "What is the capital of France?",
        "Paris is the capital of France.",
        "Photosynthesis converts light energy into chemical energy.",
    ];

    let (source, tokenizer, model) = load_model(arguments)?;
    print_formats(&model, source.as_ref())?;
    let mut cpu = Vec::with_capacity(TEXTS.len());
    let mut expected_submissions = 0usize;
    for text in TEXTS {
        let (embedding, tokens) = embed_text(&model, &tokenizer, text)?;
        cpu.push(embedding);
        expected_submissions += tokens;
    }

    rust_model_inference::ops::enable_gpu();
    let context = rust_model_inference::ops::get_vulkan_context()
        .ok_or("Vulkan backend did not initialize")?;
    let before = context.submission_count();
    let mut gpu = Vec::with_capacity(TEXTS.len());
    for (index, text) in TEXTS.into_iter().enumerate() {
        let (embedding, _) = embed_text(&model, &tokenizer, text)?;
        assert_close(&format!("embedding_{index}"), &embedding, &cpu[index])?;
        gpu.push(embedding);
    }
    let submissions = context.submission_count() - before;
    if submissions != expected_submissions as u64 {
        return Err(format!(
            "expected one submission per embedding token ({expected_submissions}), got {submissions}"
        ));
    }

    let cpu_scores = [cosine(&cpu[0], &cpu[1]), cosine(&cpu[0], &cpu[2])];
    let gpu_scores = [cosine(&gpu[0], &gpu[1]), cosine(&gpu[0], &gpu[2])];
    let cpu_ranking = ranking(cpu_scores);
    let gpu_ranking = ranking(gpu_scores);
    if gpu_ranking != cpu_ranking {
        return Err(format!(
            "embedding ranking mismatch: gpu={gpu_ranking:?} cpu={cpu_ranking:?}"
        ));
    }
    println!("check=embedding_ranking cpu={cpu_scores:?} gpu={gpu_scores:?} order={gpu_ranking:?}");
    println!(
        "device={} texts={} submissions={submissions}",
        context.device_name(),
        TEXTS.len()
    );
    Ok(())
}

#[cfg(feature = "vulkan")]
fn run_llama(arguments: &Arguments) -> Result<(), String> {
    use rust_model_inference::{
        compute::{ComputePolicy, UsedBackend},
        core::scratchpad::{KvCache, KvFormat},
        models::llama::trunk::LlamaSession,
    };
    let source = open_model_source(&arguments.model, ComponentRole::Llm)
        .map_err(|error| error.to_string())?;
    let batches = arguments
        .compare_prefill_batches
        .as_deref()
        .unwrap_or(&[1, 3, 64]);
    for &batch in batches {
        let started = std::time::Instant::now();
        let mut cpu = LlamaSession::from_source_with_compute(
            source.as_ref(),
            4,
            KvFormat::F16,
            512,
            batch,
            ComputePolicy::Cpu,
        )?;
        println!(
            "cold model=llama backend=cpu session_init_ms={:.3} batch={batch}",
            started.elapsed().as_secs_f64() * 1000.0
        );
        let started = std::time::Instant::now();
        let mut gpu = LlamaSession::from_source_with_compute(
            source.as_ref(),
            4,
            KvFormat::F16,
            512,
            batch,
            ComputePolicy::Vulkan,
        )?;
        println!(
            "cold model=llama backend=vulkan session_init_including_context_ms={:.3} batch={batch}",
            started.elapsed().as_secs_f64() * 1000.0
        );
        let tokens = cpu.tokenizer.encode(
            &PROMPT.repeat(12),
            EncodeOptions {
                add_special: true,
                parse_special: true,
            },
        );
        for repeat in 0..2 {
            let mut a = cpu.forward_logits_chunked(&tokens, batch)?;
            let mut b = gpu.forward_logits_chunked(&tokens, batch)?;
            assert_close("llama_prefill", &b, &a)?;
            for step in 0..GREEDY_TOKENS {
                let best = |v: &[f32]| {
                    v.iter()
                        .enumerate()
                        .max_by(|a, b| a.1.total_cmp(b.1))
                        .unwrap()
                        .0 as u32
                };
                let token = best(&a);
                if best(&b) != token {
                    return Err(format!("Llama greedy mismatch at {step}"));
                }
                a = cpu.forward_logits_chunked(&[token], 1)?;
                b = gpu.forward_logits_chunked(&[token], 1)?;
                assert_close("llama_decode", &b, &a)?;
            }
            if gpu.used_backend() != UsedBackend::Vulkan {
                return Err("Llama GPU check fell back".into());
            }
            if let (KvCache::F16(a), KvCache::F16(b)) = (&cpu.kv_cache, &gpu.kv_cache) {
                let av: Vec<_> =
                    a.k.iter()
                        .chain(&a.v)
                        .map(|&v| rust_model_inference::ops::f16_to_f32(v))
                        .collect();
                let bv: Vec<_> =
                    b.k.iter()
                        .chain(&b.v)
                        .map(|&v| rust_model_inference::ops::f16_to_f32(v))
                        .collect();
                assert_close("llama_kv", &bv, &av)?;
            }
            println!("model=llama scope=resident_decoder backend=Vulkan batch={batch} repeat={repeat} prompt={} greedy_tokens={GREEDY_TOKENS}",tokens.len());
            cpu.reset();
            gpu.reset();
        }
        if arguments.benchmark {
            rust_model_inference::ops::enable_gpu();
            let context =
                rust_model_inference::ops::get_vulkan_context().ok_or("Vulkan context missing")?;
            println!("benchmark model=llama kv=f16 threads=4 batch={batch} context=512 input_ids={tokens:?}");
            paired_benchmark(true, |use_gpu| {
                let session = if use_gpu { &mut gpu } else { &mut cpu };
                let before = context.compute_stats();
                let started = std::time::Instant::now();
                session.reset();
                let mut logits = session.forward_logits_chunked(&tokens, batch)?;
                let prompt_time = started.elapsed();
                let decode_started = std::time::Instant::now();
                for _ in 0..GREEDY_TOKENS {
                    let token = logits
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.total_cmp(b.1))
                        .unwrap()
                        .0 as u32;
                    logits = session.forward_logits_chunked(&[token], 1)?;
                }
                Ok(BenchmarkSample {
                    prompt: per_second(tokens.len(), prompt_time),
                    decode: per_second(GREEDY_TOKENS, decode_started.elapsed()),
                    wall_ms: started.elapsed().as_secs_f64() * 1000.0,
                    stats: context.compute_stats().since(before),
                })
            })?;
        }
    }
    Ok(())
}

#[cfg(feature = "vulkan")]
fn run() -> Result<(), String> {
    let arguments = arguments()?;
    use sha2::{Digest, Sha256};
    use std::io::Read;
    let mut file = std::fs::File::open(&arguments.model).map_err(|e| e.to_string())?;
    let mut hash = Sha256::new();
    let mut chunk = vec![0u8; 1024 * 1024];
    loop {
        let n = file.read(&mut chunk).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        hash.update(&chunk[..n]);
    }
    println!(
        "model={} sha256={:x} cold_io=hash_scan_precedes_load",
        arguments.model.display(),
        hash.finalize()
    );
    match arguments.mode {
        Mode::Llama => run_llama(&arguments),
        Mode::Qwen3 => run_qwen3(&arguments),
        Mode::Qwen35 => run_qwen35(&arguments),
        Mode::Embedding => run_embedding(&arguments),
    }
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
    use super::{format_summary, median, per_second};
    use rust_model_inference::GGMLType;
    use std::time::Duration;

    #[test]
    fn prefill_batch_comparison_requires_positive_sizes() {
        assert_eq!(super::parse_prefill_batches("1,64").unwrap(), vec![1, 64]);
        for value in ["", "0,64", "1,", "1,nope", "1"] {
            assert!(super::parse_prefill_batches(value).is_err(), "{value}");
        }
    }

    #[test]
    fn format_summary_reports_later_layer_formats() {
        let layers = [
            [
                GGMLType::Q4K,
                GGMLType::Q4K,
                GGMLType::Q6K,
                GGMLType::Q4K,
                GGMLType::Q4K,
                GGMLType::Q4K,
                GGMLType::Q6K,
            ],
            [
                GGMLType::F16,
                GGMLType::Q4_0,
                GGMLType::Q4_1,
                GGMLType::Q8_0,
                GGMLType::F16,
                GGMLType::Q6K,
                GGMLType::Q4K,
            ],
        ];

        assert_eq!(
            format_summary(&layers, GGMLType::F16).unwrap(),
            "formats=layers=2;q={Q4K,F16},k={Q4K,Q4_0},v={Q6K,Q4_1},o={Q4K,Q8_0},gate={Q4K,F16},up={Q4K,Q6K},down={Q6K,Q4K};output=F16"
        );
    }

    #[test]
    fn benchmark_statistics_use_sorted_middle_and_elapsed_seconds() {
        assert_eq!(median(&[9.0, 1.0, 5.0, 3.0, 7.0]), 5.0);
        assert_eq!(per_second(8, Duration::from_millis(500)), 16.0);
    }

    #[test]
    fn logit_check_rejects_non_finite_reference_and_output() {
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(super::assert_close("fixture", &[1.0], &[invalid]).is_err());
            assert!(super::assert_close("fixture", &[invalid], &[1.0]).is_err());
        }
    }

    #[test]
    fn qwen35_submission_check_allows_driver_safety_flushes_without_cpu_offload() {
        assert!(super::check_qwen35_submissions(0, 80, false).is_ok());
        assert!(super::check_qwen35_submissions(1, 80, false).is_err());
        assert!(super::check_qwen35_submissions(80, 80, true).is_ok());
        assert!(super::check_qwen35_submissions(560, 80, true).is_ok());
        assert!(super::check_qwen35_submissions(79, 80, true).is_err());
        assert!(super::check_qwen35_submissions(0, 80, true).is_err());
    }

    #[test]
    fn auto_gate_requires_correctness_five_pairs_and_ten_percent_wall_gain() {
        assert!(super::auto_gain_passes(&[100.0; 5], &[90.0; 5], true));
        assert!(!super::auto_gain_passes(&[100.0; 5], &[91.0; 5], true));
        assert!(!super::auto_gain_passes(&[100.0; 5], &[80.0; 5], false));
        assert!(!super::auto_gain_passes(&[100.0; 4], &[80.0; 4], true));
        assert!(!super::auto_gain_passes(&[100.0; 5], &[f64::NAN; 5], true));
        assert_eq!(super::benchmark_order(0), [false, true]);
        assert_eq!(super::benchmark_order(1), [true, false]);
        assert_eq!(super::benchmark_order(4), [false, true]);
    }
}
