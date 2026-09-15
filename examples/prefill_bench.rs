#[derive(Clone, Debug)]
struct Arguments {
    model_name: String,
    model: String,
    backend: String,
    threads: usize,
    kv: String,
    prompt_tokens: usize,
    batch: usize,
    samples: usize,
    generate: usize,
}

#[derive(Clone)]
struct BenchSample {
    sha256: String,
    model: String,
    backend: String,
    device: String,
    threads: usize,
    kv: String,
    prompt_tokens: usize,
    batch: usize,
    pp_tps: f64,
    tg_tps: f64,
    first_token_ms: f64,
    total_ms: f64,
    scratch_bytes: usize,
    submission_delta: Option<u64>,
    decode_submission_delta: Option<u64>,
}

fn arguments(mut args: impl Iterator<Item = String>) -> Result<Arguments, String> {
    let model_name = args.next().ok_or("expected qwen3, qwen35 or gemma4")?;
    if !matches!(model_name.as_str(), "qwen3" | "qwen35" | "gemma4") {
        return Err("expected qwen3, qwen35 or gemma4".into());
    }
    let mut result = Arguments {
        kv: if model_name == "qwen3" { "f16" } else { "f32" }.into(),
        model_name,
        model: String::new(),
        backend: "cpu".into(),
        threads: 4,
        prompt_tokens: 128,
        batch: 64,
        samples: 5,
        generate: 32,
    };
    while let Some(key) = args.next() {
        let value = args.next().ok_or_else(|| format!("{key} needs a value"))?;
        match key.as_str() {
            "--model" => result.model = value,
            "--backend" => result.backend = value,
            "--kv" => result.kv = value,
            "--threads" | "--prompt-tokens" | "--batch" | "--samples" | "--generate" => {
                let n = value
                    .parse::<usize>()
                    .map_err(|_| format!("invalid {key}: {value}"))?;
                if n == 0 {
                    return Err(format!("{key} must be positive"));
                }
                match key.as_str() {
                    "--threads" => result.threads = n,
                    "--prompt-tokens" => result.prompt_tokens = n,
                    "--batch" => result.batch = n,
                    "--samples" => result.samples = n,
                    _ => result.generate = n,
                }
            }
            _ => return Err(format!("unknown argument: {key}")),
        }
    }
    if result.model.is_empty() {
        return Err("--model is required".into());
    }
    if !matches!(result.backend.as_str(), "cpu" | "vulkan") {
        return Err("invalid backend".into());
    }
    if !matches!(result.kv.as_str(), "f16" | "f32")
        || (result.model_name != "qwen3" && result.kv != "f32")
    {
        return Err("invalid KV format; qwen35 and gemma4 require f32".into());
    }
    if result.generate != 32 {
        return Err("--generate must be 32".into());
    }
    result
        .prompt_tokens
        .checked_add(33)
        .ok_or("capacity overflow")?;
    Ok(result)
}

fn median(values: &mut [f64]) -> f64 {
    assert!(!values.is_empty());
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len() % 2 == 0 {
        values[middle - 1] / 2.0 + values[middle] / 2.0
    } else {
        values[middle]
    }
}

fn summary_line(kind: &str, sample: &BenchSample) -> String {
    assert!(matches!(kind, "sample" | "median"));
    let device = sample
        .device
        .split_whitespace()
        .collect::<Vec<_>>()
        .join("_");
    let mut line = format!("kind={kind} sha256={} model={} backend={} device={device} threads={} kv={} prompt_tokens={} batch={} pp_tps={:.6} tg_tps={:.6} first_token_ms={:.6} total_ms={:.6} scratch_bytes={}",
        sample.sha256, sample.model, sample.backend, sample.threads, sample.kv,
        sample.prompt_tokens, sample.batch, sample.pp_tps, sample.tg_tps,
        sample.first_token_ms, sample.total_ms, sample.scratch_bytes);
    if sample.backend == "vulkan" {
        line.push_str(&format!(
            " submission_delta={} decode_submission_delta={}",
            sample.submission_delta.expect("Vulkan prompt counter"),
            sample
                .decode_submission_delta
                .expect("Vulkan decode counter")
        ));
    }
    line
}

fn main() {
    if let Err(error) = arguments(std::env::args().skip(1)).and_then(run) {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

fn run(args: Arguments) -> Result<(), String> {
    use rust_model_inference::core::scratchpad::{KvFormat, KvLifecycle};
    use rust_model_inference::models::gemma4::{Gemma4InputRow, Gemma4Model, Gemma4Session};
    use rust_model_inference::models::qwen35::Qwen35Session;
    use rust_model_inference::{
        open_model_source, BPETokenizer, ComponentRole, ComputePool, EncodeOptions, Qwen35Model,
        Qwen3GenerateOptions, Qwen3Input, Qwen3Model, Qwen3Session, TensorSource,
    };
    use sha2::{Digest, Sha256};
    use std::{io::Read, sync::Arc, time::Instant};

    if std::env::var_os("RMI_PARITY_TRACE").is_some() {
        return Err("unset RMI_PARITY_TRACE before benchmarking".into());
    }
    #[cfg(feature = "vulkan")]
    let context = if args.backend == "vulkan" {
        rust_model_inference::ops::enable_gpu();
        Some(
            rust_model_inference::ops::get_vulkan_context()
                .ok_or("Vulkan initialization failed")?,
        )
    } else {
        None
    };
    #[cfg(not(feature = "vulkan"))]
    if args.backend == "vulkan" {
        return Err("rebuild with --features vulkan".into());
    }
    let counter = || -> u64 {
        #[cfg(feature = "vulkan")]
        if let Some(context) = &context {
            return context.submission_count();
        }
        0
    };
    let mut hash = Sha256::new();
    let mut file = std::fs::File::open(&args.model).map_err(|e| e.to_string())?;
    let mut buffer = vec![0; 1024 * 1024];
    loop {
        let n = file.read(&mut buffer).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    let source: Arc<dyn TensorSource> = Arc::from(
        open_model_source(std::path::Path::new(&args.model), ComponentRole::Llm)
            .map_err(|e| e.to_string())?,
    );
    let tokenizer = Arc::new(BPETokenizer::from_gguf_metadata(|key| {
        source.metadata(key).cloned()
    })?);
    let seed = tokenizer.encode(
        "The quick brown fox jumps over the lazy dog. ",
        EncodeOptions {
            add_special: false,
            parse_special: false,
        },
    );
    let seed = seed
        .into_iter()
        .filter(|&id| {
            Some(id) != tokenizer.bos_id()
                && Some(id) != tokenizer.eos_id()
                && !tokenizer.token_piece_bytes(id, false).is_empty()
        })
        .collect::<Vec<_>>();
    if seed.is_empty() {
        return Err("seed contains no non-special token IDs".into());
    }
    let mut tokens = Vec::with_capacity(args.prompt_tokens);
    if tokenizer.add_bos() || args.model_name == "gemma4" {
        tokens.push(tokenizer.bos_id().ok_or("required BOS missing")?);
    }
    tokens.extend(
        seed.iter()
            .copied()
            .cycle()
            .take(args.prompt_tokens - tokens.len()),
    );
    let positions = rust_model_inference::qwen_text_positions(tokens.len());
    let pool = Arc::new(ComputePool::new(args.threads));
    let kv = if args.kv == "f16" {
        KvFormat::F16
    } else {
        KvFormat::F32
    };
    let device = {
        #[cfg(feature = "vulkan")]
        if let Some(context) = &context {
            context.device_name().to_string()
        } else {
            "CPU".into()
        }
        #[cfg(not(feature = "vulkan"))]
        {
            "CPU".into()
        }
    };
    let mut sample = BenchSample {
        sha256: format!("{:x}", hash.finalize()),
        model: args.model_name.clone(),
        backend: args.backend.clone(),
        device,
        threads: args.threads,
        kv: args.kv.clone(),
        prompt_tokens: args.prompt_tokens,
        batch: args.batch,
        pp_tps: 0.0,
        tg_tps: 0.0,
        first_token_ms: 0.0,
        total_ms: 0.0,
        scratch_bytes: 0,
        submission_delta: None,
        decode_submission_delta: None,
    };
    let mut samples = Vec::with_capacity(args.samples);
    let mut collect = |measure: &mut dyn FnMut() -> Result<
        (f64, f64, f64, usize, u64, u64),
        String,
    >|
     -> Result<(), String> {
        for index in 0..=args.samples {
            let (pp, tg, total, scratch, prompt_submissions, decode_submissions) = measure()?;
            if args.backend == "vulkan"
                && args.model_name != "gemma4"
                && prompt_submissions != args.prompt_tokens.div_ceil(args.batch) as u64
            {
                return Err(format!("prompt submissions {prompt_submissions} != chunks {} (backend fallback or changed dispatch)", args.prompt_tokens.div_ceil(args.batch)));
            }
            if index == 0 {
                continue;
            }
            sample.pp_tps = args.prompt_tokens as f64 / pp;
            sample.tg_tps = 32.0 / tg;
            sample.first_token_ms = pp * 1000.0;
            sample.total_ms = total * 1000.0;
            sample.scratch_bytes = scratch;
            sample.submission_delta = (args.backend == "vulkan").then_some(prompt_submissions);
            sample.decode_submission_delta =
                (args.backend == "vulkan").then_some(decode_submissions);
            println!("{}", summary_line("sample", &sample));
            samples.push(sample.clone());
        }
        Ok(())
    };
    match args.model_name.as_str() {
        "qwen3" => {
            let model = Qwen3Model::from_source(source, tokenizer, pool)?;
            let mut session = Qwen3Session::new_with_kv_state(
                &model,
                tokens.len() + 33,
                kv,
                KvLifecycle::Ephemeral,
            )?;
            collect(&mut || {
                session.reset_kv();
                let started = Instant::now();
                let result = session.generate(
                    Qwen3Input {
                        token_ids: &tokens,
                        positions: &positions,
                        embeddings: None,
                        deepstack_embeddings: None,
                    },
                    Qwen3GenerateOptions {
                        max_new_tokens: 33,
                        temperature: 0.0,
                        prefill_batch_size: args.batch,
                    },
                )?;
                let total = started.elapsed().as_secs_f64();
                if result.token_ids.len() != 33 {
                    return Err(format!(
                        "Qwen3 stopped after {} tokens; cannot measure 32 decode evaluations",
                        result.token_ids.len()
                    ));
                }
                #[cfg(feature = "vulkan")]
                let (prompt, decode) = (result.prompt_submissions, result.decode_submissions);
                #[cfg(not(feature = "vulkan"))]
                let (prompt, decode) = (0, 0);
                Ok((
                    result.prompt_duration.as_secs_f64(),
                    result.decode_duration.as_secs_f64(),
                    total,
                    session.scratch_bytes(),
                    prompt,
                    decode,
                ))
            })?;
        }
        "qwen35" => {
            let mut model = Qwen35Model::from_source(source.as_ref())?;
            let mut session = Qwen35Session::new_with_prefill_batch_size(
                &mut model,
                tokens.len() + 32,
                args.batch,
                pool,
            )?;
            collect(&mut || {
                session.reset();
                let before = counter();
                let started = Instant::now();
                let mut logits = session.step_with_tokens(&tokens, &positions)?;
                let pp = started.elapsed().as_secs_f64();
                let after_prompt = counter();
                let decode_started = Instant::now();
                for _ in 0..32 {
                    let id = greedy(&logits)?;
                    let p = session.next_position();
                    logits = session.step_with_tokens(&[id], &[[p, p, p, 0]])?;
                }
                Ok((
                    pp,
                    decode_started.elapsed().as_secs_f64(),
                    started.elapsed().as_secs_f64(),
                    session.scratch_bytes(),
                    after_prompt - before,
                    counter() - after_prompt,
                ))
            })?;
        }
        "gemma4" => {
            let model = Gemma4Model::from_source(source, args.threads)?;
            let rows = tokens
                .iter()
                .map(|&id| Gemma4InputRow::Token(id))
                .collect::<Vec<_>>();
            let mut session = Gemma4Session::new_with_prefill_batch_size(&model, kv, args.batch)?;
            collect(&mut || {
                session.reset();
                let before = counter();
                let started = Instant::now();
                let mut logits = session.forward_rows(&rows)?;
                let pp = started.elapsed().as_secs_f64();
                let after_prompt = counter();
                if args.backend == "vulkan" {
                    let expected = args.prompt_tokens.div_ceil(args.batch)
                        * (1 + 7 * model.config.layers + 2 * model.config.base_kv_layers());
                    if after_prompt - before != expected as u64 {
                        return Err(format!(
                            "Gemma4 projection submissions {} != {expected}",
                            after_prompt - before
                        ));
                    }
                }
                let decode_started = Instant::now();
                for _ in 0..32 {
                    logits = session.forward_rows(&[Gemma4InputRow::Token(greedy(&logits)?)])?;
                }
                Ok((
                    pp,
                    decode_started.elapsed().as_secs_f64(),
                    started.elapsed().as_secs_f64(),
                    session.scratch_bytes(),
                    after_prompt - before,
                    counter() - after_prompt,
                ))
            })?;
        }
        _ => unreachable!(),
    }
    let mut summary = samples[0].clone();
    summary.pp_tps = median(&mut samples.iter().map(|s| s.pp_tps).collect::<Vec<_>>());
    summary.tg_tps = median(&mut samples.iter().map(|s| s.tg_tps).collect::<Vec<_>>());
    summary.first_token_ms =
        median(&mut samples.iter().map(|s| s.first_token_ms).collect::<Vec<_>>());
    summary.total_ms = median(&mut samples.iter().map(|s| s.total_ms).collect::<Vec<_>>());
    println!("{}", summary_line("median", &summary));
    Ok(())
}

fn greedy(logits: &[f32]) -> Result<u32, String> {
    if logits.iter().any(|value| !value.is_finite()) {
        return Err("non-finite logits".into());
    }
    logits
        .iter()
        .enumerate()
        .max_by(|(i, a), (j, b)| a.total_cmp(b).then_with(|| j.cmp(i)))
        .map(|(id, _)| id as u32)
        .ok_or_else(|| "empty logits".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_sample() -> BenchSample {
        BenchSample {
            sha256: "abc123".into(),
            model: "qwen3".into(),
            backend: "cpu".into(),
            device: "CPU".into(),
            threads: 4,
            kv: "f16".into(),
            prompt_tokens: 128,
            batch: 64,
            pp_tps: 10.0,
            tg_tps: 20.0,
            first_token_ms: 30.0,
            total_ms: 40.0,
            scratch_bytes: 4096,
            submission_delta: None,
            decode_submission_delta: None,
        }
    }

    #[test]
    fn parser_rejects_invalid_measurement_lengths() {
        let base = ["qwen3", "--model", "model.gguf"];
        let parse = |extra: &[&str]| arguments(base.iter().chain(extra).map(|s| s.to_string()));
        let defaults = parse(&[]).unwrap();
        assert_eq!(
            (defaults.batch, defaults.samples, defaults.generate),
            (64, 5, 32)
        );
        for pair in [
            ["--prompt-tokens", "0"],
            ["--batch", "0"],
            ["--samples", "0"],
            ["--generate", "31"],
            ["--generate", "33"],
            ["--threads", "0"],
            ["--backend", "metal"],
            ["--kv", "q8"],
            ["--batch", "-1"],
        ] {
            assert!(parse(&pair).is_err(), "accepted {pair:?}");
        }
        assert!(parse(&["--model"]).is_err());
        assert!(arguments(["llama", "--model", "x"].into_iter().map(String::from)).is_err());
        assert!(arguments(
            ["gemma4", "--model", "x", "--kv", "f16"]
                .into_iter()
                .map(String::from)
        )
        .is_err());
    }

    #[test]
    fn median_sorts_and_averages_the_middle_pair() {
        assert_eq!(median(&mut [9.0, 1.0, 4.0]), 4.0);
        assert_eq!(median(&mut [10.0, 2.0, 4.0, 8.0]), 6.0);
        assert_eq!(median(&mut [7.0]), 7.0);
    }

    #[test]
    fn greedy_preserves_lowest_id_ties_and_rejects_bad_logits() {
        assert_eq!(greedy(&[1.0, 2.0, 2.0]).unwrap(), 1);
        assert!(greedy(&[]).is_err());
        assert!(greedy(&[f32::NAN]).is_err());
        assert!(greedy(&[f32::INFINITY]).is_err());
    }

    #[test]
    fn benchmark_summary_contains_reproduction_fields() {
        let mut sample = fixture_sample();
        let line = summary_line("sample", &sample);
        let keys = line
            .split_whitespace()
            .map(|part| part.split('=').next().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            keys,
            [
                "kind",
                "sha256",
                "model",
                "backend",
                "device",
                "threads",
                "kv",
                "prompt_tokens",
                "batch",
                "pp_tps",
                "tg_tps",
                "first_token_ms",
                "total_ms",
                "scratch_bytes"
            ]
        );
        assert!(line.starts_with("kind=sample sha256=abc123 model=qwen3"));
        assert!(!line.contains("submission_delta="));
        sample.backend = "vulkan".into();
        sample.device = "Apple M3 Max".into();
        sample.submission_delta = Some(2);
        sample.decode_submission_delta = Some(32);
        let line = summary_line("median", &sample);
        assert!(line.starts_with("kind=median"));
        assert!(line.contains("device=Apple_M3_Max"));
        assert!(line.ends_with("submission_delta=2 decode_submission_delta=32"));
    }

    #[test]
    #[should_panic]
    fn output_rejects_unknown_record_kind() {
        summary_line("mean", &fixture_sample());
    }
}
