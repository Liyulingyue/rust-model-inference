// Standalone validation driver. The production CLI and server are tested separately.
use rust_model_inference::compute::ComputePolicy;
use rust_model_inference::models::qwen35::Qwen35Session;
use rust_model_inference::{
    qwen_text_positions, BPETokenizer, ComputePool, EncodeOptions, GGUFLoader, Qwen35Model,
    TensorSource,
};
use std::{fs, io::Write, sync::Arc, time::Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    let tokens_only = args.len() == 4 && args[2] == "tokens";
    if !tokens_only && args.len() != 6 && args.len() != 8 {
        return Err(
            "MODEL tokens PROMPT or MODEL cpu|auto|vulkan PROMPT REFERENCE.txt OUTPUT.f32 [CTX BATCH]".into(),
        );
    }
    let source = GGUFLoader::from_file(&args[1])?;
    let tokenizer = BPETokenizer::from_gguf_metadata(|key| source.metadata(key).cloned())?;
    let ids = tokenizer.encode(
        &args[3],
        EncodeOptions {
            add_special: false,
            parse_special: true,
        },
    );
    if tokens_only {
        print!("input");
        for id in ids {
            print!(" {id}");
        }
        println!();
        return Ok(());
    }
    let reference = fs::read_to_string(&args[4])?;
    let expected: Vec<u32> = reference
        .lines()
        .next()
        .ok_or("empty reference")?
        .split_whitespace()
        .skip(1)
        .map(str::parse)
        .collect::<Result<_, _>>()?;
    assert_eq!(ids, expected);
    let history: Vec<u32> = reference
        .lines()
        .skip(1)
        .map(|s| {
            s.split_whitespace()
                .nth(2)
                .ok_or("missing greedy ID")?
                .parse()
                .map_err(|_| "invalid ID")
        })
        .collect::<Result<_, _>>()?;
    let capacity = args.get(6).map_or(Ok(256), |s| s.parse::<usize>())?;
    let batch = args.get(7).map_or(Ok(8), |s| s.parse::<usize>())?;
    if batch == 0 {
        return Err("batch size must be positive".into());
    }
    if ids.is_empty() || history.is_empty() || ids.len() + history.len() - 1 > capacity {
        return Err("prompt and reference positions must fit in the context".into());
    }
    let mode = match args[2].as_str() {
        "cpu" => ComputePolicy::Cpu,
        "auto" => ComputePolicy::Auto,
        "vulkan" => ComputePolicy::Vulkan,
        _ => return Err("unknown compute policy".into()),
    };
    let gpu = if mode != ComputePolicy::Cpu {
        rust_model_inference::ops::enable_gpu();
        rust_model_inference::ops::get_vulkan_context()
    } else {
        None
    };
    let start = Instant::now();
    let mut model = Qwen35Model::from_source(&source)?;
    let mut session = Qwen35Session::new_with_compute(
        &mut model,
        capacity,
        batch,
        Arc::new(ComputePool::new(8)),
        mode,
    )?;
    println!("load_ms {}", start.elapsed().as_secs_f64() * 1000.0);
    let mut output = std::io::BufWriter::new(fs::File::create(&args[5])?);
    let mut tokens = ids;
    let mut positions = qwen_text_positions(tokens.len());
    for target in history {
        let start = Instant::now();
        let before = gpu.map_or(0, |c| c.submission_count());
        let logits = session.step_with_tokens(&tokens, &positions)?;
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        let submissions = gpu.map_or(0, |c| c.submission_count()) - before;
        if mode == ComputePolicy::Vulkan {
            assert!(submissions > 0);
        }
        assert!(logits.iter().all(|f| f.is_finite()));
        let top = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0;
        println!("step {top} {ms} {submissions}");
        for value in logits {
            output.write_all(&value.to_le_bytes())?;
        }
        let p = session.next_position();
        positions = vec![[p, p, p, 0]];
        tokens = vec![target];
    }
    output.flush()?;
    Ok(())
}
