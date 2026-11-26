//! Same-file full-vocabulary comparison with nari-labs/nari-llama.cpp's
//! test-trellis-model. See tools/oracle/nariq/README.md for the capture command.
use rust_model_inference::compute::ComputePolicy;
use rust_model_inference::models::qwen35::Qwen35Session;
use rust_model_inference::{BPETokenizer, ComputePool, EncodeOptions, GGUFLoader, Qwen35Model};
use std::{
    env, fs,
    io::{Read, Seek, SeekFrom},
    path::PathBuf,
    sync::Arc,
    time::Instant,
};

fn logsumexp(values: &[f32]) -> f64 {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    max + values
        .iter()
        .map(|&v| (v as f64 - max).exp())
        .sum::<f64>()
        .ln()
}

fn policy() -> ComputePolicy {
    match env::var("NARIQ_COMPUTE").as_deref() {
        Ok("cpu") | Err(_) => ComputePolicy::Cpu,
        Ok("auto") => ComputePolicy::Auto,
        Ok("vulkan") => ComputePolicy::Vulkan,
        value => panic!("unknown compute policy {value:?}"),
    }
}

struct Comparison {
    top: u32,
    kl: f64,
    rel_l2: f64,
    max_abs: f64,
}

fn compare(reference: &[f32], actual: &[f32]) -> Comparison {
    assert_eq!(reference.len(), actual.len());
    assert!(actual.iter().chain(reference).all(|v| v.is_finite()));
    let (a, b) = (logsumexp(reference), logsumexp(actual));
    let (mut kl, mut error, mut energy, mut max_abs) = (0.0, 0.0, 0.0, 0.0_f64);
    let mut top = 0;
    for (i, (&x, &y)) in reference.iter().zip(actual).enumerate() {
        if actual[i] > actual[top] {
            top = i;
        }
        let (x, y) = (f64::from(x), f64::from(y));
        kl += (x - a).exp() * ((x - a) - (y - b));
        error += (x - y).powi(2);
        energy += x * x;
        max_abs = max_abs.max((x - y).abs());
    }
    Comparison {
        top: top as u32,
        kl,
        rel_l2: (error / energy).sqrt(),
        max_abs,
    }
}

#[test]
#[ignore = "requires a released NariQ GGUF and matching test-trellis-model capture"]
fn nariq_prefill_and_decode_match_oracle() {
    let prefix = env::var("NARIQ_ORACLE").unwrap();
    let oracle: serde_json::Value =
        serde_json::from_slice(&fs::read(format!("{prefix}.json")).unwrap()).unwrap();
    assert_eq!(oracle["complete"], true);
    let mut capture = fs::File::open(format!("{prefix}.f32")).unwrap();
    let vocab = oracle["vocab_size"].as_u64().unwrap() as usize;
    let source = GGUFLoader::from_file(env::var("NARIQ_MODEL").unwrap()).unwrap();
    let tokenizer = BPETokenizer::from_gguf_metadata(|k| source.metadata(k).cloned()).unwrap();
    let ids = tokenizer.encode(
        oracle["prompt"].as_str().unwrap(),
        EncodeOptions {
            add_special: true,
            parse_special: true,
        },
    );
    assert_eq!(serde_json::json!(ids), oracle["prompt_ids"]);
    let expected: Vec<u32> = serde_json::from_value(oracle["greedy_ids"].clone()).unwrap();
    assert_eq!(
        serde_json::json!(expected),
        oracle["evaluated_generation_ids"]
    );
    assert!(!ids.is_empty() && !expected.is_empty());
    let rows = ids.len() + expected.len() - 1;
    let row_bytes = vocab * 4;
    assert_eq!(capture.metadata().unwrap().len(), (rows * row_bytes) as u64);
    capture
        .seek(SeekFrom::Start(((ids.len() - 1) * row_bytes) as u64))
        .unwrap();
    let mut bytes = vec![0; row_bytes];
    let context = usize::try_from(oracle["n_ctx"].as_u64().unwrap()).unwrap();
    let batch = usize::try_from(oracle["ubatch"].as_u64().unwrap()).unwrap();
    assert!(context >= rows && batch > 0);
    let policy = policy();
    let _compute_scope = policy.configure_cli().unwrap();
    let mut model = Qwen35Model::from_source(&source).unwrap();
    let mut session = Qwen35Session::new_with_compute(
        &mut model,
        context,
        batch,
        Arc::new(ComputePool::new(8)),
        policy,
    )
    .unwrap();
    #[cfg(feature = "vulkan")]
    let gpu = (policy != ComputePolicy::Cpu)
        .then(rust_model_inference::ops::get_vulkan_context)
        .flatten();
    #[cfg(feature = "vulkan")]
    let before = gpu.map_or(0, |c| c.submission_count());
    let started = Instant::now();
    let mut logits = session
        .step_with_tokens(&ids, &rust_model_inference::qwen_text_positions(ids.len()))
        .unwrap();
    let prefill = started.elapsed().as_secs_f64();
    let mut decode_s = 0.0;
    let mut worst_kl = 0.0f64;
    let mut greedy = Vec::new();
    for (step, &token) in expected.iter().enumerate() {
        capture.read_exact(&mut bytes).unwrap();
        let reference: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let Comparison {
            top,
            kl,
            rel_l2,
            max_abs,
        } = compare(&reference, &logits);
        greedy.push(top);
        worst_kl = worst_kl.max(kl);
        eprintln!("step={step} top={top} oracle={token} kl={kl:.9} rel_l2={rel_l2:.8} max_abs={max_abs:.6}");
        if step + 1 < expected.len() {
            let pos = ids.len() + step;
            let start = Instant::now();
            logits = session
                .step_with_tokens(&[token], &[[pos, pos, pos, 0]])
                .unwrap();
            decode_s += start.elapsed().as_secs_f64();
        }
    }
    eprintln!("compute={policy:?} prefill_s={prefill:.5} decode_s={decode_s:.5}");
    #[cfg(feature = "vulkan")]
    {
        let submissions = gpu.map_or(0, |c| c.submission_count() - before);
        eprintln!("submissions={submissions}");
        if policy == ComputePolicy::Vulkan {
            assert!(submissions > 0);
        }
    }
    assert_eq!(greedy, expected);
    assert!(
        worst_kl <= 1e-3,
        "max KL {worst_kl} exceeds the upstream gate"
    );
}

#[test]
#[ignore = "requires a NariQ GGUF, BF16 projector and matching mtmd media capture"]
fn nariq_image_logits_match_oracle() {
    let model = GGUFLoader::from_file(env::var("NARIQ_MODEL").unwrap()).unwrap();
    let projector = std::path::PathBuf::from(env::var_os("NARIQ_MMPROJ").unwrap());
    let capture = env::var("NARIQ_IMAGE_ORACLE").unwrap();
    let oracle: serde_json::Value =
        serde_json::from_slice(&fs::read(format!("{capture}.json")).unwrap()).unwrap();
    assert_eq!(
        oracle["oracle_pin"],
        "2644f7de461c27675eef373faa1776f20b393004"
    );
    // The original fixed-red capture predates the context field and uses 128.
    let context = usize::try_from(oracle["n_ctx"].as_u64().unwrap_or(128)).unwrap();
    let reference = fs::read(format!("{capture}.f32")).unwrap();
    assert_eq!(reference.len(), 248320 * 4);
    let reference: Vec<f32> = reference
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let image = env::var_os("NARIQ_IMAGE_INPUT").map(PathBuf::from);
    let video = env::var_os("NARIQ_VIDEO_INPUT").map(PathBuf::from);
    assert!(
        image.is_none() || video.is_none(),
        "supply one media file per capture"
    );
    let fixed_image = (image.is_none() && video.is_none()).then(|| {
        let path = env::temp_dir().join(format!("nariq-image-{}.png", std::process::id()));
        image::RgbImage::from_pixel(64, 64, image::Rgb([255, 0, 0]))
            .save(&path)
            .unwrap();
        path
    });
    let prompt = if fixed_image.is_some() {
        "What color is the image? Answer with one word.".to_owned()
    } else {
        env::var("NARIQ_MEDIA_PROMPT")
            .expect("supply the question used to capture the media logits")
    };
    let mode = policy();
    let _scope = mode.configure_cli().unwrap();
    #[cfg(feature = "vulkan")]
    let gpu = (mode != ComputePolicy::Cpu)
        .then(rust_model_inference::ops::get_vulkan_context)
        .flatten();
    #[cfg(feature = "vulkan")]
    let before = gpu.map_or(0, |context| context.submission_count());
    let result = rust_model_inference::app::run_qwen35_family_multimodal_logits(
        &model,
        &projector,
        image.as_deref().or(fixed_image.as_deref()),
        video.as_deref(),
        None,
        &prompt,
        8,
        8,
        context,
        None,
        None,
    );
    if let Some(path) = fixed_image {
        fs::remove_file(path).unwrap();
    }
    let (logits, _) = result.unwrap();
    #[cfg(feature = "vulkan")]
    {
        let submissions = gpu.map_or(0, |context| context.submission_count() - before);
        eprintln!("image submissions={submissions}");
        if mode == ComputePolicy::Vulkan {
            assert!(submissions > 0);
        }
    }
    let Comparison {
        top,
        kl,
        rel_l2,
        max_abs,
    } = compare(&reference, &logits);
    eprintln!(
        "image compute={mode:?} top={top} kl={kl:.9} rel_l2={rel_l2:.8} max_abs={max_abs:.6}"
    );
    assert_eq!(serde_json::json!(top), oracle["top1"]);
    assert!(kl <= 1e-3, "image KL {kl}");
}
