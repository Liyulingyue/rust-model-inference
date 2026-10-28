//! Unlimited-OCR-2: original safetensors, 1024px base views, native CPU/F32 compute.
mod decoder;
mod sparse;
mod vision;

use crate::app::CliOptions;
use crate::core::tensor::TensorSource;
use crate::core::thread_pool::ComputePool;
use crate::format::safetensors::SafetensorSource;
use crate::models::diffusion::dreamx::kernels::{load_float_values, Linear};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

pub const MODEL_TYPE: &str = "unlimited-ocr";
const HIDDEN: usize = 1280;
const VOCAB: usize = 129280;
const IMAGE: u32 = 128815;
const PAGE: u32 = 128827;
const VISUAL_TOKENS: usize = 273;

pub fn matches_directory(root: &Path) -> bool {
    root.is_dir()
        && std::fs::read(root.join("config.json"))
            .ok()
            .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok())
            .is_some_and(|cfg| cfg["model_type"] == MODEL_TYPE)
}

pub fn validate_config(cfg: &Value) -> Result<(), String> {
    let expected = json!({
        "model_type":"unlimited-ocr", "architectures":["UnlimitedOCRForCausalLM"],
        "hidden_size":1280,"num_hidden_layers":12,"num_attention_heads":10,
        "num_key_value_heads":10,"vocab_size":129280,"intermediate_size":6848,
        "moe_intermediate_size":896,"n_routed_experts":64,"n_shared_experts":2,
        "num_experts_per_tok":6,"first_k_dense_replace":1,"moe_layer_freq":1,
        "use_mla":false,"attention_bias":false,"hidden_act":"silu",
        "rms_norm_eps":0.000001,"norm_topk_prob":false,"routed_scaling_factor":1.0,
        "scoring_func":"softmax","topk_method":"greedy","tie_word_embeddings":false,
        "bos_token_id":0,"eos_token_id":1,"pad_token_id":2
    });
    for (key, value) in expected.as_object().unwrap() {
        require_value(&cfg[key], value, key)?;
    }
    let sp = &cfg["sparse_attention_config"];
    let expected = json!({
        "training_stage":"dash","use_sparse_attention":true,"dash_stage0":"head_cls",
        "sparse_block_size":32,"sparse_local_block":4,"sparse_index_dim":128,
        "sparse_num_index_heads":1,"dash_topk_blocks":12,"dash_scaling_factor":128.0,
        "dash_prior_floor":0.0001,"dash_sigma":1.0,"dash_image_only":true,
        "image_bucket_constraint":true,"image_token_id":128815,"page_token_id":128827,
        "per_page_image_tokens":273,"sparse_rope_mode":"page2d",
        "sparse_main_rope_mode":"page2d","sparse_rope_tok_freq_mode":"resample",
        "sparse_main_rope_tok_freq_mode":"resample","sparse_rope_tok_theta":10000.0,
        "sparse_rope_page_theta":256.0,"sparse_rope_page_pairs":1,
        "sparse_rope_page_mod":2,"sparse_rope_page_span":0,
        "sparse_attention_freq":[1,1,1,1,1,1,1,1,1,1,1,1],
        "dash_scale_per_layer":[128.0,128.0,128.0,128.0,128.0,128.0,128.0,128.0,128.0,128.0,128.0,128.0],
        "dash_floor_per_layer":[0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001]
    });
    for (key, value) in expected.as_object().unwrap() {
        require_value(&sp[key], value, &format!("sparse_attention_config.{key}"))?;
    }
    if cfg["vision_config"]["model_name"] != "deeplip_b_l"
        || cfg["vision_config"]["image_size"] != 1024
        || cfg["projector_config"]["projector_type"] != "linear"
        || cfg["projector_config"]["input_dim"] != 2048
        || cfg["projector_config"]["n_embed"] != HIDDEN
    {
        return Err("Unsupported Unlimited-OCR vision/projector config".into());
    }
    Ok(())
}

fn require_value(actual: &Value, expected: &Value, name: &str) -> Result<(), String> {
    let equal = if expected.is_number() {
        actual.as_f64() == expected.as_f64()
    } else if let (Some(a), Some(b)) = (actual.as_array(), expected.as_array()) {
        a.len() == b.len()
            && a.iter()
                .zip(b)
                .all(|(a, b)| require_value(a, b, name).is_ok())
    } else {
        actual == expected
    };
    if equal {
        Ok(())
    } else {
        Err(format!(
            "Unsupported Unlimited-OCR {name}: {actual}; expected {expected}"
        ))
    }
}

fn vector(source: &dyn TensorSource, name: &str, width: usize) -> Result<Vec<f32>, String> {
    load_float_values(source, name, &[width as u64])
}

fn linear<'a>(
    source: &'a dyn TensorSource,
    name: &str,
    input: usize,
    output: usize,
    bias: bool,
) -> Result<Linear<'a>, String> {
    Linear::from_source(
        source,
        &format!("{name}.weight"),
        bias.then(|| format!("{name}.bias")).as_deref(),
        input,
        output,
    )
}

fn trace(name: &str, layer: Option<usize>, shape: &[usize], values: &[f32]) {
    #[cfg(feature = "parity-trace")]
    crate::parity_trace::report(crate::parity_trace::checkpoint(name, layer, shape, values));
    #[cfg(not(feature = "parity-trace"))]
    let _ = (name, layer, shape, values);
}

/// Base-mode prompt, matching the official `infer(..., crop_mode=False)` path.
pub fn prepare_tokens(
    tokenizer: &tokenizers::Tokenizer,
    pages: usize,
    prompt: &str,
) -> Result<Vec<u32>, String> {
    if pages == 0 || pages > 128 {
        return Err("Unlimited-OCR expects 1..=128 pages per bounded-context request".into());
    }
    if prompt.contains("<image>") || prompt.contains("<PAGE>") {
        return Err("Unlimited-OCR --prompt is instruction text; page/image markers are inserted automatically".into());
    }
    for (text, id) in [("<image>", IMAGE), ("<PAGE>", PAGE)] {
        if tokenizer.token_to_id(text) != Some(id) {
            return Err(format!("Invalid Unlimited-OCR tokenizer token {text}"));
        }
    }
    let mut ids = vec![0];
    for _ in 0..pages {
        ids.push(PAGE);
        ids.extend(std::iter::repeat_n(IMAGE, VISUAL_TOKENS));
    }
    ids.extend(
        tokenizer
            .encode(
                if prompt.is_empty() {
                    "Multi page parsing."
                } else {
                    prompt
                },
                false,
            )
            .map_err(|e| format!("Unlimited-OCR tokenizer: {e}"))?
            .get_ids(),
    );
    Ok(ids)
}

pub fn run_cli(options: &CliOptions, threads: usize) -> Result<(), String> {
    let image = options
        .image
        .as_ref()
        .ok_or("Unlimited-OCR requires --image <first-page> [--reference <next-page> ...]")?;
    if options.compute_policy() != crate::compute::ComputePolicy::Cpu
        || options.gpu
        || options.embedding
        || options.tts
        || options.audio.is_some()
        || options.video.is_some()
        || options.mmproj.is_some()
        || options.rerank
        || options.jev
        || options.yue2
        || options.dreamx
        || options.astronex
        || options.astronex_frames.is_some()
        || options.planner.is_some()
        || options.perception.is_some()
        || options.laya_request.is_some()
        || options.timestral_request.is_some()
        || options.longcat_kind.is_some()
        || options.components.is_some()
        || options.text_encoder.is_some()
        || options.vae.is_some()
        || options.qwen_image_turbo
    {
        return Err(
            "Unlimited-OCR currently supports CPU OCR through --image/--reference only".into(),
        );
    }
    if options.temperature.is_some_and(|x| x != 0.0)
        || options.repetition_penalty.is_some_and(|x| x != 1.0)
        || options.top_k.is_some()
        || options.top_p.is_some()
    {
        return Err("Unlimited-OCR currently requires greedy decoding (--temp 0, --repetition-penalty 1, no --top-k/--top-p)".into());
    }
    let cfg: Value = serde_json::from_slice(
        &std::fs::read(options.model.join("config.json")).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    validate_config(&cfg)?;
    let tokenizer = tokenizers::Tokenizer::from_file(options.model.join("tokenizer.json"))
        .map_err(|e| e.to_string())?;
    let pages: Vec<PathBuf> = std::iter::once(image.clone())
        .chain(options.references.iter().cloned())
        .collect();
    let ids = prepare_tokens(
        &tokenizer,
        pages.len(),
        options.prompt.as_deref().unwrap_or(""),
    )?;
    let max_context = options.max_context.unwrap_or(8192);
    let max_tokens = options.max_tokens.unwrap_or(2048);
    if max_context > 131072
        || max_tokens == 0
        || ids
            .len()
            .checked_add(max_tokens)
            .is_none_or(|n| n > max_context)
    {
        return Err(
            "Unlimited-OCR prompt + --max-tokens must fit --max-context (maximum 131072)".into(),
        );
    }
    let source = SafetensorSource::open(&[
        options.model.join("model-00001-of-00002.safetensors"),
        options.model.join("model-00002-of-00002.safetensors"),
    ])?;
    let pool = ComputePool::new(threads);
    let decoder = decoder::Decoder::load(&source)?;
    let vision = vision::Vision::load(&source)?;
    let embed = linear(&source, "model.embed_tokens", HIDDEN, VOCAB, false)?;
    let mut hidden = vec![0.0; ids.len() * HIDDEN];
    for (row, &id) in hidden.chunks_exact_mut(HIDDEN).zip(&ids) {
        embed.embedding_lookup(id, row)?;
    }
    trace(
        "uocr.token_ids",
        None,
        &[ids.len()],
        &ids.iter().map(|x| *x as f32).collect::<Vec<_>>(),
    );
    for (page, path) in pages.iter().enumerate() {
        eprintln!("[unlimited-ocr] encode page {}/{}", page + 1, pages.len());
        let pixels = vision::preprocess(path)?;
        trace("uocr.pixels", Some(page), &[1024, 1024, 3], &pixels);
        let features = vision.encode(&pool, &pixels)?;
        let start = (2 + page * (VISUAL_TOKENS + 1)) * HIDDEN;
        hidden[start..start + features.len()].copy_from_slice(&features);
    }
    let generated = decoder.generate(
        &pool,
        &ids,
        hidden,
        max_tokens,
        pages.len(),
        options.dump_logits,
    )?;
    let text = tokenizer
        .decode(&generated, false)
        .map_err(|e| e.to_string())?;
    if let Some(path) = options.out.as_ref() {
        crate::app::write_output_atomically(path, text.as_bytes(), options.overwrite)?;
    } else {
        println!("{text}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn checkpoint_contract_rejects_nearby_architectures_and_changed_math() {
        let cfg: Value = serde_json::from_str(include_str!(
            "../../../tests/fixtures/unlimited_ocr_config.json"
        ))
        .unwrap();
        validate_config(&cfg).unwrap();
        for (key, value) in [
            ("model_type", json!("deepseek2")),
            ("norm_topk_prob", json!(true)),
            ("num_key_value_heads", json!(5)),
            ("hidden_size", json!(0)),
        ] {
            let mut wrong = cfg.clone();
            wrong[key] = value;
            assert!(validate_config(&wrong).is_err(), "{key}");
        }
        let mut wrong = cfg;
        wrong["sparse_attention_config"]["sparse_rope_page_mod"] = json!(0);
        assert!(validate_config(&wrong).is_err());
    }
    #[test]
    #[ignore = "requires RMI_UNLIMITED_OCR_MODEL"]
    fn real_tokenizer_preserves_markers_and_unicode() {
        let path = PathBuf::from(std::env::var("RMI_UNLIMITED_OCR_MODEL").unwrap());
        let tokenizer = tokenizers::Tokenizer::from_file(path.join("tokenizer.json")).unwrap();
        let ids = prepare_tokens(&tokenizer, 2, "Multi page parsing.").unwrap();
        assert_eq!(ids.len(), 553);
        assert_eq!(&ids[..2], &[0, PAGE]);
        assert_eq!(&ids[275..276], &[PAGE]);
        let fixture: Value = serde_json::from_str(include_str!(
            "../../../tests/fixtures/unlimited_ocr_oracle.json"
        ))
        .unwrap();
        for case in fixture["tokenizer"].as_array().unwrap() {
            let actual = tokenizer
                .encode(case["text"].as_str().unwrap(), false)
                .unwrap();
            let expected: Vec<_> = case["ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u32)
                .collect();
            assert_eq!(actual.get_ids(), expected);
        }
        assert!(prepare_tokens(&tokenizer, 0, "parse").is_err());
        assert!(prepare_tokens(&tokenizer, 1, "<image>").is_err());
    }

    #[test]
    #[ignore = "requires RMI_UNLIMITED_OCR_MODEL and RMI_UNLIMITED_OCR_ORACLE"]
    fn real_decoder_from_official_embeddings() {
        let root = PathBuf::from(std::env::var("RMI_UNLIMITED_OCR_MODEL").unwrap());
        let fixture = PathBuf::from(std::env::var("RMI_UNLIMITED_OCR_ORACLE").unwrap());
        let result: Value =
            serde_json::from_slice(&std::fs::read(fixture.join("result.json")).unwrap()).unwrap();
        let ids: Vec<_> = result["prompt_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let expected: Vec<_> = result["generated_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let checkpoints: Vec<Value> = std::fs::read_to_string(fixture.join("trace.jsonl"))
            .unwrap()
            .lines()
            .map(|v| serde_json::from_str(v).unwrap())
            .collect();
        let embedding = checkpoints
            .iter()
            .find(|v| v["name"] == "uocr.embeddings")
            .unwrap();
        let bytes = std::fs::read(embedding["binary_path"].as_str().unwrap()).unwrap();
        let hidden: Vec<_> = bytes
            .chunks_exact(4)
            .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
            .collect();
        assert_eq!(hidden.len(), ids.len() * HIDDEN);
        let source = SafetensorSource::open(&[
            root.join("model-00001-of-00002.safetensors"),
            root.join("model-00002-of-00002.safetensors"),
        ])
        .unwrap();
        let decoder = decoder::Decoder::load(&source).unwrap();
        crate::app::init_rayon_global_pool(8);
        trace(
            "uocr.token_ids",
            None,
            &[ids.len()],
            &ids.iter().map(|x| *x as f32).collect::<Vec<_>>(),
        );
        let generated = decoder
            .generate(
                &ComputePool::new(8),
                &ids,
                hidden,
                result["max_tokens"].as_u64().unwrap() as usize,
                ids.iter().filter(|&&id| id == PAGE).count(),
                false,
            )
            .unwrap();
        assert_eq!(generated, expected);
    }
}
