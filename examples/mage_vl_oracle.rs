//! Replay reference.py fixtures through native kernels, including cache continuation.
use rust_model_inference::{models::mage_vl, models::qwen3::*, *};
use std::{path::Path, sync::Arc};

fn save(path: impl AsRef<Path>, values: &[f32]) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::write(
        path,
        values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if !(5..=6).contains(&args.len()) {
        return Err(
            "usage: mage_vl_oracle TEXT.gguf VISION.gguf CASE_DIR OUT_DIR [f32|f16]".into(),
        );
    }
    let kv_format = match args.get(5).map(String::as_str).unwrap_or("f32") {
        "f32" => core::scratchpad::KvFormat::F32,
        "f16" => core::scratchpad::KvFormat::F16,
        _ => return Err("expected f32 or f16 KV".into()),
    };
    let root = Path::new(&args[3]);
    let out = Path::new(&args[4]);
    std::fs::create_dir_all(out)?;
    let case: serde_json::Value = serde_json::from_slice(&std::fs::read(root.join("case.json"))?)?;
    let source: Arc<dyn TensorSource> =
        Arc::from(open_model_source(Path::new(&args[1]), ComponentRole::Llm)?);
    let vision = open_model_source(Path::new(&args[2]), ComponentRole::Mmproj)?;
    mage_vl::validate_components(source.as_ref(), vision.as_ref())?;
    let tokenizer = Arc::new(BPETokenizer::from_gguf_metadata(|k| {
        source.metadata(k).cloned()
    })?);
    let template = prompt::jinja::template_from_source(&|k| source.metadata(k).cloned())
        .ok_or("missing template")??;
    let raw = mage_vl::prompt_tokens(
        &tokenizer,
        &template,
        &case["messages"],
        &case["tools"],
        case["thinking"].as_bool().unwrap(),
    )?;
    assert_eq!(
        serde_json::to_value(&raw)?,
        case["raw_ids"],
        "chat template/tokenizer mismatch"
    );
    let samples: serde_json::Value = serde_json::from_slice(&std::fs::read(
        root.parent().unwrap().join("tokenizer.json"),
    )?)?;
    for sample in samples.as_array().unwrap() {
        let ids = tokenizer.encode(
            sample["text"].as_str().unwrap(),
            EncodeOptions {
                add_special: false,
                parse_special: true,
            },
        );
        assert_eq!(
            serde_json::to_value(ids)?,
            sample["ids"],
            "tokenizer sample mismatch"
        );
    }
    let pool = Arc::new(ComputePool::new(8));
    let templates: serde_json::Value = serde_json::from_slice(&std::fs::read(
        root.parent().unwrap().join("templates.json"),
    )?)?;
    for case in templates.as_array().unwrap() {
        let ids = mage_vl::prompt_tokens(
            &tokenizer,
            &template,
            &case["messages"],
            &case["tools"],
            case["thinking"].as_bool().unwrap(),
        )?;
        assert_eq!(
            serde_json::to_value(ids)?,
            case["ids"],
            "multi-turn/tool template mismatch"
        );
    }
    let model = Qwen3Model::from_source(source, tokenizer, Arc::clone(&pool))?;
    let images = case["images"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| image::open(root.join(p.as_str().unwrap())))
        .collect::<Result<Vec<_>, _>>()?;
    let (tokens, embeddings, positions) = if images.is_empty() {
        let positions = qwen_text_positions(raw.len());
        (raw, None, positions)
    } else {
        std::env::set_var("RMI_MAGE_TRACE_DIR", out);
        let prepared = mage_vl::prepare(
            &model,
            vision.as_ref(),
            &raw,
            &images,
            &pool,
            model.config().n_ctx,
        )?;
        (
            prepared.tokens,
            Some(prepared.embeddings),
            prepared.positions,
        )
    };
    assert_eq!(
        serde_json::to_value(&tokens)?,
        case["input_ids"],
        "expanded image IDs mismatch"
    );
    let steps = case["steps"].as_u64().unwrap() as usize;
    #[cfg(feature = "parity-trace")]
    {
        let trace = out.join("layers.jsonl");
        if trace.exists() {
            std::fs::remove_file(&trace)?;
        }
        std::env::set_var("RMI_PARITY_TRACE", &trace);
        std::env::set_var(
            "RMI_PARITY_FILTER",
            "hidden_sequence.layer.0,hidden_sequence.layer.17,hidden_sequence.layer.35",
        );
        let mut hidden = Qwen3Session::new_with_kv_state(
            &model,
            tokens.len(),
            core::scratchpad::KvFormat::F32,
            KvLifecycle::Ephemeral,
        )?;
        let rows = hidden.forward_hidden_sequence(Qwen3Input {
            token_ids: &tokens,
            positions: &positions,
            embeddings: embeddings.as_deref(),
            deepstack_embeddings: None,
        })?;
        save(out.join("norm.f32"), &rows[rows.len() - mage_vl::WIDTH..])?;
        for layer in [0, 17, 35] {
            std::fs::copy(
                format!(
                    "{}.hidden_sequence.layer.{layer}.{}.f32",
                    trace.display(),
                    tokens.len() - 1
                ),
                out.join(format!("lm{layer}.f32")),
            )?;
        }
        std::env::remove_var("RMI_PARITY_TRACE");
        std::env::remove_var("RMI_PARITY_FILTER");
    }
    let mut session = Qwen3Session::new_with_kv_state(
        &model,
        tokens.len() + steps + 1,
        kv_format,
        KvLifecycle::Ephemeral,
    )?;
    let (mut logits, _) = session.forward_logits(
        Qwen3Input {
            token_ids: &tokens,
            positions: &positions,
            embeddings: embeddings.as_deref(),
            deepstack_embeddings: None,
        },
        32,
    )?;
    let mut generated = Vec::new();
    for step in 0..steps {
        save(out.join(format!("logits{step}.f32")), &logits)?;
        let token = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0 as u32;
        generated.push(token);
        if step + 1 < steps {
            let position = tokens.len() + step;
            (logits, _) = session.forward_logits(
                Qwen3Input {
                    token_ids: &[token],
                    positions: &[[position, position, position, 0]],
                    embeddings: None,
                    deepstack_embeddings: None,
                },
                1,
            )?;
        }
    }
    std::fs::write(out.join("greedy.json"), serde_json::to_vec(&generated)?)?;
    println!("tokens={} greedy={generated:?}", tokens.len());
    Ok(())
}
