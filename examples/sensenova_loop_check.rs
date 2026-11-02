//! Official-oracle fixture runner; weights remain on the validation host.
use rust_model_inference::compute::ComputePolicy;
use rust_model_inference::core::{
    loader::GGUFLoader,
    thread_pool::ComputePool,
    tokenizer::{BPETokenizer, EncodeOptions},
};
use rust_model_inference::models::hybrid::HybridTextModel;
use rust_model_inference::models::qwen35::trunk::HybridSession;
use serde_json::{json, Value};
use std::sync::Arc;
fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    if !(5..=6).contains(&args.len()) {
        return Err("usage: sensenova_loop_check MODEL FIXTURES OUTPUT cpu|auto [MMPROJ]".into());
    }
    let source =
        GGUFLoader::from_file(std::path::Path::new(&args[1])).map_err(|e| e.to_string())?;
    let tokenizer = BPETokenizer::from_gguf_metadata(|k| source.metadata(k).cloned())?;
    let fixtures: Vec<Value> =
        serde_json::from_str(&std::fs::read_to_string(&args[2]).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    let policy = if args[4] == "auto" {
        ComputePolicy::Auto
    } else {
        ComputePolicy::Cpu
    };
    let _scope = policy.enter_legacy_scope();
    let mut model = HybridTextModel::from_source(&source)?;
    let mut results = Vec::new();
    for (case, fixture) in fixtures.iter().enumerate() {
        let ids: Vec<u32> =
            serde_json::from_value(fixture["token_ids"].clone()).map_err(|e| e.to_string())?;
        let options = EncodeOptions {
            add_special: false,
            parse_special: true,
        };
        let actual = tokenizer.encode(
            fixture["rendered"].as_str().ok_or("missing rendered")?,
            options,
        );
        if actual != ids {
            return Err(format!("case{case} rendered token IDs differ"));
        }
        let plain = tokenizer.encode(fixture["prompt"].as_str().ok_or("missing prompt")?, options);
        if serde_json::to_value(&plain).unwrap() != fixture["plain_ids"] {
            return Err(format!("case{case} plain token IDs differ"));
        }
        let trace = format!("{}/case{case}.jsonl", args[3]);
        std::fs::create_dir_all(&args[3]).map_err(|e| e.to_string())?;
        if std::path::Path::new(&trace).exists() {
            return Err(format!("refusing existing trace {trace}"));
        }
        std::env::set_var("RMI_PARITY_TRACE", trace);
        let mut image_embedding = None;
        let mut grid = None;
        if let Some(path) = fixture["image"].as_str() {
            let _vision_cpu = ComputePolicy::Cpu.enter_legacy_scope();
            use rust_model_inference::models::qwen35::vision::{
                qwen_smart_resize, VisionEncoder, VisionScratchpad,
            };
            let mmproj = GGUFLoader::from_file(std::path::Path::new(
                args.get(5).ok_or("image requires MMPROJ")?,
            ))
            .map_err(|e| e.to_string())?;
            let pool = Arc::new(ComputePool::new(16));
            let encoder = VisionEncoder::from_source(&mmproj, pool.clone())?;
            let image = image::open(path).map_err(|e| e.to_string())?.to_rgb8();
            let g = qwen_smart_resize(
                image.width() as usize,
                image.height() as usize,
                &encoder.config,
            )?;
            if (image.width() as usize, image.height() as usize)
                != (g.image_width(), g.image_height())
            {
                return Err("oracle fixture must already have the smart-resize dimensions".into());
            }
            let pixels: Vec<f32> = image
                .as_raw()
                .iter()
                .enumerate()
                .map(|(i, &p)| {
                    (p as f32 / 255. - encoder.config.image_mean[i % 3])
                        / encoder.config.image_std[i % 3]
                })
                .collect();
            let mut scratch = VisionScratchpad::new(&encoder.config);
            encoder.encode_image(
                &pixels,
                g.image_width(),
                g.image_height(),
                &mut scratch,
                &pool,
            )?;
            let mut embeddings = model.trunk().embed_tokens(&ids)?;
            let image_id = tokenizer
                .special_token_id("image_pad")
                .ok_or("missing image_pad")?;
            let mut projected = scratch.projected.chunks_exact(model.trunk().config.n_embd);
            for (&id, row) in ids
                .iter()
                .zip(embeddings.chunks_exact_mut(model.trunk().config.n_embd))
            {
                if id == image_id {
                    row.copy_from_slice(projected.next().ok_or("missing image projection")?);
                }
            }
            if projected.next().is_some() {
                return Err("extra image projection".into());
            }
            std::fs::write(
                format!("{}/vision.f32", args[3]),
                scratch
                    .projected
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .map_err(|e| e.to_string())?;
            image_embedding = Some(embeddings);
            grid = Some(g);
        }
        let (prompt_positions, mut next_position) =
            rust_model_inference::models::qwen35::build_qwen35_positions(
                &ids,
                grid.map(|_| tokenizer.special_token_id("image_pad").unwrap()),
                &grid.into_iter().collect::<Vec<_>>(),
            )?;
        let mut session = HybridSession::new_with_compute(
            &mut model,
            ids.len() + 16,
            64,
            Arc::new(ComputePool::new(16)),
            policy,
        )?;
        let mut input = ids;
        let mut generated = Vec::new();
        for (step, _) in fixture["steps"]
            .as_array()
            .ok_or("missing steps")?
            .iter()
            .enumerate()
        {
            let positions = if step == 0 {
                prompt_positions.clone()
            } else {
                let p = next_position;
                next_position += 1;
                vec![[p, p, p, 0]]
            };
            let logits = if step == 0 && image_embedding.is_some() {
                session.step(image_embedding.as_ref().unwrap(), input.len(), &positions)?
            } else {
                session.step_with_tokens(&input, &positions)?
            };
            let id = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .ok_or("empty logits")?
                .0 as u32;
            generated.push(id);
            input = vec![id];
        }
        results.push(json!({"case":case,"generated":generated}));
        println!("case={case} tokens={generated:?}");
    }
    std::fs::write(
        format!("{}/results.json", args[3]),
        serde_json::to_vec_pretty(&results).unwrap(),
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}
