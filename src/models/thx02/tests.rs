use super::*;
use serde_json::json;

/// Reproducible CPU/device probe without adding a production binary or diagnostic API.
#[test]
#[ignore = "requires THX02_MODEL, THX02_REQUEST and THX02_OUTPUT"]
fn real_weight_readouts() {
    let source: Arc<dyn TensorSource> = Arc::new(
        crate::core::loader::GGUFLoader::from_file(std::env::var("THX02_MODEL").unwrap()).unwrap(),
    );
    let request: Value =
        serde_json::from_slice(&std::fs::read(std::env::var("THX02_REQUEST").unwrap()).unwrap())
            .unwrap();
    let requests = if let Value::Array(a) = request {
        a
    } else {
        vec![request]
    };
    let threads = std::env::var("THX02_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global();
    let compute = match std::env::var("THX02_COMPUTE").as_deref() {
        Ok("auto") => ComputePolicy::Auto,
        Ok("vulkan") => ComputePolicy::Vulkan,
        _ => ComputePolicy::Cpu,
    };
    let engine = Engine::new(
        source,
        None,
        Arc::new(ComputePool::new(threads)),
        8192,
        compute,
    )
    .unwrap();
    let _scope = compute.configure_cli().unwrap();
    let model = weights::load(
        engine.source.as_ref(),
        weights::config(engine.source.as_ref(), false).unwrap(),
    )
    .unwrap();
    let mut outputs = Vec::new();
    for request in requests {
        validate_request(&request).unwrap();
        #[cfg(feature = "vulkan")]
        let device = compute.context().unwrap();
        #[cfg(feature = "vulkan")]
        let before = device.map(|d| d.compute_stats());
        let start = std::time::Instant::now();
        let images = input::open_images(&request).unwrap();
        let visual = input::Visual::encode(engine.source.as_ref(), &images, &engine.pool).unwrap();
        let enc = engine
            .encoder
            .encode(&request, &visual.grids, false)
            .unwrap();
        let mut x = model.embed_tokens(&enc.ids).unwrap();
        visual
            .inject(&enc.ids, engine.encoder.image, &mut x)
            .unwrap();
        let hidden = forward::Session::new(&model.config, enc.ids.len(), false)
            .run(&model, x, &enc.positions, &engine.pool)
            .unwrap();
        let readouts = enc
            .markers
            .iter()
            .flat_map(|&i| hidden[i * 2560..(i + 1) * 2560].iter().copied())
            .collect::<Vec<_>>();
        let logits =
            forward::linear(model.cls_score.as_ref().unwrap(), &readouts, &engine.pool).unwrap();
        let elapsed = start.elapsed().as_secs_f64();
        #[cfg(feature = "vulkan")]
        let device_stats = device.zip(before).map(|(d,b)| {
            let s=d.compute_stats().since(b);
            if compute==ComputePolicy::Vulkan { assert!(s.submissions>0,"no GPU execution"); }
            json!({"submissions":s.submissions,"static_uploads":s.static_uploads,"host_read_bytes":s.host_read_bytes,"host_write_bytes":s.host_write_bytes})
        });
        #[cfg(not(feature = "vulkan"))]
        let device_stats = None::<Value>;
        let mut answers = serde_json::Map::new();
        for ((key, q), row) in request["questions"]
            .as_object()
            .unwrap()
            .iter()
            .zip(logits.chunks_exact(255))
        {
            let mut p = row[..protocol::options(q).unwrap().len()]
                .iter()
                .map(|&x| x as f64 / engine.temperature)
                .collect::<Vec<_>>();
            let max = p.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            for v in &mut p {
                *v = (*v - max).exp();
            }
            answers.insert(key.clone(), protocol::answer(q, &p).unwrap());
        }
        eprintln!(
            "THX-02 probe case={} tokens={} seconds={elapsed:.3}",
            outputs.len(),
            enc.ids.len()
        );
        outputs.push(json!({"request":request,"prompt":protocol::prompt(&request,&engine.encoder.codes,&engine.encoder.escapes,false).unwrap(),"input_ids":enc.ids,"positions":enc.positions,"markers":enc.markers,"logits":logits.chunks_exact(255).collect::<Vec<_>>(),"answers":answers,"seconds":elapsed,"threads":threads,"compute":format!("{compute:?}"),"device_stats":device_stats}));
        std::fs::write(
            std::env::var("THX02_OUTPUT").unwrap(),
            serde_json::to_vec_pretty(&outputs).unwrap(),
        )
        .unwrap();
    }
    std::fs::write(
        std::env::var("THX02_OUTPUT").unwrap(),
        serde_json::to_vec_pretty(&outputs).unwrap(),
    )
    .unwrap();
}

#[test]
#[ignore = "requires THX02_THINKER, THX02_REQUEST and THX02_OUTPUT"]
fn real_weight_thinker() {
    let source =
        crate::core::loader::GGUFLoader::from_file(std::env::var("THX02_THINKER").unwrap())
            .unwrap();
    let request: Value =
        serde_json::from_slice(&std::fs::read(std::env::var("THX02_REQUEST").unwrap()).unwrap())
            .unwrap();
    validate_request(&request).unwrap();
    let threads = std::env::var("THX02_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global();
    let compute = match std::env::var("THX02_COMPUTE").as_deref() {
        Ok("auto") => ComputePolicy::Auto,
        Ok("vulkan") => ComputePolicy::Vulkan,
        _ => ComputePolicy::Cpu,
    };
    let _scope = compute.configure_cli().unwrap();
    #[cfg(feature = "vulkan")]
    let device = compute.context().unwrap();
    #[cfg(feature = "vulkan")]
    let before = device.map(|d| d.compute_stats());
    let pool = Arc::new(ComputePool::new(threads));
    let encoder = input::Encoder::new(&source).unwrap();
    let model = weights::load(&source, weights::config(&source, true).unwrap()).unwrap();
    let images = input::open_images(&request).unwrap();
    let visual = input::Visual::encode(&source, &images, &pool).unwrap();
    let encoded = encoder.encode(&request, &visual.grids, true).unwrap();
    let mut session = forward::Session::new(&model.config, encoded.ids.len(), true);
    let mut embeddings = model.embed_tokens(&encoded.ids).unwrap();
    visual
        .inject(&encoded.ids, encoder.image, &mut embeddings)
        .unwrap();
    let start = std::time::Instant::now();
    let mut hidden = session
        .run(&model, embeddings, &encoded.positions, &pool)
        .unwrap();
    let prefill_seconds = start.elapsed().as_secs_f64();
    let mut forward_seconds = prefill_seconds;
    let count = std::env::var("THX02_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let mut steps = Vec::new();
    let mut generated = Vec::new();
    for step in 0..count {
        let start = std::time::Instant::now();
        let logits = forward::linear(
            &model.output_weight,
            &hidden[hidden.len() - model.config.n_embd..],
            &pool,
        )
        .unwrap();
        assert!(logits.iter().all(|v| v.is_finite()));
        let best = (1..logits.len())
            .fold(0, |best, i| if logits[i] > logits[best] { i } else { best })
            as u32;
        generated.push(best);
        steps.push(json!({"token_id":best,"logits":logits,"forward_seconds":forward_seconds,"output_projection_seconds":start.elapsed().as_secs_f64()}));
        if ["</think>", "<|im_end|>", "<|endoftext|>"]
            .iter()
            .any(|s| encoder.tokenizer.token_to_id(s) == Some(best))
            || step + 1 == count
        {
            break;
        }
        let pos = encoded.next + step;
        let start = std::time::Instant::now();
        hidden = session
            .run(
                &model,
                model.embed_tokens(&[best]).unwrap(),
                &[[pos, pos, pos, 0]],
                &pool,
            )
            .unwrap();
        forward_seconds = start.elapsed().as_secs_f64();
    }
    #[cfg(feature = "vulkan")]
    let device_stats = device.zip(before).map(|(d,b)| {
        let s=d.compute_stats().since(b);
        if compute==ComputePolicy::Vulkan { assert!(s.submissions>0,"no GPU execution"); }
        json!({"submissions":s.submissions,"static_uploads":s.static_uploads,"host_read_bytes":s.host_read_bytes,"host_write_bytes":s.host_write_bytes})
    });
    #[cfg(not(feature = "vulkan"))]
    let device_stats = None::<Value>;
    std::fs::write(std::env::var("THX02_OUTPUT").unwrap(), serde_json::to_vec(&json!({
        "prompt":protocol::prompt(&request,&encoder.codes,&encoder.escapes,true).unwrap(),
        "input_ids":encoded.ids,"positions":encoded.positions,"generated_ids":generated,
        "steps":steps,"prefill_seconds":prefill_seconds,"threads":threads,"compute":format!("{compute:?}"),"device_stats":device_stats
    })).unwrap()).unwrap();
}
