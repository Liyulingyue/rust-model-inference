use rust_model_inference::compute::ComputePolicy;
use rust_model_inference::models::qwen35::Qwen35Session;
use rust_model_inference::vulkan::VulkanContext;
use rust_model_inference::{qwen_text_positions, ComputePool, GGUFLoader, Qwen35Model};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{io::Read, sync::Arc, time::Instant};

const PROMPT: [u32; 13] = [
    248045, 846, 198, 109266, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271,
];
const EXPECTED: [u32; 4] = [109266, 6115, 98691, 95815];

fn measure(
    session: &mut Qwen35Session<'_, '_>,
    context: &VulkanContext,
    forced: Option<&[u32]>,
) -> (Value, Vec<u32>) {
    let before = context.compute_stats();
    let wall = Instant::now();
    session.reset();
    let reset_ms = wall.elapsed().as_secs_f64() * 1000.0;
    let started = Instant::now();
    let mut logits = session
        .step_with_tokens(&PROMPT, &qwen_text_positions(PROMPT.len()))
        .unwrap();
    let prefill = started.elapsed().as_secs_f64();
    let after_prompt = context.submission_count();
    let decode_start = Instant::now();
    let mut ids = Vec::new();
    for step in 0..=32 {
        assert!(logits.iter().all(|x| x.is_finite()));
        let token = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .unwrap()
            .0 as u32;
        ids.push(token);
        if step < 32 {
            let input = forced.map_or(token, |tokens| tokens[step]);
            let p = session.next_position();
            logits = session.step_with_tokens(&[input], &[[p, p, p, 0]]).unwrap();
        }
    }
    let decode = decode_start.elapsed().as_secs_f64();
    let wall_ms = wall.elapsed().as_secs_f64() * 1000.0;
    let stats = context.compute_stats().since(before);
    (
        json!({"prefill_ms": prefill * 1000.0, "prefill_tps": PROMPT.len() as f64 / prefill,
        "decode_ms": decode * 1000.0, "decode_tps": 32.0 / decode, "wall_ms": wall_ms, "reset_ms": reset_ms,
        "prefill_submissions": after_prompt - before.submissions, "decode_submissions": context.submission_count() - after_prompt,
        "uploads": stats.static_uploads, "upload_bytes": stats.static_upload_bytes,
        "host_read_bytes": stats.host_read_bytes, "host_write_bytes": stats.host_write_bytes,
        "transfers": stats.transfer_submissions, "peak_allocation_bytes": stats.peak_allocation_bytes}),
        ids,
    )
}

fn main() {
    assert!(std::env::var_os("RMI_PARITY_TRACE").is_none());
    let path = std::env::var("RMI_HOLO4_MODEL").unwrap();
    let mut file = std::fs::File::open(&path).unwrap();
    let mut hash = Sha256::new();
    let mut buffer = vec![0; 1024 * 1024];
    loop {
        let n = file.read(&mut buffer).unwrap();
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    let sha = format!("{:x}", hash.finalize());
    assert_eq!(
        sha,
        "8071d3ed5f59c88282e6466add0b344b51d49c8c630ae8f93d673ac0911ab962"
    );
    let loaded = Instant::now();
    let source = GGUFLoader::from_file(&path).unwrap();
    let mut models: Vec<_> = (0..3)
        .map(|_| Qwen35Model::from_source(&source).unwrap())
        .collect();
    let model_load_ms = loaded.elapsed().as_secs_f64() * 1000.0;
    rust_model_inference::ops::enable_gpu();
    let context = rust_model_inference::ops::get_vulkan_context().expect("real Vulkan required");
    let configs = [
        ("cpu_b64", ComputePolicy::Cpu, 64),
        ("cpu_b1", ComputePolicy::Cpu, 1),
        ("vulkan_b1", ComputePolicy::Vulkan, 1),
    ];
    let pool = Arc::new(ComputePool::new(8));
    let mut sessions = Vec::new();
    for (model, &(name, policy, batch)) in models.iter_mut().zip(&configs) {
        let init = Instant::now();
        sessions.push(
            Qwen35Session::new_with_compute(model, 128, batch, Arc::clone(&pool), policy).unwrap(),
        );
        println!(
            "{}",
            json!({"kind":"init", "backend":name,"session_init_ms":init.elapsed().as_secs_f64()*1000.0})
        );
    }
    println!(
        "{}",
        json!({"kind":"metadata", "device":context.device_name(), "model_sha256":sha, "model_load_ms":model_load_ms,
        "prompt_ids":PROMPT,"capacity":128,"threads":8,"kv":"f32","decode_evaluations":32,"measured_rounds":5,
        "scope":"resident sessions; reset included in wall; fixed CPU teacher-forced decode inputs; EOS does not stop workload; hash scan precedes load"})
    );
    let (warm, forced) = measure(&mut sessions[0], context, None);
    assert_eq!(forced[..4], EXPECTED);
    assert_eq!(warm["prefill_submissions"], 0);
    assert_eq!(warm["decode_submissions"], 0);
    let mut expected = vec![forced.clone()];
    println!(
        "{}",
        json!({"kind":"warmup", "backend":configs[0].0,"metrics":warm,"top1_ids":forced})
    );
    for i in 1..3 {
        let (warm, ids) = measure(&mut sessions[i], context, Some(&forced));
        assert_eq!(ids[..4], EXPECTED);
        expected.push(ids.clone());
        println!(
            "{}",
            json!({"kind":"warmup", "backend":configs[i].0,"metrics":warm,"top1_ids":ids,
            "top1_matches_cpu":ids==forced})
        );
    }
    for round in 0..5 {
        let order = if round % 2 == 0 { [0, 2, 1] } else { [1, 2, 0] };
        for i in order {
            let (sample, ids) = measure(&mut sessions[i], context, Some(&forced));
            assert_eq!(ids, expected[i], "same-backend greedy drift");
            let submissions = sample["prefill_submissions"].as_u64().unwrap()
                + sample["decode_submissions"].as_u64().unwrap();
            assert_eq!(submissions > 0, i == 2, "unexpected compute backend");
            println!(
                "{}",
                json!({"kind":"sample", "round":round+1, "backend":configs[i].0,"metrics":sample,"top1_matches_cpu":ids==forced})
            );
        }
    }
}
