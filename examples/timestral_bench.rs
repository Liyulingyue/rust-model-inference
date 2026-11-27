//! End-to-end and resident-model timings; no parity-trace feature is needed.
use rust_model_inference::app::init_rayon_global_pool;
use rust_model_inference::compute::ComputePolicy;
use rust_model_inference::models::timestral::{ForecastRequest, TimestralModel};
use rust_model_inference::{open_model_source, ComponentRole};
use sha2::{Digest, Sha256};
use std::{path::Path, time::Instant};

fn main() -> Result<(), String> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 6 {
        return Err(
            "usage: timestral_bench MODEL REQUEST cpu|vulkan|auto THREADS REPEATS OUTPUT.json"
                .into(),
        );
    }
    let compute = match args[2].as_str() {
        "cpu" => ComputePolicy::Cpu,
        "vulkan" => ComputePolicy::Vulkan,
        "auto" => ComputePolicy::Auto,
        _ => return Err("invalid compute policy".into()),
    };
    let threads: usize = args[3].parse().map_err(|_| "invalid threads")?;
    let repeats: usize = args[4].parse().map_err(|_| "invalid repeats")?;
    if threads == 0 || repeats == 0 {
        return Err("threads and repeats must be positive".into());
    }
    let started = Instant::now();
    let _scope = compute.configure_cli()?;
    init_rayon_global_pool(threads);
    let request: ForecastRequest =
        serde_json::from_slice(&std::fs::read(&args[1]).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    let setup_ms = started.elapsed().as_secs_f64() * 1000.;
    let step = Instant::now();
    let source =
        open_model_source(Path::new(&args[0]), ComponentRole::Llm).map_err(|e| e.to_string())?;
    let source_ms = step.elapsed().as_secs_f64() * 1000.;
    let step = Instant::now();
    let model = TimestralModel::from_source_with_compute(source.as_ref(), compute)?;
    let validate_ms = step.elapsed().as_secs_f64() * 1000.;
    let mut forecast_ms = Vec::with_capacity(repeats + 1);
    let mut serialize_ms = Vec::with_capacity(repeats + 1);
    let mut expected = Vec::new();
    for run in 0..=repeats {
        eprintln!("benchmark run={run} begin");
        let step = Instant::now();
        let result = model.forecast(&request)?;
        forecast_ms.push(step.elapsed().as_secs_f64() * 1000.);
        let step = Instant::now();
        let bytes = serde_json::to_vec(&result).map_err(|e| e.to_string())?;
        serialize_ms.push(step.elapsed().as_secs_f64() * 1000.);
        if run == 0 {
            expected = bytes;
        } else if bytes != expected {
            return Err("repeated forecasts are not identical".into());
        }
        eprintln!("benchmark run={run} end");
    }
    std::fs::write(&args[5], &expected).map_err(|e| e.to_string())?;
    println!(
        "{}",
        serde_json::json!({
            "policy": args[2], "threads": threads,
            "setup_ms": setup_ms, "source_open_ms": source_ms,
            "tensor_validation_ms": validate_ms,
            "first_forecast_ms": forecast_ms[0],
            "warm_forecast_ms": &forecast_ms[1..],
            "serialize_ms": serialize_ms,
            "output_bytes": expected.len(),
            "output_sha256": format!("{:x}", Sha256::digest(&expected)),
            "output_path": args[5],
            "stable_repeated_output": true,
        })
    );
    Ok(())
}
