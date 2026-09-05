#![cfg(feature = "vulkan")]

use rust_model_inference::ops::float::enable_gpu;
use rust_model_inference::ops::get_vulkan_context;
use rust_model_inference::vulkan::run_qwen3_operator_check;
use std::process::ExitCode;

fn main() -> ExitCode {
    let formats = match formats() {
        Ok(formats) => formats,
        Err(error) => {
            eprintln!("vk_ops_check failed: {error}");
            return ExitCode::FAILURE;
        }
    };
    enable_gpu();
    let Some(context) = get_vulkan_context() else {
        eprintln!("no vulkan context");
        return ExitCode::FAILURE;
    };

    let formats = formats.iter().map(String::as_str).collect::<Vec<_>>();
    match run_qwen3_operator_check(context, &formats) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("Vulkan operator check failed: {error}");
            ExitCode::FAILURE
        }
    }
}

fn formats() -> Result<Vec<String>, String> {
    let mut args = std::env::args().skip(1);
    let Some(flag) = args.next() else {
        return Ok(Vec::new());
    };
    let value = args
        .next()
        .ok_or("--formats needs a comma-separated list")?;
    if flag != "--formats" || args.next().is_some() {
        return Err("usage: vk_ops_check [--formats q4_0,q4_1,q4_k,q5_k,q6_k,f16,bf16]".into());
    }
    Ok(value
        .split(',')
        .filter(|format| !format.is_empty())
        .map(str::to_owned)
        .collect())
}
