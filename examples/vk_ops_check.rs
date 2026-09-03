#![cfg(feature = "vulkan")]

use rust_model_inference::ops::float::enable_gpu;
use rust_model_inference::ops::get_vulkan_context;
use rust_model_inference::vulkan::run_qwen3_operator_check;
use std::process::ExitCode;

fn main() -> ExitCode {
    enable_gpu();
    let Some(context) = get_vulkan_context() else {
        eprintln!("no vulkan context");
        return ExitCode::FAILURE;
    };

    match run_qwen3_operator_check(context) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("Vulkan operator check failed: {error}");
            ExitCode::FAILURE
        }
    }
}
