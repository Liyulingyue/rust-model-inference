#![cfg(feature = "vulkan")]

use rust_model_inference::ops::float::enable_gpu;
use rust_model_inference::ops::get_vulkan_context;
use rust_model_inference::vulkan::{run_batched_matmul_check, run_qwen3_operator_check};
use std::process::ExitCode;

fn main() -> ExitCode {
    let (formats, rows) = match parse_options(std::env::args().skip(1)) {
        Ok(options) => options,
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
    // Run row parity first so a failure in the existing CPU parity suite does
    // not prevent checking the new recorder on this device.
    let row_formats = if formats.is_empty() {
        vec!["q8_0"]
    } else {
        formats.clone()
    };
    if let Err(error) = run_batched_matmul_check(context, &row_formats, rows) {
        eprintln!("Vulkan row operator check failed: {error}");
        return ExitCode::FAILURE;
    }
    // q8_0 is checked in both suites now that the matvec operator test
    // covers it: `run_batched_matmul_check` exercises the row recorder and
    // `run_qwen3_operator_check` exercises the legacy single-row matvec.
    let legacy_formats = formats;
    match run_qwen3_operator_check(context, &legacy_formats) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("Vulkan operator check failed: {error}");
            ExitCode::FAILURE
        }
    }
}

fn parse_options(
    args: impl Iterator<Item = impl AsRef<str>>,
) -> Result<(Vec<String>, usize), String> {
    let mut args = args;
    let mut formats = Vec::new();
    let mut rows = 1;
    while let Some(flag) = args.next() {
        match flag.as_ref() {
            "--all-formats" => {
                formats = [
                    "q8_0", "q4_0", "q4_1", "q4_k", "q5_k", "q6_k", "f16", "bf16", "f32",
                ]
                .into_iter()
                .map(str::to_owned)
                .collect()
            }
            "--formats" => {
                let value = args
                    .next()
                    .ok_or("--formats needs a comma-separated list")?;
                formats = value
                    .as_ref()
                    .split(',')
                    .filter(|format| !format.is_empty())
                    .map(str::to_owned)
                    .collect();
            }
            "--rows" => {
                let value = args.next().ok_or("--rows needs a positive integer")?;
                rows = value
                    .as_ref()
                    .parse::<usize>()
                    .ok()
                    .filter(|&rows| rows > 0)
                    .ok_or("--rows needs a positive integer")?;
            }
            _ => {
                return Err("usage: vk_ops_check [--all-formats|--formats list] [--rows N]".into())
            }
        }
    }
    Ok((formats, rows))
}

#[cfg(test)]
mod tests {
    #[test]
    fn rows_option_reaches_operator_check_options() {
        let (formats, rows) =
            super::parse_options(["--all-formats", "--rows", "3"].into_iter()).unwrap();
        assert_eq!(rows, 3);
        assert_eq!(formats.len(), 9);
        assert!(formats.iter().any(|format| format == "q8_0"));
        assert!(super::parse_options(["--rows", "0"].into_iter()).is_err());
    }
}
