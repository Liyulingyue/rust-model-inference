use crate::core::tensor::TensorSource;
use crate::core::thread_pool::ComputePool;
use crate::models::diffusion::qwen_image_2_1::{matches_signature, validate_dit, QwenImage21Dit};
use std::io::{Read, Write};
use std::path::Path;
use std::sync::Arc;

/// Reads raw little-endian f32 values from a validated file path.
pub fn read_f32_file(path: &Path) -> Result<Vec<f32>, String> {
    let resolved = path
        .canonicalize()
        .map_err(|error| format!("Resolve {}: {error}", path.display()))?;
    if !resolved.is_file() {
        return Err(format!("{} is not a file", resolved.display()));
    }
    let mut bytes = Vec::new();
    std::fs::File::open(&resolved)
        .and_then(|mut file| file.read_to_end(&mut bytes))
        .map_err(|error| format!("Read {}: {error}", resolved.display()))?;
    if bytes.len() % 4 != 0 {
        return Err(format!(
            "{} must hold whole f32 values ({} bytes)",
            resolved.display(),
            bytes.len()
        ));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

pub fn qwen_image_2_1_signature(source: &dyn TensorSource) -> bool {
    matches_signature(source)
}

/// Deterministic generator identical to the oracle harness (examples/
/// qwen_image_2_1_trace/main.cpp): LCG value = ((word >> 8) & 0xFFFF) / 32768 - 1.
pub fn next_deterministic_value(state: &mut u32) -> f32 {
    *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    let mantissa = (*state >> 8) & 0xffff;
    mantissa as f32 / 32_768.0 - 1.0
}

pub const QWEN_IMAGE_2_1_DEFAULT_LATENT: usize = 16;
pub const QWEN_IMAGE_2_1_DEFAULT_CONTEXT: usize = 128;
pub const QWEN_IMAGE_2_1_DEFAULT_TIMESTEP: f32 = 500.0;
pub const QWEN_IMAGE_2_1_DEFAULT_SEED: u32 = 1_234_567;

pub struct QwenImage21Request {
    pub latent: Option<Vec<f32>>,
    pub context: Option<Vec<f32>>,
    pub timestep: f32,
    pub out: std::path::PathBuf,
}

pub fn run_qwen_image_2_1(
    source: Arc<dyn TensorSource>,
    request: QwenImage21Request,
    n_threads: usize,
) -> Result<(), String> {
    validate_dit(source.as_ref())?;
    let config = crate::models::diffusion::qwen_image_2_1::config_from_source(source.as_ref())?;
    let pool = Arc::new(ComputePool::new(n_threads.max(1)));
    let dit = QwenImage21Dit::load(Arc::clone(&source), pool)?;

    let latent_side = QWEN_IMAGE_2_1_DEFAULT_LATENT;
    let image_tokens = latent_side * latent_side;
    let latent_channels = config.in_channels;
    // One generator threads latent then context, matching the oracle harness.
    let mut state = QWEN_IMAGE_2_1_DEFAULT_SEED;
    let mut next_default = |count: usize| -> Vec<f32> {
        (0..count)
            .map(|_| next_deterministic_value(&mut state))
            .collect()
    };
    let latent = match request.latent {
        Some(values) => values,
        None => next_default(latent_channels * image_tokens),
    };
    let context = match request.context {
        Some(values) => values,
        None => next_default(config.context_dim * QWEN_IMAGE_2_1_DEFAULT_CONTEXT),
    };
    if latent.len() != latent_channels * image_tokens {
        return Err(format!(
            "Qwen-Image-2.1 latent must hold {latent_channels}x{latent_side}x{latent_side} values, got {}",
            latent.len()
        ));
    }
    if context.len() % config.context_dim != 0 || context.is_empty() {
        return Err(format!(
            "Qwen-Image-2.1 context must be rows of {} values, got {}",
            config.context_dim,
            context.len()
        ));
    }
    let context_len = context.len() / config.context_dim;

    let velocity = dit.forward(
        &latent,
        latent_side,
        latent_side,
        &context,
        context_len,
        request.timestep,
    )?;
    write_velocity_atomically(&request.out, &velocity)?;
    println!(
        "Qwen-Image-2.1 velocity written to {} ({} values, context {context_len}, timestep {})",
        request.out.display(),
        velocity.len(),
        request.timestep,
    );
    Ok(())
}

fn write_velocity_atomically(path: &Path, values: &[f32]) -> Result<(), String> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty());
    if let Some(parent) = parent {
        std::fs::create_dir_all(parent)
            .map_err(|error| format!("Create output directory {}: {error}", parent.display()))?;
    }
    let temp = path.with_extension(format!("tmp-{}", std::process::id()));
    let write = || -> std::io::Result<()> {
        let mut file = std::fs::File::create(&temp)?;
        file.write_all(&bytes)?;
        file.sync_all()?;
        drop(file);
        std::fs::rename(&temp, path)
    };
    write().map_err(|error| {
        let _ = std::fs::remove_file(&temp);
        format!(
            "Write Qwen-Image-2.1 velocity to {}: {error}",
            path.display()
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_generator_matches_the_oracle_lcg() {
        // First values of the harness generator with seed 1234567.
        let mut state = QWEN_IMAGE_2_1_DEFAULT_SEED;
        let first = next_deterministic_value(&mut state);
        let second = next_deterministic_value(&mut state);
        // state = (1234567 * 1664525 + 1013904223) % 2^32
        assert_eq!(
            state,
            (1_234_567u32)
                .wrapping_mul(1_664_525)
                .wrapping_add(1_013_904_223)
                .wrapping_mul(1_664_525)
                .wrapping_add(1_013_904_223)
        );
        assert!(first.abs() <= 1.0 && second.abs() <= 1.0);
        assert_ne!(first, second);
    }
}
