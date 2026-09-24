use std::time::Duration;

use crate::core::scratchpad::KvFormat;

pub fn validate_qwen3vl_decoder_mode(
    arch: &str,
    dump_logits: bool,
    bench: bool,
    profile: bool,
    kv_format: KvFormat,
    interactive: bool,
) -> Result<(), String> {
    if arch != "qwen3vl" {
        return Ok(());
    }
    let unsupported = if dump_logits {
        Some("--dump-logits")
    } else if bench {
        Some("--bench")
    } else if profile {
        Some("--profile")
    } else if kv_format == KvFormat::F32 {
        Some("--kv-cache f32")
    } else if interactive {
        Some("interactive mode")
    } else {
        None
    };
    match unsupported {
        Some(option) => Err(format!(
            "{option} is not supported for qwen3vl; use default F16 generation"
        )),
        None => Ok(()),
    }
}

pub const DEFAULT_THREAD_CAP: usize = 8;

pub fn resolve_thread_count(requested: usize, available: usize) -> usize {
    if requested > 0 {
        requested
    } else {
        available.clamp(1, DEFAULT_THREAD_CAP)
    }
}

/// Initialize rayon's global thread pool to match the resolved thread
/// count. Idempotent: subsequent calls (or env-var-only setups) silently
/// succeed because `build_global` errors after the first call.
///
/// See the TODO at the top of `src/core/thread_pool.rs` for the rationale
/// of the two-pool model and the preferred direction for unification.
pub fn init_rayon_global_pool(thread_count: usize) {
    let n = thread_count.max(1);
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build_global();
}

pub fn inference_step_budget(prompt_tokens: usize, max_tokens: usize, bench: bool) -> usize {
    prompt_tokens
        + if bench {
            max_tokens
        } else {
            max_tokens.saturating_sub(1)
        }
}

pub fn per_second(count: usize, elapsed: Duration) -> f64 {
    let seconds = elapsed.as_secs_f64();
    if seconds > 0.0 {
        count as f64 / seconds
    } else {
        0.0
    }
}
