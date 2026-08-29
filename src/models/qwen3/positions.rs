//! Position-id helpers for the Qwen3 LLM path.
//!
//! Both CLI text inference (`run_inference` / `run_inference_tokens`) and
//! downstream callers (VL decoder, ASR, TTS) need to materialise a 4-tuple
//! position vector per token: `[t, t, t, 0]` for pure-text runs and
//! variant forms (e.g. mrope sections) for VL/3D positions.

/// Build a contiguous text position vector of length `n_tokens`.
///
/// Each entry is `[i, i, i, 0]` so that the model treats every axis
/// uniformly. Multi-modal callers that need per-axis positions should
/// build their own vector rather than calling this.
pub fn qwen_text_positions(n_tokens: usize) -> Vec<[usize; 4]> {
    (0..n_tokens).map(|position| [position; 4]).collect()
}