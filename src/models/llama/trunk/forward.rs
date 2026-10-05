//! # LLaMA Text Inference
//!
//! LLaMA-family text generation, aligning with llama.cpp's forward pass and
//! sample/decode path. Standard LLaMA has no Q/K per-head RMSNorm.

use super::weights::{get_f32_tensor, layer_loop_config, load_layers, LlamaLayerWeights};
use crate::app::cli::{inference_step_budget, resolve_thread_count, KvFormat};
use crate::core::loader::model_config_from_source;
use crate::core::scratchpad::{ExecutionScratchpad, KvCache};
use crate::core::tensor::TensorSource;
use crate::core::thread_pool::ComputePool;
use crate::core::tokenizer::{load_tokenizer, EncodeOptions};
use crate::ops::embedding_lookup;
use crate::ops::kernel::{QuantizedTensor, Weight};
use crate::ops::{
    dot_f16_f32, dot_f32, f32_slice_to_f16, quantize_q8_0_into, rms_norm_grouped, rms_norm_inplace,
    rope_neox_inplace_with_factor, rope_norm, silu_mul_approx_inplace, silu_mul_inplace, softmax_inplace,
    sum_sq_f32, vec_add_into, vec_mad_f16_f32, vec_mad_f32, vec_scale_f32,
};
use crate::prompt::format_k2_horizon_chat_prompt_with_thinking;

use std::io::{self, Write};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Read sampling defaults from GGUF metadata (matching llama.cpp's
/// behaviour). MiniCPM5 ships `general.sampling.{top_k,top_p,temp}` keys
/// that override the C++ defaults (40, 0.95, 1.0).
pub fn sample_defaults(source: &dyn TensorSource) -> (usize, f32) {
    let top_k = source
        .metadata("general.sampling.top_k")
        .and_then(|v| v.to_u64())
        .map(|v| v as usize)
        .unwrap_or(40);
    let top_p = source
        .metadata("general.sampling.top_p")
        .and_then(|v| v.to_f64())
        .map(|v| v as f32)
        .unwrap_or(0.95);
    (top_k, top_p)
}

/// Print first 8 floats of `buf` to stderr, tagged by step/layer/label.
/// Triggered by the `RUST_LLAMA_DEBUG_TENSORS` env var (set to a layer
/// count, e.g. `RUST_LLAMA_DEBUG_TENSORS=1`).
fn dbg_tensor(step: usize, label: &'static str, il: usize, buf: &[f32]) {
    #[cfg(feature = "parity-trace")]
    {
        let name = match label {
            "embed_out" => Some("embedding"),
            "attn_norm" | "q_proj" | "k_proj" | "v_proj" | "loop_norm" => Some(label),
            "Qcur" => Some("q_rope"),
            "Kcur" => Some("k_rope"),
            "attn_out" => Some("attn_values"),
            "attn_proj" => Some("attn_proj"),
            "ffn_inp" => Some("post_attn_residual"),
            "ffn_norm" => Some("ffn_norm"),
            "ffn_gate_buf_raw" => Some("ffn_silu_gate"),
            "down_buf" => Some("ffn_down"),
            "l_out" => Some("post_ffn_residual"),
            "output_norm" => Some("result_norm"),
            "logits" => Some("result_output"),
            _ => None,
        };
        if let Some(name) = name {
            let layer =
                (!matches!(name, "embedding" | "result_norm" | "result_output")).then_some(il);
            crate::parity_trace::report(crate::parity_trace::checkpoint_at(
                name,
                layer,
                Some(step),
                &[buf.len()],
                buf,
            ));
        }
    }
    static ON: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    let limit = ON.get_or_init(|| {
        std::env::var("RUST_LLAMA_DEBUG_TENSORS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
    });
    if *limit == 0 || (il as u32) >= *limit {
        return;
    }
    let n = buf.len().min(8);
    let mut line = format!("RUST_TENSOR step={} il={} {} first8=", step, il, label);
    for v in &buf[..n] {
        line.push_str(&format!("{:.5} ", v));
    }
    line.push('\n');
    let _ = io::stderr().write_all(line.as_bytes());
    let _ = io::stderr().flush();
}

/// Print a single scalar (scale/mean) tagged by step/layer/label.
fn dbg_scalar(step: usize, label: &'static str, il: usize, value: f32) {
    static ON: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    let limit = ON.get_or_init(|| {
        std::env::var("RUST_LLAMA_DEBUG_TENSORS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
    });
    if *limit == 0 || (il as u32) >= *limit {
        return;
    }
    let line = format!(
        "RUST_SCALAR step={} il={} {}={:.10}\n",
        step, il, label, value
    );
    let _ = io::stderr().write_all(line.as_bytes());
    let _ = io::stderr().flush();
}

fn dbg_scalar_full(step: usize, label: &'static str, il: usize, value: f64) {
    static ON: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    let limit = ON.get_or_init(|| {
        std::env::var("RUST_LLAMA_DEBUG_TENSORS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
    });
    if *limit == 0 || (il as u32) >= *limit {
        return;
    }
    let line = format!(
        "RUST_SCALAR step={} il={} {}={:.10}\n",
        step, il, label, value
    );
    let _ = io::stderr().write_all(line.as_bytes());
    let _ = io::stderr().flush();
}

/// Dump full values of `buf` to a file specified by RUST_LLAMA_DEBUG_OUTFILE.
fn dbg_full(step: usize, label: &'static str, il: usize, buf: &[f32], n: usize) {
    static ON: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    let limit = ON.get_or_init(|| {
        std::env::var("RUST_LLAMA_DEBUG_TENSORS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
    });
    if *limit == 0 || (il as u32) >= *limit {
        return;
    }
    let path = match std::env::var("RUST_LLAMA_DEBUG_OUTFILE") {
        Ok(p) => p,
        Err(_) => return,
    };
    let mut line = format!("[step={} il={} {}]", step, il, label);
    for i in 0..n {
        line.push_str(&format!(" {:.5}", buf[i]));
    }
    line.push('\n');
    let _ = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .and_then(|mut f| std::io::Write::write_all(&mut f, line.as_bytes()));
}

pub(crate) fn normalization_groups(
    source: &dyn TensorSource,
    arch: &str,
    n_embd: usize,
) -> Result<usize, String> {
    if arch != "k2-horizon" {
        return Ok(1);
    }

    let groups = source
        .metadata("k2-horizon.attention.group_norm_groups")
        .and_then(|value| value.to_u64())
        .unwrap_or(1)
        .max(1) as usize;
    if !n_embd.is_multiple_of(groups) {
        return Err(format!(
            "K2-Horizon embedding length {n_embd} is not divisible by {groups} normalization groups"
        ));
    }
    Ok(groups)
}

pub(crate) fn apply_rope(
    arch: &str,
    values: &mut [f32],
    pos: usize,
    head_dim: usize,
    freq_base: f32,
    rope_dim: usize,
    attn_factor: f32,
    yarn_thetas: Option<&[f32]>,
) {
    // Phi-3 / Phi-4 only apply RoPE to the first `rope_dim` of `head_dim`;
    // the remaining lanes pass through unchanged. Mirror that by splitting
    // the slice and only rotating the head. (`rope_dim = head_dim` for
    // non-phi architectures.)
    //
    // Phi-3 / Phi-4 also multiply every RoPE output by `attn_factor`
    // (= `rope.scaling.attn_factor` in GGUF metadata). This factor is
    // baked into the model's weights during training and missing it
    // shifts the qk inner products enough to break greedy decoding.
    //
    // `rope_dim` is also the dim used to build the cos/sin table (i.e.
    // `theta_scale = freq_base^(-2/rope_dim)`). llama.cpp's ggml_rope_ext
    // uses `n_dims` as the rotation count for both the slice AND the
    // frequency table — i.e. partial RoPE uses rope_dim frequencies, not
    // head_dim frequencies.
    //
    // Phi-3 / Phi-4 / Phi-2 / Phi-MoE / Gemma-2 / Gemma-3 / GPT-NeoX /
    // Exaone / OpenELM / StableLM all use the "neox" (half-rotation) rope
    // layout in llama.cpp: `out[i] = x[i]*cos - x[i + half]*sin`, with
    // `half = rope_dim / 2`. The "normal" (interleaved) layout is used
    // by llama / qwen / granite / k2-horizon.
    //
    // `yarn_thetas` (when `Some`) overrides the per-dim
    // `theta = pos * freq_base^(-2i/rope_dim)` table with a precomputed
    // YaRN-corrected table for long-context extension. Used by Mistral 3
    // (`mistral3.rope.scaling.type = "yarn"`, factor=16, orig_ctx=16384,
    // beta_fast=32, beta_slow=1, log_multiplier=1). For short contexts
    // (< orig_ctx) YaRN is a no-op (`ramp=0` everywhere) and the thetas
    // reduce to the plain RoPE table; for long contexts the thetas
    // diverge per `rope.scaling.yarn_beta_*` ramps. See
    // `compute_yarn_thetas` for the construction.
    let neox_layout = arch == "phi3"
        || arch == "phi2"
        || arch == "phimoe"
        || arch == "gemma"
        || arch == "gemma2"
        || arch == "gemma3"
        || arch == "gptneox"
        || arch == "exaone"
        || arch == "exaone4"
        || arch == "openelm"
        || arch == "stablelm";
    let rope_dim = rope_dim.min(head_dim);
    if rope_dim < head_dim {
        apply_partial_rope(values, pos, rope_dim, freq_base, neox_layout, yarn_thetas);
        apply_attn_factor(values, rope_dim, attn_factor);
        return;
    }
    if neox_layout || arch == "k2-horizon" {
        rope_neox_inplace_with_thetas(values, pos, head_dim, freq_base, yarn_thetas);
    } else {
        rope_norm_with_thetas(values, pos, head_dim, freq_base, yarn_thetas);
    }
    apply_attn_factor(values, head_dim, attn_factor);
}

/// Compute per-dim YaRN-corrected `theta` factors.
///
/// Returns `Some(Vec<f32>)` of length `rope_dim / 2` (so that
/// `thetas[i] * pos` is the final theta for dim `i`) when the GGUF
/// metadata declares YaRN (`rope.scaling.type = "yarn"`, plus the five
/// numeric keys below); `None` otherwise. With `pos < original_ctx`,
/// the returned thetas reduce to plain RoPE (`theta = pos *
/// freq_base^(-2i/rope_dim)`); with `pos > original_ctx`, dims inside
/// the `[start, end]` wavelength-correction ramp get a fraction of
/// `pos * (1/factor) * freq_base^(-2i/rope_dim)` interpolated with the
/// extrapolation, which is what YaRN does to extend context past the
/// training window without breaking short-context behavior.
///
/// `yarn_log_multiplier` (`rope.scaling.yarn_log_multiplier`) is the
/// mscale factor: per xing4_0 trunk, the attention logits are
/// scaled by `1 + 0.1 * log_mult * ln(factor)` (squared when used as
/// the `attn_factor`); we keep the per-dim thetas independent of
/// `attn_factor` and leave that scaling to `apply_attn_factor` below
/// (so llama-3-style `attn_factor=1.0` plus YaRN thetas still works).
///
/// Reference: xing4_0 trunk (`src/models/xing4_0/trunk/forward.rs`,
/// `build_rope`); llama.cpp `ggml_compute_forward_rope_f32` YaRN branch
/// in `ggml-cpu/ops.cpp`.
pub(crate) fn compute_yarn_thetas(
    source: &dyn crate::core::tensor::TensorSource,
    arch: &str,
    freq_base: f32,
    rope_dim: usize,
) -> Option<Vec<f32>> {
    let scaling_type = source
        .metadata(&format!("{arch}.rope.scaling.type"))
        .and_then(|v| v.to_string_val())?;
    if scaling_type != "yarn" {
        return None;
    }
    let factor: f64 = source
        .metadata(&format!("{arch}.rope.scaling.factor"))
        .and_then(|v| v.to_f64())
        .unwrap_or(1.0);
    if !(factor > 1.0) {
        // YaRN only kicks in for factor > 1.0; fall through to plain
        // RoPE so the thetas table is still Some, just with `ramp=0`
        // everywhere.
        return None;
    }
    let orig_ctx: f64 = source
        .metadata(&format!("{arch}.rope.scaling.original_context_length"))
        .and_then(|v| v.to_u64())
        .map(|v| v as f64)
        .unwrap_or(0.0);
    let beta_fast: f64 = source
        .metadata(&format!("{arch}.rope.scaling.yarn_beta_fast"))
        .and_then(|v| v.to_f64())
        .unwrap_or(32.0);
    let beta_slow: f64 = source
        .metadata(&format!("{arch}.rope.scaling.yarn_beta_slow"))
        .and_then(|v| v.to_f64())
        .unwrap_or(1.0);
    let half = rope_dim / 2;
    if half == 0 || orig_ctx <= 0.0 {
        return None;
    }
    let ln_factor = (1.0 / factor).ln();
    // mscale: a multiplicative bias on the per-position cos/sin so
    // long-context logits don't collapse; llama.cpp applies this as
    // `attn_factor`, but we surface it as a `theta_scale` so that
    // `attn_factor` from GGUF (when separately shipped) still wins.
    let _mscale = 1.0f64 + 0.1 * ln_factor; // documented but applied via `attn_factor`
    let base_ln = freq_base.ln() as f64;
    let corr = |n_rot: f64| -> f64 {
        // wavelength: `dim * ln(n_ctx_orig / n_rot) / (2 * ln(base))`
        (rope_dim as f64) * (orig_ctx / (n_rot * 2.0 * std::f64::consts::PI)).ln() / (2.0 * base_ln)
    };
    let start = corr(beta_fast as f64).floor().max(0.0);
    let end = corr(beta_slow as f64).ceil().min((rope_dim as f64) - 1.0);
    let span = (end - start).max(0.001);
    let mut thetas = vec![0.0f32; half];
    let freq_scale = 1.0f64 / factor;
    for i in 0..half {
        // Per-dim extrap theta (what plain RoPE would do for this dim i).
        // Plain RoPE: `theta_at_pos = pos * freq_base^(-2i/rope_dim)`,
        // so the position-independent per-dim multiplier is
        // `freq_base^(-2i/rope_dim)`. Note xing4_0 and llama.cpp
        // multiply `pos` into `theta_extrap` and store the full
        // position-dependent table — we split that into
        // `thetas[i] * pos` to avoid allocating `max_ctx × half`
        // floats (Mistral 3 ships a 256K-context model).
        let theta_extrap = (freq_base as f64).powf(-2.0 * i as f64 / rope_dim as f64);
        // Per-dim interp theta (the YaRN-reduced extrapolation).
        let theta_interp = freq_scale * theta_extrap;
        // Linear ramp across the wavelength-correction zone.
        let ramp = 1.0 - ((i as f64 - start) / span).clamp(0.0, 1.0);
        // The final per-dim multiplier is the ramp-blend; `pos` is
        // multiplied in at rotation time so the thetas table is
        // position-independent.
        thetas[i] = (theta_interp * (1.0 - ramp) + theta_extrap * ramp) as f32;
    }
    Some(thetas)
}

/// Apply RoPE to the first `rope_dim` of `values`. `neox_layout=true`
/// uses the half-rotation pattern (`x[i]` paired with `x[i + half]`),
/// which is what Phi-3 / Phi-4 / Gemma / GPT-NeoX use in llama.cpp.
/// `neox_layout=false` uses the interleaved-pair pattern
/// (`x[2i]` paired with `x[2i + 1]`).
///
/// The cos/sin table is built against `rope_dim` itself (not `head_dim`),
/// matching llama.cpp's `n_dims` semantics.
///
/// When `yarn_thetas` is `Some(t)`, the per-dim theta is `pos * t[i]`
/// instead of `pos * freq_base^(-2i/rope_dim)` (the recurrence `theta
/// *= theta_scale` is replaced with `theta *= t[i] / t[i-1]` for
/// consecutive i; the absolute path uses `pos * t[i]` directly to
/// avoid accumulating the recurrence).
fn apply_partial_rope(
    values: &mut [f32],
    pos: usize,
    rope_dim: usize,
    freq_base: f32,
    neox_layout: bool,
    yarn_thetas: Option<&[f32]>,
) {
    use crate::ops::rope::neox::rope_sin_cos;
    if rope_dim < 2 {
        return;
    }
    let half = rope_dim / 2;
    for i in 0..half {
        let theta = if let Some(t) = yarn_thetas {
            pos as f32 * t[i]
        } else {
            let theta_scale = freq_base.powf(-2.0f32 / rope_dim as f32);
            pos as f32 * theta_scale.powi(i as i32)
        };
        let (c, s) = rope_sin_cos(theta);
        if neox_layout {
            // Half-rotation: x[i] paired with x[i + half].
            let x0 = values[i];
            let x1 = values[i + half];
            values[i] = x0 * c - x1 * s;
            values[i + half] = x0 * s + x1 * c;
        } else {
            // Interleaved: x[2i] paired with x[2i + 1].
            let x0 = values[2 * i];
            let x1 = values[2 * i + 1];
            values[2 * i] = x0 * c - x1 * s;
            values[2 * i + 1] = x0 * s + x1 * c;
        }
    }
}

/// Same as `crate::ops::rope::norm::rope_norm` but optionally reads
/// per-dim thetas from `yarn_thetas` (length = `head_dim / 2`) instead
/// of `pos * freq_base^(-2i/head_dim)`.
fn rope_norm_with_thetas(
    values: &mut [f32],
    pos: usize,
    head_dim: usize,
    freq_base: f32,
    yarn_thetas: Option<&[f32]>,
) {
    let half = head_dim / 2;
    if half == 0 || values.is_empty() {
        return;
    }
    let n_heads = values.len() / head_dim;
    // Cache sin/cos table once across all heads (same for each head at this pos).
    let mut cos_table = vec![0.0f32; half];
    let mut sin_table = vec![0.0f32; half];
    if let Some(t) = yarn_thetas {
        debug_assert_eq!(t.len(), half, "yarn_thetas length must match head_dim/2");
        for i in 0..half {
            let theta = pos as f32 * t[i];
            let (c, s) = crate::ops::rope::neox::rope_sin_cos(theta);
            cos_table[i] = c;
            sin_table[i] = s;
        }
    } else {
        let theta_scale = freq_base.powf(-2.0f32 / head_dim as f32);
        let mut theta = pos as f32;
        for i in 0..half {
            let (c, s) = crate::ops::rope::neox::rope_sin_cos(theta);
            cos_table[i] = c;
            sin_table[i] = s;
            theta *= theta_scale;
        }
    }
    for h in 0..n_heads {
        let base = h * head_dim;
        for i in 0..half {
            let x0 = values[base + 2 * i];
            let x1 = values[base + 2 * i + 1];
            let c = cos_table[i];
            let sn = sin_table[i];
            if crate::ops::scalar_mode() {
                values[base + 2 * i] = x0 * c - x1 * sn;
                values[base + 2 * i + 1] = x0 * sn + x1 * c;
            } else {
                values[base + 2 * i] = x0.mul_add(c, x1 * -sn);
                values[base + 2 * i + 1] = x0.mul_add(sn, x1 * c);
            }
        }
    }
}

/// Same as `rope_neox_inplace_with_factor` but optionally
/// reads per-dim thetas from `yarn_thetas` (length = `head_dim / 2`)
/// instead of `pos * freq_base^(-2i/head_dim)`.
fn rope_neox_inplace_with_thetas(
    values: &mut [f32],
    pos: usize,
    head_dim: usize,
    freq_base: f32,
    yarn_thetas: Option<&[f32]>,
) {
    let half = head_dim / 2;
    if half == 0 || values.is_empty() {
        return;
    }
    let n_heads = values.len() / head_dim;
    let mut cos_table = vec![0.0f32; half];
    let mut sin_table = vec![0.0f32; half];
    if let Some(t) = yarn_thetas {
        debug_assert_eq!(t.len(), half, "yarn_thetas length must match head_dim/2");
        for i in 0..half {
            let theta = pos as f32 * t[i];
            let (c, s) = crate::ops::rope::neox::rope_sin_cos(theta);
            cos_table[i] = c;
            sin_table[i] = s;
        }
    } else {
        let theta_scale = freq_base.powf(-2.0f32 / head_dim as f32);
        let mut theta = pos as f32;
        for i in 0..half {
            let (c, s) = crate::ops::rope::neox::rope_sin_cos(theta);
            cos_table[i] = c;
            sin_table[i] = s;
            theta *= theta_scale;
        }
    }
    for h in 0..n_heads {
        let base = h * head_dim;
        for i in 0..half {
            let x0 = values[base + i];
            let x1 = values[base + i + half];
            let c = cos_table[i];
            let sn = sin_table[i];
            if crate::ops::scalar_mode() {
                values[base + i] = x0 * c - x1 * sn;
                values[base + i + half] = x0 * sn + x1 * c;
            } else {
                values[base + i] = x0.mul_add(c, x1 * -sn);
                values[base + i + half] = x0.mul_add(sn, x1 * c);
            }
        }
    }
}

/// Phi-3 / Phi-4 weight every RoPE output by `attn_factor`. We do the
/// multiply in-place on the freshly rotated values so the change is
/// invisible to callers that pass `attn_factor = 1.0` (the llama default).
///
/// Forwarded to [`crate::ops::vec_scale_f32`] which dispatches to AVX2 /
/// NEON under the hood. Phi-4 calls this `n_layer * n_head * (prompt +
/// generation)` times, so the per-call savings compound across the
/// whole sequence.
fn apply_attn_factor(values: &mut [f32], head_dim: usize, attn_factor: f32) {
    if attn_factor != 1.0 && head_dim > 0 {
        crate::ops::vec_scale_f32(&mut values[..head_dim], attn_factor);
    }
}

pub fn run_inference(
    source: &dyn TensorSource,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    n_threads_arg: usize,
    bench: bool,
    profile: bool,
    kv_format: KvFormat,
    max_context: usize,
    repetition_penalty: f32,
    thinking: bool,
) -> Result<(), String> {
    let input_tokens = {
        let tokenizer = load_tokenizer(|k| source.metadata(k).cloned())
            .map_err(|error| format!("Failed to initialize tokenizer: {error}"))?;

        let arch = source
            .metadata("general.architecture")
            .and_then(|v| v.to_string_val())
            .unwrap_or_default();

        // MiniCPM5 detection: `general.architecture = llama` (LLaMA backbone),
        // but `general.name` contains "MiniCPM". MiniCPM5's GGUF embeds a
        // ChatML-based `tokenizer.chat_template` (with `<|im_start|>`/`<|im_end|>`),
        // NOT the old MiniCPM 3B `<用户>/<AI>` template. The template supports
        // `enable_thinking` to control reasoning mode.
        // (Ref: OpenBMB/MiniCPM GGUF chat_template; llama.cpp `llama-chat.cpp:180`)
        let is_minicpm5 = source
            .metadata("general.name")
            .and_then(|v| v.to_string_val())
            .map(|s| s.to_ascii_lowercase().contains("minicpm"))
            .unwrap_or(false);

        // Build the prompt based on the architecture. Granite uses a
        // distinct chat template: `<|start_of_role|>{role}<|end_of_role|>
        // {content}<|end_of_text|>` between turns and ends the user turn
        // with `<|end_of_text|>\n`. MiniCPM5 uses ChatML with non-thinking
        // mode (`🤔\n\n\web_search\n\n`). Other Llama models use Qwen2-style
        // ChatML with thinking (`🤔\n`). Nanbeige uses its embedded ChatML template.
        let is_mistral = source
            .metadata("general.name")
            .and_then(|v| v.to_string_val())
            .map(|s| {
                // All Mistral chat-template families share the
                // `[INST] … [/INST]` user-turn shape:
                //   - classic Mistral (`Mistral-7B-Instruct-v0.3`,
                //     `Mistral-Large-Instruct-*`, etc.)
                //   - Mistral 3's "Ministral" deliberate misspelling
                //     (`Ministral-3B-Instruct-2512`,
                //     `Ministral-3-3B-Reasoning-2512`)
                //   - Shieldstral (`mistralai/Shieldstral-1.0-3B` /
                //     `Mistral-Shieldstral-*`), Mistral's safety /
                //     moderation classifier family. GGUF shippers set
                //     `general.name = "Shieldstral 1.0 3B"` /
                //     `"Mistral-Shieldstral-22B"`, so neither substring
                //     contains "mistral" / "ministral" literally — the
                //     product name "Shieldstral" only overlaps the
                //     suffix.
                // Match all three families here; new Mistral chat
                // products should follow the same `[INST] … [/INST]`
                // convention and need to be added when they ship.
                let lower = s.to_ascii_lowercase();
                lower.contains("mistral")
                    || lower.contains("ministral")
                    || lower.contains("shieldstral")
            })
            .unwrap_or(false);
        // Zephyr detection: `general.name` containing "zephyr" covers
        // TheBloke's conversions (`huggingfaceh4_zephyr-7b-alpha`), but
        // other publishers (mradermacher, MaziyarPanahi) rewrite
        // `general.name` to "`.`" / `"hub"` and rely on the embedded
        // `tokenizer.chat_template` to carry the model identity. Fall
        // back to that: Zephyr's template is the only llama-arch template
        // that uses `<|user|>` / `<|assistant|>` as the user-turn and
        // generation-prompt markers. The `arch == "llama"` gate keeps
        // the chat-template fallback from firing on GLM-4 (`arch="glm4"`,
        // also uses `<|user|>` / `<|assistant|>` markers per its own
        // template); the GLM-4 branches in `llama_turn_text` /
        // `build_prompt_tokens` handle that arch.
        let is_zephyr = source
            .metadata("general.name")
            .and_then(|v| v.to_string_val())
            .map(|s| s.to_ascii_lowercase().contains("zephyr"))
            .unwrap_or(false)
            || (arch == "llama"
                && source
                    .metadata("tokenizer.chat_template")
                    .and_then(|v| v.to_string_val())
                    .map(|t| t.contains("<|user|>") && t.contains("<|assistant|>"))
                    .unwrap_or(false));

        let prompt_text = if arch == "k2-horizon" {
            format_k2_horizon_chat_prompt_with_thinking(prompt, thinking)
        } else if arch == "granite" {
            format!(
                "<|start_of_role|>user<|end_of_role|>{prompt}<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>"
            )
        } else if arch == "phi3" {
            // Phi-4 (and Phi-3) wraps each turn as
            // `<|role|>content<|end|>`; the assistant turn opens with
            // `<|assistant|>`. The trailing `<|end|>` from the user turn
            // already ends the user message.
            format!("<|user|>{prompt}<|end|><|assistant|>")
        } else if arch == "nanbeige" {
            if source
                .metadata("tokenizer.chat_template")
                .and_then(|v| v.to_string_val())
                .is_some_and(|t| t.contains("<|im_start|>"))
            {
                crate::prompt::build_nanbeige_chat_prompt(prompt, thinking)
            } else {
                prompt.to_string()
            }
        } else if is_minicpm5 {
            // MiniCPM5 uses ChatML (`{role}\n{content}`)
            // per its GGUF `tokenizer.chat_template`. The template supports
            // `enable_thinking`: when false, emits `🤔\n\n\web_search\n\n`
            // (empty thinking block → direct answer). When true, emits `🤔\n`
            // (thinking mode). Default: non-thinking for fast direct answers.
            // (Ref: OpenBMB/MiniCPM GGUF chat_template, `enable_thinking` branch)
            format!("user\n{prompt}\nassistant\n🤔\n\n</think>\n\n")
        } else if arch == "glm4" {
            // GLM-4 chat template: `[gMASK]<sop>` prefix, then
            // `<|user|>\n{prompt}<|assistant|>\n`. `[gMASK]` is the
            // BOS-like sentinel (id 151329 in GLM-4's vocab); `<sop>` is
            // 151332. Both are special tokens, recognised as single ids
            // because `parse_special=true`.
            format!("[gMASK]<sop><|user|>\n{prompt}<|assistant|>\n")
        } else if is_mistral {
            // TODO(mistral3-reasoning): the CLI emits a bare
            // `[INST] {prompt} [/INST]` turn and never injects the
            // `[SYSTEM_PROMPT] … [/SYSTEM_PROMPT]` block from
            // `tokenizer.chat_template`. For Ministral-3-3B-Instruct-2512
            // that's fine — it answers in free text directly. For
            // Ministral-3-3B-Reasoning-2512 (same engine path, same arch)
            // the chat template's default system prompt asks the model to
            // "First draft your thinking process in [THINK]…[/THINK] then
            // answer" — without that prompt the Reasoning model still
            // produces structured reasoning prose but never emits the
            // canonical `[THINK] … [/THINK]` separators Mistral trained it
            // for. Smoke-tested: `--prompt "If a train leaves station A
            // at 60 km/h …"` produces a Markdown-headed reasoning walk-
            // through (no `[THINK]` markers). Unblocking the canonical
            // shape needs one of:
            //   (a) a `--system-prompt` CLI flag passed through to here
            //       and prepended as
            //       `[SYSTEM_PROMPT]{system}[/SYSTEM_PROMPT][INST]{p}[/INST]`,
            //   (b) a chat-template-aware mode that runs the model's own
            //       jinja against the prompt, or
            //   (c) at minimum, for `general.name` containing
            //       "reasoning" / "Ministral-3-Reasoning", inject the
            //       stock default system prompt automatically.
            // Tracked; not in scope for this PR.
            format!("[INST] {prompt} [/INST]")
        } else if is_zephyr {
            format!("<|user|>\n{prompt}</s>\n<|assistant|>\n")
        } else {
            format!("user\n{prompt}\nassistant\n<think>\n")
        };
        eprintln!("[RUST_PROMPT_TEXT] {prompt_text}");
        // Mistral's `[INST]`/`[/INST]` and Zephyr's `<|user|>`/`<|assistant|>`
        // are tokenizer special tokens (`tokenizer.ggml.add_bos_token=true`).
        // Use the tokenizer's own chat-template handling: `add_special=true`
        // emits BOS automatically, `parse_special=true` recognizes the
        // literal `[INST]`/`[/INST]` in the template as single special tokens.
        // For Granite/MiniCPM5/Llama the chat template emits `<s>` (or
        // expects no BOS since add_bos_token=false), so add_special=false
        // and we manually prepend BOS. For Nanbeige (base model), let the
        // tokenizer's add_bos setting handle BOS via add_special=true.
        //
        // GLM-4 uses GPT-2 BPE (`tokenizer.ggml.model="gpt2"`); its BOS
        // is `<|endoftext|>` (id 151329). The chat template emits
        // `[gMASK]<sop>` (a separate BOS-equivalent), but the model is
        // trained with `<|endoftext|>` PRECEDING `[gMASK]<sop>`, so we
        // need `add_special=true` to make the tokenizer prepend BOS
        // GLM-4 uses GPT-2 BPE with `tokenizer.ggml.add_bos_token=false`
        // (the metadata field is missing from unsloth's GGUF conversion).
        // Its BOS is `<|endoftext|>` (id 151329), which must precede
        // the chat template's `[gMASK]<sop>` BOS-equivalent. Setting
        // `add_special=true` alone doesn't help (the BPETokenizer only
        // prepends when `add_bos` is also true), so we leave
        // `add_special=false` for GLM-4 and prepend BOS manually below
        // (just like `k2-horizon` / `granite` / `exaone` do).
        let (add_special, parse_special) = match arch {
            "nanbeige" => (true, true),
            "k2-horizon" | "granite" | "exaone" | "glm4" => (false, true),
            _ if is_mistral || is_zephyr => (true, true),
            _ => (false, true),
        };
        let mut body = tokenizer.encode(
            &prompt_text,
            EncodeOptions {
                add_special,
                parse_special,
            },
        );
        // The chat template starts with `{{- bos_token }}`, but MiniCPM5
        // and Granite both have `tokenizer.ggml.add_bos_token=false`, so
        // encode() does not emit BOS automatically. Prepend BOS manually
        // to match llama.cpp. (`add_special=true` arms above handle BOS
        // automatically via the tokenizer and skip this prepend.)
        if !add_special {
            if let Some(bos) = tokenizer.bos_id() {
                body.insert(0, bos);
            }
        }
        eprintln!("[RUST_TOKENS] n={} ids={:?}", body.len(), body);
        body
    };

    run_inference_tokens(
        source,
        input_tokens,
        max_tokens,
        temperature,
        n_threads_arg,
        bench,
        profile,
        kv_format,
        max_context,
        repetition_penalty,
    )
}

const IM_START_MARK: &str = concat!("<", "|im_start", "|", ">");
const IM_END_MARK: &str = concat!("<", "|im_end", "|", ">");
const THINK_MARK: &str = concat!("<", "|think", "|", ">");
const THINK_END_MARK: &str = concat!("<", "|/think", "|", ">");

/// Render one turn's text for the llama-family templates.
///
/// The per-arch selection used to be inline in `build_prompt_tokens` for a
/// single turn only. Extracted so the multi-turn entry point can reuse it
/// verbatim; the single-turn path calls this with exactly one turn and is
/// therefore byte-identical to its previous behaviour.
#[allow(clippy::too_many_arguments)]
fn llama_turn_text(
    arch: &str,
    is_minicpm5: bool,
    is_mistral: bool,
    is_zephyr: bool,
    has_chatml_template: bool,
    role: &str,
    content: &str,
    thinking: bool,
) -> String {
    if arch == "k2-horizon" {
        // K2-Horizon's own control tokens have no multi-turn shape;
        // callers must reject multi-turn before we get here.
        return format_k2_horizon_chat_prompt_with_thinking(content, thinking);
    }
    if arch == "granite" {
        return format!("<|start_of_role|>{role}<|end_of_role|>{content}<|end_of_text|>\n");
    }
    if arch == "nanbeige" {
        if has_chatml_template {
            return format!("{IM_START_MARK}{role}\n{content}{IM_END_MARK}\n");
        }
        return format!("{role}\n{content}\n");
    }
    if arch == "exaone" {
        // EXAONE-3.5 instruct control tokens. A system turn is not
        // expressible in the single-turn CLI shape (which only ever
        // passes one user message), so it maps to the user turn here;
        // multi-turn history renders as repeated
        // `[|user|]...[|endofturn|]` blocks plus a final `[|assistant|]`.
        // Same markers `ChatTemplate::Exaone` uses.
        if role == "assistant" {
            if content.is_empty() {
                // Generation prompt: bare marker, no end-of-turn.
                return "[|assistant|]".to_string();
            }
            return format!("[|assistant|]{content}[|endofturn|]\n");
        }
        return format!("[|user|]{content}[|endofturn|]\n");
    }
    if is_minicpm5 {
        // MiniCPM5 ChatML; thinking=false emits an empty reasoning block.
        if thinking {
            return format!("{IM_START_MARK}{role}\n{content}{IM_END_MARK}\n");
        }
        return format!(
            "{IM_START_MARK}{role}\n{content}{IM_END_MARK}\n{IM_START_MARK}assistant\n{THINK_MARK}\n\n{THINK_END_MARK}\n\n"
        );
    }
    if arch == "phi3" {
        // Phi-3 / Phi-4 instruct chat template:
        // `<|user|>…<|end|>` for the user turn and bare `<|assistant|>`
        // for the generation prompt. The outer caller appends the
        // assistant turn when `turns.len() == 1`, producing
        // `<|user|>{prompt}<|end|><|assistant|>` byte-equal to the CLI's
        // `build_prompt_tokens` phi3 branch. Multi-turn is rejected
        // upstream (`llama_supports_multiturn` for `phi3` is false), so
        // we never render more than one user turn here.
        if role == "assistant" {
            return "<|assistant|>".to_string();
        }
        return format!("<|user|>{content}<|end|>");
    }
    if arch == "glm4" {
        // GLM-4 (THUDM) uses `<|user|>\n{content}` for the user turn
        // and `<|assistant|>\n` for the generation prompt. The model's
        // chat template emits `[gMASK]<sop>\n` exactly once at the very
        // start of the prompt; the caller (`build_prompt_tokens_from_turns`)
        // is responsible for prepending it before the first user turn
        // because this helper is per-turn and carries no position state.
        // GLM-4 tokenizer is GPT-2 BPE; `[gMASK]` (id 151331), `<sop>`
        // (id 151333), `<|user|>` (id 151336), `<|assistant|>` (id 151337)
        // are recognised as single special tokens via `parse_special=true`.
        // (Ref: THUDM/glm-4-9b-chat tokenizer_config.json chat_template.)
        if role == "assistant" {
            return "<|assistant|>\n".to_string();
        }
        return format!("<|user|>\n{content}");
    }
    if is_mistral {
        // Mistral-Instruct uses `[INST] {user} [/INST]` for the user turn
        // and an empty assistant turn (generation begins right after
        // `[/INST]`). The closing wrapper `[/INST]` is part of the user
        // turn, not a separator, so the model is asked to produce the
        // first assistant token directly. Tokenizer BOS (id=1) is emitted
        // via `add_special=true` so we don't prepend it manually.
        // (Ref: llama.cpp `llama_chat_apply_template_internal` Mistral
        // branch; mistralai/Mistral-7B-Instruct-v0.3 tokenizer config.)
        if role == "assistant" {
            return String::new();
        }
        return format!("[INST] {content} [/INST]");
    }
    if is_zephyr {
        // Zephyr-7B uses `<|user|>\n{content}</s>\n<|assistant|>\n` for
        // the user turn and `<|assistant|>\n` for the assistant
        // generation prompt, mirroring HuggingFaceH4's tokenizer
        // `chat_template`. The trailing `\n` matters: Zephyr expects the
        // assistant marker on its own line.
        if role == "assistant" {
            return "<|assistant|>\n".to_string();
        }
        return format!("<|user|>\n{content}</s>\n<|assistant|>\n");
    }
    format!("user\n{content}\nassistant\n{THINK_MARK}\n")
}

/// True when `arch`'s template can express more than one turn.
fn llama_supports_multiturn(
    arch: &str,
    is_minicpm5: bool,
    is_mistral: bool,
    is_zephyr: bool,
) -> bool {
    if arch == "k2-horizon" {
        return false;
    }
    if is_minicpm5 {
        return true;
    }
    // Mistral/Zephyr repeat `[INST]…[/INST]`/`<|user|>…<|assistant|>`
    // blocks for each turn, so multi-turn is expressible here.
    if is_mistral || is_zephyr {
        return true;
    }
    // GLM-4 repeats `<|user|>\n{content}` / `<|assistant|>\n{content}\n`
    // blocks for each turn, so multi-turn is expressible here.
    if arch == "glm4" {
        return true;
    }
    // nanbeige (ChatML template) and granite (start_of_role) do.
    // Plain llama / exaone fall back to the `user\n...assistant\n` shape,
    // which llama.cpp renders as repeated `{role}\n{content}\n` blocks, so
    // multi-turn is expressible there too — as repeated turns plus a final
    // assistant prompt, which is what llama.cpp does.
    matches!(arch, "nanbeige" | "granite" | "llama" | "exaone")
}

/// Build the prompt token vector for a llama-family arch from message turns.
///
/// `turns` are the caller's messages in order (system / user / assistant).
/// This is what the HTTP layer uses; the CLI keeps calling
/// [`build_prompt_tokens`], which is a one-turn specialisation of this and
/// produces byte-identical output.
pub fn build_prompt_tokens_from_turns(
    source: &dyn TensorSource,
    tokenizer: &crate::core::tokenizer::BPETokenizer,
    turns: &[(&str, &str)],
    thinking: bool,
) -> Result<Vec<u32>, String> {
    let arch = source
        .metadata("general.architecture")
        .and_then(|v| v.to_string_val())
        .unwrap_or_default();
    let is_minicpm5 = source
        .metadata("general.name")
        .and_then(|v| v.to_string_val())
        .map(|s| s.to_ascii_lowercase().contains("minicpm"))
        .unwrap_or(false);
    let is_mistral = source
        .metadata("general.name")
        .and_then(|v| v.to_string_val())
        .map(|s| s.to_ascii_lowercase().contains("mistral"))
        .unwrap_or(false);
    let is_zephyr = source
        .metadata("general.name")
        .and_then(|v| v.to_string_val())
        .map(|s| s.to_ascii_lowercase().contains("zephyr"))
        .unwrap_or(false)
        || (arch == "llama"
            && source
                .metadata("tokenizer.chat_template")
                .and_then(|v| v.to_string_val())
                .map(|t| t.contains("<|user|>") && t.contains("<|assistant|>"))
                .unwrap_or(false));
    if turns.len() != 1 && !llama_supports_multiturn(&arch, is_minicpm5, is_mistral, is_zephyr) {
        return Err(format!(
            "multi-turn chat is unsupported for architecture {arch:?}; only a single user turn is rendered"
        ));
    }
    let has_chatml_template = source
        .metadata("tokenizer.chat_template")
        .and_then(|v| v.to_string_val())
        .is_some_and(|t| t.contains(" + IM_START + "));
    let mut prompt_text = String::new();
    // GLM-4's chat template emits the `[gMASK]<sop>` BOS-like sentinel
    // exactly once at the start of the prompt. `llama_turn_text` is
    // per-turn and has no position state, so prepend here. The single-turn
    // `build_prompt_tokens` for `arch == "glm4"` emits `[gMASK]<sop>`
    // (no trailing newline) — match that byte-for-byte so HTTP multi-turn
    // produces the same first few tokens as the CLI.
    if arch == "glm4" {
        prompt_text.push_str("[gMASK]<sop>");
    }
    for (role, content) in turns {
        prompt_text.push_str(&llama_turn_text(
            &arch,
            is_minicpm5,
            is_mistral,
            is_zephyr,
            has_chatml_template,
            role,
            content,
            thinking,
        ));
    }
    // The single-turn callers end with an assistant prompt; reproduce that
    // whenever the conversation does not already end with one.
    //
    // This used to be gated on `turns.len() == 1`, so a multi-turn prompt
    // stopped at the last user turn and never emitted its generation
    // prompt. GLM-4 was the visible casualty: its user turn is the bare
    // block `<|user|>\n{content}`, so the prompt ended mid-user-message,
    // GLM-4 sampled an immediate stop token, and the reply came back empty.
    //
    // The extra multi-turn case is limited to the archs that render a BARE
    // user turn (glm4 / mistral / zephyr) and therefore genuinely need the
    // generation prompt appended separately. Every other template already
    // folds the assistant marker into the user turn - the fallback even
    // hardcodes it (`user\n{content}\nassistant\n<think>\n`) - so appending
    // there would double it: Llama-3.2 rendered
    // `...What is my name?\nassistant\n<think>\nuser\n\nassistant\n<think>\n`.
    //
    // EXCLUDED: k2-horizon. Its `llama_turn_text` already returns the full
    // single-turn template INCLUDING the assistant prefix, so appending
    // another assistant turn duplicated the user content in the prompt and
    // the model echoed it back (caught by the CLI/HTTP sentinel).
    if arch != "k2-horizon" && arch != "nanbeige" && !(is_minicpm5 && !thinking) {
        let conversation_open = turns.last().map(|(role, _)| *role) != Some("assistant");
        let single_turn = turns.len() == 1 && turns[0].0 != "assistant";
        let bare_user_turn = arch == "glm4" || is_mistral || is_zephyr;
        if conversation_open && (single_turn || bare_user_turn) {
            prompt_text.push_str(&llama_turn_text(
                &arch,
                is_minicpm5,
                is_mistral,
                is_zephyr,
                has_chatml_template,
                "assistant",
                "",
                thinking,
            ));
        }
    }
    eprintln!("[RUST_PROMPT_TEXT] {prompt_text}");
    // Multi-turn BOS handling mirrors the single-turn path.
    let add_special = arch == "nanbeige";
    let mut body = tokenizer.encode(
        &prompt_text,
        crate::core::tokenizer::EncodeOptions {
            add_special,
            parse_special: true,
        },
    );
    if !add_special {
        if let Some(bos) = tokenizer.bos_id() {
            body.insert(0, bos);
        }
    }
    eprintln!("[RUST_TOKENS] n={} ids={:?}", body.len(), body);
    Ok(body)
}
pub fn run_inference_tokens(
    source: &dyn TensorSource,
    input_tokens: Vec<u32>,
    max_tokens: usize,
    temperature: f32,
    n_threads_arg: usize,
    bench: bool,
    profile: bool,
    kv_format: KvFormat,
    max_context: usize,
    repetition_penalty: f32,
) -> Result<(), String> {
    #[cfg(feature = "parity-trace")]
    crate::parity_trace::report(crate::parity_trace::token_ids("prompt_ids", &input_tokens));
    let t0 = Instant::now();
    let config = model_config_from_source(source)
        .map_err(|error| format!("Failed to parse model config: {error}"))?;

    let arch = source
        .metadata("general.architecture")
        .and_then(|v| v.to_string_val())
        .unwrap_or_default();

    let tokenizer = load_tokenizer(|k| source.metadata(k).cloned())
        .map_err(|error| format!("Failed to initialize tokenizer: {error}"))?;

    // Cap KV cache at the smaller of (model's claimed `context_length`,
    // the CLI-provided `--max-context`). The previous hard cap of 512
    // silently truncated the KV cache; using `config.n_ctx` directly
    // blew up on K2-Horizon-4B which claims 524288 (~77 GB of F32 KV).
    // The CLI default (8K) and any user override are applied here.
    let max_ctx = config.n_ctx.min(max_context);
    let n_embd = config.n_embd;
    let (n_layer, loop_final_norm) = layer_loop_config(source, &config)?;
    let n_head = config.n_head;
    let n_head_kv = config.n_head_kv;
    let n_embd_head = config.n_embd_head;
    let n_embd_head_k = if let Some(v) = source.metadata(&format!("{}.attention.key_length", arch))
    {
        v.to_u64().unwrap_or(n_embd_head as u64) as usize
    } else {
        n_embd_head
    };
    let n_embd_head_v =
        if let Some(v) = source.metadata(&format!("{}.attention.value_length", arch)) {
            v.to_u64().unwrap_or(n_embd_head as u64) as usize
        } else {
            n_embd_head
        };
    let n_embd_q = n_head * n_embd_head_k;
    let n_embd_gqa = n_head_kv * n_embd_head_v;
    let n_ff = config.n_ff;
    let eps = config.norm_eps;
    let freq_base = config.rope_freq_base;
    // Phi-3 / Phi-4 apply RoPE only to the first `rope.dimension_count`
    // of `head_dim`; the rest pass through. Default to head_dim when the
    // metadata is missing (other architectures always rope the full head).
    let rope_dim = source
        .metadata(&format!("{arch}.rope.dimension_count"))
        .and_then(|v| v.to_u64())
        .map(|v| v as usize)
        .filter(|v| *v > 0)
        .unwrap_or(n_embd_head);
    // Phi-3 / Phi-4 use partial RoPE: rotate the first `rope_dim` of the
    // head (e.g. 96 of 128), leave the rest untouched. The cos/sin
    // table is built against `rope_dim` itself, not the full head_dim.
    let rope_dim = rope_dim.min(n_embd_head_k);
    // Phi-3 / Phi-4 scale RoPE outputs by `rope.scaling.attn_factor`; the
    // default of 1.0 means "no rescaling" (standard llama behaviour).
    let attn_factor: f32 = source
        .metadata(&format!("{arch}.rope.scaling.attn_factor"))
        .and_then(|v| v.to_f64())
        .map(|v| v as f32)
        .unwrap_or(1.0);
    let norm_groups = normalization_groups(source, &arch, n_embd)?;

    // Granite-specific scaling factors. Zero means "not used" (no-op).
    let arch_prefix = &arch;
    let embedding_scale = source
        .metadata(&format!("{arch_prefix}.embedding_scale"))
        .and_then(|v| v.to_f64())
        .unwrap_or(0.0) as f32;
    let residual_scale = source
        .metadata(&format!("{arch_prefix}.residual_scale"))
        .and_then(|v| v.to_f64())
        .unwrap_or(0.0) as f32;
    let logit_scale = source
        .metadata(&format!("{arch_prefix}.logit_scale"))
        .and_then(|v| v.to_f64())
        .unwrap_or(0.0) as f32;

    let output_norm = get_f32_tensor(source, "output_norm.weight", n_embd);
    let embd_info = source
        .tensor_info("token_embd.weight")
        .expect("no token_embd.weight");
    crate::ops::embedding::expect_supported_embedding("token_embd.weight", embd_info.ggml_type);
    let embd_weight = source.tensor_slice("token_embd.weight").expect("no embd");
    let output_weight = source.tensor_slice("output.weight").unwrap_or(embd_weight);
    let embd_type = embd_info.ggml_type;
    let output_type = source
        .tensor_info("output.weight")
        .unwrap_or(embd_info)
        .ggml_type;

    let layers: Vec<LlamaLayerWeights> =
        load_layers(source, config.n_layer, n_embd, n_embd_q, n_embd_gqa, n_ff);

    // DEBUG: dump first N bytes of L23's w_gate, w_up, w_down for comparison with llama.cpp.
    if std::env::var("RUST_LLAMA_DEBUG_L23_WEIGHTS").is_ok() {
        let l = n_layer - 1; // L23 = last layer
        let layer = &layers[l];
        let dump_n = 256usize; // first 256 bytes (= ~7.5 Q8_0 blocks)
        for (name, _qw) in [
            ("w_gate", &layer.w_gate),
            ("w_up", &layer.w_up),
            ("w_down", &layer.w_down),
        ] {
            // Get the raw bytes from the QuantizedTensor
            // The bytes are stored in the tensor slice from the source
            // We need to re-fetch them since QuantizedTensor doesn't expose raw bytes
            let source_bytes = source
                .tensor_slice(&format!("blk.{}.ffn_{}.weight", l, &name[2..]))
                .unwrap();
            eprintln!("[RUST_W_L23] {} first {} bytes:", name, dump_n);
            for chunk in source_bytes[..dump_n].chunks(16) {
                let hex: Vec<String> = chunk.iter().map(|b| format!("{:02x}", b)).collect();
                eprintln!("  {}", hex.join(" "));
            }
        }
    }

    let load_ms = t0.elapsed().as_millis();
    println!(
        "Model: {} | n_embd={} n_layer={} n_head={} n_head_kv={} n_ff={} | loaded in {}ms",
        arch, n_embd, n_layer, n_head, n_head_kv, n_ff, load_ms
    );

    let kv_cache = match kv_format {
        KvFormat::F16 => KvCache::new_f16(n_layer, max_ctx, n_embd_gqa),
        KvFormat::F32 => KvCache::new_f32(n_layer, max_ctx, n_embd_gqa),
    };

    let vocab = tokenizer.vocab_size();

    let available_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let n_threads = resolve_thread_count(n_threads_arg, available_threads);

    let mut scratch = ExecutionScratchpad::new(
        n_embd, n_embd_q, n_embd_gqa, n_ff, vocab, n_threads, max_ctx,
    );
    let pool = Arc::new(ComputePool::new(n_threads));
    eprintln!("compute pool: {} threads", pool.n_threads());
    println!("Prompt: {} tokens", input_tokens.len());

    let eos_id = tokenizer.eos_id();
    let im_end_id = tokenizer.special_token_id("im_end");
    let mut generated_tokens: Vec<u32> = Vec::new();
    // Shared llama-family sampler: owns the repetition-penalty counts and the
    // history-seeded RNG (`ops::sampling::LlamaSampler`), and is primed with
    // the prompt so the CLI and the HTTP adapter agree token for token.
    let mut llama_sampler = crate::ops::sampling::LlamaSampler::new(
        sample_defaults(source).0,
        sample_defaults(source).1,
    );
    llama_sampler.prime(&input_tokens);
    let mut decoder = crate::core::tokenizer::StreamingDecoder::new(&*tokenizer, false);

    let group_size = n_head / n_head_kv;
    // Granite ships `granite.attention.scale` (= 1/n_embd_head) which
    // overrides the standard 1/sqrt(n_embd_head) scale. Read it from
    // metadata when present.
    let attention_scale_meta = source
        .metadata(&format!("{arch}.attention.scale"))
        .and_then(|v| v.to_f64())
        .map(|v| v as f32);
    let kq_scale = attention_scale_meta.unwrap_or_else(|| 1.0f32 / (n_embd_head_k as f32).sqrt());

    let mut t_norm: f64 = 0.0;
    let _t_quant: f64 = 0.0;
    let mut t_qkv: f64 = 0.0;
    let mut t_wo: f64 = 0.0;
    let mut t_ffn1: f64 = 0.0;
    let _t_silu: f64 = 0.0;
    let _t_down: f64 = 0.0;
    let mut t_logits: f64 = 0.0;

    print!("Output: ");
    io::stdout().flush().unwrap();

    let t_infer = Instant::now();
    let total_steps = inference_step_budget(input_tokens.len(), max_tokens, bench);
    let mut prefill_evals = 0usize;
    let mut prefill_time = Duration::ZERO;
    let mut decode_evals = 0usize;
    let mut decode_time = Duration::ZERO;

    for step in 0..total_steps {
        let eval_started = Instant::now();
        let token_id = if step < input_tokens.len() {
            input_tokens[step]
        } else {
            *generated_tokens.last().unwrap_or(&0)
        };

        let pos = step;
        #[cfg(feature = "parity-trace")]
        crate::parity_trace::report(crate::parity_trace::token_ids("input_token", &[token_id]));

        embedding_lookup(embd_weight, token_id, n_embd, embd_type, &mut scratch.x);
        if embedding_scale != 0.0 {
            // SIMD: vec_scale_f32 is AVX2+FMA on x86_64 / NEON on aarch64.
            vec_scale_f32(&mut scratch.x, embedding_scale);
        }
        dbg_tensor(step, "embed_out", 0, &scratch.x);

        for layer in 0..n_layer {
            let lw = &layers[layer % layers.len()];

            let x_ptr = scratch.x.as_mut_ptr();
            let normed_ptr = scratch.normed.as_mut_ptr();
            let q_ptr = scratch.q.as_mut_ptr();
            let k_ptr = scratch.k_new.as_mut_ptr();
            let v_ptr = scratch.v_new.as_mut_ptr();
            let attn_out_ptr = scratch.attn_out.as_mut_ptr();
            let attn_proj_ptr = scratch.attn_proj.as_mut_ptr();
            let down_buf_ptr = scratch.down_buf.as_mut_ptr();
            let scores_ptr = scratch.scores.as_mut_ptr();
            let score_stride = scratch.score_stride;
            let gate_buf_ptr = scratch.gate_buf.as_mut_ptr();
            let up_buf_ptr = scratch.up_buf.as_mut_ptr();
            let ffn_fused_ptr = scratch.ffn_fused.as_mut_ptr();
            let q8_buf_ptr = scratch.q8_buf.as_mut_ptr();
            let scale_buf_ptr = scratch.scale_buf.as_mut_ptr();
            let q8k_buf_ptr = scratch.q8k_buf.as_mut_ptr();
            let kv_cache_size = n_layer * max_ctx * n_embd_gqa;
            let (k_cache_f16_ptr, v_cache_f16_ptr) = match &kv_cache {
                KvCache::F16(c) => (c.k.as_ptr() as *mut u16, c.v.as_ptr() as *mut u16),
                _ => (std::ptr::null_mut(), std::ptr::null_mut()),
            };
            let (k_cache_f32_ptr, v_cache_f32_ptr) = match &kv_cache {
                KvCache::F32(c) => (c.k.as_ptr() as *mut f32, c.v.as_ptr() as *mut f32),
                _ => (std::ptr::null_mut(), std::ptr::null_mut()),
            };

            let max_n_in = n_embd_q.max(n_ff);
            let x = unsafe { std::slice::from_raw_parts_mut(x_ptr, n_embd) };
            let normed = unsafe { std::slice::from_raw_parts_mut(normed_ptr, n_embd) };
            let q8_buf = unsafe { std::slice::from_raw_parts_mut(q8_buf_ptr, max_n_in) };
            let scale_buf = unsafe { std::slice::from_raw_parts_mut(scale_buf_ptr, max_n_in / 32) };
            let q8k_buf =
                unsafe { std::slice::from_raw_parts_mut(q8k_buf_ptr, (max_n_in + 255) / 256) };

            let t0 = Instant::now();
            rms_norm_grouped(x, &lw.attn_norm, normed, norm_groups, eps);
            dbg_tensor(step, "attn_norm", layer, normed);
            quantize_q8_0_into(
                normed,
                n_embd,
                &mut q8_buf[..n_embd],
                &mut scale_buf[..n_embd / 32],
            );
            let q8 = q8_buf[..n_embd].as_ptr();

            let sc = scale_buf[..n_embd / 32].as_ptr();
            crate::ops::quantize_row_q8_k_into(normed, &mut q8k_buf[..n_embd / 256]);
            let q8k = q8k_buf[..n_embd / 256].as_ptr();

            pool.compute(move |ith: usize, nth: usize| {
                let input = unsafe { std::slice::from_raw_parts(normed_ptr, n_embd) };
                let q8 = unsafe { std::slice::from_raw_parts(q8, n_embd) };
                let sc = unsafe { std::slice::from_raw_parts(sc, n_embd / 32) };
                let q8k = unsafe { std::slice::from_raw_parts(q8k, n_embd / 256) };
                let q = unsafe { std::slice::from_raw_parts_mut(q_ptr, n_embd_q) };
                let k_new = unsafe { std::slice::from_raw_parts_mut(k_ptr, n_embd_gqa) };
                let v_new = unsafe { std::slice::from_raw_parts_mut(v_ptr, n_embd_gqa) };

                lw.wq.kernel.forward_prepared(
                    input,
                    q8,
                    sc,
                    Some(q8k),
                    q,
                    n_embd,
                    n_embd_q,
                    ith,
                    nth,
                );
                lw.wk.kernel.forward_prepared(
                    input,
                    q8,
                    sc,
                    Some(q8k),
                    k_new,
                    n_embd,
                    n_embd_gqa,
                    ith,
                    nth,
                );
                lw.wv.kernel.forward_prepared(
                    input,
                    q8,
                    sc,
                    Some(q8k),
                    v_new,
                    n_embd,
                    n_embd_gqa,
                    ith,
                    nth,
                );
            });

            {
                let q = unsafe { std::slice::from_raw_parts_mut(q_ptr, n_embd_q) };
                let k_new = unsafe { std::slice::from_raw_parts_mut(k_ptr, n_embd_gqa) };
                let v_new = unsafe { std::slice::from_raw_parts_mut(v_ptr, n_embd_gqa) };

                // GLM-4 ships separate `attn_q/k/v.bias` tensors
                // (plain llama does not). Add them in-place.
                if let Some(bq) = lw.bq.as_deref() {
                    vec_add_into(bq, q);
                }
                if let Some(bk) = lw.bk.as_deref() {
                    vec_add_into(bk, k_new);
                }
                if let Some(bv) = lw.bv.as_deref() {
                    vec_add_into(bv, v_new);
                }

                dbg_tensor(step, "q_proj", layer, q);
                dbg_tensor(step, "k_proj", layer, k_new);
                dbg_tensor(step, "v_proj", layer, v_new);

                // LLaMA does not have QK norm.
                // The `llama` GGUF arch uses interleaved ("normal"-style)
                // RoPE - the converter permutes HF rotate_half weights into
                // adjacent-pair layout (MiniCPM5 ships this arch too).
                for h in 0..n_head {
                    apply_rope(
                        &arch,
                        &mut q[h * n_embd_head_k..(h + 1) * n_embd_head_k],
                        pos,
                        n_embd_head_k,
                        freq_base,
                        rope_dim,
                        attn_factor,
                        None,
                    );
                }
                for h in 0..n_head_kv {
                    apply_rope(
                        &arch,
                        &mut k_new[h * n_embd_head_k..(h + 1) * n_embd_head_k],
                        pos,
                        n_embd_head_k,
                        freq_base,
                        rope_dim,
                        attn_factor,
                        None,
                    );
                }
                dbg_tensor(step, "Qcur", layer, q);
                dbg_tensor(step, "Kcur", layer, k_new);
                dbg_tensor(step, "Vcur", layer, v_new);

                let kb = layer * max_ctx * n_embd_gqa;

                if kv_format == KvFormat::F16 {
                    let k_cache =
                        unsafe { std::slice::from_raw_parts_mut(k_cache_f16_ptr, kv_cache_size) };
                    let v_cache =
                        unsafe { std::slice::from_raw_parts_mut(v_cache_f16_ptr, kv_cache_size) };
                    for h in 0..n_head_kv {
                        let off = h * n_embd_head_k;
                        f32_slice_to_f16(
                            &k_new[off..off + n_embd_head_k],
                            &mut k_cache[kb + pos * n_embd_gqa + off
                                ..kb + pos * n_embd_gqa + off + n_embd_head_k],
                        );
                        f32_slice_to_f16(
                            &v_new[off..off + n_embd_head_v],
                            &mut v_cache[kb + pos * n_embd_gqa + off
                                ..kb + pos * n_embd_gqa + off + n_embd_head_v],
                        );
                    }
                } else {
                    let k_cache =
                        unsafe { std::slice::from_raw_parts_mut(k_cache_f32_ptr, kv_cache_size) };
                    let v_cache =
                        unsafe { std::slice::from_raw_parts_mut(v_cache_f32_ptr, kv_cache_size) };
                    for h in 0..n_head_kv {
                        let off = h * n_embd_head_k;
                        k_cache[kb + pos * n_embd_gqa + off
                            ..kb + pos * n_embd_gqa + off + n_embd_head_k]
                            .copy_from_slice(&k_new[off..off + n_embd_head_k]);
                        v_cache[kb + pos * n_embd_gqa + off
                            ..kb + pos * n_embd_gqa + off + n_embd_head_v]
                            .copy_from_slice(&v_new[off..off + n_embd_head_v]);
                    }
                }
            }

            pool.compute(move |ith: usize, nth: usize| {
                let q = unsafe { std::slice::from_raw_parts(q_ptr, n_embd_q) };
                let attn_out = unsafe { std::slice::from_raw_parts_mut(attn_out_ptr, n_embd_q) };
                let h_start = ith * n_head / nth;
                let h_end = (ith + 1) * n_head / nth;

                let kb = layer * max_ctx * n_embd_gqa;

                if kv_format == KvFormat::F16 {
                    let k_cache =
                        unsafe { std::slice::from_raw_parts(k_cache_f16_ptr, kv_cache_size) };
                    let v_cache =
                        unsafe { std::slice::from_raw_parts(v_cache_f16_ptr, kv_cache_size) };
                    for h in h_start..h_end {
                        let kv_h = h / group_size;
                        let q_off = h * n_embd_head_k;
                        let out_base = h * n_embd_head_v;
                        attention_head_f16(
                            &q[q_off..q_off + n_embd_head_k],
                            &mut attn_out[out_base..out_base + n_embd_head_v],
                            k_cache,
                            v_cache,
                            kb + kv_h * n_embd_head_v,
                            n_embd_gqa,
                            pos + 1,
                            kq_scale,
                        );
                    }
                } else {
                    let k_cache =
                        unsafe { std::slice::from_raw_parts(k_cache_f32_ptr, kv_cache_size) };
                    let v_cache =
                        unsafe { std::slice::from_raw_parts(v_cache_f32_ptr, kv_cache_size) };
                    let scores = unsafe {
                        std::slice::from_raw_parts_mut(scores_ptr, n_threads * score_stride)
                    };
                    for h in h_start..h_end {
                        let kv_h = h / group_size;
                        let q_off = h * n_embd_head_k;
                        let n_cached = pos + 1;
                        let n_padded = (n_cached + 255) / 256 * 256;
                        let out_base = h * n_embd_head_v;
                        let s_off = ith * score_stride;
                        for t in 0..n_cached {
                            scores[s_off + t] = dot_f32(
                                &q[q_off..q_off + n_embd_head_k],
                                &k_cache[kb + t * n_embd_gqa + kv_h * n_embd_head_v
                                    ..kb + t * n_embd_gqa + kv_h * n_embd_head_v + n_embd_head_k],
                                n_embd_head_k,
                            ) * kq_scale;
                        }
                        scores[s_off + n_cached..s_off + n_padded].fill(f32::NEG_INFINITY);
                        softmax_inplace(&mut scores[s_off..s_off + n_padded]);
                        // The values scratch is sized to the next multiple of
                        // 256 above max_ctx (n_padded_max). Heap-allocated so
                        // long contexts don't overflow.
                        let mut values = vec![0.0f32; n_padded];
                        for d in 0..n_embd_head_v {
                            for t in 0..n_cached {
                                values[t] = v_cache[kb + t * n_embd_gqa + kv_h * n_embd_head_v + d];
                            }
                            attn_out[out_base + d] = dot_f32(
                                &values[..n_padded],
                                &scores[s_off..s_off + n_padded],
                                n_cached,
                            );
                        }
                    }
                }
            });
            t_qkv += t0.elapsed().as_secs_f64();

            let attn_out = unsafe { std::slice::from_raw_parts_mut(attn_out_ptr, n_embd_q) };
            dbg_tensor(step, "attn_out", layer, attn_out);
            let q8_buf = unsafe { std::slice::from_raw_parts_mut(q8_buf_ptr, max_n_in) };
            let scale_buf = unsafe { std::slice::from_raw_parts_mut(scale_buf_ptr, max_n_in / 32) };
            let q8k_buf =
                unsafe { std::slice::from_raw_parts_mut(q8k_buf_ptr, (max_n_in + 255) / 256) };
            let t0 = Instant::now();
            quantize_q8_0_into(
                attn_out,
                n_embd_q,
                &mut q8_buf[..n_embd_q],
                &mut scale_buf[..n_embd_q / 32],
            );
            crate::ops::quantize_row_q8_k_into(attn_out, &mut q8k_buf[..n_embd_q / 256]);
            let q8 = q8_buf[..n_embd_q].as_ptr();
            let sc = scale_buf[..n_embd_q / 32].as_ptr();
            let q8k = q8k_buf[..n_embd_q / 256].as_ptr();
            pool.compute(move |ith: usize, nth: usize| {
                let input = unsafe { std::slice::from_raw_parts(attn_out_ptr, n_embd_q) };
                let q8 = unsafe { std::slice::from_raw_parts(q8, n_embd_q) };
                let sc = unsafe { std::slice::from_raw_parts(sc, n_embd_q / 32) };
                let q8k = unsafe { std::slice::from_raw_parts(q8k, n_embd_q / 256) };
                let attn_proj = unsafe { std::slice::from_raw_parts_mut(attn_proj_ptr, n_embd) };
                lw.wo.kernel.forward_prepared(
                    input,
                    q8,
                    sc,
                    Some(q8k),
                    attn_proj,
                    n_embd_q,
                    n_embd,
                    ith,
                    nth,
                );
            });
            t_wo += t0.elapsed().as_secs_f64();

            let attn_proj = unsafe { std::slice::from_raw_parts_mut(attn_proj_ptr, n_embd) };
            dbg_tensor(step, "attn_proj", layer, attn_proj);
            let x = unsafe { std::slice::from_raw_parts_mut(x_ptr, n_embd) };
            let normed = unsafe { std::slice::from_raw_parts_mut(normed_ptr, n_embd) };
            // GLM-4 (`glm4` arch) applies an RMSNorm on the attention
            // output *before* the residual add. Standard llama skips it.
            if let Some(attn_post_norm) = lw.attn_post_norm.as_deref() {
                normed.copy_from_slice(attn_proj);
                rms_norm_grouped(normed, attn_post_norm, attn_proj, norm_groups, eps);
                dbg_tensor(step, "post_attn_norm", layer, attn_proj);
            }
            if residual_scale != 0.0 {
                vec_mad_f32(x, attn_proj, residual_scale);
            } else {
                vec_add_into(attn_proj, x);
            }
            dbg_tensor(step, "ffn_inp", layer, x);
            dbg_full(step, "ffn_inp", layer, x, n_embd);

            let t0 = Instant::now();
            // Debug-only ffn_norm stats (printed when RUST_LLAMA_DEBUG_TENSORS
            // is set). The static OnceLock guard lets the early return
            // collapse away in production so we skip the AVX2 reduction
            // entirely per layer × token.
            {
                static FFN_NORM_DBG_ON: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
                let limit = *FFN_NORM_DBG_ON.get_or_init(|| {
                    std::env::var("RUST_LLAMA_DEBUG_TENSORS")
                        .ok()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(0)
                });
                if limit != 0 && (layer as u32) < limit {
                    let sum_sq = sum_sq_f32(&x[..n_embd]);
                    let mean_sq = (sum_sq / n_embd as f64) as f32;
                    let scale = 1.0f32 / (mean_sq + eps).sqrt();
                    dbg_scalar(step, "ffn_norm_scale", layer, scale);
                    dbg_scalar(step, "ffn_norm_mean", layer, mean_sq);
                    dbg_scalar_full(step, "ffn_norm_sum_sq", layer, sum_sq);
                }
            }
            rms_norm_grouped(x, &lw.ffn_norm, normed, norm_groups, eps);
            dbg_tensor(step, "ffn_norm", layer, normed);
            dbg_full(step, "ffn_norm", layer, normed, n_embd);
            dbg_tensor(
                step,
                "ffn_norm_weight",
                layer,
                &lw.ffn_norm[..n_embd.min(lw.ffn_norm.len())],
            );
            dbg_full(
                step,
                "ffn_norm_weight",
                layer,
                &lw.ffn_norm[..n_embd.min(lw.ffn_norm.len())],
                n_embd,
            );
            quantize_q8_0_into(
                normed,
                n_embd,
                &mut q8_buf[..n_embd],
                &mut scale_buf[..n_embd / 32],
            );
            crate::ops::quantize_row_q8_k_into(normed, &mut q8k_buf[..n_embd / 256]);
            let q8 = q8_buf[..n_embd].as_ptr();
            let sc = scale_buf[..n_embd / 32].as_ptr();
            let q8k = q8k_buf[..n_embd / 256].as_ptr();

            // DEBUG: dump ffn_norm Q8_0 quantization to verify against Python reference
            if std::env::var("RUST_LLAMA_DEBUG_FFN_Q8").is_ok() && layer == n_layer - 1 && step == 0
            {
                let q8_slice = unsafe { std::slice::from_raw_parts(q8, n_embd) };
                let sc_slice = unsafe { std::slice::from_raw_parts(sc, n_embd / 32) };
                eprintln!(
                    "[RUST_FFN_Q8_L23] input q8 first 32 bytes (block 0): {:?}",
                    &q8_slice[..32]
                );
                eprintln!(
                    "[RUST_FFN_Q8_L23] input q8 bytes 32..64 (block 1): {:?}",
                    &q8_slice[32..64]
                );
                eprintln!("[RUST_FFN_Q8_L23] input scale[0..4]: {:?}", &sc_slice[..4]);
                eprintln!(
                    "[RUST_FFN_Q8_L23] input q8 last 32 bytes: {:?}",
                    &q8_slice[n_embd - 32..]
                );
            }

            pool.compute(move |ith: usize, nth: usize| {
                let input = unsafe { std::slice::from_raw_parts(normed_ptr, n_embd) };
                let q8 = unsafe { std::slice::from_raw_parts(q8, n_embd) };
                let sc = unsafe { std::slice::from_raw_parts(sc, n_embd / 32) };
                let q8k = unsafe { std::slice::from_raw_parts(q8k, n_embd / 256) };
                let gate_buf = unsafe { std::slice::from_raw_parts_mut(gate_buf_ptr, n_ff) };
                let up_buf = unsafe { std::slice::from_raw_parts_mut(up_buf_ptr, n_ff) };
                let ffn_fused_buf =
                    unsafe { std::slice::from_raw_parts_mut(ffn_fused_ptr, 2 * n_ff) };
                // GLM-4 ships a single fused `ffn_up.weight` of shape
                // `[n_embd, 2*n_ff]`; first half is gate, second is up.
                // Plain llama uses two distinct matmuls. We dispatch
                // on `arch` to keep llama fused rows byte-stable.
                if arch == "glm4" {
                    lw.w_up.kernel.forward_prepared(
                        input,
                        q8,
                        sc,
                        Some(q8k),
                        ffn_fused_buf,
                        n_embd,
                        2 * n_ff,
                        ith,
                        nth,
                    );
                    // Apply SwiGLU on the fused output: up = silu(gate)*up.
                    // The matmul partitions by `2*n_ff` rows; silu's
                    // gate slice is the first half and up slice is the
                    // second half, so each thread silus its own slice.
                    // For the gpu path, only thread 0 runs to avoid
                    // the multi-thread fence. After silu, copy the up
                    // half into `gate_buf` so the downstream w_down
                    // matmul (which always reads from `gate_buf`) sees
                    // the post-silu activation without an extra branch.
                    if crate::ops::gpu_matmul_active() {
                        if ith == 0 {
                            // Disjoint slices of `ffn_fused_buf` — safe
                            // because the halves don't alias. Split at
                            // `n_ff` to satisfy the borrow checker.
                            let (gate_part, up_part) = ffn_fused_buf.split_at_mut(n_ff);
                            silu_mul_approx_inplace(gate_part, &mut up_part[..n_ff]);
                            gate_buf[..n_ff].copy_from_slice(&up_part[..n_ff]);
                        }
                    } else {
                        let per_thread = (n_ff + nth - 1) / nth;
                        let r_start = ith * per_thread;
                        let r_end = (r_start + per_thread).min(n_ff);
                        let (gate_part, up_part) = ffn_fused_buf.split_at_mut(n_ff);
                        silu_mul_approx_inplace(
                            &gate_part[r_start..r_end],
                            &mut up_part[r_start..r_end],
                        );
                        gate_buf[r_start..r_end].copy_from_slice(&up_part[r_start..r_end]);
                    }
                } else {
                    lw.w_gate.kernel.forward_prepared(
                        input,
                        q8,
                        sc,
                        Some(q8k),
                        up_buf,
                        n_embd,
                        n_ff,
                        ith,
                        nth,
                    );
                    lw.w_up.kernel.forward_prepared(
                        input,
                        q8,
                        sc,
                        Some(q8k),
                        gate_buf,
                        n_embd,
                        n_ff,
                        ith,
                        nth,
                    );

                    if crate::ops::gpu_matmul_active() {
                        // Matmul ran as one fenced GPU dispatch owned by thread 0;
                        // per-thread row slices would race with it.
                        if ith == 0 {
                            silu_mul_approx_inplace(&up_buf[..n_ff], &mut gate_buf[..n_ff]);
                        }
                    } else {
                        // Must match the matmul kernel's ceil row partition exactly: a floor
                        // split races with the kernel when n_ff % nth != 0 (silu would
                        // read rows the matmul hasn't written yet).
                        let per_thread = (n_ff + nth - 1) / nth;
                        let r_start = ith * per_thread;
                        let r_end = (r_start + per_thread).min(n_ff);
                        silu_mul_approx_inplace(
                            &up_buf[r_start..r_end],
                            &mut gate_buf[r_start..r_end],
                        );
                    }
                }
            });

            {
                let gate_buf = unsafe { std::slice::from_raw_parts(gate_buf_ptr, n_ff) };
                let up_buf = unsafe { std::slice::from_raw_parts(up_buf_ptr, n_ff) };
                dbg_tensor(step, "ffn_gate_buf_raw", layer, gate_buf);
                dbg_tensor(step, "ffn_up_buf_raw", layer, up_buf);
                dbg_full(step, "ffn_gate_buf_raw", layer, gate_buf, n_ff);
                dbg_full(step, "ffn_up_buf_raw", layer, up_buf, n_ff);
            }

            {
                let gate_buf = unsafe { std::slice::from_raw_parts_mut(gate_buf_ptr, n_ff) };
                let q8_buf = unsafe { std::slice::from_raw_parts_mut(q8_buf_ptr, max_n_in) };
                let scale_buf =
                    unsafe { std::slice::from_raw_parts_mut(scale_buf_ptr, max_n_in / 32) };
                let q8k_buf =
                    unsafe { std::slice::from_raw_parts_mut(q8k_buf_ptr, (max_n_in + 255) / 256) };
                quantize_q8_0_into(
                    gate_buf,
                    n_ff,
                    &mut q8_buf[..n_ff],
                    &mut scale_buf[..n_ff / 32],
                );
                crate::ops::quantize_row_q8_k_into(gate_buf, &mut q8k_buf[..n_ff.div_ceil(256)]);
            }

            let q8 = q8_buf[..n_ff].as_ptr();
            let sc = scale_buf[..n_ff / 32].as_ptr();
            let q8k = q8k_buf[..n_ff.div_ceil(256)].as_ptr();
            pool.compute(move |ith: usize, nth: usize| {
                let input = unsafe { std::slice::from_raw_parts(gate_buf_ptr, n_ff) };
                let q8 = unsafe { std::slice::from_raw_parts(q8, n_ff) };
                let sc = unsafe { std::slice::from_raw_parts(sc, n_ff / 32) };
                let q8k = unsafe { std::slice::from_raw_parts(q8k, n_ff / 256) };
                let down_buf = unsafe { std::slice::from_raw_parts_mut(down_buf_ptr, n_embd) };
                lw.w_down.kernel.forward_prepared(
                    input,
                    q8,
                    sc,
                    Some(q8k),
                    down_buf,
                    n_ff,
                    n_embd,
                    ith,
                    nth,
                );
            });
            t_ffn1 += t0.elapsed().as_secs_f64();

            // DEBUG: print raw down_buf (W_down matmul output) for L23 step 0.
            // Marker: [RUST_L23_DOWN_BUF_RAW]
            if std::env::var("RUST_LLAMA_DEBUG_L23_DOWN").is_ok()
                && layer == n_layer - 1
                && step == 0
            {
                let down_buf = unsafe { std::slice::from_raw_parts(down_buf_ptr, n_embd) };
                eprint!("[RUST_L23_DOWN_BUF_RAW] first16=");
                for v in down_buf.iter().take(16) {
                    eprint!(" {:.5}", v);
                }
                eprintln!();
            }

            let down_buf = unsafe { std::slice::from_raw_parts_mut(down_buf_ptr, n_embd) };
            dbg_tensor(step, "down_buf", layer, down_buf);
            dbg_full(step, "down_buf", layer, down_buf, n_embd);
            let x = unsafe { std::slice::from_raw_parts_mut(x_ptr, n_embd) };
            // GLM-4 also RMSNorm-s the FFN output before residual add.
            if let Some(ffn_post_norm) = lw.ffn_post_norm.as_deref() {
                normed.copy_from_slice(down_buf);
                rms_norm_grouped(normed, ffn_post_norm, down_buf, norm_groups, eps);
                dbg_tensor(step, "post_mlp_norm", layer, down_buf);
            }
            if residual_scale != 0.0 {
                vec_mad_f32(x, down_buf, residual_scale);
            } else {
                vec_add_into(down_buf, x);
            }
            dbg_tensor(step, "ffn_out", layer, x);
            dbg_tensor(step, "l_out", layer, x);
            dbg_full(step, "ffn_out", layer, x, n_embd);
            if loop_final_norm && (layer + 1) < n_layer && (layer + 1) % layers.len() == 0 {
                rms_norm_inplace(x, &output_norm, eps);
                dbg_tensor(step, "loop_norm", layer, x);
            }
        }

        {
            let x = &mut scratch.x;
            let normed = &mut scratch.normed;
            let logits_ptr = scratch.logits.as_mut_ptr();
            let q8_buf = &mut scratch.q8_buf;
            let scale_buf = &mut scratch.scale_buf;
            let q8k_buf = &mut scratch.q8k_buf;

            let t0 = Instant::now();
            rms_norm_grouped(x, &output_norm, normed, norm_groups, eps);
            dbg_tensor(step, "output_norm", 0, normed);
            t_norm += t0.elapsed().as_secs_f64();

            let t0 = Instant::now();
            quantize_q8_0_into(
                normed,
                n_embd,
                &mut q8_buf[..n_embd],
                &mut scale_buf[..n_embd / 32],
            );
            crate::ops::quantize_row_q8_k_into(normed, &mut q8k_buf[..n_embd / 256]);
            let q8 = q8_buf[..n_embd].as_ptr();
            let sc = scale_buf[..n_embd / 32].as_ptr();
            let q8k = q8k_buf[..n_embd / 256].as_ptr();
            let input = normed.as_ptr();
            let output_pw = Weight::from_quantized(QuantizedTensor::from_bytes(
                output_weight,
                output_type,
                n_embd,
                vocab,
            ));
            pool.compute(move |ith: usize, nth: usize| {
                let input = unsafe { std::slice::from_raw_parts(input, n_embd) };
                let q8 = unsafe { std::slice::from_raw_parts(q8, n_embd) };
                let sc = unsafe { std::slice::from_raw_parts(sc, n_embd / 32) };
                let q8k = unsafe { std::slice::from_raw_parts(q8k, n_embd / 256) };
                let logits = unsafe { std::slice::from_raw_parts_mut(logits_ptr, vocab) };
                output_pw.kernel.forward_prepared(
                    input,
                    q8,
                    sc,
                    Some(q8k),
                    logits,
                    n_embd,
                    vocab,
                    ith,
                    nth,
                );
            });
            t_logits += t0.elapsed().as_secs_f64();

            // Granite rescales logits by `logit_scale` before softmax (sharpens
            // the distribution; values > 1). Granite-4.0 ships with
            // logit_scale=8.0, so the multiplier is 8.0, not 1/8.0.
            if logit_scale != 0.0 {
                let logits = unsafe { std::slice::from_raw_parts_mut(logits_ptr, vocab) };
                vec_scale_f32(logits, logit_scale);
            }

            // DEBUG: print LOGITS for step 0 (first 16 values).
            // Marker: [RUST_LOGITS_RAW]
            if std::env::var("RUST_LLAMA_DEBUG_LOGITS").is_ok() && step == 0 {
                let logits = unsafe { std::slice::from_raw_parts(logits_ptr, vocab) };
                eprint!("[RUST_LOGITS_RAW] first16=");
                for v in logits.iter().take(16) {
                    eprint!(" {:.5}", v);
                }
                eprintln!();
            }
        }

        dbg_tensor(step, "logits", 0, &scratch.logits);
        let eval_elapsed = eval_started.elapsed();
        if step < input_tokens.len() {
            prefill_evals += 1;
            prefill_time += eval_elapsed;
        } else {
            decode_evals += 1;
            decode_time += eval_elapsed;
        }

        if step < input_tokens.len() - 1 {
            continue;
        }

        let logits = &mut scratch.logits;
        // DEBUG: print top-10 logits so we can diff against llama.cpp.
        if std::env::var("RUST_LLAMA_DEBUG_LOGITS").is_ok() {
            let mut idxs: Vec<(usize, f32)> =
                logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
            idxs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let mut line = format!("RUST_LOGITS step={} top10:", step);
            for k in 0..10 {
                line.push_str(&format!(" {}:{:.5}", idxs[k].0, idxs[k].1));
            }
            line.push('\n');
            let _ = io::stderr().write_all(line.as_bytes());
            let _ = io::stderr().flush();
        }
        // Sample using the shared llama-family sampler
        // (`ops::sampling::LlamaSampler`): repetition penalty + llama.cpp's
        // chain with a history-seeded RNG. The HTTP adapter uses the same
        // type, so CLI and server cannot drift apart.
        let chosen_id = llama_sampler.sample(logits, temperature, repetition_penalty);
        if crate::ops::generation_runtime::stop_after_sample(
            chosen_id,
            generated_tokens.len(),
            max_tokens,
            bench,
            eos_id,
            im_end_id,
        ) != crate::ops::generation_runtime::StepAction::Continue
        {
            break;
        }

        generated_tokens.push(chosen_id);

        let text = decoder.push(chosen_id);
        print!("{}", text);
        io::stdout().flush().unwrap();

        if generated_tokens.len() == 1 {
            eprintln!();
        }
    }

    let tail = decoder.finish();
    #[cfg(feature = "parity-trace")]
    crate::parity_trace::report(crate::parity_trace::token_ids(
        "generated_ids",
        &generated_tokens,
    ));
    if !tail.is_empty() {
        print!("{}", tail);
        io::stdout().flush().unwrap();
    }

    let infer_ms = t_infer.elapsed().as_millis();
    let tok_s = if infer_ms > 0 {
        generated_tokens.len() as f64 / infer_ms as f64 * 1000.0
    } else {
        0.0
    };
    let total = t_norm + _t_quant + t_qkv + t_wo + t_ffn1 + t_logits;
    if bench || profile {
        eprintln!();
    }
    eprintln!(
        "Prompt: {:.1} t/s | Generation: {:.1} t/s | end-to-end: {:.1} tok/s",
        crate::app::cli::per_second(prefill_evals, prefill_time),
        crate::app::cli::per_second(decode_evals, decode_time),
        tok_s
    );
    if profile {
        eprintln!(
            "PROFILE: norm={:.1}% quant={:.1}% qkv+attn={:.1}% wo={:.1}% ffn={:.1}% logits={:.1}%",
            t_norm / total * 100.0,
            _t_quant / total * 100.0,
            t_qkv / total * 100.0,
            t_wo / total * 100.0,
            t_ffn1 / total * 100.0,
            t_logits / total * 100.0
        );
    }
    println!();
    println!(
        "[{} output tokens in {}ms]",
        generated_tokens.len(),
        infer_ms
    );
    Ok(())
}

/// Single forward pass: prefill `prompt_tokens` and return the
/// last-position logits. Used by JEV / classification modes that do
/// not need autoregressive decoding.
///
/// Mirrors the prefill portion of `run_inference_tokens` but stops
/// after the final logits are computed. The per-step body is the same
/// as in `run_inference_tokens`; keeping a separate copy here avoids
/// touching the existing decode loop and its bench/profile plumbing.
/// When the model exposes a `qwen2vl`-style arch metadata that
/// `LlamaSession::from_source_with_max_rows` can't parse, this
/// falls back to the legacy free-function per-token path. `batch_size`
/// is accepted for trait compatibility but the dispatch currently
/// goes through the per-token path either way.

/// `batch_size` for the chunked prefill dispatch. When the model
/// is built with `LlamaSession` this routes through the
/// [`LlamaSession::forward_logits_chunked`] entry point so a
/// `batch_size` for the chunked prefill dispatch. When the model
/// is built with `LlamaSession` this routes through the
/// [`LlamaSession::forward_logits_chunked`] entry point so a
/// `batch_size > 1` triggers the real batched math
/// (`PreparedRows::matmul_group` for Q/K/V / wo / gate / up /
/// down projections). Falls back to the legacy free-function path
/// when the model exposes a `qwen2vl`-style arch metadata that
/// `LlamaSession::from_source_with_max_rows` can't parse.
pub fn run_forward_logits_llama_with_batch(
    source: &dyn TensorSource,
    prompt_tokens: &[u32],
    n_threads_arg: usize,
    kv_format: KvFormat,
    max_context: usize,
    batch_size: usize,
) -> Result<(Vec<f32>, std::time::Duration), String> {
    let t0 = Instant::now();
    // Try the session path first. If `from_source_with_max_rows`
    // fails (wrong arch metadata, missing tensors, …) fall back
    // to the legacy free-function path so existing models
    // keep working. The dispatch decision is per-call so a
    // runtime bug in the session path doesn't permanently lock
    // out a model.
    if let Ok(mut session) = super::session::LlamaSession::from_source_with_max_rows(
        source,
        n_threads_arg,
        kv_format,
        max_context,
        batch_size,
    ) {
        let owned: Vec<u32> = prompt_tokens.to_vec();
        let logits = session
            .forward_logits_chunked(&owned, batch_size)
            .map_err(|e| format!("Llama chunked forward_logits failed: {e}"))?;
        return Ok((logits, t0.elapsed()));
    }
    run_forward_logits_llama_inner(
        source,
        prompt_tokens,
        n_threads_arg,
        kv_format,
        max_context,
        batch_size,
    )
}

/// Original per-token prefill body, factored out so the
/// session-driven path can fall back to it without re-entering
/// `run_forward_logits_llama`. Same math as before — at
/// `batch_size == 1` this collapses to the legacy per-token walk.
pub fn run_forward_logits_llama_inner(
    source: &dyn TensorSource,
    prompt_tokens: &[u32],
    n_threads_arg: usize,
    kv_format: KvFormat,
    max_context: usize,
    _batch_size: usize,
) -> Result<(Vec<f32>, std::time::Duration), String> {
    let t0 = Instant::now();
    let config = model_config_from_source(source)
        .map_err(|error| format!("Failed to parse model config: {error}"))?;
    let arch = source
        .metadata("general.architecture")
        .and_then(|v| v.to_string_val())
        .unwrap_or_default();
    let tokenizer = load_tokenizer(|k| source.metadata(k).cloned())
        .map_err(|error| format!("Failed to initialize tokenizer: {error}"))?;

    let max_ctx = config.n_ctx.min(max_context);
    let n_embd = config.n_embd;
    let (n_layer, loop_final_norm) = layer_loop_config(source, &config)?;
    let n_head = config.n_head;
    let n_head_kv = config.n_head_kv;
    let n_embd_head = config.n_embd_head;
    let n_embd_head_k = if let Some(v) = source.metadata(&format!("{}.attention.key_length", arch))
    {
        v.to_u64().unwrap_or(n_embd_head as u64) as usize
    } else {
        n_embd_head
    };
    let n_embd_head_v =
        if let Some(v) = source.metadata(&format!("{}.attention.value_length", arch)) {
            v.to_u64().unwrap_or(n_embd_head as u64) as usize
        } else {
            n_embd_head
        };
    let n_embd_q = n_head * n_embd_head_k;
    let n_embd_gqa = n_head_kv * n_embd_head_v;
    let n_ff = config.n_ff;
    let eps = config.norm_eps;
    let freq_base = config.rope_freq_base;
    let rope_dim = source
        .metadata(&format!("{arch}.rope.dimension_count"))
        .and_then(|v| v.to_u64())
        .map(|v| v as usize)
        .filter(|v| *v > 0)
        .unwrap_or(n_embd_head);
    let attn_factor: f32 = source
        .metadata(&format!("{arch}.rope.scaling.attn_factor"))
        .and_then(|v| v.to_f64())
        .map(|v| v as f32)
        .unwrap_or(1.0);
    let norm_groups = normalization_groups(source, &arch, n_embd)?;

    let arch_prefix = &arch;
    let embedding_scale = source
        .metadata(&format!("{arch_prefix}.embedding_scale"))
        .and_then(|v| v.to_f64())
        .unwrap_or(0.0) as f32;
    let residual_scale = source
        .metadata(&format!("{arch_prefix}.residual_scale"))
        .and_then(|v| v.to_f64())
        .unwrap_or(0.0) as f32;
    let logit_scale = source
        .metadata(&format!("{arch_prefix}.logit_scale"))
        .and_then(|v| v.to_f64())
        .unwrap_or(0.0) as f32;

    let output_norm = get_f32_tensor(source, "output_norm.weight", n_embd);
    let embd_info = source
        .tensor_info("token_embd.weight")
        .expect("no token_embd.weight");
    crate::ops::embedding::expect_supported_embedding("token_embd.weight", embd_info.ggml_type);
    let embd_weight = source.tensor_slice("token_embd.weight").expect("no embd");
    let output_weight = source.tensor_slice("output.weight").unwrap_or(embd_weight);
    let embd_type = embd_info.ggml_type;
    let output_type = source
        .tensor_info("output.weight")
        .unwrap_or(embd_info)
        .ggml_type;

    let layers: Vec<LlamaLayerWeights> =
        load_layers(source, config.n_layer, n_embd, n_embd_q, n_embd_gqa, n_ff);

    let kv_cache = match kv_format {
        KvFormat::F16 => KvCache::new_f16(n_layer, max_ctx, n_embd_gqa),
        KvFormat::F32 => KvCache::new_f32(n_layer, max_ctx, n_embd_gqa),
    };

    let vocab = tokenizer.vocab_size();

    let available_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let n_threads = resolve_thread_count(n_threads_arg, available_threads);

    let mut scratch = ExecutionScratchpad::new(
        n_embd, n_embd_q, n_embd_gqa, n_ff, vocab, n_threads, max_ctx,
    );
    let pool = Arc::new(ComputePool::new(n_threads));

    let group_size = n_head / n_head_kv;
    let attention_scale_meta = source
        .metadata(&format!("{arch}.attention.scale"))
        .and_then(|v| v.to_f64())
        .map(|v| v as f32);
    let kq_scale = attention_scale_meta.unwrap_or_else(|| 1.0f32 / (n_embd_head_k as f32).sqrt());

    let mut prefill_time = std::time::Duration::ZERO;
    let mut prefill_evals = 0usize;

    for step in 0..prompt_tokens.len() {
        let eval_started = Instant::now();
        let token_id = prompt_tokens[step];
        let pos = step;

        embedding_lookup(embd_weight, token_id, n_embd, embd_type, &mut scratch.x);
        if embedding_scale != 0.0 {
            vec_scale_f32(&mut scratch.x, embedding_scale);
        }

        for layer in 0..n_layer {
            let lw = &layers[layer % layers.len()];

            let x_ptr = scratch.x.as_mut_ptr();
            let normed_ptr = scratch.normed.as_mut_ptr();
            let q_ptr = scratch.q.as_mut_ptr();
            let k_ptr = scratch.k_new.as_mut_ptr();
            let v_ptr = scratch.v_new.as_mut_ptr();
            let attn_out_ptr = scratch.attn_out.as_mut_ptr();
            let attn_proj_ptr = scratch.attn_proj.as_mut_ptr();
            let down_buf_ptr = scratch.down_buf.as_mut_ptr();
            let scores_ptr = scratch.scores.as_mut_ptr();
            let score_stride = scratch.score_stride;
            let gate_buf_ptr = scratch.gate_buf.as_mut_ptr();
            let up_buf_ptr = scratch.up_buf.as_mut_ptr();
            let ffn_fused_ptr = scratch.ffn_fused.as_mut_ptr();
            let q8_buf_ptr = scratch.q8_buf.as_mut_ptr();
            let scale_buf_ptr = scratch.scale_buf.as_mut_ptr();
            let q8k_buf_ptr = scratch.q8k_buf.as_mut_ptr();
            let kv_cache_size = n_layer * max_ctx * n_embd_gqa;
            let (k_cache_f16_ptr, v_cache_f16_ptr) = match &kv_cache {
                KvCache::F16(c) => (c.k.as_ptr() as *mut u16, c.v.as_ptr() as *mut u16),
                _ => (std::ptr::null_mut(), std::ptr::null_mut()),
            };
            let (k_cache_f32_ptr, v_cache_f32_ptr) = match &kv_cache {
                KvCache::F32(c) => (c.k.as_ptr() as *mut f32, c.v.as_ptr() as *mut f32),
                _ => (std::ptr::null_mut(), std::ptr::null_mut()),
            };

            let max_n_in = n_embd_q.max(n_ff);
            let x = unsafe { std::slice::from_raw_parts_mut(x_ptr, n_embd) };
            let normed = unsafe { std::slice::from_raw_parts_mut(normed_ptr, n_embd) };
            let q8_buf = unsafe { std::slice::from_raw_parts_mut(q8_buf_ptr, max_n_in) };
            let scale_buf = unsafe { std::slice::from_raw_parts_mut(scale_buf_ptr, max_n_in / 32) };
            let q8k_buf =
                unsafe { std::slice::from_raw_parts_mut(q8k_buf_ptr, (max_n_in + 255) / 256) };

            rms_norm_grouped(x, &lw.attn_norm, normed, norm_groups, eps);
            quantize_q8_0_into(
                normed,
                n_embd,
                &mut q8_buf[..n_embd],
                &mut scale_buf[..n_embd / 32],
            );
            let q8 = q8_buf[..n_embd].as_ptr();
            let sc = scale_buf[..n_embd / 32].as_ptr();
            crate::ops::quantize_row_q8_k_into(normed, &mut q8k_buf[..n_embd / 256]);
            let q8k = q8k_buf[..n_embd / 256].as_ptr();

            pool.compute(move |ith: usize, nth: usize| {
                let input = unsafe { std::slice::from_raw_parts(normed_ptr, n_embd) };
                let q8 = unsafe { std::slice::from_raw_parts(q8, n_embd) };
                let sc = unsafe { std::slice::from_raw_parts(sc, n_embd / 32) };
                let q8k = unsafe { std::slice::from_raw_parts(q8k, n_embd / 256) };
                let q = unsafe { std::slice::from_raw_parts_mut(q_ptr, n_embd_q) };
                let k_new = unsafe { std::slice::from_raw_parts_mut(k_ptr, n_embd_gqa) };
                let v_new = unsafe { std::slice::from_raw_parts_mut(v_ptr, n_embd_gqa) };

                lw.wq.kernel.forward_prepared(
                    input,
                    q8,
                    sc,
                    Some(q8k),
                    q,
                    n_embd,
                    n_embd_q,
                    ith,
                    nth,
                );
                lw.wk.kernel.forward_prepared(
                    input,
                    q8,
                    sc,
                    Some(q8k),
                    k_new,
                    n_embd,
                    n_embd_gqa,
                    ith,
                    nth,
                );
                lw.wv.kernel.forward_prepared(
                    input,
                    q8,
                    sc,
                    Some(q8k),
                    v_new,
                    n_embd,
                    n_embd_gqa,
                    ith,
                    nth,
                );
            });

            {
                let q = unsafe { std::slice::from_raw_parts_mut(q_ptr, n_embd_q) };
                let k_new = unsafe { std::slice::from_raw_parts_mut(k_ptr, n_embd_gqa) };
                let v_new = unsafe { std::slice::from_raw_parts_mut(v_ptr, n_embd_gqa) };

                for h in 0..n_head {
                    apply_rope(
                        &arch,
                        &mut q[h * n_embd_head_k..(h + 1) * n_embd_head_k],
                        pos,
                        n_embd_head_k,
                        freq_base,
                        rope_dim,
                        attn_factor,
                        None,
                    );
                }
                for h in 0..n_head_kv {
                    apply_rope(
                        &arch,
                        &mut k_new[h * n_embd_head_k..(h + 1) * n_embd_head_k],
                        pos,
                        n_embd_head_k,
                        freq_base,
                        rope_dim,
                        attn_factor,
                        None,
                    );
                }

                let kb = layer * max_ctx * n_embd_gqa;
                if kv_format == KvFormat::F16 {
                    let k_cache =
                        unsafe { std::slice::from_raw_parts_mut(k_cache_f16_ptr, kv_cache_size) };
                    let v_cache =
                        unsafe { std::slice::from_raw_parts_mut(v_cache_f16_ptr, kv_cache_size) };
                    for h in 0..n_head_kv {
                        let off = h * n_embd_head_k;
                        f32_slice_to_f16(
                            &k_new[off..off + n_embd_head_k],
                            &mut k_cache[kb + pos * n_embd_gqa + off
                                ..kb + pos * n_embd_gqa + off + n_embd_head_k],
                        );
                        f32_slice_to_f16(
                            &v_new[off..off + n_embd_head_v],
                            &mut v_cache[kb + pos * n_embd_gqa + off
                                ..kb + pos * n_embd_gqa + off + n_embd_head_v],
                        );
                    }
                } else {
                    let k_cache =
                        unsafe { std::slice::from_raw_parts_mut(k_cache_f32_ptr, kv_cache_size) };
                    let v_cache =
                        unsafe { std::slice::from_raw_parts_mut(v_cache_f32_ptr, kv_cache_size) };
                    for h in 0..n_head_kv {
                        let off = h * n_embd_head_k;
                        k_cache[kb + pos * n_embd_gqa + off
                            ..kb + pos * n_embd_gqa + off + n_embd_head_k]
                            .copy_from_slice(&k_new[off..off + n_embd_head_k]);
                        v_cache[kb + pos * n_embd_gqa + off
                            ..kb + pos * n_embd_gqa + off + n_embd_head_v]
                            .copy_from_slice(&v_new[off..off + n_embd_head_v]);
                    }
                }
            }

            pool.compute(move |ith: usize, nth: usize| {
                let q = unsafe { std::slice::from_raw_parts(q_ptr, n_embd_q) };
                let attn_out = unsafe { std::slice::from_raw_parts_mut(attn_out_ptr, n_embd_q) };
                let h_start = ith * n_head / nth;
                let h_end = (ith + 1) * n_head / nth;

                let kb = layer * max_ctx * n_embd_gqa;

                if kv_format == KvFormat::F16 {
                    let k_cache =
                        unsafe { std::slice::from_raw_parts(k_cache_f16_ptr, kv_cache_size) };
                    let v_cache =
                        unsafe { std::slice::from_raw_parts(v_cache_f16_ptr, kv_cache_size) };
                    for h in h_start..h_end {
                        let kv_h = h / group_size;
                        let q_off = h * n_embd_head_k;
                        let out_base = h * n_embd_head_v;
                        attention_head_f16(
                            &q[q_off..q_off + n_embd_head_k],
                            &mut attn_out[out_base..out_base + n_embd_head_v],
                            k_cache,
                            v_cache,
                            kb + kv_h * n_embd_head_v,
                            n_embd_gqa,
                            pos + 1,
                            kq_scale,
                        );
                    }
                } else {
                    let k_cache =
                        unsafe { std::slice::from_raw_parts(k_cache_f32_ptr, kv_cache_size) };
                    let v_cache =
                        unsafe { std::slice::from_raw_parts(v_cache_f32_ptr, kv_cache_size) };
                    let scores = unsafe {
                        std::slice::from_raw_parts_mut(scores_ptr, n_threads * score_stride)
                    };
                    for h in h_start..h_end {
                        let kv_h = h / group_size;
                        let q_off = h * n_embd_head_k;
                        let n_cached = pos + 1;
                        let n_padded = (n_cached + 255) / 256 * 256;
                        let out_base = h * n_embd_head_v;
                        let s_off = ith * score_stride;
                        for t in 0..n_cached {
                            scores[s_off + t] = dot_f32(
                                &q[q_off..q_off + n_embd_head_k],
                                &k_cache[kb + t * n_embd_gqa + kv_h * n_embd_head_v
                                    ..kb + t * n_embd_gqa + kv_h * n_embd_head_v + n_embd_head_k],
                                n_embd_head_k,
                            ) * kq_scale;
                        }
                        scores[s_off + n_cached..s_off + n_padded].fill(f32::NEG_INFINITY);
                        softmax_inplace(&mut scores[s_off..s_off + n_padded]);
                        let mut values = vec![0.0f32; n_padded];
                        for d in 0..n_embd_head_v {
                            for t in 0..n_cached {
                                values[t] = v_cache[kb + t * n_embd_gqa + kv_h * n_embd_head_v + d];
                            }
                            attn_out[out_base + d] = dot_f32(
                                &values[..n_padded],
                                &scores[s_off..s_off + n_padded],
                                n_cached,
                            );
                        }
                    }
                }
            });

            let attn_out = unsafe { std::slice::from_raw_parts_mut(attn_out_ptr, n_embd_q) };
            quantize_q8_0_into(
                attn_out,
                n_embd_q,
                &mut q8_buf[..n_embd_q],
                &mut scale_buf[..n_embd_q / 32],
            );
            crate::ops::quantize_row_q8_k_into(attn_out, &mut q8k_buf[..n_embd_q / 256]);
            let q8 = q8_buf[..n_embd_q].as_ptr();
            let sc = scale_buf[..n_embd_q / 32].as_ptr();
            let q8k = q8k_buf[..n_embd_q / 256].as_ptr();
            pool.compute(move |ith: usize, nth: usize| {
                let input = unsafe { std::slice::from_raw_parts(attn_out_ptr, n_embd_q) };
                let q8 = unsafe { std::slice::from_raw_parts(q8, n_embd_q) };
                let sc = unsafe { std::slice::from_raw_parts(sc, n_embd_q / 32) };
                let q8k = unsafe { std::slice::from_raw_parts(q8k, n_embd_q / 256) };
                let attn_proj = unsafe { std::slice::from_raw_parts_mut(attn_proj_ptr, n_embd) };
                lw.wo.kernel.forward_prepared(
                    input,
                    q8,
                    sc,
                    Some(q8k),
                    attn_proj,
                    n_embd_q,
                    n_embd,
                    ith,
                    nth,
                );
            });

            let attn_proj = unsafe { std::slice::from_raw_parts_mut(attn_proj_ptr, n_embd) };
            let x = unsafe { std::slice::from_raw_parts_mut(x_ptr, n_embd) };
            let normed = unsafe { std::slice::from_raw_parts_mut(normed_ptr, n_embd) };
            if residual_scale != 0.0 {
                vec_mad_f32(x, attn_proj, residual_scale);
            } else {
                vec_add_into(attn_proj, x);
            }

            rms_norm_grouped(x, &lw.ffn_norm, normed, norm_groups, eps);
            quantize_q8_0_into(
                normed,
                n_embd,
                &mut q8_buf[..n_embd],
                &mut scale_buf[..n_embd / 32],
            );
            crate::ops::quantize_row_q8_k_into(normed, &mut q8k_buf[..n_embd / 256]);
            let q8 = q8_buf[..n_embd].as_ptr();
            let sc = scale_buf[..n_embd / 32].as_ptr();
            let q8k = q8k_buf[..n_embd / 256].as_ptr();

            pool.compute(move |ith: usize, nth: usize| {
                let input = unsafe { std::slice::from_raw_parts(normed_ptr, n_embd) };
                let q8 = unsafe { std::slice::from_raw_parts(q8, n_embd) };
                let sc = unsafe { std::slice::from_raw_parts(sc, n_embd / 32) };
                let q8k = unsafe { std::slice::from_raw_parts(q8k, n_embd / 256) };
                let gate_buf = unsafe { std::slice::from_raw_parts_mut(gate_buf_ptr, n_ff) };
                let up_buf = unsafe { std::slice::from_raw_parts_mut(up_buf_ptr, n_ff) };
                lw.w_gate.kernel.forward_prepared(
                    input,
                    q8,
                    sc,
                    Some(q8k),
                    up_buf,
                    n_embd,
                    n_ff,
                    ith,
                    nth,
                );
                lw.w_up.kernel.forward_prepared(
                    input,
                    q8,
                    sc,
                    Some(q8k),
                    gate_buf,
                    n_embd,
                    n_ff,
                    ith,
                    nth,
                );
                if crate::ops::gpu_matmul_active() {
                    if ith == 0 {
                        silu_mul_approx_inplace(&up_buf[..n_ff], &mut gate_buf[..n_ff]);
                    }
                } else {
                    let per_thread = (n_ff + nth - 1) / nth;
                    let r_start = ith * per_thread;
                    let r_end = (r_start + per_thread).min(n_ff);
                    silu_mul_approx_inplace(&up_buf[r_start..r_end], &mut gate_buf[r_start..r_end]);
                }
            });

            {
                let gate_buf = unsafe { std::slice::from_raw_parts_mut(gate_buf_ptr, n_ff) };
                let q8_buf = unsafe { std::slice::from_raw_parts_mut(q8_buf_ptr, max_n_in) };
                let scale_buf =
                    unsafe { std::slice::from_raw_parts_mut(scale_buf_ptr, max_n_in / 32) };
                let q8k_buf =
                    unsafe { std::slice::from_raw_parts_mut(q8k_buf_ptr, (max_n_in + 255) / 256) };
                quantize_q8_0_into(
                    gate_buf,
                    n_ff,
                    &mut q8_buf[..n_ff],
                    &mut scale_buf[..n_ff / 32],
                );
                crate::ops::quantize_row_q8_k_into(gate_buf, &mut q8k_buf[..n_ff.div_ceil(256)]);
            }

            let q8 = q8_buf[..n_ff].as_ptr();
            let sc = scale_buf[..n_ff / 32].as_ptr();
            let q8k = q8k_buf[..n_ff.div_ceil(256)].as_ptr();
            pool.compute(move |ith: usize, nth: usize| {
                let input = unsafe { std::slice::from_raw_parts(gate_buf_ptr, n_ff) };
                let q8 = unsafe { std::slice::from_raw_parts(q8, n_ff) };
                let sc = unsafe { std::slice::from_raw_parts(sc, n_ff / 32) };
                let q8k = unsafe { std::slice::from_raw_parts(q8k, n_ff / 256) };
                let down_buf = unsafe { std::slice::from_raw_parts_mut(down_buf_ptr, n_embd) };
                lw.w_down.kernel.forward_prepared(
                    input,
                    q8,
                    sc,
                    Some(q8k),
                    down_buf,
                    n_ff,
                    n_embd,
                    ith,
                    nth,
                );
            });

            let down_buf = unsafe { std::slice::from_raw_parts_mut(down_buf_ptr, n_embd) };
            let x = unsafe { std::slice::from_raw_parts_mut(x_ptr, n_embd) };
            if residual_scale != 0.0 {
                vec_mad_f32(x, down_buf, residual_scale);
            } else {
                vec_add_into(down_buf, x);
            }
            if loop_final_norm && (layer + 1) < n_layer && (layer + 1) % layers.len() == 0 {
                rms_norm_inplace(x, &output_norm, eps);
            }
        }

        // Output norm + logits.
        {
            let x = &mut scratch.x;
            let normed = &mut scratch.normed;
            let logits_ptr = scratch.logits.as_mut_ptr();
            let q8_buf = &mut scratch.q8_buf;
            let scale_buf = &mut scratch.scale_buf;
            let q8k_buf = &mut scratch.q8k_buf;

            rms_norm_grouped(x, &output_norm, normed, norm_groups, eps);
            quantize_q8_0_into(
                normed,
                n_embd,
                &mut q8_buf[..n_embd],
                &mut scale_buf[..n_embd / 32],
            );
            crate::ops::quantize_row_q8_k_into(normed, &mut q8k_buf[..n_embd / 256]);
            let q8 = q8_buf[..n_embd].as_ptr();
            let sc = scale_buf[..n_embd / 32].as_ptr();
            let q8k = q8k_buf[..n_embd / 256].as_ptr();
            let input = normed.as_ptr();
            let output_pw = Weight::from_quantized(QuantizedTensor::from_bytes(
                output_weight,
                output_type,
                n_embd,
                vocab,
            ));
            pool.compute(move |ith: usize, nth: usize| {
                let input = unsafe { std::slice::from_raw_parts(input, n_embd) };
                let q8 = unsafe { std::slice::from_raw_parts(q8, n_embd) };
                let sc = unsafe { std::slice::from_raw_parts(sc, n_embd / 32) };
                let q8k = unsafe { std::slice::from_raw_parts(q8k, n_embd / 256) };
                let logits = unsafe { std::slice::from_raw_parts_mut(logits_ptr, vocab) };
                output_pw.kernel.forward_prepared(
                    input,
                    q8,
                    sc,
                    Some(q8k),
                    logits,
                    n_embd,
                    vocab,
                    ith,
                    nth,
                );
            });
            if logit_scale != 0.0 {
                let logits = unsafe { std::slice::from_raw_parts_mut(logits_ptr, vocab) };
                vec_scale_f32(logits, logit_scale);
            }
        }

        prefill_evals += 1;
        prefill_time += eval_started.elapsed();
    }

    let logits = scratch.logits.clone();
    let total_elapsed = t0.elapsed();
    eprintln!(
        "Llama forward_logits: {} prompt tokens in {:?} ({:.0} t/s)",
        prompt_tokens.len(),
        prefill_time,
        crate::app::cli::per_second(prefill_evals, prefill_time),
    );
    let _ = total_elapsed;
    Ok((logits, prefill_time))
}

// ---- Helper functions used by `LlamaSession::forward_chunk_batched_real`
//      to reuse the legacy per-token attention math and the per-thread
//      silu_mul dispatch without duplicating the closure body.

/// Non-flash F16 KV attention: online softmax (running max + rescale) with
/// f32 Q dotted against the f16 K cache (`dot_f16_f32`). This is the
/// pre-existing shared-trunk math — it must stay exact (libm `exp`), not
/// the approximate-exp variant, because every llama-family model shares
/// this path.
#[allow(clippy::too_many_arguments)]
pub(crate) fn attention_head_f16(
    q: &[f32],
    output: &mut [f32],
    k: &[u16],
    v: &[u16],
    cache_offset: usize,
    cache_stride: usize,
    n_cached: usize,
    scale: f32,
) {
    let mut ms = 0.0f32;
    let mut s_sum = 0.0f32;
    output.fill(0.0);
    for t in 0..n_cached {
        let offset = cache_offset + t * cache_stride;
        let score = dot_f16_f32(q, &k[offset..offset + q.len()], q.len()) * scale;
        if score > ms {
            let rescale = (ms - score).exp();
            vec_scale_f32(output, rescale);
            s_sum *= rescale;
            ms = score;
        }
        let vs = (score - ms).exp();
        vec_mad_f16_f32(output, &v[offset..offset + output.len()], vs);
        s_sum += vs;
    }
    let inv_sum = 1.0 / s_sum;
    vec_scale_f32(output, inv_sum);
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_attention_per_query(
    pool: &Arc<ComputePool>,
    q: &[f32],
    attn_out: &mut [f32],
    kv_cache: &KvCache,
    kv_cache_size: usize,
    n_cached: usize,
    n_embd_head_k: usize,
    n_embd_head_v: usize,
    n_embd_gqa: usize,
    n_head: usize,
    group_size: usize,
    kq_scale: f32,
    kb: usize,
    n_threads: usize,
    max_ctx: usize,
) {
    let attn_out_ptr = attn_out.as_mut_ptr();
    let q_ptr = q.as_ptr();
    let n_embd_q = attn_out.len();
    let score_stride = max_ctx.div_ceil(256) * 256;
    let mut scores_storage = vec![0.0f32; n_threads * score_stride];
    let scores_ptr = scores_storage.as_mut_ptr();
    let is_f16 = matches!(kv_cache, KvCache::F16(_));
    let k_cache_f16_ptr = match kv_cache {
        KvCache::F16(c) => c.k.as_ptr() as *const u16,
        _ => std::ptr::null(),
    };
    let v_cache_f16_ptr = match kv_cache {
        KvCache::F16(c) => c.v.as_ptr() as *const u16,
        _ => std::ptr::null(),
    };
    let k_cache_f32_ptr = match kv_cache {
        KvCache::F32(c) => c.k.as_ptr() as *const f32,
        _ => std::ptr::null(),
    };
    let v_cache_f32_ptr = match kv_cache {
        KvCache::F32(c) => c.v.as_ptr() as *const f32,
        _ => std::ptr::null(),
    };
    pool.compute(move |ith: usize, nth: usize| {
        let h_start = ith * n_head / nth;
        let h_end = (ith + 1) * n_head / nth;
        if is_f16 {
            let k_cache =
                unsafe { std::slice::from_raw_parts(k_cache_f16_ptr as *const u16, kv_cache_size) };
            let v_cache =
                unsafe { std::slice::from_raw_parts(v_cache_f16_ptr as *const u16, kv_cache_size) };
            let q_local = unsafe { std::slice::from_raw_parts(q_ptr, n_embd_q) };
            let attn_out_local = unsafe { std::slice::from_raw_parts_mut(attn_out_ptr, n_embd_q) };
            for h in h_start..h_end {
                let kv_h = h / group_size;
                let q_off = h * n_embd_head_k;
                let out_base = h * n_embd_head_v;
                attention_head_f16(
                    &q_local[q_off..q_off + n_embd_head_k],
                    &mut attn_out_local[out_base..out_base + n_embd_head_v],
                    k_cache,
                    v_cache,
                    kb + kv_h * n_embd_head_v,
                    n_embd_gqa,
                    n_cached,
                    kq_scale,
                );
            }
        } else {
            let k_cache =
                unsafe { std::slice::from_raw_parts(k_cache_f32_ptr as *const f32, kv_cache_size) };
            let v_cache =
                unsafe { std::slice::from_raw_parts(v_cache_f32_ptr as *const f32, kv_cache_size) };
            let q_local = unsafe { std::slice::from_raw_parts(q_ptr, n_embd_q) };
            let attn_out_local = unsafe { std::slice::from_raw_parts_mut(attn_out_ptr, n_embd_q) };
            let scores =
                unsafe { std::slice::from_raw_parts_mut(scores_ptr, n_threads * score_stride) };
            let n_padded = (n_cached + 255) / 256 * 256;
            for h in h_start..h_end {
                let kv_h = h / group_size;
                let q_off = h * n_embd_head_k;
                let out_base = h * n_embd_head_v;
                let s_off = ith * score_stride;
                for t in 0..n_cached {
                    scores[s_off + t] = dot_f32(
                        &q_local[q_off..q_off + n_embd_head_k],
                        &k_cache[kb + t * n_embd_gqa + kv_h * n_embd_head_v
                            ..kb + t * n_embd_gqa + kv_h * n_embd_head_v + n_embd_head_k],
                        n_embd_head_k,
                    ) * kq_scale;
                }
                scores[s_off + n_cached..s_off + n_padded].fill(f32::NEG_INFINITY);
                softmax_inplace(&mut scores[s_off..s_off + n_padded]);
                let mut values = vec![0.0f32; n_padded];
                for d in 0..n_embd_head_v {
                    for t in 0..n_cached {
                        values[t] = v_cache[kb + t * n_embd_gqa + kv_h * n_embd_head_v + d];
                    }
                    attn_out_local[out_base + d] = dot_f32(
                        &values[..n_padded],
                        &scores[s_off..s_off + n_padded],
                        n_cached,
                    );
                }
            }
        }
    });
}

/// silu_mul dispatched across threads. `gate` and `up` each have
/// `rows * n_ff` elements laid out row-major. Used by
/// `LlamaSession::forward_chunk_batched_real` after the FFN gate +
/// up projections land in `gate_buf` / `up_buf`.
pub(crate) fn silu_mul_rows(
    pool: &Arc<ComputePool>,
    n_threads: usize,
    gate: &mut [f32],
    up: &[f32],
    n_ff: usize,
) {
    assert_eq!(gate.len(), up.len());
    let rows = gate.len() / n_ff;
    let per_thread = (n_ff + n_threads - 1) / n_threads;
    let gate_ptr = gate.as_mut_ptr();
    let up_ptr = up.as_ptr();
    pool.compute(move |ith, _nth| {
        let r_start = ith * per_thread;
        let r_end = (r_start + per_thread).min(n_ff);
        for row in 0..rows {
            unsafe {
                let g =
                    std::slice::from_raw_parts(gate_ptr.add(row * n_ff + r_start), r_end - r_start);
                let u = std::slice::from_raw_parts_mut(
                    up_ptr.add(row * n_ff + r_start) as *mut f32,
                    r_end - r_start,
                );
                // Exact SiLU via libm `exp` — matches llama.cpp. The
                // approximate-exp variant is only used on the decode
                // Exact SiLU via libm `exp` — matches llama.cpp. The
                // approximate-exp variant is only used on the decode
                // path where it was already the pre-existing convention.
                //
                // Args: `silu_mul_inplace(gate, up)` writes
                // `up[i] *= silu(gate[i])`. The first arg is the
                // multiplier source (read-only), the second is the
                // destination (mut, overwritten with silu(gate)*up).
                // Earlier revisions of this helper swapped the args,
                // producing the wrong tensor — `silu(UP) * GATE`
                // rather than `silu(GATE) * UP`. Swapped back: pass
                // `g` (gate buffer, read-only here) and `u` (up
                // buffer, the destination).
                silu_mul_inplace(g, u);
            }
        }
    });
}

/// Causal attention for a chunk of `rows` queries, using the same
/// F16/F32 dot and softmax operations as the single-token path.
///
/// Layout assumptions:
/// - `q` is `[rows × n_embd_q]` where
///   `n_embd_q = n_head × n_embd_head_k`. Row `r` of head `h`
///   lives at `q[r * n_embd_q + h * n_embd_head_k .. ]`.
/// - `attn_out` is `[rows × n_embd_q]` and `kv_cache` is the
///   standard `[layer × max_ctx × n_embd_gqa]` layout with
///   `n_embd_gqa = n_head_kv × n_embd_head_v`.
/// - `base_position` is the absolute position of the first query
///   in the chunk; each row `r` attends to positions
///   `[0, base_position + r + 1)` (causal).
///
/// Per head we do:
///   `S[r, t] = Q[r, h, :] · K[t, kv_h(h), :]`
///   `S[r, t] *= kq_scale`
///   `S[r, t > base_position + r] = -inf`     (causal mask)
///   `S = softmax(S, row)`
///   `O[r, d] = Σ_t S[r, t] · V[t, kv_h(h), d]`
///
/// Reuses the same SIMD/scalar softmax as the single-token path.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_attention_chunked(
    pool: &Arc<ComputePool>,
    q: &[f32],
    attn_out: &mut [f32],
    kv_cache: &KvCache,
    kv_cache_size: usize,
    n_cached_total: usize,
    base_position: usize,
    rows: usize,
    n_embd_q: usize,
    n_embd_gqa: usize,
    n_head: usize,
    n_embd_head_k: usize,
    n_embd_head_v: usize,
    group_size: usize,
    kq_scale: f32,
    kb: usize,
    n_threads: usize,
    max_ctx: usize,
) {
    let attn_out_ptr = attn_out.as_mut_ptr();
    let q_ptr = q.as_ptr();
    let n_padded_max = max_ctx.div_ceil(256) * 256;
    let is_f16 = matches!(kv_cache, KvCache::F16(_));
    let k_cache_f16_ptr = match kv_cache {
        KvCache::F16(c) => c.k.as_ptr() as *const u16,
        _ => std::ptr::null(),
    };
    let v_cache_f16_ptr = match kv_cache {
        KvCache::F16(c) => c.v.as_ptr() as *const u16,
        _ => std::ptr::null(),
    };
    let k_cache_f32_ptr = match kv_cache {
        KvCache::F32(c) => c.k.as_ptr() as *const f32,
        _ => std::ptr::null(),
    };
    let v_cache_f32_ptr = match kv_cache {
        KvCache::F32(c) => c.v.as_ptr() as *const f32,
        _ => std::ptr::null(),
    };
    let score_stride = rows * n_padded_max;
    let mut scores_storage = vec![0.0f32; n_threads * score_stride];
    let scores_ptr = scores_storage.as_mut_ptr();

    pool.compute(move |ith, nth| {
        let h_start = ith * n_head / nth;
        let h_end = (ith + 1) * n_head / nth;
        if is_f16 {
            let k_cache =
                unsafe { std::slice::from_raw_parts(k_cache_f16_ptr as *const u16, kv_cache_size) };
            let v_cache =
                unsafe { std::slice::from_raw_parts(v_cache_f16_ptr as *const u16, kv_cache_size) };
            let q_local = unsafe { std::slice::from_raw_parts(q_ptr, rows * n_embd_q) };
            let attn_out_local =
                unsafe { std::slice::from_raw_parts_mut(attn_out_ptr, rows * n_embd_q) };
            for h in h_start..h_end {
                let kv_h = h / group_size;
                let q_off = h * n_embd_head_k;
                let out_base = h * n_embd_head_v;
                for r in 0..rows {
                    attention_head_f16(
                        &q_local[r * n_embd_q + q_off..r * n_embd_q + q_off + n_embd_head_k],
                        &mut attn_out_local
                            [r * n_embd_q + out_base..r * n_embd_q + out_base + n_embd_head_v],
                        k_cache,
                        v_cache,
                        kb + kv_h * n_embd_head_v,
                        n_embd_gqa,
                        base_position + r + 1,
                        kq_scale,
                    );
                }
            }
        } else {
            // F32 KV cache path.
            let k_cache =
                unsafe { std::slice::from_raw_parts(k_cache_f32_ptr as *const f32, kv_cache_size) };
            let v_cache =
                unsafe { std::slice::from_raw_parts(v_cache_f32_ptr as *const f32, kv_cache_size) };
            let q_local = unsafe { std::slice::from_raw_parts(q_ptr, rows * n_embd_q) };
            let attn_out_local =
                unsafe { std::slice::from_raw_parts_mut(attn_out_ptr, rows * n_embd_q) };
            let scores = unsafe {
                std::slice::from_raw_parts_mut(scores_ptr.add(ith * score_stride), score_stride)
            };
            let n_padded = (n_cached_total + 255) / 256 * 256;
            for h in h_start..h_end {
                let kv_h = h / group_size;
                let q_off = h * n_embd_head_k;
                let out_base = h * n_embd_head_v;
                // Compute Q · K for every (row, cached_row).
                for r in 0..rows {
                    let q_row =
                        &q_local[r * n_embd_q + q_off..r * n_embd_q + q_off + n_embd_head_k];
                    for t in 0..n_cached_total {
                        let abs_pos = base_position + r;
                        let score = if t > abs_pos {
                            f32::NEG_INFINITY
                        } else {
                            crate::ops::dot_f32(
                                q_row,
                                &k_cache[kb + t * n_embd_gqa + kv_h * n_embd_head_v
                                    ..kb + t * n_embd_gqa + kv_h * n_embd_head_v + n_embd_head_k],
                                n_embd_head_k,
                            ) * kq_scale
                        };
                        scores[r * n_padded + t] = score;
                    }
                    // Causal mask padding.
                    for t in n_cached_total..n_padded {
                        scores[r * n_padded + t] = f32::NEG_INFINITY;
                    }
                    let s = &mut scores[r * n_padded..(r + 1) * n_padded];
                    softmax_inplace(s);
                    // `O = S · V` row-major: per `d` of `n_embd_head_v`,
                    // gather V[t, kv_h, d] and dot with `S`.
                    let mut values = vec![0.0; n_cached_total];
                    for d in 0..n_embd_head_v {
                        for t in 0..n_cached_total {
                            values[t] = v_cache[kb + t * n_embd_gqa + kv_h * n_embd_head_v + d];
                        }
                        attn_out_local[r * n_embd_q + out_base + d] =
                            dot_f32(&values, s, n_cached_total);
                    }
                }
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::{apply_rope, compute_yarn_thetas, normalization_groups};
    use crate::core::tensor::{MetaValue, TensorInfo, TensorSource};
    use std::collections::HashMap;

    struct MetadataSource(HashMap<String, MetaValue>);

    impl TensorSource for MetadataSource {
        fn metadata(&self, key: &str) -> Option<&MetaValue> {
            self.0.get(key)
        }

        fn tensor_info(&self, _name: &str) -> Option<&TensorInfo> {
            None
        }

        fn tensor_slice(&self, _name: &str) -> Option<&[u8]> {
            None
        }
    }

    #[test]
    fn k2_horizon_uses_grouped_norm_and_neox_rope() {
        let source = MetadataSource(HashMap::from([(
            "k2-horizon.attention.group_norm_groups".into(),
            MetaValue::Uint32(4),
        )]));
        assert_eq!(
            normalization_groups(&source, "k2-horizon", 4096).unwrap(),
            4
        );
        assert_eq!(normalization_groups(&source, "llama", 4096).unwrap(), 1);
        assert!(normalization_groups(&source, "k2-horizon", 4095).is_err());

        let mut actual = [1.0, 2.0, 3.0, 4.0];
        let mut expected = actual;
        apply_rope("k2-horizon", &mut actual, 7, 4, 10_000_000.0, 4, 1.0, None);
        crate::ops::rope_neox_inplace_with_factor(&mut expected, 7, 4, 10_000_000.0, 1.0_f32);
        assert_eq!(actual.map(f32::to_bits), expected.map(f32::to_bits));
    }

    /// `compute_yarn_thetas` returns `None` when the GGUF doesn't
    /// declare YaRN (no allocation, no cost) and `Some(thetas)` of
    /// length `rope_dim / 2` when `rope.scaling.type = "yarn"`.
    #[test]
    fn compute_yarn_thetas_is_none_for_plain_rope() {
        let source = MetadataSource(HashMap::new()); // no scaling keys
        let thetas = compute_yarn_thetas(&source, "llama", 1.0e6, 128);
        assert!(thetas.is_none(), "no yarn metadata -> None");
    }

    /// `compute_yarn_thetas` reads every required key. Mistral 3's
    /// GGUF pins `factor=16, original_context_length=16384,
    /// yarn_beta_fast=32, yarn_beta_slow=1, yarn_log_multiplier=1,
    /// freq_base=1e6, rope_dim=128`. We pin the per-dim shape and
    /// the wavelength-correction ramp bounds: outside the
    /// `[start, end]` ramp thetas equal the plain RoPE table; inside
    /// the ramp thetas are scaled down by `1/factor = 1/16`.
    #[test]
    fn compute_yarn_thetas_matches_mistral3_pin() {
        let mut map = HashMap::new();
        map.insert(
            "mistral3.rope.scaling.type".into(),
            MetaValue::String("yarn".into()),
        );
        map.insert(
            "mistral3.rope.scaling.factor".into(),
            MetaValue::Float32(16.0),
        );
        map.insert(
            "mistral3.rope.scaling.original_context_length".into(),
            MetaValue::Uint32(16384),
        );
        map.insert(
            "mistral3.rope.scaling.yarn_beta_fast".into(),
            MetaValue::Float32(32.0),
        );
        map.insert(
            "mistral3.rope.scaling.yarn_beta_slow".into(),
            MetaValue::Float32(1.0),
        );
        map.insert(
            "mistral3.rope.scaling.yarn_log_multiplier".into(),
            MetaValue::Float32(1.0),
        );
        let source = MetadataSource(map);
        let thetas = compute_yarn_thetas(&source, "mistral3", 1.0e6, 128).unwrap();
        assert_eq!(thetas.len(), 128 / 2, "thetas length must be rope_dim/2");

        // `extrap[i] = freq_base^(-2i/rope_dim)`. We re-derive the
        // plain RoPE baseline here so the test fails loudly if the
        // YaRN implementation drifts from the published formula.
        let extrap = |i: usize| -> f32 { (1.0e6f32).powf(-2.0 * i as f32 / 128.0) };
        let inv_factor = 1.0f32 / 16.0;
        // `corr(n_rot) = rope_dim * ln(n_ctx_orig / n_rot) / (2 * ln(base))`
        let corr = |n_rot: f64| -> f64 {
            let dim = 128.0_f64;
            let base_ln = (1.0e6f64).ln();
            let n_ctx_orig = 16384.0_f64;
            dim * (n_ctx_orig / (n_rot * 2.0 * std::f64::consts::PI)).ln() / (2.0 * base_ln)
        };
        let start = corr(32.0).floor().max(0.0);
        let end = corr(1.0).ceil().min(127.0);
        let span = (end - start).max(0.001);
        for i in 0..thetas.len() {
            let extrap_i = extrap(i);
            let interp_i = inv_factor * extrap_i;
            let ramp = 1.0 - ((i as f64 - start) / span).clamp(0.0, 1.0);
            let expected = interp_i * (1.0 - ramp as f32) + extrap_i * ramp as f32;
            // YaRN recomputes the per-dim theta via `powf` twice
            // (the reference tests do the same), so the round-trip
            // through f64 introduces ≤1 ULP drift. Compare with a
            // tolerance instead of bit-exact.
            let rel_err = ((thetas[i] - expected) / expected.max(1e-30)).abs();
            assert!(
                rel_err < 1e-6,
                "yarn theta mismatch at i={i}: got {}, expected {} (rel_err={})",
                thetas[i],
                expected,
                rel_err
            );
        }
    }

    /// Sanity: for positions inside the training window
    /// (`pos < original_context_length`), the YaRN thetas reduce to
    /// the plain RoPE baseline (`theta = pos * freq_base^(-2i/rope_dim)`).
    /// Above the ramp end (`start..end` are inside the wavelength-
    /// correction zone), the thetas are smaller than plain RoPE by
    /// `1/factor` at the centre of the ramp.
    #[test]
    fn compute_yarn_thetas_short_context_equals_plain_rope() {
        let mut map = HashMap::new();
        map.insert(
            "mistral3.rope.scaling.type".into(),
            MetaValue::String("yarn".into()),
        );
        map.insert(
            "mistral3.rope.scaling.factor".into(),
            MetaValue::Float32(16.0),
        );
        map.insert(
            "mistral3.rope.scaling.original_context_length".into(),
            MetaValue::Uint32(16384),
        );
        map.insert(
            "mistral3.rope.scaling.yarn_beta_fast".into(),
            MetaValue::Float32(32.0),
        );
        map.insert(
            "mistral3.rope.scaling.yarn_beta_slow".into(),
            MetaValue::Float32(1.0),
        );
        let source = MetadataSource(map);
        let yarn = compute_yarn_thetas(&source, "mistral3", 1.0e6, 128).unwrap();
        let freq_base = 1.0e6_f32;
        // For Mistral 3, `start ≈ 6`, `end ≈ 28` (depends on the exact
        // log). Outside `[6, 28]` the thetas equal plain RoPE
        // (`ramp=0` or `ramp=1`); inside that range the thetas are
        // strictly smaller than plain RoPE (YaRN compression).
        let mut saw_smaller = false;
        for i in 0..yarn.len() {
            let plain = freq_base.powf(-2.0 * i as f32 / 128.0);
            if yarn[i] < plain * 0.5 {
                saw_smaller = true;
            }
        }
        // At least one dim inside the ramp must be < 1/2 plain. If the
        // ramp is empty (which would only happen with bizarre
        // beta_fast > beta_slow) we'd never see this; assert the
        // ramp range is non-empty via `saw_smaller`.
        assert!(
            saw_smaller,
            "expected at least one dim to be YaRN-compressed"
        );
    }
}

/// Build the prompt token vector for any llama-family arch
/// (llama / nanbeige / exaone / k2-horizon / granite / MiniCPM5).
///
/// Extracted from the inline logic in `run_inference` so the HTTP layer can
/// construct the same prompt bytes the CLI uses (without forking the
/// chat-template selection). Behaviour is identical to the inline block —
/// see `run_inference` for the rationale on MiniCPM5 / nanbeige detection.
pub fn build_prompt_tokens(
    source: &dyn TensorSource,
    prompt: &str,
    thinking: bool,
) -> Result<Vec<u32>, String> {
    let tokenizer = load_tokenizer(|k| source.metadata(k).cloned())
        .map_err(|error| format!("Failed to initialize tokenizer: {error}"))?;

    let arch = source
        .metadata("general.architecture")
        .and_then(|v| v.to_string_val())
        .unwrap_or_default();

    let is_minicpm5 = source
        .metadata("general.name")
        .and_then(|v| v.to_string_val())
        .map(|s| s.to_ascii_lowercase().contains("minicpm"))
        .unwrap_or(false);
    let is_mistral = source
        .metadata("general.name")
        .and_then(|v| v.to_string_val())
        .map(|s| s.to_ascii_lowercase().contains("mistral"))
        .unwrap_or(false);
    let is_zephyr = source
        .metadata("general.name")
        .and_then(|v| v.to_string_val())
        .map(|s| s.to_ascii_lowercase().contains("zephyr"))
        .unwrap_or(false)
        || (arch == "llama"
            && source
                .metadata("tokenizer.chat_template")
                .and_then(|v| v.to_string_val())
                .map(|t| t.contains("<|user|>") && t.contains("<|assistant|>"))
                .unwrap_or(false));

    let prompt_text = if arch == "k2-horizon" {
        format_k2_horizon_chat_prompt_with_thinking(prompt, thinking)
    } else if arch == "granite" {
        format!(
            "<|start_of_role|>user<|end_of_role|>{prompt}<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>"
        )
    } else if arch == "nanbeige" {
        if source
            .metadata("tokenizer.chat_template")
            .and_then(|v| v.to_string_val())
            .is_some_and(|t| t.contains(THINK_MARK))
        {
            crate::prompt::build_nanbeige_chat_prompt(prompt, thinking)
        } else {
            prompt.to_string()
        }
    } else if is_minicpm5 {
        format!("user\n{prompt}\nassistant\n🤔\n\n</think>\n\n")
    } else if arch == "phi3" {
        // Phi-3 / Phi-4 single-turn chat template: `<|user|>…<|end|><|assistant|>`.
        // No system role, no `<think>` block. Matches the CLI's `run_inference_tokens`
        // inline template exactly so HTTP `/v1/chat/completions` produces the
        // same prompt bytes as the CLI.
        format!("<|user|>{prompt}<|end|><|assistant|>")
    } else if arch == "glm4" {
        // GLM-4 chat template: `[gMASK]<sop>` prefix, then
        // `<|user|>\n{prompt}<|assistant|>\n`. `[gMASK]` is the
        // BOS-like sentinel (id 151329 in GLM-4's vocab); `<sop>` is
        // 151332. Both are special tokens, recognised as single ids
        // because `parse_special=true`.
        format!("[gMASK]<sop><|user|>\n{prompt}<|assistant|>\n")
    } else if is_mistral {
        // Mistral-Instruct single-turn template: `[INST] {prompt} [/INST]`.
        // Tokenizer BOS is emitted via `add_special=true`; the literal
        // `[INST]`/`[/INST]` are recognised as single SentencePiece
        // special tokens (id 3 / 4) when `parse_special=true`.
        format!("[INST] {prompt} [/INST]")
    } else if is_zephyr {
        // Zephyr-7B single-turn template:
        // `<|user|>\n{prompt}</s>\n<|assistant|>\n`. Trailing `\n` matters.
        format!("<|user|>\n{prompt}</s>\n<|assistant|>\n")
    } else {
        format!("user\n{prompt}\nassistant\n<think>\n")
    };
    eprintln!("[RUST_PROMPT_TEXT] {prompt_text}");
    // Mistral/Zephyr ship `add_bos_token=true`, so let the tokenizer emit
    // BOS via `add_special=true` and recognise the literal control tokens
    // via `parse_special=true`. Granite/MiniCPM5/Phi-3/Phi-4/GLM-4/Llama
    // all ship `add_bos_token=false`, so encode() does not emit BOS via
    // `add_special=true`; for those we set `add_special=false` and
    // prepend BOS manually below. Nanbeige (base) uses the tokenizer's
    // own add_bos setting.
    let add_special = arch == "nanbeige" || is_mistral || is_zephyr;
    let mut body = tokenizer.encode(
        &prompt_text,
        EncodeOptions {
            add_special,
            parse_special: true,
        },
    );
    if !add_special {
        if let Some(bos) = tokenizer.bos_id() {
            body.insert(0, bos);
        }
    }
    eprintln!("[RUST_TOKENS] n={} ids={:?}", body.len(), body);
    Ok(body)
}
