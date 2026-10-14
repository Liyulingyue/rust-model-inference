//! `MellumConfig` — hyperparameters for the JetBrains Mellum 2.x family
//! (`general.architecture = "mellum"`).
//!
//! The contract is read straight off the GGUF and cross-checked against the
//! pinned llama.cpp oracle (`references/llama.cpp/src/models/mellum.cpp` at
//! `8e2d31e0eb658d0382d3106a61369dff81f0c080`):
//!
//! * 28 layers, `n_embd = 2304`, 32 query heads, 4 KV heads, head dim 128
//!   (`attention.key_length`, *not* `embedding_length / head_count`, which
//!   would be 72).
//! * GQA + per-head Q/K RMSNorm (`attn_q_norm` / `attn_k_norm`).
//! * 64 experts, top-8 routing, `expert_feed_forward_length = 896`, softmax
//!   gating with the selected weights re-normalised to sum to one.
//! * **Parallel** residual: `ffn_inp = attn_out + hidden`, then
//!   `hidden = moe(ffn_inp) + ffn_inp`. Not the sequential Qwen3 residual.
//! * 7 of 28 layers use standard sliding-window attention with
//!   `sliding_window = 1024`, selected by `attention.sliding_window_pattern`.
//! * Those same SWA layers use **unscaled** RoPE; the remaining layers use
//!   YaRN (`freq_scale = 1/16`, `ext_factor = 1`, `attn_factor = 1`).

use crate::core::loader::model_config_from_source;
use crate::core::tensor::{MetaValue, TensorSource};

pub const ARCH: &str = "mellum";

/// Per-layer RoPE flavour. Mellum rotates the sliding-window layers with a
/// plain table and everything else with the YaRN-corrected one, so the choice
/// is a property of the layer index rather than of the model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MellumRope {
    /// `freq_scale = 1`, `ext_factor = 0`, `mscale = 1` — identical to
    /// `crate::ops::rope::rope_neox_inplace`.
    Plain,
    /// YaRN interpolation with `freq_scale = 1 / factor`,
    /// `ext_factor = 1` and `mscale = attn_factor`.
    Yarn(YarnRope),
}

/// The scalar inputs of ggml's `rope_yarn` branch, resolved once per model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct YarnRope {
    pub n_dims: usize,
    pub n_ctx_orig: f32,
    pub freq_base: f32,
    pub freq_scale: f32,
    pub ext_factor: f32,
    pub mscale: f32,
    pub corr_dims: [f32; 2],
}

impl YarnRope {
    /// `freq_base^(-2/n_dims)` — ggml's `theta_scale`.
    pub fn theta_scale(&self) -> f32 {
        self.freq_base.powf(-2.0 / self.n_dims as f32)
    }

    /// Magnitude multiplier applied to both `cos` and `sin`.
    ///
    /// ggml's `rope_yarn` does `mscale *= 1.0f + 0.1f * logf(1.0f /
    /// freq_scale)` when `ext_factor != 0`, on top of the `attn_factor` that
    /// `llama_context` resolved. Both factors survive into the checkpoint: at
    /// `pos = 0` every YaRN lane has `theta = 0`, so the whole rotation is the
    /// identity times `mscale`.
    pub fn mscale(&self) -> f32 {
        if self.ext_factor == 0.0 {
            return self.mscale;
        }
        self.mscale * (1.0 + 0.1 * (1.0f32 / self.freq_scale).ln())
    }

    /// Rotation angle for pair `dim`, given the **un-scaled**
    /// `theta_extrap = position * theta_scale^dim` (the caller accumulates it,
    /// exactly like `ggml_rope_cache_init`).
    pub fn theta_from_extrap(&self, theta_extrap: f32, dim: usize) -> f32 {
        let theta_interp = self.freq_scale * theta_extrap;
        if self.ext_factor == 0.0 {
            return theta_interp;
        }
        let ramp = yarn_ramp(self.corr_dims[0], self.corr_dims[1], (2 * dim) as f32);
        let ramp_mix = ramp * self.ext_factor;
        theta_interp * (1.0 - ramp_mix) + theta_extrap * ramp_mix
    }

    /// cos/sin of `theta`, both multiplied by the YaRN magnitude scale
    /// ([`Self::mscale`]), exactly like ggml's
    /// `cos_theta = cosf(theta) * mscale`.
    pub fn sin_cos(&self, theta: f32) -> (f32, f32) {
        let (cos, sin) = crate::ops::rope::neox::rope_sin_cos(theta);
        let mscale = self.mscale();
        (cos * mscale, sin * mscale)
    }
}

/// `rope_yarn_ramp` — 1 at `i0/2 == low`, 0 at or past `high`.
fn yarn_ramp(low: f32, high: f32, i0: f32) -> f32 {
    let y = (i0 / 2.0 - low) / (high - low).max(0.001);
    1.0 - y.clamp(0.0, 1.0)
}

/// The only Mellum size llama.cpp names (`LLM_TYPE_12B_A2_5B`, selected by
/// `n_layer == 28`). Anything else is rejected rather than silently
/// mis-evaluated.
const KNOWN_12B_A2_5B: MellumDimensions = MellumDimensions {
    n_embd: 2304,
    n_layer: 28,
    n_head: 32,
    n_head_kv: 4,
    n_embd_head: 128,
    expert_count: 64,
    expert_used_count: 8,
    expert_ffn: 896,
    n_swa: 1024,
    n_ctx: 131_072,
    norm_eps_bits: 1e-6f32.to_bits(),
    freq_base_bits: 500_000.0f32.to_bits(),
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MellumDimensions {
    n_embd: usize,
    n_layer: usize,
    n_head: usize,
    n_head_kv: usize,
    n_embd_head: usize,
    expert_count: usize,
    expert_used_count: usize,
    expert_ffn: usize,
    n_swa: usize,
    n_ctx: usize,
    norm_eps_bits: u32,
    freq_base_bits: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MellumConfig {
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub n_embd_head_k: usize,
    pub n_embd_head_v: usize,
    /// `n_rot` — rotated lanes per head. Mellum rotates the full head.
    pub n_rot: usize,
    pub expert_count: usize,
    pub expert_used_count: usize,
    pub expert_ffn: usize,
    pub n_ctx: usize,
    /// Vocabulary width; taken from `tokenizer.ggml.tokens` because Mellum's
    /// GGUF carries no `mellum.vocab_size` key.
    pub vocab_size: usize,
    pub eps: f32,
    pub freq_base: f32,
    /// Per-layer sliding-window flag, length `n_layer`.
    pub swa: Vec<bool>,
    /// Sliding-window length in tokens; only meaningful where `swa[l]`.
    pub n_swa: usize,
    /// Per-layer RoPE flavour, length `n_layer`.
    pub rope: Vec<MellumRope>,
}

impl MellumConfig {
    /// Parse and validate the GGUF contract for `general.architecture = "mellum"`.
    pub fn from_source(source: &dyn TensorSource) -> Result<Self, String> {
        let arch = source
            .metadata("general.architecture")
            .and_then(MetaValue::to_string_val)
            .ok_or_else(|| "Missing metadata: general.architecture".to_string())?
            .to_string();
        if arch != ARCH {
            return Err(format!("Unsupported Mellum architecture: {arch}"));
        }

        let base = model_config_from_source(source)?;
        let n_layer = base.n_layer;
        if n_layer == 0 {
            return Err(format!("Invalid {ARCH}.block_count: 0"));
        }

        let get_u64 = |key: &str| -> Result<usize, String> {
            source
                .metadata(key)
                .and_then(MetaValue::to_u64)
                .and_then(|value| usize::try_from(value).ok())
                .ok_or_else(|| format!("Missing or invalid metadata: {key}"))
        };
        let n_embd_head_k = get_u64(&format!("{ARCH}.attention.key_length"))?;
        let n_embd_head_v = get_u64(&format!("{ARCH}.attention.value_length"))?;
        let n_rot = match source.metadata(&format!("{ARCH}.rope.dimension_count")) {
            Some(value) => value
                .to_u64()
                .and_then(|v| usize::try_from(v).ok())
                .unwrap_or(n_embd_head_k),
            None => n_embd_head_k,
        };
        // `src/models/mellum.cpp` asserts head_v == head_k == n_rot.
        if n_embd_head_v != n_embd_head_k || n_rot != n_embd_head_k {
            return Err(format!(
                "Invalid {ARCH} head geometry: key_length={n_embd_head_k}, value_length={n_embd_head_v}, rope.dimension_count={n_rot}"
            ));
        }

        let expert_count = get_u64(&format!("{ARCH}.expert_count"))?;
        let expert_used_count = get_u64(&format!("{ARCH}.expert_used_count"))?;
        if expert_count == 0 {
            return Err(format!("Invalid {ARCH}.expert_count: 0"));
        }
        if expert_used_count == 0 || expert_used_count > expert_count {
            return Err(format!(
                "Invalid {ARCH}.expert_used_count {expert_used_count} for {expert_count} experts"
            ));
        }
        // `src/models/mellum.cpp`: `n_ff_exp = hparams.n_ff_exp(il) ?: n_ff /
        // n_expert_used`, where `n_ff_exp(il)` comes from
        // `expert_feed_forward_length` (scalar or per-layer array).
        let expert_ffn = match read_usize_array(
            source,
            &format!("{ARCH}.expert_feed_forward_length"),
            n_layer,
        )? {
            Some(values) => {
                let value = values[0];
                if values.iter().any(|entry| *entry != value) {
                    return Err(format!(
                        "Unsupported per-layer {ARCH}.expert_feed_forward_length"
                    ));
                }
                value
            }
            None => base.n_ff / expert_used_count,
        };
        if expert_ffn == 0 {
            return Err(format!("Invalid {ARCH} expert feed-forward length: 0"));
        }

        let n_swa = get_u64(&format!("{ARCH}.attention.sliding_window"))?;
        let swa = read_swa_pattern(source, n_layer, n_swa)?;

        let rope = build_rope(source, n_layer, n_rot, base.rope_freq_base)?;

        let config = MellumConfig {
            n_embd: base.n_embd,
            n_layer,
            n_head: base.n_head,
            n_head_kv: base.n_head_kv,
            n_embd_head_k,
            n_embd_head_v,
            n_rot,
            expert_count,
            expert_used_count,
            expert_ffn,
            n_ctx: base.n_ctx,
            vocab_size: base.vocab_size,
            eps: base.norm_eps,
            freq_base: base.rope_freq_base,
            swa,
            n_swa,
            rope,
        };
        config.check_known_dimensions()?;
        Ok(config)
    }

    /// Reject configurations outside the single enumerated Mellum 2.1 size.
    fn check_known_dimensions(&self) -> Result<(), String> {
        let known = KNOWN_12B_A2_5B;
        let actual = MellumDimensions {
            n_embd: self.n_embd,
            n_layer: self.n_layer,
            n_head: self.n_head,
            n_head_kv: self.n_head_kv,
            n_embd_head: self.n_embd_head_k,
            expert_count: self.expert_count,
            expert_used_count: self.expert_used_count,
            expert_ffn: self.expert_ffn,
            n_swa: self.n_swa,
            n_ctx: self.n_ctx,
            norm_eps_bits: self.eps.to_bits(),
            freq_base_bits: self.freq_base.to_bits(),
        };
        if actual != known {
            Err(format!(
                "Unsupported {ARCH} dimensions: n_embd={}, n_layer={}, n_head={}/{}, head_dim={}, \
                 experts={}/{}, expert_ffn={}, n_swa={}, n_ctx={} (only Mellum 2.1-12B-A2.5B is verified)",
                actual.n_embd,
                actual.n_layer,
                actual.n_head,
                actual.n_head_kv,
                actual.n_embd_head,
                actual.expert_count,
                actual.expert_used_count,
                actual.expert_ffn,
                actual.n_swa,
                actual.n_ctx,
            ))
        } else {
            Ok(())
        }
    }

    /// Query width (`n_head * n_embd_head_k`) fed to `attn_output`.
    pub fn n_attn(&self) -> usize {
        self.n_head * self.n_embd_head_v
    }

    pub fn rope_for(&self, layer: usize) -> MellumRope {
        self.rope[layer]
    }

    pub fn is_swa(&self, layer: usize) -> bool {
        self.swa[layer]
    }
}

/// `attention.sliding_window_pattern` is an array of 0/1 flags, one per
/// layer. llama.cpp's `load_swa_pattern(ml, 4)` reads the array directly and
/// falls back to a 4-layer repeating pattern when it is absent.
fn read_swa_pattern(
    source: &dyn TensorSource,
    n_layer: usize,
    n_swa: usize,
) -> Result<Vec<bool>, String> {
    let key = format!("{ARCH}.attention.sliding_window_pattern");
    match source.metadata(&key) {
        Some(MetaValue::Array(_, items)) => {
            if items.len() != n_layer {
                return Err(format!(
                    "{key} has {} entries but the model declares {n_layer} layers",
                    items.len()
                ));
            }
            let mut swa = Vec::with_capacity(n_layer);
            for (index, item) in items.iter().enumerate() {
                let flag = item
                    .to_u64()
                    .or_else(|| match item {
                        MetaValue::Bool(flag) => Some(u64::from(*flag)),
                        _ => None,
                    })
                    .ok_or_else(|| format!("{key}[{index}] is not an integer or bool"))?;
                if flag > 1 {
                    return Err(format!("{key}[{index}] = {flag}; expected 0 or 1"));
                }
                swa.push(flag == 1);
            }
            Ok(swa)
        }
        Some(_) => Err(format!("{key} is not an array")),
        None => {
            if n_swa == 0 {
                return Ok(vec![false; n_layer]);
            }
            Ok((0..n_layer).map(|layer| layer % 4 != 3).collect())
        }
    }
}

fn read_usize_array(
    source: &dyn TensorSource,
    key: &str,
    n_layer: usize,
) -> Result<Option<Vec<usize>>, String> {
    match source.metadata(key) {
        Some(MetaValue::Array(_, items)) => {
            if items.len() != n_layer {
                return Err(format!(
                    "{key} has {} entries, expected {n_layer}",
                    items.len()
                ));
            }
            let mut values = Vec::with_capacity(n_layer);
            for (index, item) in items.iter().enumerate() {
                values.push(
                    item.to_u64()
                        .and_then(|value| usize::try_from(value).ok())
                        .ok_or_else(|| format!("{key}[{index}] is not a positive integer"))?,
                );
            }
            Ok(Some(values))
        }
        Some(value) => {
            // Scalar form: llama.cpp's `get_key_or_arr` broadcasts one value to
            // every layer.
            let value = value
                .to_u64()
                .and_then(|value| usize::try_from(value).ok())
                .ok_or_else(|| format!("{key} is not a positive integer"))?;
            Ok(Some(vec![value; n_layer]))
        }
        None => Ok(None),
    }
}

/// Resolve the per-layer RoPE flavour.
///
/// SWA layers rotate with `freq_scale = 1`, `ext_factor = 0`, `attn_factor = 1`
/// — i.e. the vanilla table. Every other layer uses the model's YaRN
/// parameters, matching `src/models/mellum.cpp`'s `ggml_rope_ext` calls.
fn build_rope(
    source: &dyn TensorSource,
    n_layer: usize,
    n_dims: usize,
    freq_base: f32,
) -> Result<Vec<MellumRope>, String> {
    let yarn = YarnRope::from_source(source, n_dims, freq_base)?;
    Ok((0..n_layer)
        .map(|layer| {
            if is_swa_layer(source, n_layer, layer) {
                MellumRope::Plain
            } else {
                MellumRope::Yarn(yarn)
            }
        })
        .collect())
}

fn is_swa_layer(source: &dyn TensorSource, n_layer: usize, layer: usize) -> bool {
    let key = format!("{ARCH}.attention.sliding_window_pattern");
    match source.metadata(&key) {
        Some(MetaValue::Array(_, items)) => items
            .get(layer)
            .and_then(|item| {
                item.to_u64().or_else(|| match item {
                    MetaValue::Bool(flag) => Some(u64::from(*flag)),
                    _ => None,
                })
            })
            .map(|value| value == 1)
            .unwrap_or(layer % 4 != 3),
        _ => {
            let _ = n_layer;
            layer % 4 != 3
        }
    }
}

impl YarnRope {
    /// Reproduce `llama_context`'s YaRN parameter resolution:
    ///
    /// ```text
    /// ext_factor  = GGUF yarn_ext_factor, else 1.0 for rope.scaling.type = yarn
    /// attn_factor = (0.1*log(factor) + 1) / (1 + 0.1*log(factor))      (log_mul == 0)
    ///             * GGUF rope.scaling.attn_factor
    /// beta_fast   = hparams default 32.0, beta_slow = 1.0
    /// ```
    ///
    /// Note the GGUF `rope.scaling.yarn_attn_factor` / `yarn_beta_fast` /
    /// `yarn_beta_slow` keys are **write-only** in llama.cpp (only
    /// `llama_model_saver` emits them; no loader reads them), so the 12B
    /// checkpoint's `1.2772588729858398` never reaches the graph.
    fn from_source(
        source: &dyn TensorSource,
        n_dims: usize,
        freq_base: f32,
    ) -> Result<Self, String> {
        let scaling_type = source
            .metadata(&format!("{ARCH}.rope.scaling.type"))
            .and_then(MetaValue::to_string_val)
            .unwrap_or_default()
            .to_string();
        if scaling_type != "yarn" {
            return Err(format!(
                "Unsupported {ARCH}.rope.scaling.type {scaling_type:?}; expected \"yarn\""
            ));
        }
        let factor = source
            .metadata(&format!("{ARCH}.rope.scaling.factor"))
            .and_then(MetaValue::to_f64)
            .unwrap_or(1.0) as f32;
        if factor <= 1.0 {
            return Err(format!(
                "Invalid {ARCH}.rope.scaling.factor {factor}; YaRN needs factor > 1"
            ));
        }
        let n_ctx_orig = source
            .metadata(&format!("{ARCH}.rope.scaling.original_context_length"))
            .and_then(MetaValue::to_u64)
            .and_then(|value| usize::try_from(value).ok())
            .unwrap_or(8192) as f32;
        let ext_factor = source
            .metadata(&format!("{ARCH}.rope.scaling.yarn_ext_factor"))
            .and_then(MetaValue::to_f64)
            .map(|value| value as f32)
            .unwrap_or(1.0);
        let attn_factor = source
            .metadata(&format!("{ARCH}.rope.scaling.attn_factor"))
            .and_then(MetaValue::to_f64)
            .unwrap_or(1.0) as f32;
        let beta_fast = source
            .metadata(&format!("{ARCH}.rope.scaling.yarn_beta_fast"))
            .and_then(MetaValue::to_f64)
            .unwrap_or(32.0) as f32;
        let beta_slow = source
            .metadata(&format!("{ARCH}.rope.scaling.yarn_beta_slow"))
            .and_then(MetaValue::to_f64)
            .unwrap_or(1.0) as f32;

        // `llama-context.cpp` computes
        //     attn = get_mscale(factor, 1.0);          // 0.1*log(factor) + 1
        //     attn *= 1.0f / (1.0f + 0.1f * logf(factor));
        // and `get_mscale` returns exactly `1.0f + 0.1f*logf(factor)`, so the
        // two factors cancel to 1.0 for every model that does not set
        // `rope_yarn_log_mul` (which is never read from GGUF). Write the
        // cancellation explicitly instead of relying on `x / x`, so the f32
        // rounding is visible.
        let log_mul = 0.1f32 * factor.ln();
        let get_mscale = 1.0f32 + log_mul;
        let cancelled = if get_mscale != 0.0 {
            get_mscale * (1.0f32 / (1.0f32 + log_mul))
        } else {
            1.0
        };
        let mscale = if ext_factor != 0.0 {
            cancelled * attn_factor
        } else {
            attn_factor
        };

        Ok(YarnRope {
            n_dims,
            n_ctx_orig,
            freq_base,
            freq_scale: 1.0 / factor,
            ext_factor,
            mscale,
            corr_dims: yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow),
        })
    }
}

/// `ggml_rope_yarn_corr_dims` — the `[low, high]` rotation-dim ramp window.
fn yarn_corr_dims(
    n_dims: usize,
    n_ctx_orig: f32,
    freq_base: f32,
    beta_fast: f32,
    beta_slow: f32,
) -> [f32; 2] {
    let corr_dim = |n_rot: f32| {
        n_dims as f32 * (n_ctx_orig / (n_rot * 2.0 * std::f32::consts::PI)).ln()
            / (2.0 * freq_base.ln())
    };
    let start = corr_dim(beta_fast).floor();
    let end = corr_dim(beta_slow).ceil();
    [start.max(0.0), end.min(n_dims as f32 - 1.0)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn yarn_corr_dims_match_ggml_12b_geometry() {
        // ggml_rope_yarn_corr_dims(128, 8192, 5e5, 32, 1)
        let dims = yarn_corr_dims(128, 8192.0, 500_000.0, 32.0, 1.0);
        assert_eq!(dims[0].to_bits(), 18.0f32.to_bits());
        assert_eq!(dims[1].to_bits(), 35.0f32.to_bits());
    }

    fn yarn_12b() -> YarnRope {
        YarnRope {
            n_dims: 128,
            n_ctx_orig: 8192.0,
            freq_base: 500_000.0,
            freq_scale: 1.0 / 16.0,
            ext_factor: 1.0,
            mscale: 1.0,
            corr_dims: [18.0, 35.0],
        }
    }

    /// Reference table produced by ggml's `ggml_rope_cache_init` for the 12B
    /// checkpoint's parameters (factor 16, orig ctx 8192, base 5e5,
    /// ext_factor 1, attn_factor 1).
    #[test]
    fn yarn_theta_matches_ggml_reference_table() {
        let yarn = yarn_12b();
        for (position, expected) in [
            (0.0f32, [0.0f32, 0.0, 0.0, 0.0]),
            (3.0, [3.0, 0.473_918_3, 0.033_775_63, 0.000_143_343_09]),
            (100.0, [100.0, 15.797_277, 1.125_854_1, 0.004_778_101_6]),
        ] {
            let mut theta = position;
            for (offset, want) in expected.iter().enumerate() {
                let dim = [0usize, 9, 21, 35][offset];
                let got = yarn.theta_from_extrap(theta, dim);
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "position {position} dim {dim}"
                );
                // advance to the next dim
                theta *= yarn.theta_scale();
                let _ = dim;
                break;
            }
        }
    }

    /// The YaRN magnitude scale shows up directly in the checkpoint: at
    /// `pos = 0` every lane has `theta = 0`, so the rotated value equals the
    /// normed value times `mscale`. The oracle's `cos(dim 0)` for a non-SWA
    /// layer is `0.1 * ln(16) + 1 = 1.2772589`.
    #[test]
    fn yarn_mscale_multiplies_every_lane() {
        let yarn = yarn_12b();
        let expected = 0.1f32 * (16.0f32).ln() + 1.0;
        assert_eq!(yarn.mscale(), expected);
        let (cos, sin) = yarn.sin_cos(0.0);
        assert_eq!(cos.to_bits(), expected.to_bits());
        assert_eq!(sin.to_bits(), 0.0f32.to_bits());
    }

    #[test]
    fn plain_swa_rope_uses_the_unscaled_table() {
        // freq_scale = 1 / ext_factor = 0 collapses to `theta = pos *
        // theta_scale^dim`, which is what `rope_neox_inplace` already builds.
        let yarn = YarnRope {
            freq_scale: 1.0,
            ext_factor: 0.0,
            ..yarn_12b()
        };
        assert_eq!(yarn.mscale(), 1.0);
        assert_eq!(yarn.theta_from_extrap(7.0, 0).to_bits(), 7.0f32.to_bits());
    }
}
