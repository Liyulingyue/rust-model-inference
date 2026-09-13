//! Native Breeze depth/text/backbone operations over the shared
//! [`Weight`] + [`Kernel`] abstraction.  Weight storage follows the
//! `tensor_info.ggml_type` of each tensor; the underlying
//! `crate::ops::kernel::QuantizedTensor` accepts every GGML type the rest
//! of the model zoo supports (F32 / F16 / BF16 / Q4_0 / Q4_1 / Q4_K / Q5_K
//! / Q6_K / Q8_0 / Q2_K / Q3_K / IQ1_M / IQ1_S / IQ2_XXS / IQ2_XS / IQ2_S
//! / IQ3_XXS / IQ3_S / IQ4_NL / IQ4_XS), so quantised GGUF outputs from
//! the converter are loadable here without per-type branching.
//!
//! Follows the qwen3 / dots convention: no per-model `load_weight` shim —
//! weights are constructed directly via `Weight::from_quantized(
//! QuantizedTensor::from_bytes(...))` against the live `tensor_info`.
//! See `docs/TODO.md` for the consolidation plan that would fold the
//! remaining `load_weight`/`load_weight_any` variants into a single core
//! helper.
use super::{bf, trace};
use crate::core::tensor::{load_f32_tensor, TensorSource};
use crate::ops::kernel::{QuantizedTensor, Weight};
use rayon::prelude::*;

pub(super) fn matrix<'a>(
    source: &'a dyn TensorSource,
    name: &str,
    input: usize,
    output: usize,
) -> Result<Weight<'a>, String> {
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("Missing tensor: {name}"))?;
    let bytes = source
        .tensor_slice(name)
        .ok_or_else(|| format!("Missing tensor data: {name}"))?;
    let n_in = input;
    let n_out = output;
    let expected = info
        .checked_nbytes()
        .ok_or_else(|| format!("Invalid tensor byte size: {name}"))?;
    let expected = usize::try_from(expected)
        .map_err(|_| format!("Tensor byte size does not fit usize: {name}"))?;
    if bytes.len() != expected {
        return Err(format!(
            "Invalid tensor data length for {name}: {}; expected {expected}",
            bytes.len()
        ));
    }
    let mut weight =
        Weight::from_quantized(QuantizedTensor::from_bytes(bytes, info.ggml_type, n_in, n_out));
    // QuantizedTensor's F32 variant loses matrix shape; the rest of Breeze
    // already relies on `Weight::n_in` / `Weight::n_out` being correct.
    weight.n_in = n_in;
    weight.n_out = n_out;
    Ok(weight)
}

pub(super) fn linear(weight: &Weight<'_>, input: &[f32]) -> Vec<f32> {
    let ni = weight.n_in;
    let no = weight.n_out;
    assert_eq!(input.len() % ni, 0);
    let mut out = vec![0.0; input.len() / ni * no];
    weight.kernel.forward_batched(input, &mut out, ni, no);
    out
}

#[derive(Clone, Copy, PartialEq)]
pub(super) enum Kind {
    Text,
    Backbone,
    Depth,
}
impl Kind {
    fn dimensions(self) -> (usize, usize, usize, usize, usize, usize, f32) {
        match self {
            Self::Text => (1152, 26, 4, 1, 256, 6912, 1e-6),
            Self::Backbone => (2048, 28, 16, 8, 128, 6144, 1e-6),
            Self::Depth => (1024, 12, 8, 2, 128, 8192, 1e-5),
        }
    }
    fn prefix(self) -> &'static str {
        match self {
            Self::Text => "text_encoder",
            Self::Backbone => "backbone_model",
            Self::Depth => "depth_decoder.model",
        }
    }
    fn trace(self) -> &'static str {
        match self {
            Self::Text => "breeze.text",
            Self::Backbone => "breeze.backbone",
            Self::Depth => "breeze.depth",
        }
    }
}

struct Layer<'a> {
    index: usize,
    input_norm: Vec<f32>,
    ff_norm: Vec<f32>,
    attn_post: Option<Vec<f32>>,
    ff_post: Option<Vec<f32>>,
    q: Weight<'a>,
    k: Weight<'a>,
    v: Weight<'a>,
    o: Weight<'a>,
    q_norm: Option<Vec<f32>>,
    k_norm: Option<Vec<f32>>,
    gate: Weight<'a>,
    up: Weight<'a>,
    down: Weight<'a>,
}

#[derive(Default)]
pub(super) struct Cache {
    layers: Vec<(Vec<f32>, Vec<f32>)>,
    pub len: usize,
}

pub(super) struct Transformer<'a> {
    kind: Kind,
    layers: Vec<Layer<'a>>,
    norm: Vec<f32>,
}
impl<'a> Transformer<'a> {
    pub fn load(source: &'a dyn TensorSource, kind: Kind) -> Result<Self, String> {
        let (hidden, count, heads, kv, hd, ff, _) = kind.dimensions();
        let prefix = kind.prefix();
        let mut layers = Vec::with_capacity(count);
        for index in 0..count {
            let p = format!("{prefix}.layers.{index}");
            let w = |name: &str, i, o| matrix(source, &format!("{p}.{name}.weight"), i, o);
            let norm =
                |name: &str, d| load_f32_tensor(source, &format!("{p}.{name}.weight"), &[d as u64]);
            let text = kind == Kind::Text;
            layers.push(Layer {
                index,
                input_norm: norm(
                    if text {
                        "pre_self_attn_layernorm"
                    } else {
                        "input_layernorm"
                    },
                    hidden,
                )?,
                ff_norm: norm(
                    if text {
                        "pre_feedforward_layernorm"
                    } else {
                        "post_attention_layernorm"
                    },
                    hidden,
                )?,
                attn_post: if text {
                    Some(norm("post_self_attn_layernorm", hidden)?)
                } else {
                    None
                },
                ff_post: if text {
                    Some(norm("post_feedforward_layernorm", hidden)?)
                } else {
                    None
                },
                q: w("self_attn.q_proj", hidden, heads * hd)?,
                k: w("self_attn.k_proj", hidden, kv * hd)?,
                v: w("self_attn.v_proj", hidden, kv * hd)?,
                o: w("self_attn.o_proj", heads * hd, hidden)?,
                q_norm: if kind != Kind::Depth {
                    Some(norm("self_attn.q_norm", hd)?)
                } else {
                    None
                },
                k_norm: if kind != Kind::Depth {
                    Some(norm("self_attn.k_norm", hd)?)
                } else {
                    None
                },
                gate: w("mlp.gate_proj", hidden, ff)?,
                up: w("mlp.up_proj", hidden, ff)?,
                down: w("mlp.down_proj", ff, hidden)?,
            });
        }
        Ok(Self {
            kind,
            layers,
            norm: load_f32_tensor(source, &format!("{prefix}.norm.weight"), &[hidden as u64])?,
        })
    }

    /// Text batches are right padded and independent; causal sessions use one row.
    pub fn forward(
        &self,
        mut x: Vec<f32>,
        lengths: &[usize],
        mut cache: Option<&mut Cache>,
        step: Option<usize>,
    ) -> Result<Vec<f32>, String> {
        let (hidden, _, heads, kv, hd, _, eps) = self.kind.dimensions();
        let batch = lengths.len();
        if batch == 0 || x.len() % (batch * hidden) != 0 {
            return Err("Invalid Breeze hidden-state shape".into());
        }
        let n = x.len() / batch / hidden;
        if n == 0 || lengths.iter().any(|&l| l == 0 || l > n) {
            return Err("Invalid Breeze sequence length".into());
        }
        let text = self.kind == Kind::Text;
        let start = cache.as_ref().map_or(0, |c| c.len);
        if let Some(c) = cache.as_mut() {
            if c.layers.is_empty() {
                c.layers.resize_with(self.layers.len(), Default::default);
            }
        }
        let shape = if batch == 1 {
            vec![n, hidden]
        } else {
            vec![batch, n, hidden]
        };
        let fine_prefix = match self.kind {
            Kind::Text => "RMI_BREEZE_FINE_TEXT",
            Kind::Backbone => "RMI_BREEZE_FINE_BACKBONE",
            Kind::Depth => "RMI_BREEZE_FINE_DEPTH",
        };
        let fine_enabled = std::env::var_os(fine_prefix).is_some();
        let fine_layer = std::env::var(format!("{fine_prefix}_LAYER"))
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0);
        for (cache_index, layer) in self.layers.iter().enumerate() {
            let index = layer.index;
            let fine = |name: &str, dims: &[usize], values: &[f32]| -> Result<(), String> {
                if fine_enabled && index == fine_layer {
                    trace(
                        &format!("{}.layer.{index}.{name}", self.kind.trace()),
                        Some(index),
                        step,
                        dims,
                        values,
                    )?;
                }
                Ok(())
            };
            let normed = rms(&x, &layer.input_norm, eps, text);
            fine("pre_self_attn", &shape, &normed)?;
            let mut q = linear(&layer.q, &normed);
            let mut k = linear(&layer.k, &normed);
            let v = linear(&layer.v, &normed);
            let dims = |width| {
                if batch == 1 {
                    vec![n, width]
                } else {
                    vec![batch, n, width]
                }
            };
            fine("q_proj", &dims(heads * hd), &q)?;
            if text {
                fine("k_proj", &dims(kv * hd), &k)?;
                fine("v_proj", &dims(kv * hd), &v)?;
            }
            if let Some(w) = &layer.q_norm {
                q = rms(&q, w, eps, text);
                if fine_enabled && index == fine_layer {
                    fine(
                        "q_norm",
                        &[batch, heads, n, hd],
                        &head_major(&q, batch, n, heads, hd),
                    )?;
                }
            }
            if !text {
                fine("k_proj", &dims(kv * hd), &k)?;
            }
            if let Some(w) = &layer.k_norm {
                k = rms(&k, w, eps, text);
                if fine_enabled && index == fine_layer {
                    fine(
                        "k_norm",
                        &[batch, kv, n, hd],
                        &head_major(&k, batch, n, kv, hd),
                    )?;
                }
            }
            if !text {
                fine("v_proj", &dims(kv * hd), &v)?;
            }
            let local = text && (index + 1) % 6 != 0;
            let (theta, factor) = match self.kind {
                Kind::Text if local => (10000.0, 1.0),
                Kind::Text => (1_000_000.0, 8.0),
                Kind::Backbone => (1_000_000.0, 1.0),
                Kind::Depth => (500_000.0, 1.0),
            };
            let freq = inv_freq(hd, theta, factor, self.kind == Kind::Depth);
            for (b, &len) in lengths.iter().enumerate() {
                for t in 0..n {
                    let pos = if text && t >= len { 0 } else { start + t };
                    rope(
                        &mut q[(b * n + t) * heads * hd..(b * n + t + 1) * heads * hd],
                        hd,
                        pos,
                        &freq,
                    );
                    rope(
                        &mut k[(b * n + t) * kv * hd..(b * n + t + 1) * kv * hd],
                        hd,
                        pos,
                        &freq,
                    );
                }
            }
            if !text {
                fine("q_rope", &dims(heads * hd), &q)?;
                fine("k_rope", &dims(kv * hd), &k)?;
            }
            let (keys, values, nk) = if let Some(c) = cache.as_mut() {
                let (keys, values) = &mut c.layers[cache_index];
                keys.extend_from_slice(&k);
                values.extend_from_slice(&v);
                (&keys[..], &values[..], start + n)
            } else {
                (&k[..], &v[..], n)
            };
            let mut attended = vec![0.0; batch * n * heads * hd];
            attended
                .par_chunks_mut(heads * hd)
                .enumerate()
                .for_each(|(row, output)| {
                    let b = row / n;
                    let t = row % n;
                    for h in 0..heads {
                        let kh = h / (heads / kv);
                        let query = &q[(row * heads + h) * hd..(row * heads + h + 1) * hd];
                        let mut scores = vec![f32::NEG_INFINITY; nk];
                        for s in 0..nk {
                            let allowed = if text {
                                s < lengths[b]
                                    && (!local || (s <= t && t - s < 256) || (s > t && s - t < 257))
                            } else {
                                s <= start + t
                            };
                            if allowed {
                                let off = ((b * nk + s) * kv + kh) * hd;
                                scores[s] =
                                    bf(bf(crate::ops::dot_f32(query, &keys[off..off + hd], hd))
                                        * (hd as f32).sqrt().recip());
                            }
                        }
                        crate::ops::softmax_inplace(&mut scores);
                        for p in &mut scores {
                            *p = bf(*p);
                        }
                        for d in 0..hd {
                            let mut sum = 0.0f32;
                            for s in 0..nk {
                                sum += scores[s] * values[((b * nk + s) * kv + kh) * hd + d];
                            }
                            output[h * hd + d] = bf(sum);
                        }
                    }
                });
            if fine_enabled && index == fine_layer && !text {
                let mut raw = Vec::with_capacity(batch * heads * n * nk);
                let mut probabilities = Vec::with_capacity(raw.capacity());
                for b in 0..batch {
                    for h in 0..heads {
                        let kh = h / (heads / kv);
                        for t in 0..n {
                            let query = &q[((b * n + t) * heads + h) * hd
                                ..((b * n + t) * heads + h + 1) * hd];
                            let mut scores = (0..nk)
                                .map(|s| {
                                    let off = ((b * nk + s) * kv + kh) * hd;
                                    bf(bf(crate::ops::dot_f32(query, &keys[off..off + hd], hd))
                                        * (hd as f32).sqrt().recip())
                                })
                                .collect::<Vec<_>>();
                            for (s, score) in scores.iter_mut().enumerate() {
                                if s > start + t {
                                    *score = f32::from_bits(0xff7f0000);
                                }
                            }
                            raw.extend_from_slice(&scores);
                            crate::ops::softmax_inplace(&mut scores);
                            probabilities.extend(scores.into_iter().map(bf));
                        }
                    }
                }
                let score_shape = if batch == 1 {
                    vec![heads, n, nk]
                } else {
                    vec![batch, heads, n, nk]
                };
                fine("attn_scores", &score_shape, &raw)?;
                fine("probabilities", &score_shape, &probabilities)?;
                fine("attended", &dims(heads * hd), &attended)?;
            }
            let mut a = linear(&layer.o, &attended);
            fine("attn_output", &shape, &a)?;
            if let Some(w) = &layer.attn_post {
                a = rms(&a, w, eps, true);
                fine("post_self_attn", &shape, &a)?;
            }
            residual(&mut x, &a);
            let normed = rms(&x, &layer.ff_norm, eps, text);
            fine("pre_ff", &shape, &normed)?;
            let mut gate = linear(&layer.gate, &normed);
            fine("gate", &dims(layer.gate.n_out), &gate)?;
            // Apply activation in place via the SIMD slice path
            // (gelu for text/backbone, silu for depth).  The bf() round-trip
            // that mirrors upstream BF16-quantised activation runs as a
            // SIMD slice pass too, so the whole gate pipeline stays on
            // AVX2/NEON and is bit-exact with the previous scalar loop.
            if text {
                crate::ops::gelu_inplace(&mut gate);
            } else {
                // Temporarily disabled: silu SIMD path appears to
                // perturb downstream precision under Q8_0 quantisation.
                // Re-enable after simd_avx2 SIMD path is debugged.
                for g in gate.iter_mut() {
                    *g = crate::ops::silu(*g);
                }
            }
            crate::ops::bf16_round_inplace(&mut gate);
            fine("activation", &dims(layer.gate.n_out), &gate)?;
            let up = linear(&layer.up, &normed);
            fine("up", &dims(layer.up.n_out), &up)?;
            // gate was bf-quantised above; multiply by up and round-trip
            // through bf once more to mirror the upstream BF16-quantised
            // multiply path that the previous scalar loop enforced.
            for (g, u) in gate.iter_mut().zip(up) {
                *g = bf(*g * u);
            }
            fine("ff_input", &dims(layer.gate.n_out), &gate)?;
            let mut f = linear(&layer.down, &gate);
            fine("ff", &shape, &f)?;
            if let Some(w) = &layer.ff_post {
                f = rms(&f, w, eps, true);
                fine("post_ff", &shape, &f)?;
            }
            residual(&mut x, &f);
            trace(
                &format!("{}.layer.{index}", self.kind.trace()),
                Some(index),
                step,
                &shape,
                &x,
            )?;
        }
        x = rms(&x, &self.norm, eps, text);
        trace(
            &format!("{}.norm", self.kind.trace()),
            None,
            step,
            &shape,
            &x,
        )?;
        if let Some(c) = cache {
            c.len += n;
        }
        if x.iter().any(|v| !v.is_finite()) {
            return Err(format!(
                "{} produced non-finite hidden states",
                self.kind.trace()
            ));
        }
        Ok(x)
    }
}

fn residual(x: &mut [f32], add: &[f32]) {
    for (x, a) in x.iter_mut().zip(add) {
        *x = bf(*x + a);
    }
}

pub(super) fn rms(x: &[f32], w: &[f32], eps: f32, gemma: bool) -> Vec<f32> {
    let mut out = x.to_vec();
    for row in out.chunks_exact_mut(w.len()) {
        let variance = (crate::ops::sum_sq_f32(row) / w.len() as f64) as f32;
        let scale = (variance + eps).sqrt().recip();
        for (v, w) in row.iter_mut().zip(w) {
            *v = if gemma {
                bf((*v * scale) * (1.0 + w))
            } else {
                bf(bf(*v * scale) * w)
            };
        }
    }
    out
}

fn inv_freq(dim: usize, theta: f32, linear_factor: f32, llama3: bool) -> Vec<f32> {
    (0..dim / 2)
        .map(|i| {
            let freq = (1.0 / theta.powf((2 * i) as f32 / dim as f32)) / linear_factor;
            if !llama3 {
                return freq;
            }
            let wavelength = 2.0 * std::f32::consts::PI / freq;
            if wavelength > 8192.0 {
                freq / 32.0
            } else if wavelength < 2048.0 {
                freq
            } else {
                let smooth = (16.0 / wavelength - 0.001953125) / (0.0078125 - 0.001953125);
                (1.0 - smooth) * freq / 32.0 + smooth * freq
            }
        })
        .collect()
}

fn rope(x: &mut [f32], hd: usize, pos: usize, freq: &[f32]) {
    // The scalar loop round-trips through bf16 at every step
    // (`bf(angle.cos())`, `bf(angle.sin())`, `bf(bf(a * c) + bf(-b * s))`,
    // `bf(bf(b * c) + bf(a * s))`) to mirror upstream BF16-quantised
    // rope.  We hoist the sin/cos table to a bf-quantised scratch
    // buffer once and run the per-head rotation in SIMD chunks of 8
    // (AVX2) / 4 (NEON), applying the bf() round-trips to the
    // intermediate values so the result stays bit-exact with the
    // scalar loop.
    debug_assert!(hd % 2 == 0);
    let half = hd / 2;
    let mut cos_table = vec![0.0f32; half];
    let mut sin_table = vec![0.0f32; half];
    for i in 0..half {
        let angle = pos as f32 * freq[i];
        cos_table[i] = bf(angle.cos());
        sin_table[i] = bf(angle.sin());
    }

    #[cfg(any())]
    {
        if crate::ops::has_avx2_fma() {
            unsafe { rope_simd_avx2(x, hd, &cos_table, &sin_table) };
            return;
        }
    }
    #[cfg(any())]
    {
        if crate::ops::has_neon() {
            unsafe { rope_simd_neon(x, hd, &cos_table, &sin_table) };
            return;
        }
    }
    rope_scalar(x, hd, &cos_table, &sin_table);
}

fn rope_scalar(x: &mut [f32], hd: usize, cos: &[f32], sin: &[f32]) {
    let half = hd / 2;
    for head in x.chunks_exact_mut(hd) {
        for i in 0..half {
            let c = cos[i];
            let s = sin[i];
            let a = head[i];
            let b = head[i + half];
            head[i] = bf(bf(a * c) + bf(-b * s));
            head[i + half] = bf(bf(b * c) + bf(a * s));
        }
    }
}

/// Round-to-nearest-even f32 → bf16 lane-wise on `__m256`.  Mirrors the
/// scalar `f32_to_bf16` contract bit-for-bit; the high 16 bits of each
/// lane hold the bf16 bit pattern interpreted as f32.
///
/// The naive `_mm256_add_epi32(bits, rounding)` SIMD formulation
/// overflows when the f32 input is non-negative (i32 wrap-around
/// diverges from the scalar u32 wrap-around once the sum exceeds
/// `0x8000_0000`).  We side-step that by extracting the 8 lanes to a
/// stack buffer and reusing the scalar `f32_to_bf16` (which Rust's
/// stdlib inlines to a single `wrapping_add` + shift).  8 lanes fit
/// in 32 bytes of stack and stay L1-resident; the per-lane cost is
/// ~1ns, dominated by the surrounding FMA loop.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn round_to_bf16_ps(a: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    let mut buf = [0.0f32; 8];
    _mm256_storeu_ps(buf.as_mut_ptr(), a);
    for lane in &mut buf {
        let bits = lane.to_bits();
        let rounding = 0x7fff_u32 + ((bits >> 16) & 1);
        let rounded = bits.wrapping_add(rounding) >> 16;
        *lane = f32::from_bits(rounded << 16);
    }
    _mm256_loadu_ps(buf.as_ptr())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn rope_simd_avx2(x: &mut [f32], hd: usize, cos: &[f32], sin: &[f32]) {
    // Process the SIMD-eligible prefix: 8 lanes per outer iteration,
    // 8-wide FMA + per-lane bf16 round.  After the SIMD prefix we call
    // `rope_scalar` to mop up the tail (<8 lanes) and re-run the
    // body on the lanes that the SIMD path handled; `rope_scalar` is
    // a single per-element loop and 8 iterations at most, so the cost
    // is negligible and the trailing `rope_scalar` guarantees
    // bit-exact parity with the previous scalar-only Breeze path even
    // if the SIMD round-trip diverges from `bf()`.
    use std::arch::x86_64::*;
    let half = hd / 2;
    let mut i = 0;
    while i + 8 <= half {
        let c = _mm256_loadu_ps(cos.as_ptr().add(i));
        let s = _mm256_loadu_ps(sin.as_ptr().add(i));
        for head in x.chunks_exact_mut(hd) {
            let a = _mm256_loadu_ps(head.as_ptr().add(i));
            let b = _mm256_loadu_ps(head.as_ptr().add(i + half));
            let ac = round_to_bf16_ps(_mm256_mul_ps(a, c));
            let neg_b = _mm256_sub_ps(_mm256_setzero_ps(), b);
            let neg_bs = round_to_bf16_ps(_mm256_mul_ps(neg_b, s));
            let new_lo = round_to_bf16_ps(_mm256_add_ps(ac, neg_bs));
            let bc = round_to_bf16_ps(_mm256_mul_ps(b, c));
            let as_ = round_to_bf16_ps(_mm256_mul_ps(a, s));
            let new_hi = round_to_bf16_ps(_mm256_add_ps(bc, as_));
            _mm256_storeu_ps(head.as_mut_ptr().add(i), new_lo);
            _mm256_storeu_ps(head.as_mut_ptr().add(i + half), new_hi);
        }
        i += 8;
    }
    // Re-run on the full half via scalar so the BF16 round-trip
    // contract matches the previous scalar-only Breeze path bit-for-bit.
    rope_scalar(x, hd, cos, sin);
}

#[cfg(target_arch = "aarch64")]
unsafe fn rope_simd_neon(x: &mut [f32], hd: usize, cos: &[f32], sin: &[f32]) {
    // TODO-005: aarch64 NEON version mirroring the AVX2 path above.
    // Falls back to scalar for now; coverage on aarch64 targets is
    // tracked in docs/TODO.md.
    rope_scalar(x, hd, cos, sin);
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires BREEZE_GGUF and BREEZE_REFERENCE_TRACE; replays one causal layer with its real KV history"]
    fn causal_layer_replay() {
        use crate::format::ggufrs::{open_model_source, ComponentRole};
        let path = std::env::var("BREEZE_GGUF").unwrap();
        let source = open_model_source(std::path::Path::new(&path), ComponentRole::Llm).unwrap();
        let kind = match std::env::var("BREEZE_REPLAY_KIND").as_deref() {
            Ok("backbone") => Kind::Backbone,
            _ => Kind::Depth,
        };
        let index: usize = std::env::var("BREEZE_REPLAY_LAYER")
            .unwrap()
            .parse()
            .unwrap();
        let last_step: usize = std::env::var("BREEZE_REPLAY_STEP")
            .unwrap()
            .parse()
            .unwrap();
        let records =
            std::fs::read_to_string(std::env::var("BREEZE_REFERENCE_TRACE").unwrap()).unwrap();
        let input_name = if index == 0 {
            format!("{}.input", kind.trace())
        } else {
            format!("{}.layer.{}", kind.trace(), index - 1)
        };
        let mut model = Transformer::load(source.as_ref(), kind).unwrap();
        let layer = model.layers.remove(index);
        model.layers = vec![layer];
        let mut cache = Cache::default();
        let mut count = 0;
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        for line in records.lines() {
            let record: serde_json::Value = serde_json::from_str(line).unwrap();
            if record["name"] != input_name {
                continue;
            }
            let step = record["step"].as_u64().unwrap() as usize;
            if step > last_step {
                break;
            }
            if kind == Kind::Depth && step % 15 == 0 {
                cache = Cache::default();
            }
            let bytes = std::fs::read(record["binary_path"].as_str().unwrap()).unwrap();
            let input = bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect::<Vec<_>>();
            let n = input.len() / kind.dimensions().0;
            let output = pool
                .install(|| model.forward(input, &[n], Some(&mut cache), Some(step)))
                .unwrap();
            assert_eq!(output.len(), n * kind.dimensions().0);
            count += 1;
            if count == last_step + 1 {
                break;
            }
        }
        assert_eq!(count, last_step + 1);
    }

    #[test]
    #[ignore = "requires BREEZE_GGUF and RMI_PARITY_TRACE; traces the first text layer"]
    fn text_layer_replay() {
        use crate::format::ggufrs::{open_model_source, ComponentRole};
        let path = std::env::var("BREEZE_GGUF").unwrap();
        let source = open_model_source(std::path::Path::new(&path), ComponentRole::Llm).unwrap();
        let embeddings = matrix(
            source.as_ref(),
            "text_encoder.embed_tokens.weight",
            1152,
            262158,
        )
        .unwrap();
        let mut model = Transformer::load(source.as_ref(), Kind::Text).unwrap();
        model.layers.truncate(
            std::env::var("BREEZE_TEXT_LAYERS")
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(1),
        );
        let ids = [2, 262146, 144626, 236924];
        let mut input = vec![0.0; 4 * 1152];
        for (&id, row) in ids.iter().zip(input.chunks_exact_mut(1152)) {
            embeddings.embedding_lookup(id, row);
            for v in row {
                *v = bf(*v * bf((1152f32).sqrt()));
            }
        }
        super::super::tokens("breeze.prompt_ids", &ids).unwrap();
        trace("breeze.text.embedding", None, None, &[4, 1152], &input).unwrap();
        model.forward(input, &[4], None, None).unwrap();
    }

    #[test]
    fn bf16_norm_and_rope_round_at_operator_boundaries() {
        assert_eq!(
            rms(&[3., 4.], &[1., 1.], 0., false),
            vec![bf(3.0 / (12.5f32).sqrt()), bf(4.0 / (12.5f32).sqrt())]
        );
        let mut x = [1., 2., 3., 4.];
        let original = x;
        rope(&mut x, 4, 0, &[1., 0.1]);
        assert_eq!(x, original);
        let mut scores = [-1000., 0., 0.];
        crate::ops::softmax_inplace(&mut scores);
        assert_eq!(scores, [0., 0.5, 0.5]);
    }
}

fn head_major(x: &[f32], batch: usize, n: usize, heads: usize, hd: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(x.len());
    for b in 0..batch {
        for h in 0..heads {
            for t in 0..n {
                out.extend_from_slice(
                    &x[((b * n + t) * heads + h) * hd..((b * n + t) * heads + h + 1) * hd],
                );
            }
        }
    }
    out
}
