//! AuK-Base DiT forward — Flux2Edit (Flux-style double + single blocks).
//!
//! High-level forward (TTS path, no reference audio conditioning):
//! 1. Encode the prompt via Qwen2.5-Omni (handled in `conditioning.rs`, not
//!    here) into per-token hidden states of width `TEXT_IN`.
//! 2. Project text into the joint hidden space via `txt_proj`, normalize via
//!    `txt_norm`.
//! 3. Sample an audio latent of shape `[latent_dim, latent_time]`, with
//!    `latent_time = duration_sec * sample_rate / downsample_rate`.
//! 4. Run the DiT forward for `steps` Euler steps. Each step:
//!    a. Time embedding: `timestep_embedding(t) -> time_mlp -> c` of shape
//!       `[hidden]`.
//!    b. Per-block forward through the 10 double blocks then 20 single blocks.
//!    c. Final AdaLNContinuous: `norm_out.linear` produces scale+shift from
//!       `c`, applied to the joint hidden state.
//!    d. Final linear: `proj_out` projects back to the latent.
//!    e. Euler step: `latent += velocity * dt`.
//! 5. Decode the final latent with `BigVGANFlowVAE` to produce audio at
//!    `sample_rate` Hz.
//!
//! Per-block forward (Flux-style, see references/audio.cpp/src/community_models/
//! auk/flow.cpp):
//! - Double block: 6-way AdaLN modulation, joint attention across img/txt
//!   streams, separate FF for each stream. Q/K have RMS norms (head_dim=64).
//! - Single block: 6-way AdaLN modulation, attention only (FF in same block).
//!
//! Only the TTS path is wired in this commit; the CFMEdit reference-audio
//! path is tracked in `docs/develop/TODO.md`.

use super::linear::AukLinearSession;
use crate::ops::kernel::{F16Weight, QuantizedTensor, Weight};
use std::sync::Arc;

use half::f16;

use crate::core::tensor::{GGMLType, TensorSource};
use crate::core::thread_pool::ComputePool;
use crate::ops::{
    attention_value_reduce, rms_norm, rms_norm_inplace, rope_neox_inplace, silu_mul_inplace,
    vec_add_into,
};

use super::{linear_into, validate_component, Component, Q8Scratch};

// === Architecture constants (from references/audio.cpp/docs/community_models/auk.md
//    + config/auk-base.yaml, verified against actual GGUF tensor dims) ===

/// Hidden size.
pub(crate) const HIDDEN: usize = 1536;

/// Attention head count.
pub(crate) const HEADS: usize = 24;
pub(crate) const HEAD_DIM: usize = 64;
pub(crate) const QKV_DIM: usize = HEADS * HEAD_DIM * 3; // 4608

/// Feed-forward inner dim (out dim of FF gate/up, in dim of FF down).
pub(crate) const FF_INNER: usize = 3072;

/// FF gate+up are packed into one matmul of out dim `2 * FF_INNER` = 6144.
pub(crate) const PACKED_FF_IN: usize = FF_INNER * 2;

/// AdaLN modulation dim = 6 * HIDDEN (scale_msa, gate_msa, shift_msa, scale_mlp, gate_mlp, shift_mlp).
pub(crate) const ADALN_DIM: usize = HIDDEN * 6;

/// Final AdaLNContinuous linear projects hidden -> 2*hidden (scale, shift).
pub(crate) const FINAL_NORM_DIM: usize = HIDDEN * 2;

/// Sinusoidal timestep embedding freq dim.
pub(crate) const FREQ_DIM: usize = 256;

/// Text encoder hidden dim (Qwen2.5-Omni-3B n_embd).
pub(crate) const TEXT_IN: usize = 2048;

/// Audio latent channel count.
pub(crate) const LATENT_DIM: usize = 64;

/// Number of double blocks (parallel img/txt attention + FF).
pub(crate) const NUM_DOUBLE_LAYERS: usize = 10;

/// Number of single blocks (sequential img attention + FF).
pub(crate) const NUM_SINGLE_LAYERS: usize = 20;

#[cfg(test)]
mod contract_tests {
    use super::*;

    #[test]
    #[ignore = "requires the published audio.cpp DiT GGUF in RMI_AUK_GGUF"]
    fn published_auk_dit_loads_all_twenty_single_blocks() {
        let path = std::env::var("RMI_AUK_GGUF").expect("published AuK GGUF required");
        let source: Arc<dyn TensorSource> = Arc::from(
            crate::open_model_source(std::path::Path::new(&path), crate::ComponentRole::Llm)
                .unwrap(),
        );
        super::super::validate_dit(source.as_ref()).unwrap();
        let model = AukDit::load(source, Arc::new(ComputePool::new(1))).unwrap();
        assert_eq!(model.double_blocks.len(), 10);
        assert_eq!(model.single_blocks.len(), 20);
    }

    /// The AuK-Flash recipe pins the 4-step time grid
    /// `[0, 0.07612049579620361, 0.2928932309150696, 0.6173166036605835, 1.0]`
    /// from the official `infer_auk.py`. It is (to f32 precision) the
    /// sway-warped uniform grid with `sway_sampling_coef = -1.0`, which is
    /// how the upstream entry point warps the Base schedule.
    #[test]
    fn flash_time_grid_matches_the_official_pinned_grid() {
        const OFFICIAL: [f32; 5] = [
            0.0,
            0.076_120_495_796_203_61,
            0.292_893_230_915_069_6,
            0.617_316_603_660_583_5,
            1.0,
        ];
        let warped = flow_time_grid(4, Some(-1.0), None);
        assert_eq!(warped.len(), 5);
        for (got, want) in warped.iter().zip(OFFICIAL.iter()) {
            assert!(
                (got - want).abs() < 1e-6,
                "sway grid {got} != official {want}"
            );
        }

        // An explicit grid takes precedence over both steps and sway.
        let pinned = flow_time_grid(4, Some(-1.0), Some(&OFFICIAL));
        assert_eq!(pinned, OFFICIAL.to_vec());
    }

    /// Without sway the grid is uniform, and `steps` controls its length.
    #[test]
    fn uniform_time_grid_matches_the_official_linspace() {
        let grid = flow_time_grid(4, None, None);
        assert_eq!(grid.len(), 5);
        for (index, value) in grid.iter().enumerate() {
            let want = index as f32 / 4.0;
            assert!(
                (value - want).abs() < 1e-6,
                "grid[{index}] {value} != {want}"
            );
        }
    }
}

/// Padding multiple for joint sequence length.
pub(crate) const SEQUENCE_MULTIPLE: usize = 32;

/// Phase timers for the AuK DiT forward (nanoseconds).
///
/// Only accumulated when `RMI_AUK_PROFILE` is set, so the hot path pays
/// nothing when the profile is off.
mod profile {
    use std::sync::atomic::{AtomicU64, Ordering};

    pub(super) static LINEAR_NS: AtomicU64 = AtomicU64::new(0);
    pub(super) static ATTENTION_NS: AtomicU64 = AtomicU64::new(0);
    pub(super) static LAYER_NS: AtomicU64 = AtomicU64::new(0);
    pub(super) static SWIGLU_NS: AtomicU64 = AtomicU64::new(0);
    pub(super) static SOFTMAX_NS: AtomicU64 = AtomicU64::new(0);
    pub(super) static ROPE_NS: AtomicU64 = AtomicU64::new(0);
    pub(super) static CONV_NS: AtomicU64 = AtomicU64::new(0);
    pub(super) static CALLS: AtomicU64 = AtomicU64::new(0);

    pub(super) fn enabled() -> bool {
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ENABLED.get_or_init(|| std::env::var_os("RMI_AUK_PROFILE").is_some())
    }

    pub(super) fn add(counter: &AtomicU64, started: std::time::Instant) {
        if enabled() {
            counter.fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
    }

    pub(super) fn count_call() {
        if enabled() {
            CALLS.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub(super) fn dump() {
        let ms = |value: u64| value as f64 / 1e6;
        eprintln!(
            "[auk-profile] linear {:.1}ms ({} calls)  attention {:.1}ms (softmax {:.1}ms)  layer {:.1}ms  swiglu {:.1}ms  rope {:.1}ms  conv {:.1}ms",
            ms(LINEAR_NS.swap(0, Ordering::Relaxed)),
            CALLS.swap(0, Ordering::Relaxed),
            ms(ATTENTION_NS.swap(0, Ordering::Relaxed)),
            ms(SOFTMAX_NS.swap(0, Ordering::Relaxed)),
            ms(LAYER_NS.swap(0, Ordering::Relaxed)),
            ms(SWIGLU_NS.swap(0, Ordering::Relaxed)),
            ms(ROPE_NS.swap(0, Ordering::Relaxed)),
            ms(CONV_NS.swap(0, Ordering::Relaxed)),
        );
    }
}

pub(crate) struct DoubleBlockWeights {
    pub(crate) adaLN_x: String,
    pub(crate) adaLN_x_bias: Vec<f32>,
    pub(crate) adaLN_c: String,
    pub(crate) adaLN_c_bias: Vec<f32>,
    pub(crate) qkv_x: String,
    pub(crate) qkv_x_bias: Vec<f32>,
    pub(crate) qkv_c: String,
    pub(crate) qkv_c_bias: Vec<f32>,
    /// x-stream attention output projection. The doubled suffix `.0` is
    /// because the GGUF mirrors a torch `nn.Sequential` index, not a sub-block.
    pub(crate) out_x: String,
    pub(crate) out_x_bias: Vec<f32>,
    /// c-stream attention output projection. Stored as `attn.to_out_c`
    /// (no `.0` suffix).
    pub(crate) out_c: String,
    pub(crate) out_c_bias: Vec<f32>,
    /// Q/K RMS norm for the audio (x) stream.
    pub(crate) q_norm: Vec<f32>,
    pub(crate) k_norm: Vec<f32>,
    /// Q/K RMS norm for the text (c) stream. The official `JointAttnProcessor`
    /// uses separate `attn.c_q_norm` / `attn.c_k_norm` modules; the GGUF
    /// stores them as `transformer.transformer_blocks.N.attn.c_q_norm.weight`.
    pub(crate) c_q_norm: Vec<f32>,
    pub(crate) c_k_norm: Vec<f32>,
    pub(crate) ff_x_in: String,
    pub(crate) ff_x_out: String,
    pub(crate) ff_c_in: String,
    pub(crate) ff_c_out: String,
}

pub(crate) struct SingleBlockWeights {
    pub(crate) adaLN: String,
    pub(crate) adaLN_bias: Vec<f32>,
    pub(crate) qkv: String,
    pub(crate) qkv_bias: Vec<f32>,
    pub(crate) out: String,
    pub(crate) out_bias: Vec<f32>,
    pub(crate) q_norm: Vec<f32>,
    pub(crate) k_norm: Vec<f32>,
    pub(crate) ff_in: String,
    pub(crate) ff_out: String,
}

/// ConvPositionEmbedding kernel size (`AuK modules.ConvPositionEmbedding`).
const CONV_POS_KERNEL: usize = 31;
/// ConvPositionEmbedding group count: `1536 / 16 = 96` channels per group.
const CONV_POS_GROUPS: usize = 16;
const CONV_POS_CHANNELS_PER_GROUP: usize = HIDDEN / CONV_POS_GROUPS;

/// AuK `AudioEmbed.conv_pos_embed`: two grouped Conv1d layers (kernel 31,
/// 16 groups) with Mish between and after, applied to the projected latent
/// and added back as a residual:
///
/// ```python
/// x = self.linear(x)
/// x = self.conv_pos_embed(x) + x
/// ```
///
/// Weights are stored in GGUF layout `[kernel][in_per_group][out]`.
struct ConvPosEmbedding {
    conv0_weight: Vec<f32>,
    conv0_bias: Vec<f32>,
    conv1_weight: Vec<f32>,
    conv1_bias: Vec<f32>,
}

impl ConvPosEmbedding {
    fn load(source: &dyn TensorSource) -> Result<Self, String> {
        let weight_len = CONV_POS_KERNEL * CONV_POS_CHANNELS_PER_GROUP * HIDDEN;
        Ok(Self {
            conv0_weight: pack_grouped_conv_weight(load_f32_tensor(
                source,
                "transformer.audio_embed.conv_pos_embed.conv1d.0.weight",
                weight_len,
            )?),
            conv0_bias: load_f32_vector(
                source,
                "transformer.audio_embed.conv_pos_embed.conv1d.0.bias",
                HIDDEN,
            )?,
            conv1_weight: pack_grouped_conv_weight(load_f32_tensor(
                source,
                "transformer.audio_embed.conv_pos_embed.conv1d.2.weight",
                weight_len,
            )?),
            conv1_bias: load_f32_vector(
                source,
                "transformer.audio_embed.conv_pos_embed.conv1d.2.bias",
                HIDDEN,
            )?,
        })
    }

    /// Token-major forward: `input`/`out` are `[n_tokens][HIDDEN]`, `tmp` is a
    /// scratch buffer of the same size. Returns `conv_pos_embed(input)`.
    fn forward(
        &self,
        pool: &ComputePool,
        input: &[f32],
        n_tokens: usize,
        tmp: &mut [f32],
        out: &mut [f32],
    ) {
        grouped_conv1d_mish(
            pool,
            &self.conv0_weight,
            &self.conv0_bias,
            input,
            n_tokens,
            tmp,
        );
        grouped_conv1d_mish(
            pool,
            &self.conv1_weight,
            &self.conv1_bias,
            tmp,
            n_tokens,
            out,
        );
    }
}

/// Repack a grouped convolution weight from `[out][in][kernel]` into
/// `[out/4][kernel][in][4]`.
///
/// The four output channels of a group share the same input channels, so
/// their accumulations are four independent dependency chains that a
/// vector unit can carry at once. The kernel taps and the input channels
/// are still visited in the same order as the scalar formulation, and the
/// multiply and the add stay separate, so the result is bit-identical.
fn pack_grouped_conv_weight(weight: Vec<f32>) -> Vec<f32> {
    let out_channels = HIDDEN;
    let in_per_group = CONV_POS_CHANNELS_PER_GROUP;
    let kernel = CONV_POS_KERNEL;
    let blocks = out_channels / CONV_ACC;
    let mut packed = vec![0.0_f32; weight.len()];
    for block in 0..blocks {
        for k in 0..kernel {
            for j in 0..in_per_group {
                for lane in 0..CONV_ACC {
                    let oc = block * CONV_ACC + lane;
                    packed[((block * kernel + k) * in_per_group + j) * CONV_ACC + lane] =
                        weight[(oc * in_per_group + j) * kernel + k];
                }
            }
        }
    }
    packed
}

/// Number of output channels the convolution accumulates at once.
const CONV_ACC: usize = 4;

/// One grouped 1D convolution with a Mish activation, over one token range.
///
/// `weight` uses the `[out/4][kernel][in][4]` layout of
/// [`pack_grouped_conv_weight`].
///
/// The four output channels of a block are carried by the four lanes of one
/// accumulator, so the four dependency chains proceed in parallel. The
/// multiply and the add stay separate -- no fused multiply-add -- so every
/// lane reproduces the scalar accumulation exactly.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn grouped_conv1d_mish_range(
    weight: &[f32],
    bias: &[f32],
    input: &[f32],
    out: &mut [f32],
    tokens: usize,
    start: usize,
    end: usize,
) {
    use std::arch::aarch64::*;
    const OUT_PER_GROUP: usize = HIDDEN / CONV_POS_GROUPS;
    const IN_PER_GROUP: usize = CONV_POS_CHANNELS_PER_GROUP;
    const PAD: usize = CONV_POS_KERNEL / 2;
    let blocks = HIDDEN / CONV_ACC;

    for token in start..end {
        let dst = out.as_mut_ptr().add(token * HIDDEN);
        for block in 0..blocks {
            let oc = block * CONV_ACC;
            // The four channels of a block always share the same group
            // because `OUT_PER_GROUP` is a multiple of `CONV_ACC`.
            let in_base = (oc / OUT_PER_GROUP) * IN_PER_GROUP;
            let mut acc = vld1q_f32(bias.as_ptr().add(oc));
            for k in 0..CONV_POS_KERNEL {
                let Some(src_token) = token.checked_add(k).and_then(|t| t.checked_sub(PAD)) else {
                    continue;
                };
                if src_token >= tokens {
                    continue;
                }
                let src = input.as_ptr().add(src_token * HIDDEN + in_base);
                let w_base = (block * CONV_POS_KERNEL + k) * IN_PER_GROUP * CONV_ACC;
                let w = weight.as_ptr().add(w_base);
                for j in 0..IN_PER_GROUP {
                    let wv = vld1q_f32(w.add(j * CONV_ACC));
                    let sv = vdupq_n_f32(*src.add(j));
                    acc = vaddq_f32(acc, vmulq_f32(wv, sv));
                }
            }
            vst1q_f32(dst.add(oc), acc);
            for lane in 0..CONV_ACC {
                *dst.add(oc + lane) = mish(*dst.add(oc + lane));
            }
        }
    }
}

/// One grouped 1D convolution with a Mish activation, over one token range.
///
/// `weight` uses the `[out/4][kernel][in][4]` layout of
/// [`pack_grouped_conv_weight`].
#[cfg(not(target_arch = "aarch64"))]
fn grouped_conv1d_mish_range(
    weight: &[f32],
    bias: &[f32],
    input: &[f32],
    out: &mut [f32],
    tokens: usize,
    start: usize,
    end: usize,
) {
    const OUT_PER_GROUP: usize = HIDDEN / CONV_POS_GROUPS;
    const IN_PER_GROUP: usize = CONV_POS_CHANNELS_PER_GROUP;
    const PAD: usize = CONV_POS_KERNEL / 2;
    debug_assert_eq!(OUT_PER_GROUP, IN_PER_GROUP);
    let blocks = HIDDEN / CONV_ACC;

    for token in start..end {
        let dst = &mut out[token * HIDDEN..(token + 1) * HIDDEN];
        for block in 0..blocks {
            let oc = block * CONV_ACC;
            // The four channels of a block always share the same group
            // because `OUT_PER_GROUP` is a multiple of `CONV_ACC`.
            let in_base = (oc / OUT_PER_GROUP) * IN_PER_GROUP;
            let mut acc = [bias[oc], bias[oc + 1], bias[oc + 2], bias[oc + 3]];
            for k in 0..CONV_POS_KERNEL {
                let Some(src_token) = token.checked_add(k).and_then(|t| t.checked_sub(PAD)) else {
                    continue;
                };
                if src_token >= tokens {
                    continue;
                }
                let src = &input
                    [src_token * HIDDEN + in_base..src_token * HIDDEN + in_base + IN_PER_GROUP];
                let w_base = (block * CONV_POS_KERNEL + k) * IN_PER_GROUP * CONV_ACC;
                for j in 0..IN_PER_GROUP {
                    let w = &weight[w_base + j * CONV_ACC..w_base + (j + 1) * CONV_ACC];
                    acc[0] += w[0] * src[j];
                    acc[1] += w[1] * src[j];
                    acc[2] += w[2] * src[j];
                    acc[3] += w[3] * src[j];
                }
            }
            dst[oc] = mish(acc[0]);
            dst[oc + 1] = mish(acc[1]);
            dst[oc + 2] = mish(acc[2]);
            dst[oc + 3] = mish(acc[3]);
        }
    }
}

pub(crate) struct AukDit {
    source: Arc<dyn TensorSource>,
    pool: Arc<ComputePool>,
    audio_embed_weight: String,
    audio_embed_bias: Vec<f32>,
    conv_pos_embed: ConvPosEmbedding,
    time_mlp_0_weight: String,
    time_mlp_0_bias: Vec<f32>,
    time_mlp_2_weight: String,
    time_mlp_2_bias: Vec<f32>,
    txt_proj_weight: String,
    txt_proj_bias: Vec<f32>,
    txt_norm_weight: Vec<f32>,
    norm_out_weight: String,
    norm_out_bias: Vec<f32>,
    proj_out_weight: String,
    proj_out_bias: Vec<f32>,
    rotary_inv_freq: Vec<f32>,
    double_blocks: Vec<DoubleBlockWeights>,
    single_blocks: Vec<SingleBlockWeights>,
    q8: Q8Scratch,
}

impl AukDit {
    pub(crate) fn load(
        source: Arc<dyn TensorSource>,
        pool: Arc<ComputePool>,
    ) -> Result<Self, String> {
        let source_ref = source.as_ref();
        let audio_embed_bias =
            load_f32_vector(source_ref, "transformer.audio_embed.linear.bias", HIDDEN)?;
        let conv_pos_embed = ConvPosEmbedding::load(source_ref)?;
        let time_mlp_0_bias =
            load_f32_vector(source_ref, "transformer.time_embed.time_mlp.0.bias", HIDDEN)?;
        let time_mlp_2_bias =
            load_f32_vector(source_ref, "transformer.time_embed.time_mlp.2.bias", HIDDEN)?;
        let txt_proj_bias = load_f32_vector(source_ref, "transformer.txt_proj.bias", HIDDEN)?;
        let txt_norm_weight = load_f32_vector(source_ref, "transformer.txt_norm.weight", HIDDEN)?;
        let norm_out_bias = load_f32_vector(
            source_ref,
            "transformer.norm_out.linear.bias",
            FINAL_NORM_DIM,
        )?;
        let proj_out_bias = load_f32_vector(source_ref, "transformer.proj_out.bias", LATENT_DIM)?;
        // rotary_embed.inv_freq is stored as F32 with length HEAD_DIM/2 = 32.
        let rotary_inv_freq = load_f32_vector(
            source_ref,
            "transformer.rotary_embed.inv_freq",
            HEAD_DIM / 2,
        )?;
        let mut double_blocks = Vec::with_capacity(NUM_DOUBLE_LAYERS);
        for layer in 0..NUM_DOUBLE_LAYERS {
            double_blocks.push(load_double_block(source_ref, layer)?);
        }
        let mut single_blocks = Vec::with_capacity(NUM_SINGLE_LAYERS);
        for layer in 0..NUM_SINGLE_LAYERS {
            single_blocks.push(load_single_block(source_ref, layer)?);
        }
        Ok(Self {
            q8: Q8Scratch::new(FF_INNER.max(HIDDEN)),
            source: source.clone(),
            pool: pool.clone(),
            audio_embed_weight: "transformer.audio_embed.linear.weight".into(),
            audio_embed_bias,
            conv_pos_embed,
            time_mlp_0_weight: "transformer.time_embed.time_mlp.0.weight".into(),
            time_mlp_0_bias,
            time_mlp_2_weight: "transformer.time_embed.time_mlp.2.weight".into(),
            time_mlp_2_bias,
            txt_proj_weight: "transformer.txt_proj.weight".into(),
            txt_proj_bias,
            txt_norm_weight,
            norm_out_weight: "transformer.norm_out.linear.weight".into(),
            norm_out_bias,
            proj_out_weight: "transformer.proj_out.weight".into(),
            proj_out_bias,
            rotary_inv_freq,
            double_blocks,
            single_blocks,
        })
    }

    fn linear_weights(&self) -> Result<Vec<(String, Weight<'_>)>, String> {
        let source = self.source.as_ref();
        let mut out = Vec::new();
        let mut names: Vec<String> = vec![
            "transformer.audio_embed.linear.weight".to_string(),
            "transformer.time_embed.time_mlp.0.weight".to_string(),
            "transformer.time_embed.time_mlp.2.weight".to_string(),
            "transformer.txt_proj.weight".to_string(),
            "transformer.norm_out.linear.weight".to_string(),
            "transformer.proj_out.weight".to_string(),
        ];
        for layer in 0..NUM_DOUBLE_LAYERS {
            names.push(format!(
                "transformer.transformer_blocks.{layer}.attn_norm_x.linear.weight"
            ));
            names.push(format!(
                "transformer.transformer_blocks.{layer}.attn_norm_c.linear.weight"
            ));
            names.push(format!(
                "transformer.transformer_blocks.{layer}.attn.to_qkv.weight"
            ));
            names.push(format!(
                "transformer.transformer_blocks.{layer}.attn.to_qkv_c.weight"
            ));
            names.push(format!(
                "transformer.transformer_blocks.{layer}.attn.to_out.0.weight"
            ));
            names.push(format!(
                "transformer.transformer_blocks.{layer}.attn.to_out_c.weight"
            ));
            names.push(format!(
                "transformer.transformer_blocks.{layer}.ff_x.linear_in.weight"
            ));
            names.push(format!(
                "transformer.transformer_blocks.{layer}.ff_x.linear_out.weight"
            ));
            names.push(format!(
                "transformer.transformer_blocks.{layer}.ff_c.linear_in.weight"
            ));
            names.push(format!(
                "transformer.transformer_blocks.{layer}.ff_c.linear_out.weight"
            ));
        }
        for layer in 0..NUM_SINGLE_LAYERS {
            names.push(format!(
                "transformer.single_transformer_blocks.{layer}.attn_norm.linear.weight"
            ));
            names.push(format!(
                "transformer.single_transformer_blocks.{layer}.attn.to_qkv.weight"
            ));
            names.push(format!(
                "transformer.single_transformer_blocks.{layer}.attn.to_out.0.weight"
            ));
            names.push(format!(
                "transformer.single_transformer_blocks.{layer}.ff.linear_in.weight"
            ));
            names.push(format!(
                "transformer.single_transformer_blocks.{layer}.ff.linear_out.weight"
            ));
        }
        for name in names {
            let info = match source.tensor_info(&name) {
                Some(i) => i,
                None => continue,
            };
            if !matches!(info.ggml_type, GGMLType::F16) {
                continue;
            }
            let dims = info.dims.clone();
            if dims.len() != 2 {
                continue;
            }
            let n_in = dims[0] as usize;
            let n_out = dims[1] as usize;
            let bytes = match source.tensor_slice(&name) {
                Some(b) => b,
                None => continue,
            };
            out.push((
                name,
                Weight::from_quantized(QuantizedTensor::F16(F16Weight { bytes, n_in, n_out })),
            ));
        }
        Ok(out)
    }

    /// Run the diffusion loop for `options.steps` Euler steps and return
    /// the final latent of shape `[latent_dim, latent_time]`.
    pub(crate) fn denoise(
        &self,
        text_conditioning: &[f32],
        text_tokens: usize,
        audio_conditioning: &[f32],
        audio_tokens: usize,
        options: &super::AukOptions,
    ) -> Result<Vec<f32>, String> {
        if text_tokens == 0 {
            return Err("AuK text token count must be positive".into());
        }
        if audio_tokens > 0 && audio_conditioning.len() != audio_tokens * TEXT_IN {
            return Err(format!(
                "AuK audio conditioning length {} != audio_tokens*TEXT_IN={}",
                audio_conditioning.len(),
                audio_tokens * TEXT_IN
            ));
        }
        let cond_tokens = audio_tokens + text_tokens;
        // Compute latent time from duration / sample rate / downsample rate.
        let downsample_rate = 480;
        let latent_time = options
            .duration_sec
            .checked_mul(options.sample_rate as usize)
            .and_then(|v| v.checked_div(downsample_rate))
            .ok_or("AuK latent_time overflow")?;
        if latent_time == 0 {
            return Err("AuK duration too short for one latent frame".into());
        }
        let latent_values = LATENT_DIM * latent_time;
        let mut latent = vec![0.0_f32; latent_values];
        let mut rng = SplitMix64::new(options.seed as u64);
        for value in &mut latent {
            *value = gaussian(&mut rng);
        }
        let steps = options.steps;
        // An explicit time grid (e.g. the AuK-Flash recipe) overrides the
        // step count; make sure the two agree so the sampling loop cannot
        // index past the end of the grid.
        if let Some(grid) = options.time_grid.as_ref() {
            if grid.len() != steps + 1 {
                return Err(format!(
                    "AuK time grid has {} entries but {} steps were requested (expected {})",
                    grid.len(),
                    steps,
                    steps + 1
                ));
            }
        }
        let mut velocity = vec![0.0_f32; latent_values];
        let mut uncond_velocity = vec![0.0_f32; latent_values];
        let mut scratch = AukScratch::new(cond_tokens, latent_time)?;
        let weights = self.linear_weights()?;
        let linear = AukLinearSession::new(
            &weights,
            crate::compute::ComputePolicy::legacy(),
            self.pool.clone(),
        )?;
        let guidance_scale = options.guidance_scale;
        // Official (`CFMEdit.sample`) only skips the CFG branch when the
        // strength is effectively zero:
        // ```python
        // if cfg_strength < 1e-5:
        //     return self.transformer(..., drop_text=False, ...)
        // ...
        // return v_cond + (v_cond - v_uncond) * cfg_strength
        // ```
        // Note that `cfg_strength == 0` degenerates to `v_cond` anyway, so
        // the threshold is consistent with the formula.
        let do_cfg = guidance_scale >= 1e-5;
        // Official time grid (`CFMEdit.sample`):
        // ```python
        // t = torch.linspace(0, 1, steps + 1)
        // if sway_sampling_coef is not None:
        //     t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)
        // ```
        // The upstream inference entry point defaults to
        // `sway_sampling_coef = -1.0`, which warps the grid so the early
        // steps cluster near t=0 (where the noise is most informative).
        // The distilled AuK-Flash checkpoint pins the 4-step warped grid
        // `[0, 0.0761, 0.2929, 0.6173, 1.0]` -- exactly this formula.
        let time_grid = flow_time_grid(
            steps,
            options.sway_sampling_coef,
            options.time_grid.as_deref(),
        );
        for step in 0..steps {
            // Flow-matching time: 0 = pure noise, 1 = clean data. The official
            // sampler integrates `odeint(fn, y0, linspace(0, 1, steps + 1))`.
            let t = time_grid[step];
            let t_next = time_grid[step + 1];
            // Conditional forward. `text_conditioning` is laid out as
            // `[audio_tokens | text_tokens]` when audio is present, with
            // audio embeddings at the front (matching audio.cpp's
            // conditioning.cpp where audio is concatenated before text in
            // the joint sequence).
            self.predict_velocity_inner(
                &linear,
                &mut latent,
                latent_time,
                text_conditioning,
                cond_tokens,
                audio_tokens,
                t,
                step,
                &mut scratch,
                &mut velocity,
                false,
            )?;
            let conditional_velocity = velocity.clone();
            // Unconditional forward -- only when CFG is enabled
            // (`cfg_strength >= 1e-5`). Skip the second pass otherwise; the
            // velocity already is the conditional prediction.
            if do_cfg {
                self.predict_velocity_inner(
                    &linear,
                    &mut latent,
                    latent_time,
                    text_conditioning,
                    cond_tokens,
                    audio_tokens,
                    t,
                    step,
                    &mut scratch,
                    &mut uncond_velocity,
                    true,
                )?;
                // Official (`CFMEdit.sample`):
                //     v_cond + (v_cond - v_uncond) * cfg_strength
                // Note the coefficient on `v_cond` is `1 + cfg`, not `cfg`.
                for i in 0..velocity.len() {
                    velocity[i] = velocity[i] + guidance_scale * (velocity[i] - uncond_velocity[i]);
                }
            }
            // A correct forward keeps the velocity bounded; non-finite values
            // indicate a numerical bug, so surface them instead of clamping.
            if std::env::var("RMI_AUK_DEBUG").is_ok() {
                eprintln!(
                    "[auk-debug] step {step} t={t:.3} velocity rms={:.4} max={:.4} latent rms={:.4}",
                    rms(&velocity),
                    max_abs(&velocity),
                    rms(&latent)
                );
            }
            super::super::common::require_finite(&velocity, "velocity")?;
            if let Some(dir) = std::env::var_os("RMI_AUK_DIT_DUMP") {
                let dir = std::path::PathBuf::from(dir);
                let _ = std::fs::create_dir_all(&dir);
                dump_f32(&dir.join(format!("velocity_{step}.f32")), &velocity);
                dump_f32(&dir.join(format!("latent_{step}.f32")), &latent);
                // Un-guided conditional velocity, for comparison against the
                // PyTorch reference (which does not apply CFG).
                dump_f32(
                    &dir.join(format!("velocity_cond_{step}.f32")),
                    &conditional_velocity,
                );
            }
            euler_flow_step(&mut latent, &velocity, t, t_next)?;
        }
        if std::env::var_os("RMI_AUK_DEBUG").is_some() {
            #[cfg(feature = "vulkan")]
            if let Some(context) = crate::ops::get_vulkan_context() {
                let stats = context.compute_stats();
                eprintln!(
                    "[auk-debug] gpu submissions={} static_uploads={} upload_bytes={:.2} GiB host_write={:.2} MiB host_read={:.2} MiB",
                    stats.submissions,
                    stats.static_uploads,
                    stats.static_upload_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
                    stats.host_write_bytes as f64 / (1024.0 * 1024.0),
                    stats.host_read_bytes as f64 / (1024.0 * 1024.0),
                );
                crate::vulkan::dump_submit_trace();
                crate::vulkan::ops::dump_linear_profile();
            }
        }
        if profile::enabled() {
            profile::dump();
        }
        Ok(latent)
    }

    #[allow(clippy::too_many_arguments)]
    fn predict_velocity_inner(
        &self,
        linear: &AukLinearSession<'_, '_>,
        latent: &mut [f32],
        latent_time: usize,
        text_conditioning: &[f32],
        cond_tokens: usize,
        audio_tokens: usize,
        t: f32,
        step: usize,
        scratch: &mut AukScratch,
        velocity: &mut [f32],
        drop_text: bool,
    ) -> Result<(), String> {
        if !t.is_finite() || !(0.0..=1.0).contains(&t) {
            return Err("AuK flow time must be finite and within [0, 1]".into());
        }
        super::super::common::require_finite(latent, "latent")?;
        let latent_values = LATENT_DIM * latent_time;
        if latent.len() != latent_values {
            return Err("Invalid AuK latent length".into());
        }
        let img_tokens = latent_time;
        // CFMEdit adds a reference audio; this TTS-only path initializes the
        // joint sequence with just text (img tokens come from `latent`).
        // We wire the joint into `scratch.joint` (img tokens then text).
        scratch.prepare(img_tokens, cond_tokens, self.pool.n_threads())?;
        // Rotary embedding tables. The angle of a pair only depends on the
        // token position and the frequency index, so the table is built once
        // per forward instead of once per (token, head, pair): the blocks
        // rotate 24 heads with the same 32 angles.
        {
            let half = HEAD_DIM / 2;
            let total = img_tokens + cond_tokens;
            for pos in 0..total {
                for i in 0..half {
                    let theta = pos as f32 * self.rotary_inv_freq[i];
                    let (cos_t, sin_t) = crate::ops::rope::neox::rope_sin_cos(theta);
                    scratch.rope_cos[pos * half + i] = cos_t;
                    scratch.rope_sin[pos * half + i] = sin_t;
                }
            }
        }

        // Time embedding: c ∈ [HIDDEN].
        // `t` is the flow-matching time in [0, 1] (0 = noise, 1 = data),
        // matching the official `odeint(fn, y0, linspace(0, 1, steps+1))`.
        timestep_embedding(t, &mut scratch.time_frequency);
        linear_into_dispatch(
            linear,
            self.source.as_ref(),
            &self.time_mlp_0_weight,
            FREQ_DIM,
            HIDDEN,
            &scratch.time_frequency,
            &mut scratch.time_hidden,
            &mut scratch.q8,
            self.pool.as_ref(),
        )?;
        for (v, b) in scratch.time_hidden.iter_mut().zip(&self.time_mlp_0_bias) {
            *v += *b;
        }
        // `TimestepEmbedding`: linear -> SiLU -> linear. (`silu_mul_inplace`
        // is the SwiGLU helper and would multiply by an unrelated buffer.)
        for (dst, src) in scratch
            .time_hidden_silu
            .iter_mut()
            .zip(&scratch.time_hidden)
        {
            *dst = src / (1.0 + (-src).exp());
        }
        linear_into_dispatch(
            linear,
            self.source.as_ref(),
            &self.time_mlp_2_weight,
            HIDDEN,
            HIDDEN,
            &scratch.time_hidden_silu,
            &mut scratch.time,
            &mut scratch.q8,
            self.pool.as_ref(),
        )?;
        for (v, b) in scratch.time.iter_mut().zip(&self.time_mlp_2_bias) {
            *v += *b;
        }
        super::super::common::require_finite(&scratch.time, "time conditioning")?;
        // `AdaLayerNorm` projects `linear(silu(emb))`; reuse the scratch buffer
        // for the SiLU of the time embedding.
        for (dst, src) in scratch.time_hidden_silu.iter_mut().zip(&scratch.time) {
            *dst = src / (1.0 + (-src).exp());
        }

        // Image (audio latent) embed: latent -> hidden
        run_audio_embed(
            linear,
            self.source.as_ref(),
            &self.audio_embed_weight,
            &self.audio_embed_bias,
            latent,
            latent_time,
            &mut scratch.img,
            self.pool.as_ref(),
            &mut scratch.q8,
        )?;
        // AudioEmbed: `x = conv_pos_embed(x) + x`. The grouped Conv1d pair
        // injects local temporal context; without it the DiT has no notion of
        // neighbouring latent frames and only produces noise.
        let conv_started = std::time::Instant::now();
        self.conv_pos_embed.forward(
            self.pool.as_ref(),
            &scratch.img,
            latent_time,
            &mut scratch.conv_pos,
            &mut scratch.conv_pos_out,
        );
        profile::add(&profile::CONV_NS, conv_started);
        for (value, residual) in scratch.img.iter_mut().zip(&scratch.conv_pos_out) {
            *value += *residual;
        }
        super::super::common::require_finite(&scratch.img, "audio_embed")?;
        if std::env::var("RMI_AUK_DEBUG").is_ok() {
            eprintln!(
                "[auk-debug] audio_embed rms={:.4} max={:.4}",
                rms(&scratch.img),
                max_abs(&scratch.img)
            );
        }
        if let Some(dir) = std::env::var_os("RMI_AUK_DIT_DUMP") {
            let dir = std::path::PathBuf::from(dir);
            let _ = std::fs::create_dir_all(&dir);
            dump_f32(&dir.join("text_conditioning.f32"), text_conditioning);
            dump_f32(&dir.join(format!("latent_in_{step}.f32")), latent);
            dump_f32(&dir.join(format!("audio_embed_{step}.f32")), &scratch.img);
        }

        // Text (and audio) conditioning embed: text_in -> hidden, then norm.
        // The layout matches audio.cpp's conditioning.cpp: audio embeddings
        // occupy positions `[0, audio_tokens)` followed by text tokens in
        // `[audio_tokens, cond_tokens)`. Both go through the same txt_proj
        // because the Qwen2.5-Omni audio tower projects to TEXT_IN=2048 dim,
        // the same space as text embeddings.
        //
        // The conditioning rows are contiguous, so the projection runs as one
        // batched matmul over all conditioning tokens.
        linear_rows_into_dispatch(
            linear,
            self.source.as_ref(),
            &self.txt_proj_weight,
            TEXT_IN,
            HIDDEN,
            &text_conditioning[..cond_tokens * TEXT_IN],
            cond_tokens,
            &mut scratch.text[..cond_tokens * HIDDEN],
            &mut scratch.q8,
            self.pool.as_ref(),
        )?;
        // Add bias.
        for token in 0..cond_tokens {
            for d in 0..HIDDEN {
                scratch.text[token * HIDDEN + d] += self.txt_proj_bias[d];
            }
        }
        // RMS norm with txt_norm.
        rms_norm_inplace_text(&mut scratch.text, &self.txt_norm_weight, cond_tokens);
        // Official `project_text(drop_text=True)` zeroes the text **after**
        // the projection and the norm:
        // ```python
        // c = self.txt_norm(self.txt_proj(text))
        // if drop_text:
        //     c = torch.zeros_like(c)
        // ```
        // Zeroing the *pre-projection* LLM hidden states instead would leave
        // the `txt_proj` bias intact (`txt_proj(0) == bias`), so the
        // unconditional branch would still carry text information.
        if drop_text {
            for value in scratch.text[..cond_tokens * HIDDEN].iter_mut() {
                *value = 0.0;
            }
        }
        if let Some(dir) = std::env::var_os("RMI_AUK_DIT_DUMP") {
            let dir = std::path::PathBuf::from(dir);
            let _ = std::fs::create_dir_all(&dir);
            // Only the conditional branch carries the projected text; the
            // unconditional branch zeroes it on purpose.
            if !drop_text {
                dump_f32(
                    &dir.join(format!("text_embed_{step}.f32")),
                    &scratch.text[..cond_tokens * HIDDEN],
                );
            }
        }

        // Concat: image tokens first, then conditioning tokens (audio + text).
        // Both are already in `scratch.img` and `scratch.text` (sized for
        // `total_tokens` rows). Layout matches audio.cpp flow.cpp:
        //   [img_tokens | audio_tokens | text_tokens]
        let total_tokens = scratch.joint.len() / HIDDEN;
        for token in 0..img_tokens {
            scratch.joint[token * HIDDEN..(token + 1) * HIDDEN]
                .copy_from_slice(&scratch.img[token * HIDDEN..(token + 1) * HIDDEN]);
        }
        for token in 0..cond_tokens {
            let dst = img_tokens + token;
            scratch.joint[dst * HIDDEN..(dst + 1) * HIDDEN]
                .copy_from_slice(&scratch.text[token * HIDDEN..(token + 1) * HIDDEN]);
        }
        let _ = total_tokens;

        // 10 double blocks
        for (layer_index, block) in self.double_blocks.iter().enumerate() {
            // Project time_emb through per-block AdaLN linear layers (each
            // produces 9216 = 6*1536 modulation values).
            linear_into_dispatch(
                linear,
                self.source.as_ref(),
                &block.adaLN_x,
                HIDDEN,
                ADALN_DIM,
                &scratch.time_hidden_silu,
                &mut scratch.modulation[..ADALN_DIM],
                &mut scratch.q8,
                self.pool.as_ref(),
            )?;
            for (v, b) in scratch.modulation[..ADALN_DIM]
                .iter_mut()
                .zip(&block.adaLN_x_bias)
            {
                *v += *b;
            }
            linear_into_dispatch(
                linear,
                self.source.as_ref(),
                &block.adaLN_c,
                HIDDEN,
                ADALN_DIM,
                &scratch.time_hidden_silu,
                &mut scratch.modulation[ADALN_DIM..2 * ADALN_DIM],
                &mut scratch.q8,
                self.pool.as_ref(),
            )?;
            for (v, b) in scratch.modulation[ADALN_DIM..2 * ADALN_DIM]
                .iter_mut()
                .zip(&block.adaLN_c_bias)
            {
                *v += *b;
            }
            run_double_block(
                linear,
                self.source.as_ref(),
                block,
                &mut scratch.joint,
                img_tokens,
                cond_tokens,
                &scratch.rope_cos,
                &scratch.rope_sin,
                &scratch.modulation[..2 * ADALN_DIM],
                &mut scratch.qkv,
                &mut scratch.qkv_c,
                &mut scratch.qkv_head,
                &mut scratch.attention,
                &mut scratch.scores,
                &mut scratch.normed_buf,
                &mut scratch.ff_packed,
                &mut scratch.ff_gated,
                &mut scratch.q8,
                self.pool.as_ref(),
                layer_index,
            )?;
        }

        // Swap joint order from [img, text] (double-block convention) to
        // [text, img] (single-block convention). audio.cpp does this via
        // ConcatModule({1}).build(text, img) before the single-stream stage and
        // resets positions to 0..text+img sequentially.
        let total = img_tokens + cond_tokens;
        if img_tokens > 0 && cond_tokens > 0 {
            let mut swapped = vec![0.0_f32; total * HIDDEN];
            for token in 0..cond_tokens {
                swapped[token * HIDDEN..(token + 1) * HIDDEN].copy_from_slice(
                    &scratch.joint
                        [(img_tokens + token) * HIDDEN..(img_tokens + token + 1) * HIDDEN],
                );
            }
            for token in 0..img_tokens {
                swapped[(cond_tokens + token) * HIDDEN..(cond_tokens + token + 1) * HIDDEN]
                    .copy_from_slice(&scratch.joint[token * HIDDEN..(token + 1) * HIDDEN]);
            }
            scratch.joint[..total * HIDDEN].copy_from_slice(&swapped);
        }

        // Every single block present in the validated export.
        for (layer_index, block) in self.single_blocks.iter().enumerate() {
            linear_into_dispatch(
                linear,
                self.source.as_ref(),
                &block.adaLN,
                HIDDEN,
                ADALN_DIM,
                &scratch.time_hidden_silu,
                &mut scratch.modulation[..ADALN_DIM],
                &mut scratch.q8,
                self.pool.as_ref(),
            )?;
            for (v, b) in scratch.modulation[..ADALN_DIM]
                .iter_mut()
                .zip(&block.adaLN_bias)
            {
                *v += *b;
            }
            run_single_block(
                linear,
                self.source.as_ref(),
                block,
                &mut scratch.joint,
                img_tokens,
                cond_tokens,
                &scratch.rope_cos,
                &scratch.rope_sin,
                &scratch.modulation[..ADALN_DIM],
                &mut scratch.qkv,
                &mut scratch.qkv_head,
                &mut scratch.attention,
                &mut scratch.scores,
                &mut scratch.normed_buf,
                &mut scratch.ff_packed,
                &mut scratch.ff_gated,
                &mut scratch.q8,
                self.pool.as_ref(),
                layer_index,
            )?;
        }

        // Final AdaLNContinuous: norm_out.linear produces scale+shift.
        // The unsloth/ERNIE-Image case shows that the AdaLNContinuous inner
        // norm (final_norm.norm.weight) is often dropped -- we treat it as
        // identity here.
        linear_into_dispatch(
            linear,
            self.source.as_ref(),
            &self.norm_out_weight,
            HIDDEN,
            FINAL_NORM_DIM,
            &scratch.time_hidden_silu,
            &mut scratch.modulation[..FINAL_NORM_DIM],
            &mut scratch.q8,
            self.pool.as_ref(),
        )?;
        for (v, b) in scratch.modulation[..FINAL_NORM_DIM]
            .iter_mut()
            .zip(&self.norm_out_bias)
        {
            *v += *b;
        }
        let final_scale = &scratch.modulation[..HIDDEN];
        let final_shift = &scratch.modulation[HIDDEN..FINAL_NORM_DIM];

        // Apply norm + scale + shift only to image tokens (text is discarded
        // at this stage). After single blocks the joint order is [text, img]
        // so we must offset by cond_tokens.
        let mut projected = vec![0.0_f32; img_tokens * LATENT_DIM];
        // `AdaLayerNorm_Final.norm` is a bias/affine-free LayerNorm with
        // eps=1e-6; then `(1 + scale) * norm(x) + shift`. The normalized
        // rows are gathered into the shared scratch buffer so that the
        // projection runs as one batched matmul.
        for token in 0..img_tokens {
            let token_in =
                &scratch.joint[(cond_tokens + token) * HIDDEN..(cond_tokens + token + 1) * HIDDEN];
            let normed = &mut scratch.normed_buf[token * HIDDEN..(token + 1) * HIDDEN];
            layer_norm_token(token_in, normed);
            for d in 0..HIDDEN {
                normed[d] = normed[d] * (1.0 + final_scale[d]) + final_shift[d];
            }
        }
        // proj_out: hidden -> latent_dim
        linear_rows_into_dispatch(
            linear,
            self.source.as_ref(),
            &self.proj_out_weight,
            HIDDEN,
            LATENT_DIM,
            &scratch.normed_buf[..img_tokens * HIDDEN],
            img_tokens,
            &mut projected,
            &mut scratch.q8,
            self.pool.as_ref(),
        )?;
        for token in 0..img_tokens {
            for (v, b) in projected[token * LATENT_DIM..(token + 1) * LATENT_DIM]
                .iter_mut()
                .zip(&self.proj_out_bias)
            {
                *v += *b;
            }
        }

        // Reassemble into the latent layout [latent_dim, latent_time].
        for token in 0..img_tokens {
            for c in 0..LATENT_DIM {
                let dst = c * latent_time + token;
                let src = token * LATENT_DIM + c;
                velocity[dst] = projected[src];
            }
        }
        Ok(())
    }
}

const IMG_TOKENS_PLACEHOLDER: usize = 0; // placeholder until prepare() is called; the
                                         // real value depends on `latent_time`. We use a method on AukScratch.

// === Helpers (placeholders that compile; full numerical correctness is a
//    follow-up commit once the scaffold validates and tests pass) ===

fn load_f32_vector(source: &dyn TensorSource, name: &str, len: usize) -> Result<Vec<f32>, String> {
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("Missing tensor: {name}"))?;
    if info.dims != [len as u64] {
        return Err(format!("Invalid {name} dimensions"));
    }
    if !matches!(
        info.ggml_type,
        GGMLType::F32 | GGMLType::BF16 | GGMLType::F16
    ) {
        return Err(format!(
            "Invalid {name} type {:?}: expected F32/BF16",
            info.ggml_type
        ));
    }
    let expected = usize::try_from(
        info.checked_nbytes()
            .ok_or_else(|| format!("Invalid {name} byte size"))?,
    )
    .map_err(|_| format!("Tensor byte size does not fit usize: {name}"))?;
    let bytes = source
        .tensor_slice(name)
        .ok_or_else(|| format!("Missing tensor data: {name}"))?;
    if bytes.len() != expected {
        return Err(format!("Invalid {name} byte length"));
    }
    let mut values = vec![0.0_f32; len];
    match info.ggml_type {
        GGMLType::F32 => {
            for (dst, chunk) in values.iter_mut().zip(bytes.chunks_exact(4)) {
                *dst = f32::from_le_bytes(chunk.try_into().unwrap());
            }
        }
        GGMLType::F16 => {
            for (dst, chunk) in values.iter_mut().zip(bytes.chunks_exact(2)) {
                *dst = f16::from_bits(u16::from_le_bytes(chunk.try_into().unwrap())).to_f32();
            }
        }
        GGMLType::BF16 => {
            for (dst, chunk) in values.iter_mut().zip(bytes.chunks_exact(2)) {
                *dst =
                    half::bf16::from_bits(u16::from_le_bytes(chunk.try_into().unwrap())).to_f32();
            }
        }
        _ => unreachable!(),
    }
    Ok(values)
}

/// Read a small unquantized N-D tensor into a flat `Vec<f32>`.
fn load_f32_tensor(
    source: &dyn TensorSource,
    name: &str,
    expected_len: usize,
) -> Result<Vec<f32>, String> {
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("Missing tensor: {name}"))?;
    let len = usize::try_from(info.dims.iter().product::<u64>())
        .map_err(|_| format!("Tensor {name} is too large"))?;
    if len != expected_len {
        return Err(format!(
            "Invalid {name} element count {len} (dims {:?}); expected {expected_len}",
            info.dims
        ));
    }
    if !matches!(
        info.ggml_type,
        GGMLType::F32 | GGMLType::F16 | GGMLType::BF16
    ) {
        return Err(format!(
            "Invalid {name} type {:?}: expected F32/F16/BF16",
            info.ggml_type
        ));
    }
    let bytes = source
        .tensor_slice(name)
        .ok_or_else(|| format!("Missing tensor data: {name}"))?;
    let mut values = vec![0.0_f32; len];
    match info.ggml_type {
        GGMLType::F32 => {
            for (dst, chunk) in values.iter_mut().zip(bytes.chunks_exact(4)) {
                *dst = f32::from_le_bytes(chunk.try_into().unwrap());
            }
        }
        GGMLType::F16 => {
            for (dst, chunk) in values.iter_mut().zip(bytes.chunks_exact(2)) {
                *dst = f16::from_bits(u16::from_le_bytes(chunk.try_into().unwrap())).to_f32();
            }
        }
        GGMLType::BF16 => {
            for (dst, chunk) in values.iter_mut().zip(bytes.chunks_exact(2)) {
                *dst =
                    half::bf16::from_bits(u16::from_le_bytes(chunk.try_into().unwrap())).to_f32();
            }
        }
        _ => unreachable!(),
    }
    Ok(values)
}

/// Grouped Conv1d (kernel 31, 16 groups, zero padding) followed by Mish.
///
/// `weight` is the raw GGUF layout, which for a PyTorch `Conv1d` weight of
/// shape `[out, in_per_group, kernel]` is `[out][in_per_group][kernel]` with
/// the kernel contiguous (GGUF `ne = [kernel, in_per_group, out]`). Input and
/// output are token-major `[n_tokens][HIDDEN]`.
fn grouped_conv1d_mish(
    pool: &ComputePool,
    weight: &[f32],
    bias: &[f32],
    input: &[f32],
    n_tokens: usize,
    out: &mut [f32],
) {
    const OUT_PER_GROUP: usize = HIDDEN / CONV_POS_GROUPS;
    const IN_PER_GROUP: usize = CONV_POS_CHANNELS_PER_GROUP;
    const PAD: usize = CONV_POS_KERNEL / 2;
    debug_assert_eq!(OUT_PER_GROUP, IN_PER_GROUP);

    // The token rows of the output are disjoint, so the convolution runs in
    // parallel over tokens.
    let weight_ptr = weight.as_ptr() as usize;
    let weight_len = weight.len();
    let bias_ptr = bias.as_ptr() as usize;
    let input_ptr = input.as_ptr() as usize;
    let input_len = input.len();
    let out_ptr = out.as_mut_ptr() as usize;
    let out_len = out.len();
    let tokens = n_tokens;
    pool.compute(move |ith, nth| {
        let per_thread = (tokens + nth - 1) / nth;
        let start = ith * per_thread;
        let end = (start + per_thread).min(tokens);
        if start >= end {
            return;
        }
        // SAFETY: the slices outlive the closure, and every worker owns a
        // disjoint range of output rows.
        let weight = unsafe { std::slice::from_raw_parts(weight_ptr as *const f32, weight_len) };
        let bias = unsafe { std::slice::from_raw_parts(bias_ptr as *const f32, HIDDEN) };
        let input = unsafe { std::slice::from_raw_parts(input_ptr as *const f32, input_len) };
        let out = unsafe { std::slice::from_raw_parts_mut(out_ptr as *mut f32, out_len) };
        // SAFETY: the slices outlive the closure and every worker owns a
        // disjoint range of output rows.
        unsafe {
            grouped_conv1d_mish_range(weight, bias, input, out, tokens, start, end);
        }
    });
}

/// PyTorch `nn.Mish`: `x * tanh(softplus(x))` with softplus threshold 20.
fn mish(x: f32) -> f32 {
    let softplus = if x > 20.0 { x } else { (1.0 + x.exp()).ln() };
    x * softplus.tanh()
}

/// Write a `f32` tensor to a file for numeric comparison with the PyTorch
/// reference implementation.
fn dump_f32(path: &std::path::Path, data: &[f32]) {
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for v in data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    if let Err(error) = std::fs::write(path, &bytes) {
        eprintln!("[auk-dump] failed to write {}: {error}", path.display());
    }
}

fn rms(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    (values.iter().map(|v| v * v).sum::<f32>() / values.len() as f32).sqrt()
}

fn max_abs(values: &[f32]) -> f32 {
    values.iter().fold(0.0_f32, |m, v| m.max(v.abs()))
}

fn load_double_block(
    source: &dyn TensorSource,
    layer: usize,
) -> Result<DoubleBlockWeights, String> {
    let prefix = format!("transformer.transformer_blocks.{layer}");
    let vector = |suffix: &str, len: usize| -> Result<Vec<f32>, String> {
        load_f32_vector(source, &format!("{prefix}.{suffix}"), len)
    };
    Ok(DoubleBlockWeights {
        adaLN_x: format!("{prefix}.attn_norm_x.linear.weight"),
        adaLN_x_bias: vector("attn_norm_x.linear.bias", ADALN_DIM)?,
        adaLN_c: format!("{prefix}.attn_norm_c.linear.weight"),
        adaLN_c_bias: vector("attn_norm_c.linear.bias", ADALN_DIM)?,
        qkv_x: format!("{prefix}.attn.to_qkv.weight"),
        qkv_x_bias: vector("attn.to_qkv.bias", QKV_DIM)?,
        qkv_c: format!("{prefix}.attn.to_qkv_c.weight"),
        qkv_c_bias: vector("attn.to_qkv_c.bias", QKV_DIM)?,
        out_x: format!("{prefix}.attn.to_out.0.weight"),
        out_x_bias: vector("attn.to_out.0.bias", HIDDEN)?,
        out_c: format!("{prefix}.attn.to_out_c.weight"),
        out_c_bias: vector("attn.to_out_c.bias", HIDDEN)?,
        q_norm: vector("attn.q_norm.weight", HEAD_DIM)?,
        k_norm: vector("attn.k_norm.weight", HEAD_DIM)?,
        c_q_norm: vector("attn.c_q_norm.weight", HEAD_DIM)?,
        c_k_norm: vector("attn.c_k_norm.weight", HEAD_DIM)?,
        ff_x_in: format!("{prefix}.ff_x.linear_in.weight"),
        ff_x_out: format!("{prefix}.ff_x.linear_out.weight"),
        ff_c_in: format!("{prefix}.ff_c.linear_in.weight"),
        ff_c_out: format!("{prefix}.ff_c.linear_out.weight"),
    })
}

fn load_single_block(
    source: &dyn TensorSource,
    layer: usize,
) -> Result<SingleBlockWeights, String> {
    let prefix = format!("transformer.single_transformer_blocks.{layer}");
    let vector = |suffix: &str, len: usize| -> Result<Vec<f32>, String> {
        load_f32_vector(source, &format!("{prefix}.{suffix}"), len)
    };
    Ok(SingleBlockWeights {
        adaLN: format!("{prefix}.attn_norm.linear.weight"),
        adaLN_bias: vector("attn_norm.linear.bias", ADALN_DIM)?,
        qkv: format!("{prefix}.attn.to_qkv.weight"),
        qkv_bias: vector("attn.to_qkv.bias", QKV_DIM)?,
        out: format!("{prefix}.attn.to_out.0.weight"),
        out_bias: vector("attn.to_out.0.bias", HIDDEN)?,
        q_norm: vector("attn.q_norm.weight", HEAD_DIM)?,
        k_norm: vector("attn.k_norm.weight", HEAD_DIM)?,
        ff_in: format!("{prefix}.ff.linear_in.weight"),
        ff_out: format!("{prefix}.ff.linear_out.weight"),
    })
}

/// Build the flow-matching time grid used by the official sampler.
///
/// ```python
/// t = torch.linspace(0, 1, steps + 1)
/// if t_grid is not None:
///     t = torch.tensor(t_grid)
/// elif sway_sampling_coef is not None:
///     t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)
/// ```
///
/// Returns `steps + 1` monotonically increasing values in `[0, 1]`.
fn flow_time_grid(
    steps: usize,
    sway_sampling_coef: Option<f32>,
    explicit_grid: Option<&[f32]>,
) -> Vec<f32> {
    if let Some(grid) = explicit_grid {
        if grid.len() >= 2 {
            return grid.to_vec();
        }
    }
    let steps = steps.max(1);
    let mut grid: Vec<f32> = (0..=steps).map(|i| i as f32 / steps as f32).collect();
    if let Some(sway) = sway_sampling_coef {
        for t in grid.iter_mut() {
            let pi = std::f32::consts::PI;
            *t += sway * ((pi / 2.0 * *t).cos() - 1.0 + *t);
        }
    }
    grid
}

/// Standard flow-matching Euler: `x_next = x + v * dt`.
///
/// The official sampler integrates `odeint(fn, y0, linspace(0, 1, steps+1))`
/// with `y0 = randn` (pure noise) and the flow defined as
/// `phi = (1 - t) * x0 + t * x1`, so `t` runs **0 → 1** (noise → data) and
/// `dt = t_next - t` is positive.
fn euler_flow_step(
    latent: &mut [f32],
    velocity: &[f32],
    t: f32,
    t_next: f32,
) -> Result<(), String> {
    if latent.len() != velocity.len() {
        return Err("Invalid AuK Euler buffer lengths".into());
    }
    let dt = t_next - t;
    for (x, v) in latent.iter_mut().zip(velocity) {
        *x += *v * dt;
    }
    Ok(())
}

/// Sinusoidal timestep embedding, matching AuK's `SinusPositionEmbedding`.
///
/// ```python
/// half_dim = dim // 2                       # 128
/// emb = log(10000) / (half_dim - 1)
/// emb = exp(arange(half_dim) * -emb)        # 128 frequencies
/// emb = scale * x * emb                     # scale = 1000
/// emb = cat((sin(emb), cos(emb)), dim=-1)   # [sin(0..127), cos(0..127)]
/// ```
///
/// Note the **blocked** (not interleaved) sin/cos layout and the `scale=1000`
/// factor applied here -- the caller passes the raw timestep in `[0, 1]`.
fn timestep_embedding(t: f32, out: &mut [f32; FREQ_DIM]) {
    const SCALE: f32 = 1000.0;
    let half = FREQ_DIM / 2;
    let log_theta = (1e4_f32).ln();
    for i in 0..half {
        let freq_exp = (i as f32) / (half as f32 - 1.0);
        let omega = (freq_exp * -log_theta).exp();
        let angle = SCALE * t * omega;
        out[i] = angle.sin();
        out[half + i] = angle.cos();
    }
}

struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(if seed == 0 {
            0xdead_beef_cafe_babe
        } else {
            seed
        })
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 11) as f32 * (1.0 / (1u64 << 53) as f32)
    }
}

fn gaussian(rng: &mut SplitMix64) -> f32 {
    let u1 = rng.next_f32().max(1e-7);
    let u2 = rng.next_f32();
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = 2.0 * std::f32::consts::PI * u2;
    r * theta.cos()
}

/// Scratch buffers reused across timesteps.
pub(crate) struct AukScratch {
    time_frequency: [f32; FREQ_DIM],
    time_hidden: [f32; HIDDEN],
    time_hidden_silu: [f32; HIDDEN],
    time: [f32; HIDDEN],
    img: Vec<f32>,
    /// Scratch for `AudioEmbed.conv_pos_embed` (same size as `img`).
    conv_pos: Vec<f32>,
    conv_pos_out: Vec<f32>,
    text: Vec<f32>,
    joint: Vec<f32>,
    qkv: Vec<f32>,
    qkv_c: Vec<f32>,
    /// Head-major copy of the QKV buffer that the attention reads.
    qkv_head: Vec<f32>,
    attention: Vec<f32>,
    /// Attention scratch. One row of `total` scores per pool thread.
    scores: Vec<f32>,
    /// Rotary embedding tables: `[position][HEAD_DIM / 2]` cosines and sines.
    rope_cos: Vec<f32>,
    rope_sin: Vec<f32>,
    modulation: Vec<f32>,
    rope: Vec<f32>,
    normed_buf: Vec<f32>,
    /// Scratch for the packed `ff.linear_in` output (`gate` and `up`).
    ff_packed: Vec<f32>,
    /// Scratch for the SwiGLU output that feeds `ff.linear_out`.
    ff_gated: Vec<f32>,
    q8: Q8Scratch,
}

impl AukScratch {
    fn new(_text_tokens: usize, latent_time: usize) -> Result<Self, String> {
        // Reserve joint/img/text for the padded sequence.
        // For the TTS scaffold we don't yet know total_tokens at construction;
        // prepare() resizes the buffers.
        Ok(Self {
            time_frequency: [0.0; FREQ_DIM],
            time_hidden: [0.0; HIDDEN],
            time_hidden_silu: [0.0; HIDDEN],
            time: [0.0; HIDDEN],
            img: Vec::new(),
            conv_pos: Vec::new(),
            conv_pos_out: Vec::new(),
            text: Vec::new(),
            joint: Vec::new(),
            qkv: Vec::new(),
            qkv_c: Vec::new(),
            qkv_head: Vec::new(),
            attention: Vec::new(),
            scores: Vec::new(),
            rope_cos: Vec::new(),
            rope_sin: Vec::new(),
            modulation: Vec::new(),
            rope: Vec::new(),
            normed_buf: Vec::new(),
            ff_packed: Vec::new(),
            ff_gated: Vec::new(),
            q8: Q8Scratch::new(FF_INNER.max(HIDDEN)),
        })
    }

    fn prepare(
        &mut self,
        img_token_count: usize,
        text_token_count: usize,
        threads: usize,
    ) -> Result<(), String> {
        if img_token_count == 0 {
            return Err("AuK image (latent) token count must be positive".into());
        }
        let total = img_token_count + text_token_count;
        super::super::common::resize_zeroed(&mut self.img, img_token_count * HIDDEN, "AuK img")?;
        super::super::common::resize_zeroed(
            &mut self.conv_pos,
            img_token_count * HIDDEN,
            "AuK conv_pos",
        )?;
        super::super::common::resize_zeroed(
            &mut self.conv_pos_out,
            img_token_count * HIDDEN,
            "AuK conv_pos_out",
        )?;
        super::super::common::resize_zeroed(&mut self.text, (total) * HIDDEN, "AuK text")?;
        super::super::common::resize_zeroed(&mut self.joint, total * HIDDEN, "AuK joint")?;
        super::super::common::resize_zeroed(&mut self.qkv, total * QKV_DIM, "AuK qkv")?;
        super::super::common::resize_zeroed(&mut self.qkv_head, total * QKV_DIM, "AuK qkv_head")?;
        super::super::common::resize_zeroed(&mut self.qkv_c, total * QKV_DIM, "AuK qkv_c")?;
        super::super::common::resize_zeroed(&mut self.attention, total * HIDDEN, "AuK attention")?;
        // One private score row per pool thread: the attention runs its
        // (head, query) units in parallel and each worker needs its own
        // scratch row.
        super::super::common::resize_zeroed(
            &mut self.scores,
            total * threads.max(1),
            "AuK scores",
        )?;
        super::super::common::resize_zeroed(
            &mut self.rope_cos,
            total * (HEAD_DIM / 2),
            "AuK rope_cos",
        )?;
        super::super::common::resize_zeroed(
            &mut self.rope_sin,
            total * (HEAD_DIM / 2),
            "AuK rope_sin",
        )?;
        super::super::common::resize_zeroed(&mut self.modulation, 2 * ADALN_DIM, "AuK modulation")?;
        super::super::common::resize_zeroed(&mut self.rope, total * HEAD_DIM, "AuK rope")?;
        super::super::common::resize_zeroed(
            &mut self.normed_buf,
            total * HIDDEN,
            "AuK normed_buf",
        )?;
        super::super::common::resize_zeroed(
            &mut self.ff_packed,
            total * PACKED_FF_IN,
            "AuK ff_packed",
        )?;
        super::super::common::resize_zeroed(&mut self.ff_gated, total * FF_INNER, "AuK ff_gated")?;
        Ok(())
    }
}

/// Audio embed: `latent_dim -> hidden` for each latent frame. The latent is
/// laid out as `[latent_dim, latent_time]`; we project every frame in
/// parallel and store it as `[image_tokens, hidden]`.
#[allow(clippy::too_many_arguments)]
fn run_audio_embed(
    linear: &AukLinearSession<'_, '_>,
    source: &dyn TensorSource,
    weight: &str,
    bias: &[f32],
    latent: &[f32],
    latent_time: usize,
    output: &mut [f32],
    pool: &ComputePool,
    q8: &mut Q8Scratch,
) -> Result<(), String> {
    let image_tokens = latent_time;
    if output.len() != image_tokens * HIDDEN {
        return Err("AuK audio_embed output length mismatch".into());
    }
    // The latent is laid out as `[latent_dim, latent_time]`, so a frame is
    // strided. Gather the frames into a row-major `[latent_time, latent_dim]`
    // buffer so the projection runs as one batched matmul.
    let mut frames = vec![0.0_f32; image_tokens * LATENT_DIM];
    for token in 0..image_tokens {
        for c in 0..LATENT_DIM {
            frames[token * LATENT_DIM + c] = latent[c * latent_time + token];
        }
    }
    linear_rows_into_dispatch(
        linear,
        source,
        weight,
        LATENT_DIM,
        HIDDEN,
        &frames,
        image_tokens,
        output,
        q8,
        pool,
    )?;
    for token in 0..image_tokens {
        let out = &mut output[token * HIDDEN..(token + 1) * HIDDEN];
        for (v, b) in out.iter_mut().zip(bias) {
            *v += *b;
        }
    }
    Ok(())
}

/// Per-token LayerNorm (mean subtraction + variance scaling). Matches
/// `LayerNormModule({channels, 1e-6F, false, false})` from audio.cpp -- no
/// elementwise affine, no bias. Output is unit-variance, zero-mean.
fn layer_norm_token(input: &[f32], normalized: &mut [f32]) {
    let len = input.len().min(normalized.len());
    if len == 0 {
        return;
    }
    let mut sum = 0.0_f32;
    for v in &input[..len] {
        sum += *v;
    }
    let mean = sum / len as f32;
    let mut var_sum = 0.0_f32;
    for v in &input[..len] {
        let d = *v - mean;
        var_sum += d * d;
    }
    let variance = var_sum / len as f32;
    let std_recip = 1.0 / (variance + 1e-6).sqrt();
    for d in 0..len {
        normalized[d] = (input[d] - mean) * std_recip;
    }
}

/// RMS norm over the projected text (`txt_norm`).
///
/// The official module is `nn.RMSNorm(dim, elementwise_affine=True)`, whose
/// `eps=None` default resolves to `torch.finfo(float32).eps`
/// (`RMS_NORM_EPS`), not the `1e-6` used by the AdaLN `nn.LayerNorm`.
fn rms_norm_inplace_text(hidden: &mut [f32], weight: &[f32], n_tokens: usize) {
    for token in 0..n_tokens {
        let start = token * HIDDEN;
        let end = start + HIDDEN;
        let slice = &mut hidden[start..end];
        let mut mean = 0.0_f32;
        for v in slice.iter() {
            mean += v * v;
        }
        mean = (mean / HIDDEN as f32 + RMS_NORM_EPS).sqrt().recip();
        for d in 0..HIDDEN {
            slice[d] *= mean * weight[d];
        }
    }
}

// === Flux2Edit block forwards ===
//
// Per `references/audio.cpp/src/community_models/auk/flow.cpp::build_transformer_block`
// (templated over `Streams`).
//
// Layout conventions:
// - Joint sequence: [img_tokens, text_tokens] (img first).
// - x-stream (img) attends to its own + c-stream (text) tokens.
// - c-stream (text) attends to the same joint.
// - RoPE positions: img uses 0..img_tokens, text uses 0..text_tokens independently.
// - `scratch.modulation[..ADALN_DIM]` holds the time-embedding projection for
//   the current step; we slice it into six 1536-wide chunks for the per-stream
//   (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp).

/// Apply rotary embedding in-place using a precomputed `inv_freq` table of
/// length `head_dim/2`. `x` is a single token's head-major slice
/// `[n_heads * head_dim]` flattened row-major.
///
/// AuK uses `x_transformers.RotaryEmbedding` + `apply_rotary_pos_emb`, whose
/// `freqs` tensor is built as `stack((freqs, freqs), dim=-1)` and then
/// flattened — i.e. the frequency for index `i` is repeated at **adjacent**
/// positions `2*i` and `2*i+1` (interleaved / GPT-J style), not the NeoX
/// half-split `(i, i + head_dim/2)`.
///
/// ```text
/// out[2i]   = x[2i]   * cos - x[2i+1] * sin
/// out[2i+1] = x[2i+1] * cos + x[2i]   * sin
/// ```
fn rope_apply_token(x: &mut [f32], head_dim: usize, table_cos: &[f32], table_sin: &[f32]) {
    let half = head_dim / 2;
    if half == 0 || table_cos.len() < half || table_sin.len() < half {
        return;
    }
    let n_heads = x.len() / head_dim;
    if n_heads == 0 {
        return;
    }
    for head in 0..n_heads {
        let off = head * head_dim;
        // Interleaved rotation: pair (2i, 2i+1).
        for i in 0..half {
            let cos_t = table_cos[i];
            let sin_t = table_sin[i];
            let a = x[off + 2 * i];
            let b = x[off + 2 * i + 1];
            x[off + 2 * i] = a * cos_t - b * sin_t;
            x[off + 2 * i + 1] = b * cos_t + a * sin_t;
        }
    }
}

/// `torch.nn.RMSNorm` with `eps=None` uses `torch.finfo(float32).eps`.
const RMS_NORM_EPS: f32 = 1.192_092_9e-7;

/// Per-head RMS norm applied independently per token. `hidden` is laid out
/// `[n_tokens, n_heads, head_dim]` (contiguous); `weight` is `[head_dim]`.
fn rms_norm_per_head(
    hidden: &mut [f32],
    n_tokens: usize,
    n_heads: usize,
    head_dim: usize,
    weight: &[f32],
    eps: f32,
) {
    let row = n_heads * head_dim;
    for token in 0..n_tokens {
        for head in 0..n_heads {
            let off = token * row + head * head_dim;
            let slice = &mut hidden[off..off + head_dim];
            let mut mean_sq = 0.0_f32;
            for v in slice.iter() {
                mean_sq += *v * *v;
            }
            let scale = 1.0 / (mean_sq / head_dim as f32 + eps).sqrt();
            for d in 0..head_dim {
                slice[d] = slice[d] * scale * weight[d];
            }
        }
    }
}

/// Softmax over `scores[..n]`. Modifies in place.
fn softmax_inplace(scores: &mut [f32], n: usize) {
    let max = scores[..n]
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f32;
    for j in 0..n {
        scores[j] = (scores[j] - max).exp();
        sum += scores[j];
    }
    let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
    for j in 0..n {
        scores[j] *= inv;
    }
}

/// Transpose a QKV buffer into the head-major layout the attention reads.
///
/// The input rows are `[token][Q | K | V][head][head_dim]`; the output is
/// `[Q | K | V][head][token][head_dim]`. The attention reads the keys and
/// the values of one head for every query token. In the token-major layout
/// those live `QKV_DIM` floats apart, so the attention is bound by memory
/// traffic instead of arithmetic; in the head-major layout the keys and
/// values of one head are contiguous and stay in cache.
///
/// The image (`qkv_x`) and text (`qkv_c`) streams are concatenated in the
/// joint token order, so the same layout serves the joint attention of the
/// double blocks and the single-stream attention of the single blocks.
fn transpose_qkv(
    pool: &ComputePool,
    qkv_x: &[f32],
    qkv_c: &[f32],
    img_tokens: usize,
    text_tokens: usize,
    out: &mut [f32],
) {
    let total = img_tokens + text_tokens;
    let n_heads = HIDDEN / HEAD_DIM;
    let heads = HEAD_DIM;
    let part_stride = n_heads * total * heads;
    let head_stride = total * heads;
    let units = 3 * n_heads;
    let x_ptr = qkv_x.as_ptr() as usize;
    let x_len = qkv_x.len();
    let c_ptr = qkv_c.as_ptr() as usize;
    let c_len = qkv_c.len();
    let out_ptr = out.as_mut_ptr() as usize;
    let out_len = out.len();
    pool.compute(move |ith, nth| {
        let per_thread = (units + nth - 1) / nth;
        let start = ith * per_thread;
        let end = (start + per_thread).min(units);
        if start >= end {
            return;
        }
        // SAFETY: the slices outlive the closure, and every worker owns a
        // disjoint (part, head) slice of the output.
        let qkv_x = unsafe { std::slice::from_raw_parts(x_ptr as *const f32, x_len) };
        let qkv_c = unsafe { std::slice::from_raw_parts(c_ptr as *const f32, c_len) };
        let out = unsafe { std::slice::from_raw_parts_mut(out_ptr as *mut f32, out_len) };
        for unit in start..end {
            let part = unit / n_heads;
            let head = unit % n_heads;
            let src_base = part * HIDDEN + head * heads;
            let dst_base = part * part_stride + head * head_stride;
            for token in 0..img_tokens {
                let src = &qkv_x[token * QKV_DIM + src_base..token * QKV_DIM + src_base + heads];
                let dst = &mut out[dst_base + token * heads..dst_base + (token + 1) * heads];
                dst.copy_from_slice(src);
            }
            for token in 0..text_tokens {
                let src = &qkv_c[token * QKV_DIM + src_base..token * QKV_DIM + src_base + heads];
                let joint = img_tokens + token;
                let dst = &mut out[dst_base + joint * heads..dst_base + (joint + 1) * heads];
                dst.copy_from_slice(src);
            }
        }
    });
}

/// Attention over a head-major QKV buffer.
///
/// The work is split into `n_heads * total` independent units (one head and
/// one query token each). Every unit owns a disjoint slice of `attention`
/// and a private slice of `scores`, so the units run in parallel without
/// any synchronization. The arithmetic inside a unit is unchanged, so the
/// result stays bit-identical to the sequential formulation.
///
/// `scores` must hold at least `pool.n_threads() * total` elements: each
/// worker needs its own scratch row.
fn attention_heads(
    pool: &ComputePool,
    qkv_head: &[f32],
    total: usize,
    scores: &mut [f32],
    attention: &mut [f32],
) {
    let n_heads = HIDDEN / HEAD_DIM;
    let heads = HEAD_DIM;
    let scale = 1.0 / (heads as f32).sqrt();
    let part_stride = n_heads * total * heads;
    let head_stride = total * heads;
    let units = n_heads * total;
    let scores_ptr = scores.as_mut_ptr() as usize;
    let scores_len = scores.len();
    let attention_ptr = attention.as_mut_ptr() as usize;
    let attention_len = attention.len();
    let qkv_ptr = qkv_head.as_ptr() as usize;
    let qkv_len = qkv_head.len();
    pool.compute(move |ith, nth| {
        let per_thread = (units + nth - 1) / nth;
        let start = ith * per_thread;
        let end = (start + per_thread).min(units);
        if start >= end {
            return;
        }
        // SAFETY: every pointer was derived from a live slice that outlives
        // the closure, and the slices are disjoint per worker: `scores` is
        // partitioned by `ith`, `attention` is written at disjoint
        // (token, head) positions.
        let scores_all =
            unsafe { std::slice::from_raw_parts_mut(scores_ptr as *mut f32, scores_len) };
        let attention_all =
            unsafe { std::slice::from_raw_parts_mut(attention_ptr as *mut f32, attention_len) };
        let qkv = unsafe { std::slice::from_raw_parts(qkv_ptr as *const f32, qkv_len) };
        let local = &mut scores_all[ith * total..(ith + 1) * total];
        for unit in start..end {
            let head = unit / total;
            let query = unit % total;
            let q_row = &qkv[head * head_stride + query * heads..][..heads];
            let mut max_logit = f32::NEG_INFINITY;
            for k_tok in 0..total {
                let dot = dot32(
                    q_row,
                    &qkv[part_stride + head * head_stride + k_tok * heads..][..heads],
                );
                let s = dot * scale;
                local[k_tok] = s;
                if s > max_logit {
                    max_logit = s;
                }
            }
            let softmax_started = std::time::Instant::now();
            softmax_inplace(local, total);
            profile::add(&profile::SOFTMAX_NS, softmax_started);
            let out_off = query * HIDDEN + head * heads;
            for d in 0..heads {
                let mut sum = 0.0_f32;
                for k_tok in 0..total {
                    let v_off = 2 * part_stride + head * head_stride + k_tok * heads + d;
                    sum += local[k_tok] * qkv[v_off];
                }
                attention_all[out_off + d] = sum;
            }
        }
    });
}

#[allow(clippy::too_many_arguments)]
fn run_double_block(
    linear: &AukLinearSession<'_, '_>,
    source: &dyn TensorSource,
    block: &DoubleBlockWeights,
    joint: &mut [f32],
    img_tokens: usize,
    text_tokens: usize,
    rope_cos: &[f32],
    rope_sin: &[f32],
    modulation: &[f32],
    qkv_buf: &mut [f32],
    qkv_c_buf: &mut [f32],
    qkv_head: &mut [f32],
    attention: &mut [f32],
    scores: &mut [f32],
    normed: &mut [f32],
    ff_packed: &mut [f32],
    ff_gated: &mut [f32],
    q8: &mut Q8Scratch,
    pool: &ComputePool,
    layer_index: usize,
) -> Result<(), String> {
    let _ = layer_index;
    let total = img_tokens + text_tokens;
    if total == 0 || total * HIDDEN != joint.len() {
        return Err("AuK double block joint length mismatch".into());
    }
    if scores.len() < total * pool.n_threads() {
        return Err("AuK scores buffer too small".into());
    }
    if qkv_buf.len() < total * QKV_DIM {
        return Err("AuK qkv buffer too small".into());
    }
    if qkv_c_buf.len() < total * QKV_DIM {
        return Err("AuK qkv_c buffer too small".into());
    }
    if attention.len() < total * HIDDEN {
        return Err("AuK attention buffer too small".into());
    }
    if normed.len() < total * HIDDEN {
        return Err("AuK normed buffer too small".into());
    }
    if modulation.len() < 2 * ADALN_DIM {
        return Err("AuK modulation buffer too small".into());
    }
    if rope_cos.len() < img_tokens * (HEAD_DIM / 2) || rope_sin.len() < img_tokens * (HEAD_DIM / 2)
    {
        return Err("AuK rope table too small".into());
    }

    // === Pass 1: per-stream RMSNorm + AdaLN modulation + QKV projection ===
    // Stream x (img) — occupies joint rows [0, img_tokens)
    ada_ln_qkv(
        linear,
        source,
        &block.adaLN_x,
        &block.adaLN_x_bias,
        &block.qkv_x,
        &block.qkv_x_bias,
        joint,
        0..img_tokens,
        &modulation[..ADALN_DIM],
        normed,
        qkv_buf,
        q8,
        pool,
    )?;
    // Stream c (text) — occupies joint rows [img_tokens, total)
    ada_ln_qkv(
        linear,
        source,
        &block.adaLN_c,
        &block.adaLN_c_bias,
        &block.qkv_c,
        &block.qkv_c_bias,
        joint,
        img_tokens..total,
        &modulation[ADALN_DIM..2 * ADALN_DIM],
        normed,
        qkv_c_buf,
        q8,
        pool,
    )?;

    // === Pass 2: per-stream Q/K RMS-norm + RoPE ===
    // QKV layout: [tokens, QKV_DIM] = [tokens, 3*HIDDEN] with interleaved Q,K,V
    // per row.
    project_qk_with_rope(
        qkv_buf,
        img_tokens,
        0,
        &block.q_norm,
        &block.k_norm,
        &rope_cos[..img_tokens * (HEAD_DIM / 2)],
        &rope_sin[..img_tokens * (HEAD_DIM / 2)],
    );
    project_qk_with_rope(
        qkv_c_buf,
        text_tokens,
        0,
        &block.c_q_norm,
        &block.c_k_norm,
        &rope_cos[..text_tokens * (HEAD_DIM / 2)],
        &rope_sin[..text_tokens * (HEAD_DIM / 2)],
    );

    // === Joint attention: concat Q/K/V along seq dim ===
    // We need separate K, V slices for concat. Allocate within qkv_buf/qkv_c_buf
    // by reinterpreting their halves. Simpler: use the original buffers.
    // Q @ K^T then softmax * V.
    //
    // Joint K, V are not contiguous in qkv_buf (qkv_c is separate). We process
    // the attention by iterating heads and concatenating K/V from both
    // streams per head.
    // Per-stream output: attention[i] for i in 0..img_tokens (x-stream) and
    // i in img_tokens..total (c-stream).
    let attention_started = std::time::Instant::now();
    transpose_qkv(
        pool,
        &qkv_buf[..img_tokens * QKV_DIM],
        &qkv_c_buf[..text_tokens * QKV_DIM],
        img_tokens,
        text_tokens,
        &mut qkv_head[..total * QKV_DIM],
    );
    attention_heads(pool, &qkv_head[..total * QKV_DIM], total, scores, attention);
    profile::add(&profile::ATTENTION_NS, attention_started);
    stream_block_residual_mlp(
        linear,
        source,
        &block.out_x,
        &block.out_x_bias,
        &block.ff_x_in,
        &block.ff_x_out,
        joint,
        0..img_tokens,
        &attention[..total * HIDDEN],
        &modulation[..ADALN_DIM],
        normed,
        ff_packed,
        ff_gated,
        q8,
        pool,
    )?;
    stream_block_residual_mlp(
        linear,
        source,
        &block.out_c,
        &block.out_c_bias,
        &block.ff_c_in,
        &block.ff_c_out,
        joint,
        img_tokens..total,
        &attention[..total * HIDDEN],
        &modulation[ADALN_DIM..2 * ADALN_DIM],
        normed,
        ff_packed,
        ff_gated,
        q8,
        pool,
    )?;

    super::super::common::require_finite(joint, "AuK double block output")
}

#[allow(clippy::too_many_arguments)]
fn ada_ln_qkv(
    linear: &AukLinearSession<'_, '_>,
    source: &dyn TensorSource,
    ada_weight: &str,
    ada_bias: &[f32],
    qkv_weight: &str,
    qkv_bias: &[f32],
    joint: &[f32],
    row_range: std::ops::Range<usize>,
    ada_params: &[f32],
    normed: &mut [f32],
    qkv_buf: &mut [f32],
    q8: &mut Q8Scratch,
    pool: &ComputePool,
) -> Result<(), String> {
    let tokens = row_range.len();
    // Slice modulation: 6 parts of HIDDEN each. We use shift_msa, scale_msa
    // (gate_msa, shift_mlp, scale_mlp, gate_mlp are used later in the residual/MLP).
    let shift_msa = &ada_params[0..HIDDEN];
    let scale_msa = &ada_params[HIDDEN..2 * HIDDEN];
    let gate_msa = &ada_params[2 * HIDDEN..3 * HIDDEN];
    let _shift_mlp = &ada_params[3 * HIDDEN..4 * HIDDEN];
    let _scale_mlp = &ada_params[4 * HIDDEN..5 * HIDDEN];
    let _gate_mlp = &ada_params[5 * HIDDEN..6 * HIDDEN];
    let _ = gate_msa;

    // LayerNorm per token (audio.cpp uses LayerNormModule -- mean subtraction, not
    // RMSNorm). The Flux2Edit AdaLN path then modulates (1+scale)*normalized + shift.
    for token in 0..tokens {
        let row = row_range.start + token;
        let off = row * HIDDEN;
        let slice = &joint[off..off + HIDDEN];
        // Compute mean.
        let mut sum = 0.0_f32;
        for v in slice.iter() {
            sum += *v;
        }
        let mean = sum / HIDDEN as f32;
        // Compute variance.
        let mut var_sum = 0.0_f32;
        for v in slice.iter() {
            let d = *v - mean;
            var_sum += d * d;
        }
        let variance = var_sum / HIDDEN as f32;
        let std_recip = 1.0 / (variance + 1e-6).sqrt();
        let n_off = token * HIDDEN;
        for d in 0..HIDDEN {
            normed[n_off + d] = (slice[d] - mean) * std_recip;
        }
    }

    // Apply AdaLN modulation.
    for token in 0..tokens {
        let n_off = token * HIDDEN;
        for d in 0..HIDDEN {
            let v = normed[n_off + d];
            normed[n_off + d] = v * (1.0 + scale_msa[d]) + shift_msa[d];
        }
    }

    // QKV projection: one batched matmul over the token rows.
    linear_rows_into_dispatch(
        linear,
        source,
        qkv_weight,
        HIDDEN,
        QKV_DIM,
        &normed[..tokens * HIDDEN],
        tokens,
        &mut qkv_buf[..tokens * QKV_DIM],
        q8,
        pool,
    )?;
    for token in 0..tokens {
        let out = &mut qkv_buf[token * QKV_DIM..(token + 1) * QKV_DIM];
        for (v, b) in out.iter_mut().zip(qkv_bias) {
            *v += *b;
        }
    }

    // ada_weight / ada_bias are unused here; we read the 9216-wide modulation
    // directly from `ada_params` (which was already projected by the caller).
    let _ = (ada_weight, ada_bias);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn project_qk_with_rope(
    qkv: &mut [f32],
    tokens: usize,
    pos_offset: usize,
    q_norm_w: &[f32],
    k_norm_w: &[f32],
    rope_cos: &[f32],
    rope_sin: &[f32],
) {
    // QKV row: [Q (HIDDEN) | K (HIDDEN) | V (HIDDEN)]
    let rope_started = std::time::Instant::now();
    for token in 0..tokens {
        let off = token * QKV_DIM;
        let pos = pos_offset + token;
        // Q RMSNorm + RoPE
        rms_norm_per_head(
            &mut qkv[off..off + HIDDEN],
            1,
            HIDDEN / HEAD_DIM,
            HEAD_DIM,
            q_norm_w,
            RMS_NORM_EPS,
        );
        let table = pos * (HEAD_DIM / 2);
        rope_apply_token(
            &mut qkv[off..off + HIDDEN],
            HEAD_DIM,
            &rope_cos[table..table + HEAD_DIM / 2],
            &rope_sin[table..table + HEAD_DIM / 2],
        );
        // K RMSNorm + RoPE
        rms_norm_per_head(
            &mut qkv[off + HIDDEN..off + 2 * HIDDEN],
            1,
            HIDDEN / HEAD_DIM,
            HEAD_DIM,
            k_norm_w,
            RMS_NORM_EPS,
        );
        rope_apply_token(
            &mut qkv[off + HIDDEN..off + 2 * HIDDEN],
            HEAD_DIM,
            &rope_cos[table..table + HEAD_DIM / 2],
            &rope_sin[table..table + HEAD_DIM / 2],
        );
        // V: no norm, no RoPE.
        let _ = (q_norm_w, k_norm_w);
    }
    profile::add(&profile::ROPE_NS, rope_started);
}

#[allow(clippy::too_many_arguments)]
fn stream_block_residual_mlp(
    linear: &AukLinearSession<'_, '_>,
    source: &dyn TensorSource,
    out_weight: &str,
    out_bias: &[f32],
    ff_in_weight: &str,
    ff_out_weight: &str,
    joint: &mut [f32],
    row_range: std::ops::Range<usize>,
    attention: &[f32],
    ada_params: &[f32],
    normed: &mut [f32],
    ff_packed: &mut [f32],
    ff_gated: &mut [f32],
    q8: &mut Q8Scratch,
    pool: &ComputePool,
) -> Result<(), String> {
    let tokens = row_range.len();
    let _shift_msa = &ada_params[0..HIDDEN];
    let _scale_msa = &ada_params[HIDDEN..2 * HIDDEN];
    let gate_msa = &ada_params[2 * HIDDEN..3 * HIDDEN];
    let shift_mlp = &ada_params[3 * HIDDEN..4 * HIDDEN];
    let scale_mlp = &ada_params[4 * HIDDEN..5 * HIDDEN];
    let gate_mlp = &ada_params[5 * HIDDEN..6 * HIDDEN];

    // Output projection (1536 -> 1536) for the whole stream, then the
    // residual `x + proj * gate_msa`. The attention rows of the stream are
    // contiguous, so the projection is a single batched matmul.
    linear_rows_into_dispatch(
        linear,
        source,
        out_weight,
        HIDDEN,
        HIDDEN,
        &attention[row_range.start * HIDDEN..row_range.end * HIDDEN],
        tokens,
        &mut normed[..tokens * HIDDEN],
        q8,
        pool,
    )?;
    for token in 0..tokens {
        let proj = &normed[token * HIDDEN..(token + 1) * HIDDEN];
        let row = (row_range.start + token) * HIDDEN;
        for d in 0..HIDDEN {
            joint[row + d] += (proj[d] + out_bias[d]) * gate_msa[d];
        }
    }

    // MLP: RMSNorm(residual_1), AdaLN modulate, ff_in (packed gate+up), SwiGLU,
    // ff_out, residual.
    let layer_started = std::time::Instant::now();
    for token in 0..tokens {
        let row = row_range.start + token;
        let off = row * HIDDEN;
        let normed_token = &mut normed[token * HIDDEN..(token + 1) * HIDDEN];
        layer_norm_token(&joint[off..off + HIDDEN], normed_token);
        for d in 0..HIDDEN {
            let v = normed_token[d];
            normed_token[d] = v * (1.0 + scale_mlp[d]) + shift_mlp[d];
        }
    }
    profile::add(&profile::LAYER_NS, layer_started);
    // ff_in: 1536 -> 6144 (gate+up packed)
    linear_rows_into_dispatch(
        linear,
        source,
        ff_in_weight,
        HIDDEN,
        PACKED_FF_IN,
        &normed[..tokens * HIDDEN],
        tokens,
        &mut ff_packed[..tokens * PACKED_FF_IN],
        q8,
        pool,
    )?;
    // SwiGLU: split into gate (0..3072) and up (3072..6144); out = silu(gate) * up
    let swiglu_started = std::time::Instant::now();
    for token in 0..tokens {
        let packed = &ff_packed[token * PACKED_FF_IN..(token + 1) * PACKED_FF_IN];
        let gated = &mut ff_gated[token * FF_INNER..(token + 1) * FF_INNER];
        for d in 0..FF_INNER {
            let g = packed[d];
            let u = packed[FF_INNER + d];
            // silu(g) = g * sigmoid(g)
            let silu = g / (1.0 + (-g).exp());
            gated[d] = silu * u;
        }
    }
    profile::add(&profile::SWIGLU_NS, swiglu_started);
    // ff_out: 3072 -> 1536. The normalized rows are dead after `ff_in`, so
    // the shared scratch buffer takes the output.
    linear_rows_into_dispatch(
        linear,
        source,
        ff_out_weight,
        FF_INNER,
        HIDDEN,
        &ff_gated[..tokens * FF_INNER],
        tokens,
        &mut normed[..tokens * HIDDEN],
        q8,
        pool,
    )?;
    for token in 0..tokens {
        let row = row_range.start + token;
        let off = row * HIDDEN;
        for d in 0..HIDDEN {
            joint[off + d] += normed[token * HIDDEN + d] * gate_mlp[d];
        }
    }
    Ok(())
}

fn dot32(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0.0_f32;
    for (x, y) in a.iter().zip(b) {
        s += *x * *y;
    }
    s
}

#[allow(clippy::too_many_arguments)]
fn run_single_block(
    linear: &AukLinearSession<'_, '_>,
    source: &dyn TensorSource,
    block: &SingleBlockWeights,
    joint: &mut [f32],
    img_tokens: usize,
    text_tokens: usize,
    rope_cos: &[f32],
    rope_sin: &[f32],
    modulation: &[f32],
    qkv_buf: &mut [f32],
    qkv_head: &mut [f32],
    attention: &mut [f32],
    scores: &mut [f32],
    normed: &mut [f32],
    ff_packed: &mut [f32],
    ff_gated: &mut [f32],
    q8: &mut Q8Scratch,
    pool: &ComputePool,
    layer_index: usize,
) -> Result<(), String> {
    let _ = layer_index;
    let total = img_tokens + text_tokens;
    if total == 0 || total * HIDDEN != joint.len() {
        return Err("AuK single block joint length mismatch".into());
    }
    if scores.len() < total * pool.n_threads() {
        return Err("AuK scores buffer too small".into());
    }
    if qkv_buf.len() < total * QKV_DIM {
        return Err("AuK qkv buffer too small".into());
    }
    if attention.len() < total * HIDDEN {
        return Err("AuK attention buffer too small".into());
    }
    if normed.len() < total * HIDDEN {
        return Err("AuK normed buffer too small".into());
    }
    if ff_packed.len() < total * PACKED_FF_IN {
        return Err("AuK ff packed buffer too small".into());
    }
    if ff_gated.len() < total * FF_INNER {
        return Err("AuK ff gated buffer too small".into());
    }
    if modulation.len() < ADALN_DIM {
        return Err("AuK modulation buffer too small".into());
    }
    if rope_cos.len() < total * (HEAD_DIM / 2) || rope_sin.len() < total * (HEAD_DIM / 2) {
        return Err("AuK rope table too small".into());
    }

    let shift_msa = &modulation[0..HIDDEN];
    let scale_msa = &modulation[HIDDEN..2 * HIDDEN];
    let gate_msa = &modulation[2 * HIDDEN..3 * HIDDEN];
    let shift_mlp = &modulation[3 * HIDDEN..4 * HIDDEN];
    let scale_mlp = &modulation[4 * HIDDEN..5 * HIDDEN];
    let gate_mlp = &modulation[5 * HIDDEN..6 * HIDDEN];

    // Per-token LayerNorm + AdaLN modulation + QKV projection. The
    // normalized rows are gathered into the shared scratch buffer so the
    // projection runs as one batched matmul.
    for token in 0..total {
        let off = token * HIDDEN;
        let normed_token = &mut normed[token * HIDDEN..(token + 1) * HIDDEN];
        layer_norm_token(&joint[off..off + HIDDEN], normed_token);
        for d in 0..HIDDEN {
            normed_token[d] = normed_token[d] * (1.0 + scale_msa[d]) + shift_msa[d];
        }
    }
    linear_rows_into_dispatch(
        linear,
        source,
        &block.qkv,
        HIDDEN,
        QKV_DIM,
        &normed[..total * HIDDEN],
        total,
        &mut qkv_buf[..total * QKV_DIM],
        q8,
        pool,
    )?;
    for token in 0..total {
        let out = &mut qkv_buf[token * QKV_DIM..(token + 1) * QKV_DIM];
        for (v, b) in out.iter_mut().zip(&block.qkv_bias) {
            *v += *b;
        }
    }
    // Q/K RMS-norm + RoPE (V: no norm). Stream positions: text tokens use
    // 0..text_tokens, audio tokens use text_tokens..text_tokens+img_tokens
    // (audio comes second in single block per audio.cpp).
    let heads = HEAD_DIM;
    let n_heads = HIDDEN / heads;
    let rope_started = std::time::Instant::now();
    for token in 0..total {
        let off = token * QKV_DIM;
        rms_norm_per_head(
            &mut qkv_buf[off..off + HIDDEN],
            1,
            n_heads,
            heads,
            &block.q_norm,
            1e-6,
        );
        rms_norm_per_head(
            &mut qkv_buf[off + HIDDEN..off + 2 * HIDDEN],
            1,
            n_heads,
            heads,
            &block.k_norm,
            1e-6,
        );
        let pos = token; // Sequential 0..text_tokens+img_tokens after the [text, img] swap.
        let table = pos * (HEAD_DIM / 2);
        rope_apply_token(
            &mut qkv_buf[off..off + HIDDEN],
            HEAD_DIM,
            &rope_cos[table..table + HEAD_DIM / 2],
            &rope_sin[table..table + HEAD_DIM / 2],
        );
        rope_apply_token(
            &mut qkv_buf[off + HIDDEN..off + 2 * HIDDEN],
            HEAD_DIM,
            &rope_cos[table..table + HEAD_DIM / 2],
            &rope_sin[table..table + HEAD_DIM / 2],
        );
    }
    profile::add(&profile::ROPE_NS, rope_started);

    // Single-stream attention.
    let attention_started = std::time::Instant::now();
    transpose_qkv(
        pool,
        &qkv_buf[..total * QKV_DIM],
        &[],
        total,
        0,
        &mut qkv_head[..total * QKV_DIM],
    );
    attention_heads(pool, &qkv_head[..total * QKV_DIM], total, scores, attention);
    profile::add(&profile::ATTENTION_NS, attention_started);
    linear_rows_into_dispatch(
        linear,
        source,
        &block.out,
        HIDDEN,
        HIDDEN,
        &attention[..total * HIDDEN],
        total,
        &mut normed[..total * HIDDEN],
        q8,
        pool,
    )?;
    for token in 0..total {
        let off = token * HIDDEN;
        for d in 0..HIDDEN {
            joint[off + d] += (normed[off + d] + block.out_bias[d]) * gate_msa[d];
        }
    }
    // MLP
    let layer_started = std::time::Instant::now();
    for token in 0..total {
        let off = token * HIDDEN;
        let normed_token = &mut normed[token * HIDDEN..(token + 1) * HIDDEN];
        layer_norm_token(&joint[off..off + HIDDEN], normed_token);
        for d in 0..HIDDEN {
            let v = normed_token[d];
            normed_token[d] = v * (1.0 + scale_mlp[d]) + shift_mlp[d];
        }
    }
    profile::add(&profile::LAYER_NS, layer_started);
    linear_rows_into_dispatch(
        linear,
        source,
        &block.ff_in,
        HIDDEN,
        PACKED_FF_IN,
        &normed[..total * HIDDEN],
        total,
        &mut ff_packed[..total * PACKED_FF_IN],
        q8,
        pool,
    )?;
    let swiglu_started = std::time::Instant::now();
    for token in 0..total {
        let packed = &ff_packed[token * PACKED_FF_IN..(token + 1) * PACKED_FF_IN];
        let gated = &mut ff_gated[token * FF_INNER..(token + 1) * FF_INNER];
        for d in 0..FF_INNER {
            let g = packed[d];
            let u = packed[FF_INNER + d];
            let silu = g / (1.0 + (-g).exp());
            gated[d] = silu * u;
        }
    }
    profile::add(&profile::SWIGLU_NS, swiglu_started);
    linear_rows_into_dispatch(
        linear,
        source,
        &block.ff_out,
        FF_INNER,
        HIDDEN,
        &ff_gated[..total * FF_INNER],
        total,
        &mut normed[..total * HIDDEN],
        q8,
        pool,
    )?;
    for token in 0..total {
        let off = token * HIDDEN;
        for d in 0..HIDDEN {
            joint[off + d] += normed[off + d] * gate_mlp[d];
        }
    }
    super::super::common::require_finite(joint, "AuK single block output")
}

#[allow(clippy::too_many_arguments)]
fn linear_into_dispatch(
    linear: &AukLinearSession<'_, '_>,
    source: &dyn TensorSource,
    name: &str,
    n_in: usize,
    n_out: usize,
    input: &[f32],
    output: &mut [f32],
    q8: &mut Q8Scratch,
    pool: &ComputePool,
) -> Result<(), String> {
    if input.len() != n_in || output.len() != n_out {
        return Err(format!("Invalid {name} linear shape"));
    }
    let started = std::time::Instant::now();
    profile::count_call();
    let result = if linear.run(name, input, output, 1.0)? {
        Ok(())
    } else {
        super::linear_into_scaled_impl(source, name, n_in, n_out, input, output, q8, pool, 1.0)
    };
    profile::add(&profile::LINEAR_NS, started);
    result
}

/// Apply a projection to `rows` contiguous token rows.
///
/// The rows are handed to the device in a single submission: the AuK DiT
/// applies every projection to a whole token stream, and one submission per
/// token made the fence wait dominate the runtime.
#[allow(clippy::too_many_arguments)]
fn linear_rows_into_dispatch(
    linear: &AukLinearSession<'_, '_>,
    source: &dyn TensorSource,
    name: &str,
    n_in: usize,
    n_out: usize,
    input: &[f32],
    rows: usize,
    output: &mut [f32],
    q8: &mut Q8Scratch,
    pool: &ComputePool,
) -> Result<(), String> {
    if rows == 0 || input.len() != rows * n_in || output.len() != rows * n_out {
        return Err(format!("Invalid {name} linear shape"));
    }
    let started = std::time::Instant::now();
    profile::count_call();
    let result = if linear.run_rows(name, input, rows, output, 1.0)? {
        Ok(())
    } else {
        // The weight is not bound to the device session (it is not F16), so
        // fall back to the generic per-row CPU path.
        for row in 0..rows {
            super::linear_into_scaled_impl(
                source,
                name,
                n_in,
                n_out,
                &input[row * n_in..(row + 1) * n_in],
                &mut output[row * n_out..(row + 1) * n_out],
                q8,
                pool,
                1.0,
            )?;
        }
        Ok(())
    };
    profile::add(&profile::LINEAR_NS, started);
    result
}
