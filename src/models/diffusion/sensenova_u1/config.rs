//! Fixed architecture contract for `SenseNova/SenseNova-U1.6-8B-MoT`.
//!
//! The GGUF header only carries `general.architecture` and a handful of
//! bookkeeping keys -- every architecture parameter lives in the companion
//! `config.json` of the source checkpoint. Rather than parse that JSON at
//! runtime (which would let a mismatched config silently produce garbage), the
//! parameters are pinned here as constants and the loader validates the tensor
//! contract against them, the same way `qwen_image_2_1` detects its config from
//! tensor shapes.
//!
//! See `docs/develop/SENSENOVA_U1.md` for the derivation of each value.

use crate::core::tensor::{GGMLType, TensorSource};

/// `general.architecture` of every SenseNova-U1.6 GGUF.
pub const ARCH: &str = "sensenova_u1.6";

/// `source_config_sha256` reported by the published `*.conversion.json`.
/// Recorded here so the oracle README can be cross-checked; the GGUF itself
/// does not embed it.
pub(crate) const SOURCE_CONFIG_SHA256: &str =
    "b5c28b643a4cfc771d1303acd12c94126b7a725a6c9796b3d5ecca07e9a582ef";

pub(crate) const HIDDEN: usize = 4096;
pub(crate) const LAYERS: usize = 42;
pub(crate) const QUERY_HEADS: usize = 32;
pub(crate) const KV_HEADS: usize = 8;
pub(crate) const HEAD_DIM: usize = 128;

/// Width of the time-axis slice inside `head_dim`. The remaining two quarters
/// carry the height and width axes.
pub(crate) const ROPE_T_DIM: usize = HEAD_DIM / 2;
pub(crate) const ROPE_HW_DIM: usize = HEAD_DIM / 4;

pub(crate) const ROPE_THETA_T: f32 = 5_000_000.0;
pub(crate) const ROPE_THETA_HW: f32 = 10_000.0;

pub(crate) const FFN: usize = 12_288;
pub(crate) const VOCAB: usize = 151_936;
pub(crate) const RMS_EPS: f32 = 1e-6;

/// Image patch size of the decoder head.
pub(crate) const PATCH_SIZE: usize = 16;
/// `1 / downsample_ratio`; one image token covers `PATCH_SIZE * MERGE_SIZE`
/// pixels per side.
pub(crate) const MERGE_SIZE: usize = 2;
/// Pixels per image-token side (32).
pub(crate) const TOKEN_SIDE: usize = PATCH_SIZE * MERGE_SIZE;
/// Channels in one image token (`3 * 32 * 32`).
pub(crate) const TOKEN_DIM: usize = 3 * TOKEN_SIDE * TOKEN_SIDE;

/// Vision tower width before the projection to the LLM width.
pub(crate) const VISION_HIDDEN: usize = 1024;
pub(crate) const VISION_THETA: f32 = 10_000.0;

pub(crate) const TIMESTEP_FREQ_DIM: usize = 256;
pub(crate) const CONV_DECODER_HIDDEN: usize = 1024;
pub(crate) const CONV_DECODER_MID: usize = 192;

/// Default sampler knobs, matching `examples/t2i/inference.py`.
pub(crate) const DEFAULT_STEPS: usize = 50;
pub(crate) const DEFAULT_CFG_SCALE: f32 = 4.0;
pub(crate) const DEFAULT_TIMESTEP_SHIFT: f32 = 3.0;
/// `t2i_generate`'s own default (`--t_eps` is not exposed on the CLI).
pub(crate) const DEFAULT_T_EPS: f32 = 0.02;

pub(crate) const NOISE_SCALE: f32 = 1.0;
pub(crate) const NOISE_SCALE_MAX: f32 = 16.0;
pub(crate) const NOISE_SCALE_BASE_IMAGE_SEQ_LEN: f64 = 64.0;

/// Special token ids pinned by `config.json` / the oracle tokenizer.
pub(crate) const PAD_TOKEN_ID: u32 = 151_643;
pub(crate) const EOS_TOKEN_ID: u32 = 151_645;
pub(crate) const IMG_START_TOKEN_ID: u32 = 151_670;

/// Validate that `source` really is a SenseNova-U1.6 GGUF before anything is
/// read. Checks the architecture tag and a representative tensor from each
/// component so a truncated or foreign file fails loudly instead of producing
/// noise.
pub(crate) fn validate(source: &dyn TensorSource) -> Result<(), String> {
    let arch = source
        .metadata("general.architecture")
        .and_then(|v| v.to_string_val())
        .unwrap_or_default();
    if arch != ARCH {
        return Err(format!(
            "SenseNova-U1.6 requires general.architecture={ARCH:?} (got {arch:?})"
        ));
    }

    let require = |name: &str, dims: &[u64], types: &[GGMLType]| -> Result<(), String> {
        crate::core::tensor::require_tensor_meta(source, name, dims, types).map(|_| ())
    };

    require(
        "language_model.model.embed_tokens.weight",
        &[HIDDEN as u64, VOCAB as u64],
        &[GGMLType::F32],
    )?;
    require(
        "language_model.model.norm.weight",
        &[HIDDEN as u64],
        &[GGMLType::F32],
    )?;
    require(
        "language_model.model.norm_mot_gen.weight",
        &[HIDDEN as u64],
        &[GGMLType::F32],
    )?;
    // First and last decoder layer: catches a GGUF whose layer count differs.
    for layer in [0usize, LAYERS - 1] {
        require(
            &format!("language_model.model.layers.{layer}.self_attn.q_proj.weight"),
            &[HIDDEN as u64, HIDDEN as u64],
            &[GGMLType::Q8_0, GGMLType::Q4_0, GGMLType::Q6K, GGMLType::F32],
        )?;
        require(
            &format!("language_model.model.layers.{layer}.self_attn.k_proj_mot_gen.weight"),
            &[HIDDEN as u64, (KV_HEADS * HEAD_DIM) as u64],
            &[GGMLType::Q8_0, GGMLType::Q4_0, GGMLType::Q6K, GGMLType::F32],
        )?;
        require(
            &format!("language_model.model.layers.{layer}.mlp_mot_gen.down_proj.weight"),
            &[FFN as u64, HIDDEN as u64],
            &[GGMLType::Q8_0, GGMLType::Q4_0, GGMLType::Q6K, GGMLType::F32],
        )?;
    }
    if source
        .tensor_info(&format!(
            "language_model.model.layers.{LAYERS}.self_attn.q_proj.weight"
        ))
        .is_some()
    {
        return Err(format!("SenseNova-U1.6 expects {LAYERS} decoder layers"));
    }

    require(
        "vision_model.embeddings.patch_embedding.weight",
        &[
            PATCH_SIZE as u64,
            PATCH_SIZE as u64,
            3,
            VISION_HIDDEN as u64,
        ],
        &[GGMLType::F32],
    )?;
    require(
        "fm_modules.vision_model_mot_gen.embeddings.dense_embedding.weight",
        &[
            MERGE_SIZE as u64,
            MERGE_SIZE as u64,
            VISION_HIDDEN as u64,
            HIDDEN as u64,
        ],
        &[GGMLType::F32],
    )?;
    require(
        "fm_modules.timestep_embedder.mlp.0.weight",
        &[TIMESTEP_FREQ_DIM as u64, HIDDEN as u64],
        &[GGMLType::F32],
    )?;
    require(
        "fm_modules.noise_scale_embedder.mlp.2.weight",
        &[HIDDEN as u64, HIDDEN as u64],
        &[GGMLType::F32],
    )?;
    require(
        "fm_modules.fm_head.conv1.weight",
        &[3, 3, (HIDDEN / 4) as u64, CONV_DECODER_HIDDEN as u64],
        &[GGMLType::F32],
    )?;
    require(
        "fm_modules.fm_head.conv2.weight",
        &[
            3,
            3,
            (CONV_DECODER_HIDDEN / 4) as u64,
            CONV_DECODER_MID as u64,
        ],
        &[GGMLType::F32],
    )?;

    Ok(())
}

/// Quantization formats the loader accepts. Kept next to `validate` so the
/// matrix in `docs/develop/SENSENOVA_U1.md` has a single source of truth.
pub(crate) fn supported_linear_types() -> &'static [GGMLType] {
    &[
        GGMLType::F32,
        GGMLType::F16,
        GGMLType::Q8_0,
        GGMLType::Q4_0,
        GGMLType::Q6K,
    ]
}
