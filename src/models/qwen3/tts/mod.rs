//! Text-to-Speech inference.
//!
//! Qwen3-TTS-12Hz 1.7B Base is split across two GGUF files:
//!
//! - [`Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf`] — the autoregressive "Talker" LLM
//!   that maps text tokens to a stream of audio-codebook token ids at 12 Hz.
//! - [`mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf`] — the codec decoder
//!   (speaker encoder + RVQ code predictor + DAC upsampler) that turns those
//!   audio codes into a 24 kHz waveform.
//!
//! Stage 1 ships only the Talker; the codec decoder is added in Stage 2.

use crate::core::tensor::{GGMLType, TensorSource};

pub mod codec;
pub mod speaker;
pub mod talker;

pub use talker::{
    Qwen3TtsGeneration, Qwen3TtsTalker, Qwen3TtsTalkerConfig, TtsPrompt, TtsSession,
    TtsSpecialTokens,
};

pub(crate) fn load_f16_or_f32_tensor(
    source: &dyn TensorSource,
    name: &str,
    dims: &[u64],
) -> Result<Vec<f32>, String> {
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("Missing tensor: {name}"))?;
    if info.dims != dims || !matches!(info.ggml_type, GGMLType::F16 | GGMLType::F32) {
        return Err(format!(
            "Invalid tensor {name}: shape {:?} type {:?}; expected {:?} F16/F32",
            info.dims, info.ggml_type, dims
        ));
    }
    let expected = usize::try_from(
        info.checked_nbytes()
            .ok_or_else(|| format!("Invalid tensor byte size: {name}"))?,
    )
    .map_err(|_| format!("Tensor byte size does not fit usize: {name}"))?;
    let bytes = source
        .tensor_slice(name)
        .ok_or_else(|| format!("Missing tensor data: {name}"))?;
    if bytes.len() != expected {
        return Err(format!(
            "Invalid tensor data length for {name}: {}; expected {expected}",
            bytes.len()
        ));
    }
    Ok(match info.ggml_type {
        GGMLType::F16 => bytes
            .chunks_exact(2)
            .map(|chunk| half::f16::from_le_bytes([chunk[0], chunk[1]]).to_f32())
            .collect(),
        GGMLType::F32 => bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
            .collect(),
        _ => unreachable!("validated F16/F32 tensor"),
    })
}

pub(crate) fn load_f16_tensor(
    source: &dyn TensorSource,
    name: &str,
    dims: &[u64],
) -> Result<Vec<u16>, String> {
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("Missing tensor: {name}"))?;
    if info.dims != dims || info.ggml_type != GGMLType::F16 {
        return Err(format!(
            "Invalid tensor {name}: shape {:?} type {:?}; expected {:?} F16",
            info.dims, info.ggml_type, dims
        ));
    }
    let bytes = source
        .tensor_slice(name)
        .ok_or_else(|| format!("Missing tensor data: {name}"))?;
    let expected = dims.iter().try_fold(2usize, |size, dim| {
        size.checked_mul(*dim as usize)
            .ok_or_else(|| format!("Tensor byte size overflow: {name}"))
    })?;
    if bytes.len() != expected {
        return Err(format!(
            "Invalid tensor data length for {name}: {}; expected {expected}",
            bytes.len()
        ));
    }
    Ok(bytes
        .chunks_exact(2)
        .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
        .collect())
}

/// Format used by the 12 Hz codec for its audio tokens. Each emitted token is
/// a single codebook index in the range `audio_codebook_offset..audio_codebook_offset+3072`.
pub const AUDIO_CODEBOOK_SIZE: usize = 3072;

/// Sampling temperature suggested by `general.sampling.temp = 0.9` in the
/// Qwen3-TTS Base GGUF metadata.
pub const TTS_DEFAULT_TEMP: f32 = 0.9;

pub fn predictor_top_k(temperature: f32) -> usize {
    if temperature <= 0.0 {
        1
    } else {
        50
    }
}
