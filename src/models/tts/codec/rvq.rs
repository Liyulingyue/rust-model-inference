//! Residual Vector Quantisation (RVQ) decoder for the Qwen3-TTS codec.
//!
//! The codec uses 16 codebook levels: 1 first code + 15 residual codes. Each
//! codebook contains 2048 entries of 256-dim vectors. Decoding a sequence
//! of indices reconstructs the time-domain embedding by summing the looked-up
//! vectors across all 16 levels.

use crate::core::tensor::{GGMLType, TensorSource};
use crate::models::qwen3::checked_product;

const RVQ_DIM: usize = crate::models::tts::codec::RVQ_CODE_DIM;
const RVQ_VOCAB: usize = crate::models::tts::codec::RVQ_CODEBOOK_SIZE;
const RVQ_LEVELS: usize = crate::models::tts::codec::RVQ_LEVELS;
const Q8_0_BLOCK_SIZE: usize = 34;

/// One RVQ decoder configuration. Holds the mmap-backed weight tensors for the
/// 1 first + 15 residual codebook lookup tables.
pub struct RvqDecoder {
    first_codebook: Vec<f32>,
    rest_codebook: Vec<f32>,
}

impl RvqDecoder {
    pub fn from_source(source: &dyn TensorSource) -> Result<Self, String> {
        let first_bytes = source
            .tensor_slice("a.gen.wav.quant.first.codebook.weight")
            .ok_or_else(|| {
                "Missing tensor: a.gen.wav.quant.first.codebook.weight".to_string()
            })?;
        let first_info = source
            .tensor_info("a.gen.wav.quant.first.codebook.weight")
            .ok_or_else(|| {
                "Missing tensor info: a.gen.wav.quant.first.codebook.weight".to_string()
            })?;
        let first_dims = [RVQ_DIM as u64, RVQ_VOCAB as u64];
        if first_info.dims != first_dims {
            return Err(format!(
                "first codebook dims {:?} != expected {first_dims:?}",
                first_info.dims,
            ));
        }
        if first_info.ggml_type != GGMLType::Q8_0 {
            return Err(format!(
                "first codebook type {:?} not Q8_0",
                first_info.ggml_type,
            ));
        }
        let first_codebook = dequant_q8_0_table(first_bytes, RVQ_VOCAB, RVQ_DIM)?;

        let rest_bytes = source
            .tensor_slice("a.gen.wav.quant.rest.codebook.weight")
            .ok_or_else(|| "Missing tensor: a.gen.wav.quant.rest.codebook.weight".to_string())?;
        let rest_info = source
            .tensor_info("a.gen.wav.quant.rest.codebook.weight")
            .ok_or_else(|| {
                "Missing tensor info: a.gen.wav.quant.rest.codebook.weight".to_string()
            })?;
        let rest_dims = [RVQ_DIM as u64, RVQ_VOCAB as u64, (RVQ_LEVELS - 1) as u64];
        if rest_info.dims != rest_dims {
            return Err(format!(
                "rest codebook dims {:?} != expected {rest_dims:?}",
                rest_info.dims,
            ));
        }
        if rest_info.ggml_type != GGMLType::Q8_0 {
            return Err(format!(
                "rest codebook type {:?} not Q8_0",
                rest_info.ggml_type,
            ));
        }
        let rest_codebook = dequant_q8_0_table(rest_bytes, RVQ_VOCAB * (RVQ_LEVELS - 1), RVQ_DIM)?;

        Ok(Self {
            first_codebook,
            rest_codebook,
        })
    }

    pub fn first_codebook_size(&self) -> usize {
        self.first_codebook.len() / RVQ_DIM
    }

    /// Decode a sequence of `RVQ_LEVELS` codes per timestep into the
    /// `RVQ_CODE_DIM` continuous embedding.
    ///
    /// `codes` has shape `[timesteps, RVQ_LEVELS]`. Returns `[timesteps,
    /// RVQ_CODE_DIM]` (sum of the 1 first + 15 residual codebook vectors).
    pub fn decode(&self, codes: &[u32]) -> Result<Vec<f32>, String> {
        if codes.len() % RVQ_LEVELS != 0 {
            return Err(format!(
                "RVQ codes length {} is not a multiple of {RVQ_LEVELS}",
                codes.len(),
            ));
        }
        let timesteps = codes.len() / RVQ_LEVELS;
        let mut output = vec![0.0f32; timesteps * RVQ_DIM];
        for t in 0..timesteps {
            let first_idx = codes[t * RVQ_LEVELS] as usize;
            if first_idx >= RVQ_VOCAB {
                return Err(format!(
                    "first codebook index {first_idx} exceeds vocab {RVQ_VOCAB}"
                ));
            }
            for d in 0..RVQ_DIM {
                output[t * RVQ_DIM + d] = self.first_codebook[first_idx * RVQ_DIM + d];
            }
            for level in 1..RVQ_LEVELS {
                let idx = codes[t * RVQ_LEVELS + level] as usize;
                if idx >= RVQ_VOCAB {
                    return Err(format!(
                        "rest codebook level {level} index {idx} exceeds vocab {RVQ_VOCAB}"
                    ));
                }
                let level_off = (level - 1) * RVQ_VOCAB * RVQ_DIM;
                for d in 0..RVQ_DIM {
                    output[t * RVQ_DIM + d] += self.rest_codebook[level_off + idx * RVQ_DIM + d];
                }
            }
        }
        Ok(output)
    }
}

fn dequant_q8_0_table(bytes: &[u8], vocab: usize, dim: usize) -> Result<Vec<f32>, String> {
    let blocks_per_row = dim / 32;
    let bytes_per_row = blocks_per_row * Q8_0_BLOCK_SIZE;
    let expected = checked_product("codebook bytes", vocab, bytes_per_row)?;
    if bytes.len() != expected {
        return Err(format!(
            "codebook bytes {} != expected {}",
            bytes.len(),
            expected
        ));
    }
    let mut out = vec![0.0f32; vocab * dim];
    for row in 0..vocab {
        for b in 0..blocks_per_row {
            let off = row * bytes_per_row + b * Q8_0_BLOCK_SIZE;
            let scale = half::f16::from_le_bytes([bytes[off], bytes[off + 1]]).to_f32();
            for j in 0..32usize {
                let q = bytes[off + 2 + j] as i8 as f32;
                out[row * dim + b * 32 + j] = scale * q;
            }
        }
    }
    Ok(out)
}

const _: () = {
    // Sanity: verify our constants match the Q8_0 block layout assumed by the
    // dequantization loop above (34 bytes per block: 2-byte f16 scale + 32 i8).
    assert!(Q8_0_BLOCK_SIZE == 34);
    assert!(RVQ_DIM % 32 == 0);
};