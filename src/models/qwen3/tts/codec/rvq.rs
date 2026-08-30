//! Residual Vector Quantisation (RVQ) decoder for the Qwen3-TTS codec.
//!
//! The codec uses 16 codebook levels: 1 first code + 15 residual codes. Each
//! codebook contains 2048 entries of 256-dim vectors. Decoding a sequence
//! of indices reconstructs the time-domain embedding by summing the looked-up
//! vectors across all 16 levels.

use crate::core::tensor::{GGMLType, TensorSource};
use crate::models::qwen3::trunk::util::{checked_product, static_q8_matrix};
#[cfg(target_arch = "aarch64")]
use crate::ops::kernel::q8_0::dispatch::matmul_q8_0_quantized_range_nrc1;
#[cfg(not(target_arch = "aarch64"))]
use crate::ops::matmul_q8_0_quantized_parallel_rows;
use crate::ops::quantize_q8_0_into;

const RVQ_DIM: usize = crate::models::qwen3::tts::codec::RVQ_CODE_DIM;
const RVQ_VOCAB: usize = crate::models::qwen3::tts::codec::RVQ_CODEBOOK_SIZE;
const RVQ_LEVELS: usize = crate::models::qwen3::tts::codec::RVQ_LEVELS;
const Q8_0_BLOCK_SIZE: usize = 34;

/// One RVQ decoder configuration. Holds the mmap-backed weight tensors for the
/// 1 first + 15 residual codebook lookup tables.
pub struct RvqDecoder {
    first_out_w: &'static [u8],
    first_codebook: Vec<f32>,
    rest_out_w: &'static [u8],
    rest_codebook: Vec<f32>,
}

impl RvqDecoder {
    pub fn from_source(source: &dyn TensorSource) -> Result<Self, String> {
        // First (semantic) codebook: out_proj × codebook.
        let first_out_w = static_q8_matrix(
            source,
            "a.gen.wav.quant.first.out_proj.weight",
            RVQ_DIM,
            512,
        )?;
        let first_bytes = source
            .tensor_slice("a.gen.wav.quant.first.codebook.weight")
            .ok_or_else(|| "Missing tensor: a.gen.wav.quant.first.codebook.weight".to_string())?;
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

        // Rest (acoustic) codebooks: out_proj × summed codebook.
        let rest_out_w =
            static_q8_matrix(source, "a.gen.wav.quant.rest.out_proj.weight", RVQ_DIM, 512)?;
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
            first_out_w,
            first_codebook,
            rest_out_w,
            rest_codebook,
        })
    }

    pub fn first_codebook_size(&self) -> usize {
        self.first_codebook.len() / RVQ_DIM
    }

    /// Decode a sequence of `RVQ_LEVELS` codes per timestep into a 512-dim
    /// continuous embedding (column-major: `[512, timesteps]`).
    ///
    /// `codes` has shape `[timesteps, RVQ_LEVELS]`. Mirrors the reference
    /// decoder's `quant_decode`: first (semantic) codebook lookup → `out_proj`
    /// (256 → 512), summed acoustic codebooks → `out_proj` (256 → 512), then
    /// add the two halves.
    ///
    pub fn decode(&self, codes: &[u32]) -> Result<Vec<f32>, String> {
        if codes.len() % RVQ_LEVELS != 0 {
            return Err(format!(
                "RVQ codes length {} is not a multiple of {RVQ_LEVELS}",
                codes.len(),
            ));
        }
        let frames: Vec<[u32; RVQ_LEVELS]> = codes
            .chunks_exact(RVQ_LEVELS)
            .map(|frame| frame.try_into().expect("exact RVQ frame"))
            .collect();
        self.decode_frames(&frames)
    }

    pub fn decode_frames(&self, frames: &[[u32; RVQ_LEVELS]]) -> Result<Vec<f32>, String> {
        validate_rvq_frames(frames)?;
        let timesteps = frames.len();
        let hidden_dim = 512usize;
        let mut first_hidden = vec![0.0f32; timesteps * RVQ_DIM];
        let mut rest_hidden = vec![0.0f32; timesteps * RVQ_DIM];
        for (t, frame) in frames.iter().enumerate() {
            let first_idx = frame[0] as usize;
            for d in 0..RVQ_DIM {
                first_hidden[t * RVQ_DIM + d] = self.first_codebook[first_idx * RVQ_DIM + d];
            }
            for level in 1..RVQ_LEVELS {
                let idx = frame[level] as usize;
                let level_off = (level - 1) * RVQ_VOCAB * RVQ_DIM;
                for d in 0..RVQ_DIM {
                    rest_hidden[t * RVQ_DIM + d] +=
                        self.rest_codebook[level_off + idx * RVQ_DIM + d];
                }
            }
        }
        let first_proj = matmul_2d_q8(
            self.first_out_w,
            &first_hidden,
            hidden_dim,
            RVQ_DIM,
            timesteps,
        )?;
        let rest_proj = matmul_2d_q8(
            self.rest_out_w,
            &rest_hidden,
            hidden_dim,
            RVQ_DIM,
            timesteps,
        )?;
        let mut output = vec![0.0f32; timesteps * hidden_dim];
        for t in 0..timesteps {
            for d in 0..hidden_dim {
                output[t * hidden_dim + d] =
                    first_proj[t * hidden_dim + d] + rest_proj[t * hidden_dim + d];
            }
        }
        #[cfg(feature = "parity-trace")]
        crate::parity_trace::report(crate::parity_trace::checkpoint(
            "tts.rvq_hidden",
            None,
            &[timesteps, hidden_dim],
            &output,
        ));
        Ok(output)
    }
}

fn validate_rvq_frames(frames: &[[u32; RVQ_LEVELS]]) -> Result<(), String> {
    if frames.is_empty() {
        return Err("RVQ requires at least one complete frame".into());
    }
    for (frame_index, frame) in frames.iter().enumerate() {
        for (codebook, &code) in frame.iter().enumerate() {
            if code as usize >= RVQ_VOCAB {
                return Err(format!(
                    "RVQ frame {frame_index} codebook {codebook} code {code} exceeds {RVQ_VOCAB}"
                ));
            }
        }
    }
    Ok(())
}

fn matmul_2d_q8(
    weight: &[u8],
    input: &[f32],
    out_dim: usize,
    in_dim: usize,
    n_tokens: usize,
) -> Result<Vec<f32>, String> {
    let blocks = in_dim.div_ceil(32);
    let expected = out_dim * blocks * Q8_0_BLOCK_SIZE;
    if weight.len() != expected {
        return Err(format!(
            "matmul_2d_q8: weight len {} != expected {}",
            weight.len(),
            expected,
        ));
    }
    if input.len() != n_tokens * in_dim {
        return Err("matmul_2d_q8: input length mismatch".into());
    }
    let mut out = vec![0.0f32; n_tokens * out_dim];
    let mut input_q8 = vec![0; in_dim];
    let mut input_scales = vec![0.0; blocks];
    for t in 0..n_tokens {
        quantize_q8_0_into(
            &input[t * in_dim..(t + 1) * in_dim],
            in_dim,
            &mut input_q8,
            &mut input_scales,
        );
        let output = &mut out[t * out_dim..(t + 1) * out_dim];
        #[cfg(target_arch = "aarch64")]
        {
            matmul_q8_0_quantized_range_nrc1(
                weight,
                &input_q8,
                &input_scales,
                output,
                in_dim,
                0,
                out_dim,
            );
        }
        #[cfg(not(target_arch = "aarch64"))]
        matmul_q8_0_quantized_parallel_rows(
            weight,
            &input_q8,
            &input_scales,
            output,
            in_dim,
            out_dim,
            0,
            1,
        );
    }
    Ok(out)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rvq_rejects_out_of_range_codes_instead_of_wrapping() {
        let frames = [[0u32; 16], [2048u32; 16]];
        assert!(validate_rvq_frames(&frames)
            .unwrap_err()
            .contains("frame 1 codebook 0"));
    }
}
