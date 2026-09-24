//! SenseVoice ASR: fbank → SAN-M encoder → CTC greedy decode.
//!
//! Standalone speech recognition pipeline that does not require an LLM decoder.
//! The encoder output feeds directly into a CTC head, and greedy CTC decoding
//! produces the transcription text.

use crate::core::tensor::{MetaValue, TensorSource};
use crate::core::thread_pool::ComputePool;
use crate::models::funasr::config::FunAsrConfig;
use crate::models::funasr::encoder::{load_f32_vec, load_linear, linear_fwd, Linear, SanmEncoder};
use crate::models::funasr::fbank;
use std::sync::Arc;

pub const ARCH: &str = "sensevoice-small";

pub fn is_sensevoice(source: &dyn TensorSource) -> bool {
    source
        .metadata("general.architecture")
        .and_then(MetaValue::to_string_val)
        .is_some_and(|arch| arch == ARCH)
}

pub struct SenseVoiceModel {
    encoder: SanmEncoder,
    pool: Arc<ComputePool>,
    embed_weight: Vec<f32>,
    query_tokens: Vec<usize>,
    ctc: Linear,
    blank_id: usize,
    vocab: Vec<String>,
    d_model: usize,
    input_size: usize,
}

impl SenseVoiceModel {
    pub fn new(source: Arc<dyn TensorSource>, pool: Arc<ComputePool>) -> Result<Self, String> {
        let config = FunAsrConfig::default();

        let encoder = SanmEncoder::new(source.as_ref(), Arc::clone(&pool), "encoder.", config)?;

        let embed_weight = load_f32_vec(source.as_ref(), "embed.weight")?;
        let query_tokens = load_query_tokens(source.as_ref())?;
        let blank_id = source
            .metadata("sv.blank_id")
            .and_then(MetaValue::to_u64)
            .map(|v| v as usize)
            .unwrap_or(0);

        let ctc = load_linear(source.as_ref(), "ctc.ctc_lo.")?;
        let vocab = load_vocab(source.as_ref(), "sv.vocab")?;

        Ok(Self {
            encoder,
            pool,
            embed_weight,
            query_tokens,
            ctc,
            blank_id,
            vocab,
            d_model: config.output_size,
            input_size: config.input_size,
        })
    }

    pub fn transcribe(&self, samples: &[f32]) -> Result<String, String> {
        let (fbank_data, t_fbank) = fbank::compute_fbank(samples);
        if t_fbank == 0 {
            return Ok(String::new());
        }

        let embed_cols = self.input_size;
        let n_query = self.query_tokens.len();
        let total_t = t_fbank + n_query;
        let mut input = vec![0.0f32; total_t * embed_cols];

        for (i, &tok_id) in self.query_tokens.iter().enumerate() {
            let src = &self.embed_weight[tok_id * embed_cols..(tok_id + 1) * embed_cols];
            input[i * embed_cols..(i + 1) * embed_cols].copy_from_slice(src);
        }
        input[n_query * embed_cols..].copy_from_slice(&fbank_data);

        let scale = (self.d_model as f32).sqrt();
        for v in &mut input {
            *v *= scale;
        }
        fbank::add_position_encoding(&mut input, total_t, self.input_size);

        let enc_out = self.encoder.encode(&input, total_t)?;
        let t_enc = total_t;

        let logits = linear_fwd(&self.ctc, &enc_out, t_enc, &self.pool);
        let vocab_size = self.ctc.out_dim;
        let token_ids = greedy_ctc_decode(&logits, t_enc, vocab_size, self.blank_id);

        Ok(detokenize(&token_ids, &self.vocab))
    }
}

fn load_query_tokens(source: &dyn TensorSource) -> Result<Vec<usize>, String> {
    let arr = source
        .metadata("sv.query_tokens")
        .and_then(MetaValue::to_arr)
        .ok_or_else(|| "missing sv.query_tokens metadata".to_string())?;
    Ok(arr.iter().map(|v| v.to_u64().unwrap_or(0) as usize).collect())
}

fn load_vocab(source: &dyn TensorSource, key: &str) -> Result<Vec<String>, String> {
    let arr = source
        .metadata(key)
        .and_then(MetaValue::to_arr)
        .ok_or_else(|| format!("missing metadata: {key}"))?;
    Ok(arr
        .iter()
        .map(|v| v.to_string_val().unwrap_or("").to_string())
        .collect())
}

fn greedy_ctc_decode(logits: &[f32], t: usize, vocab_size: usize, blank_id: usize) -> Vec<usize> {
    let mut prev = usize::MAX;
    let mut out = Vec::new();
    for i in 0..t {
        let row = &logits[i * vocab_size..(i + 1) * vocab_size];
        // SIMD argmax (AVX2 / NEON / scalar fallback). For SenseVoice
        // vocab_size = 25000, so each frame saves ~25k scalar
        // comparisons vs the original `for (j, &v) in row.iter()`
        // loop. Per-audio gain is ~5ms scalar → ~0.6ms SIMD.
        let best = crate::ops::argmax_f32(row);
        if best != blank_id && best != prev {
            out.push(best);
        }
        prev = best;
    }
    out
}

fn detokenize(token_ids: &[usize], vocab: &[String]) -> String {
    let mut text = String::new();
    for &id in token_ids {
        if let Some(word) = vocab.get(id) {
            if word.starts_with("<|") && word.ends_with("|>") {
                continue;
            }
            let cleaned = word.replace('\u{2581}', " ").replace('▁', " ");
            text.push_str(&cleaned);
        }
    }
    text.trim().to_string()
}
