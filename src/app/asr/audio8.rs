//! Simulated-streaming 80 ms / 480 ms Audio8 ASR CLI adapter.
//!
//! Mechanics (chunk schedule, token queue, parity-trace report sites) live in
//! [`crate::models::audio8::streaming`]; this module only parses CLI options,
//! loads the WAV file, drives the transcriber, and prints the result.

use crate::app::cli::CliOptions;
use crate::app::open_or_exit;
use crate::core::tensor::TensorSource;
use crate::core::tokenizer::BPETokenizer;
use crate::format::ggufrs::ComponentRole;
use crate::models::audio8::streaming::{
    Schedule, SpecialTokens, StreamingTranscriber, SAMPLE_RATE,
};
use crate::models::audio8::text::Audio8TextDecoder;
use crate::models::audio8::Audio8Encoder;
use crate::models::qwen3::asr::audio_processor::decode_pcm16_wav_any;
use std::sync::Arc;

fn load_samples(path: &std::path::Path) -> Result<Vec<f32>, String> {
    let bytes = std::fs::read(path).map_err(|error| format!("{}: {error}", path.display()))?;
    let decoded =
        decode_pcm16_wav_any(&bytes).map_err(|error| format!("invalid Audio8 WAV: {error:?}"))?;
    let samples = if decoded.channels == 1 {
        decoded.samples
    } else {
        decoded
            .samples
            .chunks(decoded.channels as usize)
            .map(|channels| channels.iter().sum::<f32>() / channels.len() as f32)
            .collect()
    };
    if samples.is_empty() {
        return Err("Audio8 WAV contains no audio".into());
    }
    let samples = if decoded.sample_rate as usize == SAMPLE_RATE {
        samples
    } else {
        crate::models::funasr::model::linear_resample(
            &samples,
            decoded.sample_rate as usize,
            SAMPLE_RATE,
        )
    };
    Ok(samples)
}

pub fn run_audio8_cli(options: &CliOptions) -> Result<(), String> {
    if options.mmproj.is_some() {
        return Err("Audio8 carries its audio tower in --model; omit --mmproj".into());
    }
    let source: Arc<dyn TensorSource> = Arc::from(open_or_exit(&options.model, ComponentRole::Llm));
    let encoder = Audio8Encoder::from_source(Arc::clone(&source))?;
    let decoder_source = Arc::clone(&source);
    let tokenizer = Arc::new(BPETokenizer::from_gguf_metadata(|key| {
        source.metadata(key).cloned()
    })?);
    let decoder = Audio8TextDecoder::from_source(decoder_source)?;
    let tokens = SpecialTokens::lookup(&tokenizer)?;
    let language = options.language.as_deref().unwrap_or("zh");
    let language_token = tokens.language(language)?;
    let audio_path = options.audio.as_ref().ok_or("Audio8 requires --audio")?;
    let samples = load_samples(audio_path)?;
    let stream = Schedule::new().padded_stream(&samples);
    let max_tokens = options.max_tokens.unwrap_or(512);
    let mut transcriber = StreamingTranscriber::new(
        &encoder,
        &decoder,
        stream,
        tokens,
        language_token,
        max_tokens,
    )?;
    while transcriber.next_chunk()?.is_some() {}
    let visible = transcriber.finish();
    println!("{}", tokenizer.decode(&visible, false));
    Ok(())
}
