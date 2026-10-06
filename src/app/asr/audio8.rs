//! Simulated-streaming 80 ms / 480 ms Audio8 ASR CLI adapter.
//!
//! Mechanics (chunk schedule, token queue, parity-trace report sites, WAV
//! decode) live in [`crate::models::audio8::streaming`]; this module only
//! parses CLI options and drives the transcriber.

use crate::app::cli::CliOptions;
use crate::app::open_or_exit;
use crate::core::tensor::TensorSource;
use crate::core::tokenizer::BPETokenizer;
use crate::format::ggufrs::ComponentRole;
use crate::models::audio8::streaming::{
    decode_samples, Schedule, SpecialTokens, StreamingTranscriber,
};
use crate::models::audio8::text::Audio8TextDecoder;
use crate::models::audio8::Audio8Encoder;
use std::sync::Arc;

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
    let wav_bytes =
        std::fs::read(audio_path).map_err(|error| format!("{}: {error}", audio_path.display()))?;
    let samples = decode_samples(&wav_bytes)?;
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
