use super::generation::validate_gemma4_temperature;
use super::multimodal::{run_gemma4_capture_text, run_qwen3_family_multimodal};
use crate::core::tensor::TensorSource;
use std::path::Path;
use std::sync::Arc;

/// Run multimodal inference and feed the generated text through a
/// separate TTS model (Qwen3-TTS / Qwen2.5-Omni Talker compatible) to
/// produce a 24 kHz WAV. Used to bridge Qwen2.5-Omni (no bundled Talker
/// in our GGUF set) to a usable audio output. Currently uses
/// Qwen3-TTS-12Hz-1.7B-Base as the post-processor.
///
/// Note: this implementation streams the Omni text to stdout AND
/// synthesises it for TTS in parallel. The text is captured by running
/// the multimodal flow in a subprocess-style redirect; we use a small
/// helper (`run_multimodal_with_video_capture_text`) that returns the
/// reply as a String instead of printing.
#[allow(clippy::too_many_arguments)]
pub fn run_multimodal_with_tts_postproc(
    llm_source: Arc<dyn TensorSource>,
    model_path: &Path,
    mmproj_path: Option<&Path>,
    image_path: Option<&Path>,
    video_path: Option<&Path>,
    audio_path: Option<&Path>,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    n_threads_arg: usize,
    prefill_batch_size: usize,
    max_context: usize,
    repetition_penalty: f32,
    tts_model: &Path,
    tts_mmproj: &Path,
    wav_out: &Path,
    language: &str,
) -> Result<(), String> {
    let reply = run_multimodal_with_video_capture_text(
        Arc::clone(&llm_source),
        model_path,
        mmproj_path,
        image_path,
        video_path,
        audio_path,
        prompt,
        max_tokens,
        temperature,
        n_threads_arg,
        prefill_batch_size,
        max_context,
        repetition_penalty,
    )?;
    eprintln!(
        "Omni → TTS: captured {} chars from reply; running Qwen3-TTS...",
        reply.chars().count()
    );
    // `synthesize_tts_to_wav` expects the Qwen3-TTS internal language
    // tag (e.g. "english") rather than the ISO code ("en"). Translate
    // via the same normalizer the standalone --tts path uses so the
    // Talker's `<|codec_language_english|>` literal resolves correctly.
    let internal_language = crate::app::cli::normalize_tts_language(Some(language))?;
    let wav_bytes = crate::app::tts::synthesize_tts_to_wav(
        tts_model,
        tts_mmproj,
        &reply,
        internal_language,
        // TTS frame budget: the user's --max-tokens bounds the Omni reply
        // length; the Qwen3-TTS Talker emits one 80 ms audio frame per
        // step and stops on EOS, so `max_tokens * 4` frames (~80 ms per
        // frame) caps audio at ~3.2 seconds per Omni token. Clamp to a
        // floor of 128 so short captions (e.g. 30 tokens) still produce
        // usable audio; cap at 1024 so very long replies don't run the
        // expensive DAC decoder for minutes on end.
        max_tokens.saturating_mul(4).clamp(128, 1024),
        temperature,
        n_threads_arg,
        None,
    )?;
    std::fs::write(wav_out, &wav_bytes)
        .map_err(|error| format!("Failed to write WAV {}: {error}", wav_out.display()))?;
    eprintln!(
        "Omni → TTS: wrote {} bytes ({} samples) to {}",
        wav_bytes.len(),
        wav_bytes.len() / 2,
        wav_out.display()
    );
    Ok(())
}

/// Same as `run_multimodal_with_video` but returns the generated text
/// instead of streaming it to stdout. Used by the TTS post-processor
/// pipeline so we can capture the Omni reply.
pub fn run_multimodal_with_video_capture_text(
    llm_source: Arc<dyn TensorSource>,
    model_path: &Path,
    mmproj_path: Option<&Path>,
    image_path: Option<&Path>,
    video_path: Option<&Path>,
    audio_path: Option<&Path>,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    n_threads_arg: usize,
    prefill_batch_size: usize,
    max_context: usize,
    repetition_penalty: f32,
) -> Result<String, String> {
    let owned_source = Arc::clone(&llm_source);
    let arch = llm_source
        .metadata("general.architecture")
        .and_then(|v| v.to_string_val())
        .unwrap_or_default();
    validate_gemma4_temperature(arch, temperature)?;
    if arch == "gemma4" {
        // gemma4 path uses its own `run_gemma4`; it doesn't go through
        // `run_qwen3_family_multimodal`. For now, capture the text by
        // delegating to `run_multimodal_with_video` and parsing the
        // output via stdout redirection.
        return run_gemma4_capture_text(
            model_path,
            mmproj_path,
            image_path,
            audio_path,
            prompt,
            max_tokens,
            n_threads_arg,
            prefill_batch_size,
        );
    }
    if matches!(arch, "qwen2vl" | "qwen3vl" | "qwen3vlmoe")
        && (image_path.is_some() || video_path.is_some() || audio_path.is_some())
    {
        return run_qwen3_family_multimodal(
            llm_source.as_ref(),
            owned_source,
            mmproj_path.ok_or("multimodal Qwen models require --mmproj")?,
            image_path,
            video_path,
            audio_path,
            prompt,
            max_tokens,
            temperature,
            n_threads_arg,
            prefill_batch_size,
        );
    }
    Err(format!(
        "Only qwen35, qwen3vl, qwen3vlmoe and gemma4 architectures are supported for multimodal capture, got: {arch}"
    ))
}
