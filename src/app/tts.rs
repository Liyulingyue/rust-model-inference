use crate::app::cli::{normalize_tts_language, resolve_thread_count};
use crate::app::open_or_exit;
use crate::core::tensor::TensorSource;
use crate::core::thread_pool::ComputePool;
use crate::core::tokenizer::BPETokenizer;
use crate::format::ggufrs::ComponentRole;
use crate::models::qwen3::tts::codec::{
    encode_wav_pcm16, write_wav_f32, Code2WavDecoder, CodePredictor, WAVEFORM_SAMPLE_RATE,
};
use crate::models::qwen3::tts::speaker::{reference_wav_to_mel, Qwen3TtsSpeakerEncoder};
use crate::models::qwen3::tts::{predictor_top_k, Qwen3TtsTalker, TtsPrompt, TTS_DEFAULT_TEMP};
use std::cell::Cell;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

pub fn run_tts_cli(options: &crate::app::cli::CliOptions) -> Result<(), String> {
    let started = Instant::now();
    let prompt_text = options
        .prompt
        .as_deref()
        .filter(|prompt| !prompt.trim().is_empty())
        .ok_or_else(|| "--tts requires --prompt".to_string())?;
    let language = normalize_tts_language(options.language.as_deref())?;
    let mmproj_path = options
        .mmproj
        .as_deref()
        .filter(|path| !path.as_os_str().is_empty())
        .ok_or_else(|| "--tts requires --mmproj".to_string())?;
    let out_path = options
        .out
        .as_deref()
        .ok_or_else(|| "--tts requires --out".to_string())?;

    // dots.tts uses an arch-qwen2 LLM gguf + a `dotstts` mmproj; dispatch on
    // the mmproj architecture before the Qwen3-TTS path.
    let mmproj_probe = open_or_exit(mmproj_path, ComponentRole::Mmproj);
    let mmproj_arch = mmproj_probe
        .metadata("general.architecture")
        .and_then(|value| value.to_string_val())
        .unwrap_or_default();
    if mmproj_arch == "dotstts" {
        return crate::app::dots::run_dots_tts_cli(options);
    }
    drop(mmproj_probe);

    let source: Arc<dyn TensorSource> = Arc::from(open_or_exit(&options.model, ComponentRole::Llm));
    let arch = source
        .metadata("general.architecture")
        .and_then(|value| value.to_string_val())
        .unwrap_or_default();
    crate::app::reject_incomplete_z_image_architecture(arch)?;
    eprintln!("Loading TTS talker from {}", options.model.display());
    let tokenizer = Arc::new(BPETokenizer::from_gguf_metadata(|key| {
        source.metadata(key).cloned()
    })?);
    let available = std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(1);
    let pool = Arc::new(ComputePool::new(resolve_thread_count(
        options.threads,
        available,
    )));
    let talker = Qwen3TtsTalker::from_source(source, tokenizer, pool)?;
    let max_tokens = options.max_tokens.unwrap_or(128);
    let temperature = options.temperature.unwrap_or(TTS_DEFAULT_TEMP);
    eprintln!(
        "TTS: architecture={} layers={} vocab={} audio_codebook={} eos={}",
        talker.config().architecture,
        talker.config().n_layer,
        talker.config().vocab_size,
        talker.config().audio_codebook_size,
        talker.config().eos_token_id,
    );

    eprintln!("Loading TTS codec decoder from {}", mmproj_path.display());
    let mmproj_source: Arc<dyn TensorSource> =
        Arc::from(open_or_exit(mmproj_path, ComponentRole::Mmproj));
    let speaker = if let Some(path) = options.ref_audio.as_deref() {
        eprintln!("Encoding TTS reference speaker from {}", path.display());
        let wav = std::fs::read(path)
            .map_err(|error| format!("Failed to read reference WAV {}: {error}", path.display()))?;
        let mel = reference_wav_to_mel(&wav)?;
        Some(Qwen3TtsSpeakerEncoder::from_source(mmproj_source.as_ref())?.encode(&mel)?)
    } else {
        None
    };
    let prompt = talker.prepare_prompt(prompt_text, language, speaker.as_deref())?;
    let predictor = CodePredictor::from_source(mmproj_source.as_ref())?;
    let decoder = Code2WavDecoder::from_source(mmproj_source.as_ref())?;
    let mut rng = rand::thread_rng();
    let frames = generate_tts_frames(
        &talker,
        &predictor,
        &prompt,
        max_tokens,
        temperature,
        &mut rng,
    )?;
    let waveform = decoder.decode(&frames)?;
    write_wav_f32(out_path, &waveform, WAVEFORM_SAMPLE_RATE)
        .map_err(|error| format!("WAV write failed: {error}"))?;
    eprintln!(
        "TTS: {} prompt chars, {} frames, {} samples written to {} in {:.3}s",
        prompt_text.chars().count(),
        frames.len(),
        waveform.len(),
        out_path.display(),
        started.elapsed().as_secs_f64(),
    );
    Ok(())
}

pub fn synthesize_tts_to_wav(
    model_path: &Path,
    mmproj_path: &Path,
    prompt_text: &str,
    language: &str,
    max_tokens: usize,
    temperature: f32,
    n_threads_arg: usize,
    ref_wav_bytes: Option<&[u8]>,
) -> Result<Vec<u8>, String> {
    if prompt_text.trim().is_empty() {
        return Err("TTS prompt must not be empty".into());
    }
    let source: Arc<dyn TensorSource> = Arc::from(open_or_exit(model_path, ComponentRole::Llm));
    let arch = source
        .metadata("general.architecture")
        .and_then(|value| value.to_string_val())
        .unwrap_or_default();
    crate::app::reject_incomplete_z_image_architecture(arch)?;
    let tokenizer = Arc::new(BPETokenizer::from_gguf_metadata(|key| {
        source.metadata(key).cloned()
    })?);
    let available = std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(1);
    let pool = Arc::new(ComputePool::new(resolve_thread_count(
        n_threads_arg,
        available,
    )));
    let talker = Qwen3TtsTalker::from_source(source, tokenizer, pool)?;
    let mmproj_source: Arc<dyn TensorSource> =
        Arc::from(open_or_exit(mmproj_path, ComponentRole::Mmproj));
    let speaker = if let Some(wav) = ref_wav_bytes {
        let mel = reference_wav_to_mel(wav)?;
        Some(Qwen3TtsSpeakerEncoder::from_source(mmproj_source.as_ref())?.encode(&mel)?)
    } else {
        None
    };
    let prompt = talker.prepare_prompt(prompt_text, language, speaker.as_deref())?;
    let predictor = CodePredictor::from_source(mmproj_source.as_ref())?;
    let decoder = Code2WavDecoder::from_source(mmproj_source.as_ref())?;
    let mut rng = rand::thread_rng();
    let frames = generate_tts_frames(
        &talker,
        &predictor,
        &prompt,
        max_tokens,
        temperature,
        &mut rng,
    )?;
    let waveform = decoder.decode(&frames)?;
    encode_wav_pcm16(&waveform, WAVEFORM_SAMPLE_RATE)
        .map_err(|error| format!("WAV encode failed: {error}"))
}

fn drive_frames(
    max_frames: usize,
    mut sample: impl FnMut() -> Option<u32>,
    mut predict: impl FnMut(u32) -> Result<(), String>,
) -> Result<(), String> {
    for _ in 0..max_frames {
        let Some(semantic) = sample() else {
            break;
        };
        predict(semantic)?;
    }
    Ok(())
}

fn generate_tts_frames<R: rand::Rng + ?Sized>(
    talker: &Qwen3TtsTalker,
    predictor: &CodePredictor,
    prompt: &TtsPrompt,
    max_frames: usize,
    temperature: f32,
    rng: &mut R,
) -> Result<Vec<[u32; 16]>, String> {
    if prompt.overlay.is_empty() {
        return Err("TTS prompt overlay must not be empty".into());
    }
    let mut session = talker.new_session()?;
    let t_prefill = std::time::Instant::now();
    session.prefill_prompt(prompt)?;
    eprintln!(
        "  [frame_loop] prefill_prompt took {:.3}s",
        t_prefill.elapsed().as_secs_f64()
    );
    let next_semantic = Cell::new(session.sample_semantic(temperature, rng)?);
    let mut frames = Vec::with_capacity(max_frames);
    let mut t_total = std::time::Instant::now();
    let mut t_hidden = std::time::Duration::ZERO;
    let mut t_codec = std::time::Duration::ZERO;
    let mut t_tts = std::time::Duration::ZERO;
    let mut t_sample = std::time::Duration::ZERO;
    drive_frames(
        max_frames,
        || next_semantic.take(),
        |semantic| {
            let frame_index = frames.len();
            let t_h = std::time::Instant::now();
            let hidden = session.hidden_state().to_vec();
            t_hidden += t_h.elapsed();
            let t_c = std::time::Instant::now();
            let (frame, mut feedback) =
                predictor.predict_frame(&hidden, semantic, predictor_top_k(temperature), rng)?;
            t_codec += t_c.elapsed();
            let overlay = &prompt.overlay[frame_index.min(prompt.overlay.len() - 1)];
            if feedback.len() != overlay.len() {
                return Err(format!(
                    "TTS feedback length {} != overlay length {}",
                    feedback.len(),
                    overlay.len()
                ));
            }
            for (value, text) in feedback.iter_mut().zip(overlay) {
                *value += *text;
            }
            let position = prompt
                .positions
                .len()
                .checked_add(frame_index)
                .ok_or_else(|| "TTS frame position overflow".to_string())?;
            frames.push(frame);
            let t_t = std::time::Instant::now();
            session.forward_step_with_embedding(&feedback, [position; 4])?;
            t_tts += t_t.elapsed();
            let t_s = std::time::Instant::now();
            next_semantic.set(session.sample_semantic(temperature, rng)?);
            t_sample += t_s.elapsed();
            if frame_index % 5 == 0 {
                eprintln!("  [frame_loop] frame {} done, hidden={:.3}s codec={:.3}s tts={:.3}s sample={:.3}s", 
                    frame_index, t_hidden.as_secs_f64(), t_codec.as_secs_f64(), t_tts.as_secs_f64(), t_sample.as_secs_f64());
            }
            Ok(())
        },
    )?;
    eprintln!(
        "  [frame_loop] total={:.3}s hidden={:.3}s codec={:.3}s tts={:.3}s sample={:.3}s",
        t_total.elapsed().as_secs_f64(),
        t_hidden.as_secs_f64(),
        t_codec.as_secs_f64(),
        t_tts.as_secs_f64(),
        t_sample.as_secs_f64()
    );
    Ok(frames)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_loop_keeps_first_and_last_non_eos_frames() {
        let mut semantic = vec![Some(7), Some(8), None].into_iter();
        let mut predicted = Vec::new();
        drive_frames(
            8,
            || semantic.next().unwrap(),
            |code0| {
                let mut frame = [code0; 16];
                frame[15] = code0 + 100;
                predicted.push(frame);
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(predicted.len(), 2);
        assert_eq!(predicted[0][0], 7);
        assert_eq!(predicted[1][15], 108);
    }
}
