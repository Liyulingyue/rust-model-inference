mod breeze;
pub(crate) mod dots;

use crate::app::cli::{normalize_tts_language, resolve_thread_count};
use crate::app::open_or_exit;
use crate::core::tensor::TensorSource;
use crate::core::thread_pool::ComputePool;
use crate::core::tokenizer::BPETokenizer;
use crate::format::ggufrs::ComponentRole;
use crate::format::wav::{encode_wav_pcm16_channels, write_wav_f32_channels};
use crate::models::qwen3::tts::codec::{Code2WavDecoder, CodePredictor, WAVEFORM_SAMPLE_RATE};
use crate::models::qwen3::tts::speaker::{reference_wav_to_mel, Qwen3TtsSpeakerEncoder};
use crate::models::qwen3::tts::{predictor_top_k, Qwen3TtsTalker, TtsPrompt, TTS_DEFAULT_TEMP};
use std::cell::Cell;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

/// The warning for a PiVo file whose weights are stored quantized.
///
/// PiVo computes in F32 either way, so the quantized export only changes
/// the file size. The audio it produces is not usable: the FARGAN feedback
/// loop turns the 1/127 weight grid into a carrier phase drift, and the
/// frame durations and pitch periods stop matching the official graph. The
/// measurement is in tools/oracle/pivo/validation.md.
pub(crate) fn pivo_format_notice(
    format: crate::models::pivo::WeightFormat,
) -> Option<&'static str> {
    match format {
        crate::models::pivo::WeightFormat::F32 => None,
        crate::models::pivo::WeightFormat::Q8_0 => Some(
            "PiVo: this file stores the weights in Q8_0; the audio it produces is \
             not usable, because the FARGAN feedback loop turns the 1/127 weight \
             grid into a carrier phase drift. Use the F32 export; see \
             tools/oracle/pivo/validation.md",
        ),
    }
}

pub(crate) fn pivo_request(
    options: &crate::app::cli::CliOptions,
) -> Result<crate::models::pivo::SpeechRequest, String> {
    if options.mmproj.is_some()
        || options.ref_audio.is_some()
        || options.ref_text.is_some()
        || options.edit
        || options.instruction.is_some()
        || options.temperature.is_some()
        || options.top_k.is_some()
        || options.top_p.is_some()
        || options.cfg_scale.is_some()
        || options.steps.is_some()
        || options.max_tokens.is_some()
        || options.jev
        || options.rerank
        || options.clm_head.is_some()
        || options.laya_request.is_some()
        || options.seed.is_some()
        || options.repetition_penalty.is_some()
        || options.kv_format_explicit
        || options.jinja
        || options.chat_template.is_some()
        || options.chat_template_file.is_some()
        || options.thinking
        || options.tts_model.is_some()
        || options.tts_mmproj.is_some()
        || options.max_context.is_some()
        || options.prefill_batch_size.is_some()
        || options.negative_prompt.is_some()
        || options.vae.is_some()
        || options.text_encoder.is_some()
        || !options.references.is_empty()
        || options.cfg.is_some()
        || options.noise.is_some()
        || options.resolution.is_some()
        || options.duration_seconds.is_some()
        || options.dry_run
    {
        return Err("PiVo --tts uses its bundled acoustic/vocoder weights; projector, reference/edit, sampling and other task options are unsupported".into());
    }
    if options
        .language
        .as_deref()
        .is_some_and(|v| !matches!(v.to_ascii_lowercase().as_str(), "en" | "english"))
    {
        return Err("PiVo supports English only".into());
    }
    if options
        .voice
        .as_deref()
        .is_some_and(|v| !matches!(v, "Pi" | "Vo"))
    {
        return Err("PiVo voice must be Pi or Vo".into());
    }
    let overrides = options
        .pronunciations
        .as_deref()
        .map(serde_json::from_str)
        .transpose()
        .map_err(|e| format!("Invalid --pronunciations JSON: {e}"))?
        .unwrap_or_default();
    Ok(crate::models::pivo::SpeechRequest {
        input: options.prompt.clone().unwrap_or_default(),
        voice: options.voice.clone().unwrap_or_else(|| "Pi".into()),
        speed: options.speed.unwrap_or(1.),
        overrides,
    })
}

enum TtsFrontend<'a> {
    ArkTts,
    Breeze,
    Dots,
    Qwen3 {
        prompt_text: &'a str,
        language: &'static str,
    },
}

fn tts_frontend<'a>(
    options: &'a crate::app::cli::CliOptions,
    model_arch: &str,
    codec_arch: &str,
    is_dots_mmproj: bool,
) -> Result<TtsFrontend<'a>, String> {
    // PiVo consumes these in its own request; every other TTS frontend would
    // silently ignore them, so reject before routing to a model.
    if options.voice.is_some() || options.speed.is_some() || options.pronunciations.is_some() {
        return Err("--voice/--speed/--pronunciations require a PiVo TTS model".into());
    }
    // The bridge pair belongs to the multimodal post-processor; `--tts`
    // never runs it.
    if options.tts_model.is_some() || options.tts_mmproj.is_some() {
        return Err(
            "--tts-model/--tts-mmproj are only used by the multimodal post-processor".into(),
        );
    }
    if crate::models::arktts::is_arktts_arch(model_arch)
        || crate::models::arktts::is_arktts_arch(codec_arch)
    {
        if model_arch != "arktts" || codec_arch != "arktts_codec" {
            return Err(
                "Audio8 TTS requires architecture arktts with an arktts_codec --mmproj".into(),
            );
        }
        arktts_sampling(options)?;
        return Ok(TtsFrontend::ArkTts);
    }
    if model_arch == "breeze" || codec_arch == "breeze_audio" {
        if model_arch != "breeze" || codec_arch != "breeze_audio" {
            return Err(
                "Breeze --tts requires architecture breeze with a breeze_audio --mmproj".into(),
            );
        }
        return Ok(TtsFrontend::Breeze);
    }
    if options.cfg_scale.is_some() {
        return Err("--cfg-scale requires a Breeze TTS model".into());
    }
    if options.top_k.is_some() || options.top_p.is_some() {
        return Err("--top-k/--top-p require a Breeze TTS model".into());
    }
    if options.instruction.is_some() && !(is_dots_mmproj && options.edit) {
        return Err("--instruction requires Breeze --tts or dots --tts --edit".into());
    }
    if is_dots_mmproj {
        return Ok(TtsFrontend::Dots);
    }
    let prompt_text = options
        .prompt
        .as_deref()
        .filter(|prompt| !prompt.trim().is_empty())
        .ok_or_else(|| "--tts requires --prompt".to_string())?;
    let language = normalize_tts_language(options.language.as_deref())?;
    Ok(TtsFrontend::Qwen3 {
        prompt_text,
        language,
    })
}

pub fn run_tts_cli(options: &crate::app::cli::CliOptions) -> Result<(), String> {
    let started = Instant::now();
    let source: Arc<dyn TensorSource> = Arc::from(open_or_exit(&options.model, ComponentRole::Llm));
    let arch = source
        .metadata("general.architecture")
        .and_then(|v| v.to_string_val())
        .unwrap_or_default();
    if arch == "pivo" {
        let request = pivo_request(options)?;
        let model =
            crate::models::pivo::Model::from_source(source.as_ref(), options.compute_policy())?;
        let waveform = model.synthesize(&request).map_err(|e| e.to_string())?;
        write_wav_f32_channels(
            options.out.as_deref().ok_or("--tts requires --out")?,
            &waveform,
            crate::models::pivo::SAMPLE_RATE,
            1,
        )?;
        eprintln!(
            "PiVo: format={} voice={} samples={} elapsed={:.3}s",
            model.format.name(),
            request.voice,
            waveform.len(),
            started.elapsed().as_secs_f64()
        );
        if let Some(notice) = pivo_format_notice(model.format) {
            eprintln!("{notice}");
        }
        return Ok(());
    }
    let mmproj_path = options
        .mmproj
        .as_deref()
        .filter(|path| !path.as_os_str().is_empty())
        .ok_or_else(|| "--tts requires --mmproj".to_string())?;
    let out_path = options
        .out
        .as_deref()
        .ok_or_else(|| "--tts requires --out".to_string())?;

    // dots.tts uses an arch-qwen2 LLM gguf + a clip mmproj; dispatch on the
    // exact projector pair before the Qwen3-TTS path.
    let mmproj_probe = open_or_exit(mmproj_path, ComponentRole::Mmproj);
    let codec_arch = mmproj_probe
        .metadata("general.architecture")
        .and_then(|value| value.to_string_val())
        .unwrap_or_default();
    let (prompt_text, language) = match tts_frontend(
        options,
        arch,
        codec_arch,
        crate::models::dots::is_dots_tts_mmproj(mmproj_probe.as_ref()),
    )? {
        TtsFrontend::ArkTts => {
            return run_arktts_cli(options, source.as_ref(), mmproj_probe.as_ref())
        }
        TtsFrontend::Breeze => {
            return breeze::run_breeze_tts_cli(options, source.as_ref(), mmproj_probe.as_ref())
        }
        TtsFrontend::Dots => return dots::run_dots_tts_cli(options),
        TtsFrontend::Qwen3 {
            prompt_text,
            language,
        } => (prompt_text, language),
    };
    drop(mmproj_probe);

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
    write_wav_f32_channels(out_path, &waveform, WAVEFORM_SAMPLE_RATE, 1)?;
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

pub(crate) fn arktts_sampling(
    options: &crate::app::cli::CliOptions,
) -> Result<crate::models::arktts::Sampling, String> {
    crate::models::arktts::validate_compute(options.compute_policy())?;
    if options.language.is_some()
        || options.instruction.is_some()
        || options.cfg_scale.is_some()
        || options.edit
        || options.steps.is_some()
        || options.source_audio.is_some()
    {
        return Err("Audio8 TTS detects language from text and does not support --language, --instruction, --cfg-scale, --edit, --steps or --source-audio".into());
    }
    if options.ref_audio.is_some()
        != options
            .ref_text
            .as_deref()
            .is_some_and(|s| !s.trim().is_empty())
    {
        return Err(
            "Audio8 voice cloning requires both --ref-audio and nonempty --ref-text".into(),
        );
    }
    if options.jev
        || options.jinja
        || options.chat_template.is_some()
        || options.chat_template_file.is_some()
        || options.thinking
        || options.max_context.is_some()
        || options.prefill_batch_size.is_some()
        || options.repetition_penalty.is_some()
        || options.kv_format_explicit
    {
        return Err("Audio8 TTS does not support JEV, chat templates, thinking, or text/KV generation controls".into());
    }
    // These pick a different task before `--tts` is dispatched, so accepting
    // them would silently drop the request.
    if options.rerank
        || options.clm_head.is_some()
        || options.laya_request.is_some()
        || options.tts_model.is_some()
        || options.tts_mmproj.is_some()
    {
        return Err(
            "Audio8 TTS cannot be combined with --rerank, --clm-head, --laya-request or --tts-model"
                .into(),
        );
    }
    if options
        .max_tokens
        .is_some_and(|frames| frames > crate::models::arktts::CONTEXT)
    {
        return Err(format!(
            "Audio8 TTS --max-tokens is a frame budget of at most {}",
            crate::models::arktts::CONTEXT
        ));
    }
    let defaults = crate::models::arktts::Sampling::default();
    let sampling = crate::models::arktts::Sampling {
        temperature: options.temperature.unwrap_or(defaults.temperature),
        top_k: options.top_k.unwrap_or(defaults.top_k),
        top_p: options.top_p.unwrap_or(defaults.top_p),
        seed: options.seed.map_or(defaults.seed, |s| s as u64),
    };
    sampling.validate()?;
    Ok(sampling)
}

pub(crate) fn reference_wav_at_rate(bytes: &[u8], rate: u32) -> Result<Vec<f32>, String> {
    let decoded = crate::ops::audio::frontend::decode_pcm16_wav_any(bytes)
        .map_err(|e| format!("invalid reference PCM16 WAV: {e:?}"))?;
    // Bound the sinc phase table before constructing it for untrusted WAV metadata.
    if ![
        8_000, 11_025, 16_000, 22_050, 24_000, 32_000, 44_100, 48_000, 88_200, 96_000, 176_400,
        192_000,
    ]
    .contains(&decoded.sample_rate)
    {
        return Err(
            "unsupported reference WAV sample rate; use a standard rate from 8000 to 192000 Hz"
                .into(),
        );
    }
    let frames = decoded.samples.len() / decoded.channels as usize;
    if (frames as u64 * u64::from(rate)).div_ceil(u64::from(decoded.sample_rate))
        > 2048 * crate::models::arktts::SAMPLES_PER_FRAME as u64
    {
        return Err("reference WAV exceeds 2048 codec frames".into());
    }
    let mono = decoded
        .samples
        .chunks_exact(decoded.channels as usize)
        .map(|frame| frame.iter().sum::<f32>() / decoded.channels as f32)
        .collect::<Vec<_>>();
    Ok(if decoded.sample_rate == rate {
        mono
    } else {
        crate::models::dots::speaker::Resampler::new(decoded.sample_rate, rate).resample(&mono)
    })
}

fn run_arktts_cli(
    options: &crate::app::cli::CliOptions,
    source: &dyn TensorSource,
    codec_source: &dyn TensorSource,
) -> Result<(), String> {
    use crate::models::arktts::{codec::Codec, Model, SAMPLE_RATE};
    let sampling = arktts_sampling(options)?;
    let prompt = options
        .prompt
        .as_deref()
        .ok_or("Audio8 TTS requires --prompt")?;
    let out = options.out.as_deref().ok_or("Audio8 TTS requires --out")?;
    if out.symlink_metadata().is_ok() {
        return Err(format!("output already exists: {}", out.display()));
    }
    let threads = resolve_thread_count(
        options.threads,
        std::thread::available_parallelism().map_or(1, |n| n.get()),
    );
    crate::models::arktts::validate_pair(source, codec_source)?;
    let model = Model::from_source_with_compute(source, threads, options.compute_policy())?;
    let codec = Codec::from_source_with_compute(codec_source, threads, options.compute_policy())?;
    let reference = options
        .ref_audio
        .as_deref()
        .map(|path| {
            let bytes =
                std::fs::read(path).map_err(|e| format!("failed to read reference WAV: {e}"))?;
            reference_wav_at_rate(&bytes, SAMPLE_RATE)
        })
        .transpose()?;
    let waveform = model.synthesize(
        &codec,
        prompt,
        options.ref_text.as_deref().zip(reference.as_deref()),
        options.max_tokens.unwrap_or(128),
        sampling,
    )?;
    crate::models::diffusion::dreamx::media::write_wav_atomic(out, &waveform, SAMPLE_RATE, false)?;
    eprintln!(
        "Audio8 TTS: {} frames / {} samples at {SAMPLE_RATE} Hz written to {}",
        waveform.len() / crate::models::arktts::SAMPLES_PER_FRAME,
        waveform.len(),
        out.display()
    );
    Ok(())
}

pub fn synthesize_tts_to_wav(
    model_path: &Path,
    mmproj_path: &Path,
    prompt_text: &str,
    language: &str,
    max_tokens: usize,
    temperature: Option<f32>,
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
    if crate::models::arktts::is_arktts_arch(&arch) {
        if ref_wav_bytes.is_some() {
            return Err(
                "Audio8 voice cloning requires a transcript; use --tts --ref-audio with --ref-text"
                    .into(),
            );
        }
        let policy = crate::compute::ComputePolicy::legacy();
        crate::models::arktts::validate_compute(policy)?;
        let codec_source = open_or_exit(mmproj_path, ComponentRole::Mmproj);
        crate::models::arktts::validate_pair(source.as_ref(), codec_source.as_ref())?;
        let threads = resolve_thread_count(
            n_threads_arg,
            std::thread::available_parallelism().map_or(1, |n| n.get()),
        );
        let model = crate::models::arktts::Model::from_source_with_compute(
            source.as_ref(),
            threads,
            policy,
        )?;
        let codec = crate::models::arktts::codec::Codec::from_source_with_compute(
            codec_source.as_ref(),
            threads,
            policy,
        )?;
        let default = crate::models::arktts::Sampling::default();
        let waveform = model.synthesize(
            &codec,
            prompt_text,
            None,
            crate::models::arktts::frame_budget(max_tokens),
            crate::models::arktts::Sampling {
                // An unset temperature keeps the model default instead of the
                // multimodal path's greedy fallback.
                temperature: temperature.unwrap_or(default.temperature),
                ..default
            },
        )?;
        return encode_wav_pcm16_channels(&waveform, crate::models::arktts::SAMPLE_RATE, 1);
    }
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
        temperature.unwrap_or(0.0),
        &mut rng,
    )?;
    let waveform = decoder.decode(&frames)?;
    encode_wav_pcm16_channels(&waveform, WAVEFORM_SAMPLE_RATE, 1)
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
    let t_total = std::time::Instant::now();
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
    fn arktts_routes_its_exact_codec_pair_before_qwen() {
        let mut options = crate::app::cli::CliOptions::default();
        assert!(tts_frontend(&options, "arktts", "arktts_codec", false).is_ok());
        assert!(tts_frontend(&options, "arktts", "clip", false).is_err());
        assert!(tts_frontend(&options, "qwen3", "arktts_codec", false).is_err());
        options.jev = true;
        assert!(tts_frontend(&options, "arktts", "arktts_codec", false).is_err());
    }

    #[test]
    fn arktts_rejects_flags_that_would_be_dropped() {
        assert!(arktts_sampling(&crate::app::cli::CliOptions::default()).is_ok());

        for mutate in [
            |options: &mut crate::app::cli::CliOptions| options.rerank = true,
            |options: &mut crate::app::cli::CliOptions| options.clm_head = Some("head.bin".into()),
            |options: &mut crate::app::cli::CliOptions| {
                options.laya_request = Some("request.json".into());
            },
            |options: &mut crate::app::cli::CliOptions| {
                options.tts_model = Some("bridge.gguf".into());
            },
            |options: &mut crate::app::cli::CliOptions| {
                options.max_tokens = Some(crate::models::arktts::CONTEXT + 1);
            },
        ] {
            let mut options = crate::app::cli::CliOptions::default();
            mutate(&mut options);
            assert!(arktts_sampling(&options).is_err(), "{options:?}");
        }

        let mut options = crate::app::cli::CliOptions::default();
        options.max_tokens = Some(crate::models::arktts::CONTEXT);
        assert!(arktts_sampling(&options).is_ok());

        let mut options = crate::app::cli::CliOptions::default();
        options.voice = Some("Vo".into());
        assert!(tts_frontend(&options, "arktts", "arktts_codec", false)
            .err()
            .is_some_and(|error| error.contains("PiVo")));
        assert!(tts_frontend(&options, "qwen3", "clip", false).is_err());

        let mut options = crate::app::cli::CliOptions::default();
        options.tts_model = Some("bridge.gguf".into());
        assert!(tts_frontend(&options, "qwen3", "clip", false).is_err());
        assert!(tts_frontend(&options, "breeze", "breeze_audio", false).is_err());
    }

    #[test]
    fn arktts_reference_validates_rate_before_resampling() {
        for rate in [1, 44_101] {
            let wav = encode_wav_pcm16_channels(&[0.0; 2], rate, 1).unwrap();
            assert!(reference_wav_at_rate(&wav, 44_100)
                .unwrap_err()
                .contains("sample rate"));
        }
        let wav = encode_wav_pcm16_channels(&[0.25, -0.5], 44_100, 1).unwrap();
        let samples = reference_wav_at_rate(&wav, 44_100).unwrap();
        assert!(samples
            .iter()
            .zip([0.25, -0.5])
            .all(|(a, b)| (a - b).abs() <= 1.0 / 32768.0));
        let wav = encode_wav_pcm16_channels(&[0.0; 240], 48_000, 1).unwrap();
        let samples = reference_wav_at_rate(&wav, 44_100).unwrap();
        assert_eq!(samples.len(), 221);
        assert!(samples.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn pivo_rejects_options_that_cannot_affect_speech() {
        for extra in [
            vec!["--resolution", "48000"],
            vec!["--size", "48000"],
            vec!["--duration", "10"],
            vec!["--dry-run"],
            vec!["--seed", "42"],
            vec!["--repetition-penalty", "1.1"],
            vec!["--kv-cache", "f16"],
            vec!["--jinja"],
            vec!["--thinking"],
            vec!["--tts-model", "other.gguf"],
            vec!["--max-tokens", "10"],
        ] {
            let mut args = vec!["rmi", "--tts", "--voice", "Vo", "--speed", "1.5"];
            args.extend(extra);
            let options = crate::app::cli::parse_cli_options(
                &args.into_iter().map(str::to_string).collect::<Vec<_>>(),
            )
            .unwrap();
            assert!(pivo_request(&options).is_err(), "{options:?}");
        }
        let options = crate::app::cli::CliOptions {
            voice: Some("Vo".into()),
            speed: Some(1.5),
            ..Default::default()
        };
        let request = pivo_request(&options).unwrap();
        assert_eq!(request.voice, "Vo");
        assert_eq!(request.speed, 1.5);
        assert!(tts_frontend(&options, "qwen3", "clip", false).is_err());
    }

    #[test]
    fn legal_dots_edit_routes_before_base_prompt_requirements() {
        let args = [
            "rmi",
            "--tts",
            "--edit",
            "--model",
            "edit.gguf",
            "--mmproj",
            "edit-mmproj.gguf",
            "--source-audio",
            "source.wav",
            "--instruction",
            "Hello <sub targ=\"small\">brave</sub> world.",
            "--out",
            "edited.wav",
        ]
        .map(str::to_string);
        let options = crate::app::cli::parse_cli_options(&args).unwrap();
        crate::app::cli::validate_cli_options(&options).unwrap();

        assert!(matches!(
            tts_frontend(&options, "qwen2", "clip", true).unwrap(),
            TtsFrontend::Dots
        ));
    }

    #[test]
    fn breeze_routes_before_qwen_tokenizer_and_rejects_mismatched_components() {
        let options = crate::app::cli::CliOptions::default();
        assert!(matches!(
            tts_frontend(&options, "breeze", "breeze_audio", false).unwrap(),
            TtsFrontend::Breeze
        ));
        assert!(tts_frontend(&options, "breeze", "clip", true).is_err());
        assert!(tts_frontend(&options, "qwen3", "breeze_audio", false).is_err());
        let options = crate::app::cli::CliOptions {
            prompt: Some("hello".into()),
            instruction: Some("soft voice".into()),
            ..Default::default()
        };
        assert!(tts_frontend(&options, "qwen3", "clip", false).is_err());
        assert!(tts_frontend(&options, "qwen2", "clip", true).is_err());
        let options = crate::app::cli::CliOptions {
            top_k: Some(50),
            ..Default::default()
        };
        assert!(tts_frontend(&options, "qwen3", "clip", false).is_err());
    }

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
