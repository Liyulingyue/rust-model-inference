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

pub(crate) fn is_underdog_woof_model(path: &Path) -> bool {
    crate::format::ggufrs::open_model_source(path, ComponentRole::Llm)
        .ok()
        .is_some_and(|source| {
            source
                .metadata("general.architecture")
                .and_then(|value| value.to_string_val())
                == Some(crate::models::underdog_woof::ARCHITECTURE)
        })
}

pub(crate) fn underdog_woof_request(
    options: &crate::app::cli::CliOptions,
) -> Result<crate::models::underdog_woof::SpeechRequest, String> {
    for flag in &options.supplied_flags {
        let allowed = matches!(
            flag.as_str(),
            "--tts"
                | "--model"
                | "--voice"
                | "--seed"
                | "--temperature"
                | "--temp"
                | "--threads"
                | "--compute"
                | "--gpu"
        ) || (!options.serve && matches!(flag.as_str(), "--prompt" | "--out"))
            || (options.serve && flag == "--serve");
        if !allowed {
            return Err(format!("Underdog Woof --tts does not support {flag}"));
        }
    }
    let seed = u32::try_from(options.seed.unwrap_or(0))
        .map_err(|_| "Underdog Woof --seed must be in 0..=4294967295".to_string())?;
    let request = crate::models::underdog_woof::SpeechRequest {
        input: options.prompt.clone().unwrap_or_default(),
        voice: options.voice.clone().unwrap_or_else(|| "juno".into()),
        seed,
        temperature: options.temperature.unwrap_or(0.5),
        normalize: true,
    };
    request.validate()?;
    Ok(request)
}

enum TtsFrontend<'a> {
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
    if options.voice.is_some() || options.speed.is_some() || options.pronunciations.is_some() {
        return Err("--voice/--speed/--pronunciations require a PiVo TTS model".into());
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
    let write_waveform = |waveform: &[f32], sample_rate| {
        write_wav_f32_channels(
            options.out.as_deref().ok_or("--tts requires --out")?,
            waveform,
            sample_rate,
            1,
        )
    };
    if arch == "pivo" {
        let request = pivo_request(options)?;
        let model =
            crate::models::pivo::Model::from_source(source.as_ref(), options.compute_policy())?;
        let waveform = model.synthesize(&request).map_err(|e| e.to_string())?;
        write_waveform(&waveform, crate::models::pivo::SAMPLE_RATE)?;
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
    if arch == crate::models::underdog_woof::ARCHITECTURE {
        let request = underdog_woof_request(options)?;
        let available = std::thread::available_parallelism()
            .map(std::num::NonZeroUsize::get)
            .unwrap_or(1);
        let model = crate::models::underdog_woof::Model::from_source(
            source,
            resolve_thread_count(options.threads, available),
            options.compute_policy(),
        )?;
        let waveform = model
            .synthesize(&request)
            .map_err(|error| error.to_string())?;
        write_waveform(&waveform, crate::models::underdog_woof::SAMPLE_RATE)?;
        eprintln!(
            "Underdog Woof: format={} voice={} seed={} samples={} elapsed={:.3}s",
            model.format,
            request.voice,
            request.seed,
            waveform.len(),
            started.elapsed().as_secs_f64()
        );
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
pub(crate) mod tests {
    use super::*;

    pub(crate) struct WoofMetadataFixture(pub std::path::PathBuf);

    impl WoofMetadataFixture {
        pub(crate) fn new() -> Self {
            use std::io::Write;
            let nonce = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let path =
                std::env::temp_dir().join(format!("woof-{}-{nonce}.gguf", std::process::id()));
            let mut bytes = b"GGUF".to_vec();
            bytes.extend(3u32.to_le_bytes());
            bytes.extend(0u64.to_le_bytes());
            bytes.extend(1u64.to_le_bytes());
            let key = "general.architecture";
            bytes.extend((key.len() as u64).to_le_bytes());
            bytes.extend(key.as_bytes());
            bytes.extend(8u32.to_le_bytes());
            let value = crate::models::underdog_woof::ARCHITECTURE;
            bytes.extend((value.len() as u64).to_le_bytes());
            bytes.extend(value.as_bytes());
            bytes.resize(bytes.len().div_ceil(32) * 32, 0);
            std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
                .unwrap()
                .write_all(&bytes)
                .unwrap();
            Self(path)
        }
    }

    impl Drop for WoofMetadataFixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    fn woof_options(extra: &[&str]) -> crate::app::cli::CliOptions {
        let mut args = vec!["rmi", "--tts", "--prompt", "Hi.", "--out", "speech.wav"];
        args.extend_from_slice(extra);
        crate::app::cli::parse_cli_options(
            &args.into_iter().map(str::to_string).collect::<Vec<_>>(),
        )
        .unwrap()
    }

    #[test]
    fn underdog_woof_cli_uses_voice_seed_and_temperature() {
        let options = woof_options(&[
            "--voice",
            "mira",
            "--seed",
            "4294967295",
            "--temperature",
            "0.25",
        ]);
        let request = underdog_woof_request(&options).unwrap();
        assert_eq!(request.input, "Hi.");
        assert_eq!(request.voice, "mira");
        assert_eq!(request.seed, u32::MAX);
        assert_eq!(request.temperature, 0.25);
        assert!(request.normalize);
        assert_eq!(
            underdog_woof_request(&woof_options(&["--temp", "0.75"]))
                .unwrap()
                .temperature,
            0.75
        );
        for extra in [
            vec!["--seed", "-1"],
            vec!["--seed", "4294967296"],
            vec!["--voice", "Pi"],
            vec!["--temperature", "NaN"],
        ] {
            assert!(underdog_woof_request(&woof_options(&extra)).is_err());
        }
    }

    #[test]
    fn underdog_woof_rejects_every_irrelevant_cli_option() {
        for extra in [
            vec!["--mmproj", "unused.gguf"],
            vec!["--ref-audio", "voice.wav"],
            vec!["--ref-text", "reference"],
            vec!["--edit"],
            vec!["--speed", "1.0"],
            vec!["--pronunciations", "{}"],
            vec!["--kv-cache", "f32"],
            vec!["--max-tokens", "10"],
            vec!["--max-context", "1100"],
            vec!["--top-k", "10"],
            vec!["--language", "en"],
            vec!["--dry-run"],
            vec!["--tts-model", "unused.gguf"],
            vec!["--embedding"],
            vec!["--profile"],
        ] {
            let options = woof_options(&extra);
            let error = underdog_woof_request(&options).unwrap_err();
            assert!(error.contains(extra[0]), "{error}");
        }
        let mut server = woof_options(&["--serve"]);
        server.serve = true;
        assert!(underdog_woof_request(&server)
            .unwrap_err()
            .contains("--prompt"));
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
