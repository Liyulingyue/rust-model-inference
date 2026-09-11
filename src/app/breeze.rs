use crate::app::cli::{resolve_thread_count, CliOptions};
use crate::core::tensor::TensorSource;
use crate::models::breeze::{BreezeCodec, BreezeModel};
use crate::models::diffusion::dreamx::media::write_wav_atomic;
use crate::models::dots::speaker::Resampler;
use crate::models::qwen3::asr::audio_processor::decode_pcm16_wav_any;
use crate::models::qwen3::tts::codec::WAVEFORM_SAMPLE_RATE;
use std::time::Instant;

fn validate_breeze_options(options: &CliOptions) -> Result<f32, String> {
    let temperature = options.temperature.unwrap_or(0.9);
    if !temperature.is_finite() || temperature < 0.0 {
        return Err("Breeze --temperature must be finite and nonnegative".into());
    }
    if options
        .top_p
        .is_some_and(|p| !p.is_finite() || p <= 0.0 || p > 1.0)
    {
        return Err("Breeze --top-p must be finite and in (0, 1]".into());
    }
    let unsupported = if options.edit {
        Some("--edit")
    } else if options.steps.is_some() {
        Some("--steps")
    } else if options.language.is_some() {
        Some("--language (Breeze detects language from the prompt)")
    } else if options.gpu {
        Some("--gpu")
    } else {
        None
    };
    if let Some(flag) = unsupported {
        return Err(format!("Breeze --tts does not support {flag}"));
    }
    let ref_text = options
        .ref_text
        .as_deref()
        .filter(|text| !text.trim().is_empty());
    if options.ref_audio.is_some() != ref_text.is_some() {
        return Err(
            "Breeze voice cloning requires both --ref-audio and non-empty --ref-text".into(),
        );
    }
    let has_instruction = options
        .instruction
        .as_deref()
        .is_some_and(|text| !text.trim().is_empty());
    let cfg_scale = options
        .cfg_scale
        .unwrap_or(if has_instruction { 3.0 } else { 1.0 });
    if !cfg_scale.is_finite() || cfg_scale <= 0.0 {
        return Err("--cfg-scale must be finite and greater than zero".into());
    }
    if cfg_scale != 1.0 && !has_instruction {
        return Err("Breeze --cfg-scale other than 1 requires non-empty --instruction".into());
    }
    Ok(cfg_scale)
}

fn reference_wav_to_24k(bytes: &[u8]) -> Result<Vec<f32>, String> {
    let decoded = decode_pcm16_wav_any(bytes)
        .map_err(|error| format!("Failed to decode Breeze reference WAV: {error:?}"))?;
    let mono = if decoded.channels == 1 {
        decoded.samples
    } else {
        decoded
            .samples
            .chunks_exact(usize::from(decoded.channels))
            .map(|frame| frame.iter().sum::<f32>() / f32::from(decoded.channels))
            .collect()
    };
    Ok(if decoded.sample_rate == WAVEFORM_SAMPLE_RATE {
        mono
    } else {
        Resampler::new(decoded.sample_rate, WAVEFORM_SAMPLE_RATE).resample(&mono)
    })
}

pub(crate) fn run_breeze_tts_cli(
    options: &CliOptions,
    source: &dyn TensorSource,
    codec_source: &dyn TensorSource,
) -> Result<(), String> {
    let started = Instant::now();
    let cfg_scale = validate_breeze_options(options)?;
    let prompt = options
        .prompt
        .as_deref()
        .filter(|text| !text.trim().is_empty())
        .ok_or("Breeze --tts requires a non-empty --prompt")?;
    let out = options
        .out
        .as_deref()
        .ok_or("Breeze --tts requires --out")?;
    if out.symlink_metadata().is_ok() {
        return Err(format!("Breeze output already exists: {}", out.display()));
    }
    let available = std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(1);
    let threads = resolve_thread_count(options.threads, available);
    let temperature = options.temperature.unwrap_or(0.9);
    eprintln!(
        "Loading Breeze TTS and codec (threads={threads}, temperature={temperature}, cfg_scale={cfg_scale})"
    );
    let codec = BreezeCodec::from_source(codec_source)?;
    let reference_codes = options
        .ref_audio
        .as_deref()
        .map(|path| {
            let bytes = std::fs::read(path).map_err(|error| {
                format!("Failed to read reference WAV {}: {error}", path.display())
            })?;
            codec.encode(&reference_wav_to_24k(&bytes)?)
        })
        .transpose()?;
    let reference = options.ref_text.as_deref().zip(reference_codes.as_deref());
    let model = BreezeModel::from_source(source, threads)?;
    let instruction = options
        .instruction
        .as_deref()
        .filter(|text| !text.trim().is_empty());
    let frames = model.generate_with_sampling(
        prompt,
        instruction,
        reference,
        options.max_tokens.unwrap_or(128),
        cfg_scale,
        temperature,
        options.top_k.unwrap_or(50),
        options.top_p.unwrap_or(1.0),
        options.seed.unwrap_or(42) as u64,
    )?;
    let waveform = codec.decode(&frames)?;
    write_wav_atomic(out, &waveform, WAVEFORM_SAMPLE_RATE, false)
        .map_err(|error| error.replace("DreamX", "Breeze"))?;
    eprintln!(
        "Breeze: {} frames, {} samples written to {} in {:.3}s",
        frames.len(),
        waveform.len(),
        out.display(),
        started.elapsed().as_secs_f64()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::qwen3::tts::codec::encode_wav_pcm16;

    #[test]
    fn breeze_defaults_cfg_by_instruction_and_requires_complete_clone_prompt() {
        let mut options = CliOptions::default();
        assert_eq!(validate_breeze_options(&options).unwrap(), 1.0);
        options.instruction = Some(" \n ".into());
        assert_eq!(validate_breeze_options(&options).unwrap(), 1.0);
        options.instruction = Some("温柔的女声".into());
        assert_eq!(validate_breeze_options(&options).unwrap(), 3.0);
        options.cfg_scale = Some(2.0);
        assert_eq!(validate_breeze_options(&options).unwrap(), 2.0);
        options.ref_audio = Some("voice.wav".into());
        assert!(validate_breeze_options(&options).is_err());
        options.ref_text = Some(" ".into());
        assert!(validate_breeze_options(&options).is_err());
        options.ref_text = Some("你好".into());
        assert!(validate_breeze_options(&options).is_ok());
        options.temperature = Some(0.5);
        assert!(validate_breeze_options(&options).is_ok());
        options.temperature = Some(f32::NAN);
        assert!(validate_breeze_options(&options).is_err());
        options.temperature = Some(-0.1);
        assert!(validate_breeze_options(&options).is_err());
        options.temperature = Some(0.0);
        for p in [0.0, 1.1, f32::INFINITY, f32::NAN] {
            options.top_p = Some(p);
            assert!(validate_breeze_options(&options).is_err());
        }
    }

    #[test]
    fn breeze_reference_wav_preserves_native_rate_and_resamples_other_rates() {
        let mut wav = encode_wav_pcm16(&[0.0; 2], 24_000).unwrap();
        wav[44..48].copy_from_slice(&[0, 32, 0, 192]);
        let audio = reference_wav_to_24k(&wav).unwrap();
        assert_eq!(audio, vec![0.25, -0.5]);
        let mut stereo = encode_wav_pcm16(&[0.0; 4], 24_000).unwrap();
        stereo[22..24].copy_from_slice(&2u16.to_le_bytes());
        stereo[28..32].copy_from_slice(&96_000u32.to_le_bytes());
        stereo[32..34].copy_from_slice(&4u16.to_le_bytes());
        stereo[44..52].copy_from_slice(&[0, 64, 0, 192, 0, 32, 0, 96]);
        assert_eq!(reference_wav_to_24k(&stereo).unwrap(), vec![0.0, 0.5]);
        let wav = encode_wav_pcm16(&[0.0; 16], 16_000).unwrap();
        assert_eq!(reference_wav_to_24k(&wav).unwrap(), vec![0.0; 24]);
        assert!(reference_wav_to_24k(b"invalid WAV").is_err());
    }
}
