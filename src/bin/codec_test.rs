//! Standalone test to verify the DAC pipeline produces something other than
//! noise. Skips Talker + Code Predictor and feeds a synthetic sine input
//! through the codec decoder.

use rust_model_inference::format::ggufrs::{open_model_source, ComponentRole};
use rust_model_inference::models::qwen3::tts::codec::write_wav_f32;
use rust_model_inference::models::qwen3::tts::codec::{DacDecoder, RvqDecoder};
use std::path::PathBuf;
use std::sync::Arc;

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: codec_test <mmproj.gguf> <out.wav>");
        return Err("missing args".into());
    }
    let mmproj_path = PathBuf::from(&args[1]);
    let out_path = PathBuf::from(&args[2]);

    eprintln!("Loading {}", mmproj_path.display());
    let source: Arc<dyn rust_model_inference::core::tensor::TensorSource> = Arc::from(
        open_model_source(&mmproj_path, ComponentRole::Mmproj).map_err(|e| e.to_string())?,
    );

    let _rvq = RvqDecoder::from_source(source.as_ref())?;
    let dac = DacDecoder::from_source(source.as_ref())?;

    // Feed 30 timesteps of sine wave as RVQ-decoded embedding (512-dim per step).
    // The point is to see if DAC output is structured or pure noise.
    let timesteps = 30;
    let hidden_dim = 512usize;
    let mut continuous = vec![0.0f32; timesteps * hidden_dim];
    // Try a constant (uniform) input — DAC should at least produce *something*
    // non-trivial; if it produces silence, there's a structural bug.
    for t in 0..timesteps {
        for d in 0..hidden_dim {
            continuous[t * hidden_dim + d] = 0.1;
        }
    }
    eprintln!(
        "input: {} uniform samples rms {}",
        continuous.len(),
        rms(&continuous)
    );
    eprintln!("input rms: {}", rms(&continuous));
    eprintln!(
        "Pre DAC.pre_conv: input is {} floats ({} timesteps × {} dim)",
        continuous.len(),
        timesteps,
        hidden_dim
    );
    let preconv = dac.pre_conv(&continuous, timesteps)?;
    eprintln!(
        "After pre_conv: {} floats rms {}",
        preconv.len(),
        rms(&preconv)
    );
    // Dump norm_w stats from the first upsample block for sanity.
    // (Cannot access private fields; show first 5 preconv values instead.)
    eprintln!("preconv[:5] = {:?}", &preconv[..5.min(preconv.len())]);
    // Now run DAC and inspect output stats.
    // TFM: bypass — feed preconv directly to DAC. TFM is identity-ish for
    // small inputs (we don't have sliding-window attention, but for 30
    // frames within swa=72 it's equivalent to full causal).
    eprintln!("Skipping Waveform TFM");
    let waveform = dac.decode(&preconv, timesteps)?;
    eprintln!(
        "After DAC: {} samples rms {} autocorr={}",
        waveform.len(),
        rms(&waveform),
        autocorr_max(&waveform)
    );
    eprintln!("waveform[:5] = {:?}", &waveform[..5.min(waveform.len())]);
    eprintln!(
        "waveform[max//2:max//2+5] = {:?}",
        &waveform[waveform.len() / 2..waveform.len() / 2 + 5]
    );

    write_wav_f32(&out_path, &waveform, 24000).map_err(|e| e.to_string())?;
    eprintln!("wrote {}", out_path.display());
    Ok(())
}

fn rms(samples: &[f32]) -> f32 {
    let sum: f32 = samples.iter().map(|s| s * s).sum();
    (sum / samples.len() as f32).sqrt()
}

fn autocorr_max(samples: &[f32]) -> f32 {
    let max_lag = (samples.len() / 2).min(500);
    let mut best = 0.0f32;
    for lag in 20..max_lag {
        let corr: f32 = samples
            .iter()
            .take(samples.len() - lag)
            .zip(samples.iter().skip(lag))
            .map(|(a, b)| a * b)
            .sum();
        let norm = ((samples.len() - lag) as f32) * 32768.0 * 32768.0;
        let v = (corr / norm).abs();
        if v > best {
            best = v;
        }
    }
    best
}
