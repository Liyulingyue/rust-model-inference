//! VibeVoice ASR speech-frontend parity test.
//!
//! Compares the Rust ConvNeXt encoder + speech connectors against a numpy
//! oracle (tools/vibevoice/vibevoice_oracle.py) on the same 83200-sample
//! window. Skips when the oracle dumps or the mmproj gguf are absent so CI
//! without the checkpoint stays green.

use std::path::PathBuf;
use std::sync::Arc;

use rust_model_inference::core::tensor::TensorSource;
use rust_model_inference::format::ggufrs::{open_model_source, ComponentRole};
use rust_model_inference::models::vibevoice_asr::config::VibeVoiceAsrConfig;
use rust_model_inference::models::vibevoice_asr::encoder::{SpeechConnector, TokenizerEncoder};

const REFERENCE_TOLERANCE: f32 = 1e-3;

fn oracle_dir() -> Option<PathBuf> {
    let dir = std::env::var_os("VIBEVOICE_ORACLE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp/vv-oracle"));
    dir.is_dir().then_some(dir)
}

fn read_f32(path: &std::path::Path) -> Option<Vec<f32>> {
    let bytes = std::fs::read(path).ok()?;
    Some(
        bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect(),
    )
}

fn compare(name: &str, actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{name}: length mismatch (rust {} vs oracle {})",
        actual.len(),
        expected.len()
    );
    let mut max_abs = 0.0f32;
    for (index, (&value, &reference)) in actual.iter().zip(expected).enumerate() {
        let difference = (value - reference).abs();
        max_abs = max_abs.max(difference);
        assert!(
            difference <= tolerance,
            "{name}: element {index} differs by {difference}: rust {value} vs oracle \
             {reference} (max_abs {max_abs})"
        );
    }
    eprintln!("{name}: max_abs {max_abs:.6} (tolerance {tolerance})");
}

fn open_mmproj() -> Option<Arc<dyn TensorSource>> {
    let path = std::env::var_os("VIBEVOICE_ASR_MMPROJ")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(
                "models/VibeVoice-ASR-Streaming-7B/mmproj-VibeVoice-ASR-Streaming-7B-BF16.gguf",
            )
        });
    if !path.is_file() {
        eprintln!("skipping: mmproj not found at {}", path.display());
        return None;
    }
    Some(Arc::from(
        open_model_source(&path, ComponentRole::Mmproj).expect("failed to open mmproj"),
    ))
}

#[test]
fn encoder_and_connectors_match_numpy_oracle() {
    let Some(oracle) = oracle_dir() else {
        eprintln!("skipping: oracle dumps not found (run tools/vibevoice/vibevoice_oracle.py)");
        return;
    };
    let Some(mmproj) = open_mmproj() else {
        return;
    };
    let config = VibeVoiceAsrConfig::from_source(mmproj.as_ref()).expect("mmproj config");
    let acoustic_encoder =
        TokenizerEncoder::from_source(mmproj.as_ref(), "acoustic", &config).expect("acoustic");
    let semantic_encoder =
        TokenizerEncoder::from_source(mmproj.as_ref(), "semantic", &config).expect("semantic");
    let acoustic_connector = SpeechConnector::from_source(
        mmproj.as_ref(),
        "acoustic",
        config.acoustic_vae_dim,
        config.llm_hidden_size,
        config.connector_eps,
    )
    .expect("acoustic connector");
    let semantic_connector = SpeechConnector::from_source(
        mmproj.as_ref(),
        "semantic",
        config.semantic_vae_dim,
        config.llm_hidden_size,
        config.connector_eps,
    )
    .expect("semantic connector");

    let audio = read_f32(&oracle.join("vibevoice_oracle_audio.f32")).expect("oracle audio dump");
    assert_eq!(audio.len(), 83200, "oracle window length");

    let mut traces = Vec::new();
    let acoustic = acoustic_encoder
        .forward_with_traces(&audio, &mut traces)
        .expect("acoustic encode");
    for (stage, trace) in traces.iter().enumerate() {
        let dump = oracle.join(format!("vibevoice_oracle_acoustic_stage{stage}.f32"));
        if let Some(expected) = read_f32(&dump) {
            compare(
                &format!("acoustic stage{stage}"),
                trace,
                &expected,
                REFERENCE_TOLERANCE,
            );
        }
    }
    let semantic = semantic_encoder.forward(&audio).expect("semantic encode");
    compare(
        "acoustic latents",
        &acoustic,
        &read_f32(&oracle.join("vibevoice_oracle_acoustic.f32")).expect("acoustic dump"),
        REFERENCE_TOLERANCE,
    );
    compare(
        "semantic latents",
        &semantic,
        &read_f32(&oracle.join("vibevoice_oracle_semantic.f32")).expect("semantic dump"),
        REFERENCE_TOLERANCE,
    );

    let frames = acoustic.len() / config.acoustic_vae_dim;
    let mut scratch = Vec::new();
    let mut acoustic_proj = Vec::new();
    acoustic_connector
        .forward(&acoustic, frames, &mut scratch, &mut acoustic_proj)
        .expect("acoustic projection");
    let mut semantic_proj = Vec::new();
    semantic_connector
        .forward(&semantic, frames, &mut scratch, &mut semantic_proj)
        .expect("semantic projection");
    let combined: Vec<f32> = acoustic_proj
        .iter()
        .zip(semantic_proj.iter())
        .map(|(a, s)| a + s)
        .collect();
    compare(
        "acoustic projection",
        &acoustic_proj,
        &read_f32(&oracle.join("vibevoice_oracle_acoustic_proj.f32")).expect("proj dump"),
        REFERENCE_TOLERANCE,
    );
    compare(
        "semantic projection",
        &semantic_proj,
        &read_f32(&oracle.join("vibevoice_oracle_semantic_proj.f32")).expect("proj dump"),
        REFERENCE_TOLERANCE,
    );
    compare(
        "combined features",
        &combined,
        &read_f32(&oracle.join("vibevoice_oracle_combined.f32")).expect("combined dump"),
        REFERENCE_TOLERANCE,
    );
}
