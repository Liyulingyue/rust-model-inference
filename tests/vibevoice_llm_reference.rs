//! VibeVoice ASR Qwen2 reference test against the original BF16 safetensors.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use rust_model_inference::core::tensor::TensorSource;
use rust_model_inference::core::thread_pool::ComputePool;
use rust_model_inference::format::ggufrs::{open_model_source, ComponentRole};
use rust_model_inference::models::vibevoice_asr::llm::{
    AsrInputRow, AsrLlmSession, VibeVoiceAsrLlm,
};

const PROMPT_IDS: &[u32] = &[
    2610, 525, 264, 10950, 17847, 429, 1356, 55136, 7699, 1946, 1119, 1467, 2550, 13, 5209, 1356,
    3114, 279, 2701, 6136, 3530, 4269, 11307, 448, 1493, 6894, 25, 18601, 11, 2213, 198,
];
const SPEECH_START: u32 = 151646;
const SPEECH_END: u32 = 151647;
const TEXT_CHUNK_END: u32 = 151665;
const ORACLE_FIRST_TOKEN: u32 = 58;

#[cfg(feature = "parity-trace")]
const SAFETENSORS_HIDDEN_MAX_NRMSE: f64 = 0.03;
#[cfg(feature = "parity-trace")]
const SAFETENSORS_LOGITS_MAX_NRMSE: f64 = 0.015;

fn read_f32(path: &Path) -> Result<Vec<f32>, String> {
    let bytes = std::fs::read(path).map_err(|error| error.to_string())?;
    if bytes.len() % 4 != 0 {
        return Err(format!("{} is not an f32 sidecar", path.display()));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect())
}

#[cfg(feature = "parity-trace")]
fn compare_f32(name: &str, actual: &[f32], expected: &[f32]) -> f64 {
    assert_eq!(actual.len(), expected.len(), "{name}: length mismatch");
    let mut max_abs = 0.0f32;
    let mut max_index = 0usize;
    let mut sum_sq_diff = 0.0f64;
    let mut sum_sq_reference = 0.0f64;
    for (index, (&value, &reference)) in actual.iter().zip(expected).enumerate() {
        assert!(
            value.is_finite(),
            "{name}: Rust value {index} is not finite"
        );
        assert!(
            reference.is_finite(),
            "{name}: reference value {index} is not finite"
        );
        let difference = (value - reference).abs();
        if difference > max_abs {
            max_abs = difference;
            max_index = index;
        }
        sum_sq_diff += f64::from(difference) * f64::from(difference);
        sum_sq_reference += f64::from(reference) * f64::from(reference);
    }
    let nrmse = (sum_sq_diff / sum_sq_reference.max(f64::MIN_POSITIVE)).sqrt();
    let value = actual[max_index];
    let reference = expected[max_index];
    eprintln!(
        "{name}: nrmse={nrmse:.6} max_abs={max_abs:.6} index={max_index} \
         rust={value:.7} (0x{:08x}) reference={reference:.7} (0x{:08x})",
        value.to_bits(),
        reference.to_bits()
    );
    nrmse
}

#[cfg(feature = "parity-trace")]
fn top_indices(values: &[f32], count: usize) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..values.len()).collect();
    indices.sort_unstable_by(|&left, &right| values[right].total_cmp(&values[left]));
    indices.truncate(count);
    indices
}

#[cfg(feature = "parity-trace")]
fn compare_safetensors_oracle(
    oracle_dir: &Path,
    layer_trace: &[f32],
    normed: &[f32],
    logits: &[f32],
    n_layer: usize,
    n_embd: usize,
) {
    let manifest_path = oracle_dir.join("vibevoice_llm_manifest.json");
    let manifest: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(&manifest_path)
            .unwrap_or_else(|error| panic!("{}: {error}", manifest_path.display())),
    )
    .expect("parse LLM oracle manifest");
    assert_eq!(manifest["source_kind"], "safetensors-bf16");
    assert_eq!(manifest["layers"], n_layer);
    assert_eq!(manifest["hidden_size"], n_embd);
    assert_eq!(manifest["vocab_size"], logits.len());

    let mut worst_layer = (0usize, 0.0f64);
    for layer in 0..n_layer {
        let expected = read_f32(&oracle_dir.join(format!("vibevoice_llm_layer_{layer:02}.f32")))
            .expect("read layer oracle");
        let nrmse = compare_f32(
            &format!("safetensors BF16 layer {layer}"),
            &layer_trace[layer * n_embd..(layer + 1) * n_embd],
            &expected,
        );
        if nrmse > worst_layer.1 {
            worst_layer = (layer, nrmse);
        }
    }
    let expected_normed = read_f32(&oracle_dir.join("vibevoice_llm_normed.f32"))
        .expect("read normalized hidden oracle");
    let normed_nrmse = compare_f32(
        "safetensors BF16 normalized hidden",
        normed,
        &expected_normed,
    );
    let expected_logits =
        read_f32(&oracle_dir.join("vibevoice_llm_logits.f32")).expect("read logits oracle");
    let logits_nrmse = compare_f32("safetensors BF16 logits", logits, &expected_logits);
    let actual_top = top_indices(logits, 8);
    let expected_top = top_indices(&expected_logits, 8);
    eprintln!("safetensors BF16 top8: rust={actual_top:?} reference={expected_top:?}");
    assert_eq!(actual_top, expected_top, "safetensors BF16: top-8 mismatch");
    assert!(
        worst_layer.1 <= SAFETENSORS_HIDDEN_MAX_NRMSE,
        "safetensors BF16: worst layer {} NRMSE {} exceeds {}",
        worst_layer.0,
        worst_layer.1,
        SAFETENSORS_HIDDEN_MAX_NRMSE
    );
    assert!(
        normed_nrmse <= SAFETENSORS_HIDDEN_MAX_NRMSE,
        "safetensors BF16: normalized hidden NRMSE {normed_nrmse} exceeds {}",
        SAFETENSORS_HIDDEN_MAX_NRMSE
    );
    assert!(
        logits_nrmse <= SAFETENSORS_LOGITS_MAX_NRMSE,
        "safetensors BF16: logits NRMSE {logits_nrmse} exceeds {}",
        SAFETENSORS_LOGITS_MAX_NRMSE
    );
}

#[test]
fn prompt_and_audio_rows_match_q8_oracle_token_and_position_contract() {
    let model_path = std::env::var_os("VIBEVOICE_ASR_LLM")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from("models/VibeVoice-ASR-Streaming-7B/VibeVoice-ASR-Streaming-7B-Q8_0.gguf")
        });
    let oracle_dir = std::env::var_os("VIBEVOICE_LLM_ORACLE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp/vv-oracle"));
    let embedding_path = oracle_dir.join("vibevoice_oracle_combined.f32");
    if !model_path.is_file() || !embedding_path.is_file() {
        eprintln!(
            "skipping: VibeVoice LLM/oracle missing ({}; {})",
            model_path.display(),
            embedding_path.display()
        );
        return;
    }

    let source: Arc<dyn TensorSource> =
        Arc::from(open_model_source(&model_path, ComponentRole::Llm).expect("open VibeVoice LLM"));
    let model = VibeVoiceAsrLlm::from_source(source, Arc::new(ComputePool::new(8)))
        .expect("load VibeVoice LLM");
    let embeddings = read_f32(&embedding_path).expect("read oracle embeddings");
    assert_eq!(embeddings.len(), 26 * model.config.n_embd);

    let expected_rows = PROMPT_IDS.len() + 1 + 26 + 1;
    let mut session = AsrLlmSession::new(&model, expected_rows + 1).expect("create session");
    for &token in PROMPT_IDS {
        session
            .forward_step(AsrInputRow::Token(token))
            .expect("prefill prompt token");
    }
    session
        .forward_step(AsrInputRow::Token(SPEECH_START))
        .expect("prefill speech start");
    for row in embeddings.chunks_exact(model.config.n_embd) {
        session
            .forward_step(AsrInputRow::Embedding(row))
            .expect("prefill audio embedding");
    }
    session
        .forward_step(AsrInputRow::Token(SPEECH_END))
        .expect("prefill speech end");

    assert_eq!(session.position(), expected_rows);
    #[cfg(feature = "parity-trace")]
    {
        let layer_trace = session.layer_hidden_trace().to_vec();
        let normed = session.normalized_hidden().to_vec();
        assert_eq!(
            layer_trace.len(),
            model.config.n_layer * model.config.n_embd
        );
        assert_eq!(normed.len(), model.config.n_embd);
        let logits = session.logits().expect("compute full logits").to_vec();

        if let Some(path) = std::env::var_os("VIBEVOICE_LLM_SAFETENSORS_ORACLE_DIR") {
            let dir = PathBuf::from(path);
            compare_safetensors_oracle(
                &dir,
                &layer_trace,
                &normed,
                &logits,
                model.config.n_layer,
                model.config.n_embd,
            );
        }
        assert_eq!(top_indices(&logits, 1)[0] as u32, ORACLE_FIRST_TOKEN);
    }
    #[cfg(not(feature = "parity-trace"))]
    assert_eq!(session.sample_argmax().unwrap(), ORACLE_FIRST_TOKEN);

    let before_error = session.position();
    assert!(session
        .forward_step(AsrInputRow::Token(model.vocab_size as u32))
        .is_err());
    assert_eq!(session.position(), before_error);

    session
        .forward_step(AsrInputRow::Token(TEXT_CHUNK_END))
        .expect("fill final capacity row");
    assert!(session
        .forward_step(AsrInputRow::Token(TEXT_CHUNK_END))
        .is_err());
    assert_eq!(session.position(), expected_rows + 1);
}
