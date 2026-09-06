//! VibeVoice ASR Qwen2 reference test against the Q8_0 NumPy oracle.

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
