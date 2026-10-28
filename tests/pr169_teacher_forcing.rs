//! Replay the Oracle's tokens to compare approximate logits on identical histories.
use rust_model_inference::{
    core::thread_pool::ComputePool,
    models::qwen35::{Qwen35Model, Qwen35Session},
    GGUFLoader,
};
use std::sync::Arc;

#[test]
#[ignore = "requires real qwen35/qwen35moe model and pinned Oracle log"]
fn replay_oracle_tokens() {
    assert!(std::env::var_os("RMI_PARITY_TRACE").is_some());
    let log =
        std::fs::read_to_string(std::env::var_os("RMI_LLAMA_REFERENCE_LOG").unwrap()).unwrap();
    let prompts: Vec<Vec<u32>> = log
        .lines()
        .filter_map(|line| line.strip_prefix("RMI_PROMPT "))
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(prompts.len(), 1);
    let prompt = &prompts[0];
    let generated: Vec<u32> = log
        .lines()
        .filter_map(|line| line.strip_prefix("RMI_GREEDY "))
        .map(|id| id.parse().unwrap())
        .collect();
    assert!(!prompt.is_empty() && generated.len() >= 2);
    let mut inputs = prompt.clone();
    inputs.extend_from_slice(&generated[..generated.len() - 1]);
    #[cfg(feature = "parity-trace")]
    {
        rust_model_inference::parity_trace::token_ids("prompt_ids", prompt).unwrap();
        rust_model_inference::parity_trace::token_ids("teacher_forced_ids", &inputs).unwrap();
    }
    let source = GGUFLoader::from_file(std::env::var_os("RMI_TEXT_MODEL").unwrap()).unwrap();
    let mut model = Qwen35Model::from_source(&source).unwrap();
    let pool = Arc::new(ComputePool::new(8));
    let mut session = Qwen35Session::new(&mut model, inputs.len() + 1, pool).unwrap();
    for (position, token) in inputs.into_iter().enumerate() {
        let logits = session
            .step_with_tokens(&[token], &[[position; 4]])
            .unwrap();
        assert!(logits.iter().all(|v| v.is_finite()));
    }
}
