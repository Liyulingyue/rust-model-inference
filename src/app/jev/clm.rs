//! CLM projection heads as a JEV-family decision mode.
//!
//! CLM is not a generative scorer: it embeds the state and each
//! candidate with a shared encoder (Qwen3-8B), pushes each side
//! through its own MLP head, L2-normalises, and scores
//! `logit_scale * cos(z_state, z_candidate)`.  So instead of a label
//! logit it produces one real-valued score per candidate, which we then
//! softmax into the same `JevResult` shape the logit-based scorers emit,
//! so the CLI/server output does not fork.
//!
//! Prompt layout comes from the reference client (`clm/schema.py`):
//! the state head sees `context + blank line + question`, the action
//! head sees each candidate verbatim.  Feeding a bare context leaves
//! the state head under-specified and the ranking silently degrades.

use std::path::Path;
use std::sync::Arc;

use super::{JevQuestionInput, JevResult, JevMode};
use crate::core::thread_pool::ComputePool;
use crate::core::tensor::TensorSource;
use crate::core::tokenizer::{BPETokenizer, EncodeOptions};
use crate::format::ggufrs::{open_model_source, ComponentRole};
use crate::models::clm::ClmHeads;
use crate::models::qwen3::trunk::{qwen_text_positions, Qwen3Input, Qwen3Model, Qwen3Session};

/// Labels are `A`, `B`, ... just like the logit-based JEV scorers use.
fn labels(n: usize) -> Vec<char> {
    (0..n).map(|i| (b'A' + i as u8) as char).collect()
}

fn softmax(scores: &[f32]) -> Vec<f32> {
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scores.iter().map(|&s| (s - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let sum = if sum > 0.0 { sum } else { 1.0 };
    exps.iter().map(|&e| e / sum).collect()
}

/// Embed `text` with the shared encoder and return the last-token
/// pooled, post-RMSNorm hidden state.
fn embed(
    tokenizer: &Arc<BPETokenizer>,
    model: &Qwen3Model,
    text: &str,
) -> Result<Vec<f32>, String> {
    // No chat template: the heads were trained on plain text, so the
    // candidate reaches the encoder exactly as the caller wrote it.
    let token_ids = tokenizer.encode(
        text,
        EncodeOptions { add_special: false, parse_special: false },
    );
    if token_ids.is_empty() {
        return Err(format!("empty input: {text:?}"));
    }
    let positions = qwen_text_positions(token_ids.len());
    let capacity = token_ids.len() + 4;
    let mut session =
        Qwen3Session::new(model, capacity).map_err(|e| format!("session: {e}"))?;
    session
        .forward_last_hidden(
            Qwen3Input {
                token_ids: &token_ids,
                positions: &positions,
                embeddings: None,
                deepstack_embeddings: None,
            },
            token_ids.len(),
        )
        .map_err(|e| format!("forward: {e}"))
}

/// Score with an already-loaded encoder + heads.  The server keeps both
/// alive across requests, so it uses this instead of
/// [`run_clm_decision_data`], which would re-read the head file every
/// call.
pub fn run_clm_scoring(
    model: &Qwen3Model,
    tokenizer: &Arc<BPETokenizer>,
    heads: &ClmHeads,
    context: &str,
    questions: &[JevQuestionInput],
) -> Result<Vec<JevResult>, String> {
    let prepared = super::prepare_jev_questions(questions, None)?;
    let config = model.config().clone();
    if config.n_embd != heads.encoder_dim() {
        return Err(format!(
            "encoder hidden size {} does not match the CLM heads (expected {})",
            config.n_embd,
            heads.encoder_dim()
        ));
    }

    eprintln!(
        "CLM: encoder {}x{} + heads ({} questions)",
        config.n_layer, config.n_embd, prepared.len()
    );

    let t0 = std::time::Instant::now();
    let mut results = Vec::with_capacity(prepared.len());
    let mut scratch: Vec<f32> = Vec::new();
    for q in &prepared {
        // Reference state_text(): context, blank line, question.  When
        // --jev-question is empty the caller already packed the whole
        // state into --jev-context, so appending an empty question would
        // leave a stray blank line and shift every score.
        let state_text = if q.text.trim().is_empty() {
            context.trim().to_string()
        } else {
            format!("{}\n\n{}", context.trim(), q.text.trim())
        };
        let z_state = heads
            .project_state(&embed(&tokenizer, &model, &state_text)?, &mut scratch)?;

        let mut values = Vec::with_capacity(q.descriptions.len());
        for cand in &q.descriptions {
            let z = heads.project_candidate(&embed(&tokenizer, &model, cand)?, &mut scratch)?;
            values.push(heads.score(&z_state, &z));
        }
        let probabilities = softmax(&values);

        let choice_label = probabilities
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| labels(q.descriptions.len())[i]);

        // The three summary stats match what the logit-based scorers
        // report, so a caller can read either mode the same way.
        let confidence = match choice_label {
            Some(c) => {
                let i = labels(q.descriptions.len()).iter().position(|&l| l == c).unwrap_or(0);
                probabilities[i]
                    - probabilities.iter().sum::<f32>() / probabilities.len() as f32
            }
            None => 0.0,
        };
        let entropy = -probabilities
            .iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| p * p.ln())
            .sum::<f32>();
        let mut sorted = probabilities.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let margin = sorted.first().copied().unwrap_or(0.0)
            - sorted.get(1).copied().unwrap_or(0.0);

        results.push(JevResult {
            mode: JevMode::Choice,
            question: q.text.clone(),
            labels: labels(q.descriptions.len()),
            descriptions: q.descriptions.clone(),
            values,
            probabilities,
            choice_label,
            positive_label: None,
            probability_positive: None,
            score: None,
            confidence,
            entropy,
            margin,
            prefill_ms: t0.elapsed().as_millis(),
        });
    }
    Ok(results)
}


/// Load the head file, then delegate to [`run_clm_scoring`].  CLI only.
pub fn run_clm_decision_data(
    source: Arc<dyn TensorSource>,
    head_path: &Path,
    context: &str,
    questions: &[JevQuestionInput],
    n_threads_arg: usize,
) -> Result<Vec<JevResult>, String> {
    let head_source: Box<dyn TensorSource> =
        open_model_source(head_path, ComponentRole::Llm)
            .map_err(|e| format!("open CLM heads ({}): {e}", head_path.display()))?;
    let heads = ClmHeads::from_source(head_source.as_ref())?;

    let tokenizer = Arc::new(
        BPETokenizer::from_gguf_metadata(|k| source.metadata(k).cloned())
            .map_err(|e| format!("init tokenizer: {e}"))?,
    );
    let pool = Arc::new(ComputePool::new(n_threads_arg.max(1)));
    let model = Qwen3Model::from_source(source, Arc::clone(&tokenizer), pool)
        .map_err(|e| format!("load encoder: {e}"))?;
    if model.config().architecture != "qwen3" {
        return Err(format!(
            "CLM needs a qwen3 encoder, got {:?}",
            model.config().architecture
        ));
    }
    run_clm_scoring(&model, &tokenizer, &heads, context, questions)
}

/// CLI entry: score then print exactly the way the logit-based JEV
/// scorers do (`--jev-output-json` included), so the two modes are
/// indistinguishable from the shell.
pub fn run_clm_decision(
    source: Arc<dyn TensorSource>,
    head_path: &Path,
    context: &str,
    questions: &[JevQuestionInput],
    n_threads_arg: usize,
    output_json: bool,
) -> Result<(), String> {
    let t0 = std::time::Instant::now();
    let results = run_clm_decision_data(source, head_path, context, questions, n_threads_arg)?;
    if output_json {
        for r in &results {
            let line = serde_json::to_string(r).map_err(|e| format!("json encode: {e}"))?;
            println!("{line}");
        }
        return Ok(());
    }
    for r in &results {
        println!("Q: {}", r.question);
        for (label, (desc, (p, v))) in r
            .labels
            .iter()
            .zip(r.descriptions.iter().zip(r.probabilities.iter().zip(r.values.iter())))
        {
            println!("  {label}. {desc}  p={p:.4}  score={v:.4}");
        }
        if let Some(c) = r.choice_label {
            println!("  -> choice: {c}");
        }
        println!("  ({:.0} ms)", t0.elapsed().as_millis());
    }
    Ok(())
}
