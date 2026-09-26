use std::sync::Arc;

use rust_model_inference::core::thread_pool::ComputePool;
use rust_model_inference::core::tokenizer::{BPETokenizer, EncodeOptions};
use rust_model_inference::models::qwen3::trunk::{
    qwen_text_positions, Qwen3Input, Qwen3Model, Qwen3Session,
};
use rust_model_inference::{GGUFLoader, TensorSource};

fn loader() -> Option<GGUFLoader> {
    let path = std::env::var_os("RMI_QWEN3_RERANK_Q8_MODEL")?;
    Some(GGUFLoader::from_file(path).unwrap())
}

fn model_source() -> Option<Arc<dyn TensorSource>> {
    loader().map(|inner| Arc::new(inner) as Arc<dyn TensorSource>)
}

fn tokenizer() -> Option<BPETokenizer> {
    let loader = loader()?;
    Some(BPETokenizer::from_gguf_metadata(|k| loader.metadata(k).cloned()).unwrap())
}

fn model() -> Option<(&'static Qwen3Model, BPETokenizer)> {
    let source = model_source()?;
    let tok = Arc::new(tokenizer()?);
    let tok_for_model = Arc::clone(&tok);
    let m = Qwen3Model::from_source(source, tok_for_model, Arc::new(ComputePool::new(4)))
        .ok()?;
    // Recover the (inner) tokenizer via Arc::try_unwrap when possible,
    // else dereference. For our tests, the Arc is uniquely owned here so
    // try_unwrap succeeds.
    Some((Box::leak(Box::new(m)), Arc::try_unwrap(tok).ok()?))
}

fn render_prompt(instruction: &str, query: &str, document: &str) -> String {
    format!(
        "system\nJudge whether the Document meets the requirements based on \
         the Query and the Instruct provided. Note that the answer can only be \
         \"yes\" or \"no\".\nuser\n<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}\n"
    )
}

#[test]
fn q8_rerank_contract_loads() {
    let Some(model) = model() else { return };
    let m = &model.0;
    assert_eq!(m.config().architecture, "qwen3");
    assert_eq!(m.config().n_layer, 28);
    assert_eq!(m.config().n_embd, 1024);
    assert!(m.is_rerank(), "rerank flag should be set when cls.output.weight is present");
}

#[test]
fn q8_rerank_score_ranks_paris_over_photosynthesis() {
    let Some((model, tok)) = model() else { return };
    let instruction = "Given a web search query, retrieve relevant passages that answer the query";
    let query = "What is the capital of France?";
    let docs = [
        "Paris is the capital and most populous city of France.",
        "Photosynthesis is the process used by plants to convert light energy into chemical energy.",
    ];
    let mut scores = Vec::with_capacity(docs.len());
    for doc in &docs {
        let prompt = render_prompt(instruction, query, doc);
        let token_ids = tok.encode(
            &prompt,
            EncodeOptions {
                add_special: false,
                parse_special: false,
            },
        );
        let positions = qwen_text_positions(token_ids.len());
        let mut session = Qwen3Session::new(&model, token_ids.len() + 4).unwrap();
        let hidden = session
            .forward_rerank(
                Qwen3Input {
                    token_ids: &token_ids,
                    positions: &positions,
                    embeddings: None,
                    deepstack_embeddings: None,
                },
                token_ids.len(),
            )
            .unwrap();
        assert!(!hidden.is_empty());
        assert!(hidden.iter().all(|v| v.is_finite()));
        let logits = model.score_logits(&hidden).unwrap();
        // logits[0] = yes, logits[1] = no (per Qwen3-Reranker convention).
        let yes = logits.first().copied().unwrap();
        scores.push((doc, yes));
    }
    let paris = scores[0].1;
    let photo = scores[1].1;
    assert!(
        paris > photo,
        "Paris relevance ({paris:.3}) should exceed Photosynthesis ({photo:.3})",
    );
    // The actual gap is large (>10 logit); assert a generous lower bound to
    // tolerate future kernel refactors that might cost a few logit.
    assert!(
        paris - photo > 2.0,
        "Paris - Photosynthesis gap should be large; got {:.3}",
        paris - photo,
    );
}

#[test]
fn q8_rerank_chunked_and_incremental_match() {
    let Some((model, tok)) = model() else { return };
    let instruction = "Given a web search query, retrieve relevant passages that answer the query";
    let query = "What is the capital of France?";
    let doc = "Paris is the capital and most populous city of France, home to the Eiffel Tower.";
    let prompt = render_prompt(instruction, query, doc);
    let token_ids = tok.encode(
        &prompt,
        EncodeOptions {
            add_special: false,
            parse_special: false,
        },
    );
    let positions = qwen_text_positions(token_ids.len());
    // Single-shot reference.
    let mut ref_session = Qwen3Session::new(&model, token_ids.len() + 4).unwrap();
    let expected = ref_session
        .forward_rerank(
            Qwen3Input {
                token_ids: &token_ids,
                positions: &positions,
                embeddings: None,
                deepstack_embeddings: None,
            },
            token_ids.len(),
        )
        .unwrap();
    // Two-shot incremental (first half, then second half).
    let half = token_ids.len() / 2;
    let mut incremental = Qwen3Session::new(&model, token_ids.len() + 4).unwrap();
    let _ = incremental
        .forward_rerank(
            Qwen3Input {
                token_ids: &token_ids[..half],
                positions: &positions[..half],
                embeddings: None,
                deepstack_embeddings: None,
            },
            half,
        )
        .unwrap();
    let actual = incremental
        .forward_rerank(
            Qwen3Input {
                token_ids: &token_ids[half..],
                positions: &positions[half..],
                embeddings: None,
                deepstack_embeddings: None,
            },
            token_ids.len() - half,
        )
        .unwrap();
    // Bit-exact last hidden.
    assert_eq!(expected.len(), actual.len());
    assert!(expected
        .iter()
        .zip(&actual)
        .all(|(a, b)| a.to_bits() == b.to_bits()));
    // And the score is bit-exact too.
    let expected_score = model.score_logits(&expected).unwrap();
    let actual_score = model.score_logits(&actual).unwrap();
    let exp_bits: Vec<u32> = expected_score.iter().map(|v| v.to_bits()).collect();
    let act_bits: Vec<u32> = actual_score.iter().map(|v| v.to_bits()).collect();
    assert_eq!(exp_bits, act_bits);
}

#[test]
#[ignore = "requires a non-rerank qwen3 Q8 GGUF (none available locally); positive coverage is in the main qwen3 tests"]
fn q8_rerank_rejects_non_rerank_model() {
    let Some(source) = model_source() else { return };
    let tok = Arc::new(tokenizer().unwrap());
    let m = Qwen3Model::from_source(source, Arc::clone(&tok), Arc::new(ComputePool::new(4))).unwrap();
    assert!(!m.is_rerank(), "model should not be flagged as rerank without cls head");
}

#[test]
#[ignore = "requires a non-rerank qwen3 Q8 GGUF (none available locally)"]
fn q8_rerank_score_logits_works_without_head() {
    let Some(source) = model_source() else { return };
    let tok = Arc::new(tokenizer().unwrap());
    let m = Qwen3Model::from_source(source, Arc::clone(&tok), Arc::new(ComputePool::new(4))).unwrap();
    let dummy_hidden = vec![0.0f32; m.config().n_embd];
    let err = m.score_logits(&dummy_hidden).unwrap_err();
    assert!(
        err.contains("not a rerank") || err.contains("rerank"),
        "expected rerank error, got {err}"
    );
}