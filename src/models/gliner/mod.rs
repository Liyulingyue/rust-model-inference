//! GLiNER2.5-Decide: a DeBERTa-v3-large encoder with a small classification
//! head on the hidden states at the `[L]` marker positions.
//!
//! The model is not a generator. `classify_text(text, {head: labels})` runs one
//! encoder pass and reads a logit per label off the marker rows, which is the
//! whole of `gliner2/classification/scoring.py::ClassificationScorer`:
//!
//! ```text
//! hidden = DebertaV2Model(input_ids)                    # no [CLS]/[SEP]
//! label_rows = hidden[[L] positions of this task]
//! logits = classifier(label_rows).squeeze(-1)
//! probs  = softmax(logits) or sigmoid(logits)
//! ```
//!
//! The NER heads (`span_rep`, `count_embed`, `count_pred`) are not converted and
//! not needed: they never run on the classification path.
//!
//! Layer by layer the encoder is standard DeBERTa-v3, documented in
//! [`compute::encode`].

pub mod compute;
pub mod prompt;
pub mod weights;

use crate::core::sentencepiece::SentencePieceTokenizer;
use crate::core::tensor::{MetaValue, TensorSource};

use compute::EncoderConfig;
use prompt::{EncodedPrompt, Label, Task};

pub const ARCH: &str = "gliner2";

/// One label's logits and decoded probability.
#[derive(Debug, Clone, PartialEq)]
pub struct LabelScore {
    pub label: String,
    pub logit: f32,
    pub probability: f32,
}

/// The decoded result of one task.
#[derive(Debug, Clone, PartialEq)]
pub struct TaskResult {
    pub task: String,
    pub scores: Vec<LabelScore>,
    /// Argmax label, or every label at/above the threshold for a multi-label
    /// task. Never empty: a multi-label head with nothing over the threshold
    /// falls back to its argmax, matching `_extract_classification_result`.
    pub selected: Vec<String>,
    pub multi_label: bool,
}

/// A loaded GLiNER2.5-Decide model.
pub struct GlinerModel<'a> {
    source: &'a dyn TensorSource,
    config: EncoderConfig,
    spm: SentencePieceTokenizer,
    weights: weights::ModelWeights<'a>,
}

impl<'a> GlinerModel<'a> {
    /// Load from a GGUF produced by `tools/converter/gliner/convert_gliner.py`.
    pub fn from_source(source: &'a dyn TensorSource) -> Result<Self, String> {
        let spm = load_spm(source)?;
        Self::from_source_with_tokenizer(source, spm)
    }

    /// Same as [`GlinerModel::from_source`], reusing an already-built
    /// tokenizer. Rebuilding it costs a 128k-piece trie, which a server pays
    /// once at startup rather than per request.
    pub fn from_source_with_tokenizer(
        source: &'a dyn TensorSource,
        spm: SentencePieceTokenizer,
    ) -> Result<Self, String> {
        let arch = meta_str(source, "general.architecture")?;
        if arch != ARCH {
            return Err(format!("expected {ARCH} architecture, got {arch:?}"));
        }
        let n_embd = meta_usize(source, &key("embedding_length"))?;
        let n_head = meta_usize(source, &key("attention.head_count"))?;
        let head_dim = meta_usize(source, &key("attention.head_dim"))?;
        let scale_divisor = meta_usize(source, &key("attention.scale_divisor"))?;
        if scale_divisor != 3 {
            return Err(format!(
                "{ARCH}.attention.scale_divisor must be 3 (1 + |p2c|c2p|), got {scale_divisor}"
            ));
        }
        let bucket_size = meta_usize(source, &key("relative_attention.bucket_size"))?;
        let max_relative = meta_usize(source, &key("relative_attention.max_relative_positions"))?;
        let activation = meta_str(source, &key("classifier.activation"))?;
        if activation != "relu" {
            return Err(format!(
                "{ARCH}.classifier.activation must be relu, got {activation:?}"
            ));
        }
        let config = EncoderConfig {
            n_embd,
            n_layer: meta_usize(source, &key("block_count"))?,
            n_head,
            n_ff: meta_usize(source, &key("feed_forward_length"))?,
            head_dim,
            eps: meta_f32(source, &key("attention.layer_norm_epsilon"))?,
            att_span: bucket_size,
            pos_ebd_size: bucket_size * 2,
            bucket_size,
            max_relative_positions: max_relative,
            norm_rel_embeddings: meta_bool(source, &key("norm_rel_embeddings"))?,
            vocab_size: meta_usize(source, &key("vocab_size"))?,
        };
        if meta_str(source, "tokenizer.ggml.model")? != "spm" {
            return Err("gliner2 requires a SentencePiece tokenizer".into());
        }
        if spm.len() + 11 != config.vocab_size {
            return Err(format!(
                "vocab_size {} does not match {} SentencePiece pieces plus 11 added tokens",
                config.vocab_size,
                spm.len()
            ));
        }
        let classifier_intermediate = meta_usize(source, &key("classifier.intermediate_size"))?;
        let weights = weights::load_weights(
            source,
            &weights::LoadSpec {
                n_layer: config.n_layer,
                n_embd: config.n_embd,
                n_head: config.n_head,
                n_ff: config.n_ff,
                pos_ebd_size: config.pos_ebd_size,
                classifier_intermediate,
            },
        )?;
        Ok(Self {
            source,
            config,
            spm,
            weights,
        })
    }

    pub fn config(&self) -> &EncoderConfig {
        &self.config
    }

    pub fn tokenizer(&self) -> &SentencePieceTokenizer {
        &self.spm
    }

    /// `SchemaTransformer` output for `tasks` and `text`.
    pub fn encode_prompt(&self, tasks: &[Task], text: &str) -> Result<EncodedPrompt, String> {
        prompt::build_prompt(tasks, text, &self.spm)
    }

    /// Hidden states for an already-encoded prompt.
    pub fn forward(&self, input_ids: &[u32], n_threads_arg: usize) -> Result<Vec<f32>, String> {
        compute::encode(&self.weights, &self.config, input_ids, n_threads_arg)
    }

    /// `classify_text(text, tasks)`: one encoder pass, one logit per label.
    pub fn classify_text(
        &self,
        text: &str,
        tasks: &[Task],
        n_threads_arg: usize,
    ) -> Result<Vec<TaskResult>, String> {
        let encoded = self.encode_prompt(tasks, text)?;
        let hidden = self.forward(&encoded.input_ids, n_threads_arg)?;
        self.score_prompt(tasks, &encoded, &hidden)
    }

    /// Scores for a prompt that was encoded and run separately.
    pub fn score_prompt(
        &self,
        tasks: &[Task],
        encoded: &EncodedPrompt,
        hidden: &[f32],
    ) -> Result<Vec<TaskResult>, String> {
        if encoded.markers.len() != tasks.len() {
            return Err("task/marker count mismatch".into());
        }
        let d = self.config.n_embd;
        if hidden.len() != encoded.input_ids.len() * d {
            return Err("hidden state does not match the encoded prompt".into());
        }
        let available = std::thread::available_parallelism()
            .map(|value| value.get())
            .unwrap_or(4);
        let pool = std::sync::Arc::new(crate::core::thread_pool::ComputePool::new(
            crate::app::resolve_thread_count(0, available),
        ));
        let mut scratch: Vec<f32> = Vec::new();
        let mut results = Vec::with_capacity(tasks.len());
        for (task, markers) in tasks.iter().zip(&encoded.markers) {
            // `embs[1:]` drops the `[P]` prompt row; the rest are the labels.
            let mut logits = Vec::with_capacity(task.labels.len());
            for position in markers.positions.iter().skip(1) {
                logits.push(self.classifier(
                    &hidden[*position * d..(*position + 1) * d],
                    &pool,
                    &mut scratch,
                )?);
            }
            let scores = decode(task, &logits);
            let selected = select(task, &scores);
            results.push(TaskResult {
                task: task.name.clone(),
                scores,
                selected,
                multi_label: task.multi_label,
            });
        }
        Ok(results)
    }

    /// `classifier.0 -> ReLU -> classifier.2`, squeezed.
    ///
    /// The widths come from the loaded biases rather than `Weight::n_out`,
    /// which reports 1 for every F32 tensor (see `compute::matmul_into`).
    fn classifier(
        &self,
        row: &[f32],
        pool: &std::sync::Arc<crate::core::thread_pool::ComputePool>,
        hidden: &mut Vec<f32>,
    ) -> Result<f32, String> {
        let intermediate = self.weights.classifier_0_bias.len();
        if row.len() != self.config.n_embd {
            return Err(format!(
                "classifier input is {} wide, expected {}",
                row.len(),
                self.config.n_embd
            ));
        }
        hidden.resize(intermediate, 0.0);
        compute::matmul_into(
            &self.weights.classifier_0,
            row,
            self.config.n_embd,
            intermediate,
            hidden,
            pool,
        );
        for (value, bias) in hidden.iter_mut().zip(&self.weights.classifier_0_bias) {
            *value = (*value + bias).max(0.0);
        }
        let mut out = [0.0f32; 1];
        compute::matmul_into(
            &self.weights.classifier_2,
            hidden,
            intermediate,
            1,
            &mut out,
            pool,
        );
        Ok(out[0] + self.weights.classifier_2_bias[0])
    }
}

fn decode(task: &Task, logits: &[f32]) -> Vec<LabelScore> {
    let scaled: Vec<f32> = logits
        .iter()
        .map(|value| value / task.temperature)
        .collect();
    let multi = task.multi_label;
    let probabilities: Vec<f32> = match task.activation.as_deref() {
        Some("softmax") => softmax(&scaled),
        Some("sigmoid") => scaled.iter().map(|value| sigmoid(*value)).collect(),
        // `class_act = "auto"`: sigmoid for a multi-label head, else softmax.
        _ if multi => scaled.iter().map(|value| sigmoid(*value)).collect(),
        _ => softmax(&scaled),
    };
    task.labels
        .iter()
        .zip(probabilities)
        .zip(scaled)
        .map(|((label, probability), logit)| LabelScore {
            label: label.name.clone(),
            logit,
            probability,
        })
        .collect()
}

fn select(task: &Task, scores: &[LabelScore]) -> Vec<String> {
    let mut best = 0usize;
    for (index, score) in scores.iter().enumerate() {
        if score.probability > scores[best].probability {
            best = index;
        }
    }
    if !task.multi_label {
        return vec![scores[best].label.clone()];
    }
    let chosen: Vec<String> = scores
        .iter()
        .filter(|score| score.probability >= task.cls_threshold)
        .map(|score| score.label.clone())
        .collect();
    if chosen.is_empty() {
        return vec![scores[best].label.clone()];
    }
    chosen
}

fn softmax(values: &[f32]) -> Vec<f32> {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut exps: Vec<f32> = values.iter().map(|value| (value - max).exp()).collect();
    let total: f64 = exps.iter().map(|value| f64::from(*value)).sum();
    let scale = (1.0 / total) as f32;
    for value in &mut exps {
        *value *= scale;
    }
    exps
}

fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exponential = value.exp();
        exponential / (1.0 + exponential)
    }
}

fn key(suffix: &str) -> String {
    format!("{ARCH}.{suffix}")
}

fn meta_str(source: &dyn TensorSource, name: &str) -> Result<String, String> {
    source
        .metadata(name)
        .and_then(MetaValue::to_string_val)
        .map(str::to_string)
        .ok_or_else(|| format!("missing or invalid metadata: {name}"))
}

fn meta_usize(source: &dyn TensorSource, name: &str) -> Result<usize, String> {
    source
        .metadata(name)
        .and_then(MetaValue::to_u64)
        .map(|value| value as usize)
        .ok_or_else(|| format!("missing or invalid metadata: {name}"))
}

fn meta_f32(source: &dyn TensorSource, name: &str) -> Result<f32, String> {
    source
        .metadata(name)
        .and_then(|value| value.to_f64())
        .map(|value| value as f32)
        .ok_or_else(|| format!("missing or invalid metadata: {name}"))
}

fn meta_bool(source: &dyn TensorSource, name: &str) -> Result<bool, String> {
    match source.metadata(name) {
        Some(MetaValue::Bool(value)) => Ok(*value),
        Some(MetaValue::Uint32(value)) => Ok(*value != 0),
        Some(MetaValue::Uint64(value)) => Ok(*value != 0),
        _ => Err(format!("missing or invalid metadata: {name}")),
    }
}

/// Rebuild the SentencePiece tokenizer from the GGUF metadata the converter
/// wrote: pieces, scores, types, the compiled charsmap, and the flags.
///
/// `tokenizer.ggml.tokens` / `.scores` / `spm.piece_types` all run to
/// `vocab_size` — the SentencePiece pieces followed by the eleven added GLiNER
/// tokens — so everything is cut back to `spm.piece_count` first. The added
/// tokens never reach the lattice: `prompt::encode_token` resolves them by name.
pub fn load_spm(source: &dyn TensorSource) -> Result<SentencePieceTokenizer, String> {
    let piece_count = meta_usize(source, &key("spm.piece_count"))?;
    let pieces = meta_bytes_array(source, piece_count)?;
    let types_raw = meta_uint_array(source, &key("spm.piece_types"))?;
    let scores_raw = meta_f64_array(source, "tokenizer.ggml.scores")?;
    if types_raw.len() < piece_count || scores_raw.len() < piece_count {
        return Err(format!(
            "SentencePiece scores/types hold {} entries, expected at least {piece_count}",
            types_raw.len().min(scores_raw.len())
        ));
    }
    let charsmap = base64_decode(&meta_str(source, &key("spm.charsmap_b64"))?)?;
    let normalizer = crate::core::sentencepiece::Normalizer::from_parts(
        &charsmap,
        meta_bool(source, &key("spm.add_dummy_prefix"))?,
        meta_bool(source, &key("spm.remove_extra_whitespaces"))?,
        meta_bool(source, &key("spm.escape_whitespaces"))?,
        meta_bool(source, &key("spm.treat_whitespace_as_suffix"))?,
    )?;
    SentencePieceTokenizer::from_parts(
        pieces,
        scores_raw[..piece_count]
            .iter()
            .map(|value| *value as f32)
            .collect(),
        types_raw[..piece_count]
            .iter()
            .map(|value| *value as u8)
            .collect(),
        meta_bool(source, &key("spm.byte_fallback"))?,
        normalizer,
    )
}

fn meta_array<'a>(source: &'a dyn TensorSource, name: &str) -> Result<&'a Vec<MetaValue>, String> {
    source
        .metadata(name)
        .and_then(MetaValue::to_arr)
        .ok_or_else(|| format!("missing metadata array: {name}"))
}

fn meta_uint_array(source: &dyn TensorSource, name: &str) -> Result<Vec<u32>, String> {
    meta_array(source, name)?
        .iter()
        .map(|value| {
            value
                .to_u64()
                .map(|number| number as u32)
                .ok_or_else(|| format!("{name} must hold unsigned integers"))
        })
        .collect()
}

fn meta_f64_array(source: &dyn TensorSource, name: &str) -> Result<Vec<f64>, String> {
    meta_array(source, name)?
        .iter()
        .map(|value| {
            value
                .to_f64()
                .ok_or_else(|| format!("{name} must hold floats"))
        })
        .collect()
}

/// `tokenizer.ggml.tokens` carries the SentencePiece pieces followed by the
/// eleven added GLiNER tokens, so the two are split on `spm.piece_count`.
fn meta_bytes_array(source: &dyn TensorSource, piece_count: usize) -> Result<Vec<Vec<u8>>, String> {
    let tokens = meta_array(source, "tokenizer.ggml.tokens")?;
    if piece_count > tokens.len() {
        return Err(format!(
            "spm.piece_count {piece_count} exceeds {} tokens",
            tokens.len()
        ));
    }
    let mut pieces = Vec::with_capacity(piece_count);
    for value in tokens.iter().take(piece_count) {
        pieces.push(
            value
                .to_string_val()
                .ok_or_else(|| "tokenizer.ggml.tokens must hold strings".to_string())?
                .as_bytes()
                .to_vec(),
        );
    }
    Ok(pieces)
}

fn base64_decode(text: &str) -> Result<Vec<u8>, String> {
    use base64::Engine;
    base64::engine::general_purpose::STANDARD
        .decode(text.as_bytes())
        .map_err(|error| format!("invalid base64 in spm.charsmap_b64: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_is_normalised() {
        let values = softmax(&[1.0, 2.0, 3.0]);
        let total: f32 = values.iter().sum();
        assert!((total - 1.0).abs() < 1e-6, "sum {total}");
        assert!(values[2] > values[1] && values[1] > values[0]);
    }

    #[test]
    fn sigmoid_matches_the_math() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(4.0) > 0.98);
        assert!(sigmoid(-4.0) < 0.02);
    }

    #[test]
    fn single_label_head_picks_the_argmax() {
        let task = Task::new("intent", vec![Label::new("a"), Label::new("b")]);
        let scores = decode(&task, &[-1.0, 3.0]);
        // `class_act = "auto"` on a single-label head is softmax, not sigmoid.
        let total: f32 = scores.iter().map(|score| score.probability).sum();
        assert!(
            (total - 1.0).abs() < 1e-6,
            "single-label head must softmax, got {total}"
        );
        assert!(scores[1].probability > 0.97);
        assert_eq!(select(&task, &scores), vec!["b".to_string()]);
    }

    #[test]
    fn multi_label_head_falls_back_to_argmax() {
        let mut task = Task::new("topics", vec![Label::new("a"), Label::new("b")]);
        task.multi_label = true;
        task.cls_threshold = 0.9;
        let scores = decode(&task, &[-9.0, -8.0]);
        assert_eq!(select(&task, &scores), vec!["b".to_string()]);
    }

    #[test]
    fn multi_label_head_keeps_every_label_over_the_threshold() {
        let mut task = Task::new(
            "topics",
            vec![Label::new("a"), Label::new("b"), Label::new("c")],
        );
        task.multi_label = true;
        task.cls_threshold = 0.4;
        let scores = decode(&task, &[2.0, -3.0, 1.0]);
        assert_eq!(
            select(&task, &scores),
            vec!["a".to_string(), "c".to_string()]
        );
    }

    #[test]
    fn temperature_scales_the_logits() {
        let task = Task::new("rating", vec![Label::new("0"), Label::new("10")]);
        let scores = decode(&task, &[0.0, 2.0]);
        assert!(scores[1].probability > scores[0].probability);
    }

    #[test]
    fn base64_round_trips_the_charsmap_header() {
        // 4 bytes -> 8 chars, verified against the standard alphabet.
        assert_eq!(
            base64_decode("AAAEAA==").unwrap(),
            vec![0x00, 0x00, 0x04, 0x00]
        );
        assert_eq!(
            base64_decode("//79/A==").unwrap(),
            vec![0xFF, 0xFE, 0xFD, 0xFC]
        );
        assert!(base64_decode("not base64!").is_err());
    }
}
