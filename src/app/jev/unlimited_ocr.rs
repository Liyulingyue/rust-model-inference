//! Native-tokenizer label scoring for Unlimited-OCR's causal language decoder.
use super::{grouped, single, types::*, JevInputs};
use crate::models::unlimited_ocr::Model;
use std::time::Instant;

fn label_ids(tokenizer: &tokenizers::Tokenizer, labels: &[char]) -> Result<Vec<u32>, String> {
    labels
        .iter()
        .map(|c| {
            let encoded = tokenizer
                .encode(c.to_string(), false)
                .map_err(|e| e.to_string())?;
            match encoded.get_ids() {
                [id] if (*id as usize) < 129280 => Ok(*id),
                _ => Err(format!(
                    "Unlimited-OCR JEV label {c} must encode as exactly one token"
                )),
            }
        })
        .collect()
}

fn prompt(system: &str, payload: &str) -> String {
    // The original checkpoint has no chat template; retain its instruction prefix convention.
    format!("{system}\n{payload}\nAnswer:")
}

pub fn score(
    model: &Model,
    context: &str,
    questions: &[JevQuestionInput],
    positive: Option<&str>,
    images: &[Vec<u8>],
) -> Result<Vec<JevResult>, String> {
    single::prepare_jev_questions(questions, positive)?
        .iter()
        .map(|q| {
            let labels = single::jev_labels(q);
            let ids = label_ids(&model.tokenizer, &labels)?;
            let instruction = prompt(
                single::jev_system_prompt(q.mode),
                &single::jev_payload_json(context, q)?,
            );
            let tokens = model.tokens(images.len(), &instruction)?;
            let started = Instant::now();
            let logits = model.logits(&tokens, images)?;
            Ok(single::compute_jev_result_with_ids(
                q,
                &labels,
                &ids,
                &logits,
                started.elapsed().as_millis(),
            ))
        })
        .collect()
}

pub fn score_grouped(
    model: &Model,
    context: &str,
    questions: &[JevGroupedQuestionInput],
    mode: JevMode,
    images: &[Vec<u8>],
) -> Result<Vec<JevGroupedResult>, String> {
    grouped::prepare_jev_grouped_questions(questions, mode)?
        .iter()
        .map(|q| {
            let labels = grouped::allocate_group_labels(q);
            let ids = labels
                .iter()
                .map(|v| label_ids(&model.tokenizer, v))
                .collect::<Result<Vec<_>, _>>()?;
            let instruction = prompt(
                grouped::build_grouped_system(),
                &grouped::build_grouped_payload(context, q)?,
            );
            let tokens = model.tokens(images.len(), &instruction)?;
            let started = Instant::now();
            let logits = model.logits(&tokens, images)?;
            Ok(grouped::compute_grouped_jev_result_with_ids(
                q,
                &labels,
                &ids,
                &logits,
                started.elapsed().as_millis(),
            ))
        })
        .collect()
}

pub fn run_cli(
    model: &Model,
    options: &crate::app::CliOptions,
    images: &[Vec<u8>],
) -> Result<(), String> {
    match super::build_jev_inputs(options)?.ok_or("Missing --jev inputs")? {
        JevInputs::Single {
            context,
            questions,
            positive,
        } => {
            for result in score(model, &context, &questions, positive.as_deref(), images)? {
                if options.jev_output_json {
                    println!(
                        "{}",
                        serde_json::to_string(&result).map_err(|e| e.to_string())?
                    );
                } else {
                    println!("{}: {}", result.question, result.choice_label.unwrap());
                    for (label, probability) in result.labels.iter().zip(&result.probabilities) {
                        println!("  {label}: {probability:.6}");
                    }
                    if let Some(score) = result.score {
                        println!("score: {score:.6}");
                    }
                    if let Some(probability) = result.probability_positive {
                        println!("probability_positive: {probability:.6}");
                    }
                }
            }
        }
        JevInputs::Grouped {
            context,
            questions,
            mode,
        } => {
            for result in score_grouped(model, &context, &questions, mode, images)? {
                // The shared serializer preserves per-group probabilities and score fields.
                println!(
                    "{}",
                    serde_json::to_string(&result).map_err(|e| e.to_string())?
                );
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn missing_label_tokens_are_rejected() {
        let tokenizer = tokenizers::Tokenizer::new(tokenizers::models::bpe::BPE::default());
        assert!(label_ids(&tokenizer, &['A', 'B'])
            .unwrap_err()
            .contains("exactly one token"));
    }
    #[test]
    fn scores_use_actual_label_logits_and_independent_group_normalization() {
        let questions = [JevQuestionInput {
            text: "Q".into(),
            options: vec!["yes".into(), "no".into()],
        }];
        let q = single::prepare_jev_questions(&questions, Some("A")).unwrap();
        let result = single::compute_jev_result_with_ids(
            &q[0],
            &['A', 'B'],
            &[2, 0],
            &[0.0, 1000.0, 2.0],
            7,
        );
        assert!((result.probabilities[0] - 0.880797).abs() < 1e-6);
        assert_eq!(result.prefill_ms, 7);
        let qs = [JevGroupedQuestionInput {
            text: "Q".into(),
            groups: vec![
                JevGroupInput {
                    label: "one".into(),
                    options: vec!["a".into(), "b".into()],
                },
                JevGroupInput {
                    label: "two".into(),
                    options: vec!["c".into(), "d".into()],
                },
            ],
        }];
        let q = grouped::prepare_jev_grouped_questions(&qs, JevMode::BlockChoice).unwrap();
        let result = grouped::compute_grouped_jev_result_with_ids(
            &q[0],
            &[vec!['A', 'B'], vec!['C', 'D']],
            &[vec![0, 1], vec![2, 3]],
            &[2.0, 0.0, 100.0, 100.0],
            0,
        );
        assert!((result.groups[0].probabilities[0] - 0.880797).abs() < 1e-6);
        assert_eq!(result.groups[1].probabilities, [0.5, 0.5]);
    }
}
