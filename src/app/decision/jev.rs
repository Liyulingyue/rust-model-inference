//! The JEV adapter for native decision engines.
//!
//! The JEV framework is model-agnostic: it knows how to score one question
//! against a context and how to group questions, but nothing about a
//! decision model that answers several typed questions from one state. That
//! translation lives here, next to the entry points that use it, instead of
//! inside `app::jev`, so the shared framework stays free of model special
//! cases.

use std::sync::Arc;

use serde_json::{json, Map, Value};

use super::super::jev::single::PreparedQuestion;
use super::super::jev::types::{JevGroupResult, JevGroupedResult};
use super::super::{CliOptions, JevInputs, JevMode, JevResult};
use crate::TensorSource;

struct Readout {
    id: String,
    question: PreparedQuestion,
    labels: Vec<char>,
    group: Option<String>,
}

/// One joint native decision request plus the existing JEV output layout.
pub(crate) struct JevPlan {
    pub request: Value,
    context: String,
    mode: Option<JevMode>,
    readouts: Vec<Vec<Readout>>,
}

impl JevPlan {
    pub fn new(inputs: JevInputs, images: Vec<String>) -> Result<Self, String> {
        let (context, mode, readouts) = match inputs {
            JevInputs::Single {
                context,
                questions,
                positive,
            } => {
                let prepared =
                    super::super::prepare_jev_questions(&questions, positive.as_deref())?;
                let readouts = prepared
                    .into_iter()
                    .enumerate()
                    .map(|(i, question)| {
                        let labels = (0..question.descriptions.len())
                            .map(|j| (b'A' + j as u8) as char)
                            .collect();
                        vec![Readout {
                            id: format!("q{i}"),
                            question,
                            labels,
                            group: None,
                        }]
                    })
                    .collect();
                (context, None, readouts)
            }
            JevInputs::Grouped {
                context,
                questions,
                mode,
            } => {
                if !matches!(mode, JevMode::MultiSelect | JevMode::BlockChoice) {
                    return Err("invalid grouped JEV mode".into());
                }
                let prepared = super::super::prepare_jev_grouped_questions(&questions, mode)?;
                let mut readouts = Vec::with_capacity(prepared.len());
                for (i, (question, input)) in prepared.into_iter().zip(&questions).enumerate() {
                    let mut label = b'A';
                    let mut groups = Vec::with_capacity(question.groups.len());
                    for (j, (group, input)) in
                        question.groups.into_iter().zip(&input.groups).enumerate()
                    {
                        let labels = (label..label + group.descriptions.len() as u8)
                            .map(char::from)
                            .collect();
                        label += group.descriptions.len() as u8;
                        let score = input.options.iter().any(|option| option.contains(':'));
                        groups.push(Readout {
                            id: format!("q{i}g{j}"),
                            labels,
                            group: Some(group.label.clone()),
                            question: PreparedQuestion {
                                mode: if score {
                                    JevMode::Score
                                } else {
                                    JevMode::Choice
                                },
                                text: question.text.clone(),
                                descriptions: group.descriptions,
                                values: group.values,
                                positive_label: None,
                            },
                        });
                    }
                    readouts.push(groups);
                }
                (context, Some(mode), readouts)
            }
        };
        let mut questions = Map::new();
        for readout in readouts.iter().flatten() {
            if readout
                .question
                .values
                .iter()
                .any(|value| !value.is_finite())
            {
                return Err("JEV score values must be finite".into());
            }
            if let Some(positive) = readout.question.positive_label {
                if !readout.labels.contains(&positive) {
                    return Err("--jev-positive must name an existing option".into());
                }
            }
            let criteria = if readout.question.mode == JevMode::Score {
                json!(readout.question.descriptions)
            } else {
                Value::Object(
                    readout
                        .labels
                        .iter()
                        .zip(&readout.question.descriptions)
                        .map(|(label, description)| (label.to_string(), json!(description)))
                        .collect(),
                )
            };
            let instructions = match &readout.group {
                Some(group) => format!("{}\n{group}", readout.question.text),
                None => readout.question.text.clone(),
            };
            questions.insert(
                readout.id.clone(),
                json!({
                    "type":if readout.question.mode == JevMode::Score {"score"} else {"choice"},
                    "instructions":instructions,"criteria":criteria,
                }),
            );
        }
        if questions.is_empty() {
            return Err("JEV needs at least one question".into());
        }
        let request = json!({"state":context,"questions":questions,"images":images});
        Ok(Self {
            request,
            context,
            mode,
            readouts,
        })
    }

    pub fn format(&self, native: &Value) -> Result<Value, String> {
        let answers = native
            .get("answers")
            .and_then(Value::as_object)
            .ok_or("decision response is missing answers")?;
        let prefill_ms = native
            .get("latency_ms")
            .and_then(Value::as_f64)
            .unwrap_or(0.0) as u128;
        let mut results = Vec::with_capacity(self.readouts.len());
        for question in &self.readouts {
            let mut readouts = Vec::with_capacity(question.len());
            for readout in question {
                let probabilities = answers
                    .get(&readout.id)
                    .and_then(|answer| answer.get("probabilities"))
                    .and_then(Value::as_object)
                    .ok_or("decision response is missing JEV probabilities")?;
                let q = &readout.question;
                let probabilities =
                    readout
                        .labels
                        .iter()
                        .enumerate()
                        .map(|(i, label)| {
                            let key = if q.mode == JevMode::Score {
                                i.to_string()
                            } else {
                                label.to_string()
                            };
                            let value =
                                probabilities.get(&key).and_then(Value::as_f64).ok_or_else(
                                    || format!("THX-02 response is missing probability {key}"),
                                )?;
                            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                                return Err(format!("invalid THX-02 probability {value}"));
                            }
                            Ok(value as f32)
                        })
                        .collect::<Result<Vec<_>, String>>()?;
                let chosen = probabilities
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .map(|(i, _)| i)
                    .ok_or("THX-02 response has no probabilities")?;
                let mut sorted = probabilities.clone();
                sorted.sort_by(|a, b| b.total_cmp(a));
                let positive = q.positive_label.and_then(|label| {
                    readout
                        .labels
                        .iter()
                        .position(|candidate| *candidate == label)
                });
                readouts.push(JevResult {
                    mode: q.mode,
                    question: q.text.clone(),
                    labels: readout.labels.clone(),
                    descriptions: q.descriptions.clone(),
                    values: q.values.clone(),
                    choice_label: Some(readout.labels[chosen]),
                    positive_label: q.positive_label,
                    probability_positive: positive.map(|i| probabilities[i]),
                    score: (q.mode == JevMode::Score).then(|| {
                        probabilities
                            .iter()
                            .zip(&q.values)
                            .map(|(p, v)| p * v)
                            .sum()
                    }),
                    confidence: probabilities[chosen],
                    entropy: -probabilities
                        .iter()
                        .filter(|p| **p > 0.0)
                        .map(|p| p * p.ln())
                        .sum::<f32>(),
                    margin: sorted[0] - sorted.get(1).copied().unwrap_or(0.0),
                    probabilities,
                    prefill_ms,
                    selected: Vec::new(),
                });
            }
            let result = if let Some(mode) = self.mode {
                serde_json::to_value(JevGroupedResult {
                    mode,
                    question: question[0].question.text.clone(),
                    prefill_ms,
                    groups: readouts
                        .into_iter()
                        .zip(question)
                        .map(|(result, readout)| JevGroupResult {
                            label: readout.group.clone().unwrap_or_default(),
                            labels: result.labels,
                            descriptions: result.descriptions,
                            values: result.values,
                            probabilities: result.probabilities,
                            choice_label: result.choice_label.unwrap(),
                            score: result.score,
                            confidence: result.confidence,
                            entropy: result.entropy,
                            margin: result.margin,
                        })
                        .collect(),
                })
            } else {
                serde_json::to_value(&readouts[0])
            }
            .map_err(|error| error.to_string())?;
            results.push(result);
        }
        Ok(
            json!({"mode":match self.mode {Some(JevMode::MultiSelect)=>"multi_select",Some(_)=>"block_choice",None=>"single"},
            "context":self.context,"results":results}),
        )
    }
}

pub fn run_jev_cli(
    source: Option<Arc<dyn TensorSource>>,
    options: &CliOptions,
    n_threads: usize,
) -> Result<(), String> {
    super::validate_options(options)?;
    if options.jev_multi
        && options
            .jev_questions
            .iter()
            .any(|question| question.options.len() % 2 != 0)
    {
        return Err("native --jev-multi needs an even number of paired options".into());
    }
    let inputs = super::super::build_jev_inputs(options)?
        .ok_or("native decisions require --decision-request or --jev")?;
    let images = options
        .image
        .as_deref()
        .map(|path| path.to_string_lossy().into_owned())
        .into_iter()
        .collect();
    let mut plan = JevPlan::new(inputs, images)?;
    super::normalize_cli_images(&mut plan.request)?;
    let engine = if let Some(source) = source {
        super::from_source(options, source, n_threads)?
    } else {
        super::from_path(options, n_threads)?
    };
    let result = plan.format(&engine.decide(&plan.request)?)?;
    for value in result["results"].as_array().ok_or("missing JEV results")? {
        println!(
            "{}",
            if options.jev_output_json {
                serde_json::to_string(value)
            } else {
                serde_json::to_string_pretty(value)
            }
            .map_err(|error| error.to_string())?
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::{JevGroupInput, JevGroupedQuestionInput, JevQuestionInput};

    #[test]
    fn decision_jev_defers_model_specific_media_limits_to_the_engine() {
        let plan = JevPlan::new(
            JevInputs::Single {
                context: "state".into(),
                questions: vec![JevQuestionInput {
                    text: "pick".into(),
                    options: vec!["yes".into(), "no".into()],
                }],
                positive: None,
            },
            vec!["image.png".into(); 5],
        )
        .unwrap();
        assert!(crate::models::d3_turbo::validate_request(&plan.request).is_ok());
        assert!(crate::models::thx02::validate_request(&plan.request).is_err());
    }

    #[test]
    fn decision_jev_binary_preserves_native_choice_probabilities() {
        let plan = JevPlan::new(
            JevInputs::Single {
                context: "state".into(),
                questions: vec![JevQuestionInput {
                    text: "pick".into(),
                    options: vec!["yes".into(), "no".into()],
                }],
                positive: Some("B".into()),
            },
            vec![],
        )
        .unwrap();
        assert_eq!(plan.request["questions"]["q0"]["type"], "choice");
        let result = plan
            .format(&json!({"answers":{"q0":{"probabilities":{"A":0.25,"B":0.75}}},"latency_ms":3}))
            .unwrap();
        assert_eq!(result["results"][0]["probability"], 0.75);
        assert_eq!(result["results"][0]["choice"], "B");
    }

    #[test]
    fn decision_jev_scores_map_native_indices_to_user_values() {
        let plan = JevPlan::new(
            JevInputs::Single {
                context: "state".into(),
                questions: vec![JevQuestionInput {
                    text: "quality".into(),
                    options: vec!["bad:-2".into(), "good:4".into()],
                }],
                positive: None,
            },
            vec![],
        )
        .unwrap();
        assert_eq!(
            plan.request["questions"]["q0"]["criteria"],
            json!(["bad", "good"])
        );
        let result = plan
            .format(&json!({"answers":{"q0":{"probabilities":{"0":0.25,"1":0.75}}}}))
            .unwrap();
        assert_eq!(result["results"][0]["score"], 2.5);
        assert_eq!(
            result["results"][0]["probabilities"],
            json!({"A":0.25,"B":0.75})
        );
    }

    #[test]
    fn decision_jev_groups_share_one_joint_request_and_keep_global_labels() {
        let plan = JevPlan::new(
            JevInputs::Grouped {
                context: "state".into(),
                mode: JevMode::BlockChoice,
                questions: vec![JevGroupedQuestionInput {
                    text: "pick".into(),
                    groups: vec![
                        JevGroupInput {
                            label: "first".into(),
                            options: vec!["a".into(), "b".into()],
                        },
                        JevGroupInput {
                            label: "second".into(),
                            options: vec!["c".into(), "d".into()],
                        },
                    ],
                }],
            },
            vec!["cat.png".into()],
        )
        .unwrap();
        assert_eq!(plan.request["questions"].as_object().unwrap().len(), 2);
        assert_eq!(
            plan.request["questions"]["q0g1"]["criteria"],
            json!({"C":"c","D":"d"})
        );
        assert_eq!(plan.request["images"], json!(["cat.png"]));
        let result = plan
            .format(&json!({"answers":{
                "q0g0":{"probabilities":{"A":0.25,"B":0.75}},
                "q0g1":{"probabilities":{"C":0.5,"D":0.5}}
            }}))
            .unwrap();
        assert_eq!(result["mode"], "block_choice");
        assert_eq!(
            result["results"][0]["groups"][1]["probabilities"],
            json!({"C":0.5,"D":0.5})
        );
    }
}
