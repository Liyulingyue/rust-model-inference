//! JEV (Joint Embedding Verification) decision scoring.
//!
//! Two modes are supported:
//!
//! - **Single** (`run_jev_decision`): one question → softmax over
//!   candidates A..Z. Uses [`JevMode::Choice` | `Binary` | `Score`].
//! - **Grouped** (`run_jev_grouped_decision`): one question with
//!   multiple independent groups → per-group softmax. Uses
//!   [`JevMode::MultiSelect` | `BlockChoice`].
//!
//! Submodules:
//! - [`types`]: shared data types ([`JevMode`], [`JevResult`],
//!   [`JevGroupedResult`], etc.).
//! - [`single`]: single-mode scoring (entry point + 9 per-arch
//!   `JevScorer` implementations).
//! - [`grouped`]: grouped-mode scoring (entry point + 8 per-arch
//!   `JevGroupedScorer` implementations).
//! - [`adapters`]: backends that cannot be expressed by [`single::JevScorer`]
//!   — a different architecture, a different scoring rule, or both. See
//!   [`adapters`] for why this exists.
//! - `clm`: the earlier instance of that same pattern, not moved (it predates
//!   the `adapters` layout and is already merged/verified).

pub mod adapters;
pub(crate) mod clm;
pub(crate) mod grouped;
pub(crate) mod single;
pub(crate) mod types;

pub use adapters::gliner2::{
    gliner2_schema, load_gliner2_source, parse_schema, run_gliner2_decision, run_gliner2_scoring,
    schema_from_label_sets, schema_from_questions, LabelSet,
};
pub use adapters::gliner2_boundary::{
    extract as run_gliner2_boundary_extract, parse_boundary_schema, run_gliner2_boundary,
    BoundaryDecodeOptions,
};
pub use clm::{run_clm_decision, run_clm_scoring};
pub use grouped::{
    build_grouped_payload, build_grouped_system, prepare_jev_grouped_questions,
    run_jev_grouped_decision, run_jev_grouped_decision_data,
};
pub use single::{
    image_supported_arch, jev_payload_json, jev_system_prompt, prepare_jev_questions,
    run_jev_decision, run_jev_decision_data,
};
pub use types::{JevGroupInput, JevGroupedQuestionInput, JevMode, JevQuestionInput, JevResult};

use crate::app::cli::CliOptions;

/// JEV input variants constructed from CLI options.
pub enum JevInputs {
    /// Grouped mode (multi-select or block-choice).
    Grouped {
        context: String,
        questions: Vec<JevGroupedQuestionInput>,
        mode: JevMode,
    },
    /// Single mode (one question → softmax over candidates).
    Single {
        context: String,
        questions: Vec<JevQuestionInput>,
        positive: Option<String>,
    },
}

/// Build JEV inputs from CLI options. Returns Ok(None) if JEV mode is not active.
pub fn build_jev_inputs(options: &CliOptions) -> Result<Option<JevInputs>, String> {
    if !options.jev {
        return Ok(None);
    }
    let context = options
        .jev_context
        .clone()
        .ok_or_else(|| "--jev requires --jev-context <text>".to_string())?;
    let output_json = options.jev_output_json;
    let _ = output_json; // consumed by caller

    if options.jev_multi || !options.jev_blocks.is_empty() {
        if options.jev_multi && !options.jev_blocks.is_empty() {
            return Err("--jev-multi and --jev-block are mutually exclusive".into());
        }
        if options.jev_questions.is_empty() {
            return Err("--jev requires at least one --jev-question".into());
        }
        let mode = if options.jev_multi {
            JevMode::MultiSelect
        } else {
            JevMode::BlockChoice
        };
        let inputs: Vec<JevGroupedQuestionInput> = options
            .jev_questions
            .iter()
            .map(|q| {
                if !options.jev_blocks.is_empty() {
                    JevGroupedQuestionInput {
                        text: q.text.clone(),
                        groups: options
                            .jev_blocks
                            .iter()
                            .map(|b| JevGroupInput {
                                label: b.label.clone(),
                                options: b.options.clone(),
                            })
                            .collect(),
                    }
                } else {
                    let pairs: Vec<JevGroupInput> = q
                        .options
                        .chunks(2)
                        .enumerate()
                        .filter_map(|(i, chunk)| {
                            if chunk.len() == 2 {
                                Some(JevGroupInput {
                                    label: format!("pair_{}", i + 1),
                                    options: chunk.to_vec(),
                                })
                            } else {
                                None
                            }
                        })
                        .collect();
                    if pairs.is_empty() {
                        JevGroupedQuestionInput {
                            text: q.text.clone(),
                            groups: Vec::new(),
                        }
                    } else {
                        JevGroupedQuestionInput {
                            text: q.text.clone(),
                            groups: pairs,
                        }
                    }
                }
            })
            .collect();
        Ok(Some(JevInputs::Grouped {
            context,
            questions: inputs,
            mode,
        }))
    } else {
        if options.jev_questions.is_empty() {
            return Err("--jev requires at least one --jev-question".into());
        }
        let questions: Vec<JevQuestionInput> = options
            .jev_questions
            .iter()
            .map(|q| JevQuestionInput {
                text: q.text.clone(),
                options: q.options.clone(),
            })
            .collect();
        Ok(Some(JevInputs::Single {
            context,
            questions,
            positive: options.jev_positive.clone(),
        }))
    }
}
