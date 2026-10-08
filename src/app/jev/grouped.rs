use super::types::{
    JevGroupResult, JevGroupedQuestionInput, JevGroupedResult, JevMode, PreparedGroup,
    PreparedGroupedQuestion,
};
use crate::core::tensor::TensorSource;
use crate::core::tokenizer::{BPETokenizer, EncodeOptions};
use crate::prompt::{append_qwen_assistant_prefix, append_qwen_message_tokens};
use std::sync::Arc;
use std::time::Instant;

pub(crate) trait JevGroupedScorer {
    /// Display name used in the dispatch table's error message
    /// (the `eprintln!("compute pool: {} threads (...)")` is owned
    /// by the per-arch wrapper, not the scorer).
    fn scorer_label(&self) -> &'static str;

    /// Build the per-question grouped chat-template token ids
    /// for `q`. Returns the per-group A..Z labels (one vec per
    /// group, in the same order as `q.groups`) and the token id
    /// sequence to feed into `forward_logits`.
    fn build_grouped_prompt(
        &self,
        context: &str,
        q: &PreparedGroupedQuestion,
    ) -> Result<(Vec<Vec<char>>, Vec<u32>), String>;

    /// Run the prefill + LM-head forward for a single grouped
    /// question, returning the final logits and elapsed duration.
    fn forward_logits(
        &mut self,
        token_ids: Vec<u32>,
    ) -> Result<(Vec<f32>, std::time::Duration), String>;

    /// Borrow the tokenizer for `compute_grouped_jev_result`.
    fn tokenizer(&self) -> &BPETokenizer;
}

/// Per-question dispatch shared by every `run_jev_grouped_<arch>`
/// function. Calls `scorer.build_grouped_prompt` for token ids +
/// per-group labels, `scorer.forward_logits` for the prefill,
/// and finally `compute_grouped_jev_result` to score each group.
fn run_jev_grouped_core<S: JevGroupedScorer>(
    _source: Arc<dyn TensorSource>,
    context: &str,
    per_question: &[PreparedGroupedQuestion],
    output_json: bool,
    scorer: &mut S,
) -> Result<Vec<JevGroupedResult>, String> {
    let mut results = Vec::with_capacity(per_question.len());
    for q in per_question {
        let (group_labels, token_ids) = scorer.build_grouped_prompt(context, q)?;
        if !output_json {
            eprintln!("\n--- JEV grouped question ({} groups) ---", q.groups.len());
            eprintln!("Q: {}", q.text);
        }
        let (logits, prefill_dur) = scorer.forward_logits(token_ids)?;
        results.push(compute_grouped_jev_result(
            q,
            scorer.tokenizer(),
            &group_labels,
            &logits,
            prefill_dur.as_millis(),
        ));
    }
    Ok(results)
}

pub(crate) mod gemma4;
pub(crate) mod hunyuan;
pub(crate) mod lfm2;
pub(crate) mod lfm25;
pub(crate) mod llama;
pub(crate) mod nemotron_h;
pub(crate) mod qwen3;
pub(crate) mod qwen35;
pub(crate) mod spark;

pub fn prepare_jev_grouped_questions(
    questions: &[JevGroupedQuestionInput],
    mode: JevMode,
) -> Result<Vec<PreparedGroupedQuestion>, String> {
    if questions.is_empty() {
        return Err("--jev requires at least one --jev-question".into());
    }
    let mut per_question = Vec::with_capacity(questions.len());
    for q in questions {
        if q.groups.is_empty() {
            return Err(format!(
                "Question {:?} has no groups; use --jev-block to define at least one block",
                q.text
            ));
        }
        let mut total_options = 0usize;
        let mut prepared_groups = Vec::with_capacity(q.groups.len());
        for (gi, g) in q.groups.iter().enumerate() {
            if g.options.len() < 2 {
                return Err(format!(
                    "Question {:?} group {} ({:?}) needs at least 2 options, got {}",
                    q.text,
                    gi + 1,
                    g.label,
                    g.options.len()
                ));
            }
            total_options += g.options.len();
            if total_options > 26 {
                return Err(format!(
                    "Question {:?} has {} total options across all groups; JEV supports at most 26 (A..Z)",
                    q.text, total_options
                ));
            }
            let has_colon = g.options.iter().any(|opt| opt.contains(':'));
            let mut descriptions = Vec::with_capacity(g.options.len());
            let mut values = Vec::with_capacity(g.options.len());
            for opt in &g.options {
                if has_colon {
                    let (desc, val_str) = opt.rsplit_once(':').ok_or_else(|| {
                        format!(
                            "Group {:?} option {:?} must be \"description:value\" (mixed with/without colon is not allowed)",
                            g.label, opt
                        )
                    })?;
                    let v: f32 = val_str
                        .parse()
                        .map_err(|e| format!("Score value {:?} is not a float: {}", val_str, e))?;
                    descriptions.push(desc.to_string());
                    values.push(v);
                } else {
                    descriptions.push(opt.clone());
                    values.push(0.0);
                }
            }
            prepared_groups.push(PreparedGroup {
                label: if g.label.is_empty() {
                    format!("group_{}", gi + 1)
                } else {
                    g.label.clone()
                },
                descriptions,
                values,
            });
        }
        per_question.push(PreparedGroupedQuestion {
            mode,
            text: q.text.clone(),
            groups: prepared_groups,
        });
    }
    Ok(per_question)
}

pub fn build_grouped_system() -> &'static str {
    "For each group, select the best option. Reply with only a letter label."
}

pub fn build_grouped_payload(context: &str, q: &PreparedGroupedQuestion) -> Result<String, String> {
    let mut payload = String::from("{\"context\": ");
    payload.push_str(&serde_json::to_string(context).map_err(|e| format!("context json: {e}"))?);
    payload.push_str(", \"question\": ");
    payload.push_str(&serde_json::to_string(&q.text).map_err(|e| format!("question json: {e}"))?);
    payload.push_str(", \"groups\": [");
    let mut label_char = b'A';
    for (gi, group) in q.groups.iter().enumerate() {
        if gi > 0 {
            payload.push_str(", ");
        }
        payload.push('{');
        for (oi, desc) in group.descriptions.iter().enumerate() {
            if oi > 0 {
                payload.push_str(", ");
            }
            payload.push('"');
            payload.push(label_char as char);
            payload.push_str("\": ");
            payload.push_str(&serde_json::to_string(desc).map_err(|e| format!("desc json: {e}"))?);
            label_char += 1;
        }
        payload.push('}');
    }
    payload.push_str("]}");
    Ok(payload)
}

fn allocate_group_labels(q: &PreparedGroupedQuestion) -> Vec<Vec<char>> {
    let mut all = Vec::with_capacity(q.groups.len());
    let mut next = b'A';
    for group in &q.groups {
        let count = group.descriptions.len();
        let labels: Vec<char> = (next..next + count as u8).map(|b| b as char).collect();
        next += count as u8;
        all.push(labels);
    }
    all
}

/// Render the grouped prompt through the model's own template.
///
/// Grouped scorers build their prompt here rather than in the single-mode
/// `build_prompt`, which is why `--jinja` used to be stored on the scorer and
/// then never read: grouped JEV silently ignored the flag. Returns `None` when
/// Jinja2 is off, so the caller keeps its hand-built ChatML.
fn grouped_prompt_via_jinja(
    tokenizer: &dyn crate::core::tokenizer::Tokenizer,
    system: &str,
    payload: &str,
    jinja: Option<&crate::prompt::jinja::JinjaChatTemplate>,
) -> Result<Option<Vec<u32>>, String> {
    let Some(template) = jinja else {
        return Ok(None);
    };
    crate::prompt::jinja::render_text_conversation(
        tokenizer,
        template,
        Some(system),
        payload,
        false,
    )
    .map(Some)
}

fn build_jev_token_ids_for_arch(
    arch: &str,
    tokenizer: &BPETokenizer,
    system: &str,
    payload: &str,
    jinja: Option<&crate::prompt::jinja::JinjaChatTemplate>,
) -> Result<Vec<u32>, String> {
    if let Some(ids) = grouped_prompt_via_jinja(tokenizer, system, payload, jinja)? {
        return Ok(ids);
    }
    match arch {
        "qwen3" | "qwen3vl" | "hunyuan-dense" => {
            let mut token_ids = Vec::new();
            append_qwen_message_tokens(
                &mut token_ids,
                tokenizer,
                "system",
                &tokenizer.encode(
                    system,
                    EncodeOptions {
                        add_special: false,
                        parse_special: false,
                    },
                ),
            )?;
            append_qwen_message_tokens(
                &mut token_ids,
                tokenizer,
                "user",
                &tokenizer.encode(
                    payload,
                    EncodeOptions {
                        add_special: false,
                        parse_special: false,
                    },
                ),
            )?;
            append_qwen_assistant_prefix(&mut token_ids, tokenizer, false)?;
            Ok(token_ids)
        }
        "qwen35" => {
            let mut token_ids = Vec::new();
            append_qwen_message_tokens(
                &mut token_ids,
                tokenizer,
                "system",
                &tokenizer.encode(
                    system,
                    EncodeOptions {
                        add_special: false,
                        parse_special: false,
                    },
                ),
            )?;
            append_qwen_message_tokens(
                &mut token_ids,
                tokenizer,
                "user",
                &tokenizer.encode(
                    payload,
                    EncodeOptions {
                        add_special: false,
                        parse_special: false,
                    },
                ),
            )?;
            append_qwen_assistant_prefix(&mut token_ids, tokenizer, false)?;
            Ok(token_ids)
        }
        // Mistral 3 (`mistral3` arch — covers Ministral-3-3B-Instruct /
        // -3B-Reasoning / -8B-Instruct / Mistral-Shieldstral-1.0-3B,
        // all of which ride the llama trunk and use the Tekken / Mistral
        // `[INST] … [/INST]` chat-template family). Same
        // `[INST] {system} {payload} [/INST]` shape the single scorer
        // uses (`src/app/jev/single/llama.rs::build_prompt` `mistral3`
        // arm); folds the JEV system back into the user turn the same
        // way (Mistral 3's own chat_template splits system out via
        // `[SYSTEM_PROMPT]`, but folding it back into `[INST]` is what
        // the 3B checkpoint was actually post-trained on for plain
        // question answering). The same ~56-81% always-A positional
        // bias documented in `docs/usage/ministral3.md` §3.1 applies
        // to the JSON instruction shape; call out here so future
        // maintainers don't think a mistral3 arm is missing by mistake.
        "mistral3" => {
            let prompt = format!("[INST] {system} {payload} [/INST]");
            Ok(tokenizer.encode(
                &prompt,
                EncodeOptions {
                    add_special: false,
                    parse_special: true,
                },
            ))
        }
        "llama" | "k2-horizon" | "granite" | "nanbeige" | "qwen2_2" | "glm4" => {
            if arch == "k2-horizon" || arch == "granite" {
                let prompt = format!(
                    "<|start_of_role|>system<|end_of_role|>{system}<|end_of_text|>\n\
                     <|start_of_role|>user<|end_of_role|>{payload}<|end_of_text|>\n\
                     <|start_of_role|>assistant<|end_of_role|>"
                );
                let mut ids = tokenizer.encode(
                    &prompt,
                    EncodeOptions {
                        add_special: false,
                        parse_special: true,
                    },
                );
                if let Some(bos) = tokenizer.bos_id() {
                    if ids.first() != Some(&bos) {
                        ids.insert(0, bos);
                    }
                }
                Ok(ids)
            } else if arch == "nanbeige" {
                let prompt = format!("{system}\n\n{payload}\n\nAnswer:");
                Ok(tokenizer.encode(
                    &prompt,
                    EncodeOptions {
                        add_special: true,
                        parse_special: true,
                    },
                ))
            } else if arch == "glm4" {
                let prompt = format!("[gMASK]<sop><|user|>\n{payload}<|assistant|>\n");
                let mut ids = tokenizer.encode(
                    &prompt,
                    EncodeOptions {
                        add_special: false,
                        parse_special: true,
                    },
                );
                Ok(ids)
            } else {
                let prompt = format!("system\n{system}\nuser\n{payload}\nassistant\n");
                let mut ids = tokenizer.encode(
                    &prompt,
                    EncodeOptions {
                        add_special: false,
                        parse_special: true,
                    },
                );
                if let Some(bos) = tokenizer.bos_id() {
                    ids.insert(0, bos);
                }
                Ok(ids)
            }
        }
        // Phi-3 / Phi-4 chat template is single-turn `<|user|>…<|end|><|assistant|>`
        // with no system role. `payload` already folds the system instructions
        // and question into the user message.
        "phi3" => {
            let prompt = format!("<|user|>{payload}<|end|><|assistant|>");
            Ok(tokenizer.encode(
                &prompt,
                EncodeOptions {
                    add_special: true,
                    parse_special: true,
                },
            ))
        }
        "gemma4" => {
            let prompt = format!("{system}\n\n{payload}\n\n<turn|>\n<|turn>model\n");
            let bos = tokenizer.bos_id().ok_or("Gemma4 tokenizer missing BOS")?;
            let mut ids = tokenizer.encode(
                &prompt,
                EncodeOptions {
                    add_special: false,
                    parse_special: true,
                },
            );
            if ids.first() != Some(&bos) {
                ids.insert(0, bos);
            }
            Ok(ids)
        }
        "lfm2" | "lfm25" => {
            let mut token_ids = Vec::new();
            if let Some(bos) = tokenizer.bos_id() {
                token_ids.push(bos);
            }
            token_ids.extend(tokenizer.encode(
                &format!("system\n{system}\n"),
                EncodeOptions {
                    add_special: false,
                    parse_special: false,
                },
            ));
            token_ids.extend(tokenizer.encode(
                &format!("user\n{payload}\n"),
                EncodeOptions {
                    add_special: false,
                    parse_special: false,
                },
            ));
            token_ids.extend(tokenizer.encode(
                "assistant\n",
                EncodeOptions {
                    add_special: false,
                    parse_special: false,
                },
            ));
            Ok(token_ids)
        }
        "spark2_5" => {
            let sos = "<｜start▁of▁sentence｜>";
            let eos = "<｜end▁of▁sentence｜>";
            let prompt = format!(
                "{sos}<|System|>\n{system}{eos}\
                 {sos}<|User|>{payload}{eos}\
                 {sos}<|Bot|>\n"
            );
            let mut token_ids = tokenizer.encode(
                &prompt,
                EncodeOptions {
                    add_special: false,
                    parse_special: true,
                },
            );
            if tokenizer.add_bos() {
                if let Some(bos) = tokenizer.bos_id() {
                    token_ids.insert(0, bos);
                }
            }
            Ok(token_ids)
        }
        "nemotron_h" => {
            let prompt = format!("{system}\n\n{payload}\n\nAnswer:");
            Ok(tokenizer.encode(
                &prompt,
                EncodeOptions {
                    add_special: true,
                    parse_special: true,
                },
            ))
        }
        other => Err(format!(
            "--jev grouped is not yet supported for architecture {:?}; \
             currently supported: qwen3 / qwen3vl / qwen35 / llama / k2-horizon / \
             granite / nanbeige / qwen2_2 / gemma4 / lfm2 / lfm25 / spark2_5 / \
             mistral3 / hunyuan-dense / nemotron_h",
            other
        )),
    }
}

fn compute_grouped_jev_result(
    q: &PreparedGroupedQuestion,
    tokenizer: &BPETokenizer,
    group_labels: &[Vec<char>],
    logits: &[f32],
    prefill_ms: u128,
) -> JevGroupedResult {
    let mut groups = Vec::with_capacity(q.groups.len());
    for (gi, group) in q.groups.iter().enumerate() {
        let labels = &group_labels[gi];
        let label_ids: Vec<u32> = labels
            .iter()
            .map(|c| {
                let s = c.to_string();
                tokenizer
                    .encode(
                        &s,
                        EncodeOptions {
                            add_special: false,
                            parse_special: false,
                        },
                    )
                    .into_iter()
                    .next()
                    .unwrap_or(0)
            })
            .collect();
        let group_logits: Vec<f32> = label_ids.iter().map(|&id| logits[id as usize]).collect();
        let max_logit = group_logits
            .iter()
            .fold(f32::NEG_INFINITY, |a, &b| f32::max(a, b));
        let mut exps: Vec<f32> = group_logits
            .iter()
            .map(|&z| (z - max_logit).exp())
            .collect();
        let sum: f32 = exps.iter().sum();
        for v in exps.iter_mut() {
            *v /= sum;
        }
        let chosen_idx = exps
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        let confidence = exps[chosen_idx];
        let entropy: f32 = -exps
            .iter()
            .filter(|p| **p > 0.0)
            .map(|p| p * p.ln())
            .sum::<f32>();
        let mut sorted = exps.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let margin = if sorted.len() >= 2 {
            sorted[0] - sorted[1]
        } else {
            sorted[0]
        };
        let has_values = group.values.iter().any(|v| *v != 0.0);
        let score = if has_values {
            Some(
                exps.iter()
                    .zip(group.values.iter())
                    .map(|(p, v)| p * v)
                    .sum(),
            )
        } else {
            None
        };
        groups.push(JevGroupResult {
            label: group.label.clone(),
            labels: labels.to_vec(),
            descriptions: group.descriptions.clone(),
            values: group.values.clone(),
            probabilities: exps,
            choice_label: labels[chosen_idx],
            score,
            confidence,
            entropy,
            margin,
        });
    }
    JevGroupedResult {
        mode: q.mode,
        question: q.text.clone(),
        groups,
        prefill_ms,
    }
}

/// Same as [`run_jev_grouped_decision`] but returns the structured
/// `Vec<JevGroupedResult>` instead of formatting output. The HTTP
/// server's `/v1/jev/grouped` endpoint uses this; the CLI
/// `--jev-multi` / `--jev-block` paths still go through the printer
/// wrapper below.
pub fn run_jev_grouped_decision_data(
    source: Arc<dyn TensorSource>,
    context: &str,
    questions: &[JevGroupedQuestionInput],
    mode: JevMode,
    n_threads_arg: usize,
    prefill_batch_size: usize,
    jinja: crate::prompt::jinja::Options,
) -> Result<Vec<JevGroupedResult>, String> {
    let prepared = prepare_jev_grouped_questions(questions, mode)?;
    let arch = source
        .metadata("general.architecture")
        .and_then(|v| v.to_string_val())
        .unwrap_or_default();
    eprintln!("JEV grouped: arch = {:?}, mode = {:?}", arch, mode);

    let results = match &*arch {
        "qwen3" | "qwen3vl" => qwen3::run_jev_grouped_qwen3(
            source.clone(),
            context,
            &prepared,
            n_threads_arg,
            prefill_batch_size,
            false,
            jinja.clone(),
        )?,
        "qwen35" => qwen35::run_jev_grouped_qwen35(
            source.clone(),
            context,
            &prepared,
            n_threads_arg,
            prefill_batch_size,
            false,
            jinja.clone(),
        )?,
        "llama" | "k2-horizon" | "granite" | "nanbeige" | "qwen2_2" | "phi3" | "glm4"
        | "mistral3" => llama::run_jev_grouped_llama(
            source.clone(),
            context,
            &prepared,
            n_threads_arg,
            prefill_batch_size,
            false,
            jinja.clone(),
        )?,
        "gemma4" => gemma4::run_jev_grouped_gemma4(
            source.clone(),
            context,
            &prepared,
            n_threads_arg,
            prefill_batch_size,
            false,
            jinja.clone(),
        )?,
        "lfm2" => lfm2::run_jev_grouped_lfm2(
            source.clone(),
            context,
            &prepared,
            n_threads_arg,
            prefill_batch_size,
            false,
            jinja.clone(),
        )?,
        "lfm25" => lfm25::run_jev_grouped_lfm25(
            source.clone(),
            context,
            &prepared,
            n_threads_arg,
            prefill_batch_size,
            false,
            jinja.clone(),
        )?,
        "spark2_5" => spark::run_jev_grouped_spark(
            source.clone(),
            context,
            &prepared,
            n_threads_arg,
            prefill_batch_size,
            false,
            jinja.clone(),
        )?,
        "nemotron_h" => nemotron_h::run_jev_grouped_nemotron_h(
            source.clone(),
            context,
            &prepared,
            n_threads_arg,
            prefill_batch_size,
            false,
            jinja.clone(),
        )?,
        "hunyuan-dense" => hunyuan::run_jev_grouped_hunyuan(
            source.clone(),
            context,
            &prepared,
            n_threads_arg,
            prefill_batch_size,
            false,
            jinja.clone(),
        )?,
        other => {
            return Err(format!(
                "--jev grouped is not yet supported for architecture {:?}; \
                 currently supported: qwen3 / qwen3vl / qwen35 / llama / k2-horizon / \
                 granite / nanbeige / qwen2_2 / gemma4 / lfm2 / lfm25 / spark2_5 / \
                 mistral3 / hunyuan-dense / nemotron_h",
                other
            ));
        }
    };

    Ok(results)
}

pub fn run_jev_grouped_decision(
    source: Arc<dyn TensorSource>,
    context: &str,
    questions: &[JevGroupedQuestionInput],
    mode: JevMode,
    n_threads_arg: usize,
    prefill_batch_size: usize,
    output_json: bool,
    jinja: crate::prompt::jinja::Options,
) -> Result<(), String> {
    let t0 = Instant::now();
    let results = run_jev_grouped_decision_data(
        source.clone(),
        context,
        questions,
        mode,
        n_threads_arg,
        prefill_batch_size,
        jinja,
    )?;

    if output_json {
        for r in &results {
            let line = serde_json::to_string(r).map_err(|e| format!("json encode: {e}"))?;
            println!("{}", line);
        }
    } else if results.len() == 1 {
        print_grouped_result_text(&results[0]);
    } else {
        println!(
            "\n--- JEV grouped decisions ({} questions) ---",
            results.len()
        );
        for r in &results {
            println!("\nQ: {}", r.question);
            print_grouped_result_text(r);
        }
    }

    let total_ms = t0.elapsed().as_millis();
    eprintln!(
        "\nJEV grouped total: {} ms ({} questions)",
        total_ms,
        results.len()
    );
    Ok(())
}

fn print_grouped_result_text(r: &JevGroupedResult) {
    println!("\n--- JEV decision ({:?}) ---", r.mode);
    for g in &r.groups {
        if let Some(score) = g.score {
            println!("  [{}] score: {:.4}", g.label, score);
            for (i, p) in g.probabilities.iter().enumerate() {
                println!(
                    "    {}: {:.4} × {} = {:.4} — {}",
                    g.labels[i],
                    p,
                    g.values[i],
                    p * g.values[i],
                    g.descriptions[i]
                );
            }
        } else {
            println!("  [{}] choice: {}", g.label, g.choice_label);
            for (i, p) in g.probabilities.iter().enumerate() {
                println!("    {}: {:.4} — {}", g.labels[i], p, g.descriptions[i]);
            }
        }
        println!(
            "    confidence: {:.4} | entropy: {:.4} | margin: {:.4}",
            g.confidence, g.entropy, g.margin
        );
    }
    println!("prefill: {} ms", r.prefill_ms);
}

#[cfg(test)]
mod tests {
    use super::grouped_prompt_via_jinja;
    use crate::core::tokenizer::{MockTokenizer, Tokenizer};
    use crate::prompt::jinja::JinjaChatTemplate;

    /// Grouped JEV used to store `--jinja` on the scorer and then never read
    /// it, because the prompt is built by this free function rather than by the
    /// single-mode `build_prompt`. The marker text proves the template shaped
    /// the prompt.
    #[test]
    fn grouped_prompt_uses_the_jinja_template() {
        let template = JinjaChatTemplate::compile(
            "JINJA-MARKER|{{ messages[0].content }}|{{ messages[1].content }}",
            "test",
        )
        .unwrap();
        let tok = MockTokenizer::new();
        let ids = grouped_prompt_via_jinja(&tok, "SYS", "PAY", Some(&template))
            .unwrap()
            .expect("jinja path must render");
        assert_eq!(tok.decode(&ids, false), "JINJA-MARKER|SYS|PAY");
    }

    #[test]
    fn grouped_prompt_without_jinja_is_left_to_the_caller() {
        // `None` means "keep hand-building ChatML", so enabling the flag never
        // changes the default prompts.
        let tok = MockTokenizer::new();
        assert!(grouped_prompt_via_jinja(&tok, "SYS", "PAY", None)
            .unwrap()
            .is_none());
    }
}
