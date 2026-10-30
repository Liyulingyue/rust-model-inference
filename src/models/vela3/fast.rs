//! Native Vela3-fast decision, router and word span inference.
use super::tensor::{layer_norm, linear, open_weights, safetensor_metadata, values, OverlaySource};
use super::{text, ultrabert};
use crate::core::tensor::TensorSource;
use crate::core::thread_pool::ComputePool;
use crate::format::safetensors::SafetensorSource;
use serde_json::{json, Map, Value};
use std::collections::HashMap;
use std::path::Path;
use tokenizers::Tokenizer;

#[derive(Clone, Debug)]
struct Segment {
    role: String,
    ids: Vec<u32>,
}
#[derive(Clone, Debug)]
struct Record {
    ids: Vec<u32>,
    ranges: HashMap<String, (usize, usize, usize)>,
    truncated: bool,
}
#[derive(Clone)]
struct Question {
    key: String,
    typ: String,
    qpos: usize,
    positions: Vec<usize>,
    names: Vec<String>,
    abstain: bool,
}
struct Adapter {
    source: SafetensorSource,
    scale: f32,
}
struct FastModel<'a> {
    root: &'a Path,
    source: SafetensorSource,
    config: Value,
    runtime: Value,
    routers: Map<String, Value>,
    tok: Tokenizer,
}
struct Raw {
    logits: Vec<Vec<f64>>,
    span: Vec<f64>,
    offsets: Vec<(usize, usize)>,
    target_tokens: usize,
    input_tokens: usize,
}

fn read_json(path: &Path) -> Result<Value, String> {
    serde_json::from_slice(
        &std::fs::read(path).map_err(|e| format!("Read {}: {e}", path.display()))?,
    )
    .map_err(|e| format!("Parse {}: {e}", path.display()))
}
fn render(value: &Value) -> String {
    value
        .as_str()
        .map(str::to_owned)
        .unwrap_or_else(|| value.to_string())
}
fn norm(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}
fn described(name: &str, description: &str) -> String {
    if description.is_empty() {
        name.into()
    } else {
        format!("{name}: {description}")
    }
}
fn string_options(value: &Value) -> Result<Vec<(String, String)>, String> {
    match value {
        Value::Object(m) => Ok(m
            .iter()
            .map(|(n, d)| {
                (
                    n.clone(),
                    if d.is_null() {
                        String::new()
                    } else {
                        render(d)
                    },
                )
            })
            .collect()),
        Value::Array(a) => Ok(a.iter().map(|n| (render(n), String::new())).collect()),
        _ => Err("criteria must be an object or list".into()),
    }
}
fn parts(state: &Value, strict: bool) -> Result<Vec<(String, String)>, String> {
    if let Some(s) = state.as_str() {
        return Ok(vec![("user".into(), s.into())]);
    }
    if let Some(m) = state
        .as_object()
        .filter(|m| !m.is_empty() && m.values().all(Value::is_string))
    {
        if m.keys()
            .all(|k| matches!(k.as_str(), "user" | "context" | "answer"))
        {
            return Ok(m
                .iter()
                .map(|(r, t)| (r.clone(), t.as_str().unwrap().into()))
                .collect());
        }
        if strict {
            return Err("span / router state parts must be named user, context, answer".into());
        }
    }
    if strict {
        Err(
            "span questions need a text state or an object of user/context/answer text parts"
                .into(),
        )
    } else {
        Ok(vec![("user".into(), state.to_string())])
    }
}
fn error(typ: &Value, kind: &str, message: impl AsRef<str>) -> Value {
    json!({"type":typ,"error":kind,"message":message.as_ref()})
}

pub(crate) fn predict(root: &Path, request: &Value, pool: &ComputePool) -> Result<Value, String> {
    let questions = request
        .get("questions")
        .and_then(Value::as_object)
        .ok_or("Vela3 questions must be an object")?;
    if request
        .get("images")
        .is_some_and(|v| !v.is_null() && v.as_array().is_none_or(|a| !a.is_empty()))
    {
        return Err("vela3-fast accepts text; use a vision model for images".into());
    }
    let model = FastModel::load(root)?;
    let state = request.get("state").unwrap_or(&Value::Null);
    let mut answers = Map::new();
    let mut decisions = Vec::new();
    let mut router_groups: Vec<(String, Option<String>, Vec<(String, Value, Value)>)> = Vec::new();
    let mut spans = Vec::new();
    for (key, q) in questions {
        let typ = q.get("type").unwrap_or(&Value::Null);
        match typ.as_str() {
            Some("span") => spans.push((key, q)),
            Some(t @ ("choice" | "set"))
                if model
                    .routers
                    .contains_key(&norm(q["instructions"].as_str().unwrap_or(""))) =>
            {
                let schema = &model.routers[&norm(q["instructions"].as_str().unwrap_or(""))];
                if schema["type"] != t {
                    answers.insert(
                        key.clone(),
                        error(
                            typ,
                            "invalid_question",
                            "router question asked with another type than trained",
                        ),
                    );
                    continue;
                }
                let roles = match parts(state, false) {
                    Ok(p) => p.into_iter().map(|(r, _)| r).collect::<Vec<_>>(),
                    Err(e) => {
                        answers.insert(key.clone(), error(typ, "invalid_question", e));
                        continue;
                    }
                };
                let target = q["over"].as_str().map(str::to_owned).unwrap_or_else(|| {
                    schema["over"]
                        .as_str()
                        .filter(|s| roles.iter().any(|r| r == s))
                        .unwrap_or(roles.last().unwrap())
                        .into()
                });
                let mut task = None;
                if let Some(tasks) = model.runtime["router"]["task_adapters"].as_object() {
                    for (name, ta) in tasks {
                        if ta["ids"]
                            .as_array()
                            .is_some_and(|a| a.contains(&schema["id"]))
                            && model.task_files(ta).iter().all(|f| root.join(f).is_file())
                        {
                            task = Some(name.clone());
                            break;
                        }
                    }
                }
                if let Some((_, _, group)) = router_groups
                    .iter_mut()
                    .find(|(r, t, _)| r == &target && t == &task)
                {
                    group.push((key.clone(), q.clone(), schema.clone()));
                } else {
                    router_groups.push((
                        target,
                        task,
                        vec![(key.clone(), q.clone(), schema.clone())],
                    ));
                }
            }
            Some("noul") => decisions.push((key, q)),
            Some("choice")
                if q["criteria"].as_object().is_some_and(|m| !m.is_empty())
                    || q["criteria"].as_array().is_some_and(|a| !a.is_empty()) =>
            {
                decisions.push((key, q))
            }
            Some("choice") => {
                answers.insert(key.clone(),error(typ,"invalid_question","choice questions need criteria: {option: description or null} or a list of options"));
            }
            _ => {
                answers.insert(
                    key.clone(),
                    error(
                        typ,
                        "invalid_question",
                        "unsupported question type (choice, noul, span; set for router questions)",
                    ),
                );
            }
        }
    }
    let mut input_tokens = 0;
    let mut decision_answers = Map::new();
    let mut decision_tokens = 0;
    let mut decision_error = None;
    for (key, q) in &decisions {
        match model.decision(state, q, pool) {
            Ok((a, n)) => {
                decision_answers.insert((*key).clone(), a);
                decision_tokens += n;
            }
            Err(e) => {
                decision_error = Some(e);
                break;
            }
        }
    }
    if let Some(e) = decision_error {
        for (key, q) in decisions {
            answers.insert(key.clone(), error(&q["type"], "unsupported", &e));
        }
    } else {
        answers.extend(decision_answers);
        input_tokens += decision_tokens;
    }
    for (target, task, group) in router_groups {
        match model.router(state, &target, task.as_deref(), &group, pool) {
            Ok((a, n)) => {
                answers.extend(a);
                input_tokens += n;
            }
            Err(e) => {
                for (key, q, _) in group {
                    answers.insert(key, error(&q["type"], "invalid_question", &e));
                }
            }
        }
    }
    for (key, q) in spans {
        match model.span(state, q, pool) {
            Ok((answer, n)) => {
                answers.insert(key.clone(), answer);
                input_tokens += n;
            }
            Err(e) => {
                answers.insert(key.clone(), error(&q["type"], "invalid_question", e));
            }
        }
    }
    let ordered: Map<_, _> = questions
        .keys()
        .map(|k| (k.clone(), answers.remove(k).unwrap()))
        .collect();
    Ok(
        json!({"model":format!("vllm-sr/{}",model.runtime["name"].as_str().unwrap_or("vela3-fast")),"answers":ordered,"usage":{"input_tokens":input_tokens,"output_tokens":0}}),
    )
}

impl<'a> FastModel<'a> {
    fn load(root: &'a Path) -> Result<Self, String> {
        let config = read_json(&root.join("config.json"))?;
        let runtime = read_json(&root.join("vela3_config.json"))?;
        let routers = read_json(&root.join("router_questions.json"))?["questions"]
            .as_object()
            .ok_or("missing router questions")?
            .iter()
            .map(|(k, v)| (norm(k), v.clone()))
            .collect();
        let mut tokenizer = read_json(&root.join("tokenizer.json"))?;
        if let Some(added) = tokenizer["added_tokens"].as_array_mut() {
            added.retain(|t| {
                config["decision"]["marker_ids"]
                    .get(t["content"].as_str().unwrap_or(""))
                    .is_none()
            });
        }
        let mut tok = Tokenizer::from_bytes(tokenizer.to_string()).map_err(|e| e.to_string())?;
        tok.with_padding(None);
        tok.with_truncation(None).map_err(|e| e.to_string())?;
        Ok(Self {
            root,
            source: open_weights(root)?,
            config,
            runtime,
            routers,
            tok,
        })
    }
    fn ids(&self, text: &str) -> Result<Vec<u32>, String> {
        Ok(self
            .tok
            .encode(text, false)
            .map_err(|e| e.to_string())?
            .get_ids()
            .to_vec())
    }
    fn marker(&self, name: &str) -> Result<u32, String> {
        self.config["decision"]["marker_ids"][name]
            .as_u64()
            .and_then(|v| u32::try_from(v).ok())
            .ok_or_else(|| format!("missing Vela3 marker {name}"))
    }
    fn boundary(&self, name: &str) -> Result<u32, String> {
        self.config[name]
            .as_u64()
            .and_then(|v| u32::try_from(v).ok())
            .ok_or_else(|| format!("invalid Vela3 {name}"))
    }
    fn adapter(&self, file: &str) -> Result<Adapter, String> {
        let path = self.root.join(file);
        let metadata = safetensor_metadata(&path)?;
        let rank = metadata["lora_rank"]
            .as_str()
            .and_then(|s| s.parse::<f32>().ok())
            .or_else(|| metadata["lora_rank"].as_f64().map(|x| x as f32))
            .ok_or("missing adapter lora_rank")?;
        let alpha = metadata["lora_alpha"]
            .as_str()
            .and_then(|s| s.parse::<f32>().ok())
            .or_else(|| metadata["lora_alpha"].as_f64().map(|x| x as f32))
            .ok_or("missing adapter lora_alpha")?;
        if rank <= 0. || !rank.is_finite() || !alpha.is_finite() {
            return Err("invalid adapter rank or alpha".into());
        }
        Ok(Adapter {
            source: SafetensorSource::open(&[path])?,
            scale: alpha / rank,
        })
    }
    fn task_files(&self, task: &Value) -> Vec<String> {
        task["files"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(Value::as_str)
                    .map(str::to_owned)
                    .collect()
            })
            .unwrap_or_else(|| {
                task["file"]
                    .as_str()
                    .map(|s| vec![s.into()])
                    .unwrap_or_default()
            })
    }
    fn decision(
        &self,
        state: &Value,
        q: &Value,
        pool: &ComputePool,
    ) -> Result<(Value, usize), String> {
        let options = if q["type"] == "noul" {
            vec![("no".into(), String::new()), ("yes".into(), String::new())]
        } else {
            string_options(&q["criteria"])?
        };
        let instructions = q.get("instructions").map(render).unwrap_or_default();
        let mut ids = vec![self.boundary("bos_token_id")?, self.marker("[Q]")?];
        ids.extend(self.ids(&instructions)?);
        let mut positions = Vec::new();
        let mut names = Vec::new();
        for (n, d) in options {
            if names.contains(&n) {
                continue;
            }
            positions.push(ids.len());
            names.push(n.clone());
            ids.push(self.marker("[O]")?);
            ids.extend(self.ids(&described(&n, &d))?);
        }
        ids.push(self.marker("[SEP_SCHEMA]")?);
        let ps = ids.len();
        ids.push(self.marker("[SEG_user]")?);
        ids.extend(self.ids(&if state.is_null() {
            String::new()
        } else {
            render(state)
        })?);
        let pe = ids.len();
        ids.push(self.boundary("eos_token_id")?);
        let max = self.config["decision"]["max_len"].as_u64().unwrap_or(32768) as usize;
        if ids.len() > max {
            return Err(format!("input longer than the {max}-token context"));
        }
        let h = ultrabert::encode(&self.source, &self.config, "encoder.", &ids, None, pool)?;
        let logits = head_logits(
            &self.source,
            &h,
            self.hidden()?,
            1,
            &positions,
            (ps, pe),
            pool,
        )?;
        let probabilities = softmax(&logits)?;
        let answer = if q["type"] == "noul" {
            json!({"type":"noul","noul":probabilities[1]})
        } else {
            choice_answer(&names, &probabilities, None)?
        };
        Ok((answer, ids.len()))
    }
    fn hidden(&self) -> Result<usize, String> {
        self.config["hidden_size"]
            .as_u64()
            .and_then(|v| usize::try_from(v).ok())
            .filter(|&v| v > 0)
            .ok_or("invalid hidden_size".into())
    }
    fn router(
        &self,
        state: &Value,
        target: &str,
        task: Option<&str>,
        group: &[(String, Value, Value)],
        pool: &ComputePool,
    ) -> Result<(Map<String, Value>, usize), String> {
        let parts = parts(state, false)?;
        let mut schema = Vec::new();
        let mut questions = Vec::new();
        for (key, q, trained) in group {
            let default_options = string_options(&trained["options"])?;
            let supplied = q.get("criteria").filter(|v| !v.is_null());
            let options = if let Some(crit) = supplied {
                let opts = string_options(crit)?;
                if opts.is_empty() {
                    default_options.clone()
                } else {
                    opts.into_iter()
                        .map(|(n, d)| {
                            let desc = if d.is_empty() {
                                default_options
                                    .iter()
                                    .find(|(name, _)| name == &n)
                                    .map(|(_, d)| d.clone())
                                    .unwrap_or_default()
                            } else {
                                d
                            };
                            (n, desc)
                        })
                        .collect()
                }
            } else {
                default_options
            };
            if options.is_empty() {
                return Err("router questions need at least one option".into());
            }
            let qpos = 1 + schema.len();
            schema.push(self.marker("[Q]")?);
            schema.extend(
                self.ids(
                    q["instructions"]
                        .as_str()
                        .ok_or("router instructions must be text")?,
                )?,
            );
            let mut positions = Vec::new();
            let mut names = Vec::new();
            for (n, d) in options {
                if names.contains(&n) {
                    continue;
                }
                positions.push(1 + schema.len());
                names.push(n.clone());
                schema.push(self.marker("[O]")?);
                schema.extend(self.ids(&described(&n, &d))?);
            }
            let abstain = trained["abstain"].as_bool().unwrap_or(false);
            if abstain {
                positions.push(1 + schema.len());
                schema.push(self.marker("[ABS]")?);
            }
            questions.push(Question {
                key: key.clone(),
                typ: trained["type"]
                    .as_str()
                    .ok_or("invalid trained router type")?
                    .into(),
                qpos,
                positions,
                names,
                abstain,
            });
        }
        let files = task
            .map(|t| self.task_files(&self.runtime["router"]["task_adapters"][t]))
            .unwrap_or_else(|| vec!["router_adapter.safetensors".into()]);
        if files.is_empty() {
            return Err("router adapter ensemble is empty".into());
        }
        let mut average = Vec::<Vec<f64>>::new();
        let mut usage = 0;
        for (file_index, file) in files.iter().enumerate() {
            let adapter = self.adapter(file)?;
            let raw = self.schema_raw(
                &parts,
                target,
                &schema,
                &questions,
                &[],
                8192,
                &adapter,
                pool,
            )?;
            if file_index == 0 {
                usage = raw.input_tokens;
                average = raw.logits;
            } else {
                for (acc, other) in average.iter_mut().zip(raw.logits) {
                    for (a, b) in acc.iter_mut().zip(other) {
                        *a += b;
                    }
                }
            }
        }
        let mut answers = Map::new();
        let head = task
            .map(|t| format!("router:{t}"))
            .unwrap_or_else(|| "router".into());
        for (q, mut logits) in questions.iter().zip(average) {
            for v in &mut logits {
                *v /= files.len() as f64;
            }
            if logits.len() != q.names.len() + usize::from(q.abstain) {
                return Err("router head shape mismatch".into());
            }
            logits.truncate(q.names.len());
            let answer = if q.typ == "set" {
                let p = logits.iter().map(|&v| sigmoid(v)).collect::<Vec<_>>();
                let threshold = self.runtime["router"]["set_threshold"]
                    .as_f64()
                    .unwrap_or(0.5);
                let probabilities: Map<_, _> = q
                    .names
                    .iter()
                    .zip(&p)
                    .map(|(n, &v)| (n.clone(), json!(v)))
                    .collect();
                json!({"type":"set","selected":q.names.iter().zip(&p).filter(|(_,v)|**v>threshold).map(|(n,_)|n).collect::<Vec<_>>(),"probabilities":probabilities,"threshold":threshold,"head":head})
            } else {
                choice_answer(&q.names, &softmax(&logits)?, Some(&head))?
            };
            answers.insert(q.key.clone(), answer);
        }
        Ok((answers, usage))
    }
    fn label_set(&self, names: &[String]) -> Option<String> {
        if let Some(sets) = self.runtime["span"]["router_label_sets"].as_object() {
            for (set, list) in sets {
                if list.as_array().is_some_and(|a| {
                    names
                        .iter()
                        .all(|n| a.iter().any(|x| x.as_str() == Some(n)))
                }) {
                    return Some(set.clone());
                }
            }
        }
        if self.runtime["span"]["evidence"]["labels"]
            .as_array()
            .is_some_and(|a| {
                names.iter().all(|n| {
                    a.iter().any(|x| {
                        x.as_str()
                            .is_some_and(|x| x.to_lowercase() == n.trim().to_lowercase())
                    })
                })
            })
        {
            return Some("evidence".into());
        }
        None
    }
    fn span(&self, state: &Value, q: &Value, pool: &ComputePool) -> Result<(Value, usize), String> {
        let parts = parts(state, true)?;
        let sp = &self.runtime["span"];
        let target = q["target"]
            .as_str()
            .or_else(|| q["over"].as_str())
            .unwrap_or_else(|| {
                if parts.iter().any(|(r, _)| r == "answer") {
                    "answer"
                } else {
                    parts.last().unwrap().0.as_str()
                }
            });
        let target_text = parts
            .iter()
            .find(|(r, _)| r == target)
            .map(|(_, t)| t.as_str())
            .ok_or_else(|| format!("target {target:?} is not a part of the state"))?;
        let criteria = q
            .get("criteria")
            .filter(|v| {
                v.as_object().is_some_and(|m| !m.is_empty())
                    || v.as_array().is_some_and(|a| !a.is_empty())
            })
            .or_else(|| q.get("labels"))
            .ok_or("span questions need criteria or labels")?;
        let mut labels = string_options(criteria)?;
        let names = labels.iter().map(|(n, _)| n.clone()).collect::<Vec<_>>();
        let distinct = names.iter().collect::<std::collections::HashSet<_>>();
        if labels.is_empty() || distinct.len() != labels.len() {
            return Err("span labels must be distinct and at least one".into());
        }
        let label_set = self.label_set(&names);
        let set = label_set.as_deref().unwrap_or("");
        let mut question = q["instructions"].as_str().unwrap_or("").to_owned();
        if let Some(schema) = sp["schemas"].get(set) {
            for (n, d) in &mut labels {
                if d.is_empty() {
                    *d = schema["labels"][n.as_str()].as_str().unwrap_or("").into();
                }
            }
            if question.is_empty() {
                question = schema["question"].as_str().unwrap_or("").into();
            }
        }
        if set == "evidence" && question.is_empty() {
            question = sp["evidence"]["question"].as_str().unwrap_or("").into();
        }
        if question.is_empty() {
            question = sp["default_question"]
                .as_str()
                .unwrap_or("Which spans are named entities of these types?")
                .into();
        }
        let normalized = norm(&question).to_lowercase();
        let evidence = set == "evidence"
            || (!matches!(set, "pii" | "halu" | "toxic") && evidence_question(&normalized));
        let relation = relation_question(&normalized);
        let names_only = labels.iter().all(|(_, d)| d.is_empty());
        let mut file = "span_adapter.safetensors".to_owned();
        let mut head = "span";
        if evidence {
            if let Some(f) = sp["evidence"]["adapter"]
                .as_str()
                .filter(|f| self.root.join(f).is_file())
            {
                file = f.into();
            }
        }
        if label_set.is_none() && !evidence {
            let route = std::env::var("VELA3_ENTITY_ROUTE")
                .ok()
                .unwrap_or_else(|| sp["entity_route"].as_str().unwrap_or("ner").into());
            let task = if relation {
                Some("relation")
            } else if names_only && route == "ner" {
                Some("ner")
            } else {
                None
            };
            if let Some(task) = task {
                if let Some(f) = sp["task_adapters"][task]
                    .as_str()
                    .filter(|f| self.root.join(f).is_file())
                {
                    file = f.into();
                    head = task;
                }
            }
        }
        let set_adapter = &sp["set_adapter"];
        let use_set = set_adapter["label_sets"]
            .as_array()
            .is_some_and(|a| a.iter().any(|v| v.as_str() == Some(set)))
            && set_adapter["file"]
                .as_str()
                .is_some_and(|f| self.root.join(f).is_file());
        if use_set {
            file = set_adapter["file"].as_str().unwrap().into();
            head = "span_set";
        }
        let mut schema = Vec::new();
        let mut entities = Vec::new();
        for (n, d) in labels.iter() {
            entities.push((n.clone(), 1 + schema.len()));
            schema.push(self.marker("[E]")?);
            schema.extend(self.ids(&described(n, d))?);
        }
        let adapter = self.adapter(&file)?;
        let max_len = if evidence {
            sp["evidence"]["max_len"].as_u64().unwrap_or(32768) as usize
        } else {
            sp["window"]["max_len"].as_u64().unwrap_or(8192) as usize
        };
        let raw = self.schema_raw(
            &parts,
            target,
            &schema,
            &[],
            &entities,
            max_len,
            &adapter,
            pool,
        )?;
        let mut temperature = 1.;
        let threshold;
        let why;
        if set == "pii" {
            let buckets = if use_set {
                &set_adapter["pii_buckets"]
            } else {
                &sp["pii_buckets"]
            };
            let bucket = length_bucket(buckets, raw.target_tokens)?;
            threshold = bucket["threshold"]
                .as_f64()
                .ok_or("missing PII threshold")?;
            temperature = if use_set {
                &set_adapter["router_set_temperature"]
            } else {
                &sp["router_set_temperature"]
            }
            .as_f64()
            .ok_or("missing router span temperature")?;
            why = format!(
                "pii label set, {} bucket ({} target tokens)",
                bucket["name"].as_str().unwrap_or(""),
                raw.target_tokens
            );
        } else if matches!(set, "halu" | "toxic") {
            threshold = if use_set {
                set_adapter["thresholds_by_set"][set].as_f64()
            } else {
                None
            }
            .or_else(|| sp["thresholds_by_set"][set].as_f64())
            .ok_or("missing label-set threshold")?;
            temperature = if use_set {
                &set_adapter["router_set_temperature"]
            } else {
                &sp["router_set_temperature"]
            }
            .as_f64()
            .ok_or("missing router span temperature")?;
            why = format!("{set} label set");
        } else if evidence {
            if let Some(b) = sp["evidence"].get("buckets") {
                let bucket = length_bucket(b, raw.target_tokens)?;
                threshold = bucket["threshold"]
                    .as_f64()
                    .ok_or("missing evidence threshold")?;
                why = format!(
                    "evidence question, {} bucket ({} target tokens)",
                    bucket["name"].as_str().unwrap_or(""),
                    raw.target_tokens
                );
            } else {
                threshold = sp["evidence"]["threshold"].as_f64().unwrap_or(0.5);
                why = "evidence question".into();
            }
        } else if head == "relation" {
            let form = if labels.len() == 1 { "pair" } else { "multi" };
            threshold = sp["relation"][format!("threshold_{form}")]
                .as_f64()
                .ok_or("missing relation threshold")?;
            why = format!("relation question ({form})");
        } else if relation {
            threshold = sp["relation_threshold"].as_f64().unwrap_or(0.55);
            why = "relation question".into();
        } else if head == "ner" {
            threshold = sp["ner"]["threshold"].as_f64().unwrap_or(0.6);
            why = "open labels, names only".into();
        } else {
            threshold = sp["threshold"].as_f64().unwrap_or(0.6);
            why = "open labels".into();
        }
        let threshold = match q.get("threshold") {
            Some(v) => text::threshold_value(v)?,
            None => threshold,
        };
        if !temperature.is_finite() || temperature <= 0. || !threshold.is_finite() {
            return Err("invalid span threshold or temperature".into());
        }
        let probabilities = raw
            .span
            .iter()
            .map(|&v| sigmoid(v / temperature) as f32)
            .collect::<Vec<_>>();
        let mut spans = text::decode_fast_spans(
            target_text,
            &raw.offsets,
            &probabilities,
            &names,
            threshold.clamp(-1., 2.) as f32,
        )?;
        if let Some(spans) = spans.as_array_mut() {
            for span in spans {
                let start = span["start"].as_u64().ok_or("invalid decoded span start")? as usize;
                let end = span["end"].as_u64().ok_or("invalid decoded span end")? as usize;
                let start_byte = char_to_byte(target_text, start)?;
                let end_byte = char_to_byte(target_text, end)?;
                let label = names
                    .iter()
                    .position(|n| span["label"] == n.as_str())
                    .ok_or("invalid decoded span label")?;
                let mut total = 0.;
                let mut count = 0;
                for (i, &(a, b)) in raw.offsets.iter().enumerate() {
                    if a < end_byte && b > start_byte {
                        total += probabilities[i * names.len() + label] as f64;
                        count += 1;
                    }
                }
                span["text"] = json!(&target_text[start_byte..end_byte]);
                span["score"] = json!(if count == 0 { 0. } else { total / count as f64 });
            }
        }
        Ok((
            json!({"type":"span","target":target,"spans":spans,"head":head,"head_reason":format!("{head} adapter; {why}"),"threshold":threshold}),
            raw.input_tokens,
        ))
    }
    #[allow(clippy::too_many_arguments)]
    fn schema_raw(
        &self,
        parts: &[(String, String)],
        target: &str,
        schema: &[u32],
        questions: &[Question],
        entities: &[(String, usize)],
        max_len: usize,
        adapter: &Adapter,
        pool: &ComputePool,
    ) -> Result<Raw, String> {
        let mut segments = Vec::new();
        let mut word_offsets = Vec::new();
        let mut first_tokens = Vec::new();
        let mut target_tokens = None;
        for (role, part) in parts {
            let encoded = self
                .tok
                .encode(part.as_str(), false)
                .map_err(|e| e.to_string())?;
            if role == target {
                target_tokens = Some(encoded.len());
                let offsets = encoded.get_offsets();
                let mut ends = Vec::with_capacity(offsets.len());
                let mut largest = 0;
                for &(a, b) in offsets {
                    if b > a {
                        largest = largest.max(b);
                    }
                    ends.push(largest);
                }
                for (a, b) in text::split_words(part) {
                    let mut t = ends.partition_point(|&e| e <= a);
                    while t < offsets.len() && offsets[t].1 <= offsets[t].0 {
                        t += 1;
                    }
                    if t < offsets.len() && offsets[t].0 < b {
                        word_offsets.push((a, b));
                        first_tokens.push(t);
                    }
                }
            }
            segments.push(Segment {
                role: role.clone(),
                ids: encoded.get_ids().to_vec(),
            });
        }
        let target_tokens = target_tokens
            .ok_or_else(|| format!("question target {target:?} is not in the state"))?;
        let protected = vec![target.to_owned()];
        let markers = &self.config["decision"]["marker_ids"];
        let first = assemble(
            schema,
            &segments,
            &protected,
            max_len,
            self.boundary("bos_token_id")?,
            self.boundary("eos_token_id")?,
            markers,
        )?;
        let retained = first.ranges[target].2;
        let windowed = first.truncated && retained < target_tokens;
        let ranges = if windowed {
            window_ranges(
                target_tokens,
                retained,
                self.runtime["span"]["window"]["overlap"]
                    .as_u64()
                    .unwrap_or(512) as usize,
            )
        } else {
            vec![(0, target_tokens)]
        };
        if ranges.is_empty() {
            return Err("span schema leaves no target token capacity".into());
        }
        let n_labels = entities.len();
        let mut span = vec![
            0.;
            word_offsets
                .len()
                .checked_mul(n_labels)
                .ok_or("span score shape overflow")?
        ];
        let mut counts = vec![0usize; word_offsets.len()];
        let mut all_logits = questions
            .iter()
            .map(|q| {
                vec![
                    if q.typ == "set" {
                        f64::NEG_INFINITY
                    } else {
                        0.
                    };
                    q.positions.len()
                ]
            })
            .collect::<Vec<_>>();
        let mut input_tokens = 0;
        let source = OverlaySource::new(&self.source, &adapter.source);
        let hidden = self.hidden()?;
        for (a, b) in ranges.iter().copied() {
            let record = if windowed {
                let mut seg = segments.clone();
                let target_segment = seg.iter_mut().find(|s| s.role == target).unwrap();
                target_segment.ids = target_segment.ids[a..b].to_vec();
                assemble(
                    schema,
                    &seg,
                    &protected,
                    max_len,
                    self.boundary("bos_token_id")?,
                    self.boundary("eos_token_id")?,
                    markers,
                )?
            } else {
                first.clone()
            };
            input_tokens += record.ids.len();
            let h = ultrabert::encode_adapted(
                &self.source,
                &adapter.source,
                adapter.scale,
                &self.config,
                "encoder.",
                &record.ids,
                None,
                pool,
            )?;
            let (start, end, kept) = record.ranges[target];
            for (q, acc) in questions.iter().zip(&mut all_logits) {
                let logits = head_logits(
                    &source,
                    &h,
                    hidden,
                    q.qpos,
                    &q.positions,
                    (start, end),
                    pool,
                )?;
                for (value, lg) in acc.iter_mut().zip(logits) {
                    if q.typ == "set" {
                        *value = value.max(lg);
                    } else {
                        *value += lg;
                    }
                }
            }
            if n_labels > 0 {
                let selected = first_tokens
                    .iter()
                    .enumerate()
                    .filter(|(_, t)| **t >= a && **t < a + kept)
                    .map(|(i, &t)| (i, start + 1 + t - a))
                    .collect::<Vec<_>>();
                if !selected.is_empty() {
                    let words = gather(
                        &h,
                        hidden,
                        &selected.iter().map(|(_, p)| *p).collect::<Vec<_>>(),
                    )?;
                    let labels = gather(
                        &h,
                        hidden,
                        &entities.iter().map(|(_, p)| *p).collect::<Vec<_>>(),
                    )?;
                    let t =
                        normalized_projection(&source, "w_t.weight", &words, selected.len(), pool)?;
                    let e = normalized_projection(&source, "w_e.weight", &labels, n_labels, pool)?;
                    let width = t.len() / selected.len();
                    if e.len() != n_labels * width {
                        return Err("span projection width mismatch".into());
                    }
                    let tau = temperature(&source, "log_tau_span")?;
                    for (i, (global, _)) in selected.iter().enumerate() {
                        for j in 0..n_labels {
                            span[global * n_labels + j] += (crate::ops::dot_f32(
                                &t[i * width..(i + 1) * width],
                                &e[j * width..(j + 1) * width],
                                width,
                            ) / tau)
                                as f64;
                        }
                        counts[*global] += 1;
                    }
                }
            }
        }
        for (q, logits) in questions.iter().zip(&mut all_logits) {
            if q.typ != "set" {
                for lg in logits {
                    *lg = (*lg / ranges.len() as f64) as f32 as f64;
                }
            }
        }
        for (i, &count) in counts.iter().enumerate() {
            for j in 0..n_labels {
                span[i * n_labels + j] = if count > 0 {
                    span[i * n_labels + j] / count as f64
                } else {
                    -30.
                };
            }
        }
        Ok(Raw {
            logits: all_logits,
            span,
            offsets: word_offsets,
            target_tokens,
            input_tokens,
        })
    }
}

fn length_bucket(buckets: &Value, tokens: usize) -> Result<&Value, String> {
    buckets
        .as_array()
        .and_then(|b| {
            b.iter()
                .find(|b| b["max_tokens"].as_u64().is_some_and(|n| tokens as u64 <= n))
        })
        .ok_or("no span length bucket covers the target".into())
}
fn char_to_byte(text: &str, index: usize) -> Result<usize, String> {
    if index == text.chars().count() {
        Ok(text.len())
    } else {
        text.char_indices()
            .nth(index)
            .map(|(p, _)| p)
            .ok_or("decoded span offset exceeds text".into())
    }
}
fn relation_question(question: &str) -> bool {
    static PATTERN: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    PATTERN.get_or_init(||regex::Regex::new(r"\bspans? (?:are|is) the .+ (?:of|for)\b|\brelat(?:ed|ion|ions|es|ing)?\b|\b(?:head|tail)\b|\bstands? in\b|\bwith respect to\b|\bcounterpart\b|\bsuch that\b|\brole of\b").unwrap()).is_match(question)
}
fn evidence_question(question: &str) -> bool {
    static PATTERN: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    PATTERN
        .get_or_init(|| {
            regex::Regex::new(r"\b(?:evidence|support(?:s|ing|ed)?|cite[sd]?|citations?)\b")
                .unwrap()
        })
        .is_match(question)
}
fn gather(hidden: &[f32], width: usize, positions: &[usize]) -> Result<Vec<f32>, String> {
    let mut out = Vec::with_capacity(
        positions
            .len()
            .checked_mul(width)
            .ok_or("readout shape overflow")?,
    );
    for &pos in positions {
        let start = pos.checked_mul(width).ok_or("readout position overflow")?;
        out.extend_from_slice(
            hidden
                .get(start..start + width)
                .ok_or("readout position exceeds hidden states")?,
        );
    }
    Ok(out)
}
fn head_norm(source: &dyn TensorSource, x: &[f32]) -> Result<Vec<f32>, String> {
    layer_norm(
        x,
        &values(source, "norm.weight")?,
        &values(source, "norm.bias")?,
        1e-5,
    )
}
fn normalized_projection(
    source: &dyn TensorSource,
    name: &str,
    input: &[f32],
    rows: usize,
    pool: &ComputePool,
) -> Result<Vec<f32>, String> {
    if rows == 0 {
        return Ok(Vec::new());
    }
    let mut output = linear(source, name, &head_norm(source, input)?, rows, pool)?;
    if !output.len().is_multiple_of(rows) {
        return Err("head projection shape mismatch".into());
    }
    let width = output.len() / rows;
    if width == 0 {
        return Err("head projection width must be nonzero".into());
    }
    for row in output.chunks_exact_mut(width) {
        let norm = row.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
        for v in row {
            *v /= norm;
        }
    }
    Ok(output)
}
fn temperature(source: &dyn TensorSource, name: &str) -> Result<f32, String> {
    let t = values(source, name)?;
    if t.len() != 1 {
        return Err(format!("{name} must be scalar"));
    }
    let tau = t[0].exp();
    if !tau.is_finite() || tau <= 0. {
        return Err(format!("invalid {name}"));
    }
    Ok(tau)
}
#[allow(clippy::too_many_arguments)]
fn head_logits(
    source: &dyn TensorSource,
    h: &[f32],
    hidden: usize,
    qpos: usize,
    positions: &[usize],
    range: (usize, usize),
    pool: &ComputePool,
) -> Result<Vec<f64>, String> {
    let (ps, pe) = range;
    if ps >= pe || pe.checked_mul(hidden).is_none_or(|n| n > h.len()) {
        return Err("invalid decision pooling range".into());
    }
    let mut pooled = vec![0.; hidden];
    let mut prefix = vec![0.; hidden];
    for (index, row) in h.chunks_exact(hidden).take(pe).enumerate() {
        if index == ps {
            prefix.copy_from_slice(&pooled);
        }
        for (x, v) in pooled.iter_mut().zip(row) {
            *x += v;
        }
    }
    for (x, p) in pooled.iter_mut().zip(prefix) {
        *x = (*x - p) / (pe - ps) as f32;
    }
    let v = normalized_projection(source, "w_p.weight", &pooled, 1, pool)?;
    let q = linear(
        source,
        "w_q.weight",
        &head_norm(source, &gather(h, hidden, &[qpos])?)?,
        1,
        pool,
    )?;
    let ho = gather(h, hidden, positions)?;
    let mut u = linear(
        source,
        "w_o.weight",
        &head_norm(source, &ho)?,
        positions.len(),
        pool,
    )?;
    if q.len() != v.len() || u.len() != positions.len() * v.len() {
        return Err("decision projection width mismatch".into());
    }
    for row in u.chunks_exact_mut(v.len()) {
        for (x, q) in row.iter_mut().zip(&q) {
            *x += q;
        }
        let norm = row.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
        for x in row {
            *x /= norm;
        }
    }
    let mut mlp = linear(source, "cls_mlp.0.weight", &ho, positions.len(), pool)?;
    for v in &mut mlp {
        *v = v.max(0.);
    }
    let mlp = linear(source, "cls_mlp.3.weight", &mlp, positions.len(), pool)?;
    if mlp.len() != positions.len() {
        return Err("decision MLP output width mismatch".into());
    }
    let tau = temperature(source, "log_tau")?;
    Ok(u.chunks_exact(v.len())
        .zip(mlp)
        .map(|(u, m)| (crate::ops::dot_f32(u, &v, v.len()) / tau + m) as f64)
        .collect())
}
fn sigmoid(value: f64) -> f64 {
    if value >= 0. {
        1. / (1. + (-value).exp())
    } else {
        let e = value.exp();
        e / (1. + e)
    }
}
fn softmax(logits: &[f64]) -> Result<Vec<f64>, String> {
    if logits.is_empty() || logits.iter().any(|v| !v.is_finite()) {
        return Err("decision logits must be finite and nonempty".into());
    }
    let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut p = logits.iter().map(|v| (v - max).exp()).collect::<Vec<_>>();
    let sum = p.iter().sum::<f64>();
    for v in &mut p {
        *v /= sum;
    }
    Ok(p)
}
fn choice_answer(
    names: &[String],
    probabilities: &[f64],
    head: Option<&str>,
) -> Result<Value, String> {
    if names.is_empty() || names.len() != probabilities.len() {
        return Err("choice names/probabilities shape mismatch".into());
    }
    let mut best = 0;
    for i in 1..names.len() {
        if probabilities[i] > probabilities[best] {
            best = i;
        }
    }
    let p: Map<_, _> = names
        .iter()
        .zip(probabilities)
        .map(|(n, &p)| (n.clone(), json!(p)))
        .collect();
    let mut answer = json!({"type":"choice","choice":names[best],"probabilities":p});
    if let Some(head) = head {
        answer["head"] = json!(head);
    }
    Ok(answer)
}
fn assemble(
    schema: &[u32],
    segments: &[Segment],
    protected: &[String],
    max_len: usize,
    bos: u32,
    eos: u32,
    markers: &Value,
) -> Result<Record, String> {
    let mut segments = segments.to_vec();
    let fixed = 3usize
        .checked_add(segments.len())
        .and_then(|v| v.checked_add(schema.len()))
        .ok_or("Vela3 schema size overflow")?;
    let excess = |segments: &[Segment]| -> Result<usize, String> {
        let total = segments.iter().try_fold(fixed, |n, s| {
            n.checked_add(s.ids.len())
                .ok_or("Vela3 sequence size overflow")
        })?;
        Ok(total.saturating_sub(max_len))
    };
    if let Some(index) = segments.iter().position(|s| !protected.contains(&s.role)) {
        for floor in [64, 1] {
            let over = excess(&segments)?;
            if over > 0 {
                let n = segments[index].ids.len();
                let keep = n.saturating_sub(over).max(floor.min(n));
                segments[index].ids.truncate(keep);
            }
        }
    }
    let mut truncated = false;
    for index in 0..segments.len() {
        if protected.contains(&segments[index].role) {
            let over = excess(&segments)?;
            if over > 0 {
                let n = segments[index].ids.len();
                let keep = n.saturating_sub(over).max(1.min(n));
                segments[index].ids.truncate(keep);
                truncated |= n > keep;
            }
        }
    }
    if excess(&segments)? > 0 {
        return Err("schema alone does not fit max_length".into());
    }
    let marker = |name: &str| -> Result<u32, String> {
        markers[name]
            .as_u64()
            .and_then(|v| u32::try_from(v).ok())
            .ok_or_else(|| format!("missing Vela3 marker {name}"))
    };
    let mut ids = vec![bos];
    ids.extend_from_slice(schema);
    ids.push(marker("[SEP_SCHEMA]")?);
    let mut ranges = HashMap::new();
    for segment in segments {
        let start = ids.len();
        ids.push(marker(&format!("[SEG_{}]", segment.role))?);
        ids.extend_from_slice(&segment.ids);
        if ranges
            .insert(segment.role, (start, ids.len(), segment.ids.len()))
            .is_some()
        {
            return Err("one text part per role is required".into());
        }
    }
    ids.push(eos);
    Ok(Record {
        ids,
        ranges,
        truncated,
    })
}
fn window_ranges(n: usize, length: usize, overlap: usize) -> Vec<(usize, usize)> {
    if length == 0 {
        return Vec::new();
    }
    let overlap = overlap.min(length / 4);
    let stride = (length - overlap).max(1);
    let mut ranges = Vec::new();
    for start in (0..n.saturating_sub(overlap).max(1)).step_by(stride) {
        let end = n.min(start.saturating_add(length));
        ranges.push((start, end));
        if end >= n {
            break;
        }
    }
    ranges
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn assembly_shrinks_unprotected_context_before_windowing_target() {
        let segments = vec![
            Segment {
                role: "user".into(),
                ids: vec![1; 100],
            },
            Segment {
                role: "answer".into(),
                ids: vec![2; 20],
            },
        ];
        let mk = serde_json::json!({"[SEP_SCHEMA]":8,"[SEG_user]":9,"[SEG_answer]":10});
        let r = assemble(&[4, 5], &segments, &["answer".into()], 40, 0, 0, &mk).unwrap();
        assert_eq!(r.ids.len(), 40);
        assert_eq!(r.ranges["answer"].2, 20);
        assert_eq!(r.ranges["user"].2, 13);
        assert!(!r.truncated);
    }
    #[test]
    fn target_windows_cover_tail_and_limit_overlap_to_quarter() {
        assert_eq!(window_ranges(24, 10, 512), vec![(0, 10), (8, 18), (16, 24)]);
        assert_eq!(window_ranges(3, 10, 512), vec![(0, 3)]);
        assert_eq!(window_ranges(3, 1, 512), vec![(0, 1), (1, 2), (2, 3)]);
    }
}

#[cfg(test)]
mod oracle_tests {
    use super::*;
    fn close_json(actual: &Value, expected: &Value, path: &str) {
        match (actual, expected) {
            (Value::Number(a), Value::Number(b)) => {
                let (a, b) = (a.as_f64().unwrap(), b.as_f64().unwrap());
                assert!((a - b).abs() <= 5e-4 + 5e-4 * b.abs(), "{path}: {a} vs {b}");
            }
            (Value::Array(a), Value::Array(b)) => {
                assert_eq!(a.len(), b.len(), "{path}");
                for (i, (a, b)) in a.iter().zip(b).enumerate() {
                    close_json(a, b, &format!("{path}[{i}]"));
                }
            }
            (Value::Object(a), Value::Object(b)) => {
                assert_eq!(a.len(), b.len(), "{path}: {a:?} vs {b:?}");
                for (k, b) in b {
                    close_json(a.get(k).unwrap_or(&Value::Null), b, &format!("{path}.{k}"));
                }
            }
            _ => assert_eq!(actual, expected, "{path}"),
        }
    }
    #[test]
    #[ignore = "requires RMI_VELA3_FAST_ROOT and official fast_oracle.py fixture RMI_VELA3_FAST_FIXTURE"]
    fn native_fast_matches_official_hidden_logits_and_system_one() {
        let root = std::path::PathBuf::from(std::env::var("RMI_VELA3_FAST_ROOT").unwrap());
        let fixture =
            read_json(Path::new(&std::env::var("RMI_VELA3_FAST_FIXTURE").unwrap())).unwrap();
        let model = FastModel::load(&root).unwrap();
        let pool = ComputePool::new(2);
        for row in fixture["records"].as_array().unwrap() {
            let ids = row["record"]["ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u32)
                .collect::<Vec<_>>();
            let hidden =
                ultrabert::encode(&model.source, &model.config, "encoder.", &ids, None, &pool)
                    .unwrap();
            let expected = row["hidden"]
                .as_array()
                .unwrap()
                .iter()
                .flat_map(|r| {
                    r.as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v.as_f64().unwrap() as f32)
                })
                .collect::<Vec<_>>();
            assert_eq!(hidden.len(), expected.len());
            let max_error = hidden
                .iter()
                .zip(&expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(
                max_error < 1e-3,
                "{} hidden max error {max_error}",
                row["key"]
            );
            let rec = &row["record"];
            let pos = rec["opt_pos"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect::<Vec<_>>();
            let range = rec["pool"].as_array().unwrap();
            let logits = head_logits(
                &model.source,
                &hidden,
                model.hidden().unwrap(),
                rec["q_pos"].as_u64().unwrap() as usize,
                &pos,
                (
                    range[0].as_u64().unwrap() as usize,
                    range[1].as_u64().unwrap() as usize,
                ),
                &pool,
            )
            .unwrap();
            let max_logit_error = logits
                .iter()
                .zip(row["logits"].as_array().unwrap())
                .map(|(a, b)| (a - b.as_f64().unwrap()).abs())
                .fold(0f64, f64::max);
            eprintln!(
                "{}: hidden_max_abs={max_error:.9e}, logits_max_abs={max_logit_error:.9e}",
                row["key"]
            );
            close_json(
                &json!(logits),
                &row["logits"],
                &format!("{}.logits", row["key"]),
            );
        }
        let result = predict(&root, &fixture["request"], &pool).unwrap();
        close_json(
            &result,
            fixture.get("system_one").unwrap_or(&fixture["decision"]),
            "system_one",
        );
    }
}
