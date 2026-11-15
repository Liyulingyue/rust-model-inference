use super::{
    forward::{linear, trace, Session},
    input::{Encoded, Visual},
    protocol, weights, Engine,
};
use crate::models::qwen35::trunk::HybridTrunk;
use serde_json::{json, Map, Value};
use std::{collections::HashMap, time::Instant};

#[derive(Default)]
struct Usage {
    input: usize,
    output: usize,
    sequences: usize,
}

fn orders(n: usize, k: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    for order in std::iter::once((0..n).collect())
        .chain(std::iter::once((0..n).rev().collect()))
        .chain((1..n).map(|s| (0..n).map(|i| (i + s) % n).collect()))
    {
        if !out.contains(&order) {
            out.push(order);
        }
        if out.len() == k {
            break;
        }
    }
    out
}

fn same_criteria(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Object(a), Value::Object(b)) => {
            a.len() == b.len()
                && a.iter()
                    .all(|(k, v)| b.get(k).is_some_and(|w| same_criteria(v, w)))
        }
        (Value::Array(a), Value::Array(b)) => {
            a.len() == b.len() && a.iter().zip(b).all(|(a, b)| same_criteria(a, b))
        }
        _ => a == b,
    }
}

fn facet_joint(qs: &Map<String, Value>) -> bool {
    let first = qs.values().next().unwrap();
    qs.len() > 1
        && qs.values().all(|q| {
            q["instructions"].is_string()
                && q["type"] == first["type"]
                && same_criteria(&q["criteria"], &first["criteria"])
        })
}

fn single(base: &Value, key: &str, q: &Value) -> Value {
    let mut req = base.clone();
    req["questions"] = json!({key:q});
    req
}

fn plan(
    engine: &Engine,
    request: &Value,
    visual: &Visual,
) -> Result<Vec<(Value, Encoded)>, String> {
    let enc = engine.encoder.encode(request, &visual.grids, false)?;
    if enc.ids.len() <= engine.max_context {
        return Ok(vec![(request.clone(), enc)]);
    }
    let qs = request["questions"].as_object().unwrap();
    if qs.len() == 1 {
        return Err(format!(
            "THX-02 question needs {} tokens, over context {}; nothing was truncated",
            enc.ids.len(),
            engine.max_context
        ));
    }
    let items = qs.iter().collect::<Vec<_>>();
    let mut out = Vec::new();
    let (left, right) = items.split_at(items.len() / 2);
    for part in [left, right] {
        let mut req = request.clone();
        req["questions"] = Value::Object(
            part.iter()
                .map(|(k, v)| ((*k).clone(), (*v).clone()))
                .collect(),
        );
        out.extend(plan(engine, &req, visual)?);
    }
    Ok(out)
}

fn read(
    engine: &Engine,
    model: &HybridTrunk<'_>,
    req: &Value,
    visual: &Visual,
    usage: &mut Usage,
) -> Result<HashMap<String, Vec<f64>>, String> {
    let chunks = plan(engine, req, visual)?;
    let mut out = HashMap::new();
    for (request, encoded) in chunks {
        let mut x = model.embed_tokens(&encoded.ids)?;
        visual.inject(&encoded.ids, engine.encoder.image, &mut x)?;
        let hidden = Session::new(&model.config, encoded.ids.len(), false).run(
            model,
            x,
            &encoded.positions,
            &engine.pool,
        )?;
        let width = model.config.n_embd;
        let readouts = encoded
            .markers
            .iter()
            .flat_map(|&m| hidden[m * width..(m + 1) * width].iter().copied())
            .collect::<Vec<_>>();
        let logits = linear(
            model
                .cls_score
                .as_ref()
                .ok_or("THX-02 missing decision head")?,
            &readouts,
            &engine.pool,
        )?;
        trace("thx02.logits", None, &[encoded.markers.len(), 255], &logits);
        let temperature = req["temperature"].as_f64().unwrap_or(engine.temperature);
        for ((key, q), row) in request["questions"]
            .as_object()
            .unwrap()
            .iter()
            .zip(logits.chunks_exact(255))
        {
            let count = protocol::options(q)?.len();
            let mut p = row[..count]
                .iter()
                .map(|&v| v as f64 / temperature)
                .collect::<Vec<_>>();
            let max = p.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let sum = p
                .iter_mut()
                .map(|v| {
                    *v = (*v - max).exp();
                    *v
                })
                .sum::<f64>();
            if !sum.is_finite() || sum <= 0. {
                return Err("THX-02 produced non-finite decision logits".into());
            }
            for v in &mut p {
                *v /= sum;
            }
            out.insert(key.clone(), p);
        }
        usage.input += encoded.ids.len();
        usage.sequences += 1;
    }
    Ok(out)
}

fn gate_request(state: &Value, q: &Value) -> Result<Value, String> {
    let desc = protocol::options(q)?
        .into_iter()
        .take(12)
        .map(|(_, v)| {
            protocol::python_str(&v)
                .chars()
                .take(200)
                .collect::<String>()
        })
        .collect::<Vec<_>>()
        .join("\n");
    Ok(
        json!({"state":format!("Question: {}\nOptions:\n{desc}\n\nContext excerpt:\n{}",protocol::describe(&protocol::instructions(q)),protocol::describe(state).chars().take(1200).collect::<String>()),"questions":{"g":{"type":"noul","instructions":protocol::GATE_QUESTION}}}),
    )
}

fn think(
    engine: &Engine,
    request: &Value,
    images: &[image::DynamicImage],
    cap: usize,
    usage: &mut Usage,
) -> Result<String, String> {
    let source = engine.thinker.as_ref().ok_or("THX-02 missing thinker")?;
    let enc = engine.thinker_encoder.as_ref().unwrap();
    let mut cfg = weights::config(source.as_ref(), true)?;
    cfg.n_ctx = cfg.n_ctx.min(engine.max_context);
    let model = weights::load(source.as_ref(), cfg)?;
    let visual = Visual::encode(source.as_ref(), images, &engine.pool)?;
    let encoded = enc.encode(request, &visual.grids, true)?;
    if encoded.ids.len() >= engine.max_context {
        return Err("THX-02 thinker prompt exceeds context".into());
    }
    let mut x = model.embed_tokens(&encoded.ids)?;
    visual.inject(&encoded.ids, enc.image, &mut x)?;
    let mut session = Session::new(&model.config, encoded.ids.len(), true);
    let mut hidden = session.run(&model, x, &encoded.positions, &engine.pool)?;
    let mut generated = Vec::new();
    usage.input += encoded.ids.len();
    usage.sequences += 1;
    let limit = cap.min(engine.max_context - encoded.ids.len());
    for step in 0..limit {
        let last = &hidden[hidden.len() - model.config.n_embd..];
        let logits = linear(&model.output_weight, last, &engine.pool)?;
        let best = (1..logits.len())
            .fold(0, |best, i| if logits[i] > logits[best] { i } else { best })
            as u32;
        if logits.iter().any(|v| !v.is_finite()) {
            return Err("THX-02 thinker produced non-finite logits".into());
        }
        usage.output += 1;
        if ["</think>", "<|im_end|>", "<|endoftext|>"]
            .iter()
            .any(|s| enc.tokenizer.token_to_id(s) == Some(best))
        {
            break;
        }
        generated.push(best);
        if step + 1 == limit {
            break;
        }
        let pos = encoded.next + step;
        hidden = session.run(
            &model,
            model.embed_tokens(&[best])?,
            &[[pos, pos, pos, 0]],
            &engine.pool,
        )?;
    }
    enc.tokenizer
        .decode(&generated, false)
        .map(|s| s.replace("<think>", "").trim().to_owned())
        .map_err(|e| e.to_string())
}

pub(super) fn decide(engine: &Engine, request: &Value) -> Result<Value, String> {
    let start = Instant::now();
    let images = super::input::open_images(request)?;
    let visual = Visual::encode(engine.source.as_ref(), &images, &engine.pool)?;
    let mut cfg = weights::config(engine.source.as_ref(), false)?;
    cfg.n_ctx = engine.max_context;
    let model = weights::load(engine.source.as_ref(), cfg)?;
    let mut usage = Usage::default();
    let mut answers = Map::new();
    let qs = request["questions"].as_object().unwrap();
    if images.is_empty() && facet_joint(qs) {
        let ps = read(engine, &model, request, &visual, &mut usage)?;
        for (key, q) in qs {
            answers.insert(key.clone(), protocol::answer(q, &ps[key])?);
        }
    } else {
        for (key, q) in qs {
            let opts = protocol::options(q)?;
            let variants = if q["type"] == "choice" {
                orders(opts.len(), request["tta"].as_u64().unwrap_or(1) as usize)
            } else {
                vec![(0..opts.len()).collect()]
            };
            let mut logps = vec![0.; opts.len()];
            let mut direct = None;
            for order in &variants {
                let mut question = q.clone();
                if q["type"] == "choice" {
                    let old = q["criteria"].as_object().unwrap();
                    question["criteria"] = Value::Object(
                        order
                            .iter()
                            .map(|&i| (opts[i].0.clone(), old[&opts[i].0].clone()))
                            .collect(),
                    );
                }
                let ps = read(
                    engine,
                    &model,
                    &single(request, key, &question),
                    &visual,
                    &mut usage,
                )?;
                if variants.len() == 1 {
                    direct = Some(ps[key].clone());
                    break;
                }
                for (new, &original) in order.iter().enumerate() {
                    logps[original] += ps[key][new].max(1e-9).ln() / variants.len() as f64;
                }
            }
            let mut p = direct.unwrap_or_else(|| logps.into_iter().map(f64::exp).collect());
            let sum = p.iter().sum::<f64>();
            for v in &mut p {
                *v /= sum;
            }
            let always = request["think"] == true;
            let auto = request["think"] == "auto";
            let mut thought = None;
            if (always || auto)
                && p.iter().copied().fold(0., f64::max) < if always { 0.97 } else { 0.6 }
            {
                let gate = if auto {
                    read(
                        engine,
                        &model,
                        &gate_request(&request["state"], q)?,
                        &Visual::default(),
                        &mut usage,
                    )?["g"][1]
                } else {
                    1.
                };
                if gate >= if auto { 0.2 } else { 0. } {
                    let th = think(
                        engine,
                        &single(request, key, q),
                        &images,
                        if always { 320 } else { 96 },
                        &mut usage,
                    )?;
                    if !th.is_empty() {
                        let mut refined = q.clone();
                        refined["instructions"] = json!(format!(
                            "{}{}{}",
                            protocol::describe(&protocol::instructions(q)),
                            protocol::THOUGHT_NOTE,
                            th
                        ));
                        let p1 = read(
                            engine,
                            &model,
                            &single(request, key, &refined),
                            &visual,
                            &mut usage,
                        )?;
                        let alpha = if always { 0.5 } else { 0.7 };
                        for (p, y) in p.iter_mut().zip(&p1[key]) {
                            *p = alpha * *p + (1. - alpha) * y;
                        }
                        thought = Some(th);
                    }
                }
            }
            let mut a = protocol::answer(q, &p)?;
            if request["return_thoughts"] == true {
                if let Some(th) = thought {
                    a["thought"] = json!(th);
                }
            }
            answers.insert(key.clone(), a);
        }
    }
    Ok(
        json!({"model":"THX-02","answers":answers,"usage":{"input_tokens":usage.input,"output_tokens":usage.output,"sequences":usage.sequences},"latency_ms":(start.elapsed().as_secs_f64()*10000.).round()/10.}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn option_orders_are_distinct_and_bounded() {
        assert_eq!(orders(2, 8), vec![vec![0, 1], vec![1, 0]]);
        assert_eq!(
            orders(3, 3),
            vec![vec![0, 1, 2], vec![2, 1, 0], vec![1, 2, 0]]
        );
    }
    #[test]
    fn joint_facets_ignore_object_key_order_but_require_text_instructions() {
        let qs = json!({"a":{"type":"choice","instructions":"a","criteria":{"x":1,"y":2}},"b":{"type":"choice","instructions":"b","criteria":{"y":2,"x":1}}});
        assert!(facet_joint(qs.as_object().unwrap()));
        let mut q = qs;
        q["b"]["instructions"] = json!({"text":"b"});
        assert!(!facet_joint(q.as_object().unwrap()));
    }

    #[test]
    fn gate_uses_python_str_for_structured_descriptions() {
        let q = json!({"type":"score","instructions":"pick","criteria":[{"x":true},{"x":false}]});
        assert_eq!(
            gate_request(&json!("state"), &q).unwrap()["state"],
            "Question: pick\nOptions:\n{'x': True}\n{'x': False}\n\nContext excerpt:\nstate"
        );
    }
}
