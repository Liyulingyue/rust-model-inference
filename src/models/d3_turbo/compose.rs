//! The checkpoint's System One renderer and spatial readout, kept out of app I/O.
use super::{decoder, python_json};
use serde_json::{json, Map, Value};

const BINS: usize = 128;
const EDGES: [&str; 4] = ["top", "bottom", "left", "right"];

fn describe(v: &Value) -> String {
    v.as_str()
        .map(str::to_owned)
        .unwrap_or_else(|| python_json(v, false, false))
}

fn truth(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
        Value::Number(n) => n.as_f64() != Some(0.),
    }
}

pub(super) fn references(assets: &Map<String, Value>) -> Result<Vec<String>, String> {
    let mut refs: Vec<_> = assets.keys().cloned().collect();
    if refs.iter().any(|r| {
        r.trim().is_empty() || r.len() > 1024 || r.contains(['\u{e000}', '\u{e001}', '\u{e002}'])
    }) || refs.iter().map(String::len).sum::<usize>() > 65536
    {
        return Err("asset references must be nonblank, at most 1024 bytes each and 65536 in total, without reserved characters".into());
    }
    refs.sort_by(|a, b| b.chars().count().cmp(&a.chars().count()).then(a.cmp(b)));
    Ok(refs)
}

fn split(text: &str, refs: &[String]) -> Vec<(bool, String)> {
    let mut parts = Vec::new();
    let (mut at, mut start) = (0, 0);
    while at < text.len() {
        if let Some(r) = refs.iter().find(|r| text[at..].starts_with(r.as_str())) {
            if at > start {
                parts.push((false, text[start..at].into()));
            }
            parts.push((true, r.clone()));
            at += r.len();
            start = at;
        } else {
            at += text[at..].chars().next().unwrap().len_utf8();
        }
    }
    if start < text.len() {
        parts.push((false, text[start..].into()));
    }
    parts
}

pub(super) fn hoist(
    value: &Value,
    refs: &[String],
    front: &mut Vec<Value>,
    assets: &Map<String, Value>,
) -> Value {
    match value {
        Value::String(s) => Value::String(
            split(s, refs)
                .into_iter()
                .map(|(image, item)| {
                    if image {
                        front.push(assets[&item].clone());
                        format!("[image {}]", front.len())
                    } else {
                        item
                    }
                })
                .collect(),
        ),
        Value::Array(a) => Value::Array(a.iter().map(|v| hoist(v, refs, front, assets)).collect()),
        Value::Object(o) => Value::Object(
            o.iter()
                .map(|(k, v)| (k.clone(), hoist(v, refs, front, assets)))
                .collect(),
        ),
        _ => value.clone(),
    }
}

fn push(parts: &mut Vec<Value>, text: String) {
    if text.is_empty() {
        return;
    }
    if let Some(last) = parts.last_mut().filter(|v| v["type"] == "text") {
        let mut combined = last["text"].as_str().unwrap().to_owned();
        combined.push_str(&text);
        last["text"] = json!(combined);
    } else {
        parts.push(json!({"type":"text","text":text}));
    }
}

fn mark(value: &Value, refs: &[String], order: &mut Vec<String>) -> Value {
    match value {
        Value::String(s) => {
            let parts = split(s, refs);
            if !parts.iter().any(|(image, _)| *image) {
                return value.clone();
            }
            let whole = parts.len() == 1;
            Value::String(
                parts
                    .into_iter()
                    .map(|(image, item)| {
                        if !image {
                            return item;
                        }
                        let index = order.len();
                        order.push(item);
                        if whole {
                            format!("\u{e002}{index}\u{e002}")
                        } else {
                            format!("\u{e000}{index}\u{e001}")
                        }
                    })
                    .collect(),
            )
        }
        Value::Array(a) => Value::Array(a.iter().map(|v| mark(v, refs, order)).collect()),
        Value::Object(o) => Value::Object(
            o.iter()
                .map(|(k, v)| (k.clone(), mark(v, refs, order)))
                .collect(),
        ),
        _ => value.clone(),
    }
}

fn describe_parts(
    value: &Value,
    refs: &[String],
    assets: &Map<String, Value>,
    parts: &mut Vec<Value>,
    images: &mut Vec<Value>,
) -> Result<(), String> {
    let mut order = Vec::new();
    let text = describe(&mark(value, refs, &mut order));
    let mut at = 0;
    while let Some(offset) = text[at..].find(['\u{e000}', '\u{e002}']) {
        let start = at + offset;
        let marker = text[start..].chars().next().unwrap();
        let end_marker = if marker == '\u{e000}' {
            '\u{e001}'
        } else {
            marker
        };
        let digits = start + marker.len_utf8();
        let end = digits
            + text[digits..]
                .find(end_marker)
                .ok_or("invalid asset marker")?;
        let index: usize = text[digits..end]
            .parse()
            .map_err(|_| "invalid asset marker index")?;
        let mut before = &text[at..start];
        let mut next = end + end_marker.len_utf8();
        if marker == '\u{e002}' && before.ends_with('"') && text[next..].starts_with('"') {
            before = &before[..before.len() - 1];
            next += 1;
        }
        push(parts, before.to_owned());
        parts.push(json!({"type":"image"}));
        images.push(
            assets
                .get(order.get(index).ok_or("invalid asset marker reference")?)
                .ok_or("missing asset")?
                .clone(),
        );
        at = next;
    }
    push(parts, text[at..].to_owned());
    Ok(())
}

pub(super) struct Prompt {
    pub content: Vec<Value>,
    pub own_images: Vec<Value>,
    pub keys: Vec<String>,
    pub suffix: String,
}

pub(super) fn prompt(
    state: &Value,
    question: &Value,
    codes: &[String],
    n_front: usize,
    n_videos: usize,
    assets: &Map<String, Value>,
    refs: &[String],
) -> Result<Prompt, String> {
    if question["type"] == "score" && (n_front > 0 || n_videos > 0 || !assets.is_empty()) {
        return Err("score questions take no images here".into());
    }
    let spatial = question["type"] == "box" || question["type"] == "track";
    let presence = question["type"] == "track";
    let keys = if spatial {
        if !question["instructions"].is_string()
            || question
                .as_object()
                .unwrap()
                .keys()
                .any(|k| !matches!(k.as_str(), "type" | "instructions" | "at"))
        {
            return Err("a box question takes type and instructions".into());
        }
        Vec::new()
    } else {
        let (keys, _) = decoder::options(question)?;
        keys
    };
    let mut content = vec![json!({"type":"image"}); n_front];
    content.extend(vec![json!({"type":"video"}); n_videos]);
    let mut p = Prompt {
        content,
        own_images: Vec::new(),
        keys,
        suffix: String::new(),
    };
    if assets.is_empty() && !spatial {
        push(
            &mut p.content,
            decoder::user_prompt(state, question, codes)?,
        );
        return Ok(p);
    }
    let body = if state.is_null() || state == "" {
        "(empty)".into()
    } else {
        describe(state)
    };
    push(&mut p.content, format!("State:\n{body}\n\nQuestion:\n"));
    let instructions = question
        .get("instructions")
        .filter(|v| truth(v))
        .cloned()
        .unwrap_or_else(|| json!("Choose the best matching option."));
    describe_parts(
        &instructions,
        refs,
        assets,
        &mut p.content,
        &mut p.own_images,
    )?;
    if spatial {
        let when = question["at"]
            .as_f64()
            .map(|at| format!(" in the video frame at {at:.1} seconds"))
            .unwrap_or_default();
        let visible = if presence {
            format!(
                "\nAlso say whether the target is visible then: {} no, {} yes.",
                codes[0], codes[1]
            )
        } else {
            String::new()
        };
        push(&mut p.content, format!("\n\nBox:\nFind the target{when}. Give its top, bottom, left and right edges, each as a code on a 128-step scale across the image: {} is the top or left side, {} the bottom or right side.{visible}\n\nReply with the edge codes.",codes[0],codes[127]));
        p.suffix =
            if presence { "visible:\n" } else { "" }.to_owned() + "top:\nbottom:\nleft:\nright:";
        return Ok(p);
    }
    push(&mut p.content, "\n\nOptions:".into());
    for (i, key) in p.keys.iter().enumerate() {
        push(&mut p.content, format!("\n{}: ", codes[i]));
        let value = &question["criteria"][key];
        if question["type"] == "choice" {
            push(&mut p.content, key.clone());
            if !value.is_null() {
                push(&mut p.content, ": ".into());
                describe_parts(value, refs, assets, &mut p.content, &mut p.own_images)?;
            }
        } else {
            let value = if truth(value) {
                value.clone()
            } else {
                json!(if i == 0 { "No / false" } else { "Yes / true" })
            };
            describe_parts(&value, refs, assets, &mut p.content, &mut p.own_images)?;
        }
    }
    push(
        &mut p.content,
        "\n\nReply with only the code of the best option.".into(),
    );
    Ok(p)
}

pub(super) fn readouts(
    text: &str,
    suffix: &str,
    encoded: &tokenizers::Encoding,
) -> Result<Vec<(usize, usize)>, String> {
    if suffix.is_empty() {
        return Ok(vec![(encoded.len() - 1, 0)]);
    }
    let base = text.len() - suffix.len();
    let mut end = base;
    suffix
        .lines()
        .map(|marker| {
            end += marker.len();
            let row = encoded
                .get_offsets()
                .iter()
                .rposition(|&(_, stop)| stop == end)
                .ok_or("a box marker's colon must end a token")?;
            end += 1;
            Ok((row, if marker == "visible:" { 2 } else { BINS }))
        })
        .collect()
}

fn order(a: f64, b: f64) -> (f64, f64) {
    if a < b {
        (a, b)
    } else {
        let mid = (a + b) / 2.;
        let half = 0.5 / BINS as f64;
        ((mid - half).max(0.), (mid + half).min(1.))
    }
}

pub(super) fn box_answer(logits: &[Vec<f32>], calibration: &[(f64, f64)]) -> Result<Value, String> {
    if logits.len() != 4
        || logits
            .iter()
            .any(|x| x.len() != BINS || x.iter().any(|v| !v.is_finite()))
    {
        return Err("invalid box readout".into());
    }
    let probabilities: Vec<Vec<f64>> = logits
        .iter()
        .map(|x| {
            let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut p: Vec<f64> = x.iter().map(|v| (v - max).exp() as f64).collect();
            let sum: f64 = p.iter().sum();
            for v in &mut p {
                *v /= sum;
            }
            p
        })
        .collect();
    let edges: Vec<f64> = probabilities
        .iter()
        .map(|p| {
            p.iter()
                .enumerate()
                .map(|(i, v)| v * (i as f64 + 0.5) / BINS as f64)
                .sum()
        })
        .collect();
    let (top, bottom) = order(edges[0], edges[1]);
    let (left, right) = order(edges[2], edges[3]);
    let mut phi = 1f64;
    for _ in 0..64 {
        phi = (1. + phi).powf(0.2);
    }
    let alpha: [f64; 4] = std::array::from_fn(|d| phi.recip().powi(d as i32 + 1) % 1.);
    let mut hits = 0;
    for i in 0..512 {
        let draw: [f64; 4] = std::array::from_fn(|d| {
            let point = (0.5 + alpha[d] * (i + 1) as f64) % 1.;
            let mut sum = 0.;
            let index = probabilities[d]
                .iter()
                .position(|v| {
                    sum += v;
                    sum >= point
                })
                .unwrap_or(BINS - 1);
            (index as f64 + 0.5) / BINS as f64
        });
        let [t, b, l, r] = draw;
        let inter = (b.min(bottom) - t.max(top)).max(0.) * (r.min(right) - l.max(left)).max(0.);
        let union = (b - t).max(0.) * (r - l).max(0.) + (bottom - top) * (right - left) - inter;
        if t < b && l < r && union > 0. && inter / union >= 0.5 {
            hits += 1;
        }
    }
    let mut confidence = hits as f64 / 512.;
    if let Some(&(x, y)) = calibration.first() {
        if confidence <= x {
            confidence = y;
        } else if let Some(pair) = calibration.windows(2).find(|p| confidence <= p[1].0) {
            let [(x0, y0), (x1, y1)] = [pair[0], pair[1]];
            confidence = if x1 == x0 {
                y0
            } else {
                y0 + (y1 - y0) * (confidence - x0) / (x1 - x0)
            };
        } else {
            confidence = calibration.last().unwrap().1;
        }
    }
    Ok(
        json!({"type":"box","box":{"top":top,"bottom":bottom,"left":left,"right":right},"confidence":confidence}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn score_media_is_rejected_before_inference() {
        let question = json!({"type":"score","criteria":["bad","good"]});
        let codes = vec!["A".into(), "B".into()];
        for (images, videos, assets) in [
            (1, 0, json!({})),
            (0, 1, json!({})),
            (0, 0, json!({"@x":"x.png"})),
        ] {
            let assets = assets.as_object().unwrap();
            assert!(prompt(
                &json!("state"),
                &question,
                &codes,
                images,
                videos,
                assets,
                &references(assets).unwrap()
            )
            .is_err());
        }
        assert!(prompt(&json!("state"), &question, &codes, 0, 0, &Map::new(), &[]).is_ok());
    }

    #[test]
    fn renderer_and_box_readout_match_pinned_upstream_goldens() {
        let fixture: Value =
            serde_json::from_str(include_str!("../../../tools/oracle/d3_turbo/protocol.json"))
                .unwrap();
        let codes: Vec<String> = serde_json::from_value(fixture["codes"].clone()).unwrap();
        for case in fixture["cases"].as_array().unwrap() {
            let assets = case["assets"].as_object().unwrap();
            let p = prompt(
                &case["state"],
                &case["question"],
                &codes,
                case["front"].as_u64().unwrap() as usize,
                case["videos"].as_u64().unwrap_or(0) as usize,
                assets,
                &references(assets).unwrap(),
            )
            .unwrap();
            assert_eq!(json!(p.content), case["content"]);
            assert_eq!(json!(p.own_images), case["own_images"]);
            assert_eq!(p.suffix, case["suffix"].as_str().unwrap());
        }
        for case in fixture["boxes"].as_array().unwrap() {
            let logits: Vec<Vec<f32>> = serde_json::from_value(case["logits"].clone()).unwrap();
            let calibration =
                serde_json::from_value::<Vec<(f64, f64)>>(case["calibration"].clone()).unwrap();
            let answer = box_answer(&logits, &calibration).unwrap();
            for edge in EDGES {
                assert!(
                    (answer["box"][edge].as_f64().unwrap() - case["box"][edge].as_f64().unwrap())
                        .abs()
                        < 1e-6
                );
            }
            assert!(
                (answer["confidence"].as_f64().unwrap() - case["confidence"].as_f64().unwrap())
                    .abs()
                    < 1e-6
            );
        }
    }
    #[test]
    fn assets_are_longest_first_and_never_replace_object_keys() {
        let assets = json!({"@a":"a.png","@ab":"b.png"})
            .as_object()
            .unwrap()
            .clone();
        let refs = references(&assets).unwrap();
        let mut images = vec![json!("front.png")];
        let state = hoist(&json!({"@a":["@ab", "x @a"]}), &refs, &mut images, &assets);
        assert_eq!(state, json!({"@a":["[image 2]", "x [image 3]"]}));
        assert_eq!(
            images,
            json!(["front.png", "b.png", "a.png"])
                .as_array()
                .unwrap()
                .clone()
        );
        let mut parts = Vec::new();
        let mut own = Vec::new();
        describe_parts(
            &json!(["@ab", "text @a"]),
            &refs,
            &assets,
            &mut parts,
            &mut own,
        )
        .unwrap();
        assert_eq!(parts,json!([{"type":"text","text":"["},{"type":"image"},{"type":"text","text":", \"text "},{"type":"image"},{"type":"text","text":"\"]"}]).as_array().unwrap().clone());
        assert_eq!(own, vec![json!("b.png"), json!("a.png")]);
    }
    #[test]
    fn uniform_box_is_ordered_and_spatial_readout_rejects_nan() {
        let answer = box_answer(&vec![vec![0.; BINS]; 4], &[]).unwrap();
        assert!(answer["box"]["top"].as_f64().unwrap() < answer["box"]["bottom"].as_f64().unwrap());
        assert!(answer["confidence"].as_f64().unwrap() <= 1.);
        assert!(box_answer(&vec![vec![f32::NAN; BINS]; 4], &[]).is_err());
    }
}
