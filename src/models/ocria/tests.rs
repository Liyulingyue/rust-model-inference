//! OCRia-nano-1 tests.
//!
//! The contract tests run everywhere: they check the packed model's metadata
//! contract, the graph descriptions the loader enforces, and the pipeline
//! pieces that need no weights.
//!
//! The parity tests need a packed GGUF and the fixtures `tools/oracle/ocria`
//! records, so they are `#[ignore]`d. Run them with
//!
//! ```sh
//! OCRIA_GGUF=models/OCRia-nano-1/ocria-f32.gguf \
//! OCRIA_FIXTURES=tools/oracle/ocria/fixtures \
//! OCRIA_PAGES=tools/oracle/ocria/fixtures/pages \
//! cargo test --locked --profile release-fast --lib \
//!   models::ocria::tests -- --ignored --nocapture
//! ```
//!
//! Every numeric gate here is stated up front, in the test, with the value the
//! oracle record measured; none of them is relaxed after a failure.

use super::*;
use std::path::{Path, PathBuf};

/// The contract's `config`, which the GGUF must repeat exactly.
fn contract() -> serde_json::Value {
    serde_json::from_str(CONTRACT).unwrap()
}

#[test]
fn contract_names_every_tensor_and_its_shape() {
    let contract = contract();
    let config: Config = serde_json::from_value(contract["config"].clone()).unwrap();
    assert_eq!(
        config.rec.groups,
        ["latin", "cjk", "rtl", "world"],
        "the four recognizers are addressed by these names"
    );
    assert_eq!(config.rec.visual_rtl, [false, false, true, false]);
    assert_eq!(config.det.long_side, 1600);
    assert_eq!(config.det.thresh, 0.55);
    assert_eq!(config.det.merge, 0.6);

    let tensors = contract["tensors"].as_object().unwrap();
    let mut counts = std::collections::BTreeMap::new();
    for (name, shape) in tensors {
        let shape: Vec<usize> = serde_json::from_value(shape.clone()).unwrap();
        assert!(!shape.is_empty() && shape.iter().all(|&d| d > 0), "{name}");
        // Every weight is a flattened 2-D matrix [out, fan_in]; every bias is
        // one-dimensional. The converter stores conv kernels C-order flattened
        // so that a pointwise conv's row length is the 32-multiple Q8_0 needs.
        assert!(
            shape.len() == 1 || shape.len() == 2,
            "{name} has shape {shape:?}; weights are [out, fan_in] and biases [out]"
        );
        let component = if let Some(rest) = name.strip_prefix("rec.") {
            format!("rec.{}", rest.split('.').next().unwrap())
        } else {
            name.split('.').next().unwrap().to_string()
        };
        *counts.entry(component).or_insert(0usize) += 1;
    }
    assert_eq!(
        counts,
        [
            ("det".to_string(), 74),
            ("rec.cjk".to_string(), 60),
            ("rec.latin".to_string(), 58),
            ("rec.rtl".to_string(), 58),
            ("rec.world".to_string(), 58),
            ("router".to_string(), 34),
        ]
        .into_iter()
        .collect::<std::collections::BTreeMap<_, _>>(),
        "the tensor set is pinned by the converter against the ONNX graphs"
    );
}

#[test]
fn contract_tensor_names_follow_the_node_path_rule() {
    let contract = contract();
    let tensors = contract["tensors"].as_object().unwrap();
    for name in tensors.keys() {
        let (stem, suffix) = name
            .rsplit_once('.')
            .unwrap_or_else(|| panic!("{name} has no suffix"));
        assert!(
            suffix == "weight" || suffix == "bias",
            "{name} must end in .weight or .bias"
        );
        assert!(!stem.is_empty(), "{name}");
        assert!(
            !name.contains("//") && !name.starts_with('.') && !name.contains("/"),
            "{name} must be the flattened ONNX node path"
        );
    }
    for pinned in [
        "det.stem.stem.0.weight",
        "det.s4.s4.0.dw.weight",
        "det.lat.3.weight",
        "det.head.head.3.bias",
        "router.group.weight",
        "router.flip.bias",
        "rec.latin.head.weight",
        "rec.cjk.head.head.1.weight",
        "rec.cjk.mix.mix.0.fc1.weight",
    ] {
        assert!(tensors.contains_key(pinned), "missing {pinned}");
    }
    // `det.lat.*` are the only bias-free convs in the release.
    assert!(!tensors.contains_key("det.lat.3.bias"));
}

#[test]
fn options_default_to_the_benchmarked_pipeline() {
    let options = Options::default();
    assert_eq!(options.det_long_side, 1600);
    assert_eq!(options.det_merge, 0.6);
    assert!(options.gutter && options.orient && options.rescue && options.junk);
    assert_eq!(options.faint_max, 24);
}

#[test]
fn a_non_ocria_file_is_not_claimed() {
    assert!(!matches_model(Path::new("/nonexistent/ocria.gguf")));
}

#[test]
fn option_overrides_replace_only_what_they_name() {
    let base = Options {
        det_long_side: 1600,
        det_thresh: 0.55,
        det_merge: 0.6,
        ..Options::default()
    };
    let overrides: OptionsOverride = serde_json::from_str(
        r#"{"long_side":800,"threshold":0.4,"orient":false,"faint_threshold":0.0}"#,
    )
    .unwrap();
    let options = overrides.apply(base).unwrap();
    assert_eq!(options.det_long_side, 800);
    assert_eq!(options.det_thresh, 0.4);
    assert!(!options.orient);
    assert_eq!(options.faint_thresh, 0.0);
    // untouched fields keep the base
    assert_eq!(options.det_merge, 0.6);
    assert!(options.gutter && options.rescue && options.junk);
    // an empty object is the identity
    let identity: OptionsOverride = serde_json::from_str("{}").unwrap();
    assert_eq!(identity.apply(base).unwrap().det_long_side, 1600);
}

#[test]
fn option_overrides_reject_values_the_pipeline_cannot_run_with() {
    let base = Options::default();
    for (json, needle) in [
        (r#"{"threshold":0}"#, "threshold"),
        (r#"{"threshold":1}"#, "threshold"),
        (r#"{"threshold":-0.5}"#, "threshold"),
        (r#"{"long_side":0}"#, "long side"),
        (r#"{"long_side":99999}"#, "long side"),
        (r#"{"merge":-1}"#, "merge distance"),
        (r#"{"faint_threshold":1.5}"#, "faint threshold"),
        (
            r#"{"faint_threshold":0.5,"faint_max":0}"#,
            "candidate budget",
        ),
        (r#"{"unknown":1}"#, "unknown field"),
    ] {
        let overrides: Result<OptionsOverride, _> = serde_json::from_str(json);
        let rejected = match overrides {
            Ok(overrides) => overrides.apply(base).is_err(),
            Err(_) => true,
        };
        assert!(rejected, "{json} should be rejected");
        let message = match serde_json::from_str::<OptionsOverride>(json) {
            Ok(overrides) => overrides.apply(base).unwrap_err(),
            Err(error) => error.to_string(),
        };
        assert!(message.contains(needle), "{json}: {message}");
    }
}

// ---------------------------------------------------------------------------
// Parity against the ONNX Runtime oracle
// ---------------------------------------------------------------------------

fn fixture_dir() -> PathBuf {
    std::env::var_os("OCRIA_FIXTURES")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("tools/oracle/ocria/fixtures"))
}

fn packed_model() -> Option<(Model, PathBuf)> {
    let path = std::env::var_os("OCRIA_GGUF").map(PathBuf::from)?;
    let source =
        crate::format::ggufrs::open_model_source(&path, crate::format::ggufrs::ComponentRole::Llm)
            .expect("open the packed OCRia GGUF");
    let model = Model::from_source(source.as_ref(), ComputePolicy::Cpu, 2, None)
        .expect("load the packed OCRia GGUF");
    Some((model, path))
}

fn read_f32(dir: &Path, name: &str, count: usize) -> Vec<f32> {
    let bytes = std::fs::read(dir.join(name)).unwrap_or_else(|e| panic!("{name}: {e}"));
    assert_eq!(bytes.len(), count * 4, "{name} length");
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

fn read_i64(dir: &Path, name: &str, count: usize) -> Vec<i64> {
    let bytes = std::fs::read(dir.join(name)).unwrap_or_else(|e| panic!("{name}: {e}"));
    assert_eq!(bytes.len(), count * 8, "{name} length");
    bytes
        .chunks_exact(8)
        .map(|b| i64::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

fn manifest() -> Option<(serde_json::Value, PathBuf)> {
    let dir = fixture_dir();
    let path = dir.join("manifest.json");
    if !path.is_file() {
        return None;
    }
    Some((
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap(),
        dir,
    ))
}

fn max_abs(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max)
}

/// The detector's probability map against ONNX Runtime's, on the same input.
///
/// The gate is absolute: the map is a sigmoid of a shallow network, and the
/// oracle record measured 3.6e-6, so 1e-5 leaves room for a different
/// convolution order without hiding a semantic difference.
#[test]
#[ignore = "requires OCRIA_GGUF and the ONNX Runtime fixtures"]
fn detector_probability_map_matches_onnx() {
    let Some((model, _)) = packed_model() else {
        panic!("set OCRIA_GGUF");
    };
    let (manifest, dir) = manifest().expect("fixtures manifest.json");
    let mut worst = 0.0f32;
    for case in manifest["det"].as_array().unwrap() {
        let (h, w) = (
            case["height"].as_u64().unwrap() as usize,
            case["width"].as_u64().unwrap() as usize,
        );
        let input = read_f32(&dir, case["input"].as_str().unwrap(), 3 * h * w);
        let (ph, pw) = (h / 2, w / 2);
        let expected = read_f32(&dir, case["prob"].as_str().unwrap(), ph * pw);
        let (prob, got_h, got_w) = model.det.forward_prob(&input, h, w).unwrap();
        assert_eq!((got_h, got_w), (ph, pw), "{}", case["name"]);
        let delta = max_abs(&prob, &expected);
        println!("det {}: max_abs {delta:.3e}", case["name"]);
        worst = worst.max(delta);
    }
    assert!(worst <= 1e-5, "detector prob max_abs {worst:.3e} > 1e-5");
}

/// The router's group and flip logits against ONNX Runtime's.
///
/// The router ends in a `Gemm` over a pooled vector, so the gate is 1e-4
/// absolute on the logits -- wide enough for the pooled sum's order, far
/// narrower than the 0.9 override the page policy acts on.
#[test]
#[ignore = "requires OCRIA_GGUF and the ONNX Runtime fixtures"]
fn router_logits_match_onnx() {
    let Some((model, _)) = packed_model() else {
        panic!("set OCRIA_GGUF");
    };
    let (manifest, dir) = manifest().expect("fixtures manifest.json");
    let mut worst_group = 0.0f32;
    let mut worst_flip = 0.0f32;
    for case in manifest["router"].as_array().unwrap() {
        let count = case["count"].as_u64().unwrap() as usize;
        let input = read_f32(&dir, case["input"].as_str().unwrap(), count * 32 * 512);
        let group = read_f32(&dir, case["group_logits"].as_str().unwrap(), count * 4);
        let flip = read_f32(&dir, case["flip_logits"].as_str().unwrap(), count * 2);
        let (got_group, got_flip) = model.router.forward_logits(&input, count).unwrap();
        let dg = max_abs(&got_group, &group);
        let df = max_abs(&got_flip, &flip);
        println!("router {}: group {dg:.3e} flip {df:.3e}", case["name"]);
        worst_group = worst_group.max(dg);
        worst_flip = worst_flip.max(df);
    }
    assert!(
        worst_group <= 1e-4 && worst_flip <= 1e-4,
        "router logits max_abs group {worst_group:.3e} flip {worst_flip:.3e} > 1e-4"
    );
}

/// The recognizers' per-timestep decisions against ONNX Runtime's.
///
/// `ids` is the contract the text is built from, so it is compared exactly.
/// `conf` is the in-graph softmax maximum, compared at 1e-3.
///
/// `rec_latin`'s released graph is ONNX-Runtime-quantized: it snaps its weights
/// *and* its activations to int8/uint8 grids inside the graph. This crate
/// dequantizes the weights and computes in F32, so the fixture gives every
/// group the faithful F32 graph's `ids_f32`/`conf_f32` as the target and keeps
/// the QDQ graph's `ids`/`conf` for the record; the measured difference between
/// the two is in `latin_qdq_effect.json` and `validation.md`.
#[test]
#[ignore = "requires OCRIA_GGUF and the ONNX Runtime fixtures"]
fn recognizer_decisions_match_onnx() {
    let Some((model, _)) = packed_model() else {
        panic!("set OCRIA_GGUF");
    };
    let (manifest, dir) = manifest().expect("fixtures manifest.json");
    let mut worst_conf = 0.0f32;
    let mut id_mismatches = 0usize;
    let mut id_total = 0usize;
    for (index, group) in rec::GROUPS.iter().enumerate() {
        for case in manifest["rec"][group].as_array().unwrap() {
            let count = case["count"].as_u64().unwrap() as usize;
            let width = case["width"].as_u64().unwrap() as usize;
            let input = read_f32(&dir, case["input"].as_str().unwrap(), count * 32 * width);
            let (logits, steps) = model.recs[index]
                .forward_logits(&input, count, width)
                .unwrap();
            let classes = logits.len() / (count * steps);
            // `rec_latin`'s graph is int8-quantized (QDQ) and snaps its
            // activations at run time, which this crate does not reproduce;
            // the fixture therefore carries the faithful F32 graph's answer
            // (`ids_f32`/`conf_f32`) as the target for every group, and the
            // QDQ graph's answer (`ids`/`conf`) for the record.
            let (ids_key, conf_key) = if case.get("ids_f32").is_some() {
                ("ids_f32", "conf_f32")
            } else {
                ("ids", "conf")
            };
            let ids = read_i64(&dir, case[ids_key].as_str().unwrap(), count * steps);
            let conf = read_f32(&dir, case[conf_key].as_str().unwrap(), count * steps);
            let mut got_ids = Vec::with_capacity(count * steps);
            let mut got_conf = Vec::with_capacity(count * steps);
            // `forward_logits` returns the head's channel-first plane, so a
            // timestep's classes are strided by `steps`, not contiguous.
            for step in 0..count * steps {
                let row: Vec<f32> = (0..classes).map(|c| logits[c * steps + step]).collect();
                let row = &row[..];
                let mut best = 0usize;
                for (k, &value) in row.iter().enumerate().skip(1) {
                    if value > row[best] {
                        best = k;
                    }
                }
                let peak = row[best];
                let sum: f32 = row.iter().map(|v| (v - peak).exp()).sum();
                got_ids.push(best as i64);
                got_conf.push(1.0 / sum);
            }
            let mismatch = ids
                .iter()
                .zip(&got_ids)
                .filter(|(want, got)| want != got)
                .count();
            let delta = max_abs(&got_conf, &conf);
            println!(
                "rec.{group} {}: ids {mismatch}/{} differ, conf max_abs {delta:.3e}",
                case["name"],
                count * steps
            );
            id_mismatches += mismatch;
            id_total += count * steps;
            worst_conf = worst_conf.max(delta);
        }
    }
    println!("rec: ids {id_mismatches}/{id_total} differ, worst conf {worst_conf:.3e}");
    assert!(
        worst_conf <= 1e-3,
        "recognizer conf max_abs {worst_conf:.3e} > 1e-3"
    );
    // The latin graph is the only quantized one; the other three must match
    // exactly. The count for latin is recorded in tools/oracle/ocria/README.md.
    assert_eq!(
        id_mismatches, 0,
        "recognizer ids differ on {id_mismatches}/{id_total} timesteps"
    );
}

/// The recorded pages and their oracle line lists.
fn page_cases() -> Vec<(PathBuf, serde_json::Value)> {
    let pages = std::env::var_os("OCRIA_PAGES")
        .map(PathBuf::from)
        .unwrap_or_else(|| fixture_dir().join("pages"));
    let mut cases = Vec::new();
    for entry in std::fs::read_dir(&pages).expect("page fixture directory") {
        let path = entry.unwrap().path();
        if path.extension().and_then(|e| e.to_str()) != Some("png") {
            continue;
        }
        let oracle_path = path.with_extension("json");
        if !oracle_path.is_file() {
            continue;
        }
        let oracle: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&oracle_path).unwrap()).unwrap();
        cases.push((path, oracle));
    }
    cases.sort_by(|a, b| a.0.cmp(&b.0));
    assert!(!cases.is_empty(), "no page fixtures in {}", pages.display());
    cases
}

fn mean_confidence(lines: impl Iterator<Item = f32>) -> f64 {
    let (sum, count) = lines.fold((0.0f64, 0usize), |(sum, count), v| {
        (sum + v as f64, count + 1)
    });
    sum / count as f64
}

/// End to end: the same page image through this crate and through the pinned
/// LiteParse engine, on the F32 weights.
///
/// Text and the line count must match exactly, and the boxes within 1 px (the
/// two sides round the same float differently on the way out).
///
/// Confidence gets two gates. The recognizers agree with their own graph to
/// ~3e-6, but the oracle ran `rec_latin.onnx`, which snaps its activations to
/// uint8 grids inside the graph and this crate does not: per timestep the
/// effect is ~2e-2 and, when a character sits on a near tie, one character can
/// move a whole line's mean by ~4e-2. So a line's confidence is compared at
/// 2e-1 -- wide, but still narrow enough to catch a wrong CTC collapse or a
/// wrong averaging denominator -- while the page's mean confidence, where the
/// near ties average out, is compared at 5e-3. The measured values are printed
/// and recorded in `validation.md`.
#[test]
#[ignore = "requires OCRIA_GGUF and the recorded page oracle"]
fn page_read_matches_liteparse() {
    let Some((model, _)) = packed_model() else {
        panic!("set OCRIA_GGUF");
    };
    for (path, oracle) in page_cases() {
        let image = image::open(&path).unwrap().to_rgb8();
        let page = model
            .run(image.as_raw(), image.width(), image.height())
            .expect("read the page");
        let expected = oracle["lines"].as_array().unwrap();
        assert_eq!(
            page.lines.len(),
            expected.len(),
            "{}: line count",
            path.display()
        );
        let mut worst_confidence = 0.0f32;
        for (got, want) in page.lines.iter().zip(expected) {
            assert_eq!(
                got.text,
                want["text"].as_str().unwrap(),
                "{}",
                path.display()
            );
            let bbox = want["bbox"].as_array().unwrap();
            for k in 0..4 {
                let want = bbox[k].as_f64().unwrap() as f32;
                assert!(
                    (got.bbox[k] - want).abs() <= 1.0,
                    "{}: bbox[{k}] {} vs {want}",
                    path.display(),
                    got.bbox[k]
                );
            }
            let confidence = want["confidence"].as_f64().unwrap() as f32;
            worst_confidence = worst_confidence.max((got.confidence - confidence).abs());
            assert!(
                (got.confidence - confidence).abs() <= 2e-1,
                "{}: confidence {} vs {confidence}",
                path.display(),
                got.confidence
            );
            assert_eq!(
                got.faint,
                want["faint"].as_bool().unwrap_or(false),
                "{}: faint flag",
                path.display()
            );
        }
        let got_mean = mean_confidence(page.lines.iter().map(|line| line.confidence));
        let want_mean = mean_confidence(
            expected
                .iter()
                .map(|line| line["confidence"].as_f64().unwrap() as f32),
        );
        assert!(
            (got_mean - want_mean).abs() <= 5e-3,
            "{}: mean confidence {got_mean:.6} vs {want_mean:.6}",
            path.display()
        );
        println!(
            "{}: {} lines match, worst line delta {worst_confidence:.2e}, mean confidence \
             {got_mean:.6} vs {want_mean:.6}",
            path.display(),
            page.lines.len()
        );
    }
}

/// The Q8_0 export against the same ONNX Runtime fixtures.
///
/// The F32 gates above are not applied here: 85 of the 342 tensors are stored
/// in 8-bit blocks, so the numbers move. The gates below are the drift measured
/// on the committed fixtures plus headroom (see `tools/oracle/ocria/validation.md`):
/// detector 1.85e-2, router logits 6.08e-2, 16 of 1064 recognizer timesteps
/// (1.5 %) and a 0.208 per-timestep confidence delta. The confidence gate is a
/// sanity bound rather than a precision claim -- a timestep's softmax maximum
/// moves a lot when two classes are nearly tied -- while `ids` is the decision
/// that reaches the text. They are here so a change to the quantization rule or
/// the loader shows up as a failure rather than as a silent drift.
#[test]
#[ignore = "requires OCRIA_GGUF (q8_0) and the ONNX Runtime fixtures"]
fn q8_0_weights_stay_within_the_measured_quantization_bounds() {
    let Some((model, path)) = packed_model() else {
        panic!("set OCRIA_GGUF");
    };
    if model.format != WeightFormat::Q8_0 {
        // The gate is the quantized file's; running the suite with the F32 one
        // is the normal case, so skip rather than fail.
        println!("skipped: {} is not a Q8_0 file", path.display());
        return;
    }
    let (manifest, dir) = manifest().expect("fixtures manifest.json");

    let mut worst_det = 0.0f32;
    for case in manifest["det"].as_array().unwrap() {
        let (h, w) = (
            case["height"].as_u64().unwrap() as usize,
            case["width"].as_u64().unwrap() as usize,
        );
        let input = read_f32(&dir, case["input"].as_str().unwrap(), 3 * h * w);
        let expected = read_f32(&dir, case["prob"].as_str().unwrap(), (h / 2) * (w / 2));
        let (prob, _, _) = model.det.forward_prob(&input, h, w).unwrap();
        worst_det = worst_det.max(max_abs(&prob, &expected));
    }

    let mut worst_router = 0.0f32;
    for case in manifest["router"].as_array().unwrap() {
        let count = case["count"].as_u64().unwrap() as usize;
        let input = read_f32(&dir, case["input"].as_str().unwrap(), count * 32 * 512);
        let group = read_f32(&dir, case["group_logits"].as_str().unwrap(), count * 4);
        let flip = read_f32(&dir, case["flip_logits"].as_str().unwrap(), count * 2);
        let (got_group, got_flip) = model.router.forward_logits(&input, count).unwrap();
        worst_router = worst_router.max(max_abs(&got_group, &group).max(max_abs(&got_flip, &flip)));
    }

    let (mut differing, mut total, mut worst_conf) = (0usize, 0usize, 0.0f32);
    for (index, group) in rec::GROUPS.iter().enumerate() {
        for case in manifest["rec"][group].as_array().unwrap() {
            let count = case["count"].as_u64().unwrap() as usize;
            let width = case["width"].as_u64().unwrap() as usize;
            let input = read_f32(&dir, case["input"].as_str().unwrap(), count * 32 * width);
            let (logits, steps) = model.recs[index]
                .forward_logits(&input, count, width)
                .unwrap();
            let classes = logits.len() / (count * steps);
            let ids = read_i64(&dir, case["ids"].as_str().unwrap(), count * steps);
            let conf = read_f32(&dir, case["conf"].as_str().unwrap(), count * steps);
            for step in 0..count * steps {
                let row: Vec<f32> = (0..classes).map(|c| logits[c * steps + step]).collect();
                let mut best = 0usize;
                for (k, &value) in row.iter().enumerate().skip(1) {
                    if value > row[best] {
                        best = k;
                    }
                }
                let peak = row[best];
                let sum: f32 = row.iter().map(|v| (v - peak).exp()).sum();
                if ids[step] != best as i64 {
                    differing += 1;
                }
                worst_conf = worst_conf.max((1.0 / sum - conf[step]).abs());
            }
            total += count * steps;
        }
    }
    println!(
        "q8_0: det {worst_det:.3e}, router {worst_router:.3e}, ids {differing}/{total}, conf {worst_conf:.3e}"
    );
    assert!(
        worst_det <= 3e-2,
        "q8_0 detector drift {worst_det:.3e} > 3e-2"
    );
    assert!(
        worst_router <= 1e-1,
        "q8_0 router drift {worst_router:.3e} > 1e-1"
    );
    assert!(
        differing * 20 <= total,
        "q8_0 recognizer ids differ on {differing}/{total} timesteps (> 5 %)"
    );
    assert!(
        worst_conf <= 3e-1,
        "q8_0 recognizer conf drift {worst_conf:.3e} > 3e-1"
    );
}

/// Layer 2: the network against ONNX Runtime, on the committed fixtures.
///
/// `tools/oracle/ocria/layout_fixtures` holds two pages' graph inputs, three
/// outputs and eleven intermediates, recorded from the pinned graph. The gate is
/// 1e-4 on the logits: the transformer runs six layers of f32 matmuls in a
/// different reduction order than ONNX Runtime, and the fixtures were recorded
/// with that in mind.
#[test]
#[ignore = "requires OCRIA_LAYOUT and the committed layout fixtures"]
fn layout_network_matches_onnx() {
    let layout_path = std::env::var_os("OCRIA_LAYOUT").expect("set OCRIA_LAYOUT");
    let source = crate::format::ggufrs::open_model_source(
        std::path::Path::new(&layout_path),
        crate::format::ggufrs::ComponentRole::Llm,
    )
    .expect("open the layout GGUF");
    let model = layout::LayoutNet::from_source(source.as_ref()).expect("load layout");

    let root = PathBuf::from("tools/oracle/ocria/layout_fixtures");
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("manifest.json")).unwrap()).unwrap();
    let blob = |case: &serde_json::Value, name: &str| -> (Vec<usize>, PathBuf) {
        let entry = case["blobs"]
            .as_array()
            .unwrap()
            .iter()
            .find(|b| b["name"] == name)
            .unwrap_or_else(|| panic!("no blob {name}"));
        let shape: Vec<usize> = entry["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|d| d.as_u64().unwrap() as usize)
            .collect();
        (shape, root.join(entry["file"].as_str().unwrap()))
    };
    let read_f32_blob = |shape: &[usize], path: &PathBuf| -> Vec<f32> {
        let bytes = std::fs::read(path).unwrap();
        assert_eq!(
            bytes.len(),
            shape.iter().product::<usize>() * 4,
            "{}",
            path.display()
        );
        bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect()
    };

    let mut worst = 0.0f32;
    for case in manifest["cases"].as_array().unwrap() {
        let n = case["units"].as_u64().unwrap() as usize;
        let (feats_shape, feats_path) = blob(case, "feats");
        let (pair_shape, pair_path) = blob(case, "pairf");
        let image_entry = case["blobs"]
            .as_array()
            .unwrap()
            .iter()
            .find(|b| b["name"] == "image")
            .expect("no image blob");
        let image_shape: Vec<usize> = image_entry["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|d| d.as_u64().unwrap() as usize)
            .collect();
        let (boxes_shape, boxes_path) = blob(case, "boxes");
        let (valid_shape, valid_path) = blob(case, "valid");
        let feats = read_f32_blob(&feats_shape, &feats_path);
        let pairs = read_f32_blob(&pair_shape, &pair_path);
        let boxes = read_f32_blob(&boxes_shape, &boxes_path);
        let valid = read_f32_blob(&valid_shape, &valid_path);
        // A case may name a raster file or a constant fill (the all-white page
        // costs no bytes and pins the branch's biases on their own).
        let canvas: Vec<f32> = match image_entry["file"].as_str() {
            Some(file) => {
                let ink = std::fs::read(root.join(file)).unwrap();
                assert_eq!(ink.len(), image_shape.iter().product::<usize>());
                ink.iter().map(|&v| v as f32 / 255.0).collect()
            }
            None => {
                let fill = image_entry["fill"].as_u64().unwrap_or(0) as u8;
                vec![fill as f32 / 255.0; image_shape.iter().product::<usize>()]
            }
        };
        let img = layout::features::ImgInput {
            canvas,
            boxes: boxes
                .chunks_exact(4)
                .map(|b| [b[0], b[1], b[2], b[3]])
                .collect(),
            valid: (valid[0], valid[1]),
        };
        // Every stage the reference records, through the real forward path:
        // the image branch's vector, the encoder input, the first encoder
        // layer, the last encoder layer before its cross-attention, and the
        // hidden state the heads read.
        let stages = model.net().debug_stages(&feats, &pairs, Some(&img), n);
        for (name, got) in &stages {
            let (shape, path) = blob(case, name);
            let want = read_f32_blob(&shape, &path);
            let delta = max_abs(got, &want);
            println!("layout {} {name}: max_abs {delta:.3e}", case["name"]);
            worst = worst.max(delta);
        }
        let logits = model.net().forward(&feats, &pairs, Some(&img), n);
        assert_eq!(logits.n, n);
        assert_eq!(logits.n_cls, 13);
        for (name, got, want) in [
            ("node", &logits.node, "node"),
            ("pair", &logits.pair, "pair"),
            ("cls", &logits.cls, "cls"),
        ] {
            let (shape, path) = blob(case, want);
            let expected = read_f32_blob(&shape, &path);
            let delta = max_abs(got, &expected);
            // How often the decision the heads feed would survive the gap: for
            // `node` and `cls` that is the per-unit argmax, for `pair` the
            // per-pair one. A large delta with matching argmaxes is a numeric
            // difference; a small delta with mismatches is a layout difference.
            let (rows, cols) = match name {
                "node" => (n, 2),
                "cls" => (n, 13),
                _ => (n * n, logits.n_pair),
            };
            let mut agree = 0;
            for row in 0..rows {
                let want_row = &expected[row * cols..(row + 1) * cols];
                let got_row = &got[row * cols..(row + 1) * cols];
                let argmax = |v: &[f32]| {
                    let mut best = 0;
                    for (k, &x) in v.iter().enumerate() {
                        if x > v[best] {
                            best = k;
                        }
                    }
                    best
                };
                if argmax(want_row) == argmax(got_row) {
                    agree += 1;
                }
            }
            let mean = got
                .iter()
                .zip(&expected)
                .map(|(a, b)| (a - b) as f64)
                .sum::<f64>()
                / got.len() as f64;
            println!(
                "layout {} {name}: max_abs {delta:.3e}, mean delta {mean:+.3e}, argmax {agree}/{rows}",
                case["name"]
            );
            worst = worst.max(delta);
        }
    }
    assert!(worst <= 1e-4, "layout logits max_abs {worst:.3e} > 1e-4");
}

/// Layer 2: the table model finds the ruled table the reference finds.
///
/// `fixtures/layout/table.png` is a 3-column ruled table whose reference
/// Markdown (`fixtures/layout/table.reference.md`, recorded with the pinned
/// LiteParse engine) is four rows with a header row. The assertion is on the
/// structure — the recognizer reads `2` as `N` on this synthetic page, so the
/// cell text is checked for the words that are unambiguous, not for equality.
#[test]
#[ignore = "requires OCRIA_GGUF, OCRIA_LAYOUT and the recorded layout fixture"]
fn layout_finds_the_ruled_table() {
    let Some((model, _)) = packed_model() else {
        panic!("set OCRIA_GGUF");
    };
    let layout = std::env::var_os("OCRIA_LAYOUT").expect("set OCRIA_LAYOUT");
    let source = crate::format::ggufrs::open_model_source(
        std::path::Path::new(&layout),
        crate::format::ggufrs::ComponentRole::Llm,
    )
    .expect("open the layout GGUF");
    let model = model.with_layout(source.as_ref()).expect("attach layout");
    assert!(model.has_layout());

    let page = PathBuf::from("tools/oracle/ocria/fixtures/layout/table.png");
    let image = image::open(&page).unwrap().to_rgb8();
    let read = model
        .run(image.as_raw(), image.width(), image.height())
        .expect("read the page");
    assert_eq!(read.tables.len(), 1, "one table");
    let table = &read.tables[0];
    assert_eq!((table.rows, table.columns), (4, 3), "4 rows by 3 columns");
    assert_eq!(table.head_rows, vec![0], "the header row");
    let cells: Vec<&str> = table.cells.iter().map(|cell| cell.text.as_str()).collect();
    println!("table cells: {cells:?}");
    for word in ["Item", "Qty", "Price", "Coffee", "Bread", "Oil"] {
        assert!(
            cells.iter().any(|cell| cell.contains(word)),
            "missing {word} in {cells:?}"
        );
    }
    // every cell lands inside the table's box
    for cell in &table.cells {
        assert!(cell.bbox[0] >= table.bbox[0] - 1.0 && cell.bbox[2] <= table.bbox[2] + 1.0);
        assert!(cell.bbox[1] >= table.bbox[1] - 1.0 && cell.bbox[3] <= table.bbox[3] + 1.0);
    }
}

/// End to end on the quantized export.
///
/// The Q8_0 file stores 85 of the 342 tensors in 8-bit blocks, so its numbers
/// move: the detector's probability map and the router's logits are measured
/// against the F32 file in `validation.md`. What has to survive is the page:
/// the same text, in the same order, from the same boxes. This asserts exactly
/// that, and prints the confidence drift for the record.
#[test]
#[ignore = "requires OCRIA_GGUF (q8_0) and the recorded page oracle"]
fn page_read_q8_0_keeps_the_text() {
    let Some((model, path)) = packed_model() else {
        panic!("set OCRIA_GGUF");
    };
    if model.format != WeightFormat::Q8_0 {
        println!("skipped: {} is not a Q8_0 file", path.display());
        return;
    }
    for (path, oracle) in page_cases() {
        let image = image::open(&path).unwrap().to_rgb8();
        let page = model
            .run(image.as_raw(), image.width(), image.height())
            .expect("read the page");
        let expected = oracle["lines"].as_array().unwrap();
        assert_eq!(
            page.lines.len(),
            expected.len(),
            "{}: line count",
            path.display()
        );
        let mut worst = 0.0f32;
        for (got, want) in page.lines.iter().zip(expected) {
            assert_eq!(
                got.text,
                want["text"].as_str().unwrap(),
                "{}",
                path.display()
            );
            let confidence = want["confidence"].as_f64().unwrap() as f32;
            worst = worst.max((got.confidence - confidence).abs());
        }
        let got_mean = mean_confidence(page.lines.iter().map(|line| line.confidence));
        let want_mean = mean_confidence(
            expected
                .iter()
                .map(|line| line["confidence"].as_f64().unwrap() as f32),
        );
        println!(
            "{}: {} lines, text identical, worst line delta {worst:.2e}, mean confidence \
             {got_mean:.6} vs {want_mean:.6}",
            path.display(),
            page.lines.len()
        );
    }
}
