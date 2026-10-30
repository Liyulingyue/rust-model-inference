//! Native Vela3 System One inference from the official safetensors packages.
pub(crate) mod d3;
pub(crate) mod decoder;
pub(crate) mod fast;
pub(crate) mod tensor;
pub(crate) mod text;
pub(crate) mod ultrabert;
pub(crate) mod vision;

use crate::core::thread_pool::ComputePool;
use serde_json::Value;
use std::path::Path;
use std::sync::Arc;

#[derive(Debug, PartialEq)]
enum Family {
    D3,
    Fast,
    Vision,
}

fn family(config: &Value, model: &Value) -> Result<Family, String> {
    match (config["format"].as_str(), model["model_type"].as_str()) {
        (Some("vela3/3"), Some("qwen3_5")) => Ok(Family::D3),
        (Some("vela3-encoder/1"), Some("ultrabert")) => Ok(Family::Fast),
        (Some("vela3v-encoder/1"), Some("ultravbert")) => Ok(Family::Vision),
        (format, model_type) => Err(format!(
            "unsupported Vela format/model_type: {format:?}/{model_type:?}"
        )),
    }
}

/// Answer a System One `{state, questions, images?}` request.
/// d3 variants require the separate base directory pinned by `vela3_config.json`.
/// All computation uses the repository's CPU operators and thread pool.
pub fn system_one(
    model_dir: &Path,
    base_dir: Option<&Path>,
    request: &Value,
    threads: usize,
) -> Result<Value, String> {
    if !request.is_object() || !request["questions"].is_object() {
        return Err("request must contain a questions object".into());
    }
    if let Some(images) = request.get("images") {
        if !images.is_null() && !images.is_array() {
            return Err("images must be an array".into());
        }
    }
    for field in ["videos", "assets", "long_context"] {
        if request.get(field).is_some_and(|value| match value {
            Value::Null | Value::Bool(false) => false,
            Value::Array(items) => !items.is_empty(),
            Value::Object(items) => !items.is_empty(),
            _ => true,
        }) {
            return Err(format!("unsupported Vela3 request field: {field}"));
        }
    }
    if threads == 0 {
        return Err("threads must be positive".into());
    }
    let read = |file: &str| -> Result<Value, String> {
        let path = model_dir.join(file);
        serde_json::from_slice(
            &std::fs::read(&path).map_err(|e| format!("Read {}: {e}", path.display()))?,
        )
        .map_err(|e| format!("Parse {}: {e}", path.display()))
    };
    let config_path = if model_dir.join("vela3v_config.json").is_file() {
        "vela3v_config.json"
    } else {
        "vela3_config.json"
    };
    let kind = family(&read(config_path)?, &read("config.json")?)?;
    let pool = Arc::new(ComputePool::new(threads));
    match kind {
        Family::D3 => decoder::predict(
            model_dir,
            base_dir.ok_or("d3 variants need --base-model pointing to the pinned d3 snapshot")?,
            request,
            pool,
        ),
        Family::Fast => fast::predict(model_dir, request, &pool),
        Family::Vision => vision::predict(model_dir, request, &pool),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn explicit_formats_never_fall_back_by_name() {
        for (format, model_type, expected) in [
            ("vela3/3", "qwen3_5", Family::D3),
            ("vela3-encoder/1", "ultrabert", Family::Fast),
            ("vela3v-encoder/1", "ultravbert", Family::Vision),
        ] {
            assert_eq!(
                family(&json!({"format":format}), &json!({"model_type":model_type})).unwrap(),
                expected
            );
        }
        assert!(family(
            &json!({"format":"vela3/3"}),
            &json!({"model_type":"qwen3","name":"vela3-mini"})
        )
        .is_err());
        assert!(family(
            &json!({"format":"future"}),
            &json!({"model_type":"ultrabert"})
        )
        .is_err());
    }
    #[test]
    fn invalid_requests_fail_before_loading_weights() {
        assert!(
            system_one(Path::new("/missing"), None, &json!({"questions":[]}), 1)
                .unwrap_err()
                .contains("questions")
        );
        assert!(system_one(
            Path::new("/missing"),
            None,
            &json!({"questions":{},"images":"x"}),
            1
        )
        .unwrap_err()
        .contains("images"));
        assert!(
            system_one(Path::new("/missing"), None, &json!({"questions":{}}), 0)
                .unwrap_err()
                .contains("threads")
        );
    }

    #[test]
    fn unsupported_video_assets_and_long_context_fail_before_loading() {
        for (field, value) in [
            ("videos", json!(["clip.mp4"])),
            ("assets", json!({"clip": "clip.mp4"})),
            ("long_context", json!(true)),
        ] {
            let mut request = json!({"questions": {}});
            request[field] = value;
            assert!(
                system_one(Path::new("/missing"), None, &request, 1)
                    .unwrap_err()
                    .contains("unsupported"),
                "{field} was ignored"
            );
        }
        let empty = json!({"questions":{},"videos":[],"assets":{},"long_context":false});
        assert!(!system_one(Path::new("/missing"), None, &empty, 1)
            .unwrap_err()
            .contains("unsupported"));
    }
}
