//! Native Vela3 System One inference from the official safetensors packages.
pub(crate) mod d3;
pub(crate) mod decoder;
pub(crate) mod fast;
mod identity;
pub(crate) mod tensor;
pub(crate) mod text;
pub(crate) mod ultrabert;
pub(crate) mod vision;

use crate::compute::ComputePolicy;
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
///
/// `model_dir` holds the model weights: the d3 base snapshot for the d3
/// variants, or the self-contained package for `fast` and `fast-vision`.
/// `adapter_dir` holds the Vela3 adapter package (LoRA, heads and config)
/// that the d3 variants need; the self-contained variants carry their own
/// adapters and pass `None`.
/// Uses CPU operators regardless of other sessions' compute policies.
pub fn system_one(
    model_dir: &Path,
    adapter_dir: Option<&Path>,
    request: &Value,
    threads: usize,
) -> Result<Value, String> {
    system_one_with_policy(model_dir, adapter_dir, request, threads, ComputePolicy::Cpu)
}

/// System One with shared Vulkan projection offload and CPU postprocessing.
/// Auto falls back to CPU; Vulkan requires a usable device and reports device failure.
pub fn system_one_with_policy(
    model_dir: &Path,
    adapter_dir: Option<&Path>,
    request: &Value,
    threads: usize,
    policy: ComputePolicy,
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
    let _compute = policy.configure_cli()?;
    if policy == ComputePolicy::Vulkan {
        eprintln!("compute requested=Vulkan backend=partial mode=system-one offload=linear note=no complete device graph; remaining operators use CPU");
    }
    // The family marker (`vela3_config.json`) ships with the adapter package;
    // the self-contained variants keep it next to their weights.
    let config_dir = adapter_dir.unwrap_or(model_dir);
    let read = |file: &str| -> Result<Value, String> {
        let path = config_dir.join(file);
        serde_json::from_slice(
            &std::fs::read(&path).map_err(|e| format!("Read {}: {e}", path.display()))?,
        )
        .map_err(|e| format!("Parse {}: {e}", path.display()))
    };
    let config_path = if config_dir.join("vela3v_config.json").is_file() {
        "vela3v_config.json"
    } else {
        "vela3_config.json"
    };
    let kind = family(&read(config_path)?, &read("config.json")?)?;
    let pool = Arc::new(ComputePool::new(threads));
    let answer = match kind {
        Family::D3 => decoder::predict(
            adapter_dir
                .ok_or("d3 variants need --adapter pointing to the Vela3 adapter package")?,
            model_dir,
            request,
            pool,
        ),
        Family::Fast => fast::predict(model_dir, request, &pool),
        Family::Vision => vision::predict(model_dir, request, &pool),
    }?;
    // Legacy projection kernels repair a failed dispatch on CPU; strict requests still report it.
    #[cfg(feature = "vulkan")]
    if policy == ComputePolicy::Vulkan {
        policy.context().map_err(|e| e.to_string())?;
    }
    Ok(answer)
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
    fn explicit_compute_policy_validates_input_before_device_initialization() {
        for policy in [
            ComputePolicy::Cpu,
            ComputePolicy::Auto,
            ComputePolicy::Vulkan,
        ] {
            assert!(system_one_with_policy(
                Path::new("/missing"),
                None,
                &json!({"questions": []}),
                1,
                policy
            )
            .unwrap_err()
            .contains("questions"));
        }
    }

    #[cfg(not(feature = "vulkan"))]
    #[test]
    fn forced_vulkan_requires_the_feature_before_loading_weights() {
        assert!(system_one_with_policy(
            Path::new("/missing"),
            None,
            &json!({"questions": {}}),
            1,
            ComputePolicy::Vulkan
        )
        .unwrap_err()
        .contains("--features vulkan"));
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
