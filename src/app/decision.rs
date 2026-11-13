//! Native decision entry points.
//!
//! Two shapes reach the same engine, and the difference is the protocol
//! rather than the model. The native request carries one state and several
//! typed questions, and the JEV adapter ([`jev`]) drives the same engine
//! through the shared `--jev` and `/v1/jev/*` routes, so each backend answers a
//! scorer request the way every other model does. The model semantics live
//! in [`crate::models::decision`].

pub mod jev;

use std::io::Read;
use std::sync::Arc;

use base64::Engine as _;
use serde_json::{json, Value};

use super::CliOptions;
use crate::core::thread_pool::ComputePool;
use crate::format::ggufrs::{open_model_source, ComponentRole};
use crate::models::decision::Engine;
use crate::TensorSource;

pub(crate) fn validate_options(options: &CliOptions) -> Result<(), String> {
    let mode = if options.decision_request.is_some() {
        "--decision-request"
    } else {
        "native decisions"
    };
    if options.model.as_os_str().is_empty() {
        return Err(format!("{mode} requires --model"));
    }
    for flag in &options.supplied_flags {
        let allowed = matches!(
            flag.as_str(),
            "--model"
                | "--threads"
                | "--compute"
                | "--gpu"
                | "--max-context"
                | "--decision-thinker"
        ) || (options.decision_request.is_some() && flag == "--decision-request")
            || (options.decision_request.is_none() && options.serve && flag == "--serve")
            || (options.decision_request.is_none()
                && options.jev
                && matches!(
                    flag.as_str(),
                    "--jev"
                        | "--jev-context"
                        | "--jev-question"
                        | "--jev-option"
                        | "--jev-positive"
                        | "--jev-output"
                        | "--jev-multi"
                        | "--jev-block"
                        | "--image"
                ));
        if !allowed {
            return Err(format!(
                "{mode} cannot be combined with {flag}; put native task inputs in the JSON request"
            ));
        }
    }
    if options.effective_max_context() == 0 {
        return Err("decision --max-context must be greater than zero".into());
    }
    Ok(())
}

pub(crate) fn from_source(
    options: &CliOptions,
    source: Arc<dyn TensorSource>,
    n_threads: usize,
) -> Result<Engine, String> {
    validate_options(options)?;
    let thinker = options
        .decision_thinker
        .as_deref()
        .map(|path| {
            open_model_source(path, ComponentRole::Llm)
                .map(Arc::from)
                .map_err(|error| format!("decision thinker {}: {error}", path.display()))
        })
        .transpose()?;
    Engine::from_source(
        &options.model,
        source,
        thinker,
        Arc::new(ComputePool::new(n_threads)),
        n_threads,
        options.effective_max_context(),
        options.compute_policy(),
    )
}

pub(crate) fn from_path(options: &CliOptions, n_threads: usize) -> Result<Engine, String> {
    validate_options(options)?;
    Engine::from_path(
        &options.model,
        options.decision_thinker.as_deref(),
        n_threads,
        options.effective_max_context(),
        options.compute_policy(),
    )
}

fn normalize_cli_images(request: &mut Value) -> Result<(), String> {
    if let Some(images) = request.get_mut("images").and_then(Value::as_array_mut) {
        for source in images {
            let source_text = source.as_str().ok_or("images must contain strings")?.trim();
            if source_text.len() > 48 * 1024 * 1024 {
                return Err("THX-02 image source exceeds the base64 size limit".into());
            }
            let local_path = source_text.strip_prefix("file://").unwrap_or(source_text);
            if !source_text.starts_with("data:")
                && !source_text.starts_with("http://")
                && !source_text.starts_with("https://")
            {
                if let Ok(metadata) = std::fs::metadata(local_path) {
                    if metadata.len() > 32 * 1024 * 1024 {
                        return Err("THX-02 images must not exceed 32 MiB".into());
                    }
                }
            }
            let image = super::server::api::image_input::decode_image_source(source_text, false)?;
            if image.bytes.len() > 32 * 1024 * 1024 {
                return Err("THX-02 images must not exceed 32 MiB".into());
            }
            super::media::decode_image_bytes(&image.bytes)?;
            let format = image::guess_format(&image.bytes).map_err(|error| error.to_string())?;
            *source = json!(format!(
                "data:{};base64,{}",
                format.to_mime_type(),
                base64::engine::general_purpose::STANDARD.encode(&image.bytes)
            ));
        }
    }
    Ok(())
}

pub fn run_cli(options: &CliOptions, n_threads: usize) -> Result<(), String> {
    validate_options(options)?;
    let path = options
        .decision_request
        .as_deref()
        .ok_or("missing --decision-request")?;
    let text = if path == std::path::Path::new("-") {
        let mut text = String::new();
        std::io::stdin()
            .read_to_string(&mut text)
            .map_err(|error| format!("decision request stdin: {error}"))?;
        text
    } else {
        std::fs::read_to_string(path)
            .map_err(|error| format!("decision request {}: {error}", path.display()))?
    };
    let mut request: Value =
        serde_json::from_str(&text).map_err(|error| format!("decision request JSON: {error}"))?;
    let source: Option<Arc<dyn TensorSource>> =
        if crate::models::decision::is_model_directory(&options.model) {
            None
        } else {
            Some(Arc::from(
                open_model_source(&options.model, ComponentRole::Llm).map_err(|error| {
                    format!("decision model {}: {error}", options.model.display())
                })?,
            ))
        };
    crate::models::decision::validate_request(&options.model, source.as_deref(), &request)?;
    normalize_cli_images(&mut request)?;
    let engine = if let Some(source) = source {
        from_source(options, source, n_threads)?
    } else {
        from_path(options, n_threads)?
    };
    let result = engine.decide(&request)?;
    println!(
        "{}",
        serde_json::to_string(&result).map_err(|error| error.to_string())?
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::{JevGroupInput, JevGroupedQuestionInput, JevQuestionInput};

    #[test]
    fn decision_cli_images_validate_files_data_urls_and_refuse_remote_urls() {
        let directory =
            std::env::temp_dir().join(format!("thx02-cli-image-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let mut png = std::io::Cursor::new(Vec::new());
        image::DynamicImage::new_rgb8(2, 2)
            .write_to(&mut png, image::ImageFormat::Png)
            .unwrap();
        let path = directory.join("valid.png");
        std::fs::write(&path, png.get_ref()).unwrap();
        let data_url = format!(
            "data:image/png;base64,{}",
            base64::engine::general_purpose::STANDARD.encode(png.get_ref())
        );
        let mut request = json!({"images":[path.to_str().unwrap(), data_url]});
        normalize_cli_images(&mut request).unwrap();
        assert_eq!(request["images"], json!([data_url, data_url]));

        let invalid = directory.join("invalid.png");
        std::fs::write(&invalid, b"not an image").unwrap();
        let truncated = directory.join("truncated.png");
        std::fs::write(&truncated, &png.get_ref()[..16]).unwrap();
        let large = directory.join("large.png");
        std::fs::File::create(&large)
            .unwrap()
            .set_len(32 * 1024 * 1024 + 1)
            .unwrap();
        assert!(
            normalize_cli_images(&mut json!({"images":[large.to_str().unwrap()]}))
                .unwrap_err()
                .contains("32 MiB")
        );
        for source in [
            directory.join("missing.png").to_string_lossy().into_owned(),
            invalid.to_string_lossy().into_owned(),
            truncated.to_string_lossy().into_owned(),
            "http://127.0.0.1/image.png".into(),
            "https://example.com/image.png".into(),
        ] {
            assert!(normalize_cli_images(&mut json!({"images":[source]})).is_err());
        }
        std::fs::remove_dir_all(directory).unwrap();
    }
}
