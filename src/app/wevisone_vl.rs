//! WeVisOne-VL CLI entry point.
//!
//! The model itself lives in [`crate::models::wevisone_vl`]: the released
//! pixel decoder, the task prompts and the prediction pipeline. This
//! module only wires the command line to that pipeline.
use crate::models::wevisone_vl::{predict, PredictRequest, Prediction, PredictionOptions};
use crate::core::thread_pool::ComputePool;
use crate::core::tensor::TensorSource;
use crate::core::tokenizer::BPETokenizer;
use std::sync::Arc;

pub fn run_cli(options: &crate::app::CliOptions, threads: usize) -> Result<(), String> {
    use crate::format::ggufrs::{open_model_source, ComponentRole};
    let request_path = options
        .wevisone_request
        .as_ref()
        .ok_or("missing WeVisOne request")?;
    if std::fs::metadata(request_path)
        .map_err(|e| e.to_string())?
        .len()
        > 1024 * 1024
    {
        return Err("WeVisOne request exceeds 1MiB".into());
    }
    let bytes = std::fs::read(request_path).map_err(|e| e.to_string())?;
    let request: PredictRequest = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
    request.validate()?;
    let image_ref =
        crate::app::server::api::image_input::decode_image_source(&request.image, false)?;
    let image = crate::app::server::api::image_input::decode_image_bytes(&image_ref.bytes)?;
    let source: Arc<dyn TensorSource> = Arc::from(
        open_model_source(&options.model, ComponentRole::Llm).map_err(|e| e.to_string())?,
    );
    let vision = open_model_source(
        options
            .mmproj
            .as_ref()
            .ok_or("WeVisOne requires --mmproj")?,
        ComponentRole::Mmproj,
    )
    .map_err(|e| e.to_string())?;
    let decoder = options
        .wevisone_decoder
        .as_ref()
        .map(|path| open_model_source(path, ComponentRole::Llm))
        .transpose()
        .map_err(|e| e.to_string())?;
    let tokenizer = Arc::new(BPETokenizer::from_gguf_metadata(|k| {
        source.metadata(k).cloned()
    })?);
    let result = predict(
        source,
        vision.as_ref(),
        decoder.as_deref(),
        Arc::new(ComputePool::new(threads)),
        tokenizer,
        PredictionOptions::from_cli(options)?,
        &request,
        &image,
    )?;
    let json = serde_json::to_vec_pretty(&result).map_err(|e| e.to_string())?;
    if let Some(path) = &options.output {
        crate::app::write_output_atomically(path, &json, options.overwrite)?;
    } else {
        println!("{}", String::from_utf8_lossy(&json));
    }
    Ok(())
}
