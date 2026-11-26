//! CLI entry for the OCRia page reader: read `--image`, run the model, write
//! the page as JSON.
//!
//! This is the app layer's job -- file IO and output -- and the model itself
//! stays in [`crate::models::ocria`]. The shape (one page image in, one JSON
//! document out) serves this one model, which is why it is not a generic
//! capability module.
//!
//! `--model <ocria.gguf> --image page.png [--out lines.json]` reads one page
//! image and writes the recognised lines as JSON. The model is selected by its
//! GGUF architecture, the same way the other capability backends are, so the
//! flag surface stays the one `--image` already has.
//!
//! The input is a page image: this repository has no PDF renderer, and the
//! upstream release's PDF text layer and LayoutNet stages are the second
//! delivery layer (`tools/oracle/ocria/layout-plan.md`).

use crate::app::CliOptions;
use crate::format::ggufrs::{open_model_source, ComponentRole};
use crate::models::ocria::Model;

/// Reads the page `--image` names and writes the lines as JSON.
pub fn run_cli(options: &CliOptions, threads: usize) -> Result<(), String> {
    if options.mmproj.is_some() {
        return Err("OCRia needs no --mmproj: the one GGUF holds every component".into());
    }
    let path = options
        .image
        .as_ref()
        .ok_or("OCRia needs --image <page.png>")?;
    let bytes = std::fs::read(path).map_err(|e| format!("Read page {}: {e}", path.display()))?;
    let decoded = crate::app::media::decode_image_bytes(&bytes)
        .map_err(|e| format!("Decode page {}: {e}", path.display()))?;
    let rgb = decoded.to_rgb8();
    let (width, height) = (rgb.width(), rgb.height());
    let source = open_model_source(&options.model, ComponentRole::Llm)
        .map_err(|e| format!("Open {}: {e}", options.model.display()))?;
    let model = Model::from_source(source.as_ref(), options.compute_policy(), threads)?;
    let page = model
        .run(rgb.as_raw(), width, height)
        .map_err(|e| e.to_string())?;
    let mut json = serde_json::to_vec_pretty(&page).map_err(|e| format!("Encode result: {e}"))?;
    json.push(b'\n');
    match options.out.as_ref() {
        Some(path) => crate::app::write_output_atomically(path, &json, options.overwrite)?,
        None => print!("{}", String::from_utf8_lossy(&json)),
    }
    Ok(())
}
