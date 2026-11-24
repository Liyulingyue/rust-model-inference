//! Pixel-art CLI and startup options shared with the image-generation server.
use super::{write_output_atomically, CliOptions};
use crate::{
    core::tensor::TensorSource,
    models::tessera::{Options, Pipeline, PixelOptions, GRID},
};
use std::{
    io::{BufReader, Cursor},
    sync::Arc,
};

pub(super) fn options(cli: &CliOptions, server: bool) -> Result<Options, String> {
    for flag in &cli.supplied_flags {
        match flag.as_str() {
            "--model" | "--steps" | "--seed" | "--guidance" | "--cfg" | "--cfg-scale"
            | "--temperature" | "--width" | "--height" | "--resolution" | "--compute" | "--gpu"
            | "--threads" => {}
            "--prompt" | "--out" | "--input" | "--overwrite" if !server => {}
            "--serve" if server => {}
            other => return Err(format!("Tessera does not accept {other} in this mode")),
        }
    }
    if [cli.width, cli.height, cli.resolution]
        .into_iter()
        .flatten()
        .any(|side| side != GRID)
    {
        return Err("Tessera-small generates 16x16 sprites".into());
    }
    let guidance: Vec<_> = [cli.guidance, cli.cfg, cli.cfg_scale]
        .into_iter()
        .flatten()
        .collect();
    if guidance.len() > 1 {
        return Err("supply only one guidance option".into());
    }
    let defaults = Options::default();
    let pixel: PixelOptions = if let Some(path) = &cli.input {
        let file =
            std::fs::File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
        serde_json::from_reader(BufReader::new(file))
            .map_err(|e| format!("invalid pixel options: {e}"))?
    } else {
        PixelOptions::default()
    };
    let out = Options {
        pixel,
        steps: cli.steps.unwrap_or(defaults.steps),
        seed: u64::try_from(cli.seed.unwrap_or(0)).map_err(|_| "seed must be non-negative")?,
        guidance: guidance.first().copied().unwrap_or(defaults.guidance),
        temperature: cli.temperature.unwrap_or(defaults.temperature),
    };
    out.validate()?;
    Ok(out)
}

pub(super) fn png(img: image::RgbaImage) -> Result<Vec<u8>, String> {
    let mut bytes = Cursor::new(Vec::new());
    img.write_to(&mut bytes, image::ImageFormat::Png)
        .map_err(|e| e.to_string())?;
    Ok(bytes.into_inner())
}

pub fn run_pixel_art_cli(
    source: Arc<dyn TensorSource>,
    cli: &CliOptions,
    threads: usize,
) -> Result<(), String> {
    let opt = options(cli, false)?;
    let prompt = cli.prompt.as_deref().ok_or("Tessera requires --prompt")?;
    crate::models::tessera::validate_prompt(prompt)?;
    let path = cli.out.as_ref().ok_or("Tessera requires --out")?;
    if path.exists() && !cli.overwrite {
        return Err(format!("Output already exists: {}", path.display()));
    }
    let pipeline = Pipeline::load(source, threads, cli.compute_policy())?;
    let bytes = png(pipeline.generate(prompt, &opt)?)?;
    write_output_atomically(path, &bytes, cli.overwrite)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_conflicting_modes_sizes_and_server_only_input_sources() {
        for (extra, server) in [
            (vec!["--jev"], false),
            (vec!["--resolution", "32"], false),
            (vec!["--cfg", "1", "--guidance", "2"], false),
            (vec!["--prompt", "sprite"], true),
            (vec!["--input", "local.json"], true),
            (vec!["--temperature", "NaN"], false),
        ] {
            let args: Vec<_> = ["rmi", "--model", "tessera.gguf"]
                .into_iter()
                .chain(extra)
                .map(String::from)
                .collect();
            let cli = crate::app::parse_cli_options(&args).unwrap();
            assert!(options(&cli, server).is_err());
        }
    }
}
