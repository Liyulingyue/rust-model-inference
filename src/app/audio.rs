//! Audio-to-audio CLI orchestration.
use super::CliOptions;

pub fn run_enhancement(options: &CliOptions) -> Result<(), String> {
    let model =
        crate::models::nemotron_reuse::Model::load(&options.model, options.compute_policy())?;
    let input = options
        .audio
        .as_deref()
        .ok_or("--enhance requires --audio")?;
    let output = options.out.as_deref().ok_or("--enhance requires --out")?;
    let bytes = std::fs::read(input).map_err(|e| format!("{}: {e}", input.display()))?;
    let enhanced = model.enhance_wav(&bytes)?;
    super::write_output_atomically(output, &enhanced, options.overwrite)
}
