use rust_model_inference::app::run_longcat_image_edit;
use rust_model_inference::models::diffusion::longcat::LongCatKind;
use std::path::PathBuf;

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let mut model = None;
    let mut components = None;
    let mut input = None;
    let mut output = None;
    let mut instruction = None;
    let mut kind = None;
    let mut side = 1024;
    let mut steps = None;
    let mut guidance = None;
    let mut seed = 42;
    let mut threads = 1;
    while let Some(flag) = args.next() {
        let value = args
            .next()
            .ok_or_else(|| format!("Missing value for {flag}"))?;
        match flag.as_str() {
            "--model" => model = Some(PathBuf::from(value)),
            "--components" => components = Some(PathBuf::from(value)),
            "--input" => input = Some(PathBuf::from(value)),
            "--output" => output = Some(PathBuf::from(value)),
            "--instruction" => instruction = Some(value),
            "--kind" => {
                kind = Some(match value.as_str() {
                    "edit" => LongCatKind::Edit,
                    "turbo" => LongCatKind::EditTurbo,
                    _ => return Err("--kind must be edit or turbo".into()),
                })
            }
            "--side" => side = value.parse().map_err(|_| "Invalid --side")?,
            "--steps" => steps = Some(value.parse().map_err(|_| "Invalid --steps")?),
            "--guidance" => guidance = Some(value.parse().map_err(|_| "Invalid --guidance")?),
            "--seed" => seed = value.parse().map_err(|_| "Invalid --seed")?,
            "--threads" => threads = value.parse().map_err(|_| "Invalid --threads")?,
            _ => return Err(format!("Unknown option {flag}")),
        }
    }
    let kind = kind.ok_or("Missing --kind edit|turbo")?;
    run_longcat_image_edit(
        &model.ok_or("Missing --model")?,
        &components.ok_or("Missing --components")?,
        &input.ok_or("Missing --input")?,
        &output.ok_or("Missing --output")?,
        &instruction.ok_or("Missing --instruction")?,
        kind,
        side,
        steps.unwrap_or(if kind == LongCatKind::Edit { 50 } else { 8 }),
        guidance.unwrap_or(if kind == LongCatKind::Edit { 4.5 } else { 1.0 }),
        seed,
        threads,
    )
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}\nUsage: longcat-image-edit --kind edit|turbo --model FILE.gguf --components DIR --input INPUT.png --output OUTPUT.png --instruction TEXT [--side 1024] [--steps N] [--guidance F32] [--seed U64] [--threads N]");
        std::process::exit(2);
    }
}
