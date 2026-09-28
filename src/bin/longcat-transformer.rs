//! Explicit packed-input LongCat Transformer CLI; not an image editing pipeline.
use rust_model_inference::models::diffusion::longcat::{LongCatKind, LongCatTransformer};
use rust_model_inference::GGUFLoader;
use std::{fs, path::Path};

fn read(path: &Path) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let bytes = fs::read(path)?;
    if bytes.len() % 4 != 0 {
        return Err(format!("{} is not an F32 buffer", path.display()).into());
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect())
}
fn write(path: &Path, values: &[f32]) -> std::io::Result<()> {
    fs::write(
        path,
        values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )
}
fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() == 2 && args[0] == "--fixture" {
        let dir = Path::new(&args[1]);
        fs::create_dir_all(dir)?;
        let values = |n: usize, offset: usize| {
            (0..n)
                .map(|i| (((i + offset) % 29) as i32 - 14) as f32 / 32.0)
                .collect::<Vec<_>>()
        };
        write(&dir.join("img.f32"), &values(128, 3))?;
        write(&dir.join("txt.f32"), &values(7168, 7))?;
        write(
            &dir.join("positions.f32"),
            &[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0],
        )?;
        return Ok(());
    }
    if args.len() != 5 {
        return Err("usage: longcat-transformer edit|turbo MODEL INPUT_DIR OUTPUT.f32 TIMESTEP\n       longcat-transformer --fixture INPUT_DIR\nRun with RMI_SCALAR=1; input: img.f32, txt.f32, positions.f32 (little-endian, row-major).".into());
    }
    let kind = match args[0].as_str() {
        "edit" => LongCatKind::Edit,
        "turbo" => LongCatKind::EditTurbo,
        _ => return Err("model kind must be edit or turbo".into()),
    };
    let dir = Path::new(&args[2]);
    let image = read(&dir.join("img.f32"))?;
    let text = read(&dir.join("txt.f32"))?;
    let flat = read(&dir.join("positions.f32"))?;
    if flat.len() % 3 != 0 {
        return Err("positions must have three coordinates per token".into());
    }
    let positions: Vec<[f32; 3]> = flat
        .chunks_exact(3)
        .map(|v| v.try_into().unwrap())
        .collect();
    let timestep: f32 = args[4].parse()?;
    let loader = GGUFLoader::from_file(&args[1])?;
    let model = LongCatTransformer::load(&loader, kind)?;
    let output = model.forward(&image, &text, &positions, timestep)?;
    write(Path::new(&args[3]), &output)?;
    println!(
        "LongCat {kind:?}: {} image tokens, {} text tokens, {} output F32 values",
        image.len() / 64,
        text.len() / 3584,
        output.len()
    );
    Ok(())
}
fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
