//! Test-only packed-input runner for the pinned LongCat scalar Oracle.
use rust_model_inference::models::diffusion::longcat::{LongCatKind, LongCatTransformer};
use rust_model_inference::GGUFLoader;
use std::{env, fs, path::Path};

fn read(path: &Path) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let bytes = fs::read(path)?;
    if bytes.len() % 4 != 0 {
        return Err(format!("{} is not an F32 buffer", path.display()).into());
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect())
}

#[test]
#[ignore = "requires real LongCat GGUFs and the pinned scalar Oracle"]
fn longcat_transformer_case() -> Result<(), Box<dyn std::error::Error>> {
    let kind = match env::var("RMI_LONGCAT_KIND")?.as_str() {
        "edit" => LongCatKind::Edit,
        "turbo" => LongCatKind::EditTurbo,
        kind => return Err(format!("unsupported LongCat kind: {kind}").into()),
    };
    let model_path = env::var("RMI_LONGCAT_MODEL")?;
    let input_dir = env::var("RMI_LONGCAT_INPUT")?;
    let output_path = env::var("RMI_LONGCAT_OUTPUT")?;
    let timestep: f32 = env::var("RMI_LONGCAT_TIMESTEP")?.parse()?;
    let dir = Path::new(&input_dir);
    let image = read(&dir.join("img.f32"))?;
    let text = read(&dir.join("txt.f32"))?;
    let flat = read(&dir.join("positions.f32"))?;
    if flat.len() % 3 != 0 {
        return Err("positions must have three coordinates per token".into());
    }
    let positions: Vec<[f32; 3]> = flat
        .chunks_exact(3)
        .map(|row| row.try_into().unwrap())
        .collect();
    let loader = GGUFLoader::from_file(model_path)?;
    let model = LongCatTransformer::load(&loader, kind)?;
    let output = model.forward(&image, &text, &positions, timestep)?;
    fs::write(
        output_path,
        output
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    Ok(())
}
