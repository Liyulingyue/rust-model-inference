use crate::app::cli::ZImageCliOptions;
use crate::core::tensor::TensorSource;
use crate::models::diffusion::z_image::{ZImageOptions, ZImagePipeline, ZImageRgb};
use image::ImageEncoder;
use std::ffi::OsString;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

pub fn run_z_image_cli(
    diffusion: Arc<dyn TensorSource>,
    text: Arc<dyn TensorSource>,
    vae: Arc<dyn TensorSource>,
    prompt: &str,
    options: ZImageCliOptions,
    n_threads: usize,
) -> Result<(), String> {
    let started = Instant::now();
    let ZImageCliOptions {
        steps,
        resolution,
        seed,
        out,
    } = options;
    let pipeline = ZImagePipeline::load(diffusion, text, vae, n_threads)?;
    println!(
        "Z-Image components loaded in {}ms",
        started.elapsed().as_millis()
    );
    let rgb = pipeline.generate_rgb(
        prompt,
        &ZImageOptions {
            steps,
            resolution,
            seed,
        },
    )?;
    write_png_atomically(&out, &rgb)?;
    println!(
        "Z-Image PNG saved to {} in {}ms",
        out.display(),
        started.elapsed().as_millis()
    );
    Ok(())
}

pub fn write_png_atomically(path: &Path, rgb: &ZImageRgb) -> Result<(), String> {
    let expected = usize::try_from(rgb.width)
        .ok()
        .and_then(|width| {
            usize::try_from(rgb.height)
                .ok()
                .and_then(|height| width.checked_mul(height))
        })
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or("Z-Image RGB size overflow")?;
    if rgb.bytes.len() != expected {
        return Err(format!(
            "Invalid Z-Image RGB length: expected {expected}, got {}",
            rgb.bytes.len()
        ));
    }
    let mut encoded = Vec::new();
    image::codecs::png::PngEncoder::new(&mut encoded)
        .write_image(
            &rgb.bytes,
            rgb.width,
            rgb.height,
            image::ColorType::Rgb8.into(),
        )
        .map_err(|error| format!("Encode Z-Image PNG: {error}"))?;
    write_sibling_temp_then_rename(path, &encoded)
}

fn write_sibling_temp_then_rename(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let file_name = path
        .file_name()
        .filter(|name| !name.is_empty())
        .ok_or("Z-Image output path requires a file name")?;
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));

    for counter in 0..256u16 {
        let mut temp_name = OsString::from(".");
        temp_name.push(file_name);
        temp_name.push(format!(".tmp-{}-{counter}", std::process::id()));
        let temp_path: PathBuf = parent.join(temp_name);
        let mut file = match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp_path)
        {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => {
                return Err(format!(
                    "Create Z-Image temporary output {}: {error}",
                    temp_path.display()
                ));
            }
        };
        let result = (|| -> std::io::Result<()> {
            file.write_all(bytes)?;
            file.sync_all()?;
            drop(file);
            std::fs::rename(&temp_path, path)
        })();
        if let Err(error) = result {
            let _ = std::fs::remove_file(&temp_path);
            return Err(format!(
                "Publish Z-Image PNG to {}: {error}",
                path.display()
            ));
        }
        return Ok(());
    }
    Err("Could not create a unique Z-Image temporary output".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::ZImageCliOptions;
    use crate::core::tensor::{GGMLType, MetaValue, TensorInfo};
    use crate::models::diffusion::z_image::ZImageRgb;
    use std::path::{Path, PathBuf};

    struct TextSignatureSource {
        info: TensorInfo,
    }

    impl TextSignatureSource {
        fn new() -> Self {
            Self {
                info: TensorInfo {
                    name: "model.embed_tokens.weight".into(),
                    dims: vec![2560, 151936],
                    ggml_type: GGMLType::Q8_0,
                    offset: 0,
                },
            }
        }
    }

    impl TensorSource for TextSignatureSource {
        fn metadata(&self, _key: &str) -> Option<&MetaValue> {
            None
        }

        fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
            (name == self.info.name).then_some(&self.info)
        }

        fn tensor_slice(&self, _name: &str) -> Option<&[u8]> {
            None
        }
    }

    struct MustNotBeRead;

    impl TensorSource for MustNotBeRead {
        fn metadata(&self, _key: &str) -> Option<&MetaValue> {
            panic!("later Z-Image component was read")
        }

        fn tensor_info(&self, _name: &str) -> Option<&TensorInfo> {
            panic!("later Z-Image component was read")
        }

        fn tensor_slice(&self, _name: &str) -> Option<&[u8]> {
            panic!("later Z-Image component was read")
        }
    }

    fn test_temp_dir(line: u32) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "rust-model-inference-z-image-{}-{line}",
            std::process::id()
        ));
        std::fs::create_dir(&path).unwrap();
        path
    }

    fn valid_rgb() -> ZImageRgb {
        ZImageRgb {
            width: 2,
            height: 1,
            bytes: vec![255, 0, 0, 0, 255, 0],
        }
    }

    #[test]
    fn text_signature_cannot_be_dispatched_as_a_dit() {
        let options = ZImageCliOptions {
            steps: 1,
            resolution: 16,
            seed: 7,
            out: "not-reached.png".into(),
        };
        let result = run_z_image_cli(
            Arc::new(TextSignatureSource::new()),
            Arc::new(MustNotBeRead),
            Arc::new(MustNotBeRead),
            "fox",
            options,
            1,
        );
        assert!(result.unwrap_err().contains("cap_embedder.0.weight"));
    }

    #[test]
    fn failed_png_encoding_preserves_the_existing_output() {
        let dir = test_temp_dir(line!());
        let output = dir.join("image.png");
        std::fs::write(&output, b"old").unwrap();
        let invalid = ZImageRgb {
            width: 2,
            height: 2,
            bytes: vec![0; 11],
        };

        assert!(write_png_atomically(&output, &invalid).is_err());
        assert_eq!(std::fs::read(&output).unwrap(), b"old");
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 1);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn successful_publication_is_a_decodable_png_at_the_requested_size() {
        let dir = test_temp_dir(line!());
        let output = dir.join("image.png");

        write_png_atomically(&output, &valid_rgb()).unwrap();

        let decoded = image::open(&output).unwrap().to_rgb8();
        assert_eq!(decoded.dimensions(), (2, 1));
        assert_eq!(decoded.into_raw(), vec![255, 0, 0, 0, 255, 0]);
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 1);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn publication_rejects_invalid_paths_and_rgb_overflow() {
        let dir = test_temp_dir(line!());
        let missing_parent = dir.join("missing").join("image.png");
        let overflow = ZImageRgb {
            width: u32::MAX,
            height: u32::MAX,
            bytes: Vec::new(),
        };

        assert!(write_png_atomically(Path::new(""), &valid_rgb()).is_err());
        assert!(write_png_atomically(&missing_parent, &valid_rgb()).is_err());
        assert!(write_png_atomically(&dir.join("overflow.png"), &overflow)
            .unwrap_err()
            .contains("overflow"));
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 0);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn failed_rename_removes_only_its_sibling_temp() {
        let dir = test_temp_dir(line!());
        let output = dir.join("image.png");
        std::fs::create_dir(&output).unwrap();

        assert!(write_png_atomically(&output, &valid_rgb()).is_err());
        assert!(output.is_dir());
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 1);
        std::fs::remove_dir_all(dir).unwrap();
    }
}
