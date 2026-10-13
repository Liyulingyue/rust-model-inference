//! BGE visual retrieval graphs; image and text branches fuse before L2.
mod clip;
mod llava;
mod screenshot;
mod visualized;

use crate::core::tensor::{load_f32_tensor, GGMLType, TensorSource};
use crate::core::thread_pool::ComputePool;
use crate::ops::kernel::{PreparedRows, QuantizedTensor, Weight};
use serde::de::DeserializeOwned;
use std::path::Path;

pub fn is_visual(source: &dyn TensorSource) -> bool {
    crate::app::arch_of(source) == "bge-vl-clip"
        || marked(source, "bge.visualized")
        || marked(source, "bge.llava_next")
        || marked(source, "bge.screenshot")
}

fn marked(source: &dyn TensorSource, key: &str) -> bool {
    matches!(
        source.metadata(key),
        Some(crate::core::tensor::MetaValue::Bool(true))
    )
}

pub fn compute_embedding(
    source: &dyn TensorSource,
    prompt: &str,
    image: Option<&Path>,
    threads: usize,
) -> Result<Vec<f32>, String> {
    let available = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let pool = ComputePool::new(crate::app::resolve_thread_count(threads, available));
    let arch = crate::app::arch_of(source);
    if marked(source, "bge.visualized") {
        if arch != "bert" {
            return Err("Visualized BGE requires bert architecture".into());
        }
        return visualized::compute(source, prompt, image, &pool);
    }
    if marked(source, "bge.llava_next") {
        if arch != "llama" {
            return Err("BGE LLaVA-NeXT requires llama architecture".into());
        }
        return llava::compute(source, prompt, image, &pool);
    }
    if marked(source, "bge.screenshot") {
        return screenshot::compute(source, prompt, image, &pool);
    }
    match crate::app::arch_of(source).as_str() {
        "bge-vl-clip" => clip::compute(source, prompt, image, &pool),
        other => Err(format!("unsupported BGE visual architecture {other:?}")),
    }
}

fn json_metadata<T: DeserializeOwned>(source: &dyn TensorSource, key: &str) -> Result<T, String> {
    let raw = source
        .metadata(key)
        .and_then(|v| v.to_string_val())
        .ok_or_else(|| format!("missing metadata {key}"))?;
    serde_json::from_str(raw).map_err(|e| format!("{key}: {e}"))
}

struct Linear<'a> {
    weight: Weight<'a>,
    bias: Option<Vec<f32>>,
}
impl<'a> Linear<'a> {
    fn load(
        source: &'a dyn TensorSource,
        prefix: &str,
        n_in: usize,
        n_out: usize,
        bias: bool,
    ) -> Result<Self, String> {
        Self::load_shape(
            source,
            &format!("{prefix}.weight"),
            &[n_in as u64, n_out as u64],
            n_in,
            n_out,
            if bias {
                Some(format!("{prefix}.bias"))
            } else {
                None
            },
        )
    }
    fn load_shape(
        source: &'a dyn TensorSource,
        name: &str,
        dims: &[u64],
        n_in: usize,
        n_out: usize,
        bias: Option<String>,
    ) -> Result<Self, String> {
        let info = source
            .tensor_info(name)
            .ok_or_else(|| format!("missing tensor {name}"))?;
        if info.dims != dims
            || !matches!(
                info.ggml_type,
                GGMLType::F32 | GGMLType::F16 | GGMLType::BF16 | GGMLType::Q8_0
            )
        {
            return Err(format!(
                "invalid BGE tensor {name}: {:?} {:?}",
                info.dims, info.ggml_type
            ));
        }
        let raw = source
            .tensor_slice(name)
            .ok_or_else(|| format!("missing tensor data {name}"))?;
        if Some(raw.len() as u64) != info.checked_nbytes() {
            return Err(format!("invalid tensor byte length {name}"));
        }
        let weight = Weight::from_quantized(QuantizedTensor::from_bytes(
            raw,
            info.ggml_type,
            n_in,
            n_out,
        ));
        let bias = bias
            .map(|name| load_f32_tensor(source, &name, &[n_out as u64]))
            .transpose()?;
        Ok(Self { weight, bias })
    }
    fn apply(&self, input: &[f32], pool: &ComputePool) -> Result<Vec<f32>, String> {
        let n_in = self.weight.n_in;
        if input.is_empty() || !input.len().is_multiple_of(n_in) {
            return Err("BGE linear input shape mismatch".into());
        }
        let rows = input.len() / n_in;
        let mut output = vec![0.0; rows * self.weight.n_out];
        let mut prepared = PreparedRows::new(rows, n_in);
        prepared.prepare(
            input,
            rows,
            n_in,
            self.weight.needs_q8_0_activation(),
            self.weight.uses_q8_k(),
        )?;
        prepared.matmul(&self.weight, input, &mut output, pool)?;
        if let Some(bias) = &self.bias {
            for row in output.chunks_exact_mut(bias.len()) {
                for (x, b) in row.iter_mut().zip(bias) {
                    *x += *b;
                }
            }
        }
        Ok(output)
    }
}
struct Norm {
    weight: Vec<f32>,
    bias: Vec<f32>,
    eps: f32,
}
impl Norm {
    fn load(source: &dyn TensorSource, prefix: &str, h: usize, eps: f32) -> Result<Self, String> {
        Ok(Self {
            weight: load_f32_tensor(source, &format!("{prefix}.weight"), &[h as u64])?,
            bias: load_f32_tensor(source, &format!("{prefix}.bias"), &[h as u64])?,
            eps,
        })
    }
    fn apply(&self, input: &[f32]) -> Vec<f32> {
        let mut result = vec![0.0; input.len()];
        for (x, y) in input
            .chunks_exact(self.weight.len())
            .zip(result.chunks_exact_mut(self.weight.len()))
        {
            crate::ops::layer_norm(x, &self.weight, &self.bias, self.eps, y);
        }
        result
    }
}

fn preprocess(
    source: &dyn TensorSource,
    path: &Path,
    side: usize,
    stretch: bool,
) -> Result<Vec<f32>, String> {
    let cfg: serde_json::Value = json_metadata(source, "bge.preprocessor")?;
    let array = |key: &str| -> Result<[f32; 3], String> {
        let a = cfg
            .get(key)
            .and_then(|v| v.as_array())
            .ok_or_else(|| format!("missing image {key}"))?;
        if a.len() != 3 {
            return Err(format!("invalid image {key}"));
        }
        let mut result = [0.; 3];
        for (x, v) in result.iter_mut().zip(a) {
            *x = v.as_f64().ok_or("invalid image normalization value")? as f32;
        }
        Ok(result)
    };
    let mean = array("image_mean")?;
    let std = array("image_std")?;
    if std.iter().any(|x| !x.is_finite() || *x <= 0.) || mean.iter().any(|x| !x.is_finite()) {
        return Err("invalid image normalization".into());
    }
    let image = crate::app::media::decode_image(path)?;
    let (width, height) = if stretch {
        (side, side)
    } else {
        let scale = side as f64 / image.width().min(image.height()) as f64;
        (
            (image.width() as f64 * scale).floor().max(side as f64) as usize,
            (image.height() as f64 * scale).floor().max(side as f64) as usize,
        )
    };
    let resized = crate::app::media::normalize_resized_image(&image, width, height, &mean, &std)?;
    let left = (width - side) / 2;
    let top = (height - side) / 2;
    let mut cropped = Vec::with_capacity(side * side * 3);
    for y in top..top + side {
        cropped.extend_from_slice(&resized[(y * width + left) * 3..(y * width + left + side) * 3]);
    }
    Ok(cropped)
}

fn trace(name: &str, values: &[f32]) -> Result<(), String> {
    #[cfg(feature = "parity-trace")]
    if crate::parity_trace::enabled(name) {
        crate::parity_trace::checkpoint(name, None, &[values.len()], values)
            .map_err(|e| e.to_string())?;
    }
    let _ = (name, values);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::tensor::{MetaValue, TensorInfo};
    use std::collections::HashMap;

    struct Source(HashMap<String, MetaValue>);
    impl TensorSource for Source {
        fn metadata(&self, key: &str) -> Option<&MetaValue> {
            self.0.get(key)
        }
        fn tensor_info(&self, _: &str) -> Option<&TensorInfo> {
            None
        }
        fn tensor_slice(&self, _: &str) -> Option<&[u8]> {
            None
        }
    }

    #[test]
    fn visual_markers_require_true_and_the_matching_architecture() {
        for (key, expected) in [
            ("bge.visualized", "bert"),
            ("bge.llava_next", "llama"),
            ("bge.screenshot", "qwen2vl"),
        ] {
            let mut source = Source(HashMap::from([
                (
                    "general.architecture".into(),
                    MetaValue::String("qwen3".into()),
                ),
                (key.into(), MetaValue::Bool(false)),
            ]));
            assert!(!is_visual(&source));
            source
                .0
                .insert(key.into(), MetaValue::String("true".into()));
            assert!(!is_visual(&source));
            source.0.insert(key.into(), MetaValue::Bool(true));
            assert!(is_visual(&source));
            let error = compute_embedding(&source, "hello", None, 1).unwrap_err();
            assert!(error.contains(expected), "{error}");
        }
        let source = Source(HashMap::from([
            ("general.architecture".into(), MetaValue::String("bert".into())),
            ("bge.visualized".into(), MetaValue::Bool(true)),
            ("bge.config".into(), MetaValue::String(
                r#"{"hidden_size":32,"intermediate_size":0,"num_hidden_layers":1,"num_attention_heads":4,"image_size":4,"patch_size":2}"#.into(),
            )),
        ]));
        let error =
            compute_embedding(&source, "hello", Some(Path::new("missing.png")), 1).unwrap_err();
        assert!(error.contains("invalid EVA"), "{error}");
    }
}
