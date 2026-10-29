//! Validated safetensors operations shared by the Vela3 backends.
use crate::core::tensor::{GGMLType, TensorInfo, TensorSource};
use crate::core::thread_pool::ComputePool;
use crate::format::safetensors::SafetensorSource;
use crate::ops::kernel::{PreparedRows, QuantizedTensor, Weight};
use serde_json::Value;
use std::collections::BTreeSet;
use std::io::Read;
use std::path::Path;

pub(crate) fn open_weights(dir: &Path) -> Result<SafetensorSource, String> {
    let single = dir.join("model.safetensors");
    if single.is_file() {
        return SafetensorSource::open(&[single]);
    }
    let index_path = dir.join("model.safetensors.index.json");
    let index: Value = serde_json::from_slice(
        &std::fs::read(&index_path).map_err(|e| format!("Read {}: {e}", index_path.display()))?,
    )
    .map_err(|e| format!("Parse {}: {e}", index_path.display()))?;
    let map = index["weight_map"]
        .as_object()
        .filter(|m| !m.is_empty())
        .ok_or("missing safetensors weight_map")?;
    let mut files = BTreeSet::new();
    for filename in map.values() {
        let file = filename.as_str().ok_or("invalid safetensors shard name")?;
        if Path::new(file)
            .components()
            .any(|c| !matches!(c, std::path::Component::Normal(_)))
        {
            return Err("safetensors shard must be relative to model directory".into());
        }
        files.insert(dir.join(file));
    }
    let source = SafetensorSource::open(&files.into_iter().collect::<Vec<_>>())?;
    for name in map.keys() {
        if source.tensor_info(name).is_none() {
            return Err(format!(
                "safetensors index tensor missing from shards: {name}"
            ));
        }
    }
    Ok(source)
}

pub(crate) fn safetensor_metadata(path: &Path) -> Result<Value, String> {
    let mut file =
        std::fs::File::open(path).map_err(|e| format!("Open {}: {e}", path.display()))?;
    let mut len = [0; 8];
    file.read_exact(&mut len)
        .map_err(|e| format!("Read safetensors header: {e}"))?;
    let len =
        usize::try_from(u64::from_le_bytes(len)).map_err(|_| "safetensors header overflow")?;
    if len > 100 * 1024 * 1024
        || len as u64
            > file
                .metadata()
                .map_err(|e| e.to_string())?
                .len()
                .saturating_sub(8)
    {
        return Err("invalid safetensors header length".into());
    }
    let mut bytes = vec![0; len];
    file.read_exact(&mut bytes).map_err(|e| e.to_string())?;
    let header: Value = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
    Ok(header.get("__metadata__").cloned().unwrap_or(Value::Null))
}

fn checked<'a>(
    source: &'a dyn TensorSource,
    name: &str,
) -> Result<(&'a TensorInfo, &'a [u8]), String> {
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("missing tensor: {name}"))?;
    if !matches!(
        info.ggml_type,
        GGMLType::F32 | GGMLType::F16 | GGMLType::BF16
    ) {
        return Err(format!(
            "unsupported tensor dtype for {name}: {:?}",
            info.ggml_type
        ));
    }
    let bytes = source
        .tensor_slice(name)
        .ok_or_else(|| format!("missing tensor bytes: {name}"))?;
    if info.checked_nbytes() != Some(bytes.len() as u64) {
        return Err(format!("invalid tensor byte count: {name}"));
    }
    Ok((info, bytes))
}

pub(crate) fn values(source: &dyn TensorSource, name: &str) -> Result<Vec<f32>, String> {
    let (info, bytes) = checked(source, name)?;
    let out: Vec<f32> = match info.ggml_type {
        GGMLType::F32 => bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect(),
        GGMLType::F16 => bytes
            .chunks_exact(2)
            .map(|b| crate::ops::f16_to_f32(u16::from_le_bytes(b.try_into().unwrap())))
            .collect(),
        GGMLType::BF16 => bytes
            .chunks_exact(2)
            .map(|b| f32::from_bits((u16::from_le_bytes(b.try_into().unwrap()) as u32) << 16))
            .collect(),
        _ => unreachable!(),
    };
    if out.iter().any(|x| !x.is_finite()) {
        return Err(format!("non-finite tensor: {name}"));
    }
    Ok(out)
}

pub(crate) fn shape(source: &dyn TensorSource, name: &str) -> Result<Vec<usize>, String> {
    let (info, _) = checked(source, name)?;
    info.dims
        .iter()
        .rev()
        .map(|&n| usize::try_from(n).map_err(|_| format!("oversized tensor dimension: {name}")))
        .collect()
}

pub(crate) fn linear(
    source: &dyn TensorSource,
    name: &str,
    input: &[f32],
    rows: usize,
    pool: &ComputePool,
) -> Result<Vec<f32>, String> {
    let dims = shape(source, name)?;
    if dims.len() != 2 || dims[0] == 0 || dims[1] == 0 {
        return Err(format!("linear tensor {name} must be a nonempty matrix"));
    }
    let (out, width) = (dims[0], dims[1]);
    if rows.checked_mul(width) != Some(input.len()) || input.iter().any(|x| !x.is_finite()) {
        return Err(format!("invalid linear input shape or values: {name}"));
    }
    let count = rows
        .checked_mul(out)
        .ok_or("linear output shape overflow")?;
    let (info, bytes) = checked(source, name)?;
    let weight = Weight::from_quantized(QuantizedTensor::from_bytes(
        bytes,
        info.ggml_type,
        width,
        out,
    ));
    let mut scratch = PreparedRows::new(rows.max(1), width);
    let mut output = vec![0.; count];
    if rows > 0 {
        scratch.prepare(input, rows, width, false, false)?;
        scratch.matmul(&weight, input, &mut output, pool)?;
    }
    if let Some(stem) = name.strip_suffix(".weight") {
        let bias_name = format!("{stem}.bias");
        if source.tensor_info(&bias_name).is_some() {
            if shape(source, &bias_name)? != [out] {
                return Err(format!("invalid bias shape: {bias_name}"));
            }
            let bias = values(source, &bias_name)?;
            for row in output.chunks_exact_mut(out) {
                for (x, b) in row.iter_mut().zip(&bias) {
                    *x += b;
                }
            }
        }
    }
    Ok(output)
}

pub(crate) fn embed(
    source: &dyn TensorSource,
    name: &str,
    ids: &[u32],
) -> Result<Vec<f32>, String> {
    let dims = shape(source, name)?;
    if dims.len() != 2 || dims[0] == 0 || dims[1] == 0 {
        return Err(format!("embedding tensor {name} must be a nonempty matrix"));
    }
    let (info, bytes) = checked(source, name)?;
    let (vocab, width) = (dims[0], dims[1]);
    let count = ids
        .len()
        .checked_mul(width)
        .ok_or("embedding shape overflow")?;
    let element_bytes = if info.ggml_type == GGMLType::F32 {
        4
    } else {
        2
    };
    let mut output = Vec::with_capacity(count);
    for &id in ids {
        let row = id as usize;
        if row >= vocab {
            return Err(format!("token {id} exceeds {name} vocabulary {vocab}"));
        }
        let data = &bytes[row * width * element_bytes..(row + 1) * width * element_bytes];
        match info.ggml_type {
            GGMLType::F32 => output.extend(
                data.chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap())),
            ),
            GGMLType::F16 => output.extend(
                data.chunks_exact(2)
                    .map(|b| crate::ops::f16_to_f32(u16::from_le_bytes(b.try_into().unwrap()))),
            ),
            GGMLType::BF16 => {
                output.extend(data.chunks_exact(2).map(|b| {
                    f32::from_bits((u16::from_le_bytes(b.try_into().unwrap()) as u32) << 16)
                }))
            }
            _ => unreachable!(),
        }
    }
    if output.iter().any(|x| !x.is_finite()) {
        return Err(format!("non-finite embedding: {name}"));
    }
    Ok(output)
}

pub(crate) fn layer_norm(
    x: &[f32],
    weight: &[f32],
    bias: &[f32],
    eps: f32,
) -> Result<Vec<f32>, String> {
    norm_shape(x, weight, eps)?;
    if bias.len() != weight.len() || bias.iter().any(|v| !v.is_finite()) {
        return Err("layer norm bias shape mismatch".into());
    }
    let mut out = Vec::with_capacity(x.len());
    for row in x.chunks_exact(weight.len()) {
        let mean = row.iter().sum::<f32>() / weight.len() as f32;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / weight.len() as f32;
        let inv = (var + eps).sqrt().recip();
        out.extend(
            row.iter()
                .zip(weight)
                .zip(bias)
                .map(|((&v, &w), &b)| (v - mean) * inv * w + b),
        );
    }
    Ok(out)
}
pub(crate) fn rms_norm(x: &[f32], weight: &[f32], eps: f32) -> Result<Vec<f32>, String> {
    norm_shape(x, weight, eps)?;
    let mut out = Vec::with_capacity(x.len());
    for row in x.chunks_exact(weight.len()) {
        let inv = (row.iter().map(|v| v * v).sum::<f32>() / weight.len() as f32 + eps)
            .sqrt()
            .recip();
        out.extend(row.iter().zip(weight).map(|(&v, &w)| v * inv * w));
    }
    Ok(out)
}
fn norm_shape(x: &[f32], weight: &[f32], eps: f32) -> Result<(), String> {
    if weight.is_empty()
        || !x.len().is_multiple_of(weight.len())
        || !eps.is_finite()
        || eps <= 0.
        || x.iter().chain(weight).any(|v| !v.is_finite())
    {
        return Err("invalid normalization shape, epsilon or values".into());
    }
    Ok(())
}

/// The adapter stores fully tuned projection weights under `.base.weight`.
pub(crate) struct OverlaySource<'a> {
    base: &'a dyn TensorSource,
    adapter: &'a dyn TensorSource,
}
impl<'a> OverlaySource<'a> {
    pub(crate) fn new(base: &'a dyn TensorSource, adapter: &'a dyn TensorSource) -> Self {
        Self { base, adapter }
    }
    fn adapter_name(&self, name: &str) -> Option<String> {
        if self.adapter.tensor_info(name).is_some() {
            return Some(name.into());
        }
        for suffix in [".weight", ".bias"] {
            if let Some(stem) = name.strip_suffix(suffix) {
                let wrapped = format!("{stem}.base{suffix}");
                if self.adapter.tensor_info(&wrapped).is_some() {
                    return Some(wrapped);
                }
            }
        }
        None
    }
}
impl TensorSource for OverlaySource<'_> {
    fn metadata(&self, key: &str) -> Option<&crate::core::tensor::MetaValue> {
        self.adapter
            .metadata(key)
            .or_else(|| self.base.metadata(key))
    }
    fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        if let Some(n) = self.adapter_name(name) {
            self.adapter.tensor_info(&n)
        } else {
            self.base.tensor_info(name)
        }
    }
    fn tensor_slice(&self, name: &str) -> Option<&[u8]> {
        if let Some(n) = self.adapter_name(name) {
            self.adapter.tensor_slice(&n)
        } else {
            self.base.tensor_slice(name)
        }
    }
}
#[allow(clippy::too_many_arguments)]
pub(crate) fn linear_lora(
    base: &dyn TensorSource,
    adapter: &dyn TensorSource,
    scale: f32,
    name: &str,
    input: &[f32],
    rows: usize,
    pool: &ComputePool,
) -> Result<Vec<f32>, String> {
    if !scale.is_finite() {
        return Err("invalid LoRA scale".into());
    }
    let overlay = OverlaySource::new(base, adapter);
    let mut output = linear(&overlay, name, input, rows, pool)?;
    let stem = name
        .strip_suffix(".weight")
        .ok_or("LoRA projection requires .weight name")?;
    let a = format!("{stem}.lora_A");
    let b = format!("{stem}.lora_B");
    match (adapter.tensor_info(&a), adapter.tensor_info(&b)) {
        (None, None) => {}
        (Some(_), Some(_)) => {
            let low = linear(adapter, &a, input, rows, pool)?;
            let delta = linear(adapter, &b, &low, rows, pool)?;
            if delta.len() != output.len() {
                return Err(format!("LoRA output shape mismatch: {name}"));
            }
            for (x, d) in output.iter_mut().zip(delta) {
                *x += d * scale;
            }
        }
        _ => return Err(format!("incomplete LoRA pair: {name}")),
    }
    Ok(output)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::core::tensor::{GGMLType, MetaValue, TensorInfo};
    use std::collections::HashMap;
    #[derive(Default)]
    pub(crate) struct Source(pub(crate) HashMap<String, (TensorInfo, Vec<u8>)>);
    impl Source {
        pub(crate) fn add(&mut self, name: &str, dims: &[usize], vals: &[f32]) {
            assert_eq!(dims.iter().product::<usize>(), vals.len());
            self.0.insert(
                name.into(),
                (
                    TensorInfo {
                        name: name.into(),
                        dims: dims.iter().rev().map(|&n| n as u64).collect(),
                        ggml_type: GGMLType::F32,
                        offset: 0,
                    },
                    vals.iter().flat_map(|v| v.to_le_bytes()).collect(),
                ),
            );
        }
    }
    impl TensorSource for Source {
        fn metadata(&self, _: &str) -> Option<&MetaValue> {
            None
        }
        fn tensor_info(&self, n: &str) -> Option<&TensorInfo> {
            self.0.get(n).map(|t| &t.0)
        }
        fn tensor_slice(&self, n: &str) -> Option<&[u8]> {
            self.0.get(n).map(|t| t.1.as_slice())
        }
    }
    #[test]
    fn pytorch_linear_shape_bias_and_embedding_bounds() {
        let mut s = Source::default();
        s.add("proj.weight", &[2, 3], &[1., 2., 3., -1., 0., 2.]);
        s.add("proj.bias", &[2], &[0.5, -0.5]);
        assert_eq!(shape(&s, "proj.weight").unwrap(), [2, 3]);
        assert_eq!(
            linear(
                &s,
                "proj.weight",
                &[1., 2., 3., 3., 2., 1.],
                2,
                &ComputePool::new(1)
            )
            .unwrap(),
            [14.5, 4.5, 10.5, -1.5]
        );
        assert_eq!(
            embed(&s, "proj.weight", &[1, 0]).unwrap(),
            [-1., 0., 2., 1., 2., 3.]
        );
        assert!(embed(&s, "proj.weight", &[2]).is_err());
        assert!(linear(&s, "proj.weight", &[1., 2.], 1, &ComputePool::new(1)).is_err());
        s.0.get_mut("proj.weight").unwrap().1.pop();
        assert!(values(&s, "proj.weight").is_err());
    }
    #[test]
    fn normalization_matches_independent_torch_f32_reference() {
        let x = [1., 2., 4., -2., -1., 0.];
        let gamma = [1., 2., 0.5];
        let beta = [0.1, -0.2, 0.3];
        let expected = [
            -0.96904147,
            -0.73452073,
            0.96815085,
            -1.12473559,
            -0.2,
            0.91236782,
        ];
        let actual = layer_norm(&x, &gamma, &beta, 1e-5).unwrap();
        for (a, b) in actual.iter().zip(expected) {
            assert!((a - b).abs() < 2e-6, "{a} vs {b}");
        }
        assert!(rms_norm(&x, &[], 1e-6).is_err());
        assert!(layer_norm(&x, &gamma, &[1.], 1e-5).is_err());
    }
    #[test]
    fn adapter_overlays_full_weight_and_adds_the_lora_residual() {
        let mut base = Source::default();
        base.add("proj.weight", &[2, 2], &[1., 0., 0., 1.]);
        base.add("proj.bias", &[2], &[0.1, 0.2]);
        let mut delta = Source::default();
        delta.add("proj.base.weight", &[2, 2], &[2., 0., 0., 3.]);
        delta.add("proj.lora_A", &[1, 2], &[1., 1.]);
        delta.add("proj.lora_B", &[2, 1], &[1., -1.]);
        let output = linear_lora(
            &base,
            &delta,
            2.,
            "proj.weight",
            &[1., 2.],
            1,
            &ComputePool::new(1),
        )
        .unwrap();
        for (a, b) in output.iter().zip([8.1, 0.2]) {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
        delta.0.remove("proj.lora_B");
        assert!(linear_lora(
            &base,
            &delta,
            2.,
            "proj.weight",
            &[1., 2.],
            1,
            &ComputePool::new(1)
        )
        .is_err());
    }
    #[test]
    fn normalization_rejects_nonfinite_bias() {
        assert!(layer_norm(&[1., 2.], &[1., 1.], &[f32::NAN, 0.], 1e-5).is_err());
    }
}
