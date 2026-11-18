use std::collections::BTreeMap;

use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::core::tensor::{f32_tensor_bytes, GGMLType, MetaValue, TensorSource};
use crate::ops::kernel::Weight;
use crate::ops::nn::{load_float_values, Linear};

const CONTRACT: &str = include_str!("contract.json");

#[derive(Deserialize)]
struct TensorSpec {
    shape: Vec<u64>,
}
#[derive(Deserialize)]
struct Schema {
    tensors: BTreeMap<String, TensorSpec>,
    block_prefix: String,
    block_count: usize,
    block: BTreeMap<String, TensorSpec>,
}
#[derive(Deserialize)]
struct Contract {
    schemas: BTreeMap<String, Schema>,
}

pub(super) fn shapes() -> BTreeMap<String, Vec<u64>> {
    let contract: Contract = serde_json::from_str(CONTRACT).expect("embedded FF-L contract");
    let mut shapes = BTreeMap::new();
    for (component, schema) in contract.schemas {
        let prefix = match component.as_str() {
            "kv_refiner" => "refiner",
            "pixel_decoder" => "decoder",
            other => other,
        };
        let mut tensors: Vec<_> = schema.tensors.into_iter().collect();
        for layer in 0..schema.block_count {
            for (key, spec) in &schema.block {
                tensors.push((
                    format!("{}{layer}.{key}", schema.block_prefix),
                    TensorSpec {
                        shape: spec.shape.clone(),
                    },
                ));
            }
        }
        for (name, spec) in tensors {
            let mut shape = spec.shape;
            if let Some(field) = name.strip_prefix("heads_grouped.gemms.0.") {
                for layer in 0..shape[0] {
                    let (field, dims) = if let Some(field) = field.strip_suffix("_w") {
                        (format!("{field}.weight"), vec![shape[2], shape[1]])
                    } else if let Some(field) = field.strip_suffix("_b") {
                        (format!("{field}.bias"), vec![shape[2]])
                    } else {
                        (field.to_owned(), vec![shape[2]])
                    };
                    shapes.insert(format!("refiner.heads.{layer}.{field}"), dims);
                }
                continue;
            }
            if name.ends_with("modulation") {
                shape = vec![shape.iter().product()];
            } else if name == "patch_embedding.weight"
                || (shape.len() == 5 && shape[2..] == [1, 1, 1])
            {
                shape = vec![shape[0], shape[1..].iter().product()];
            }
            shapes.insert(format!("{prefix}.{name}"), shape);
        }
    }
    shapes
}

pub fn matches_model(source: &dyn TensorSource) -> bool {
    source
        .metadata("general.architecture")
        .and_then(MetaValue::to_string_val)
        == Some("fastframes_l")
}

pub(super) fn validate_metadata(source: &dyn TensorSource) -> Result<(), String> {
    if !matches_model(source) {
        return Err("expected explicit fastframes_l architecture".into());
    }
    let hash = format!("{:x}", Sha256::digest(CONTRACT.as_bytes()));
    for (key, expected) in [
        ("fastframes_l.contract_sha256", hash.as_str()),
        (
            "fastframes_l.source_revision",
            "651fb104ea405268a4c7400b01c1859718090c5d",
        ),
        (
            "fastframes_l.text_revision",
            "349065d9c04b2608b85121f4b76f5fcbd03f7dab",
        ),
        (
            "fastframes_l.oracle_commit",
            "6044a56f44bffec500d04f2c21be3c5066891431",
        ),
    ] {
        if source.metadata(key).and_then(MetaValue::to_string_val) != Some(expected) {
            return Err(format!("FastFrames-L incompatible or missing {key}"));
        }
    }
    if source
        .metadata("fastframes_l.contract_version")
        .and_then(MetaValue::to_u64)
        != Some(1)
    {
        return Err("FastFrames-L contract version must be 1".into());
    }
    Ok(())
}

pub(super) struct Graph<'a> {
    pub linear: BTreeMap<String, Linear<'a>>,
    pub floats: BTreeMap<String, Vec<f32>>,
}

pub(super) fn load_linear<'a>(
    source: &'a dyn TensorSource,
    name: &str,
    bias: Option<&str>,
    n_in: usize,
    n_out: usize,
) -> Result<Linear<'a>, String> {
    if source
        .tensor_info(name)
        .is_some_and(|info| info.ggml_type == GGMLType::F32)
    {
        // The full F32 bundle exceeds host RAM; keep aligned GGUF matrices mmap-backed.
        let bytes = f32_tensor_bytes(source, name, &[n_in as u64, n_out as u64])?;
        let weight = Weight::from_f32_bytes(bytes, n_in, n_out)
            .ok_or_else(|| format!("invalid F32 matrix {name}"))?;
        let bias = bias
            .map(|name| load_float_values(source, name, &[n_out as u64]))
            .transpose()?;
        return Linear::from_weight(weight, bias);
    }
    Linear::from_source(source, name, bias, n_in, n_out)
}

impl<'a> Graph<'a> {
    pub fn executor<'m>(
        &'m self,
        policy: crate::compute::ComputePolicy,
        rows: usize,
        pool: std::sync::Arc<crate::core::thread_pool::ComputePool>,
    ) -> Result<crate::compute::linear::LinearExecutor<'m, 'a>, String> {
        let bindings = self
            .linear
            .iter()
            .filter(|(name, _)| name.as_str() != "text.token_embedding")
            .map(|(_, w)| w.binding())
            .collect();
        crate::compute::linear::LinearExecutor::new(policy, bindings, rows, pool)
            .map_err(|e| e.to_string())
    }

    pub fn load(source: &'a dyn TensorSource) -> Result<Self, String> {
        validate_metadata(source)?;
        let shapes = shapes();
        let mut linear = BTreeMap::new();
        let mut floats = BTreeMap::new();
        for (name, shape) in &shapes {
            if shape.len() == 2 && !name.contains("pos_embedding") {
                let bias = format!(
                    "{}.bias",
                    name.strip_suffix(".weight").ok_or("invalid linear name")?
                );
                linear.insert(
                    name.trim_end_matches(".weight").to_owned(),
                    load_linear(
                        source,
                        name,
                        shapes.contains_key(&bias).then_some(bias.as_str()),
                        shape[1] as usize,
                        shape[0] as usize,
                    )?,
                );
            } else {
                let dims: Vec<_> = shape.iter().rev().copied().collect();
                floats.insert(name.clone(), load_float_values(source, name, &dims)?);
            }
        }
        Ok(Self { linear, floats })
    }
}
