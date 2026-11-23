use std::collections::BTreeMap;

use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::core::tensor::{f32_tensor_bytes, GGMLType, MetaValue, TensorSource};
use crate::ops::kernel::Weight;
use crate::ops::nn::{load_float_values, Linear};

const CONTRACT: &str = include_str!("contract.json");
const CONTRACT_S: &str = include_str!("contract_s.json");

/// Fixed per-variant numerical recipe (timesteps and noise convention).
///
/// The values are the released FastFrames recipes; FF-S draws transition noise
/// per frame when it is needed (`Sequential`), FF-L draws all three blocks
/// right after the initial noise (`Pregenerated`).
#[derive(Clone, Copy)]
pub(super) enum NoiseMode {
    Sequential,
    Pregenerated,
}
#[derive(Clone, Copy)]
pub(super) struct Recipe {
    /// Timesteps of the first frame's clean-up pass (4 steps, 3 transitions).
    pub first_frame: [f32; 4],
    /// Timesteps of every later frame (2 steps, 1 transition).
    pub timesteps: [f32; 2],
    /// Timestep the KV refiner embeds.
    pub refine: f32,
    pub noise: NoiseMode,
}
#[derive(Clone, Copy)]
pub(super) struct Dims {
    /// Wan DiT width (FF-L 5120, FF-S 1536).
    pub dim: usize,
    /// DiT attention heads; head_dim is `dim / heads` (128 for both).
    pub heads: usize,
    /// DiT blocks (FF-L 40, FF-S 30).
    pub layers: usize,
    /// KV refiner stem width, shared by both variants.
    pub refiner_dim: usize,
    /// Pixel decoder width, shared by both variants.
    pub decoder_dim: usize,
    /// FF-L projects 5120-wide features through a learned bridge; FF-S decodes
    /// its 1536-wide features directly.
    pub bridge: bool,
}
#[derive(Clone, Copy)]
pub(super) struct Spec {
    /// Human-readable variant name for diagnostics ("FastFrames-L"/"FastFrames-S").
    pub name: &'static str,
    pub dims: Dims,
    pub recipe: Recipe,
}

impl Spec {
    pub fn head_dim(&self) -> usize {
        self.dims.dim / self.dims.heads
    }
    /// Times whose conditioning every generation needs (schedules plus the
    /// frame-0 clean pass at t=0).
    pub fn conditioning_times(&self) -> Vec<f32> {
        let mut times = self.recipe.first_frame.to_vec();
        times.push(0.);
        times
    }
}

pub(super) const L: Spec = Spec {
    name: "FastFrames-L",
    dims: Dims {
        dim: 5120,
        heads: 40,
        layers: 40,
        refiner_dim: 768,
        decoder_dim: 1536,
        bridge: true,
    },
    recipe: Recipe {
        first_frame: [1000., 937., 833., 625.],
        timesteps: [1000., 833.],
        refine: 833.,
        noise: NoiseMode::Pregenerated,
    },
};
pub(super) const S: Spec = Spec {
    name: "FastFrames-S",
    dims: Dims {
        dim: 1536,
        heads: 12,
        layers: 30,
        refiner_dim: 768,
        decoder_dim: 1536,
        bridge: false,
    },
    recipe: Recipe {
        first_frame: [1000., 937.5, 833.3333129882812, 625.],
        timesteps: [1000., 833.3333129882812],
        refine: 833.3333129882812,
        noise: NoiseMode::Sequential,
    },
};

/// Variant selection by the GGUF's explicit `general.architecture`.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Variant {
    L,
    S,
}

impl Variant {
    pub fn of(source: &dyn TensorSource) -> Option<Self> {
        match source
            .metadata("general.architecture")
            .and_then(MetaValue::to_string_val)
            .as_deref()
        {
            Some("fastframes_l") => Some(Self::L),
            Some("fastframes_s") => Some(Self::S),
            _ => None,
        }
    }
    pub fn spec(&self) -> Spec {
        match self {
            Self::L => L,
            Self::S => S,
        }
    }
    /// Contract text pinned to this variant.
    fn contract(&self) -> &'static str {
        match self {
            Self::L => CONTRACT,
            Self::S => CONTRACT_S,
        }
    }
    /// Metadata key prefix, e.g. `fastframes_l.contract_sha256`.
    fn keys(&self) -> &'static str {
        match self {
            Self::L => "fastframes_l",
            Self::S => "fastframes_s",
        }
    }
}

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
struct Revisions {
    revision: String,
}
#[derive(Deserialize)]
struct Oracle {
    commit: String,
}
#[derive(Deserialize)]
struct Config {
    dim: usize,
    heads: usize,
    layers: usize,
}
#[derive(Deserialize)]
struct Contract {
    schemas: BTreeMap<String, Schema>,
    config: Config,
    source: Revisions,
    text_source: Revisions,
    oracle: Oracle,
}

pub(super) fn shapes(variant: &Variant) -> BTreeMap<String, Vec<u64>> {
    let contract: Contract = serde_json::from_str(variant.contract()).expect("embedded contract");
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
    Variant::of(source).is_some()
}

pub(super) fn validate_metadata(
    source: &dyn TensorSource,
    variant: &Variant,
) -> Result<(), String> {
    let keys = variant.keys();
    let name = variant.spec().name;
    if Variant::of(source) != Some(*variant) {
        return Err(format!("expected explicit {name} architecture"));
    }
    let contract: Contract = serde_json::from_str(variant.contract()).expect("embedded contract");
    let hash = format!("{:x}", Sha256::digest(variant.contract().as_bytes()));
    for (key, expected) in [
        (format!("{keys}.contract_sha256"), hash),
        (format!("{keys}.source_revision"), contract.source.revision),
        (
            format!("{keys}.text_revision"),
            contract.text_source.revision,
        ),
        (format!("{keys}.oracle_commit"), contract.oracle.commit),
    ] {
        if source.metadata(&key).and_then(MetaValue::to_string_val) != Some(expected.as_str()) {
            return Err(format!("{name} incompatible or missing {key}"));
        }
    }
    if source
        .metadata(&format!("{keys}.contract_version"))
        .and_then(MetaValue::to_u64)
        != Some(1)
    {
        return Err(format!("{name} contract version must be 1"));
    }
    let spec = variant.spec();
    if contract.config.dim != spec.dims.dim
        || contract.config.heads != spec.dims.heads
        || contract.config.layers != spec.dims.layers
    {
        return Err(format!(
            "{name} contract config disagrees with the fixed recipe"
        ));
    }
    Ok(())
}

pub(super) struct Graph<'a> {
    pub linear: BTreeMap<String, Linear<'a>>,
    pub floats: BTreeMap<String, Vec<f32>>,
    pub spec: Spec,
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
        let variant = Variant::of(source)
            .ok_or_else(|| "expected explicit fastframes architecture".to_string())?;
        validate_metadata(source, &variant)?;
        let shapes = shapes(&variant);
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
        Ok(Self {
            linear,
            floats,
            spec: variant.spec(),
        })
    }
}
