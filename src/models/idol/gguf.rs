//! A single GGUF carries all six Idol components, using their original tensor names under roles.
use super::{background, conditioning, decoder, denoiser, Model};
use crate::core::{
    loader::GGUFLoader,
    tensor::{GGMLType, MetaValue, TensorInfo, TensorSource},
};
use std::{collections::HashSet, path::Path};

pub(super) const ROLES: [&str; 6] = [
    "denoiser",
    "octree",
    "gaussian",
    "dino",
    "vae",
    "background",
];
const FORMATS: [&str; 10] = [
    "original", "f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "q4_k", "q5_k", "q6_k",
];

fn string<'a>(source: &'a dyn TensorSource, key: &str) -> Result<&'a str, String> {
    match source.metadata(key) {
        Some(MetaValue::String(value)) => Ok(value),
        _ => Err(format!("Idol GGUF requires string metadata {key}")),
    }
}

fn validate(source: &dyn TensorSource, tensors: &[TensorInfo]) -> Result<(), String> {
    if string(source, "general.architecture")? != "idol" {
        return Err("Idol GGUF architecture must be idol".into());
    }
    if !matches!(
        source.metadata("idol.format_version"),
        Some(MetaValue::Uint32(1) | MetaValue::Uint64(1))
    ) {
        return Err("Unsupported Idol GGUF format_version; expected 1".into());
    }
    if string(source, "idol.oracle_commit")? != "4702a7f00850e34b6e83a2170cc123c3e9570744" {
        return Err("Unsupported Idol GGUF oracle contract".into());
    }
    if !FORMATS.contains(&string(source, "idol.weight_format")?) {
        return Err("Unsupported Idol GGUF weight_format".into());
    }
    let config = |key: &str| -> Result<serde_json::Value, String> {
        serde_json::from_str(string(source, key)?)
            .map_err(|e| format!("Invalid Idol GGUF {key}: {e}"))
    };
    super::validate_config(&config("idol.config_json")?)?;
    super::check_component_config(
        "dino",
        &config("idol.dino_config_json")?,
        super::expected_dino_config(),
    )?;
    super::check_component_config(
        "vae",
        &config("idol.vae_config_json")?,
        super::expected_vae_config(),
    )?;
    let mut names = HashSet::new();
    let mut present = HashSet::new();
    for info in tensors {
        if !names.insert(&info.name) {
            return Err(format!("Duplicate Idol GGUF tensor {}", info.name));
        }
        let (role, suffix) = info
            .name
            .split_once('.')
            .ok_or_else(|| format!("Idol GGUF tensor needs a component prefix: {}", info.name))?;
        if !ROLES.contains(&role) || suffix.is_empty() {
            return Err(format!("Unknown Idol GGUF component in {}", info.name));
        }
        present.insert(role);
        match info.ggml_type {
            GGMLType::F32 | GGMLType::F16 | GGMLType::BF16 => (),
            GGMLType::I64 if matches!(role, "background" | "vae") => (),
            GGMLType::Q8_0
            | GGMLType::Q4_0
            | GGMLType::Q4_1
            | GGMLType::Q4K
            | GGMLType::Q5K
            | GGMLType::Q6K
                if info.dims.len() == 2 =>
            {
                ()
            }
            _ => {
                return Err(format!(
                    "Unsupported Idol GGUF tensor {}: type {:?}, dims {:?}",
                    info.name, info.ggml_type, info.dims
                ))
            }
        }
    }
    for role in ROLES {
        if !present.contains(role) {
            return Err(format!("Missing Idol GGUF component {role}"));
        }
    }
    let scalars: Vec<String> = serde_json::from_str(string(source, "idol.scalar_tensors_json")?)
        .map_err(|e| format!("Invalid Idol GGUF scalar list: {e}"))?;
    let mut scalar_names = HashSet::new();
    for name in &scalars {
        if !scalar_names.insert(name)
            || source.tensor_info(name).is_none_or(|info| info.dims != [1])
        {
            return Err(format!("Invalid Idol GGUF scalar tensor {name}"));
        }
    }
    Ok(())
}

pub(super) struct Bundle {
    source: GGUFLoader,
}
impl Bundle {
    pub fn open(path: &Path) -> Result<Self, String> {
        let source = GGUFLoader::from_file(path)?;
        validate(&source, source.tensors())?;
        Ok(Self { source })
    }
    pub fn component(&self, role: &'static str) -> Component<'_> {
        Component {
            source: &self.source,
            role,
        }
    }
    pub fn load(&self, compute: crate::compute::ComputePolicy) -> Result<Model, String> {
        eprintln!(
            "[idol] weights=gguf format={} runtime_storage=F32 component_precision=original",
            string(&self.source, "idol.weight_format")?
        );
        Ok(Model {
            compute,
            denoiser: denoiser::Denoiser::load(&self.component("denoiser"))
                .map_err(|e| format!("Idol denoiser: {e}"))?,
            decoder: decoder::Decoder::load(&self.component("octree"), &self.component("gaussian"))
                .map_err(|e| format!("Idol decoder: {e}"))?,
            conditioning: conditioning::Conditioning::load(
                &self.component("dino"),
                &self.component("vae"),
            )
            .map_err(|e| format!("Idol conditioning: {e}"))?,
            background: background::Background::load(&self.component("background"))
                .map_err(|e| format!("Idol background: {e}"))?,
        })
    }
}

pub(super) struct Component<'a> {
    source: &'a dyn TensorSource,
    role: &'static str,
}
impl TensorSource for Component<'_> {
    fn metadata(&self, _: &str) -> Option<&MetaValue> {
        None
    }
    fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        self.source.tensor_info(&format!("{}.{name}", self.role))
    }
    fn tensor_slice(&self, name: &str) -> Option<&[u8]> {
        self.source.tensor_slice(&format!("{}.{name}", self.role))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    struct Source {
        metadata: BTreeMap<String, MetaValue>,
        tensors: Vec<TensorInfo>,
    }
    impl TensorSource for Source {
        fn metadata(&self, key: &str) -> Option<&MetaValue> {
            self.metadata.get(key)
        }
        fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
            self.tensors.iter().find(|t| t.name == name)
        }
        fn tensor_slice(&self, name: &str) -> Option<&[u8]> {
            self.tensor_info(name).map(|_| &[0, 0, 128, 63][..])
        }
    }
    fn source() -> Source {
        let mut metadata = BTreeMap::new();
        for (key, value) in [
            ("general.architecture", "idol".into()),
            (
                "idol.oracle_commit",
                "4702a7f00850e34b6e83a2170cc123c3e9570744".into(),
            ),
            ("idol.weight_format", "original".into()),
            ("idol.scalar_tensors_json", "[]".into()),
            (
                "idol.config_json",
                super::super::expected_config().to_string(),
            ),
            (
                "idol.dino_config_json",
                super::super::expected_dino_config().to_string(),
            ),
            (
                "idol.vae_config_json",
                super::super::expected_vae_config().to_string(),
            ),
        ] {
            metadata.insert(key.into(), MetaValue::String(value));
        }
        metadata.insert("idol.format_version".into(), MetaValue::Uint64(1));
        Source {
            metadata,
            tensors: ROLES
                .iter()
                .map(|role| TensorInfo {
                    name: format!("{role}.weight"),
                    dims: vec![1],
                    ggml_type: GGMLType::F32,
                    offset: 0,
                })
                .collect(),
        }
    }
    #[test]
    fn bundle_contract_checks_architecture_components_and_configuration() {
        let mut s = source();
        validate(&s, &s.tensors).unwrap();
        s.metadata.insert(
            "general.architecture".into(),
            MetaValue::String("qwen3".into()),
        );
        assert!(validate(&s, &s.tensors)
            .unwrap_err()
            .contains("architecture"));
        let mut s = source();
        s.tensors.retain(|t| !t.name.starts_with("vae."));
        assert!(validate(&s, &s.tensors)
            .unwrap_err()
            .contains("component vae"));
        let mut s = source();
        let mut cfg = super::super::expected_dino_config();
        cfg["rope_theta"] = 200.into();
        s.metadata.insert(
            "idol.dino_config_json".into(),
            MetaValue::String(cfg.to_string()),
        );
        assert!(validate(&s, &s.tensors).unwrap_err().contains("rope_theta"));
    }
    #[test]
    fn bundle_contract_rejects_duplicate_unknown_and_unsupported_tensors() {
        let mut s = source();
        s.tensors.push(s.tensors[0].clone());
        assert!(validate(&s, &s.tensors).unwrap_err().contains("Duplicate"));
        let mut s = source();
        s.tensors[0].name = "wrong.weight".into();
        assert!(validate(&s, &s.tensors).unwrap_err().contains("Unknown"));
        let mut s = source();
        s.tensors[0].ggml_type = GGMLType::Q2K;
        assert!(validate(&s, &s.tensors)
            .unwrap_err()
            .contains("Unsupported"));
    }
    #[test]
    fn components_preserve_tensor_identity_and_do_not_cross_roles() {
        let s = source();
        let c = Component {
            source: &s,
            role: "denoiser",
        };
        assert_eq!(c.tensor_info("weight").unwrap().name, "denoiser.weight");
        assert_eq!(c.tensor_slice("weight").unwrap(), &[0, 0, 128, 63]);
        assert!(c.tensor_info("gaussian.weight").is_none());
    }
}
