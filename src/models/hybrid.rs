//! Explicit runtime dispatch for the two hybrid text architectures.

use crate::core::scratchpad::KvCache;
use crate::core::tensor::TensorSource;
use crate::core::thread_pool::ComputePool;
use crate::models::edge0::Edge0Model;
use crate::models::qwen35::trunk::session::HybridTrunkModel;
use crate::models::qwen35::trunk::HybridTrunk;
use crate::models::qwen35::{Qwen35Model, Qwen35Scratchpad};

pub enum HybridTextModel<'a> {
    Qwen35(Qwen35Model<'a>),
    Edge0(Edge0Model<'a>),
    Occamy(crate::models::occamy::weights::OccamyModel<'a>),
}

impl<'a> HybridTextModel<'a> {
    pub fn from_source(source: &'a dyn TensorSource) -> Result<Self, String> {
        match source
            .metadata("general.architecture")
            .and_then(|value| value.to_string_val())
        {
            Some("qwen35") => Ok(Self::Qwen35(Qwen35Model::from_source(source)?)),
            Some("edge0") => Ok(Self::Edge0(Edge0Model::from_source(source)?)),
            Some("qwen35moe") => Ok(Self::Occamy(
                crate::models::occamy::weights::OccamyModel::from_source(source)?,
            )),
            arch => Err(format!("Unsupported hybrid architecture: {arch:?}")),
        }
    }

    pub fn trunk(&self) -> &HybridTrunk<'a> {
        match self {
            Self::Qwen35(model) => model,
            Self::Edge0(model) => &model.trunk,
            Self::Occamy(model) => &model.trunk,
        }
    }
}

impl<'m> HybridTrunkModel<'m> for HybridTextModel<'m> {
    fn trunk(&self) -> &HybridTrunk<'m> {
        self.trunk()
    }

    fn forward_at(
        &mut self,
        n_tokens: usize,
        base_position: usize,
        kv_cache: &mut KvCache,
        scratch: &mut Qwen35Scratchpad,
        pool: &ComputePool,
        positions: &[[usize; 4]],
    ) -> Result<Vec<f32>, String> {
        match self {
            Self::Qwen35(model) => model.forward_at(
                n_tokens,
                base_position,
                kv_cache,
                scratch,
                pool,
                positions,
                None,
            ),
            Self::Edge0(model) => {
                model.forward_at(n_tokens, base_position, kv_cache, scratch, pool, positions)
            }
            Self::Occamy(model) => {
                crate::models::qwen35::trunk::session::HybridTrunkModel::forward_at(
                    model,
                    n_tokens,
                    base_position,
                    kv_cache,
                    scratch,
                    pool,
                    positions,
                )
            }
        }
    }

    #[cfg(feature = "vulkan")]
    fn supports_vulkan(&self) -> bool {
        matches!(self, Self::Qwen35(_))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::tensor::{MetaValue, TensorInfo};

    struct Edge0Source(MetaValue);
    impl TensorSource for Edge0Source {
        fn metadata(&self, key: &str) -> Option<&MetaValue> {
            (key == "general.architecture").then_some(&self.0)
        }
        fn tensor_info(&self, _: &str) -> Option<&TensorInfo> {
            None
        }
        fn tensor_slice(&self, _: &str) -> Option<&[u8]> {
            None
        }
    }

    #[test]
    fn edge0_dispatch_uses_edge0_metadata() {
        let source = Edge0Source(MetaValue::String("edge0".into()));
        let error = match HybridTextModel::from_source(&source) {
            Ok(_) => panic!("incomplete Edge0 source must fail"),
            Err(error) => error,
        };
        assert!(error.contains("edge0.embedding_length"), "{error}");
    }
}
