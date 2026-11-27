//! Helpers shared by the diffusion pipelines.
//!
//! These are the tensor-contract checks that every pipeline performs when
//! loading its components. They were duplicated verbatim across `auk`,
//! `ernie_image` and `z_image`, which meant a change to the contract had to be
//! made in three places and could silently diverge.
//!
//! Only genuinely identical code lives here. Where a pipeline has a richer
//! diagnostic or different behaviour -- Z-Image's `require_finite` reports
//! nan/inf counts and the first bad index, and its `resize_zeroed` avoids
//! clearing a buffer it can reuse -- it keeps its own version and calls the
//! shared predicate instead.

use crate::core::tensor::GGMLType;
use crate::core::tensor::TensorSource;

/// Whether every element is finite. NaN and infinities are both rejected.
pub(crate) fn all_finite(values: &[f32]) -> bool {
    values.iter().all(|v| v.is_finite())
}

/// Assert that a tensor exists with the expected shape, dtype and byte length.
///
/// The dtype and byte-length checks are what catch a GGUF that was quantized
/// differently from the one a pipeline was validated against, which otherwise
/// shows up much later as an out-of-bounds read.
pub(crate) fn require_tensor(
    source: &dyn TensorSource,
    name: &str,
    dims: &[u64],
    ggml_type: GGMLType,
) -> Result<(), String> {
    // Metadata only: the contracts are validated before any payload is read,
    // and the loader measures the bytes when it loads the tensor.
    crate::core::tensor::require_tensor_meta(source, name, dims, &[ggml_type]).map(|_| ())
}

/// Reject a buffer containing NaN or infinities.
pub(crate) fn require_finite(values: &[f32], name: &str) -> Result<(), String> {
    if all_finite(values) {
        Ok(())
    } else {
        Err(format!("Non-finite {name}"))
    }
}

/// Resize `dst` to `len` zeros, reporting allocation failures by `name`.
///
/// This always clears first, so a buffer that shrinks discards its tail.
pub(crate) fn resize_zeroed(dst: &mut Vec<f32>, len: usize, name: &str) -> Result<(), String> {
    dst.clear();
    dst.try_reserve_exact(len)
        .map_err(|e| format!("Failed to allocate {name}: {e}"))?;
    dst.resize(len, 0.0);
    Ok(())
}

/// Test-only helpers shared by the diffusion model probes.
#[cfg(test)]
pub(crate) mod testing {
    use crate::core::tensor::{GGMLType, MetaValue, TensorInfo, TensorSource};
    use std::collections::HashMap;

    /// A source that answers metadata and tensor names but holds no payload.
    ///
    /// The signature probes only look at the header, so a test can describe
    /// a model's shape without building a GGUF.
    pub(crate) struct NameOnlySource {
        arch: Option<MetaValue>,
        info: HashMap<String, TensorInfo>,
    }

    impl NameOnlySource {
        /// Change the stored type of a tensor the source already carries.
        pub(crate) fn with_type(mut self, name: &str, ggml_type: GGMLType) -> Self {
            if let Some(info) = self.info.get_mut(name) {
                info.ggml_type = ggml_type;
            }
            self
        }

        /// A source with the given architecture tag and tensor names.
        pub(crate) fn new(arch: Option<&str>, tensors: &[&str]) -> Self {
            let mut info = HashMap::new();
            for name in tensors {
                info.insert(
                    (*name).to_string(),
                    TensorInfo {
                        name: (*name).to_string(),
                        dims: vec![2, 2],
                        ggml_type: GGMLType::F16,
                        offset: 0,
                    },
                );
            }
            Self {
                arch: arch.map(|value| MetaValue::String(value.to_string())),
                info,
            }
        }
    }

    impl TensorSource for NameOnlySource {
        fn metadata(&self, key: &str) -> Option<&MetaValue> {
            if key == "general.architecture" {
                return self.arch.as_ref();
            }
            None
        }

        fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
            self.info.get(name)
        }

        fn tensor_slice(&self, _name: &str) -> Option<&[u8]> {
            None
        }
    }
}
