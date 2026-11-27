//! Qwen3 model family
//!
//! Per [`MODEL_ORGANIZATION.md`](../../../../docs/MODEL_ORGANIZATION.md) §2.1:
//! - `trunk/` contains the pure transformer decoder (`Qwen3Model` + `Qwen3Session`).
//! - `asr/` and `tts/` are siblings to `trunk/` (not nested).
//!
//! ## Public surface (re-exported from `trunk/`)
//!
//! `Qwen3Model`, `Qwen3Session`, `Qwen3Config`, `Qwen3Rope`, `Qwen3Input`,
//! `Qwen3GenerateOptions`, `Qwen3Generation`, `text_encode`,
//! `run_shared_inference`, `qwen_text_positions`, `Qwen3LayerWeights`,
//! `get_f32_tensor`, `load_layers`, `load_layers_static`, `static_weight`.

use crate::core::tensor::TensorSource;

pub mod asr;
pub mod embedding;
pub mod omni;
pub mod omni_audio;
pub mod trunk;
pub mod tts;
pub mod vision;

pub use embedding::run_embedding;
pub use trunk::{
    get_f32_tensor, load_layers, load_layers_static, qwen_text_positions, run_shared_inference,
    static_weight, text_encode, Qwen3Config, Qwen3GenerateOptions, Qwen3Generation, Qwen3Input,
    Qwen3LayerWeights, Qwen3Model, Qwen3Rope, Qwen3Session,
};

/// GGUF metadata key that marks a checkpoint as the WeVisOne-VL variant.
///
/// WeVisOne ships the `qwen3vl` architecture, so the architecture name alone
/// cannot tell it apart from plain Qwen3-VL; the publisher writes this key
/// into the GGUF metadata instead.
pub const WEVISONE_SOURCE_REVISION_KEY: &str = "wevisone_vl.source_revision";

/// Whether `source` is a WeVisOne-VL checkpoint.
///
/// The checkpoint is not interchangeable with plain Qwen3-VL: its trained
/// chat template has no thinking mechanism, its vision tower's precompute has
/// to stay in F32, and its resident decode path keeps KV on the CPU. Every
/// one of those decisions keys off the same metadata entry, so the entry is
/// defined once here and this predicate is the only thing that reads it.
///
/// Call sites must go through this function rather than repeating
/// `source.metadata(WEVISONE_SOURCE_REVISION_KEY).is_some()`: the same
/// question used to be spelled out at a dozen places, and a new WeVisOne
/// variant that missed one of them would diverge silently. The
/// `wevisone_marker_is_read_in_one_place` test keeps it that way.
pub fn is_wevisone<S: TensorSource + ?Sized>(source: &S) -> bool {
    source.metadata(WEVISONE_SOURCE_REVISION_KEY).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::tensor::{MetaValue, TensorInfo};

    struct MetadataSource(Option<MetaValue>);

    impl TensorSource for MetadataSource {
        fn metadata(&self, key: &str) -> Option<&MetaValue> {
            (key == WEVISONE_SOURCE_REVISION_KEY)
                .then_some(self.0.as_ref())
                .flatten()
        }

        fn tensor_info(&self, _name: &str) -> Option<&TensorInfo> {
            None
        }

        fn tensor_slice(&self, _name: &str) -> Option<&[u8]> {
            None
        }
    }

    #[test]
    fn is_wevisone_follows_the_metadata_key() {
        let marked = MetadataSource(Some(MetaValue::String("855d2a73".into())));
        assert!(is_wevisone(&marked));
        assert!(is_wevisone(&marked as &dyn TensorSource));

        let unmarked = MetadataSource(None);
        assert!(!is_wevisone(&unmarked));
        assert!(!is_wevisone(&unmarked as &dyn TensorSource));
    }

    /// The WeVisOne question was spelled out at a dozen call sites, and
    /// forgetting one of them diverges silently. Scan the tree and fail if a
    /// new site reads the marker key directly instead of asking
    /// [`is_wevisone`].
    #[test]
    fn wevisone_marker_is_read_in_one_place() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut offenders = Vec::new();
        let mut stack = vec![root.clone()];
        while let Some(dir) = stack.pop() {
            for entry in std::fs::read_dir(&dir).expect("read src/") {
                let path = entry.expect("dir entry").path();
                if path.is_dir() {
                    stack.push(path);
                    continue;
                }
                if path.extension().and_then(|e| e.to_str()) != Some("rs") {
                    continue;
                }
                let text = std::fs::read_to_string(&path).expect("read source file");
                if !text.contains("\"wevisone_vl.source_revision\"") {
                    continue;
                }
                let relative = path
                    .strip_prefix(&root)
                    .expect("path under src/")
                    .display()
                    .to_string()
                    .replace('\\', "/");
                // The definition site is the one file allowed to spell the key
                // out; everything else must use the constant or the predicate.
                if relative != "models/qwen3/mod.rs" {
                    offenders.push(relative);
                }
            }
        }
        offenders.sort();
        assert!(
            offenders.is_empty(),
            "the WeVisOne marker key must only be spelled out in models/qwen3/mod.rs; \
             these files read it directly instead of calling models::qwen3::is_wevisone: \
             {offenders:?}"
        );
    }
}
