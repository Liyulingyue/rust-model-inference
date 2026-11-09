//! HAL-X THX-02 decision model: a noncausal readout, not a chat decoder.
mod forward;
mod input;
pub mod protocol;
mod runtime;
#[cfg(test)]
mod tests;
mod weights;

use crate::{
    compute::ComputePolicy,
    core::{
        tensor::{MetaValue, TensorSource},
        thread_pool::ComputePool,
    },
};
use serde_json::Value;
use std::sync::Arc;

pub const ARCH: &str = "thx02";
pub use protocol::validate_request;

pub struct Engine {
    source: Arc<dyn TensorSource>,
    thinker: Option<Arc<dyn TensorSource>>,
    encoder: input::Encoder,
    thinker_encoder: Option<input::Encoder>,
    pool: Arc<ComputePool>,
    max_context: usize,
    compute: ComputePolicy,
    temperature: f64,
}

impl Engine {
    pub fn new(
        source: Arc<dyn TensorSource>,
        thinker: Option<Arc<dyn TensorSource>>,
        pool: Arc<ComputePool>,
        max_context: usize,
        compute: ComputePolicy,
    ) -> Result<Self, String> {
        compute.check_build()?;
        if max_context == 0 {
            return Err("THX-02 context must be nonzero".into());
        }
        let config = weights::config(source.as_ref(), false)?;
        let temperature = source
            .metadata("thx02.temperature")
            .and_then(MetaValue::to_f64)
            .filter(|t| t.is_finite() && *t > 0.)
            .ok_or("THX-02 requires a positive calibrated temperature")?;
        let encoder = input::Encoder::new(source.as_ref())?;
        let thinker_encoder = if let Some(s) = &thinker {
            weights::config(s.as_ref(), true)?;
            for key in ["thx02.model_sha256", "thx02.source_revision"] {
                if weights::string(source.as_ref(), key)? != weights::string(s.as_ref(), key)? {
                    return Err(format!(
                        "THX-02 thinker {key} does not match decision checkpoint"
                    ));
                }
            }
            Some(input::Encoder::new(s.as_ref())?)
        } else {
            None
        };
        Ok(Self {
            source,
            thinker,
            encoder,
            thinker_encoder,
            pool,
            max_context: max_context.min(config.n_ctx),
            compute,
            temperature,
        })
    }

    pub fn decide(&self, request: &Value) -> Result<Value, String> {
        validate_request(request)?;
        if request
            .get("think")
            .is_some_and(|v| v == true || v == "auto")
            && self.thinker.is_none()
        {
            return Err(
                "THX-02 think requires the matching merged thinker GGUF (--decision-thinker)"
                    .into(),
            );
        }
        let _scope = self.compute.configure_cli()?;
        self.compute.validate_text_arch(ARCH)?;
        #[cfg(feature = "vulkan")]
        let device = self.compute.context().map_err(|e| e.to_string())?;
        #[cfg(feature = "vulkan")]
        let before = device.map(|d| d.compute_stats());
        let result = runtime::decide(self, request)?;
        #[cfg(feature = "vulkan")]
        if let (Some(device), Some(before)) = (device, before) {
            let stats = device.compute_stats().since(before);
            if self.compute == ComputePolicy::Vulkan && stats.submissions == 0 {
                return Err("THX-02 Vulkan request executed no device operators".into());
            }
            if std::env::var_os("RMI_COMPUTE_TRACE").is_some() {
                eprintln!("compute scope=thx02 requested={:?} device_submissions={} static_uploads={} cpu=gated_delta_scan,norms,small_projections device=eligible_projections,verified_attention_with_cpu_fallback", self.compute, stats.submissions, stats.static_uploads);
            }
        }
        Ok(result)
    }
}
