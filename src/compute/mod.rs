//! Session-local compute selection; CPU kernels retain their own ISA dispatch.

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ComputePolicy {
    #[default]
    Cpu,
    Auto,
    Vulkan,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UsedBackend {
    Cpu,
    Vulkan,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ComputeError {
    InvalidInput(String),
    Unsupported(String),
    Device(String),
    State(String),
}

impl std::fmt::Display for ComputeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (kind, message) = match self {
            Self::InvalidInput(s) => ("invalid compute input", s),
            Self::Unsupported(s) => ("unsupported compute request", s),
            Self::Device(s) => ("compute device failure", s),
            Self::State(s) => ("invalid compute state", s),
        };
        write!(f, "{kind}: {message}")
    }
}

impl std::error::Error for ComputeError {}

impl ComputePolicy {
    fn context_with<T>(
        self,
        initialize: impl FnOnce() -> Result<T, String>,
    ) -> Result<Option<T>, ComputeError> {
        if self == Self::Cpu {
            return Ok(None);
        }
        match initialize() {
            Ok(context) => Ok(Some(context)),
            Err(error) if self == Self::Auto => {
                log::info!("compute: CPU fallback: {error}");
                Ok(None)
            }
            Err(error) => Err(ComputeError::Device(error)),
        }
    }

    #[cfg(feature = "vulkan")]
    pub(crate) fn context(
        self,
    ) -> Result<Option<&'static crate::vulkan::VulkanContext>, ComputeError> {
        self.context_with(|| {
            if crate::vulkan::gpu_broken() {
                return Err("Vulkan device is disabled after a queue failure".into());
            }
            let context = crate::ops::float::shared_vulkan_context()?;
            if context.is_software_icd() {
                return Err("software Vulkan device is not an accelerator".into());
            }
            Ok(context)
        })
    }

    /// Compatibility constructors still honor the legacy request, within a CPU scope.
    pub(crate) fn legacy() -> Self {
        #[cfg(feature = "vulkan")]
        if crate::core::thread_pool::gpu_matmul_disabled() {
            return Self::Cpu;
        }
        if crate::ops::gpu_requested() {
            Self::Auto
        } else {
            Self::Cpu
        }
    }

    pub(crate) fn cpu_scope(self) -> ComputeScope {
        ComputeScope {
            #[cfg(feature = "vulkan")]
            _guard: (self == Self::Cpu)
                .then(crate::core::thread_pool::ComputePool::disable_gpu_matmul_for_scope),
        }
    }
}

pub(crate) struct ComputeScope {
    #[cfg(feature = "vulkan")]
    _guard: Option<crate::core::thread_pool::GpuMatmulScope>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    #[test]
    fn cpu_policy_never_initializes_vulkan() {
        let calls = Cell::new(0);
        let selected = ComputePolicy::Cpu
            .context_with(|| {
                calls.set(calls.get() + 1);
                Ok(7)
            })
            .unwrap();
        assert_eq!(selected, None);
        assert_eq!(calls.get(), 0);
    }

    #[test]
    fn cpu_auto_sessions_do_not_share_policy() {
        crate::ops::enable_gpu();
        for (policy, expected) in [
            (ComputePolicy::Auto, Some(7)),
            (ComputePolicy::Cpu, None),
            (ComputePolicy::Auto, Some(7)),
        ] {
            assert_eq!(policy.context_with(|| Ok(7)).unwrap(), expected);
        }
    }

    #[test]
    fn forced_vulkan_reports_initialization_failure() {
        assert_eq!(
            ComputePolicy::Auto
                .context_with::<()>(|| Err("no device".into()))
                .unwrap(),
            None
        );
        assert!(matches!(
            ComputePolicy::Vulkan.context_with::<()>(|| Err("no device".into())),
            Err(ComputeError::Device(_))
        ));
    }
}
