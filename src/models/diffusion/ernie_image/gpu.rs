//! Device-side projections for the ERNIE-Image DiT.
//!
//! One runtime serves every projection in the model. The runtime holds the
//! weights it has uploaded, so a weight travels to the device once and
//! every later call that needs it reuses the buffer.
//!
//! That reuse is the whole point of keeping the runtime around. A runtime
//! built per call would start with an empty cache, and the DiT would send
//! the same weight to the device again for every projection that needs it:
//! the transfer costs more than the projection.
//!
//! The runtime does not borrow the model, so the caller can hold it next to
//! the tensors it reads from without the two borrowing each other.

#[cfg(feature = "vulkan")]
use crate::vulkan::ops::{BatchedLinearRuntime, GpuMatmulSpec};

/// The device projections, or nothing when there is no device.
///
/// The type exists in both builds so the callers do not need a `cfg` on
/// every signature that carries it. Without the `vulkan` feature the
/// device is not compiled in at all, so the projections there are empty
/// and the DiT runs on the CPU.
#[cfg(feature = "vulkan")]
pub(super) struct ErnieGpuProjections {
    runtime: BatchedLinearRuntime,
}

#[cfg(feature = "vulkan")]
impl ErnieGpuProjections {
    /// Build the runtime, or report that the device is not there.
    ///
    /// The device is optional: without one the DiT runs on the CPU, which is
    /// the path the exact CPU scope selects.
    pub(super) fn new(
        max_rows: usize,
        max_n_in: usize,
        max_n_out: usize,
        descriptor_capacity: usize,
    ) -> Result<Option<Self>, String> {
        crate::compute::ComputePolicy::legacy()
            .context_with(|| {
                let context = crate::compute::ComputePolicy::Vulkan
                    .context()
                    .map_err(|e| e.to_string())?
                    .ok_or("ERNIE-Image requires a hardware Vulkan device")?;
                Ok(Self {
                    runtime: BatchedLinearRuntime::new(
                        context,
                        max_rows,
                        max_n_in,
                        max_n_out,
                        descriptor_capacity,
                    )
                    .map_err(|e| e.to_string())?,
                })
            })
            .map_err(|e| e.to_string())
    }

    /// Project `rows` rows in one dispatch.
    pub(super) fn run(
        &mut self,
        weight: &[u8],
        spec: impl Into<GpuMatmulSpec>,
        input: &[f32],
        rows: usize,
        n_in: usize,
        n_out: usize,
        output: &mut [f32],
    ) -> Result<(), String> {
        self.runtime
            .matmul_rows(weight, spec, input, rows, n_in, n_out, output)
            .map_err(|e| e.to_string())
    }
}

/// Without the `vulkan` feature there is no device to project on.
#[cfg(not(feature = "vulkan"))]
pub(super) struct ErnieGpuProjections;

#[cfg(not(feature = "vulkan"))]
impl ErnieGpuProjections {
    /// No device in this build, so there is never a runtime to build.
    pub(super) fn new(
        _max_rows: usize,
        _max_n_in: usize,
        _max_n_out: usize,
        _descriptor_capacity: usize,
    ) -> Result<Option<Self>, String> {
        Ok(None)
    }
}
