//! Device-side projections for the ERNIE-Image DiT.
//!
//! One runtime serves every projection in the model. The runtime holds the
//! weights it has uploaded, so a weight travels to the device once and
//! every later call that needs it reuses the buffer.
//!
//! That reuse is the whole point of keeping the runtime around. A runtime
//! built per call would start with an empty cache, and the DiT would send
//! the same weight to the device again for every projection that needs it:
//! the transfer costs more than the projection, and the device path ends up
//! slower than the CPU one it replaced.
//!
//! The runtime does not borrow the model, so the caller can hold it next to
//! the tensors it reads from without the two borrowing each other.

use crate::vulkan::ops::{BatchedLinearRuntime, GpuMatmulSpec};

pub(super) struct ErnieGpuProjections {
    runtime: BatchedLinearRuntime,
}

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
