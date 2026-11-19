//! Explicit component offload. Layout, pointwise functions and sampling stay on the host.
use crate::{compute::ComputePolicy, core::thread_pool::ComputePool, ops::nn::Linear};

pub(super) struct Execution {
    policy: ComputePolicy,
    #[cfg(feature = "vulkan")]
    gpu: std::sync::Mutex<Option<super::device::Device>>,
}

impl Execution {
    pub fn attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        heads: usize,
    ) -> Result<Option<Vec<f32>>, String> {
        #[cfg(feature = "vulkan")]
        return self.run("sophon_attention", |gpu| gpu.attention(q, k, v, heads));
        #[cfg(not(feature = "vulkan"))]
        {
            let _ = (q, k, v, heads);
            Ok(None)
        }
    }

    pub fn log_mel(&self, samples: &[f32]) -> Result<crate::ops::audio::frontend::LogMel, String> {
        use crate::ops::audio::frontend;
        #[cfg(feature = "vulkan")]
        if let Some(raw) = self.run("sophon_frontend", |gpu| {
            let padded = frontend::reflect_pad(samples).map_err(|e| format!("{e:?}"))?;
            let frames = samples.len() / frontend::HOP;
            let power = gpu.spectrum(
                &padded,
                &frontend::periodic_hann_window(),
                frames,
                frontend::HOP,
                true,
            )?;
            let filters = frontend::mel_filters().map_err(|e| format!("{e:?}"))?;
            let count = frames
                .checked_mul(frontend::MEL_BINS)
                .ok_or("mel size overflow")?;
            gpu.custom(
                4,
                &[&power, filters],
                count,
                &[u32::try_from(frames).map_err(|_| "mel frames overflow")?],
                count.div_ceil(64),
            )
        })? {
            let maximum = raw.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let normalized = raw
                .iter()
                .map(|v| (v.max(maximum - 8.0) + 4.0) / 4.0)
                .collect();
            return Ok(frontend::LogMel {
                frames: samples.len() / frontend::HOP,
                raw,
                normalized,
            });
        }
        frontend::compute_log_mel_trimmed(samples).map_err(|e| format!("Sophon frontend: {e:?}"))
    }

    pub fn cpu() -> Self {
        Self {
            policy: ComputePolicy::Cpu,
            #[cfg(feature = "vulkan")]
            gpu: std::sync::Mutex::new(None),
        }
    }

    pub fn new(policy: ComputePolicy) -> Result<Self, String> {
        policy.check_build().map_err(|e| e.to_string())?;
        #[cfg(feature = "vulkan")]
        let gpu = if crate::ops::scalar_mode() {
            policy.context_with(|| Err("Sophon scalar mode requires CPU".into()))
        } else {
            policy.context()
        }
        .map_err(|e| e.to_string())?
        .map(super::device::Device::new);
        Ok(Self {
            policy,
            #[cfg(feature = "vulkan")]
            gpu: std::sync::Mutex::new(gpu),
        })
    }

    #[cfg(feature = "vulkan")]
    pub fn run(
        &self,
        stage: &str,
        op: impl FnOnce(&mut super::device::Device) -> Result<Vec<f32>, String>,
    ) -> Result<Option<Vec<f32>>, String> {
        let mut guard = self.gpu.lock().map_err(|_| "Sophon device lock poisoned")?;
        let Some(gpu) = guard.as_mut() else {
            if self.policy == ComputePolicy::Vulkan {
                return Err("Sophon Vulkan execution is unavailable".into());
            }
            return Ok(None);
        };
        match op(gpu) {
            Ok(values) if values.iter().all(|v| v.is_finite()) => {
                self.policy
                    .trace(stage, crate::compute::UsedBackend::Vulkan, values.len());
                Ok(Some(values))
            }
            result => {
                let error = result
                    .err()
                    .unwrap_or_else(|| "nonfinite device result".into());
                if self.policy == ComputePolicy::Vulkan {
                    return Err(format!("{stage}: {error}"));
                }
                log::warn!("{stage}: CPU fallback: {error}");
                *guard = None;
                Ok(None)
            }
        }
    }

    pub fn project(
        &self,
        weight: &crate::ops::kernel::Weight<'_>,
        input: &[f32],
        rows: usize,
    ) -> Result<Option<Vec<f32>>, String> {
        #[cfg(feature = "vulkan")]
        return self.run("sophon_projection", |gpu| gpu.project(weight, input, rows));
        #[cfg(not(feature = "vulkan"))]
        {
            let _ = (weight, input, rows, self.policy);
            Ok(None)
        }
    }

    pub fn linear(
        &self,
        linear: &Linear<'_>,
        pool: &ComputePool,
        input: &[f32],
        rows: usize,
    ) -> Result<Vec<f32>, String> {
        if let Some(mut output) = self.project(linear.binding().weight, input, rows)? {
            if let Some(bias) = linear.bias() {
                for row in output.chunks_exact_mut(linear.n_out()) {
                    for (v, b) in row.iter_mut().zip(bias) {
                        *v += b;
                    }
                }
            }
            return Ok(output);
        }
        let _scope = ComputePolicy::Cpu.cpu_scope();
        linear.forward(pool, input, rows)
    }
}

#[cfg(all(test, feature = "vulkan"))]
mod tests {
    use super::*;
    use crate::models::sophon_realtime::tests::assert_close;
    use crate::ops::kernel::{QuantizedTensor, Weight};

    #[test]
    fn strict_unavailable_device_cannot_fall_back() {
        let strict = Execution {
            policy: ComputePolicy::Vulkan,
            gpu: std::sync::Mutex::new(None),
        };
        assert!(strict.run("test", |_| Ok(vec![1.0])).is_err());
        assert!(Execution::cpu()
            .run("test", |_| panic!("CPU initialized device"))
            .unwrap()
            .is_none());
    }

    #[test]
    #[ignore = "requires a hardware Vulkan device"]
    fn vulkan_frontend_matches_cpu_short_and_boundary_audio() {
        let gpu = Execution::new(ComputePolicy::Vulkan).unwrap();
        for length in [320, 4000, 16001] {
            let samples: Vec<_> = (0..length)
                .map(|n| (n as f32 * 0.173).sin() * 0.25 + (n as f32 * 0.019).cos() * 0.1)
                .collect();
            let expected = Execution::cpu().log_mel(&samples).unwrap();
            let actual = gpu.log_mel(&samples).unwrap();
            assert_eq!(actual.frames, expected.frames);
            assert_close(&actual.normalized, &expected.normalized);
        }
    }

    #[test]
    #[ignore = "requires a hardware Vulkan device"]
    fn vulkan_f16_projection_keeps_opted_in_f32_activations_across_batches() {
        let gpu = Execution::new(ComputePolicy::Vulkan).unwrap();
        let bytes: Vec<_> = (0..128 * 7)
            .flat_map(|n| {
                crate::ops::f32_to_f16(((n * 17 % 127) as f32 - 63.0) / 64.0).to_le_bytes()
            })
            .collect();
        let weight = Weight::from_quantized(QuantizedTensor::from_bytes(
            &bytes,
            crate::GGMLType::F16,
            128,
            7,
        ));
        assert!(!weight.kernel.keeps_f32_activation());
        let linear = Linear::from_weight(weight, None)
            .unwrap()
            .with_f32_activations();
        let weight = linear.binding().weight;
        assert!(weight.kernel.keeps_f32_activation());
        let input: Vec<_> = (0..65 * 128)
            .map(|n| ((n * 23 % 163) as f32 - 81.0) / 113.0)
            .collect();
        for rows in [1, 32, 65, 3] {
            let actual = gpu
                .project(weight, &input[..rows * 128], rows)
                .unwrap()
                .unwrap();
            let mut expected = vec![0.0; rows * 7];
            for (x, y) in input[..rows * 128]
                .chunks_exact(128)
                .zip(expected.chunks_exact_mut(7))
            {
                weight
                    .kernel
                    .forward_prepared(x, &[], &[], None, y, 128, 7, 0, 1);
            }
            assert_close(&actual, &expected);
        }
    }

    #[test]
    #[ignore = "requires a hardware Vulkan device"]
    fn auto_discards_nonfinite_output_and_disables_failed_component() {
        let auto = Execution::new(ComputePolicy::Auto).unwrap();
        assert!(auto.gpu.lock().unwrap().is_some());
        assert!(auto
            .run("test", |_| Ok(vec![1.0, f32::NAN]))
            .unwrap()
            .is_none());
        assert!(auto
            .run("test", |_| panic!("failed device reused"))
            .unwrap()
            .is_none());
        let strict = Execution::new(ComputePolicy::Vulkan).unwrap();
        assert!(strict
            .run("test", |_| Err("injected failure".into()))
            .unwrap_err()
            .contains("injected failure"));
        let bytes = 2.0f32.to_le_bytes().repeat(4);
        let weight = Weight::from_f32_bytes(&bytes, 2, 2).unwrap();
        let linear = Linear::from_weight(weight, Some(vec![0.5, 1.0])).unwrap();
        assert_eq!(
            auto.linear(&linear, &ComputePool::new(2), &[1.0, 3.0], 1)
                .unwrap(),
            vec![8.5, 9.0]
        );
    }
}
