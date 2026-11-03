//! Render-owned projection uploads; activations are tiled to bound scratch memory.
use crate::vulkan::ops::{BatchedLinearRuntime, GpuMatmulSpec};

pub(super) struct GpuProjections {
    runtime: BatchedLinearRuntime,
}

impl GpuProjections {
    pub(super) fn new(
        max_n_in: usize,
        max_n_out: usize,
        weights: usize,
    ) -> Result<Option<Self>, String> {
        crate::compute::ComputePolicy::legacy()
            .context_with(|| {
                let context = crate::compute::ComputePolicy::Vulkan
                    .context()
                    .map_err(|e| e.to_string())?
                    .ok_or("Qwen-Image-2.1 requires a hardware Vulkan device")?;
                Ok(Self {
                    runtime: BatchedLinearRuntime::new(
                        context,
                        64,
                        max_n_in,
                        max_n_out,
                        weights + 1,
                    )
                    .map_err(|e| e.to_string())?,
                })
            })
            .map_err(|e| e.to_string())
    }

    pub(super) fn run(
        &mut self,
        weight: &[u8],
        spec: impl Into<GpuMatmulSpec>,
        input: &[f32],
        output: &mut [f32],
        n_in: usize,
        n_out: usize,
    ) -> Result<(), String> {
        if n_in == 0
            || n_out == 0
            || input.is_empty()
            || input.len() % n_in != 0
            || (input.len() / n_in).checked_mul(n_out) != Some(output.len())
            || input.iter().any(|v| !v.is_finite())
        {
            return Err("Invalid Qwen-Image-2.1 GPU projection input".into());
        }
        let spec = spec.into();
        for (x, y) in input.chunks(64 * n_in).zip(output.chunks_mut(64 * n_out)) {
            self.runtime
                .matmul_rows(weight, spec, x, x.len() / n_in, n_in, n_out, y)
                .map_err(|e| e.to_string())?;
        }
        if output.iter().any(|v| !v.is_finite()) {
            return Err("Non-finite Qwen-Image-2.1 GPU projection output".into());
        }
        crate::compute::ComputePolicy::Vulkan.trace(
            "qwen_image_projection",
            crate::compute::UsedBackend::Vulkan,
            input.len() / n_in,
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vulkan::ops::{GpuMatmulMode, GpuWeightFormat};

    #[test]
    fn cpu_scope_skips_gpu_after_global_enable() {
        // Isolate the permanent global flag from other tests in this process.
        if std::env::var_os("RMI_QI21_CPU_SCOPE_TEST").is_none() {
            assert!(std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    concat!(module_path!(), "::cpu_scope_skips_gpu_after_global_enable")
                        .split_once("::")
                        .unwrap()
                        .1,
                ])
                .env("RMI_QI21_CPU_SCOPE_TEST", "1")
                .status()
                .unwrap()
                .success());
            return;
        }
        crate::ops::enable_gpu();
        let _cpu = crate::compute::ComputePolicy::Cpu.enter_legacy_scope();
        assert!(GpuProjections::new(64, 32, 2).unwrap().is_none());
    }

    #[test]
    fn auto_falls_back_without_a_hardware_vulkan_device() {
        // A child process isolates both loader configuration and cached Vulkan state.
        if std::env::var_os("RMI_QI21_NO_DEVICE_TEST").is_none() {
            let missing = std::env::temp_dir()
                .join(format!("rmi-qi21-missing-icd-{}.json", std::process::id()));
            assert!(!missing.exists());
            assert!(std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    concat!(
                        module_path!(),
                        "::auto_falls_back_without_a_hardware_vulkan_device"
                    )
                    .split_once("::")
                    .unwrap()
                    .1,
                ])
                .env("RMI_QI21_NO_DEVICE_TEST", "1")
                .env("VK_DRIVER_FILES", &missing)
                .env("VK_ICD_FILENAMES", &missing)
                .status()
                .unwrap()
                .success());
            return;
        }
        {
            let _auto = crate::compute::ComputePolicy::Auto.enter_legacy_scope();
            assert!(GpuProjections::new(64, 32, 2).unwrap().is_none());
        }
        let _forced = crate::compute::ComputePolicy::Vulkan.enter_legacy_scope();
        assert!(GpuProjections::new(64, 32, 2).is_err());
    }

    #[test]
    #[ignore = "requires a hardware Vulkan device"]
    fn projection_tiles_and_rounded_bf16_match_cpu() {
        crate::ops::enable_gpu();
        let mut gpu = GpuProjections::new(64, 32, 2).unwrap().unwrap();
        let weights: Vec<u8> = (0..64 * 32)
            .flat_map(|i| crate::ops::f32_to_bf16((i % 23) as f32 / 23.0 - 0.5).to_le_bytes())
            .collect();
        let input: Vec<f32> = (0..65 * 64).map(|i| (i % 37) as f32 / 37.0).collect();
        let mut output = vec![f32::NAN; 65 * 32];
        gpu.run(
            &weights,
            GpuWeightFormat::BF16.with_mode(GpuMatmulMode::RoundedBf16),
            &input,
            &mut output,
            64,
            32,
        )
        .unwrap();
        for (i, &actual) in output.iter().enumerate() {
            let expected: f64 = (0..64)
                .map(|k| {
                    let j = (i % 32 * 64 + k) * 2;
                    let w =
                        crate::ops::bf16_to_f32(u16::from_le_bytes([weights[j], weights[j + 1]]));
                    let x =
                        crate::ops::bf16_to_f32(crate::ops::f32_to_bf16(input[i / 32 * 64 + k]));
                    f64::from(w * x)
                })
                .sum();
            assert!((actual - expected as f32).abs() <= 1e-5);
        }
        assert!(gpu
            .run(
                &weights,
                GpuWeightFormat::BF16,
                &[f32::NAN; 64],
                &mut [0.0; 32],
                64,
                32
            )
            .is_err());
        assert!(gpu
            .run(
                &weights,
                GpuWeightFormat::BF16,
                &[0.0; 63],
                &mut [0.0; 32],
                64,
                32
            )
            .is_err());
    }
}
