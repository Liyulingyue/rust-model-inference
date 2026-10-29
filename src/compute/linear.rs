//! Validated, model-borrowed linear execution with shared CPU activation preparation.
use super::{ComputeError, ComputePolicy, UsedBackend};
use crate::core::thread_pool::ComputePool;
use crate::ops::kernel::{PreparedRows, Weight};
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LinearMode {
    Prepared,
    Forward,
    F16Strict,
}

#[derive(Clone, Copy)]
pub(crate) struct LinearBinding<'model, 'weights> {
    pub weight: &'model Weight<'weights>,
    pub mode: LinearMode,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct LinearId {
    owner: u64,
    slot: usize,
}

pub(crate) struct LinearExecutor<'model, 'weights> {
    bindings: Vec<LinearBinding<'model, 'weights>>,
    owner: u64,
    policy: ComputePolicy,
    max_rows: usize,
    pool: Arc<ComputePool>,
    prepared: PreparedRows,
    #[cfg(feature = "vulkan")]
    runtime: Option<crate::vulkan::ops::BatchedLinearRuntime>,
    #[cfg(feature = "vulkan")]
    staging: Vec<f32>,
}

impl<'model, 'weights> LinearExecutor<'model, 'weights> {
    pub(crate) fn uses_vulkan(&self) -> bool {
        #[cfg(feature = "vulkan")]
        {
            self.runtime.is_some()
        }
        #[cfg(not(feature = "vulkan"))]
        {
            false
        }
    }

    pub(crate) fn new(
        policy: ComputePolicy,
        bindings: Vec<LinearBinding<'model, 'weights>>,
        max_rows: usize,
        pool: Arc<ComputePool>,
    ) -> Result<Self, ComputeError> {
        let policy = if crate::ops::scalar_mode() {
            ComputePolicy::Cpu
        } else {
            policy
        };
        if max_rows == 0 || bindings.is_empty() {
            return Err(invalid("linear requires bindings and nonzero rows"));
        }
        let mut max_n_in = 0;
        let mut max_n_out = 0;
        for binding in &bindings {
            let w = binding.weight;
            if w.n_in == 0 || w.n_out == 0 {
                return Err(invalid("linear dimensions must be nonzero"));
            }
            for width in [w.n_in, w.n_out] {
                checked_elements(max_rows, width)?;
            }
            let (block, bytes) = w.ggml_type.type_traits();
            if !w.n_in.is_multiple_of(block) {
                return Err(invalid("weight width is not block aligned"));
            }
            let len = w
                .n_in
                .checked_div(block)
                .and_then(|n| n.checked_mul(w.n_out))
                .and_then(|n| n.checked_mul(bytes))
                .ok_or_else(|| invalid("weight size overflow"))?;
            if w.kernel
                .weight_bytes()
                .is_some_and(|data| data.len() != len)
            {
                return Err(invalid("weight storage length mismatch"));
            }
            if binding.mode == LinearMode::F16Strict && !w.kernel.supports_f16_strict() {
                return Err(ComputeError::Unsupported(
                    "kernel does not implement F16Strict".into(),
                ));
            }
            if policy == ComputePolicy::Vulkan {
                gpu_compatible(*binding)?;
            }
            max_n_in = max_n_in.max(w.n_in);
            max_n_out = max_n_out.max(w.n_out);
        }
        static NEXT_OWNER: AtomicU64 = AtomicU64::new(1);
        let owner = NEXT_OWNER
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
            .map_err(|_| ComputeError::State("linear executor identity exhausted".into()))?;
        #[cfg(feature = "vulkan")]
        let runtime = if use_vulkan_for_outputs(policy, max_rows * max_n_out)
            && bindings.iter().any(|b| gpu_compatible(*b).is_ok())
        {
            match policy.context()? {
                Some(context) => {
                    match crate::vulkan::ops::BatchedLinearRuntime::new_with_output_slots(
                        context,
                        max_rows,
                        max_n_in,
                        max_n_out,
                        bindings
                            .len()
                            .checked_add(bindings.len().div_ceil(2))
                            .and_then(|n| n.checked_add(1))
                            .ok_or_else(|| invalid("binding count overflow"))?,
                        bindings.len().min(3),
                    ) {
                        Ok(runtime) => Some(runtime),
                        Err(error) if policy == ComputePolicy::Auto => {
                            log::info!("compute linear: CPU fallback: {error}");
                            None
                        }
                        Err(error) => return Err(error.into()),
                    }
                }
                None => None,
            }
        } else {
            None
        };
        #[cfg(not(feature = "vulkan"))]
        if policy == ComputePolicy::Vulkan {
            return Err(ComputeError::Unsupported(
                "build with --features vulkan".into(),
            ));
        }
        Ok(Self {
            bindings,
            owner,
            policy,
            max_rows,
            pool,
            prepared: PreparedRows::new(max_rows, max_n_in),
            #[cfg(feature = "vulkan")]
            runtime,
            #[cfg(feature = "vulkan")]
            staging: Vec::new(),
        })
    }

    pub(crate) fn id(&self, index: usize) -> Result<LinearId, ComputeError> {
        if index >= self.bindings.len() {
            return Err(invalid("linear binding index out of range"));
        }
        Ok(LinearId {
            owner: self.owner,
            slot: index,
        })
    }

    pub(crate) fn id_for(&self, weight: &Weight<'_>) -> Result<LinearId, ComputeError> {
        let slot = self
            .bindings
            .iter()
            .position(|b| std::ptr::eq(b.weight, weight))
            .ok_or_else(|| invalid("unbound linear weight"))?;
        self.id(slot)
    }

    pub(crate) fn run(
        &mut self,
        id: LinearId,
        input: &[f32],
        rows: usize,
        output: &mut [f32],
    ) -> Result<UsedBackend, ComputeError> {
        self.run_group([id], input, rows, [output])
            .map(|backends| backends[0])
    }

    pub(crate) fn run_group<const N: usize>(
        &mut self,
        ids: [LinearId; N],
        input: &[f32],
        rows: usize,
        mut outputs: [&mut [f32]; N],
    ) -> Result<[UsedBackend; N], ComputeError> {
        if N == 0 || rows == 0 || rows > self.max_rows {
            return Err(invalid("linear rows exceed capacity or group is empty"));
        }
        for (id, output) in ids.iter().zip(&outputs) {
            if id.owner != self.owner || id.slot >= self.bindings.len() {
                return Err(invalid("linear ID belongs to another executor"));
            }
            let weight = self.bindings[id.slot].weight;
            if input.len() != checked_elements(rows, weight.n_in)?
                || output.len() != checked_elements(rows, weight.n_out)?
            {
                return Err(invalid("linear input/output length mismatch"));
            }
        }
        if input.iter().any(|v| !v.is_finite()) {
            return Err(invalid("linear input contains non-finite values"));
        }
        let bindings = ids.map(|id| self.bindings[id.slot]);
        if std::env::var_os("RMI_COMPUTE_TRACE").is_some() {
            for binding in &bindings {
                eprintln!(
                    "compute scope=host_linear mode={:?} format={:?} shape={}x{} rows={rows}",
                    binding.mode,
                    binding.weight.ggml_type,
                    binding.weight.n_out,
                    binding.weight.n_in
                );
            }
        }
        #[cfg(feature = "vulkan")]
        if self.runtime.is_some()
            && !crate::core::thread_pool::gpu_matmul_disabled()
            && use_vulkan_for_outputs(
                self.policy,
                outputs.iter().map(|output| output.len()).max().unwrap_or(0),
            )
        {
            let attempt = self.run_gpu(&bindings, input, rows, &mut outputs);
            match attempt {
                Ok(()) => {
                    self.policy.trace("host_linear", UsedBackend::Vulkan, rows);
                    return Ok([UsedBackend::Vulkan; N]);
                }
                Err(error) if self.policy == ComputePolicy::Vulkan => return Err(error),
                Err(error) => {
                    log::info!("compute linear: CPU fallback: {error}");
                    if !matches!(error, ComputeError::Unsupported(_)) {
                        self.runtime = None;
                    }
                }
            }
        }
        if self.policy == ComputePolicy::Vulkan {
            return Err(ComputeError::Unsupported(
                "Vulkan execution is unavailable in this scope".into(),
            ));
        }
        // The legacy Q8 kernel must not dispatch GPU again from a CPU fallback.
        let _scope = ComputePolicy::Cpu.cpu_scope();
        let prepared = bindings.iter().filter(|b| b.mode == LinearMode::Prepared);
        let need_q8 = prepared.clone().any(|b| b.weight.needs_q8_0_activation());
        let need_q8k = prepared.clone().any(|b| b.weight.uses_q8_k());
        if prepared.count() != 0 {
            self.prepared
                .prepare(input, rows, bindings[0].weight.n_in, need_q8, need_q8k)
                .map_err(ComputeError::InvalidInput)?;
        }
        if bindings.iter().all(|b| b.mode == LinearMode::Prepared) {
            let mut slots = outputs.into_iter();
            self.prepared
                .matmul_group(
                    input,
                    bindings.map(|b| (b.weight, slots.next().unwrap())),
                    &self.pool,
                )
                .map_err(ComputeError::InvalidInput)?;
        } else {
            for (binding, output) in bindings.into_iter().zip(outputs) {
                let weight = binding.weight;
                match binding.mode {
                    LinearMode::Prepared => self
                        .prepared
                        .matmul(weight, input, output, &self.pool)
                        .map_err(ComputeError::InvalidInput)?,
                    mode => {
                        for (x, y) in input
                            .chunks_exact(weight.n_in)
                            .zip(output.chunks_exact_mut(weight.n_out))
                        {
                            match mode {
                                LinearMode::Forward => {
                                    weight.kernel.forward(x, y, weight.n_in, weight.n_out)
                                }
                                LinearMode::F16Strict => {
                                    if !weight.kernel.forward_f16_strict(
                                        x,
                                        y,
                                        weight.n_in,
                                        weight.n_out,
                                    ) {
                                        return Err(ComputeError::State(
                                            "kernel revoked F16Strict support".into(),
                                        ));
                                    }
                                }
                                LinearMode::Prepared => unreachable!(),
                            }
                        }
                    }
                }
            }
        }
        self.policy.trace("host_linear", UsedBackend::Cpu, rows);
        Ok([UsedBackend::Cpu; N])
    }

    #[cfg(feature = "vulkan")]
    fn run_gpu<const N: usize>(
        &mut self,
        bindings: &[LinearBinding<'model, 'weights>; N],
        input: &[f32],
        rows: usize,
        outputs: &mut [&mut [f32]; N],
    ) -> Result<(), ComputeError> {
        use crate::vulkan::ops::GpuMatmulSpec;
        let runtime = self.runtime.as_mut().unwrap();
        let mut total = 0usize;
        // Validate the entire group before the first upload or output write.
        for (index, (binding, output)) in bindings.iter().zip(outputs.iter()).enumerate() {
            gpu_compatible(*binding)?;
            let w = binding.weight;
            let bytes = w.kernel.weight_bytes().unwrap();
            for previous in &bindings[..index] {
                if previous
                    .weight
                    .kernel
                    .weight_bytes()
                    .is_some_and(|other| std::ptr::eq(other, bytes))
                    && previous.weight.ggml_type != w.ggml_type
                {
                    return Err(ComputeError::Unsupported(
                        "aliased weights have different Vulkan formats".into(),
                    ));
                }
            }
            runtime.validate(
                w.kernel.weight_bytes().unwrap(),
                GpuMatmulSpec::prepared(w)?,
                input.len(),
                rows,
                w.n_in,
                w.n_out,
                output.len(),
            )?;
            total = total
                .checked_add(output.len())
                .ok_or_else(|| invalid("group output size overflow"))?;
        }
        if N > 1 {
            let specs: Vec<_> = bindings
                .iter()
                .map(|binding| {
                    Ok((
                        binding.weight.kernel.weight_bytes().ok_or_else(|| {
                            ComputeError::Unsupported("kernel has no immutable storage view".into())
                        })?,
                        GpuMatmulSpec::prepared(binding.weight)?,
                        binding.weight.n_in,
                        binding.weight.n_out,
                    ))
                })
                .collect::<Result<_, ComputeError>>()?;
            runtime.matmul_rows_group(&specs, input, rows, &mut outputs[..])?;
            if outputs
                .iter()
                .any(|output| output.iter().any(|v| !v.is_finite()))
            {
                return Err(ComputeError::Device(
                    "linear produced non-finite output".into(),
                ));
            }
            return Ok(());
        }
        self.staging.resize(total, 0.0);
        let mut offset = 0;
        for (binding, output) in bindings.iter().zip(outputs.iter()) {
            let w = binding.weight;
            runtime.matmul_rows(
                w.kernel.weight_bytes().unwrap(),
                GpuMatmulSpec::prepared(w)?,
                input,
                rows,
                w.n_in,
                w.n_out,
                &mut self.staging[offset..offset + output.len()],
            )?;
            offset += output.len();
        }
        if self.staging.iter().any(|v| !v.is_finite()) {
            return Err(ComputeError::Device(
                "linear produced non-finite output".into(),
            ));
        }
        let mut offset = 0;
        for output in outputs {
            output.copy_from_slice(&self.staging[offset..offset + output.len()]);
            offset += output.len();
        }
        Ok(())
    }
}

#[cfg(feature = "vulkan")]
fn use_vulkan_for_outputs(policy: ComputePolicy, outputs: usize) -> bool {
    match policy {
        ComputePolicy::Cpu => false,
        ComputePolicy::Auto => outputs >= crate::ops::kernel::vulkan::min_gpu_matmul_outputs(),
        ComputePolicy::Vulkan => true,
    }
}

fn invalid(message: &str) -> ComputeError {
    ComputeError::InvalidInput(message.into())
}
fn checked_elements(rows: usize, width: usize) -> Result<usize, ComputeError> {
    let count = rows
        .checked_mul(width)
        .ok_or_else(|| invalid("linear shape overflow"))?;
    if count > isize::MAX as usize / std::mem::size_of::<f32>() {
        return Err(invalid("linear allocation size overflow"));
    }
    Ok(count)
}
fn gpu_compatible(binding: LinearBinding<'_, '_>) -> Result<(), ComputeError> {
    use crate::GGMLType;
    if binding.weight.ggml_type == GGMLType::F16
        && crate::ops::f16_uses_neon_accumulators(binding.weight.n_in)
        && !binding.weight.n_in.is_multiple_of(32)
    {
        return Err(ComputeError::Unsupported(
            "F16 NEON reduction with an F64 tail requires CPU".into(),
        ));
    }
    if binding.mode == LinearMode::F16Strict
        || (binding.mode == LinearMode::Forward && binding.weight.ggml_type == GGMLType::F16)
    {
        return Err(ComputeError::Unsupported(
            "Vulkan has no matching F16 activation/reduction contract".into(),
        ));
    }
    if binding.weight.kernel.weight_bytes().is_none() {
        return Err(ComputeError::Unsupported(
            "kernel has no immutable storage view".into(),
        ));
    }
    if !matches!(
        binding.weight.ggml_type,
        GGMLType::F32
            | GGMLType::F16
            | GGMLType::BF16
            | GGMLType::Q8_0
            | GGMLType::Q4_0
            | GGMLType::Q4_1
            | GGMLType::Q4K
            | GGMLType::Q5K
            | GGMLType::Q6K
    ) {
        return Err(ComputeError::Unsupported(
            "weight format has no Vulkan kernel".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::kernel::{F16Weight, QuantizedTensor};

    fn pool() -> Arc<ComputePool> {
        Arc::new(ComputePool::new(2))
    }

    #[test]
    fn tiny_auto_executor_stays_cpu_across_row_sizes() {
        #[cfg(feature = "vulkan")]
        if crate::ops::kernel::vulkan::min_gpu_matmul_outputs() <= 6 {
            return;
        }
        let a = Weight::from_quantized(QuantizedTensor::F32 {
            data: vec![2.0, 3.0],
            n_in: 2,
            n_out: 1,
        });
        let b = Weight::from_quantized(QuantizedTensor::F32 {
            data: vec![1.0, -1.0, 4.0, 0.5],
            n_in: 2,
            n_out: 2,
        });
        let mut executor = LinearExecutor::new(
            ComputePolicy::Auto,
            [&a, &b]
                .map(|weight| LinearBinding {
                    weight,
                    mode: LinearMode::Prepared,
                })
                .to_vec(),
            3,
            pool(),
        )
        .unwrap();
        assert!(!executor.uses_vulkan());
        let ids = [executor.id(0).unwrap(), executor.id(1).unwrap()];
        let input = [4.0, 5.0, 1.0, 2.0, -2.0, 1.0];
        for rows in [1, 3, 2, 1] {
            let mut out_a = vec![f32::NAN; rows];
            let mut out_b = vec![f32::NAN; rows * 2];
            assert_eq!(
                executor
                    .run_group(ids, &input[..rows * 2], rows, [&mut out_a, &mut out_b])
                    .unwrap(),
                [UsedBackend::Cpu; 2],
            );
            assert_eq!(out_a, [23.0, 8.0, -1.0][..rows]);
            assert_eq!(out_b, [-1.0, 18.5, -1.0, 5.0, -3.0, -7.5][..rows * 2]);
        }
    }

    #[cfg(feature = "vulkan")]
    #[test]
    fn auto_gpu_threshold_preserves_forced_vulkan_and_configured_boundary() {
        let minimum = crate::ops::kernel::vulkan::min_gpu_matmul_outputs();
        assert!(!use_vulkan_for_outputs(ComputePolicy::Cpu, usize::MAX));
        assert!(use_vulkan_for_outputs(ComputePolicy::Vulkan, 1));
        assert!(use_vulkan_for_outputs(ComputePolicy::Auto, minimum));
        if minimum > 0 {
            assert!(!use_vulkan_for_outputs(ComputePolicy::Auto, minimum - 1));
        }
    }

    #[cfg(feature = "vulkan")]
    #[test]
    #[ignore = "requires a Vulkan device"]
    fn auto_gpu_threshold_interleaves_rows_and_mixed_group_sizes() {
        let minimum = crate::ops::kernel::vulkan::min_gpu_matmul_outputs();
        if minimum <= 128 || crate::ops::scalar_mode() {
            return;
        }
        let rows = minimum.div_ceil(128);
        let big = Weight::from_quantized(QuantizedTensor::F32 {
            data: [1.0, 2.0, 3.0, 4.0].repeat(128),
            n_in: 4,
            n_out: 128,
        });
        let small = Weight::from_quantized(QuantizedTensor::F32 {
            data: vec![1.0, 0.0, 0.0, 1.0],
            n_in: 4,
            n_out: 1,
        });
        let mut executor = LinearExecutor::new(
            ComputePolicy::Auto,
            [&big, &small]
                .map(|weight| LinearBinding {
                    weight,
                    mode: LinearMode::Prepared,
                })
                .to_vec(),
            rows,
            pool(),
        )
        .unwrap();
        assert!(executor.uses_vulkan());
        let context = crate::ops::float::shared_vulkan_context().unwrap();
        let ids = [executor.id(0).unwrap(), executor.id(1).unwrap()];
        for actual_rows in [1, rows, 1, rows] {
            let before = context.submission_count();
            let mut out_big = vec![f32::NAN; actual_rows * 128];
            let mut out_small = vec![f32::NAN; actual_rows];
            let expected = if actual_rows == rows {
                UsedBackend::Vulkan
            } else {
                UsedBackend::Cpu
            };
            assert_eq!(
                executor
                    .run_group(
                        ids,
                        &vec![1.0; actual_rows * 4],
                        actual_rows,
                        [&mut out_big, &mut out_small],
                    )
                    .unwrap(),
                [expected; 2]
            );
            assert_eq!(out_big, vec![10.0; actual_rows * 128]);
            assert_eq!(out_small, vec![2.0; actual_rows]);
            if expected == UsedBackend::Cpu {
                assert_eq!(context.submission_count(), before);
            } else {
                assert!(context.submission_count() > before);
            }
            assert!(executor.uses_vulkan());
        }
        let before = context.submission_count();
        let mut output = vec![f32::NAN; rows];
        assert_eq!(
            executor
                .run(ids[1], &vec![1.0; rows * 4], rows, &mut output)
                .unwrap(),
            UsedBackend::Cpu
        );
        assert_eq!(output, vec![2.0; rows]);
        assert_eq!(context.submission_count(), before);
    }

    #[test]
    #[ignore = "run in a fresh process with RMI_SCALAR=1"]
    fn scalar_linear_selection_does_not_initialize_a_device() {
        assert!(crate::ops::scalar_mode());
        let weight = Weight::from_quantized(QuantizedTensor::F32 {
            data: vec![2.0, 3.0],
            n_in: 2,
            n_out: 1,
        });
        for policy in [ComputePolicy::Auto, ComputePolicy::Vulkan] {
            let mut executor = LinearExecutor::new(
                policy,
                vec![LinearBinding {
                    weight: &weight,
                    mode: LinearMode::Prepared,
                }],
                1,
                pool(),
            )
            .unwrap();
            assert!(!executor.uses_vulkan());
            let mut output = [f32::NAN];
            assert_eq!(
                executor
                    .run(executor.id(0).unwrap(), &[4.0, 5.0], 1, &mut output)
                    .unwrap(),
                UsedBackend::Cpu
            );
            assert_eq!(output, [23.0]);
        }
    }

    #[cfg(feature = "vulkan")]
    #[test]
    #[ignore = "requires a Vulkan device"]
    fn rounded_bf16_prepared_linear_matches_cpu_with_tails() {
        const WIDTH: usize = 261;
        const OUTPUTS: usize = 65;
        const ROWS: usize = 3;
        let bytes: Vec<u8> = (0..WIDTH * OUTPUTS)
            .flat_map(|index| {
                crate::ops::f32_to_bf16(((index * 17 % 97) as f32 - 48.0) / 19.0).to_le_bytes()
            })
            .collect();
        let rounded = Weight {
            kernel: Box::new(crate::ops::kernel::bf16::BF16Kernel::with_bf16_input(
                &bytes,
            )),
            ggml_type: crate::GGMLType::BF16,
            n_in: WIDTH,
            n_out: OUTPUTS,
        };
        let input: Vec<_> = (0..ROWS * WIDTH)
            .map(|index| ((index * 43 % 191) as f32 - 95.0) / 37.0)
            .collect();
        let run = |policy| {
            let mut executor = LinearExecutor::new(
                policy,
                vec![LinearBinding {
                    weight: &rounded,
                    mode: LinearMode::Prepared,
                }],
                ROWS,
                pool(),
            )
            .unwrap();
            let mut output = vec![0.0; ROWS * OUTPUTS];
            let backend = executor
                .run(executor.id(0).unwrap(), &input, ROWS, &mut output)
                .unwrap();
            (backend, output)
        };
        let (cpu_backend, expected) = run(ComputePolicy::Cpu);
        let (gpu_backend, actual) = run(ComputePolicy::Vulkan);
        assert_eq!(cpu_backend, UsedBackend::Cpu);
        assert_eq!(gpu_backend, UsedBackend::Vulkan);
        assert_eq!(
            actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        let raw = crate::ops::kernel::bf16::BF16Kernel::new(&bytes);
        let mut unrounded = vec![0.0; OUTPUTS];
        crate::ops::kernel::Kernel::forward(&raw, &input[..WIDTH], &mut unrounded, WIDTH, OUTPUTS);
        assert!(unrounded
            .iter()
            .zip(&expected)
            .any(|(a, b)| a.to_bits() != b.to_bits()));
    }

    #[test]
    fn f16_linear_modes_preserve_existing_rounding() {
        let bytes: Vec<_> = (0..64)
            .flat_map(|i| crate::ops::f32_to_f16((i as f32 - 31.0) / 17.0).to_le_bytes())
            .collect();
        let weight = Weight::from_quantized(QuantizedTensor::F16(F16Weight {
            bytes: &bytes,
            n_in: 32,
            n_out: 2,
        }));
        let input: Vec<_> = (0..96).map(|i| 1.0 + (i as f32 - 47.0) / 65536.0).collect();
        for mode in [
            LinearMode::Prepared,
            LinearMode::Forward,
            LinearMode::F16Strict,
        ] {
            let mut executor = LinearExecutor::new(
                ComputePolicy::Cpu,
                vec![LinearBinding {
                    weight: &weight,
                    mode,
                }],
                3,
                pool(),
            )
            .unwrap();
            let mut actual = [f32::NAN; 6];
            let mut expected = [0.0; 6];
            for (input, output) in input.chunks_exact(32).zip(expected.chunks_exact_mut(2)) {
                match mode {
                    LinearMode::Prepared => {
                        weight
                            .kernel
                            .forward_prepared(input, &[], &[], None, output, 32, 2, 0, 1)
                    }
                    LinearMode::Forward => weight.kernel.forward(input, output, 32, 2),
                    LinearMode::F16Strict => {
                        assert!(weight.kernel.forward_f16_strict(input, output, 32, 2))
                    }
                }
            }
            assert_eq!(
                executor
                    .run(executor.id(0).unwrap(), &input, 3, &mut actual)
                    .unwrap(),
                UsedBackend::Cpu
            );
            assert_eq!(
                actual.map(f32::to_bits),
                expected.map(f32::to_bits),
                "{mode:?}"
            );
        }
    }

    #[test]
    fn linear_executor_rejects_invalid_shape_before_dispatch() {
        let weight = Weight::from_quantized(QuantizedTensor::F32 {
            data: vec![2.0; 6],
            n_in: 3,
            n_out: 2,
        });
        let bindings = || {
            vec![LinearBinding {
                weight: &weight,
                mode: LinearMode::Prepared,
            }]
        };
        let mut executor = LinearExecutor::new(ComputePolicy::Cpu, bindings(), 3, pool()).unwrap();
        let other = LinearExecutor::new(ComputePolicy::Cpu, bindings(), 3, pool()).unwrap();
        let id = executor.id(0).unwrap();
        let mut output = [123.0; 2];
        for (id, rows, input) in [
            (id, 0, &[][..]),
            (id, usize::MAX, &[][..]),
            (id, 1, &[1.0][..]),
            (other.id(0).unwrap(), 1, &[1.0; 3][..]),
        ] {
            assert!(executor.run(id, input, rows, &mut output).is_err());
            assert_eq!(output, [123.0; 2]);
        }
        let mut short = [456.0];
        assert!(executor
            .run_group([id, id], &[1.0; 3], 1, [&mut output, &mut short])
            .is_err());
        assert_eq!(output, [123.0; 2]);
        assert_eq!(short, [456.0]);
        assert!(executor.id(1).is_err());
        assert!(LinearExecutor::new(ComputePolicy::Cpu, bindings(), usize::MAX, pool()).is_err());
        executor.run(id, &[1.0; 3], 1, &mut output).unwrap();
        assert_eq!(output, [6.0; 2]);
    }

    #[test]
    fn linear_executor_preserves_cpu_modes_and_grouped_quantization() {
        use crate::ops::kernel::Kernel;
        use std::sync::Mutex;
        struct Observed(Arc<Mutex<Vec<usize>>>);
        impl Kernel for Observed {
            fn forward_prequantized(
                &self,
                q8: &[u8],
                _: &[f32],
                out: &mut [f32],
                _: usize,
                _: usize,
                ith: usize,
                _: usize,
            ) {
                if ith == 0 {
                    self.0.lock().unwrap().push(q8.as_ptr() as usize);
                    out[0] = q8[0] as i8 as f32;
                }
            }
        }
        let seen = Arc::new(Mutex::new(Vec::new()));
        let weight = Weight {
            kernel: Box::new(Observed(seen.clone())),
            ggml_type: crate::GGMLType::Q8_0,
            n_in: 32,
            n_out: 1,
        };
        let mut executor = LinearExecutor::new(
            ComputePolicy::Cpu,
            vec![LinearBinding {
                weight: &weight,
                mode: LinearMode::Prepared,
            }],
            3,
            pool(),
        )
        .unwrap();
        let id = executor.id(0).unwrap();
        let (mut a, mut b) = ([0.0; 3], [0.0; 3]);
        executor
            .run_group([id, id], &[1.0; 96], 3, [&mut a, &mut b])
            .unwrap();
        assert_eq!(a, [127.0; 3]);
        assert_eq!(a, b);
        let seen = seen.lock().unwrap();
        assert_eq!(seen.len(), 6);
        for pair in seen.chunks_exact(2) {
            assert_eq!(pair[0], pair[1]);
        }
    }

    #[test]
    fn linear_executor_auto_falls_back_without_partial_output() {
        let weight = Weight::from_quantized(QuantizedTensor::F32 {
            data: vec![2.0, 3.0],
            n_in: 2,
            n_out: 1,
        });
        let mut executor = LinearExecutor::new(
            ComputePolicy::Auto,
            vec![LinearBinding {
                weight: &weight,
                mode: LinearMode::Forward,
            }],
            1,
            pool(),
        )
        .unwrap();
        let mut out = [123.0];
        executor
            .run(executor.id(0).unwrap(), &[4.0, 5.0], 1, &mut out)
            .unwrap();
        assert_eq!(out, [23.0]);
        assert!(LinearExecutor::new(
            ComputePolicy::Cpu,
            vec![LinearBinding {
                weight: &weight,
                mode: LinearMode::F16Strict
            }],
            1,
            pool()
        )
        .is_err());
    }

    #[cfg(feature = "vulkan")]
    #[test]
    #[ignore = "requires a Vulkan device"]
    fn f16_prepared_device_preserves_half_accumulation_bits() {
        if !crate::ops::has_neon() {
            return;
        }
        for n_in in [32, 256, 512, 2048] {
            let bytes: Vec<_> = (0..n_in * 9)
                .flat_map(|i| {
                    crate::ops::f32_to_f16(((i * 17 % 101) as f32 - 50.0) / 47.0).to_le_bytes()
                })
                .collect();
            let weight = Weight::from_quantized(QuantizedTensor::F16(F16Weight {
                bytes: &bytes,
                n_in,
                n_out: 9,
            }));
            let bindings = vec![LinearBinding {
                weight: &weight,
                mode: LinearMode::Prepared,
            }];
            let mut cpu =
                LinearExecutor::new(ComputePolicy::Cpu, bindings.clone(), 64, pool()).unwrap();
            let mut gpu = LinearExecutor::new(ComputePolicy::Vulkan, bindings, 64, pool()).unwrap();
            for rows in [1, 3, 64] {
                let input: Vec<_> = (0..rows * n_in)
                    .map(|i| ((i * 29 % 251) as f32 - 125.0) / 97.0)
                    .collect();
                let mut expected = vec![0.0; rows * 9];
                let mut actual = expected.clone();
                cpu.run(cpu.id(0).unwrap(), &input, rows, &mut expected)
                    .unwrap();
                assert_eq!(
                    gpu.run(gpu.id(0).unwrap(), &input, rows, &mut actual)
                        .unwrap(),
                    UsedBackend::Vulkan
                );
                for (i, (&a, &b)) in actual.iter().zip(&expected).enumerate() {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "n_in={n_in} rows={rows} index={i}: {a} vs {b}"
                    );
                }
            }
        }
    }

    #[cfg(feature = "vulkan")]
    #[test]
    #[ignore = "requires a Vulkan device"]
    fn linear_device_failure_falls_back_only_for_auto() {
        let n_out = crate::ops::kernel::vulkan::min_gpu_matmul_outputs().max(1);
        let weight = Weight::from_quantized(QuantizedTensor::F32 {
            data: [2.0, 3.0].repeat(n_out),
            n_in: 2,
            n_out,
        });
        let context = Box::leak(Box::new(crate::vulkan::VulkanContext::new().unwrap()));
        for policy in [ComputePolicy::Auto, ComputePolicy::Vulkan] {
            let mut executor = LinearExecutor::new(
                ComputePolicy::Cpu,
                vec![LinearBinding {
                    weight: &weight,
                    mode: LinearMode::Prepared,
                }],
                1,
                pool(),
            )
            .unwrap();
            executor.policy = policy;
            let mut runtime =
                crate::vulkan::ops::BatchedLinearRuntime::new(context, 1, 2, n_out, 2).unwrap();
            runtime.begin_commands = |_| Err(crate::vulkan::VulkanError::Timeout);
            executor.runtime = Some(runtime);
            let mut out = vec![123.0; n_out];
            let before = context.submission_count();
            let result = executor.run(executor.id(0).unwrap(), &[4.0, 5.0], 1, &mut out);
            if policy == ComputePolicy::Auto {
                assert_eq!(result.unwrap(), UsedBackend::Cpu);
                assert_eq!(out, vec![23.0; n_out]);
                assert!(executor.runtime.is_none());
            } else {
                assert!(matches!(result, Err(ComputeError::Device(_))));
                assert_eq!(out, vec![123.0; n_out]);
            }
            assert_eq!(context.submission_count(), before);
        }
    }
}
