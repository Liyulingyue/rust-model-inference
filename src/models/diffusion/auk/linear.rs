use crate::compute::{
    linear::{LinearBinding, LinearExecutor, LinearId, LinearMode},
    ComputePolicy,
};
use crate::ops::kernel::Weight;
use crate::ComputePool;
use std::{cell::RefCell, collections::HashMap, sync::Arc};

/// Borrowed weight bindings and their uploads live only for this render.
pub(super) struct AukLinearSession<'model, 'weights> {
    executor: Option<RefCell<LinearExecutor<'model, 'weights>>>,
    ids: HashMap<&'model str, LinearId>,
    scaled: RefCell<Vec<f32>>,
}

impl<'model, 'weights> AukLinearSession<'model, 'weights> {
    pub(super) fn new(
        weights: &'model [(String, Weight<'weights>)],
        policy: ComputePolicy,
        pool: Arc<ComputePool>,
    ) -> Result<Self, String> {
        let executor = if weights.is_empty() {
            None
        } else {
            Some(
                LinearExecutor::new(
                    policy,
                    weights
                        .iter()
                        .map(|(_, weight)| LinearBinding {
                            weight,
                            mode: LinearMode::Prepared,
                        })
                        .collect(),
                    1,
                    pool,
                )
                .map_err(|e| e.to_string())?,
            )
        };
        let ids = match &executor {
            Some(executor) => weights
                .iter()
                .enumerate()
                .map(|(index, (name, _))| (name.as_str(), executor.id(index).unwrap()))
                .collect(),
            None => HashMap::new(),
        };
        Ok(Self {
            executor: executor.map(RefCell::new),
            ids,
            scaled: RefCell::new(Vec::new()),
        })
    }

    pub(super) fn run(
        &self,
        name: &str,
        input: &[f32],
        output: &mut [f32],
        scale: f32,
    ) -> Result<bool, String> {
        let Some(&id) = self.ids.get(name) else {
            return Ok(false);
        };
        if !scale.is_finite() || scale == 0.0 {
            return Err("invalid AuK linear scale".into());
        }
        let mut executor = self.executor.as_ref().unwrap().borrow_mut();
        if scale == 1.0 {
            executor
                .run(id, input, 1, output)
                .map_err(|e| e.to_string())?;
        } else {
            let mut scaled = self.scaled.borrow_mut();
            scaled.resize(input.len(), 0.0);
            for (to, from) in scaled.iter_mut().zip(input) {
                *to = *from * scale;
            }
            executor
                .run(id, &scaled, 1, output)
                .map_err(|e| e.to_string())?;
            let inverse = scale.recip();
            for value in output {
                *value *= inverse;
            }
        }
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::kernel::{F16Weight, QuantizedTensor};

    #[test]
    fn auk_f16_sessions_keep_weights_and_activation_scaling_independent() {
        check_sessions(crate::compute::ComputePolicy::Cpu);
    }

    #[cfg(feature = "vulkan")]
    #[test]
    #[ignore = "requires a Vulkan device"]
    fn auk_f16_device_sessions_do_not_reuse_dropped_model_weights() {
        check_sessions(crate::compute::ComputePolicy::Vulkan);
    }

    fn check_sessions(policy: crate::compute::ComputePolicy) {
        let pool = std::sync::Arc::new(crate::ComputePool::new(2));
        for value in [2.0, -3.0] {
            let bytes: Vec<_> = [value, value + 0.5]
                .into_iter()
                .flat_map(|v| crate::ops::f32_to_f16(v).to_le_bytes())
                .collect();
            let weights = vec![(
                "projection".into(),
                crate::ops::kernel::Weight::from_quantized(QuantizedTensor::F16(F16Weight {
                    bytes: &bytes,
                    n_in: 2,
                    n_out: 1,
                })),
            )];
            let session = AukLinearSession::new(&weights, policy, pool.clone()).unwrap();
            assert_eq!(
                session.executor.as_ref().unwrap().borrow().uses_vulkan(),
                policy == crate::compute::ComputePolicy::Vulkan
            );
            for scale in [1.0, 4.0, 0.25] {
                let input = [1.0003, -0.12345];
                let mut expected = [0.0];
                let mut actual = [123.0];
                crate::ops::kernel::f16::F16Kernel::new(&bytes).forward_scaled(
                    &input,
                    &mut expected,
                    2,
                    1,
                    scale,
                    &mut Vec::new(),
                );
                assert!(session
                    .run("projection", &input, &mut actual, scale)
                    .unwrap());
                assert_eq!(actual.map(f32::to_bits), expected.map(f32::to_bits));
            }
            assert!(!session.run("missing", &[1.0; 2], &mut [0.0], 1.0).unwrap());
        }
    }
}
