use super::config::HyMoeConfig;
use super::weights::HyMoe;
use crate::compute::linear::{LinearBinding, LinearExecutor, LinearMode};

fn route(
    logits: &[f32],
    bias: &[f32],
    used: usize,
    scale: f32,
) -> Result<(Vec<usize>, Vec<f32>), String> {
    if logits.is_empty()
        || logits.len() != bias.len()
        || used == 0
        || used > logits.len()
        || !scale.is_finite()
        || scale <= 0.0
        || logits.iter().chain(bias).any(|v| !v.is_finite())
    {
        return Err("invalid Hy-SimulMT router dimensions, values or scale".into());
    }
    let probs: Vec<f32> = logits.iter().map(|v| 1.0 / (1.0 + (-v).exp())).collect();
    let mut ids: Vec<usize> = (0..logits.len()).collect();
    ids.sort_by(|&a, &b| {
        (probs[b] + bias[b])
            .total_cmp(&(probs[a] + bias[a]))
            .then(a.cmp(&b))
    });
    ids.truncate(used);
    let sum = (ids.iter().map(|&i| f64::from(probs[i])).sum::<f64>() as f32).max(6.103515625e-5);
    let weights = ids.iter().map(|&i| (probs[i] / sum) * scale).collect();
    Ok((ids, weights))
}

impl HyMoe {
    pub(super) fn bindings(&self) -> Vec<LinearBinding<'_, 'static>> {
        self.experts
            .iter()
            .chain([&self.shared])
            .flat_map(|e| [&e.gate, &e.up, &e.down])
            .map(|weight| LinearBinding {
                weight,
                mode: LinearMode::Prepared,
            })
            .collect()
    }

    pub(super) fn forward(
        &self,
        input: &[f32],
        cfg: &HyMoeConfig,
        executor: &mut LinearExecutor<'_, 'static>,
        output: &mut [f32],
    ) -> Result<(), String> {
        let logits: Vec<f32> = self
            .router
            .chunks_exact(input.len())
            .map(|row| crate::ops::dot::dot_f32_ggml(input, row, input.len()))
            .collect();
        let (ids, probabilities) = route(&logits, &self.bias, cfg.expert_used_count, cfg.scale)?;
        output.fill(0.0);
        for (id, probability) in ids
            .into_iter()
            .map(Some)
            .zip(probabilities)
            .chain(std::iter::once((None, 1.0)))
        {
            let expert = id.map_or(&self.shared, |id| &self.experts[id]);
            let mut gate = vec![0.0; expert.gate.n_out];
            let mut up = vec![0.0; expert.up.n_out];
            let ids = [
                executor.id_for(&expert.gate).map_err(|e| e.to_string())?,
                executor.id_for(&expert.up).map_err(|e| e.to_string())?,
            ];
            executor
                .run_group(ids, input, 1, [&mut gate, &mut up])
                .map_err(|e| e.to_string())?;
            crate::ops::silu_mul_approx_inplace(&gate, &mut up);
            let mut value = vec![0.0; output.len()];
            let id = executor.id_for(&expert.down).map_err(|e| e.to_string())?;
            executor
                .run(id, &up, 1, &mut value)
                .map_err(|e| e.to_string())?;
            for (out, v) in output.iter_mut().zip(value) {
                *out += probability * v;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::weights::Expert;
    use super::*;
    use crate::compute::ComputePolicy;
    use crate::core::thread_pool::ComputePool;
    use crate::ops::kernel::{QuantizedTensor, Weight};
    use std::sync::Arc;

    #[test]
    fn hy_router_bias_only_selects_experts() {
        let (ids, weights) = route(
            &[0.0, 3.0f32.ln(), -3.0f32.ln()],
            &[0.0, 0.0, 1.0],
            2,
            2.826,
        )
        .unwrap();
        assert_eq!(ids, [2, 1]);
        for (actual, expected) in weights.iter().zip([0.7065, 2.1195]) {
            assert!((actual - expected).abs() < 3e-7);
        }
        assert_eq!(
            route(&[0.0; 4], &[0.0; 4], 2, 1.0).unwrap(),
            (vec![0, 1], vec![0.5, 0.5])
        );
        assert_eq!(route(&[-1000.0], &[0.0], 1, 2.826).unwrap().1, [0.0]);
    }

    #[test]
    fn hy_routed_plus_ungated_shared_expert() {
        let w =
            |data, n_in, n_out| Weight::from_quantized(QuantizedTensor::F32 { data, n_in, n_out });
        let moe = HyMoe {
            router: vec![0.0; 6],
            bias: vec![0.0, 0.0, 1.0],
            experts: (0..3)
                .map(|i| Expert {
                    gate: w(vec![1., 0.], 2, 1),
                    up: w(vec![1., 0.], 2, 1),
                    down: w(vec![(i + 1) as f32, (i + 2) as f32], 1, 2),
                })
                .collect(),
            shared: Expert {
                gate: w(vec![0., 1.], 2, 1),
                up: w(vec![1., 0.], 2, 1),
                down: w(vec![1., -1.], 1, 2),
            },
        };
        let cfg = HyMoeConfig {
            expert_count: 3,
            expert_used_count: 2,
            expert_ffn: 1,
            shared_expert_ffn: 1,
            scale: 2.826,
        };
        let mut executor = LinearExecutor::new(
            ComputePolicy::Cpu,
            moe.bindings(),
            1,
            Arc::new(ComputePool::new(2)),
        )
        .unwrap();
        let mut out = [0.; 2];
        moe.forward(&[1., 2.], &cfg, &mut executor, &mut out)
            .unwrap();
        let sigmoid = 1.0 / (1.0 + (-1.0f32).exp());
        let shared = 2.0 / (1.0 + (-2.0f32).exp());
        assert!((out[0] - (5.652 * sigmoid + shared)).abs() < 2e-6);
        assert!((out[1] - (8.478 * sigmoid - shared)).abs() < 2e-6);
    }

    #[test]
    fn hy_moe_swiglu_matches_pinned_ggml_fixture() {
        let diagonal = |values: [f32; 4]| {
            let mut data = vec![0.0; 16];
            for (i, value) in values.into_iter().enumerate() {
                data[i * 4 + i] = value;
            }
            Weight::from_quantized(QuantizedTensor::F32 {
                data,
                n_in: 4,
                n_out: 4,
            })
        };
        let moe = HyMoe {
            router: vec![0.0; 4],
            bias: vec![0.0],
            experts: vec![Expert {
                gate: diagonal(
                    [0xbf46_e0d2, 0xbf47_ebea, 0xbeee_8b6b, 0xbe1b_692b].map(f32::from_bits),
                ),
                up: diagonal(
                    [0xbdbd_ab3d, 0xbdf0_16eb, 0x3e08_064c, 0xbf87_c095].map(f32::from_bits),
                ),
                down: diagonal([1.0; 4]),
            }],
            shared: Expert {
                gate: diagonal([0.0; 4]),
                up: diagonal([0.0; 4]),
                down: diagonal([0.0; 4]),
            },
        };
        let cfg = HyMoeConfig {
            expert_count: 1,
            expert_used_count: 1,
            expert_ffn: 4,
            shared_expert_ffn: 4,
            scale: 1.0,
        };
        let mut executor = LinearExecutor::new(
            ComputePolicy::Cpu,
            moe.bindings(),
            1,
            Arc::new(ComputePool::new(1)),
        )
        .unwrap();
        let mut out = [0.0; 4];
        moe.forward(&[1.0; 4], &cfg, &mut executor, &mut out)
            .unwrap();
        let expected = [0x3cb9_a7c4, 0x3ceb_9536, 0xbcc3_7dcb, 0x3d98_56f8].map(f32::from_bits);
        for (got, expected) in out.into_iter().zip(expected) {
            assert!(
                (got - expected).abs() < 1e-7,
                "got={got}, expected={expected}"
            );
        }
    }

    #[test]
    fn hy_router_rejects_invalid_dimensions_and_nonfinite_values() {
        for (logits, bias, used, scale) in [
            (&[][..], &[][..], 1, 1.0),
            (&[0.0][..], &[][..], 1, 1.0),
            (&[0.0][..], &[0.0][..], 2, 1.0),
            (&[f32::NAN][..], &[0.0][..], 1, 1.0),
            (&[0.0][..], &[f32::INFINITY][..], 1, 1.0),
            (&[0.0][..], &[0.0][..], 1, f32::NAN),
        ] {
            assert!(route(logits, bias, used, scale).is_err());
        }
    }
}
