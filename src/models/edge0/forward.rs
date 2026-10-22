//! Edge0 MoE computation and model forward entry points.

use super::weights::{Edge0Model, Edge0MoeWeights};
use crate::core::scratchpad::KvCache;
use crate::core::thread_pool::ComputePool;
use crate::models::qwen35::trunk::session::HybridTrunkModel;
use crate::models::qwen35::trunk::HybridTrunk;
use crate::models::qwen35::Qwen35Scratchpad;
use crate::ops::kernel::PreparedRows;
use crate::ops::{silu_inplace, sum_sq_f32};

/// Edge0's Q/K path uses RMSNorm with additive epsilon before its head scaling.
pub(crate) fn normalize_recurrent_qk(values: &mut [f32], eps: f32, factor: f32) {
    let sum = sum_sq_f32(values);
    let mean = (sum / values.len() as f64) as f32;
    let scale = 1.0f32 / (mean + eps).sqrt();
    for value in values {
        *value = (*value * scale) * factor;
    }
}

#[cfg(test)]
#[test]
fn recurrent_qk_norm_adds_epsilon_to_mean_square() {
    let mut values = [0.02f32, 0.01];
    normalize_recurrent_qk(&mut values, 1e-6, 0.5);
    assert_eq!(values.map(f32::to_bits), [0x3f2195f4, 0x3ea195f4]);
}

impl<'a> Edge0Model<'a> {
    pub(crate) fn forward_at(
        &mut self,
        n_tokens: usize,
        base_position: usize,
        kv_cache: &mut KvCache,
        scratch: &mut Qwen35Scratchpad,
        pool: &ComputePool,
        mrope_positions: &[[usize; 4]],
    ) -> Result<Vec<f32>, String> {
        self.trunk.forward_impl(
            n_tokens,
            Some(base_position),
            kv_cache,
            scratch,
            pool,
            mrope_positions,
            Some(&self.moe),
        )
    }
}

impl<'m> HybridTrunkModel<'m> for Edge0Model<'m> {
    fn trunk(&self) -> &HybridTrunk<'m> {
        &self.trunk
    }
    fn forward_at(
        &mut self,
        n_tokens: usize,
        base_position: usize,
        kv_cache: &mut KvCache,
        scratch: &mut Qwen35Scratchpad,
        pool: &ComputePool,
        positions: &[[usize; 4]],
    ) -> Result<Vec<f32>, String> {
        Edge0Model::forward_at(
            self,
            n_tokens,
            base_position,
            kv_cache,
            scratch,
            pool,
            positions,
        )
    }
    #[cfg(feature = "vulkan")]
    fn supports_vulkan(&self) -> bool {
        false
    }
}

pub(crate) fn forward_edge0_moe_token(
    moe: &Edge0MoeWeights<'_>,
    input: &[f32],
    shared: &mut [f32],
    prepared: &mut PreparedRows,
    pool: &ComputePool,
    layer: usize,
    token: usize,
) -> Result<(), String> {
    let mut logits = vec![0.0; moe.router.n_out];
    let mut shared_gate = [0.0];
    prepared.prepare(
        input,
        1,
        input.len(),
        moe.router.needs_q8_0_activation() || moe.shared_gate.needs_q8_0_activation(),
        moe.router.uses_q8_k() || moe.shared_gate.uses_q8_k(),
    )?;
    prepared.matmul_group(
        input,
        [
            (&moe.router, &mut logits),
            (&moe.shared_gate, &mut shared_gate),
        ],
        pool,
    )?;
    #[cfg(feature = "parity-trace")]
    if layer == 0 && token == 0 {
        crate::parity_trace::report(crate::parity_trace::checkpoint(
            "edge0.router-0",
            Some(layer),
            &[logits.len()],
            &logits,
        ));
    }
    #[cfg(not(feature = "parity-trace"))]
    let _ = (layer, token);
    if logits.iter().any(|value| !value.is_finite()) {
        return Err("Edge0 router produced non-finite logits".into());
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp = logits
        .iter()
        .map(|value| (*value - max).exp())
        .collect::<Vec<_>>();
    let total = exp.iter().sum::<f32>();
    let probabilities = exp.iter().map(|value| *value / total).collect::<Vec<_>>();
    let mut indices = (0..logits.len()).collect::<Vec<_>>();
    indices.sort_by(|&a, &b| {
        probabilities[b]
            .total_cmp(&probabilities[a])
            .then_with(|| a.cmp(&b))
    });
    let chosen = &indices[..moe.used];
    #[cfg(feature = "parity-trace")]
    if layer == 0 && token == 0 {
        let values = chosen.iter().map(|&index| index as f32).collect::<Vec<_>>();
        crate::parity_trace::report(crate::parity_trace::checkpoint(
            "edge0.chosen-0",
            Some(layer),
            &[values.len()],
            &values,
        ));
    }
    let chosen_total = chosen
        .iter()
        .map(|&index| probabilities[index])
        .sum::<f32>();
    let mut routed = vec![0.0f32; shared.len()];
    for &expert in chosen {
        let gate_weight = &moe.gate[expert];
        let up_weight = &moe.up[expert];
        let down_weight = &moe.down[expert];
        let mut gate = vec![0.0; gate_weight.n_out];
        let mut up = vec![0.0; up_weight.n_out];
        prepared.prepare(
            input,
            1,
            input.len(),
            gate_weight.needs_q8_0_activation() || up_weight.needs_q8_0_activation(),
            gate_weight.uses_q8_k() || up_weight.uses_q8_k(),
        )?;
        prepared.matmul_group(
            input,
            [(gate_weight, &mut gate), (up_weight, &mut up)],
            pool,
        )?;
        silu_inplace(&mut gate);
        for (gate_value, up_value) in gate.iter_mut().zip(up) {
            *gate_value *= up_value;
        }
        let mut down = vec![0.0; down_weight.n_out];
        prepared.prepare(
            &gate,
            1,
            gate.len(),
            down_weight.needs_q8_0_activation(),
            down_weight.uses_q8_k(),
        )?;
        prepared.matmul(down_weight, &gate, &mut down, pool)?;
        let score = probabilities[expert] / chosen_total;
        for (out, value) in routed.iter_mut().zip(down) {
            *out += score * value;
        }
    }
    let shared_scale = 1.0 / (1.0 + (-shared_gate[0]).exp());
    #[cfg(feature = "parity-trace")]
    if layer == 0 && token == 0 {
        crate::parity_trace::report(crate::parity_trace::checkpoint(
            "edge0.routed-0",
            Some(layer),
            &[routed.len()],
            &routed,
        ));
        crate::parity_trace::report(crate::parity_trace::checkpoint(
            "edge0.shared-0",
            Some(layer),
            &[shared.len()],
            shared,
        ));
        crate::parity_trace::report(crate::parity_trace::checkpoint(
            "edge0.shared_gate-0",
            Some(layer),
            &[1],
            &[shared_scale],
        ));
    }
    for (out, value) in shared.iter_mut().zip(routed) {
        *out = value + shared_scale * *out;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::tensor::GGMLType;
    use crate::ops::kernel::{f32::F32Kernel, Kernel, Weight};
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct ObservedKernel<'a> {
        inner: F32Kernel,
        workers: &'a AtomicUsize,
    }

    impl Kernel for ObservedKernel<'_> {
        fn forward_prequantized(
            &self,
            _: &[u8],
            _: &[f32],
            _: &mut [f32],
            _: usize,
            _: usize,
            _: usize,
            _: usize,
        ) {
            panic!("F32 projections require the original activations");
        }

        fn forward_prepared(
            &self,
            input: &[f32],
            input_q8: &[u8],
            scales: &[f32],
            q8_k: Option<&[crate::ops::quant::BlockQ8K]>,
            output: &mut [f32],
            n_in: usize,
            n_out: usize,
            ith: usize,
            nth: usize,
        ) {
            self.workers.fetch_or(1 << ith, Ordering::Relaxed);
            self.inner
                .forward_prepared(input, input_q8, scales, q8_k, output, n_in, n_out, ith, nth);
        }
    }

    #[test]
    fn moe_projections_use_all_compute_workers() {
        let workers: [AtomicUsize; 8] = std::array::from_fn(|_| AtomicUsize::new(0));
        let weight = |n_out, projection: usize| Weight {
            kernel: Box::new(ObservedKernel {
                inner: F32Kernel::new(vec![1.0 / 64.0; 64 * n_out], 64, n_out),
                workers: &workers[projection],
            }),
            ggml_type: GGMLType::F32,
            n_in: 64,
            n_out,
        };
        let moe = Edge0MoeWeights {
            router: weight(2, 0),
            shared_gate: weight(1, 1),
            gate: vec![weight(64, 2), weight(64, 3)],
            up: vec![weight(64, 4), weight(64, 5)],
            down: vec![weight(64, 6), weight(64, 7)],
            used: 1,
        };
        let mut shared = [2.0; 64];
        forward_edge0_moe_token(
            &moe,
            &[1.0; 64],
            &mut shared,
            &mut PreparedRows::new(1, 64),
            &ComputePool::new(4),
            0,
            0,
        )
        .unwrap();
        for value in shared {
            assert!((value - 2.1931758).abs() < 1e-5, "MoE output {value}");
        }
        assert_eq!(
            workers
                .each_ref()
                .map(|workers| workers.load(Ordering::Relaxed)),
            [15, 15, 15, 0, 15, 0, 15, 0],
        );
    }
}
