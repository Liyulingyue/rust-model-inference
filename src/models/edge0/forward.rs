//! Edge0 MoE computation and model forward entry points.

use super::weights::{Edge0Model, Edge0MoeWeights};
use crate::core::scratchpad::KvCache;
use crate::core::thread_pool::ComputePool;
use crate::models::qwen35::trunk::session::HybridTrunkModel;
use crate::models::qwen35::trunk::HybridTrunk;
use crate::models::qwen35::Qwen35Scratchpad;
use crate::ops::silu;

/// Edge0's Q/K path uses RMSNorm with additive epsilon before its head scaling.
pub(crate) fn normalize_recurrent_qk(values: &mut [f32], eps: f32, factor: f32) {
    let sum = values
        .iter()
        .map(|&value| f64::from(value * value))
        .sum::<f64>();
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
    layer: usize,
    token: usize,
) -> Result<(), String> {
    let logits = moe.router.matmul(input);
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
        let mut gate = moe.gate[expert].matmul(input);
        let up = moe.up[expert].matmul(input);
        for (gate_value, up_value) in gate.iter_mut().zip(up) {
            *gate_value = silu(*gate_value) * up_value;
        }
        let down = moe.down[expert].matmul(&gate);
        let score = probabilities[expert] / chosen_total;
        for (out, value) in routed.iter_mut().zip(down) {
            *out += score * value;
        }
    }
    let shared_gate = moe.shared_gate.matmul(input)[0];
    let shared_scale = 1.0 / (1.0 + (-shared_gate).exp());
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
