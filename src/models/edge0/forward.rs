//! Edge0 MoE computation and model forward entry points.

use super::weights::{Edge0Model, Edge0MoeWeights};
use crate::core::scratchpad::KvCache;
use crate::core::thread_pool::ComputePool;
use crate::models::qwen35::trunk::session::HybridTrunkModel;
use crate::models::qwen35::trunk::HybridTrunk;
use crate::models::qwen35::Qwen35Scratchpad;
use crate::ops::silu;

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
) -> Result<(), String> {
    let logits = moe.router.matmul(input);
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
    for (out, value) in shared.iter_mut().zip(routed) {
        *out = value + shared_scale * *out;
    }
    Ok(())
}
