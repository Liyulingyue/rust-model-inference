use crate::core::thread_pool::ComputePool;
use crate::models::occamy::weights::OccamyMoeWeights;
use crate::ops::kernel::PreparedRows;
use crate::ops::silu_inplace;

/// MoE FFN for one Occamy token.
///
/// The routed half is plain softmax top-k over `ffn_gate_inp`, renormalised
/// across the chosen experts, and the shared expert is gated by its own
/// sigmoid before being added. That last part is what separates Occamy from
/// Edge0, which adds the shared expert unweighted; it follows the Qwen3-Next
/// reference in `references/llama.cpp/src/models/qwen35moe.cpp`.
pub(crate) fn forward_occamy_moe_token(
    moe: &OccamyMoeWeights<'_>,
    input: &[f32],
    out: &mut [f32],
    prepared: &mut PreparedRows,
    pool: &ComputePool,
    layer: usize,
    token: usize,
) -> Result<(), String> {
    let mut logits = vec![0.0; moe.router.n_out];
    let mut shared_gate = [0.0f32];
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
    let _ = (layer, token);
    if logits.iter().any(|value| !value.is_finite()) || !shared_gate[0].is_finite() {
        return Err("Occamy router produced non-finite logits".into());
    }

    // Softmax gating, matching LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX upstream.
    let hparams = &moe.hparams;
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
    // This GGUF declares neither `expert_weights_norm` nor
    // `expert_weights_scale`, so upstream keeps the raw softmax
    // probabilities and does not renormalise across the chosen experts.
    let scale = hparams.expert_weights_scale.unwrap_or(1.0);
    let renormalise = hparams.expert_weights_norm.unwrap_or(false);
    let chosen_total = if renormalise {
        chosen
            .iter()
            .map(|&index| probabilities[index])
            .sum::<f32>()
    } else {
        1.0
    };

    out.fill(0.0);
    for &expert in chosen {
        let gate_weight = &moe.gate[expert];
        let up_weight = &moe.up[expert];
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
        for (slot, value) in gate.iter_mut().zip(up.iter()) {
            *slot *= *value;
        }
        let hidden = gate;

        let down = &moe.down[expert];
        let mut projected = vec![0.0; down.n_out];
        prepared.prepare(
            &hidden,
            1,
            hidden.len(),
            down.needs_q8_0_activation(),
            down.uses_q8_k(),
        )?;
        prepared.matmul(down, &hidden, &mut projected, pool)?;

        let score = probabilities[expert] * scale / chosen_total;
        for (dst, value) in out.iter_mut().zip(projected.iter()) {
            *dst += score * value;
        }
    }

    // Shared expert: separate gate/up tensors (not fused), then the scalar
    // ffn_gate_inp_shexp value gates the whole thing through a sigmoid.
    let shared_hidden = moe.shared_up.n_out;
    let mut shared_scalar = vec![0.0f32; moe.shared_gate.n_out];
    let mut shared_gate_proj = vec![0.0f32; shared_hidden];
    let mut shared_up_proj = vec![0.0f32; shared_hidden];
    prepared.prepare(
        input,
        1,
        input.len(),
        moe.shared_gate.needs_q8_0_activation() || moe.shared_up.needs_q8_0_activation(),
        moe.shared_gate.uses_q8_k() || moe.shared_up.uses_q8_k(),
    )?;
    prepared.matmul_group(
        input,
        [
            (&moe.shared_gate, &mut shared_scalar),
            (&moe.shared_up, &mut shared_up_proj),
        ],
        pool,
    )?;
    silu_inplace(&mut shared_gate_proj);
    for (slot, value) in shared_gate_proj.iter_mut().zip(shared_up_proj.iter()) {
        *slot *= *value;
    }
    let mut shared_out = vec![0.0; out.len()];
    prepared.prepare(
        &shared_gate_proj,
        1,
        shared_gate_proj.len(),
        moe.shared_down.needs_q8_0_activation(),
        moe.shared_down.uses_q8_k(),
    )?;
    prepared.matmul(&moe.shared_down, &shared_gate_proj, &mut shared_out, pool)?;

    // 1 / (1 + e^-g)
    let gate_scale = 1.0 / (1.0 + (-shared_scalar[0]).exp());
    for (dst, value) in out.iter_mut().zip(shared_out.iter()) {
        *dst += gate_scale * value;
    }
    Ok(())
}
