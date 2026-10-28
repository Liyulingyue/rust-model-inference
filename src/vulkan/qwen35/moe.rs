//! Resident expert projections; host routing and ordered reduction share CPU semantics.
// shortcut: each MoE row synchronizes with the host; move routing/reduction to shaders after numerical and driver validation.
use super::*;
use crate::models::qwen35::trunk::{config::Qwen35Moe, weights::Qwen35MoeWeights};

struct ExpertBindings {
    gate_up: WeightBindings<2>,
    down: OperatorBindings,
    width: usize,
}

pub(super) struct MoeBindings {
    router: OperatorBindings,
    experts: Vec<ExpertBindings>,
    shared: Option<(OperatorBindings, ExpertBindings)>,
    config: Qwen35Moe,
}

impl MoeBindings {
    pub(super) fn new(
        ops: &mut Qwen3Ops<'_>,
        buffers: &mut UploadedBuffers,
        weights: &Qwen35MoeWeights<'_>,
        config: Qwen35Moe,
    ) -> Result<Self, VulkanError> {
        let router = bind_weight(ops, buffers, &weights.gate_inp, "MoE router")?;
        let gate = buffers.upload(weights.gate_exps)?;
        let up = buffers.upload(weights.up_exps)?;
        let down = buffers.upload(weights.down_exps)?;
        let formats = [weights.gate_type, weights.up_type, weights.down_type]
            .map(GpuWeightFormat::from_ggml_type);
        let [gate_format, up_format, down_format] = formats;
        let (gate_format, up_format, down_format) = (gate_format?, up_format?, down_format?);
        let mut experts = Vec::with_capacity(weights.n_expert);
        for expert in 0..weights.n_expert {
            let view = |buffer, stride: usize| (buffer, (expert * stride) as u64, stride as u64);
            let gate_up = if gate_format == up_format {
                WeightBindings::Grouped(ops.bind_weight_ranges(
                    &[view(gate, weights.stride_gate), view(up, weights.stride_up)],
                    &[gate_format, up_format],
                )?)
            } else {
                WeightBindings::Split([
                    ops.bind_weight_ranges(&[view(gate, weights.stride_gate)], &[gate_format])?,
                    ops.bind_weight_ranges(&[view(up, weights.stride_up)], &[up_format])?,
                ])
            };
            experts.push(ExpertBindings {
                gate_up,
                down: ops.bind_weight_ranges(&[view(down, weights.stride_down)], &[down_format])?,
                width: config.n_ff_exp,
            });
        }
        let shared = if config.n_ff_shexp == 0 {
            None
        } else {
            let missing = || VulkanError::UnsupportedShape("incomplete MoE shared expert".into());
            let score_values = weights.gate_inp_shexp.as_deref().ok_or_else(missing)?;
            let score_weight = Weight::from_quantized(crate::ops::kernel::QuantizedTensor::F32 {
                data: score_values.to_vec(),
                n_in: weights.n_embd,
                n_out: 1,
            });
            let score = bind_weight(ops, buffers, &score_weight, "shared gate")?;
            let gate_up = bind_weight_group(
                ops,
                buffers,
                [
                    (
                        weights.gate_shexp.as_ref().ok_or_else(missing)?,
                        "shared gate projection",
                    ),
                    (
                        weights.up_shexp.as_ref().ok_or_else(missing)?,
                        "shared up projection",
                    ),
                ],
            )?;
            let down = bind_weight(
                ops,
                buffers,
                weights.down_shexp.as_ref().ok_or_else(missing)?,
                "shared down",
            )?;
            Some((
                score,
                ExpertBindings {
                    gate_up,
                    down,
                    width: config.n_ff_shexp,
                },
            ))
        };
        Ok(Self {
            router,
            experts,
            shared,
            config,
        })
    }

    fn record_expert(
        &self,
        session: &Qwen35VulkanSession,
        commands: &TokenCommands<'_>,
        expert: &ExpertBindings,
        input: ArenaRegion,
        slot: usize,
    ) -> Result<(), VulkanError> {
        let gate = row_region(session.layout.ffn_gate, slot, expert.width)?;
        let up = row_region(session.layout.ffn_up, slot, expert.width)?;
        session.record_weight_group(
            commands,
            expert.gate_up,
            input,
            [(gate, expert.width), (up, expert.width)],
            session.config.n_embd,
            1,
        )?;
        session
            .ops
            .record_silu_mul_rows_into(commands, gate, up, gate, expert.width, 1, false)?;
        let output = row_region(session.layout.down, slot, session.config.n_embd)?;
        session.record_weights(
            commands,
            expert.down,
            gate,
            &[(output, session.config.n_embd)],
            expert.width,
            1,
        )
    }

    pub(super) fn forward(
        &self,
        session: &Qwen35VulkanSession,
        mut commands: TokenCommands<'static>,
        rows: usize,
        layer: usize,
        position: usize,
    ) -> Result<TokenCommands<'static>, VulkanError> {
        let width = session.config.n_embd;
        session.record_weights(
            &commands,
            self.router,
            session.layout.normed,
            &[(session.layout.router_logits, self.experts.len())],
            width,
            rows,
        )?;
        if let Some((score, _)) = &self.shared {
            session.record_weights(
                &commands,
                *score,
                session.layout.normed,
                &[(session.layout.shared_gate, 1)],
                width,
                rows,
            )?;
        }
        commands = commands.flush()?;
        let logits = session
            .ops
            .read_f32(session.layout.router_logits, rows * self.experts.len())?
            .to_vec();
        let shared_scores = session
            .ops
            .read_f32(session.layout.shared_gate, rows)?
            .to_vec();
        let experts_per_submission = session.context.moe_experts_per_submission();
        let host_reduce = ["ffn_moe_out", "ffn_out"]
            .iter()
            .any(|name| crate::models::qwen35::trunk::trace::trace_enabled(name));
        let mut output = if host_reduce {
            vec![0.0; rows * width]
        } else {
            Vec::new()
        };
        let count = self.config.n_expert_used + usize::from(self.shared.is_some());
        for row in 0..rows {
            let trace = &crate::models::qwen35::trunk::trace::layer_1d;
            if crate::models::qwen35::trunk::trace::trace_enabled("ffn_moe_input") {
                let input = session.ops.read_f32(session.layout.normed, rows * width)?;
                trace(
                    "ffn_moe_input",
                    layer,
                    position + row,
                    &input[row * width..(row + 1) * width],
                );
            }
            let (chosen, mut scores) = crate::models::qwen35::trunk::moe::route_experts(
                &logits[row * self.experts.len()..(row + 1) * self.experts.len()],
                self.config.n_expert_used,
                self.config.weights_scale,
                layer,
                position + row,
            )
            .map_err(VulkanError::UnsupportedShape)?;
            if !host_reduce {
                if self.shared.is_some() {
                    scores.push(1.0 / (1.0 + (-shared_scores[row]).exp()));
                }
                session
                    .ops
                    .write_f32(row_region(session.layout.moe_scores, row, count)?, &scores)?;
            }
            let input = row_region(session.layout.normed, row, width)?;
            for (slot, &expert) in chosen.iter().enumerate() {
                if slot > 0 && slot % experts_per_submission == 0 {
                    commands = commands.flush()?;
                }
                self.record_expert(session, &commands, &self.experts[expert], input, slot)?;
            }
            if let Some((_, shared)) = &self.shared {
                if !chosen.is_empty() && chosen.len() % experts_per_submission == 0 {
                    commands = commands.flush()?;
                }
                self.record_expert(session, &commands, shared, input, chosen.len())?;
            }
            if !host_reduce {
                session.ops.record_moe_reduce(
                    &commands,
                    session.layout.down,
                    row_region(session.layout.moe_scores, row, count)?,
                    row_region(session.layout.projection, row, width)?,
                    row_region(session.layout.x, row, width)?,
                    session.layout.moe_status,
                    width,
                    count,
                )?;
                // Preserve the shorter submissions on drivers outside the validated nine-expert set.
                if row + 1 < rows || experts_per_submission < count {
                    commands = commands.flush()?;
                }
                continue;
            }
            commands = commands.flush()?;
            let count = chosen.len() + usize::from(self.shared.is_some());
            let values = session.ops.read_f32(session.layout.down, count * width)?;
            let target = &mut output[row * width..(row + 1) * width];
            for (slot, score) in scores.iter().enumerate() {
                for (o, v) in target
                    .iter_mut()
                    .zip(&values[slot * width..(slot + 1) * width])
                {
                    *o += score * v;
                }
            }
            trace("ffn_moe_out", layer, position + row, target);
            if self.shared.is_some() {
                let score = 1.0 / (1.0 + (-shared_scores[row]).exp());
                for (o, v) in target.iter_mut().zip(&values[chosen.len() * width..]) {
                    *o += score * v;
                }
            }
            trace("ffn_out", layer, position + row, target);
        }
        if !host_reduce {
            return Ok(commands);
        }
        if output.iter().any(|v| !v.is_finite()) {
            return Err(VulkanError::UnsupportedShape(
                "non-finite MoE output".into(),
            ));
        }
        session.ops.write_f32(session.layout.projection, &output)?;
        session.ops.record_add_rows(
            &commands,
            session.layout.x,
            session.layout.projection,
            width,
            rows,
        )?;
        Ok(commands)
    }
}

fn row_region(region: ArenaRegion, row: usize, width: usize) -> Result<ArenaRegion, VulkanError> {
    let size = qwen35_product("expert row", &[width, 4])?;
    let offset = row.checked_mul(size).ok_or(VulkanError::OutOfMemory)?;
    if offset.checked_add(size).is_none_or(|end| end > region.size) {
        return Err(VulkanError::UnsupportedShape(
            "expert row exceeds arena region".into(),
        ));
    }
    Ok(ArenaRegion {
        offset: region.offset + offset,
        size,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{core::thread_pool::ComputePool, ops::kernel::PreparedRows, GGUFLoader};

    #[test]
    #[ignore = "requires Vulkan and RMI_QWEN35_MODEL with real MoE weights"]
    fn qwen35_real_moe_layers_run_on_vulkan() {
        let source = GGUFLoader::from_file(std::env::var("RMI_QWEN35_MODEL").unwrap()).unwrap();
        let model = Qwen35Model::from_source(&source).unwrap();
        let config = model.config.moe.expect("requires routed experts");
        let context = Box::leak(Box::new(VulkanContext::new().unwrap()));
        let session = Qwen35VulkanSession::try_new_rows(&model, 8, 3, context)
            .unwrap()
            .unwrap();
        let width = model.config.n_embd;
        let pool = ComputePool::new(8);
        let _cpu_scope = ComputePool::disable_gpu_matmul_for_scope();
        let mut prepared = PreparedRows::new(1, width.max(config.n_ff_exp).max(config.n_ff_shexp));
        let input: Vec<f32> = (0..3 * width)
            .map(|i| ((i * 29 % 251) as f32 - 125.0) / 97.0)
            .collect();
        let mut maximum = 0.0f64;
        for (layer, weights) in model.layers.iter().enumerate() {
            let weights = weights.moe.as_ref().unwrap();
            let mut expected = vec![0.0; input.len()];
            for row in 0..3 {
                crate::models::qwen35::trunk::moe::forward_qwen35moe_token(
                    weights,
                    &config,
                    config.weights_scale,
                    &input[row * width..(row + 1) * width],
                    &mut expected[row * width..(row + 1) * width],
                    &mut prepared,
                    &pool,
                    layer,
                    row,
                )
                .unwrap();
            }
            session
                .ops
                .write_f32(session.layout.x, &vec![0.0; input.len()])
                .unwrap();
            session
                .ops
                .write_f32(session.layout.normed, &input)
                .unwrap();
            let before = context.compute_stats();
            let FfnBindings::MoeHostRouted(bindings) = &session.layers[layer].ffn else {
                panic!("requires host-routed MoE expert formats");
            };
            bindings
                .forward(
                    &session,
                    TokenCommands::begin(context).unwrap(),
                    3,
                    layer,
                    0,
                )
                .unwrap()
                .submit_and_wait()
                .unwrap();
            let experts = config.n_expert_used + usize::from(config.n_ff_shexp > 0);
            let budget = context.moe_experts_per_submission();
            let expected_submissions =
                1 + 3 * experts.div_ceil(budget) + usize::from(budget < experts);
            assert_eq!(
                context.compute_stats().since(before).submissions,
                expected_submissions as u64,
                "router + expert batches + residual, layer={layer}"
            );
            let actual = session.ops.read_f32(session.layout.x, input.len()).unwrap();
            assert!(actual.iter().all(|v| v.is_finite()));
            let error = (expected
                .iter()
                .zip(actual)
                .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
                .sum::<f64>()
                / expected.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>())
            .sqrt();
            maximum = maximum.max(error);
            println!("MoE layer={layer} rows=3 relative_l2={error}");
            assert!(error < 0.02, "layer={layer} relative_l2={error}");
        }
        println!("MoE same-input CPU/Vulkan max_relative_l2={maximum}");
    }
}
