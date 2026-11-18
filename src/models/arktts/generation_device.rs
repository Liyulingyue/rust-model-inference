use super::device::{Graph, Op, Result, Tensor};
use super::{tokens, Frame, Matrix, Model, Sampling, EOS, SEMANTIC, VOCAB, WIDTH};
use rand::{Rng, SeedableRng};

fn table(graph: &mut Graph, matrix: &Matrix<'_>) -> Result<crate::vulkan::GpuBuffer> {
    // Decode immutable lookup tables once; projection weights retain native storage and kernels.
    let mut values = vec![0.0; matrix.weight.n_in * matrix.weight.n_out];
    for (id, row) in values.chunks_exact_mut(matrix.weight.n_in).enumerate() {
        matrix.weight.embedding_lookup(id as u32, row);
    }
    graph.floats(&values)
}

fn packed(graph: &mut Graph, binding: usize, columns: &[[u32; 11]]) -> Result<Tensor> {
    let ids = graph.input_ids(&columns.iter().flatten().copied().collect::<Vec<_>>(), 11)?;
    let output = graph.alloc(columns.len(), WIDTH)?;
    graph.custom(
        Op::PackedEmbedding,
        Some(binding),
        ids,
        output,
        output,
        output.len(),
        WIDTH,
        &[SEMANTIC],
        None,
    )?;
    graph.free(ids);
    Ok(output)
}

pub(super) fn sample(
    graph: &mut Graph,
    logits: Tensor,
    semantic: bool,
    sampling: Sampling,
    rng: &mut rand::rngs::StdRng,
) -> Result<u32> {
    let count = if semantic { 4097 } else { logits.width };
    let ids = graph.alloc(1, count)?;
    let values = graph.alloc(1, count)?;
    let random = graph.alloc(1, count)?;
    if sampling.temperature > 0.0 {
        let mut cloned = rng.clone();
        graph.write(random, (0..count).map(|_| cloned.gen::<f32>()).collect());
    }
    let result = graph.alloc(1, 3)?;
    graph.custom(
        Op::Rank,
        None,
        logits,
        ids,
        values,
        count,
        0,
        &[u32::from(semantic), SEMANTIC, EOS],
        None,
    )?;
    graph.custom(
        Op::Sample,
        None,
        ids,
        values,
        random,
        count,
        sampling.top_k.min(count),
        &[
            result.word(),
            sampling.temperature.to_bits(),
            sampling.top_p.to_bits(),
        ],
        Some(1),
    )?;
    let result_values = graph.read(result)?;
    let id = result_values[0].to_bits();
    let consumed = result_values[1].to_bits() as usize;
    if result_values[2].to_bits() != 0 || consumed > count {
        return Err("ArkTTS device logits must be finite and nonempty".into());
    }
    for _ in 0..consumed {
        let _ = rng.gen::<f32>();
    }
    for tensor in [ids, values, random, result] {
        graph.free(tensor);
    }
    Ok(id)
}

impl Model<'_> {
    pub(super) fn generate_device(
        &self,
        context: &'static crate::vulkan::VulkanContext,
        prompt: &[[u32; 11]],
        max_frames: usize,
        sampling: Sampling,
        history: Option<&[Frame]>,
    ) -> Result<Vec<Frame>> {
        let mut graph = Graph::new(context, "generate");
        let mut cache = self
            .slow
            .device_cache(&mut graph, prompt.len() + max_frames)?;
        let mut fast_cache = self.fast.device_cache(&mut graph, 10)?;
        let embeddings = table(&mut graph, &self.embeddings)?;
        let codes = table(&mut graph, &self.code_embeddings)?;
        let packed_binding = graph.binding(vec![embeddings, codes], None);
        let fast_embeddings = table(&mut graph, &self.fast_embeddings)?;
        let fast_binding = graph.binding(vec![fast_embeddings], None);
        let mut rng = rand::rngs::StdRng::seed_from_u64(sampling.seed);
        let input = packed(&mut graph, packed_binding, prompt)?;
        let mut hidden = self.slow.device_forward(&mut graph, input, &mut cache)?;
        let mut frames = Vec::new();
        let mut previous: Option<Frame> = None;
        for step in 0..max_frames {
            let normalized = self
                .slow
                .device_normalize(&mut graph, hidden.row(hidden.rows - 1))?;
            graph.free(hidden);
            let logits = graph.matmul(&self.embeddings, normalized)?;
            graph.trace("arktts.slow.logits", None, Some(step), logits, &[VOCAB]);
            let mut semantic = sample(&mut graph, logits, true, sampling, &mut rng)?;
            if let Some(history) = history {
                semantic = SEMANTIC + history[step][0];
            }
            if sampling.temperature > 0.0 {
                let high = sample(
                    &mut graph,
                    logits,
                    true,
                    Sampling {
                        temperature: 1.0,
                        top_p: 0.9,
                        ..sampling
                    },
                    &mut rng,
                )?;
                if previous.is_some_and(|p| p.contains(&semantic)) && semantic != EOS {
                    semantic = high;
                }
            }
            graph.free(logits);
            if semantic == EOS {
                break;
            }
            if !(SEMANTIC..SEMANTIC + 4096).contains(&semantic) {
                return Err("invalid ArkTTS device semantic ID".into());
            }
            let mut frame = [0; 10];
            frame[0] = semantic - SEMANTIC;
            fast_cache.position = 0;
            let warmup = self
                .fast
                .device_forward(&mut graph, normalized, &mut fast_cache)?;
            graph.free(warmup);
            for codebook in 1..10 {
                let id = graph.input_ids(&[frame[codebook - 1]], 1)?;
                let input = graph.alloc(1, WIDTH)?;
                graph.custom(
                    Op::Lookup,
                    Some(fast_binding),
                    id,
                    input,
                    input,
                    WIDTH,
                    WIDTH,
                    &[4096, 1, 0],
                    None,
                )?;
                graph.free(id);
                let hidden = self
                    .fast
                    .device_forward(&mut graph, input, &mut fast_cache)?;
                let normalized = self.fast.device_normalize(&mut graph, hidden)?;
                graph.free(hidden);
                let logits = graph.matmul(&self.fast_output, normalized)?;
                graph.free(normalized);
                graph.trace(
                    "arktts.fast.logits",
                    Some(codebook),
                    Some(step),
                    logits,
                    &[4096],
                );
                frame[codebook] = sample(&mut graph, logits, false, sampling, &mut rng)?;
                graph.free(logits);
                if let Some(history) = history {
                    frame[codebook] = history[step][codebook];
                }
            }
            tokens("arktts.frame", &frame)?;
            frames.push(frame);
            if let Some(previous) = &mut previous {
                previous.rotate_left(1);
                previous[9] = semantic;
            } else {
                previous = Some([0; 10]);
            }
            if step + 1 < max_frames {
                let mut column = [0; 11];
                column[0] = semantic;
                column[1..].copy_from_slice(&frame);
                let input = packed(&mut graph, packed_binding, &[column])?;
                hidden = self.slow.device_forward(&mut graph, input, &mut cache)?;
            }
        }
        Ok(frames)
    }
}
