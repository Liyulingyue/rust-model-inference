use super::*;
use crate::models::arktts::device::{Graph, Op, Result, Tensor};

fn convolution(graph: &mut Graph, conv: &CausalConv, x: Tensor) -> Result<Tensor> {
    let p = conv.view();
    if p.input != x.width {
        return Err("invalid ArkTTS device convolution width".into());
    }
    let out = graph.alloc(x.rows.div_ceil(p.stride), p.output)?;
    let binding = graph.pair(p.weight, p.bias)?;
    graph.custom(
        Op::Conv,
        Some(binding),
        x,
        out,
        out,
        out.len(),
        p.input,
        &[
            p.output as u32,
            p.kernel as u32,
            p.stride as u32,
            p.dilation as u32,
            p.groups as u32,
            x.rows as u32,
            u32::from(p.replicate),
        ],
        None,
    )?;
    Ok(out)
}
fn conv_replace(graph: &mut Graph, conv: &CausalConv, x: Tensor) -> Result<Tensor> {
    let out = convolution(graph, conv, x)?;
    graph.free(x);
    Ok(out)
}
fn transpose(graph: &mut Graph, conv: &Transpose, x: Tensor) -> Result<Tensor> {
    let rows = x
        .rows
        .checked_mul(conv.stride)
        .ok_or("ArkTTS device transpose size overflow")?;
    let out = graph.alloc(rows, conv.output)?;
    let binding = graph.pair(&conv.weight, &conv.bias)?;
    graph.custom(
        Op::Transpose,
        Some(binding),
        x,
        out,
        out,
        out.len(),
        conv.input,
        &[
            conv.output as u32,
            conv.kernel as u32,
            conv.stride as u32,
            x.rows as u32,
        ],
        None,
    )?;
    graph.free(x);
    Ok(out)
}
fn snake(graph: &mut Graph, snake: &Snake, x: Tensor) -> Result<()> {
    let binding = graph.pair(&snake.alpha, &snake.inverse)?;
    graph.unary(Op::Snake, x, Some(binding))
}
fn residual(graph: &mut Graph, block: &Residual, x: Tensor) -> Result<Tensor> {
    let mut branch = graph.copy(x)?;
    snake(graph, &block.first_snake, branch)?;
    branch = conv_replace(graph, &block.first, branch)?;
    snake(graph, &block.second_snake, branch)?;
    branch = conv_replace(graph, &block.second, branch)?;
    graph.add(x, branch, None)?;
    graph.free(branch);
    Ok(x)
}
fn next(graph: &mut Graph, block: &ConvNeXt<'_>, x: Tensor) -> Result<Tensor> {
    let depth = convolution(graph, &block.depth, x)?;
    let norm = graph.alloc(x.rows, x.width)?;
    let binding = graph.pair(&block.norm, &block.bias)?;
    graph.custom(
        Op::LayerNorm,
        Some(binding),
        depth,
        norm,
        norm,
        x.rows,
        x.width,
        &[],
        Some(x.rows),
    )?;
    graph.free(depth);
    let branch = graph.matmul(&block.first, norm)?;
    graph.free(norm);
    graph.unary(Op::Gelu, branch, None)?;
    let out = graph.matmul(&block.second, branch)?;
    graph.free(branch);
    graph.add(x, out, Some(&block.gamma))?;
    graph.free(out);
    Ok(x)
}
fn transformer(graph: &mut Graph, block: &Transformer<'_>, x: Tensor) -> Result<Tensor> {
    let mut cache = block.device_cache(graph, x.rows)?;
    let hidden = block.device_forward(graph, x, &mut cache)?;
    let out = block.device_normalize(graph, hidden)?;
    graph.free(hidden);
    block.release_device_cache(graph, cache);
    Ok(out)
}

impl Codec<'_> {
    pub(super) fn encode_device(
        &self,
        context: &'static crate::vulkan::VulkanContext,
        audio: &[f32],
    ) -> Result<Vec<Frame>> {
        let mut graph = Graph::new(context, "codec_encode");
        let mut padded = audio.to_vec();
        padded.resize(
            audio.len().div_ceil(SAMPLES_PER_FRAME) * SAMPLES_PER_FRAME,
            0.0,
        );
        let input = graph.input(&padded, 1)?;
        let mut x = conv_replace(&mut graph, &self.encoder.input, input)?;
        for (index, (blocks, activation, down)) in self.encoder.stages.iter().enumerate() {
            for block in blocks {
                x = residual(&mut graph, block, x)?;
            }
            snake(&mut graph, activation, x)?;
            x = conv_replace(&mut graph, down, x)?;
            if index == 3 {
                x = transformer(&mut graph, &self.encoder.transformer, x)?;
            }
        }
        snake(&mut graph, &self.encoder.snake, x)?;
        x = conv_replace(&mut graph, &self.encoder.output, x)?;
        graph.trace("arktts.codec.encoder", None, None, x, &[x.rows, 1024]);
        for (conv, block) in &self.down {
            x = conv_replace(&mut graph, conv, x)?;
            x = next(&mut graph, block, x)?;
        }
        x = transformer(&mut graph, &self.pre, x)?;
        graph.trace("arktts.codec.pre", None, None, x, &[x.rows, 1024]);
        let codes = graph.alloc(x.rows, 10)?;
        for (index, book) in self.codebooks.iter().enumerate() {
            let projected = convolution(&mut graph, &book.input, x)?;
            let quantized = graph.alloc(x.rows, 8)?;
            let binding = graph.pair(&book.normalized, &book.embeddings)?;
            graph.custom(
                Op::Codebook,
                Some(binding),
                projected,
                codes,
                quantized,
                x.rows,
                book.size,
                &[index as u32],
                Some(x.rows),
            )?;
            graph.free(projected);
            let branch = conv_replace(&mut graph, &book.output, quantized)?;
            graph.custom(
                Op::Subtract,
                None,
                x,
                branch,
                x,
                x.len(),
                x.width,
                &[],
                None,
            )?;
            graph.free(branch);
        }
        let values = graph.read(codes)?;
        let frames: Vec<Frame> = values
            .chunks_exact(10)
            .map(|row| std::array::from_fn(|i| row[i].to_bits()))
            .collect();
        super::super::tokens(
            "arktts.reference_codes",
            &frames.iter().flatten().copied().collect::<Vec<_>>(),
        )?;
        Ok(frames)
    }
    pub(super) fn decode_device(
        &self,
        context: &'static crate::vulkan::VulkanContext,
        frames: &[Frame],
    ) -> Result<Vec<f32>> {
        let mut graph = Graph::new(context, "codec_decode");
        let codes = graph.input_ids(&frames.iter().flatten().copied().collect::<Vec<_>>(), 10)?;
        let mut x = graph.alloc(frames.len(), 1024)?;
        graph.unary(Op::Zero, x, None)?;
        for (index, book) in self.codebooks.iter().enumerate() {
            let projected = graph.alloc(frames.len(), 8)?;
            let binding = graph.vector(&book.embeddings)?;
            graph.custom(
                Op::Lookup,
                Some(binding),
                codes,
                projected,
                projected,
                projected.len(),
                8,
                &[book.size as u32, 10, index as u32],
                None,
            )?;
            let branch = conv_replace(&mut graph, &book.output, projected)?;
            graph.add(x, branch, None)?;
            graph.free(branch);
        }
        graph.free(codes);
        graph.trace(
            "arktts.codec.quantized",
            None,
            None,
            x,
            &[frames.len(), 1024],
        );
        x = transformer(&mut graph, &self.post, x)?;
        graph.trace("arktts.codec.post", None, None, x, &[frames.len(), 1024]);
        for (conv, block) in &self.up {
            x = transpose(&mut graph, conv, x)?;
            x = next(&mut graph, block, x)?;
        }
        graph.trace("arktts.codec.upsampled", None, None, x, &[x.rows, 1024]);
        x = conv_replace(&mut graph, &self.decoder.input, x)?;
        for (index, stage) in self.decoder.stages.iter().enumerate() {
            snake(&mut graph, &stage.snake, x)?;
            x = transpose(&mut graph, &stage.up, x)?;
            for block in &stage.residual {
                x = residual(&mut graph, block, x)?;
            }
            graph.trace(
                "arktts.codec.decoder",
                Some(index),
                None,
                x,
                &[x.rows, x.width],
            );
        }
        snake(&mut graph, &self.decoder.snake, x)?;
        x = conv_replace(&mut graph, &self.decoder.output, x)?;
        graph.unary(Op::Tanh, x, None)?;
        let waveform = graph.read(x)?;
        if waveform.len() != frames.len() * SAMPLES_PER_FRAME
            || waveform.iter().any(|v| !v.is_finite())
        {
            return Err("ArkTTS device codec produced invalid PCM".into());
        }
        trace(
            "arktts.codec.audio",
            None,
            None,
            &[waveform.len()],
            &waveform,
        )?;
        Ok(waveform)
    }
}
