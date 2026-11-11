use super::Transformer;
use crate::models::arktts::device::{Graph, Op, Result, Tensor};

pub(in crate::models::arktts) struct Cache {
    keys: Vec<Tensor>,
    values: Vec<Tensor>,
    rope: usize,
    capacity: usize,
    pub position: usize,
}
impl Transformer<'_> {
    pub(in crate::models::arktts) fn device_cache(
        &self,
        graph: &mut Graph,
        capacity: usize,
    ) -> Result<Cache> {
        if capacity == 0 || capacity > super::super::CONTEXT * 4 {
            return Err("invalid ArkTTS device cache capacity".into());
        }
        let mut keys = Vec::new();
        let mut values = Vec::new();
        for _ in &self.layers {
            keys.push(graph.alloc(capacity, self.shape.kv * 64)?);
            values.push(graph.alloc(capacity, self.shape.kv * 64)?);
        }
        let mut coefficients = Vec::with_capacity(capacity * 64);
        for position in 0..capacity {
            for index in 0..32 {
                let phase =
                    position as f32 * (1.0 / self.shape.rope_base.powf((2 * index) as f32 / 64.0));
                coefficients.push(half::bf16::from_f32(phase.cos()).to_f32());
                coefficients.push(half::bf16::from_f32(phase.sin()).to_f32());
            }
        }
        let buffer = graph.floats(&coefficients)?;
        let rope = graph.binding(vec![buffer], None);
        Ok(Cache {
            keys,
            values,
            rope,
            capacity,
            position: 0,
        })
    }
    pub(in crate::models::arktts) fn device_normalize(
        &self,
        graph: &mut Graph,
        input: Tensor,
    ) -> Result<Tensor> {
        graph.rms(input, &self.norm, self.shape.epsilon)
    }
    pub(in crate::models::arktts) fn device_forward(
        &self,
        graph: &mut Graph,
        hidden: Tensor,
        cache: &mut Cache,
    ) -> Result<Tensor> {
        let s = self.shape;
        let rows = hidden.rows;
        let start = cache.position;
        let end = start
            .checked_add(rows)
            .filter(|&n| n <= cache.capacity)
            .ok_or("ArkTTS device KV capacity exceeded")?;
        if hidden.width != s.width {
            return Err("invalid ArkTTS device transformer input".into());
        }
        graph.trace(
            &format!("{}.input", self.name),
            None,
            Some(start),
            hidden,
            &[rows, s.width],
        );
        for (index, layer) in self.layers.iter().enumerate() {
            let norm = graph.rms(hidden, &layer.norm1, s.epsilon)?;
            let qkv = graph.matmul(&layer.qkv, norm)?;
            graph.free(norm);
            let query = graph.alloc(rows, s.heads * 64)?;
            graph.custom(
                Op::RopeKv,
                Some(cache.rope),
                qkv,
                cache.keys[index],
                cache.values[index],
                rows * (s.heads + s.kv) * 32,
                qkv.width,
                &[
                    s.heads as u32,
                    s.kv as u32,
                    start as u32,
                    cache.capacity as u32,
                    query.word(),
                ],
                None,
            )?;
            graph.free(qkv);
            let attention = graph.alloc(rows, s.heads * 64)?;
            graph.custom(
                Op::Attention,
                None,
                query,
                cache.keys[index],
                cache.values[index],
                rows,
                s.heads,
                &[
                    s.kv as u32,
                    start as u32,
                    cache.capacity as u32,
                    s.window.unwrap_or(0) as u32,
                    attention.word(),
                ],
                Some(rows * s.heads),
            )?;
            graph.free(query);
            let update = graph.matmul(&layer.output, attention)?;
            graph.free(attention);
            graph.add(hidden, update, Some(&layer.attn_scale))?;
            graph.free(update);
            let norm = graph.rms(hidden, &layer.norm2, s.epsilon)?;
            let gate = graph.matmul(&layer.gate, norm)?;
            let up = graph.matmul(&layer.up, norm)?;
            graph.free(norm);
            graph.custom(
                Op::SiluMul,
                None,
                gate,
                up,
                gate,
                gate.len(),
                gate.width,
                &[],
                None,
            )?;
            graph.free(up);
            let update = graph.matmul(&layer.down, gate)?;
            graph.free(gate);
            graph.add(hidden, update, Some(&layer.ffn_scale))?;
            graph.free(update);
            graph.trace(
                &self.name,
                Some(index),
                Some(start),
                hidden,
                &[rows, s.width],
            );
        }
        cache.position = end;
        Ok(hidden)
    }
    pub(in crate::models::arktts) fn release_device_cache(&self, graph: &mut Graph, cache: Cache) {
        for tensor in cache.keys.into_iter().chain(cache.values) {
            graph.free(tensor);
        }
    }
}
