use super::{
    linear,
    sparse::{self, Pages},
    trace, vector, HIDDEN, PAGE, VOCAB,
};
use crate::models::diffusion::dreamx::kernels::{rms_norm_rows, Linear};
use crate::{ComputePool, TensorSource};
use rayon::prelude::*;

struct Mlp<'a> {
    gate: Linear<'a>,
    up: Linear<'a>,
    down: Linear<'a>,
}
impl<'a> Mlp<'a> {
    fn load(s: &'a dyn TensorSource, p: &str, width: usize) -> Result<Self, String> {
        Ok(Self {
            gate: linear(s, &format!("{p}.gate_proj"), HIDDEN, width, false)?,
            up: linear(s, &format!("{p}.up_proj"), HIDDEN, width, false)?,
            down: linear(s, &format!("{p}.down_proj"), width, HIDDEN, false)?,
        })
    }
    fn forward(&self, pool: &ComputePool, x: &[f32]) -> Result<Vec<f32>, String> {
        let rows = x.len() / HIDDEN;
        let gate = self.gate.forward_verified(pool, x, rows)?;
        let mut up = self.up.forward_verified(pool, x, rows)?;
        crate::ops::silu_mul_inplace(&gate, &mut up);
        self.down.forward_verified(pool, &up, rows)
    }
}
struct Layer<'a> {
    q: Linear<'a>,
    k: Linear<'a>,
    v: Linear<'a>,
    o: Linear<'a>,
    rq: Linear<'a>,
    rk: Linear<'a>,
    rqn: Vec<f32>,
    rkn: Vec<f32>,
    cls: Vec<f32>,
    norm: Vec<f32>,
    post: Vec<f32>,
    dense: Option<Mlp<'a>>,
    gate: Option<Linear<'a>>,
    experts: Vec<Mlp<'a>>,
    shared: Option<Mlp<'a>>,
}
#[derive(Default)]
// shortcut: retain full F32 KV within the context budget; add eviction for unlimited streaming.
struct Cache {
    k: Vec<f32>,
    v: Vec<f32>,
    index: Vec<f32>,
}

pub(super) struct Decoder<'a> {
    layers: Vec<Layer<'a>>,
    norm: Vec<f32>,
    head: Linear<'a>,
    embedding: Linear<'a>,
}
impl<'a> Decoder<'a> {
    pub fn load(s: &'a dyn TensorSource) -> Result<Self, String> {
        let mut layers = Vec::new();
        for i in 0..12 {
            let p = format!("model.layers.{i}");
            let a = format!("{p}.self_attn");
            let r = format!("{a}.dash_router");
            let m = format!("{p}.mlp");
            let mut experts = Vec::new();
            if i > 0 {
                for j in 0..64 {
                    experts.push(Mlp::load(s, &format!("{m}.experts.{j}"), 896)?);
                }
            }
            layers.push(Layer {
                q: linear(s, &format!("{a}.q_proj"), HIDDEN, HIDDEN, false)?,
                k: linear(s, &format!("{a}.k_proj"), HIDDEN, HIDDEN, false)?,
                v: linear(s, &format!("{a}.v_proj"), HIDDEN, HIDDEN, false)?,
                o: linear(s, &format!("{a}.o_proj"), HIDDEN, HIDDEN, false)?,
                rq: linear(s, &format!("{r}.q_proj"), HIDDEN, 128, false)?,
                rk: linear(s, &format!("{r}.k_proj"), HIDDEN, 128, false)?,
                rqn: vector(s, &format!("{r}.q_norm.weight"), 128)?,
                rkn: vector(s, &format!("{r}.k_norm.weight"), 128)?,
                cls: vector(s, &format!("{r}.head_cls"), 128)?,
                norm: vector(s, &format!("{p}.input_layernorm.weight"), HIDDEN)?,
                post: vector(s, &format!("{p}.post_attention_layernorm.weight"), HIDDEN)?,
                dense: if i == 0 {
                    Some(Mlp::load(s, &m, 6848)?)
                } else {
                    None
                },
                gate: if i > 0 {
                    Some(linear(s, &format!("{m}.gate"), HIDDEN, 64, false)?)
                } else {
                    None
                },
                experts,
                shared: if i > 0 {
                    Some(Mlp::load(s, &format!("{m}.shared_experts"), 1792)?)
                } else {
                    None
                },
            });
        }
        Ok(Self {
            layers,
            norm: vector(s, "model.norm.weight", HIDDEN)?,
            head: linear(s, "lm_head", HIDDEN, VOCAB, false)?,
            embedding: linear(s, "model.embed_tokens", HIDDEN, VOCAB, false)?,
        })
    }
    fn forward(
        &self,
        pool: &ComputePool,
        mut x: Vec<f32>,
        pages: &Pages,
        start: usize,
        caches: &mut [Cache],
    ) -> Result<Vec<f32>, String> {
        let rows = x.len() / HIDDEN;
        trace("uocr.embeddings", None, &[rows, HIDDEN], &x);
        for (i, (layer, cache)) in self.layers.iter().zip(caches).enumerate() {
            let norm = rms_norm_rows(&x, rows, &layer.norm, 1e-6)?;
            let mut q = layer.q.forward_verified(pool, &norm, rows)?;
            let mut k = layer.k.forward_verified(pool, &norm, rows)?;
            let v = layer.v.forward_verified(pool, &norm, rows)?;
            let mut rq = router_norm(layer.rq.forward_verified(pool, &norm, rows)?, &layer.rqn)?;
            let mut rk = router_norm(layer.rk.forward_verified(pool, &norm, rows)?, &layer.rkn)?;
            let coords = &pages.coords[start..start + rows];
            sparse::rope(&mut q, coords, 10, pages.page_rope);
            sparse::rope(&mut k, coords, 10, pages.page_rope);
            sparse::rope(&mut rq, coords, 1, pages.page_rope);
            sparse::rope(&mut rk, coords, 1, pages.page_rope);
            cache.k.extend(k);
            cache.v.extend(v);
            cache.index.extend(rk);
            let keys = cache.k.len() / HIDDEN;
            let summaries = summarize(&cache.index, &layer.cls);
            let nblocks = keys.div_ceil(32);
            let routes: Vec<_> = (0..rows)
                .map(|row| {
                    let pos = start + row;
                    let iq = &rq[row * 128..(row + 1) * 128];
                    let scores: Vec<_> = summaries
                        .chunks_exact(128)
                        .enumerate()
                        .map(|(b, k)| {
                            if b + 4 <= pos / 32 {
                                (crate::ops::dot_f32(iq, k, 128) / 128_f32.sqrt()).clamp(-1e4, 1e4)
                            } else {
                                -1e9
                            }
                        })
                        .collect();
                    sparse::route(&scores, &pages.candidate_blocks(pos, nblocks))
                })
                .collect();
            #[cfg(feature = "parity-trace")]
            if crate::parity_trace::enabled("uocr.routes") {
                let mut routed = vec![-1.0; rows * 24];
                for (row, route) in routes.iter().enumerate() {
                    for (j, &(b, bias)) in route.iter().enumerate() {
                        routed[row * 24 + j * 2] = b as f32;
                        routed[row * 24 + j * 2 + 1] = bias;
                    }
                }
                trace("uocr.routes", Some(i), &[rows, 12, 2], &routed);
            }
            let mut attended = vec![0.0; rows * HIDDEN];
            attended
                .par_chunks_mut(128)
                .enumerate()
                .for_each(|(j, out)| {
                    let row = j / 10;
                    let head = j % 10;
                    let pos = start + row;
                    let query = &q[j * 128..(j + 1) * 128];
                    let mut visible = Vec::new();
                    for &(block, bias) in &routes[row] {
                        for key in block * 32..((block + 1) * 32).min(keys).min(pos + 1) {
                            if key + 128 <= pos {
                                visible.push((key, bias));
                            }
                        }
                    }
                    for key in pos.saturating_sub(127)..=pos {
                        visible.push((key, 0.0));
                    }
                    visible.sort_unstable_by_key(|x| x.0);
                    let mut scores: Vec<_> = visible
                        .iter()
                        .map(|&(t, b)| {
                            let off = t * HIDDEN + head * 128;
                            crate::ops::dot_f32(query, &cache.k[off..off + 128], 128)
                                / 128_f32.sqrt()
                                + b
                        })
                        .collect();
                    crate::ops::softmax_inplace(&mut scores);
                    for ((t, _), p) in visible.iter().zip(scores) {
                        let off = t * HIDDEN + head * 128;
                        for d in 0..128 {
                            out[d] += p * cache.v[off + d];
                        }
                    }
                });
            trace("uocr.attention", Some(i), &[rows, HIDDEN], &attended);
            let projected = layer.o.forward_verified(pool, &attended, rows)?;
            for (dst, value) in x.iter_mut().zip(projected) {
                *dst += value;
            }
            let norm = rms_norm_rows(&x, rows, &layer.post, 1e-6)?;
            let mlp = if let Some(dense) = &layer.dense {
                dense.forward(pool, &norm)?
            } else {
                moe(layer, pool, &norm, i)?
            };
            for (dst, value) in x.iter_mut().zip(mlp) {
                *dst += value;
            }
            if x.iter().any(|v| !v.is_finite()) {
                return Err(format!("Non-finite Unlimited-OCR layer {i}"));
            }
            trace("uocr.layer", Some(i), &[rows, HIDDEN], &x);
        }
        let output = rms_norm_rows(&x, rows, &self.norm, 1e-6)?;
        trace("uocr.norm", None, &[rows, HIDDEN], &output);
        let logits = self
            .head
            .forward_verified(pool, &output[(rows - 1) * HIDDEN..], 1)?;
        if logits.iter().any(|v| !v.is_finite()) {
            return Err("Non-finite Unlimited-OCR logits".into());
        }
        trace("uocr.logits", None, &[VOCAB], &logits);
        Ok(logits)
    }
    pub fn prefill(
        &self,
        pool: &ComputePool,
        ids: &[u32],
        hidden: Vec<f32>,
    ) -> Result<Vec<f32>, String> {
        let pages = Pages::new(ids)?;
        self.forward(
            pool,
            hidden,
            &pages,
            0,
            &mut (0..12).map(|_| Cache::default()).collect::<Vec<_>>(),
        )
    }

    #[cfg(test)]
    pub fn generate(
        &self,
        pool: &ComputePool,
        ids: &[u32],
        hidden: Vec<f32>,
        max_tokens: usize,
        expected_pages: usize,
        dump_logits: bool,
    ) -> Result<Vec<u32>, String> {
        self.generate_stream(
            pool,
            ids,
            hidden,
            max_tokens,
            expected_pages,
            dump_logits,
            &mut |_| Ok(crate::ops::generation_runtime::Flow::Continue),
        )
        .map(|r| r.0)
    }

    pub fn generate_stream(
        &self,
        pool: &ComputePool,
        ids: &[u32],
        hidden: Vec<f32>,
        max_tokens: usize,
        expected_pages: usize,
        dump_logits: bool,
        on_token: &mut dyn FnMut(u32) -> Result<crate::ops::generation_runtime::Flow, String>,
    ) -> Result<(Vec<u32>, crate::ops::generation_runtime::Finish), String> {
        let mut pages = Pages::new(ids)?;
        let mut caches: Vec<_> = (0..12).map(|_| Cache::default()).collect();
        let mut logits = self.forward(pool, hidden, &pages, 0, &mut caches)?;
        let mut generated = Vec::new();
        let mut finish = crate::ops::generation_runtime::Finish::Limit;
        for step in 0..max_tokens {
            if dump_logits {
                println!("{}", serde_json::json!({"step":step,"logits":logits}));
            }
            page_guard(&mut logits, &generated, expected_pages, 20000);
            let next = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1).then(b.0.cmp(&a.0)))
                .unwrap()
                .0 as u32;
            if next == 1 {
                finish = crate::ops::generation_runtime::Finish::Eos;
                break;
            }
            generated.push(next);
            if on_token(next)? == crate::ops::generation_runtime::Flow::Stop {
                finish = crate::ops::generation_runtime::Finish::Cancelled;
                break;
            }
            if step + 1 == max_tokens {
                break;
            }
            pages.push(next);
            let mut x = vec![0.0; HIDDEN];
            self.embedding.embedding_lookup(next, &mut x)?;
            logits = self.forward(pool, x, &pages, ids.len() + step, &mut caches)?;
        }
        trace(
            "uocr.generated",
            None,
            &[generated.len()],
            &generated.iter().map(|x| *x as f32).collect::<Vec<_>>(),
        );
        Ok((generated, finish))
    }
}
fn router_norm(input: Vec<f32>, weight: &[f32]) -> Result<Vec<f32>, String> {
    let mut out = rms_norm_rows(&input, input.len() / 128, &[1.0; 128], 1e-6)?;
    for row in out.chunks_exact_mut(128) {
        for (x, w) in row.iter_mut().zip(weight) {
            *x = ((*x).clamp(-1e4, 1e4) * (1.0 + w)).clamp(-1e4, 1e4);
        }
    }
    Ok(out)
}
fn summarize(index: &[f32], cls: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0; index.len().div_ceil(32 * 128) * 128];
    for (block, dst) in index.chunks(32 * 128).zip(out.chunks_exact_mut(128)) {
        let mut scores: Vec<_> = block
            .chunks_exact(128)
            .map(|k| (crate::ops::dot_f32(k, cls, 128) / 128_f32.sqrt()).clamp(-1e4, 1e4))
            .collect();
        crate::ops::softmax_inplace(&mut scores);
        for (key, p) in block.chunks_exact(128).zip(scores) {
            for d in 0..128 {
                dst[d] += p * key[d];
            }
        }
    }
    out
}
fn moe(
    layer: &Layer<'_>,
    pool: &ComputePool,
    x: &[f32],
    layer_index: usize,
) -> Result<Vec<f32>, String> {
    let rows = x.len() / HIDDEN;
    let scores = layer
        .gate
        .as_ref()
        .unwrap()
        .forward_verified(pool, x, rows)?;
    trace("uocr.moe_gate", Some(layer_index), &[rows, 64], &scores);
    let mut assignments: Vec<Vec<(usize, f32)>> = (0..64).map(|_| Vec::new()).collect();
    for (row, logits) in scores.chunks_exact(64).enumerate() {
        let mut prob = logits.to_vec();
        crate::ops::softmax_inplace(&mut prob);
        let mut ranked: Vec<_> = (0..64).collect();
        ranked.sort_by(|&a, &b| prob[b].total_cmp(&prob[a]).then(a.cmp(&b)));
        for expert in ranked.into_iter().take(6) {
            assignments[expert].push((row, prob[expert]));
        }
    }
    let mut routed = vec![0.0; x.len()];
    for (expert, selected) in layer.experts.iter().zip(assignments) {
        if selected.is_empty() {
            continue;
        }
        let mut input = Vec::with_capacity(selected.len() * HIDDEN);
        for &(row, _) in &selected {
            input.extend_from_slice(&x[row * HIDDEN..(row + 1) * HIDDEN]);
        }
        let values = expert.forward(pool, &input)?;
        for ((row, weight), value) in selected.into_iter().zip(values.chunks_exact(HIDDEN)) {
            for d in 0..HIDDEN {
                routed[row * HIDDEN + d] += weight * value[d];
            }
        }
    }
    let shared = layer.shared.as_ref().unwrap().forward(pool, x)?;
    for (x, y) in routed.iter_mut().zip(shared) {
        *x += y;
    }
    Ok(routed)
}
fn page_guard(logits: &mut [f32], generated: &[u32], pages: usize, max_page_tokens: usize) {
    let count = generated.iter().filter(|&&id| id == PAGE).count();
    let page_len = generated
        .iter()
        .rposition(|&id| id == PAGE)
        .map_or(generated.len(), |i| generated.len() - i - 1);
    let forced = if count > pages {
        Some(1)
    } else if page_len >= max_page_tokens {
        Some(if count < pages { PAGE as usize } else { 1 })
    } else {
        None
    };
    if let Some(id) = forced {
        let keep = logits[id];
        logits.fill(f32::NEG_INFINITY);
        logits[id] = keep;
    } else if count < pages {
        logits[1] = f32::NEG_INFINITY;
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn page_guard_prevents_early_eos_and_caps_unfinished_pages() {
        let mut logits = vec![1.0; VOCAB];
        page_guard(&mut logits, &[], 2, 10);
        assert_eq!(logits[1], f32::NEG_INFINITY);
        let mut logits = vec![1.0; VOCAB];
        page_guard(&mut logits, &[PAGE, 4, 5], 2, 2);
        assert_eq!(logits[PAGE as usize], 1.0);
        assert_eq!(logits[3], f32::NEG_INFINITY);
        let mut logits = vec![1.0; VOCAB];
        page_guard(&mut logits, &[PAGE, PAGE, PAGE], 2, 10);
        assert_eq!(logits[1], 1.0);
        assert_eq!(logits[PAGE as usize], f32::NEG_INFINITY);
    }
}
