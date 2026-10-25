use super::forecast::{scale_row, Row};
use super::graph::{self, Slot, TimestralOp, TimestralOpKind};
use super::*;

pub(crate) struct Layout {
    pub(crate) rows: usize,
    pub(crate) enc: usize,
    pub(crate) dec: usize,
    pub(crate) context: usize,
    pub(crate) covariates: bool,
    pub(crate) roles: Vec<u8>,
    pub(crate) valid: Vec<bool>,
    pub(crate) first: usize,
}

impl Layout {
    pub(crate) fn seq(&self) -> usize {
        self.enc + self.dec
    }
    pub(crate) fn time_allowed(&self, row: usize, q: usize, k: usize) -> bool {
        let valid = k < self.enc && self.valid[row * self.enc + k];
        if q < self.enc {
            return k < self.enc && ((valid && (self.roles[row] == 2 || k <= q)) || k == q);
        }
        let p = q - self.enc;
        if !self.covariates {
            return if k < self.enc {
                valid
            } else {
                k - self.enc <= p
            };
        }
        if k < self.enc {
            return p >= self.context && k < self.context && valid;
        }
        let kp = k - self.enc;
        kp == p || (p >= self.context && kp >= self.context && kp <= p)
    }
    pub(crate) fn variate_allowed(&self, slot: usize, q: usize, k: usize) -> bool {
        q == k
            || ((slot >= self.enc || self.valid[k * self.enc + slot])
                && !(self.roles[q] != 0 && self.roles[k] == 0)
                && !(self.roles[q] == 2 && self.roles[k] != 2))
    }
    pub(crate) fn position(&self, slot: usize) -> f32 {
        let p = if self.covariates {
            slot % self.enc
        } else {
            slot
        };
        p as f32 - self.first as f32
    }
}

// Timestral pairs the two halves of the full head, even with partial rotation.
fn rotate(x: &mut [f32], pos: f32) {
    for head in x.chunks_exact_mut(HD) {
        for i in 0..16 {
            let theta = pos / 10000f32.powf((2 * i) as f32 / 32.);
            let (sin, cos) = theta.sin_cos();
            let (a, b) = (head[i], head[i + HD / 2]);
            head[i] = a * cos - b * sin;
            head[i + HD / 2] = b * cos + a * sin;
        }
    }
}

impl TimestralModel<'_> {
    pub(super) fn predict_block(
        &self,
        rows: &[Row],
        future: &[Vec<Option<f32>>],
        offset: usize,
        h: usize,
    ) -> Result<Vec<Vec<f32>>, String> {
        let r = rows.len();
        let context = rows[0].values.len() / PS;
        let cov = rows.iter().any(|row| row.role != 0);
        let enc = context + if cov { h } else { 0 };
        let dec = if cov { enc } else { h };
        let s = enc + dec;
        let mut inputs = Vec::new();
        let mut locs = Vec::new();
        let mut scales = Vec::new();
        let mut valid = Vec::new();
        let mut patch_inputs = Vec::new();
        let mut first = enc;
        let mut observations = Vec::new();
        for (i, row) in rows.iter().enumerate() {
            let mut values = row.values.clone();
            let mut mask = row.observed.clone();
            if cov {
                for t in 0..h * PS {
                    let v = future[i].get(offset + t).copied().flatten();
                    values.push(v.unwrap_or(0.));
                    mask.push(v.is_some());
                }
            }
            let (scaled, loc, scale) = scale_row(&values, &mask, row.role == 2);
            let mut started = false;
            for p in 0..enc {
                let observed = mask[p * PS..(p + 1) * PS].iter().any(|b| *b);
                if observed {
                    first = first.min(p);
                }
                started |= observed;
                valid.push(started);
                patch_inputs.extend_from_slice(&scaled[p * PS..(p + 1) * PS]);
                patch_inputs.extend(mask[p * PS..(p + 1) * PS].iter().map(|b| {
                    if *b {
                        0.
                    } else {
                        1.
                    }
                }));
            }
            inputs.extend(scaled.iter().copied());
            locs.extend_from_slice(&loc);
            scales.extend_from_slice(&scale);
            observations.extend(mask.iter().map(|b| if *b { 1. } else { 0. }));
        }
        let layout = Layout {
            rows: r,
            enc,
            dec,
            context,
            covariates: cov,
            roles: rows.iter().map(|r| r.role).collect(),
            valid,
            first: if first == enc { 0 } else { first },
        };
        trace("observed", &[r, enc * PS], &observations)?;
        trace("scaled", &[r, enc * PS], &inputs)?;
        trace("loc", &[r, enc * PS], &locs)?;
        trace("scale", &[r, enc * PS], &scales)?;
        trace("patch_input", &[r, enc, 2 * PS], &patch_inputs)?;
        #[cfg(feature = "parity-trace")]
        if std::env::var_os("RMI_PARITY_TRACE").is_some() {
            let tm = (0..r)
                .flat_map(|row| (0..s).flat_map(move |q| (0..s).map(move |k| (row, q, k))))
                .map(|(row, q, k)| {
                    if layout.time_allowed(row, q, k) {
                        1.
                    } else {
                        0.
                    }
                })
                .collect::<Vec<_>>();
            let vm = (0..s)
                .flat_map(|slot| (0..r).flat_map(move |q| (0..r).map(move |k| (slot, q, k))))
                .map(|(slot, q, k)| {
                    if layout.variate_allowed(slot, q, k) {
                        1.
                    } else {
                        0.
                    }
                })
                .collect::<Vec<_>>();
            let pos = (0..r)
                .flat_map(|_| (0..s).map(|p| layout.position(p)))
                .collect::<Vec<_>>();
            trace("time_mask", &[r, 1, s, s], &tm)?;
            trace("variate_mask", &[s, 1, r, r], &vm)?;
            trace("positions", &[r, s], &pos)?;
        }
        let embed = self.residual_block("representation.patch_proj", &patch_inputs, 2 * PS, D);
        let query = self.weights["representation.forecast_query"];
        let mut x = Vec::with_capacity(r * s * D);
        for row in embed.chunks_exact(enc * D) {
            x.extend_from_slice(row);
            for _ in 0..dec {
                x.extend_from_slice(query);
            }
        }
        trace("embedding", &[r, s, D], &x)?;
        let out = self.transform(x, &layout)?;
        let mut decoder = Vec::with_capacity(r * dec * D);
        for row in out.chunks_exact(s * D) {
            decoder.extend_from_slice(&row[enc * D..]);
        }
        let raw = self.residual_block("head.projection", &decoder, D, PS * Q);
        trace("head_raw", &[r, dec * PS, Q], &raw)?;
        let mut preds = raw;
        for q in preds.chunks_exact_mut(Q) {
            monotone(q);
        }
        trace("head_monotone", &[r, dec * PS, Q], &preds)?;
        let mut result = Vec::with_capacity(r);
        for row in 0..r {
            let loc = locs[row * enc * PS + context * PS - 1];
            let scale = scales[row * enc * PS + context * PS - 1];
            let start = (row * dec * PS + if cov { context * PS } else { 0 }) * Q;
            let mut values = preds[start..start + h * PS * Q]
                .iter()
                .map(|v| v.sinh() * scale + loc)
                .collect::<Vec<_>>();
            for q in values.chunks_exact_mut(Q) {
                clamp_infinities(q);
            }
            result.push(values);
        }
        trace("block_quantiles", &[r, h * PS, Q], &result.concat())?;
        Ok(result)
    }

    fn transform(&self, x: Vec<f32>, layout: &Layout) -> Result<Vec<f32>, String> {
        use crate::compute::UsedBackend;
        #[cfg(feature = "vulkan")]
        if let Some(context) = self.compute.context().map_err(|e| e.to_string())? {
            let mut state = self
                .gpu
                .lock()
                .map_err(|_| "Timestral GPU state poisoned")?;
            let result = (|| {
                let count = layout.rows * layout.seq();
                if state.as_ref().is_none_or(|gpu| gpu.capacity() < count) {
                    // shortcut: growing the arena reuploads weights; retain uploads if resizing becomes frequent.
                    *state = None;
                    *state = Some(crate::vulkan::timestral::Transformer::new(
                        context,
                        &self.weights,
                        count,
                    )?);
                }
                state.as_ref().unwrap().forward(&x, layout)
            })();
            match result {
                Ok(out) => {
                    self.compute.trace(
                        "timestral.transformer",
                        UsedBackend::Vulkan,
                        layout.rows * layout.seq(),
                    );
                    return Ok(out);
                }
                Err(error) if self.compute == crate::compute::ComputePolicy::Auto => {
                    *state = None;
                    eprintln!("compute requested=Auto scope=timestral.transformer backend=Cpu reason={error}");
                }
                Err(error) => return Err(format!("Timestral Vulkan: {error}")),
            }
        }
        self.compute.trace(
            "timestral.transformer",
            UsedBackend::Cpu,
            layout.rows * layout.seq(),
        );
        self.transform_cpu(x, layout)
    }

    /// Execute the shared op list on the CPU.
    ///
    /// This is the CPU twin of `crate::vulkan::timestral::Transformer`: both
    /// consume `graph::transformer_ops()`, so the layer order, the weight
    /// names and the tensor layout are defined exactly once.
    fn transform_cpu(&self, x: Vec<f32>, layout: &Layout) -> Result<Vec<f32>, String> {
        let mut executor = CpuExecutor::new(self, layout, x)?;
        for op in graph::transformer_ops() {
            executor.execute(op)?;
        }
        Ok(executor.take(Slot::Norm))
    }
}

/// CPU executor for the shared Timestral op list.
///
/// Slots hold owned `Vec<f32>` buffers; the residual stream lives in
/// `Slot::X` for the whole run, every other slot is scratch.
struct CpuExecutor<'a, 'w> {
    model: &'a TimestralModel<'w>,
    layout: &'a Layout,
    slots: Vec<Vec<f32>>,
    count: usize,
    seq: usize,
    rows: usize,
}

impl<'a, 'w> CpuExecutor<'a, 'w> {
    fn new(
        model: &'a TimestralModel<'w>,
        layout: &'a Layout,
        input: Vec<f32>,
    ) -> Result<Self, String> {
        let count = layout.rows * layout.seq();
        let mut slots = vec![Vec::new(); Slot::COUNT];
        slots[Slot::X.index()] = input;
        Ok(Self {
            model,
            layout,
            slots,
            count,
            seq: layout.seq(),
            rows: layout.rows,
        })
    }

    fn take(&mut self, slot: Slot) -> Vec<f32> {
        std::mem::take(&mut self.slots[slot.index()])
    }

    fn put(&mut self, slot: Slot, values: Vec<f32>) {
        self.slots[slot.index()] = values;
    }

    fn borrow(&self, slot: Slot) -> &[f32] {
        &self.slots[slot.index()]
    }

    fn execute(&mut self, op: TimestralOp) -> Result<(), String> {
        let checkpoints = op.trace.clone();
        match op.kind {
            TimestralOpKind::RmsNorm {
                input,
                weight,
                output,
            } => {
                let dst = {
                    let src = &self.slots[input.index()];
                    let scale = self.model.weight(&weight);
                    // `rms_norm` only touches `min(input.len(), weight.len())`
                    // elements, so each row has to be normalized on its own.
                    let mut dst = vec![0.; src.len()];
                    for (out, inp) in dst.chunks_exact_mut(D).zip(src.chunks_exact(D)) {
                        crate::ops::rms_norm(inp, scale, out, 1e-6);
                    }
                    dst
                };
                self.put(output, dst);
            }
            TimestralOpKind::Linear {
                input,
                weight,
                output,
                n_in,
                n_out,
            } => {
                let dst = {
                    let src = &self.slots[input.index()];
                    self.model.linear_named(&weight, src, n_in, n_out)
                };
                self.put(output, dst);
            }
            TimestralOpKind::Qkv {
                fused,
                q,
                k,
                v,
                rope: _,
                time,
            } => {
                let fused = self.take(fused);
                let (mut qv, mut kv, mut vv) = (
                    vec![0.; self.count * D],
                    vec![0.; self.count * D],
                    vec![0.; self.count * D],
                );
                for token in 0..self.count {
                    for head in 0..HEADS {
                        for dim in 0..HD {
                            let dst = token * D + head * HD + dim;
                            let src = token * 3 * D + dim * HEADS + head;
                            qv[dst] = fused[src];
                            kv[dst] = fused[src + D];
                            vv[dst] = fused[src + 2 * D];
                        }
                    }
                    if time {
                        rotate(
                            &mut qv[token * D..(token + 1) * D],
                            self.layout.position(token % self.seq),
                        );
                        rotate(
                            &mut kv[token * D..(token + 1) * D],
                            self.layout.position(token % self.seq),
                        );
                    }
                }
                self.put(q, qv);
                self.put(k, kv);
                self.put(v, vv);
            }
            TimestralOpKind::Attention {
                q,
                k,
                v,
                out,
                masks: _,
                roles: _,
                time,
            } => {
                let values = {
                    let q = &self.slots[q.index()];
                    let k = &self.slots[k.index()];
                    let v = &self.slots[v.index()];
                    self.attention(q, k, v, time)
                };
                self.put(out, values);
            }
            TimestralOpKind::Gate { attn, gate, .. } => {
                let mut values = self.take(attn);
                let gate = self.take(gate);
                for (o, g) in values.iter_mut().zip(gate) {
                    *o *= 1. / (1. + (-g).exp());
                }
                self.put(attn, values);
            }
            TimestralOpKind::SwiGlu { fused, hidden } => {
                let fused = self.take(fused);
                let mut hidden_values = vec![0.; self.count * F];
                for (out, inp) in hidden_values
                    .chunks_exact_mut(F)
                    .zip(fused.chunks_exact(2 * F))
                {
                    for i in 0..F {
                        out[i] = inp[i] * crate::ops::silu(inp[F + i]);
                    }
                }
                self.put(hidden, hidden_values);
            }
            TimestralOpKind::Add { input, output } => {
                let addition = self.take(input);
                let mut acc = self.take(output);
                for (a, b) in acc.iter_mut().zip(addition) {
                    *a += b;
                }
                self.put(output, acc);
            }
        }
        // Record the parity checkpoints this op produced.
        for (name, slot) in checkpoints {
            let values = self.borrow(slot);
            let shape: Vec<usize> = match slot {
                Slot::Q | Slot::K | Slot::V => vec![self.rows, self.seq, HEADS, HD],
                _ => vec![self.rows, self.seq, D],
            };
            super::trace(&name, &shape, values)?;
        }
        Ok(())
    }

    /// Dual-axis masked attention.
    ///
    /// The time axis attends over the sequence, the variate axis over the
    /// target rows; `Layout` owns both masks.
    fn attention(&self, q: &[f32], k: &[f32], v: &[f32], time: bool) -> Vec<f32> {
        let layout = self.layout;
        let s = self.seq;
        let rows = self.rows;
        let count = self.count;
        let mut out_values = vec![0.; count * D];
        // Tokens are independent, so the rayon pool splits them.
        out_values
            .par_chunks_mut(D)
            .enumerate()
            .for_each(|(token, token_out)| {
                let row = token / s;
                let slot = token % s;
                let keys = if time { s } else { rows };
                let mut scores = vec![0.; keys];
                for head in 0..HEADS {
                    let query = &q[token * D + head * HD..token * D + (head + 1) * HD];
                    for (j, score) in scores.iter_mut().enumerate() {
                        let allowed = if time {
                            layout.time_allowed(row, slot, j)
                        } else {
                            layout.variate_allowed(slot, row, j)
                        };
                        let key = if time { row * s + j } else { j * s + slot };
                        *score = if allowed {
                            crate::ops::dot_f32(
                                query,
                                &k[key * D + head * HD..key * D + (head + 1) * HD],
                                HD,
                            ) / 8.
                        } else {
                            f32::NEG_INFINITY
                        };
                    }
                    crate::ops::softmax_inplace(&mut scores);
                    for (j, score) in scores.iter().enumerate() {
                        let key = if time { row * s + j } else { j * s + slot };
                        for dim in 0..HD {
                            token_out[head * HD + dim] += score * v[key * D + head * HD + dim];
                        }
                    }
                }
            });
        out_values
    }
}

fn monotone(q: &mut [f32]) {
    let median = q[MEDIAN];
    let mut total = 0.;
    for i in (0..MEDIAN).rev() {
        total += 3. * (LEVELS[i + 1] - LEVELS[i]) * softplus(q[i]);
        q[i] = median - total;
    }
    total = 0.;
    for i in MEDIAN + 1..Q {
        total += 3. * (LEVELS[i] - LEVELS[i - 1]) * softplus(q[i]);
        q[i] = median + total;
    }
    for v in q {
        *v = v.clamp(-f32::MAX, f32::MAX);
    }
}
fn softplus(x: f32) -> f32 {
    if x > 20. {
        x
    } else {
        x.exp().ln_1p()
    }
}
pub(super) fn clamp_infinities(q: &mut [f32]) {
    let min = q
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(f32::INFINITY, f32::min);
    let max = q
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(f32::NEG_INFINITY, f32::max);
    for x in q {
        if *x == f32::INFINITY {
            *x = if max.is_finite() { max } else { 0. };
        }
        if *x == f32::NEG_INFINITY {
            *x = if min.is_finite() { min } else { 0. };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn partial_rope_pairs_full_head_halves() {
        let mut x = (0..D).map(|i| i as f32).collect::<Vec<_>>();
        let old = x.clone();
        rotate(&mut x, 1.);
        assert_eq!(x[16..32], old[16..32]);
        assert_eq!(x[48..64], old[48..64]);
        assert!((x[0] + 32. * 1f32.sin()).abs() < 1e-5);
        assert!((x[32] - 32. * 1f32.cos()).abs() < 1e-5);
    }
    #[test]
    fn quantiles_are_monotone_and_overflow_clamps_to_finite_neighbor() {
        let mut x = vec![0.; Q];
        monotone(&mut x);
        assert!(x.windows(2).all(|w| w[0] < w[1]));
        assert_eq!(x[MEDIAN], 0.);
        let mut extreme = vec![f32::MAX; Q];
        monotone(&mut extreme);
        assert!(extreme.iter().all(|v| v.is_finite()));
        let mut x = [f32::NEG_INFINITY, -2., 1., f32::INFINITY];
        clamp_infinities(&mut x);
        assert_eq!(x, [-2., -2., 1., 1.]);
    }
    #[test]
    fn masks_keep_covariate_firewalls_and_causality() {
        let l = Layout {
            rows: 3,
            enc: 3,
            dec: 3,
            context: 2,
            covariates: true,
            roles: vec![0, 1, 2],
            valid: vec![false, true, false, true, true, false, true, true, true],
            first: 0,
        };
        assert!(!l.time_allowed(0, 0, 1));
        assert!(l.time_allowed(0, 0, 0));
        assert!(!l.time_allowed(0, 5, 2));
        assert!(l.time_allowed(2, 0, 2));
        assert!(!l.time_allowed(0, 3, 0));
        assert!(l.time_allowed(0, 5, 1));
        assert!(!l.variate_allowed(5, 1, 0));
        assert!(!l.variate_allowed(5, 2, 1));
        assert!(l.variate_allowed(5, 0, 2));
        assert!(l.variate_allowed(0, 0, 0));
    }
}
