use super::{add, silu, Model};

impl Model {
    fn attention_block(&self, name: &str, input: &[f32], causal: bool, rotary: bool) -> Vec<f32> {
        let n = input.len() / 96;
        let normalized = self.norm(&format!("{name}.norm1"), input);
        let p = format!("{name}.attention");
        let mut q = self.norm(
            &format!("{p}.q_norm"),
            &self.linear(&format!("{p}.q"), &normalized),
        );
        let mut k = self.norm(
            &format!("{p}.k_norm"),
            &self.linear(&format!("{p}.k"), &normalized),
        );
        let v = self.linear(&format!("{p}.v"), &normalized);
        if rotary {
            rope(&mut q, 4);
            rope(&mut k, 2);
        }
        let mut attended = vec![0.; n * 96];
        for t in 0..n {
            for h in 0..4 {
                let start = if causal { t.saturating_sub(63) } else { 0 };
                let end = if causal { t + 1 } else { n };
                let query = &q[t * 96 + h * 24..t * 96 + (h + 1) * 24];
                let mut scores: Vec<f32> = (start..end)
                    .map(|s| {
                        crate::ops::dot_f32(query, &k[s * 48 + (h / 2) * 24..][..24], 24)
                            / 24f32.sqrt()
                    })
                    .collect();
                crate::ops::softmax_inplace(&mut scores);
                for (s, score) in (start..end).zip(scores) {
                    for d in 0..24 {
                        attended[t * 96 + h * 24 + d] += score * v[s * 48 + (h / 2) * 24 + d];
                    }
                }
            }
        }
        let residual = add(input, &self.linear(&format!("{p}.out"), &attended));
        let normalized = self.norm(&format!("{name}.norm2"), &residual);
        let mut gate = self.linear(&format!("{name}.gate"), &normalized);
        silu(&mut gate);
        let up = self.linear(&format!("{name}.up"), &normalized);
        gate.iter_mut().zip(up).for_each(|(g, u)| *g *= u);
        add(&residual, &self.linear(&format!("{name}.down"), &gate))
    }

    // Gather the temporal window in the checkpoint's [channel, kernel] order.
    pub(super) fn conv(
        &self,
        name: &str,
        input: &[f32],
        dilation: usize,
        padding: usize,
        groups: usize,
    ) -> Vec<f32> {
        let weight = &self.weights[&format!("{name}.weight")];
        let (out, group_width, kernel) = (weight.shape[0], weight.shape[1], weight.shape[2]);
        let width = group_width * groups;
        let n = input.len() / width;
        let rows = n + 2 * padding - dilation * (kernel - 1);
        let bias = self.weights.get(&format!("{name}.bias"));
        let mut result = vec![0.; rows * out];
        let mut window = vec![0.; group_width * kernel];
        for t in 0..rows {
            for g in 0..groups {
                window.fill(0.);
                for c in 0..group_width {
                    for j in 0..kernel {
                        let pos = (t + j * dilation) as isize - padding as isize;
                        if (0..n as isize).contains(&pos) {
                            window[c * kernel + j] =
                                input[pos as usize * width + g * group_width + c];
                        }
                    }
                }
                for o in g * (out / groups)..(g + 1) * (out / groups) {
                    let w = &weight.data[o * window.len()..][..window.len()];
                    result[t * out + o] = crate::ops::dot_f32(&window, w, window.len())
                        + bias.map_or(0., |b| b.data[o]);
                }
            }
        }
        result
    }

    /// The token embeddings with the speaker vector added.
    ///
    /// The gather stays on the host in the device path too: the token stream
    /// is short and the copy is a handful of elements per token.
    pub(super) fn embed(&self, tokens: &[usize], voice: usize) -> Vec<f32> {
        let embeddings = self.w("acoustic.tokens.weight");
        let speaker = &self.w("acoustic.voices.weight")[voice * 96..][..96];
        tokens
            .iter()
            .flat_map(|&id| add(&embeddings[id * 96..][..96], speaker))
            .collect()
    }

    /// The six text blocks and the final norm.
    pub(super) fn text_blocks(&self, x: &[f32]) -> Vec<f32> {
        let mut x = x.to_vec();
        for i in 0..6 {
            let p = format!("acoustic.text_blocks.{i}");
            x = if i % 2 == 0 {
                self.attention_block(&p, &x, false, true)
            } else {
                let normalized = self.norm(&format!("{p}.norm1"), &x);
                let dilation = 1 << (i / 2);
                let conv = self.conv(&format!("{p}.conv"), &normalized, dilation, 2 * dilation, 1);
                let mut output = self.norm(&format!("{p}.norm2"), &conv);
                silu(&mut output);
                add(&x, &output)
            };
        }
        self.norm("acoustic.text_norm", &x)
    }

    /// The whole encoder on the host.
    pub(super) fn encode(&self, tokens: &[usize], voice: usize) -> (Vec<f32>, Vec<f32>) {
        let x = self.embed(tokens, voice);
        let encoded = self.text_blocks(&x);
        let log_durations = self.duration_head(&encoded);
        (encoded, log_durations)
    }

    /// The frame duration head.
    ///
    /// The head produces the integer frame durations, and the device path
    /// keeps it on the host: the rounding of the durations is sensitive, and
    /// the host and the device have to agree on the log duration to far
    /// better than the rounding margin.
    pub(super) fn duration_head(&self, encoded: &[f32]) -> Vec<f32> {
        let mut durations = self.linear("acoustic.durations.in_proj", encoded);
        silu(&mut durations);
        for i in 0..2 {
            let p = format!("acoustic.durations.layers.{i}");
            durations = self.conv(&format!("{p}.0"), &durations, 1, 1, 64);
            durations = self.linear(&format!("{p}.1"), &durations);
            silu(&mut durations);
            durations = self.norm(&format!("acoustic.durations.norms.{i}"), &durations);
        }
        self.linear("acoustic.durations.out", &durations)
    }

    pub(super) fn decode(&self, expanded: &[f32]) -> Vec<f32> {
        let mut x = Vec::with_capacity(expanded.len() / 100 * 96);
        for row in expanded.chunks_exact(100) {
            x.extend(add(
                &row[..96],
                &self.linear("acoustic.phase_projection", &row[96..]),
            ));
        }
        for i in 0..2 {
            // The pinned export uses RoPE only in the first decoder block.
            x = self.attention_block(&format!("acoustic.decoder.{i}"), &x, true, i == 0);
        }
        let mut features = self.linear(
            "acoustic.feature_head",
            &self.norm("acoustic.feature_norm", &x),
        );
        let mean = self.w("acoustic.feature_mean");
        let std = self.w("acoustic.feature_std");
        for (i, v) in features.iter_mut().enumerate() {
            *v = *v * std[i % 20] + mean[i % 20];
        }
        features
    }
}

fn rope(values: &mut [f32], heads: usize) {
    for (t, row) in values.chunks_exact_mut(heads * 24).enumerate() {
        for head in row.chunks_exact_mut(24) {
            for i in 0..12 {
                let theta = t as f32 * ((2 * i) as f32 * (-10000f32.ln() / 24.)).exp();
                let (sin, cos) = theta.sin_cos();
                let (a, b) = (head[i], head[i + 12]);
                head[i] = a * cos - b * sin;
                head[i + 12] = b * cos + a * sin;
            }
        }
    }
}

pub(super) fn durations(tokens: &[usize], logs: &[f32], speed: f32) -> Result<Vec<usize>, String> {
    if logs.iter().any(|v| !v.is_finite()) {
        return Err("Non-finite PiVo durations".into());
    }
    let result: Vec<usize> = tokens
        .iter()
        .zip(logs)
        .map(|(&id, &value)| {
            if id <= 2 {
                return 0;
            }
            let frames = (value.clamp(0., 201f32.ln()).exp_m1() / speed)
                .round_ties_even()
                .clamp(0., 200.) as usize;
            frames.max(usize::from(id >= 10))
        })
        .collect();
    let count: usize = result.iter().sum();
    if count == 0 || count > 3000 {
        return Err("PiVo text exceeds the utterance frame budget or produces no frames; split it into clauses".into());
    }
    Ok(result)
}

pub(super) fn expand(encoded: &[f32], durations: &[usize]) -> Vec<f32> {
    let mut expanded = Vec::with_capacity(durations.iter().sum::<usize>() * 100);
    for (row, &count) in encoded.chunks_exact(96).zip(durations) {
        for t in 0..count {
            expanded.extend_from_slice(row);
            let fraction = (t as f32 + 0.5) / count as f32;
            expanded.extend_from_slice(&[
                2. * fraction - 1.,
                (std::f32::consts::PI * fraction).sin(),
                (std::f32::consts::PI * fraction).cos(),
                (count as f32).ln_1p() / 201f32.ln(),
            ]);
        }
    }
    expanded
}
