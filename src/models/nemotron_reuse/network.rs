use super::{trace, Model, CHANNELS, INNER, LAYERS, STATE};
use rayon::prelude::*;

struct Map {
    values: Vec<f32>,
    t: usize,
    f: usize,
    c: usize,
}

impl Map {
    fn transpose(self) -> Self {
        let mut values = vec![0.0; self.values.len()];
        for t in 0..self.t {
            for f in 0..self.f {
                values[(f * self.t + t) * self.c..(f * self.t + t + 1) * self.c].copy_from_slice(
                    &self.values[(t * self.f + f) * self.c..(t * self.f + f + 1) * self.c],
                );
            }
        }
        Self {
            values,
            t: self.f,
            f: self.t,
            c: self.c,
        }
    }
}

impl Model {
    fn linear(&self, prefix: &str, input: &[f32]) -> Vec<f32> {
        let tensor = &self.tensors[&format!("{prefix}.weight")];
        let (cout, cin) = (tensor.shape[0], tensor.shape[1]);
        let bias = self.tensors.get(&format!("{prefix}.bias"));
        let mut output = vec![0.0; input.len() / cin * cout];
        output
            .par_chunks_mut(cout)
            .zip(input.par_chunks(cin))
            .for_each(|(y, x)| {
                for (c, out) in y.iter_mut().enumerate() {
                    *out = crate::ops::dot_f32(x, &tensor.values[c * cin..(c + 1) * cin], cin)
                        + bias.map_or(0.0, |b| b.values[c]);
                }
            });
        output
    }

    fn conv(
        &self,
        prefix: &str,
        x: &Map,
        stride: [usize; 2],
        dilation: usize,
        pad: [usize; 2],
    ) -> Map {
        let tensor = &self.tensors[&format!("{prefix}.weight")];
        let [cout, cin, kt, kf] = <[usize; 4]>::try_from(tensor.shape.as_slice()).unwrap();
        debug_assert_eq!(cin, x.c);
        let t = (x.t + 2 * pad[0] - dilation * (kt - 1) - 1) / stride[0] + 1;
        let f = (x.f + 2 * pad[1] - (kf - 1) - 1) / stride[1] + 1;
        let width = cin * kt * kf;
        let bias = self.values(&format!("{prefix}.bias"));
        let mut values = vec![0.0; t * f * cout];
        values.par_chunks_mut(cout).enumerate().for_each_init(
            || vec![0.0; width],
            |patch, (row, out)| {
                for c in 0..cin {
                    for it in 0..kt {
                        for jf in 0..kf {
                            let ti =
                                ((row / f) * stride[0] + it * dilation) as isize - pad[0] as isize;
                            let fi = ((row % f) * stride[1] + jf) as isize - pad[1] as isize;
                            patch[(c * kt + it) * kf + jf] =
                                if ti >= 0 && fi >= 0 && ti < x.t as isize && fi < x.f as isize {
                                    x.values[(ti as usize * x.f + fi as usize) * cin + c]
                                } else {
                                    0.0
                                };
                        }
                    }
                }
                for (c, v) in out.iter_mut().enumerate() {
                    *v = crate::ops::dot_f32(
                        patch,
                        &tensor.values[c * width..(c + 1) * width],
                        width,
                    ) + bias[c];
                }
            },
        );
        Map {
            values,
            t,
            f,
            c: cout,
        }
    }

    fn activate(&self, prefix: &str, mut x: Map) -> Map {
        let weight = self.values(&format!("{prefix}.1.weight"));
        let bias = self.values(&format!("{prefix}.1.bias"));
        let prelu = self.values(&format!("{prefix}.2.weight"));
        let size = x.t * x.f;
        let stats: Vec<_> = (0..x.c)
            .map(|c| {
                let mean = x
                    .values
                    .iter()
                    .skip(c)
                    .step_by(x.c)
                    .map(|&v| v as f64)
                    .sum::<f64>()
                    / size as f64;
                let var = x
                    .values
                    .iter()
                    .skip(c)
                    .step_by(x.c)
                    .map(|&v| (v as f64 - mean).powi(2))
                    .sum::<f64>()
                    / size as f64;
                (mean as f32, 1.0 / (var as f32 + 1e-5).sqrt())
            })
            .collect();
        x.values.par_chunks_mut(x.c).for_each(|row| {
            for (c, v) in row.iter_mut().enumerate() {
                *v = (*v - stats[c].0) * stats[c].1 * weight[c] + bias[c];
                if *v < 0.0 {
                    *v *= prelu[c];
                }
            }
        });
        x
    }

    fn dense(&self, prefix: &str, x: Map) -> Map {
        let mut skip = x;
        for i in 0..4 {
            let p = format!("{prefix}.dense_block.{i}");
            let dilation = 1 << i;
            let y = self.activate(
                &p,
                self.conv(&format!("{p}.0"), &skip, [1, 1], dilation, [dilation, 1]),
            );
            if i == 3 {
                return y;
            }
            let mut values = Vec::with_capacity(y.values.len() + skip.values.len());
            for (new, old) in y
                .values
                .chunks_exact(y.c)
                .zip(skip.values.chunks_exact(skip.c))
            {
                values.extend_from_slice(new);
                values.extend_from_slice(old);
            }
            skip = Map {
                values,
                t: y.t,
                f: y.f,
                c: y.c + skip.c,
            };
        }
        unreachable!()
    }

    fn upsample(&self, prefix: &str, x: Map, scale: usize) -> Map {
        let y = self.conv(&format!("{prefix}.0.conv"), &x, [1, 1], 1, [0, 1]);
        let mut values = vec![0.0; y.values.len()];
        for row in 0..y.t * y.f {
            for r in 0..scale {
                // PyTorch splits channels as [r, channel], then interleaves frequency.
                values[(row * scale + r) * CHANNELS..(row * scale + r + 1) * CHANNELS]
                    .copy_from_slice(
                        &y.values[row * y.c + r * CHANNELS..row * y.c + (r + 1) * CHANNELS],
                    );
            }
        }
        self.activate(
            prefix,
            Map {
                values,
                t: y.t,
                f: y.f * scale,
                c: CHANNELS,
            },
        )
    }

    fn decode(&self, root: &str, x: Map) -> Map {
        let x = self.dense(&format!("{root}.dense_block"), x);
        let x = self.upsample(&format!("{root}.up_conv1"), x, 2);
        self.upsample(&format!("{root}.up_conv2"), x.transpose(), 4)
            .transpose()
    }

    fn mamba(&self, prefix: &str, input: &[f32], length: usize, reverse: bool) -> Vec<f32> {
        let mut x = input.to_vec();
        if reverse {
            for sequence in x.chunks_exact_mut(length * CHANNELS) {
                for t in 0..length / 2 {
                    for c in 0..CHANNELS {
                        sequence.swap(t * CHANNELS + c, (length - 1 - t) * CHANNELS + c);
                    }
                }
            }
        }
        let xz = self.linear(&format!("{prefix}.in_proj"), &x);
        let kernel = self.values(&format!("{prefix}.conv1d.weight"));
        let bias = self.values(&format!("{prefix}.conv1d.bias"));
        let mut convolved = vec![0.0; x.len() / CHANNELS * INNER];
        convolved
            .par_chunks_mut(length * INNER)
            .enumerate()
            .for_each(|(b, out)| {
                for t in 0..length {
                    for c in 0..INNER {
                        let mut value = bias[c];
                        for k in 0..4 {
                            if t + k >= 3 {
                                value += xz[((b * length + t + k - 3) * 2 * INNER) + c]
                                    * kernel[c * 4 + k];
                            }
                        }
                        out[t * INNER + c] = value;
                    }
                }
                crate::ops::silu_inplace(out);
            });
        let projected = self.linear(&format!("{prefix}.x_proj"), &convolved);
        let dt_input: Vec<_> = projected
            .chunks_exact(4 + 2 * STATE)
            .flat_map(|r| r[..4].iter().copied())
            .collect();
        let mut dt = self.linear(&format!("{prefix}.dt_proj"), &dt_input);
        for v in &mut dt {
            *v = if *v > 20.0 { *v } else { v.exp().ln_1p() };
        }
        let a: Vec<_> = self
            .values(&format!("{prefix}.A_log"))
            .iter()
            .map(|v| -v.exp())
            .collect();
        let d = self.values(&format!("{prefix}.D"));
        let mut scanned = vec![0.0; convolved.len()];
        scanned
            .par_chunks_mut(length * INNER)
            .enumerate()
            .for_each(|(b, out)| {
                let mut state = vec![0.0; INNER * STATE];
                for t in 0..length {
                    let row = b * length + t;
                    let bc = &projected[row * (4 + 2 * STATE) + 4..(row + 1) * (4 + 2 * STATE)];
                    for c in 0..INNER {
                        let u = convolved[row * INNER + c];
                        let delta = dt[row * INNER + c];
                        let state = &mut state[c * STATE..(c + 1) * STATE];
                        for n in 0..STATE {
                            state[n] =
                                (delta * a[c * STATE + n]).exp() * state[n] + delta * bc[n] * u;
                        }
                        let value = crate::ops::dot_f32(state, &bc[STATE..], STATE) + d[c] * u;
                        let z = xz[row * 2 * INNER + INNER + c];
                        out[t * INNER + c] = value * (z / (1.0 + (-z).exp()));
                    }
                }
            });
        let mut out = self.linear(&format!("{prefix}.out_proj"), &scanned);
        crate::ops::vec_add_into(&x, &mut out);
        if reverse {
            for sequence in out.chunks_exact_mut(length * CHANNELS) {
                for t in 0..length / 2 {
                    for c in 0..CHANNELS {
                        sequence.swap(t * CHANNELS + c, (length - 1 - t) * CHANNELS + c);
                    }
                }
            }
        }
        out
    }

    fn bidirectional(&self, prefix: &str, input: &[f32], length: usize) -> Vec<f32> {
        let fw = self.mamba(&format!("{prefix}.forward_blocks"), input, length, false);
        let bw = self.mamba(&format!("{prefix}.backward_blocks"), input, length, true);
        let mut cat = Vec::with_capacity(2 * input.len());
        for (a, b) in fw.chunks_exact(CHANNELS).zip(bw.chunks_exact(CHANNELS)) {
            cat.extend_from_slice(a);
            cat.extend_from_slice(b);
        }
        let projected = self.linear(&format!("{prefix}.output_proj"), &cat);
        let weight = self.values(&format!("{prefix}.norm.weight"));
        let bias = self.values(&format!("{prefix}.norm.bias"));
        let mut output = vec![0.0; input.len()];
        output
            .par_chunks_mut(CHANNELS)
            .zip(projected.par_chunks(CHANNELS))
            .zip(input.par_chunks(CHANNELS))
            .for_each(|((out, x), residual)| {
                crate::ops::layer_norm(x, weight, bias, 1e-5, out);
                crate::ops::vec_add_into(residual, out);
            });
        output
    }

    pub(super) fn forward(
        &self,
        mag: &[f32],
        phase: &[f32],
        t: usize,
        f: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let mut values = vec![0.0; (t + 2) * (f + 2) * 2];
        for i in 0..t {
            for j in 0..f {
                values[(i * (f + 2) + j) * 2] = mag[i * f + j];
                values[(i * (f + 2) + j) * 2 + 1] = phase[i * f + j];
            }
        }
        let x = Map {
            values,
            t: t + 2,
            f: f + 2,
            c: 2,
        };
        let x = self.activate(
            "dense_encoder.dense_conv_1",
            self.conv("dense_encoder.dense_conv_1.0", &x, [1, 1], 1, [0, 0]),
        );
        let x = self.dense("dense_encoder.dense_block", x);
        let mut x = self.activate(
            "dense_encoder.dense_conv_2",
            self.conv("dense_encoder.dense_conv_2.0", &x, [4, 2], 1, [0, 0]),
        );
        trace("reuse.encoder", &[x.t, x.f, x.c], &x.values);
        for i in 0..LAYERS {
            x = x.transpose();
            x.values = self.bidirectional(&format!("TSMamba.{i}.time_mamba"), &x.values, x.f);
            x = x.transpose();
            x.values = self.bidirectional(&format!("TSMamba.{i}.freq_mamba"), &x.values, x.f);
            trace(&format!("reuse.block.{i}"), &[x.t, x.f, x.c], &x.values);
        }
        let mag_features = self.decode(
            "mask_decoder",
            Map {
                values: x.values.clone(),
                t: x.t,
                f: x.f,
                c: x.c,
            },
        );
        let magnitude = self.conv("mask_decoder.final_conv", &mag_features, [1, 1], 1, [0, 0]);
        let phase_features = self.decode("phase_decoder", x);
        let re = self.conv(
            "phase_decoder.phase_conv_r",
            &phase_features,
            [1, 1],
            1,
            [0, 0],
        );
        let im = self.conv(
            "phase_decoder.phase_conv_i",
            &phase_features,
            [1, 1],
            1,
            [0, 0],
        );
        let mut mag = vec![0.0; t * f];
        let mut phase = vec![0.0; t * f];
        for ti in 0..t {
            for fi in 0..f {
                mag[ti * f + fi] = magnitude.values[ti * magnitude.f + fi];
                phase[ti * f + fi] = im.values[ti * im.f + fi].atan2(re.values[ti * re.f + fi]);
            }
        }
        trace("reuse.magnitude", &[t, f], &mag);
        trace("reuse.phase", &[t, f], &phase);
        (mag, phase)
    }
}
