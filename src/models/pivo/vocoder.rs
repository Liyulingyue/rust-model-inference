use super::{sigmoid, Model};

pub(super) struct State {
    pub memory: Vec<f32>,
    pub gru: [Vec<f32>; 3],
    pub frame: Vec<f32>,
}

impl Default for State {
    fn default() -> Self {
        Self {
            memory: vec![0.; 256],
            gru: [vec![0.; 160], vec![0.; 128], vec![0.; 128]],
            frame: vec![0.; 164],
        }
    }
}

pub(super) fn period(features: &[f32]) -> usize {
    (256. / 2f32.powf(features[18].clamp(-10., 10.) + 1.5))
        .round_ties_even()
        .clamp(32., 255.) as usize
}

impl Model {
    pub(super) fn conditioning(&self, context: &[f32]) -> Vec<f32> {
        let mut input = Vec::with_capacity(3 * 32);
        for row in context.chunks_exact(20) {
            input.extend_from_slice(row);
            let index = period(row) - 32;
            input.extend_from_slice(
                &self.w("vocoder.conditioning.pitch_embedding.weight")[index * 12..][..12],
            );
        }
        let mut x = self.linear("vocoder.conditioning.dense1", &input);
        x.iter_mut().for_each(|v| *v = v.tanh());
        let mut x = self.conv("vocoder.conditioning.conv", &x, 1, 0, 1);
        x.iter_mut().for_each(|v| *v = v.tanh());
        let mut x = self.linear("vocoder.conditioning.dense2", &x);
        x.iter_mut().for_each(|v| *v = v.tanh());
        x
    }

    fn glu(&self, name: &str, input: &[f32]) -> Vec<f32> {
        input
            .iter()
            .zip(self.linear(&format!("{name}.gate"), input))
            .map(|(x, g)| (x * sigmoid(g)).clamp(-1., 1.))
            .collect()
    }

    fn gru(&self, name: &str, input: &[f32], hidden: &mut [f32]) {
        // PyTorch GRUCell stores reset, update, new gates; biases are absent.
        let project = |suffix: &str, x: &[f32]| {
            let w = self.w(&format!("{name}.weight_{suffix}"));
            w.chunks_exact(x.len())
                .map(|row| crate::ops::dot_f32(x, row, x.len()))
                .collect::<Vec<_>>()
        };
        let i = project("ih", input);
        let h = project("hh", hidden);
        let n = hidden.len();
        for j in 0..n {
            let reset = sigmoid(h[j] + i[j]);
            let update = sigmoid(h[n + j] + i[n + j]);
            let new = (i[2 * n + j] + h[2 * n + j] * reset).tanh();
            hidden[j] = update * (hidden[j] - new) + new;
        }
    }

    pub(super) fn vocoder_step(
        &self,
        conditioning: &[f32],
        period: usize,
        state: &mut State,
    ) -> Vec<f32> {
        let gain = self.linear("vocoder.step.cond_gain", conditioning)[0].exp();
        let pitch: Vec<f32> = (0..44)
            .map(|i| {
                let mut index = 256 + i as isize - period as isize - 2;
                if index >= 256 {
                    index -= period as isize;
                }
                (state.memory[index.max(0) as usize] / (gain + 1e-5)).clamp(-1., 1.)
            })
            .collect();
        let previous: Vec<f32> = state.memory[216..]
            .iter()
            .map(|v| (v / (gain + 1e-5)).clamp(-1., 1.))
            .collect();
        let frame = [conditioning, &pitch, &previous].concat();
        let input = [&state.frame[..], &frame].concat();
        state.frame = frame;
        let mut x = self.linear("vocoder.step.frame_conv", &input);
        x.iter_mut().for_each(|v| *v = v.tanh());
        let frame = self.glu("vocoder.step.frame_glu", &x);
        let gains: Vec<f32> = self
            .linear("vocoder.step.pitch_gain", &frame)
            .into_iter()
            .map(sigmoid)
            .collect();
        let mut outputs = Vec::with_capacity(3);
        for (i, gain) in gains.iter().take(3).enumerate() {
            let previous_layer = if i == 0 { &frame } else { &outputs[i - 1] };
            let scaled: Vec<f32> = pitch[2..42].iter().map(|p| p * gain).collect();
            let input = [&previous_layer[..], &scaled, &previous].concat();
            self.gru(
                &format!("vocoder.step.gru{}", i + 1),
                &input,
                &mut state.gru[i],
            );
            let clamped: Vec<f32> = state.gru[i].iter().map(|h| h.clamp(-1., 1.)).collect();
            outputs.push(self.glu(&format!("vocoder.step.glu{}", i + 1), &clamped));
        }
        let scaled: Vec<f32> = pitch[2..42].iter().map(|p| p * gains[3]).collect();
        let input = [
            &outputs[0][..],
            &outputs[1],
            &outputs[2],
            &frame,
            &scaled,
            &previous,
        ]
        .concat();
        let mut x = self.linear("vocoder.step.skip_dense", &input);
        x.iter_mut().for_each(|v| *v = v.tanh());
        let mut excitation = self.linear(
            "vocoder.step.signal_out",
            &self.glu("vocoder.step.skip_glu", &x),
        );
        excitation.iter_mut().for_each(|v| *v = v.tanh() * gain);
        state.memory.copy_within(40.., 0);
        state.memory[216..].copy_from_slice(&excitation);
        excitation
    }

    pub(super) fn vocode(&self, features: &[f32]) -> Vec<f32> {
        let frames = features.len() / 20;
        let mut state = State::default();
        let mut deemphasis = 0f64;
        let mut pcm = Vec::with_capacity(frames * 160);
        for t in 0..frames {
            let mut context = Vec::with_capacity(60);
            for p in [t.saturating_sub(1), t, (t + 1).min(frames - 1)] {
                context.extend_from_slice(&features[p * 20..][..20]);
            }
            let conditioning = self.conditioning(&context);
            let period = period(&context[20..40]);
            for subframe in conditioning.chunks_exact(80) {
                for v in self.vocoder_step(subframe, period, &mut state) {
                    deemphasis = f64::from(v) + 0.85 * deemphasis;
                    pcm.push(deemphasis as f32);
                }
            }
        }
        pcm
    }
}
