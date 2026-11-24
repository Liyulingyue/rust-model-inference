use super::{palette::Pixel, COLORS, GRID, MASK, PIXELS};
use crate::{models::diffusion::z_image::dit::TorchMt19937, ops::softmax_inplace};
use serde::Deserialize;

#[derive(Clone, Copy, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GuidanceSchedule {
    #[default]
    Constant,
    Linear,
    Cosine,
}

/// Optional controls for the published confidence-ordered sampler. A null
/// canvas entry is regenerated; a palette ID (including transparent 0) is kept.
#[derive(Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PixelOptions {
    pub canvas: Option<Vec<Option<Pixel>>>,
    pub choice_temperature: f32,
    pub temperature_end: Option<f32>,
    pub cfg_schedule: GuidanceSchedule,
    pub top_k: usize,
    pub top_p: f32,
    pub min_p: f32,
    pub spatial_nms: bool,
    pub nms_threshold: f32,
    pub commit_threshold: f32,
    pub polish_steps: usize,
    pub polish_frac: f32,
    pub self_cond: bool,
}

impl Default for PixelOptions {
    fn default() -> Self {
        Self {
            canvas: None,
            choice_temperature: 2.,
            temperature_end: None,
            cfg_schedule: GuidanceSchedule::Constant,
            top_k: 0,
            top_p: 1.,
            min_p: 0.,
            spatial_nms: true,
            nms_threshold: 0.9,
            commit_threshold: 0.,
            polish_steps: 0,
            polish_frac: 0.08,
            self_cond: true,
        }
    }
}

#[derive(Clone)]
pub struct Options {
    pub steps: usize,
    pub seed: u64,
    pub guidance: f32,
    pub temperature: f32,
    pub pixel: PixelOptions,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            steps: 16,
            seed: 0,
            guidance: 3.,
            temperature: 1.,
            pixel: PixelOptions::default(),
        }
    }
}

impl Options {
    pub fn validate(&self) -> Result<(), String> {
        let p = &self.pixel;
        if !(1..=PIXELS).contains(&self.steps) || p.polish_steps > PIXELS {
            return Err("Tessera accepts 1..=256 steps and 0..=256 polish steps".into());
        }
        if [
            self.temperature,
            self.guidance,
            p.choice_temperature,
            p.temperature_end.unwrap_or(0.),
        ]
        .iter()
        .any(|x| !x.is_finite() || *x < 0.)
        {
            return Err("guidance and temperatures must be finite and non-negative".into());
        }
        if p.top_k > COLORS
            || [
                p.top_p,
                p.min_p,
                p.nms_threshold,
                p.commit_threshold,
                p.polish_frac,
            ]
            .iter()
            .any(|x| !(0.0..=1.0).contains(x))
        {
            return Err("invalid top_k or pixel sampling probability".into());
        }
        if p.canvas.as_ref().is_some_and(|c| {
            c.len() != PIXELS
                || c.iter()
                    .flatten()
                    .any(|v| matches!(v, Pixel::Id(id) if *id >= MASK))
        }) {
            return Err("canvas must have 256 row-major entries, each null, RGBA bytes or a palette ID 0..511".into());
        }
        Ok(())
    }

    fn guidance_at(&self, frac: f32) -> f32 {
        let factor = match self.pixel.cfg_schedule {
            GuidanceSchedule::Constant => return self.guidance,
            GuidanceSchedule::Linear => frac,
            GuidanceSchedule::Cosine => (1. - (std::f32::consts::PI * frac).cos()) / 2.,
        };
        1. + (self.guidance - 1.) * factor
    }
}

fn argmax(row: &[f32]) -> usize {
    let mut best = 0;
    for i in 1..row.len() {
        if row[i] > row[best] {
            best = i;
        }
    }
    best
}

fn filter(row: &mut [f32], p: &PixelOptions) {
    if p.top_k > 0 && p.top_k < COLORS {
        let mut sorted = row.to_vec();
        let (_, kth, _) = sorted.select_nth_unstable_by(p.top_k - 1, |a, b| b.total_cmp(a));
        for v in row.iter_mut().filter(|v| **v < *kth) {
            *v = f32::NEG_INFINITY;
        }
    }
    if p.min_p > 0. {
        let cutoff = row[argmax(row)] + p.min_p.ln();
        for v in row.iter_mut().filter(|v| **v < cutoff) {
            *v = f32::NEG_INFINITY;
        }
    }
    softmax_inplace(row);
    if p.top_p < 1. {
        let mut order: Vec<_> = (0..COLORS).collect();
        order.sort_by(|&a, &b| row[b].total_cmp(&row[a]));
        let mut sum = 0.;
        for i in order {
            if sum > p.top_p {
                row[i] = 0.;
            } else {
                sum += row[i];
            }
        }
        for v in row {
            *v /= sum;
        }
    }
}

fn priorities(mask: &[u16], score: &[f32], confidence: &[f32], opt: &PixelOptions) -> Vec<usize> {
    let mut priority = score.to_vec();
    let mut order = Vec::new();
    for (i, &token) in mask.iter().enumerate() {
        if token != MASK {
            continue;
        }
        order.push(i);
        if !opt.spatial_nms {
            continue;
        }
        let (y, x) = (i / GRID, i % GRID);
        let eligible = confidence[i] >= opt.nms_threshold
            || (y.saturating_sub(1)..=(y + 1).min(GRID - 1)).all(|ny| {
                (x.saturating_sub(1)..=(x + 1).min(GRID - 1)).all(|nx| {
                    let j = ny * GRID + nx;
                    mask[j] != MASK || confidence[j] >= opt.nms_threshold || score[j] <= score[i]
                })
            });
        if eligible {
            priority[i] += 1e4;
        }
    }
    order.sort_by(|&a, &b| priority[b].total_cmp(&priority[a]));
    order
}

/// Predict returns guided, unnormalized logits in [pixel, palette ID] order.
pub(super) fn sample(
    opt: &Options,
    mut canvas: Vec<u16>,
    mut predict: impl FnMut(&[u16], Option<&[f32]>, f32) -> Result<Vec<f32>, String>,
) -> Result<Vec<u16>, String> {
    opt.validate()?;
    let p = &opt.pixel;

    let start = canvas.iter().filter(|&&v| v == MASK).count();
    if start == 0 {
        return Ok(canvas);
    }
    let mut rng = TorchMt19937::new(opt.seed);
    let mut commit_conf = vec![f32::INFINITY; PIXELS];
    let mut sc = None;
    for k in 0..opt.steps + p.polish_steps {
        let polish = k >= opt.steps;
        let frac = ((k + 1) as f64 / opt.steps as f64).min(1.);
        if polish {
            let mut low: Vec<_> = (0..PIXELS)
                .filter(|&i| commit_conf[i].is_finite())
                .collect();
            low.sort_by(|&a, &b| commit_conf[a].total_cmp(&commit_conf[b]));
            for &i in low
                .iter()
                .take((p.polish_frac * PIXELS as f32).round_ties_even().max(1.) as usize)
            {
                canvas[i] = MASK;
            }
        }
        let masked = canvas.iter().filter(|&&id| id == MASK).count();
        if masked == 0 {
            continue;
        }
        let logits = predict(&canvas, sc.as_deref(), opt.guidance_at(frac as f32))?;
        if logits.len() != PIXELS * COLORS || logits.iter().any(|v| !v.is_finite()) {
            return Err("invalid Tessera logits".into());
        }
        let mut probs = logits.clone();
        for row in probs.chunks_exact_mut(COLORS) {
            softmax_inplace(row);
        }
        let temp = p.temperature_end.map_or(opt.temperature, |end| {
            opt.temperature + (end - opt.temperature) * k as f32 / (opt.steps - 1).max(1) as f32
        });
        let mut ids = vec![0; PIXELS];
        let mut confidence = vec![0.; PIXELS];
        for (i, row) in probs.chunks_exact(COLORS).enumerate() {
            let (id, prob) = if temp == 0. || polish {
                let j = argmax(&logits[i * COLORS..(i + 1) * COLORS]);
                (j, row[j])
            } else {
                let mut filtered;
                let distribution = if temp == 1. && p.top_k == 0 && p.top_p == 1. && p.min_p == 0. {
                    row
                } else {
                    filtered = logits[i * COLORS..(i + 1) * COLORS].to_vec();
                    let shift = filtered[argmax(&filtered)];
                    for v in &mut filtered {
                        *v = (*v - shift) / temp;
                    }
                    filter(&mut filtered, p);
                    &filtered
                };
                // PyTorch CPU multinomial with one sample uses an exponential
                // race, consuming one double uniform per color, even on kept pixels.
                let mut best = 0;
                let mut high = -1.;
                for (j, &v) in distribution.iter().enumerate() {
                    let exp = -(-rng.sample_uniform_f64()).ln_1p() as f32;
                    let val = v / exp;
                    if val > high {
                        best = j;
                        high = val;
                    }
                }
                (best, distribution[best])
            };
            ids[i] = id as u16;
            confidence[i] = prob;
        }
        let mut score = confidence
            .iter()
            .map(|v| v.max(1e-12).ln())
            .collect::<Vec<_>>();
        if !polish && p.choice_temperature > 0. {
            for v in &mut score {
                let u = rng.sample_uniform_f32().max(1e-10);
                *v -= p.choice_temperature * (1. - frac as f32) * (-u.ln()).ln();
            }
        }
        let remaining = if frac == 1. {
            0
        } else {
            (start as f32 * (frac * std::f64::consts::FRAC_PI_2).cos() as f32).floor() as usize
        };
        let threshold = if p.commit_threshold > 0. {
            (0..PIXELS)
                .filter(|&i| canvas[i] == MASK && confidence[i] >= p.commit_threshold)
                .count()
        } else {
            0
        };
        let commit = masked.saturating_sub(remaining).max(1).max(threshold);
        for i in priorities(&canvas, &score, &confidence, p)
            .into_iter()
            .take(commit)
        {
            canvas[i] = ids[i];
            commit_conf[i] = confidence[i];
        }
        if p.self_cond {
            sc = Some(probs);
        }
    }
    Ok(canvas)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn inpaint_preserves_known_pixels_including_transparency_during_polish() {
        let mut canvas = vec![0; PIXELS];
        canvas[77] = MASK;
        canvas[31] = 15;
        let mut opt = Options::default();
        opt.pixel.polish_steps = 1;
        opt.temperature = 0.;
        let mut calls = 0;
        let result = sample(&opt, canvas, |ids, sc, _| {
            assert_eq!(ids.iter().filter(|&&id| id == MASK).count(), 1);
            assert_eq!(sc.is_some(), calls > 0);
            calls += 1;
            let mut logits = vec![0.; PIXELS * COLORS];
            logits[77 * COLORS + 13] = 10.;
            Ok(logits)
        })
        .unwrap();
        assert_eq!(calls, 2);
        assert_eq!(result[77], 13);
        assert_eq!(result[31], 15);
        assert_eq!(result.iter().filter(|&&t| t == 0).count(), 254);
    }
}
