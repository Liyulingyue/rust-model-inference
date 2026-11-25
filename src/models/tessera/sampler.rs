use super::{palette::Pixel, COLORS, GRID, MASK, PIXELS};
use crate::{models::diffusion::z_image::dit::TorchMt19937, ops::softmax_inplace};
use serde::Deserialize;

#[derive(Debug, Clone, Copy, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GuidanceSchedule {
    #[default]
    Constant,
    Linear,
    Cosine,
}

/// Optional controls for the published confidence-ordered sampler. A null
/// canvas entry is regenerated; a palette ID (including transparent 0) is kept.
#[derive(Debug, Clone, Deserialize)]
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

#[derive(Debug, Clone)]
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

pub(super) struct DrawOptions<'a> {
    pub pixel: &'a PixelOptions,
    pub temperature: f32,
    pub guidance: f32,
}

pub(super) struct Draw {
    pub probs: Vec<f32>,
    pub ids: Vec<u16>,
    pub confidence: Vec<f32>,
    pub best: Vec<u16>,
}

pub(super) fn draw_cpu(
    mut cond: Vec<f32>,
    uncond: Option<Vec<f32>>,
    opts: &DrawOptions<'_>,
    noise: &[f32],
) -> Result<Draw, String> {
    if let Some(uncond) = uncond {
        for (c, u) in cond.iter_mut().zip(uncond) {
            *c = u + opts.guidance * (*c - u);
        }
    }
    if cond.len() != PIXELS * COLORS || cond.iter().any(|v| !v.is_finite()) {
        return Err("invalid Tessera logits".into());
    }
    let p = opts.pixel;
    let mut probs = cond.clone();
    let mut ids = Vec::with_capacity(PIXELS);
    let mut best = Vec::with_capacity(PIXELS);
    let mut confidence = Vec::with_capacity(PIXELS);
    for (i, (row, logits)) in probs
        .chunks_exact_mut(COLORS)
        .zip(cond.chunks_exact_mut(COLORS))
        .enumerate()
    {
        let peak = argmax(logits);
        best.push(peak as u16);
        softmax_inplace(row);
        let distribution = if opts.temperature > 0.
            && (opts.temperature != 1. || p.top_k > 0 || p.top_p < 1. || p.min_p > 0.)
        {
            let shift = logits[peak];
            for v in logits.iter_mut() {
                *v = (*v - shift) / opts.temperature;
            }
            filter(logits, p);
            &*logits
        } else {
            &*row
        };
        let mut chosen = peak;
        if opts.temperature > 0. {
            let mut high = -1.;
            for (j, (&v, &exp)) in distribution.iter().zip(&noise[i * COLORS..]).enumerate() {
                let val = v / exp;
                if val > high {
                    chosen = j;
                    high = val;
                }
            }
        }
        ids.push(chosen as u16);
        confidence.push(distribution[chosen]);
    }
    Ok(Draw {
        probs,
        ids,
        confidence,
        best,
    })
}

pub(super) fn commit_cpu(
    canvas: &mut [u16],
    commit_conf: &mut [f32],
    draw: &Draw,
    gumbels: &[f32],
    remaining: usize,
    p: &PixelOptions,
) {
    let score: Vec<_> = draw
        .confidence
        .iter()
        .zip(gumbels)
        .map(|(v, g)| v.max(1e-12).ln() + g)
        .collect();
    let masked = canvas.iter().filter(|&&v| v == MASK).count();
    let threshold = if p.commit_threshold > 0. {
        (0..PIXELS)
            .filter(|&i| canvas[i] == MASK && draw.confidence[i] >= p.commit_threshold)
            .count()
    } else {
        0
    };
    let count = masked.saturating_sub(remaining).max(1).max(threshold);
    for i in priorities(canvas, &score, &draw.confidence, p)
        .into_iter()
        .take(count)
    {
        canvas[i] = draw.ids[i];
        commit_conf[i] = draw.confidence[i];
    }
}

pub(super) fn remask_cpu(canvas: &mut [u16], conf: &[f32], count: usize) {
    let mut low: Vec<_> = (0..PIXELS).filter(|&i| conf[i].is_finite()).collect();
    low.sort_by(|&a, &b| conf[a].total_cmp(&conf[b]));
    for i in low.into_iter().take(count) {
        canvas[i] = MASK;
    }
}

pub(super) trait SamplingModel {
    fn remask(&mut self, canvas: &mut [u16], conf: &[f32], count: usize) -> Result<(), String> {
        remask_cpu(canvas, conf, count);
        Ok(())
    }
    fn draw(
        &mut self,
        ids: &[u16],
        states: &[f32],
        pooled: &[f32],
        sc: Option<&[f32]>,
        opts: &DrawOptions<'_>,
        noise: &[f32],
    ) -> Result<Draw, String>;
    fn commit(
        &mut self,
        canvas: &mut [u16],
        commit_conf: &mut [f32],
        draw: &Draw,
        gumbels: &[f32],
        remaining: usize,
        p: &PixelOptions,
    ) -> Result<(), String> {
        commit_cpu(canvas, commit_conf, draw, gumbels, remaining, p);
        Ok(())
    }
}

/// Indexed like the author's SampleStep: decode steps first, then polish steps.
#[derive(Clone, Copy, Debug)]
pub struct Step {
    pub index: usize,
    pub total: usize,
    pub polish: bool,
}

struct Canvas {
    ids: Vec<u16>,
    confidence: Vec<f32>,
    sc: Option<Vec<f32>>,
}

pub(super) fn sample(
    opt: &Options,
    initial: &[u16],
    conditioning: &[(&[f32], &[f32])],
    model: &mut impl SamplingModel,
    mut preview: impl FnMut(Step, usize, &[u16]) -> Result<(), String>,
) -> Result<Vec<Vec<u16>>, String> {
    let p = &opt.pixel;
    let start = initial.iter().filter(|&&v| v == MASK).count();
    let mut canvases: Vec<_> = conditioning
        .iter()
        .map(|_| Canvas {
            ids: initial.to_vec(),
            confidence: vec![f32::INFINITY; PIXELS],
            sc: None,
        })
        .collect();
    let mut rng = TorchMt19937::new(opt.seed);
    let total = opt.steps + p.polish_steps;
    for k in 0..total {
        let polish = k >= opt.steps;
        if !polish && canvases.iter().all(|c| !c.ids.contains(&MASK)) {
            continue;
        }
        let frac = ((k + 1) as f64 / opt.steps as f64).min(1.);
        let remaining = if frac == 1. {
            0
        } else {
            (start as f32 * (frac * std::f64::consts::FRAC_PI_2).cos() as f32).floor() as usize
        };
        let temperature = if polish {
            0.
        } else {
            p.temperature_end.map_or(opt.temperature, |end| {
                opt.temperature + (end - opt.temperature) * k as f32 / (opt.steps - 1).max(1) as f32
            })
        };
        let opts = DrawOptions {
            pixel: p,
            temperature,
            guidance: opt.guidance_at(frac as f32),
        };
        let mut draws = Vec::with_capacity(canvases.len());
        // PyTorch multinomial consumes a double uniform per color, including
        // kept pixels, for the entire batch before drawing spatial Gumbels.
        for (canvas, &(states, pooled)) in canvases.iter_mut().zip(conditioning) {
            if polish {
                let count = (p.polish_frac * PIXELS as f32).round_ties_even().max(1.) as usize;
                model.remask(&mut canvas.ids, &canvas.confidence, count)?;
            }
            let noise = if temperature > 0. {
                (0..PIXELS * COLORS)
                    .map(|_| -(-rng.sample_uniform_f64()).ln_1p() as f32)
                    .collect()
            } else {
                Vec::new()
            };
            draws.push(model.draw(
                &canvas.ids,
                states,
                pooled,
                canvas.sc.as_deref(),
                &opts,
                &noise,
            )?);
        }
        for (i, (canvas, draw)) in canvases.iter_mut().zip(draws).enumerate() {
            let mut gumbels = [0.; PIXELS];
            if !polish && p.choice_temperature > 0. {
                for g in &mut gumbels {
                    let u = rng.sample_uniform_f32().max(1e-10);
                    *g = -p.choice_temperature * (1. - frac as f32) * (-u.ln()).ln();
                }
            }
            model.commit(
                &mut canvas.ids,
                &mut canvas.confidence,
                &draw,
                &gumbels,
                remaining,
                p,
            )?;
            preview(
                Step {
                    index: k,
                    total,
                    polish,
                },
                i,
                &draw.best,
            )?;
            if p.self_cond {
                canvas.sc = Some(draw.probs);
            }
        }
    }
    Ok(canvases.into_iter().map(|c| c.ids).collect())
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
        struct Mock(usize);
        impl SamplingModel for Mock {
            fn draw(
                &mut self,
                ids: &[u16],
                _: &[f32],
                _: &[f32],
                sc: Option<&[f32]>,
                opts: &DrawOptions,
                noise: &[f32],
            ) -> Result<Draw, String> {
                assert_eq!(ids.iter().filter(|&&id| id == MASK).count(), 1);
                assert_eq!(sc.is_some(), self.0 > 0);
                self.0 += 1;
                let mut logits = vec![0.; PIXELS * COLORS];
                logits[77 * COLORS + 13] = 10.;
                draw_cpu(logits, None, opts, noise)
            }
        }
        let mut mock = Mock(0);
        let result = sample(&opt, &canvas, &[(&[], &[])], &mut mock, |_, _, _| Ok(()))
            .unwrap()
            .pop()
            .unwrap();
        assert_eq!(mock.0, 2);
        assert_eq!(result[77], 13);
        assert_eq!(result[31], 15);
        assert_eq!(result.iter().filter(|&&t| t == 0).count(), 254);
    }
}
