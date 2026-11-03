use super::network::clamp_infinities;
use super::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum Series {
    Univariate(Vec<Option<f32>>),
    Multivariate(Vec<Vec<Option<f32>>>),
}
impl Series {
    fn rows(&self) -> Vec<Vec<Option<f32>>> {
        match self {
            Self::Univariate(row) => vec![row.clone()],
            Self::Multivariate(rows) => rows.clone(),
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ForecastRequest {
    pub context: Series,
    pub prediction_length: usize,
    #[serde(default)]
    pub observation_mask: Option<Vec<Vec<bool>>>,
    #[serde(default)]
    pub past_covariates: Vec<Vec<Option<f32>>>,
    #[serde(default)]
    pub known_covariates: Vec<Vec<Option<f32>>>,
    #[serde(default = "yes")]
    pub symmetric_averaging: bool,
    #[serde(default = "yes")]
    pub clip_nonnegative: bool,
    #[serde(default = "max_context")]
    pub max_context: usize,
    #[serde(default = "block_size")]
    pub per_shot_horizon_patches: usize,
}
fn yes() -> bool {
    true
}
fn max_context() -> usize {
    32768
}
fn block_size() -> usize {
    32
}

#[derive(Debug, Serialize)]
pub struct Forecast {
    pub quantile_levels: Vec<f32>,
    /// [target variate, horizon step, quantile].
    pub quantiles: Vec<Vec<Vec<f32>>>,
}

#[derive(Clone)]
pub(super) struct Row {
    pub values: Vec<f32>,
    pub observed: Vec<bool>,
    pub role: u8,
}
struct Prepared {
    rows: Vec<Row>,
    future: Vec<Vec<Option<f32>>>,
    targets: usize,
    nonnegative: Vec<bool>,
}

impl ForecastRequest {
    fn prepare(&self, negate: bool) -> Result<Prepared, String> {
        let targets = self.context.rows();
        let t = targets.first().map_or(0, Vec::len);
        if targets.is_empty() || t == 0 || self.prediction_length == 0 {
            return Err(
                "Timestral requires non-empty context and a positive prediction_length".into(),
            );
        }
        if !(1..=32768).contains(&self.max_context)
            || !(1..=32).contains(&self.per_shot_horizon_patches)
        {
            return Err(
                "max_context must be in 1..=32768; per_shot_horizon_patches in 1..=32".into(),
            );
        }
        self.prediction_length
            .checked_add(PS - 1)
            .and_then(|h| h.checked_mul(targets.len()))
            .and_then(|h| h.checked_mul(Q * 4))
            .filter(|n| *n <= isize::MAX as usize)
            .ok_or("Timestral forecast dimensions overflow")?;
        if let Some(mask) = &self.observation_mask {
            if mask.len() != targets.len() || mask.iter().any(|row| row.len() != t) {
                return Err("observation_mask must match [target variates, context length]".into());
            }
        }
        let known_len = t
            .checked_add(self.prediction_length)
            .ok_or("Known covariate length overflow")?;
        let mut rows = Vec::new();
        let mut future = Vec::new();
        let mut nonnegative = Vec::new();
        let start = t.saturating_sub(self.max_context);
        let pad = (PS - (t - start) % PS) % PS;
        for (role, series) in [
            (0, &targets),
            (1, &self.past_covariates),
            (2, &self.known_covariates),
        ] {
            for (row_index, series) in series.iter().enumerate() {
                if series.len() != if role == 2 { known_len } else { t } {
                    return Err(format!("Invalid length for Timestral row role {role}"));
                }
                if series.iter().flatten().any(|v| !v.is_finite()) {
                    return Err("Use null for missing values; numeric inputs must be finite".into());
                }
                let mut values = vec![0.; pad];
                let mut observed = vec![false; pad];
                let mut previous = 0.;
                let mut any_observed = false;
                let mut all_nonnegative = true;
                for (j, v) in series[..t].iter().enumerate() {
                    let valid = v.is_some()
                        && (role != 0
                            || self
                                .observation_mask
                                .as_ref()
                                .is_none_or(|m| m[row_index][j]));
                    let value = if valid {
                        v.unwrap()
                    } else if role == 0 {
                        previous
                    } else {
                        0.
                    };
                    if valid {
                        previous = value;
                        any_observed = true;
                        all_nonnegative &= value >= 0.;
                    }
                    if j >= start {
                        values.push(if negate { -value } else { value });
                        observed.push(valid);
                    }
                }
                rows.push(Row {
                    values,
                    observed,
                    role,
                });
                if role == 0 {
                    nonnegative.push(any_observed && all_nonnegative);
                }
                future.push(if role == 2 {
                    series[t..]
                        .iter()
                        .map(|v| v.map(|v| if negate { -v } else { v }))
                        .collect()
                } else {
                    Vec::new()
                });
            }
        }
        Ok(Prepared {
            rows,
            future,
            targets: targets.len(),
            nonnegative,
        })
    }
}

impl TimestralModel<'_> {
    pub fn forecast(&self, request: &ForecastRequest) -> Result<Forecast, String> {
        let prepared = request.prepare(false)?;
        let mut values = self.forecast_raw(request, &prepared)?;
        if request.symmetric_averaging {
            let mirrored = self.forecast_raw(request, &request.prepare(true)?)?;
            for (row, reflected) in values.iter_mut().zip(mirrored) {
                for (q, other) in row.chunks_exact_mut(Q).zip(reflected.chunks_exact(Q)) {
                    for i in 0..Q {
                        q[i] = 0.5 * (q[i] - other[Q - 1 - i]);
                    }
                }
            }
        }
        if request.clip_nonnegative {
            for (out, eligible) in values.iter_mut().zip(&prepared.nonnegative) {
                if *eligible {
                    for v in out {
                        *v = v.max(0.);
                    }
                }
            }
        }
        if values.iter().flatten().any(|v| !v.is_finite()) {
            return Err("Timestral produced a non-finite forecast; input is degenerate".into());
        }
        trace(
            "forecast",
            &[prepared.targets, request.prediction_length, Q],
            &values.concat(),
        )?;
        Ok(Forecast {
            quantile_levels: LEVELS.to_vec(),
            quantiles: values
                .into_iter()
                .map(|r| r.chunks_exact(Q).map(|q| q.to_vec()).collect())
                .collect(),
        })
    }

    fn forecast_raw(
        &self,
        request: &ForecastRequest,
        prepared: &Prepared,
    ) -> Result<Vec<Vec<f32>>, String> {
        let mut rows = prepared.rows.clone();
        let mut result = vec![Vec::new(); prepared.targets];
        let patches = request.prediction_length.div_ceil(PS);
        let mut offset = 0;
        while offset < patches {
            let h = request.per_shot_horizon_patches.min(patches - offset);
            let mut preds = self.predict_block(&rows, &prepared.future, offset * PS, h)?;
            for (row, pred) in rows.iter().zip(&mut preds) {
                refine(row, pred, h);
            }
            trace("refined", &[rows.len(), h * PS, Q], &preds.concat())?;
            for (out, pred) in result.iter_mut().zip(&preds) {
                out.extend_from_slice(pred);
            }
            offset += h;
            if offset < patches {
                for (i, (row, pred)) in rows.iter_mut().zip(&preds).enumerate() {
                    for t in 0..h * PS {
                        let (v, obs) = match row.role {
                            0 => {
                                let v = pred[t * Q + MEDIAN];
                                (
                                    if v.is_finite() && v.abs() < 1e30 {
                                        v
                                    } else {
                                        0.
                                    },
                                    true,
                                )
                            }
                            2 => {
                                let v = prepared.future[i]
                                    .get((offset - h) * PS + t)
                                    .copied()
                                    .flatten();
                                (v.unwrap_or(0.), v.is_some())
                            }
                            _ => (0., false),
                        };
                        row.values.push(v);
                        row.observed.push(obs);
                    }
                    if row.values.len() > request.max_context {
                        let keep = (request.max_context / PS * PS).max(PS);
                        let drop = row.values.len() - keep;
                        row.values.drain(..drop);
                        row.observed.drain(..drop);
                    }
                }
            }
        }
        for (out, row) in result.iter_mut().zip(&prepared.rows) {
            out.truncate(request.prediction_length * Q);
            let (_, _, scale) = scale_row(&row.values, &row.observed, false);
            let cap = scale[scale.len() - 1] * 1e4;
            let mut lo = f32::INFINITY;
            let mut hi = f32::NEG_INFINITY;
            for (v, obs) in row.values.iter().zip(&row.observed) {
                if *obs {
                    lo = lo.min(*v);
                    hi = hi.max(*v);
                }
            }
            if !lo.is_finite() {
                lo = 0.;
                hi = 0.;
            }
            for v in out {
                *v = v.clamp(lo - cap, hi + cap);
            }
        }
        Ok(result)
    }
}

pub(super) fn scale_row(
    values: &[f32],
    observed: &[bool],
    known: bool,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut loc = vec![0.; values.len()];
    let mut scale = vec![1.; values.len()];
    let mut count = 0f64;
    let mut sum = 0f64;
    let mut m2 = 0f64;
    let mut previous_mean = 0.;
    for p in 0..values.len() / PS {
        for i in p * PS..(p + 1) * PS {
            let x = f64::from(values[i]);
            let weight = if observed[i] { 1. } else { 0. };
            count += weight;
            sum += x * weight;
            let mean = sum / count.max(1.);
            m2 += (x - previous_mean) * (x - mean) * weight;
            previous_mean = mean;
        }
        loc[p * PS..(p + 1) * PS].fill(previous_mean as f32);
        scale[p * PS..(p + 1) * PS].fill((m2 / (count - 1.).max(1.)).sqrt().max(1e-6) as f32);
    }
    let binary = known
        && observed.iter().any(|b| *b)
        && values
            .iter()
            .zip(observed)
            .all(|(v, b)| !*b || *v == 0. || *v == 1.);
    if known {
        let n = observed.iter().filter(|b| **b).count().max(1) as f32;
        let mean = values
            .iter()
            .zip(observed)
            .filter(|(_, b)| **b)
            .map(|(v, _)| *v)
            .sum::<f32>()
            / n;
        let var = values
            .iter()
            .zip(observed)
            .filter(|(_, b)| **b)
            .map(|(v, _)| (*v - mean).powi(2))
            .sum::<f32>()
            / n;
        loc.fill(if binary { 0. } else { mean });
        scale.fill(if binary { 1. } else { var.sqrt().max(1e-6) });
    }
    let scaled = values
        .iter()
        .enumerate()
        .map(|(i, v)| {
            if !observed[i] {
                0.
            } else {
                let v = (*v - loc[i]) / scale[i].max(f32::EPSILON);
                if binary {
                    v
                } else {
                    v.asinh()
                }
            }
        })
        .collect();
    (scaled, loc, scale)
}

fn refine(row: &Row, quantiles: &mut [f32], h: usize) {
    if h <= 1 || row.role != 0 {
        return;
    }
    let (_, loc, scale) = scale_row(&row.values, &row.observed, false);
    let loc0 = loc[loc.len() - 1];
    let scale0 = scale[scale.len() - 1];
    let mut count = 0f64;
    let mut mean = 0f64;
    let mut m2 = 0f64;
    for (v, b) in row.values.iter().zip(&row.observed) {
        if *b {
            count += 1.;
            let delta = f64::from(*v) - mean;
            mean += delta / count;
            m2 += delta * (f64::from(*v) - mean);
        }
    }
    let origin = mean;
    let clip = f64::from(f32::MAX).sqrt() / 1e3;
    for p in 0..h {
        let frame_scale = (m2 / (count - 1.).max(1.)).max(0.).sqrt().max(1e-6);
        let block = &mut quantiles[p * PS * Q..(p + 1) * PS * Q];
        let folded = block
            .chunks_exact(Q)
            .map(|q| {
                let norm = (f64::from(q[MEDIAN]) - f64::from(loc0)) / f64::from(scale0);
                (norm * frame_scale + mean - origin).clamp(-clip, clip) + origin
            })
            .collect::<Vec<_>>();
        let b_mean = folded.iter().sum::<f64>() / PS as f64;
        let b_m2 = folded.iter().map(|v| (*v - b_mean).powi(2)).sum::<f64>();
        let (j_loc, j_scale) = if p == 0 {
            (loc0, scale0)
        } else {
            (mean as f32, frame_scale as f32)
        };
        for q in block.chunks_exact_mut(Q) {
            for v in q.iter_mut() {
                *v = (*v - loc0) / scale0 * j_scale + j_loc;
            }
            clamp_infinities(q);
        }
        let delta = b_mean - mean;
        let total = count + PS as f64;
        m2 += b_m2 + delta * delta * count * PS as f64 / total;
        mean += delta * PS as f64 / total;
        count = total;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn request(value: serde_json::Value) -> ForecastRequest {
        serde_json::from_value(value).unwrap()
    }
    #[test]
    fn preparation_preserves_masks_and_ffills_before_context_truncation() {
        let r = request(
            serde_json::json!({"context":[2,null,4,null],"prediction_length":1,"max_context":2}),
        );
        let p = r.prepare(false).unwrap();
        assert_eq!(&p.rows[0].values[30..], &[4., 4.]);
        assert_eq!(&p.rows[0].observed[30..], &[true, false]);
        assert!(p.rows[0].values[..30].iter().all(|v| *v == 0.));
        let r = request(
            serde_json::json!({"context":[2,null,null],"prediction_length":1,"max_context":1}),
        );
        let p = r.prepare(false).unwrap();
        assert_eq!(p.rows[0].values[31], 2.);
        let r = request(
            serde_json::json!({"context":[-2,2,null],"prediction_length":1,"max_context":1}),
        );
        assert!(!r.prepare(false).unwrap().nonnegative[0]);
    }
    #[test]
    fn invalid_requests_fail_before_allocating_forecasts() {
        for value in [
            serde_json::json!({"context":[],"prediction_length":1}),
            serde_json::json!({"context":[1],"prediction_length":0}),
            serde_json::json!({"context":[[1],[1,2]],"prediction_length":1}),
            serde_json::json!({"context":[1],"prediction_length":1,"known_covariates":[[1]]}),
            serde_json::json!({"context":[1],"prediction_length":1,"max_context":0}),
            serde_json::json!({"context":[1],"prediction_length":1,"per_shot_horizon_patches":33}),
        ] {
            assert!(request(value).prepare(false).is_err());
        }
        assert!(serde_json::from_value::<ForecastRequest>(serde_json::json!({"context":[1],"prediction_length":1,"categorical_past_covariates":[[1]]})).is_err());
    }
    #[test]
    fn causal_scaling_does_not_read_future_and_binary_covariates_skip_asinh() {
        let mut x = (0..64).map(|i| i as f32).collect::<Vec<_>>();
        let mask = vec![true; 64];
        let (a, loc, scale) = scale_row(&x, &mask, false);
        x[32..].fill(1e6);
        let (b, _, _) = scale_row(&x, &mask, false);
        assert_eq!(a[..32], b[..32]);
        assert_eq!(loc[0], 15.5);
        assert!((scale[0] - (88f32).sqrt()).abs() < 1e-5);
        let x = (0..32).map(|i| (i % 2) as f32).collect::<Vec<_>>();
        let (scaled, loc, scale) = scale_row(&x, &mask[..32], true);
        assert_eq!(scaled, x);
        assert_eq!(loc, vec![0.; 32]);
        assert_eq!(scale, vec![1.; 32]);
    }
}
