//! Astronex-World's fixed eight-step flow-matching UniPC scheduler.

/// Flow-matching UniPC (order 2) used by the pinned Astronex reference.
///
/// This is deliberately kept local to Astronex instead of pretending to be a
/// general diffusion scheduler: the model is trained for the eight-step
/// `predict_x0`/`bh2` contract below.  All tensor arithmetic stays F32, just
/// like the reference scheduler tensors.  The scheduler is not part of the
/// operator parity gate because platform `ln`/`exp_m1` implementations are
/// allowed to differ by a last bit.
pub(super) struct FlowUniPc {
    pub(super) sigmas: Vec<f32>,
    pub(super) timesteps: Vec<u32>,
    step_index: usize,
    model_outputs: [Option<Vec<f32>>; 2],
    lower_order_nums: usize,
    this_order: usize,
    last_sample: Option<Vec<f32>>,
}

impl FlowUniPc {
    pub(super) fn new(steps: usize, shift: f32) -> Result<Self, String> {
        if steps == 0 || !shift.is_finite() || shift <= 0.0 {
            return Err("Astronex UniPC requires positive finite steps and shift".into());
        }
        // The official scheduler constructs this linspace in F64, applies the
        // flow shift, then stores the sigma table as F32. Preserve that order
        // so timestep selection does not move at the boundary.
        let sigma_max = 1.0f64 - 1.0 / 1000.0;
        let mut sigmas = Vec::with_capacity(steps + 1);
        let mut timesteps = Vec::with_capacity(steps);
        for index in 0..steps {
            let base = sigma_max * (1.0 - index as f64 / steps as f64);
            let shifted = shift as f64 * base / (1.0 + (shift as f64 - 1.0) * base);
            sigmas.push(shifted as f32);
            timesteps.push((shifted * 1000.0) as u32);
        }
        sigmas.push(0.0);
        Ok(Self {
            sigmas,
            timesteps,
            step_index: 0,
            model_outputs: std::array::from_fn(|_| None),
            lower_order_nums: 0,
            this_order: 1,
            last_sample: None,
        })
    }

    pub(super) fn step(
        &mut self,
        model_output: &[f32],
        timestep: u32,
        sample: &[f32],
    ) -> Result<Vec<f32>, String> {
        if self.step_index >= self.timesteps.len() {
            return Err("Astronex UniPC was stepped after completion".into());
        }
        if self.timesteps[self.step_index] != timestep {
            return Err(format!(
                "Astronex UniPC timestep mismatch: got {timestep}, expected {}",
                self.timesteps[self.step_index]
            ));
        }
        if model_output.is_empty() || model_output.len() != sample.len() {
            return Err("Astronex UniPC sample/model shape mismatch".into());
        }

        // The official scheduler converts the flow prediction before the
        // UniC corrector.  Passing the raw flow here changes both the history
        // residual and every later latent.
        let converted = self.convert_model_output(model_output, sample)?;
        let mut current = sample.to_vec();
        if self.step_index > 0 && self.last_sample.is_some() {
            current = self.corrector(&converted, &current)?;
        }

        self.model_outputs[0] = self.model_outputs[1].take();
        self.model_outputs[1] = Some(converted);
        let order = 2
            .min(self.timesteps.len() - self.step_index)
            .min(self.lower_order_nums + 1);
        self.this_order = order;
        self.last_sample = Some(current.clone());
        let next = self.predictor(&current, order)?;
        self.lower_order_nums = (self.lower_order_nums + 1).min(2);
        self.step_index += 1;
        if next.iter().any(|value| !value.is_finite()) {
            return Err("Astronex UniPC produced a non-finite latent".into());
        }
        Ok(next)
    }

    fn convert_model_output(
        &self,
        model_output: &[f32],
        sample: &[f32],
    ) -> Result<Vec<f32>, String> {
        let sigma = self.sigmas[self.step_index];
        let mut x0 = Vec::with_capacity(sample.len());
        for (&x, &flow) in sample.iter().zip(model_output) {
            x0.push(x - sigma * flow);
        }
        Ok(x0)
    }

    fn predictor(&self, sample: &[f32], order: usize) -> Result<Vec<f32>, String> {
        let index = self.step_index;
        let m0 = self.model_outputs[1]
            .as_deref()
            .ok_or("Astronex UniPC predictor has no current model output")?;
        let (alpha_t, sigma_t) = alpha_sigma(self.sigmas[index + 1]);
        let (alpha_s0, sigma_s0) = alpha_sigma(self.sigmas[index]);
        let h = lambda(alpha_t, sigma_t) - lambda(alpha_s0, sigma_s0);
        let hh = -h;
        let h_phi_1 = hh.exp_m1();
        let b_h = hh.exp_m1();
        let mut residual = vec![0.0; sample.len()];
        if order == 2 {
            let previous = self.model_outputs[0]
                .as_deref()
                .ok_or("Astronex UniPC order-2 predictor has no history")?;
            let sigma_si = self.sigmas[index - 1];
            let (alpha_si, sigma_si) = alpha_sigma(sigma_si);
            let rk = (lambda(alpha_si, sigma_si) - lambda(alpha_s0, sigma_s0)) / h;
            if !rk.is_finite() || rk == 0.0 {
                return Err("Astronex UniPC predictor has a degenerate history ratio".into());
            }
            for ((value, &older), &current) in residual.iter_mut().zip(previous).zip(m0) {
                *value = 0.5 * (older - current) / rk;
            }
        }
        let mut out = Vec::with_capacity(sample.len());
        for ((&x, &current), &residual) in sample.iter().zip(m0).zip(&residual) {
            let base = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * current;
            out.push(base - alpha_t * b_h * residual);
        }
        Ok(out)
    }

    fn corrector(&self, model_output: &[f32], sample: &[f32]) -> Result<Vec<f32>, String> {
        let index = self.step_index;
        let previous_sample = self
            .last_sample
            .as_deref()
            .ok_or("Astronex UniPC corrector has no previous sample")?;
        let m0 = self.model_outputs[1]
            .as_deref()
            .ok_or("Astronex UniPC corrector has no previous model output")?;
        let (alpha_t, sigma_t) = alpha_sigma(self.sigmas[index]);
        let (alpha_s0, sigma_s0) = alpha_sigma(self.sigmas[index - 1]);
        let h = lambda(alpha_t, sigma_t) - lambda(alpha_s0, sigma_s0);
        let hh = -h;
        let h_phi_1 = hh.exp_m1();
        let b_h = hh.exp_m1();
        let order = self.this_order;
        let mut residuals: Vec<Vec<f32>> = Vec::new();
        let rhos = if order == 1 {
            vec![0.5]
        } else {
            let previous = self.model_outputs[0]
                .as_deref()
                .ok_or("Astronex UniPC order-2 corrector has no history")?;
            let sigma_si = self.sigmas[index - 2];
            let (alpha_si, sigma_si) = alpha_sigma(sigma_si);
            let rk = (lambda(alpha_si, sigma_si) - lambda(alpha_s0, sigma_s0)) / h;
            if !rk.is_finite() || rk == 0.0 {
                return Err("Astronex UniPC corrector has a degenerate history ratio".into());
            }
            let h_phi_k = h_phi_1 / hh - 1.0;
            let h_phi_k2 = h_phi_k / hh - 0.5;
            let b0 = h_phi_k / b_h;
            let b1 = h_phi_k2 * 2.0 / b_h;
            let determinant = 1.0 - rk;
            if determinant == 0.0 || !determinant.is_finite() {
                return Err("Astronex UniPC corrector linear system is singular".into());
            }
            residuals.push(
                previous
                    .iter()
                    .zip(m0)
                    .map(|(&older, &current)| (older - current) / rk)
                    .collect(),
            );
            vec![(b0 - rk * b1) / determinant, (b1 - b0) / determinant]
        };
        let mut out = Vec::with_capacity(sample.len());
        for index in 0..sample.len() {
            let correction = residuals
                .first()
                .map_or(0.0, |values| rhos[0] * values[index])
                + rhos[order - 1] * (model_output[index] - m0[index]);
            let base = sigma_t / sigma_s0 * previous_sample[index] - alpha_t * h_phi_1 * m0[index];
            out.push(base - alpha_t * b_h * correction);
        }
        Ok(out)
    }
}

fn alpha_sigma(sigma: f32) -> (f32, f32) {
    (1.0 - sigma, sigma)
}

fn lambda(alpha: f32, sigma: f32) -> f32 {
    alpha.ln() - sigma.ln()
}
