//! Sampling utilities aligned with llama.cpp's sampler chain:
//!   repetition_penalty -> top_k -> top_p -> temperature -> dist sample.
//!
//! Repetition penalty divides the logit of any token that has already
//! appeared by `penalty^count` (Hugging Face / llama.cpp definition).
//! `penalty == 1.0` is a no-op; values > 1.0 suppress repeats, < 1.0
//! encourage them. Callers must pass a count map that tracks each
//! generated token.

use crate::ops::vec_scale_f32;

pub fn apply_repetition_penalty(
    logits: &mut [f32],
    token_counts: &std::collections::HashMap<u32, u32>,
    penalty: f32,
) {
    if penalty == 1.0 || token_counts.is_empty() {
        return;
    }
    debug_assert!(penalty > 0.0, "repetition penalty must be positive");
    for (&token, &count) in token_counts {
        if count == 0 {
            continue;
        }
        let idx = token as usize;
        if let Some(l) = logits.get_mut(idx) {
            // Negative logits get multiplied by penalty (closer to 0);
            // positive logits get divided (smaller). Both make the token
            // less likely to win argmax / sampling again.
            let factor = penalty.powi(count as i32);
            if *l > 0.0 {
                *l /= factor;
            } else {
                *l *= factor;
            }
        }
    }
}

pub fn argmax(x: &[f32]) -> usize {
    let mut best_idx = 0;
    let mut best_val = x[0];
    for (i, &v) in x.iter().enumerate().skip(1) {
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    best_idx
}

pub fn sample_top_k(logits: &[f32], k: usize) -> Vec<(usize, f32)> {
    let n = logits.len();
    let keep = k.min(n);
    let mut top: Vec<(usize, f32)> = Vec::with_capacity(keep);
    let mut min_in_top = f32::NEG_INFINITY;
    let mut worst_idx = 0;
    for (i, &v) in logits.iter().enumerate() {
        if top.len() < keep {
            top.push((i, v));
            if top.len() == keep {
                let mut w = 0;
                for j in 1..keep {
                    if top[j].1 < top[w].1 {
                        w = j;
                    }
                }
                worst_idx = w;
                min_in_top = top[w].1;
            }
        } else if v > min_in_top {
            top[worst_idx] = (i, v);
            let mut w = 0;
            for j in 1..keep {
                if top[j].1 < top[w].1 {
                    w = j;
                }
            }
            worst_idx = w;
            min_in_top = top[w].1;
        }
    }
    let max_val = top
        .iter()
        .map(|&(_, v)| v)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for (_, v) in top.iter_mut() {
        *v = (*v - max_val).exp();
        sum += *v;
    }
    if sum > 0.0 {
        for (_, p) in top.iter_mut() {
            *p /= sum;
        }
    }
    top
}

/// Sample a token id using top-K + random draw (matches the reference codec
/// decoder's on-device sampler).
pub fn sample_top_k_draw<R: rand::Rng>(logits: &[f32], k: usize, rng: &mut R) -> usize {
    let candidates = sample_top_k(logits, k);
    let target: f32 = rng.gen();
    let mut cumulative = 0.0f32;
    for &(idx, p) in &candidates {
        cumulative += p;
        if cumulative >= target {
            return idx;
        }
    }
    candidates.last().map(|&(i, _)| i).unwrap_or(0)
}

/// Single-pass sampler that mirrors llama.cpp's default chain:
///   1. Top-K filtering (k = 0 = vocab size, no filter)
///   2. Top-P filtering (p < 1.0 = nucleus; p = 1.0 = disabled)
///   3. Temperature scaling (temp = 0 = argmax, temp = 1 = no scale)
///   4. Stochastic draw over the resulting distribution
///
/// `rng_u64` is the random number used for the final draw; pass
/// `rand::random()` or any other source of entropy.
pub fn sample_llama_cpp(
    logits: &mut [f32],
    top_k: usize,
    top_p: f32,
    temperature: f32,
    rng_u64: u64,
) -> usize {
    // 1. Argmax / temperature = 0 path: pick the highest logit, no RNG needed.
    if temperature <= 0.0 {
        let mut max_i = 0usize;
        let mut max_l = logits[0];
        for i in 1..logits.len() {
            if logits[i] > max_l {
                max_l = logits[i];
                max_i = i;
            }
        }
        return max_i;
    }

    // 2. Temperature scaling. llama.cpp's temp_impl divides logits by temp.
    let inv_temp = 1.0f32 / temperature;
    for l in logits.iter_mut() {
        *l *= inv_temp;
    }

    // 3. Find the max logit (for softmax numerical stability).
    let _max_l = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    // 4. Build a partial sorted vector: first by top_k (if > 0), then by
    //    top_p nucleus. llama.cpp does top_k first, then top_p, on the
    //    already-reduced set. We do both in one pass for simplicity.
    let vocab = logits.len();

    // Apply top-k: keep only top-k logits.
    let mut candidates: Vec<(usize, f32)> = if top_k > 0 && top_k < vocab {
        let mut partial: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
        // Partial sort by logit desc.
        partial.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        partial.truncate(top_k);
        partial
    } else {
        logits.iter().copied().enumerate().collect()
    };

    // Apply top-p (nucleus): softmax then keep smallest set with cumsum >= p.
    if top_p < 1.0 {
        // Compute softmax probabilities.
        let max_l = candidates
            .iter()
            .map(|&(_, l)| l)
            .fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for &mut (_, ref mut p) in candidates.iter_mut() {
            *p = (*p - max_l).exp();
            sum += *p;
        }
        if sum > 0.0 {
            for &mut (_, ref mut p) in candidates.iter_mut() {
                *p /= sum;
            }
        }

        // Sort by probability desc (already softmaxed).
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Cumsum and find last index where cumsum >= top_p (or include at
        // least one token).
        let mut cum = 0.0f32;
        let mut last_idx = candidates.len();
        for i in 0..candidates.len() {
            cum += candidates[i].1;
            if cum >= top_p {
                last_idx = i + 1;
                break;
            }
        }
        candidates.truncate(last_idx);
    }

    // 5. Renormalise the logits (the temperature-scaled logits were
    //    already passed in, but after truncation we need to do softmax).
    let max_l = candidates
        .iter()
        .map(|&(_, l)| l)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for &mut (_, ref mut p) in candidates.iter_mut() {
        *p = (*p - max_l).exp();
        sum += *p;
    }
    if sum > 0.0 {
        for &mut (_, ref mut p) in candidates.iter_mut() {
            *p /= sum;
        }
    }

    // 6. Dist sample: pick uniformly in [0, 1), find smallest i where
    //    cumsum >= target. Matches llama.cpp's dist sampler.
    let target = (rng_u64 as f64 / u64::MAX as f64) as f32;
    let mut cum = 0.0f32;
    let mut chosen = candidates.last().map(|&(i, _)| i).unwrap_or(0);
    for &(idx, p) in &candidates {
        cum += p;
        if cum >= target {
            chosen = idx;
            break;
        }
    }
    chosen
}

/// Canonical greedy / temperature sampler for the text front-ends.
///
/// This is the implementation the qwen3 trunk has always used, moved here so
/// the CLI (qwen3 + qwen35 paths) and every HTTP adapter share one sampler.
/// It validates its input instead of panicking or silently returning 0:
///
/// * empty logits      -> `Err`
/// * non-finite logits -> `Err`
/// * exact logit ties  -> the FIRST index wins (strict `>`), which is what
///   `max_by(partial_cmp)` did *not* guarantee (it keeps the last maximum).
///
/// `temperature <= 0` is greedy; above that it softmaxes with max subtraction
/// and draws against `rand::random()`.
pub fn sample_greedy_or_temperature(logits: &[f32], temperature: f32) -> Result<u32, String> {
    if temperature <= 0.0 {
        return greedy_checked(logits);
    }
    let (&first, rest) = logits
        .split_first()
        .ok_or_else(|| "Cannot sample empty logits".to_string())?;
    if !first.is_finite() {
        return Err("Cannot sample non-finite logits".into());
    }
    let mut max_logit = first;
    for &logit in rest {
        if !logit.is_finite() {
            return Err("Cannot sample non-finite logits".into());
        }
        max_logit = max_logit.max(logit);
    }
    let sum: f32 = logits
        .iter()
        .map(|logit| ((logit - max_logit) / temperature).exp())
        .sum();
    if !sum.is_finite() || sum <= 0.0 {
        return Err("Sampling probability sum is not finite and positive".into());
    }
    let target = rand::random::<f32>() * sum;
    let mut cumulative = 0.0f32;
    for (index, &logit) in logits.iter().enumerate() {
        cumulative += ((logit - max_logit) / temperature).exp();
        if cumulative >= target {
            return u32::try_from(index).map_err(|_| "Token ID does not fit u32".into());
        }
    }
    u32::try_from(logits.len() - 1).map_err(|_| "Token ID does not fit u32".into())
}

/// Greedy argmax with the same validation as [`sample_greedy_or_temperature`].
/// First index wins on exact ties (strict `>`).
pub fn greedy_checked(logits: &[f32]) -> Result<u32, String> {
    let (&first, rest) = logits
        .split_first()
        .ok_or_else(|| "Cannot sample empty logits".to_string())?;
    if !first.is_finite() {
        return Err("Cannot sample non-finite logits".into());
    }
    let mut best_id = 0usize;
    let mut best = first;
    for (index, &logit) in rest.iter().enumerate() {
        if !logit.is_finite() {
            return Err("Cannot sample non-finite logits".into());
        }
        if logit > best {
            best = logit;
            best_id = index + 1;
        }
    }
    u32::try_from(best_id).map_err(|_| "Token ID does not fit u32".into())
}

/// Stateful llama-family sampler: repetition penalty + llama.cpp chain
/// (`top_k -> top_p -> temperature -> dist sample`) with an RNG seeded from
/// the generated history so a prompt is reproducible.
///
/// Extracted verbatim from `llama::trunk::run_inference_tokens` so the CLI
/// and the HTTP `LlamaTextRuntime` adapter cannot drift apart: one
/// implementation, two front-ends.
#[derive(Default)]
pub struct LlamaSampler {
    /// Every token seen so far (prompt + generated), used to seed the RNG.
    all_tokens: Vec<u32>,
    /// Per-token repeat counts, used by `apply_repetition_penalty`.
    token_counts: std::collections::HashMap<u32, u32>,
    top_k: usize,
    top_p: f32,
}

impl LlamaSampler {
    /// `top_k` / `top_p` usually come from the GGUF `general.sampling.*`
    /// metadata (see `llama::trunk::sample_defaults`).
    pub fn new(top_k: usize, top_p: f32) -> Self {
        Self {
            all_tokens: Vec::new(),
            token_counts: std::collections::HashMap::new(),
            top_k,
            top_p,
        }
    }

    /// Seed the history RNG with the prompt tokens before the first call, so
    /// the CLI (which seeds from prompt + generated) and the HTTP adapter
    /// (which seeds identically) agree.
    pub fn prime(&mut self, prompt_tokens: &[u32]) {
        self.all_tokens.extend_from_slice(prompt_tokens);
    }

    /// Sample the next token id from `logits`.
    pub fn sample(&mut self, logits: &mut [f32], temperature: f32, repetition_penalty: f32) -> u32 {
        apply_repetition_penalty(logits, &self.token_counts, repetition_penalty);
        let rng_u64 = if temperature <= 0.0 {
            0
        } else {
            // Deterministic per prompt: seed from the full token history.
            let mut rng = 0u64.wrapping_add(0x9E3779B97F4A7C15);
            for &t in &self.all_tokens {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(t as u64);
            }
            rng
        };
        let chosen = sample_llama_cpp(logits, self.top_k, self.top_p, temperature, rng_u64) as u32;
        *self.token_counts.entry(chosen).or_insert(0) += 1;
        self.all_tokens.push(chosen);
        chosen
    }
}

/// Stateful lfm2moe-family sampler: repetition penalty, then greedy (when
/// `temperature <= 0`) or top-40 + temperature-scaled softmax draw with an
/// RNG seeded from the generated history.
///
/// Extracted verbatim from `lfm2moe::run_inference_with_batch` so the CLI and
/// the HTTP `Lfm2MoeTextRuntime` adapter share one implementation (they used
/// to diverge: the server used temperature-only sampling).
#[derive(Default)]
pub struct Lfm2MoeSampler {
    all_tokens: Vec<u32>,
    token_counts: std::collections::HashMap<u32, u32>,
}

impl Lfm2MoeSampler {
    pub fn new() -> Self {
        Self::default()
    }

    /// Seed the history RNG with the prompt tokens before the first call.
    pub fn prime(&mut self, prompt_tokens: &[u32]) {
        self.all_tokens.extend_from_slice(prompt_tokens);
    }

    /// Sample the next token id from `logits` (mutated in place).
    pub fn sample(&mut self, logits: &mut [f32], temperature: f32, repetition_penalty: f32) -> u32 {
        apply_repetition_penalty(logits, &self.token_counts, repetition_penalty);
        let chosen = if temperature <= 0.0 {
            logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0)
        } else {
            vec_scale_f32(logits, 1.0 / temperature);
            let top = sample_top_k(logits, 40);
            let mut rng = 0u64;
            for &t in &self.all_tokens {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(t as u64);
            }
            let r = ((rng >> 33) as f32) / (1u32 << 31) as f32;
            let mut cum = 0.0f32;
            let mut chosen = top[0].0;
            for &(idx, prob) in &top {
                cum += prob;
                if cum >= r {
                    chosen = idx;
                    break;
                }
            }
            chosen
        };
        *self.token_counts.entry(chosen as u32).or_insert(0) += 1;
        self.all_tokens.push(chosen as u32);
        chosen as u32
    }
}

#[cfg(test)]
mod sampler_unification_tests {
    use super::{greedy_checked, sample_greedy_or_temperature};

    #[test]
    fn greedy_picks_first_index_on_exact_ties() {
        // The old server-derived sampler used `max_by(partial_cmp)`, which keeps
        // the LAST maximum; the canonical one keeps the FIRST. Pin it.
        let logits = [1.0f32, 5.0, 5.0, 0.0];
        assert_eq!(greedy_checked(&logits).unwrap(), 1);
    }

    #[test]
    fn greedy_rejects_empty_and_non_finite() {
        assert!(greedy_checked(&[]).is_err(), "empty must be an error, not 0");
        assert!(greedy_checked(&[1.0, f32::NAN]).is_err(), "NaN must error, not panic");
        assert!(sample_greedy_or_temperature(&[1.0, f32::INFINITY], 0.0).is_err());
    }

    #[test]
    fn temperature_zero_is_greedy_everywhere() {
        let logits = [0.25f32, 9.0, 0.5];
        assert_eq!(sample_greedy_or_temperature(&logits, 0.0).unwrap(), 1);
        assert_eq!(sample_greedy_or_temperature(&logits, -1.0).unwrap(), 1);
    }

    #[test]
    fn temperature_negative_is_treated_as_greedy() {
        // The old copies disagreed here: one used `== 0.0`, another `<= 0.0`.
        let logits = [0.25f32, 9.0, 0.5];
        assert_eq!(sample_greedy_or_temperature(&logits, -0.5).unwrap(), 1);
    }

    #[test]
    fn temperature_draw_stays_in_range() {
        // Smoke: the draw must return a valid index for a normal distribution
        // (the exact token depends on RNG, so only bounds are asserted).
        let logits: Vec<f32> = (0..64).map(|i| (i as f32) * 0.1).collect();
        for _ in 0..32 {
            let id = sample_greedy_or_temperature(&logits, 0.8).unwrap();
            assert!(id < 64, "id {id} out of range");
        }
    }
}
