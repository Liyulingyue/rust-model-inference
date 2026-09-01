//! Sampling utilities aligned with llama.cpp's sampler chain:
//!   top_k -> top_p -> temperature -> dist sample.

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
    let max_l = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);

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
