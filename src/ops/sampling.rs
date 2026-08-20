//! Sampling utilities: argmax + top-K with softmax.

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