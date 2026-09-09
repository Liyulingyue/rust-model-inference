// Torch CPU LayerNorm uses four Welford lanes and cascades 16-vector chunks.
// Keep these F32 associations separate from the shared F64 norm reductions.
#[derive(Clone, Copy, Default)]
struct Moments {
    count: usize,
    mean: [f32; 4],
    variance: [f32; 4],
}

impl Moments {
    fn add(&mut self, other: Self) {
        let count = self.count + other.count;
        let factor = if count == 0 {
            0.
        } else {
            other.count as f32 / count as f32
        };
        for lane in 0..4 {
            let delta = other.mean[lane] - self.mean[lane];
            self.mean[lane] += factor * delta;
            self.variance[lane] +=
                other.variance[lane] + delta * delta * factor * self.count as f32;
        }
        self.count = count;
    }
}

pub(super) fn row_moments(input: &[f32]) -> (f32, f32) {
    debug_assert!(!input.is_empty() && input.len() % 4 == 0);
    let chunks = input.len().div_ceil(64);
    let depth = chunks.next_power_of_two().trailing_zeros() as usize;
    let mut stack = vec![Moments::default(); depth.max(1)];
    for (index, chunk) in input.chunks(64).enumerate() {
        let mut moments = Moments::default();
        for (j, row) in chunk.chunks_exact(4).enumerate() {
            let factor = 1. / (j + 1) as f32;
            for lane in 0..4 {
                let delta = row[lane] - moments.mean[lane];
                moments.mean[lane] += delta * factor;
                moments.variance[lane] += delta * (row[lane] - moments.mean[lane]);
            }
            moments.count += 1;
        }
        stack[0].add(moments);
        let mut mask = index + 1;
        for level in 1..depth {
            if mask & 1 != 0 {
                break;
            }
            let previous = stack[level - 1];
            stack[level].add(previous);
            stack[level - 1] = Moments::default();
            mask >>= 1;
        }
    }
    for level in 1..depth {
        let other = stack[level];
        stack[0].add(other);
    }
    let mut count = 0;
    let mut mean = 0.;
    let mut variance = 0.;
    let lane_count = input.len() / 4;
    for lane in 0..4 {
        let total = count + lane_count;
        let factor = lane_count as f32 / total as f32;
        let delta = stack[0].mean[lane] - mean;
        mean = factor.mul_add(delta, mean);
        variance += (delta * delta * factor).mul_add(count as f32, stack[0].variance[lane]);
        count = total;
    }
    (mean, variance / input.len() as f32)
}
