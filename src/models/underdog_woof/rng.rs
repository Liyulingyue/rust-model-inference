//! The source runtime draws 32 MT19937 uniforms per frame before Box–Muller.
pub(super) struct Noise {
    state: [u32; 624],
    index: usize,
}

impl Noise {
    pub(super) fn new(seed: u32) -> Self {
        let mut state = [0; 624];
        state[0] = seed;
        for i in 1..624 {
            state[i] = 1_812_433_253u32
                .wrapping_mul(state[i - 1] ^ (state[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        Self { state, index: 624 }
    }

    fn next(&mut self) -> u32 {
        if self.index == 624 {
            for i in 0..624 {
                let y = (self.state[i] & 0x8000_0000) | (self.state[(i + 1) % 624] & 0x7fff_ffff);
                self.state[i] = self.state[(i + 397) % 624]
                    ^ (y >> 1)
                    ^ if y & 1 == 0 { 0 } else { 0x9908_b0df };
            }
            self.index = 0;
        }
        let mut y = self.state[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }

    pub(super) fn frame(&mut self) -> [f32; 32] {
        let uniforms: [f64; 32] =
            std::array::from_fn(|_| (self.next() & 0xff_ffff) as f64 / 16_777_216.0);
        let mut out = [0.0; 32];
        for base in [0, 16] {
            for i in 0..8 {
                let radius = (-2.0 * (1.0 - uniforms[base + i]).ln()).sqrt();
                let (sin, cos) = (std::f64::consts::TAU * uniforms[base + i + 8]).sin_cos();
                out[base + i] = (radius * cos) as f32;
                out[base + i + 8] = (radius * sin) as f32;
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pinned_numpy_noise_and_request_local_reset() {
        let expected = [
            -1.1258398294448853,
            -1.152360200881958,
            -0.2505785822868347,
            -0.4338788390159607,
            0.848710298538208,
            0.6920092701911926,
            -0.31601276993751526,
            -2.1152195930480957,
        ];
        let mut noise = Noise::new(0);
        let first = noise.frame();
        for (&actual, expected) in first.iter().zip(expected) {
            assert!((f64::from(actual) - expected).abs() < 1e-7);
        }
        assert_ne!(first, noise.frame());
        assert_eq!(first, Noise::new(0).frame());
        assert_ne!(first, Noise::new(1).frame());
    }
}
