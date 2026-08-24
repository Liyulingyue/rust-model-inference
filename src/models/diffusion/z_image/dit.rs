const MT_N: usize = 624;
const MT_M: usize = 397;
const MT_MATRIX_A: u32 = 0x9908_b0df;
const MT_UPPER_MASK: u32 = 0x8000_0000;
const MT_LOWER_MASK: u32 = 0x7fff_ffff;
const PATCH_SIZE: usize = 2;
const ROPE_THETA: f32 = 256.0;
const ROPE_AXES: [usize; 3] = [32, 48, 48];
const ROPE_HEAD_WIDTH: usize = 128;
const SEQUENCE_MULTIPLE: usize = 32;

pub(crate) struct TorchMt19937 {
    state: [u32; MT_N],
    left: usize,
    next: usize,
    has_next_gauss: bool,
    next_gauss: f64,
}

impl TorchMt19937 {
    pub(crate) fn new(seed: u64) -> Self {
        let mut state = [0; MT_N];
        state[0] = seed as u32;
        for index in 1..MT_N {
            let previous = state[index - 1];
            state[index] = 1_812_433_253u32
                .wrapping_mul(previous ^ (previous >> 30))
                .wrapping_add(index as u32);
        }
        Self {
            state,
            left: 1,
            next: 0,
            has_next_gauss: false,
            next_gauss: 0.0,
        }
    }

    fn twist(u: u32, v: u32) -> u32 {
        (((u & MT_UPPER_MASK) | (v & MT_LOWER_MASK)) >> 1)
            ^ if v & 1 == 0 { 0 } else { MT_MATRIX_A }
    }

    fn next_state(&mut self) {
        self.left = MT_N;
        self.next = 0;
        for index in 0..MT_N - MT_M {
            self.state[index] =
                self.state[index + MT_M] ^ Self::twist(self.state[index], self.state[index + 1]);
        }
        for index in MT_N - MT_M..MT_N - 1 {
            self.state[index] = self.state[index + MT_M - MT_N]
                ^ Self::twist(self.state[index], self.state[index + 1]);
        }
        self.state[MT_N - 1] =
            self.state[MT_M - 1] ^ Self::twist(self.state[MT_N - 1], self.state[0]);
    }

    fn rand_u32(&mut self) -> u32 {
        self.left -= 1;
        if self.left == 0 {
            self.next_state();
        }
        let mut value = self.state[self.next];
        self.next += 1;
        value ^= value >> 11;
        value ^= (value << 7) & 0x9d2c_5680;
        value ^= (value << 15) & 0xefc6_0000;
        value ^ (value >> 18)
    }

    fn rand_u64(&mut self) -> u64 {
        (u64::from(self.rand_u32()) << 32) | u64::from(self.rand_u32())
    }

    fn uniform_f32(value: u32) -> f32 {
        (value & 0x00ff_ffff) as f32 * (1.0f32 / (1u32 << 24) as f32)
    }

    fn uniform_f64(value: u64) -> f64 {
        (value & ((1u64 << 53) - 1)) as f64 * (1.0f64 / (1u64 << 53) as f64)
    }

    fn normal_double(&mut self) -> f64 {
        if self.has_next_gauss {
            self.has_next_gauss = false;
            return self.next_gauss;
        }
        let u1 = Self::uniform_f64(self.rand_u64());
        let u2 = Self::uniform_f64(self.rand_u64());
        let radius = (-2.0 * (-u2).ln_1p()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u1;
        self.next_gauss = radius * theta.sin();
        self.has_next_gauss = true;
        radius * theta.cos()
    }

    fn normal_fill_16(values: &mut [f32]) {
        debug_assert_eq!(values.len(), 16);
        for index in 0..8 {
            let u1 = 1.0 - values[index];
            let u2 = values[index + 8];
            let radius = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f32::consts::PI * u2;
            values[index] = radius * theta.cos();
            values[index + 8] = radius * theta.sin();
        }
    }

    pub(crate) fn fill_normal(&mut self, output: &mut [f32]) {
        if output.len() < 16 {
            for value in output {
                *value = self.normal_double() as f32;
            }
            return;
        }

        for value in output.iter_mut() {
            *value = Self::uniform_f32(self.rand_u32());
        }
        for start in (0..output.len() - 15).step_by(16) {
            Self::normal_fill_16(&mut output[start..start + 16]);
        }
        if output.len() % 16 != 0 {
            let tail = output.len() - 16;
            for value in &mut output[tail..] {
                *value = Self::uniform_f32(self.rand_u32());
            }
            Self::normal_fill_16(&mut output[tail..]);
        }
    }
}

pub(crate) fn time_snr_shift(alpha: f32, t: f32) -> f32 {
    if alpha == 1.0 {
        t
    } else {
        alpha * t / (1.0 + (alpha - 1.0) * t)
    }
}

pub(crate) fn z_image_sigmas(steps: usize) -> Result<Vec<f32>, String> {
    if steps == 0 {
        return Err("Z-Image steps must be positive".into());
    }
    if steps == 1 {
        return Ok(vec![time_snr_shift(3.0, 1.0), 0.0]);
    }
    let stride = 999.0 / (steps - 1) as f32;
    let mut result = (0..steps)
        .map(|index| time_snr_shift(3.0, (1000.0 - stride * index as f32) / 1000.0))
        .collect::<Vec<_>>();
    result.push(0.0);
    Ok(result)
}

fn zeroed_f32(name: &str, len: usize) -> Result<Vec<f32>, String> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|error| format!("Failed to allocate {name}: {error}"))?;
    values.resize(len, 0.0);
    Ok(values)
}

fn patch_shape(
    channels: usize,
    height: usize,
    width: usize,
) -> Result<(usize, usize, usize), String> {
    if channels == 0 || height == 0 || width == 0 {
        return Err("Z-Image latent dimensions must be positive".into());
    }
    let patch_height = height
        .checked_add(PATCH_SIZE - 1)
        .ok_or("Z-Image latent shape overflow")?
        / PATCH_SIZE;
    let patch_width = width
        .checked_add(PATCH_SIZE - 1)
        .ok_or("Z-Image latent shape overflow")?
        / PATCH_SIZE;
    let patch_width_channels = channels
        .checked_mul(PATCH_SIZE * PATCH_SIZE)
        .ok_or("Z-Image patch width overflow")?;
    Ok((patch_height, patch_width, patch_width_channels))
}

pub(crate) fn patchify_latent(
    latent: &[f32],
    channels: usize,
    height: usize,
    width: usize,
) -> Result<Vec<f32>, String> {
    let expected = channels
        .checked_mul(height)
        .and_then(|value| value.checked_mul(width))
        .ok_or("Z-Image latent shape overflow")?;
    if latent.len() != expected {
        return Err(format!(
            "Invalid Z-Image latent length: expected {expected}, got {}",
            latent.len()
        ));
    }
    let (patch_height, patch_width, patch_width_channels) = patch_shape(channels, height, width)?;
    let output_len = patch_height
        .checked_mul(patch_width)
        .and_then(|value| value.checked_mul(patch_width_channels))
        .ok_or("Z-Image patch shape overflow")?;
    let mut output = zeroed_f32("Z-Image patches", output_len)?;

    for patch_y in 0..patch_height {
        for patch_x in 0..patch_width {
            let token = patch_y * patch_width + patch_x;
            for inner_y in 0..PATCH_SIZE {
                let y = patch_y * PATCH_SIZE + inner_y;
                if y >= height {
                    continue;
                }
                for inner_x in 0..PATCH_SIZE {
                    let x = patch_x * PATCH_SIZE + inner_x;
                    if x >= width {
                        continue;
                    }
                    let patch_offset = (inner_y * PATCH_SIZE + inner_x) * channels;
                    for channel in 0..channels {
                        let source = (channel * height + y) * width + x;
                        output[token * patch_width_channels + patch_offset + channel] =
                            latent[source];
                    }
                }
            }
        }
    }
    Ok(output)
}

pub(crate) fn unpatchify_latent(
    patches: &[f32],
    channels: usize,
    height: usize,
    width: usize,
) -> Result<Vec<f32>, String> {
    let (patch_height, patch_width, patch_width_channels) = patch_shape(channels, height, width)?;
    let expected = patch_height
        .checked_mul(patch_width)
        .and_then(|value| value.checked_mul(patch_width_channels))
        .ok_or("Z-Image patch shape overflow")?;
    if patches.len() != expected {
        return Err(format!(
            "Invalid Z-Image patch length: expected {expected}, got {}",
            patches.len()
        ));
    }
    let output_len = channels
        .checked_mul(height)
        .and_then(|value| value.checked_mul(width))
        .ok_or("Z-Image latent shape overflow")?;
    let mut output = zeroed_f32("Z-Image latent", output_len)?;

    for channel in 0..channels {
        for y in 0..height {
            for x in 0..width {
                let token = (y / PATCH_SIZE) * patch_width + x / PATCH_SIZE;
                let patch_offset = ((y % PATCH_SIZE) * PATCH_SIZE + x % PATCH_SIZE) * channels;
                output[(channel * height + y) * width + x] =
                    patches[token * patch_width_channels + patch_offset + channel];
            }
        }
    }
    Ok(output)
}

fn padded_to_sequence_multiple(value: usize) -> Result<usize, String> {
    value
        .checked_add(SEQUENCE_MULTIPLE - 1)
        .map(|value| value / SEQUENCE_MULTIPLE * SEQUENCE_MULTIPLE)
        .ok_or_else(|| "Z-Image sequence length overflow".into())
}

pub(crate) fn z_image_rope(
    text_tokens: usize,
    image_width: usize,
    image_height: usize,
) -> Result<Vec<f32>, String> {
    if text_tokens == 0 || image_width == 0 || image_height == 0 {
        return Err("Z-Image RoPE dimensions must be positive".into());
    }
    let axes_sum = ROPE_AXES.iter().try_fold(0usize, |sum, axis| {
        if axis % 2 != 0 {
            return Err("Z-Image RoPE axes must be even".to_string());
        }
        sum.checked_add(*axis)
            .ok_or_else(|| "Z-Image RoPE head width overflow".into())
    })?;
    if axes_sum != ROPE_HEAD_WIDTH {
        return Err("Z-Image RoPE axes must match the attention head width".into());
    }

    let patch_width = image_width
        .checked_add(PATCH_SIZE - 1)
        .ok_or("Z-Image image shape overflow")?
        / PATCH_SIZE;
    let patch_height = image_height
        .checked_add(PATCH_SIZE - 1)
        .ok_or("Z-Image image shape overflow")?
        / PATCH_SIZE;
    let image_tokens = patch_width
        .checked_mul(patch_height)
        .ok_or("Z-Image image token count overflow")?;
    let padded_text = padded_to_sequence_multiple(text_tokens)?;
    let padded_image = padded_to_sequence_multiple(image_tokens)?;
    let position_count = padded_text
        .checked_add(padded_image)
        .ok_or("Z-Image position count overflow")?;
    let output_len = position_count
        .checked_mul(ROPE_HEAD_WIDTH)
        .ok_or("Z-Image RoPE output size overflow")?;
    let mut output = zeroed_f32("Z-Image RoPE", output_len)?;

    for position_index in 0..position_count {
        let positions = if position_index < padded_text {
            [(position_index + 1) as f32, 0.0, 0.0]
        } else {
            let image_index = position_index - padded_text;
            if image_index < image_tokens {
                [
                    (padded_text + 1) as f32,
                    (image_index / patch_width) as f32,
                    (image_index % patch_width) as f32,
                ]
            } else {
                [0.0, 0.0, 0.0]
            }
        };

        let mut output_index = position_index * ROPE_HEAD_WIDTH;
        for (axis, dimension) in ROPE_AXES.iter().copied().enumerate() {
            let half = dimension / 2;
            let end = (dimension as f32 - 2.0) / dimension as f32;
            let step = end / (half - 1) as f32;
            for frequency in 0..half {
                let scale = frequency as f32 * step;
                let omega = 1.0 / ROPE_THETA.powf(scale);
                let angle = positions[axis] * omega;
                output[output_index] = angle.cos();
                output[output_index + 1] = angle.sin();
                output_index += 2;
            }
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::{
        patchify_latent, time_snr_shift, unpatchify_latent, z_image_rope, z_image_sigmas,
        TorchMt19937,
    };

    fn expected_seed_42_20_bits() -> Vec<u32> {
        vec![
            0x3ff6_a52a,
            0x3fbe_5f53,
            0x3f66_9567,
            0xc006_c0db,
            0xbf42_14e2,
            0x3f8a_0650,
            0x3f4d_0143,
            0x3fd7_1e93,
            0x3eb6_3345,
            0xbf2f_c686,
            0xbefc_9934,
            0x3e77_4894,
            0xbe6d_2eed,
            0x3d2b_0c00,
            0xbe80_ce7a,
            0x3f5c_1fb0,
            0xbe9e_9482,
            0xbeca_9a91,
            0x3f4d_ac3c,
            0xbf1f_20e0,
        ]
    }

    #[test]
    fn torch_mt19937_recomputes_the_final_sixteen_values() {
        let mut values = vec![0.0; 20];
        TorchMt19937::new(42).fill_normal(&mut values);
        assert_eq!(
            values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected_seed_42_20_bits()
        );
    }

    #[test]
    fn torch_mt19937_uses_the_double_fallback_for_short_vectors() {
        let mut values = vec![0.0; 5];
        TorchMt19937::new(42).fill_normal(&mut values);
        assert_eq!(
            values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            [
                0x3eac_62ae,
                0x3e03_e69d,
                0x3e70_16e7,
                0x3e6b_dc6c,
                0xbf8f_b9c2
            ]
        );
    }

    #[test]
    fn discrete_flow_schedule_has_eight_steps_and_a_zero_tail() {
        let sigmas = z_image_sigmas(8).unwrap();
        assert_eq!(sigmas.len(), 9);
        assert_eq!(sigmas[0].to_bits(), time_snr_shift(3.0, 1.0).to_bits());
        assert_eq!(sigmas[8].to_bits(), 0.0f32.to_bits());
    }

    #[test]
    fn discrete_flow_schedule_rejects_zero_steps() {
        assert_eq!(
            z_image_sigmas(0).unwrap_err(),
            "Z-Image steps must be positive"
        );
    }

    #[test]
    fn z_image_rope_axes_sum_to_the_128_wide_head() {
        assert_eq!(z_image_rope(32, 64, 64).unwrap().len(), (32 + 1024) * 128);
    }

    #[test]
    fn z_image_rope_pads_text_and_image_positions_independently() {
        let rope = z_image_rope(1, 2, 2).unwrap();
        assert_eq!(rope.len(), 64 * 128);
        assert!((rope[0] - 1.0f32.cos()).abs() < 1e-6);
        assert!((rope[1] - 1.0f32.sin()).abs() < 1e-6);

        let image = 32 * 128;
        assert!((rope[image] - 33.0f32.cos()).abs() < 1e-6);
        assert!((rope[image + 1] - 33.0f32.sin()).abs() < 1e-6);

        let image_padding = 33 * 128;
        assert_eq!(&rope[image_padding..image_padding + 2], &[1.0, 0.0]);
    }

    #[test]
    fn z_image_rope_rejects_invalid_shapes() {
        assert!(z_image_rope(0, 64, 64).is_err());
        assert!(z_image_rope(1, 0, 64).is_err());
        assert!(z_image_rope(1, usize::MAX, 64).is_err());
    }

    #[test]
    fn patch_layout_keeps_channels_inside_each_spatial_position() {
        let latent = [0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0];
        assert_eq!(
            patchify_latent(&latent, 2, 2, 2).unwrap(),
            [0.0, 10.0, 1.0, 11.0, 2.0, 12.0, 3.0, 13.0]
        );
    }

    #[test]
    fn latent_patch_round_trip_preserves_spatial_and_channel_order() {
        let latent = (0..16 * 64 * 64)
            .map(|value| value as f32)
            .collect::<Vec<_>>();
        let patches = patchify_latent(&latent, 16, 64, 64).unwrap();
        assert_eq!(patches.len(), 1024 * 64);
        assert_eq!(unpatchify_latent(&patches, 16, 64, 64).unwrap(), latent);
    }

    #[test]
    fn patching_rejects_bad_lengths_and_overflowing_shapes() {
        assert!(patchify_latent(&[], 16, 64, 64).is_err());
        assert!(patchify_latent(&[], usize::MAX, 2, 2).is_err());
        assert!(unpatchify_latent(&[], 16, 64, 64).is_err());
    }
}
