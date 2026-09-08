//! CAM++ speaker (x-vector) encoder with its frontend.
//!
//! Frontend (reference `speaker/encoder.py` + `utils/audio.py`):
//!   prompt WAV at 48 kHz → stored 41-tap FIR (torchaudio 3:1 sinc) → 16 kHz
//!   → Kaldi fbank (povey window, pre-emph 0.97, 80 HTK mel bins, log,
//!   mean-normalized) → CAM++ (FCM stem, TDNN, 3 dense blocks, masked
//!   statistics pooling, dense → 512-dim x-vector).

use super::blas::sys;

use crate::core::tensor::TensorSource;
use crate::models::dots::patch_encoder::load_f16_f32;

pub(in crate::models::dots) mod exp;
mod log;
mod melbank;

const MEL_BINS: usize = 80;
const SR_16K: usize = 16_000;
const FRAME_LEN: usize = 400; // 25 ms
const FRAME_SHIFT: usize = 160; // 10 ms
const FFT_SIZE: usize = 512;
const PREEMPH: f32 = 0.97;
const MEL_FLOOR: f32 = f32::EPSILON;
const BN_EPS: f32 = 1e-5;

unsafe extern "C" {
    fn powf(x: f32, y: f32) -> f32;
}

// ---------------------------------------------------------------------------
// Generic high-quality resampler (kaiser-windowed sinc, 64 taps, rolloff 0.95)
// ---------------------------------------------------------------------------

pub struct Resampler {
    kernel: Vec<f32>,
    ratio: f64, // orig / new (>= 1 assumed; downsampling path used for 48k->16k)
    k_max: usize,
    phases: usize,
    stride: usize,
}

fn bessel_i0(x: f64) -> f64 {
    if x == 0.0 {
        return 1.0;
    }
    let mut sum = 1.0f64;
    let mut term = 1.0f64;
    let half = x / 2.0;
    for k in 1..=80 {
        term *= half * half / (k as f64 * k as f64);
        sum += term;
        if term.abs() < 1e-18 * sum.abs() {
            break;
        }
    }
    sum
}

const TORCH_2_8_KAISER_64_HALVES: [u32; 138] = [
    0x29daef1a, 0xb2e1ec5e, 0x3391d3a8, 0xb40b9767, 0x345eba5b, 0xb497de38, 0x34ac02a0, 0xb48a8308,
    0xad3e08f2, 0x351e5f2e, 0xb5e35413, 0x366d87b5, 0xb6d6aba4, 0x37314b7a, 0xb78934bb, 0x37c9b276,
    0xb80df03e, 0x3840342e, 0xb87b23b4, 0x389e89ff, 0xb8c16b84, 0x38e3b29b, 0xb900e3cc, 0x390b711d,
    0xb90e7e02, 0x39066bb0, 0xb8de231b, 0x3887b58f, 0x30c225bb, 0xb8c27779, 0x396429e0, 0xb9c627ef,
    0x3a16f9ae, 0xba54d576, 0x3a8e1658, 0xbab5e7b9, 0x3ae0db1d, 0xbb06bb9e, 0x3b1ce13e, 0xbb31a037,
    0x3b4379c4, 0xbb50b3d3, 0x3b576134, 0xbb556d33, 0x3b48aebd, 0xbb2ef785, 0x3b062eef, 0xba98d29a,
    0xb24c9eb6, 0x3ac09fc9, 0xbb555295, 0x3bafa050, 0xbbfeda13, 0x3c2be57f, 0xbc5cc610, 0x3c88b6eb,
    0xbca4883e, 0x3cc15cba, 0xbcdea91b, 0x3cfbd4fe, 0xbd0c1f87, 0x3d19a0f5, 0xbd261d37, 0x3d314616,
    0xbd3ad40a, 0x3d428852, 0xbd482f7e, 0x3d4ba372, 0x3f733333, 0x29daef1a, 0x29daef1a, 0xb2ea7838,
    0x33b0fea4, 0xb4529d7e, 0x34d88246, 0xb5489bbf, 0x35ab83b8, 0xb6093e62, 0x364f6598, 0xb694c3f1,
    0x36cb2796, 0xb70410d4, 0x372306f2, 0xb73dbaba, 0x374d148e, 0xb7473d08, 0x371f587b, 0xb68aecde,
    0xb6b26fd2, 0x37a8e5c5, 0xb82ea92b, 0x38953d08, 0xb8e66df8, 0x3926659f, 0xb9649d9c, 0x3996ce74,
    0xb9c01a9e, 0x39ed0741, 0xba0ddcbb, 0x3a24c7e2, 0xba398a42, 0x3a49fbc0, 0xba538a77, 0x3a53472e,
    0xba45f8e1, 0x3a283a56, 0xb9ed4250, 0x3937b6df, 0x3952f271, 0xba339730, 0x3aa7fe35, 0xbb028495,
    0x3b3827a3, 0xbb742b6b, 0x3b9aaed0, 0xbbbd037b, 0x3bdff307, 0xbc010b0f, 0x3c10df6b, 0xbc1e7dd3,
    0x3c28cd5d, 0xbc2e97e4, 0x3c2e8c2e, 0xbc273ebc, 0x3c172979, 0xbbf94c19, 0x3babc16c, 0xbb02f48e,
    0xbb15680b, 0x3bff9c5a, 0xbc7387f2, 0x3cc41e66, 0xbd12cc32, 0x3d556895, 0xbd9bc7e9, 0x3dee8e41,
    0xbe528cc9, 0x3f226897,
];

const TORCH_2_8_KAISER_128_HALVES: [u32; 272] = [
    0x29daef1a, 0xb24d2243, 0x32c7ead4, 0xb32b6eee, 0x3386963e, 0xb3c56988, 0x3408c8e1, 0xb43430de,
    0x34623f68, 0xb48750c9, 0x34998906, 0xb4a39775, 0x34a028f7, 0xb488ce7e, 0x342c4be0, 0x2d3e08f2,
    0xb4822c20, 0x351c6bfe, 0xb58aebfe, 0x35d833a8, 0xb61b7cb9, 0x3653a44f, 0xb68a0213, 0x36adabd2,
    0xb6d3c156, 0x36fabb15, 0xb710457c, 0x37214b1f, 0xb72edc1d, 0x373723ad, 0xb738128a, 0x372f72d1,
    0xb71af167, 0x36f085c3, 0xb68a7f3e, 0xb0381b09, 0x36b1e2fd, 0xb746674e, 0x37a43b0f, 0xb7ef1408,
    0x38215d7e, 0xb84ebe8e, 0x387e7e0d, 0xb89784d1, 0x38af38a7, 0xb8c53351, 0x38d82265, 0xb8e68ea9,
    0x38eee64a, 0xb8ef8a1a, 0x38e6d80b, 0xb8d343b8, 0x38b361bb, 0xb88609d8, 0x3814c933, 0x30c225bb,
    0xb8321378, 0x38c01288, 0xb919df2a, 0x3958ff27, 0xb98e01ff, 0x39b08efc, 0xb9d3123e, 0x39f4483e,
    0xba096272, 0x3a167ecc, 0xba20a82e, 0x3a2708a4, 0xba28cc8d, 0x3a2529f8, 0xba1b6c1c, 0x3a0af7e1,
    0xb9e6b420, 0x39a89d07, 0xb93725e6, 0xb1af83f9, 0x39524bf6, 0xb9de4c64, 0x3a2ea0f0, 0xba719fb3,
    0x3a9b347f, 0xbabd7fda, 0x3ade91ed, 0xbafd316d, 0x3b0c0824, 0xbb16ea8c, 0x3b1e9366, 0xbb225b00,
    0x3b21a4b7, 0xbb1be586, 0x3b10aa32, 0xbaff3b5a, 0x3ad11e90, 0xba96f0eb, 0x3a21fdd9, 0x324c9eb6,
    0xba35de92, 0x3abe40af, 0xbb13fedf, 0x3b4ae1b3, 0xbb813254, 0x3b9c7bf1, 0xbbb6735f, 0x3bce2df0,
    0xbbe2b828, 0x3bf3191b, 0xbbfe593e, 0x3c01c467, 0xbc00e310, 0x3bf844e0, 0xbbe65227, 0x3bcb5f96,
    0xbba7028a, 0x3b71fcf3, 0xbb028d36, 0xb2a612d1, 0x3b14f224, 0xbb9d950c, 0x3bf88c57, 0xbc2d33ac,
    0x3c60fe48, 0xbc8b7e02, 0x3ca73803, 0xbcc3475e, 0x3cdf4069, 0xbcfab3fc, 0x3d0a98fe, 0xbd1725b3,
    0x3d22ca5d, 0xbd2d5468, 0x3d369521, 0xbd3e6344, 0x3d449b92, 0xbd49219e, 0x3d4be0fb, 0x3f733333,
    0x29daef1a, 0xb1e93ed6, 0x32330446, 0xb25bae6c, 0x32503fa5, 0xb1ced965, 0xb2134e90, 0x3317aea6,
    0xb3a845bb, 0x3418952e, 0xb477f5ad, 0x34bb36fe, 0xb505c319, 0x3536c5f1, 0xb5704fa9, 0x359897f2,
    0xb5bb90e2, 0x35df4cee, 0xb600a521, 0x360f1f04, 0xb6190326, 0x361bdc1d, 0xb614d46f, 0x3600c910,
    0xb5b8dc87, 0x35119533, 0x352a00df, 0xb6130a21, 0x368baf95, 0xb6dc55f6, 0x371db66a, 0xb7540a6a,
    0x37881dd6, 0xb7a86a3a, 0x37c9decd, 0xb7eb1d97, 0x380538fe, 0xb812e9f6, 0x381d764f, 0xb8239321,
    0x3823db2c, 0xb81cdec7, 0x380d3204, 0xb7e6ef6f, 0x379d046d, 0xb6eae152, 0xb7026d48, 0x37d71bcb,
    0xb8432c58, 0x38933f02, 0xb8c9f4f8, 0x390242f6, 0xb920afc9, 0x393f4647, 0xb95cd84e, 0x39780cc5,
    0xb987b13b, 0x39909d38, 0xb995f178, 0x3996d394, 0xb9926da1, 0x3987faf0, 0xb96da33a, 0x393cd144,
    0xb8f9a7a0, 0x3835a440, 0x384472e0, 0xb91dd76f, 0x398ba0cb, 0xb9cd8a61, 0x3a0997a8, 0xba2d5cde,
    0x3a51010b, 0xba73460f, 0x3a89675b, 0xba971243, 0x3aa1e06d, 0xbaa90c9d, 0x3aabd6de, 0xbaa98a3c,
    0x3aa18604, 0xba934512, 0x3a7ccb51, 0xba45631c, 0x3a0049a6, 0xb937a1c6, 0xb9436e1d, 0x3a1a9a9e,
    0xba86b37c, 0x3ac36637, 0xbb00f4d9, 0x3b2044ac, 0xbb3eac8d, 0x3b5b1f73, 0xbb74826c, 0x3b84d823,
    0xbb8cc19b, 0x3b916dfd, 0xbb925688, 0x3b8f01ab, 0xbb870779, 0x3b742eab, 0xbb4ff86f, 0x3b214179,
    0xbad04cbc, 0x3a1443a4, 0x3a1d0bb2, 0xbaf785d5, 0x3b570a83, 0xbb9bac36, 0x3bcd5015, 0xbbff3f6b,
    0x3c1813fa, 0xbc2f4a94, 0x3c447b9c, 0xbc56d36d, 0x3c657523, 0xbc6f7cd1, 0x3c740090, 0xbc72121b,
    0x3c68bd7c, 0xbc570572, 0x3c3bdeac, 0xbc16232a, 0x3bc8ff3f, 0xbb1543e9, 0xbb2646f1, 0x3c0b3831,
    0xbc82244f, 0x3cce287e, 0xbd182298, 0x3d5a919d, 0xbd9e0c32, 0x3df051d7, 0xbe531bdd, 0x3f2274ce,
];

fn pinned_torch_2_8_kaiser_kernel(width: usize) -> Option<Vec<f32>> {
    let words: &[u32] = match width {
        64 => &TORCH_2_8_KAISER_64_HALVES,
        128 => &TORCH_2_8_KAISER_128_HALVES,
        _ => return None,
    };
    let half = words.len() / 2;
    let taps = half * 2 - 1;
    let mut kernel = Vec::with_capacity(taps * 2);
    kernel.extend(words[..half].iter().copied().map(f32::from_bits));
    kernel.extend(words[..half - 1].iter().rev().copied().map(f32::from_bits));
    kernel.extend(words[half..].iter().copied().map(f32::from_bits));
    kernel.extend(words[half + 1..].iter().rev().copied().map(f32::from_bits));
    Some(kernel)
}

impl Resampler {
    /// Resampler driven by a stored FIR kernel (the checkpoint's
    /// `resample.kernel`, a 41-tap 3:1 lowpass): `out[n] = Σ_k kernel[k]·x[n·3 + k − 19]`.
    pub fn from_kernel(kernel: &[f32]) -> Result<Self, String> {
        if kernel.len() != 41 {
            return Err(format!(
                "speaker resample kernel must have 41 taps, got {}",
                kernel.len()
            ));
        }
        Ok(Self {
            kernel: kernel.to_vec(),
            ratio: 3.0,
            k_max: 19,
            phases: 0,
            stride: 0,
        })
    }

    /// Build a kaiser-windowed sinc resampler (torchaudio-style:
    /// lowpass_filter_width=64, rolloff=0.95, sinc_interp_kaiser).
    pub fn new(orig: u32, new: u32) -> Self {
        Self::with_width(orig, new, 64)
    }

    pub(crate) fn with_width(orig: u32, new: u32, lowpass_filter_width: usize) -> Self {
        assert!(orig > 0 && new > 0 && lowpass_filter_width > 0);
        let mut a = orig;
        let mut b = new;
        while b != 0 {
            (a, b) = (b, a % b);
        }
        let stride = (orig / a) as usize;
        let phases = (new / a) as usize;
        let ratio = orig as f64 / new as f64;
        let base_freq = stride.min(phases) as f64 * 0.95;
        let k_max = (lowpass_filter_width as f64 * stride as f64 / base_freq).ceil() as usize;
        let taps = 2 * k_max + stride;
        // ponytail: exact Torch 2.8 parity is intentionally capped at the two
        // official 24 kHz -> 48 kHz Dots widths; other pairs use this generic
        // high-quality fallback rather than vendoring ATen Cephes and SLEEF.
        if orig == 24_000 && new == 48_000 {
            if let Some(kernel) = pinned_torch_2_8_kaiser_kernel(lowpass_filter_width) {
                return Self {
                    kernel,
                    ratio,
                    k_max,
                    phases,
                    stride,
                };
            }
        }
        let beta = 14.769_656_459_379_492;
        let i0_beta = bessel_i0(beta);
        let mut kernel = Vec::with_capacity(phases * taps);
        for phase in 0..phases {
            for tap in 0..taps {
                let index = (tap as f64 - k_max as f64) / stride as f64;
                let t = ((-(phase as f64) / phases as f64 + index) * base_freq)
                    .clamp(-(lowpass_filter_width as f64), lowpass_filter_width as f64);
                let rel = t / lowpass_filter_width as f64;
                let window = bessel_i0(beta * (1.0 - rel * rel).max(0.0).sqrt()) / i0_beta;
                let angle = std::f64::consts::PI * t;
                let sinc = if angle == 0.0 {
                    1.0
                } else {
                    angle.sin() / angle
                };
                kernel.push((sinc * window * base_freq / stride as f64) as f32);
            }
        }
        Self {
            kernel,
            ratio,
            k_max,
            phases,
            stride,
        }
    }

    pub fn resample(&self, input: &[f32]) -> Vec<f32> {
        if self.phases != 0 {
            let out_len = input
                .len()
                .saturating_mul(self.phases)
                .div_ceil(self.stride);
            let taps = self.kernel.len() / self.phases;
            let mut out = vec![0.0f32; out_len];
            for (index, value) in out.iter_mut().enumerate() {
                let frame = index / self.phases;
                let phase = index % self.phases;
                let source_start = frame * self.stride;
                let phase_kernel = &self.kernel[phase * taps..(phase + 1) * taps];
                for (tap, &weight) in phase_kernel.iter().enumerate() {
                    let source = source_start as isize + tap as isize - self.k_max as isize;
                    if (0..input.len() as isize).contains(&source) {
                        *value = input[source as usize].mul_add(weight, *value);
                    }
                }
            }
            return out;
        }
        let out_len = input.len().div_ceil(3);
        let mut out = vec![0.0f32; out_len];
        let k_max = self.k_max as f64;
        for n in 0..out_len {
            let center = n as f64 * self.ratio;
            let mut sum = 0.0f32;
            let start = (center - k_max).floor() as i64;
            for tap in 0..self.kernel.len() {
                let src = start + tap as i64;
                if (0..input.len() as i64).contains(&src) {
                    sum = input[src as usize].mul_add(self.kernel[tap], sum);
                }
            }
            out[n] = sum;
        }
        out
    }
}

#[cfg(test)]
mod resampler_tests {
    use super::Resampler;

    #[test]
    fn stored_kernel_resampler_matches_torch_offset_and_fma_contract() {
        let mut kernel = vec![0.0; 41];
        kernel[19] = 1.0;
        kernel[20] = f32::from_bits(0x3f80_0001);
        let input = [
            f32::from_bits(0xbf80_0002),
            f32::from_bits(0x3f80_0001),
            0.0,
        ];
        let output = Resampler::from_kernel(&kernel).unwrap().resample(&input);
        assert_eq!(output[0].to_bits(), 0x2880_0000);
    }

    #[test]
    fn stored_kernel_resampler_keeps_torch_ceil_length_tail() {
        let mut kernel = vec![0.0; 41];
        kernel[19] = 1.0;
        let output = Resampler::from_kernel(&kernel)
            .unwrap()
            .resample(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(output, vec![1.0, 4.0]);
    }

    #[test]
    fn torchaudio_24k_to_48k_matches_f32_kernel_and_fma_contract() {
        let input: Vec<f32> = (0..80)
            .map(|index| (((index * 37) % 101) as f32 - 50.0) / 32768.0)
            .collect();
        let cases = [
            (
                64,
                [
                    0xbacb_e7ac,
                    0xbabf_2232,
                    0xb9be_8180,
                    0x3a37_911e,
                    0x3a36_79f8,
                    0xb993_b501,
                    0xba9a_f2b0,
                    0xba92_3b61,
                    0xb909_d6dd,
                    0x3a6f_9ad3,
                    0x3a8d_4907,
                    0x3924_d090,
                    0xba7a_6d00,
                    0xba82_366c,
                    0x3984_0d85,
                    0x3ac0_e5ee,
                ],
            ),
            (
                128,
                [
                    0xbaca_fdc5,
                    0xbac1_223c,
                    0xb9c3_f998,
                    0x3a3b_1e69,
                    0x3a3a_0c82,
                    0xb999_8858,
                    0xba9d_18d0,
                    0xba91_2e4b,
                    0xb8ec_b62d,
                    0x3a6e_74f8,
                    0x3a8a_a835,
                    0x3925_2953,
                    0xba74_ff92,
                    0xba81_b1b1,
                    0x3972_a579,
                    0x3abf_cf8d,
                ],
            ),
        ];
        for (width, expected) in cases {
            let actual = Resampler::with_width(24_000, 48_000, width).resample(&input);
            let actual_bits: [u32; 16] = std::array::from_fn(|index| actual[index].to_bits());
            assert_eq!(actual_bits, expected, "width={width}");
        }
    }
}

#[cfg(test)]
mod fbank_tests {
    use super::{
        kaldi_fbank, log::torch28_log, mel_matmul, melbank::torch28_mel_filterbank, povey_window,
        prepare_kaldi_frame, torch28_arm_mean_400, torch28_column_mean, torch28_rfft_power_512,
    };

    fn reduction_fixture() -> [f32; 400] {
        let mut state = 1u32;
        std::array::from_fn(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            f32::from_bits((state & 0x8000_0000) | 0x3f00_0000 | (state & 0x007f_ffff))
        })
    }

    #[test]
    fn torch28_arm_mean_matches_sumkernel_reduction_order() {
        let frame = reduction_fixture();
        let mean = torch28_arm_mean_400(&frame);
        assert_eq!(mean.to_bits(), 0xbcd9_310d);
        let post_dc: [u32; 6] = std::array::from_fn(|index| (frame[index] - mean).to_bits());
        assert_eq!(
            post_dc,
            [
                0x3f0f_22f4,
                0x3f0f_4f63,
                0xbf0f_37f6,
                0xbf6c_713d,
                0x3f77_36e8,
                0x3f1f_8ac7,
            ]
        );
    }

    #[test]
    fn kaldi_frame_removes_dc_before_replicated_left_preemphasis() {
        let frame = prepare_kaldi_frame(&reduction_fixture());
        let head: [u32; 6] = std::array::from_fn(|index| frame[index].to_bits());
        assert_eq!(
            head,
            [
                0x3c89_6940,
                0x3c8e_f720,
                0xbf8d_1d5d,
                0xbec3_0a64,
                0x3fee_4822,
                0xbea0_830c,
            ]
        );
    }

    #[test]
    fn povey_window_matches_torch28_float32_words() {
        let window = povey_window();
        let head: [u32; 8] = std::array::from_fn(|index| window[index].to_bits());
        assert_eq!(
            head,
            [
                0x0000_0000,
                0x398b_01c2,
                0x3a61_d13c,
                0x3ae0_ed7f,
                0x3b37_6219,
                0x3b85_f871,
                0x3bb6_9cf0,
                0x3bed_450d,
            ]
        );
    }

    #[test]
    fn torch28_rfft_matches_pocketfft_real_imag_and_power() {
        let mut input = [0.0f32; 512];
        let head = [
            0x0000_0000,
            0x3351_4aaa,
            0xb390_51c7,
            0x33d3_cf2d,
            0xb448_d49e,
            0xb550_bd88,
            0x3522_32b8,
            0x3393_bf70,
            0xb4e2_3168,
            0xb5ac_0a4a,
            0x349e_ca65,
            0x34ef_f66d,
            0x3560_4879,
            0xb617_e844,
            0xb63e_f7c6,
            0xb686_34b4,
        ];
        for (value, word) in input.iter_mut().zip(head) {
            *value = f32::from_bits(word);
        }
        let (real, imag, power) = torch28_rfft_power_512(&input);
        let actual_real: [u32; 16] = std::array::from_fn(|index| real[index].to_bits());
        let actual_imag: [u32; 16] = std::array::from_fn(|index| imag[index].to_bits());
        let actual_power: [u32; 16] = std::array::from_fn(|index| power[index].to_bits());
        assert_eq!(
            actual_real,
            [
                0xb71e76de, 0xb71c3df1, 0xb715a491, 0xb70ade6f, 0xb6f87fbf, 0xb6d47675, 0xb6aaba7f,
                0xb67922d3, 0xb616c516, 0xb5455e6c, 0x35505236, 0x3616718c, 0x3672cd28, 0x36a33423,
                0x36c76648, 0x36e4f00c
            ]
        );
        assert_eq!(
            actual_imag,
            [
                0x00000000, 0x35cf7713, 0x364c61f0, 0x36957b4c, 0x36c0569a, 0x36e57ffa, 0x3701f0e4,
                0x370d4e1d, 0x3714861a, 0x37176828, 0x3715e5d5, 0x3710135c, 0x370626de, 0x36f0ed48,
                0x36ceec9a, 0x36a7662e
            ]
        );
        assert_eq!(
            actual_power,
            [
                0x2ec42de2, 0x2ec3f827, 0x2ec3577c, 0x2ec24d84, 0x2ec0dce3, 0x2ebf094f, 0x2ebcd76d,
                0x2eba4cdf, 0x2eb7701b, 0x2eb44869, 0x2eb0ddbb, 0x2ead38ba, 0x2ea9627e, 0x2ea56499,
                0x2ea148e8, 0x2e9d1978
            ]
        );
        assert_eq!(real[256].to_bits(), 0x36c5_1809);
        assert_eq!(imag[256].to_bits(), 0x0000_0000);
        assert_eq!(power[256].to_bits(), 0x2e17_be00);
    }

    #[test]
    #[ignore = "requires DOTS_FBANK_INPUT sidecar"]
    fn real_input_mel_matmul_matches_pinned_oracle_bitwise() {
        let bytes = std::fs::read(std::env::var_os("DOTS_FBANK_INPUT").unwrap()).unwrap();
        let input = bytes
            .chunks_exact(4)
            .map(|word| f32::from_le_bytes(word.try_into().unwrap()))
            .collect::<Vec<_>>();
        let raw_frame = std::array::from_fn(|index| input[index]);
        let prepared = prepare_kaldi_frame(&raw_frame);
        let window = povey_window();
        let mut fft_input = [0.0f32; 512];
        for index in 0..400 {
            fft_input[index] = prepared[index] * window[index];
        }
        let (_, _, power) = torch28_rfft_power_512(&fft_input);
        let actual = mel_matmul(&power, &torch28_mel_filterbank(), 1);
        let expected = [
            0x32c5_b0a2,
            0x331d_f98f,
            0x348b_473a,
            0x34dc_0c3b,
            0x34c3_0ff0,
            0x3506_c181,
            0x35cc_33e7,
            0x362b_671c,
            0x368e_b959,
            0x369f_c58a,
            0x367a_d858,
            0x3668_4be4,
            0x36da_a7b3,
            0x3760_39f1,
            0x363a_3329,
            0x35fc_c422,
            0x3629_5782,
            0x35cd_0576,
            0x35ab_a603,
            0x353b_1e59,
            0x359d_d3bf,
            0x36a7_ff98,
            0x368b_a0c5,
            0x35f1_f470,
            0x3510_67e4,
            0x35aa_6fd9,
            0x3510_f7f3,
            0x3559_bef5,
            0x35b1_2763,
            0x3533_4a15,
            0x3631_9d06,
            0x3658_5cdf,
            0x3587_5d57,
            0x35d9_1b24,
            0x3725_0358,
            0x3666_d682,
            0x364b_be27,
            0x3676_8824,
            0x36a4_523e,
            0x3683_f173,
            0x3663_844e,
            0x36cd_5120,
            0x3710_c065,
            0x36a3_5d56,
            0x3741_0d26,
            0x3777_d3ff,
            0x372e_18c5,
            0x36da_bf85,
            0x36af_42f8,
            0x3583_4fa5,
            0x35a0_9672,
            0x35fe_976b,
            0x361a_5a52,
            0x36b7_12c9,
            0x3727_8ee5,
            0x37d0_bb01,
            0x375f_0207,
            0x3762_1ca3,
            0x3680_f5ec,
            0x3741_fe6f,
            0x3818_f688,
            0x3726_6e4e,
            0x3756_7c6b,
            0x3707_4773,
            0x370c_de61,
            0x376e_b86d,
            0x37d3_8553,
            0x374a_b6da,
            0x36df_5de0,
            0x36a1_174b,
            0x372a_efb8,
            0x36b4_5bb8,
            0x3704_e6d0,
            0x3791_4b01,
            0x3755_a348,
            0x37dd_0ba8,
            0x3780_5833,
            0x3756_d307,
            0x3683_4a0b,
            0x36da_4d4f,
        ];
        assert_eq!(
            actual
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected
        );
    }

    #[test]
    #[ignore = "requires DOTS_FBANK_MATMUL, DOTS_FBANK_CLAMP, and DOTS_FBANK_LOG sidecars"]
    fn clamp_and_log_match_pinned_torch28_arm_words() {
        let read = |name: &str| {
            std::fs::read(std::env::var_os(name).unwrap())
                .unwrap()
                .chunks_exact(4)
                .map(|word| f32::from_le_bytes(word.try_into().unwrap()))
                .collect::<Vec<_>>()
        };
        let matmul = read("DOTS_FBANK_MATMUL");
        let expected_clamp = read("DOTS_FBANK_CLAMP");
        let expected_log = read("DOTS_FBANK_LOG");
        assert_eq!(matmul.len(), expected_clamp.len());
        assert_eq!(matmul.len(), expected_log.len());
        for index in 0..matmul.len() {
            let clamped = matmul[index].max(f32::EPSILON);
            assert_eq!(
                clamped.to_bits(),
                expected_clamp[index].to_bits(),
                "clamp[{index}]"
            );
        }
        for index in 0..expected_clamp.len() {
            assert_eq!(
                torch28_log(expected_clamp[index]).to_bits(),
                expected_log[index].to_bits(),
                "log[{index}]"
            );
        }
    }

    #[test]
    #[ignore = "requires DOTS_FBANK_LOG, DOTS_FBANK_MEAN, and DOTS_FBANK_FINAL sidecars"]
    fn column_mean_and_centering_match_pinned_torch28_arm_words() {
        let read = |name: &str| {
            std::fs::read(std::env::var_os(name).unwrap())
                .unwrap()
                .chunks_exact(4)
                .map(|word| f32::from_le_bytes(word.try_into().unwrap()))
                .collect::<Vec<_>>()
        };
        let input = read("DOTS_FBANK_LOG");
        let expected_mean = read("DOTS_FBANK_MEAN");
        let expected_final = read("DOTS_FBANK_FINAL");
        let mean = torch28_column_mean(&input, 590, 80);
        assert_eq!(mean.len(), expected_mean.len());
        for (index, (&actual, &expected)) in mean.iter().zip(&expected_mean).enumerate() {
            assert_eq!(actual.to_bits(), expected.to_bits(), "mean[{index}]");
        }
        let actual = input
            .iter()
            .enumerate()
            .map(|(index, &value)| value - mean[index % 80])
            .collect::<Vec<_>>();
        for (index, (&actual, &expected)) in actual.iter().zip(&expected_final).enumerate() {
            assert_eq!(actual.to_bits(), expected.to_bits(), "fbank[{index}]");
        }
    }

    #[test]
    #[ignore = "requires DOTS_FBANK_INPUT and DOTS_FBANK_ORACLE sidecars"]
    fn real_input_fbank_matches_pinned_oracle_bitwise() {
        let read = |name: &str| {
            let bytes = std::fs::read(std::env::var_os(name).unwrap()).unwrap();
            bytes
                .chunks_exact(4)
                .map(|word| f32::from_le_bytes(word.try_into().unwrap()))
                .collect::<Vec<_>>()
        };
        let input = read("DOTS_FBANK_INPUT");
        let expected = read("DOTS_FBANK_ORACLE");
        let actual = kaldi_fbank(&input);
        assert_eq!(actual.len(), expected.len());
        for (index, (left, right)) in actual.iter().zip(expected).enumerate() {
            assert_eq!(
                left.to_bits(),
                right.to_bits(),
                "fbank[{index}] rust={:08x} oracle={:08x}",
                left.to_bits(),
                right.to_bits()
            );
        }
    }
}

#[cfg(test)]
mod campplus_tests {
    use super::{
        cam_context, cam_gate, cam_stats_pooling, channel_major_to_time_major, conv1d_time_major,
        conv2d_forward, conv2d_forward_stride, dense_layer_forward, dense_layer_front,
        fcm_input_layout, fcm_output_layout, torch28_batch_norm_terms, torch28_contiguous_mean,
        transit_forward,
    };

    #[test]
    fn fcm_layouts_bridge_public_time_major_and_internal_channel_major_buffers() {
        let public = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [T=3, F=2]
        assert_eq!(
            fcm_input_layout(&public, 3, 2),
            [1.0, 3.0, 5.0, 2.0, 4.0, 6.0]
        );

        // [C=2, H=2, T=3] -> [T=3, C*H=4]
        let internal = [
            0.0, 1.0, 2.0, 10.0, 11.0, 12.0, 100.0, 101.0, 102.0, 110.0, 111.0, 112.0,
        ];
        assert_eq!(
            fcm_output_layout(&internal, 3, 2, 2),
            [0.0, 10.0, 100.0, 110.0, 1.0, 11.0, 101.0, 111.0, 2.0, 12.0, 102.0, 112.0]
        );
    }

    fn fma_fixture() -> (Vec<f32>, Vec<f32>) {
        let pairs = [
            (0x3f1f_f38a, 0x3f5e_5b40),
            (0xbf20_f634, 0xbf96_afa8),
            (0x3ce1_63e2, 0xbf2f_75f0),
            (0xbe5f_7a9f, 0x3f32_1f80),
            (0x3e41_6f1b, 0xbf34_7ce0),
            (0x3cd5_d346, 0xbea8_7f20),
        ];
        let mut weight = vec![0.0; 9];
        let mut input = vec![0.0; 6];
        for (index, &(w, x)) in pairs.iter().enumerate() {
            weight[index + 3] = f32::from_bits(w);
            input[index] = f32::from_bits(x);
        }
        (weight, input)
    }

    #[test]
    fn regular_and_stride_conv2d_use_source_order_fma() {
        let (weight, input) = fma_fixture();
        let regular = conv2d_forward(&weight, None, &input, 3, 2, 1, 1, 3, 3, 1, 1);
        let stride = conv2d_forward_stride(&weight, None, &input, 3, 2, 1, 1, 3, 3, 1, 1, 1, 1);
        assert_eq!(regular[1].to_bits(), 0x3f78_6d04);
        assert_eq!(stride[1].to_bits(), 0x3f78_6d04);
    }

    #[test]
    fn batch_norm_fuses_beta_but_not_final_affine() {
        let input = f32::from_bits(0x3e03_f690);
        let weight = f32::from_bits(0x3ebf_6ada);
        let bias = f32::from_bits(0xbde8_ace1);
        let mean = f32::from_bits(0xbe39_2875);
        let variance = f32::from_bits(0x3d81_7df1);
        let (alpha, beta) = torch28_batch_norm_terms(weight, bias, mean, variance);
        assert_eq!(alpha.to_bits(), 0x3fbe_4be3);
        assert_eq!(beta.to_bits(), 0x3e1e_ef63);
        assert_eq!((input * alpha + beta).to_bits(), 0x3eb1_8fce);
        assert_ne!(input.mul_add(alpha, beta).to_bits(), 0x3eb1_8fce);
    }

    #[test]
    fn tdnn_conv1d_uses_source_order_fma() {
        let pairs = [
            (0x3f1f_f38a, 0x3f5e_5b40),
            (0xbf20_f634, 0xbf96_afa8),
            (0x3ce1_63e2, 0xbf2f_75f0),
            (0xbe5f_7a9f, 0x3f32_1f80),
            (0x3e41_6f1b, 0xbf34_7ce0),
            (0x3cd5_d346, 0xbea8_7f20),
        ];
        let weight = pairs.map(|(weight, _)| f32::from_bits(weight));
        let input = [
            pairs[0].1, pairs[3].1, pairs[1].1, pairs[4].1, pairs[2].1, pairs[5].1,
        ]
        .map(f32::from_bits);
        let actual = conv1d_time_major(&weight, &input, 3, 1, 2, 1, 3, 1, 0, 1);
        assert_eq!(actual[0].to_bits(), 0x3f78_6d04);
    }

    #[test]
    fn torch28_contiguous_mean_and_cam_segments_keep_reduction_order() {
        let mut state = 1u32;
        let input: [f32; 295] = std::array::from_fn(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            f32::from_bits((state & 0x8000_0000) | 0x3f00_0000 | (state & 0x007f_ffff))
        });
        assert_eq!(torch28_contiguous_mean(&input).to_bits(), 0xbbeb_44f6);
        let context = cam_context(&input, 295, 1);
        assert_eq!(context[0].to_bits(), 0x3c3a_2dd3);
        assert_eq!(context[100].to_bits(), 0xbc5c_6872);
        assert_eq!(context[200].to_bits(), 0xbd2d_a2b7);
    }

    #[test]
    #[ignore = "requires DOTS_CAMPLUS_OUT_NONLINEAR and DOTS_CAMPLUS_STATS sidecars"]
    fn real_input_stats_pooling_matches_pinned_oracle_bitwise() {
        let read = |name: &str| {
            std::fs::read(std::env::var_os(name).unwrap())
                .unwrap()
                .chunks_exact(4)
                .map(|word| f32::from_le_bytes(word.try_into().unwrap()))
                .collect::<Vec<_>>()
        };
        let out_nonlinear =
            channel_major_to_time_major(&read("DOTS_CAMPLUS_OUT_NONLINEAR"), 295, 512);
        let expected = read("DOTS_CAMPLUS_STATS");
        let actual = cam_stats_pooling(&out_nonlinear, 295, 512);
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "stats[{index}] rust={:08x} oracle={:08x}",
                actual.to_bits(),
                expected.to_bits()
            );
        }
    }

    #[test]
    #[ignore = "requires DOTS_CAMPLUS_* model and dense checkpoint sidecars"]
    fn real_input_dense_projection_matches_pinned_oracle_bitwise() {
        use crate::{open_model_source, ComponentRole};

        let read = |name: &str| {
            std::fs::read(std::env::var_os(name).unwrap())
                .unwrap()
                .chunks_exact(4)
                .map(|word| f32::from_le_bytes(word.try_into().unwrap()))
                .collect::<Vec<_>>()
        };
        let mmproj = std::path::PathBuf::from(std::env::var_os("DOTS_CAMPLUS_MMPROJ").unwrap());
        let source = open_model_source(&mmproj, ComponentRole::Mmproj).unwrap();
        let model = super::CamPlus::from_source(source.as_ref()).unwrap();
        let stats = read("DOTS_CAMPLUS_STATS");
        let expected_final = read("DOTS_CAMPLUS_FINAL");
        let actual = model.dense_projection(&stats);
        assert_eq!(actual.len(), expected_final.len());
        for (index, (&actual, &expected)) in actual.iter().zip(&expected_final).enumerate() {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "final[{index}] rust={:08x} oracle={:08x}",
                actual.to_bits(),
                expected.to_bits()
            );
        }
    }

    #[test]
    #[ignore = "requires DOTS_CAMPLUS_* dense block checkpoint sidecars"]
    fn real_input_dense_blocks_and_transits_match_pinned_oracle_bitwise() {
        use crate::{open_model_source, ComponentRole};

        let read = |name: &str| {
            std::fs::read(std::env::var_os(name).unwrap())
                .unwrap()
                .chunks_exact(4)
                .map(|word| f32::from_le_bytes(word.try_into().unwrap()))
                .collect::<Vec<_>>()
        };
        let assert_bits = |name: &str, actual: &[f32], expected: &[f32]| {
            assert_eq!(actual.len(), expected.len(), "{name} length");
            for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
                assert_eq!(actual.to_bits(), expected.to_bits(), "{name}[{index}]");
            }
        };
        let mmproj = std::path::PathBuf::from(std::env::var_os("DOTS_CAMPLUS_MMPROJ").unwrap());
        let source = open_model_source(&mmproj, ComponentRole::Mmproj).unwrap();
        let model = super::CamPlus::from_source(source.as_ref()).unwrap();
        let mut x = read("DOTS_CAMPLUS_TDNN");
        for (block_index, (input_channels, output_channels)) in
            [(128usize, 512usize), (256, 1024), (512, 1024)]
                .into_iter()
                .enumerate()
        {
            for (layer_index, layer) in model.blocks[block_index].layers.iter().enumerate() {
                x = dense_layer_forward(layer, &x, 295, input_channels + layer_index * 32);
            }
            let expected_block = channel_major_to_time_major(
                &read(&format!("DOTS_CAMPLUS_BLOCK{}", block_index + 1)),
                295,
                output_channels,
            );
            assert_bits(&format!("block{}", block_index + 1), &x, &expected_block);
            let mut next = vec![0.0f32; 295 * (output_channels / 2)];
            transit_forward(
                &model.transits[block_index],
                &x,
                295,
                output_channels,
                &mut next,
            );
            let expected_transit = channel_major_to_time_major(
                &read(&format!("DOTS_CAMPLUS_TRANSIT{}", block_index + 1)),
                295,
                output_channels / 2,
            );
            assert_bits(
                &format!("transit{}", block_index + 1),
                &next,
                &expected_transit,
            );
            x = next;
        }
    }

    #[test]
    #[ignore = "requires DOTS_CAMPLUS_* model and checkpoint sidecars"]
    fn real_input_fcm_and_tdnn_match_pinned_oracle_bitwise() {
        use crate::{open_model_source, ComponentRole};

        let read = |name: &str| {
            std::fs::read(std::env::var_os(name).unwrap())
                .unwrap()
                .chunks_exact(4)
                .map(|word| f32::from_le_bytes(word.try_into().unwrap()))
                .collect::<Vec<_>>()
        };
        let mmproj = std::path::PathBuf::from(std::env::var_os("DOTS_CAMPLUS_MMPROJ").unwrap());
        let source = open_model_source(&mmproj, ComponentRole::Mmproj).unwrap();
        let model = super::CamPlus::from_source(source.as_ref()).unwrap();
        let input = read("DOTS_CAMPLUS_FBANK");
        let expected_fcm = read("DOTS_CAMPLUS_FCM");
        let expected_tdnn = read("DOTS_CAMPLUS_TDNN");
        let expected_xvector =
            std::env::var_os("DOTS_CAMPLUS_XVECTOR").map(|_| read("DOTS_CAMPLUS_XVECTOR"));
        let expected_dense_nl2 = read("DOTS_CAMPLUS_DENSE_NL2");
        let expected_dense_local = read("DOTS_CAMPLUS_DENSE_LOCAL");
        let expected_dense_context = read("DOTS_CAMPLUS_DENSE_CONTEXT");
        let expected_gate_linear1 = read("DOTS_CAMPLUS_GATE_LINEAR1");
        let expected_gate_relu1 = read("DOTS_CAMPLUS_GATE_RELU1");
        let expected_gate_linear2 = read("DOTS_CAMPLUS_GATE_LINEAR2");
        let expected_gate_sigmoid = read("DOTS_CAMPLUS_GATE_SIGMOID");
        let expected_dense_gated = read("DOTS_CAMPLUS_DENSE_GATED");
        let expected_dense_concat = read("DOTS_CAMPLUS_DENSE_CONCAT");
        let frames = input.len() / super::MEL_BINS;
        let fcm = model.fcm(&input, frames);
        assert_eq!(fcm.len(), expected_fcm.len());
        for (index, (&actual, &expected)) in fcm.iter().zip(&expected_fcm).enumerate() {
            assert_eq!(actual.to_bits(), expected.to_bits(), "fcm[{index}]");
        }
        let tdnn = model.tdnn(&fcm, frames, super::conv1d_length(frames, 5, 2, 2));
        assert_eq!(tdnn.len(), expected_tdnn.len());
        for (index, (&actual, &expected)) in tdnn.iter().zip(&expected_tdnn).enumerate() {
            assert_eq!(actual.to_bits(), expected.to_bits(), "tdnn[{index}]");
        }
        let front = dense_layer_front(&model.blocks[0].layers[0], &expected_tdnn, 295, 128);
        for (name, actual, expected) in [
            ("dense_nl2", &front.nonlinear2, &expected_dense_nl2),
            ("dense_local", &front.local, &expected_dense_local),
            ("dense_context", &front.context, &expected_dense_context),
        ] {
            assert_eq!(actual.len(), expected.len(), "{name} length");
            for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
                assert_eq!(actual.to_bits(), expected.to_bits(), "{name}[{index}]");
            }
        }
        let layer = &model.blocks[0].layers[0];
        let assert_bits = |name: &str, actual: &[f32], expected: &[f32]| {
            assert_eq!(actual.len(), expected.len(), "{name} length");
            for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
                assert_eq!(
                    actual.to_bits(),
                    expected.to_bits(),
                    "{name}[{index}] rust={:08x} oracle={:08x}",
                    actual.to_bits(),
                    expected.to_bits()
                );
            }
        };
        let gate = cam_gate(layer, &front.local, &front.context, 295);
        assert_bits("gate_linear1", &gate.linear1, &expected_gate_linear1);
        assert_bits("gate_relu1", &gate.relu1, &expected_gate_relu1);
        assert_bits("gate_linear2", &gate.linear2, &expected_gate_linear2);
        assert_bits("gate_sigmoid", &gate.sigmoid, &expected_gate_sigmoid);
        assert_bits("dense_gated", &gate.gated, &expected_dense_gated);
        let dense_concat =
            dense_layer_forward(&model.blocks[0].layers[0], &expected_tdnn, 295, 128);
        assert_eq!(dense_concat.len(), expected_dense_concat.len());
        for (index, (&actual, &expected)) in
            dense_concat.iter().zip(&expected_dense_concat).enumerate()
        {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "dense_concat[{index}] rust={:08x} oracle={:08x}",
                actual.to_bits(),
                expected.to_bits()
            );
        }
        if let Some(expected_xvector) = expected_xvector {
            let xvector = model.encode(&input).unwrap();
            assert_eq!(xvector.len(), expected_xvector.len());
            for (index, (&actual, &expected)) in xvector.iter().zip(&expected_xvector).enumerate() {
                assert_eq!(actual.to_bits(), expected.to_bits(), "xvector[{index}]");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Kaldi-style fbank
// ---------------------------------------------------------------------------

fn mel_matmul(power: &[f32], filterbank: &[f32], frames: usize) -> Vec<f32> {
    const FFT_BINS: usize = FFT_SIZE / 2 + 1;
    debug_assert_eq!(power.len(), frames * FFT_BINS);
    debug_assert_eq!(filterbank.len(), MEL_BINS * FFT_BINS);
    let mut output = vec![0.0f32; frames * MEL_BINS];
    for frame in 0..frames {
        let power_row = &power[frame * FFT_BINS..(frame + 1) * FFT_BINS];
        for mel in 0..MEL_BINS {
            let mel_row = &filterbank[mel * FFT_BINS..(mel + 1) * FFT_BINS];
            let mut accumulator = 0.0f32;
            for bin in 0..FFT_BINS {
                accumulator = power_row[bin].mul_add(mel_row[bin], accumulator);
            }
            output[frame * MEL_BINS + mel] = accumulator;
        }
    }
    output
}

fn torch28_column_mean(input: &[f32], rows: usize, columns: usize) -> Vec<f32> {
    debug_assert_eq!(input.len(), rows * columns);
    let mut means = vec![0.0f32; columns];
    for column in 0..columns {
        let mut levels = [0.0f32; 4];
        for row in 0..rows {
            levels[0] += input[row * columns + column];
            let completed = row + 1;
            if completed % 16 == 0 {
                levels[1] += levels[0];
                levels[0] = 0.0;
            }
            if completed % 256 == 0 {
                levels[2] += levels[1];
                levels[1] = 0.0;
            }
            if completed % 4096 == 0 {
                levels[3] += levels[2];
                levels[2] = 0.0;
            }
        }
        for level in 1..levels.len() {
            levels[0] += levels[level];
        }
        means[column] = levels[0] / rows as f32;
    }
    means
}

fn torch28_contiguous_sum(input: &[f32]) -> f32 {
    const LANES: usize = 4;
    const ILP: usize = 4;
    const LEVELS: usize = 4;

    let vector_count = input.len() / LANES;
    let group_count = vector_count / ILP;
    let level_power = 4usize.max(
        (usize::BITS as usize - group_count.saturating_sub(1).leading_zeros() as usize) / LEVELS,
    );
    let level_step = 1usize << level_power;
    let level_mask = level_step - 1;
    let mut levels = [[[0.0f32; LANES]; ILP]; LEVELS];
    let mut group = 0;
    while group + level_step <= group_count {
        for offset in 0..level_step {
            for partial in 0..ILP {
                for lane in 0..LANES {
                    levels[0][partial][lane] +=
                        input[(group + offset) * ILP * LANES + partial * LANES + lane];
                }
            }
        }
        group += level_step;
        for level in 1..LEVELS {
            for partial in 0..ILP {
                for lane in 0..LANES {
                    levels[level][partial][lane] += levels[level - 1][partial][lane];
                    levels[level - 1][partial][lane] = 0.0;
                }
            }
            if group & (level_mask << (level * level_power)) != 0 {
                break;
            }
        }
    }
    while group < group_count {
        for partial in 0..ILP {
            for lane in 0..LANES {
                levels[0][partial][lane] += input[group * ILP * LANES + partial * LANES + lane];
            }
        }
        group += 1;
    }
    for level in 1..LEVELS {
        for partial in 0..ILP {
            for lane in 0..LANES {
                levels[0][partial][lane] += levels[level][partial][lane];
            }
        }
    }
    for vector in group_count * ILP..vector_count {
        for lane in 0..LANES {
            levels[0][0][lane] += input[vector * LANES + lane];
        }
    }
    for partial in 1..ILP {
        for lane in 0..LANES {
            levels[0][0][lane] += levels[0][partial][lane];
        }
    }
    let mut sum = 0.0f32;
    for &value in &input[vector_count * LANES..] {
        sum += value;
    }
    for lane in 0..LANES {
        sum += levels[0][0][lane];
    }
    sum
}

fn torch28_contiguous_mean(input: &[f32]) -> f32 {
    torch28_contiguous_sum(input) / input.len() as f32
}

fn cam_context(input: &[f32], time: usize, channels: usize) -> Vec<f32> {
    debug_assert_eq!(input.len(), time * channels);
    let segment_count = time.div_ceil(100);
    let mut segment_means = vec![0.0f32; channels * segment_count];
    let mut global_means = vec![0.0f32; channels];
    let mut row = vec![0.0f32; time];
    for channel in 0..channels {
        for frame in 0..time {
            row[frame] = input[frame * channels + channel];
        }
        global_means[channel] = torch28_contiguous_mean(&row);
        for segment in 0..segment_count {
            let start = segment * 100;
            let end = (start + 100).min(time);
            let mut sum = 0.0f32;
            for &value in &row[start..end] {
                sum += value;
            }
            segment_means[channel * segment_count + segment] = sum / (end - start) as f32;
        }
    }
    let mut context = vec![0.0f32; input.len()];
    for frame in 0..time {
        for channel in 0..channels {
            context[frame * channels + channel] =
                global_means[channel] + segment_means[channel * segment_count + frame / 100];
        }
    }
    context
}

/// In-place complex FFT over a `[re0, im0, re1, im1, ...]` buffer
/// (radix-2, iterative Cooley-Tukey).
pub fn fft_complex(re_im: &mut [f64]) {
    let n = re_im.len() / 2;
    debug_assert!(n.is_power_of_two());
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            re_im.swap(2 * i, 2 * j);
            re_im.swap(2 * i + 1, 2 * j + 1);
        }
    }
    let mut len = 2usize;
    while len <= n {
        let angle = -2.0 * std::f64::consts::PI / len as f64;
        let (wr, wi) = (angle.cos(), angle.sin());
        let half = len / 2;
        for start in (0..n).step_by(len) {
            let (mut w_r, mut w_i) = (1.0f64, 0.0f64);
            for j in 0..half {
                let k = start + j;
                let l = k + half;
                let (u_r, u_i) = (re_im[2 * k], re_im[2 * k + 1]);
                let (v_r, v_i) = (
                    re_im[2 * l] * w_r - re_im[2 * l + 1] * w_i,
                    re_im[2 * l] * w_i + re_im[2 * l + 1] * w_r,
                );
                re_im[2 * k] = u_r + v_r;
                re_im[2 * k + 1] = u_i + v_i;
                re_im[2 * l] = u_r - v_r;
                re_im[2 * l + 1] = u_i - v_i;
                let nw_r = w_r * wr - w_i * wi;
                w_i = w_r * wi + w_i * wr;
                w_r = nw_r;
            }
        }
        len <<= 1;
    }
}

const TORCH28_POVEY_WINDOW: [u32; FRAME_LEN] = [
    0x00000000, 0x398b01c2, 0x3a61d13c, 0x3ae0ed7f, 0x3b376219, 0x3b85f871, 0x3bb69cf0, 0x3bed450d,
    0x3c14d3ee, 0x3c35c461, 0x3c595962, 0x3c7f7c2a, 0x3c940c1c, 0x3ca98d87, 0x3cc039e2, 0x3cd8097f,
    0x3cf0f521, 0x3d057b01, 0x3d1302e1, 0x3d210f2b, 0x3d2f9d09, 0x3d3ea9ba, 0x3d4e3290, 0x3d5e3507,
    0x3d6eaeb6, 0x3d7f9d37, 0x3d887f1b, 0x3d9167b1, 0x3d9a8753, 0x3da3dce8, 0x3dad675c, 0x3db725aa,
    0x3dc116c9, 0x3dcb39bf, 0x3dd58d83, 0x3de01125, 0x3deac3a6, 0x3df5a410, 0x3e0058b9, 0x3e05f570,
    0x3e0ba7b2, 0x3e116f06, 0x3e174afd, 0x3e1d3b1c, 0x3e233eee, 0x3e295605, 0x3e2f7fe5, 0x3e35bc22,
    0x3e3c0a45, 0x3e4269db, 0x3e48da79, 0x3e4f5ba4, 0x3e55ecf1, 0x3e5c8ded, 0x3e633e23, 0x3e69fd2c,
    0x3e70ca8e, 0x3e77a5dc, 0x3e7e8ea9, 0x3e82c241, 0x3e86437a, 0x3e89cacb, 0x3e8d57fb, 0x3e90ead2,
    0x3e948318, 0x3e982096, 0x3e9bc315, 0x3e9f6a5e, 0x3ea31636, 0x3ea6c669, 0x3eaa7abf, 0x3eae3301,
    0x3eb1eef9, 0x3eb5ae6e, 0x3eb97129, 0x3ebd36f4, 0x3ec0ff9a, 0x3ec4cadf, 0x3ec89894, 0x3ecc687c,
    0x3ed03a62, 0x3ed40e11, 0x3ed7e352, 0x3edbb9f1, 0x3edf91b5, 0x3ee36a68, 0x3ee743d6, 0x3eeb1dc8,
    0x3eeef80a, 0x3ef2d267, 0x3ef6aca7, 0x3efa8697, 0x3efe6001, 0x3f011c58, 0x3f03083a, 0x3f04f389,
    0x3f06de2c, 0x3f08c80a, 0x3f0ab109, 0x3f0c990e, 0x3f0e8000, 0x3f1065c6, 0x3f124a44, 0x3f142d64,
    0x3f160f0b, 0x3f17ef21, 0x3f19cd8b, 0x3f1baa31, 0x3f1d84fa, 0x3f1f5dcf, 0x3f213498, 0x3f230939,
    0x3f24db9c, 0x3f26abaa, 0x3f287949, 0x3f2a4463, 0x3f2c0ce1, 0x3f2dd2a9, 0x3f2f95a5, 0x3f3155bf,
    0x3f3312de, 0x3f34ccef, 0x3f3683d8, 0x3f383783, 0x3f39e7dc, 0x3f3b94cb, 0x3f3d3e3c, 0x3f3ee418,
    0x3f40864a, 0x3f4224bc, 0x3f43bf5b, 0x3f455612, 0x3f46e8cb, 0x3f487773, 0x3f4a01f6, 0x3f4b883f,
    0x3f4d0a3b, 0x3f4e87d6, 0x3f5000fe, 0x3f51759f, 0x3f52e5a7, 0x3f545105, 0x3f55b7a5, 0x3f571974,
    0x3f587664, 0x3f59ce60, 0x3f5b2158, 0x3f5c6f3b, 0x3f5db7f8, 0x3f5efb80, 0x3f6039c2, 0x3f6172ae,
    0x3f62a635, 0x3f63d448, 0x3f64fcd6, 0x3f661fd2, 0x3f673d2e, 0x3f6854db, 0x3f6966ca, 0x3f6a72ee,
    0x3f6b793d, 0x3f6c79a5, 0x3f6d741c, 0x3f6e6896, 0x3f6f5705, 0x3f703f5f, 0x3f712198, 0x3f71fda3,
    0x3f72d378, 0x3f73a30c, 0x3f746c52, 0x3f752f43, 0x3f75ebd4, 0x3f76a1fc, 0x3f7751b2, 0x3f77faed,
    0x3f789da6, 0x3f7939d3, 0x3f79cf6e, 0x3f7a5e6f, 0x3f7ae6ce, 0x3f7b6886, 0x3f7be38f, 0x3f7c57e4,
    0x3f7cc57f, 0x3f7d2c5b, 0x3f7d8c72, 0x3f7de5c0, 0x3f7e383f, 0x3f7e83ee, 0x3f7ec8c7, 0x3f7f06c7,
    0x3f7f3deb, 0x3f7f6e31, 0x3f7f9795, 0x3f7fba16, 0x3f7fd5b4, 0x3f7fea6b, 0x3f7ff83b, 0x3f7fff23,
    0x3f7fff23, 0x3f7ff83b, 0x3f7fea6b, 0x3f7fd5b4, 0x3f7fba17, 0x3f7f9796, 0x3f7f6e31, 0x3f7f3deb,
    0x3f7f06c7, 0x3f7ec8c7, 0x3f7e83ee, 0x3f7e3840, 0x3f7de5c0, 0x3f7d8c72, 0x3f7d2c5b, 0x3f7cc57f,
    0x3f7c57e5, 0x3f7be390, 0x3f7b6887, 0x3f7ae6ce, 0x3f7a5e6f, 0x3f79cf6e, 0x3f7939d3, 0x3f789da7,
    0x3f77faee, 0x3f7751b2, 0x3f76a1fc, 0x3f75ebd5, 0x3f752f44, 0x3f746c54, 0x3f73a30c, 0x3f72d378,
    0x3f71fda4, 0x3f712199, 0x3f703f5f, 0x3f6f5706, 0x3f6e6896, 0x3f6d741e, 0x3f6c79a6, 0x3f6b793e,
    0x3f6a72f0, 0x3f6966ca, 0x3f6854db, 0x3f673d2f, 0x3f661fd3, 0x3f64fcd8, 0x3f63d448, 0x3f62a637,
    0x3f6172b0, 0x3f6039c3, 0x3f5efb82, 0x3f5db7f8, 0x3f5c6f3b, 0x3f5b2158, 0x3f59ce60, 0x3f587666,
    0x3f571976, 0x3f55b7a5, 0x3f545107, 0x3f52e5a8, 0x3f5175a1, 0x3f5000fe, 0x3f4e87d8, 0x3f4d0a3b,
    0x3f4b8840, 0x3f4a01f6, 0x3f487775, 0x3f46e8cd, 0x3f455612, 0x3f43bf5d, 0x3f4224bd, 0x3f40864c,
    0x3f3ee418, 0x3f3d3e3f, 0x3f3b94cc, 0x3f39e7df, 0x3f383785, 0x3f3683db, 0x3f34ccf1, 0x3f3312df,
    0x3f3155c0, 0x3f2f95a5, 0x3f2dd2ab, 0x3f2c0ce1, 0x3f2a4466, 0x3f28794b, 0x3f26abac, 0x3f24db9d,
    0x3f230939, 0x3f213499, 0x3f1f5dd0, 0x3f1d84fd, 0x3f1baa32, 0x3f19cd8d, 0x3f17ef21, 0x3f160f0e,
    0x3f142d65, 0x3f124a48, 0x3f1065c7, 0x3f0e8000, 0x3f0c9910, 0x3f0ab109, 0x3f08c80d, 0x3f06de2d,
    0x3f04f38c, 0x3f03083b, 0x3f011c5c, 0x3efe6004, 0x3efa869d, 0x3ef6acaa, 0x3ef2d266, 0x3eeef80f,
    0x3eeb1dc9, 0x3ee743db, 0x3ee36a6a, 0x3edf91ba, 0x3edbb9f3, 0x3ed7e35a, 0x3ed40e15, 0x3ed03a69,
    0x3ecc687f, 0x3ec89893, 0x3ec4cae5, 0x3ec0ff9a, 0x3ebd36fa, 0x3eb9712b, 0x3eb5ae73, 0x3eb1eefa,
    0x3eae3307, 0x3eaa7ac2, 0x3ea6c670, 0x3ea31639, 0x3e9f6a5c, 0x3e9bc319, 0x3e982097, 0x3e94831c,
    0x3e90ead3, 0x3e8d5800, 0x3e89cace, 0x3e864380, 0x3e82c243, 0x3e7e8eb6, 0x3e77a5e2, 0x3e70ca8e,
    0x3e69fd33, 0x3e633e25, 0x3e5c8df3, 0x3e55ecf3, 0x3e4f5bad, 0x3e48da7c, 0x3e4269e7, 0x3e3c0a49,
    0x3e35bc2e, 0x3e2f7fec, 0x3e295605, 0x3e233ef7, 0x3e1d3b1c, 0x3e174b04, 0x3e116f0b, 0x3e0ba7b9,
    0x3e05f575, 0x3e0058c3, 0x3df5a41a, 0x3deac3a6, 0x3de0112f, 0x3dd58d83, 0x3dcb39c9, 0x3dc116ce,
    0x3db725b9, 0x3dad6761, 0x3da3dcf8, 0x3d9a875d, 0x3d9167c1, 0x3d887f20, 0x3d7f9d37, 0x3d6eaecd,
    0x3d5e3513, 0x3d4e32a7, 0x3d3ea9ba, 0x3d2f9d20, 0x3d210f37, 0x3d1302fa, 0x3d057b0e, 0x3cf0f554,
    0x3cd80999, 0x3cc039e2, 0x3ca98da2, 0x3c940c1c, 0x3c7f7c63, 0x3c595962, 0x3c35c49d, 0x3c14d42d,
    0x3bed458f, 0x3bb69d78, 0x3b85f900, 0x3b376219, 0x3ae0ed7f, 0x3a61d13c, 0x398b01c2, 0x00000000,
];

fn povey_window() -> [f32; FRAME_LEN] {
    TORCH28_POVEY_WINDOW.map(f32::from_bits)
}

fn pocketfft_twiddle(index: usize) -> (f32, f32) {
    fn calculate(mut index: usize) -> (f64, f64) {
        const N: usize = FFT_SIZE;
        let angle = 0.25 * std::f64::consts::PI / N as f64;
        index <<= 3;
        if index < 4 * N {
            if index < 2 * N {
                if index < N {
                    let value = index as f64 * angle;
                    return (value.cos(), value.sin());
                }
                let value = (2 * N - index) as f64 * angle;
                return (value.sin(), value.cos());
            }
            index -= 2 * N;
            if index < N {
                let value = index as f64 * angle;
                return (-value.sin(), value.cos());
            }
            let value = (2 * N - index) as f64 * angle;
            return (-value.cos(), value.sin());
        }
        index = 8 * N - index;
        if index < 2 * N {
            if index < N {
                let value = index as f64 * angle;
                return (value.cos(), -value.sin());
            }
            let value = (2 * N - index) as f64 * angle;
            return (value.sin(), -value.cos());
        }
        index -= 2 * N;
        if index < N {
            let value = index as f64 * angle;
            return (-value.sin(), -value.cos());
        }
        let value = (2 * N - index) as f64 * angle;
        (-value.cos(), -value.sin())
    }

    const SHIFT: usize = 5;
    const MASK: usize = (1 << SHIFT) - 1;
    let mirrored = 2 * index > FFT_SIZE;
    let index = if mirrored { FFT_SIZE - index } else { index };
    let (x1r, x1i) = if index & MASK == 0 {
        (1.0, 0.0)
    } else {
        calculate(index & MASK)
    };
    let upper = index >> SHIFT;
    let (x2r, x2i) = if upper == 0 {
        (1.0, 0.0)
    } else {
        calculate(upper * (MASK + 1))
    };
    let real = (x1r * x2r - x1i * x2i) as f32;
    let imag = (x1r * x2i + x1i * x2r) as f32;
    (real, if mirrored { -imag } else { imag })
}

fn pocketfft_radf4(ido: usize, l1: usize, twiddle_stride: usize, cc: &[f32], ch: &mut [f32]) {
    let cc_index = |a: usize, b: usize, c: usize| a + ido * (b + l1 * c);
    let ch_index = |a: usize, b: usize, c: usize| a + ido * (b + 4 * c);
    for k in 0..l1 {
        let left = cc[cc_index(0, k, 3)];
        let right = cc[cc_index(0, k, 1)];
        let tr1 = left + right;
        ch[ch_index(0, 2, k)] = left - right;
        let left = cc[cc_index(0, k, 0)];
        let right = cc[cc_index(0, k, 2)];
        let tr2 = left + right;
        ch[ch_index(ido - 1, 1, k)] = left - right;
        ch[ch_index(0, 0, k)] = tr2 + tr1;
        ch[ch_index(ido - 1, 3, k)] = tr2 - tr1;
    }
    if ido & 1 == 0 {
        const HALF_SQRT_TWO: f32 = 0.707_106_77;
        for k in 0..l1 {
            let ti1 = -HALF_SQRT_TWO * (cc[cc_index(ido - 1, k, 1)] + cc[cc_index(ido - 1, k, 3)]);
            let tr1 = HALF_SQRT_TWO * (cc[cc_index(ido - 1, k, 1)] - cc[cc_index(ido - 1, k, 3)]);
            let left = cc[cc_index(ido - 1, k, 0)];
            ch[ch_index(ido - 1, 0, k)] = left + tr1;
            ch[ch_index(ido - 1, 2, k)] = left - tr1;
            let right = cc[cc_index(ido - 1, k, 2)];
            ch[ch_index(0, 3, k)] = ti1 + right;
            ch[ch_index(0, 1, k)] = ti1 - right;
        }
    }
    if ido <= 2 {
        return;
    }
    for k in 0..l1 {
        for i in (2..ido).step_by(2) {
            let inverse = ido - i;
            let multiply = |factor: usize, real: f32, imag: f32| {
                let (wr, wi) = pocketfft_twiddle(factor * twiddle_stride * (i / 2));
                (wr.mul_add(real, wi * imag), wr.mul_add(imag, -(wi * real)))
            };
            let (cr2, ci2) = multiply(1, cc[cc_index(i - 1, k, 1)], cc[cc_index(i, k, 1)]);
            let (cr3, ci3) = multiply(2, cc[cc_index(i - 1, k, 2)], cc[cc_index(i, k, 2)]);
            let (cr4, ci4) = multiply(3, cc[cc_index(i - 1, k, 3)], cc[cc_index(i, k, 3)]);
            let tr1 = cr4 + cr2;
            let tr4 = cr4 - cr2;
            let ti1 = ci2 + ci4;
            let ti4 = ci2 - ci4;
            let base_real = cc[cc_index(i - 1, k, 0)];
            let tr2 = base_real + cr3;
            let tr3 = base_real - cr3;
            let base_imag = cc[cc_index(i, k, 0)];
            let ti2 = base_imag + ci3;
            let ti3 = base_imag - ci3;
            ch[ch_index(i - 1, 0, k)] = tr2 + tr1;
            ch[ch_index(inverse - 1, 3, k)] = tr2 - tr1;
            ch[ch_index(i, 0, k)] = ti1 + ti2;
            ch[ch_index(inverse, 3, k)] = ti1 - ti2;
            ch[ch_index(i - 1, 2, k)] = tr3 + ti4;
            ch[ch_index(inverse - 1, 1, k)] = tr3 - ti4;
            ch[ch_index(i, 2, k)] = tr4 + ti3;
            ch[ch_index(inverse, 1, k)] = tr4 - ti3;
        }
    }
}

fn pocketfft_radf2(ido: usize, l1: usize, twiddle_stride: usize, cc: &[f32], ch: &mut [f32]) {
    let cc_index = |a: usize, b: usize, c: usize| a + ido * (b + l1 * c);
    let ch_index = |a: usize, b: usize, c: usize| a + ido * (b + 2 * c);
    for k in 0..l1 {
        let left = cc[cc_index(0, k, 0)];
        let right = cc[cc_index(0, k, 1)];
        ch[ch_index(0, 0, k)] = left + right;
        ch[ch_index(ido - 1, 1, k)] = left - right;
    }
    if ido & 1 == 0 {
        for k in 0..l1 {
            ch[ch_index(0, 1, k)] = -cc[cc_index(ido - 1, k, 1)];
            ch[ch_index(ido - 1, 0, k)] = cc[cc_index(ido - 1, k, 0)];
        }
    }
    if ido <= 2 {
        return;
    }
    for k in 0..l1 {
        for i in (2..ido).step_by(2) {
            let inverse = ido - i;
            let (wr, wi) = pocketfft_twiddle(twiddle_stride * (i / 2));
            let real = cc[cc_index(i - 1, k, 1)];
            let imag = cc[cc_index(i, k, 1)];
            let tr2 = wr.mul_add(real, wi * imag);
            let ti2 = wr.mul_add(imag, -(wi * real));
            let base_real = cc[cc_index(i - 1, k, 0)];
            ch[ch_index(i - 1, 0, k)] = base_real + tr2;
            ch[ch_index(inverse - 1, 1, k)] = base_real - tr2;
            let base_imag = cc[cc_index(i, k, 0)];
            ch[ch_index(i, 0, k)] = ti2 + base_imag;
            ch[ch_index(inverse, 1, k)] = ti2 - base_imag;
        }
    }
}

fn torch28_rfft_power_512(input: &[f32; FFT_SIZE]) -> ([f32; 257], [f32; 257], [f32; 257]) {
    let mut current = *input;
    let mut scratch = [0.0f32; FFT_SIZE];
    pocketfft_radf4(1, 128, 128, &current, &mut scratch);
    pocketfft_radf4(4, 32, 32, &scratch, &mut current);
    pocketfft_radf4(16, 8, 8, &current, &mut scratch);
    pocketfft_radf4(64, 2, 2, &scratch, &mut current);
    pocketfft_radf2(256, 1, 1, &current, &mut scratch);
    let mut real = [0.0f32; 257];
    let mut imag = [0.0f32; 257];
    let mut power = [0.0f32; 257];
    real[0] = scratch[0];
    real[256] = scratch[511];
    for bin in 1..256 {
        real[bin] = scratch[2 * bin - 1];
        imag[bin] = scratch[2 * bin];
    }
    for bin in 0..257 {
        let magnitude = f32::hypot(real[bin], imag[bin]);
        power[bin] = unsafe { powf(magnitude, 2.0) };
    }
    (real, imag, power)
}

/// Torch 2.8 ARM64 `sum_out(...).div_(400)` reduction order for one Kaldi frame.
fn torch28_arm_mean_400(frame: &[f32; FRAME_LEN]) -> f32 {
    let mut partial = [[0.0f32; 4]; 4];
    let mut cascade = [[0.0f32; 4]; 4];
    for group in 0..16 {
        for accumulator in 0..4 {
            let base = (4 * group + accumulator) * 4;
            for lane in 0..4 {
                partial[accumulator][lane] += frame[base + lane];
            }
        }
    }
    for accumulator in 0..4 {
        for lane in 0..4 {
            cascade[accumulator][lane] += partial[accumulator][lane];
            partial[accumulator][lane] = 0.0;
        }
    }
    for group in 16..25 {
        for accumulator in 0..4 {
            let base = (4 * group + accumulator) * 4;
            for lane in 0..4 {
                partial[accumulator][lane] += frame[base + lane];
            }
        }
    }
    for accumulator in 0..4 {
        for lane in 0..4 {
            partial[accumulator][lane] += cascade[accumulator][lane];
        }
    }
    for accumulator in 1..4 {
        for lane in 0..4 {
            partial[0][lane] += partial[accumulator][lane];
        }
    }
    (((partial[0][0] + partial[0][1]) + partial[0][2]) + partial[0][3]) / FRAME_LEN as f32
}

fn prepare_kaldi_frame(frame: &[f32; FRAME_LEN]) -> [f32; FRAME_LEN] {
    let mean = torch28_arm_mean_400(frame);
    let mut centered = std::array::from_fn(|index| frame[index] - mean);
    for index in (1..FRAME_LEN).rev() {
        centered[index] -= PREEMPH * centered[index - 1];
    }
    centered[0] -= PREEMPH * centered[0];
    centered
}

/// 16 kHz waveform → `[frames, 80]` log-mel (mean-normalized).
pub fn kaldi_fbank(waveform: &[f32]) -> Vec<f32> {
    let n = waveform.len();
    let n_frames = if n < FRAME_LEN {
        1
    } else {
        1 + (n - FRAME_LEN) / FRAME_SHIFT
    };
    let filterbank = melbank::torch28_mel_filterbank();
    let window = povey_window();
    let fft_bins = FFT_SIZE / 2 + 1;
    let mut power = vec![0.0f32; n_frames * fft_bins];
    for frame in 0..n_frames {
        let start = frame * FRAME_SHIFT;
        let raw_frame =
            std::array::from_fn(|index| waveform.get(start + index).copied().unwrap_or(0.0));
        let prepared = prepare_kaldi_frame(&raw_frame);
        let mut fft_input = [0.0f32; FFT_SIZE];
        for i in 0..FRAME_LEN {
            fft_input[i] = prepared[i] * window[i];
        }
        let (_, _, spectrum) = torch28_rfft_power_512(&fft_input);
        power[frame * fft_bins..(frame + 1) * fft_bins].copy_from_slice(&spectrum);
    }
    let mut features = mel_matmul(&power, &filterbank, n_frames);
    for value in &mut features {
        *value = log::torch28_log(value.max(MEL_FLOOR));
    }
    let means = torch28_column_mean(&features, n_frames, MEL_BINS);
    for (index, value) in features.iter_mut().enumerate() {
        *value -= means[index % MEL_BINS];
    }
    features
}

// ---------------------------------------------------------------------------
// CAM++ x-vector
// ---------------------------------------------------------------------------

pub struct BatchNorm {
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
    pub running_mean: Vec<f32>,
    pub running_var: Vec<f32>,
}

impl BatchNorm {
    /// BatchNorm with affine=False (only normalization).
    fn apply_no_affine(&self, x: &mut [f32]) {
        for c in 0..self.running_mean.len() {
            let inv = 1.0 / (self.running_var[c] + BN_EPS).sqrt();
            x[c] = (x[c] - self.running_mean[c]) * inv;
        }
    }
}

pub struct CamPlus {
    // FCM stem (fbank [80, T] → [320, T])
    pub head_conv1: Vec<f32>, // [32,1,3,3]
    pub head_bn1: BatchNorm,
    pub res_l1_0: ResBlock2d,
    pub res_l1_1: ResBlock2d,
    pub res_l2_0: ResBlock2d,
    pub res_l2_1: ResBlock2d,
    pub head_conv2: Vec<f32>, // [32,32,3,3] stride (2,1)
    pub head_bn2: BatchNorm,
    // TDNN
    pub tdnn_w: Vec<f32>, // [128,320,5]
    pub tdnn_bn: BatchNorm,
    // three dense blocks + transits
    pub blocks: Vec<DenseBlock>,
    pub transits: Vec<Transit>,
    pub out_bn: BatchNorm,
    pub dense_w: Vec<f32>, // [512,1024,1]
    pub dense_bn: BatchNorm,
}

pub struct ResBlock2d {
    pub conv1: Vec<f32>,
    pub bn1: BatchNorm,
    pub conv2: Vec<f32>,
    pub bn2: BatchNorm,
    pub shortcut: Option<(Vec<f32>, BatchNorm)>, // stride != 1
    pub stride: usize,
}

pub struct DenseBlock {
    pub layers: Vec<DenseLayer>,
}

pub struct DenseLayer {
    pub nl1: BatchNorm,      // bn(in) + relu
    pub linear1: Vec<f32>,   // [128, in, 1]
    pub nl2: BatchNorm,      // bn(128) + relu
    pub cam_local: Vec<f32>, // [32, 128, 3]
    pub cam_local_dilation: usize,
    pub cam_lin1: Vec<f32>, // [64, 128, 1]
    pub cam_lin1_bias: Vec<f32>,
    pub cam_lin2: Vec<f32>, // [32, 64, 1]
    pub cam_lin2_bias: Vec<f32>,
}

pub struct Transit {
    pub nl: BatchNorm,    // bn(in) + relu
    pub linear: Vec<f32>, // [out, in, 1]
}

impl CamPlus {
    pub fn from_source(source: &dyn TensorSource) -> Result<Self, String> {
        let s = |name: &str, dims: &[u64]| -> Result<Vec<f32>, String> {
            load_f16_f32(source, name, dims)
        };
        let bn = |prefix: &str, channels: usize| -> Result<BatchNorm, String> {
            Ok(BatchNorm {
                weight: s(&format!("{prefix}.weight"), &[channels as u64])?,
                bias: s(&format!("{prefix}.bias"), &[channels as u64])?,
                running_mean: s(&format!("{prefix}.running_mean"), &[channels as u64])?,
                running_var: s(&format!("{prefix}.running_var"), &[channels as u64])?,
            })
        };
        let head_conv1 = s("dotstts.speaker.head.conv1.weight", &[3, 3, 1, 32])?;
        let head_bn1 = bn("dotstts.speaker.head.bn1", 32)?;
        let res = |block: &str, stride: usize| -> Result<ResBlock2d, String> {
            let prefix = format!("dotstts.speaker.head.{block}");
            let conv1 = s(&format!("{prefix}.conv1.weight"), &[3, 3, 32, 32])?;
            let bn1 = bn(&format!("{prefix}.bn1"), 32)?;
            let conv2 = s(&format!("{prefix}.conv2.weight"), &[3, 3, 32, 32])?;
            let bn2 = bn(&format!("{prefix}.bn2"), 32)?;
            let shortcut = if stride != 1 {
                Some((
                    s(&format!("{prefix}.shortcut.0.weight"), &[1, 1, 32, 32])?,
                    bn(&format!("{prefix}.shortcut.1"), 32)?,
                ))
            } else {
                None
            };
            Ok(ResBlock2d {
                conv1,
                bn1,
                conv2,
                bn2,
                shortcut,
                stride,
            })
        };
        let res_l1_0 = res("layer1.0", 2)?;
        let res_l1_1 = res("layer1.1", 1)?;
        let res_l2_0 = res("layer2.0", 2)?;
        let res_l2_1 = res("layer2.1", 1)?;
        let head_conv2 = s("dotstts.speaker.head.conv2.weight", &[3, 3, 32, 32])?;
        let head_bn2 = bn("dotstts.speaker.head.bn2", 32)?;
        let tdnn_w = s("dotstts.speaker.xvector.tdnn.linear.weight", &[5, 320, 128])?;
        let tdnn_bn = bn("dotstts.speaker.xvector.tdnn.nonlinear.batchnorm", 128)?;

        let mut blocks = Vec::new();
        let mut transits = Vec::new();
        let mut channels = 128usize;
        for (bi, (num_layers, dilation)) in [(12usize, 1usize), (24, 2), (16, 2)].iter().enumerate()
        {
            let mut layers = Vec::new();
            for layer in 0..*num_layers {
                let dil = *dilation;
                let in_ch = channels + layer * 32;
                let prefix = format!("dotstts.speaker.xvector.block{}.tdnnd{}", bi + 1, layer + 1);
                let linear1 = s(&format!("{prefix}.linear1.weight"), &[1, in_ch as u64, 128])?;
                let nl1 = bn(&format!("{prefix}.nonlinear1.batchnorm"), in_ch)?;
                let nl2 = bn(&format!("{prefix}.nonlinear2.batchnorm"), 128)?;
                let cam_local = s(
                    &format!("{prefix}.cam_layer.linear_local.weight"),
                    &[3, 128, 32],
                )?;
                let cam_lin1 = s(&format!("{prefix}.cam_layer.linear1.weight"), &[1, 128, 64])?;
                let cam_lin1_bias = s(&format!("{prefix}.cam_layer.linear1.bias"), &[64])?;
                let cam_lin2 = s(&format!("{prefix}.cam_layer.linear2.weight"), &[1, 64, 32])?;
                let cam_lin2_bias = s(&format!("{prefix}.cam_layer.linear2.bias"), &[32])?;
                layers.push(DenseLayer {
                    nl1,
                    linear1,
                    nl2,
                    cam_local,
                    cam_local_dilation: dil,
                    cam_lin1,
                    cam_lin1_bias,
                    cam_lin2,
                    cam_lin2_bias,
                });
            }
            blocks.push(DenseBlock { layers });
            channels += num_layers * 32;
            let nl = bn(
                &format!(
                    "dotstts.speaker.xvector.transit{}.nonlinear.batchnorm",
                    bi + 1
                ),
                channels,
            )?;
            let linear = s(
                &format!("dotstts.speaker.xvector.transit{}.linear.weight", bi + 1),
                &[1, channels as u64, (channels / 2) as u64],
            )?;
            transits.push(Transit { nl, linear });
            channels /= 2;
        }
        let out_bn = bn("dotstts.speaker.xvector.out_nonlinear.batchnorm", channels)?;
        let dense_w = s(
            "dotstts.speaker.xvector.dense.linear.weight",
            &[1, (channels * 2) as u64, 512],
        )?;
        let dense_bn = BatchNorm {
            weight: vec![0.0; 512],
            bias: vec![0.0; 512],
            running_mean: s(
                "dotstts.speaker.xvector.dense.nonlinear.batchnorm.running_mean",
                &[512],
            )?,
            running_var: s(
                "dotstts.speaker.xvector.dense.nonlinear.batchnorm.running_var",
                &[512],
            )?,
        };
        Ok(Self {
            head_conv1,
            head_bn1,
            res_l1_0,
            res_l1_1,
            res_l2_0,
            res_l2_1,
            head_conv2,
            head_bn2,
            tdnn_w,
            tdnn_bn,
            blocks,
            transits,
            out_bn,
            dense_w,
            dense_bn,
        })
    }

    /// Encode mel frames `[frames, 80]` → 512-dim x-vector.
    pub fn encode(&self, mel: &[f32]) -> Result<Vec<f32>, String> {
        let frames = mel.len() / MEL_BINS;
        if frames == 0 {
            return Err("speaker encoder needs at least one frame".into());
        }
        // FCM: [80, T] → [320, T]
        let mut x = self.fcm(mel, frames);
        // TDNN: [320, T] → [128, T2]
        let t2 = conv1d_length(frames, 5, 2, 2);
        x = self.tdnn(&x, frames, t2);
        let t = t2;
        // dense blocks + transits
        for (block, transit) in self.blocks.iter().zip(self.transits.iter()) {
            let mut channels = 0usize;
            for (layer_idx, layer) in block.layers.iter().enumerate() {
                if layer_idx == 0 {
                    channels = match self.blocks.iter().position(|b| std::ptr::eq(b, block)) {
                        Some(0) => 128usize,
                        Some(1) => 256usize,
                        _ => 512usize,
                    };
                }
                let in_ch = channels + layer_idx * 32;
                x = dense_layer_forward(layer, &x, t, in_ch);
            }
            let channels_after = channels + block.layers.len() * 32;
            let mut trans = vec![0.0f32; t * (channels_after / 2)];
            transit_forward(transit, &x, t, channels_after, &mut trans);
            x = trans;
        }
        // out_nonlinear: bn(512) + relu
        for c in 0..512 {
            let (scale, bias) = torch28_batch_norm_terms(
                self.out_bn.weight[c],
                self.out_bn.bias[c],
                self.out_bn.running_mean[c],
                self.out_bn.running_var[c],
            );
            for j in 0..t {
                let idx = j * 512 + c;
                x[idx] = (x[idx] * scale + bias).max(0.0);
            }
        }
        // masked stats pooling (all frames valid, unbiased std, floor 1e-2)
        let stats = cam_stats_pooling(&x, t, 512);
        // dense: [1024, 1] → 512 + batchnorm_ (affine=False)
        Ok(self.dense_projection(&stats))
    }

    fn dense_projection(&self, stats: &[f32]) -> Vec<f32> {
        debug_assert_eq!(stats.len(), 1024);
        let mut dense = vec![0.0f32; 512];
        #[cfg(any(
            target_os = "macos",
            all(feature = "openblas", target_os = "linux", target_arch = "x86_64"),
        ))]
        unsafe {
            const CBLAS_ROW_MAJOR: i32 = 101;
            const CBLAS_NO_TRANSPOSE: i32 = 111;
            sys::cblas_sgemm(
                CBLAS_ROW_MAJOR,
                CBLAS_NO_TRANSPOSE,
                CBLAS_NO_TRANSPOSE,
                512,
                1,
                1024,
                1.0,
                self.dense_w.as_ptr(),
                1024,
                stats.as_ptr(),
                1,
                0.0,
                dense.as_mut_ptr(),
                1,
            );
        }
        #[cfg(not(any(
            target_os = "macos",
            all(feature = "openblas", target_os = "linux", target_arch = "x86_64"),
        )))]
        for output in 0..512 {
            for input in 0..1024 {
                dense[output] =
                    self.dense_w[output * 1024 + input].mul_add(stats[input], dense[output]);
            }
        }

        let mut final_output = vec![0.0f32; 512];
        for channel in 0..512 {
            let alpha = 1.0 / (self.dense_bn.running_var[channel] + BN_EPS).sqrt();
            let beta = (-self.dense_bn.running_mean[channel]).mul_add(alpha, 0.0);
            final_output[channel] = dense[channel] * alpha + beta;
        }
        final_output
    }

    fn fcm(&self, mel: &[f32], frames: usize) -> Vec<f32> {
        // conv2d [1→32] on [80, T]
        let input = fcm_input_layout(mel, frames, MEL_BINS);
        let mut x = conv2d_forward(
            &self.head_conv1,
            None,
            &input,
            frames,
            80,
            1,
            32,
            3,
            3,
            1,
            1,
        );
        self.apply_bn2d(&self.head_bn1, &mut x, 32, 80, frames);
        relu_inplace(&mut x);
        x = self.resblock(&self.res_l1_0, &x, 32, 80, frames);
        x = self.resblock(&self.res_l1_1, &x, 32, 40, frames);
        x = self.resblock(&self.res_l2_0, &x, 32, 40, frames);
        x = self.resblock(&self.res_l2_1, &x, 32, 20, frames);
        // conv2 stride (2,1)
        let mut y = conv2d_forward_stride(
            &self.head_conv2,
            None,
            &x,
            frames,
            20,
            32,
            32,
            3,
            3,
            2,
            1,
            1,
            1,
        );
        self.apply_bn2d(&self.head_bn2, &mut y, 32, 10, frames);
        relu_inplace(&mut y);
        // Natural FCM [32,10,T] storage -> Rust TDNN [T,320] storage.
        fcm_output_layout(&y, frames, 32, 10)
    }

    fn resblock(
        &self,
        block: &ResBlock2d,
        x: &[f32],
        channels: usize,
        h: usize,
        t: usize,
    ) -> Vec<f32> {
        let h_out = if block.stride == 1 { h } else { h / 2 };
        let mut out = if block.stride == 1 {
            conv2d_forward(&block.conv1, None, x, t, h, channels, channels, 3, 3, 1, 1)
        } else {
            conv2d_forward_stride(
                &block.conv1,
                None,
                x,
                t,
                h,
                channels,
                channels,
                3,
                3,
                2,
                1,
                1,
                1,
            )
        };
        self.apply_bn2d(&block.bn1, &mut out, channels, h_out, t);
        relu_inplace(&mut out);
        out = conv2d_forward(
            &block.conv2,
            None,
            &out,
            t,
            h_out,
            channels,
            channels,
            3,
            3,
            1,
            1,
        );
        self.apply_bn2d(&block.bn2, &mut out, channels, h_out, t);
        let residual = match &block.shortcut {
            Some((w, bn_out)) => {
                let mut sc =
                    conv2d_forward_stride(w, None, x, t, h, channels, channels, 1, 1, 2, 1, 0, 0);
                self.apply_bn2d(bn_out, &mut sc, channels, h_out, t);
                sc
            }
            None => x.to_vec(),
        };
        for (o, &r) in out.iter_mut().zip(residual.iter()) {
            *o += r;
        }
        relu_inplace(&mut out);
        out
    }

    fn apply_bn2d(&self, bn: &BatchNorm, x: &mut [f32], channels: usize, h: usize, t: usize) {
        for c in 0..channels {
            let (scale, bias) = torch28_batch_norm_terms(
                bn.weight[c],
                bn.bias[c],
                bn.running_mean[c],
                bn.running_var[c],
            );
            for pos in 0..h * t {
                let idx = c * h * t + pos;
                x[idx] = x[idx] * scale + bias;
            }
        }
    }

    fn tdnn(&self, x: &[f32], t: usize, t2: usize) -> Vec<f32> {
        let mut out = conv1d_time_major(&self.tdnn_w, x, t, t2, 320, 128, 5, 2, 2, 1);
        for c in 0..128 {
            let (scale, bias) = torch28_batch_norm_terms(
                self.tdnn_bn.weight[c],
                self.tdnn_bn.bias[c],
                self.tdnn_bn.running_mean[c],
                self.tdnn_bn.running_var[c],
            );
            for j in 0..t2 {
                let idx = j * 128 + c;
                out[idx] = out[idx] * scale + bias;
            }
        }
        for value in out.iter_mut() {
            *value = value.max(0.0);
        }
        out
    }
}

fn cam_stats_pooling(input: &[f32], time: usize, channels: usize) -> Vec<f32> {
    debug_assert_eq!(input.len(), time * channels);
    let mut stats = vec![0.0f32; channels * 2];
    let mut row = vec![0.0f32; time];
    let mut squared_centered = vec![0.0f32; time];
    for channel in 0..channels {
        for frame in 0..time {
            row[frame] = input[frame * channels + channel];
        }
        let mean = torch28_contiguous_mean(&row);
        for frame in 0..time {
            let centered = row[frame] - mean;
            squared_centered[frame] = centered * centered;
        }
        let variance = torch28_contiguous_sum(&squared_centered) / (time - 1).max(1) as f32;
        stats[channel] = mean;
        stats[channels + channel] = variance.max(1e-2).sqrt();
    }
    stats
}

fn conv1d_length(t: usize, kernel: usize, stride: usize, pad: usize) -> usize {
    (t + 2 * pad).saturating_sub(kernel) / stride + 1
}

fn relu_inplace(x: &mut [f32]) {
    for value in x.iter_mut() {
        *value = value.max(0.0);
    }
}

fn fcm_input_layout(input: &[f32], frames: usize, features: usize) -> Vec<f32> {
    debug_assert_eq!(input.len(), frames * features);
    let mut output = vec![0.0f32; input.len()];
    for frame in 0..frames {
        for feature in 0..features {
            output[feature * frames + frame] = input[frame * features + feature];
        }
    }
    output
}

fn fcm_output_layout(input: &[f32], frames: usize, channels: usize, height: usize) -> Vec<f32> {
    debug_assert_eq!(input.len(), frames * channels * height);
    let mut output = vec![0.0f32; input.len()];
    for frame in 0..frames {
        for channel in 0..channels {
            for row in 0..height {
                output[frame * channels * height + channel * height + row] =
                    input[channel * height * frames + row * frames + frame];
            }
        }
    }
    output
}

fn torch28_batch_norm_terms(
    weight: f32,
    bias: f32,
    running_mean: f32,
    running_var: f32,
) -> (f32, f32) {
    let inverse_std = 1.0 / (running_var + BN_EPS).sqrt();
    let alpha = inverse_std * weight;
    let beta = (-running_mean).mul_add(alpha, bias);
    (alpha, beta)
}

#[allow(clippy::too_many_arguments)]
fn conv1d_time_major(
    weight: &[f32],
    input: &[f32],
    input_time: usize,
    output_time: usize,
    input_channels: usize,
    output_channels: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; output_time * output_channels];
    for output_channel in 0..output_channels {
        for time in 0..output_time {
            let mut accumulator = 0.0f32;
            for input_channel in 0..input_channels {
                for tap in 0..kernel {
                    let source = time as isize * stride as isize + tap as isize * dilation as isize
                        - padding as isize;
                    if (0..input_time as isize).contains(&source) {
                        accumulator = weight[output_channel * input_channels * kernel
                            + input_channel * kernel
                            + tap]
                            .mul_add(
                                input[source as usize * input_channels + input_channel],
                                accumulator,
                            );
                    }
                }
            }
            output[time * output_channels + output_channel] = accumulator;
        }
    }
    output
}

/// Conv2d over a `[channels, h, t]` layout, stride (1,1), pad 1.
/// `weight` gguf-dims `[kw, kh, in, out]`.
fn conv2d_forward(
    weight: &[f32],
    bias: Option<&[f32]>,
    x: &[f32],
    t: usize,
    h: usize,
    in_ch: usize,
    out_ch: usize,
    _kw: usize,
    _kh: usize,
    sh: usize,
    sw: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; out_ch * h * t];
    for o in 0..out_ch {
        let b = bias.map_or(0.0, |b| b[o]);
        for i in 0..in_ch {
            for kh in 0..3usize {
                for kw in 0..3usize {
                    let w = weight[o * in_ch * 9 + i * 9 + kh * 3 + kw];
                    if w == 0.0 {
                        continue;
                    }
                    for hp in 0..h {
                        let src_h = hp as isize * sh as isize + kh as isize - 1;
                        for tp in 0..t {
                            let src_t = tp as isize * sw as isize + kw as isize - 1;
                            if src_h < 0 || src_t < 0 || src_h >= h as isize || src_t >= t as isize
                            {
                                continue;
                            }
                            let index = o * h * t + hp * t + tp;
                            out[index] = w.mul_add(
                                x[i * h * t + src_h as usize * t + src_t as usize],
                                out[index],
                            );
                        }
                    }
                }
            }
        }
        for pos in 0..h * t {
            out[o * h * t + pos] += b;
        }
    }
    out
}

/// Conv2d over a `[channels, h, t]` layout with explicit kernel/pad/stride.
/// `weight` gguf-dims `[kw, kh, in, out]`; pad is symmetric (ph top/bottom,
/// pw left/right), out h = saturating formula, stride may differ per axis.
fn conv2d_forward_stride(
    weight: &[f32],
    bias: Option<&[f32]>,
    x: &[f32],
    t: usize,
    h_in: usize,
    in_ch: usize,
    out_ch: usize,
    kw: usize,
    kh: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
) -> Vec<f32> {
    let h_out = (h_in + 2 * ph).saturating_sub(kh) / sh + 1;
    let mut out = vec![0.0f32; out_ch * h_out * t];
    for o in 0..out_ch {
        let b = bias.map_or(0.0, |b| b[o]);
        for i in 0..in_ch {
            for kk_h in 0..kh {
                for kk_w in 0..kw {
                    let w = weight[o * in_ch * kh * kw + i * kh * kw + kk_h * kw + kk_w];
                    if w == 0.0 {
                        continue;
                    }
                    for hp in 0..h_out {
                        let src_h = hp as isize * sh as isize + kk_h as isize - ph as isize;
                        if src_h < 0 || src_h >= h_in as isize {
                            continue;
                        }
                        for tp in 0..t {
                            let src_t = tp as isize * sw as isize + kk_w as isize - pw as isize;
                            if src_t < 0 || src_t >= t as isize {
                                continue;
                            }
                            let index = o * h_out * t + hp * t + tp;
                            out[index] = w.mul_add(
                                x[i * h_in * t + src_h as usize * t + src_t as usize],
                                out[index],
                            );
                        }
                    }
                }
            }
        }
        for pos in 0..h_out * t {
            out[o * h_out * t + pos] += b;
        }
    }
    out
}

/// One dense-layer step: [in, T] → [in+32, T] (concat), with the CAM
/// attention gate (reference `CAMDenseTDNNLayer` + `CAMLayer`).
struct DenseLayerFront {
    nonlinear2: Vec<f32>,
    local: Vec<f32>,
    context: Vec<f32>,
}

struct CamGate {
    linear1: Vec<f32>,
    relu1: Vec<f32>,
    linear2: Vec<f32>,
    sigmoid: Vec<f32>,
    gated: Vec<f32>,
}

fn cam_gate(layer: &DenseLayer, local: &[f32], context: &[f32], time: usize) -> CamGate {
    let mut linear1 = cam_gate_linear(
        &layer.cam_lin1,
        &layer.cam_lin1_bias,
        context,
        time,
        128,
        64,
    );
    let mut relu1 = linear1.clone();
    relu_inplace(&mut relu1);
    let relu1_time_major = channel_major_to_time_major(&relu1, time, 64);
    let linear2 = cam_gate_linear(
        &layer.cam_lin2,
        &layer.cam_lin2_bias,
        &relu1_time_major,
        time,
        64,
        32,
    );
    let mut sigmoid = linear2.clone();
    for value in &mut sigmoid {
        *value = exp::torch28_sigmoid(*value);
    }
    let mut gated = local.to_vec();
    for channel in 0..32 {
        for frame in 0..time {
            gated[frame * 32 + channel] *= sigmoid[channel * time + frame];
        }
    }
    CamGate {
        linear1: std::mem::take(&mut linear1),
        relu1,
        linear2,
        sigmoid,
        gated,
    }
}

fn dense_layer_front(layer: &DenseLayer, x: &[f32], t: usize, in_ch: usize) -> DenseLayerFront {
    // nonlinear1 (bn(in) + relu)
    let mut h = vec![0.0f32; in_ch * t];
    for c in 0..in_ch {
        let (scale, bias) = torch28_batch_norm_terms(
            layer.nl1.weight[c],
            layer.nl1.bias[c],
            layer.nl1.running_mean[c],
            layer.nl1.running_var[c],
        );
        for j in 0..t {
            let idx = j * in_ch + c;
            h[idx] = (x[j * in_ch + c] * scale + bias).max(0.0);
        }
    }
    // linear1: 1x1 conv [in → 128]
    let mut bn_in = conv1d_time_major(&layer.linear1, &h, t, t, in_ch, 128, 1, 1, 0, 1);
    // nonlinear2 (bn(128) + relu)
    for c in 0..128 {
        let (scale, bias) = torch28_batch_norm_terms(
            layer.nl2.weight[c],
            layer.nl2.bias[c],
            layer.nl2.running_mean[c],
            layer.nl2.running_var[c],
        );
        for j in 0..t {
            let idx = j * 128 + c;
            bn_in[idx] = (bn_in[idx] * scale + bias).max(0.0);
        }
    }
    // cam_layer: linear_local 3x1 with dilation d (pad = d)
    let d = layer.cam_local_dilation;
    let local = conv1d_time_major(&layer.cam_local, &bn_in, t, t, 128, 32, 3, 1, d, d);
    // attention gate: context = seg-pooled + global mean → 1x1 convs → sigmoid
    let context = cam_context(&bn_in, t, 128);
    DenseLayerFront {
        nonlinear2: bn_in,
        local,
        context,
    }
}

fn dense_layer_forward(layer: &DenseLayer, x: &[f32], t: usize, in_ch: usize) -> Vec<f32> {
    let front = dense_layer_front(layer, x, t, in_ch);
    let out = cam_gate(layer, &front.local, &front.context, t).gated;
    // concat: [in + 32, T]
    let mut combined = vec![0.0f32; (in_ch + 32) * t];
    for j in 0..t {
        combined[j * (in_ch + 32)..j * (in_ch + 32) + in_ch]
            .copy_from_slice(&x[j * in_ch..(j + 1) * in_ch]);
        combined[j * (in_ch + 32) + in_ch..(j + 1) * (in_ch + 32)]
            .copy_from_slice(&out[j * 32..(j + 1) * 32]);
    }
    combined
}

fn channel_major_to_time_major(input: &[f32], time: usize, channels: usize) -> Vec<f32> {
    debug_assert_eq!(input.len(), time * channels);
    let mut output = vec![0.0f32; input.len()];
    for frame in 0..time {
        for channel in 0..channels {
            output[frame * channels + channel] = input[channel * time + frame];
        }
    }
    output
}

/// CAM's two biased 1x1 convolutions use the pinned Torch macOS Slow2d path:
/// bias-prefilled row-major Accelerate SGEMM. Inputs are time-major `[T, in]`;
/// outputs stay channel-major `[out, T]` for the following gate operation.
fn cam_gate_linear(
    weight: &[f32],
    bias: &[f32],
    input: &[f32],
    time: usize,
    in_channels: usize,
    out_channels: usize,
) -> Vec<f32> {
    debug_assert_eq!(weight.len(), out_channels * in_channels);
    debug_assert_eq!(bias.len(), out_channels);
    debug_assert_eq!(input.len(), time * in_channels);

    let mut output = vec![0.0f32; out_channels * time];
    for out_channel in 0..out_channels {
        output[out_channel * time..(out_channel + 1) * time].fill(bias[out_channel]);
    }

    #[cfg(any(
        target_os = "macos",
        all(feature = "openblas", target_os = "linux", target_arch = "x86_64"),
    ))]
    {
        let mut channel_major = vec![0.0f32; input.len()];
        for frame in 0..time {
            for channel in 0..in_channels {
                channel_major[channel * time + frame] = input[frame * in_channels + channel];
            }
        }
        const CBLAS_ROW_MAJOR: i32 = 101;
        const CBLAS_NO_TRANSPOSE: i32 = 111;
        unsafe {
            sys::cblas_sgemm(
                CBLAS_ROW_MAJOR,
                CBLAS_NO_TRANSPOSE,
                CBLAS_NO_TRANSPOSE,
                out_channels as i32,
                time as i32,
                in_channels as i32,
                1.0,
                weight.as_ptr(),
                in_channels as i32,
                channel_major.as_ptr(),
                time as i32,
                1.0,
                output.as_mut_ptr(),
                time as i32,
            );
        }
    }

    #[cfg(not(any(
        target_os = "macos",
        all(feature = "openblas", target_os = "linux", target_arch = "x86_64"),
    )))]
    {
        for out_channel in 0..out_channels {
            for frame in 0..time {
                let index = out_channel * time + frame;
                for in_channel in 0..in_channels {
                    output[index] = weight[out_channel * in_channels + in_channel]
                        .mul_add(input[frame * in_channels + in_channel], output[index]);
                }
            }
        }
    }

    output
}

fn transit_forward(transit: &Transit, x: &[f32], t: usize, in_ch: usize, out: &mut [f32]) {
    let out_ch = in_ch / 2;
    // bn + relu
    let mut h = vec![0.0f32; in_ch * t];
    for c in 0..in_ch {
        let (scale, bias) = torch28_batch_norm_terms(
            transit.nl.weight[c],
            transit.nl.bias[c],
            transit.nl.running_mean[c],
            transit.nl.running_var[c],
        );
        for j in 0..t {
            let idx = j * in_ch + c;
            h[idx] = (x[idx] * scale + bias).max(0.0);
        }
    }
    out.copy_from_slice(&conv1d_time_major(
        &transit.linear,
        &h,
        t,
        t,
        in_ch,
        out_ch,
        1,
        1,
        0,
        1,
    ));
}

#[cfg(test)]
mod tests {
    use super::Resampler;

    #[test]
    fn sinc_resampler_keeps_the_fractional_final_phase() {
        let output = Resampler::new(3, 2).resample(&[1.0, 0.0, 0.0, 0.0]);
        assert_eq!(output.len(), 3);
    }
}
