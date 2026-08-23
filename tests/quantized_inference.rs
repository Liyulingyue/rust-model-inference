use rust_model_inference::ops::kernel::q4_0::Q4_0Kernel;
use rust_model_inference::ops::kernel::q4_1::Q4_1Kernel;
use rust_model_inference::ops::kernel::q4_k::{Q4_KKernel, Q4_KWeight};
use rust_model_inference::ops::kernel::q6_k::Q6_KKernel;
use rust_model_inference::ops::kernel::Kernel;

#[cfg(feature = "parity-trace")]
#[test]
fn parity_dot_uses_llama_scalar_double_accumulator() {
    let left = [100_000_000.0f32, 1.0, -100_000_000.0];
    let right = [1.0f32; 3];

    assert_eq!(
        rust_model_inference::ops::dot_f32(&left, &right, 3).to_bits(),
        1.0f32.to_bits()
    );
}

#[cfg(feature = "parity-trace")]
#[test]
fn parity_silu_mul_uses_llama_scalar_order() {
    let gate = [0.0, 0.0, 0.0, f32::from_bits(0xbf44_c6b5)];
    let mut up = [0.0, 0.0, 0.0, f32::from_bits(0x3ddf_57e5)];

    rust_model_inference::ops::silu_mul_inplace(&gate, &mut up);

    assert_eq!(up[3].to_bits(), 0xbcd9_8667);
}

#[test]
fn q4_0_dot_scales_in_llama_scalar_order() {
    let mut weight = vec![0x80u8; 18];
    weight[..2].copy_from_slice(&half::f16::from_f32(-3.76953125).to_bits().to_le_bytes());
    weight[2..9].fill(0x00);
    weight[9] = 0x70;
    let mut input_q8 = vec![0u8; 32];
    input_q8[..22].fill(127);
    input_q8[22] = 94;
    input_q8[23] = 1;
    let mut output = [0.0f32];

    Q4_0Kernel::new(&weight).forward_prequantized(
        &input_q8,
        &[3.439453125],
        &mut output,
        32,
        1,
        0,
        1,
    );

    assert_eq!(output[0].to_bits(), 0x4892_44e7);
}

#[cfg(feature = "parity-trace")]
#[test]
fn parity_q8_0_uses_llama_scalar_rounding() {
    let mut input = vec![0.0f32; 32];
    input[0] = 127.0;
    input[1] = 0.5;
    input[2] = -0.5;

    let (quantized, _) = rust_model_inference::ops::quantize_q8_0(&input, input.len());

    assert_eq!(&quantized[..3], &[127, 1, (-1i8) as u8]);
}

#[test]
fn q4_1_prepared_path_uses_llama_q8_1_sum_scale() {
    let mut weight = vec![0u8; 20];
    weight[2..4].copy_from_slice(&half::f16::from_f32(1.0).to_bits().to_le_bytes());
    let input = [1.0f32; 32];
    let input_q8 = [127u8; 32];
    let input_scale = half::f16::from_f32(1.0 / 127.0).to_f32();
    let mut output = [0.0f32];

    Q4_1Kernel::new(&weight).forward_prepared(
        &input,
        &input_q8,
        &[input_scale],
        &mut output,
        32,
        1,
        0,
        1,
    );

    assert_eq!(output[0].to_bits(), 32.0f32.to_bits());
}

#[cfg(feature = "parity-trace")]
#[test]
fn parity_rope_neox_uses_llama_scalar_non_fused_rotation() {
    let mut values = [0.0f32; 128];
    values[0] = f32::from_bits(0x3f24_bed8);
    values[64] = f32::from_bits(0xbd9d_4940);

    rust_model_inference::ops::rope_neox(&mut values, 1, 128, 1_000_000.0);

    assert_eq!(values[0].to_bits(), 0x3ed3_1cd6);
}

#[cfg(all(feature = "parity-trace", target_os = "macos"))]
#[test]
fn parity_rope_neox_uses_llama_combined_sincos() {
    let mut values = [0.0f32; 128];
    values[14] = f32::from_bits(0xbfcd_db0e);
    values[78] = f32::from_bits(0xc038_7005);

    rust_model_inference::ops::rope_neox(&mut values, 6, 128, 1_000_000.0);

    assert_eq!(
        [values[14], values[78]].map(f32::to_bits),
        [0xbf35_c294, 0xc04e_44a7]
    );
}

#[cfg(feature = "parity-trace")]
#[test]
fn parity_softmax_uses_llama_scalar_exp_and_double_sum() {
    let mut values = [0x4100_5f5f, 0x40e9_d754, 0x411f_b16f, 0x4110_0e1d].map(f32::from_bits);

    rust_model_inference::ops::softmax_inplace(&mut values);

    assert_eq!(
        values.map(f32::to_bits),
        [0x3db6_4746, 0x3d32_346f, 0x3f21_5bc5, 0x3e72_e02c]
    );
}

#[test]
fn q6_k_embedding_matches_canonical_row_decoder_bit_for_bit() {
    let mut rows = vec![0u8; 2 * rust_model_inference::ops::quant::BLOCK_Q6K_SIZE];
    for (index, byte) in rows.iter_mut().enumerate() {
        *byte = index.wrapping_mul(37).wrapping_add(11) as u8;
    }
    rows[208..210].copy_from_slice(&half::f16::from_f32(0.03125).to_bits().to_le_bytes());
    rows[418..420].copy_from_slice(&half::f16::from_f32(-0.0625).to_bits().to_le_bytes());

    let mut expected = [0.0f32; 256];
    rust_model_inference::ops::quant::dequantize_row_q6_k(&rows[210..420], &mut expected);
    let mut actual = [0.0f32; 256];
    rust_model_inference::ops::embedding::embedding_lookup_q6_k(&rows, 1, 256, &mut actual);

    assert_eq!(
        actual.map(f32::to_bits),
        expected.map(f32::to_bits),
        "Q6_K embedding must use the canonical ql/qh/scale layout"
    );
}

#[test]
fn q8_k_quantizer_returns_one_complete_block() {
    let input: Vec<f32> = (0..256)
        .map(|index| ((index as i32 - 127) as f32) * 0.03125)
        .collect();
    let blocks = rust_model_inference::ops::quant::quantize_row_q8_k(&input);

    assert_eq!(blocks.len(), 1);
    assert_eq!(
        blocks[0]
            .bsums
            .iter()
            .map(|&value| i32::from(value))
            .sum::<i32>(),
        blocks[0]
            .qs
            .iter()
            .map(|&value| i32::from(value))
            .sum::<i32>()
    );
}

#[test]
fn q8_k_quantizer_uses_llama_nearest_even_rounding() {
    let mut input = vec![0.0f32; 256];
    input[0] = -127.0;
    input[1] = 0.5;
    input[2] = 1.5;
    input[3] = 2.5;
    input[4] = -0.5;

    let blocks = rust_model_inference::ops::quant::quantize_row_q8_k(&input);

    assert_eq!(&blocks[0].qs[..5], &[-127, 0, 2, 2, 0]);
}

#[test]
fn q4_k_prepared_path_matches_existing_scalar_dot_bits() {
    let weight: Vec<u8> = (0usize..144)
        .map(|index| index.wrapping_mul(29).wrapping_add(7) as u8)
        .collect();
    let input: Vec<f32> = (0..256)
        .map(|index| ((index as i32 % 23) - 11) as f32 / 7.0)
        .collect();
    let expected = rust_model_inference::ops::quant::vec_dot_q4k_q8k_scalar(
        &weight,
        &rust_model_inference::ops::quant::quantize_row_q8_k(&input),
    );
    let kernel = Q4_KKernel::new(Q4_KWeight {
        data: &weight,
        n_in: 256,
        n_out: 1,
    });
    let mut actual = [0.0f32];

    kernel.forward_prepared(&input, &[], &[], &mut actual, 256, 1, 0, 1);

    assert_eq!(actual[0].to_bits(), expected.to_bits());
}

#[test]
fn q6_k_prepared_path_matches_existing_scalar_dot_bits() {
    let mut weight: Vec<u8> = (0usize..210)
        .map(|index| index.wrapping_mul(17).wrapping_add(3) as u8)
        .collect();
    weight[208..210].copy_from_slice(&half::f16::from_f32(0.015625).to_bits().to_le_bytes());
    let input: Vec<f32> = (0..256)
        .map(|index| ((index as i32 % 19) - 9) as f32 / 5.0)
        .collect();
    let expected = rust_model_inference::ops::quant::vec_dot_q6k_q8k_scalar(
        &weight,
        &rust_model_inference::ops::quant::quantize_row_q8_k(&input),
    );
    let kernel = Q6_KKernel::new(&weight);
    let mut actual = [0.0f32];

    kernel.forward_prepared(&input, &[], &[], &mut actual, 256, 1, 0, 1);

    assert_eq!(actual[0].to_bits(), expected.to_bits());
}

#[test]
fn q4_k_kernel_multiplies_uniform_block() {
    let mut weight = vec![0u8; 144];
    weight[..2].copy_from_slice(&half::f16::from_f32(1.0).to_bits().to_le_bytes());
    weight[4..16].copy_from_slice(&[1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]);
    weight[16..].fill(0x11);

    let kernel = Q4_KKernel::new(Q4_KWeight {
        data: &weight,
        n_in: 256,
        n_out: 1,
    });
    let mut output = [0.0];
    kernel.forward_prequantized(&[1; 256], &[1.0; 8], &mut output, 256, 1, 0, 1);

    assert_eq!(output, [256.0]);
}

#[test]
fn q6_k_kernel_multiplies_uniform_block() {
    let mut weight = vec![0x11u8; 210];
    weight[128..192].fill(0xaa);
    weight[192..208].fill(1);
    weight[208..].copy_from_slice(&half::f16::from_f32(1.0).to_bits().to_le_bytes());

    let kernel = Q6_KKernel::new(&weight);
    let mut output = [0.0];
    kernel.forward_prequantized(&[1; 1024], &[1.0; 8], &mut output, 256, 1, 0, 1);

    assert_eq!(output, [256.0]);
}
