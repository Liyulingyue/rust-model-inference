use rust_model_inference::models::diffusion::qwen_image_2_1::{flow_sigmas, rgba_bytes};

#[test]
fn flow_schedule_keeps_the_last_evaluation_and_final_zero_distinct() {
    let sigmas = flow_sigmas(4, 256).unwrap();
    assert_eq!(sigmas.len(), 5);
    assert_eq!(sigmas[0], 1.0);
    assert!((sigmas[3] - 0.02).abs() < 1e-6);
    assert_eq!(sigmas[4], 0.0);
    assert!(sigmas.windows(2).all(|pair| pair[0] > pair[1]));
    assert_eq!(flow_sigmas(1, 256).unwrap(), [1.0, 0.0]);
    assert!(flow_sigmas(0, 256).is_err());
    assert!(flow_sigmas(4, 0).is_err());
}

#[test]
fn rgba_output_preserves_alpha_and_rejects_bad_tensor_shapes() {
    // Planar R, G, B, A; two pixels, including transparent saturated RGB.
    assert_eq!(
        rgba_bytes(&[-1.0, 1.0, 0.0, 0.0, 1.0, -1.0, -1.0, 1.0], 2, 1).unwrap(),
        [0, 127, 255, 0, 255, 127, 0, 255]
    );
    assert!(rgba_bytes(&[0.0; 4], 2, 1).is_err());
    assert!(rgba_bytes(&[f32::NAN; 4], 1, 1).is_err());
    assert!(rgba_bytes(&[], usize::MAX, 2).is_err());
}
