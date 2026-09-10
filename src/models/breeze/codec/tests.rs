use super::*;

#[test]
fn tensor_contract_rejects_bf16_and_wrong_axis_order() {
    use crate::core::tensor::TensorInfo;
    struct Source {
        info: TensorInfo,
        bytes: Vec<u8>,
    }
    impl TensorSource for Source {
        fn metadata(&self, _: &str) -> Option<&MetaValue> {
            None
        }
        fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
            (name == self.info.name).then_some(&self.info)
        }
        fn tensor_slice(&self, name: &str) -> Option<&[u8]> {
            (name == self.info.name).then_some(self.bytes.as_slice())
        }
    }
    let mut source = Source {
        info: TensorInfo {
            name: "weight".into(),
            dims: vec![3, 2],
            ggml_type: GGMLType::BF16,
            offset: 0,
        },
        bytes: vec![0; 24],
    };
    assert!(tensor(&source, "weight", &[2, 3]).is_err());
    source.info.ggml_type = GGMLType::F32;
    assert_eq!(tensor(&source, "weight", &[2, 3]).unwrap(), vec![0.; 6]);
    assert!(tensor(&source, "weight", &[3, 2]).is_err());
    source.bytes[..4].copy_from_slice(&f32::NAN.to_le_bytes());
    assert!(tensor(&source, "weight", &[2, 3]).is_err());
    assert!(validate_config(&source).is_err());
}

#[test]
fn causal_convolution_padding_stride_dilation_and_groups() {
    let mut conv = Conv {
        weight: vec![1., 2., 3., 4.],
        bias: vec![0.],
        input: 1,
        output: 1,
        kernel: 4,
        stride: 2,
        dilation: 1,
        groups: 1,
        replicate: false,
    };
    assert_eq!(conv.forward(&[1., 2., 3.]).unwrap(), vec![11., 14.]);
    conv.replicate = true;
    assert_eq!(conv.forward(&[1., 2., 3.]).unwrap(), vec![14., 26.]);
    conv = Conv {
        weight: vec![1., 2., 3., 4.],
        bias: vec![0., 10.],
        input: 2,
        output: 2,
        kernel: 2,
        stride: 1,
        dilation: 2,
        groups: 2,
        replicate: false,
    };
    assert_eq!(
        conv.forward(&[1., 2., 3., 4., 5., 6.]).unwrap(),
        vec![2., 18., 6., 26., 11., 40.]
    );
    assert!(conv.forward(&[1.]).is_err());
}

#[test]
fn codebook_keeps_first_tie_and_subtracts_selected_centroid() {
    let book = Codebook {
        values: vec![0., 0., 2., 0.],
        dim: 2,
        norms: vec![0., 4.],
    };
    let mut residual = vec![1., 0., 1.75, 0.];
    assert_eq!(book.encode(&mut residual), vec![0, 1]);
    assert_eq!(residual, vec![1., 0., -0.25, 0.]);
}

#[test]
fn input_contract_rejects_invalid_pcm_and_ids() {
    assert!(validate_audio(&[]).is_err());
    assert!(validate_audio(&[f32::NAN]).is_err());
    assert!(validate_frames(&[]).is_err());
    assert!(validate_frames(&[[2048; 16]]).is_err());
    assert_eq!(validate_audio(&[0.; 1921]).unwrap(), 2);
    assert_eq!(validate_frames(&[[2047; 16]]).unwrap(), 1920);
}

#[test]
#[ignore = "requires BREEZE_CODEC_GGUF; optionally BREEZE_CODEC_REFERENCE_DIR for numeric comparison"]
fn codec_real_weights_and_reference() {
    use crate::format::ggufrs::{open_model_source, ComponentRole};
    let path = std::env::var("BREEZE_CODEC_GGUF").unwrap();
    let source = open_model_source(std::path::Path::new(&path), ComponentRole::Mmproj).unwrap();
    let codec = BreezeCodec::from_source(source.as_ref()).unwrap();
    let mut frames = vec![
        [
            404, 1380, 1234, 2018, 681, 179, 1453, 1610, 770, 1245, 1839, 1223, 848, 1771, 602,
            1102,
        ],
        [
            1630, 997, 439, 810, 783, 396, 293, 975, 870, 1384, 1619, 344, 1459, 170, 1558, 1861,
        ],
    ];
    if let Ok(path) = std::env::var("BREEZE_CODEC_CODES") {
        frames = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    }
    let output = codec.decode(&frames).unwrap();
    assert_eq!(output.len(), frames.len() * SAMPLES_PER_FRAME);
    assert!(output.iter().all(|v| v.is_finite() && v.abs() <= 1.));
    assert!(output.iter().any(|v| *v != 0.));
    let audio: Vec<f32> = (0..1921).map(|i| ((i % 97) as f32 - 48.) / 256.).collect();
    let encoded = codec.encode(&audio).unwrap();
    assert_eq!(encoded.len(), 2);
    assert!(encoded.iter().flatten().all(|&v| v < 2048));
    eprintln!("Breeze codec encoded fixture: {encoded:?}");
    if let Ok(dir) = std::env::var("BREEZE_CODEC_REFERENCE_DIR") {
        let expected_ids: Vec<[u32; 16]> =
            serde_json::from_slice(&std::fs::read(format!("{dir}/encoded.json")).unwrap()).unwrap();
        assert_eq!(encoded, expected_ids);
        let bytes = std::fs::read(format!("{dir}/decoded.f32")).unwrap();
        let expected: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        assert_eq!(output.len(), expected.len());
        for (index, (actual, expected)) in output.iter().zip(expected).enumerate() {
            let tolerance = 1e-4 + 1e-4 * expected.abs();
            assert!(
                (actual - expected).abs() <= tolerance,
                "PCM mismatch at sample {index}: {actual} != {expected}"
            );
        }
    }
}
