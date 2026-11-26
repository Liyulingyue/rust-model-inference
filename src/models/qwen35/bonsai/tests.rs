use super::*;

#[test]
fn signed_gdn_map_transposes_heads_and_preserves_channels() {
    let signs = [1, -1, -1, 1, 1, 1, -1, -1, 1, -1, 1, 1];
    let map = signed_map(&signs, Some((2, 6))).unwrap();
    assert_eq!(&*map, &[1, -2, -5, 6, 9, 10, -3, -4, 7, -8, 11, 12]);
    assert!(signed_map(&signs, Some((4, 6))).is_err());
    assert!(signed_map(&signs, Some((2, 8))).is_err());
}

#[test]
fn packed_codes_and_scale_survive_repacking() {
    for (ty, size) in [
        (GGMLType::Q2_0, 64),
        (GGMLType::PQ2_0, 128),
        (GGMLType::PTQ1_0, 128),
    ] {
        let values: Vec<u8> = (0..size)
            .map(|i| ((i * 17 + i / 7) % if ty == GGMLType::PTQ1_0 { 3 } else { 4 }) as u8)
            .collect();
        let scale = [0x37, 0x38];
        let mut input = vec![];
        if ty == GGMLType::PTQ1_0 {
            let mut cursor = 0;
            for (stride, digits) in [(16, 5), (8, 5), (2, 4)] {
                for lane in 0..stride {
                    let code = (0..digits).fold(0u16, |code, digit| {
                        code * 3 + u16::from(values[cursor + lane + digit * stride])
                    });
                    let code = if digits == 4 { code * 3 } else { code };
                    input.push((code * 256).div_ceil(243) as u8);
                }
                cursor += stride * digits;
            }
            input.extend(scale);
        } else {
            input.extend(scale);
            input.extend(
                values
                    .chunks_exact(4)
                    .map(|q| q[0] | q[1] << 2 | q[2] << 4 | q[3] << 6),
            );
        }
        let packed = repack(ty, &input).unwrap();
        for (block, values) in packed.chunks_exact(18).zip(values.chunks_exact(32)) {
            assert_eq!(&block[..2], &scale);
            for i in 0..16 {
                assert_eq!(block[2 + i] & 15, values[i] + 7);
                assert_eq!(block[2 + i] >> 4, values[i + 16] + 7);
            }
        }
    }
}

#[test]
fn walsh_transform_matches_dense_matrix() {
    let input: Vec<f32> = (0..1024)
        .map(|i| ((37 * i % 119) as f32 - 51.0) / 128.0)
        .collect();
    let mut actual = input.clone();
    hadamard(&mut actual);
    for (i, &result) in actual.iter().enumerate() {
        let expected = input
            .iter()
            .enumerate()
            .map(|(j, x)| {
                if (i & j).count_ones() % 2 == 0 {
                    *x as f64
                } else {
                    -(*x as f64)
                }
            })
            .sum::<f64>()
            / 32.0;
        assert!((result as f64 - expected).abs() < 1e-5);
    }
}
