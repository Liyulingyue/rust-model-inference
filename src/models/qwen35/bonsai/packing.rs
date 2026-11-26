use crate::core::tensor::GGMLType;
use rayon::prelude::*;

/// Re-express the codes as (value + 8) in Q4_0's two 16-element planes.
/// The scale is copied for each group of 32; no floating point is involved.
pub(super) fn repack(ty: GGMLType, data: &[u8]) -> Result<Vec<u8>, String> {
    let (elements, bytes) = ty.type_traits();
    if !matches!(ty, GGMLType::PQ2_0 | GGMLType::Q2_0 | GGMLType::PTQ1_0)
        || !data.len().is_multiple_of(bytes)
    {
        return Err("invalid Bonsai block type or length".into());
    }
    let stride = elements / 32 * 18;
    let mut out = vec![
        0;
        (data.len() / bytes)
            .checked_mul(stride)
            .ok_or("Bonsai repack overflow")?
    ];
    out.par_chunks_mut(stride)
        .zip(data.par_chunks(bytes))
        .for_each(|(dst, src)| {
            let mut codes = [0u8; 128];
            let scale = if ty == GGMLType::PTQ1_0 {
                // 16 lanes x 5 trits, then 8 x 5, then the two high bytes x 4.
                let mut cursor = 0;
                for (packed, digits) in [(&src[..16], 5), (&src[16..24], 5), (&src[24..26], 4)] {
                    for power in [1u8, 3, 9, 27, 81].iter().take(digits) {
                        for &v in packed {
                            codes[cursor] = ((u16::from(v.wrapping_mul(*power)) * 3) >> 8) as u8;
                            cursor += 1;
                        }
                    }
                }
                &src[26..28]
            } else {
                for (chunk, &packed) in codes.chunks_exact_mut(4).zip(&src[2..]) {
                    for (shift, code) in chunk.iter_mut().enumerate() {
                        *code = (packed >> (2 * shift)) & 3;
                    }
                }
                &src[..2]
            };
            for (block, codes) in dst.chunks_exact_mut(18).zip(codes.chunks_exact(32)) {
                block[..2].copy_from_slice(scale);
                for i in 0..16 {
                    block[i + 2] = (codes[i] + 7) | ((codes[i + 16] + 7) << 4);
                }
            }
        });
    Ok(out)
}
