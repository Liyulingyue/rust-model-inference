//! Convert user-supplied inpainting colors with the release's OKLab metric.
use super::{COLORS, MASK, PIXELS};
use serde::Deserialize;

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(untagged)]
pub enum Pixel {
    Id(u16),
    Rgba([u8; 4]),
}

fn oklab([r, g, b]: [u8; 3]) -> [f32; 3] {
    let [r, g, b] = [r, g, b].map(|value| {
        let s = f64::from(value) / 255.;
        if s <= 0.04045 {
            s / 12.92
        } else {
            ((s + 0.055) / 1.055).powf(2.4)
        }
    });
    let l = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b).cbrt();
    let m = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b).cbrt();
    let s = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b).cbrt();
    [
        (0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s) as f32,
        (1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s) as f32,
        (0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s) as f32,
    ]
}

pub(super) fn initial_canvas(entries: Option<&[Option<Pixel>]>, palette: &[[u8; 3]]) -> Vec<u16> {
    let Some(entries) = entries else {
        return vec![MASK; PIXELS];
    };
    // Token zero is reserved for transparency, including when an opaque
    // palette entry also encodes black.
    let colors = std::cell::LazyCell::new(|| {
        palette[1..COLORS]
            .iter()
            .copied()
            .map(oklab)
            .collect::<Vec<_>>()
    });
    entries
        .iter()
        .map(|entry| match entry {
            None => MASK,
            Some(Pixel::Id(id)) => *id,
            Some(Pixel::Rgba([_, _, _, a])) if *a < 128 => 0,
            Some(Pixel::Rgba([r, g, b, _])) => {
                let point = oklab([*r, *g, *b]);
                let distance =
                    |color: &[f32; 3]| (0..3).map(|i| (color[i] - point[i]).powi(2)).sum::<f32>();
                let (_, index) = colors
                    .iter()
                    .enumerate()
                    .map(|(i, c)| (distance(c), i + 1))
                    .min_by(|a, b| a.0.total_cmp(&b.0))
                    .expect("fixed 511-color opaque palette");
                index as u16
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn masked_transparent_and_opaque_black_are_distinct() {
        let mut palette = vec![[255; 3]; COLORS];
        palette[2] = [0; 3];
        let entries = [
            None,
            Some(Pixel::Id(8)),
            Some(Pixel::Rgba([0, 0, 0, 127])),
            Some(Pixel::Rgba([0, 0, 0, 128])),
            Some(Pixel::Rgba([254, 254, 254, 255])),
        ];
        assert_eq!(initial_canvas(Some(&entries), &palette), [MASK, 8, 0, 2, 1]);
    }
}
