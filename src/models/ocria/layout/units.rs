//! Building LayoutNet's units, ruling lines and raster from a page image and the
//! lines the OCR stack read.
//!
//! The reference reaches the table model through its PDF text layer and its
//! projection pass; this path has neither. Its `ocr_units` (`table_model/mod.rs:100-161`)
//! is the shape this follows: one unit per OCR line, box in the page's coordinate
//! system, height un-padded (`h = (y1 - y0) / 1.4`), no font information and no
//! projection signals.

use super::features::{self, PageImage, Rules};
use super::Unit;

/// One page's layout inputs.
pub(crate) struct PageLayout {
    pub units: Vec<Unit>,
    pub rules: Rules,
    pub image: PageImage,
}

/// Builds the layout inputs from the OCR lines of one page.
///
/// `rgb` is the page image (RGB, or grey with `w * h` bytes) and `w`/`h` its
/// size in pixels; that pixel grid is the coordinate system the units, the
/// ruling lines and the feature normalisation all use.
///
/// Every line becomes a unit, in order: the decoded `Block::units` are indices
/// into this list, and `super::Model` reads a block's text back from the page's
/// lines by the same index, so the two lists have to stay parallel. The
/// reference can drop the items it cannot use because its units come from the
/// projection pass rather than from the lines themselves; here a whitespace-only
/// line or a degenerate box stays in the list as a unit the model ignores.
pub(crate) fn page_layout(
    lines: &[super::super::Line],
    rgb: &[u8],
    w: usize,
    h: usize,
) -> PageLayout {
    let (rules, image) = features::process_raster(rgb, w, h);
    let units = lines.iter().map(unit_of_line).collect();
    PageLayout {
        units,
        rules,
        image,
    }
}

/// One OCR line as a layout unit.
///
/// The reference trims the text of the items it keeps (`mod.rs:113-118`); the box
/// becomes both frames, because the OCR stack reads upright lines, so the reading
/// frame is the page frame. The reference also skips items with empty text or a
/// non-positive box — that cannot be done here without breaking the index
/// correspondence with `Line`s (see [`page_layout`]), and every feature
/// expression stays finite for an empty text and a zero-sized box.
fn unit_of_line(line: &super::super::Line) -> Unit {
    let bx = line.bbox;
    Unit {
        bx,
        pbox: bx,
        text: line.text.trim().to_string(),
        // the recognizer's crop carries 1.4x the glyph height; the reference's
        // `ocr_units` divides it back out (`mod.rs:142`)
        h: (bx[3] - bx[1]) / 1.4,
        bold: None,
        faint: line.faint,
    }
}

/// The page's ruling lines and raster, without the units.
#[allow(dead_code)] // the model's path is `page_layout`, which also builds the units
pub(crate) fn raster(rgb: &[u8], w: usize, h: usize) -> (Rules, PageImage) {
    features::process_raster(rgb, w, h)
}

#[cfg(test)]
mod tests {
    use super::features::{unit_features, FUnit, FEAT_DIM};
    use super::*;

    fn line(text: &str, bbox: [f32; 4], faint: bool) -> crate::models::ocria::Line {
        let [x0, y0, x1, y1] = bbox;
        crate::models::ocria::Line {
            text: text.to_string(),
            bbox,
            confidence: 0.9,
            polygon: [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
            faint,
        }
    }

    /// A white page with one black rectangle: `(x0, y0, x1, y1)` inclusive, RGB.
    fn ruled_page(w: usize, h: usize, rect: (usize, usize, usize, usize)) -> Vec<u8> {
        let mut rgb = vec![255u8; w * h * 3];
        let (x0, y0, x1, y1) = rect;
        for y in y0..=y1.min(h - 1) {
            for x in x0..=x1.min(w - 1) {
                let o = (y * w + x) * 3;
                rgb[o..o + 3].copy_from_slice(&[0, 0, 0]);
            }
        }
        rgb
    }

    #[test]
    fn page_layout_maps_ocr_lines_to_units_in_order() {
        let lines = [
            line("  Item ", [10.0, 20.0, 60.0, 40.0], false),
            line("   ", [0.0, 0.0, 10.0, 10.0], false), // whitespace-only: kept, empty text
            line("7", [100.0, 20.0, 110.0, 34.0], true), // faint short token
            line("x", [5.0, 5.0, 5.0, 9.0], false),     // zero-width box: kept
            line("y", [7.0, 7.0, 9.0, 7.0], false),     // zero-height box: kept
        ];
        let page = page_layout(&lines, &vec![255u8; 64 * 48 * 3], 64, 48);
        // one unit per line, same order: Block::units indexes this list and the
        // page's text is read back from `lines` by the same index
        assert_eq!(page.units.len(), lines.len());
        for (u, l) in page.units.iter().zip(&lines) {
            assert_eq!(u.bx, l.bbox);
            assert_eq!(u.pbox, u.bx, "upright line: one frame");
            assert_eq!(u.bold, None, "no font information from OCR");
            assert_eq!(u.faint, l.faint, "the faint flag rides on the unit");
            assert_eq!(u.text, l.text.trim());
        }

        let u0 = &page.units[0];
        assert_eq!(u0.text, "Item", "trimmed");
        assert!((u0.h - 20.0 / 1.4).abs() < 1e-6, "un-padded height");
        assert!(!u0.faint);
        assert_eq!(page.units[1].text, "", "a whitespace-only line has no text");
        assert!(page.units[2].faint);
        assert!((page.units[2].h - 14.0 / 1.4).abs() < 1e-6);
        assert!((page.units[3].h - 4.0 / 1.4).abs() < 1e-6);
        assert_eq!(page.units[4].h, 0.0, "a flat box has no height");

        // the raster is the page's grey at the image branch's size
        assert_eq!((page.image.w, page.image.h), (1024, 768));
        assert!((page.image.k - 1024.0 / 64.0).abs() < 1e-6);
        assert!(page.rules.h.is_empty() && page.rules.v.is_empty());

        // degenerate lines must still produce finite features, not NaN
        let funits: Vec<FUnit> = page
            .units
            .iter()
            .map(|u| FUnit {
                bx: u.bx,
                pbox: u.pbox,
                text: u.text.clone(),
                h: u.h,
                bold: u.bold,
                proj: None,
            })
            .collect();
        let f = unit_features(&funits, (64.0, 48.0), Some(&page.rules));
        assert_eq!(f.len(), 5 * FEAT_DIM);
        assert!(f.iter().all(|v| v.is_finite()), "finite features");
    }

    #[test]
    fn raster_is_the_page_layout_raster() {
        let rgb = ruled_page(200, 100, (10, 48, 189, 49));
        let (rules, image) = raster(&rgb, 200, 100);
        assert_eq!(rules.h.len(), 1, "the 2 px rule survives the cap");
        let page = page_layout(&[], &rgb, 200, 100);
        assert_eq!(page.rules.h, rules.h);
        assert_eq!(page.rules.v, rules.v);
        assert_eq!(page.image.px, image.px);
        assert_eq!((page.image.w, page.image.h), (image.w, image.h));
    }

    /// End to end over the two files: the rule under a line becomes feature 61.
    #[test]
    fn a_rule_under_a_line_flags_the_unit() {
        let rgb = ruled_page(200, 100, (10, 48, 189, 49));
        let lines = [
            line("Item", [10.0, 50.0, 60.0, 70.0], false),
            line("Qty", [120.0, 50.0, 170.0, 70.0], false),
        ];
        let page = page_layout(&lines, &rgb, 200, 100);
        let funits: Vec<FUnit> = page
            .units
            .iter()
            .map(|u| FUnit {
                bx: u.bx,
                pbox: u.pbox,
                text: u.text.clone(),
                h: u.h,
                bold: u.bold,
                proj: None,
            })
            .collect();
        let f = unit_features(&funits, (200.0, 100.0), Some(&page.rules));
        assert_eq!(f.len(), 2 * FEAT_DIM);
        for i in 0..2 {
            let row = &f[i * FEAT_DIM..(i + 1) * FEAT_DIM];
            assert_eq!(row[61], 1.0, "unit {i}: rule above");
            assert_eq!(row[62], 0.0, "unit {i}: nothing below");
            assert_eq!(row[65], 1.0, "unit {i}: rules present");
            // h = 20/1.4 = 14.29, so the rule 1 pt above is inside the 1.5h band
            assert!(row[8] > 0.9 && row[8] < 1.1, "unit {i}: h / median h");
        }
    }
}
