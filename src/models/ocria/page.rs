//! Page-level geometry and the reading heuristics: crops, the column-gutter
//! split, the sideways-page test, the reading frame, the junk filter and the
//! script statistics the rescue needs.
//!
//! Every routine mirrors the one LiteParse runs (`ocr/ocria/page.rs` in the
//! pinned revision recorded in `tools/oracle/ocria/sources.json`). These are
//! the pipeline's discrete decisions -- which crop a line becomes, whether a
//! page is retried a quarter turn round, which lines are dropped and which
//! characters are folded -- so they change the text, not just the numbers.

use super::img::{otsu, perspective_transform, warp_perspective_cubic, Img};
use super::Quad;

fn dist(a: [f32; 2], b: [f32; 2]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

/// The crop a line quad covers, resampled onto an axis-aligned box.
///
/// The output is as wide as the quad's longer horizontal pair and as tall as its longer
/// vertical pair (at least 2 px each), and the quad's corners map onto that box's corners;
/// a box more than 1.5x taller than wide is turned a quarter counter-clockwise so the
/// recognizer reads it along the line.
pub(crate) fn crop(img: &Img, q: &Quad) -> Img {
    let w = (dist(q[0], q[1]).max(dist(q[2], q[3])) as usize).max(2);
    let h = (dist(q[0], q[3]).max(dist(q[1], q[2])) as usize).max(2);
    let dst = [
        [0.0, 0.0],
        [w as f32, 0.0],
        [w as f32, h as f32],
        [0.0, h as f32],
    ];
    let m = perspective_transform(q, &dst);
    let c = warp_perspective_cubic(img, &m, w, h);
    if h as f64 / w as f64 >= 1.5 {
        // vertical box: turn it horizontal
        c.rot90(1)
    } else {
        c
    }
}

/// The quad's axis-aligned extents: `(x0, y0, x1, y1)`.
fn extent(q: &Quad) -> (f64, f64, f64, f64) {
    let xs = q.iter().map(|p| p[0] as f64);
    let ys = q.iter().map(|p| p[1] as f64);
    (
        xs.clone().fold(f64::INFINITY, f64::min),
        ys.clone().fold(f64::INFINITY, f64::min),
        xs.fold(f64::NEG_INFINITY, f64::max),
        ys.fold(f64::NEG_INFINITY, f64::max),
    )
}

/// Boxes that are clearly horizontal text lines (at least 3x wider than tall, >= 30 px).
pub(crate) fn long_lines(boxes: &[Quad]) -> usize {
    boxes
        .iter()
        .filter(|q| {
            let (x0, y0, x1, y1) = extent(q);
            let (dx, dy) = (x1 - x0, y1 - y0);
            dx >= 30.0 && dx >= 3.0 * dy
        })
        .count()
}

/// Whether the page looks sideways: many detected lines, almost none of them horizontal.
pub(crate) fn looks_sideways(boxes: &[Quad]) -> bool {
    boxes.len() >= 15 && (long_lines(boxes) as f64) < 0.25 * boxes.len() as f64
}

/// The quad in the page's reading frame: `flip` undoes an upside-down page and
/// `sideways` a quarter turn.
///
/// A box quad (screen order in the image it was read in) becomes a quad in the original
/// image whose corners run in reading order: an upside-down page swaps corners; a page
/// read after a quarter turn maps back with `x = w0 - 1 - y'`, `y = x'`, the inverse of
/// [`Img::rot90`]`(1)`.
pub(crate) fn reading_frame(q: &Quad, flip: bool, sideways: bool, w0: usize) -> Quad {
    let mut p = *q;
    if flip {
        p = [p[2], p[3], p[0], p[1]];
    }
    if sideways {
        for pt in p.iter_mut() {
            *pt = [(w0 as f64 - 1.0 - pt[1] as f64) as f32, pt[0]];
        }
    }
    p
}

/// Minimum blank-run width, in line heights, for a column gutter.
const GUTTER_MIN: f64 = 0.4;

/// Cut near-horizontal boxes at blank vertical runs that line up with other boxes' edges
/// and stay clear in the neighbouring lines.
pub(crate) fn split_gutters(img: &Img, boxes: &[Quad]) -> Vec<Quad> {
    if boxes.len() < 3 {
        return boxes.to_vec();
    }
    let gray = img.mean_gray();
    let (gw, gh) = (gray.w as i64, gray.h as i64);
    let px = |x: i64, y: i64| gray.data[(y * gw + x) as usize];
    let ext: Vec<(f64, f64, f64, f64)> = boxes.iter().map(extent).collect();
    let mut out = Vec::with_capacity(boxes.len());
    for (i, q) in boxes.iter().enumerate() {
        let (x0, y0, x1, y1) = ext[i];
        let h = y1 - y0;
        let tilt = (q[1][1] as f64 - q[0][1] as f64).abs();
        if h < 6.0 || x1 - x0 < 4.0 * h || tilt > 0.25 * h {
            out.push(*q);
            continue;
        }
        let ya = ((y0 + 0.15 * h) as i64).max(0).min(gh);
        let yb = ((y1 - 0.15 * h).ceil() as i64).clamp(0, gh);
        let xa = (x0 as i64).max(0);
        let xb = (x1.ceil() as i64).min(gw);
        if yb <= ya || xb <= xa {
            out.push(*q);
            continue;
        }
        let (mut lo, mut hi) = (255u8, 0u8);
        for y in ya..yb {
            for x in xa..xb {
                let v = px(x, y);
                lo = lo.min(v);
                hi = hi.max(v);
            }
        }
        if (hi as i32) - (lo as i32) < 60 {
            out.push(*q);
            continue;
        }
        let thr = otsu(
            (ya..yb)
                .flat_map(|y| (xa..xb).map(move |x| (x, y)))
                .map(|(x, y)| px(x, y)),
        );
        // A printed column rule (a thin solid stroke that carries on above and below the
        // line, through the space between lines) belongs to the gutter, not the text:
        // glyph stems stop at the line.
        let rule = rule_columns(&px, (xa, xb), (y0, y1), (gw, gh), thr);
        // ink columns (relative to xa)
        let idx: Vec<i64> = (xa..xb)
            .filter(|&x| !rule.contains(&x) && (ya..yb).any(|y| px(x, y) < thr))
            .map(|x| x - xa)
            .collect();
        if idx.len() < 2 {
            out.push(*q);
            continue;
        }
        let (cy, tol) = ((y0 + y1) / 2.0, 0.6 * h);
        let mut cuts: Vec<(f64, f64)> = Vec::new();
        for g in 0..idx.len() - 1 {
            if ((idx[g + 1] - idx[g]) as f64) <= GUTTER_MIN * h {
                continue;
            }
            // blank columns [a, b)
            let (a, b) = ((xa + idx[g] + 1) as f64, (xa + idx[g + 1]) as f64);
            let near = ext.iter().enumerate().filter(|&(j, e)| {
                j != i
                    && ((e.1 + e.3) / 2.0 - cy).abs() < 25.0 * h
                    && 0.5 * h < e.3 - e.1
                    && e.3 - e.1 < 2.0 * h
            });
            let (mut left, mut right) = (0, 0);
            for (_, e) in near {
                left += ((e.0 - b).abs() < tol) as usize;
                right += ((e.2 - a).abs() < tol) as usize;
            }
            // both sides need a column of edges; a hanging-indent marker has only one
            if left < 1 || right < 1 || left + right < 3 || a - x0 < 3.0 * h {
                continue;
            }
            // the gutter stays clear in the neighbouring lines, judged per line
            let m = (((b - a) * 0.25) as i64).max(1) as f64;
            let (sa, sb) = ((a + m) as i64, (b - m) as i64);
            if sb <= sa {
                continue;
            }
            let (mut ok, mut bad) = (0, 0);
            for (j, e) in ext.iter().enumerate() {
                if j == i
                    || ((e.1 + e.3) / 2.0 - cy).abs() > 4.0 * h
                    || !(0.5 * h < e.3 - e.1 && e.3 - e.1 < 2.0 * h)
                {
                    continue;
                }
                if e.2 <= sa as f64 || e.0 >= sb as f64 {
                    ok += 1;
                    continue;
                }
                let eh = e.3 - e.1;
                let (ra, rb) = (
                    ((e.1 + 0.15 * eh) as i64).clamp(0, gh),
                    ((e.3 - 0.15 * eh).ceil() as i64).clamp(0, gh),
                );
                let (ca, cb) = (sa.clamp(0, gw), sb.clamp(0, gw));
                let inked =
                    (ra..rb).any(|y| (ca..cb).any(|x| !rule.contains(&x) && px(x, y) < thr));
                if inked {
                    bad += 1;
                } else {
                    ok += 1;
                }
            }
            if !(ok >= 2 && (bad == 0 || (bad == 1 && ok >= 3))) {
                continue;
            }
            cuts.push((a, b));
        }
        if cuts.is_empty() {
            out.push(*q);
            continue;
        }
        let pad = 0.15 * h;
        let mut edges = vec![x0];
        for &(a, b) in &cuts {
            edges.push(a + pad);
            edges.push(b - pad);
        }
        edges.push(x1);
        for pair in edges.chunks(2) {
            let (s, e) = (pair[0], pair[1]);
            if e - s > 0.5 * h {
                let (s, e, ya, yb) = (s as f32, e as f32, y0 as f32, y1 as f32);
                out.push([[s, ya], [e, ya], [e, yb], [s, yb]]);
            }
        }
    }
    out
}

/// Columns of a printed vertical rule crossing the line box `y0..y1`: runs at most ~0.15
/// line heights wide that are inked down the whole line and through bands of 0.35 line
/// heights above and below it, with blank columns on both sides, widened by their
/// anti-aliased edges.
fn rule_columns(
    px: &impl Fn(i64, i64) -> u8,
    (xa, xb): (i64, i64),
    (y0, y1): (f64, f64),
    (gw, gh): (i64, i64),
    thr: u8,
) -> std::collections::HashSet<i64> {
    let h = y1 - y0;
    let band = (0.35 * h).ceil() as i64;
    let (top, bot) = (y0 as i64, y1.ceil() as i64);
    let inked_share = |x: i64, ya: i64, yb: i64| -> f64 {
        let (ya, yb) = (ya.clamp(0, gh), yb.clamp(0, gh));
        if yb <= ya || x < 0 || x >= gw {
            return 0.0;
        }
        (ya..yb).filter(|&y| px(x, y) < thr).count() as f64 / (yb - ya) as f64
    };
    let solid: Vec<bool> = (xa..xb)
        .map(|x| {
            inked_share(x, top, bot) >= 0.9
                && inked_share(x, top - band, top) >= 0.8
                && inked_share(x, bot, bot + band) >= 0.8
        })
        .collect();
    let max_w = ((0.15 * h).ceil() as i64).max(3);
    let mut out = std::collections::HashSet::new();
    let mut i = 0usize;
    while i < solid.len() {
        if !solid[i] {
            i += 1;
            continue;
        }
        let start = i;
        while i < solid.len() && solid[i] {
            i += 1;
        }
        // a stroke between blank columns (a light ground): on a dark panel every gap
        // between light letters is "inked" top to bottom too
        let flank = ((0.25 * h).ceil() as i64).max(3);
        let blank_side = |from: i64, to: i64| {
            (from..to)
                .filter(|&x| inked_share(x, top, bot) < 0.1)
                .count() as i64
                * 2
                >= to - from
        };
        let (rs, re) = (xa + start as i64, xa + i as i64);
        if (i - start) as i64 <= max_w && blank_side(rs - flank, rs) && blank_side(re, re + flank) {
            // with its anti-aliased edges, inked in only part of the line
            let pad = ((0.06 * h).ceil() as i64).max(2);
            out.extend((xa + start as i64 - pad)..(xa + i as i64 + pad));
        }
    }
    out
}

/// The most common group (`np.bincount(..).argmax()`: the lowest index on ties).
pub(crate) fn majority(groups: &[usize]) -> usize {
    let mut n = [0usize; 4];
    for &g in groups {
        n[g] += 1;
    }
    let mut best = 0;
    for i in 1..4 {
        if n[i] > n[best] {
            best = i;
        }
    }
    best
}

/// `np.mean` over the values; NaN for none.
pub(crate) fn mean(values: impl Iterator<Item = f32>) -> f64 {
    let (s, n) = values.fold((0f64, 0usize), |(s, n), x| (s + x as f64, n + 1));
    if n == 0 {
        f64::NAN
    } else {
        s / n as f64
    }
}

/// Characters that show a reading really is in `group`'s script.
///
/// The groups are the recognizer order: Latin (with Latin-1 / Extended and Greek /
/// Cyrillic, whose lookalikes the Latin model reads), CJK, right-to-left, and the rest of
/// the world's scripts.
pub(crate) fn script_chars(group: usize, text: &str) -> usize {
    text.chars()
        .filter(|&c| {
            let u = c as u32;
            match group {
                0 => {
                    c.is_ascii_alphabetic()
                        || (0x00c0..=0x024f).contains(&u)
                        || (0x0370..=0x04ff).contains(&u)
                }
                1 => [
                    (0x3040, 0x30ff),
                    (0x3400, 0x9fff),
                    (0xac00, 0xd7af),
                    (0xf900, 0xfaff),
                ]
                .iter()
                .any(|&(a, b)| (a..=b).contains(&u)),
                2 => [(0x0590, 0x08ff), (0xfb1d, 0xfdff), (0xfe70, 0xfeff)]
                    .iter()
                    .any(|&(a, b)| (a..=b).contains(&u)),
                _ => [
                    (0x0900, 0x0dff),
                    (0x0e00, 0x0eff),
                    (0x0f00, 0x109f),
                    (0x10a0, 0x10ff),
                    (0x0530, 0x058f),
                    (0x1200, 0x139f),
                    (0x1780, 0x17ff),
                ]
                .iter()
                .any(|&(a, b)| (a..=b).contains(&u)),
            }
        })
        .count()
}

fn is_latin_letter(c: char) -> bool {
    c.is_ascii_alphabetic() || (c.is_alphabetic() && ('\u{00c0}'..='\u{024f}').contains(&c))
}

/// Latin letters in `text` (ASCII and Latin-1 / Extended-A/B; not Greek or Cyrillic).
pub(crate) fn latin_letters(text: &str) -> usize {
    text.chars().filter(|&c| is_latin_letter(c)).count()
}

/// Cyrillic / Greek lookalikes of Latin letters.
const HOMO_FROM: &str = "аеорсухкіјѕԁԛԝАВЕКМНОРСТХІЈЅΑΒΕΖΗΙΚΜΝΟΡΤΥΧοαν";
const HOMO_TO: &str = "aeopcyxkijsdqwABEKMHOPCTXIJSABEZHIKMNOPTYXoav";
/// Lookalikes from the other scripts a Latin line picks up: the Latin recognizer's own
/// foreign classes (Thai น ท ร, Armenian ա) and the world recognizer reading a Latin line
/// ("REPՍBLQՍE", "C๐mmunicatoทร"). Zero digits that look like "o" count here: inside a
/// Latin word they are the letter.
const HOMO_EXTRA_FROM: &str = "ՍսՕօոհցզաԼՏนทรอบ๐ບນຣ໐၀ဝ";
const HOMO_EXTRA_TO: &str = "UuOonhgqwLSunsououusooo";

/// First code point of each run of decimal digits the recognizers can write besides ASCII
/// (Arabic-Indic, Extended Arabic-Indic, the Indic scripts, Thai, Lao, Tibetan, Myanmar,
/// Khmer).
const DIGIT_ZEROS: [u32; 18] = [
    0x0660, 0x06f0, 0x0966, 0x09e6, 0x0a66, 0x0ae6, 0x0b66, 0x0be6, 0x0c66, 0x0ce6, 0x0d66, 0x0de6,
    0x0e50, 0x0ed0, 0x0f20, 0x1040, 0x1090, 0x17e0,
];

/// Latin and Greek lookalikes of Cyrillic letters, for a Cyrillic word the Latin recognizer
/// wrote partly in Latin ("aрганізацый", "ФOPYMA", "ПяТAГΑ").
const CYR_FROM: &str = "aceopxyiABCEHKMOPTXYIΑΒΓΕΗΚΜΟΠΡΤΧαο";
const CYR_TO: &str = "асеорхуіАВСЕНКМОРТХУІАВГЕНКМОПРТХао";

fn homoglyph(c: char) -> Option<char> {
    HOMO_FROM
        .chars()
        .zip(HOMO_TO.chars())
        .chain(HOMO_EXTRA_FROM.chars().zip(HOMO_EXTRA_TO.chars()))
        .find(|m| m.0 == c)
        .map(|m| m.1)
}

/// The Latin letter a Cyrillic or other-script lookalike stands for in a Latin word.
///
/// Small-cap and italic Cyrillic forms (т н м п г л) only look Latin in context, so they
/// are mapped here and not in the table; т н м take the word's case ("ZQX272003тOKEN",
/// "OнMS").
fn latin_lookalike(c: char, upper: bool) -> Option<char> {
    homoglyph(c).or(match c {
        'т' => Some(if upper { 'T' } else { 't' }),
        'н' => Some(if upper { 'H' } else { 'h' }),
        'м' => Some(if upper { 'M' } else { 'm' }),
        'п' | 'л' => Some('n'),
        'г' => Some('r'),
        _ => None,
    })
}

/// Clean a reading of other scripts' lookalike characters.
///
/// In a line that is mostly Latin, inside a word with at least as many Latin letters as
/// other letters, lookalikes become the Latin letter ("REPՍBLQՍE" -> "REPUBLQUE"), a word
/// of lookalikes alone does too ("В - Linear"), and other scripts' digits become ASCII
/// digits ("(UA०८" -> "(UA08"). Lines mostly in another script (a Cyrillic page read by
/// the Latin recognizer, a bilingual Gurmukhi line) are left alone.
pub(crate) fn clean_latin(text: &str) -> String {
    let lat = latin_letters(text);
    let oth = text
        .chars()
        .filter(|&c| c.is_alphabetic() && !is_latin_letter(c))
        .count();
    let latin_line = lat > 0 && lat >= 2 * oth;
    let cyr_line = text
        .chars()
        .filter(|&c| ('\u{0400}'..='\u{04ff}').contains(&c))
        .count()
        > lat;
    let mut out = String::with_capacity(text.len());
    for (k, word) in text.split(char::is_whitespace).enumerate() {
        if k > 0 {
            // split() drops the separator; every separator here is one whitespace char
            out.push(' ');
        }
        let wl = latin_letters(word);
        let wo = word
            .chars()
            .filter(|&c| c.is_alphabetic() && !is_latin_letter(c))
            .count();
        let upper = word.chars().filter(|c| c.is_uppercase()).count()
            > word.chars().filter(|c| c.is_lowercase()).count();
        // Greek letters standing alone are real ("α = 0.05"), never lookalikes
        let all_lookalike = wo > 0
            && word.chars().all(|c| {
                !c.is_alphabetic()
                    || is_latin_letter(c)
                    || (latin_lookalike(c, upper).is_some()
                        && !('\u{0370}'..='\u{03ff}').contains(&c))
            });
        // Cyrillic / Greek inside a Latin word fold anywhere; the other scripts'
        // lookalikes only in a Latin line, since Thai and Lao glue Latin words to their
        // own without a space ("Windowsนี้")
        let cg = word
            .chars()
            .filter(|&c| ('\u{0370}'..='\u{04ff}').contains(&c))
            .count();
        let fold_cg = cg > 0 && wl >= cg;
        let fold = latin_line && wo > 0 && (wl >= wo || all_lookalike);
        // the other way: a Cyrillic word with Latin / Greek lookalikes inside
        let cyr = word
            .chars()
            .filter(|&c| ('\u{0400}'..='\u{04ff}').contains(&c))
            .count();
        let lookalikes_only = word
            .chars()
            .filter(|&c| c.is_alphabetic() && !('\u{0400}'..='\u{04ff}').contains(&c))
            .all(|c| CYR_FROM.contains(c));
        let to_cyr =
            (cyr >= 2 && cyr > wl + (cg - cyr)) || (cyr >= 1 && cyr_line && lookalikes_only);
        for c in word.chars() {
            let c = if to_cyr {
                CYR_FROM
                    .chars()
                    .zip(CYR_TO.chars())
                    .find(|m| m.0 == c)
                    .map_or(c, |m| m.1)
            } else if fold || (fold_cg && ('\u{0370}'..='\u{04ff}').contains(&c)) {
                latin_lookalike(c, upper).unwrap_or(c)
            } else {
                c
            };
            let u = c as u32;
            let digit = DIGIT_ZEROS.iter().find(|&&z| (z..z + 10).contains(&u));
            match digit {
                Some(&z) if latin_line => out.push(char::from(b'0' + (u - z) as u8)),
                _ => out.push(c),
            }
        }
    }
    out
}

/// A single short token ("1", "12", "x3"): no space, at least one letter or digit, at most
/// six characters. Faint detector candidates are only kept as one.
pub(crate) fn is_short_token(text: &str) -> bool {
    let t = text.trim();
    !t.is_empty()
        && t.chars().count() <= 6
        && !t.chars().any(char::is_whitespace)
        && t.chars().any(char::is_alphanumeric)
}

/// A detector false positive on a picture: no letter or digit, or 1-2 unsure characters.
pub(crate) fn is_junk(text: &str, conf: f32) -> bool {
    const JUNK_CONF: f32 = 0.8;
    let word = text.chars().filter(|c| c.is_alphanumeric()).count();
    word == 0 || (word <= 2 && conf < JUNK_CONF)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quad(x0: f32, y0: f32, x1: f32, y1: f32) -> Quad {
        [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    }

    #[test]
    fn short_tokens() {
        for t in ["1", "12", " 7 ", "x3", "€1", "1,0"] {
            assert!(is_short_token(t), "{t:?}");
        }
        for t in ["", "|", "--", "1 2", "1234567"] {
            assert!(!is_short_token(t), "{t:?}");
        }
    }

    #[test]
    fn homoglyphs_fold_only_inside_latin_words() {
        assert_eq!(clean_latin("pt6а-42 Привет"), "pt6a-42 Привет");
    }

    #[test]
    fn clean_latin_folds_other_scripts_in_latin_lines() {
        // world-recognizer readings of Latin words
        assert_eq!(clean_latin("REPՍBLQՍE FRANÇAISE"), "REPUBLQUE FRANÇAISE");
        assert_eq!(
            clean_latin("C๐mmunicatoทร Security"),
            "Communicatons Security"
        );
        // a lone lookalike word in a Latin line
        assert_eq!(clean_latin("В - Linear 1- Center"), "B - Linear 1- Center");
        // other scripts' digits in a Latin line
        assert_eq!(
            clean_latin("KLAR 08 (UA०८ SHIP TERM"),
            "KLAR 08 (UA08 SHIP TERM"
        );
        assert_eq!(clean_latin("GSTIN: 27ADC300६/१ZK"), "GSTIN: 27ADC3006/1ZK");
        // small-cap / italic Cyrillic forms in Latin words
        assert_eq!(clean_latin("ZQX272003тOKEN OнMS"), "ZQX272003TOKEN OHMS");
        assert_eq!(
            clean_latin("airborлe пoise to гeduce"),
            "airborne noise to reduce"
        );
        // Latin / Greek lookalikes inside Cyrillic words
        assert_eq!(
            clean_latin("aрганізацый ФOPYMA ПяТAГΑ"),
            "арганізацый ФОРУМА ПяТАГА"
        );
        // a Greek letter in an English line stays Greek
        assert_eq!(
            clean_latin("significant at α = 0.05 level"),
            "significant at α = 0.05 level"
        );
    }

    #[test]
    fn clean_latin_leaves_other_script_lines() {
        for s in [
            "Привет мир, как дела",
            "ReturnToWorkSA ਦਾ ਨਿਯਮ ਸਮੂਹ (Regulation Group) ਰਿਟਰਨ ਟੂ ਵਰਕ",
            "३ मार्च २०२४ को",
            "การบริหารความเสี่ยง ๒๕๖๗",
            "Հայաստանի Հանրապետություն",
            "ติดตั้ง Windowsนี้ ได้ที่ร้าน",
        ] {
            assert_eq!(clean_latin(s), s);
        }
    }

    #[test]
    fn clean_latin_keeps_the_whitespace_it_was_given() {
        assert_eq!(clean_latin(" a  b "), " a  b ");
        assert_eq!(clean_latin(""), "");
    }

    #[test]
    fn reading_frame_maps_quarter_turn_back() {
        let q = quad(0.0, 0.0, 10.0, 2.0);
        let r = reading_frame(&q, false, true, 100);
        assert_eq!(r[0], [99.0, 0.0]);
        assert_eq!(r[1], [99.0, 10.0]);
        assert_eq!(r[2], [97.0, 10.0]);
        assert_eq!(r[3], [97.0, 0.0]);
        // without the turn the quad is the one the detector gave
        assert_eq!(reading_frame(&q, false, false, 100), q);
    }

    #[test]
    fn reading_frame_flips_an_upside_down_page() {
        let q = quad(0.0, 0.0, 10.0, 2.0);
        let r = reading_frame(&q, true, false, 100);
        assert_eq!(r, [[10.0, 2.0], [0.0, 2.0], [0.0, 0.0], [10.0, 0.0]]);
        // both: flip first, then the quarter turn
        let both = reading_frame(&q, true, true, 100);
        assert_eq!(both[0], [97.0, 10.0]);
        assert_eq!(both[1], [97.0, 0.0]);
    }

    #[test]
    fn rot90_and_reading_frame_are_inverses_on_an_asymmetric_page() {
        // The sideways retry detects on `page.rot90(1)` and maps boxes back with
        // `reading_frame(.., sideways = true, page_w)`. Pixel by pixel the two must undo
        // each other: a clockwise `rot90`, or a `y = h0 - 1 - x` inverse, fails at once.
        let (w0, h0) = (6usize, 4usize);
        let data: Vec<u8> = (0..w0 * h0).map(|i| (i * 7 + 3) as u8).collect();
        let page = Img {
            w: w0,
            h: h0,
            ch: 1,
            data,
        };
        let turned = page.rot90(1);
        assert_eq!((turned.w, turned.h), (h0, w0));
        for y in 0..turned.h {
            for x in 0..turned.w {
                let q = [[x as f32, y as f32]; 4];
                let back = reading_frame(&q, false, true, w0)[0];
                assert_eq!(
                    turned.at(x, y, 0),
                    page.at(back[0] as usize, back[1] as usize, 0),
                    "({x},{y}) -> {back:?}"
                );
            }
        }
    }

    #[test]
    fn reading_frame_reproduces_the_recorded_sideways_page() {
        // `tools/oracle/ocria/fixtures/pages/`: the sideways page is the receipt turned a
        // quarter clockwise, and the recorded output polygons are the receipt's polygons
        // mapped back through the reading frame with `w0 = 800` (the page it was given).
        // First line of each file, verbatim.
        let upright: Quad = [
            [55.263157, 50.0],
            [357.23685, 50.0],
            [357.23685, 90.0],
            [55.263157, 90.0],
        ];
        let recorded: Quad = [
            [749.0, 55.263157],
            [749.0, 357.23685],
            [709.0, 357.23685],
            [709.0, 55.263157],
        ];
        let got = reading_frame(&upright, false, true, 800);
        for (a, b) in got.iter().zip(&recorded) {
            assert!(
                (a[0] - b[0]).abs() < 1e-3 && (a[1] - b[1]).abs() < 1e-3,
                "{got:?} vs {recorded:?}"
            );
        }
    }

    #[test]
    fn long_lines_counts_level_lines() {
        let boxes = vec![
            quad(0.0, 0.0, 100.0, 10.0),   // 100 x 10: level and long
            quad(0.0, 20.0, 90.0, 30.0),   // 90 x 10: level and long
            quad(0.0, 40.0, 10.0, 140.0),  // 10 x 100: vertical
            quad(0.0, 160.0, 20.0, 165.0), // 20 wide: under the 30 px floor
            quad(0.0, 170.0, 40.0, 190.0), // 40 x 20: not 3x wider than tall
        ];
        assert_eq!(long_lines(&boxes), 2);
        assert_eq!(long_lines(&[]), 0);
    }

    #[test]
    fn looks_sideways_needs_fifteen_boxes_and_few_level_lines() {
        let vertical: Vec<Quad> = (0..15)
            .map(|i| quad(10.0 * i as f32, 0.0, 10.0 * i as f32 + 8.0, 100.0))
            .collect();
        assert!(looks_sideways(&vertical));
        assert!(!looks_sideways(&vertical[..14]));
        // 4 level lines out of 15 is more than a quarter: not sideways
        let mut level = vertical.clone();
        for (i, b) in level.iter_mut().take(4).enumerate() {
            *b = quad(0.0, 10.0 * i as f32, 100.0, 10.0 * i as f32 + 8.0);
        }
        assert!(!looks_sideways(&level));
    }

    #[test]
    fn crop_of_an_axis_aligned_quad_is_that_region() {
        let mut data = vec![0u8; 64];
        for y in 0..8 {
            for x in 0..8 {
                data[y * 8 + x] = (x + 10 * y) as u8;
            }
        }
        let page = Img {
            w: 8,
            h: 8,
            ch: 1,
            data,
        };
        let c = crop(&page, &quad(1.0, 2.0, 5.0, 6.0));
        assert_eq!((c.w, c.h), (4, 4));
        for y in 0..4 {
            for x in 0..4 {
                assert_eq!(c.at(x, y, 0), page.at(x + 1, y + 2, 0), "({x},{y})");
            }
        }
    }

    #[test]
    fn crop_of_a_vertical_quad_turns_it_counter_clockwise() {
        // 3 wide x 8 tall, every pixel distinct: h / w >= 1.5, so the crop is turned a
        // quarter counter-clockwise, which is the reference's `c.rot90(1)`. A clockwise
        // turn would give page.at(y', 8 - 1 - x') and is caught by every pixel below.
        let mut data = vec![0u8; 24];
        for y in 0..8 {
            for x in 0..3 {
                data[y * 3 + x] = (x + 10 * y) as u8;
            }
        }
        let page = Img {
            w: 3,
            h: 8,
            ch: 1,
            data,
        };
        let c = crop(&page, &quad(0.0, 0.0, 3.0, 8.0));
        assert_eq!((c.w, c.h), (8, 3));
        for y in 0..3 {
            for x in 0..8 {
                assert_eq!(c.at(x, y, 0), page.at(2 - y, x, 0), "({x},{y})");
            }
        }
    }

    #[test]
    fn crop_of_a_tiny_quad_is_at_least_two_pixels() {
        let page = Img {
            w: 4,
            h: 4,
            ch: 1,
            data: vec![5; 16],
        };
        let c = crop(&page, &quad(1.0, 1.0, 1.5, 1.5));
        assert_eq!((c.w, c.h), (2, 2));
    }

    /// A two-column page: 240 x 120, paper 240, glyph cores 40 with one-pixel 140
    /// anti-aliased borders, three rows of eight glyphs per column.
    fn gutter_page() -> Img {
        let (w, h) = (240, 120);
        let mut data = vec![240u8; w * h];
        let mut glyph = |x: usize, y: usize| {
            for yy in y - 1..y + 13 {
                for xx in x - 1..x + 5 {
                    data[yy * w + xx] = 140;
                }
            }
            for yy in y..y + 12 {
                for xx in x..x + 4 {
                    data[yy * w + xx] = 40;
                }
            }
        };
        for &row in &[22usize, 50, 78] {
            for k in 0..8 {
                glyph(30 + 8 * k, row);
                glyph(150 + 8 * k, row);
            }
        }
        Img { w, h, ch: 1, data }
    }

    #[test]
    fn split_gutters_cuts_a_line_spanning_two_columns() {
        let page = gutter_page();
        let boxes = vec![
            quad(24.0, 18.0, 96.0, 38.0),
            quad(144.0, 18.0, 216.0, 38.0),
            quad(24.0, 46.0, 96.0, 66.0),
            quad(144.0, 46.0, 216.0, 66.0),
            // a merged detection across the gutter, on the third row
            quad(24.0, 74.0, 216.0, 94.0),
        ];
        let out = split_gutters(&page, &boxes);
        assert_eq!(out.len(), 6, "{out:?}");
        assert_eq!(&out[..4], &boxes[..4]);
        assert_eq!(out[4], quad(24.0, 74.0, 93.0, 94.0));
        assert_eq!(out[5], quad(147.0, 74.0, 216.0, 94.0));
    }

    #[test]
    fn split_gutters_leaves_a_single_column_alone() {
        let page = gutter_page();
        // the same five boxes, but no box is wider than four line heights
        let boxes = vec![
            quad(24.0, 18.0, 96.0, 38.0),
            quad(144.0, 18.0, 216.0, 38.0),
            quad(24.0, 46.0, 96.0, 66.0),
        ];
        assert_eq!(split_gutters(&page, &boxes), boxes);
        // fewer than three boxes are never examined
        assert_eq!(split_gutters(&page, &boxes[..2]), boxes[..2]);
        // a narrow box in a wide page is left alone
        let wide = vec![
            quad(24.0, 18.0, 40.0, 38.0),
            quad(144.0, 18.0, 216.0, 38.0),
            quad(24.0, 46.0, 96.0, 66.0),
        ];
        assert_eq!(split_gutters(&page, &wide), wide);
    }

    #[test]
    fn is_junk_drops_empty_and_unsure_readings() {
        // no letter or digit at all
        assert!(is_junk("", 0.99));
        assert!(is_junk("|", 0.99));
        assert!(is_junk(" - ", 0.99));
        // 1-2 characters are junk below the 0.8 confidence bound
        assert!(is_junk("ab", 0.79));
        assert!(is_junk("12", 0.0));
        assert!(!is_junk("ab", 0.8));
        assert!(!is_junk("ab", 0.95));
        // two characters are junk whatever the confidence, and three or more are kept
        assert!(is_junk("1 2", 0.1));
        assert!(!is_junk("abc", 0.1));
        assert!(!is_junk("abc def", 0.1));
    }

    #[test]
    fn script_chars_counts_each_groups_ranges() {
        // Latin: ASCII letters, Latin-1 / Extended-A/B, and Greek / Cyrillic lookalikes
        assert_eq!(script_chars(0, "Abc 123 ÀÉ"), 5);
        assert_eq!(script_chars(0, "αβ"), 2);
        assert_eq!(script_chars(0, "中文"), 0);
        // CJK: kana, Han, Hangul, compatibility ideographs
        assert_eq!(script_chars(1, "中文ひらがなカナ한글"), 10);
        assert_eq!(script_chars(1, "abc"), 0);
        // right-to-left: Hebrew, Arabic, Syriac, Thaana, presentation forms
        assert_eq!(script_chars(2, "שלום"), 4);
        assert_eq!(script_chars(2, "مرحبا"), 5);
        // the world's other scripts
        assert_eq!(script_chars(3, "हिन्दी"), 6);
        assert_eq!(script_chars(3, "ไทย"), 3);
        assert_eq!(script_chars(3, "ქართული"), 7);
        assert_eq!(script_chars(3, "Հայ"), 3);
        assert_eq!(script_chars(3, "abc"), 0);
    }

    #[test]
    fn latin_letters_counts_only_latin() {
        assert_eq!(latin_letters("Abc ÀÉ αβ 中文"), 5);
        assert_eq!(latin_letters("Привет"), 0);
        assert_eq!(latin_letters(""), 0);
    }

    #[test]
    fn majority_is_the_most_common_group_lowest_index_on_ties() {
        assert_eq!(majority(&[0, 1, 1, 2]), 1);
        assert_eq!(majority(&[3, 3, 1]), 3);
        // equal counts: the lower index wins, and an unmentioned group counts as zero
        assert_eq!(majority(&[2, 1]), 1);
        assert_eq!(majority(&[]), 0);
    }

    #[test]
    fn mean_is_the_arithmetic_mean_and_nan_when_empty() {
        assert_eq!(mean([1.0, 2.0].into_iter()), 1.5);
        assert!(mean(std::iter::empty()).is_nan());
    }
}
