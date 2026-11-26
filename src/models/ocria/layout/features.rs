//! Unit features and the page raster for LayoutNet.
//!
//! Feature layout `[N, 68]`: 21 geometry / text / style, 32 Fourier terms of the
//! centre, 8 projection signals (always zero on this path), 5 ruling-line flags,
//! then block id and leaf id.
//!
//! A port of the reference runtime's `table_model/features.rs` (`tn_core.unit_features`,
//! `pair_inputs` and `rules.py`). The reference works in PDF points against a 150 dpi
//! render; this path has a page image and no dpi, so the page's pixels are the coordinate
//! system everything is expressed in (units, ruling lines, page size) and the only place a
//! physical length survives is the ruling-line thickness cap, which [`process_raster`]
//! turns into pixels from the page's long side.
//!
//! The last two columns are the unit's block and leaf ids, `-1` when it has no projection
//! signal. `-1` is the reference's sentinel and it has to stay negative: the model's pair
//! geometry reads `id_i == id_j && id_i >= 0`, so writing `0` for "no id" would make every
//! pair on the page claim the same block.

/// LiteParse's grid-projection signals for one unit.
///
/// The OCR path has none: the ablation in `tools/oracle/ocria/layout-plan.md`
/// shows the table model reproduces the reference's table with these absent, so
/// `FUnit::proj` is always `None` here and the eight features stay zero.
#[derive(Clone, Debug, Default)]
pub(crate) struct Proj {
    pub feat_text: Option<String>,
    pub flow: f32,
    pub lc: f32,
    pub rc: f32,
    pub cc: f32,
    pub snap: char,
    pub block: f32,
    pub leaf: f32,
}

/// A text unit with what the features need.
#[derive(Clone, Debug)]
pub(crate) struct FUnit {
    pub bx: [f32; 4],
    pub pbox: [f32; 4],
    pub text: String,
    pub h: f32,
    pub bold: Option<bool>,
    pub proj: Option<Proj>,
}

/// Ruling lines in page points: horizontal `[y, x0, x1]`, vertical `[x, y0, y1]`.
#[derive(Clone, Debug, Default)]
pub(crate) struct Rules {
    pub h: Vec<[f32; 3]>,
    pub v: Vec<[f32; 3]>,
}

/// The page raster the image branch reads: ink on a square canvas.
#[derive(Clone, Debug, Default)]
pub(crate) struct PageImage {
    pub w: usize,
    pub h: usize,
    pub px: Vec<u8>,
    pub k: f32,
}

/// The image branch's input tensor.
#[derive(Clone, Debug, Default)]
pub(crate) struct ImgInput {
    pub canvas: Vec<f32>,
    pub boxes: Vec<[f32; 4]>,
    pub valid: (f32, f32),
}

pub(crate) const N_FOURIER: usize = 8;
pub(crate) const FEAT_DIM: usize = 21 + 4 * N_FOURIER + 8 + 5 + 2;

/// Side of the square canvas the image branch reads (`tablenet.py`'s `IMG_SIDE`).
///
/// The network patchifies this canvas at stride 16, so it is a model constant:
/// `net.rs` reads it for the CNN and [`PageImage::input`] puts the page's ink in
/// its top-left corner.
pub(crate) const IMG_SIDE: usize = 1024;

/// Ruling lines are dropped above this thickness, in points (`features.rs:424`).
const THICK_MAX_PT: f32 = 2.5;

/// The page long side, in points, [`process_raster`] assumes to turn
/// [`THICK_MAX_PT`] into pixels: A4's 842 pt.
///
/// The reference's `scale` is the render's dpi / 72, which an image-only caller does
/// not have, so it is reconstructed from the page's long side. That reproduces the
/// reference exactly on a 150 dpi A4 render (cap 5.2 px) and keeps the cap at a
/// constant fraction of the page at any other resolution. The `max(1, ..)` floor
/// treats a page shorter than 842 px as a 72 dpi scan, so the cap never falls below
/// [`THICK_MAX_PT`] pixels — without it, a small crop would lose its own 2 px grid.
/// Measured consequence of the floor: on synthetic pages under 842 px with 3-5 px
/// bars we keep fewer rules than a 150 dpi interpretation of the same page would
/// (700x500: 1 horizontal + 0 vertical here, 3 + 1 at 150 dpi), while the committed
/// fixture's 2 px table grid survives in both.
const ASSUMED_PAGE_LONG_SIDE_PT: f32 = 842.0;

/// The median of `v` (`0.0` for an empty slice). `total_cmp` keeps NaN from
/// panicking, unlike the reference's `partial_cmp().unwrap()`.
fn median_f32(v: &[f32]) -> f32 {
    let mut s = v.to_vec();
    s.sort_by(f32::total_cmp);
    let n = s.len();
    if n == 0 {
        return 0.0;
    }
    if n % 2 == 1 {
        s[n / 2]
    } else {
        (s[n / 2 - 1] + s[n / 2]) / 2.0
    }
}

/// `tn_core._NUM`: `^[\s(\[$€£¥+\-–−±<>≤≥~]*[\d.,]+[%)\]*†‡§a-z]*$` (case-insensitive).
fn is_numeric_text(t: &str) -> bool {
    let cs: Vec<char> = t.chars().collect();
    let lead = |c: char| c.is_whitespace() || "([$€£¥+-–−±<>≤≥~".contains(c);
    let body = |c: char| {
        c.is_ascii_digit() || c == '.' || c == ',' || (c.is_numeric() && c.is_ascii_digit())
    };
    let tail = |c: char| "%)]*†‡§".contains(c) || c.is_ascii_alphabetic();
    let mut i = 0;
    while i < cs.len() && lead(cs[i]) && !body(cs[i]) {
        i += 1;
    }
    // the body needs at least one char; regex backtracking can give leading chars back only if
    // they could be body chars, and none of the lead set is a digit, '.' or ','
    let start = i;
    while i < cs.len() && body(cs[i]) {
        i += 1;
    }
    if i == start {
        return false;
    }
    cs[i..].iter().all(|&c| tail(c))
}

/// Python `str.isdigit` (Unicode decimal / digit), close enough for features.
fn is_digit(c: char) -> bool {
    c.is_ascii_digit() || matches!(c, '²' | '³' | '¹' | '⁰'..='⁹' | '₀'..='₉')
}

/// `[N, 4]`: a horizontal rule just above / below the unit, a vertical rule just left / right.
fn unit_rule_features(boxes: &[[f32; 4]], rules: &Rules, h: &[f32]) -> Vec<[f32; 4]> {
    let mut out = vec![[0f32; 4]; boxes.len()];
    for (i, b) in boxes.iter().enumerate() {
        let hh = h.get(i).copied().unwrap_or(0.0).max(0.5);
        for r in &rules.h {
            let ov = b[2].min(r[2]) - b[0].max(r[1]) > 0.0;
            if !ov {
                continue;
            }
            let above = b[1] - r[0];
            let below = r[0] - b[3];
            if above >= -0.3 * hh && above < 1.5 * hh {
                out[i][0] = 1.0;
            }
            if below >= -0.3 * hh && below < 1.5 * hh {
                out[i][1] = 1.0;
            }
        }
        for r in &rules.v {
            let ov = b[3].min(r[2]) - b[1].max(r[1]) > 0.0;
            if !ov {
                continue;
            }
            let left = b[0] - r[0];
            let right = r[0] - b[2];
            if left >= -0.3 * hh && left < 2.0 * hh {
                out[i][2] = 1.0;
            }
            if right >= -0.3 * hh && right < 2.0 * hh {
                out[i][3] = 1.0;
            }
        }
    }
    out
}

/// `[N, N, 2]`: rules strictly between two units (spanning both), counts capped at 3, scaled.
fn pair_rule_features(boxes: &[[f32; 4]], rules: &Rules) -> Vec<f32> {
    let n = boxes.len();
    let mut out = vec![0f32; n * n * 2];
    let cap = 3.0f32;
    let cy: Vec<f32> = boxes.iter().map(|b| (b[1] + b[3]) / 2.0).collect();
    let cx: Vec<f32> = boxes.iter().map(|b| (b[0] + b[2]) / 2.0).collect();
    // covers[i][r]: rule r runs under unit i's x-range (h) / beside its y-range (v)
    let hcov: Vec<Vec<bool>> = boxes
        .iter()
        .map(|b| {
            rules
                .h
                .iter()
                .map(|r| r[1] <= b[2] && r[2] >= b[0])
                .collect()
        })
        .collect();
    let vcov: Vec<Vec<bool>> = boxes
        .iter()
        .map(|b| {
            rules
                .v
                .iter()
                .map(|r| r[1] <= b[3] && r[2] >= b[1])
                .collect()
        })
        .collect();
    for i in 0..n {
        for j in 0..n {
            if !rules.h.is_empty() {
                let (lo, hi) = (cy[i].min(cy[j]), cy[i].max(cy[j]));
                let c = rules
                    .h
                    .iter()
                    .enumerate()
                    .filter(|(r, rr)| rr[0] > lo && rr[0] < hi && hcov[i][*r] && hcov[j][*r])
                    .count();
                out[(i * n + j) * 2] = (c as f32).min(cap) / cap;
            }
            if !rules.v.is_empty() {
                let (lo, hi) = (cx[i].min(cx[j]), cx[i].max(cx[j]));
                let c = rules
                    .v
                    .iter()
                    .enumerate()
                    .filter(|(r, rr)| rr[0] > lo && rr[0] < hi && vcov[i][*r] && vcov[j][*r])
                    .count();
                out[(i * n + j) * 2 + 1] = (c as f32).min(cap) / cap;
            }
        }
    }
    out
}

/// `[N, FEAT_DIM]` unit features.
pub(crate) fn unit_features(units: &[FUnit], wh: (f32, f32), rules: Option<&Rules>) -> Vec<f32> {
    let (w_pg, h_pg) = wh;
    let n = units.len();
    let mut f = vec![0f32; n * FEAT_DIM];
    if n == 0 {
        return f;
    }
    let b: Vec<[f32; 4]> = units.iter().map(|u| u.bx).collect();
    let h: Vec<f32> = units.iter().map(|u| u.h.max(0.5)).collect();
    let hm = median_f32(&h);
    let mid: Vec<f32> = b.iter().map(|x| (x[1] + x[3]) / 2.0).collect();
    let rule_f = rules.map(|r| {
        let pb: Vec<[f32; 4]> = units.iter().map(|u| u.pbox).collect();
        let hs: Vec<f32> = units.iter().map(|u| u.h).collect();
        unit_rule_features(&pb, r, &hs)
    });
    for i in 0..n {
        let mut n_row = 0f32;
        let (mut gap_l, mut gap_r) = (20f32, 20f32);
        let (mut lmax, mut rmin) = (f32::NEG_INFINITY, f32::INFINITY);
        let (mut has_l, mut has_r) = (false, false);
        for j in 0..n {
            let same = (mid[j] - mid[i]).abs() < 0.5 * h[j].max(h[i]);
            if !same {
                continue;
            }
            n_row += 1.0;
            if j == i {
                continue;
            }
            if b[j][2] <= b[i][0] + 0.5 * h[i] {
                has_l = true;
                lmax = lmax.max(b[j][2]);
            }
            if b[j][0] >= b[i][2] - 0.5 * h[i] {
                has_r = true;
                rmin = rmin.min(b[j][0]);
            }
        }
        if has_l {
            gap_l = 20f32.min((b[i][0] - lmax) / hm);
        }
        if has_r {
            gap_r = 20f32.min((rmin - b[i][2]) / hm);
        }
        let cx = (b[i][0] + b[i][2]) / 2.0 / w_pg;
        let cy = (b[i][1] + b[i][3]) / 2.0 / h_pg;
        let t = units[i]
            .proj
            .as_ref()
            .and_then(|p| p.feat_text.as_deref())
            .unwrap_or(&units[i].text);
        let chars: Vec<char> = t.chars().collect();
        let len = chars.len() as f32;
        let n_alpha = chars.iter().filter(|c| c.is_alphabetic()).count() as f32;
        let dig = chars.iter().filter(|&&c| is_digit(c)).count() as f32 / len.max(1.0);
        let alpha = n_alpha / len.max(1.0);
        let upper = chars.iter().filter(|c| c.is_uppercase()).count() as f32 / n_alpha.max(1.0);
        let words = t.split_whitespace().count() as f32;
        let numeric = if is_numeric_text(t) { 1.0 } else { 0.0 };
        let tr = t.trim_end();
        let endp =
            if tr.ends_with('.') || tr.ends_with(':') || tr.ends_with(';') || tr.ends_with(',') {
                1.0
            } else {
                0.0
            };
        let row = &mut f[i * FEAT_DIM..(i + 1) * FEAT_DIM];
        let base = [
            cx,
            cy,
            (b[i][2] - b[i][0]) / w_pg,
            (b[i][3] - b[i][1]) / h_pg,
            b[i][0] / w_pg,
            b[i][2] / w_pg,
            b[i][1] / h_pg,
            b[i][3] / h_pg,
            h[i] / hm,
            len.ln_1p() / 5.0,
            words.ln_1p() / 3.0,
            dig,
            alpha,
            upper,
            numeric,
            endp,
            n_row.ln_1p() / 3.0,
            gap_l / 20.0,
            gap_r / 20.0,
            if units[i].bold == Some(true) {
                1.0
            } else {
                0.0
            },
            if units[i].bold.is_none() { 0.0 } else { 1.0 },
        ];
        row[..21].copy_from_slice(&base);
        let mut k = 21;
        for q in 0..N_FOURIER {
            let fr = (2f32).powi(q as i32) * std::f32::consts::PI;
            row[k] = (fr * cx).sin();
            row[k + 1] = (fr * cx).cos();
            row[k + 2] = (fr * cy).sin();
            row[k + 3] = (fr * cy).cos();
            k += 4;
        }
        if let Some(p) = &units[i].proj {
            row[k] = 1.0;
            row[k + 1] = p.flow;
            row[k + 2] = p.lc.ln_1p() / 3.0;
            row[k + 3] = p.rc.ln_1p() / 3.0;
            row[k + 4] = p.cc.ln_1p() / 3.0;
            row[k + 5] = (p.snap == 'L') as u8 as f32;
            row[k + 6] = (p.snap == 'R') as u8 as f32;
            row[k + 7] = (p.snap == 'C') as u8 as f32;
        }
        k += 8;
        if let Some(rf) = &rule_f {
            row[k..k + 4].copy_from_slice(&rf[i]);
            row[k + 4] = 1.0;
        }
        k += 5;
        // -1 is the reference's "no id" (`features.rs:298-299`); `net.rs`'s
        // pair geometry reads it as "never the same block / leaf".
        row[k] = units[i].proj.as_ref().map_or(-1.0, |p| p.block);
        row[k + 1] = units[i].proj.as_ref().map_or(-1.0, |p| p.leaf);
    }
    f
}

/// `[N, N, 2]` pairwise features.
pub(crate) fn pair_features(units: &[FUnit], _wh: (f32, f32), rules: Option<&Rules>) -> Vec<f32> {
    let n = units.len();
    match rules {
        Some(rules) => {
            let pb: Vec<[f32; 4]> = units.iter().map(|u| u.pbox).collect();
            pair_rule_features(&pb, rules)
        }
        None => vec![0f32; n * n * 2],
    }
}

/// Gray threshold below which a pixel is ink.
const DARK: u8 = 170;

/// Ink runs per scanline, `(start, end)` inclusive: one row of the table per image
/// row (horizontal runs) or per image column (vertical runs).
type Runs = Vec<Vec<(usize, usize)>>;

/// A page's ink mask (grey < [`DARK`]) as one bitset per row, so the run scan only visits the
/// pixels where ink starts or stops (pages are mostly paper).
struct InkBits {
    w: usize,
    h: usize,
    words: usize,
    bits: Vec<u64>,
}

impl InkBits {
    fn from_grey(gray: &[u8], w: usize, h: usize) -> Self {
        let words = w.div_ceil(64);
        let mut bits = vec![0u64; words * h];
        for (row, dst) in gray
            .chunks_exact(w.max(1))
            .take(h)
            .zip(bits.chunks_exact_mut(words.max(1)))
        {
            for (k, px) in row.chunks(64).enumerate() {
                let mut word = 0u64;
                for (b, &g) in px.iter().enumerate() {
                    word |= ((g < DARK) as u64) << b;
                }
                dst[k] = word;
            }
        }
        InkBits { w, h, words, bits }
    }

    fn row(&self, y: usize) -> &[u64] {
        &self.bits[y * self.words..(y + 1) * self.words]
    }

    /// Horizontal ink runs per row and vertical ink runs per column, as `(start, end inclusive)`
    /// in scan order, keeping those at least 3% of the page side (15 px floor) long.
    fn runs(&self) -> (Runs, Runs) {
        let (w, h, words) = (self.w, self.h, self.words);
        let lh = 15usize.max((w as f32 * 0.03) as usize);
        let lv = 15usize.max((h as f32 * 0.03) as usize);
        let mut hruns: Vec<Vec<(usize, usize)>> = vec![Vec::new(); h];
        let mut vruns: Vec<Vec<(usize, usize)>> = vec![Vec::new(); w];
        // where each column's current vertical run began (valid while its bit was set last row)
        let mut open = vec![0usize; w];
        let zero = vec![0u64; words];
        for y in 0..h {
            let cur = self.row(y);
            let prev = if y == 0 { &zero[..] } else { self.row(y - 1) };
            // vertical runs start or end where a column's ink differs from the row above
            for k in 0..words {
                let mut changed = cur[k] ^ prev[k];
                while changed != 0 {
                    let x = k * 64 + changed.trailing_zeros() as usize;
                    changed &= changed - 1;
                    if cur[k] >> (x % 64) & 1 == 1 {
                        open[x] = y;
                    } else if y - open[x] >= lv {
                        vruns[x].push((open[x], y - 1));
                    }
                }
            }
            // horizontal runs: maximal stretches of set bits
            let mut x = next_bit(cur, 0, w, true);
            while x < w {
                let end = next_bit(cur, x, w, false);
                if end - x >= lh {
                    hruns[y].push((x, end - 1));
                }
                x = next_bit(cur, end, w, true);
            }
        }
        if h > 0 {
            let last = self.row(h - 1);
            for (x, &o) in open.iter().enumerate() {
                if last[x / 64] >> (x % 64) & 1 == 1 && h - o >= lv {
                    vruns[x].push((o, h - 1));
                }
            }
        }
        (hruns, vruns)
    }
}

/// The first position at or after `from` (below `w`) whose bit is `set`; `w` when there is none.
fn next_bit(row: &[u64], from: usize, w: usize, set: bool) -> usize {
    let mut k = from / 64;
    if k >= row.len() {
        return w;
    }
    let flip = if set { 0 } else { u64::MAX };
    let mut word = (row[k] ^ flip) & (u64::MAX << (from % 64));
    loop {
        if word != 0 {
            return (k * 64 + word.trailing_zeros() as usize).min(w);
        }
        k += 1;
        if k >= row.len() {
            return w;
        }
        word = row[k] ^ flip;
    }
}

/// The union-find root of `k`.
fn root(p: &mut [usize], mut k: usize) -> usize {
    while p[k] != k {
        p[k] = p[p[k]];
        k = p[k];
    }
    k
}

/// Ruling lines from ink runs: each 8-connected component of runs on adjacent rows (columns) is
/// one rule, unless it is thicker than `thick_max_px` pixels.
fn rules_from_runs((hruns, vruns): (Runs, Runs), scale: f32, thick_max_px: f32) -> Rules {
    let mut out = Rules::default();
    for (axis, runs) in [(0, hruns), (1, vruns)] {
        let lines_n = runs.len();
        // 8-connected components over runs of adjacent lines (union-find)
        let mut ids: Vec<(usize, usize, usize)> = Vec::new(); // (line, start, end)
        let mut first: Vec<usize> = vec![0; lines_n + 1];
        for (line, rs) in runs.iter().enumerate() {
            first[line] = ids.len();
            for &(s, e) in rs {
                ids.push((line, s, e));
            }
        }
        first[lines_n] = ids.len();
        let mut parent: Vec<usize> = (0..ids.len()).collect();
        for line in 1..lines_n {
            for a in first[line - 1]..first[line] {
                for b in first[line]..first[line + 1] {
                    let (_, s0, e0) = ids[a];
                    let (_, s1, e1) = ids[b];
                    if s1 <= e0 + 1 && e1 + 1 >= s0 {
                        let (ra, rb) = (root(&mut parent, a), root(&mut parent, b));
                        if ra != rb {
                            parent[ra] = rb;
                        }
                    }
                }
            }
        }
        // component extents, in root order (deterministic)
        let mut comp: Vec<Option<(usize, usize, usize, usize)>> = vec![None; ids.len()];
        for k in 0..ids.len() {
            let r = root(&mut parent, k);
            let (line, s, e) = ids[k];
            let c = comp[r].get_or_insert((usize::MAX, 0, usize::MAX, 0));
            c.0 = c.0.min(line);
            c.1 = c.1.max(line);
            c.2 = c.2.min(s);
            c.3 = c.3.max(e);
        }
        let round1 = |v: f32| (v * 10.0).round() / 10.0;
        for (l0, l1, s, e) in comp.into_iter().flatten() {
            let thick = (l1 - l0 + 1) as f32;
            if thick > thick_max_px {
                continue;
            }
            let along = (e - s + 1) as f32;
            let mid = (l0 as f32 + thick / 2.0) / scale;
            let (a0, a1) = (s as f32 / scale, (s as f32 + along) / scale);
            if axis == 0 {
                out.h.push([round1(mid), round1(a0), round1(a1)]);
            } else {
                out.v.push([round1(mid), round1(a0), round1(a1)]);
            }
        }
    }
    out
}

/// Ruling lines from a greyscale page.
///
/// `scale` is raster pixels per page point: rule coordinates come back in page
/// points (pixel positions divided by `scale`), the units the unit boxes are in,
/// and the thickness cap is [`THICK_MAX_PT`] points. Pass `1.0` when the boxes
/// are already in pixels.
///
/// The model's own path enters through [`process_raster`], which keeps the rule
/// coordinates in page pixels; this is the reference's scale-aware entry point,
/// kept for callers that know their dpi and for the tests.
#[allow(dead_code)] // the OCR path goes through `process_raster`
pub(crate) fn detect_rules(gray: &[u8], w: usize, h: usize, scale: f32) -> Rules {
    if w == 0 || h == 0 {
        return Rules::default();
    }
    let scale = if scale > 0.0 { scale } else { 1.0 };
    let ink = InkBits::from_grey(gray, w, h);
    rules_from_runs(ink.runs(), scale, THICK_MAX_PT * scale)
}

/// Raster pixels per page point, from the page's long side (see
/// [`ASSUMED_PAGE_LONG_SIDE_PT`]).
fn px_per_point(w: usize, h: usize) -> f32 {
    (w.max(h) as f32 / ASSUMED_PAGE_LONG_SIDE_PT).max(1.0)
}

/// The page in grey (0 = black), from RGB or already-grey bytes.
///
/// Grey follows training exactly: OpenCV's RGB2GRAY integer weights, which is what
/// the reference applies to its render (`features.rs:613-629`).
fn grey(rgb: &[u8], w: usize, h: usize) -> Option<Vec<u8>> {
    let n = w.checked_mul(h)?;
    if rgb.len() == n {
        return Some(rgb.to_vec());
    }
    if rgb.len() != n.checked_mul(3)? {
        return None;
    }
    let mut out = vec![0u8; n];
    for (o, p) in out.iter_mut().zip(rgb.chunks_exact(3)) {
        let v = 4899 * p[0] as u32 + 9617 * p[1] as u32 + 1868 * p[2] as u32 + (1 << 13);
        *o = (v >> 14) as u8;
    }
    Some(out)
}

/// Area-average resample (OpenCV INTER_AREA when shrinking) of a grey image to (tw, th).
fn resample_area(g: &[u8], w: usize, h: usize, tw: usize, th: usize) -> Vec<u8> {
    // separable box filter with fractional coverage: rows, then columns. Each output's taps are
    // (source index, weight) in source order, flattened, with `at[o]..at[o + 1]` its range.
    let weights = |n_in: usize, n_out: usize| -> (Vec<usize>, Vec<(usize, f32)>) {
        let f = n_in as f32 / n_out as f32;
        let mut at = vec![0];
        let mut taps = Vec::new();
        for o in 0..n_out {
            let (a, b) = (o as f32 * f, (o as f32 + 1.0) * f);
            let mut i = a.floor() as usize;
            while (i as f32) < b && i < n_in {
                let cov = (b.min(i as f32 + 1.0) - a.max(i as f32)).max(0.0);
                if cov > 0.0 {
                    taps.push((i, cov / f));
                }
                i += 1;
            }
            at.push(taps.len());
        }
        (at, taps)
    };
    let ((ax, wx), (ay, wy)) = (weights(w, tw), weights(h, th));
    let mut tmp = vec![0f32; h * tw];
    for (row, dst) in g.chunks_exact(w).zip(tmp.chunks_exact_mut(tw)) {
        for (x, d) in dst.iter_mut().enumerate() {
            let mut s = 0f32;
            for &(i, c) in &wx[ax[x]..ax[x + 1]] {
                s += row[i] as f32 * c;
            }
            *d = s;
        }
    }
    // columns: the same per-output sums in the same tap order, accumulated a row at a time
    let mut out = vec![0u8; th * tw];
    let mut acc = vec![0f32; tw];
    for (y, dst) in out.chunks_exact_mut(tw).enumerate() {
        acc.iter_mut().for_each(|a| *a = 0.0);
        for &(i, c) in &wy[ay[y]..ay[y + 1]] {
            for (a, &v) in acc.iter_mut().zip(&tmp[i * tw..(i + 1) * tw]) {
                *a += v * c;
            }
        }
        for (d, &a) in dst.iter_mut().zip(&acc) {
            *d = a.round().clamp(0.0, 255.0) as u8;
        }
    }
    out
}

/// Ruling lines and the image-branch raster from a page image.
///
/// The raster is the page's grey, longest side [`IMG_SIDE`], area-resampled from the
/// page image (`features.rs:650-681`); `k` is its scale, image pixels per page pixel.
/// The ruling lines come out in page pixels, the units' coordinate system, with the
/// 2.5 pt thickness cap taken from the page's long side.
pub(crate) fn process_raster(rgb: &[u8], w: usize, h: usize) -> (Rules, PageImage) {
    if w == 0 || h == 0 {
        return (Rules::default(), PageImage::default());
    }
    let Some(gray) = grey(rgb, w, h) else {
        return (Rules::default(), PageImage::default());
    };
    let side = IMG_SIDE as f32;
    let s = side / w.max(h) as f32;
    let (tw, th) = (
        ((w as f32 * s).round() as usize).max(8),
        ((h as f32 * s).round() as usize).max(8),
    );
    let image = PageImage {
        w: tw,
        h: th,
        px: resample_area(&gray, w, h, tw, th),
        k: s,
    };
    let ink = InkBits::from_grey(&gray, w, h);
    let rules = rules_from_runs(ink.runs(), 1.0, THICK_MAX_PT * px_per_point(w, h));
    (rules, image)
}

impl PageImage {
    /// The image branch's input: ink on the square canvas, unit boxes as fractions.
    pub(crate) fn input(&self, pboxes: &[[f32; 4]]) -> ImgInput {
        let side = IMG_SIDE;
        let mut canvas = vec![0f32; side * side];
        for y in 0..self.h.min(side) {
            for x in 0..self.w.min(side) {
                let px = self.px.get(y * self.w + x).copied().unwrap_or(255);
                canvas[y * side + x] = (255.0 - px as f32) / 255.0;
            }
        }
        let f = self.k / side as f32;
        let boxes = pboxes
            .iter()
            .map(|b| [b[0] * f, b[1] * f, b[2] * f, b[3] * f].map(|v| v.clamp(0.0, 1.0)))
            .collect();
        ImgInput {
            canvas,
            boxes,
            valid: (self.w as f32 / side as f32, self.h as f32 / side as f32),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit(bx: [f32; 4], text: &str, h: f32, bold: Option<bool>) -> FUnit {
        FUnit {
            bx,
            pbox: bx,
            text: text.to_string(),
            h,
            bold,
            proj: None,
        }
    }

    /// A white page with black rectangles: `rects` are `(x0, y0, x1, y1)` inclusive.
    fn page(w: usize, h: usize, rects: &[(usize, usize, usize, usize)]) -> Vec<u8> {
        let mut g = vec![255u8; w * h];
        for &(x0, y0, x1, y1) in rects {
            for y in y0..=y1.min(h - 1) {
                for x in x0..=x1.min(w - 1) {
                    g[y * w + x] = 0;
                }
            }
        }
        g
    }

    #[test]
    fn feature_layout_is_68_values() {
        assert_eq!(N_FOURIER, 8);
        assert_eq!(FEAT_DIM, 68);
        assert_eq!(21 + 4 * N_FOURIER + 8 + 5 + 2, 68);
    }

    #[test]
    fn numeric_text_matches_the_reference_class() {
        for t in [
            "1", "12", "1,234.50", "€1", "$ 12", "(12)", "[3.5]", "12%", "-3", "+2.5", "1†", "12a",
            "1.2.3", "≤5", "~7", " 42", "3)", "9]",
        ] {
            assert!(is_numeric_text(t), "{t:?} is numeric");
        }
        for t in [
            "", " ", "abc", "a1", "1 2", "1-2", "†", "%", "12 #", "x12", "1/2", "Item", "1 2 3",
            // the reference's tail class has no whitespace: a trailing space breaks it
            " 42 ",
        ] {
            assert!(!is_numeric_text(t), "{t:?} is not numeric");
        }
    }

    #[test]
    fn digit_class_counts_unicode_digits() {
        for c in ['0', '7', '²', '³', '¹', '⁰', '⁵', '₀', '₉'] {
            assert!(is_digit(c), "{c:?} is a digit");
        }
        for c in ['a', 'A', '.', ',', '½', 'Ⅷ', ' '] {
            assert!(!is_digit(c), "{c:?} is not a digit");
        }
    }

    /// Features 11-15 and 19-20: digit / alpha / upper fractions, the numeric and
    /// end-punctuation classes, and the bold / bold-known flags.
    #[test]
    fn text_and_style_features() {
        let u = [
            unit([0.0, 0.0, 10.0, 10.0], "ABc 12.", 10.0, Some(true)),
            unit([0.0, 20.0, 10.0, 30.0], "12,5%", 10.0, Some(false)),
            unit([0.0, 40.0, 10.0, 50.0], "word", 10.0, None),
        ];
        let f = unit_features(&u, (100.0, 100.0), None);
        let row = |i: usize| &f[i * FEAT_DIM..(i + 1) * FEAT_DIM];
        // "ABc 12." -> 7 chars, 3 alphabetic, 2 digits, 2 uppercase, 2 words, endp '.'
        assert!((row(0)[9] - 7f32.ln_1p() / 5.0).abs() < 1e-6);
        assert!((row(0)[10] - 2f32.ln_1p() / 3.0).abs() < 1e-6);
        assert!((row(0)[11] - 2.0 / 7.0).abs() < 1e-6);
        assert!((row(0)[12] - 3.0 / 7.0).abs() < 1e-6);
        assert!((row(0)[13] - 2.0 / 3.0).abs() < 1e-6);
        assert_eq!(row(0)[14], 0.0, "mixed text is not numeric");
        assert_eq!(row(0)[15], 1.0, "ends in '.'");
        assert_eq!(row(0)[19], 1.0, "bold");
        assert_eq!(row(0)[20], 1.0, "bold known");
        // "12,5%" -> numeric, no end punctuation, bold false
        assert_eq!(row(1)[14], 1.0, "numeric");
        assert_eq!(row(1)[15], 0.0);
        assert_eq!(row(1)[19], 0.0);
        assert_eq!(row(1)[20], 1.0, "bold known (false)");
        // "word" -> no bold information at all
        assert_eq!(row(2)[14], 0.0);
        assert_eq!(row(2)[19], 0.0);
        assert_eq!(row(2)[20], 0.0);
    }

    /// Features 21-52: `sin/cos(2^q π c)` for the centre fraction, 4 per axis per term.
    #[test]
    fn fourier_terms_of_the_centre() {
        let u = [unit([20.0, 40.0, 60.0, 80.0], "x", 20.0, None)];
        let f = unit_features(&u, (200.0, 400.0), None);
        let (cx, cy) = (40.0f32 / 200.0, 60.0f32 / 400.0);
        let mut k = 21;
        for q in 0..N_FOURIER {
            let fr = (2f32).powi(q as i32) * std::f32::consts::PI;
            assert!((f[k] - (fr * cx).sin()).abs() < 1e-6, "q{q} sin x");
            assert!((f[k + 1] - (fr * cx).cos()).abs() < 1e-6, "q{q} cos x");
            assert!((f[k + 2] - (fr * cy).sin()).abs() < 1e-6, "q{q} sin y");
            assert!((f[k + 3] - (fr * cy).cos()).abs() < 1e-6, "q{q} cos y");
            k += 4;
        }
        // q = 0 is the constant term: sin(π c), cos(π c)
        assert!((f[21] - (std::f32::consts::PI * cx).sin()).abs() < 1e-6);
        assert!((f[22] - (std::f32::consts::PI * cx).cos()).abs() < 1e-6);
    }

    /// The whole 68-value row of a two-unit page, computed by hand from the
    /// reference formula (`features.rs:177-302`) with no rules and no projection.
    #[test]
    fn two_unit_page_features_are_hand_computed() {
        let u = [
            unit([10.0, 20.0, 60.0, 40.0], "Item", 20.0, None),
            unit([100.0, 20.0, 160.0, 40.0], "Qty 2", 20.0, Some(true)),
        ];
        let f = unit_features(&u, (200.0, 100.0), None);
        assert_eq!(f.len(), 2 * FEAT_DIM);

        // U0: box 50x20 at (10,20); centre (35,30); one same-line neighbour to the right at
        // x=100 -> gap_r = (100-60)/20 = 2 -> 0.1; no left neighbour -> gap_l stays 20 -> 1.0.
        let mut want0 = vec![0f32; FEAT_DIM];
        want0[0] = 35.0 / 200.0;
        want0[1] = 30.0 / 100.0;
        want0[2] = 50.0 / 200.0;
        want0[3] = 20.0 / 100.0;
        want0[4] = 10.0 / 200.0;
        want0[5] = 60.0 / 200.0;
        want0[6] = 20.0 / 100.0;
        want0[7] = 40.0 / 100.0;
        want0[8] = 1.0; // h / median h
        want0[9] = 4f32.ln_1p() / 5.0; // "Item": 4 chars
        want0[10] = 1f32.ln_1p() / 3.0; // 1 word
        want0[11] = 0.0; // digits
        want0[12] = 1.0; // 4/4 alphabetic
        want0[13] = 0.25; // "I" of 4
        want0[14] = 0.0; // not numeric
        want0[15] = 0.0; // no end punctuation
        want0[16] = 2f32.ln_1p() / 3.0; // both units on the line
        want0[17] = 1.0;
        want0[18] = 0.1;
        want0[19] = 0.0;
        want0[20] = 0.0;
        let (cx0, cy0) = (35.0f32 / 200.0, 30.0f32 / 100.0);
        let (cx1, cy1) = (130.0f32 / 200.0, 30.0f32 / 100.0);
        for q in 0..N_FOURIER {
            let fr = (2f32).powi(q as i32) * std::f32::consts::PI;
            let k = 21 + 4 * q;
            want0[k] = (fr * cx0).sin();
            want0[k + 1] = (fr * cx0).cos();
            want0[k + 2] = (fr * cy0).sin();
            want0[k + 3] = (fr * cy0).cos();
        }
        // no projection signals, no rules; ids are the reference's "absent"
        for v in &mut want0[53..61] {
            *v = 0.0;
        }
        for v in &mut want0[61..66] {
            *v = 0.0;
        }
        want0[66] = -1.0;
        want0[67] = -1.0;

        // U1: box 60x20 at (100,20); "Qty 2" -> 5 chars, 3 alphabetic, 1 digit, 1 upper,
        // 2 words; a left neighbour at x=60 -> gap_l = (100-60)/20 = 2 -> 0.1, no right one.
        let mut want1 = vec![0f32; FEAT_DIM];
        want1[0] = 130.0 / 200.0;
        want1[1] = 30.0 / 100.0;
        want1[2] = 60.0 / 200.0;
        want1[3] = 20.0 / 100.0;
        want1[4] = 100.0 / 200.0;
        want1[5] = 160.0 / 200.0;
        want1[6] = 20.0 / 100.0;
        want1[7] = 40.0 / 100.0;
        want1[8] = 1.0;
        want1[9] = 5f32.ln_1p() / 5.0;
        want1[10] = 2f32.ln_1p() / 3.0;
        want1[11] = 0.2; // one digit of five
        want1[12] = 0.6; // three alphabetic of five
        want1[13] = 1.0 / 3.0; // "Q"
        want1[14] = 0.0;
        want1[15] = 0.0;
        want1[16] = 2f32.ln_1p() / 3.0;
        want1[17] = 0.1;
        want1[18] = 1.0;
        want1[19] = 1.0; // bold
        want1[20] = 1.0; // bold known
        for q in 0..N_FOURIER {
            let fr = (2f32).powi(q as i32) * std::f32::consts::PI;
            let k = 21 + 4 * q;
            want1[k] = (fr * cx1).sin();
            want1[k + 1] = (fr * cx1).cos();
            want1[k + 2] = (fr * cy1).sin();
            want1[k + 3] = (fr * cy1).cos();
        }
        want1[66] = -1.0;
        want1[67] = -1.0;

        for (i, want) in [want0, want1].iter().enumerate() {
            let got = &f[i * FEAT_DIM..(i + 1) * FEAT_DIM];
            for (k, (a, b)) in got.iter().zip(want.iter()).enumerate() {
                assert!((a - b).abs() < 1e-6, "unit {i} feature {k}: {a} != {b}");
            }
        }
    }

    /// Features 61-65 and the pair rule features: rules just outside each edge of a
    /// unit, and rules strictly between two units' centres that span both boxes.
    #[test]
    fn rule_features_flag_edges_and_count_between_pairs() {
        let u = [
            unit([10.0, 20.0, 60.0, 40.0], "a", 20.0, None),
            unit([100.0, 100.0, 150.0, 120.0], "b", 20.0, None),
        ];
        let rules = Rules {
            // above U0 (y=19, 1 above its top), below U0 (y=41, 1 below its bottom),
            // between the two centres and spanning both, and one that misses U1.
            h: vec![
                [19.0, 5.0, 70.0],
                [41.0, 5.0, 70.0],
                [50.0, 0.0, 200.0],
                [70.0, 0.0, 60.0],
            ],
            // left of U0 (x=8), right of U0 (x=61), between the centres spanning both.
            v: vec![[8.0, 15.0, 45.0], [61.0, 15.0, 45.0], [80.0, 0.0, 200.0]],
        };
        let f = unit_features(&u, (200.0, 200.0), Some(&rules));
        let row = |i: usize| &f[i * FEAT_DIM..(i + 1) * FEAT_DIM];
        // U0 has a rule on every side; U1 only the ones that reach it.
        assert_eq!(&row(0)[61..66], &[1.0, 1.0, 1.0, 1.0, 1.0]);
        // U1: above? no (nothing at y in [-6, 130) overlapping x 100..150 -> y=50 spans 0..200,
        // so above = 100-50 = 50 < 30? no) -> 0; below? y=150 none -> 0;
        // left: x=80, left = 100-80 = 20 < 2*20 -> 1; right: none.
        assert_eq!(&row(1)[61..66], &[0.0, 0.0, 1.0, 0.0, 1.0]);

        let pairs = pair_features(&u, (200.0, 200.0), Some(&rules));
        assert_eq!(pairs.len(), 2 * 2 * 2);
        // (0,1) and (1,0): one horizontal rule between the centres spanning both, one vertical
        let one_third = 1.0 / 3.0;
        for (i, j) in [(0, 1), (1, 0)] {
            let k = (i * 2 + j) * 2;
            assert!((pairs[k] - one_third).abs() < 1e-6, "h rule {i}->{j}");
            assert!((pairs[k + 1] - one_third).abs() < 1e-6, "v rule {i}->{j}");
        }
        // a unit against itself has no rule between its own centre and itself
        assert_eq!(pairs[0], 0.0);
        assert_eq!(pairs[1], 0.0);

        // no rules at all: every rule feature is zero
        let zeros = unit_features(&u, (200.0, 200.0), None);
        assert_eq!(&zeros[61..66], &[0.0, 0.0, 0.0, 0.0, 0.0]);
        assert!(pair_features(&u, (200.0, 200.0), None)
            .iter()
            .all(|&v| v == 0.0));
    }

    #[test]
    fn detect_rules_finds_a_ruled_grid() {
        let (w, h) = (240, 160);
        let gray = page(
            w,
            h,
            &[
                (10, 20, 229, 21),   // horizontal rule, 2 px thick
                (10, 100, 229, 101), // horizontal rule, 2 px thick
                (30, 10, 31, 149),   // vertical rule, 2 px thick
                (200, 10, 201, 149), // vertical rule, 2 px thick
            ],
        );
        let rules = detect_rules(&gray, w, h, 1.0);
        // mid = l0 + thick/2 = l0 + 1; spans in pixels
        assert_eq!(rules.h, vec![[21.0, 10.0, 230.0], [101.0, 10.0, 230.0]]);
        assert_eq!(rules.v, vec![[31.0, 10.0, 150.0], [201.0, 10.0, 150.0]]);
    }

    #[test]
    fn detect_rules_is_scale_aware_for_thickness() {
        let (w, h) = (240, 160);
        // a 6 px band: 3 pt at 2 px/pt, but only 1.2 pt at 5 px/pt
        let gray = page(w, h, &[(10, 50, 229, 55)]);
        assert!(detect_rules(&gray, w, h, 1.0).h.is_empty(), "6 px > 2.5 px");
        assert!(
            detect_rules(&gray, w, h, 2.0).h.is_empty(),
            "6 px > 5 px at 2 px/pt"
        );
        let fine = detect_rules(&gray, w, h, 5.0);
        assert_eq!(fine.h.len(), 1, "6 px = 1.2 pt at 5 px/pt");
        // and the coordinate comes back in points: (50 + 3) / 5
        assert!((fine.h[0][0] - 53.0 / 5.0).abs() < 1e-6);
        assert!((fine.h[0][1] - 2.0).abs() < 1e-6);
        assert!((fine.h[0][2] - 46.0).abs() < 1e-6);
    }

    #[test]
    fn detect_rules_ignores_text_sized_ink() {
        let (w, h) = (240, 160);
        // 10x4 blobs: too short along either axis (min run is 15 px)
        let mut rects = Vec::new();
        for row in 0..6 {
            for col in 0..8 {
                let (x, y) = (10 + col * 28, 10 + row * 24);
                rects.push((x, y, x + 9, y + 3));
            }
        }
        let gray = page(w, h, &rects);
        let rules = detect_rules(&gray, w, h, 1.0);
        assert!(rules.h.is_empty() && rules.v.is_empty());
    }

    /// `PageImage::input` on a 4x4 page: ink in the canvas corner, boxes as canvas
    /// fractions scaled by `k / IMG_SIDE` and clamped.
    #[test]
    fn page_image_input_puts_ink_and_boxes_on_the_canvas() {
        let img = PageImage {
            w: 4,
            h: 4,
            px: vec![
                0, 51, 102, 153, //
                255, 204, 153, 102, //
                51, 0, 255, 128, //
                64, 32, 16, 8,
            ],
            k: 2.0,
        };
        let boxes = [
            [0.0, 0.0, 512.0, 512.0],
            [100.0, 10.0, 600.0, 40.0],
            [-10.0, -10.0, 5.0, 5.0],
        ];
        let input = img.input(&boxes);
        assert_eq!(input.canvas.len(), IMG_SIDE * IMG_SIDE);
        assert_eq!(input.canvas[0], 1.0);
        assert!((input.canvas[1] - (255.0 - 51.0) / 255.0).abs() < 1e-6);
        assert_eq!(input.canvas[4], 0.0);
        // row 4 is past the page: the canvas stays empty
        assert_eq!(input.canvas[4 * IMG_SIDE], 0.0);
        assert_eq!(input.canvas[3 * IMG_SIDE + 3], (255.0 - 8.0) / 255.0);
        // f = k / IMG_SIDE = 1/512: the full-page box is the unit square
        assert_eq!(input.boxes[0], [0.0, 0.0, 1.0, 1.0]);
        assert!((input.boxes[1][0] - 100.0 / 512.0).abs() < 1e-6);
        assert!((input.boxes[1][1] - 10.0 / 512.0).abs() < 1e-6);
        assert_eq!(input.boxes[1][2], 1.0, "600/512 clamps at 1");
        assert!((input.boxes[1][3] - 40.0 / 512.0).abs() < 1e-6);
        // a box that starts left of the page clamps at 0
        assert_eq!(&input.boxes[2][..2], &[0.0, 0.0]);
        assert!((input.boxes[2][2] - 5.0 / 512.0).abs() < 1e-6);
        assert_eq!(input.valid, (4.0 / 1024.0, 4.0 / 1024.0));
    }

    #[test]
    fn process_raster_resamples_to_the_long_side_and_keeps_page_pixel_rules() {
        // 200x100 RGB page, black 2 px rule across the middle
        let (w, h) = (200, 100);
        let mut rgb = vec![255u8; w * h * 3];
        for x in 10..190 {
            for y in 48..50 {
                let o = (y * w + x) * 3;
                rgb[o..o + 3].copy_from_slice(&[0, 0, 0]);
            }
        }
        let (rules, img) = process_raster(&rgb, w, h);
        assert_eq!((img.w, img.h), (1024, 512));
        assert_eq!(img.px.len(), 1024 * 512);
        assert!((img.k - 1024.0 / 200.0).abs() < 1e-6);
        // the resampled page is white at the top-left and dark on the rule
        assert_eq!(img.px[0], 255);
        assert!(img.px[248 * 1024 + 512] < 40, "the rule's rows are dark");
        // rules stay in page pixels, so they line up with the units
        assert_eq!(rules.h.len(), 1);
        assert!((rules.h[0][0] - 49.0).abs() < 1e-6, "{:?}", rules.h);
        assert!((rules.h[0][1] - 10.0).abs() < 1e-6);
        assert!((rules.h[0][2] - 190.0).abs() < 1e-6);
    }

    #[test]
    fn process_raster_accepts_grey_bytes_and_rejects_bad_lengths() {
        let (w, h) = (40, 30);
        let mut g = vec![255u8; w * h];
        for x in 0..40 {
            g[15 * w + x] = 0;
        }
        let (rules, img) = process_raster(&g, w, h);
        assert_eq!((img.w, img.h), (1024, 768));
        assert_eq!(rules.h.len(), 1);
        // a 2-byte page is neither grey nor RGB
        let (rules, img) = process_raster(&[0, 1], w, h);
        assert!(rules.h.is_empty() && rules.v.is_empty());
        assert_eq!((img.w, img.h), (0, 0));
        assert!(img.px.is_empty());
        // zero-sized pages are empty, not a panic
        let (rules, img) = process_raster(&[], 0, 0);
        assert!(rules.h.is_empty());
        assert_eq!(img.w, 0);
    }

    /// The layer-2 regression page (`tools/oracle/ocria/fixtures/layout/table.png`, the
    /// one `layout_finds_the_ruled_table` reads): the 3-column grid's 2 px rules must
    /// survive the thickness cap and land on the grid's pixels.
    #[test]
    fn the_recorded_table_fixture_keeps_its_grid_rules() {
        let bytes = include_bytes!("../../../../tools/oracle/ocria/fixtures/layout/table.png");
        let decoded = image::load_from_memory(bytes)
            .expect("decode the fixture")
            .to_rgb8();
        let (w, h) = (decoded.width() as usize, decoded.height() as usize);
        assert_eq!((w, h), (700, 500));
        let (rules, image) = process_raster(decoded.as_raw(), w, h);
        assert_eq!((image.w, image.h), (1024, 731));
        for y in [81.0f32, 125.0, 169.0, 213.0, 257.0] {
            assert!(
                rules
                    .h
                    .iter()
                    .any(|r| (r[0] - y).abs() < 0.5 && r[1] <= 61.0 && r[2] >= 601.0),
                "no horizontal rule at y={y}: {:?}",
                rules.h
            );
        }
        for x in [61.0f32, 241.0, 421.0, 601.0] {
            assert!(
                rules
                    .v
                    .iter()
                    .any(|r| (r[0] - x).abs() < 0.5 && r[1] <= 81.0 && r[2] >= 257.0),
                "no vertical rule at x={x}: {:?}",
                rules.v
            );
        }
    }

    #[test]
    fn empty_pages_have_no_features() {
        assert!(unit_features(&[], (10.0, 10.0), None).is_empty());
        assert!(pair_features(&[], (10.0, 10.0), None).is_empty());
    }
}
