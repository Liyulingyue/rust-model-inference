//! Decoding LayoutNet's logits into tables and layout blocks.
//!
//! A port of LiteParse's `table_model/decode.rs` (pinned `44ceb6ab`) together with the
//! block / class half of its `table_model/mod.rs` (`block_groups`, `split_at_gaps`,
//! `split_paragraphs`, and the loop of `run_page_framed` that turns a table's unit group
//! into a `Grid` and the class logits into blocks).
//!
//! Order-sensitive steps (argmax ties, stable sorts, insertion order) follow the reference
//! exactly, so parity fixtures compare cell for cell: `avg_link` merges the pair with the
//! highest mean (ties: the first live group in `alive` order) and the fast incremental
//! partner cache below is the reference's, not a re-derivation. `span_link` splits bridging
//! units (a header spanning several columns) out of the linkage and re-attaches them after,
//! `decode_grid` places cells in `(rmin, cmin)` order into an owner-slot grid where a cell
//! whose span is already taken either shrinks to its first free slot or appends its text to
//! the occupant, and the two wrap folds rewrite rows in place.
//!
//! `DUnit.minor` is kept, and with it the reference's `mask_minor` path (units running against
//! the page's main direction have their logits forced to -9 before decoding). This
//! repository's OCR path has no rotation, so `minor` is always false and the mask is never
//! applied; the code path stays for parity with the reference's page model.
//!
//! The table keep/drop predicates [`Grid::is_toc`], [`Grid::is_chart`] and [`Grid::is_prose`]
//! are decode-time filters: the reference calls them from `run_page_framed` to decide whether
//! a decoded grid is a table at all, so they are applied here and `Layout::tables` is exactly
//! the reference's `PageModel::tables`.

use std::cmp::Ordering;
use std::collections::BTreeMap;

use super::net::Logits;
use super::{Block, Layout, Table};

/// A unit as the decoder sees it.
#[derive(Clone, Debug)]
pub(crate) struct DUnit {
    pub bx: [f64; 4],
    pub text: String,
    /// The unit's line height without the recognizer's padding (`Unit::h`), which is what the
    /// reference's `median_h` thresholds are measured against.
    pub h: f64,
    /// Runs against the page's main direction (a sidebar, a rotated label). Always false on
    /// this repository's OCR path; see the module comment for what it does when set.
    pub minor: bool,
}

/// Decoder switches, at the release's defaults.
#[derive(Clone, Copy, Debug)]
pub(crate) struct DecodeOptions {
    pub table_thresh: f32,
    pub head_band: bool,
    pub forced_link: bool,
    pub squeeze: bool,
    pub fold_wraps: bool,
}

impl Default for DecodeOptions {
    fn default() -> Self {
        DecodeOptions {
            table_thresh: 0.0,
            head_band: true,
            forced_link: true,
            squeeze: true,
            fold_wraps: true,
        }
    }
}

/// One placed cell.
#[derive(Clone, Debug)]
pub(crate) struct GridCell {
    pub r0: usize,
    pub c0: usize,
    pub r1: usize,
    pub c1: usize,
    pub text: String,
    pub bx: [f64; 4],
}

/// A decoded table: cells in placement order, header rows, grid size.
#[derive(Clone, Debug, Default)]
pub(crate) struct Grid {
    pub cells: Vec<GridCell>,
    pub head_rows: Vec<usize>,
    pub n_rows: usize,
    pub n_cols: usize,
}

/// A stacked-cell link is only trusted when the weaker of `same_row` / `same_col` clears
/// this margin (`decode.rs:20`).
const CELL_MARGIN: f32 = 1.5;
/// Two boxes stack (read as one cell's lines) when their vertical gap is under this share of
/// the smaller line height (`decode.rs:21`).
const STACK_GAP: f64 = 0.8;
/// The same, when both units are header units (`decode.rs:22`).
const STACK_GAP_HDR: f64 = 0.8;
/// Cell pairs that do not overlap on an axis lose this much of their logit (`decode.rs:23`).
const GATE: f64 = 3.0;

/// `partial_cmp` for finite boxes, falling back to a total order for NaN instead of panicking
/// (`f64::partial_cmp(..).unwrap()` in the reference). For finite values this is exactly the
/// reference's comparison, including `-0.0 == 0.0` in the stable sorts.
#[inline]
fn cmp_f64(a: f64, b: f64) -> Ordering {
    a.partial_cmp(&b).unwrap_or_else(|| a.total_cmp(&b))
}

/// Median of `v`, or NaN when it is empty (`decode.rs:48`).
fn median(v: &mut [f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| cmp_f64(*a, *b));
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

/// The unit's height for the decoder's `*_h` thresholds: the reference's `DUnit.h`, i.e. the
/// line height without the recognizer's padding. (Box heights are used where the reference
/// uses them, in `decode_grid`'s `hh`.)
#[inline]
fn unit_h(u: &DUnit) -> f64 {
    u.h
}

/// Median unit height, or 1 when it is 0 (`decode.rs:62`).
fn median_h(units: &[DUnit], idx: impl Iterator<Item = usize>) -> f64 {
    let mut hs: Vec<f64> = idx.map(|i| unit_h(&units[i])).collect();
    let m = median(&mut hs);
    if m == 0.0 || m.is_nan() {
        1.0
    } else {
        m
    }
}

/// Average-linkage clusters of a symmetric n x n matrix: merge while the mean logit > thresh.
/// Groups come back sorted, in the order of their surviving representative (`decode.rs:75`).
///
/// Each merge joins the pair with the highest mean (ties: the first in row-major order of the
/// live groups) and adds the absorbed group's sums into the survivor's row and column. Every
/// live group caches its best partner, so a merge rescans only the rows whose cached partner
/// it touched: O(n^2) on typical pages instead of rescanning every pair per merge.
pub(crate) fn avg_link(p: &[f64], n: usize, thresh: f64) -> Vec<Vec<usize>> {
    if p.len() < n * n {
        return Vec::new();
    }
    let mut groups: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
    let mut s = p.to_vec();
    let mut cnt = vec![1f64; n * n];
    let mut alive: Vec<usize> = (0..n).collect();
    // best partner of each live group g: (mean, partner), the first maximum in `alive` order
    let row_best = |s: &[f64], cnt: &[f64], alive: &[usize], g: usize| -> (f64, usize) {
        let mut best = (f64::NEG_INFINITY, usize::MAX);
        for &h in alive {
            if h != g {
                let v = s[g * n + h] / cnt[g * n + h];
                if v > best.0 {
                    best = (v, h);
                }
            }
        }
        best
    };
    let mut best: Vec<(f64, usize)> = (0..n).map(|g| row_best(&s, &cnt, &alive, g)).collect();
    while alive.len() > 1 {
        // the first live group (in `alive` order) holding the overall maximum
        let mut ba = 0;
        for a in 1..alive.len() {
            if best[alive[a]].0 > best[alive[ba]].0 {
                ba = a;
            }
        }
        let (v, gb) = best[alive[ba]];
        if v <= thresh {
            break;
        }
        let ga = alive[ba];
        let Some(bb) = alive.iter().position(|&g| g == gb) else {
            break;
        };
        let moved = std::mem::take(&mut groups[gb]);
        groups[ga].extend(moved);
        for k in 0..n {
            s[ga * n + k] += s[gb * n + k];
            cnt[ga * n + k] += cnt[gb * n + k];
        }
        for k in 0..n {
            s[k * n + ga] += s[k * n + gb];
            cnt[k * n + ga] += cnt[k * n + gb];
        }
        alive.remove(bb);
        // Only row ga and column ga changed (and gb is gone): a row whose partner was ga or gb
        // is rescanned; any other row compares its old best with its new entry for ga.
        best[ga] = row_best(&s, &cnt, &alive, ga);
        for &r in &alive {
            if r == ga {
                continue;
            }
            let (bv, bc) = best[r];
            if bc == ga || bc == gb {
                best[r] = row_best(&s, &cnt, &alive, r);
            } else {
                let c = s[r * n + ga] / cnt[r * n + ga];
                if c > bv || (c == bv && ga < bc) {
                    best[r] = (c, ga);
                }
            }
        }
    }
    alive
        .iter()
        .map(|&g| {
            let mut v = groups[g].clone();
            v.sort_unstable();
            v
        })
        .collect()
}

/// Units positive with two groups that are negative with each other (a spanning header)
/// (`decode.rs:151`).
fn bridges(p: &[f64], n: usize) -> Vec<usize> {
    let (neg, frac) = (-1.0, 0.3);
    let mut out = Vec::new();
    for i in 0..n {
        let nb: Vec<usize> = (0..n).filter(|&j| j != i && p[i * n + j] > 0.0).collect();
        if nb.len() < 2 {
            continue;
        }
        let (mut tot, mut negc) = (0usize, 0usize);
        for a in 0..nb.len() {
            for b in a + 1..nb.len() {
                tot += 1;
                if p[nb[a] * n + nb[b]] < neg {
                    negc += 1;
                }
            }
        }
        if tot > 0 && (negc as f64 / tot as f64) > frac {
            out.push(i);
        }
    }
    out
}

/// Rows or columns: linkage over the non-bridging units; each bridging unit then joins every
/// group it is positive with on average (a span), or its best group (`decode.rs:177`).
fn span_link(p: &[f64], n: usize) -> Vec<Vec<usize>> {
    let br = bridges(p, n);
    let core: Vec<usize> = (0..n).filter(|i| !br.contains(i)).collect();
    if core.is_empty() {
        return avg_link(p, n, 0.0);
    }
    let m = core.len();
    let mut sub = vec![0f64; m * m];
    for a in 0..m {
        for b in 0..m {
            sub[a * m + b] = p[core[a] * n + core[b]];
        }
    }
    let mut groups: Vec<Vec<usize>> = avg_link(&sub, m, 0.0)
        .into_iter()
        .map(|g| g.into_iter().map(|i| core[i]).collect())
        .collect();
    for &i in &br {
        let means: Vec<f64> = groups
            .iter()
            .map(|g| g.iter().map(|&j| p[i * n + j]).sum::<f64>() / g.len() as f64)
            .collect();
        let mut hit: Vec<usize> = (0..groups.len()).filter(|&k| means[k] > 0.0).collect();
        if hit.is_empty() {
            let mut best = 0;
            for k in 1..means.len() {
                if means[k] > means[best] {
                    best = k;
                }
            }
            hit = vec![best];
        }
        for k in hit {
            groups[k].push(i);
        }
    }
    for g in groups.iter_mut() {
        g.sort_unstable();
    }
    groups
}

/// Box indices in reading order: lines (vertical overlap over half the shorter box), top to
/// bottom, left to right within a line (`decode.rs:221`).
pub(crate) fn reading_order(boxes: &[[f64; 4]]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..boxes.len()).collect();
    order.sort_by(|&a, &b| cmp_f64(boxes[a][1], boxes[b][1]));
    let mut lines: Vec<(f64, f64, Vec<usize>)> = Vec::new();
    for k in order {
        let b = boxes[k];
        let mut placed = false;
        for ln in lines.iter_mut() {
            let ov = ln.1.min(b[3]) - ln.0.max(b[1]);
            if ov > 0.5 * (ln.1 - ln.0).min(b[3] - b[1]) {
                ln.2.push(k);
                ln.0 = ln.0.min(b[1]);
                ln.1 = ln.1.max(b[3]);
                placed = true;
                break;
            }
        }
        if !placed {
            lines.push((b[1], b[3], vec![k]));
        }
    }
    lines.sort_by(|a, b| cmp_f64(a.0, b.0));
    let mut out = Vec::new();
    for ln in lines {
        let mut ks = ln.2;
        ks.sort_by(|&a, &b| cmp_f64(boxes[a][0], boxes[b][0]));
        out.extend(ks);
    }
    out
}

fn bbox_of(units: &[DUnit], g: &[usize]) -> [f64; 4] {
    let mut b = [
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    ];
    for &i in g {
        if let Some(u) = units.get(i) {
            let u = &u.bx;
            b[0] = b[0].min(u[0]);
            b[1] = b[1].min(u[1]);
            b[2] = b[2].max(u[2]);
            b[3] = b[3].max(u[3]);
        }
    }
    b
}

/// `re.match(r"^\s*(table|tab\.|figure|fig\.|exhibit|chart|source|note)s?\b", t, re.I)`: the
/// alternatives in order, the optional "s" tried first (greedy), then a word boundary
/// (`decode.rs:271`).
fn is_caption(text: &str) -> bool {
    let t: Vec<char> = text.trim_start().to_lowercase().chars().collect();
    let word = |c: char| c.is_alphanumeric() || c == '_';
    let boundary = |pos: usize| {
        let prev = pos.checked_sub(1).map(|i| word(t[i])).unwrap_or(false);
        let next = t.get(pos).map(|&c| word(c)).unwrap_or(false);
        prev != next
    };
    for w in [
        "table", "tab.", "figure", "fig.", "exhibit", "chart", "source", "note",
    ] {
        let wc: Vec<char> = w.chars().collect();
        if t.len() < wc.len() || t[..wc.len()] != wc[..] {
            continue;
        }
        let end = wc.len();
        if t.get(end) == Some(&'s') && boundary(end + 1) {
            return true;
        }
        if boundary(end) {
            return true;
        }
    }
    false
}

/// Grow each table up over its header band (`decode.rs:298`).
fn head_band(units: &[DUnit], l: &Logits, groups: &mut [Vec<usize>], taken: &mut [bool]) {
    let hm = median_h(units, 0..units.len());
    for g in groups.iter_mut() {
        for _ in 0..3 {
            let b = bbox_of(units, g);
            let (x0, y0, x1) = (b[0], b[1], b[2]);
            let above: Vec<usize> = (0..units.len())
                .filter(|&i| {
                    let u = &units[i].bx;
                    !taken[i]
                        && l.node(i, 0) > -8.0
                        && u[3] <= y0 + 0.2 * hm
                        && y0 - u[3] < 4.0 * 1.4 * hm
                })
                .collect();
            if above.is_empty() {
                break;
            }
            // the nearest line above: units overlapping vertically with the lowest one (max()'s first)
            let mut low = above[0];
            for &i in &above[1..] {
                if units[i].bx[3] > units[low].bx[3] {
                    low = i;
                }
            }
            let lb = units[low].bx;
            let line: Vec<usize> = above
                .iter()
                .copied()
                .filter(|&i| units[i].bx[3].min(lb[3]) - units[i].bx[1].max(lb[1]) > 0.3 * hm)
                .collect();
            let w = x1 - x0;
            let mut ok = line
                .iter()
                .all(|&i| units[i].bx[0] >= x0 - 0.05 * w && units[i].bx[2] <= x1 + 0.05 * w);
            let near = y0
                - line
                    .iter()
                    .map(|&i| units[i].bx[3])
                    .fold(f64::NEG_INFINITY, f64::max)
                < 1.5 * 1.4 * hm;
            let strong = line.iter().all(|&i| l.node(i, 0) > -1.5);
            let over = line.iter().all(|&i| {
                g.iter().any(|&j| {
                    units[i].bx[2].min(units[j].bx[2]) - units[i].bx[0].max(units[j].bx[0]) > 0.0
                })
            });
            ok = ok && (strong || (near && over && line.iter().all(|&i| l.node(i, 0) > -4.0)));
            ok = ok && !line.iter().any(|&i| is_caption(&units[i].text));
            let hdr = line.iter().filter(|&&i| l.node(i, 1) > 0.0).count() as f64
                / line.len().max(1) as f64;
            ok = ok && line.len() >= 2 && hdr > 0.5;
            ok = ok
                && !line.iter().any(|&i| {
                    let t = &units[i].text;
                    t.split_whitespace().count() > 6
                        || t.trim_end().ends_with('.')
                        || t.trim_end().ends_with(':')
                });
            if !ok {
                break;
            }
            for &i in &line {
                g.push(i);
                taken[i] = true;
            }
        }
        g.sort_unstable();
    }
}

/// Tables as lists of unit indices (`decode.rs:370`, `decode_tables` with `absorb=True`).
pub(crate) fn decode_tables(units: &[DUnit], l: &Logits, opt: &DecodeOptions) -> Vec<Vec<usize>> {
    let n = units.len();
    let on: Vec<usize> = (0..n)
        .filter(|&i| l.node(i, 0) > opt.table_thresh)
        .collect();
    if on.len() < 2 {
        return Vec::new();
    }
    let m = on.len();
    let mut p = vec![0f64; m * m];
    for a in 0..m {
        for b in 0..m {
            p[a * m + b] = l.pair(0, on[a], on[b]) as f64;
        }
    }
    let mut groups: Vec<Vec<usize>> = avg_link(&p, m, 0.0)
        .into_iter()
        .filter(|g| g.len() >= 2)
        .map(|g| g.into_iter().map(|i| on[i]).collect())
        .collect();
    // stacked pieces that line up across and sit within 3 lines merge back
    if groups.len() > 1 {
        let hm = median_h(units, 0..n);
        loop {
            let ext: Vec<[f64; 4]> = groups.iter().map(|g| bbox_of(units, g)).collect();
            let mut merged = None;
            'outer: for a in 0..groups.len() {
                for b in 0..groups.len() {
                    if a == b {
                        continue;
                    }
                    let (aa, bb) = (ext[a], ext[b]);
                    let gap = bb[1] - aa[3];
                    let ov = aa[2].min(bb[2]) - aa[0].max(bb[0]);
                    if gap >= 0.0
                        && gap < 3.0 * 1.4 * hm
                        && ov > 0.7 * (aa[2] - aa[0]).min(bb[2] - bb[0])
                    {
                        merged = Some((a, b));
                        break 'outer;
                    }
                }
            }
            match merged {
                Some((a, b)) => {
                    let gb = groups[b].clone();
                    groups[a].extend(gb);
                    groups[a].sort_unstable();
                    groups.remove(b);
                }
                None => break,
            }
        }
    }
    // a small group lying mostly inside a bigger table's box (grown by a line) joins it
    if groups.len() > 1 {
        let hm = median_h(units, 0..n);
        loop {
            groups.sort_by(|a, b| b.len().cmp(&a.len()));
            let ext: Vec<[f64; 4]> = groups
                .iter()
                .map(|g| {
                    let b = bbox_of(units, g);
                    [
                        b[0] - 1.4 * hm,
                        b[1] - 1.4 * hm,
                        b[2] + 1.4 * hm,
                        b[3] + 1.4 * hm,
                    ]
                })
                .collect();
            let mut merged = None;
            'outer2: for s in (1..groups.len()).rev() {
                let sb = bbox_of(units, &groups[s]);
                let area = ((sb[2] - sb[0]) * (sb[3] - sb[1])).max(1e-6);
                for b in 0..s {
                    if groups[s].len() as f64 > 0.5 * groups[b].len() as f64 {
                        continue;
                    }
                    let e = ext[b];
                    let ov = (sb[2].min(e[2]) - sb[0].max(e[0])).max(0.0)
                        * (sb[3].min(e[3]) - sb[1].max(e[1])).max(0.0);
                    if ov >= 0.5 * area {
                        merged = Some((s, b));
                        break 'outer2;
                    }
                }
            }
            match merged {
                Some((s, b)) => {
                    let gs = groups[s].clone();
                    groups[b].extend(gs);
                    groups[b].sort_unstable();
                    groups.remove(s);
                }
                None => break,
            }
        }
    }
    if !groups.is_empty() {
        let mut taken = vec![false; n];
        for g in &groups {
            for &i in g {
                if let Some(t) = taken.get_mut(i) {
                    *t = true;
                }
            }
        }
        for g in groups.iter_mut() {
            let b = bbox_of(units, g);
            for i in 0..n {
                if taken[i] || l.node(i, 0) < -8.0 {
                    continue;
                }
                let u = &units[i].bx;
                let (cx, cy) = ((u[0] + u[2]) / 2.0, (u[1] + u[3]) / 2.0);
                if b[0] <= cx && cx <= b[2] && b[1] <= cy && cy <= b[3] {
                    g.push(i);
                    taken[i] = true;
                }
            }
            g.sort_unstable();
        }
        if opt.head_band {
            head_band(units, l, &mut groups, &mut taken);
        }
    }
    groups
}

/// Fold near-empty columns into their nearest neighbour (`decode.rs:519`).
fn squeeze_cols(cells: &mut Vec<GridCell>, head: &[usize], n_cols: &mut usize, hh: f64) {
    let mut changed = true;
    while changed && *n_cols >= 3 {
        changed = false;
        for q in 0..*n_cols {
            let col: Vec<usize> = (0..cells.len())
                .filter(|&k| cells[k].c0 == q && cells[k].c1 == q && !cells[k].text.is_empty())
                .collect();
            if col
                .iter()
                .filter(|&&k| !head.contains(&cells[k].r0))
                .count()
                > 1
            {
                continue;
            }
            let boxes: Vec<[f64; 4]> = col.iter().map(|&k| cells[k].bx).collect();
            let mut best: Option<(f64, usize)> = None;
            for t in [q as isize - 1, q as isize + 1] {
                if t < 0 || t as usize >= *n_cols {
                    continue;
                }
                let t = t as usize;
                let nb: Vec<[f64; 4]> = cells
                    .iter()
                    .filter(|x| x.c0 <= t && t <= x.c1 && x.c0 == x.c1 && !x.text.is_empty())
                    .map(|x| x.bx)
                    .collect();
                if boxes.is_empty() || nb.is_empty() {
                    continue;
                }
                let mut gap = f64::INFINITY;
                for b in &boxes {
                    for nn in &nb {
                        gap = gap.min((b[0] - nn[2]).max(nn[0] - b[2]).max(0.0));
                    }
                }
                let closer = match best {
                    None => true,
                    Some((g, _)) => gap < g,
                };
                if gap < 1.5 * hh && closer {
                    best = Some((gap, t));
                }
            }
            if let Some((_, t)) = best {
                fold_col(cells, q, t);
                *n_cols -= 1;
                changed = true;
                break;
            }
        }
    }
}

fn fold_col(cells: &mut Vec<GridCell>, k: usize, t: usize) {
    let movers: Vec<usize> = (0..cells.len())
        .filter(|&i| cells[i].c0 == k && cells[i].c1 == k)
        .collect();
    let mut remove = Vec::new();
    for &xi in &movers {
        let r = cells[xi].r0;
        let tgt = (0..cells.len()).find(|&j| {
            cells[j].r0 <= r && r <= cells[j].r1 && cells[j].c0 <= t && t <= cells[j].c1
        });
        match tgt {
            Some(j) if j != xi => {
                if !cells[xi].text.is_empty() {
                    let joined = if t < k {
                        format!("{} {}", cells[j].text, cells[xi].text)
                    } else {
                        format!("{} {}", cells[xi].text, cells[j].text)
                    };
                    cells[j].text = joined.trim().to_string();
                }
                remove.push(xi);
            }
            _ => {
                cells[xi].c0 = t;
                cells[xi].c1 = t;
            }
        }
    }
    remove.sort_unstable();
    for &i in remove.iter().rev() {
        cells.remove(i);
    }
    for x in cells.iter_mut() {
        if x.c0 <= k && k <= x.c1 && x.c0 < x.c1 {
            x.c1 -= 1;
        } else if x.c0 > k {
            x.c0 -= 1;
            x.c1 -= 1;
        }
    }
}

/// Union-find root with path halving (the reference's nested `root`, `decode.rs:660`).
fn uf_root(parent: &mut [usize], mut k: usize) -> usize {
    while parent[k] != k {
        parent[k] = parent[parent[k]];
        k = parent[k];
    }
    k
}

/// One table's grid from its units (global indices `idx`) and the row / col / cell heads
/// (`decode.rs:613`).
pub(crate) fn decode_grid(units: &[DUnit], idx: &[usize], l: &Logits, opt: &DecodeOptions) -> Grid {
    let m = idx.len();
    if m == 0 || idx.iter().any(|&i| i >= units.len()) {
        return Grid::default();
    }
    let bxs: Vec<[f32; 4]> = idx
        .iter()
        .map(|&i| {
            let b = units[i].bx;
            [b[0] as f32, b[1] as f32, b[2] as f32, b[3] as f32]
        })
        .collect();
    let hh: Vec<f32> = bxs.iter().map(|b| (b[3] - b[1]).max(1e-3)).collect();
    let pr = |k: usize, a: usize, b: usize| l.pair(k, idx[a], idx[b]);
    // cell logits: same_cell, or for stacked neighbours the weaker of same_row / same_col - margin
    let mut cell_p = vec![0f64; m * m];
    let mut forced = vec![false; m * m];
    for a in 0..m {
        for b in 0..m {
            let vgap = (bxs[b][1] - bxs[a][3]).max(bxs[a][1] - bxs[b][3]);
            let xov = bxs[a][2].min(bxs[b][2]) - bxs[a][0].max(bxs[b][0]);
            let both_hdr = l.node(idx[a], 1) > 0.0 && l.node(idx[b], 1) > 0.0;
            let lim = if both_hdr { STACK_GAP_HDR } else { STACK_GAP } * hh[a].min(hh[b]) as f64;
            let stacked = (vgap as f64) < lim && vgap > -0.5 * hh[a] && xov > 0.0;
            let f = if stacked {
                pr(1, a, b).min(pr(2, a, b)) - CELL_MARGIN
            } else {
                f32::NEG_INFINITY
            };
            forced[a * m + b] = f > 0.0;
            cell_p[a * m + b] = pr(3, a, b).max(f) as f64;
        }
    }
    let mut cells = avg_link(&cell_p, m, 0.0);
    if opt.forced_link && cells.len() > 1 {
        let mut of = vec![0usize; m];
        for (k, g) in cells.iter().enumerate() {
            for &i in g {
                of[i] = k;
            }
        }
        let mut parent: Vec<usize> = (0..cells.len()).collect();
        for a in 0..m {
            for b in 0..m {
                if forced[a * m + b] {
                    let ra = uf_root(&mut parent, of[a]);
                    let rb = uf_root(&mut parent, of[b]);
                    if ra != rb {
                        parent[ra] = rb;
                    }
                }
            }
        }
        let mut keys: Vec<usize> = Vec::new();
        let mut merged: Vec<Vec<usize>> = Vec::new();
        for (k, g) in cells.iter().enumerate() {
            let r = uf_root(&mut parent, k);
            match keys.iter().position(|&x| x == r) {
                Some(p) => merged[p].extend(g.iter().copied()),
                None => {
                    keys.push(r);
                    merged.push(g.clone());
                }
            }
        }
        for g in merged.iter_mut() {
            g.sort_unstable();
        }
        cells = merged;
    }
    let nc = cells.len();
    let cb: Vec<[f64; 4]> = cells
        .iter()
        .map(|g| {
            let mut b = [
                f64::INFINITY,
                f64::INFINITY,
                f64::NEG_INFINITY,
                f64::NEG_INFINITY,
            ];
            for &i in g {
                let u = units[idx[i]].bx;
                b[0] = b[0].min(u[0]);
                b[1] = b[1].min(u[1]);
                b[2] = b[2].max(u[2]);
                b[3] = b[3].max(u[3]);
            }
            b
        })
        .collect();
    // cell-pair logits on an axis: the mean of their unit pairs, minus GATE without overlap
    let cell_logits = |k: usize, lo: usize, hi: usize| -> Vec<f64> {
        let mut out = vec![0f64; nc * nc];
        for a in 0..nc {
            for b in a + 1..nc {
                let mut s = 0f64;
                for &i in &cells[a] {
                    for &j in &cells[b] {
                        s += pr(k, i, j) as f64;
                    }
                }
                let mut v = s / (cells[a].len() * cells[b].len()) as f64;
                if cb[a][hi].min(cb[b][hi]) - cb[a][lo].max(cb[b][lo]) <= 0.0 {
                    v -= GATE;
                }
                out[a * nc + b] = v;
                out[b * nc + a] = v;
            }
        }
        out
    };
    let (mut rows, mut cols) = if nc > 1 {
        (
            span_link(&cell_logits(1, 1, 3), nc),
            span_link(&cell_logits(2, 0, 2), nc),
        )
    } else {
        (vec![vec![0]], vec![vec![0]])
    };
    let mean_of = |g: &[usize], f: &dyn Fn(usize) -> f64| {
        g.iter().map(|&k| f(k)).sum::<f64>() / g.len().max(1) as f64
    };
    let ycen = |k: usize| (cb[k][1] + cb[k][3]) / 2.0;
    let xcen = |k: usize| (cb[k][0] + cb[k][2]) / 2.0;
    rows.sort_by(|a, b| cmp_f64(mean_of(a, &ycen), mean_of(b, &ycen)));
    cols.sort_by(|a, b| cmp_f64(mean_of(a, &xcen), mean_of(b, &xcen)));
    let mut r_in = vec![Vec::new(); nc];
    let mut c_in = vec![Vec::new(); nc];
    for (k, g) in rows.iter().enumerate() {
        for &c in g {
            r_in[c].push(k);
        }
    }
    for (k, g) in cols.iter().enumerate() {
        for &c in g {
            c_in[c].push(k);
        }
    }
    let n_rows0 = rows.len();
    let n_cols0 = cols.len();
    let mut placed: Vec<GridCell> = Vec::new();
    let mut owner = vec![usize::MAX; n_rows0 * n_cols0]; // slot -> index in `placed`
    let mut order: Vec<usize> = (0..nc).collect();
    // every cell belongs to at least one row and one column group; 0 is a safe stand-in for
    // the impossible empty group (the reference unwraps here)
    let rmin = |c: usize| {
        r_in.get(c)
            .and_then(|v| v.iter().copied().min())
            .unwrap_or(0)
    };
    let rmax = |c: usize| {
        r_in.get(c)
            .and_then(|v| v.iter().copied().max())
            .unwrap_or(0)
    };
    let cmin = |c: usize| {
        c_in.get(c)
            .and_then(|v| v.iter().copied().min())
            .unwrap_or(0)
    };
    let cmax = |c: usize| {
        c_in.get(c)
            .and_then(|v| v.iter().copied().max())
            .unwrap_or(0)
    };
    order.sort_by_key(|&a| (rmin(a), cmin(a)));
    for c in order {
        let (mut r0, mut r1) = (rmin(c), rmax(c));
        let (mut c0, mut c1) = (cmin(c), cmax(c));
        let boxes: Vec<[f64; 4]> = cells[c].iter().map(|&i| units[idx[i]].bx).collect();
        let text = reading_order(&boxes)
            .into_iter()
            .map(|k| units[idx[cells[c][k]]].text.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        let mut span = Vec::new();
        for r in r0..=r1 {
            for q in c0..=c1 {
                span.push((r, q));
            }
        }
        if span
            .iter()
            .any(|&(r, q)| owner[r * n_cols0 + q] != usize::MAX)
        {
            match span
                .iter()
                .find(|&&(r, q)| owner[r * n_cols0 + q] == usize::MAX)
            {
                Some(&(r, q)) => {
                    r0 = r;
                    r1 = r;
                    c0 = q;
                    c1 = q;
                }
                None => {
                    let o = owner[r0 * n_cols0 + c0];
                    if let Some(cell) = placed.get_mut(o) {
                        cell.text = format!("{} {}", cell.text, text);
                    }
                    continue;
                }
            }
        }
        let pi = placed.len();
        placed.push(GridCell {
            r0,
            c0,
            r1,
            c1,
            text,
            bx: cb[c],
        });
        for r in r0..=r1 {
            for q in c0..=c1 {
                owner[r * n_cols0 + q] = pi;
            }
        }
    }
    let mut head_rows = Vec::new();
    for (k, g) in rows.iter().enumerate() {
        let us: Vec<usize> = g.iter().flat_map(|&c| cells[c].iter().copied()).collect();
        let pos = us.iter().filter(|&&i| l.node(idx[i], 1) > 0.0).count() as f64;
        if pos / us.len().max(1) as f64 > 0.5 {
            head_rows.push(k);
        }
    }
    let mut n_cols = n_cols0;
    if opt.squeeze {
        let hh = median_h(units, idx.iter().copied());
        squeeze_cols(&mut placed, &head_rows, &mut n_cols, hh);
    }
    let mut head_rows = head_rows;
    let mut n_rows = n_rows0;
    if opt.fold_wraps {
        fold_wrapped_rows(&mut placed, &mut head_rows, &mut n_rows);
        fold_header_wraps(&mut placed, &mut head_rows, &mut n_rows);
    }
    Grid {
        cells: placed,
        head_rows,
        n_rows,
        n_cols,
    }
}

/// A lone number (thousands separators, sign, decimal, percent): values don't wrap
/// (`decode.rs:850`).
fn is_number(t: &str) -> bool {
    let t = t.trim().trim_end_matches('%').replace([',', ' '], "");
    !t.is_empty()
        && t.chars().any(|c| c.is_ascii_digit())
        && t.chars()
            .all(|c| c.is_ascii_digit() || "+-.()$€£".contains(c))
}

/// Text opening a new list item ("1.2 ...", "(a) ...", "• ..."): a sub-row, never a wrapped
/// line (`decode.rs:859`).
fn starts_item(t: &str) -> bool {
    let t = t.trim_start();
    let Some(first) = t.split_whitespace().next() else {
        return false;
    };
    if first.chars().count() == 1 && "•·▪◦‣-–—*●○■□".contains(first) {
        return true;
    }
    let core = first
        .trim_start_matches('(')
        .trim_end_matches([')', '.', ':']);
    let enumerator = !core.is_empty()
        && (core
            .split('.')
            .all(|p| !p.is_empty() && p.chars().all(|c| c.is_ascii_digit()))
            || (core.chars().count() == 1
                && core.chars().all(|c| c.is_ascii_alphabetic())
                && first != core));
    enumerator && first != t
}

/// Rows that are a wrapped cell's continuation lines fold into the row above (`decode.rs:884`).
fn fold_wrapped_rows(cells: &mut Vec<GridCell>, head: &mut Vec<usize>, n_rows: &mut usize) {
    let filled = |cells: &[GridCell], r: usize| -> Vec<usize> {
        (0..cells.len())
            .filter(|&k| cells[k].r0 == r && cells[k].r1 == r && !cells[k].text.trim().is_empty())
            .collect()
    };
    let top = |cells: &[GridCell], ks: &[usize]| {
        ks.iter()
            .map(|&k| cells[k].bx[1])
            .fold(f64::INFINITY, f64::min)
    };
    let mut pitches = Vec::new();
    for r in 1..*n_rows {
        let (a, b) = (filled(cells, r - 1), filled(cells, r));
        if a.len() >= 2 && b.len() >= 2 {
            pitches.push(top(cells, &b) - top(cells, &a));
        }
    }
    if pitches.len() < 2 {
        return;
    }
    let pitch = median(&mut pitches);
    let mut r = 1;
    while r < *n_rows {
        let (above, here) = (filled(cells, r - 1), filled(cells, r));
        let same_kind = head.contains(&r) == head.contains(&(r - 1));
        let mut targets = Vec::new();
        let span_starts = cells.iter().any(|c| c.r0 == r && c.r1 > r);
        let ok = same_kind
            && !span_starts
            && !here.is_empty()
            && here.len() < above.len()
            && here.iter().all(|&k| {
                let c = &cells[k];
                // the filled cell covering this column in the row above
                let t = (0..cells.len()).find(|&j| {
                    cells[j].r0 < r
                        && r - 1 <= cells[j].r1
                        && cells[j].c0 <= c.c0
                        && c.c1 <= cells[j].c1
                });
                match t {
                    Some(j)
                        if !cells[j].text.trim().is_empty()
                            && !is_number(&c.text)
                            && !starts_item(&c.text) =>
                    {
                        let ok = c.bx[1] - cells[j].bx[1] < 0.8 * pitch
                            && c.bx[1] - cells[j].bx[3] < 0.5 * pitch;
                        targets.push(j);
                        ok
                    }
                    _ => false,
                }
            });
        if !ok {
            r += 1;
            continue;
        }
        for (&k, &j) in here.iter().zip(&targets) {
            let (t, b) = (cells[k].text.trim().to_string(), cells[k].bx);
            let c = &mut cells[j];
            c.text = format!("{} {}", c.text.trim(), t);
            c.bx = [
                c.bx[0].min(b[0]),
                c.bx[1].min(b[1]),
                c.bx[2].max(b[2]),
                c.bx[3].max(b[3]),
            ];
        }
        remove_row(cells, head, n_rows, r);
    }
}

/// Drop row `r` after its text moved up (`decode.rs:960`).
fn remove_row(cells: &mut Vec<GridCell>, head: &mut Vec<usize>, n_rows: &mut usize, r: usize) {
    cells.retain(|c| !(c.r0 == r && c.r1 == r));
    for c in cells.iter_mut() {
        if c.r0 > r {
            c.r0 -= 1;
            c.r1 -= 1;
        } else if c.r1 >= r {
            c.r1 -= 1;
        }
    }
    head.retain(|&h| h != r);
    for h in head.iter_mut() {
        if *h > r {
            *h -= 1;
        }
    }
    *n_rows = n_rows.saturating_sub(1);
}

/// Whether `below` reads as the next line of the same phrase as `above` (`decode.rs:982`).
fn continues(above: &str, below: &str) -> bool {
    let (a, b) = (above.trim(), below.trim());
    let opens = b
        .chars()
        .next()
        .is_some_and(|ch| ch.is_lowercase() || "([{".contains(ch));
    let open_brackets = a.matches('(').count() > a.matches(')').count();
    let last_word = a
        .rsplit(char::is_whitespace)
        .next()
        .unwrap_or("")
        .to_lowercase();
    let breaks = a.ends_with(['-', '/', ',', '(', '&'])
        || open_brackets
        || [
            "and", "of", "the", "for", "to", "in", "on", "or", "per", "by", "with", "from",
        ]
        .contains(&last_word.as_str());
    opens || breaks
}

/// Header cells wrapped over several lines come out as one header row per line; in the header
/// area a row folds into the one above when every filled cell sits under a filled cell, none is
/// a number, each starts within 0.6x its own line height of the cell above's bottom, and each
/// reads as a continuation (`decode.rs:1011`).
fn fold_header_wraps(cells: &mut Vec<GridCell>, head: &mut Vec<usize>, n_rows: &mut usize) {
    let filled = |cells: &[GridCell], r: usize| -> Vec<usize> {
        (0..cells.len())
            .filter(|&k| cells[k].r0 == r && cells[k].r1 == r && !cells[k].text.trim().is_empty())
            .collect()
    };
    // the header area: the leading rows without numbers (at most four), and any marked header
    // rows at the top
    let zone = |cells: &[GridCell], head: &[usize], n_rows: usize| -> usize {
        let mut marked = 0;
        while marked < n_rows && head.contains(&marked) {
            marked += 1;
        }
        let mut z = 0;
        while z < n_rows.min(4) {
            let f = filled(cells, z);
            if f.is_empty() || f.iter().any(|&k| is_number(&cells[k].text)) {
                break;
            }
            z += 1;
        }
        z.max(marked).min(4)
    };
    let mut r = 1;
    while r < zone(cells, head, *n_rows) {
        let here = filled(cells, r);
        let span_starts = cells.iter().any(|c| c.r0 == r && c.r1 > r);
        let mut pairs = Vec::new();
        // one lone filled cell in a wider row is a body cell's wrapped tail more often than a
        // header line (a header wraps in several columns at once)
        let n_cols = cells.iter().map(|c| c.c1 + 1).max().unwrap_or(0);
        let ok = (here.len() >= 2 || n_cols <= 2)
            && !span_starts
            && here.iter().all(|&k| {
                let c = &cells[k];
                let above = (0..cells.len()).find(|&j| {
                    cells[j].r0 < r
                        && r - 1 <= cells[j].r1
                        && cells[j].c0 <= c.c0
                        && c.c1 <= cells[j].c1
                        && !cells[j].text.trim().is_empty()
                });
                match above {
                    Some(j) if !is_number(&c.text) && !starts_item(&c.text) => {
                        let line_h = (c.bx[3] - c.bx[1]).max(1e-6);
                        pairs.push((k, j));
                        c.bx[1] - cells[j].bx[3] < 0.6 * line_h
                            && continues(&cells[j].text, &c.text)
                    }
                    _ => false,
                }
            });
        if !ok {
            r += 1;
            continue;
        }
        for (k, j) in pairs {
            let (t, b) = (cells[k].text.trim().to_string(), cells[k].bx);
            let c = &mut cells[j];
            c.text = format!("{} {}", c.text.trim(), t);
            c.bx = [
                c.bx[0].min(b[0]),
                c.bx[1].min(b[1]),
                c.bx[2].max(b[2]),
                c.bx[3].max(b[3]),
            ];
        }
        remove_row(cells, head, n_rows, r);
    }
}

impl Grid {
    /// A table of contents: at most three columns, and most rows end in a page number (arabic
    /// or roman) with the arabic ones non-decreasing down the table (`mod.rs:784`).
    pub(crate) fn is_toc(&self) -> bool {
        if self.n_cols > 3 {
            return false;
        }
        let mut rows: Vec<Vec<(usize, &str)>> = vec![Vec::new(); self.n_rows];
        for c in &self.cells {
            let t = c.text.trim();
            if !t.is_empty() {
                if let Some(row) = rows.get_mut(c.r0) {
                    row.push((c.c0, t));
                }
            }
        }
        let rows: Vec<_> = rows.into_iter().filter(|r| !r.is_empty()).collect();
        let mut hits = 0;
        let mut arabic = Vec::new();
        for r in &rows {
            let Some(last) = r.iter().max_by_key(|(q, _)| *q).map(|(_, t)| *t) else {
                continue;
            };
            let roman = last.len() <= 7 && last.chars().all(|ch| "ivxlcdmIVXLCDM".contains(ch));
            let num = last.len() <= 4 && last.chars().all(|ch| ch.is_ascii_digit());
            if r.len() >= 2 && (roman || num) {
                hits += 1;
                if num {
                    arabic.push(last.parse::<u32>().unwrap_or(0));
                }
            }
        }
        let rising = arabic.windows(2).filter(|w| w[1] < w[0]).count() <= arabic.len() / 10;
        hits >= 3 && hits * 5 >= rows.len() * 3 && rising
    }

    /// Multi-column prose read as a table (`mod.rs:817`).
    pub(crate) fn is_prose(&self) -> bool {
        let words = |t: &str| t.split_whitespace().count();
        let cells: Vec<&GridCell> = self
            .cells
            .iter()
            .filter(|c| !c.text.trim().is_empty())
            .collect();
        if cells.len() < 6 {
            return false;
        }
        let mut wc: Vec<usize> = cells.iter().map(|c| words(&c.text)).collect();
        wc.sort_unstable();
        if wc[wc.len() / 2] < 4 {
            return false;
        }
        let numeric = cells
            .iter()
            .filter(|c| words(&c.text) <= 3 && c.text.chars().any(|ch| ch.is_ascii_digit()))
            .count();
        if numeric * 10 >= cells.len() * 3 {
            return false;
        }
        let mut cols: BTreeMap<usize, Vec<&GridCell>> = BTreeMap::new();
        for c in &cells {
            if c.c0 == c.c1 {
                cols.entry(c.c0).or_default().push(c);
            }
        }
        let (mut pairs, mut flow) = (0usize, 0usize);
        for col in cols.values_mut() {
            col.sort_by_key(|c| c.r0);
            for w in col.windows(2) {
                if w[1].r0 > w[0].r1 + 2 {
                    continue; // not adjacent: more than one empty row between them
                }
                pairs += 1;
                let up = w[0].text.trim_end();
                let open = !up.ends_with(['.', '!', '?', ':', ';']);
                let lower = w[1]
                    .text
                    .trim_start()
                    .chars()
                    .next()
                    .is_some_and(|ch| ch.is_lowercase());
                flow += (open && lower) as usize;
            }
        }
        pairs >= 3 && flow * 10 >= pairs * 4
    }

    /// A chart read as a table: a sparse, mostly numeric grid with an axis (`mod.rs:870`).
    pub(crate) fn is_chart(&self) -> bool {
        let slots = self.n_rows * self.n_cols;
        if slots == 0 {
            return false;
        }
        let parse = |t: &str| -> Option<f64> {
            let t = t.trim().trim_end_matches('%').replace(',', "");
            if t.is_empty()
                || !t
                    .chars()
                    .all(|ch| ch.is_ascii_digit() || ch == '.' || ch == '-' || ch == '+')
            {
                return None;
            }
            t.parse::<f64>().ok()
        };
        let (mut filled, mut nonempty, mut numeric) = (0, 0, 0);
        let mut cols: BTreeMap<usize, Vec<(usize, f64)>> = BTreeMap::new();
        let mut flow: Vec<((usize, usize), f64)> = Vec::new();
        for c in &self.cells {
            if c.text.trim().is_empty() {
                continue;
            }
            nonempty += 1;
            filled += (c.r1 - c.r0 + 1) * (c.c1 - c.c0 + 1);
            if let Some(v) = parse(&c.text) {
                numeric += 1;
                if c.c0 == c.c1 {
                    cols.entry(c.c0).or_default().push((c.r0, v));
                }
                flow.push(((c.r0, c.c0), v));
            }
        }
        if filled * 2 >= slots || numeric * 2 < nonempty {
            return false;
        }
        let stepped = |vs: &[f64]| {
            vs.len() >= 3 && {
                let d: Vec<f64> = vs.windows(2).map(|w| w[1] - w[0]).collect();
                d[0] != 0.0
                    && d.iter()
                        .all(|&x| (x - d[0]).abs() <= 1e-6 * d[0].abs().max(1.0))
            }
        };
        // a column of ticks (y axis), or every number on the grid in reading order (an x axis of
        // years strewn diagonally, one per row)
        flow.sort_by_key(|&(rc, _)| rc);
        let all: Vec<f64> = flow.iter().map(|&(_, v)| v).collect();
        stepped(&all)
            || cols.values_mut().any(|vs| {
                vs.sort_by_key(|&(r, _)| r);
                stepped(&vs.iter().map(|&(_, v)| v).collect::<Vec<_>>())
            })
    }
}

/// The highest class index; ties go to the last maximum, as the reference's `max_by` does
/// (`mod.rs:716`). An empty slice gives 0.
fn argmax(v: &[f32]) -> usize {
    let mut best = 0;
    for k in 1..v.len() {
        if v[k].total_cmp(&v[best]) != Ordering::Less {
            best = k;
        }
    }
    best
}

/// Summed class logits over `g`, argmax (`mod.rs:729`).
fn argmax_sum(g: &[usize], cls: &[f32], c: usize) -> usize {
    let mut s = vec![0f32; c];
    for &i in g {
        if let Some(row) = cls.get(i * c..(i + 1) * c) {
            for (k, v) in row.iter().enumerate() {
                s[k] += v;
            }
        }
    }
    argmax(&s)
}

/// A block's units split where their horizontal extents leave a gap wider than the units'
/// median height (`mod.rs:281`).
fn split_at_gaps(g: &[usize], units: &[DUnit]) -> Vec<Vec<usize>> {
    let mut hs: Vec<f64> = g.iter().map(|&i| units[i].bx[3] - units[i].bx[1]).collect();
    hs.sort_by(f64::total_cmp);
    let gap = hs.get(hs.len() / 2).copied().unwrap_or(0.0).max(1.0);
    let mut by_x: Vec<usize> = g.to_vec();
    by_x.sort_by(|&a, &b| units[a].bx[0].total_cmp(&units[b].bx[0]));
    let mut parts: Vec<Vec<usize>> = Vec::new();
    let mut right = f64::NEG_INFINITY;
    for i in by_x {
        let p = units[i].bx;
        if parts.is_empty() || p[0] - right > gap {
            parts.push(Vec::new());
        }
        if let Some(part) = parts.last_mut() {
            part.push(i);
        }
        right = right.max(p[2]);
    }
    parts
}

/// A block's units split into its lines' runs: a new part starts where the vertical gap to the
/// next line is more than the block's median line height, or where the lines change between
/// heading-like and body text (`mod.rs:222`).
fn split_paragraphs(g: &[usize], units: &[DUnit], cls: &[f32], c: usize) -> Vec<Vec<usize>> {
    if g.len() < 2 || c < 3 {
        return vec![g.to_vec()];
    }
    let mut by_y: Vec<usize> = g.to_vec();
    by_y.sort_by(|&a, &b| {
        let (pa, pb) = (units[a].bx, units[b].bx);
        ((pa[1] + pa[3]) / 2.0).total_cmp(&((pb[1] + pb[3]) / 2.0))
    });
    // lines: units overlapping the current line by more than half the smaller height
    let mut lines: Vec<(f64, f64, Vec<usize>)> = Vec::new();
    for i in by_y {
        let p = units[i].bx;
        match lines.last_mut() {
            Some(l) if (l.1.min(p[3]) - l.0.max(p[1])) > 0.5 * (l.1 - l.0).min(p[3] - p[1]) => {
                l.0 = l.0.min(p[1]);
                l.1 = l.1.max(p[3]);
                l.2.push(i);
            }
            _ => lines.push((p[1], p[3], vec![i])),
        }
    }
    if lines.len() < 2 {
        return vec![g.to_vec()];
    }
    let mut hs: Vec<f64> = lines.iter().map(|l| l.1 - l.0).collect();
    hs.sort_by(f64::total_cmp);
    let line_h = hs[hs.len() / 2].max(1.0);
    let headingish = |l: &[usize]| {
        let (mut head, mut all) = (0f32, 0f32);
        for &i in l {
            let Some(row) = cls.get(i * c..(i + 1) * c) else {
                continue;
            };
            let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let e: Vec<f32> = row.iter().map(|v| (v - m).exp()).collect();
            let z: f32 = e.iter().sum();
            let n = units[i]
                .text
                .chars()
                .filter(|ch| !ch.is_whitespace())
                .count()
                .max(1) as f32;
            let share = if z > 0.0 {
                (e.get(1).copied().unwrap_or(0.0) + e.get(2).copied().unwrap_or(0.0)) / z
            } else {
                0.0
            };
            head += n * share;
            all += n;
        }
        head * 2.0 > all
    };
    let mut parts: Vec<Vec<usize>> = vec![lines[0].2.clone()];
    for k in 1..lines.len() {
        let gap = lines[k].0 - lines[k - 1].1;
        if gap > line_h || headingish(&lines[k].2) != headingish(&lines[k - 1].2) {
            parts.push(Vec::new());
        }
        if let Some(part) = parts.last_mut() {
            part.extend_from_slice(&lines[k].2);
        }
    }
    parts
}

/// Units grouped into blocks from same_block logits: connected components of positive links,
/// then average linkage inside each component (`mod.rs:306`).
fn block_groups(l: &Logits, keep: &[usize]) -> Vec<Vec<usize>> {
    let m = keep.len();
    let mut parent: Vec<usize> = (0..m).collect();
    for a in 0..m {
        for b in a + 1..m {
            if l.pair(4, keep[a], keep[b]) > 0.0 {
                let (ra, rb) = (uf_root(&mut parent, a), uf_root(&mut parent, b));
                if ra != rb {
                    parent[ra] = rb;
                }
            }
        }
    }
    let mut comps: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for a in 0..m {
        let r = uf_root(&mut parent, a);
        comps.entry(r).or_default().push(a);
    }
    let mut out = Vec::new();
    for (_, c) in comps {
        let k = c.len();
        let mut p = vec![0f64; k * k];
        for x in 0..k {
            for y in 0..k {
                p[x * k + y] = l.pair(4, keep[c[x]], keep[c[y]]) as f64;
            }
        }
        for g in avg_link(&p, k, 0.0) {
            out.push(g.into_iter().map(|x| keep[c[x]]).collect());
        }
    }
    out
}

/// Sets the logits of units running against the page's main direction to -9, as
/// `tablelab.run_model` does (`mod.rs:13`). Only the heads actually present are masked, so a
/// four-head model cannot index past its pair tensor.
fn mask_minor(l: &mut Logits, minor: &[bool]) {
    let n = l.n;
    let heads = l.n_pair.min(4);
    if l.node.len() < 2 * n || l.pair.len() < heads * n * n {
        return;
    }
    for (i, &m) in minor.iter().enumerate().take(n) {
        if !m {
            continue;
        }
        l.node[i * 2] = -9.0;
        l.node[i * 2 + 1] = -9.0;
        for k in 0..heads {
            for j in 0..n {
                l.pair[(k * n + i) * n + j] = -9.0;
                l.pair[(k * n + j) * n + i] = -9.0;
            }
        }
    }
}

/// Decodes one page: tables (rows, columns, cells, spans), every unit's layout class, and the
/// layout blocks.
pub(crate) fn decode(
    units: &[DUnit],
    logits: &super::net::Logits,
    options: &DecodeOptions,
) -> Layout {
    let mut layout = Layout::default();
    let n = units.len();
    if n < 2 || logits.n != n {
        return layout;
    }
    if logits.node.len() < 2 * n
        || logits.cls.len() < logits.n_cls.saturating_mul(n)
        || logits.pair.len() < logits.n_pair.max(1) * n * n
    {
        return layout;
    }
    // no model output at all (an all-zero buffer, or a model that did not run): nothing to
    // decode rather than a page of one-unit blocks
    if logits.node.iter().all(|&v| v == 0.0) && logits.cls.iter().all(|&v| v == 0.0) {
        return layout;
    }
    // `mask_minor`: the reference masks before decoding. Never taken on this repository's OCR
    // path (`minor` is always false), and the logits are only copied when it is.
    let minor: Vec<bool> = units.iter().map(|u| u.minor).collect();
    let masked;
    let l = if minor.iter().any(|&m| m) {
        let mut m = Logits {
            n: logits.n,
            node: logits.node.clone(),
            pair: logits.pair.clone(),
            cls: logits.cls.clone(),
            n_cls: logits.n_cls,
            n_pair: logits.n_pair,
        };
        mask_minor(&mut m, &minor);
        masked = m;
        &masked
    } else {
        logits
    };
    for g in decode_tables(units, l, options) {
        let g: Vec<usize> = g.into_iter().filter(|&i| !units[i].minor).collect();
        if g.len() < 2 {
            continue;
        }
        let grid = decode_grid(units, &g, l, options);
        // one column is a list or a stack of lines, not a table; tables of contents, charts and
        // multi-column prose read as grids are no tables either
        if grid.n_cols < 2 || grid.is_toc() || grid.is_chart() || grid.is_prose() {
            continue;
        }
        layout.tables.push(Table {
            bbox: bbox_of(units, &g),
            grid,
        });
    }
    // the class head: one index per unit (0 when the model has no class head)
    if l.n_cls > 0 {
        let c = l.n_cls;
        layout.classes.reserve(n);
        for (i, u) in units.iter().enumerate() {
            if u.minor {
                layout.classes.push(0);
            } else {
                layout.classes.push(argmax(&l.cls[i * c..(i + 1) * c]));
            }
        }
    }
    // blocks: only a joint model (the fifth pair head) has same_block logits to group by
    if l.n_pair == 5 && l.n_cls > 0 {
        let keep: Vec<usize> = (0..n).filter(|&i| !units[i].minor).collect();
        let c = l.n_cls;
        for g in block_groups(l, &keep) {
            let parts: Vec<Vec<usize>> = split_at_gaps(&g, units)
                .into_iter()
                .flat_map(|part| split_paragraphs(&part, units, &l.cls, c))
                .collect();
            for part in parts {
                if part.is_empty() {
                    continue;
                }
                // each part takes its own class
                let class = argmax_sum(&part, &l.cls, c);
                let boxes: Vec<[f64; 4]> = part.iter().map(|&i| units[i].bx).collect();
                let order: Vec<usize> = reading_order(&boxes)
                    .into_iter()
                    .filter_map(|k| part.get(k).copied())
                    .collect();
                layout.blocks.push(Block {
                    bbox: bbox_of(units, &part),
                    class,
                    units: order,
                });
            }
        }
    }
    layout
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit(x0: f64, y0: f64, x1: f64, y1: f64, text: &str) -> DUnit {
        DUnit {
            bx: [x0, y0, x1, y1],
            text: text.into(),
            // the OCR path's unpadded line height (`ocr_units`)
            h: (y1 - y0) / 1.4,
            minor: false,
        }
    }

    /// All-zero logits with five pair heads and thirteen classes, to be filled in by a case.
    fn logits(n: usize) -> Logits {
        Logits {
            n,
            node: vec![0.0; 2 * n],
            pair: vec![0.0; 5 * n * n],
            cls: vec![0.0; n * 13],
            n_cls: 13,
            n_pair: 5,
        }
    }

    fn set_pair(l: &mut Logits, k: usize, i: usize, j: usize, v: f32) {
        let n = l.n;
        l.pair[(k * n + i) * n + j] = v;
        l.pair[(k * n + j) * n + i] = v;
    }

    /// 2 rows x 3 columns with a header spanning all three columns: one wide unit on top, one
    /// unit per column below.
    fn spanning_header_case() -> (Vec<DUnit>, Logits) {
        let units = vec![
            unit(0.0, 0.0, 300.0, 10.0, "Header"),
            unit(0.0, 20.0, 100.0, 30.0, "a"),
            unit(100.0, 20.0, 200.0, 30.0, "b"),
            unit(200.0, 20.0, 300.0, 30.0, "c"),
        ];
        let mut l = logits(4);
        for i in 0..4 {
            l.node[i * 2] = 2.0; // in_table
            l.node[i * 2 + 1] = if i == 0 { 2.0 } else { -2.0 }; // header
            l.cls[i * 13] = 2.0; // Text
        }
        for i in 0..4 {
            for j in 0..4 {
                if i != j {
                    set_pair(&mut l, 0, i, j, 2.0); // same_table
                    set_pair(&mut l, 3, i, j, -2.0); // same_cell
                    set_pair(&mut l, 4, i, j, 2.0); // same_block
                }
            }
        }
        // same_row: the body units share a row, the header is a row of its own
        for i in 1..4 {
            for j in 1..4 {
                if i != j {
                    set_pair(&mut l, 1, i, j, 2.0);
                }
            }
        }
        for j in 1..4 {
            set_pair(&mut l, 1, 0, j, -2.0);
        }
        // same_col: the header is in every column, the body cells in one each
        for j in 1..4 {
            set_pair(&mut l, 2, 0, j, 2.0);
        }
        for i in 1..4 {
            for j in 1..4 {
                if i != j {
                    set_pair(&mut l, 2, i, j, -2.0);
                }
            }
        }
        (units, l)
    }

    #[test]
    fn avg_link_hand_matrix() {
        // 0-1 and 2-3 link; 4 joins 2-3 at 0.45; the rest average under 0.5
        let n = 5;
        let mut p = vec![-0.1f64; n * n];
        let v = [
            [1.0, 0.9, 0.2, 0.1, 0.1],
            [0.9, 1.0, 0.2, 0.1, 0.1],
            [0.2, 0.2, 1.0, 0.85, 0.1],
            [0.1, 0.1, 0.85, 1.0, 0.8],
            [0.1, 0.1, 0.1, 0.8, 1.0],
        ];
        for i in 0..n {
            for j in 0..n {
                p[i * n + j] = v[i][j];
            }
        }
        // {2,3} merges at 0.85, then its mean with 4 is (0.1 + 0.8) / 2 = 0.45
        assert_eq!(avg_link(&p, n, 0.4), vec![vec![0, 1], vec![2, 3, 4]]);
        assert_eq!(avg_link(&p, n, 0.5), vec![vec![0, 1], vec![2, 3], vec![4]]);
        // the merge test is strict (`v <= thresh` stops): 0.9 does not merge at 0.9
        assert_eq!(
            avg_link(&p, n, 0.9),
            vec![vec![0], vec![1], vec![2], vec![3], vec![4]]
        );
        // ties go to the first live group in `alive` order, and groups stay sorted
        let mut q = vec![-1f64; 4 * 4];
        for i in 0..4 {
            q[i * 4 + i] = 1.0;
        }
        q[0 * 4 + 2] = 1.0;
        q[2 * 4 + 0] = 1.0;
        q[1 * 4 + 3] = 1.0;
        q[3 * 4 + 1] = 1.0;
        assert_eq!(avg_link(&q, 4, 0.0), vec![vec![0, 2], vec![1, 3]]);
    }

    /// The exhaustive average linkage `avg_link` replaced (rescans every live pair per merge):
    /// merges, order and sums must match exactly.
    fn avg_link_exhaustive(p: &[f64], n: usize, thresh: f64) -> Vec<Vec<usize>> {
        let mut groups: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
        let mut s = p.to_vec();
        let mut cnt = vec![1f64; n * n];
        let mut alive: Vec<usize> = (0..n).collect();
        while alive.len() > 1 {
            let m = alive.len();
            let (mut best, mut ba, mut bb) = (f64::NEG_INFINITY, 0, 0);
            for a in 0..m {
                for b in 0..m {
                    if a == b {
                        continue;
                    }
                    let (ga, gb) = (alive[a], alive[b]);
                    let v = s[ga * n + gb] / cnt[ga * n + gb];
                    if v > best {
                        best = v;
                        ba = a;
                        bb = b;
                    }
                }
            }
            if best <= thresh {
                break;
            }
            let (ga, gb) = (alive[ba], alive[bb]);
            let moved = std::mem::take(&mut groups[gb]);
            groups[ga].extend(moved);
            for k in 0..n {
                s[ga * n + k] += s[gb * n + k];
                cnt[ga * n + k] += cnt[gb * n + k];
            }
            for k in 0..n {
                s[k * n + ga] += s[k * n + gb];
                cnt[k * n + ga] += cnt[k * n + gb];
            }
            alive.remove(bb);
        }
        alive
            .iter()
            .map(|&g| {
                let mut v = groups[g].clone();
                v.sort_unstable();
                v
            })
            .collect()
    }

    #[test]
    fn avg_link_matches_exhaustive_reference() {
        // xorshift: deterministic matrices without a rand dependency
        let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let levels = [-2.0, -1.0, -0.5, 0.0, 0.25, 0.5, 1.0, 2.0];
        for case in 0..150 {
            let n = 1 + (next() % 40) as usize;
            let symmetric = case % 4 != 3;
            let discrete = case % 2 == 0; // small value set: many exact ties
            let mut p = vec![0f64; n * n];
            for i in 0..n {
                for j in 0..n {
                    if symmetric && j < i {
                        p[i * n + j] = p[j * n + i];
                        continue;
                    }
                    p[i * n + j] = if discrete {
                        levels[(next() % levels.len() as u64) as usize]
                    } else {
                        (next() % 20_001) as f64 / 5_000.0 - 2.0
                    };
                }
            }
            for thresh in [0.0, -0.75, 0.6] {
                assert_eq!(
                    avg_link(&p, n, thresh),
                    avg_link_exhaustive(&p, n, thresh),
                    "case {case} n {n} symmetric {symmetric} discrete {discrete} thresh {thresh}"
                );
            }
        }
    }

    #[test]
    fn reading_order_two_column_page() {
        // units inserted right column first: reading order is line by line, left to right
        let boxes = vec![
            [200.0, 0.0, 300.0, 10.0],  // right, line 0
            [0.0, 0.0, 100.0, 10.0],    // left, line 0
            [200.0, 20.0, 300.0, 30.0], // right, line 1
            [0.0, 20.0, 100.0, 30.0],   // left, line 1
        ];
        assert_eq!(reading_order(&boxes), vec![1, 0, 3, 2]);
        assert_eq!(reading_order(&[]), Vec::<usize>::new());
    }

    #[test]
    fn decode_grid_spanning_header() {
        let (units, l) = spanning_header_case();
        // two rows of body text: the near-empty column fold would otherwise pull the three
        // one-cell columns together (see `decode_grid_squeezes_near_empty_column`)
        let opt = DecodeOptions {
            squeeze: false,
            ..DecodeOptions::default()
        };
        assert_eq!(decode_tables(&units, &l, &opt), vec![vec![0, 1, 2, 3]]);
        let grid = decode_grid(&units, &[0, 1, 2, 3], &l, &opt);
        assert_eq!((grid.n_rows, grid.n_cols), (2, 3));
        assert_eq!(grid.head_rows, vec![0]);
        assert_eq!(grid.cells.len(), 4);
        let head = &grid.cells[0];
        assert_eq!((head.r0, head.c0, head.r1, head.c1), (0, 0, 0, 2));
        assert_eq!(head.text, "Header");
        assert_eq!(head.bx, [0.0, 0.0, 300.0, 10.0]);
        for (k, (c, text)) in [(0usize, "a"), (1, "b"), (2, "c")].into_iter().enumerate() {
            let cell = &grid.cells[k + 1];
            assert_eq!(
                (cell.r0, cell.c0, cell.r1, cell.c1),
                (1, c, 1, c),
                "cell {k}"
            );
            assert_eq!(cell.text, text, "cell {k}");
        }
    }

    #[test]
    fn decode_grid_squeezes_near_empty_column() {
        // the same table at the release defaults: every body column holds one non-header cell,
        // so the first column folds into the second (text appended left-to-right)
        let (units, l) = spanning_header_case();
        let grid = decode_grid(&units, &[0, 1, 2, 3], &l, &DecodeOptions::default());
        assert_eq!(grid.n_cols, 2);
        assert_eq!(grid.n_rows, 2);
        let body: Vec<&str> = grid
            .cells
            .iter()
            .filter(|c| c.r0 == 1)
            .map(|c| c.text.as_str())
            .collect();
        assert_eq!(body, vec!["a b", "c"]);
    }

    #[test]
    fn forced_links_join_stacked_groups() {
        // three stacked lines of one column: A-B and B-C are forced, A-C is not, and the
        // average linkage splits them because A-C is strongly negative
        let units = vec![
            unit(0.0, 0.0, 100.0, 10.0, "l1"),
            unit(0.0, 12.0, 100.0, 22.0, "l2"),
            unit(0.0, 24.0, 100.0, 34.0, "l3"),
        ];
        let mut l = logits(3);
        for i in 0..3 {
            l.node[i * 2] = 2.0;
            l.cls[i * 13] = 2.0;
        }
        for (i, j) in [(0, 1), (1, 2), (0, 2)] {
            set_pair(&mut l, 0, i, j, 2.0);
            set_pair(&mut l, 3, i, j, -10.0); // same_cell: never links
            set_pair(&mut l, 4, i, j, 2.0);
        }
        for (i, j) in [(0, 1), (1, 2)] {
            set_pair(&mut l, 1, i, j, 4.0); // same_row
            set_pair(&mut l, 2, i, j, 4.0); // same_col
        }
        set_pair(&mut l, 1, 0, 2, -10.0);
        set_pair(&mut l, 2, 0, 2, -10.0);
        let joined = decode_grid(&units, &[0, 1, 2], &l, &DecodeOptions::default());
        assert_eq!((joined.n_rows, joined.n_cols), (1, 1));
        assert_eq!(joined.cells.len(), 1);
        assert_eq!(joined.cells[0].text, "l1 l2 l3");
        // without the forced-link repair the same logits split into a 2x2 grid
        let opt = DecodeOptions {
            forced_link: false,
            ..DecodeOptions::default()
        };
        let split = decode_grid(&units, &[0, 1, 2], &l, &opt);
        assert_eq!((split.n_rows, split.n_cols), (2, 2));
        assert_eq!(split.cells.len(), 2);
    }

    #[test]
    fn decode_returns_tables_classes_and_blocks() {
        let (units, l) = spanning_header_case();
        let opt = DecodeOptions {
            squeeze: false,
            ..DecodeOptions::default()
        };
        let layout = decode(&units, &l, &opt);
        assert_eq!(layout.tables.len(), 1);
        assert_eq!(layout.tables[0].bbox, [0.0, 0.0, 300.0, 30.0]);
        assert_eq!(layout.tables[0].grid.n_cols, 3);
        assert_eq!(layout.classes, vec![0, 0, 0, 0]);
        assert_eq!(layout.blocks.len(), 1);
        assert_eq!(layout.blocks[0].class, 0);
        assert_eq!(layout.blocks[0].units, vec![0, 1, 2, 3]);
        assert_eq!(layout.blocks[0].bbox, [0.0, 0.0, 300.0, 30.0]);
    }

    #[test]
    fn block_units_are_in_reading_order() {
        // one block of two lines, two columns, inserted right-then-left per line: the block's
        // units must be in reading order, so joining their texts with one space reproduces the
        // reference's `ModelBlock.text`
        let units = vec![
            unit(105.0, 0.0, 205.0, 10.0, "r0"),
            unit(0.0, 0.0, 100.0, 10.0, "l0"),
            unit(105.0, 20.0, 205.0, 30.0, "r1"),
            unit(0.0, 20.0, 100.0, 30.0, "l1"),
        ];
        let mut l = logits(4);
        for i in 0..4 {
            l.cls[i * 13] = 2.0; // Text
            for j in 0..4 {
                if i != j {
                    set_pair(&mut l, 4, i, j, 2.0); // same_block: one block
                }
            }
        }
        let layout = decode(&units, &l, &DecodeOptions::default());
        assert!(layout.tables.is_empty());
        assert_eq!(layout.blocks.len(), 1);
        assert_eq!(layout.blocks[0].units, vec![1, 0, 3, 2]);
        let text = layout.blocks[0]
            .units
            .iter()
            .map(|&i| units[i].text.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        assert_eq!(text, "l0 r0 l1 r1");
    }

    #[test]
    fn wrapped_header_lines_fold() {
        let hcell = |r: usize, c: usize, text: &str, y0: f64, y1: f64| GridCell {
            r0: r,
            c0: c,
            r1: r,
            c1: c,
            text: text.into(),
            bx: [c as f64 * 100.0, y0, c as f64 * 100.0 + 90.0, y1],
        };
        // "Product" / "(active substance(s))" and "Recommendation" / "(additions to text)"
        let mut cells = vec![
            hcell(0, 0, "Product", 76.6, 83.6),
            hcell(0, 1, "Date", 76.6, 83.6),
            hcell(0, 2, "Recommendation", 76.6, 83.6),
            hcell(1, 0, "(active substance(s))", 87.5, 96.2),
            hcell(1, 2, "(additions to text)", 87.5, 96.2),
            hcell(2, 0, "Purevax", 110.0, 118.0),
            hcell(2, 1, "5-7 October 2021", 110.0, 118.0),
            hcell(2, 2, "Section 4.6", 110.0, 118.0),
        ];
        let (mut head, mut n) = (vec![0], 3);
        fold_header_wraps(&mut cells, &mut head, &mut n);
        assert_eq!(n, 2);
        assert!(cells
            .iter()
            .any(|c| c.text == "Product (active substance(s))"));
        // a text-only list: no continuation cue, no fold
        let mut list = vec![
            hcell(0, 0, "Alfa Romeo", 10.0, 18.0),
            hcell(0, 1, "Audi", 10.0, 18.0),
            hcell(1, 0, "Buick Reatta", 20.0, 28.0),
            hcell(1, 1, "Cadillac", 20.0, 28.0),
        ];
        let (mut head, mut n) = (vec![], 2);
        fold_header_wraps(&mut list, &mut head, &mut n);
        assert_eq!(n, 2);
        // one lowercase tail in a wide row: a body cell's wrap, not a header line
        let mut tail = vec![
            hcell(0, 0, "Country", 10.0, 18.0),
            hcell(0, 1, "Sector", 10.0, 18.0),
            hcell(0, 2, "Restrictions", 10.0, 18.0),
            hcell(1, 2, "right required to acquire land", 20.0, 28.0),
        ];
        let (mut head, mut n) = (vec![0], 2);
        fold_header_wraps(&mut tail, &mut head, &mut n);
        assert_eq!(n, 2);
    }

    #[test]
    fn captions_are_recognised() {
        for t in [
            "Table 1: results",
            "tab.2",
            "Figures 3-4",
            "NOTE: x",
            "chart of y",
        ] {
            assert!(is_caption(t), "{t}");
        }
        // "tab. 2" does not match: the reference's `\b` sits between "." and " ", which is
        // not a word boundary (the same as Python's `re.match`)
        for t in ["tablet sales", "notebook", "Tab", "The table", "tab. 2"] {
            assert!(!is_caption(t), "{t}");
        }
    }

    #[test]
    fn minor_units_are_masked_out() {
        // three units on one line; the middle one runs against the page's main direction, so
        // `mask_minor` sets its logits to -9 before decoding and it takes no part
        let mut units = vec![
            unit(0.0, 0.0, 100.0, 10.0, "a"),
            unit(100.0, 0.0, 200.0, 10.0, "b"),
            unit(200.0, 0.0, 300.0, 10.0, "c"),
        ];
        units[1].minor = true;
        let mut l = logits(3);
        for i in 0..3 {
            l.node[i * 2] = 2.0;
            l.cls[i * 13] = 2.0;
            for j in 0..3 {
                if i != j {
                    set_pair(&mut l, 0, i, j, 2.0); // same_table
                    set_pair(&mut l, 1, i, j, 2.0); // same_row
                    set_pair(&mut l, 2, i, j, -2.0); // same_col
                    set_pair(&mut l, 3, i, j, -2.0); // same_cell
                    set_pair(&mut l, 4, i, j, 2.0); // same_block
                }
            }
        }
        let layout = decode(&units, &l, &DecodeOptions::default());
        assert_eq!(layout.tables.len(), 1);
        let texts: Vec<&str> = layout.tables[0]
            .grid
            .cells
            .iter()
            .map(|c| c.text.as_str())
            .collect();
        assert_eq!(texts, vec!["a", "c"]);
        assert_eq!(layout.classes.len(), 3);
        assert_eq!(layout.classes[1], 0);
        assert!(layout.blocks.iter().all(|b| !b.units.contains(&1)));
    }

    #[test]
    fn degenerate_inputs_decode_to_nothing() {
        let opt = DecodeOptions::default();
        // no units
        let empty = logits(0);
        assert!(decode(&[], &empty, &opt).tables.is_empty());
        assert_eq!(decode(&[], &empty, &opt).blocks.len(), 0);
        assert_eq!(decode(&[], &empty, &opt).classes.len(), 0);
        assert!(avg_link(&[], 0, 0.0).is_empty());
        assert!(decode_grid(&[], &[], &empty, &opt).cells.is_empty());
        assert_eq!(decode_grid(&[], &[], &empty, &opt).n_rows, 0);
        // one unit: not a table and not a block page
        let one = vec![unit(0.0, 0.0, 10.0, 10.0, "x")];
        let l1 = logits(1);
        let layout = decode(&one, &l1, &opt);
        assert!(layout.tables.is_empty());
        assert!(layout.blocks.is_empty());
        // all-zero logits: the model produced nothing, so nothing decodes
        let two = vec![
            unit(0.0, 0.0, 10.0, 10.0, "a"),
            unit(20.0, 0.0, 30.0, 10.0, "b"),
        ];
        let l2 = logits(2);
        let layout = decode(&two, &l2, &opt);
        assert!(layout.tables.is_empty());
        assert!(layout.blocks.is_empty());
        assert!(layout.classes.is_empty());
    }

    #[test]
    fn degenerate_boxes_and_flat_logits() {
        let opt = DecodeOptions::default();
        // every box identical and every pair logit identical: one cell, one column, so the
        // page has no table and nothing panics
        let same = vec![
            unit(0.0, 0.0, 10.0, 10.0, "a"),
            unit(0.0, 0.0, 10.0, 10.0, "b"),
            unit(0.0, 0.0, 10.0, 10.0, "c"),
        ];
        let mut l = logits(3);
        for i in 0..3 {
            l.node[i * 2] = 1.0;
            l.cls[i * 13] = 2.0;
            for j in 0..3 {
                if i != j {
                    for k in 0..5 {
                        set_pair(&mut l, k, i, j, 1.0);
                    }
                }
            }
        }
        let layout = decode(&same, &l, &opt);
        assert!(layout.tables.is_empty());
        assert_eq!(layout.classes.len(), same.len());
        // a real 2x2 arrangement with flat logits: every cell merges into one, so again no table
        let grid = vec![
            unit(0.0, 0.0, 100.0, 10.0, "a"),
            unit(100.0, 0.0, 200.0, 10.0, "b"),
            unit(0.0, 20.0, 100.0, 30.0, "c"),
            unit(100.0, 20.0, 200.0, 30.0, "d"),
        ];
        let mut l = logits(4);
        for i in 0..4 {
            l.node[i * 2] = 1.0;
            l.cls[i * 13] = 2.0;
            for j in 0..4 {
                if i != j {
                    for k in 0..5 {
                        set_pair(&mut l, k, i, j, 1.0);
                    }
                }
            }
        }
        let layout = decode(&grid, &l, &opt);
        assert!(layout.tables.is_empty());
        assert_eq!(layout.classes.len(), grid.len());
        assert!(!layout.blocks.is_empty());
        // no class head (n_cls == 0): `classes` is empty, which the caller reads as "the model
        // did not run", and there are no blocks to group
        let (units, mut l) = spanning_header_case();
        l.n_cls = 0;
        l.cls.clear();
        let layout = decode(&units, &l, &opt);
        assert!(layout.classes.is_empty());
        assert!(layout.blocks.is_empty());
        assert_eq!(layout.tables.len(), 1);
    }
}
