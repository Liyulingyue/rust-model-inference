//! `resolve_overlaps` — shared, deterministic span-conflict resolution.
//!
//! Mirrors `gliner2.inference.overlap.resolve_overlaps`
//! (`target/gliner2-oracle/gliner2/inference/overlap.py:56`).
//!
//! `gliner2.5-base-v1` ships `overlap_policy = "flat"`, which normalizes to
//! `disallow`: the maximum-total-score set of non-overlapping spans. That is a
//! weighted-interval-scheduling problem, not a greedy pass, so thresholding and
//! sorting alone leaves overlapping spans in the output.
//!
//! The contract, in order:
//!  1. Rank by `(-score, start, end, original_index)`.
//!  2. Collapse exact-boundary duplicates to their best-ranked representative.
//!  3. Apply the policy.
//!  4. Return ranked by the same key.
//!
//! Policies:
//!  - `allow` — keep every distinct span.
//!  - `nested` — keep disjoint and containment overlaps, reject crossings.
//!  - `flat` / `disallow` — maximum-total-score non-overlapping set.
//!  - `longest` — drop spans strictly contained in another candidate.
//!
//! `disallow`'s dynamic program has three tie-breaks in a fixed order (total
//! score, then number of spans, then the lexicographically better ranking), and
//! they are load-bearing: a chunked merge can hand it zero-confidence entries
//! that would otherwise win by being first. All three are reproduced.

/// A scored half-open span, the minimal input the resolver needs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScoredSpan {
    pub score: f32,
    pub start: usize,
    pub end: usize,
}

/// The canonical policy names, after alias resolution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OverlapPolicy {
    Allow,
    Nested,
    Disallow,
    Longest,
}

/// Normalize a policy name, or fall back to `default` when it is `None`.
///
/// `None` is resolved *only* through `default` so a caller can preserve an
/// architecture default rather than accidentally imposing one. Mirrors
/// `normalize_overlap_policy`.
pub fn normalize_overlap_policy(
    policy: Option<&str>,
    default: &str,
) -> Result<OverlapPolicy, String> {
    let selected = policy.unwrap_or(default);
    let key = selected.trim().to_lowercase().replace('-', "_");
    Ok(match key.as_str() {
        "allow" | "all" | "none" => OverlapPolicy::Allow,
        "nested" | "allow_nested" => OverlapPolicy::Nested,
        "flat" | "disallow" | "no_overlap" | "non_overlapping" => OverlapPolicy::Disallow,
        "longest" | "keep_longest" => OverlapPolicy::Longest,
        other => {
            return Err(format!(
                "unknown overlap_policy {other:?}; expected one of: \
                 allow, nested, flat/disallow, longest"
            ))
        }
    })
}

/// `(-score, start, end, original_index)` — the single ranking used everywhere.
fn rank_key(span: &ScoredSpan, index: usize) -> (OrderedF32, usize, usize, usize) {
    (OrderedF32(-span.score), span.start, span.end, index)
}

/// `total_cmp` on the negated score, so NaN sorts the same way torch/Python's
/// `-score` tuple comparison does rather than panicking.
#[derive(Clone, Copy, Debug, PartialEq)]
struct OrderedF32(f32);

impl Eq for OrderedF32 {}

impl PartialOrd for OrderedF32 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrderedF32 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

/// Resolve overlaps among `items` under `policy`.
///
/// `items` are ranked by `(-score, start, end, index)`; the returned indices are
/// into `items`, in ranked order. Duplicate boundaries collapse to the
/// best-ranked representative before the policy runs.
pub fn resolve_overlaps(items: &[ScoredSpan], policy: OverlapPolicy) -> Vec<usize> {
    if items.is_empty() {
        return Vec::new();
    }
    let mut ranked: Vec<usize> = (0..items.len()).collect();
    ranked.sort_by_key(|&index| rank_key(&items[index], index));

    // Collapse exact-boundary duplicates, keeping the best-ranked occurrence.
    let mut distinct: Vec<usize> = Vec::with_capacity(ranked.len());
    let mut seen: Vec<(usize, usize)> = Vec::with_capacity(ranked.len());
    for index in ranked {
        let boundaries = (items[index].start, items[index].end);
        if seen.contains(&boundaries) {
            continue;
        }
        seen.push(boundaries);
        distinct.push(index);
    }

    let kept = match policy {
        OverlapPolicy::Allow => distinct,
        OverlapPolicy::Nested => {
            let mut kept: Vec<usize> = Vec::new();
            for candidate in distinct {
                let span = items[candidate];
                let crossing = kept.iter().any(|&existing| {
                    let other = items[existing];
                    let overlaps = span.start < other.end && other.start < span.end;
                    let contains = (span.start <= other.start && other.end <= span.end)
                        || (other.start <= span.start && span.end <= other.end);
                    overlaps && !contains
                });
                if !crossing {
                    kept.push(candidate);
                }
            }
            kept
        }
        OverlapPolicy::Longest => {
            // Every distinct span is a candidate container, not just the ones
            // already kept — the reference scans `distinct`, so a contained span
            // is dropped even when its container is itself contained.
            distinct
                .iter()
                .copied()
                .filter(|&candidate| {
                    let span = items[candidate];
                    !distinct.iter().any(|&other| {
                        let wider = items[other];
                        wider.start <= span.start
                            && span.end <= wider.end
                            && (wider.start < span.start || span.end < wider.end)
                    })
                })
                .collect()
        }
        OverlapPolicy::Disallow => maximum_score_non_overlapping(items, &distinct),
    };

    let mut result = kept;
    result.sort_by_key(|&index| rank_key(&items[index], index));
    result
}

/// Weighted interval scheduling, matching the reference's tie-breaks exactly.
///
/// The DP runs over spans sorted by `(end, start, -score, index)`. For each
/// span, the candidate optimum either takes it (extending the best solution
/// among its predecessors) or skips it. Ties go to: higher total score, then
/// the larger set, then the lexicographically better ranked selection — the
/// last one because two equal-scoring sets must resolve deterministically.
fn maximum_score_non_overlapping(items: &[ScoredSpan], distinct: &[usize]) -> Vec<usize> {
    let mut by_end: Vec<usize> = distinct.to_vec();
    by_end.sort_by_key(|&index| {
        let span = items[index];
        (span.end, span.start, OrderedF32(-span.score), index)
    });
    let ends: Vec<usize> = by_end.iter().map(|&index| items[index].end).collect();
    // `bisect_right(ends, start, 0, index) - 1`: the last span that ends at or
    // before this one starts. Half-open spans may touch.
    let predecessors: Vec<usize> = by_end
        .iter()
        .enumerate()
        .map(|(position, &index)| {
            let start = items[index].start;
            ends[..position].partition_point(|end| *end <= start) - 1
        })
        .collect();

    // `best[k]` is the optimum over the first `k` spans: (score, selection).
    let mut best: Vec<(f32, Vec<usize>)> = vec![(0.0, Vec::new())];
    for (position, &index) in by_end.iter().enumerate() {
        let (previous_score, previous_selection) = &best[predecessors[position] + 1];
        let with_item = (previous_score + items[index].score, {
            let mut selection = previous_selection.clone();
            selection.push(position);
            selection
        });
        let without_item = best[position].clone();
        let winner = if with_item.0 > without_item.0 {
            with_item
        } else if with_item.0 < without_item.0 {
            without_item
        } else if with_item.1.len() > without_item.1.len() {
            with_item
        } else if with_item.1.len() < without_item.1.len() {
            without_item
        } else if selection_key(items, &by_end, &with_item.1)
            < selection_key(items, &by_end, &without_item.1)
        {
            with_item
        } else {
            without_item
        };
        best.push(winner);
    }

    best.pop()
        .map(|(_, selection)| selection.into_iter().map(|slot| by_end[slot]).collect())
        .unwrap_or_default()
}

/// The ranked-selection key the final tie-break compares: the rank keys of the
/// chosen spans, sorted. Smaller is better, and a prefix is better than the
/// longer selection that extends it.
fn selection_key(
    items: &[ScoredSpan],
    by_end: &[usize],
    selection: &[usize],
) -> Vec<(OrderedF32, usize, usize, usize)> {
    let mut keys: Vec<(OrderedF32, usize, usize, usize)> = selection
        .iter()
        .map(|&slot| {
            let index = by_end[slot];
            rank_key(&items[index], index)
        })
        .collect();
    keys.sort();
    keys
}
