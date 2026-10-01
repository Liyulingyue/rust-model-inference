//! `TypedRelationPairGenerator` + `SparseRelationScorer` — the `[R]` head.
//!
//! Mirrors `gliner2.models.boundary.relations`
//! (`target/gliner2-oracle/gliner2/models/boundary/relations.py`).
//!
//! Relations do **not** introduce a second extraction representation: they
//! reuse the entity mention candidates the span path already produced. Pair
//! generation is *typed and capped* — for each relation type, keep the
//! top `relation_heads_per_type` head-typed mentions and the top
//! `relation_tails_per_type` tail-typed ones, then score their capped cross
//! product. Work is therefore `O(Rh*Rt)` per relation type with fixed caps,
//! never the `O(N^2)` all-pairs matrix.
//!
//! Three details here are load-bearing and easy to get wrong:
//!
//! 1. **Argument probabilities are un-temperatured.** The span decode path uses
//!    `sigmoid(pair_logits / pair_temperature)`
//!    (`engine.py:84`), but `generate_batched` re-derives its own
//!    `probs = sigmoid(candidates.pair_logits)` (`relations.py:141`) for the
//!    argument threshold. With base-v1's `pair_temperature = 1.0` the two
//!    coincide, so no fixture on this checkpoint can tell the difference — this
//!    port keeps them distinct anyway, because a checkpoint with
//!    `pair_temperature != 1.0` would otherwise silently threshold arguments
//!    differently from the reference.
//!
//! 2. **The mention ordering is a lexicographic `(start, end)` sort, not a
//!    score sort.** `select` does two stable argsorts — by end, then by start —
//!    which is exactly a stable sort on the key `(start, end)`. It exists only
//!    to give the later score sort a deterministic tie-break: mentions with
//!    equal score come out in document order.
//!
//! 3. **The final pair top-k is over `head_prob * tail_prob`**, with a stable
//!    descending sort so ties break head-major, then tail. It is *not* a sort by
//!    the relation logit — the scorer runs afterwards, on the survivors.

use crate::core::tensor::TensorSource;
use crate::ops::kernel::{QuantizedTensor, Weight};

use super::settings::BoundarySettings;

/// Capping rules for one relation type (`RelationProposalSettings`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RelationProposalSettings {
    pub heads_per_relation: usize,
    pub tails_per_relation: usize,
    pub pair_cap: usize,
    /// A mention only qualifies as an argument at or above this probability.
    /// base-v1 uses 0.2, not the reference's dataclass default of 0.0.
    pub argument_threshold: f32,
}

impl RelationProposalSettings {
    /// Read the caps from the checkpoint's transcribed settings. Inference uses
    /// only this constructor, so there is nowhere for a default to creep in.
    pub fn from_settings(settings: &BoundarySettings) -> Self {
        Self {
            heads_per_relation: settings.relation_heads_per_type,
            tails_per_relation: settings.relation_tails_per_type,
            pair_cap: settings.relation_pair_cap,
            argument_threshold: settings.relation_argument_proposal_threshold,
        }
    }
}

/// One relation type and the entity queries allowed to fill each role.
///
/// `RelationTypeSpec`. The reference builds one spec per relation group, with
/// `head_query_ids` / `tail_query_ids` naming the group's first two fields and
/// `allow_self` left at `False` (`model.py:1396-1407`, `model.py:1494-1498`).
#[derive(Clone, Debug, PartialEq)]
pub struct RelationTypeSpec {
    pub relation_type: String,
    pub head_query_ids: Vec<usize>,
    pub tail_query_ids: Vec<usize>,
    pub allow_self: bool,
}

impl RelationTypeSpec {
    /// The common case the reference builds: the group's first field is the
    /// head, its second is the tail.
    pub fn two_role(relation_type: impl Into<String>, head: usize, tail: usize) -> Self {
        Self {
            relation_type: relation_type.into(),
            head_query_ids: vec![head],
            tail_query_ids: vec![tail],
            allow_self: false,
        }
    }
}

/// One proposed relation pair, already compacted: invalid pairs are absent.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RelationPair {
    /// Index into the relation-spec list.
    pub relation_index: usize,
    /// Half-open word/token offsets. `head_end - 1` is the last covered token,
    /// which is how `SparseRelationScorer` gathers the endpoint states.
    pub head_start: usize,
    pub head_end: usize,
    pub tail_start: usize,
    pub tail_end: usize,
    /// `sigmoid(head pair logit)` and `sigmoid(tail pair logit)`, the un-
    /// temperatured mention probabilities the pair score is the product of.
    pub head_prob: f32,
    pub tail_prob: f32,
    /// The query slot each mention came from, kept for the reference's
    /// `head_keys` / `tail_keys` presentation metadata.
    pub head_query: usize,
    pub tail_query: usize,
}

/// The `DocumentCandidatePool` view the generator consumes.
///
/// Only the fields the generator reads: the mention offsets, the raw pair
/// logits, and the validity mask (which the reference ANDs with `query_mask`,
/// so padding and inactive queries arrive already folded together).
pub struct RelationCandidates<'a> {
    /// `candidates.indices`, `[Q, C, 2]`.
    pub indices: &'a [usize],
    /// `candidates.pair_logits`, `[Q, C]`.
    pub pair_logits: &'a [f32],
    /// `candidates.valid_mask & query_mask[..., None]`, `[Q, C]`.
    pub valid_mask: &'a [bool],
    pub q_count: usize,
    pub c_count: usize,
}

/// Generate typed, capped relation pairs for one document.
///
/// Batch size is 1 because that is what decode does: `_decode_relations` slices
/// a single sample out of the batch and calls `generate` on it
/// (`engine.py:816-822`). The reference's training path runs the same routine
/// over a whole batch, which this port does not attempt.
pub fn generate_typed_relation_pairs(
    candidates: &RelationCandidates<'_>,
    specs: &[RelationTypeSpec],
    settings: &RelationProposalSettings,
) -> Vec<RelationPair> {
    let q_count = candidates.q_count;
    let c_count = candidates.c_count;
    if specs.is_empty() || q_count == 0 || c_count == 0 {
        return Vec::new();
    }
    let flat_len = q_count * c_count;

    // `probs = sigmoid(candidates.pair_logits)` — see the module note on why
    // this is deliberately not divided by `pair_temperature`.
    let mut probs = Vec::with_capacity(flat_len);
    for &logit in &candidates.pair_logits[..flat_len] {
        probs.push(sigmoid(logit));
    }

    let mut out = Vec::new();
    for (relation_index, spec) in specs.iter().enumerate() {
        // `head_member` / `tail_member`: which queries may fill each role, with
        // out-of-range ids dropped exactly as the reference does.
        let head_member: Vec<bool> = (0..q_count)
            .map(|q| spec.head_query_ids.contains(&q))
            .collect();
        let tail_member: Vec<bool> = (0..q_count)
            .map(|q| spec.tail_query_ids.contains(&q))
            .collect();

        let head_valid = |slot: usize| {
            let (q, c) = (slot / c_count, slot % c_count);
            candidates.valid_mask[q * c_count + c]
                && head_member[q]
                && probs[slot] >= settings.argument_threshold
        };
        let tail_valid = |slot: usize| {
            let (q, c) = (slot / c_count, slot % c_count);
            candidates.valid_mask[q * c_count + c]
                && tail_member[q]
                && probs[slot] >= settings.argument_threshold
        };

        let heads = select_mentions(
            candidates,
            &probs,
            &head_valid,
            settings.heads_per_relation,
        );
        let tails = select_mentions(
            candidates,
            &probs,
            &tail_valid,
            settings.tails_per_relation,
        );

        // `pair_score = hp[..., None] * tp[..., None, :]` over the capped cross
        // product, minus the same-span pairs, then a stable top-`pair_cap`.
        let mut scored: Vec<(usize, usize, f32)> = Vec::with_capacity(heads.len() * tails.len());
        for (hi, head) in heads.iter().enumerate() {
            if !head.valid {
                continue;
            }
            for (ti, tail) in tails.iter().enumerate() {
                if !tail.valid {
                    continue;
                }
                // `same_span`: the reference compares both endpoints, so a
                // head and tail with identical offsets are the same span no
                // matter which query produced them. `allow_self` is never set
                // on the decode path, but honour it rather than assuming.
                let same_span = head.start == tail.start && head.end == tail.end;
                if same_span && !spec.allow_self {
                    continue;
                }
                scored.push((hi, ti, head.prob * tail.prob));
            }
        }
        // `torch.argsort(..., descending=True, stable=True)`: ties keep the
        // flattened order, which is head-major then tail.
        let mut order: Vec<usize> = (0..scored.len()).collect();
        order.sort_by(|&a, &b| {
            f32_order(scored[b].2, scored[a].2).then(a.cmp(&b))
        });
        order.truncate(settings.pair_cap);
        for slot in order {
            let (hi, ti, _) = scored[slot];
            let head = heads[hi];
            let tail = tails[ti];
            out.push(RelationPair {
                relation_index,
                head_start: head.start,
                head_end: head.end,
                tail_start: tail.start,
                tail_end: tail.end,
                head_prob: head.prob,
                tail_prob: tail.prob,
                head_query: head.query,
                tail_query: tail.query,
            });
        }
    }
    out
}

/// A selected mention.
///
/// The `valid` flag is load-bearing, not bookkeeping. `select` always returns
/// the setting's full width, padding with invalid slots when the document has
/// fewer qualifying mentions than the cap. The reference then masks
/// `pair_valid = hvalid & tvalid` out of the pair top-k
/// (`relations.py:212-213, 227-228`), so a *real* head paired with a *padded*
/// tail is dropped — even though their spans differ and their product is a
/// harmless 0.0. Inferring validity from the score instead would let those
/// zero-score pairs into the top-k whenever the document runs out of
/// arguments, which is exactly when they would start displacing real pairs.
#[derive(Clone, Copy, Debug)]
struct SelectedMention {
    start: usize,
    end: usize,
    prob: f32,
    query: usize,
    valid: bool,
}

/// `select(...)` from `relations.py:160-201`.
///
/// Order of operations, all of it observable in the output:
/// 1. stable sort by `end`, then stable sort by `start` — equal to one stable
///    sort on `(start, end)`;
/// 2. stable descending sort by probability, keep `min(requested, Q*C)`;
///
///    the reference masks invalid mentions to `FLOOR` before its sort rather
///    than dropping them, which sorts them to the bottom — so filtering first
///    and keeping `min(requested, n_valid)` is the same selection, and the
///    leftover slots stay the padding from step 3 either way;
/// 3. pad up to `requested` with invalid slots, so downstream indexing can use
///    the setting's width unconditionally.
fn select_mentions(
    candidates: &RelationCandidates<'_>,
    probs: &[f32],
    valid: &dyn Fn(usize) -> bool,
    requested: usize,
) -> Vec<SelectedMention> {
    let q_count = candidates.q_count;
    let c_count = candidates.c_count;
    let flat_len = q_count * c_count;
    let mut out = vec![
        SelectedMention {
            start: 0,
            end: 0,
            prob: 0.0,
            query: 0,
            valid: false,
        };
        requested
    ];
    if requested == 0 {
        return out;
    }
    let take = requested.min(flat_len);

    // Step 1: `torch.sort(stable=True)` twice is a lexicographic sort on the
    // keys, and Rust's `sort_by` is stable, so this is a single sort whose
    // stability supplies the final tie-break on the flat slot.
    let mut order: Vec<usize> = (0..flat_len).collect();
    order.sort_by(|&a, &b| {
        let sa = candidates.indices[a * 2];
        let sb = candidates.indices[b * 2];
        let ea = candidates.indices[a * 2 + 1];
        let eb = candidates.indices[b * 2 + 1];
        sa.cmp(&sb).then(ea.cmp(&eb))
    });

    // Step 2: stable descending sort of the *reordered* scores.
    let mut ranked: Vec<usize> = order
        .iter()
        .copied()
        .filter(|&slot| valid(slot))
        .collect();
    ranked.sort_by(|&a, &b| f32_order(probs[b], probs[a]));

    for (position, &slot) in ranked.iter().take(take).enumerate() {
        out[position] = SelectedMention {
            start: candidates.indices[slot * 2],
            end: candidates.indices[slot * 2 + 1],
            prob: probs[slot],
            query: (slot / c_count).min(q_count - 1),
            valid: true,
        };
    }
    out
}

// ---------------------------------------------------------------------------
// `SparseRelationScorer`
// ---------------------------------------------------------------------------

/// `SparseRelationScorer` (`relations.py:265`). Scores a proposed pair from
/// four endpoint boundary states, the relation query, the relative order and a
/// normalized distance — no dense pair matrix.
pub struct SparseRelationScorer<'a> {
    hidden_size: usize,
    relation_query_dim: usize,
    use_biaffine_content: bool,
    mlp_in: Weight<'a>,
    mlp_in_bias: Vec<f32>,
    mlp_out: Weight<'a>,
    mlp_out_bias: Vec<f32>,
    head_content_projection: Option<(Weight<'a>, Vec<f32>)>,
    tail_content_projection: Option<(Weight<'a>, Vec<f32>)>,
    relation_content_gate: Option<(Weight<'a>, Vec<f32>)>,
    content_linear: Option<(Weight<'a>, Vec<f32>)>,
}

impl<'a> SparseRelationScorer<'a> {
    pub fn load(
        source: &'a dyn TensorSource,
        hidden_size: usize,
        relation_query_dim: usize,
        use_biaffine_content: bool,
    ) -> Result<Self, String> {
        // Four endpoint states + relation query + order + normalized distance.
        let in_dim = 4 * hidden_size + relation_query_dim + 2;
        // `nn.Sequential(Linear, GELU, Dropout, Linear)`: Dropout occupies an
        // index, so the output projection is `mlp.3`, not `mlp.2`.
        let mlp_in = load_weight(source, "relation_scorer.mlp.0.weight", in_dim, hidden_size)?;
        let mlp_in_bias = load_vec(source, "relation_scorer.mlp.0.bias", hidden_size)?;
        let mlp_out = load_weight(source, "relation_scorer.mlp.3.weight", hidden_size, 1)?;

        let mut scorer = Self {
            hidden_size,
            relation_query_dim,
            use_biaffine_content,
            mlp_in,
            mlp_in_bias,
            mlp_out,
            mlp_out_bias: load_vec(source, "relation_scorer.mlp.3.bias", 1)?,
            head_content_projection: None,
            tail_content_projection: None,
            relation_content_gate: None,
            content_linear: None,
        };
        if use_biaffine_content {
            scorer.head_content_projection = Some((
                load_weight(
                    source,
                    "relation_scorer.head_content_projection.weight",
                    hidden_size,
                    hidden_size,
                )?,
                load_vec(
                    source,
                    "relation_scorer.head_content_projection.bias",
                    hidden_size,
                )?,
            ));
            scorer.tail_content_projection = Some((
                load_weight(
                    source,
                    "relation_scorer.tail_content_projection.weight",
                    hidden_size,
                    hidden_size,
                )?,
                load_vec(
                    source,
                    "relation_scorer.tail_content_projection.bias",
                    hidden_size,
                )?,
            ));
            scorer.relation_content_gate = Some((
                load_weight(
                    source,
                    "relation_scorer.relation_content_gate.weight",
                    relation_query_dim,
                    hidden_size,
                )?,
                load_vec(
                    source,
                    "relation_scorer.relation_content_gate.bias",
                    hidden_size,
                )?,
            ));
            scorer.content_linear = Some((
                load_weight(
                    source,
                    "relation_scorer.content_linear.weight",
                    2 * hidden_size + relation_query_dim,
                    1,
                )?,
                load_vec(source, "relation_scorer.content_linear.bias", 1)?,
            ));
        }
        Ok(scorer)
    }

    /// Score every pair. `boundary_states` is `[L, H]` and `relation_states` is
    /// `[R, relation_query_dim]`, both for the single document decode handles.
    ///
    /// Returns one logit per entry of `pairs`, in order. The reference masks
    /// invalid pairs to `0.0` and the compact form (`compact=True`) has already
    /// dropped them, so no mask is returned.
    pub fn forward(
        &self,
        boundary_states: &[f32],
        relation_states: &[f32],
        pairs: &[RelationPair],
    ) -> Vec<f32> {
        if pairs.is_empty() {
            return Vec::new();
        }
        let length = boundary_states.len() / self.hidden_size.max(1);
        if length == 0 || relation_states.is_empty() {
            return vec![0.0; pairs.len()];
        }
        let in_dim = 4 * self.hidden_size + self.relation_query_dim + 2;
        let mut logits = Vec::with_capacity(pairs.len());

        // The biaffine branch needs a cumulative sum of the text states, built
        // once for the whole document rather than per pair.
        let prefix = self
            .use_biaffine_content
            .then(|| build_prefix(boundary_states, length, self.hidden_size));

        let mut features = vec![0.0f32; in_dim];
        for pair in pairs {
            let gather = |pos: usize| {
                let pos = pos.min(length - 1);
                &boundary_states[pos * self.hidden_size..][..self.hidden_size]
            };
            let h_start = gather(pair.head_start);
            let h_end = gather(pair.head_end.saturating_sub(1));
            let t_start = gather(pair.tail_start);
            let t_end = gather(pair.tail_end.saturating_sub(1));
            let rel = &relation_states[pair.relation_index * self.relation_query_dim..]
                [..self.relation_query_dim];

            let mut at = 0;
            features[at..at + self.hidden_size].copy_from_slice(h_start);
            at += self.hidden_size;
            features[at..at + self.hidden_size].copy_from_slice(h_end);
            at += self.hidden_size;
            features[at..at + self.hidden_size].copy_from_slice(t_start);
            at += self.hidden_size;
            features[at..at + self.hidden_size].copy_from_slice(t_end);
            at += self.hidden_size;
            features[at..at + self.relation_query_dim].copy_from_slice(rel);
            at += self.relation_query_dim;
            // `delta = tail_start - head_start` in integers, cast before
            // concatenating so the MLP stays dtype-consistent.
            let delta = pair.tail_start as f32 - pair.head_start as f32;
            features[at] = delta.signum();
            at += 1;
            features[at] = delta.abs() / length as f32;

            let mut hidden = vec![0.0f32; self.hidden_size];
            apply_linear_full(
                &features,
                &self.mlp_in,
                &self.mlp_in_bias,
                &mut hidden,
            );
            for value in hidden.iter_mut() {
                *value = gelu(*value);
            }
            let mut score = vec![0.0f32; 1];
            apply_linear_full(&hidden, &self.mlp_out, &self.mlp_out_bias, &mut score);
            let mut total = score[0];

            if let (Some(prefix), Some((head_w, head_b)), Some((tail_w, tail_b))) = (
                prefix.as_ref(),
                self.head_content_projection.as_ref(),
                self.tail_content_projection.as_ref(),
            ) {
                let gate = self.relation_content_gate.as_ref();
                let head_content = project_pooled(
                    prefix,
                    pair.head_start,
                    pair.head_end,
                    length,
                    self.hidden_size,
                    head_w,
                    head_b,
                );
                let tail_content = project_pooled(
                    prefix,
                    pair.tail_start,
                    pair.tail_end,
                    length,
                    self.hidden_size,
                    tail_w,
                    tail_b,
                );
                let mut gate_values = vec![0.0f32; self.hidden_size];
                if let Some((gate_w, gate_b)) = gate {
                    apply_linear_full(rel, gate_w, gate_b, &mut gate_values);
                    for value in gate_values.iter_mut() {
                        *value = sigmoid(*value);
                    }
                }
                // `head_content * gate * tail_content`, summed and divided by
                // sqrt(hidden_size).
                let mut dot = 0.0f32;
                for i in 0..self.hidden_size {
                    dot += head_content[i] * gate_values[i] * tail_content[i];
                }
                let scale = 1.0 / (self.hidden_size as f32).sqrt();
                total += dot * scale;

                if let Some((linear_w, linear_b)) = self.content_linear.as_ref() {
                    let mut wide = Vec::with_capacity(2 * self.hidden_size + self.relation_query_dim);
                    wide.extend_from_slice(&head_content);
                    wide.extend_from_slice(&tail_content);
                    wide.extend_from_slice(rel);
                    let mut value = vec![0.0f32; 1];
                    apply_linear_full(&wide, linear_w, linear_b, &mut value);
                    total += value[0];
                }
            }
            logits.push(total);
        }
        logits
    }
}

/// `prefix[end] - prefix[start]` gives the span sum, so the pooled vector is the
/// mean over the span (`relations.py:317-325`). `build_prefix` is the same
/// convention as `SpanContentPooler::build_prefix`, on the text states rather
/// than the content projections.
fn build_prefix(states: &[f32], length: usize, hidden: usize) -> Vec<f32> {
    let mut prefix = vec![0.0f32; (length + 1) * hidden];
    for pos in 0..length {
        for dim in 0..hidden {
            prefix[(pos + 1) * hidden + dim] =
                prefix[pos * hidden + dim] + states[pos * hidden + dim];
        }
    }
    prefix
}

#[allow(clippy::too_many_arguments)]
fn project_pooled(
    prefix: &[f32],
    start: usize,
    end: usize,
    length: usize,
    hidden: usize,
    weight: &Weight<'_>,
    bias: &[f32],
) -> Vec<f32> {
    let start = start.min(length);
    let end = end.min(length);
    let width = end.saturating_sub(start).max(1) as f32;
    let mut pooled = vec![0.0f32; hidden];
    for dim in 0..hidden {
        pooled[dim] = (prefix[end * hidden + dim] - prefix[start * hidden + dim]) / width;
    }
    let mut out = vec![0.0f32; hidden];
    apply_linear_full(&pooled, weight, bias, &mut out);
    out
}

/// `nn.GELU()` is the exact erf-based form, not the tanh approximation.
fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + erf(x * std::f32::consts::FRAC_1_SQRT_2))
}

/// Abramowitz & Stegun 7.1.26. `f32::erf` is not in std, and the reference's
/// `nn.GELU()` default is exact, so a tanh approximation is not equivalent.
fn erf(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-x * x).exp();
    sign * y
}

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn f32_order(a: f32, b: f32) -> std::cmp::Ordering {
    a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal)
}

// ---------------------------------------------------------------------------
// Tensor helpers
// ---------------------------------------------------------------------------

fn load_vec(source: &dyn TensorSource, name: &str, len: usize) -> Result<Vec<f32>, String> {
    crate::core::tensor::load_f32_tensor(source, name, &[len as u64])
        .map_err(|e| format!("{name}: {e}"))
}

fn load_weight<'a>(
    source: &'a dyn TensorSource,
    name: &str,
    n_in: usize,
    n_out: usize,
) -> Result<Weight<'a>, String> {
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("missing tensor {name}"))?;
    if info.dims != [n_in as u64, n_out as u64] {
        return Err(format!(
            "tensor {name} has dims {:?}, expected [{n_in}, {n_out}]",
            info.dims
        ));
    }
    let bytes = source
        .tensor_slice(name)
        .ok_or_else(|| format!("missing tensor data {name}"))?;
    Ok(Weight::from_quantized(QuantizedTensor::from_bytes(
        bytes,
        info.ggml_type,
        n_in,
        n_out,
    )))
}

fn apply_linear_full(input: &[f32], weight: &Weight<'_>, bias: &[f32], output: &mut [f32]) {
    if let Some(rows) = weight.kernel.f32_slice() {
        let n_in = input.len();
        let n_out = output.len();
        debug_assert_eq!(bias.len(), n_out);
        for (out_index, row) in rows.chunks_exact(n_in).take(n_out).enumerate() {
            output[out_index] = crate::ops::dot_f32(row, input, n_in) + bias[out_index];
        }
    } else {
        weight
            .kernel
            .forward(input, output, weight.n_in, weight.n_out);
        for (out, b) in output.iter_mut().zip(bias.iter()) {
            *out += *b;
        }
    }
}
