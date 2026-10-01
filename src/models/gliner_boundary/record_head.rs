//! `RecordHead` — instance formation and null-aware field assignment.
//!
//! Mirrors `gliner2.models.boundary.records.RecordHead`
//! (`target/gliner2-oracle/gliner2/models/boundary/records.py:246`) and its
//! `decode_group` (`records.py:714`).
//!
//! Records reuse the entity mention candidates rather than proposing their own
//! spans, so the only tensor this needs from the pool is
//! `DocumentCandidateBatch::candidate_states` — the `hidden_size`-wide
//! `candidate_encoder` output, not the scorer's internal `pair_dim` features.
//!
//! All three modes reduce to the same two things, which is the point of the
//! design: **(instance states, object logits)** plus a per-field assignment
//! matrix. Only the first two differ:
//!
//! * `natural` — the instances *are* the anchor field's candidates, and their
//!   object logits are the anchor candidate's `pair_logits` straight from the
//!   pool. `object_head` is not applied, which is easy to miss.
//! * `latent` — every field's candidates are pooled and scored by
//!   `latent_seed_head` to decide which ones seed an instance.
//! * `anchorless` — the `instance_embed` table is the instance state, refined by
//!   one attention pass over all candidates (`_anchorless_states`), and scored by
//!   `object_head`. Nothing anchors to a span, which is also why the decoder
//!   switches to `object_threshold` instead of `anchor_threshold` here.
//!
//! The decoder's interesting half is the exclusive-field assignment, which is a
//! global optimisation rather than a per-instance argmax; see [`decode_group`].

use std::collections::BTreeMap;

use crate::core::tensor::TensorSource;
use crate::ops::kernel::{QuantizedTensor, Weight};

use super::matching::linear_sum_assignment;
use super::record_spec::RecordSpec;

/// `torch.finfo(torch.float32).eps`, the floor the reference clamps softmax
/// outputs to before taking `-log`. Without it a zero probability gives `inf`
/// and poisons the assignment cost for every row.
const PROB_EPS: f32 = f32::EPSILON;

/// The three decode thresholds, all read from the checkpoint's transcribed
/// settings (`record_anchor_threshold`, `record_field_threshold`,
/// `record_anchor_proposal_threshold`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RecordDecodeSettings {
    /// `anchor_threshold` — gates instances in `natural`/`latent` mode.
    pub anchor_threshold: f32,
    /// `object_threshold` — gates instances in `anchorless` mode, which have no
    /// anchor to score.
    pub object_threshold: f32,
    pub field_threshold: f32,
    pub temperature: f32,
}

/// `RecordHead` (`records.py:254`).
pub struct RecordHead<'a> {
    hidden_size: usize,
    record_dim: usize,
    instance_queries: usize,
    inst_proj: Weight<'a>,
    inst_proj_bias: Vec<f32>,
    field_proj: Weight<'a>,
    field_proj_bias: Vec<f32>,
    cand_proj: Weight<'a>,
    cand_proj_bias: Vec<f32>,
    null_embed: Vec<f32>,
    object_head: Weight<'a>,
    object_head_bias: Vec<f32>,
    latent_seed_head: Weight<'a>,
    latent_seed_head_bias: Vec<f32>,
    instance_embed: Vec<f32>,
    q_proj: Weight<'a>,
    q_proj_bias: Vec<f32>,
    k_proj: Weight<'a>,
    k_proj_bias: Vec<f32>,
    v_proj: Weight<'a>,
    v_proj_bias: Vec<f32>,
}

/// The per-field candidate views `forward_group` gathers. Owns everything, so
/// the [`RecordGroup`] it lands in does not borrow the source.
pub struct FieldCandidates {
    /// `candidates.indices[query]`, `[C, 2]`, valid entries only.
    pub spans: Vec<Vec<usize>>,
    /// `candidates.pair_logits[query]`, valid entries only.
    pub pair_logits: Vec<f32>,
    /// `candidates.candidate_states[query]`, valid entries only,
    /// `[C, hidden_size]`.
    pub states: Vec<f32>,
}

/// The candidate batch `forward_group` reads, in the reference's
/// `CandidateTensorBatch` shape. Grouped so the call site is not nine positional
/// tensors that have to be supplied in exactly the right order.
pub struct RecordCandidates<'a> {
    /// `[Q, C, 2]`, flattened.
    pub indices: &'a [usize],
    /// `[Q, C]`.
    pub pair_logits: &'a [f32],
    /// `[Q, C]`. Already ANDed with `query_mask`, so an inactive query's
    /// candidates arrive here as invalid.
    pub valid_mask: &'a [bool],
    /// `[C, hidden_size]` — the `candidate_encoder` output, not the scorer's
    /// internal `pair_dim` features.
    ///
    /// **Candidate-major, unlike the three fields above**, and deliberately so.
    /// The reference's `to_candidate_batch` reaches `[B, Q, C, H]` with an
    /// `expand`, a broadcast view whose values do not depend on `q`, and
    /// `DocumentCandidateBatch` keeps the narrower `[B, C, H]`. Indexing this one
    /// query-major reads past the end of the buffer, or silently reads another
    /// query's candidates — so the asymmetry is spelled out here rather than left
    /// to be inferred from the three neighbours.
    pub states: &'a [f32],
    pub q_count: usize,
    pub c_count: usize,
}

/// `[field][instance][1 + candidates]`: column 0 is the null/ABSENT column.
pub type AssignLogits = Vec<Vec<Vec<f32>>>;

/// A record's dedup key: field query id -> its spans, sorted so a reordering
/// cannot disguise a duplicate. Mirrors `_dedup_key`.
type DedupKey = Vec<(usize, Vec<(usize, usize)>)>;

/// The `RecordGroupOutput` that [`decode_group`] consumes.
pub struct RecordGroup {
    pub spec: RecordSpec,
    pub object_logits: Vec<f32>,
    pub assign_logits: AssignLogits,
    pub field_query_ids: Vec<usize>,
    pub field_candidates: Vec<FieldCandidates>,
    /// `(field index, candidate index)` per instance, or `None` for anchorless
    /// instances, which seed from no span.
    pub instance_seed: Vec<Option<(usize, usize)>>,
    pub instance_spans: Vec<Option<(usize, usize)>>,
}

impl<'a> RecordHead<'a> {
    pub fn load(
        source: &'a dyn TensorSource,
        hidden_size: usize,
        record_dim: usize,
        instance_queries: usize,
    ) -> Result<Self, String> {
        const P: &str = "record_decoder";
        let linear = |name: &str, n_in: usize, n_out: usize| -> Result<Weight<'a>, String> {
            load_weight(source, &format!("{P}.{name}.weight"), n_in, n_out)
        };
        let bias = |name: &str, len: usize| -> Result<Vec<f32>, String> {
            load_vec(source, &format!("{P}.{name}.bias"), len)
        };
        Ok(Self {
            hidden_size,
            record_dim,
            instance_queries,
            inst_proj: linear("inst_proj", hidden_size, record_dim)?,
            inst_proj_bias: bias("inst_proj", record_dim)?,
            field_proj: linear("field_proj", hidden_size, record_dim)?,
            field_proj_bias: bias("field_proj", record_dim)?,
            cand_proj: linear("cand_proj", hidden_size, record_dim)?,
            cand_proj_bias: bias("cand_proj", record_dim)?,
            null_embed: load_vec(source, &format!("{P}.null_embed"), record_dim)?,
            object_head: linear("object_head", hidden_size, 1)?,
            object_head_bias: bias("object_head", 1)?,
            latent_seed_head: linear("latent_seed_head", hidden_size, 1)?,
            latent_seed_head_bias: bias("latent_seed_head", 1)?,
            instance_embed: load_instance_embed(source, P, hidden_size, instance_queries)?,
            q_proj: linear("q_proj", hidden_size, record_dim)?,
            q_proj_bias: bias("q_proj", record_dim)?,
            k_proj: linear("k_proj", hidden_size, record_dim)?,
            k_proj_bias: bias("k_proj", record_dim)?,
            v_proj: linear("v_proj", hidden_size, hidden_size)?,
            v_proj_bias: bias("v_proj", hidden_size)?,
        })
    }

    /// `RecordHead.forward_group` (`records.py:572`).
    ///
    /// `candidate_states` is `[Q, C, hidden_size]`, `indices` `[Q, C, 2]`,
    /// `pair_logits` and `valid_mask` `[Q, C]`, `query_states` `[Q, hidden_size]`.
    pub fn forward_group(
        &self,
        spec: &RecordSpec,
        query_states: &[f32],
        candidates: &RecordCandidates<'_>,
    ) -> Result<RecordGroup, String> {
        let RecordCandidates {
            indices,
            pair_logits,
            valid_mask,
            states: candidate_states,
            q_count: query_count,
            c_count,
        } = *candidates;
        if query_count == 0 || c_count == 0 {
            return Err("record routing requires at least one boundary query".into());
        }
        // `query_count` is the min over every tensor the reference takes it from,
        // so a short tensor must not read past its end. Clamping the ids is what
        // the reference does; an id the layout invented lands on query 0.
        let field_query_ids: Vec<usize> = spec.fields.iter().map(|f| f.query_id).collect();
        let mut field_candidates = Vec::with_capacity(field_query_ids.len());
        for &qid in &field_query_ids {
            let safe = qid.min(query_count - 1);
            let mut entry = FieldCandidates {
                spans: Vec::new(),
                pair_logits: Vec::new(),
                states: Vec::new(),
            };
            for slot in 0..c_count {
                let flat = safe * c_count + slot;
                if !valid_mask.get(flat).copied().unwrap_or(false) {
                    continue;
                }
                entry
                    .spans
                    .push(vec![indices[flat * 2], indices[flat * 2 + 1]]);
                entry.pair_logits.push(pair_logits[flat]);
                let state_base = slot * self.hidden_size;
                entry
                    .states
                    .extend_from_slice(&candidate_states[state_base..][..self.hidden_size]);
            }
            field_candidates.push(entry);
        }

        // `(instance states, object logits, seeds, spans)` by mode.
        let mut inst_states: Vec<f32> = Vec::new();
        let mut object_logits: Vec<f32> = Vec::new();
        let mut instance_seed: Vec<Option<(usize, usize)>> = Vec::new();
        let mut instance_spans: Vec<Option<(usize, usize)>> = Vec::new();

        match spec.mode.as_str() {
            "natural" => {
                let anchor_query_id = spec.anchor_query_id.ok_or_else(|| {
                    format!("record {:?} is natural but has no anchor", spec.task_name)
                })?;
                let anchor_field = spec
                    .fields
                    .iter()
                    .position(|f| f.query_id == anchor_query_id)
                    .ok_or_else(|| {
                        format!(
                            "record {:?} anchor_query_id {anchor_query_id} matches no field",
                            spec.task_name
                        )
                    })?;
                let anchor = &field_candidates[anchor_field];
                // The instances *are* the anchor candidates, and their object
                // logits are the pool's pair_logits straight through — not
                // `object_head`. That is the whole difference from latent mode.
                inst_states = anchor.states.clone();
                object_logits = anchor.pair_logits.clone();
                for slot in 0..anchor.spans.len() {
                    instance_seed.push(Some((anchor_field, slot)));
                    instance_spans.push(Some((anchor.spans[slot][0], anchor.spans[slot][1])));
                }
            }
            "latent" => {
                for (field_index, candidates) in field_candidates.iter().enumerate() {
                    for slot in 0..candidates.pair_logits.len() {
                        let base = slot * self.hidden_size;
                        let mut score = vec![0.0f32; 1];
                        apply_linear_full(
                            &candidates.states[base..][..self.hidden_size],
                            &self.latent_seed_head,
                            &self.latent_seed_head_bias,
                            &mut score,
                        );
                        inst_states
                            .extend_from_slice(&candidates.states[base..][..self.hidden_size]);
                        object_logits.push(score[0]);
                        instance_seed.push(Some((field_index, slot)));
                        instance_spans
                            .push(Some((candidates.spans[slot][0], candidates.spans[slot][1])));
                    }
                }
            }
            "anchorless" => {
                inst_states = self.anchorless_states(&field_candidates);
                for row in 0..self.instance_queries {
                    let mut score = vec![0.0f32; 1];
                    apply_linear_full(
                        &inst_states[row * self.hidden_size..][..self.hidden_size],
                        &self.object_head,
                        &self.object_head_bias,
                        &mut score,
                    );
                    object_logits.push(score[0]);
                    instance_seed.push(None);
                    instance_spans.push(None);
                }
            }
            other => return Err(format!("unknown record mode {other:?}")),
        }

        // `_assign_logits`: one row per instance, one column per candidate, plus
        // a leading null column. The instance and field projections are *added*
        // before the dot products, so a field's preference is shared across
        // instances and an instance's preference across fields.
        let instances = object_logits.len();
        let mut inst_q = vec![0.0f32; instances * self.record_dim];
        if instances > 0 {
            apply_linear_rows(
                &inst_states,
                &self.inst_proj,
                &self.inst_proj_bias,
                &mut inst_q,
                self.record_dim,
            );
        }
        let mut field_q = vec![0.0f32; field_query_ids.len() * self.record_dim];
        for (index, &qid) in field_query_ids.iter().enumerate() {
            let safe = qid.min(query_count - 1);
            apply_linear_full(
                &query_states[safe * self.hidden_size..][..self.hidden_size],
                &self.field_proj,
                &self.field_proj_bias,
                &mut field_q[index * self.record_dim..][..self.record_dim],
            );
        }
        let mut assign_logits: AssignLogits = Vec::with_capacity(field_query_ids.len());
        for (index, candidates) in field_candidates.iter().enumerate() {
            let mut rows = Vec::with_capacity(instances);
            for row in 0..instances {
                let base = row * self.record_dim;
                let mut query = vec![0.0f32; self.record_dim];
                for dim in 0..self.record_dim {
                    query[dim] = inst_q[base + dim] + field_q[index * self.record_dim + dim];
                }
                let null: f32 = (0..self.record_dim)
                    .map(|dim| query[dim] * self.null_embed[dim])
                    .sum();
                let mut line = Vec::with_capacity(1 + candidates.pair_logits.len());
                line.push(null);
                for slot in 0..candidates.pair_logits.len() {
                    let state_base = slot * self.hidden_size;
                    let mut projected = vec![0.0f32; self.record_dim];
                    apply_linear_full(
                        &candidates.states[state_base..][..self.hidden_size],
                        &self.cand_proj,
                        &self.cand_proj_bias,
                        &mut projected,
                    );
                    line.push(
                        (0..self.record_dim)
                            .map(|dim| query[dim] * projected[dim])
                            .sum(),
                    );
                }
                rows.push(line);
            }
            assign_logits.push(rows);
        }

        Ok(RecordGroup {
            spec: spec.clone(),
            object_logits,
            assign_logits,
            field_query_ids,
            field_candidates,
            instance_seed,
            instance_spans,
        })
    }

    /// `_anchorless_states` (`records.py:294`).
    ///
    /// `instance_embed` is the instance state, then one attention pass over the
    /// concatenation of every field's candidates, added back. The `record_dim`
    /// sqrt is on the attention logits only, and `v_proj` stays at `hidden_size`,
    /// so the residual is in the encoder's width.
    fn anchorless_states(&self, fields: &[FieldCandidates]) -> Vec<f32> {
        let instances = self.instance_queries;
        let mut inst = Vec::with_capacity(instances * self.hidden_size);
        for row in 0..instances {
            inst.extend_from_slice(
                &self.instance_embed[row * self.hidden_size..][..self.hidden_size],
            );
        }
        let context: usize = fields.iter().map(|f| f.pair_logits.len()).sum();
        if context == 0 {
            return inst;
        }
        let mut q = vec![0.0f32; instances * self.record_dim];
        apply_linear_rows(
            &inst,
            &self.q_proj,
            &self.q_proj_bias,
            &mut q,
            self.record_dim,
        );
        let mut flat_context = Vec::with_capacity(context * self.hidden_size);
        for candidates in fields {
            flat_context.extend_from_slice(&candidates.states);
        }
        let mut k = vec![0.0f32; context * self.record_dim];
        apply_linear_rows(
            &flat_context,
            &self.k_proj,
            &self.k_proj_bias,
            &mut k,
            self.record_dim,
        );
        let mut v = vec![0.0f32; context * self.hidden_size];
        apply_linear_rows(
            &flat_context,
            &self.v_proj,
            &self.v_proj_bias,
            &mut v,
            self.hidden_size,
        );
        let scale = 1.0 / (self.record_dim as f32).sqrt();
        let mut out = inst.clone();
        for row in 0..instances {
            let base = row * self.record_dim;
            let mut attn = Vec::with_capacity(context);
            for column in 0..context {
                attn.push(
                    (0..self.record_dim)
                        .map(|dim| q[base + dim] * k[column * self.record_dim + dim])
                        .sum::<f32>()
                        * scale,
                );
            }
            softmax_into(&attn.clone(), &mut attn);
            for (column, &weight) in attn.iter().enumerate() {
                if weight == 0.0 {
                    continue;
                }
                for dim in 0..self.hidden_size {
                    out[row * self.hidden_size + dim] +=
                        weight * v[column * self.hidden_size + dim];
                }
            }
        }
        out
    }
}

/// One decoded record: field query id -> selected half-open word spans.
#[derive(Clone, Debug, PartialEq)]
pub struct DecodedRecord {
    pub fields: BTreeMap<usize, Vec<(usize, usize)>>,
    pub field_scores: BTreeMap<usize, Vec<f32>>,
    /// Set for `natural` mode: the span the instance was seeded from.
    pub anchor_span: Option<(usize, usize)>,
    pub score: f32,
}

/// `decode_group` (`records.py:714`).
///
/// The interesting part is the exclusive-field assignment, which is a **global**
/// optimisation rather than a per-instance argmax. Greedily letting the
/// highest-object instance claim its favourite candidate can force later instances
/// onto unrelated spans, so exclusive scalar fields are solved jointly with
/// [`linear_sum_assignment`] and exclusive list candidates are each awarded to
/// their single strongest instance.
pub fn decode_group(
    group: &RecordGroup,
    settings: RecordDecodeSettings,
) -> Result<Vec<DecodedRecord>, String> {
    if settings.temperature <= 0.0 {
        return Err("temperature must be > 0".into());
    }
    let instance_count = group.object_logits.len();
    if instance_count == 0 {
        return Ok(Vec::new());
    }
    let obj_prob: Vec<f32> = group
        .object_logits
        .iter()
        .map(|logit| sigmoid(logit / settings.temperature))
        .collect();
    // `anchorless` instances have no anchor span, so `object_threshold` gates
    // them instead.
    let select_threshold = if group.spec.mode == "anchorless" {
        settings.object_threshold
    } else {
        settings.anchor_threshold
    };
    let mut order: Vec<usize> = (0..instance_count).collect();
    order.sort_by(|&a, &b| f32_order(obj_prob[b], obj_prob[a]).then(a.cmp(&b)));
    let selected: Vec<usize> = order
        .into_iter()
        .filter(|&index| obj_prob[index] >= select_threshold)
        .collect();

    // `scalar_choices[(instance, field)]` and `list_owners[(field, candidate)]`
    // are only populated for exclusive fields.
    let mut scalar_choices: BTreeMap<(usize, usize), Option<(usize, f32)>> = BTreeMap::new();
    let mut list_owners: BTreeMap<(usize, usize), (usize, f32)> = BTreeMap::new();

    for (field_index, field) in group.spec.fields.iter().enumerate() {
        if !field.exclusive || selected.is_empty() {
            continue;
        }
        // `[selected, 1 + candidates]`, temperature-scaled.
        let rows = selected.len();
        let width = group.assign_logits[field_index]
            .first()
            .map(Vec::len)
            .unwrap_or(0);
        if width == 0 {
            continue;
        }
        let mut logits = vec![0.0f32; rows * width];
        for (row, &instance) in selected.iter().enumerate() {
            let source = &group.assign_logits[field_index][instance];
            for column in 0..width {
                logits[row * width + column] = source[column] / settings.temperature;
            }
        }
        let candidate_count = width - 1;

        if field.cardinality.is_scalar() {
            if candidate_count == 0 {
                for &instance in &selected {
                    scalar_choices.insert((instance, field_index), None);
                }
                continue;
            }
            // Softmax per row, then cost = -log(prob).
            // Softmax per row, into a separate buffer: the temperature-scaled
            // logits stay intact for the list path's sigmoid.
            let mut probs = vec![0.0f32; rows * width];
            for row in 0..rows {
                let source = logits[row * width..(row + 1) * width].to_vec();
                softmax_into(&source, &mut probs[row * width..(row + 1) * width]);
            }
            let mut candidate_cost = vec![0.0f32; rows * candidate_count];
            let mut max_candidate_cost = f32::NEG_INFINITY;
            for row in 0..rows {
                for column in 0..candidate_count {
                    let cost = -probs[row * width + column + 1].max(PROB_EPS).ln();
                    candidate_cost[row * candidate_count + column] = cost;
                    max_candidate_cost = max_candidate_cost.max(cost);
                }
            }
            // Column 0 is the null/ABSENT option. When the field may not be
            // absent it gets one emergency cost for *every* row — a scalar
            // broadcast, not a per-row max. Making it per-row would let a row
            // with cheap candidates escape the assignment that a row with
            // expensive ones has to take.
            let mut diagonal = vec![0.0f32; rows];
            if field.cardinality.allows_absent() {
                for row in 0..rows {
                    diagonal[row] = -probs[row * width].max(PROB_EPS).ln();
                }
            } else {
                let emergency = max_candidate_cost + 50.0;
                diagonal.fill(emergency);
            }
            let mut max_diagonal = f32::NEG_INFINITY;
            for &value in &diagonal {
                max_diagonal = max_diagonal.max(value);
            }
            let invalid_cost = max_candidate_cost.max(max_diagonal) + 1_000.0;
            // `row_count` extra columns so every row has a slot to fall into;
            // only the diagonal one is cheap enough to win. The diagonal lives at
            // column `candidate_count + row` — inside the *absent* block. Writing it
            // at plain `row` lands it in the real-candidate block, where it
            // silently overwrites a real cost and leaves the row's own absent slot
            // at `invalid_cost`. Nothing errors; the matrix is simply a different
            // problem, and the solver returns a valid matching of it.
            // `matrix_width` is 1 more than `width` times `rows`: the `rows`
            // absent columns appended to the `candidate_count` real ones. Named
            // distinctly from `width` (the softmax width) because shadowing it here
            // silently reinterprets every later `probs[...]` index.
            let matrix_width = candidate_count + rows;
            let mut cost = vec![0.0f32; rows * matrix_width];
            for row in 0..rows {
                for column in 0..candidate_count {
                    cost[row * matrix_width + column] =
                        candidate_cost[row * candidate_count + column];
                }
                for column in candidate_count..matrix_width {
                    cost[row * matrix_width + column] = invalid_cost;
                }
                cost[row * matrix_width + candidate_count + row] = diagonal[row];
            }
            let matrix: Vec<Vec<f64>> = (0..rows)
                .map(|row| {
                    (0..matrix_width)
                        .map(|column| cost[row * matrix_width + column] as f64)
                        .collect()
                })
                .collect();
            let (assigned_rows, assigned_cols) = linear_sum_assignment(&matrix)
                .map_err(|error| format!("record assignment failed: {error}"))?;
            // The solver returns (row, col) *pairs*: `assigned_rows[i]` is paired
            // with `assigned_cols[i]`, and neither array is indexed by the other.
            // A matrix row is a position in `selected`, not an instance id, since
            // `selected` is ordered by probability while the instances are not.
            let mut unassigned: Vec<usize> = (0..rows).collect();
            for pair in 0..assigned_rows.len() {
                let matrix_row = assigned_rows[pair];
                let column = assigned_cols[pair];
                unassigned.retain(|&row| row != matrix_row);
                let instance = selected[matrix_row];
                if column >= candidate_count {
                    // The row fell into an absent column.
                    scalar_choices.insert((instance, field_index), None);
                    continue;
                }
                let probability = probs[matrix_row * width + column + 1];
                if probability < settings.field_threshold && field.cardinality.allows_absent() {
                    scalar_choices.insert((instance, field_index), None);
                    continue;
                }
                scalar_choices.insert((instance, field_index), Some((column, probability)));
            }
            // A row the solver did not mention (it always returns exactly
            // `min(rows, cols)` pairs, so this is defensive) is treated as absent
            // rather than left to default into a bogus candidate.
            for row in unassigned {
                scalar_choices.insert((selected[row], field_index), None);
            }
        } else if candidate_count > 0 {
            // List fields do not go through the solver: each candidate is awarded
            // to its single strongest instance.
            for candidate in 0..candidate_count {
                let mut best = (f32::NEG_INFINITY, 0usize);
                for row in 0..rows {
                    let probability = sigmoid(logits[row * width + candidate + 1]);
                    if probability > best.0 {
                        best = (probability, row);
                    }
                }
                if best.0 >= settings.field_threshold {
                    list_owners.insert((field_index, candidate), (selected[best.1], best.0));
                }
            }
        }
    }

    let anchor_field_index = if group.spec.mode == "natural" {
        group
            .spec
            .anchor_query_id
            .and_then(|anchor| group.field_query_ids.iter().position(|&q| q == anchor))
    } else {
        None
    };

    let mut records: Vec<DecodedRecord> = Vec::new();
    for &instance in &selected {
        let mut record = DecodedRecord {
            fields: BTreeMap::new(),
            field_scores: BTreeMap::new(),
            anchor_span: None,
            score: obj_prob[instance],
        };
        if group.spec.mode == "natural" {
            record.anchor_span = group.instance_spans[instance];
        }

        for (field_index, field) in group.spec.fields.iter().enumerate() {
            let query_id = field.query_id;
            let spans = &group.field_candidates[field_index].spans;
            let logits_row: Vec<f32> = group.assign_logits[field_index][instance]
                .iter()
                .map(|logit| logit / settings.temperature)
                .collect();

            // The anchor field takes the instance's own seed span, at the
            // instance's score, with no threshold applied.
            if anchor_field_index == Some(field_index) {
                if let Some(anchor) = record.anchor_span {
                    record.fields.entry(query_id).or_default().push(anchor);
                    record
                        .field_scores
                        .entry(query_id)
                        .or_default()
                        .push(record.score);
                }
                continue;
            }

            if field.cardinality.is_scalar() {
                if field.exclusive {
                    if let Some(Some((candidate, probability))) =
                        scalar_choices.get(&(instance, field_index))
                    {
                        record
                            .fields
                            .entry(query_id)
                            .or_default()
                            .push((spans[*candidate][0], spans[*candidate][1]));
                        record
                            .field_scores
                            .entry(query_id)
                            .or_default()
                            .push(*probability);
                    }
                    continue;
                }
                // Non-exclusive scalar: softmax over the candidates plus the
                // null column, then the first acceptable choice in score order.
                let mut probs = vec![0.0f32; logits_row.len()];
                softmax_into(&logits_row, &mut probs);
                let mut ranking: Vec<usize> = (0..probs.len()).collect();
                ranking.sort_by(|&a, &b| f32_order(probs[b], probs[a]).then(a.cmp(&b)));
                let mut chosen = None;
                for column in ranking {
                    if column == 0 {
                        if field.cardinality.allows_absent() {
                            chosen = Some(0);
                            break;
                        }
                        // A field that may not be absent *skips* the null column
                        // and takes the next-best candidate. Breaking here instead
                        // leaves the field empty, which is exactly the ABSENT
                        // outcome the cardinality forbids — so a `required_one`
                        // field silently lost its value while still looking like
                        // a decode that simply found nothing.
                        continue;
                    }
                    chosen = Some(column);
                    break;
                }
                let Some(column) = chosen.filter(|&column| column != 0) else {
                    continue;
                };
                if probs[column] < settings.field_threshold && field.cardinality.allows_absent() {
                    continue;
                }
                let candidate = column - 1;
                record
                    .fields
                    .entry(query_id)
                    .or_default()
                    .push((spans[candidate][0], spans[candidate][1]));
                record
                    .field_scores
                    .entry(query_id)
                    .or_default()
                    .push(probs[column]);
            } else {
                // List field. Column 0 is the null column and is never a
                // candidate here.
                for candidate in 1..logits_row.len() {
                    let probability = if field.exclusive {
                        match list_owners.get(&(field_index, candidate - 1)) {
                            Some(&(owner, probability)) if owner == instance => probability,
                            _ => continue,
                        }
                    } else {
                        let probability = sigmoid(logits_row[candidate]);
                        if probability < settings.field_threshold {
                            continue;
                        }
                        probability
                    };
                    record
                        .fields
                        .entry(query_id)
                        .or_default()
                        .push((spans[candidate - 1][0], spans[candidate - 1][1]));
                    record
                        .field_scores
                        .entry(query_id)
                        .or_default()
                        .push(probability);
                }
            }
        }
        if !record.fields.is_empty() {
            records.push(record);
        }
    }

    match group.spec.mode.as_str() {
        // `latent` and `anchorless` produce an unordered set, so identical field
        // sets collapse to the highest-scoring instance. The surviving order is
        // the order of *first appearance*, not a sort: the reference keeps its
        // dict's insertion order, and a replacement keeps the slot the key
        // already occupied. Sorting by key here would reorder the records.
        "latent" | "anchorless" => {
            let mut keys: Vec<DedupKey> = Vec::new();
            let mut best: Vec<DecodedRecord> = Vec::new();
            for record in records {
                let key = dedup_key(&record);
                match keys.iter().position(|seen| *seen == key) {
                    Some(index) => {
                        if record.score > best[index].score {
                            best[index] = record;
                        }
                    }
                    None => {
                        keys.push(key);
                        best.push(record);
                    }
                }
            }
            Ok(best)
        }
        // `natural` has a meaningful order: the anchor span in the document.
        "natural" => {
            records.sort_by(|a, b| {
                a.anchor_span
                    .is_none()
                    .cmp(&b.anchor_span.is_none())
                    .then(a.anchor_span.cmp(&b.anchor_span))
            });
            Ok(records)
        }
        other => Err(format!("unknown record mode {other:?}")),
    }
}

/// `_dedup_key` (`records.py:707`): the field set, with spans sorted inside each
/// field so a reordering cannot disguise a duplicate.
fn dedup_key(record: &DecodedRecord) -> DedupKey {
    record
        .fields
        .iter()
        .map(|(query_id, spans)| {
            let mut sorted = spans.clone();
            sorted.sort_unstable();
            (*query_id, sorted)
        })
        .collect()
}

/// `torch.softmax(x, -1)` into a separate destination.
///
/// The destination is separate because the decoder needs both at once: the
/// temperature-scaled logits stay intact for the list path's sigmoid while the
/// scalar path consumes the normalized copy.
fn softmax_into(source: &[f32], destination: &mut [f32]) {
    let max = source.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut total = 0.0f32;
    for (index, value) in source.iter().enumerate() {
        let exponent = (value - max).exp();
        destination[index] = exponent;
        total += exponent;
    }
    for value in destination.iter_mut() {
        *value /= total;
    }
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

/// `record_decoder.instance_embed` is a bare `[instance_queries, hidden_size]`
/// parameter, so the converter relabelled its shape to GGUF's `(in, out)`
/// convention and the metadata now says `[hidden_size, instance_queries]`.
///
/// But it relabelled only the *shape*: the bytes are still the checkpoint's
/// `[instance_queries, hidden_size]` row-major, not a transpose of it. Reading
/// them per the declared dims gives a table whose 32 rows are a permutation of
/// the real ones, and since the table is `randn * 0.02` every row is nearly
/// identical, so the resulting logits differ by only ~0.01 — small enough to hide
/// behind a loose tolerance, and with no structure to hint that a permutation is
/// what happened. `anchorless_without_candidates_isolates_instance_embed` exists
/// to catch exactly that: it leaves the attention path out, so the parameter is
/// the only input.
fn load_instance_embed(
    source: &dyn TensorSource,
    prefix: &str,
    hidden_size: usize,
    instance_queries: usize,
) -> Result<Vec<f32>, String> {
    let name = format!("{prefix}.instance_embed");
    let info = source
        .tensor_info(&name)
        .ok_or_else(|| format!("missing tensor {name}"))?;
    // The declared dims are the transposed shape, so this checks the metadata
    // rather than the layout: it is what catches a converter that starts
    // transposing the bytes, which would then be silently wrong here.
    if info.dims != [hidden_size as u64, instance_queries as u64] {
        return Err(format!(
            "tensor {name} has dims {:?}, expected [{hidden_size}, {instance_queries}]",
            info.dims
        ));
    }
    crate::core::tensor::load_f32_tensor(
        source,
        &name,
        &[hidden_size as u64, instance_queries as u64],
    )
    .map_err(|e| format!("{name}: {e}"))
}

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

fn apply_linear_rows(
    input: &[f32],
    weight: &Weight<'_>,
    bias: &[f32],
    output: &mut [f32],
    n_out: usize,
) {
    let n_in = weight.n_in.max(1);
    debug_assert!(output.len() >= input.len() / n_in * n_out);
    for row in 0..input.len() / n_in {
        apply_linear_full(
            &input[row * n_in..][..n_in],
            weight,
            bias,
            &mut output[row * n_out..][..n_out],
        );
    }
}
