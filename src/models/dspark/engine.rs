use super::DSparkSession;
use crate::models::qwen3::trunk::util::greedy_token;
use crate::models::qwen3::trunk::Qwen3Session;

#[derive(Clone, Debug, PartialEq)]
pub struct TargetBatch {
    pub logits: Vec<Vec<f32>>,
    pub features: Vec<Vec<f32>>,
}

pub trait DSparkTarget {
    type Checkpoint;

    fn checkpoint(&self) -> Self::Checkpoint;
    fn restore(&mut self, checkpoint: &Self::Checkpoint);
    fn position(&self) -> usize;
    fn evaluate(
        &mut self,
        token_ids: &[u32],
        target_layers: &[usize],
    ) -> Result<TargetBatch, String>;
    fn current_logits(&self) -> &[f32];
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verification {
    Accepted(usize),
    Rejected { accepted: usize, target: u32 },
}

#[derive(Clone, Debug, PartialEq)]
pub struct RunOptions {
    pub max_tokens: usize,
    pub draft_n_max: usize,
    pub confidence_min: f32,
    pub stop_tokens: Vec<u32>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DSparkStats {
    pub drafted: usize,
    pub accepted: usize,
    pub target_evaluations: usize,
}

pub fn verify_ids(draft: &[u32], target: &[u32]) -> Verification {
    for (accepted, (&draft, &target)) in draft.iter().zip(target).enumerate() {
        if draft != target {
            return Verification::Rejected { accepted, target };
        }
    }
    Verification::Accepted(draft.len().min(target.len()))
}

pub fn prefill<T: DSparkTarget>(
    target: &mut T,
    draft: &mut DSparkSession<'_>,
    token_ids: &[u32],
    batch_size: usize,
) -> Result<(), String> {
    if token_ids.is_empty() || batch_size == 0 {
        return Err("DSpark prefill requires tokens and a positive batch size".into());
    }
    if target.position() != 0 || draft.position() != 0 {
        return Err("DSpark prefill requires empty target and draft sessions".into());
    }
    let target_layers = draft.target_layers().to_vec();
    for chunk in token_ids.chunks(batch_size) {
        let base = target.position();
        let batch = target.evaluate(chunk, &target_layers)?;
        validate_batch(&batch, chunk.len())?;
        inject_features(draft, base, &batch.features)?;
    }
    ensure_aligned(target, draft)
}

pub fn run_greedy<T: DSparkTarget>(
    target: &mut T,
    draft: &mut DSparkSession<'_>,
    options: RunOptions,
    mut on_token: impl FnMut(u32),
) -> Result<DSparkStats, String> {
    if options.max_tokens == 0
        || options.draft_n_max == 0
        || options.draft_n_max > draft.block_size()
        || !options.confidence_min.is_finite()
        || !(0.0..=1.0).contains(&options.confidence_min)
    {
        return Err("Invalid DSpark run options".into());
    }
    ensure_aligned(target, draft)?;

    let mut stats = DSparkStats::default();
    let mut generated = 0usize;
    let mut pending = greedy_token(target.current_logits())?;
    if options.stop_tokens.contains(&pending) {
        return Ok(stats);
    }
    on_token(pending);
    generated += 1;

    let target_layers = draft.target_layers().to_vec();
    let mut catch_up = false;
    while generated < options.max_tokens {
        ensure_aligned(target, draft)?;
        let remaining = options.max_tokens - generated;
        let keep = if catch_up {
            0
        } else {
            options.draft_n_max.min(remaining.saturating_sub(1))
        };
        let base = draft.position();
        let draft_ids = if keep == 0 {
            Vec::new()
        } else {
            let mut token_ids = draft
                .draft(pending, options.draft_n_max, options.confidence_min)?
                .token_ids;
            token_ids.truncate(keep);
            token_ids
        };
        stats.drafted = stats
            .drafted
            .checked_add(draft_ids.len())
            .ok_or("DSpark drafted token counter overflow")?;

        let step = run_step(target, pending, &draft_ids, &target_layers)?;
        stats.target_evaluations = stats
            .target_evaluations
            .checked_add(step.target_evaluations)
            .ok_or("DSpark target evaluation counter overflow")?;
        let accepted = match step.verification {
            Verification::Accepted(accepted) | Verification::Rejected { accepted, .. } => accepted,
        };
        if !draft_ids.is_empty() && std::env::var_os("RUST_DSPARK_TRACE_IDS").is_some() {
            eprintln!(
                "[DSPARK_BLOCK] {}",
                serde_json::json!({
                    "position": base, "draft_ids": draft_ids, "accepted": accepted,
                })
            );
        }
        stats.accepted = stats
            .accepted
            .checked_add(accepted)
            .ok_or("DSpark accepted token counter overflow")?;
        catch_up = matches!(step.verification, Verification::Rejected { .. });
        inject_features(draft, base, &step.features)?;
        ensure_aligned(target, draft)?;

        pending = step.next_pending;
        for token in step.output_ids {
            if options.stop_tokens.contains(&token) {
                return Ok(stats);
            }
            on_token(token);
            generated += 1;
            if generated == options.max_tokens {
                return Ok(stats);
            }
        }
    }
    Ok(stats)
}

struct Step {
    verification: Verification,
    output_ids: Vec<u32>,
    next_pending: u32,
    features: Vec<Vec<f32>>,
    target_evaluations: usize,
}

fn run_step<T: DSparkTarget>(
    target: &mut T,
    pending: u32,
    draft_ids: &[u32],
    target_layers: &[usize],
) -> Result<Step, String> {
    let checkpoint = target.checkpoint();
    let base = target.position();
    let mut inputs = Vec::with_capacity(draft_ids.len() + 1);
    inputs.push(pending);
    inputs.extend_from_slice(draft_ids);

    let batch = match target.evaluate(&inputs, target_layers) {
        Ok(batch) => batch,
        Err(error) => {
            target.restore(&checkpoint);
            return Err(error);
        }
    };
    if let Err(error) = validate_target_result(target, base, &batch, inputs.len()) {
        target.restore(&checkpoint);
        return Err(error);
    }
    let target_ids = batch
        .logits
        .iter()
        .map(|logits| greedy_token(logits))
        .collect::<Result<Vec<_>, _>>()?;
    let verification = verify_ids(draft_ids, &target_ids[..draft_ids.len()]);

    match verification {
        Verification::Accepted(accepted) if accepted == draft_ids.len() => {
            let next_pending = target_ids[draft_ids.len()];
            let mut output_ids = draft_ids.to_vec();
            output_ids.push(next_pending);
            Ok(Step {
                verification,
                output_ids,
                next_pending,
                features: batch.features,
                target_evaluations: 1,
            })
        }
        Verification::Rejected {
            accepted,
            target: replacement,
        } => {
            target.restore(&checkpoint);
            let replay = &inputs[..accepted + 1];
            let replay_batch = match target.evaluate(replay, target_layers) {
                Ok(batch) => batch,
                Err(error) => {
                    target.restore(&checkpoint);
                    return Err(error);
                }
            };
            if let Err(error) = validate_target_result(target, base, &replay_batch, replay.len()) {
                target.restore(&checkpoint);
                return Err(error);
            }
            if greedy_token(replay_batch.logits.last().unwrap())? != replacement {
                target.restore(&checkpoint);
                return Err("DSpark replay changed the target replacement token".into());
            }
            let mut output_ids = draft_ids[..accepted].to_vec();
            output_ids.push(replacement);
            Ok(Step {
                verification,
                output_ids,
                next_pending: replacement,
                features: replay_batch.features,
                target_evaluations: 2,
            })
        }
        Verification::Accepted(_) => Err("DSpark verification length mismatch".into()),
    }
}

fn validate_target_result<T: DSparkTarget>(
    target: &T,
    base: usize,
    batch: &TargetBatch,
    rows: usize,
) -> Result<(), String> {
    validate_batch(batch, rows)?;
    let expected = base
        .checked_add(rows)
        .ok_or("DSpark target position overflow")?;
    if target.position() != expected {
        return Err(format!(
            "DSpark target position mismatch: expected {expected}, got {}",
            target.position()
        ));
    }
    Ok(())
}

fn validate_batch(batch: &TargetBatch, rows: usize) -> Result<(), String> {
    if batch.logits.len() != rows
        || batch.features.len() != rows
        || batch.logits.iter().any(Vec::is_empty)
        || batch.features.iter().any(Vec::is_empty)
    {
        return Err("DSpark target returned invalid batch shapes".into());
    }
    Ok(())
}

fn inject_features(
    draft: &mut DSparkSession<'_>,
    base: usize,
    features: &[Vec<f32>],
) -> Result<(), String> {
    for (row, features) in features.iter().enumerate() {
        let position = base
            .checked_add(row)
            .ok_or("DSpark injection position overflow")?;
        draft.inject(position, features)?;
    }
    Ok(())
}

fn ensure_aligned<T: DSparkTarget>(target: &T, draft: &DSparkSession<'_>) -> Result<(), String> {
    if target.position() != draft.position() {
        return Err(format!(
            "DSpark session position mismatch: target {}, draft {}",
            target.position(),
            draft.position()
        ));
    }
    Ok(())
}

impl DSparkTarget for Qwen3Session<'_> {
    type Checkpoint = usize;

    fn checkpoint(&self) -> Self::Checkpoint {
        self.dspark_position()
    }

    fn restore(&mut self, checkpoint: &Self::Checkpoint) {
        self.dspark_restore(*checkpoint);
    }

    fn position(&self) -> usize {
        self.dspark_position()
    }

    fn evaluate(
        &mut self,
        token_ids: &[u32],
        target_layers: &[usize],
    ) -> Result<TargetBatch, String> {
        let capture = self.forward_causal_capture(token_ids, target_layers)?;
        let hidden = self.dspark_hidden_size();
        let vocab = self.last_logits().len();
        let mut features = vec![Vec::new(); token_ids.len()];
        for layer in capture.layer_inputs {
            if layer.len() != token_ids.len() * hidden {
                return Err("Qwen3 returned an invalid DSpark layer capture".into());
            }
            for (row, features) in features.iter_mut().enumerate() {
                features.extend_from_slice(&layer[row * hidden..(row + 1) * hidden]);
            }
        }
        if capture.logits.len() != token_ids.len() * vocab {
            return Err("Qwen3 returned invalid DSpark logits".into());
        }
        let logits = capture
            .logits
            .chunks_exact(vocab)
            .map(<[f32]>::to_vec)
            .collect();
        Ok(TargetBatch { logits, features })
    }

    fn current_logits(&self) -> &[f32] {
        self.last_logits()
    }
}

#[cfg(test)]
mod tests {
    use super::{run_step, verify_ids, DSparkTarget, TargetBatch, Verification};
    use std::collections::VecDeque;

    struct FakeTarget {
        position: usize,
        logits: Vec<f32>,
        plans: VecDeque<Vec<u32>>,
        evaluations: Vec<Vec<u32>>,
        restores: usize,
    }

    impl FakeTarget {
        fn new(plans: impl IntoIterator<Item = Vec<u32>>) -> Self {
            Self {
                position: 0,
                logits: logits(0),
                plans: plans.into_iter().collect(),
                evaluations: Vec::new(),
                restores: 0,
            }
        }
    }

    impl DSparkTarget for FakeTarget {
        type Checkpoint = usize;

        fn checkpoint(&self) -> Self::Checkpoint {
            self.position
        }

        fn restore(&mut self, checkpoint: &Self::Checkpoint) {
            self.position = *checkpoint;
            self.restores += 1;
        }

        fn position(&self) -> usize {
            self.position
        }

        fn evaluate(
            &mut self,
            token_ids: &[u32],
            _target_layers: &[usize],
        ) -> Result<TargetBatch, String> {
            let plan = self.plans.pop_front().ok_or("Missing fake target plan")?;
            if plan.len() != token_ids.len() {
                return Err("Invalid fake target plan".into());
            }
            self.evaluations.push(token_ids.to_vec());
            self.position += token_ids.len();
            self.logits = logits(*plan.last().unwrap());
            Ok(TargetBatch {
                logits: plan.into_iter().map(logits).collect(),
                features: token_ids.iter().map(|&id| vec![id as f32]).collect(),
            })
        }

        fn current_logits(&self) -> &[f32] {
            &self.logits
        }
    }

    fn logits(token: u32) -> Vec<f32> {
        let mut logits = vec![-1.0; 16];
        logits[token as usize] = 1.0;
        logits
    }

    #[test]
    fn verification_reports_full_acceptance_and_first_mismatch() {
        assert_eq!(
            verify_ids(&[2, 3, 4], &[2, 3, 4]),
            Verification::Accepted(3)
        );
        assert_eq!(
            verify_ids(&[2, 3, 4], &[2, 9, 4]),
            Verification::Rejected {
                accepted: 1,
                target: 9,
            }
        );
    }

    #[test]
    fn mismatch_restores_and_replays_only_committed_inputs() {
        let mut target = FakeTarget::new([vec![2, 9, 4, 8], vec![2, 9], vec![10]]);
        let step = run_step(&mut target, 1, &[2, 3, 4], &[0]).unwrap();

        assert_eq!(
            step.verification,
            Verification::Rejected {
                accepted: 1,
                target: 9,
            }
        );
        assert_eq!(step.output_ids, vec![2, 9]);
        assert_eq!(step.next_pending, 9);
        assert_eq!(step.features, vec![vec![1.0], vec![2.0]]);
        assert_eq!(target.evaluations, vec![vec![1, 2, 3, 4], vec![1, 2]]);
        assert_eq!(target.restores, 1);
        assert_eq!(target.position, 2);

        let catch_up = run_step(&mut target, step.next_pending, &[], &[0]).unwrap();
        assert_eq!(catch_up.output_ids, vec![10]);
        assert_eq!(catch_up.next_pending, 10);
        assert_eq!(target.evaluations.last(), Some(&vec![9]));
    }
}
