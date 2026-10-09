//! The semantic order shared by CPU execution and Vulkan command recording.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DenseStep {
    AttnNorm,
    Qkv,
    QkNormRope,
    AppendKv,
    Attention,
    AttnOut,
    AttnResidual,
    FfnNorm,
    GateUp,
    SiluMul,
    Down,
    FfnResidual,
}

pub(crate) trait DenseBlockOps {
    type Error;
    fn run_step(&mut self, layer: usize, step: DenseStep) -> Result<(), Self::Error>;
}

impl<F, E> DenseBlockOps for F
where
    F: FnMut(usize, DenseStep) -> Result<(), E>,
{
    type Error = E;
    fn run_step(&mut self, layer: usize, step: DenseStep) -> Result<(), E> {
        self(layer, step)
    }
}

pub(crate) fn run_dense_layer<E: DenseBlockOps>(
    executor: &mut E,
    layer: usize,
) -> Result<(), E::Error> {
    use DenseStep::*;
    for step in [
        AttnNorm,
        Qkv,
        QkNormRope,
        AppendKv,
        Attention,
        AttnOut,
        AttnResidual,
        FfnNorm,
        GateUp,
        SiluMul,
        Down,
        FfnResidual,
    ] {
        executor.run_step(layer, step)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_recipe_records_one_semantic_sequence() {
        let mut seen = Vec::new();
        run_dense_layer(
            &mut |layer, step| -> Result<(), crate::compute::ComputeError> {
                seen.push((layer, step));
                Ok(())
            },
            3,
        )
        .unwrap();
        use DenseStep::*;
        assert_eq!(
            seen,
            [
                AttnNorm,
                Qkv,
                QkNormRope,
                AppendKv,
                Attention,
                AttnOut,
                AttnResidual,
                FfnNorm,
                GateUp,
                SiluMul,
                Down,
                FfnResidual
            ]
            .map(|step| (3, step))
        );
    }

    #[test]
    fn dense_recipe_does_not_execute_past_a_failed_step() {
        let mut seen = Vec::new();
        let result = run_dense_layer(
            &mut |_, step| {
                if step == DenseStep::AppendKv {
                    return Err(crate::compute::ComputeError::State("injected".into()));
                }
                seen.push(step);
                Ok(())
            },
            0,
        );
        assert!(result.is_err());
        assert_eq!(
            seen,
            [DenseStep::AttnNorm, DenseStep::Qkv, DenseStep::QkNormRope]
        );
    }
}
