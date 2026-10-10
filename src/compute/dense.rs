//! One dense expression, interpreted by CPU execution and Vulkan recording.
use crate::core::thread_pool::ComputePool;
use crate::ops::kernel::{PreparedRows, Weight};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DenseTensor {
    X,
    Normed,
    Q,
    K,
    V,
    Attn,
    Projection,
    Gate,
    Up,
    Down,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DenseMatrix {
    Q,
    K,
    V,
    AttnOut,
    Gate,
    Up,
    Down,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DenseNorm {
    Attn,
    Ffn,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DenseOp {
    RmsNorm {
        input: DenseTensor,
        weight: DenseNorm,
        output: DenseTensor,
    },
    Linear {
        input: DenseTensor,
        projections: &'static [(DenseMatrix, DenseTensor)],
    },
    QkNormRope,
    AppendKv,
    Attention,
    Add {
        input: DenseTensor,
        output: DenseTensor,
    },
    /// Overwrite up with SiLU(gate) * up.
    SiluMul {
        gate: DenseTensor,
        up: DenseTensor,
    },
    Moe {
        input: DenseTensor,
        output: DenseTensor,
    },
}

pub(crate) trait DenseBlockOps {
    type Error;
    fn execute(&mut self, layer: usize, op: DenseOp) -> Result<(), Self::Error>;
}

impl<F, E> DenseBlockOps for F
where
    F: FnMut(usize, DenseOp) -> Result<(), E>,
{
    type Error = E;
    fn execute(&mut self, layer: usize, op: DenseOp) -> Result<(), E> {
        self(layer, op)
    }
}

pub(crate) fn run_dense_layer<E: DenseBlockOps>(
    executor: &mut E,
    layer: usize,
    moe: bool,
) -> Result<(), E::Error> {
    use DenseTensor::*;
    for op in [
        DenseOp::RmsNorm {
            input: X,
            weight: DenseNorm::Attn,
            output: Normed,
        },
        DenseOp::Linear {
            input: Normed,
            projections: &[
                (DenseMatrix::Q, Q),
                (DenseMatrix::K, K),
                (DenseMatrix::V, V),
            ],
        },
        DenseOp::QkNormRope,
        DenseOp::AppendKv,
        DenseOp::Attention,
        DenseOp::Linear {
            input: Attn,
            projections: &[(DenseMatrix::AttnOut, Projection)],
        },
        DenseOp::Add {
            input: Projection,
            output: X,
        },
        DenseOp::RmsNorm {
            input: X,
            weight: DenseNorm::Ffn,
            output: Normed,
        },
    ] {
        executor.execute(layer, op)?;
    }
    if moe {
        executor.execute(
            layer,
            DenseOp::Moe {
                input: Normed,
                output: Down,
            },
        )?;
    } else {
        executor.execute(
            layer,
            DenseOp::Linear {
                input: Normed,
                projections: &[(DenseMatrix::Gate, Gate), (DenseMatrix::Up, Up)],
            },
        )?;
        executor.execute(layer, DenseOp::SiluMul { gate: Gate, up: Up })?;
        executor.execute(
            layer,
            DenseOp::Linear {
                input: Up,
                projections: &[(DenseMatrix::Down, Down)],
            },
        )?;
    }
    executor.execute(
        layer,
        DenseOp::Add {
            input: Down,
            output: X,
        },
    )
}

/// Borrow existing scratch in DenseTensor order; taking output views avoids aliasing
/// without copying activations or constructing raw pointers.
pub(crate) struct DenseCpu<'a, 'w> {
    pub buffers: [&'a mut [f32]; 10],
    pub matrices: [&'a Weight<'w>; 7],
    pub norms: [&'a [f32]; 2],
    pub prepared: &'a mut PreparedRows,
    pub pool: &'a ComputePool,
    pub rows: usize,
    pub eps: f32,
    pub approximate_silu: bool,
}

impl DenseCpu<'_, '_> {
    fn with_outputs<const N: usize>(
        &mut self,
        input: DenseTensor,
        outputs: [DenseTensor; N],
        f: impl FnOnce(&mut Self, &[f32], [&mut [f32]; N]) -> Result<(), String>,
    ) -> Result<(), String> {
        for (i, out) in outputs.iter().enumerate() {
            if *out == input || outputs[..i].contains(out) {
                return Err("dense operator has aliased input/output buffers".into());
            }
        }
        let mut taken = outputs.map(|out| std::mem::take(&mut self.buffers[out as usize]));
        let x = std::mem::take(&mut self.buffers[input as usize]);
        let result = f(self, x, taken.each_mut().map(|out| &mut **out));
        self.buffers[input as usize] = x;
        for (out, values) in outputs.into_iter().zip(taken) {
            self.buffers[out as usize] = values;
        }
        result
    }

    fn linear<const N: usize>(
        &mut self,
        input: DenseTensor,
        projections: [(DenseMatrix, DenseTensor); N],
    ) -> Result<(), String> {
        self.with_outputs(
            input,
            projections.map(|(_, out)| out),
            |this, x, outputs| {
                if this.rows == 0 || x.len() % this.rows != 0 {
                    return Err("dense linear input shape mismatch".into());
                }
                let weights = projections.map(|(weight, _)| this.matrices[weight as usize]);
                for (weight, out) in weights.iter().zip(&outputs) {
                    if weight.n_in != x.len() / this.rows
                        || this.rows.checked_mul(weight.n_out) != Some(out.len())
                    {
                        return Err("dense linear projection shape mismatch".into());
                    }
                }
                this.prepared.prepare(
                    x,
                    this.rows,
                    x.len() / this.rows,
                    weights.iter().any(|w| w.needs_q8_0_activation()),
                    weights.iter().any(|w| w.uses_q8_k()),
                )?;
                let mut outputs = outputs.into_iter();
                this.prepared.matmul_group(
                    x,
                    weights.map(|w| (w, outputs.next().unwrap())),
                    this.pool,
                )
            },
        )
    }

    pub fn execute(&mut self, op: DenseOp) -> Result<(), String> {
        match op {
            DenseOp::RmsNorm {
                input,
                weight,
                output,
            } => self.with_outputs(input, [output], |this, x, [out]| {
                let norm = this.norms[weight as usize];
                if norm.is_empty()
                    || this.rows.checked_mul(norm.len()) != Some(x.len())
                    || out.len() != x.len()
                {
                    return Err("dense RMSNorm shape mismatch".into());
                }
                for (x, out) in x
                    .chunks_exact(norm.len())
                    .zip(out.chunks_exact_mut(norm.len()))
                {
                    crate::ops::rms_norm(x, norm, out, this.eps);
                }
                Ok(())
            }),
            DenseOp::Linear { input, projections } => match *projections {
                [a] => self.linear(input, [a]),
                [a, b] => self.linear(input, [a, b]),
                [a, b, c] => self.linear(input, [a, b, c]),
                _ => Err("dense linear group must contain 1..=3 projections".into()),
            },
            DenseOp::Add { input, output } => self.with_outputs(input, [output], |_, x, [out]| {
                if x.len() != out.len() {
                    return Err("dense residual shape mismatch".into());
                }
                crate::ops::vec_add_into(x, out);
                Ok(())
            }),
            DenseOp::SiluMul { gate, up } => self.with_outputs(gate, [up], |this, gate, [up]| {
                if this.rows == 0 || gate.len() % this.rows != 0 || up.len() != gate.len() {
                    return Err("dense SiLU shape mismatch".into());
                }
                let width = gate.len() / this.rows;
                if width == 0 {
                    return Err("dense SiLU width must be nonzero".into());
                }
                for (gate, up) in gate.chunks_exact(width).zip(up.chunks_exact_mut(width)) {
                    if this.approximate_silu {
                        crate::ops::silu_mul_approx_inplace(gate, up);
                    } else {
                        crate::ops::silu_mul_inplace(gate, up);
                    }
                }
                Ok(())
            }),
            _ => Err("dense CPU operation requires a model state adapter".into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dense_expression_carries_operands_and_moe_replaces_only_ffn() {
        use DenseTensor::*;
        let mut dense = Vec::new();
        run_dense_layer(
            &mut |_, op| -> Result<(), String> {
                dense.push(op);
                Ok(())
            },
            0,
            false,
        )
        .unwrap();
        assert!(dense.contains(&DenseOp::Linear {
            input: Normed,
            projections: &[
                (DenseMatrix::Q, Q),
                (DenseMatrix::K, K),
                (DenseMatrix::V, V)
            ]
        }));
        assert!(dense.contains(&DenseOp::SiluMul { gate: Gate, up: Up }));
        assert!(dense.contains(&DenseOp::Linear {
            input: Up,
            projections: &[(DenseMatrix::Down, Down)]
        }));
        assert!(dense.contains(&DenseOp::Add {
            input: Projection,
            output: X
        }));
        assert!(dense.contains(&DenseOp::Add {
            input: Down,
            output: X
        }));
        let mut moe = Vec::new();
        run_dense_layer(
            &mut |_, op| -> Result<(), String> {
                moe.push(op);
                Ok(())
            },
            0,
            true,
        )
        .unwrap();
        assert_eq!(&dense[..8], &moe[..8]);
        assert_eq!(
            moe[8],
            DenseOp::Moe {
                input: Normed,
                output: Down
            }
        );
        assert_eq!(moe[9], *dense.last().unwrap());
    }
    #[test]
    fn dense_recipe_does_not_execute_past_a_failed_step() {
        let mut seen = Vec::new();
        let result = run_dense_layer(
            &mut |_, op| {
                if op == DenseOp::AppendKv {
                    return Err("injected");
                }
                seen.push(op);
                Ok(())
            },
            0,
            false,
        );
        assert!(result.is_err());
        assert_eq!(seen.len(), 3);
        assert_eq!(seen.last(), Some(&DenseOp::QkNormRope));
    }
    #[test]
    fn cpu_operators_use_operands_and_restore_views_after_validation_error() {
        use crate::ops::kernel::QuantizedTensor;
        use DenseTensor::*;
        let weight = Weight::from_quantized(QuantizedTensor::F32 {
            data: vec![1.0; 16],
            n_in: 4,
            n_out: 4,
        });
        let mut values: [Vec<f32>; 10] = std::array::from_fn(|_| vec![123.0; 8]);
        values[X as usize] = vec![1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0];
        let mut prepared = PreparedRows::new(2, 4);
        let pool = ComputePool::new(2);
        let mut cpu = DenseCpu {
            buffers: values.each_mut().map(Vec::as_mut_slice),
            matrices: [&weight; 7],
            norms: [&[1.0; 4]; 2],
            prepared: &mut prepared,
            pool: &pool,
            rows: 2,
            eps: 1e-5,
            approximate_silu: true,
        };
        let input = cpu.buffers[X as usize].to_vec();
        assert!(cpu
            .execute(DenseOp::Linear {
                input: X,
                projections: &[(DenseMatrix::Q, Q), (DenseMatrix::K, Q)]
            })
            .is_err());
        assert_eq!(cpu.buffers[Q as usize], &[123.0; 8]);
        assert_eq!(cpu.buffers[X as usize], input);
        cpu.execute(DenseOp::Linear {
            input: X,
            projections: &[(DenseMatrix::Q, Q), (DenseMatrix::K, K)],
        })
        .unwrap();
        assert_eq!(
            cpu.buffers[Q as usize],
            &[10.0, 10.0, 10.0, 10.0, -10.0, -10.0, -10.0, -10.0]
        );
        assert_eq!(cpu.buffers[K as usize], cpu.buffers[Q as usize]);
        cpu.buffers[Gate as usize].copy_from_slice(&input);
        let mut expected = cpu.buffers[Up as usize].to_vec();
        for (gate, up) in input.chunks_exact(4).zip(expected.chunks_exact_mut(4)) {
            crate::ops::silu_mul_approx_inplace(gate, up);
        }
        cpu.execute(DenseOp::SiluMul { gate: Gate, up: Up })
            .unwrap();
        assert_eq!(
            cpu.buffers[Up as usize]
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>(),
            expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
        cpu.execute(DenseOp::Add {
            input: Up,
            output: X,
        })
        .unwrap();
        for ((actual, input), add) in cpu.buffers[X as usize].iter().zip(input).zip(expected) {
            assert_eq!(actual.to_bits(), (input + add).to_bits());
        }
        assert!(cpu
            .execute(DenseOp::RmsNorm {
                input: Q,
                weight: DenseNorm::Attn,
                output: Q
            })
            .is_err());
        cpu.execute(DenseOp::RmsNorm {
            input: X,
            weight: DenseNorm::Attn,
            output: Normed,
        })
        .unwrap();
        assert_eq!(cpu.buffers[Normed as usize].len(), 8);
    }
}
