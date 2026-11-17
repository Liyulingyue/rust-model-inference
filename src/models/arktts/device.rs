//! ArkTTS's resident graph. Only request inputs, sampled IDs and final outputs cross the host boundary.
use super::Matrix;
use crate::compute::ComputePolicy;
use crate::vulkan::dense::UploadedBuffers;
use crate::vulkan::ops::{ArenaRegion, GpuMatmulSpec, OperatorBindings, Qwen3Ops, TokenCommands};
use crate::vulkan::{GpuBuffer, VulkanContext};
use std::collections::HashMap;

pub(super) type Result<T> = std::result::Result<T, String>;
fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}

// Values match the private shader's PC.op; tensor offsets are F32 words.
#[derive(Clone, Copy)]
#[repr(u32)]
pub(super) enum Op {
    Copy = 0,
    Add = 1,
    Bias = 2,
    SiluMul = 3,
    Snake = 4,
    Gelu = 5,
    Tanh = 6,
    LayerNorm = 7,
    RopeKv = 8,
    Attention = 9,
    Conv = 10,
    Transpose = 11,
    Lookup = 12,
    PackedEmbedding = 13,
    Codebook = 14,
    Rank = 15,
    Sample = 16,
    Subtract = 17,
    Zero = 18,
}

#[derive(Clone, Copy)]
pub(super) struct Tensor {
    pub region: ArenaRegion,
    pub rows: usize,
    pub width: usize,
}
impl Tensor {
    pub fn len(self) -> usize {
        self.rows * self.width
    }
    pub fn word(self) -> u32 {
        (self.region.offset / 4) as u32
    }
    pub fn row(self, row: usize) -> Self {
        assert!(row < self.rows);
        Self {
            region: ArenaRegion {
                offset: self.region.offset + row * self.width * 4,
                size: self.width * 4,
            },
            rows: 1,
            width: self.width,
        }
    }
}

#[derive(Default)]
struct Arena {
    end: usize,
    free: Vec<ArenaRegion>,
}
impl Arena {
    fn alloc(&mut self, rows: usize, width: usize) -> Result<Tensor> {
        let size = rows
            .checked_mul(width)
            .and_then(|n| n.checked_mul(4))
            .filter(|&n| n > 0)
            .ok_or("invalid ArkTTS device tensor size")?;
        let aligned = size.checked_add(15).ok_or("ArkTTS arena overflow")? & !15;
        let offset = if let Some(index) = self.free.iter().position(|r| r.size >= aligned) {
            let r = &mut self.free[index];
            let offset = r.offset;
            r.offset += aligned;
            r.size -= aligned;
            if r.size == 0 {
                self.free.remove(index);
            }
            offset
        } else {
            let offset = self.end;
            self.end = offset
                .checked_add(aligned)
                .filter(|&n| n / 4 <= u32::MAX as usize)
                .ok_or("ArkTTS device arena exceeds addressable size")?;
            offset
        };
        Ok(Tensor {
            region: ArenaRegion { offset, size },
            rows,
            width,
        })
    }
    fn release(&mut self, tensor: Tensor) {
        self.free.push(ArenaRegion {
            offset: tensor.region.offset,
            size: (tensor.region.size + 15) & !15,
        });
        self.free.sort_unstable_by_key(|r| r.offset);
        let mut merged: Vec<ArenaRegion> = Vec::new();
        for r in self.free.drain(..) {
            if let Some(previous) = merged.last_mut() {
                assert!(
                    previous.offset + previous.size <= r.offset,
                    "overlapping ArkTTS arena releases"
                );
                if previous.offset + previous.size == r.offset {
                    previous.size += r.size;
                    continue;
                }
            }
            merged.push(r);
        }
        self.free = merged;
    }
}

struct Binding {
    buffers: Vec<GpuBuffer>,
    spec: Option<GpuMatmulSpec>,
    ready: Option<OperatorBindings>,
}
enum Action {
    Write {
        tensor: Tensor,
        values: Vec<f32>,
    },
    Custom {
        binding: Option<usize>,
        push: Vec<u32>,
        groups: usize,
    },
    Matmul {
        binding: usize,
        input: Tensor,
        output: Tensor,
        scratch: [Tensor; 5],
    },
    Norm {
        binding: usize,
        input: Tensor,
        output: Tensor,
        epsilon: f32,
    },
    Trace {
        name: String,
        layer: Option<usize>,
        step: Option<usize>,
        tensor: Tensor,
        shape: Vec<usize>,
    },
}

pub(super) struct Graph {
    // Descriptors are destroyed before their immutable buffers.
    ops: Option<Qwen3Ops<'static>>,
    buffers: UploadedBuffers,
    context: &'static VulkanContext,
    arena: Arena,
    allocated_bytes: usize,
    bindings: Vec<Binding>,
    matrices: HashMap<usize, usize>,
    vectors: HashMap<(usize, usize), usize>,
    actions: Vec<Action>,
    pipeline: usize,
    dispatches: usize,
    reads: usize,
    uploads: usize,
    stage: &'static str,
    operators: [usize; 19],
    norm_ops: usize,
    trace_bytes: usize,
}
impl Graph {
    pub fn context(policy: ComputePolicy) -> Result<Option<&'static VulkanContext>> {
        if crate::ops::scalar_mode() {
            return policy
                .context_with(|| Err("ArkTTS scalar mode requires CPU".into()))
                .map_err(err);
        }
        policy.context().map_err(err)
    }
    pub fn new(context: &'static VulkanContext, stage: &'static str) -> Self {
        Self {
            ops: None,
            buffers: UploadedBuffers::new(context),
            context,
            arena: Arena::default(),
            allocated_bytes: 0,
            bindings: Vec::new(),
            matrices: HashMap::new(),
            vectors: HashMap::new(),
            actions: Vec::new(),
            pipeline: 0,
            dispatches: 0,
            reads: 0,
            uploads: 0,
            stage,
            operators: [0; 19],
            norm_ops: 0,
            trace_bytes: 0,
        }
    }
    pub fn alloc(&mut self, rows: usize, width: usize) -> Result<Tensor> {
        let tensor = self.arena.alloc(rows, width)?;
        if self.ops.is_some() && self.arena.end > self.allocated_bytes {
            return Err("ArkTTS device workspace capacity exceeded".into());
        }
        Ok(tensor)
    }
    pub fn free(&mut self, tensor: Tensor) {
        self.arena.release(tensor);
    }
    pub fn input(&mut self, values: &[f32], width: usize) -> Result<Tensor> {
        if width == 0 || values.len() % width != 0 {
            return Err("invalid ArkTTS device input".into());
        }
        let x = self.alloc(values.len() / width, width)?;
        self.write(x, values.to_vec());
        Ok(x)
    }
    pub fn input_ids(&mut self, values: &[u32], width: usize) -> Result<Tensor> {
        self.input(
            &values
                .iter()
                .map(|&v| f32::from_bits(v))
                .collect::<Vec<_>>(),
            width,
        )
    }
    pub fn write(&mut self, tensor: Tensor, values: Vec<f32>) {
        assert_eq!(tensor.len(), values.len());
        self.actions.push(Action::Write { tensor, values });
    }
    pub fn binding(&mut self, buffers: Vec<GpuBuffer>, spec: Option<GpuMatmulSpec>) -> usize {
        let id = self.bindings.len();
        self.bindings.push(Binding {
            buffers,
            spec,
            ready: None,
        });
        id
    }
    pub fn floats(&mut self, values: &[f32]) -> Result<GpuBuffer> {
        self.buffers.upload_f32(values).map_err(err)
    }
    pub fn vector(&mut self, values: &[f32]) -> Result<usize> {
        let key = (values.as_ptr() as usize, values.len());
        if let Some(&id) = self.vectors.get(&key) {
            return Ok(id);
        }
        let buffer = self.floats(values)?;
        let id = self.binding(vec![buffer], None);
        self.vectors.insert(key, id);
        Ok(id)
    }
    pub fn pair(&mut self, first: &[f32], second: &[f32]) -> Result<usize> {
        let a = self.floats(first)?;
        let b = self.floats(second)?;
        Ok(self.binding(vec![a, b], None))
    }
    pub fn custom(
        &mut self,
        op: Op,
        binding: Option<usize>,
        a: Tensor,
        b: Tensor,
        c: Tensor,
        count: usize,
        width: usize,
        parameters: &[u32],
        groups: Option<usize>,
    ) -> Result<()> {
        let mut push = vec![
            op as u32,
            a.word(),
            b.word(),
            c.word(),
            u32::try_from(count).map_err(err)?,
            u32::try_from(width).map_err(err)?,
        ];
        push.extend_from_slice(parameters);
        push.resize(13, 0);
        self.actions.push(Action::Custom {
            binding,
            push,
            groups: groups.unwrap_or(count.div_ceil(64)),
        });
        Ok(())
    }
    pub fn unary(&mut self, op: Op, x: Tensor, binding: Option<usize>) -> Result<()> {
        self.custom(op, binding, x, x, x, x.len(), x.width, &[], None)
    }
    pub fn copy(&mut self, input: Tensor) -> Result<Tensor> {
        let output = self.alloc(input.rows, input.width)?;
        self.custom(
            Op::Copy,
            None,
            input,
            output,
            output,
            input.len(),
            input.width,
            &[],
            None,
        )?;
        Ok(output)
    }
    pub fn add(&mut self, output: Tensor, branch: Tensor, scale: Option<&[f32]>) -> Result<()> {
        assert_eq!(output.len(), branch.len());
        let binding = scale.map(|s| self.vector(s)).transpose()?;
        self.custom(
            Op::Add,
            binding,
            output,
            branch,
            output,
            output.len(),
            output.width,
            &[u32::from(binding.is_some())],
            None,
        )
    }
    pub fn rms(&mut self, input: Tensor, weights: &[f32], epsilon: f32) -> Result<Tensor> {
        let output = self.alloc(input.rows, input.width)?;
        let binding = self.vector(weights)?;
        self.actions.push(Action::Norm {
            binding,
            input,
            output,
            epsilon,
        });
        Ok(output)
    }
    pub fn matmul(&mut self, matrix: &Matrix<'_>, input: Tensor) -> Result<Tensor> {
        let w = &matrix.weight;
        assert_eq!(input.width, w.n_in);
        let key = w as *const _ as usize;
        let binding = if let Some(&id) = self.matrices.get(&key) {
            id
        } else {
            let spec = GpuMatmulSpec::prepared(w).map_err(err)?;
            let bytes = w
                .kernel
                .weight_bytes()
                .ok_or("ArkTTS matrix has no native device storage")?;
            let buffer = self.buffers.upload(bytes).map_err(err)?;
            let id = self.binding(vec![buffer], Some(spec));
            self.matrices.insert(key, id);
            id
        };
        let output = self.alloc(input.rows, w.n_out)?;
        // The shared prepared kernels select activation quantization from the weight binding.
        let rows = input.rows.min(32);
        let q8 = self.alloc(rows, w.n_in.div_ceil(4))?;
        let scales = self.alloc(rows, w.n_in.div_ceil(32))?;
        let sums = self.alloc(rows, w.n_in.div_ceil(32))?;
        let q8k = self.alloc(rows, w.n_in.div_ceil(4))?;
        let scales_k = self.alloc(rows, w.n_in.div_ceil(256))?;
        let scratch = [q8, scales, sums, q8k, scales_k];
        self.actions.push(Action::Matmul {
            binding,
            input,
            output,
            scratch,
        });
        for t in scratch {
            self.free(t);
        }
        if !matrix.bias.is_empty() {
            let bias = self.vector(&matrix.bias)?;
            self.unary(Op::Bias, output, Some(bias))?;
        }
        Ok(output)
    }
    pub fn trace(
        &mut self,
        name: &str,
        layer: Option<usize>,
        step: Option<usize>,
        tensor: Tensor,
        shape: &[usize],
    ) {
        #[cfg(feature = "parity-trace")]
        if std::env::var_os("RMI_PARITY_TRACE").is_some() {
            self.actions.push(Action::Trace {
                name: name.into(),
                layer,
                step,
                tensor,
                shape: shape.to_vec(),
            });
        }
        let _ = (name, layer, step, tensor, shape);
    }
    pub fn execute(&mut self) -> Result<()> {
        if self.ops.is_none() {
            self.allocated_bytes = self.arena.end;
            let mut ops = Qwen3Ops::new_device_local_with_size(self.context, self.arena.end, 2048)
                .map_err(err)?;
            self.pipeline = ops
                .add_shader(include_bytes!("../../../shaders/bin/arktts.spv"))
                .map_err(err)?;
            self.ops = Some(ops);
        }
        let ops = self.ops.as_mut().unwrap();
        for binding in &mut self.bindings {
            if binding.ready.is_none() {
                binding.ready = Some(
                    match binding.spec {
                        Some(spec) => ops.bind_weight_buffers_mode(
                            &binding.buffers,
                            &[spec.format],
                            spec.mode,
                        ),
                        None => ops.bind_buffers(&binding.buffers),
                    }
                    .map_err(err)?,
                );
            }
        }
        let mut commands = None;
        let mut pending = 0;
        for action in self.actions.drain(..) {
            // Ordered writes cannot clobber an earlier input whose arena slot was reused.
            if let Action::Write { tensor, values } = action {
                if let Some(c) = commands.take() {
                    TokenCommands::submit_and_wait(c).map_err(err)?;
                }
                ops.write_f32(tensor.region, &values).map_err(err)?;
                self.uploads += values.len() * 4;
                continue;
            }
            if let Action::Trace {
                name,
                layer,
                step,
                tensor,
                shape,
            } = action
            {
                if let Some(c) = commands.take() {
                    TokenCommands::submit_and_wait(c).map_err(err)?;
                }
                let mut values = vec![0.0; tensor.len()];
                ops.read_f32_into(tensor.region, &mut values).map_err(err)?;
                self.trace_bytes += values.len() * 4;
                super::trace(&name, layer, step, &shape, &values)?;
                continue;
            }
            if commands.is_none() {
                commands = Some(TokenCommands::begin(self.context).map_err(err)?);
            }
            let c = commands.as_ref().unwrap();
            match action {
                Action::Custom {
                    binding,
                    push,
                    groups,
                } => {
                    self.operators[push[0] as usize] += 1;
                    // Vulkan 1.1 guarantees at least 65535 workgroups per dimension.
                    let max = 65535;
                    let x = groups.min(max);
                    ops.record_custom(
                        c,
                        self.pipeline,
                        binding.map(|id| self.bindings[id].ready.unwrap()),
                        &push,
                        [x as u32, groups.div_ceil(x) as u32, 1],
                    )
                    .map_err(err)?;
                }
                Action::Norm {
                    binding,
                    input,
                    output,
                    epsilon,
                } => {
                    self.norm_ops += 1;
                    ops.record_rms_norm_rows(
                        c,
                        self.bindings[binding].ready.unwrap(),
                        input.region,
                        output.region,
                        input.width,
                        epsilon,
                        input.rows,
                        input.width,
                        output.width,
                    )
                    .map_err(err)?;
                }
                Action::Matmul {
                    binding,
                    input,
                    output,
                    scratch,
                } => {
                    let [q8, scales, sums, q8k, scales_k] = scratch;
                    for start in (0..input.rows).step_by(32) {
                        let rows = (input.rows - start).min(32);
                        let xr = ArenaRegion {
                            offset: input.region.offset + start * input.width * 4,
                            size: rows * input.width * 4,
                        };
                        let yr = ArenaRegion {
                            offset: output.region.offset + start * output.width * 4,
                            size: rows * output.width * 4,
                        };
                        ops.record_weight_matmul_rows(
                            c,
                            self.bindings[binding].ready.unwrap(),
                            xr,
                            q8.region,
                            scales.region,
                            sums.region,
                            q8k.region,
                            scales_k.region,
                            &[(yr, output.width, output.width * 4)],
                            input.width,
                            rows,
                            input.width,
                        )
                        .map_err(err)?;
                    }
                }
                Action::Trace { .. } | Action::Write { .. } => unreachable!(),
            }
            self.dispatches += 1;
            pending += 1;
            // Long codec convolutions otherwise exceed RADV's command-buffer watchdog.
            if pending == 8 {
                commands = Some(commands.take().unwrap().flush().map_err(err)?);
                pending = 0;
            }
        }
        if let Some(c) = commands {
            c.submit_and_wait().map_err(err)?;
        }
        Ok(())
    }
    pub fn read(&mut self, tensor: Tensor) -> Result<Vec<f32>> {
        self.execute()?;
        let mut values = vec![0.0; tensor.len()];
        self.ops
            .as_ref()
            .unwrap()
            .read_f32_into(tensor.region, &mut values)
            .map_err(err)?;
        self.reads += values.len() * 4;
        Ok(values)
    }
}
impl Drop for Graph {
    fn drop(&mut self) {
        if std::env::var_os("RMI_COMPUTE_TRACE").is_some() {
            eprintln!("arktts-device stage={} backend=Vulkan graph_ops={} host_input_bytes={} host_output_bytes={} arena_bytes={}",
                self.stage,self.dispatches,self.uploads,self.reads,self.arena.end);
            eprintln!("arktts-operators stage={} attention={} norm={} rope_kv={} convolution={} embedding={} codebook_search={} sampling={} activation={} trace_read_bytes={}",
                self.stage,self.operators[9],self.norm_ops+self.operators[7],self.operators[8],
                self.operators[10]+self.operators[11],self.operators[12]+self.operators[13],
                self.operators[14],self.operators[16],self.operators[3]+self.operators[4]+self.operators[5]+self.operators[6],self.trace_bytes);
            eprintln!(
                "compute counters=device_cumulative {:?}",
                self.context.compute_stats()
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn arena_reuses_aligned_regions_and_rejects_overflow() {
        let mut arena = Arena::default();
        let a = arena.alloc(1, 3).unwrap();
        let b = arena.alloc(2, 5).unwrap();
        assert_eq!(b.region.offset, 16);
        arena.release(a);
        arena.release(b);
        assert_eq!(arena.alloc(1, 16).unwrap().region.offset, 0);
        assert_eq!(arena.end, 64);
        assert!(arena.alloc(usize::MAX, 4).is_err());
        assert!(arena.alloc(0, 1).is_err());
    }

    #[test]
    #[ignore = "requires a hardware Vulkan device"]
    fn attention_crosses_subgroup_and_window_boundaries() {
        let context = Graph::context(ComputePolicy::Vulkan).unwrap().unwrap();
        for (time, window) in [(31, 0), (32, 0), (33, 0), (129, 128), (513, 512)] {
            let mut graph = Graph::new(context, "attention_test");
            let mut query = vec![0.0; 64];
            query[0] = 1.0;
            let mut keys = vec![0.0; (time + 1) * 64];
            let mut values = vec![0.0; keys.len()];
            for t in 0..=time {
                keys[t * 64] = if t == 0 || t == time {
                    40.0
                } else {
                    (t % 3) as f32 * 8.0 - 16.0
                };
                for d in 0..64 {
                    values[t * 64 + d] = (t * 7 + d) as f32 * 0.03125;
                }
            }
            let first = if window == 0 {
                0
            } else {
                (time + 1).saturating_sub(window)
            };
            let mut probabilities: Vec<f32> =
                (first..=time).map(|t| keys[t * 64] * 0.125).collect();
            crate::ops::softmax_inplace(&mut probabilities);
            let expected: Vec<f32> = (0..64)
                .map(|d| {
                    probabilities
                        .iter()
                        .enumerate()
                        .map(|(j, &p)| p * values[(first + j) * 64 + d])
                        .sum()
                })
                .collect();
            let q = graph.input(&query, 64).unwrap();
            let k = graph.input(&keys, 64).unwrap();
            let v = graph.input(&values, 64).unwrap();
            let out = graph.alloc(1, 64).unwrap();
            graph
                .custom(
                    Op::Attention,
                    None,
                    q,
                    k,
                    v,
                    1,
                    1,
                    &[1, time as u32, (time + 1) as u32, window as u32, out.word()],
                    Some(1),
                )
                .unwrap();
            let actual = graph.read(out).unwrap();
            for (a, b) in actual.iter().zip(expected) {
                assert!(
                    (a - b).abs() < 2e-5,
                    "time={time} window={window} {a} != {b}"
                );
            }
        }
    }

    #[test]
    #[ignore = "requires a hardware Vulkan device"]
    fn device_primitives_and_sampling_match_cpu() {
        use rand::{Rng, SeedableRng};
        let context = Graph::context(ComputePolicy::Vulkan).unwrap().unwrap();
        let mut graph = Graph::new(context, "primitive_test");
        // Reuse of an uploaded input must preserve command ordering.
        let input = graph.input(&[1.0, 2.0, 3.0, 4.0], 1).unwrap();
        let copied = graph.copy(input).unwrap();
        graph.free(input);
        let reused = graph.input(&[9.0, 8.0, 7.0, 6.0], 1).unwrap();
        let weights = [0.25, -0.5, 0.75];
        let bias = [0.125];
        let binding = graph.pair(&weights, &bias).unwrap();
        let output = graph.alloc(4, 1).unwrap();
        graph
            .custom(
                Op::Conv,
                Some(binding),
                copied,
                output,
                output,
                4,
                1,
                &[1, 3, 1, 1, 1, 4, 0],
                None,
            )
            .unwrap();
        let expected = [0.875, 1.125, 1.625, 2.125];
        let actual = graph.read(output).unwrap();
        for (a, b) in actual.iter().zip(expected) {
            assert!((a - b).abs() < 1e-6, "{actual:?}");
        }
        assert_eq!(graph.read(reused).unwrap(), [9.0, 8.0, 7.0, 6.0]);
        for temperature in [0.0, 0.8] {
            for top_p in [0.01, 0.9, 1.0] {
                let mut graph = Graph::new(context, "sampling_test");
                let logits: Vec<f32> = (0..256).map(|i| ((i * 17) % 23) as f32 * 0.25).collect();
                let x = graph.input(&logits, 256).unwrap();
                let sampling = super::super::Sampling {
                    temperature,
                    top_p,
                    top_k: 50,
                    seed: 42,
                };
                let mut cpu = rand::rngs::StdRng::seed_from_u64(42);
                let mut gpu = cpu.clone();
                let expected = super::super::draw(
                    &logits
                        .iter()
                        .enumerate()
                        .map(|(i, &v)| (i as u32, v))
                        .collect::<Vec<_>>(),
                    sampling,
                    &mut cpu,
                )
                .unwrap();
                let actual = super::super::generation_device::sample(
                    &mut graph, x, false, sampling, &mut gpu,
                )
                .unwrap();
                assert_eq!(actual, expected, "temperature={temperature} top_p={top_p}");
                assert_eq!(cpu.gen::<u64>(), gpu.gen::<u64>(), "RNG consumption");
            }
        }
        let mut graph = Graph::new(context, "invalid_sampling_test");
        let x = graph.input(&[0.0, f32::NAN, f32::INFINITY], 3).unwrap();
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        assert!(super::super::generation_device::sample(
            &mut graph,
            x,
            false,
            super::super::Sampling::default(),
            &mut rng
        )
        .is_err());
    }
}
