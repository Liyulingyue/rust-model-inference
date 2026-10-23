//! Resident Dory graph: projections, loop state and KV stay on the device.
use super::dense::UploadedBuffers;
use super::ops::{ArenaLayout, ArenaRegion, OperatorBindings, Qwen3Ops, TokenCommands};
use super::{VulkanContext, VulkanError};
use crate::app::cli::KvFormat;
use crate::core::tensor::GGMLType;
use crate::models::dory::weights::DoryLayerWeights;
use crate::models::dory::{DoryConfig, DoryModel};
use crate::ops::kernel::Weight;

type Result<T> = std::result::Result<T, VulkanError>;

struct Matrix {
    binding: OperatorBindings,
    n_in: usize,
    n_out: usize,
    format: u32,
}

enum Layer {
    Attention {
        vectors: OperatorBindings,
        q: Matrix,
        k: Matrix,
        v: Matrix,
        gate: Matrix,
        out: Matrix,
    },
    Mlp {
        vectors: OperatorBindings,
        gate: Matrix,
        up: Matrix,
        down: Matrix,
    },
}

pub(crate) struct DoryVulkanSession {
    ops: Qwen3Ops<'static>,
    _buffers: UploadedBuffers,
    context: &'static VulkanContext,
    layout: ArenaLayout,
    input_h: ArenaRegion,
    state: ArenaRegion,
    rope_sliding: ArenaRegion,
    layers: Vec<Layer>,
    slots: Vec<Option<usize>>,
    output: Matrix,
    output_scale: OperatorBindings,
    graph_pipeline: usize,
    matmul_pipeline: usize,
    capacity: usize,
    pub(crate) failed: bool,
}

fn matrix(ops: &mut Qwen3Ops<'_>, buffers: &mut UploadedBuffers, w: &Weight<'_>) -> Result<Matrix> {
    let format = match w.ggml_type {
        GGMLType::Q5K => 13,
        GGMLType::Q6K => 14,
        GGMLType::BF16 => 30,
        _ => {
            return Err(VulkanError::UnsupportedShape(
                "Dory Vulkan currently supports Q5_K/Q6_K/BF16 projections".into(),
            ))
        }
    };
    if w.n_in % 256 != 0 {
        return Err(VulkanError::UnsupportedShape(
            "Dory Vulkan projection width must be a multiple of 256".into(),
        ));
    }
    let bytes = w
        .kernel
        .weight_bytes()
        .ok_or_else(|| VulkanError::UnsupportedShape("missing Dory weight bytes".into()))?;
    let buffer = buffers.upload(bytes)?;
    Ok(Matrix {
        binding: ops.bind_buffers(&[buffer])?,
        n_in: w.n_in,
        n_out: w.n_out,
        format,
    })
}

impl DoryVulkanSession {
    pub(crate) fn new(
        model: &DoryModel,
        context: &'static VulkanContext,
        capacity: usize,
    ) -> Result<Self> {
        if !context.shader_float64 {
            return Err(VulkanError::UnsupportedShape(
                "Dory Vulkan requires shaderFloat64 for recurrent normalization and attention"
                    .into(),
            ));
        }
        let cfg = &model.config;
        let mut slots = vec![None; cfg.n_layer];
        let mut count = 0;
        for (layer, slot) in slots.iter_mut().enumerate() {
            if cfg.is_attention(layer) {
                *slot = Some(count);
                count += if cfg.in_recurrent_block(layer) {
                    cfg.n_loops
                } else {
                    1
                };
            }
        }
        let layout = ArenaLayout::build_rows(
            cfg.n_embd,
            cfg.n_ff,
            cfg.n_head,
            cfg.n_head_kv,
            cfg.head_dim,
            cfg.vocab_size,
            count,
            capacity,
            1,
        )?;
        let input_h = ArenaRegion {
            offset: layout.total_size(),
            size: cfg.n_embd * 4,
        };
        let state = ArenaRegion {
            offset: input_h
                .offset
                .checked_add(input_h.size)
                .ok_or(VulkanError::OutOfMemory)?,
            size: cfg.n_embd * 4,
        };
        let rope_sliding = ArenaRegion {
            offset: state
                .offset
                .checked_add(state.size)
                .ok_or(VulkanError::OutOfMemory)?,
            size: cfg.head_dim * 4,
        };
        let bytes = rope_sliding
            .offset
            .checked_add(rope_sliding.size)
            .ok_or(VulkanError::OutOfMemory)?;
        if bytes / 4 > u32::MAX as usize
            || bytes as u64 > context.limits.max_storage_buffer_range as u64
        {
            return Err(VulkanError::UnsupportedShape("Dory Vulkan context exceeds the device storage buffer range; reduce context capacity".into()));
        }
        let mut ops = Qwen3Ops::new_device_local_with_size(context, bytes, cfg.n_layer * 7 + 4)?;
        let graph_pipeline = ops.add_shader(include_bytes!("../../shaders/bin/dory.spv"))?;
        let matmul_pipeline =
            ops.add_shader(include_bytes!("../../shaders/bin/dory_matmul.spv"))?;
        let mut buffers = UploadedBuffers::new(context);
        let mut layers = Vec::with_capacity(cfg.n_layer);
        for w in &model.layers {
            let (alpha, rest) = match w {
                DoryLayerWeights::Attention {
                    residual_alpha,
                    gate_scale,
                    qk_scale,
                    ..
                } => (residual_alpha, [gate_scale, qk_scale]),
                DoryLayerWeights::Mlp {
                    residual_alpha,
                    suv_gate,
                    suv_up,
                    ..
                } => (residual_alpha, [suv_gate, suv_up]),
            };
            let mut vectors: Vec<f32> = alpha
                .iter()
                .map(|v| v * (cfg.alpha_init / cfg.initializer_range))
                .collect();
            vectors.extend(rest[0]);
            vectors.extend(rest[1]);
            let buffer = buffers.upload_f32(&vectors)?;
            let vectors = ops.bind_buffers(&[buffer])?;
            layers.push(match w {
                DoryLayerWeights::Attention {
                    wq,
                    wk,
                    wv,
                    w_gate,
                    wo,
                    ..
                } => Layer::Attention {
                    vectors,
                    q: matrix(&mut ops, &mut buffers, wq)?,
                    k: matrix(&mut ops, &mut buffers, wk)?,
                    v: matrix(&mut ops, &mut buffers, wv)?,
                    gate: matrix(&mut ops, &mut buffers, w_gate)?,
                    out: matrix(&mut ops, &mut buffers, wo)?,
                },
                DoryLayerWeights::Mlp {
                    w_gate,
                    w_up,
                    w_down,
                    ..
                } => Layer::Mlp {
                    vectors,
                    gate: matrix(&mut ops, &mut buffers, w_gate)?,
                    up: matrix(&mut ops, &mut buffers, w_up)?,
                    down: matrix(&mut ops, &mut buffers, w_down)?,
                },
            });
        }
        let output = matrix(&mut ops, &mut buffers, &model.output)?;
        let scales: Vec<f32> = model
            .output_scale
            .iter()
            .map(|s| s * (1.0 / cfg.initializer_range))
            .collect();
        let buffer = buffers.upload_f32(&scales)?;
        let output_scale = ops.bind_buffers(&[buffer])?;
        Ok(Self {
            ops,
            _buffers: buffers,
            context,
            layout,
            input_h,
            state,
            rope_sliding,
            layers,
            slots,
            output,
            output_scale,
            graph_pipeline,
            matmul_pipeline,
            capacity,
            failed: false,
        })
    }

    fn word(region: ArenaRegion) -> u32 {
        (region.offset / 4) as u32
    }

    fn graph(
        &self,
        commands: &TokenCommands<'_>,
        binding: Option<OperatorBindings>,
        push: [u32; 14],
        threads: usize,
    ) -> Result<()> {
        self.ops.record_custom(
            commands,
            self.graph_pipeline,
            binding,
            &push,
            [threads.div_ceil(64) as u32, 1, 1],
        )
    }

    fn norm(
        &self,
        commands: &TokenCommands<'_>,
        region: ArenaRegion,
        heads: usize,
        width: usize,
    ) -> Result<()> {
        self.graph(
            commands,
            None,
            [
                0,
                Self::word(region),
                0,
                0,
                heads as u32,
                width as u32,
                0,
                0,
                0,
                0,
                0,
                0,
                (1.0 / (width as f32).sqrt()).to_bits(),
                0,
            ],
            heads * 64,
        )
    }

    fn copy_add(
        &self,
        commands: &TokenCommands<'_>,
        a: ArenaRegion,
        b: Option<ArenaRegion>,
        c: ArenaRegion,
        n: usize,
    ) -> Result<()> {
        self.graph(
            commands,
            None,
            [
                if b.is_some() { 2 } else { 1 },
                Self::word(a),
                b.map(Self::word).unwrap_or(0),
                Self::word(c),
                n as u32,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            n,
        )
    }

    fn project(
        &self,
        commands: &TokenCommands<'_>,
        w: &Matrix,
        input: ArenaRegion,
        output: ArenaRegion,
    ) -> Result<()> {
        if w.format != 30 {
            self.ops.record_quantize_q8_k_ggml(
                commands,
                input,
                self.layout.q8k,
                self.layout.q8k_scales,
                w.n_in,
            )?;
        }
        self.ops.record_custom(
            commands,
            self.matmul_pipeline,
            Some(w.binding),
            &[
                Self::word(input),
                Self::word(self.layout.q8k),
                Self::word(self.layout.q8k_scales),
                Self::word(output),
                w.n_in as u32,
                w.n_out as u32,
                w.format,
            ],
            [w.n_out.div_ceil(8) as u32, 1, 1],
        )
    }

    fn layer(
        &self,
        commands: &TokenCommands<'_>,
        cfg: &DoryConfig,
        layer: usize,
        loop_idx: usize,
        position: usize,
        kv: KvFormat,
    ) -> Result<()> {
        let l = &self.layout;
        let vectors = match &self.layers[layer] {
            Layer::Attention {
                vectors,
                q,
                k,
                v,
                gate,
                out,
            } => {
                let qn = cfg.n_head * cfg.head_dim;
                let kn = cfg.n_head_kv * cfg.head_dim;
                self.project(commands, q, l.x, l.q)?;
                self.project(commands, k, l.x, l.k)?;
                self.project(commands, v, l.x, l.v)?;
                self.project(commands, gate, l.x, l.gate)?;
                let rope = if cfg.attention_is_sliding(layer) {
                    self.rope_sliding
                } else {
                    l.rope
                };
                for (r, n, h) in [(l.q, qn, cfg.n_head), (l.k, kn, cfg.n_head_kv)] {
                    self.graph(
                        commands,
                        None,
                        [
                            3,
                            Self::word(r),
                            Self::word(rope),
                            0,
                            n as u32,
                            cfg.head_dim as u32,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                        ],
                        n / 2,
                    )?;
                    self.norm(commands, r, h, cfg.head_dim)?;
                }
                let half = u32::from(matches!(kv, KvFormat::F16));
                self.graph(
                    commands,
                    Some(*vectors),
                    [
                        4,
                        Self::word(l.q),
                        0,
                        0,
                        qn as u32,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        (cfg.n_embd + qn) as u32,
                        ((cfg.head_dim as f32).sqrt() / cfg.initializer_range).to_bits(),
                        half,
                    ],
                    qn,
                )?;
                let slot = self.slots[layer].unwrap()
                    + if cfg.in_recurrent_block(layer) {
                        loop_idx
                    } else {
                        0
                    };
                let window = if cfg.attention_is_sliding(layer) {
                    cfg.sliding_window.min(self.capacity)
                } else {
                    self.capacity
                };
                let seq = (position + 1).min(window);
                let cache = slot * self.capacity * kn;
                let kb = Self::word(l.kv_k) + cache as u32;
                let vb = Self::word(l.kv_v) + cache as u32;
                let offset = ((position % window) * kn) as u32;
                self.graph(
                    commands,
                    None,
                    [
                        5,
                        Self::word(l.k),
                        Self::word(l.v),
                        kb + offset,
                        kn as u32,
                        0,
                        vb + offset,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        half,
                    ],
                    kn,
                )?;
                let mut push = [
                    6,
                    Self::word(l.q),
                    kb,
                    Self::word(l.scores),
                    seq as u32,
                    cfg.head_dim as u32,
                    cfg.n_head as u32,
                    cfg.n_head_kv as u32,
                    0,
                    window as u32,
                    ((position + 1 - seq) % window) as u32,
                    0,
                    (1.0 / (cfg.head_dim as f32).sqrt()).to_bits(),
                    half,
                ];
                self.graph(commands, None, push, seq * cfg.n_head)?;
                push[0] = 7;
                push[1] = Self::word(l.scores);
                self.graph(commands, None, push, cfg.n_head)?;
                push[0] = 8;
                push[2] = vb;
                push[3] = Self::word(l.attn);
                self.graph(commands, None, push, qn)?;
                self.graph(
                    commands,
                    Some(*vectors),
                    [
                        9,
                        Self::word(l.attn),
                        Self::word(l.gate),
                        0,
                        qn as u32,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        cfg.n_embd as u32,
                        (cfg.gate_init.sqrt() / cfg.initializer_range).to_bits(),
                        0,
                    ],
                    qn,
                )?;
                self.norm(commands, l.attn, 1, qn)?;
                self.project(commands, out, l.attn, l.projection)?;
                *vectors
            }
            Layer::Mlp {
                vectors,
                gate,
                up,
                down,
            } => {
                self.project(commands, gate, l.x, l.gate)?;
                self.project(commands, up, l.x, l.up)?;
                self.graph(
                    commands,
                    Some(*vectors),
                    [
                        10,
                        Self::word(l.gate),
                        Self::word(l.up),
                        0,
                        cfg.n_ff as u32,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        cfg.n_embd as u32,
                        (cfg.n_embd as f32).sqrt().to_bits(),
                        0,
                    ],
                    cfg.n_ff,
                )?;
                self.norm(commands, l.up, 1, cfg.n_ff)?;
                self.project(commands, down, l.up, l.projection)?;
                *vectors
            }
        };
        self.norm(commands, l.x, 1, cfg.n_embd)?;
        self.norm(commands, l.projection, 1, cfg.n_embd)?;
        self.graph(
            commands,
            Some(vectors),
            [
                11,
                Self::word(l.x),
                Self::word(l.projection),
                0,
                cfg.n_embd as u32,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            cfg.n_embd,
        )?;
        self.norm(commands, l.x, 1, cfg.n_embd)
    }

    pub(crate) fn prefill(
        &mut self,
        model: &DoryModel,
        tokens: &[u32],
        position: usize,
        kv: KvFormat,
    ) -> Result<Vec<f32>> {
        if self.failed {
            return Err(VulkanError::UnsupportedShape(
                "Dory GPU session requires reset after a failed submission".into(),
            ));
        }
        let result = self.run(model, tokens, position, kv);
        if result.is_err() {
            self.failed = true;
        }
        result
    }

    fn run(
        &self,
        model: &DoryModel,
        tokens: &[u32],
        position: usize,
        kv: KvFormat,
    ) -> Result<Vec<f32>> {
        let cfg = &model.config;
        let l = &self.layout;
        let mut embedding = vec![0.0; cfg.n_embd];
        for (step, &token) in tokens.iter().enumerate() {
            let pos = position + step;
            model
                .token_embedding
                .embedding_lookup(token, &mut embedding);
            self.ops.write_f32(l.x, &embedding)?;
            crate::models::dory::trace("embd", None, pos, &embedding);
            for (r, base) in [
                (l.rope, cfg.rope_freq_base),
                (self.rope_sliding, cfg.rope_freq_base_2),
            ] {
                let mut coefficients = vec![0.0; cfg.head_dim];
                coefficients[..cfg.head_dim / 2].fill(1.0);
                crate::ops::rope_neox_inplace_precise(&mut coefficients, pos, cfg.head_dim, base);
                self.ops.write_f32(r, &coefficients)?;
            }
            let mut commands = TokenCommands::begin(self.context)?;
            for layer in 0..cfg.n_input_layers {
                self.layer(&commands, cfg, layer, 0, pos, kv)?;
                commands = self.checkpoint(commands, layer, pos)?;
            }
            self.copy_add(&commands, l.x, None, self.input_h, cfg.n_embd)?;
            self.copy_add(&commands, l.x, None, self.state, cfg.n_embd)?;
            for loop_idx in 0..cfg.n_loops {
                self.copy_add(&commands, self.state, Some(self.input_h), l.x, cfg.n_embd)?;
                self.norm(&commands, l.x, 1, cfg.n_embd)?;
                for layer in cfg.n_input_layers..cfg.n_input_layers + cfg.n_recurrent_layers {
                    self.layer(&commands, cfg, layer, loop_idx, pos, kv)?;
                    if layer + 1 == cfg.n_input_layers + cfg.n_recurrent_layers {
                        self.norm(&commands, l.x, 1, cfg.n_embd)?;
                    }
                    commands =
                        self.checkpoint(commands, layer + loop_idx * cfg.n_recurrent_layers, pos)?;
                }
                self.copy_add(&commands, l.x, None, self.state, cfg.n_embd)?;
            }
            for layer in cfg.n_input_layers + cfg.n_recurrent_layers..cfg.n_layer {
                self.layer(&commands, cfg, layer, 0, pos, kv)?;
                commands = self.checkpoint(
                    commands,
                    layer + (cfg.n_loops - 1) * cfg.n_recurrent_layers,
                    pos,
                )?;
            }
            commands.submit_and_wait()?;
        }
        self.trace_region(
            "result_norm",
            None,
            position + tokens.len() - 1,
            l.x,
            cfg.n_embd,
        )?;
        let commands = TokenCommands::begin(self.context)?;
        self.project(&commands, &self.output, l.x, l.logits)?;
        self.graph(
            &commands,
            Some(self.output_scale),
            [
                12,
                Self::word(l.logits),
                0,
                0,
                cfg.vocab_size as u32,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            cfg.vocab_size,
        )?;
        commands.submit_and_wait()?;
        let mut logits = vec![0.0; cfg.vocab_size];
        self.ops.read_f32_into(l.logits, &mut logits)?;
        if logits.iter().any(|v| !v.is_finite()) {
            return Err(VulkanError::UnsupportedShape(
                "non-finite Dory Vulkan logits".into(),
            ));
        }
        crate::models::dory::trace("result_output", None, position + tokens.len() - 1, &logits);
        Ok(logits)
    }

    fn checkpoint<'a>(
        &self,
        commands: TokenCommands<'a>,
        layer: usize,
        position: usize,
    ) -> Result<TokenCommands<'a>> {
        // Bound recordings even without tracing; long command buffers can be truncated by drivers.
        if Self::tracing("l_out") || layer % 4 == 3 {
            commands.submit_and_wait()?;
            self.trace_region(
                "l_out",
                Some(layer),
                position,
                self.layout.x,
                self.input_h.size / 4,
            )?;
            TokenCommands::begin(self.context)
        } else {
            Ok(commands)
        }
    }

    fn tracing(name: &str) -> bool {
        #[cfg(feature = "parity-trace")]
        {
            crate::parity_trace::enabled(name)
        }
        #[cfg(not(feature = "parity-trace"))]
        {
            let _ = name;
            false
        }
    }
    fn trace_region(
        &self,
        name: &str,
        layer: Option<usize>,
        position: usize,
        r: ArenaRegion,
        n: usize,
    ) -> Result<()> {
        if Self::tracing(name) {
            let mut v = vec![0.0; n];
            self.ops.read_f32_into(r, &mut v)?;
            crate::models::dory::trace(name, layer, position, &v);
        }
        Ok(())
    }
}
