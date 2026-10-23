//! Resident Dory graph: projections, loop state and KV stay on the device.
use super::dense::UploadedBuffers;
use super::ops::{ArenaRegion, OperatorBindings, Qwen3Ops, TokenCommands};
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_kv_layout_uses_half_the_cache_bytes() {
        let mut cfg = DoryConfig {
            n_embd: 256,
            n_layer: 6,
            n_head: 2,
            n_head_kv: 1,
            head_dim: 256,
            n_ff: 512,
            n_ctx: 4096,
            vocab_size: 256,
            rope_freq_base: 10000.0,
            rope_freq_base_2: 1000.0,
            sliding_window: 1024,
            n_input_layers: 2,
            n_recurrent_layers: 2,
            n_output_layers: 2,
            n_loops: 4,
            recurrent_kv_cache_mode: "per_loop".into(),
            initializer_range: 0.0177,
            alpha_init: 0.05,
            gate_init: 2.0,
            rope_profile: vec![2, 1, 1, 1, 2, 1],
            attention_sliding: vec![true, false, true],
        };
        let f32 = DoryLayout::new(&cfg, 2048, KvFormat::F32).unwrap();
        let f16 = DoryLayout::new(&cfg, 2048, KvFormat::F16).unwrap();
        let bytes = |l: &DoryLayout| {
            l.caches
                .iter()
                .map(|c| c.keys.size + c.values.size)
                .sum::<usize>()
        };
        assert_eq!(bytes(&f32), bytes(&f16) * 2);
        assert_eq!(f32.bytes - f16.bytes, bytes(&f16));
        for c in &f16.caches {
            assert!(c.slots <= 2048);
        }
        assert_eq!(f32.caches.len(), 6);
        cfg.recurrent_kv_cache_mode = "last_loop".into();
        let last = DoryLayout::new(&cfg, 2048, KvFormat::F16).unwrap();
        assert_eq!(last.caches.len(), 3);
        assert!(bytes(&last) < bytes(&f16));
        assert!(DoryLayout::new(&cfg, usize::MAX, KvFormat::F16).is_err());
    }

    #[test]
    #[ignore = "requires a hardware Vulkan device with shaderFloat64"]
    fn cooperative_quantization_matches_signed_q8k() {
        crate::ops::enable_gpu();
        let context = crate::ops::get_vulkan_context().expect("Vulkan context");
        assert!(!context.is_software_icd() && context.shader_float64);
        for width in [256, 512, 2560, 7168] {
            let input = ArenaRegion {
                offset: 0,
                size: width * 4,
            };
            let q8 = ArenaRegion {
                offset: input.size,
                size: width,
            };
            let scales = ArenaRegion {
                offset: q8.offset + q8.size,
                size: width / 256 * 4,
            };
            let mut ops =
                Qwen3Ops::new_device_local_with_size(context, scales.offset + scales.size, 1)
                    .unwrap();
            let pipeline = ops
                .add_shader(include_bytes!("../../shaders/bin/dory.spv"))
                .unwrap();
            for pattern in 0..4 {
                let mut values: Vec<f32> = (0..width)
                    .map(|i| {
                        if pattern == 0 {
                            0.0
                        } else {
                            ((i * 73 % 251) as f32 - 125.0) * 0.137
                        }
                    })
                    .collect();
                if pattern == 1 || pattern == 2 {
                    for block in values.chunks_exact_mut(256) {
                        let maximum = if pattern == 1 { 127.0 } else { -127.0 };
                        block[3] = maximum;
                        block[252] = -maximum;
                        block[8..14].copy_from_slice(&[
                            0.5,
                            -0.5,
                            1.5,
                            -1.5,
                            f32::from_bits(0x3effffff),
                            -0.0,
                        ]);
                    }
                }
                ops.write_f32(input, &values).unwrap();
                let mut expected = Vec::new();
                for cooperative in [false, true] {
                    let commands = TokenCommands::begin(context).unwrap();
                    if cooperative {
                        let mut push = [0; 17];
                        push[..4].copy_from_slice(&[
                            13,
                            0,
                            (q8.offset / 4) as u32,
                            (scales.offset / 4) as u32,
                        ]);
                        ops.record_custom(
                            &commands,
                            pipeline,
                            None,
                            &push,
                            [(width / 256) as u32, 1, 1],
                        )
                        .unwrap();
                    } else {
                        ops.record_quantize_q8_k_ggml(&commands, input, q8, scales, width)
                            .unwrap();
                    }
                    commands.submit_and_wait().unwrap();
                    let mut actual = Vec::new();
                    for region in [q8, scales] {
                        let mut data = vec![0.0; region.size / 4];
                        ops.read_f32_into(region, &mut data).unwrap();
                        actual.extend(data.into_iter().map(f32::to_bits));
                    }
                    if cooperative {
                        assert_eq!(actual, expected, "width={width} pattern={pattern}");
                    } else {
                        expected = actual;
                    }
                }
            }
        }
    }
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

struct CacheRegion {
    keys: ArenaRegion,
    values: ArenaRegion,
    slots: usize,
}

struct DoryLayout {
    x: ArenaRegion,
    q: ArenaRegion,
    k: ArenaRegion,
    v: ArenaRegion,
    attn: ArenaRegion,
    projection: ArenaRegion,
    gate: ArenaRegion,
    up: ArenaRegion,
    logits: ArenaRegion,
    q8k: ArenaRegion,
    q8k_scales: ArenaRegion,
    scores: ArenaRegion,
    rope: ArenaRegion,
    rope_sliding: ArenaRegion,
    input_h: ArenaRegion,
    state: ArenaRegion,
    caches: Vec<CacheRegion>,
    slots: Vec<Option<usize>>,
    bytes: usize,
    rows: usize,
}

impl DoryLayout {
    fn new(cfg: &DoryConfig, capacity: usize, kv: KvFormat) -> Result<Self> {
        let rows = capacity.min(crate::models::dory::PREFILL_BATCH);
        let q = cfg
            .n_head
            .checked_mul(cfg.head_dim)
            .ok_or(VulkanError::OutOfMemory)?;
        let k = cfg
            .n_head_kv
            .checked_mul(cfg.head_dim)
            .ok_or(VulkanError::OutOfMemory)?;
        if rows == 0 || k % 2 != 0 {
            return Err(VulkanError::UnsupportedShape(
                "invalid Dory KV dimensions".into(),
            ));
        }
        let max_in = cfg.n_ff.max(q).max(cfg.n_embd);
        let mut bytes = 0usize;
        let mut alloc = |dimensions: &[usize], element_bytes: usize| -> Result<ArenaRegion> {
            let elements = dimensions
                .iter()
                .try_fold(1usize, |n, &d| n.checked_mul(d))
                .ok_or(VulkanError::OutOfMemory)?;
            let offset = bytes.checked_add(15).ok_or(VulkanError::OutOfMemory)? & !15;
            let size = elements
                .checked_mul(element_bytes)
                .ok_or(VulkanError::OutOfMemory)?;
            bytes = offset.checked_add(size).ok_or(VulkanError::OutOfMemory)?;
            Ok(ArenaRegion { offset, size })
        };
        let x = alloc(&[rows, cfg.n_embd], 4)?;
        let q = alloc(&[rows, q], 4)?;
        let k_region = alloc(&[rows, k], 4)?;
        let v = alloc(&[rows, k], 4)?;
        let attn = alloc(&[rows, cfg.n_head, cfg.head_dim], 4)?;
        let projection = alloc(&[rows, cfg.n_embd], 4)?;
        let gate = alloc(&[rows, max_in], 4)?;
        let up = alloc(&[rows, cfg.n_ff], 4)?;
        let logits = alloc(&[cfg.vocab_size], 4)?;
        let q8k = alloc(&[rows, max_in], 1)?;
        let q8k_scales = alloc(&[rows, max_in.div_ceil(256)], 4)?;
        let scores = alloc(&[cfg.n_head, capacity], 4)?;
        let rope = alloc(&[rows, cfg.head_dim], 4)?;
        let rope_sliding = alloc(&[rows, cfg.head_dim], 4)?;
        let input_h = alloc(&[rows, cfg.n_embd], 4)?;
        let state = alloc(&[rows, cfg.n_embd], 4)?;
        let mut caches = Vec::new();
        let mut slots = vec![None; cfg.n_layer];
        let element_bytes = if matches!(kv, KvFormat::F16) { 2 } else { 4 };
        for (layer, slot) in slots.iter_mut().enumerate() {
            if !cfg.is_attention(layer) {
                continue;
            }
            *slot = Some(caches.len());
            let cache_slots = if cfg.attention_is_sliding(layer) {
                capacity.min(cfg.sliding_window.saturating_add(rows - 1))
            } else {
                capacity
            };
            let elements = cache_slots.checked_mul(k).ok_or(VulkanError::OutOfMemory)?;
            for _ in 0..if cfg.separate_loop_cache(layer) {
                cfg.n_loops
            } else {
                1
            } {
                caches.push(CacheRegion {
                    keys: alloc(&[elements], element_bytes)?,
                    values: alloc(&[elements], element_bytes)?,
                    slots: cache_slots,
                });
            }
        }
        Ok(Self {
            x,
            q,
            k: k_region,
            v,
            attn,
            projection,
            gate,
            up,
            logits,
            q8k,
            q8k_scales,
            scores,
            rope,
            rope_sliding,
            input_h,
            state,
            caches,
            slots,
            bytes,
            rows,
        })
    }

    fn row(region: ArenaRegion, row: usize, width: usize) -> ArenaRegion {
        ArenaRegion {
            offset: region.offset + row * width * 4,
            size: width * 4,
        }
    }
}

pub(crate) struct DoryVulkanSession {
    ops: Qwen3Ops<'static>,
    _buffers: UploadedBuffers,
    context: &'static VulkanContext,
    layout: DoryLayout,
    layers: Vec<Layer>,
    output: Matrix,
    output_scale: OperatorBindings,
    graph_pipeline: usize,
    matmul_pipeline: usize,
    capacity: usize,
    kv_format: KvFormat,
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
    pub(crate) fn kv_cache_bytes(&self) -> usize {
        self.layout
            .caches
            .iter()
            .map(|c| c.keys.size + c.values.size)
            .sum()
    }
    pub(crate) fn new(
        model: &DoryModel,
        context: &'static VulkanContext,
        capacity: usize,
        kv_format: KvFormat,
    ) -> Result<Self> {
        if !context.shader_float64 {
            return Err(VulkanError::UnsupportedShape(
                "Dory Vulkan requires shaderFloat64 for recurrent normalization and attention"
                    .into(),
            ));
        }
        let cfg = &model.config;
        let layout = DoryLayout::new(cfg, capacity, kv_format)?;
        let bytes = layout.bytes;
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
            layers,
            output,
            output_scale,
            graph_pipeline,
            matmul_pipeline,
            capacity,
            kv_format,
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
        self.graph_rows(commands, binding, push, threads, 1, [0; 3])
    }

    fn graph_rows(
        &self,
        commands: &TokenCommands<'_>,
        binding: Option<OperatorBindings>,
        push: [u32; 14],
        threads: usize,
        rows: usize,
        strides: [usize; 3],
    ) -> Result<()> {
        let mut data = [0u32; 17];
        data[..14].copy_from_slice(&push);
        for (v, stride) in data[14..].iter_mut().zip(strides) {
            *v = stride as u32;
        }
        self.ops.record_custom(
            commands,
            self.graph_pipeline,
            binding,
            &data,
            [threads.div_ceil(64) as u32, rows as u32, 1],
        )
    }

    fn norm(
        &self,
        commands: &TokenCommands<'_>,
        region: ArenaRegion,
        heads: usize,
        width: usize,
        rows: usize,
    ) -> Result<()> {
        self.graph_rows(
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
            rows,
            [heads * width, 0, 0],
        )
    }

    fn copy_add(
        &self,
        commands: &TokenCommands<'_>,
        a: ArenaRegion,
        b: Option<ArenaRegion>,
        c: ArenaRegion,
        n: usize,
        rows: usize,
    ) -> Result<()> {
        self.graph_rows(
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
            rows,
            [n, n, n],
        )
    }

    fn project(
        &self,
        commands: &TokenCommands<'_>,
        input: ArenaRegion,
        projections: &[(&Matrix, ArenaRegion)],
        rows: usize,
    ) -> Result<()> {
        let l = &self.layout;
        let q8_stride = l.q8k.size / l.rows;
        let scale_stride = l.q8k_scales.size / l.rows / 4;
        if let Some((w, _)) = projections.iter().find(|(w, _)| w.format != 30) {
            self.graph_rows(
                commands,
                None,
                [
                    13,
                    Self::word(input),
                    Self::word(l.q8k),
                    Self::word(l.q8k_scales),
                    0,
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
                w.n_in / 4,
                rows,
                [w.n_in, q8_stride / 4, scale_stride],
            )?;
        }
        for &(w, output) in projections {
            let stride = if output.offset == l.gate.offset {
                l.gate.size / l.rows / 4
            } else {
                w.n_out
            };
            self.ops.record_custom(
                commands,
                self.matmul_pipeline,
                Some(w.binding),
                &[
                    Self::word(input),
                    Self::word(l.q8k),
                    Self::word(l.q8k_scales),
                    Self::word(output),
                    w.n_in as u32,
                    w.n_out as u32,
                    w.format,
                    w.n_in as u32,
                    (q8_stride / 4) as u32,
                    scale_stride as u32,
                    stride as u32,
                ],
                [w.n_out.div_ceil(8) as u32, rows as u32, 1],
            )?;
        }
        Ok(())
    }

    fn layer(
        &self,
        commands: &TokenCommands<'_>,
        cfg: &DoryConfig,
        layer: usize,
        loop_idx: usize,
        position: usize,
        kv: KvFormat,
        rows: usize,
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
                self.project(
                    commands,
                    l.x,
                    &[(q, l.q), (k, l.k), (v, l.v), (gate, l.gate)],
                    rows,
                )?;
                let rope = if cfg.attention_is_sliding(layer) {
                    l.rope_sliding
                } else {
                    l.rope
                };
                for (r, n, h) in [(l.q, qn, cfg.n_head), (l.k, kn, cfg.n_head_kv)] {
                    self.graph_rows(
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
                        rows,
                        [n, cfg.head_dim, 0],
                    )?;
                    self.norm(commands, r, h, cfg.head_dim, rows)?;
                }
                let half = u32::from(matches!(kv, KvFormat::F16));
                self.graph_rows(
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
                    rows,
                    [qn, 0, 0],
                )?;
                let slot = l.slots[layer].unwrap()
                    + if cfg.separate_loop_cache(layer) {
                        loop_idx
                    } else {
                        0
                    };
                let cache = &l.caches[slot];
                let window = if cfg.attention_is_sliding(layer) {
                    cfg.sliding_window.min(self.capacity)
                } else {
                    self.capacity
                };
                for row in 0..rows {
                    let pos = position + row;
                    let seq = (pos + 1).min(window);
                    let divisor = if half != 0 { 2 } else { 1 };
                    let offset = ((pos % cache.slots) * kn / divisor) as u32;
                    self.graph(
                        commands,
                        None,
                        [
                            5,
                            Self::word(l.k) + (row * kn) as u32,
                            Self::word(l.v) + (row * kn) as u32,
                            Self::word(cache.keys) + offset,
                            kn as u32,
                            0,
                            Self::word(cache.values) + offset,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            half,
                        ],
                        kn / divisor,
                    )?;
                    let mut push = [
                        6,
                        Self::word(l.q) + (row * qn) as u32,
                        Self::word(cache.keys),
                        Self::word(l.scores),
                        seq as u32,
                        cfg.head_dim as u32,
                        cfg.n_head as u32,
                        cfg.n_head_kv as u32,
                        0,
                        cache.slots as u32,
                        ((pos + 1 - seq) % cache.slots) as u32,
                        0,
                        (1.0 / (cfg.head_dim as f32).sqrt()).to_bits(),
                        half,
                    ];
                    self.graph(commands, None, push, seq * cfg.n_head)?;
                    push[0] = 7;
                    push[1] = Self::word(l.scores);
                    self.graph(commands, None, push, cfg.n_head)?;
                    push[0] = 8;
                    push[2] = Self::word(cache.values);
                    push[3] = Self::word(l.attn) + (row * qn) as u32;
                    self.graph(commands, None, push, qn)?;
                }
                self.graph_rows(
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
                    rows,
                    [qn, l.gate.size / l.rows / 4, 0],
                )?;
                self.norm(commands, l.attn, 1, qn, rows)?;
                self.project(commands, l.attn, &[(out, l.projection)], rows)?;
                *vectors
            }
            Layer::Mlp {
                vectors,
                gate,
                up,
                down,
            } => {
                self.project(commands, l.x, &[(gate, l.gate), (up, l.up)], rows)?;
                self.graph_rows(
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
                    rows,
                    [l.gate.size / l.rows / 4, cfg.n_ff, 0],
                )?;
                self.norm(commands, l.up, 1, cfg.n_ff, rows)?;
                self.project(commands, l.up, &[(down, l.projection)], rows)?;
                *vectors
            }
        };
        self.norm(commands, l.x, 1, cfg.n_embd, rows)?;
        self.norm(commands, l.projection, 1, cfg.n_embd, rows)?;
        self.graph_rows(
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
            rows,
            [cfg.n_embd, cfg.n_embd, 0],
        )?;
        self.norm(commands, l.x, 1, cfg.n_embd, rows)
    }

    pub(crate) fn prefill(
        &mut self,
        model: &DoryModel,
        tokens: &[u32],
        position: usize,
        kv: KvFormat,
    ) -> Result<Vec<f32>> {
        if kv != self.kv_format {
            return Err(VulkanError::UnsupportedShape(
                "Dory KV storage format requires reset".into(),
            ));
        }
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
        for (chunk_idx, chunk) in tokens.chunks(l.rows).enumerate() {
            let pos = position + chunk_idx * l.rows;
            let rows = chunk.len();
            let mut embedding = vec![0.0; rows * cfg.n_embd];
            for (row, (&token, h)) in chunk
                .iter()
                .zip(embedding.chunks_exact_mut(cfg.n_embd))
                .enumerate()
            {
                model.token_embedding.embedding_lookup(token, h);
                crate::models::dory::trace("embd", None, pos + row, h);
            }
            self.ops
                .write_f32(DoryLayout::row(l.x, 0, rows * cfg.n_embd), &embedding)?;
            for (r, base) in [
                (l.rope, cfg.rope_freq_base),
                (l.rope_sliding, cfg.rope_freq_base_2),
            ] {
                let mut coefficients = vec![0.0; rows * cfg.head_dim];
                for (row, c) in coefficients.chunks_exact_mut(cfg.head_dim).enumerate() {
                    c[..cfg.head_dim / 2].fill(1.0);
                    crate::ops::rope_neox_inplace_precise(c, pos + row, cfg.head_dim, base);
                }
                self.ops
                    .write_f32(DoryLayout::row(r, 0, rows * cfg.head_dim), &coefficients)?;
            }
            let mut commands = TokenCommands::begin(self.context)?;
            for layer in 0..cfg.n_input_layers {
                self.layer(&commands, cfg, layer, 0, pos, kv, rows)?;
                commands = self.checkpoint(commands, layer, pos, rows)?;
            }
            self.copy_add(&commands, l.x, None, l.input_h, cfg.n_embd, rows)?;
            self.copy_add(&commands, l.x, None, l.state, cfg.n_embd, rows)?;
            for loop_idx in 0..cfg.n_loops {
                self.copy_add(&commands, l.state, Some(l.input_h), l.x, cfg.n_embd, rows)?;
                self.norm(&commands, l.x, 1, cfg.n_embd, rows)?;
                for layer in cfg.n_input_layers..cfg.n_input_layers + cfg.n_recurrent_layers {
                    self.layer(&commands, cfg, layer, loop_idx, pos, kv, rows)?;
                    if layer + 1 == cfg.n_input_layers + cfg.n_recurrent_layers {
                        self.norm(&commands, l.x, 1, cfg.n_embd, rows)?;
                    }
                    commands = self.checkpoint(
                        commands,
                        layer + loop_idx * cfg.n_recurrent_layers,
                        pos,
                        rows,
                    )?;
                }
                self.copy_add(&commands, l.x, None, l.state, cfg.n_embd, rows)?;
            }
            for layer in cfg.n_input_layers + cfg.n_recurrent_layers..cfg.n_layer {
                self.layer(&commands, cfg, layer, 0, pos, kv, rows)?;
                commands = self.checkpoint(
                    commands,
                    layer + (cfg.n_loops - 1) * cfg.n_recurrent_layers,
                    pos,
                    rows,
                )?;
            }
            commands.submit_and_wait()?;
        }
        let final_row = (tokens.len() - 1) % l.rows;
        let final_hidden = DoryLayout::row(l.x, final_row, cfg.n_embd);
        self.trace_region(
            "result_norm",
            None,
            position + tokens.len() - 1,
            final_hidden,
            cfg.n_embd,
        )?;
        let commands = TokenCommands::begin(self.context)?;
        self.project(&commands, final_hidden, &[(&self.output, l.logits)], 1)?;
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
        rows: usize,
    ) -> Result<TokenCommands<'a>> {
        // Bound recordings even without tracing; long command buffers can be truncated by drivers.
        let trace = Self::tracing_step("l_out", position + rows - 1);
        if trace || layer % 4 == 3 {
            commands.submit_and_wait()?;
            if trace {
                let width = self.layout.x.size / self.layout.rows / 4;
                let mut values = vec![0.0; rows * width];
                self.ops
                    .read_f32_into(DoryLayout::row(self.layout.x, 0, values.len()), &mut values)?;
                for (row, h) in values.chunks_exact(width).enumerate() {
                    crate::models::dory::trace("l_out", Some(layer), position + row, h);
                }
            }
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
    fn tracing_step(name: &str, step: usize) -> bool {
        Self::tracing(name)
            && step
                >= std::env::var("RMI_DORY_TRACE_FROM")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(0)
    }
    fn trace_region(
        &self,
        name: &str,
        layer: Option<usize>,
        position: usize,
        r: ArenaRegion,
        n: usize,
    ) -> Result<()> {
        if Self::tracing_step(name, position) {
            let mut v = vec![0.0; n];
            self.ops.read_f32_into(r, &mut v)?;
            crate::models::dory::trace(name, layer, position, &v);
        }
        Ok(())
    }
}
