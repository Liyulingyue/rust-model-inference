//! Resident 2+8+2 hourglass. A denoising pass records all blocks in one command
//! buffer; both classifier-free guidance branches can feed the sampler in that same submission.
use super::{
    color_features, Block, Projection, Rope, Weights, D, FF, GRID, HEADS, HEAD_DIM, PIXELS, PREFIX,
    VOCAB,
};
use crate::models::tessera::{
    device::{word, Graph, Layout, Matrix},
    sampler::{Draw, DrawOptions, PixelOptions},
};
use crate::ops::nn::AttentionSpec;
use crate::vulkan::{
    ops::{ArenaRegion as Region, TokenCommands},
    VulkanContext, VulkanError,
};

type Result<T> = std::result::Result<T, VulkanError>;
const NONE: u32 = u32::MAX;
const ROWS: usize = PREFIX + PIXELS;

#[derive(Default)]
struct Constants(Vec<f32>);
impl Constants {
    fn store(&mut self, values: &[f32]) -> Region {
        let offset = self.0.len() * 4;
        self.0.extend_from_slice(values);
        Region {
            offset,
            size: values.len() * 4,
        }
    }
    fn rope(&mut self, grid: usize, stride: usize) -> Region {
        let mut table = Vec::with_capacity((PREFIX + grid * grid) * HEAD_DIM);
        for _ in 0..PREFIX * HEAD_DIM / 2 {
            table.extend([0., 1.]);
        }
        for (sin, cos) in Rope::new(grid, stride).angles {
            table.extend([sin, cos]);
        }
        self.store(&table)
    }
}

struct DeviceBlock {
    qkv: Projection,
    proj: Projection,
    w12: Projection,
    w3: Projection,
    q_norm: Region,
    k_norm: Region,
    kernel: Region,
    bias: Region,
    mod_bias: Region,
}
impl DeviceBlock {
    fn load(src: &Block, c: &mut Constants) -> Self {
        Self {
            qkv: src.qkv,
            proj: src.proj,
            w12: src.w12,
            w3: src.w3,
            q_norm: c.store(&src.q_norm),
            k_norm: c.store(&src.k_norm),
            kernel: c.store(&src.kernel),
            bias: c.store(&src.bias),
            mod_bias: c.store(&src.mod_bias),
        }
    }
}

// Once a binding is recorded, all later operators refer to arena offsets.
// The prefix occupies the first 28 rows at both spatial resolutions.
struct Slots {
    ids: Region,
    text: Region,
    pool: Region,
    sc: Region,
    ratio: Region,
    t: Region,
    g: Region,
    cond: Region,
    shared_mod: Region,
    sc_hidden: Region,
    x: Region,
    core: Region,
    merged: Region,
    norm: Region,
    tmp: Region,
    qkv: Region,
    q: Region,
    k: Region,
    v: Region,
    fused: Region,
    hidden: Region,
    logits: Region,
    uncond: Region,
    noise: Region,
    probs: Region,
    triplets: Region,
    commit_ids: Region,
    confidence: Region,
    commit_conf: Region,
    gumbels: Region,
}
impl Slots {
    fn new(l: &mut Layout) -> Self {
        Self {
            ids: l.floats(PIXELS),
            text: l.floats(24 * D),
            pool: l.floats(D),
            sc: l.floats(PIXELS * VOCAB),
            ratio: l.floats(16),
            t: l.floats(24 * D),
            g: l.floats(D),
            cond: l.floats(D),
            shared_mod: l.floats(6 * D),
            sc_hidden: l.floats(PIXELS * D),
            x: l.floats(ROWS * D),
            core: l.floats((PREFIX + 64) * D),
            merged: l.floats(PIXELS * D),
            norm: l.floats(VOCAB * D),
            tmp: l.floats(VOCAB * D),
            qkv: l.floats(ROWS * 3 * D),
            q: l.floats(ROWS * D),
            k: l.floats(ROWS * D),
            v: l.floats(ROWS * D),
            fused: l.floats(ROWS * 2 * FF),
            hidden: l.floats(ROWS * FF),
            logits: l.floats(PIXELS * VOCAB),
            uncond: l.floats(PIXELS * VOCAB),
            noise: l.floats(PIXELS * VOCAB),
            probs: l.floats(PIXELS * VOCAB),
            triplets: l.floats(PIXELS * 3),
            commit_ids: l.floats(PIXELS),
            confidence: l.floats(PIXELS),
            commit_conf: l.floats(PIXELS),
            gumbels: l.floats(PIXELS),
        }
    }
}

pub(super) struct Device<'m, 'w> {
    graph: Graph,
    weights: &'m Weights<'w>,
    matrices: Vec<Matrix>,
    biases: Vec<Option<Region>>,
    stages: [Vec<DeviceBlock>; 3],
    embedding: Region,
    positions: Region,
    registers: Region,
    null_text: Region,
    null_cond: Region,
    text_norm: Region,
    pool_norm: Region,
    down_norm: Region,
    pixel_rope: Region,
    patch_rope: Region,
    s: Slots,
    sample_shader: usize,
    commit_shader: usize,
}

fn prefix_end(region: Region) -> Region {
    region.shifted(PREFIX * D * 4)
}
fn opt(r: Option<Region>) -> u32 {
    r.map(word).unwrap_or(NONE)
}
fn groups(n: usize) -> [u32; 3] {
    [n.div_ceil(64) as u32, 1, 1]
}

impl<'m, 'w> Device<'m, 'w> {
    pub(super) fn new(context: &'static VulkanContext, w: &'m Weights<'w>) -> Result<Self> {
        let mut constants = Constants::default();
        let embeddings = constants.store(&w.embedding);
        let positions = constants.store(&w.positions);
        let registers = constants.store(&w.registers);
        let null_text = constants.store(&w.null_text);
        let null_cond = constants.store(&w.null_cond);
        let text_norm = constants.store(&w.text_norm);
        let pool_norm = constants.store(&w.pool_norm);
        let down_norm = constants.store(&w.down_norm);
        let pixel_rope = constants.rope(GRID, 1);
        let patch_rope = constants.rope(GRID / 2, 2);
        let stages = w.stages.each_ref().map(|blocks| {
            blocks
                .iter()
                .map(|b| DeviceBlock::load(b, &mut constants))
                .collect()
        });
        let colors = constants.store(&color_features(&w.lab));
        let biases = w
            .linears
            .iter()
            .map(|m| m.bias().map(|v| constants.store(v)))
            .collect();
        let mut layout = Layout::default();
        let static_data = layout.floats(constants.0.len());
        let s = Slots::new(&mut layout);
        let mut graph = Graph::new(
            context,
            layout,
            VOCAB,
            4 * D,
            w.linears.len(),
            include_bytes!("../../../../shaders/bin/tessera_block.spv"),
        )?;
        graph.ops.write_f32(static_data, &constants.0)?;
        let matrices = w
            .linears
            .iter()
            .map(|lin| graph.bind(lin))
            .collect::<Result<_>>()?;
        let sample_shader = graph
            .ops
            .add_shader(include_bytes!("../../../../shaders/bin/tessera_sample.spv"))?;
        let commit_shader = graph
            .ops
            .add_shader(include_bytes!("../../../../shaders/bin/tessera_commit.spv"))?;
        let device = Self {
            graph,
            weights: w,
            matrices,
            biases,
            stages,
            embedding: embeddings,
            positions,
            registers,
            null_text,
            null_cond,
            text_norm,
            pool_norm,
            down_norm,
            pixel_rope,
            patch_rope,
            s,
            sample_shader,
            commit_shader,
        };
        let cmd = TokenCommands::begin(context)?;
        device.mlp(
            &cmd,
            w.color_mlp,
            colors,
            device.s.norm,
            device.s.tmp,
            VOCAB,
        )?;
        device.graph.custom(
            &cmd,
            &[
                8,
                word(device.embedding),
                word(device.s.tmp),
                (VOCAB * D) as u32,
            ],
            groups(VOCAB * D),
        )?;
        cmd.submit_and_wait()?;
        Ok(device)
    }
    #[cfg(test)]
    pub(super) fn counters(&self) -> (crate::vulkan::ComputeStats, usize) {
        (
            self.graph.ops.context().compute_stats(),
            self.graph.ops.recorded_dispatch_count(),
        )
    }
    pub(super) fn counts(&self) -> (usize, usize) {
        (0, self.graph.projection_count.get())
    }

    fn project(
        &self,
        cmd: &TokenCommands<'_>,
        slot: Projection,
        input: Region,
        out: Region,
        rows: usize,
    ) -> Result<()> {
        self.graph
            .project(cmd, &self.matrices[slot.0], input, out, rows)
    }
    fn activate(
        &self,
        cmd: &TokenCommands<'_>,
        slot: Projection,
        out: Region,
        extra: Option<Region>,
        rows: usize,
        silu: bool,
    ) -> Result<()> {
        let width = self.weights.linears[slot.0].n_out();
        self.graph.custom(
            cmd,
            &[
                0,
                word(out),
                opt(self.biases[slot.0]),
                opt(extra),
                width as u32,
                silu as u32,
                (rows * width) as u32,
            ],
            groups(rows * width),
        )
    }
    fn linear_bias(
        &self,
        cmd: &TokenCommands<'_>,
        slot: Projection,
        x: Region,
        y: Region,
        rows: usize,
    ) -> Result<()> {
        self.project(cmd, slot, x, y, rows)?;
        self.activate(cmd, slot, y, None, rows, false)
    }
    fn mlp(
        &self,
        cmd: &TokenCommands<'_>,
        layers: [Projection; 2],
        input: Region,
        tmp: Region,
        out: Region,
        rows: usize,
    ) -> Result<()> {
        self.project(cmd, layers[0], input, tmp, rows)?;
        self.activate(cmd, layers[0], tmp, None, rows, true)?;
        self.linear_bias(cmd, layers[1], tmp, out, rows)
    }
    #[allow(clippy::too_many_arguments)]
    fn norm(
        &self,
        cmd: &TokenCommands<'_>,
        x: Region,
        y: Region,
        width: usize,
        rows: usize,
        gamma: Option<Region>,
        mods: Option<Region>,
        offset: Option<Region>,
    ) -> Result<()> {
        self.graph.custom(
            cmd,
            &[
                1,
                word(x),
                word(y),
                width as u32,
                opt(gamma),
                opt(mods),
                opt(offset),
            ],
            [rows as u32, 1, 1],
        )
    }
    fn gate(
        &self,
        cmd: &TokenCommands<'_>,
        x: Region,
        delta: Region,
        offset: Region,
        index: usize,
        rows: usize,
    ) -> Result<()> {
        self.graph.custom(
            cmd,
            &[
                3,
                word(x),
                word(delta),
                word(self.s.shared_mod.shifted(index * D * 4)),
                word(offset.shifted(index * D * 4)),
                (rows * D) as u32,
            ],
            groups(rows * D),
        )
    }
    fn block(
        &self,
        cmd: &TokenCommands<'_>,
        x: Region,
        b: &DeviceBlock,
        grid: usize,
        rope: Region,
    ) -> Result<()> {
        let s = &self.s;
        let rows = PREFIX + grid * grid;
        self.norm(
            cmd,
            x,
            s.norm,
            D,
            rows,
            None,
            Some(s.shared_mod),
            Some(b.mod_bias),
        )?;
        self.project(cmd, b.qkv, s.norm, s.qkv, rows)?;
        self.graph.custom(
            cmd,
            &[
                2,
                word(s.qkv),
                word(s.q),
                word(s.k),
                word(s.v),
                word(b.q_norm),
                word(b.k_norm),
                word(rope),
            ],
            [HEADS as u32, rows as u32, 1],
        )?;
        self.graph.ops.record_online_attention(
            cmd,
            s.q,
            s.k,
            s.v,
            s.tmp,
            AttentionSpec {
                query_tokens: rows,
                key_tokens: rows,
                query_heads: HEADS,
                key_value_heads: HEADS,
                head_dim: HEAD_DIM,
                scale: 1. / (HEAD_DIM as f32).sqrt(),
                causal: false,
            },
            0,
            0,
        )?;
        self.project(cmd, b.proj, s.tmp, s.norm, rows)?;
        self.gate(cmd, x, s.norm, b.mod_bias, 2, rows)?;
        self.norm(
            cmd,
            x,
            s.norm,
            D,
            rows,
            None,
            Some(s.shared_mod.shifted(3 * D * 4)),
            Some(b.mod_bias.shifted(3 * D * 4)),
        )?;
        self.project(cmd, b.w12, s.norm, s.fused, rows)?;
        self.graph.custom(
            cmd,
            &[
                4,
                word(s.fused),
                word(s.hidden),
                word(b.kernel),
                word(b.bias),
                grid as u32,
                (rows * FF) as u32,
            ],
            groups(rows * FF),
        )?;
        self.project(cmd, b.w3, s.hidden, s.tmp, rows)?;
        self.gate(cmd, x, s.tmp, b.mod_bias, 5, rows)
    }

    // Inputs are written before beginning commands, including on discrete GPUs where
    // the device-local arena performs a staging transfer. All layer intermediates stay resident.
    fn upload(
        &self,
        ids: &[u16],
        states: &[f32],
        pooled: &[f32],
        sc: Option<&[f32]>,
    ) -> Result<()> {
        let s = &self.s;
        let ops = &self.graph.ops;
        let canvas: Vec<_> = ids
            .iter()
            .map(|&id| f32::from_bits(u32::from(id)))
            .collect();
        ops.write_f32(s.ids, &canvas)?;
        ops.write_f32(s.text, states)?;
        ops.write_f32(s.pool, pooled)?;
        if let Some(sc) = sc {
            ops.write_f32(s.sc, sc)?;
        }
        Ok(())
    }

    fn record(
        &self,
        cmd: &TokenCommands<'_>,
        has_sc: bool,
        unconditional: bool,
        logits: Region,
    ) -> Result<()> {
        let s = &self.s;
        let w = self.weights;
        self.graph
            .custom(cmd, &[9, word(s.ids), word(s.ratio)], [1, 1, 1])?;
        let (text, global) = if unconditional {
            (self.null_text, self.null_cond)
        } else {
            self.norm(cmd, s.text, s.norm, D, 24, Some(self.text_norm), None, None)?;
            self.linear_bias(cmd, w.text_proj, s.norm, s.t, 24)?;
            self.norm(cmd, s.pool, s.norm, D, 1, Some(self.pool_norm), None, None)?;
            self.mlp(cmd, w.pool_mlp, s.norm, s.tmp, s.cond, 1)?;
            (s.t, s.cond)
        };
        if has_sc {
            self.project(cmd, w.self_cond, s.sc, s.sc_hidden, PIXELS)?;
        }
        self.graph.custom(
            cmd,
            &[
                5,
                word(s.x),
                word(s.ids),
                word(self.embedding),
                word(self.positions),
                word(self.registers),
                word(text),
                opt(has_sc.then_some(s.sc_hidden)),
            ],
            groups(ROWS * D),
        )?;
        self.project(cmd, w.ratio_mlp[0], s.ratio, s.tmp, 1)?;
        self.activate(cmd, w.ratio_mlp[0], s.tmp, None, 1, true)?;
        self.project(cmd, w.ratio_mlp[1], s.tmp, s.g, 1)?;
        self.activate(cmd, w.ratio_mlp[1], s.g, Some(global), 1, true)?;
        self.linear_bias(cmd, w.modulation, s.g, s.shared_mod, 1)?;
        for b in &self.stages[0] {
            self.block(cmd, s.x, b, GRID, self.pixel_rope)?;
        }
        self.graph.custom(
            cmd,
            &[6, word(s.x), word(s.core), word(s.merged)],
            groups(ROWS * D),
        )?;
        self.norm(
            cmd,
            s.merged,
            s.merged,
            4 * D,
            64,
            Some(self.down_norm),
            None,
            None,
        )?;
        self.linear_bias(cmd, w.down, s.merged, prefix_end(s.core), 64)?;
        for b in &self.stages[1] {
            self.block(cmd, s.core, b, GRID / 2, self.patch_rope)?;
        }
        self.linear_bias(cmd, w.up, prefix_end(s.core), s.merged, 64)?;
        self.graph.custom(
            cmd,
            &[7, word(s.core), word(s.x), word(s.merged)],
            groups(ROWS * D),
        )?;
        for b in &self.stages[2] {
            self.block(cmd, s.x, b, GRID, self.pixel_rope)?;
        }
        self.linear_bias(cmd, w.final_mod, s.g, s.tmp, 1)?;
        self.norm(
            cmd,
            prefix_end(s.x),
            s.norm,
            D,
            PIXELS,
            None,
            Some(s.tmp),
            None,
        )?;
        self.linear_bias(cmd, w.head, s.norm, logits, PIXELS)
    }

    #[cfg(test)]
    pub(super) fn forward(
        &self,
        ids: &[u16],
        states: &[f32],
        pooled: &[f32],
        sc: Option<&[f32]>,
        unconditional: bool,
    ) -> Result<Vec<f32>> {
        self.upload(ids, states, pooled, sc)?;
        let cmd = TokenCommands::begin(self.graph.ops.context())?;
        self.record(&cmd, sc.is_some(), unconditional, self.s.logits)?;
        cmd.submit_and_wait()?;
        self.graph.read(self.s.logits, PIXELS * VOCAB)
    }

    pub(super) fn draw(
        &self,
        ids: &[u16],
        states: &[f32],
        pooled: &[f32],
        sc: Option<&[f32]>,
        opts: &DrawOptions<'_>,
        noise: &[f32],
    ) -> Result<Draw> {
        let s = &self.s;
        let ops = &self.graph.ops;
        self.upload(ids, states, pooled, sc)?;
        if !noise.is_empty() {
            ops.write_f32(s.noise, noise)?;
        }
        let cmd = TokenCommands::begin(ops.context())?;
        self.record(&cmd, sc.is_some(), false, s.logits)?;
        let uncond = if (opts.guidance - 1.).abs() > 1e-6 {
            self.record(&cmd, sc.is_some(), true, s.uncond)?;
            s.uncond
        } else {
            s.logits
        };
        let p = opts.pixel;
        ops.record_custom(
            &cmd,
            self.sample_shader,
            None,
            &[
                word(s.logits),
                word(uncond),
                word(s.noise),
                word(s.probs),
                word(s.triplets),
                opts.guidance.to_bits(),
                opts.temperature.to_bits(),
                p.top_k as u32,
                p.top_p.to_bits(),
                p.min_p.to_bits(),
            ],
            [PIXELS as u32, 1, 1],
        )?;
        cmd.submit_and_wait()?;
        let probs = if p.self_cond {
            self.graph.read(s.probs, PIXELS * VOCAB)?
        } else {
            Vec::new()
        };
        let mut best = Vec::with_capacity(PIXELS);
        let mut ids = Vec::with_capacity(PIXELS);
        let mut confidence = Vec::with_capacity(PIXELS);
        for row in self.graph.read(s.triplets, PIXELS * 3)?.chunks_exact(3) {
            if !row.iter().all(|x| x.is_finite())
                || !(0.0..VOCAB as f32).contains(&row[0])
                || !(0.0..VOCAB as f32).contains(&row[1])
            {
                return Err(VulkanError::UnsupportedShape(
                    "invalid Tessera device draw".into(),
                ));
            }
            best.push(row[0] as u16);
            ids.push(row[1] as u16);
            confidence.push(row[2]);
        }
        Ok(Draw {
            probs,
            ids,
            confidence,
            best,
        })
    }

    pub(super) fn remask(&self, canvas: &mut [u16], conf: &[f32], count: usize) -> Result<()> {
        let ops = &self.graph.ops;
        let values: Vec<_> = canvas.iter().map(|&i| f32::from_bits(i as u32)).collect();
        ops.write_f32(self.s.ids, &values)?;
        ops.write_f32(self.s.commit_conf, conf)?;
        let cmd = TokenCommands::begin(ops.context())?;
        ops.record_custom(
            &cmd,
            self.commit_shader,
            None,
            &[
                word(self.s.ids),
                word(self.s.commit_conf),
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                count as u32,
            ],
            [1, 1, 1],
        )?;
        cmd.submit_and_wait()?;
        for (id, x) in canvas.iter_mut().zip(self.graph.read(self.s.ids, PIXELS)?) {
            *id = x.to_bits() as u16;
        }
        Ok(())
    }
    pub(super) fn commit(
        &self,
        canvas: &mut [u16],
        commit_conf: &mut [f32],
        draw: &Draw,
        gumbels: &[f32],
        remaining: usize,
        p: &PixelOptions,
    ) -> Result<()> {
        let ops = &self.graph.ops;
        let s = &self.s;
        for (target, ids) in [(s.ids, &*canvas), (s.commit_ids, draw.ids.as_slice())] {
            let values: Vec<_> = ids
                .iter()
                .map(|&id| f32::from_bits(u32::from(id)))
                .collect();
            ops.write_f32(target, &values)?;
        }
        ops.write_f32(s.confidence, &draw.confidence)?;
        ops.write_f32(s.commit_conf, commit_conf)?;
        ops.write_f32(s.gumbels, gumbels)?;
        let cmd = TokenCommands::begin(ops.context())?;
        ops.record_custom(
            &cmd,
            self.commit_shader,
            None,
            &[
                word(s.ids),
                word(s.commit_conf),
                word(s.commit_ids),
                word(s.confidence),
                word(s.gumbels),
                remaining as u32,
                p.spatial_nms as u32,
                p.nms_threshold.to_bits(),
                p.commit_threshold.to_bits(),
                0,
            ],
            [1, 1, 1],
        )?;
        cmd.submit_and_wait()?;
        let ids = self
            .graph
            .read(s.ids, PIXELS)?
            .iter()
            .map(|f| f.to_bits() as u16)
            .collect::<Vec<_>>();
        ops.read_f32_into(s.commit_conf, commit_conf)?;
        canvas.copy_from_slice(&ids);
        Ok(())
    }
}
