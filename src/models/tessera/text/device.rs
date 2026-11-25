//! Request-local CLIP device graph. Hidden states, attention and MLP intermediates
//! stay in this arena across all twelve blocks.
use super::{Block, Norm, TextEncoder, CONTEXT, FF, HEADS, WIDTH};
use crate::{
    models::tessera::device::{word, Graph, Layout, Matrix},
    ops::{
        kernel::Weight,
        nn::{AttentionSpec, Linear},
    },
    vulkan::{
        ops::{ArenaRegion, TokenCommands},
        VulkanContext, VulkanError,
    },
};

type Result<T> = std::result::Result<T, VulkanError>;

struct Affine {
    scale: u32,
    bias: u32,
}
struct Projection {
    matrix: Matrix,
    bias: u32,
}
struct Layer {
    norm1: Affine,
    norm2: Affine,
    q: Projection,
    k: Projection,
    v: Projection,
    out: Projection,
    fc1: Projection,
    fc2: Projection,
}

/// All small biases and normalization parameters share one upload.
struct Parameters {
    base: u32,
    data: Vec<f32>,
}
impl Parameters {
    fn push(&mut self, values: &[f32]) -> u32 {
        let at = self.base + self.data.len() as u32;
        self.data.extend_from_slice(values);
        at
    }
    fn affine(&mut self, norm: &Norm) -> Affine {
        Affine {
            scale: self.push(&norm.weight),
            bias: self.push(&norm.bias),
        }
    }
    fn layer(&mut self, graph: &mut Graph, block: &Block<'_>) -> Result<Layer> {
        let norm1 = self.affine(&block.norm1);
        let norm2 = self.affine(&block.norm2);
        let mut projection = |linear: &Linear<'_>| -> Result<Projection> {
            let bias = linear.bias().ok_or_else(|| {
                VulkanError::UnsupportedShape("CLIP projection lacks its bias".into())
            })?;
            Ok(Projection {
                matrix: graph.bind(linear)?,
                bias: self.push(bias),
            })
        };
        Ok(Layer {
            norm1,
            norm2,
            q: projection(&block.q)?,
            k: projection(&block.k)?,
            v: projection(&block.v)?,
            out: projection(&block.out)?,
            fc1: projection(&block.fc1)?,
            fc2: projection(&block.fc2)?,
        })
    }
}

struct Slots {
    tokens: ArenaRegion,
    positions: ArenaRegion,
    hidden: ArenaRegion,
    norm: ArenaRegion,
    q: ArenaRegion,
    k: ArenaRegion,
    v: ArenaRegion,
    attn: ArenaRegion,
    projection: ArenaRegion,
    ff: ArenaRegion,
}

pub(super) struct Device {
    graph: Graph,
    slots: Slots,
    blocks: Vec<Layer>,
    final_norm: Affine,
    context: &'static VulkanContext,
    position_format: u32,
}

fn row_bytes(weight: &Weight<'_>) -> usize {
    let (elements, bytes) = weight.ggml_type.type_traits();
    weight.n_in / elements * bytes
}

/// Copy the requested rows in their original storage type. Only the selected
/// bytes cross the host boundary; their GGUF decoding and addition run on GPU.
fn packed_rows(weight: &Weight<'_>, ids: impl Iterator<Item = usize>) -> Result<Vec<f32>> {
    let src = weight.kernel.weight_bytes().ok_or_else(|| {
        VulkanError::UnsupportedShape("CLIP embedding has no native weight storage".into())
    })?;
    let stride = row_bytes(weight);
    let mut data = vec![0.; CONTEXT * stride / 4];
    let dst: &mut [u8] = bytemuck::cast_slice_mut(&mut data);
    for (dest, id) in dst.chunks_exact_mut(stride).zip(ids) {
        dest.copy_from_slice(&src[id * stride..(id + 1) * stride]);
    }
    Ok(data)
}

impl Device {
    pub(super) fn new(encoder: &TextEncoder<'_>, context: &'static VulkanContext) -> Result<Self> {
        let mut layout = Layout::default();
        let param_count = encoder.blocks.len() * (9 * WIDTH + FF) + 2 * WIDTH;
        let param_region = layout.floats(param_count);
        let slots = Slots {
            tokens: layout.floats(CONTEXT * row_bytes(&encoder.tokens) / 4),
            positions: layout.floats(CONTEXT * row_bytes(&encoder.positions) / 4),
            hidden: layout.floats(CONTEXT * WIDTH),
            norm: layout.floats(CONTEXT * WIDTH),
            q: layout.floats(CONTEXT * WIDTH),
            k: layout.floats(CONTEXT * WIDTH),
            v: layout.floats(CONTEXT * WIDTH),
            attn: layout.floats(CONTEXT * WIDTH),
            projection: layout.floats(CONTEXT * WIDTH),
            ff: layout.floats(CONTEXT * FF),
        };
        let mut graph = Graph::new(
            context,
            layout,
            CONTEXT,
            FF,
            6 * encoder.blocks.len(),
            include_bytes!("../../../../shaders/bin/tessera_clip.spv"),
        )?;
        let mut params = Parameters {
            base: word(param_region),
            data: Vec::with_capacity(param_count),
        };
        let blocks = encoder
            .blocks
            .iter()
            .map(|b| params.layer(&mut graph, b))
            .collect::<Result<Vec<_>>>()?;
        let final_norm = params.affine(&encoder.norm);
        graph.ops.write_f32(param_region, &params.data)?;
        graph.ops.write_f32(
            slots.positions,
            &packed_rows(&encoder.positions, 0..CONTEXT)?,
        )?;
        Ok(Self {
            graph,
            slots,
            blocks,
            final_norm,
            context,
            position_format: encoder.positions.ggml_type as u32,
        })
    }

    fn norm(&self, cmd: &TokenCommands<'_>, input: ArenaRegion, norm: &Affine) -> Result<()> {
        self.graph.custom(
            cmd,
            &[
                1,
                word(input),
                word(self.slots.norm),
                0,
                norm.scale,
                norm.bias,
            ],
            [CONTEXT as u32, 1, 1],
        )
    }

    fn residual(&self, cmd: &TokenCommands<'_>, branch: &Projection) -> Result<()> {
        let s = &self.slots;
        self.graph.custom(
            cmd,
            &[3, word(s.projection), word(s.hidden), 0, branch.bias],
            [(CONTEXT * WIDTH / 64) as u32, 1, 1],
        )
    }

    pub(super) fn encode(&self, ids: &[u32], tokens: &Weight<'_>) -> Result<Vec<f32>> {
        let s = &self.slots;
        let graph = &self.graph;
        let packed = packed_rows(tokens, ids.iter().map(|&id| id as usize))?;
        graph.ops.write_f32(s.tokens, &packed)?;
        let start = graph.projection_count.get();
        let cmd = TokenCommands::begin(self.context)?;
        let groups = [(CONTEXT * WIDTH / 64) as u32, 1, 1];
        graph.custom(
            &cmd,
            &[
                0,
                word(s.tokens),
                word(s.positions),
                word(s.hidden),
                tokens.ggml_type as u32,
                self.position_format,
            ],
            groups,
        )?;
        for b in &self.blocks {
            self.norm(&cmd, s.hidden, &b.norm1)?;
            for (p, out) in [(&b.q, s.q), (&b.k, s.k), (&b.v, s.v)] {
                graph.project(&cmd, &p.matrix, s.norm, out, CONTEXT)?;
            }
            graph.custom(
                &cmd,
                &[
                    2,
                    word(s.q),
                    word(s.k),
                    word(s.v),
                    b.q.bias,
                    b.k.bias,
                    b.v.bias,
                ],
                groups,
            )?;
            graph.ops.record_online_attention(
                &cmd,
                s.q,
                s.k,
                s.v,
                s.attn,
                AttentionSpec {
                    query_tokens: CONTEXT,
                    key_tokens: CONTEXT,
                    query_heads: HEADS,
                    key_value_heads: HEADS,
                    head_dim: WIDTH / HEADS,
                    scale: ((WIDTH / HEADS) as f32).sqrt().recip(),
                    causal: true,
                },
                0,
                0,
            )?;
            graph.project(&cmd, &b.out.matrix, s.attn, s.projection, CONTEXT)?;
            self.residual(&cmd, &b.out)?;
            self.norm(&cmd, s.hidden, &b.norm2)?;
            graph.project(&cmd, &b.fc1.matrix, s.norm, s.ff, CONTEXT)?;
            graph.custom(
                &cmd,
                &[4, word(s.ff), 0, 0, b.fc1.bias],
                [(CONTEXT * FF / 64) as u32, 1, 1],
            )?;
            graph.project(&cmd, &b.fc2.matrix, s.ff, s.projection, CONTEXT)?;
            self.residual(&cmd, &b.fc2)?;
        }
        self.norm(&cmd, s.hidden, &self.final_norm)?;
        cmd.submit_and_wait()?;
        let mut states = vec![0.; CONTEXT * WIDTH];
        graph.ops.read_f32_into(s.norm, &mut states)?;
        if std::env::var_os("RMI_COMPUTE_TRACE").is_some() {
            let n = self.blocks.len();
            eprintln!("Tessera CLIP backend=Vulkan projections={} embedding=1 norm={} attention={n} residual={} quick_gelu={n} host_input_bytes={} host_output_bytes={} submissions=1",
                graph.projection_count.get()-start, 2*n+1,2*n,packed.len()*4,states.len()*4);
        }
        Ok(states)
    }
}
