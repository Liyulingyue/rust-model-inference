//! Vulkan resident session for the dense `nemotron_h` trunk
//! (Nemotron-3 Nano 4B).
//!
//! The 4B is a hybrid Mamba-2 / attention tower: 42 layers made of 21
//! SSM-only layers, 17 FFN-only layers and 4 attention-only layers. Every
//! layer still carries a pre-norm on its input, and the three branch types
//! are mutually exclusive per layer.
//!
//! This module follows the shape of `qwen35.rs` (the other hybrid SSM
//! trunk) but the semantics differ in three places that matter:
//!
//! * **SSM**: Nemotron uses canonical Mamba-2 (`state = state * exp(dt*A) +
//!   B*x*dt`, `y = state·C + D*x`, then `silu(z)*y`, then a group RMSNorm),
//!   not Qwen3.5's recurrent form (`beta`/`alpha` gates, RMS-norm'd Q/K).
//!   The Qwen3.5 SSM shaders cannot be reused.
//! * **Attention**: Nemotron's attention has no output gate; the Qwen3.5
//!   attention shader applies a SiLU gate after the value mix.
//! * **FFN**: Nemotron uses `ReLU(x)^2` (LLM_FFN_RELU_SQR), not SwiGLU.
//!
//! All 42 layers run on the device. The Mamba-2 branch is the reason
//! this module is not a copy of `qwen35.rs`: its recurrence is serial in
//! the token dimension, so an SSM layer is recorded as
//!
//! 1. one RMS norm for every row,
//! 2. the `ssm_in` projection for every row,
//! 3. the conv1d, the time step, the scan, the z gate and the group norm
//!    for **one token at a time** (the conv history and the scan state
//!    belong to the sequence, not to the row),
//! 4. the `ssm_out` projection for every row,
//! 5. the residual add.
//!
//! The device keeps the Mamba-2 state (the conv history ring and the
//! scan state) next to the attention KV cache, so both are cleared
//! together when a new sequence starts.

use crate::models::nemotron_h::trunk::config::NemotronConfig;
use crate::models::nemotron_h::trunk::forward::NemotronModel;
use crate::ops::kernel::Weight;
use crate::vulkan::dense::UploadedBuffers;
use crate::vulkan::ops::{
    row_dispatch, ArenaRegion, ExpertSelection, GpuMatmulSpec, GpuWeightFormat, GpuWeightLayout,
    OperatorBindings, Qwen3Ops, TokenCommands,
};
use crate::vulkan::{GpuBuffer, VulkanContext, VulkanError};

/// Bindings for one attention layer (Q/K/V/O projections).
#[derive(Clone, Copy)]
struct AttentionBindings {
    q: OperatorBindings,
    k: OperatorBindings,
    v: OperatorBindings,
    /// Q/K RMSNorm weights packed as `[q_norm, k_norm]`. `None` when the
    /// layer has no Q/K norm (the 4B Nano attention layers ship neither).
    qk_norm: Option<OperatorBindings>,
    output: OperatorBindings,
}

/// Bindings for one FFN layer (up/down projections).
#[derive(Clone, Copy)]
struct FfnBindings {
    up: OperatorBindings,
    down: OperatorBindings,
}

/// Bindings for one mixture-of-experts layer.
///
/// The 30B-A3B's FFN layers are routed-expert layers: the dense
/// `ffn_up`/`ffn_down` weights are absent, and the layer holds 128
/// routed experts plus one shared expert instead.
#[derive(Clone, Copy)]
struct MoeBindings {
    /// The router, `ffn_gate_inp`: F32 `[n_embd, n_expert]`.
    router: OperatorBindings,
    /// The routed expert up projections, `ffn_up_exps`.
    ///
    /// The whole 3-D tensor lives in one buffer: the kernel reads the
    /// expert the routing chose and offsets the weight by one expert
    /// matrix, so a layer needs one descriptor set instead of one per
    /// expert.
    up_exps: OperatorBindings,
    /// The routed expert down projections, `ffn_down_exps`.
    down_exps: OperatorBindings,
    /// The shared expert up projection, `ffn_up_shexp`.
    up_shexp: OperatorBindings,
    /// The shared expert down projection, `ffn_down_shexp`.
    down_shexp: OperatorBindings,
}

/// Bindings for one Mamba-2 SSM layer.
#[derive(Clone, Copy)]
struct SsmBindings {
    /// The `ssm_in` projection, `Q8_0`.
    input_projection: OperatorBindings,
    /// The `ssm_out` projection, `Q8_0`.
    output_projection: OperatorBindings,
    /// The group RMSNorm weights, F32.
    group_norm: OperatorBindings,
}

enum LayerBindings {
    Attention {
        norm: OperatorBindings,
        attention: AttentionBindings,
    },
    Ffn {
        norm: OperatorBindings,
        ffn: FfnBindings,
    },
    /// SSM-only layer.
    Ssm {
        norm: OperatorBindings,
        ssm: SsmBindings,
    },
    /// A mixture-of-experts layer (the 30B-A3B).
    Moe {
        norm: OperatorBindings,
        moe: MoeBindings,
    },
}

/// Arena regions that belong to one Mamba-2 SSM layer.
///
/// The scan state and the conv history are *persistent*: they carry the
/// recurrence from one token to the next, exactly like the CPU path's
/// `scratch.ssm_scan_state` and `scratch.ssm_conv_hist`. The F32 weights
/// never change, so they are written into the arena once when the session
/// is built.
#[derive(Clone, Copy)]
struct SsmArena {
    /// Persistent scan state, `heads * headdim * d_state` floats.
    state: ArenaRegion,
    /// Persistent conv history ring, `3 * conv_cols` floats.
    history: ArenaRegion,
    /// Fused conv1d weights, `conv_cols * kernel` floats.
    conv_weights: ArenaRegion,
    /// Fused conv1d bias, `conv_cols` floats.
    conv_bias: ArenaRegion,
    /// Time step bias, `heads` floats.
    dt_bias: ArenaRegion,
    /// `A_log`, `heads` floats.
    a_log: ArenaRegion,
    /// D skip, `heads` floats.
    d: ArenaRegion,
}

/// The Mamba-2 dimensions of one SSM layer.
#[derive(Clone, Copy)]
struct SsmShape {
    /// `d_inner`, the width of the scan output.
    inner: usize,
    /// `n_group`, the number of groups sharing a `B`/`C` pair.
    groups: usize,
    /// `d_state`, the width of the state vector.
    d_state: usize,
    /// `dt_rank`, which is also the head count.
    heads: usize,
    /// `d_inner / heads`.
    headdim: usize,
    /// The causal conv1d kernel width.
    kernel: usize,
    /// `d_inner + 2 * n_group * d_state`, the fused conv1d width.
    conv_cols: usize,
    /// `2 * d_inner + 2 * n_group * d_state + dt_rank`, the in-projection
    /// width.
    in_projection: usize,
    /// `d_inner / n_group`, the group norm width.
    group_len: usize,
}

impl SsmShape {
    fn new(config: &NemotronConfig) -> Result<Self, VulkanError> {
        let inner = config.ssm_inner_size;
        let groups = config.ssm_group_count;
        let d_state = config.ssm_state_size;
        let heads = config.ssm_time_step_rank;
        let kernel = config.ssm_conv_kernel;
        if inner == 0
            || groups == 0
            || d_state == 0
            || heads == 0
            || kernel < 2
            || inner % heads != 0
            || inner % groups != 0
            || d_state % 16 != 0
        {
            return Err(VulkanError::UnsupportedShape(
                "invalid Nemotron-H SSM dimensions".into(),
            ));
        }
        // `d_inner + 2 * n_group * d_state`, the fused conv1d width.
        let conv_cols = inner
            .checked_add(2 * groups * d_state)
            .ok_or(VulkanError::OutOfMemory)?;
        // `2 * d_inner + 2 * n_group * d_state + dt_rank`, the in-projection
        // width.
        let in_projection = 2 * inner + 2 * groups * d_state + heads;
        Ok(Self {
            inner,
            groups,
            d_state,
            heads,
            headdim: inner / heads,
            kernel,
            conv_cols,
            in_projection,
            group_len: inner / groups,
        })
    }
}

/// Arena layout for the Nemotron-3 Nano device session.
///
/// The KV cache is one contiguous region covering every attention layer,
/// because the KV ops index it by layer; `slots` maps a global layer index
/// to its position within that region.
struct NemotronHArenaLayout {
    x: ArenaRegion,
    normed: ArenaRegion,
    q: ArenaRegion,
    k: ArenaRegion,
    v: ArenaRegion,
    attn: ArenaRegion,
    projection: ArenaRegion,
    ffn_up: ArenaRegion,
    ffn_down: ArenaRegion,
    logits: ArenaRegion,
    rope: ArenaRegion,
    scores: ArenaRegion,
    q8: ArenaRegion,
    q8_scales: ArenaRegion,
    q4_1_input_sums: ArenaRegion,
    q8k: ArenaRegion,
    q8k_scales: ArenaRegion,
    kv_k: ArenaRegion,
    kv_v: ArenaRegion,
    kv_delta_k: ArenaRegion,
    kv_delta_v: ArenaRegion,
    /// The Mamba-2 in-projection output, `rows * in_projection` floats.
    /// The layout is `[z | x | B | C | dt]`, which is what
    /// `split_mamba2_projection` returns on the CPU.
    ssm_in_projection: ArenaRegion,
    /// The post-SiLU conv1d output, `rows * conv_cols` floats, holding
    /// `[x | B | C]`.
    ssm_conv: ArenaRegion,
    /// The softplus time step, `rows * heads` floats.
    ssm_dt: ArenaRegion,
    /// The per-head decay, `rows * heads` floats.
    ssm_decay: ArenaRegion,
    /// The scan output plus the D skip, `rows * inner` floats.
    ssm_scan: ArenaRegion,
    /// The z-gated scan output, `rows * inner` floats.
    ssm_gated: ArenaRegion,
    /// The group-normalised scan output, `rows * inner` floats.
    ssm_normed: ArenaRegion,
    /// The SSM output projection, `rows * n_embd` floats.
    ssm_projection: ArenaRegion,
    /// Per-layer SSM regions, indexed by the SSM layer's slot.
    ssm_layers: Vec<SsmArena>,
    /// The router logits, `rows * n_expert` floats.
    router_logits: ArenaRegion,
    /// The sigmoid probabilities of the router, `rows * n_expert` floats.
    router_probs: ArenaRegion,
    /// The ranking scores, `probs + exp_probs_b.bias`, `rows * n_expert`
    /// floats.
    router_selection: ArenaRegion,
    /// The chosen expert ids, `rows * n_expert_used` words.
    expert_ids: ArenaRegion,
    /// The routing weights of the chosen experts, `rows * n_expert_used`
    /// floats.
    expert_weights: ArenaRegion,
    /// The routed expert up projections, `rows * n_expert_used * n_ff_exp`
    /// floats. The layout is `[row][slot][n_ff_exp]`.
    expert_up: ArenaRegion,
    /// The routed expert down projections, `rows * n_expert_used * n_embd`
    /// floats. The layout is `[row][slot][n_embd]`.
    expert_down: ArenaRegion,
    /// The shared expert up projection, `rows * n_ff_shexp` floats.
    shared_up: ArenaRegion,
    /// The shared expert down projection, `rows * n_embd` floats.
    shared_down: ArenaRegion,
    /// The per-layer router selection bias, `exp_probs_b.bias`: one
    /// region of `n_expert` floats per MoE layer.
    moe_bias: Vec<ArenaRegion>,
    slots: Vec<Option<usize>>,
    total_size: usize,
}

fn product(label: &str, values: &[usize]) -> Result<usize, VulkanError> {
    values.iter().try_fold(1usize, |product, &value| {
        product.checked_mul(value).ok_or_else(|| {
            VulkanError::UnsupportedShape(format!("Nemotron-H {label} size overflows usize"))
        })
    })
}

fn f32_region(cursor: &mut usize, elements: usize) -> Result<ArenaRegion, VulkanError> {
    let bytes = elements.checked_mul(4).ok_or(VulkanError::OutOfMemory)?;
    region(cursor, bytes)
}

fn region(cursor: &mut usize, size: usize) -> Result<ArenaRegion, VulkanError> {
    let offset = cursor
        .checked_add(15)
        .map(|value| value & !15)
        .ok_or(VulkanError::OutOfMemory)?;
    *cursor = offset.checked_add(size).ok_or(VulkanError::OutOfMemory)?;
    Ok(ArenaRegion { offset, size })
}

impl NemotronHArenaLayout {
    fn new(config: &NemotronConfig, capacity: usize, rows: usize) -> Result<Self, VulkanError> {
        let n_embd = config.n_embd;
        let n_head = config.n_head;
        let n_head_kv = config.n_head_kv;
        let head_dim_k = config.n_embd_head_k;
        let head_dim_v = config.n_embd_head_v;
        let n_attn_q = n_head * head_dim_k;
        let n_attn_kv = n_head_kv * head_dim_k;
        let n_attn_v_kv = n_head_kv * head_dim_v;
        let n_ff = config.n_ff;
        if [
            n_embd,
            config.n_layer,
            n_head,
            n_head_kv,
            head_dim_k,
            head_dim_v,
            n_ff,
            capacity,
            rows,
            config.rope_dim,
            config.vocab_size,
        ]
        .contains(&0)
            || !n_head.is_multiple_of(n_head_kv)
            || config.rope_dim > head_dim_k
            || !config.rope_dim.is_multiple_of(2)
        {
            return Err(VulkanError::UnsupportedShape(
                "invalid Nemotron-H Vulkan arena dimensions".into(),
            ));
        }
        // The Q8 activation buffer has to hold the widest quantized
        // input of the model: the projections, and for the 30B the shared
        // expert (3712) and the expert down projection (1856).
        let quant_len = n_embd
            .max(n_ff)
            .max(n_attn_q)
            .max(n_attn_v_kv)
            .max(config.moe.as_ref().map_or(0, |moe| moe.n_ff_exp))
            .max(config.moe.as_ref().map_or(0, |moe| moe.n_ff_shexp));
        let kv_cache = product(
            "KV cache",
            &[config.attention_layers.len(), capacity, n_attn_kv],
        )?;
        let kv_delta = product(
            "KV delta",
            &[config.attention_layers.len(), rows, n_attn_kv],
        )?;
        let mut cursor = 0usize;
        let mut row_region =
            |width| f32_region(&mut cursor, product("row buffer", &[rows, width])?);
        let x = row_region(n_embd)?;
        let normed = row_region(n_embd)?;
        let q = row_region(n_attn_q)?;
        let k = row_region(n_attn_kv)?;
        let v = row_region(n_attn_v_kv)?;
        let attn = row_region(n_attn_q)?;
        let projection = row_region(n_embd)?;
        let ffn_up = row_region(n_ff)?;
        let ffn_down = row_region(n_embd)?;
        let rope = row_region(config.rope_dim)?;
        // The mixture-of-experts scratch. The dense architecture has none
        // of it, so the regions stay empty there.
        let (n_expert, n_expert_used, n_ff_exp, n_ff_shexp) =
            config.moe.as_ref().map_or((0, 0, 0, 0), |moe| {
                (
                    moe.n_expert,
                    moe.n_expert_used,
                    moe.n_ff_exp,
                    moe.n_ff_shexp,
                )
            });
        let router_logits = row_region(n_expert)?;
        let router_probs = row_region(n_expert)?;
        let router_selection = row_region(n_expert)?;
        let expert_ids = row_region(n_expert_used)?;
        let expert_weights = row_region(n_expert_used)?;
        let expert_up = row_region(product("expert up", &[n_expert_used, n_ff_exp])?)?;
        let expert_down = row_region(product("expert down", &[n_expert_used, n_embd])?)?;
        let shared_up = row_region(n_ff_shexp)?;
        let shared_down = row_region(n_embd)?;
        let scores = f32_region(
            &mut cursor,
            product("attention scores", &[rows, n_head, capacity])?,
        )?;
        let logits = f32_region(&mut cursor, config.vocab_size)?;
        let quant_rows = product("quantized rows", &[rows, quant_len.div_ceil(4) * 4])?;
        let q8 = region(&mut cursor, quant_rows)?;
        let q8_scales = f32_region(
            &mut cursor,
            product("Q8 scales", &[rows, quant_len.div_ceil(32)])?,
        )?;
        let q4_1_input_sums = f32_region(
            &mut cursor,
            product("Q4 sums", &[rows, quant_len.div_ceil(32)])?,
        )?;
        let q8k = region(&mut cursor, quant_rows)?;
        let q8k_scales = f32_region(
            &mut cursor,
            product("Q8K scales", &[rows, quant_len.div_ceil(256)])?,
        )?;
        let kv_k = f32_region(&mut cursor, kv_cache)?;
        let kv_v = f32_region(&mut cursor, kv_cache)?;
        let kv_delta_k = f32_region(&mut cursor, kv_delta)?;
        let kv_delta_v = f32_region(&mut cursor, kv_delta)?;
        // The Mamba-2 scratch is shared by every SSM layer: the layers run
        // one after another, so only one layer's intermediate values are
        // live at a time. The scan state and the conv history are the
        // exception -- they carry the recurrence across tokens, so they
        // get one region per layer.
        let ssm = SsmShape::new(config)?;
        let ssm_in_projection = f32_region(
            &mut cursor,
            product("SSM in-projection rows", &[rows, ssm.in_projection])?,
        )?;
        let ssm_conv = f32_region(
            &mut cursor,
            product("SSM conv rows", &[rows, ssm.conv_cols])?,
        )?;
        let ssm_dt = f32_region(&mut cursor, product("SSM dt rows", &[rows, ssm.heads])?)?;
        let ssm_decay = f32_region(&mut cursor, product("SSM decay rows", &[rows, ssm.heads])?)?;
        let ssm_scan = f32_region(&mut cursor, product("SSM scan rows", &[rows, ssm.inner])?)?;
        let ssm_gated = f32_region(&mut cursor, product("SSM gated rows", &[rows, ssm.inner])?)?;
        let ssm_normed = f32_region(
            &mut cursor,
            product("SSM normalised rows", &[rows, ssm.inner])?,
        )?;
        let ssm_projection = f32_region(
            &mut cursor,
            product("SSM projection rows", &[rows, n_embd])?,
        )?;
        let state_elements = product("SSM state", &[ssm.heads, ssm.headdim, ssm.d_state])?;
        let history_elements = product("SSM conv history", &[3, ssm.conv_cols])?;
        let conv_weight_elements = product("SSM conv weights", &[ssm.conv_cols, ssm.kernel])?;
        let mut ssm_layers = Vec::new();
        for layer in 0..config.n_layer {
            if config.attention_layers.contains(&layer) || config.ffn_layers.contains(&layer) {
                continue;
            }
            ssm_layers.push(SsmArena {
                state: f32_region(&mut cursor, state_elements)?,
                history: f32_region(&mut cursor, history_elements)?,
                conv_weights: f32_region(&mut cursor, conv_weight_elements)?,
                conv_bias: f32_region(&mut cursor, ssm.conv_cols)?,
                dt_bias: f32_region(&mut cursor, ssm.heads)?,
                a_log: f32_region(&mut cursor, ssm.heads)?,
                d: f32_region(&mut cursor, ssm.heads)?,
            });
        }
        let mut slots = vec![None; config.n_layer];
        for (layer, slot) in slots.iter_mut().enumerate() {
            if config.attention_layers.contains(&layer) {
                *slot = Some(
                    config
                        .attention_layers
                        .iter()
                        .position(|&l| l == layer)
                        .expect("attention layer is in the list"),
                );
            }
        }
        // The router selection bias is a per-layer constant, so it gets a
        // region of its own per MoE layer.
        let mut moe_bias = Vec::new();
        for layer in 0..config.n_layer {
            if !config.moe.is_some() || !config.ffn_layers.contains(&layer) {
                continue;
            }
            moe_bias.push(f32_region(&mut cursor, n_expert)?);
        }
        Ok(Self {
            x,
            normed,
            q,
            k,
            v,
            attn,
            projection,
            ffn_up,
            ffn_down,
            logits,
            rope,
            scores,
            q8,
            q8_scales,
            q4_1_input_sums,
            q8k,
            q8k_scales,
            kv_k,
            kv_v,
            kv_delta_k,
            kv_delta_v,
            ssm_in_projection,
            ssm_conv,
            ssm_dt,
            ssm_decay,
            ssm_scan,
            ssm_gated,
            ssm_normed,
            ssm_projection,
            ssm_layers,
            router_logits,
            router_probs,
            router_selection,
            expert_ids,
            expert_weights,
            expert_up,
            expert_down,
            shared_up,
            shared_down,
            moe_bias,
            slots,
            total_size: cursor,
        })
    }
}

pub(crate) struct NemotronHVulkanSession {
    context: &'static VulkanContext,
    ops: Qwen3Ops<'static>,
    _buffers: UploadedBuffers,
    layers: Vec<LayerBindings>,
    output_norm: OperatorBindings,
    output: OperatorBindings,
    layout: NemotronHArenaLayout,
    config: NemotronConfig,
    /// The Mamba-2 dimensions shared by every SSM layer.
    ssm: SsmShape,
    /// The SSM slot of each layer, `None` for the attention and FFN
    /// layers.
    ssm_slots: Vec<Option<usize>>,
    /// The mixture-of-experts slot of each layer, `None` for the layers
    /// that are not routed-expert layers.
    moe_slots: Vec<Option<usize>>,
    pub(crate) capacity: usize,
    max_rows: usize,
    rope: Vec<f32>,
    logits: Vec<f32>,
    /// The number of tokens the recurrent state on the device covers.
    ///
    /// The Mamba-2 recurrence (the conv history and the scan state) and
    /// the attention KV cache belong to a *sequence*, not to a call. The
    /// host keeps the same state in the scratch pad, so the two must
    /// agree on where the sequence starts: a fresh scratch pad means a
    /// fresh sequence, and the device state has to be cleared to match.
    committed_position: usize,
    /// Pipeline for the `ReLU(x)^2` activation, added at construction time.
    relu_square_pipeline: usize,
}

fn required_weight<'a>(
    weight: Option<&'a Weight<'a>>,
    layer: usize,
    name: &str,
) -> Result<&'a Weight<'a>, VulkanError> {
    weight.ok_or_else(|| {
        VulkanError::UnsupportedShape(format!("missing Nemotron-H layer {layer} {name} weight"))
    })
}

/// Fetch one of a layer's F32 SSM weights and check its width.
///
/// The SSM weights are tiny next to the projections, so they live in the
/// arena rather than in their own buffers.
fn required_ssm_vector<'a>(
    values: Option<&'a Vec<f32>>,
    layer: usize,
    name: &str,
    expected: usize,
) -> Result<&'a [f32], VulkanError> {
    let values = values.ok_or_else(|| {
        VulkanError::UnsupportedShape(format!("missing Nemotron-H layer {layer} {name} weight"))
    })?;
    if values.len() != expected {
        return Err(VulkanError::UnsupportedShape(format!(
            "Nemotron-H layer {layer} {name} has {} values, expected {expected}",
            values.len()
        )));
    }
    Ok(values.as_slice())
}

fn upload_weight(
    buffers: &mut UploadedBuffers,
    weight: &Weight<'_>,
    label: &str,
) -> Result<(GpuBuffer, GpuMatmulSpec, GpuWeightLayout), VulkanError> {
    let format = GpuMatmulSpec::prepared(weight)?;
    let bytes = weight.kernel.weight_bytes().ok_or_else(|| {
        VulkanError::UnsupportedShape(format!(
            "{label} does not expose {:?} storage bytes",
            weight.ggml_type
        ))
    })?;
    // The Q8_0 weights go into the layout the device kernels read. That is
    // ggml's own layout unless the transposed layout is turned on, and the
    // decision is the shared one, so this path and the generic weight
    // storage agree on it.
    let layout = format.format.device_layout(buffers.context());
    let buffer = if layout == GpuWeightLayout::Transposed {
        let (n_in, n_out) = (weight.n_in, weight.n_out);
        let size = crate::vulkan::q8_layout::transposed_size(n_in, n_out);
        buffers.upload_filled(size, |packed| {
            crate::vulkan::q8_layout::transpose_q8_0_weight_into(packed, bytes, n_in, n_out);
        })?
    } else {
        buffers.upload(bytes)?
    };
    Ok((buffer, format, layout))
}

fn bind_weight(
    ops: &mut Qwen3Ops<'_>,
    buffers: &mut UploadedBuffers,
    weight: &Weight<'_>,
    label: &str,
) -> Result<OperatorBindings, VulkanError> {
    let (buffer, format, layout) = upload_weight(buffers, weight, label)?;
    ops.bind_weight_buffers_mode_layout(
        std::slice::from_ref(&buffer),
        std::slice::from_ref(&format.format),
        format.mode,
        layout,
    )
}

/// Bind a whole 3-D expert tensor as a single buffer.
///
/// ggml stores the expert axis last and contiguous, so one expert is one
/// matrix after the other. The kernel reads the expert the routing chose
/// and offsets the weight by one expert matrix, which keeps the layer at
/// one descriptor set instead of one per expert.
fn bind_expert_tensor(
    ops: &mut Qwen3Ops<'_>,
    buffers: &mut UploadedBuffers,
    bytes: &[u8],
    ggml_type: crate::core::tensor::GGMLType,
) -> Result<OperatorBindings, VulkanError> {
    let format = GpuWeightFormat::from_ggml_type(ggml_type)?;
    let buffer = buffers.upload(bytes)?;
    let size = u64::try_from(bytes.len()).map_err(|_| VulkanError::OutOfMemory)?;
    ops.bind_weight_ranges(&[(buffer, 0, size)], &[format])
}

/// Whether a weight format has a device kernel that can read the expert
/// the routing chose.
///
/// The mixture-of-experts layers keep all the experts of one matrix in a
/// single buffer and let the kernel offset the weight by the selected
/// expert, so the format has to support that addressing.
fn check_expert_format(value: crate::core::tensor::GGMLType, label: &str) -> Result<(), String> {
    let format = GpuWeightFormat::from_ggml_type(value).map_err(|error| error.to_string())?;
    if !matches!(
        format,
        GpuWeightFormat::Q8_0
            | GpuWeightFormat::Q5_0
            | GpuWeightFormat::Q5_1
            | GpuWeightFormat::F32
    ) {
        return Err(format!("{label}: {value:?} has no device expert kernel"));
    }
    Ok(())
}

fn check_eligibility(config: &NemotronConfig, model: &NemotronModel) -> Result<(), String> {
    if model.layers.len() != config.n_layer {
        return Err("layer count mismatch".into());
    }
    for (layer_index, layer) in model.layers.iter().enumerate() {
        let is_attention = config.attention_layers.contains(&layer_index);
        let is_ffn = config.ffn_layers.contains(&layer_index);
        if is_attention {
            for weight in [&layer.wq, &layer.wk, &layer.wv, &layer.wo] {
                if let Some(weight) = weight {
                    GpuWeightFormat::from_ggml_type(weight.ggml_type)
                        .map_err(|error| error.to_string())?;
                } else {
                    return Err(format!("layer {layer_index} missing attention weight"));
                }
            }
        } else if is_ffn {
            if let Some(moe) = &layer.moe {
                // The routed experts and the shared expert. The router
                // itself is F32, which the device matmul takes as it is.
                check_expert_format(moe.up_type, "MoE expert up")?;
                check_expert_format(moe.down_type, "MoE expert down")?;
                let moe_cfg = config.moe.as_ref().ok_or_else(|| {
                    format!("layer {layer_index} has MoE weights but no MoE config")
                })?;
                if moe.n_expert != moe_cfg.n_expert
                    || moe.n_ff_exp != moe_cfg.n_ff_exp
                    || moe.n_embd != config.n_embd
                {
                    return Err(format!("layer {layer_index} MoE dimensions disagree"));
                }
                for (name, weight) in [
                    ("router", &moe.gate_inp),
                    ("shared expert up", &moe.up_shexp),
                    ("shared expert down", &moe.down_shexp),
                ] {
                    GpuWeightFormat::from_ggml_type(weight.ggml_type)
                        .map_err(|error| format!("layer {layer_index} {name}: {error}"))?;
                }
                if moe.exp_probs_b.len() != moe_cfg.n_expert {
                    return Err(format!("layer {layer_index} has a short selection bias"));
                }
            } else {
                for weight in [&layer.w_up, &layer.w_down] {
                    if let Some(weight) = weight {
                        GpuWeightFormat::from_ggml_type(weight.ggml_type)
                            .map_err(|error| error.to_string())?;
                    } else {
                        return Err(format!("layer {layer_index} missing FFN weight"));
                    }
                }
            }
        } else {
            // Mamba-2 SSM layer.
            for weight in [&layer.ssm_in, &layer.ssm_out] {
                if let Some(weight) = weight {
                    GpuWeightFormat::from_ggml_type(weight.ggml_type)
                        .map_err(|error| error.to_string())?;
                } else {
                    return Err(format!("layer {layer_index} missing SSM weight"));
                }
            }
            let ssm = SsmShape::new(config).map_err(|error| error.to_string())?;
            for (name, values, expected) in [
                (
                    "ssm_conv1d_w",
                    layer.ssm_conv1d_w.as_ref(),
                    ssm.conv_cols * ssm.kernel,
                ),
                ("ssm_conv1d_b", layer.ssm_conv1d_b.as_ref(), ssm.conv_cols),
                ("ssm_dt_bias", layer.ssm_dt_bias.as_ref(), ssm.heads),
                ("ssm_a_log", layer.ssm_a_log.as_ref(), ssm.heads),
                ("ssm_d", layer.ssm_d.as_ref(), ssm.heads),
                ("ssm_norm", layer.ssm_norm.as_ref(), ssm.inner),
            ] {
                let values =
                    values.ok_or_else(|| format!("layer {layer_index} missing {name} weight"))?;
                if values.len() != expected {
                    return Err(format!(
                        "layer {layer_index} {name} has {} values, expected {expected}",
                        values.len()
                    ));
                }
            }
        }
    }
    Ok(())
}

impl NemotronHVulkanSession {
    pub(crate) fn try_new_rows(
        model: &NemotronModel,
        capacity: usize,
        max_rows: usize,
        context: &'static VulkanContext,
    ) -> Result<Option<Self>, VulkanError> {
        if let Err(reason) = check_device_eligibility(context.supports_shader_float16())
            .and_then(|()| check_eligibility(&model.config, model))
        {
            eprintln!("[GPU] Nemotron-H Vulkan unavailable: {reason}. Falling back to CPU.");
            return Ok(None);
        }
        let config = model.config.clone();
        let ssm = SsmShape::new(&config)?;
        let layout = NemotronHArenaLayout::new(&config, capacity, max_rows)?;
        let descriptor_capacity = model.layers.iter().try_fold(3usize, |count, _| {
            count.checked_add(6).ok_or(VulkanError::OutOfMemory)
        })?;
        let mut ops =
            Qwen3Ops::new_device_local_with_size(context, layout.total_size, descriptor_capacity)?;
        let relu_square_pipeline =
            ops.add_shader(include_bytes!("../../shaders/bin/relu_square.spv"))?;
        let mut buffers = UploadedBuffers::new(context);
        let mut layers = Vec::with_capacity(config.n_layer);
        let mut ssm_slots: Vec<Option<usize>> = vec![None; config.n_layer];
        let mut ssm_slot = 0usize;
        let mut moe_slots: Vec<Option<usize>> = vec![None; config.n_layer];
        let mut moe_slot = 0usize;

        for (layer_index, layer) in model.layers.iter().enumerate() {
            let norm_buffer = buffers.upload_f32(&layer.attn_norm)?;
            let norm = ops.bind_buffers(&[norm_buffer])?;
            let is_attention = config.attention_layers.contains(&layer_index);
            let is_ffn = config.ffn_layers.contains(&layer_index);
            if is_attention {
                let wq = required_weight(layer.wq.as_ref(), layer_index, "attn_q")?;
                let wk = required_weight(layer.wk.as_ref(), layer_index, "attn_k")?;
                let wv = required_weight(layer.wv.as_ref(), layer_index, "attn_v")?;
                let wo = required_weight(layer.wo.as_ref(), layer_index, "attn_output")?;
                let q = bind_weight(&mut ops, &mut buffers, wq, "Nemotron-H Q")?;
                let k = bind_weight(&mut ops, &mut buffers, wk, "Nemotron-H K")?;
                let v = bind_weight(&mut ops, &mut buffers, wv, "Nemotron-H V")?;
                // Q/K norm weights are optional and packed into one
                // descriptor set: the shader reads Q from binding 0 and
                // K from binding 1.
                let qk_norm = match (layer.attn_q_norm.as_ref(), layer.attn_k_norm.as_ref()) {
                    (Some(qn), Some(kn)) => {
                        let q_buffer = buffers.upload_f32(qn)?;
                        let k_buffer = buffers.upload_f32(kn)?;
                        Some(ops.bind_buffers(&[q_buffer, k_buffer])?)
                    }
                    (None, None) => None,
                    _ => {
                        return Err(VulkanError::UnsupportedShape(format!(
                            "Nemotron-H layer {layer_index} has only one of Q/K norm"
                        )))
                    }
                };
                let output =
                    bind_weight(&mut ops, &mut buffers, wo, "Nemotron-H attention output")?;
                layers.push(LayerBindings::Attention {
                    norm,
                    attention: AttentionBindings {
                        q,
                        k,
                        v,
                        qk_norm,
                        output,
                    },
                });
            } else if is_ffn {
                if let Some(moe_weights) = layer.moe.as_ref() {
                    // A routed-expert layer: the dense up/down weights are
                    // absent and the experts take their place.
                    let router = bind_weight(
                        &mut ops,
                        &mut buffers,
                        &moe_weights.gate_inp,
                        "Nemotron-H MoE router",
                    )?;
                    let up_exps = bind_expert_tensor(
                        &mut ops,
                        &mut buffers,
                        moe_weights.up_exps,
                        moe_weights.up_type,
                    )?;
                    let down_exps = bind_expert_tensor(
                        &mut ops,
                        &mut buffers,
                        moe_weights.down_exps,
                        moe_weights.down_type,
                    )?;
                    let up_shexp = bind_weight(
                        &mut ops,
                        &mut buffers,
                        &moe_weights.up_shexp,
                        "Nemotron-H shared expert up",
                    )?;
                    let down_shexp = bind_weight(
                        &mut ops,
                        &mut buffers,
                        &moe_weights.down_shexp,
                        "Nemotron-H shared expert down",
                    )?;
                    // The selection bias is a per-layer constant, so it
                    // lives in the arena and is written once.
                    let bias = layout.moe_bias.get(moe_slot).copied().ok_or_else(|| {
                        VulkanError::UnsupportedShape(
                            "Nemotron-H MoE layer without a bias region".into(),
                        )
                    })?;
                    ops.write_f32(bias, &moe_weights.exp_probs_b)?;
                    moe_slots[layer_index] = Some(moe_slot);
                    moe_slot += 1;
                    layers.push(LayerBindings::Moe {
                        norm,
                        moe: MoeBindings {
                            router,
                            up_exps,
                            down_exps,
                            up_shexp,
                            down_shexp,
                        },
                    });
                    continue;
                }
                let w_up = required_weight(layer.w_up.as_ref(), layer_index, "ffn_up")?;
                let w_down = required_weight(layer.w_down.as_ref(), layer_index, "ffn_down")?;
                let up = bind_weight(&mut ops, &mut buffers, w_up, "Nemotron-H FFN up")?;
                let down = bind_weight(&mut ops, &mut buffers, w_down, "Nemotron-H FFN down")?;
                layers.push(LayerBindings::Ffn {
                    norm,
                    ffn: FfnBindings { up, down },
                });
            } else {
                // Mamba-2 SSM layer.
                let ssm_in = required_weight(layer.ssm_in.as_ref(), layer_index, "ssm_in")?;
                let ssm_out = required_weight(layer.ssm_out.as_ref(), layer_index, "ssm_out")?;
                let input_projection = bind_weight(
                    &mut ops,
                    &mut buffers,
                    ssm_in,
                    "Nemotron-H SSM in projection",
                )?;
                let output_projection = bind_weight(
                    &mut ops,
                    &mut buffers,
                    ssm_out,
                    "Nemotron-H SSM out projection",
                )?;
                let group_norm_buffer = buffers.upload_f32(required_ssm_vector(
                    layer.ssm_norm.as_ref(),
                    layer_index,
                    "ssm_norm",
                    ssm.inner,
                )?)?;
                let group_norm = ops.bind_buffers(&[group_norm_buffer])?;
                // The F32 weights live in the arena; the scan state and the
                // conv history are zeroed here because they are the
                // recurrence's initial condition.
                let arena = layout.ssm_layers[ssm_slot];
                ops.write_f32(
                    arena.conv_weights,
                    required_ssm_vector(
                        layer.ssm_conv1d_w.as_ref(),
                        layer_index,
                        "ssm_conv1d_w",
                        ssm.conv_cols * ssm.kernel,
                    )?,
                )?;
                ops.write_f32(
                    arena.conv_bias,
                    required_ssm_vector(
                        layer.ssm_conv1d_b.as_ref(),
                        layer_index,
                        "ssm_conv1d_b",
                        ssm.conv_cols,
                    )?,
                )?;
                ops.write_f32(
                    arena.dt_bias,
                    required_ssm_vector(
                        layer.ssm_dt_bias.as_ref(),
                        layer_index,
                        "ssm_dt_bias",
                        ssm.heads,
                    )?,
                )?;
                ops.write_f32(
                    arena.a_log,
                    required_ssm_vector(
                        layer.ssm_a_log.as_ref(),
                        layer_index,
                        "ssm_a_log",
                        ssm.heads,
                    )?,
                )?;
                ops.write_f32(
                    arena.d,
                    required_ssm_vector(layer.ssm_d.as_ref(), layer_index, "ssm_d", ssm.heads)?,
                )?;
                ops.zero_region(arena.state)?;
                ops.zero_region(arena.history)?;
                ssm_slots[layer_index] = Some(ssm_slot);
                ssm_slot += 1;
                layers.push(LayerBindings::Ssm {
                    norm,
                    ssm: SsmBindings {
                        input_projection,
                        output_projection,
                        group_norm,
                    },
                });
            }
        }

        let output_norm_buffer = buffers.upload_f32(&model.output_norm)?;
        let output_norm = ops.bind_buffers(&[output_norm_buffer])?;
        let output = bind_weight(&mut ops, &mut buffers, &model.output, "Nemotron-H output")?;
        let rope_dim = config.rope_dim;

        Ok(Some(Self {
            context,
            ops,
            _buffers: buffers,
            layers,
            output_norm,
            output,
            layout,
            config,
            ssm,
            ssm_slots,
            moe_slots,
            capacity,
            max_rows,
            rope: vec![0.0; product("RoPE rows", &[rope_dim, max_rows])?],
            logits: vec![0.0; model.config.vocab_size],
            committed_position: 0,
            relu_square_pipeline,
        }))
    }

    /// Run a contiguous range of device-recorded layers over the given
    /// hidden-state rows.
    ///
    /// The whole stack is recorded in one call. The hidden state is
    /// written to and read back from the arena, so the host can run a
    /// CPU-only branch in between two calls (the `RMI_NH_CPU_*` overrides
    /// exist for exactly that).
    pub(crate) fn forward_layers(
        &mut self,
        input: &[f32],
        base_position: usize,
        positions: &[usize],
        rows: usize,
        layer_start: usize,
        layer_end: usize,
    ) -> Result<(), VulkanError> {
        let config = &self.config;
        let n_embd = config.n_embd;
        let input_count = product("input rows", &[rows, n_embd])?;
        if rows > self.max_rows
            || positions.len() != rows
            || input.len() != input_count
            || input.iter().any(|v| !v.is_finite())
            || layer_start >= layer_end
            || layer_end > config.n_layer
        {
            return Err(VulkanError::UnsupportedShape(format!(
                "invalid Nemotron-H token input={} position={base_position}/{} layers={layer_start}..{layer_end}",
                input.len(),
                self.capacity
            )));
        }
        self.ops.write_f32(self.layout.x, input)?;
        // RoPE is only applied when the layer has Q/K norm weights; the
        // 4B Nano attention layers ship neither, so the rope buffer is
        // left untouched for them.
        let has_rope = self
            .layers
            .iter()
            .any(|layer| matches!(layer, LayerBindings::Attention { attention, .. } if attention.qk_norm.is_some()));
        if has_rope && config.rope_dim > 0 {
            for (row, &position) in positions.iter().enumerate() {
                let coeff = &mut self.rope[row * config.rope_dim..(row + 1) * config.rope_dim];
                coeff[..config.rope_dim / 2].fill(1.0);
                crate::ops::rope_neox_inplace_precise(
                    coeff,
                    position,
                    config.rope_dim,
                    config.rope_freq_base,
                );
            }
            self.ops
                .write_f32(self.layout.rope, &self.rope[..rows * config.rope_dim])?;
        }
        let commands = TokenCommands::begin(self.context)?;
        for layer_index in layer_start..layer_end {
            let bindings = &self.layers[layer_index];
            match bindings {
                LayerBindings::Attention { norm, attention } => {
                    self.ops.record_rms_norm_rows(
                        &commands,
                        *norm,
                        self.layout.x,
                        self.layout.normed,
                        n_embd,
                        config.norm_eps,
                        rows,
                        n_embd,
                        n_embd,
                    )?;
                    self.record_qkv(&commands, attention, rows)?;
                    self.record_attention(&commands, attention, layer_index, base_position, rows)?;
                    self.ops.record_add_rows(
                        &commands,
                        self.layout.x,
                        self.layout.projection,
                        n_embd,
                        rows,
                    )?;
                }
                LayerBindings::Ffn { norm, ffn } => {
                    self.ops.record_rms_norm_rows(
                        &commands,
                        *norm,
                        self.layout.x,
                        self.layout.normed,
                        n_embd,
                        config.norm_eps,
                        rows,
                        n_embd,
                        n_embd,
                    )?;
                    self.record_ffn(&commands, ffn, rows)?;
                    self.ops.record_add_rows(
                        &commands,
                        self.layout.x,
                        self.layout.ffn_down,
                        n_embd,
                        rows,
                    )?;
                }
                LayerBindings::Moe { norm, moe } => {
                    self.ops.record_rms_norm_rows(
                        &commands,
                        *norm,
                        self.layout.x,
                        self.layout.normed,
                        n_embd,
                        config.norm_eps,
                        rows,
                        n_embd,
                        n_embd,
                    )?;
                    // The reduction adds the routed experts and the shared
                    // expert to the residual on its own.
                    self.record_moe(&commands, moe, layer_index, rows)?;
                }
                LayerBindings::Ssm {
                    norm,
                    ssm: ssm_bindings,
                } => {
                    // The pre-norm runs for every row at once.
                    self.ops.record_rms_norm_rows(
                        &commands,
                        *norm,
                        self.layout.x,
                        self.layout.normed,
                        n_embd,
                        config.norm_eps,
                        rows,
                        n_embd,
                        n_embd,
                    )?;
                    // The in-projection is a plain matmul, so it runs for
                    // every row at once. Its output is
                    // `[z | x | B | C | dt]`.
                    self.ops.record_weight_matmul_rows(
                        &commands,
                        ssm_bindings.input_projection,
                        self.layout.normed,
                        self.layout.q8,
                        self.layout.q8_scales,
                        self.layout.q4_1_input_sums,
                        self.layout.q8k,
                        self.layout.q8k_scales,
                        &[(
                            self.layout.ssm_in_projection,
                            self.ssm.in_projection,
                            self.ssm.in_projection * 4,
                        )],
                        n_embd,
                        rows,
                        n_embd,
                    )?;
                    // The conv1d, the scan and the group norm carry the
                    // recurrence, so they run one token at a time: the
                    // conv history and the scan state belong to the
                    // sequence, not to the row.
                    for row in 0..rows {
                        self.record_ssm_row(
                            &commands,
                            layer_index,
                            ssm_bindings,
                            row,
                            base_position + row,
                        )?;
                    }
                    // The output projection is a plain matmul again.
                    self.ops.record_weight_matmul_rows(
                        &commands,
                        ssm_bindings.output_projection,
                        self.layout.ssm_normed,
                        self.layout.q8,
                        self.layout.q8_scales,
                        self.layout.q4_1_input_sums,
                        self.layout.q8k,
                        self.layout.q8k_scales,
                        &[(self.layout.ssm_projection, n_embd, n_embd * 4)],
                        self.ssm.inner,
                        rows,
                        self.ssm.inner,
                    )?;
                    self.ops.record_add_rows(
                        &commands,
                        self.layout.x,
                        self.layout.ssm_projection,
                        n_embd,
                        rows,
                    )?;
                }
            }
        }
        commands.submit_and_wait()?;
        Ok(())
    }

    /// Read the hidden state back from the arena after a `forward_layers`
    /// call, so the host can run the CPU SSM layers in between.
    pub(crate) fn read_hidden(&self, rows: usize) -> Result<Vec<f32>, VulkanError> {
        let mut hidden = vec![0.0; rows * self.config.n_embd];
        self.ops.read_f32_into(self.layout.x, &mut hidden)?;
        Ok(hidden)
    }

    /// The number of tokens the device's recurrent state covers.
    pub(crate) fn committed_position(&self) -> usize {
        self.committed_position
    }

    /// The number of rows the arena's row buffers hold.
    ///
    /// A call that passes more rows than this would write past the
    /// arena, so the host has to know the limit before it hands a
    /// prompt to the device.
    pub(crate) fn max_rows(&self) -> usize {
        self.max_rows
    }

    /// Record that the device's recurrent state now covers `position`
    /// tokens.
    pub(crate) fn set_committed_position(&mut self, position: usize) {
        self.committed_position = position;
    }

    /// Clear the recurrent state: the KV cache, the Mamba-2 conv history
    /// and the Mamba-2 scan state.
    ///
    /// The host's scratch pad carries the same state, and a fresh scratch
    /// pad means a fresh sequence. Without this the device would carry the
    /// previous sequence's recurrence into the new one, and the conv
    /// history alone is enough to move every SSM layer's output.
    pub(crate) fn reset(&mut self) -> Result<(), VulkanError> {
        let mut regions = vec![
            self.layout.kv_k,
            self.layout.kv_v,
            self.layout.kv_delta_k,
            self.layout.kv_delta_v,
        ];
        for arena in &self.layout.ssm_layers {
            regions.push(arena.state);
            regions.push(arena.history);
        }
        for region in regions {
            self.ops.zero_region(region)?;
        }
        self.committed_position = 0;
        Ok(())
    }

    /// The logits produced by the last `finish_logits` call.
    pub(crate) fn logits(&self) -> &[f32] {
        &self.logits
    }

    /// Write the hidden state back into the arena before the next
    /// `forward_layers` call.
    pub(crate) fn write_hidden(&self, hidden: &[f32]) -> Result<(), VulkanError> {
        self.ops.write_f32(self.layout.x, hidden)
    }

    /// Run the final norm and the output projection, returning the logits.
    pub(crate) fn finish_logits(&mut self, rows: usize) -> Result<(), VulkanError> {
        let config = &self.config;
        let n_embd = config.n_embd;
        let last_offset = (rows - 1) * n_embd * 4;
        let last = ArenaRegion {
            offset: self.layout.x.offset + last_offset,
            size: self.layout.x.size - last_offset,
        };
        let commands = TokenCommands::begin(self.context)?;
        self.ops.record_rms_norm(
            &commands,
            self.output_norm,
            last,
            self.layout.normed,
            n_embd,
            config.norm_eps,
        )?;
        commands.submit_and_wait()?;
        let commands = TokenCommands::begin(self.context)?;
        self.record_weights(
            &commands,
            self.output,
            self.layout.normed,
            &[(self.layout.logits, config.vocab_size)],
            n_embd,
            1,
        )?;
        commands.submit_and_wait()?;
        self.ops
            .read_f32_into(self.layout.logits, &mut self.logits)?;
        if self.logits.iter().any(|v| !v.is_finite()) {
            return Err(VulkanError::UnsupportedShape(
                "Nemotron-H Vulkan chunk produced non-finite logits".into(),
            ));
        }
        Ok(())
    }

    /// Record one token of a Mamba-2 SSM layer.
    ///
    /// The order follows the CPU path (`src/models/nemotron_h/trunk/forward.rs`):
    /// the in-projection has already run for every row, and this records the
    /// conv1d, the time step, the scan, the z gate and the group norm for
    /// one token.
    fn record_ssm_row(
        &self,
        commands: &TokenCommands<'_>,
        layer_index: usize,
        ssm_bindings: &SsmBindings,
        row: usize,
        position: usize,
    ) -> Result<(), VulkanError> {
        let ssm = self.ssm;
        let slot = self.ssm_slots[layer_index].ok_or_else(|| {
            VulkanError::UnsupportedShape("Nemotron-H SSM layer has no arena slot".into())
        })?;
        let arena = self.layout.ssm_layers[slot];
        let row_base = row * ssm.in_projection;
        let in_projection = self.layout.ssm_in_projection;
        let conv_output = ArenaRegion {
            offset: self.layout.ssm_conv.offset + row * ssm.conv_cols * 4,
            size: ssm.conv_cols * 4,
        };
        let dt_output = ArenaRegion {
            offset: self.layout.ssm_dt.offset + row * ssm.heads * 4,
            size: ssm.heads * 4,
        };
        let decay_output = ArenaRegion {
            offset: self.layout.ssm_decay.offset + row * ssm.heads * 4,
            size: ssm.heads * 4,
        };
        let scan_output = ArenaRegion {
            offset: self.layout.ssm_scan.offset + row * ssm.inner * 4,
            size: ssm.inner * 4,
        };
        let gated_output = ArenaRegion {
            offset: self.layout.ssm_gated.offset + row * ssm.inner * 4,
            size: ssm.inner * 4,
        };

        // The in-projection output is `[z | x | B | C | dt]`; the conv1d
        // reads the contiguous `[x | B | C]` part.
        let conv_input = ArenaRegion {
            offset: in_projection.offset + (row_base + ssm.inner) * 4,
            size: ssm.conv_cols * 4,
        };
        self.ops.record_nemotron_h_ssm_conv(
            commands,
            conv_input,
            conv_output,
            arena.conv_weights,
            arena.conv_bias,
            arena.history,
            ssm.conv_cols,
            ssm.kernel,
            position % 3,
        )?;

        // The time step is the tail of the in-projection output.
        let dt_input = ArenaRegion {
            offset: in_projection.offset + (row_base + ssm.inner + ssm.conv_cols) * 4,
            size: ssm.heads * 4,
        };
        self.ops.record_nemotron_h_ssm_dt(
            commands,
            dt_input,
            arena.dt_bias,
            arena.a_log,
            dt_output,
            decay_output,
            ssm.heads,
        )?;

        // The scan reads `x`, `B` and `C` out of the conv1d output.
        let group_width = ssm.groups * ssm.d_state * 4;
        let x = ArenaRegion {
            offset: conv_output.offset,
            size: ssm.inner * 4,
        };
        let b = ArenaRegion {
            offset: conv_output.offset + ssm.inner * 4,
            size: group_width,
        };
        let c = ArenaRegion {
            offset: conv_output.offset + ssm.inner * 4 + group_width,
            size: group_width,
        };
        self.ops.record_nemotron_h_ssm_scan(
            commands,
            arena.state,
            x,
            b,
            c,
            dt_output,
            decay_output,
            arena.d,
            scan_output,
            ssm.inner,
            ssm.heads,
            ssm.headdim,
            ssm.d_state,
            ssm.groups,
        )?;

        // The z gate reads the first `d_inner` values of the
        // in-projection row.
        let z = ArenaRegion {
            offset: in_projection.offset + row_base * 4,
            size: ssm.inner * 4,
        };
        self.ops
            .record_nemotron_h_ssm_gate(commands, z, scan_output, gated_output, ssm.inner)?;

        // The group norm writes into this row's slot of the normalised
        // buffer, which the output projection reads.
        let normed = ArenaRegion {
            offset: self.layout.ssm_normed.offset + row * ssm.inner * 4,
            size: ssm.inner * 4,
        };
        self.ops.record_nemotron_h_ssm_norm(
            commands,
            ssm_bindings.group_norm,
            gated_output,
            normed,
            ssm.group_len,
            ssm.groups,
            self.config.norm_eps,
        )
    }

    fn record_qkv(
        &self,
        commands: &TokenCommands<'_>,
        attention: &AttentionBindings,
        rows: usize,
    ) -> Result<(), VulkanError> {
        let config = &self.config;
        let n_embd = config.n_embd;
        let n_attn_q = config.n_head * config.n_embd_head_k;
        let n_attn_kv = config.n_head_kv * config.n_embd_head_k;
        let n_attn_v_kv = config.n_head_kv * config.n_embd_head_v;
        // Three separate single-weight matmuls rather than one grouped
        // three-weight matmul. The grouped kernel writes each output with
        // its own row count, and the Q/K/V widths differ (5120/1024/1024),
        // so the split keeps every output on the plain one-weight path.
        for (bindings, region, n_out) in [
            (attention.q, self.layout.q, n_attn_q),
            (attention.k, self.layout.k, n_attn_kv),
            (attention.v, self.layout.v, n_attn_v_kv),
        ] {
            self.ops.record_weight_matmul_rows(
                commands,
                bindings,
                self.layout.normed,
                self.layout.q8,
                self.layout.q8_scales,
                self.layout.q4_1_input_sums,
                self.layout.q8k,
                self.layout.q8k_scales,
                &[(region, n_out, n_out * 4)],
                n_embd,
                rows,
                n_embd,
            )?;
        }
        Ok(())
    }

    fn record_attention(
        &self,
        commands: &TokenCommands<'_>,
        attention: &AttentionBindings,
        layer_index: usize,
        base_position: usize,
        rows: usize,
    ) -> Result<(), VulkanError> {
        let config = &self.config;
        let n_embd = config.n_embd;
        let n_attn_q = config.n_head * config.n_embd_head_k;
        let n_attn_kv = config.n_head_kv * config.n_embd_head_k;
        let slot = self.layout.slots[layer_index].ok_or_else(|| {
            VulkanError::UnsupportedShape("attention layer has no KV cache slot".into())
        })?;
        let layer_count = self.config.attention_layers.len();
        let cache_k = self.layout.kv_k;
        let cache_v = self.layout.kv_v;
        // Q/K norm + RoPE. The 4B Nano attention layers ship neither, so
        // both are skipped when the weights are absent.
        if let Some(qk_norm) = attention.qk_norm {
            self.ops.record_qk_norm_rope_rows(
                commands,
                qk_norm,
                self.layout.q,
                self.layout.k,
                config.n_head,
                config.n_head_kv,
                config.n_embd_head_k,
                self.layout.rope,
                config.norm_eps,
                true,
                true,
                rows,
            )?;
        }
        // The KV cache keeps F32. llama.cpp's default cache type is F16, but
        // this model's reference harness runs with an F32 cache, and the F16
        // rounding is amplified by the Mamba-2 layers that follow each
        // attention layer.
        self.ops.record_kv_write_rows_f32(
            commands,
            self.layout.k,
            self.layout.v,
            cache_k,
            cache_v,
            self.layout.kv_delta_k,
            self.layout.kv_delta_v,
            slot,
            base_position,
            layer_count,
            self.capacity,
            n_attn_kv,
            rows,
            n_attn_kv,
        )?;
        // Scores, softmax and values all stay in F32, matching the CPU path.
        self.ops.record_attention_scores_f32_rows(
            commands,
            self.layout.q,
            cache_k,
            self.layout.scores,
            slot,
            layer_count,
            base_position,
            self.capacity,
            config.n_head,
            config.n_head_kv,
            config.n_embd_head_k,
            rows,
        )?;
        self.ops.record_softmax_f32_rows(
            commands,
            self.layout.scores,
            config.n_head,
            base_position,
            rows,
            false,
        )?;
        self.ops.record_attention_values_f32_kv_rows(
            commands,
            self.layout.scores,
            cache_v,
            self.layout.attn,
            slot,
            layer_count,
            base_position,
            self.capacity,
            config.n_head,
            config.n_head_kv,
            config.n_embd_head_k,
            rows,
            false,
        )?;
        self.ops.record_weight_matmul_rows(
            commands,
            attention.output,
            self.layout.attn,
            self.layout.q8,
            self.layout.q8_scales,
            self.layout.q4_1_input_sums,
            self.layout.q8k,
            self.layout.q8k_scales,
            &[(self.layout.projection, n_embd, n_embd * 4)],
            n_attn_q,
            rows,
            n_attn_q,
        )
    }

    fn record_ffn(
        &self,
        commands: &TokenCommands<'_>,
        ffn: &FfnBindings,
        rows: usize,
    ) -> Result<(), VulkanError> {
        let config = &self.config;
        let n_embd = config.n_embd;
        let n_ff = config.n_ff;
        self.ops.record_weight_matmul_rows(
            commands,
            ffn.up,
            self.layout.normed,
            self.layout.q8,
            self.layout.q8_scales,
            self.layout.q4_1_input_sums,
            self.layout.q8k,
            self.layout.q8k_scales,
            &[(self.layout.ffn_up, n_ff, n_ff * 4)],
            n_embd,
            rows,
            n_embd,
        )?;
        // ReLU(x)^2: llama.cpp's LLM_FFN_RELU_SQR. The shader squares the
        // clamped value, so the result is `max(x, 0)^2`.
        let push = [
            Self::word(self.layout.ffn_up),
            n_ff as u32,
            rows as u32,
            n_ff as u32,
        ];
        let [x, y, z] = row_dispatch(n_ff.div_ceil(64), rows, &self.context.limits)?;
        self.ops
            .record_custom(commands, self.relu_square_pipeline, None, &push, [x, y, z])?;
        self.ops.record_weight_matmul_rows(
            commands,
            ffn.down,
            self.layout.ffn_up,
            self.layout.q8,
            self.layout.q8_scales,
            self.layout.q4_1_input_sums,
            self.layout.q8k,
            self.layout.q8k_scales,
            &[(self.layout.ffn_down, n_embd, n_embd * 4)],
            n_ff,
            rows,
            n_ff,
        )
    }

    /// Record one mixture-of-experts layer.
    ///
    /// The layer follows the reference (`src/models/nemotron-h.cpp::
    /// build_ffn_layer`):
    ///
    /// ```text
    /// logits    = ffn_gate_inp @ cur                 # [n_expert]
    /// probs     = sigmoid(logits)
    /// selection = probs + exp_probs_b.bias           # the bias ranks only
    /// chosen    = topk(selection, n_expert_used)
    /// weights   = probs[chosen] / max(sum, 2^-14) * scale
    /// up        = relu(ffn_up_exps[chosen] @ cur) ** 2
    /// down      = ffn_down_exps[chosen] @ up
    /// out       = sum_i weights[i] * down_i
    ///            + ffn_down_shexp @ relu(ffn_up_shexp @ cur) ** 2
    /// ```
    ///
    /// The routing stays on the device: the chosen ids are written into
    /// the arena and the expert kernels read them from there, so a layer
    /// does not synchronise with the host. The shared expert has no gate
    /// and no weight, so the reducer adds it as it is.
    fn record_moe(
        &self,
        commands: &TokenCommands<'_>,
        moe: &MoeBindings,
        layer_index: usize,
        rows: usize,
    ) -> Result<(), VulkanError> {
        let config = &self.config;
        let moe_cfg = config.moe.as_ref().ok_or_else(|| {
            VulkanError::UnsupportedShape("Nemotron-H MoE layer without MoE metadata".into())
        })?;
        let n_embd = config.n_embd;
        let n_expert = moe_cfg.n_expert;
        let used = moe_cfg.n_expert_used;
        let n_ff_exp = moe_cfg.n_ff_exp;
        let n_ff_shexp = moe_cfg.n_ff_shexp;

        // The router: `ffn_gate_inp`, F32 [n_embd, n_expert].
        self.record_weights(
            commands,
            moe.router,
            self.layout.normed,
            &[(self.layout.router_logits, n_expert)],
            n_embd,
            rows,
        )?;

        // The routing, entirely on the device.
        let slot = self.moe_slots[layer_index].ok_or_else(|| {
            VulkanError::UnsupportedShape("Nemotron-H MoE layer has no arena slot".into())
        })?;
        self.ops.record_nemotron_moe_route(
            commands,
            self.layout.router_logits,
            self.layout.router_probs,
            self.layout.router_selection,
            self.layout.expert_ids,
            self.layout.expert_weights,
            self.layout.moe_bias[slot],
            n_expert,
            used,
            moe_cfg.expert_weights_scale,
            moe_cfg.expert_weights_norm,
            rows,
        )?;

        // The routed experts. One dispatch per slot, each covering every
        // row: the kernel reads the expert the routing chose for that row
        // and offsets the weight by one expert matrix.
        for expert_slot in 0..used {
            let output = self.layout.expert_up.shifted(expert_slot * n_ff_exp * 4);
            self.ops.record_selected_weight_matmul_rows(
                commands,
                moe.up_exps,
                self.layout.normed,
                self.layout.q8,
                self.layout.q8_scales,
                self.layout.q4_1_input_sums,
                self.layout.q8k,
                self.layout.q8k_scales,
                &[(output, n_ff_exp, used * n_ff_exp * 4)],
                n_embd,
                rows,
                n_embd,
                Some(ExpertSelection {
                    ids: self.layout.expert_ids,
                    count: n_expert,
                    used,
                    slot: expert_slot,
                }),
            )?;
        }
        // ReLU(x)^2 on the expert up projections. The buffer is laid out
        // as [row][slot][n_ff_exp], so one row of the activation is the
        // whole slot block of a token.
        self.record_relu_square(
            commands,
            self.layout.expert_up,
            used * n_ff_exp,
            rows,
            used * n_ff_exp,
        )?;

        // The expert down projections read the activations of their own
        // slot, so the input region starts at that slot's offset and the
        // row stride stays the width of a whole row.
        for expert_slot in 0..used {
            let input = self.layout.expert_up.shifted(expert_slot * n_ff_exp * 4);
            let output = self.layout.expert_down.shifted(expert_slot * n_embd * 4);
            self.ops.record_selected_weight_matmul_rows(
                commands,
                moe.down_exps,
                input,
                self.layout.q8,
                self.layout.q8_scales,
                self.layout.q4_1_input_sums,
                self.layout.q8k,
                self.layout.q8k_scales,
                &[(output, n_embd, used * n_embd * 4)],
                n_ff_exp,
                rows,
                used * n_ff_exp,
                Some(ExpertSelection {
                    ids: self.layout.expert_ids,
                    count: n_expert,
                    used,
                    slot: expert_slot,
                }),
            )?;
        }

        // The shared expert: no gate, no weight.
        self.record_weights(
            commands,
            moe.up_shexp,
            self.layout.normed,
            &[(self.layout.shared_up, n_ff_shexp)],
            n_embd,
            rows,
        )?;
        self.record_relu_square(
            commands,
            self.layout.shared_up,
            n_ff_shexp,
            rows,
            n_ff_shexp,
        )?;
        self.record_weights(
            commands,
            moe.down_shexp,
            self.layout.shared_up,
            &[(self.layout.shared_down, n_embd)],
            n_ff_shexp,
            rows,
        )?;

        // The reduction: the weighted sum of the routed experts plus the
        // shared expert, added to the residual.
        self.ops.record_nemotron_moe_reduce(
            commands,
            self.layout.x,
            self.layout.expert_down,
            self.layout.expert_weights,
            self.layout.shared_down,
            n_embd,
            used,
            rows,
            true,
        )
    }

    /// Record the `ReLU(x)^2` activation over a region.
    ///
    /// llama.cpp builds the Nemotron-H FFN as `LLM_FFN_RELU_SQR`
    /// (`src/models/nemotron-h.cpp`): the projection is followed by
    /// `max(x, 0)^2`. The shader squares the clamped value, so the result
    /// is `max(x, 0)^2`.
    fn record_relu_square(
        &self,
        commands: &TokenCommands<'_>,
        region: ArenaRegion,
        width: usize,
        rows: usize,
        row_stride: usize,
    ) -> Result<(), VulkanError> {
        let push = [
            Self::word(region),
            width as u32,
            rows as u32,
            row_stride as u32,
        ];
        let [x, y, z] = row_dispatch(width.div_ceil(64), rows, &self.context.limits)?;
        self.ops
            .record_custom(commands, self.relu_square_pipeline, None, &push, [x, y, z])
    }

    fn record_weights(
        &self,
        commands: &TokenCommands<'_>,
        bindings: OperatorBindings,
        input: ArenaRegion,
        outputs: &[(ArenaRegion, usize)],
        n_in: usize,
        rows: usize,
    ) -> Result<(), VulkanError> {
        let outputs = outputs
            .iter()
            .map(|&(region, width)| {
                Ok((
                    region,
                    width,
                    width.checked_mul(4).ok_or(VulkanError::OutOfMemory)?,
                ))
            })
            .collect::<Result<Vec<_>, VulkanError>>()?;
        self.ops.record_weight_matmul_rows(
            commands,
            bindings,
            input,
            self.layout.q8,
            self.layout.q8_scales,
            self.layout.q4_1_input_sums,
            self.layout.q8k,
            self.layout.q8k_scales,
            &outputs,
            n_in,
            rows,
            n_in,
        )
    }

    fn word(region: ArenaRegion) -> u32 {
        (region.offset / 4) as u32
    }
}

fn check_device_eligibility(shader_float16: bool) -> Result<(), String> {
    shader_float16
        .then_some(())
        .ok_or_else(|| "selected Vulkan device does not support shaderFloat16".into())
}

#[cfg(test)]
mod tests {
    //! Shader-level parity checks for the Mamba-2 SSM kernels.
    //!
    //! The full model test compares the logits, which is a poor debugging
    //! tool when the difference is one ULP in a SiLU: the recurrence
    //! amplifies it by four orders of magnitude and the logits are then
    //! far apart. These tests run each kernel against the CPU
    //! implementation on the same inputs and compare the bits, so a
    //! failure names the kernel that moved.

    use super::*;
    use crate::vulkan::ops::TokenCommands;

    /// A deterministic pseudo-random sequence, so a failure is
    /// reproducible.
    struct Rng(u64);

    impl Rng {
        fn next_u32(&mut self) -> u32 {
            // xorshift64*
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            (self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 32) as u32
        }

        /// A value in `[-scale, scale]`.
        fn value(&mut self, scale: f32) -> f32 {
            let raw = self.next_u32() as f32 / (u32::MAX as f32 + 1.0);
            (raw * 2.0 - 1.0) * scale
        }
    }

    struct Harness {
        ops: Qwen3Ops<'static>,
        cursor: usize,
    }

    impl Harness {
        fn new(context: &'static VulkanContext, bytes: usize) -> Self {
            let ops = Qwen3Ops::new_device_local_with_size(context, bytes, 8).expect("arena");
            Self { ops, cursor: 0 }
        }

        /// Reserve a region of `elements` F32 values.
        fn region(&mut self, elements: usize) -> ArenaRegion {
            let offset = (self.cursor + 15) & !15;
            self.cursor = offset + elements * 4;
            ArenaRegion {
                offset,
                size: elements * 4,
            }
        }
    }

    fn context() -> Option<&'static VulkanContext> {
        let context = Box::leak(Box::new(
            crate::vulkan::VulkanContext::new().expect("Vulkan context"),
        ));
        Some(context)
    }

    /// Compare two buffers bit for bit and report the first difference in
    /// ULPs, so a one-bit rounding difference is distinguishable from a
    /// structural one.
    fn assert_bits_match(actual: &[f32], expected: &[f32], label: &str) {
        assert_eq!(actual.len(), expected.len(), "{label} length");
        for (index, (&a, &b)) in actual.iter().zip(expected).enumerate() {
            if a.to_bits() != b.to_bits() {
                let ulp = (a.to_bits() as i64 - b.to_bits() as i64).abs();
                let relative = (f64::from(a) - f64::from(b)) / f64::from(b).abs().max(1e-30);
                panic!(
                    "{label} element {index} differs: actual={a:e} expected={b:e} \
                     ulp={ulp} relative={relative:e}"
                );
            }
        }
    }

    /// The CPU's `mamba2_conv_step`, transcribed.
    fn cpu_conv_step(
        weights: &[f32],
        bias: &[f32],
        history: &mut [f32],
        current: &[f32],
        kernel: usize,
    ) -> Vec<f32> {
        let cols = current.len();
        let mut output = vec![0.0; cols];
        for c in 0..cols {
            for k in 0..kernel - 1 {
                output[c] += weights[c * kernel + k] * history[k * cols + c];
            }
            output[c] += weights[c * kernel + kernel - 1] * current[c];
            output[c] += bias[c];
        }
        if kernel > 2 {
            history.copy_within(cols.., 0);
        }
        history[(kernel - 2) * cols..].copy_from_slice(current);
        output
    }

    #[test]
    fn ssm_conv_matches_cpu_bits() {
        let Some(context) = context() else { return };
        const COLS: usize = 9728;
        const KERNEL: usize = 4;
        let mut harness = Harness::new(context, 4 * 1024 * 1024);
        let input = harness.region(COLS);
        let output = harness.region(COLS);
        let weights = harness.region(COLS * KERNEL);
        let bias = harness.region(COLS);
        let history = harness.region(COLS * 3);
        let mut rng = Rng(0x1234_5678_9abc_def0);
        let weight_values: Vec<f32> = (0..COLS * KERNEL).map(|_| rng.value(0.5)).collect();
        let bias_values: Vec<f32> = (0..COLS).map(|_| rng.value(0.5)).collect();
        harness.ops.write_f32(weights, &weight_values).unwrap();
        harness.ops.write_f32(bias, &bias_values).unwrap();
        harness.ops.zero_region(history).unwrap();

        // Three tokens, so the conv history's ring wraps once.
        let tokens: Vec<Vec<f32>> = (0..3)
            .map(|_| (0..COLS).map(|_| rng.value(2.0)).collect())
            .collect();
        for (token, current) in tokens.iter().enumerate() {
            harness.ops.write_f32(input, current).unwrap();
            let commands = TokenCommands::begin(context).unwrap();
            harness
                .ops
                .record_nemotron_h_ssm_conv(
                    &commands,
                    input,
                    output,
                    weights,
                    bias,
                    history,
                    COLS,
                    KERNEL,
                    token % 3,
                )
                .unwrap();
            commands.submit_and_wait().unwrap();

            // The CPU path keeps a shifting history, so replay it from
            // the start for every token.
            let mut history_cpu = vec![0.0; COLS * (KERNEL - 1)];
            for previous in tokens.iter().take(token) {
                let _ = cpu_conv_step(
                    &weight_values,
                    &bias_values,
                    &mut history_cpu,
                    previous,
                    KERNEL,
                );
            }
            let mut expected = cpu_conv_step(
                &weight_values,
                &bias_values,
                &mut history_cpu,
                current,
                KERNEL,
            );
            crate::ops::activation::silu_ggml_inplace(&mut expected);
            let mut actual = vec![0.0; COLS];
            harness.ops.read_f32_into(output, &mut actual).unwrap();
            assert_bits_match(&actual, &expected, &format!("SSM conv1d token {token}"));
        }
    }

    #[test]
    fn ssm_dt_matches_cpu_bits() {
        let Some(context) = context() else { return };
        const HEADS: usize = 96;
        let mut harness = Harness::new(context, 64 * 1024);
        let dt = harness.region(HEADS);
        let dt_bias = harness.region(HEADS);
        let a_log = harness.region(HEADS);
        let out_dt = harness.region(HEADS);
        let out_decay = harness.region(HEADS);
        let mut rng = Rng(0x0fed_cba9_8765_4321);
        let dt_values: Vec<f32> = (0..HEADS).map(|_| rng.value(3.0)).collect();
        let bias_values: Vec<f32> = (0..HEADS).map(|_| rng.value(1.0)).collect();
        let a_log_values: Vec<f32> = (0..HEADS).map(|_| -rng.value(20.0)).collect();
        harness.ops.write_f32(dt, &dt_values).unwrap();
        harness.ops.write_f32(dt_bias, &bias_values).unwrap();
        harness.ops.write_f32(a_log, &a_log_values).unwrap();
        let commands = TokenCommands::begin(context).unwrap();
        harness
            .ops
            .record_nemotron_h_ssm_dt(&commands, dt, dt_bias, a_log, out_dt, out_decay, HEADS)
            .unwrap();
        commands.submit_and_wait().unwrap();
        let mut actual_dt = vec![0.0; HEADS];
        let mut actual_decay = vec![0.0; HEADS];
        harness.ops.read_f32_into(out_dt, &mut actual_dt).unwrap();
        harness
            .ops
            .read_f32_into(out_decay, &mut actual_decay)
            .unwrap();
        for head in 0..HEADS {
            let z = dt_values[head] + bias_values[head];
            let expected_dt = if z > 20.0 { z } else { (1.0 + z.exp()).ln() };
            let expected_decay = (expected_dt * a_log_values[head]).exp();
            assert_bits_match(
                &actual_dt[head..head + 1],
                &[expected_dt],
                &format!("SSM dt head {head}"),
            );
            assert_bits_match(
                &actual_decay[head..head + 1],
                &[expected_decay],
                &format!("SSM decay head {head}"),
            );
        }
    }

    #[test]
    fn ssm_scan_matches_cpu_bits() {
        let Some(context) = context() else { return };
        const HEADS: usize = 96;
        const HEADDIM: usize = 80;
        const D_STATE: usize = 128;
        const GROUPS: usize = 8;
        const INNER: usize = HEADS * HEADDIM;
        let mut harness = Harness::new(context, 16 * 1024 * 1024);
        let state = harness.region(HEADS * HEADDIM * D_STATE);
        let x = harness.region(INNER);
        let b = harness.region(GROUPS * D_STATE);
        let c = harness.region(GROUPS * D_STATE);
        let dt = harness.region(HEADS);
        let decay = harness.region(HEADS);
        let d = harness.region(HEADS);
        let output = harness.region(INNER);
        let mut rng = Rng(0xdead_beef_cafe_0001);
        let x_values: Vec<f32> = (0..INNER).map(|_| rng.value(2.0)).collect();
        let b_values: Vec<f32> = (0..GROUPS * D_STATE).map(|_| rng.value(1.0)).collect();
        let c_values: Vec<f32> = (0..GROUPS * D_STATE).map(|_| rng.value(1.0)).collect();
        let dt_values: Vec<f32> = (0..HEADS).map(|_| rng.value(0.5)).collect();
        let decay_values: Vec<f32> = (0..HEADS).map(|_| rng.value(0.5)).collect();
        let d_values: Vec<f32> = (0..HEADS).map(|_| rng.value(1.0)).collect();
        let mut state_values: Vec<f32> = (0..HEADS * HEADDIM * D_STATE)
            .map(|_| rng.value(0.5))
            .collect();
        harness.ops.write_f32(x, &x_values).unwrap();
        harness.ops.write_f32(b, &b_values).unwrap();
        harness.ops.write_f32(c, &c_values).unwrap();
        harness.ops.write_f32(dt, &dt_values).unwrap();
        harness.ops.write_f32(decay, &decay_values).unwrap();
        harness.ops.write_f32(d, &d_values).unwrap();
        harness.ops.write_f32(state, &state_values).unwrap();

        let commands = TokenCommands::begin(context).unwrap();
        harness
            .ops
            .record_nemotron_h_ssm_scan(
                &commands, state, x, b, c, dt, decay, d, output, INNER, HEADS, HEADDIM, D_STATE,
                GROUPS,
            )
            .unwrap();
        commands.submit_and_wait().unwrap();

        // The CPU path, transcribed.
        let heads_per_group = HEADS / GROUPS;
        let mut expected = vec![0.0; INNER];
        for head in 0..HEADS {
            let group = head / heads_per_group;
            for channel in 0..HEADDIM {
                let row = head * HEADDIM + channel;
                let start = row * D_STATE;
                let mut state_row = state_values[start..start + D_STATE].to_vec();
                let sumf = crate::models::nemotron_h::trunk::forward::ssm_scan_row(
                    &mut state_row,
                    &b_values[group * D_STATE..(group + 1) * D_STATE],
                    &c_values[group * D_STATE..(group + 1) * D_STATE],
                    decay_values[head],
                    x_values[row] * dt_values[head],
                );
                expected[row] = sumf + d_values[head] * x_values[row];
                state_values[start..start + D_STATE].copy_from_slice(&state_row);
            }
        }
        let mut actual = vec![0.0; INNER];
        harness.ops.read_f32_into(output, &mut actual).unwrap();
        assert_bits_match(&actual, &expected, "SSM scan");
    }

    #[test]
    fn ssm_gate_matches_cpu_bits() {
        let Some(context) = context() else { return };
        const INNER: usize = 7680;
        let mut harness = Harness::new(context, 1024 * 1024);
        let z = harness.region(INNER);
        let y = harness.region(INNER);
        let output = harness.region(INNER);
        let mut rng = Rng(0x5555_aaaa_3333_7777);
        let z_values: Vec<f32> = (0..INNER).map(|_| rng.value(3.0)).collect();
        let y_values: Vec<f32> = (0..INNER).map(|_| rng.value(3.0)).collect();
        harness.ops.write_f32(z, &z_values).unwrap();
        harness.ops.write_f32(y, &y_values).unwrap();
        let commands = TokenCommands::begin(context).unwrap();
        harness
            .ops
            .record_nemotron_h_ssm_gate(&commands, z, y, output, INNER)
            .unwrap();
        commands.submit_and_wait().unwrap();
        let mut expected = z_values.clone();
        crate::ops::activation::silu_ggml_inplace(&mut expected);
        for index in 0..INNER {
            expected[index] *= y_values[index];
        }
        let mut actual = vec![0.0; INNER];
        harness.ops.read_f32_into(output, &mut actual).unwrap();
        assert_bits_match(&actual, &expected, "SSM gate");
    }

    #[test]
    fn ssm_norm_matches_cpu_bits() {
        let Some(context) = context() else { return };
        const INNER: usize = 7680;
        const GROUPS: usize = 8;
        const GROUP_LEN: usize = INNER / GROUPS;
        let mut harness = Harness::new(context, 1024 * 1024);
        let input = harness.region(INNER);
        let output = harness.region(INNER);
        let mut rng = Rng(0x7777_2222_9999_0000);
        let input_values: Vec<f32> = (0..INNER).map(|_| rng.value(3.0)).collect();
        let norm_values: Vec<f32> = (0..INNER).map(|_| rng.value(1.0)).collect();
        harness.ops.write_f32(input, &input_values).unwrap();
        let mut buffers = crate::vulkan::dense::UploadedBuffers::new(context);
        let weight_buffer = buffers.upload_f32(&norm_values).unwrap();
        let bindings = harness.ops.bind_buffers(&[weight_buffer]).unwrap();
        let commands = TokenCommands::begin(context).unwrap();
        harness
            .ops
            .record_nemotron_h_ssm_norm(&commands, bindings, input, output, GROUP_LEN, GROUPS, 1e-5)
            .unwrap();
        commands.submit_and_wait().unwrap();
        let mut expected = input_values.clone();
        for group in 0..GROUPS {
            let range = group * GROUP_LEN..(group + 1) * GROUP_LEN;
            let slice = &mut expected[range.clone()];
            let mean_sq = (crate::ops::sum_sq_f32(slice) / GROUP_LEN as f64) as f32;
            let rstd = 1.0 / (mean_sq + 1e-5).sqrt();
            for (offset, value) in slice.iter_mut().enumerate() {
                *value = *value * rstd * norm_values[range.start + offset];
            }
        }
        let mut actual = vec![0.0; INNER];
        harness.ops.read_f32_into(output, &mut actual).unwrap();

        assert_bits_match(&actual, &expected, "SSM group norm");
    }
}
