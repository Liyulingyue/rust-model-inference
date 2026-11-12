//! The device path of the PiVo acoustic encoder.
//!
//! The encoder is the part of PiVo that can run on a device: the attention
//! blocks and the convolution blocks work on whole token streams, and every
//! weight matrix is uploaded once and stays resident. The vocoder is left on
//! the host on purpose. Its FARGAN step feeds its own output back into the
//! next step, so the frames of an utterance are strictly sequential and a
//! device would only add a dispatch per frame.
//!
//! The duration head stays on the host as well. It produces the integer frame
//! durations, and those are rounded: the rounding margin of the shortest
//! oracle case is 0.0024 frames, so the device and the host have to agree on
//! the log duration to far better than that before the frame count can
//! differ. The device path reproduces the host's frame counts, and the
//! waveform that comes out of them is the host's waveform to within the
//! oracle's gate.

use std::collections::HashMap;
use std::sync::{Mutex, MutexGuard, OnceLock};

use crate::vulkan::ops::{ArenaRegion, PivoCapacity, PivoRuntime, TokenCommands};

use super::Model;

/// The width of the acoustic network.
const WIDTH: usize = 96;
/// The feed-forward width.
const FF_WIDTH: usize = 192;
/// The number of query heads.
const Q_HEADS: usize = 4;
/// The number of key and value heads.
const KV_HEADS: usize = 2;
/// The width of one head.
const HEAD_DIM: usize = 24;
/// The longest token stream the frontend accepts.
const MAX_ROWS: usize = 512;
/// The causal window of the decoder.
const ATTENTION_WINDOW: usize = 64;

/// The device session of the PiVo acoustic encoder.
///
/// The runtime and its pipelines are expensive to build, so the session is
/// kept between requests. The weights are uploaded once and stay resident.
pub(super) struct PivoDevice {
    runtime: PivoRuntime,
    /// The resident weight regions, by tensor name.
    weights: HashMap<String, ArenaRegion>,
    /// The fingerprint of the weights that are resident.
    fingerprint: u64,
}

/// The device session shared by the requests.
static SESSION: OnceLock<Mutex<Option<PivoDevice>>> = OnceLock::new();

/// The rotation table of the rotary embedding.
///
/// The host's own kernel computes the angle the same way and fills the
/// table with the same sine and cosine, so the device rotates a head exactly
/// the way the host does. A device's transcendental is a couple of ULP away
/// from the host's, and the frame durations are rounded tightly enough for
/// that to matter.
fn rope_table(rows: usize) -> Vec<f32> {
    let mut table = vec![0f32; rows * 24];
    for (position, row) in table.chunks_exact_mut(24).enumerate() {
        for pair in 0..12 {
            let theta = position as f32 * ((2 * pair) as f32 * (-10000f32.ln() / 24.)).exp();
            let (sin, cos) = theta.sin_cos();
            row[pair] = sin;
            row[12 + pair] = cos;
        }
    }
    table
}

/// Whether a tensor belongs to the encoder and has to be resident.
fn is_encoder_weight(name: &str) -> bool {
    name.starts_with("acoustic.text_blocks.") || name == "acoustic.text_norm.weight"
}

/// The number of floats the encoder weights take.
fn encoder_weight_floats(model: &Model) -> usize {
    model
        .weights
        .iter()
        .filter(|(name, _)| is_encoder_weight(name))
        .map(|(_, tensor)| tensor.data.len())
        .sum()
}

/// A fingerprint of the encoder weights.
///
/// The session is shared, so the weights have to be checked before they are
/// used: a quantized export has the same shapes as the F32 one and would
/// otherwise be served with the other's values. The fingerprint looks at the
/// size and at the two ends of every tensor, which is enough to tell the
/// formats apart and costs nothing next to the upload.
fn weight_fingerprint(model: &Model) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    let mut mix = |value: u64| {
        hash ^= value;
        hash = hash.wrapping_mul(0x1000_0000_01b3);
    };
    for (name, tensor) in &model.weights {
        if !is_encoder_weight(name) {
            continue;
        }
        mix(tensor.data.len() as u64);
        mix(name.len() as u64);
        if let Some(first) = tensor.data.first() {
            mix(first.to_bits() as u64);
        }
        if let Some(last) = tensor.data.last() {
            mix(last.to_bits() as u64);
        }
    }
    hash
}

impl PivoDevice {
    /// Build the session and upload the encoder weights.
    fn build(model: &Model) -> Result<Self, String> {
        let context = crate::ops::get_vulkan_context()
            .ok_or_else(|| "PiVo needs a Vulkan device".to_string())?;
        let weight_floats = encoder_weight_floats(model);
        let runtime = PivoRuntime::new(
            context,
            PivoCapacity {
                activation: MAX_ROWS * FF_WIDTH,
                rope_table: MAX_ROWS * 24,
                weight: weight_floats,
                bias: 16,
            },
        )
        .map_err(|e| e.to_string())?;
        let mut device = Self {
            runtime,
            weights: HashMap::new(),
            fingerprint: weight_fingerprint(model),
        };
        for (name, tensor) in &model.weights {
            if !is_encoder_weight(name) {
                continue;
            }
            let region = device
                .runtime
                .upload_weight(&tensor.data)
                .map_err(|e| e.to_string())?;
            device.weights.insert(name.clone(), region);
        }
        Ok(device)
    }

    /// The shared session, built on the first request.
    fn session(model: &Model) -> Result<MutexGuard<'static, Option<PivoDevice>>, String> {
        let slot = SESSION.get_or_init(Mutex::default);
        let mut guard = slot
            .lock()
            .map_err(|_| "PiVo device session lock poisoned".to_string())?;
        let fingerprint = weight_fingerprint(model);
        let stale = guard
            .as_ref()
            .is_some_and(|device| device.fingerprint != fingerprint);
        if guard.is_none() || stale {
            *guard = Some(Self::build(model)?);
        }
        Ok(guard)
    }

    /// The resident region of a weight.
    fn weight(&self, name: &str) -> Result<ArenaRegion, String> {
        self.weights
            .get(name)
            .copied()
            .ok_or_else(|| format!("PiVo device weight is missing: {name}"))
    }

    /// One attention block of the encoder.
    fn attention_block(
        &self,
        commands: &TokenCommands<'_>,
        name: &str,
        rows: usize,
    ) -> Result<(), String> {
        let norm1 = self.weight(&format!("{name}.norm1.weight"))?;
        let q = self.weight(&format!("{name}.attention.q.weight"))?;
        let k = self.weight(&format!("{name}.attention.k.weight"))?;
        let v = self.weight(&format!("{name}.attention.v.weight"))?;
        let q_norm = self.weight(&format!("{name}.attention.q_norm.weight"))?;
        let k_norm = self.weight(&format!("{name}.attention.k_norm.weight"))?;
        let out = self.weight(&format!("{name}.attention.out.weight"))?;
        let norm2 = self.weight(&format!("{name}.norm2.weight"))?;
        let gate = self.weight(&format!("{name}.gate.weight"))?;
        let up = self.weight(&format!("{name}.up.weight"))?;
        let down = self.weight(&format!("{name}.down.weight"))?;
        let runtime = &self.runtime;
        runtime
            .rms_norm(
                commands,
                runtime.x,
                runtime.normed,
                norm1,
                WIDTH,
                1e-6,
                rows,
                WIDTH,
                WIDTH,
            )
            .map_err(|e| e.to_string())?;

        for (width, output, weight) in [
            (Q_HEADS * HEAD_DIM, runtime.q, q),
            (KV_HEADS * HEAD_DIM, runtime.k, k),
            (KV_HEADS * HEAD_DIM, runtime.v, v),
        ] {
            runtime
                .linear(
                    commands,
                    runtime.normed,
                    output,
                    weight,
                    None,
                    WIDTH,
                    width,
                    rows,
                    WIDTH,
                    width,
                )
                .map_err(|e| e.to_string())?;
        }

        // The query and key norms are per head, so they run over the 24 wide
        // units of the row.
        for (buffer, heads, weight) in [(runtime.q, Q_HEADS, q_norm), (runtime.k, KV_HEADS, k_norm)]
        {
            runtime
                .rms_norm(
                    commands,
                    buffer,
                    buffer,
                    weight,
                    HEAD_DIM,
                    1e-6,
                    rows * heads,
                    HEAD_DIM,
                    HEAD_DIM,
                )
                .map_err(|e| e.to_string())?;
        }

        runtime
            .rope(
                commands,
                runtime.q,
                runtime.rope_table,
                rows,
                Q_HEADS * HEAD_DIM,
                Q_HEADS,
            )
            .map_err(|e| e.to_string())?;
        runtime
            .rope(
                commands,
                runtime.k,
                runtime.rope_table,
                rows,
                KV_HEADS * HEAD_DIM,
                KV_HEADS,
            )
            .map_err(|e| e.to_string())?;

        runtime
            .attention(
                commands,
                runtime.q,
                runtime.k,
                runtime.v,
                runtime.attn,
                rows,
                Q_HEADS,
                KV_HEADS,
                HEAD_DIM,
                ATTENTION_WINDOW,
                false,
            )
            .map_err(|e| e.to_string())?;

        runtime
            .linear(
                commands,
                runtime.attn,
                runtime.projection,
                out,
                None,
                WIDTH,
                WIDTH,
                rows,
                WIDTH,
                WIDTH,
            )
            .map_err(|e| e.to_string())?;
        runtime
            .add(commands, runtime.x, runtime.projection, rows * WIDTH)
            .map_err(|e| e.to_string())?;

        runtime
            .rms_norm(
                commands,
                runtime.x,
                runtime.normed,
                norm2,
                WIDTH,
                1e-6,
                rows,
                WIDTH,
                WIDTH,
            )
            .map_err(|e| e.to_string())?;

        runtime
            .linear(
                commands,
                runtime.normed,
                runtime.gate,
                gate,
                None,
                WIDTH,
                FF_WIDTH,
                rows,
                WIDTH,
                FF_WIDTH,
            )
            .map_err(|e| e.to_string())?;
        runtime
            .linear(
                commands,
                runtime.normed,
                runtime.up,
                up,
                None,
                WIDTH,
                FF_WIDTH,
                rows,
                WIDTH,
                FF_WIDTH,
            )
            .map_err(|e| e.to_string())?;
        runtime
            .swiglu(
                commands,
                runtime.gate,
                runtime.up,
                runtime.gate,
                rows * FF_WIDTH,
            )
            .map_err(|e| e.to_string())?;

        runtime
            .linear(
                commands,
                runtime.gate,
                runtime.down,
                down,
                None,
                FF_WIDTH,
                WIDTH,
                rows,
                FF_WIDTH,
                WIDTH,
            )
            .map_err(|e| e.to_string())?;
        runtime
            .add(commands, runtime.x, runtime.down, rows * WIDTH)
            .map_err(|e| e.to_string())
    }

    /// One convolution block of the encoder.
    fn conv_block(
        &self,
        commands: &TokenCommands<'_>,
        name: &str,
        rows: usize,
        dilation: usize,
    ) -> Result<(), String> {
        let norm1 = self.weight(&format!("{name}.norm1.weight"))?;
        let weight = self.weight(&format!("{name}.conv.weight"))?;
        let norm2 = self.weight(&format!("{name}.norm2.weight"))?;
        let runtime = &self.runtime;
        let kernel = 5;
        let padding = 2 * dilation;
        let out_frames = rows + 2 * padding - dilation * (kernel - 1);
        runtime
            .rms_norm(
                commands,
                runtime.x,
                runtime.normed,
                norm1,
                WIDTH,
                1e-6,
                rows,
                WIDTH,
                WIDTH,
            )
            .map_err(|e| e.to_string())?;
        runtime
            .conv1d(
                commands,
                runtime.normed,
                runtime.conv,
                weight,
                None,
                WIDTH,
                WIDTH,
                rows,
                out_frames,
                kernel,
                dilation,
                padding,
            )
            .map_err(|e| e.to_string())?;
        runtime
            .rms_norm(
                commands,
                runtime.conv,
                runtime.conv,
                norm2,
                WIDTH,
                1e-6,
                out_frames,
                WIDTH,
                WIDTH,
            )
            .map_err(|e| e.to_string())?;
        runtime
            .silu(commands, runtime.conv, out_frames * WIDTH)
            .map_err(|e| e.to_string())?;
        runtime
            .add(commands, runtime.x, runtime.conv, rows * WIDTH)
            .map_err(|e| e.to_string())
    }
}

impl Model {
    /// Run the acoustic encoder on the device and return the encoded frames.
    ///
    /// The result has the same shape and the same meaning as the host
    /// encoder's: one row of 96 values per input token.
    pub(super) fn encode_on_device(
        &self,
        tokens: &[usize],
        voice: usize,
    ) -> Result<Vec<f32>, String> {
        let rows = tokens.len();
        if rows == 0 || rows > MAX_ROWS {
            return Err(format!(
                "PiVo device encoder needs 1 to {MAX_ROWS} tokens, got {rows}"
            ));
        }
        let mut session = PivoDevice::session(self)?;
        let device = session
            .as_mut()
            .ok_or_else(|| "PiVo device session is missing".to_string())?;

        // The embedding gather stays on the host: it is a handful of copies
        // and the token stream is short.
        let embeddings = &self.weights["acoustic.tokens.weight"].data;
        let voices = &self.weights["acoustic.voices.weight"].data;
        let speaker = &voices[voice * WIDTH..][..WIDTH];
        let mut x = Vec::with_capacity(rows * WIDTH);
        for &id in tokens {
            let embedding = &embeddings[id * WIDTH..][..WIDTH];
            x.extend(embedding.iter().zip(speaker).map(|(a, b)| a + b));
        }

        // The host writes happen before the command buffer starts. A write
        // that needs a staging buffer takes the submission lock, and the
        // command buffer holds it.
        device
            .runtime
            .upload(device.runtime.x, &x)
            .map_err(|e| e.to_string())?;
        let table = rope_table(rows);
        device
            .runtime
            .upload(device.runtime.rope_table, &table)
            .map_err(|e| e.to_string())?;

        // The whole encoder is recorded into one command buffer. A dispatch
        // per operator with a submission each would spend more time waiting
        // for the queue than the operators take.
        let commands = TokenCommands::begin(device.runtime.context()).map_err(|e| e.to_string())?;
        for block in 0..6 {
            let name = format!("acoustic.text_blocks.{block}");
            if block % 2 == 0 {
                device.attention_block(&commands, &name, rows)?;
            } else {
                device.conv_block(&commands, &name, rows, 1 << (block / 2))?;
            }
        }
        let text_norm = device.weight("acoustic.text_norm.weight")?;
        let runtime = &device.runtime;
        runtime
            .rms_norm(
                &commands,
                runtime.x,
                runtime.normed,
                text_norm,
                WIDTH,
                1e-6,
                rows,
                WIDTH,
                WIDTH,
            )
            .map_err(|e| e.to_string())?;
        commands.submit_and_wait().map_err(|e| e.to_string())?;
        runtime
            .download(runtime.normed, rows * WIDTH)
            .map_err(|e| e.to_string())
    }
}
