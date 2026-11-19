use crate::ops::kernel::Weight;
use crate::vulkan::ops::{
    ArenaRegion, BatchedLinearRuntime, GpuMatmulSpec, Qwen3Ops, TokenCommands,
};
use crate::vulkan::VulkanContext;

pub(super) struct Device {
    context: &'static VulkanContext,
    linear: Option<(BatchedLinearRuntime, usize, usize)>,
    custom: Option<(Qwen3Ops<'static>, usize, usize)>,
}

impl Device {
    pub fn new(context: &'static VulkanContext) -> Self {
        Self {
            context,
            linear: None,
            custom: None,
        }
    }

    pub fn project(
        &mut self,
        weight: &Weight<'_>,
        input: &[f32],
        rows: usize,
    ) -> Result<Vec<f32>, String> {
        if rows == 0
            || weight.n_in.checked_mul(rows) != Some(input.len())
            || input.iter().any(|v| !v.is_finite())
        {
            return Err("invalid Sophon projection input".into());
        }
        let spec = GpuMatmulSpec::prepared(weight).map_err(|e| e.to_string())?;
        let bytes = weight
            .kernel
            .weight_bytes()
            .ok_or("projection has no native device storage")?;
        let n_in = weight
            .n_in
            .checked_next_power_of_two()
            .ok_or("projection width overflow")?;
        let n_out = weight
            .n_out
            .checked_next_power_of_two()
            .ok_or("projection width overflow")?;
        if self
            .linear
            .as_ref()
            .is_none_or(|(_, i, o)| *i < n_in || *o < n_out)
        {
            let (i, o) = self
                .linear
                .as_ref()
                .map(|(_, i, o)| (n_in.max(*i), n_out.max(*o)))
                .unwrap_or((n_in, n_out));
            self.linear = Some((
                BatchedLinearRuntime::new(self.context, 32, i, o, 512)
                    .map_err(|e| e.to_string())?,
                i,
                o,
            ));
        }
        let length = rows
            .checked_mul(weight.n_out)
            .ok_or("projection output overflow")?;
        let mut output = vec![0.0; length];
        let runtime = &mut self.linear.as_mut().unwrap().0;
        for start in (0..rows).step_by(32) {
            let end = (start + 32).min(rows);
            runtime
                .matmul_rows(
                    bytes,
                    spec,
                    &input[start * weight.n_in..end * weight.n_in],
                    end - start,
                    weight.n_in,
                    weight.n_out,
                    &mut output[start * weight.n_out..end * weight.n_out],
                )
                .map_err(|e| e.to_string())?;
        }
        Ok(output)
    }

    // All inputs are written before beginning the command buffer's host-write barrier.
    pub fn custom(
        &mut self,
        op: u32,
        inputs: &[&[f32]],
        count: usize,
        parameters: &[u32],
        groups: usize,
    ) -> Result<Vec<f32>, String> {
        if op > 4 || inputs.len() > 3 || parameters.len() > 9 || count == 0 || groups == 0 {
            return Err("invalid Sophon device dispatch".into());
        }
        let mut push = [0u32; 16];
        push[0] = op;
        let mut end = 0usize;
        let mut regions = Vec::new();
        for (i, values) in inputs.iter().enumerate() {
            if values.iter().any(|v| !v.is_finite()) {
                return Err("nonfinite Sophon device input".into());
            }
            push[i + 1] = u32::try_from(end).map_err(|_| "device offset overflow")?;
            let size = values.len().checked_mul(4).ok_or("device size overflow")?;
            regions.push(ArenaRegion {
                offset: end.checked_mul(4).ok_or("device size overflow")?,
                size,
            });
            end = end
                .checked_add(values.len())
                .ok_or("device size overflow")?;
        }
        push[4] = u32::try_from(end).map_err(|_| "device output offset overflow")?;
        push[5] = u32::try_from(count).map_err(|_| "device output count overflow")?;
        u32::try_from(end.checked_add(count).ok_or("device span overflow")?)
            .map_err(|_| "device word span overflow")?;
        push[6..6 + parameters.len()].copy_from_slice(parameters);
        let output = ArenaRegion {
            offset: end.checked_mul(4).ok_or("device size overflow")?,
            size: count.checked_mul(4).ok_or("device size overflow")?,
        };
        let size = output
            .offset
            .checked_add(output.size)
            .ok_or("device size overflow")?;
        if self
            .custom
            .as_ref()
            .is_none_or(|(_, capacity, _)| *capacity < size)
        {
            let capacity = size
                .checked_next_power_of_two()
                .ok_or("device capacity overflow")?
                .max(4096);
            let mut ops =
                Qwen3Ops::new_with_size(self.context, capacity, 8).map_err(|e| e.to_string())?;
            let shader = ops
                .add_shader(include_bytes!("../../../shaders/bin/sophon_audio.spv"))
                .map_err(|e| e.to_string())?;
            self.custom = Some((ops, capacity, shader));
        }
        let (ops, _, shader) = self.custom.as_ref().unwrap();
        let x = groups.min(65535);
        let y = groups.div_ceil(x);
        if y > 65535 {
            return Err("Sophon device grid exceeds limits".into());
        }
        push[15] = x as u32;
        for (region, values) in regions.into_iter().zip(inputs) {
            ops.write_f32(region, values).map_err(|e| e.to_string())?;
        }
        let commands = TokenCommands::begin(self.context).map_err(|e| e.to_string())?;
        ops.record_custom(&commands, *shader, None, &push, [x as u32, y as u32, 1])
            .map_err(|e| e.to_string())?;
        commands.submit_and_wait().map_err(|e| e.to_string())?;
        let mut result = vec![0.0; count];
        ops.read_f32_into(output, &mut result)
            .map_err(|e| e.to_string())?;
        Ok(result)
    }

    pub fn attention(
        &mut self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        heads: usize,
    ) -> Result<Vec<f32>, String> {
        let width = heads
            .checked_mul(64)
            .filter(|w| *w > 0)
            .ok_or("invalid attention heads")?;
        if q.is_empty()
            || k.is_empty()
            || !q.len().is_multiple_of(width)
            || !k.len().is_multiple_of(width)
            || k.len() != v.len()
        {
            return Err("invalid Sophon attention shape".into());
        }
        let queries = q.len() / width;
        let keys = k.len() / width;
        self.custom(
            1,
            &[q, k, v],
            q.len(),
            &[
                u32::try_from(queries).map_err(|_| "attention queries overflow")?,
                u32::try_from(keys).map_err(|_| "attention keys overflow")?,
                u32::try_from(heads).map_err(|_| "attention heads overflow")?,
            ],
            queries * heads,
        )
    }

    pub fn spectrum(
        &mut self,
        padded: &[f32],
        window: &[f32],
        frames: usize,
        hop: usize,
        power: bool,
    ) -> Result<Vec<f32>, String> {
        let n = window.len();
        let mut table = window.to_vec();
        for cosine in [true, false] {
            table.extend((0..n).map(|i| {
                let a = (std::f64::consts::TAU * i as f64 / n as f64) as f32;
                let (s, c) = a.sin_cos();
                if cosine {
                    c
                } else {
                    s
                }
            }));
        }
        let bins = n / 2 + 1;
        let count = frames
            .checked_mul(bins)
            .and_then(|c| c.checked_mul(if power { 1 } else { 2 }))
            .ok_or("spectrum size overflow")?;
        if n == 0
            || frames == 0
            || (frames - 1)
                .checked_mul(hop)
                .and_then(|v| v.checked_add(n))
                .is_none_or(|len| len > padded.len())
        {
            return Err("truncated spectrum input".into());
        }
        self.custom(
            2,
            &[padded, &table],
            count,
            &[
                u32::try_from(n).map_err(|_| "spectrum width overflow")?,
                u32::try_from(frames).map_err(|_| "spectrum frames overflow")?,
                u32::try_from(hop).map_err(|_| "spectrum hop overflow")?,
                u32::from(power),
            ],
            count.div_ceil(64),
        )
    }
}
