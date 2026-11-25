use super::{
    as_u32, f32_region, product, row_dispatch, ArenaRegion, Qwen3Ops, TokenCommands, VulkanContext,
    VulkanError, CACHED_ATTENTION,
};
use crate::core::scratchpad::KvFormat;

/// One layer's persistent device KV cache. Scratch regions are sized for the largest
/// append, and a sliding window keeps only the span needed by that entire batch.
pub(crate) struct CachedAttention {
    ops: Qwen3Ops<'static>,
    q: ArenaRegion,
    new_k: ArenaRegion,
    new_v: ArenaRegion,
    output: ArenaRegion,
    k: ArenaRegion,
    v: ArenaRegion,
    capacity: usize,
    max_rows: usize,
    window: usize,
    slots: usize,
    heads: usize,
    kv_heads: usize,
    dim: usize,
    packed: bool,
    position: usize,
    last_start: usize,
}

impl CachedAttention {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        context: &'static VulkanContext,
        capacity: usize,
        max_rows: usize,
        window: Option<usize>,
        heads: usize,
        kv_heads: usize,
        dim: usize,
        format: KvFormat,
    ) -> Result<Self, VulkanError> {
        let window = window.unwrap_or(capacity).min(capacity);
        let packed = format == KvFormat::F16;
        if max_rows == 0
            || max_rows > capacity
            || window == 0
            || heads == 0
            || kv_heads == 0
            || !heads.is_multiple_of(kv_heads)
            || !(1..=128).contains(&dim)
            || capacity > (u32::MAX - 32) as usize
        {
            return Err(VulkanError::UnsupportedShape(
                "invalid cached GQA dimensions".into(),
            ));
        }
        let slots = capacity.min(window.saturating_add(max_rows - 1));
        let q_elements = product("cached attention queries", &[max_rows, heads, dim])?;
        let kv_width = product("cached attention KV width", &[kv_heads, dim])?;
        let per_word = if packed { 2 } else { 1 };
        if !kv_width.is_multiple_of(per_word) {
            return Err(VulkanError::UnsupportedShape(
                "packed F16 KV rows must be pair aligned".into(),
            ));
        }
        let rows_elements = product("cached attention appended KV", &[max_rows, kv_width])?;
        let cache_elements = product("cached attention KV", &[slots, kv_width])?;
        // Shader indices address individual F32 or F16 elements before conversion to words.
        as_u32(cache_elements, "cached attention KV elements")?;
        row_dispatch(kv_width.div_ceil(32 * per_word), max_rows, &context.limits)?;
        row_dispatch(max_rows, heads, &context.limits)?;

        let mut cursor = 0;
        let q = f32_region(&mut cursor, q_elements)?;
        let new_k = f32_region(&mut cursor, rows_elements)?;
        let new_v = f32_region(&mut cursor, rows_elements)?;
        let output = f32_region(&mut cursor, q_elements)?;
        let k = f32_region(&mut cursor, cache_elements / per_word)?;
        let v = f32_region(&mut cursor, cache_elements / per_word)?;
        Ok(Self {
            ops: Qwen3Ops::new_with_size(context, cursor, 1)?,
            q,
            new_k,
            new_v,
            output,
            k,
            v,
            capacity,
            max_rows,
            window,
            slots,
            heads,
            kv_heads,
            dim,
            packed,
            position: 0,
            last_start: 0,
        })
    }

    pub(crate) fn forward(
        &mut self,
        position: usize,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        output: &mut [f32],
    ) -> Result<(), VulkanError> {
        let width = self.heads * self.dim;
        let kv_width = self.kv_heads * self.dim;
        let rows = q.len() / width;
        if rows == 0
            || rows > self.max_rows
            || position > self.capacity - rows
            || (position != self.position && position != self.last_start)
            || !q.len().is_multiple_of(width)
            || output.len() != q.len()
            || k.len() != rows * kv_width
            || v.len() != k.len()
        {
            return Err(VulkanError::UnsupportedShape(
                "invalid cached GQA append".into(),
            ));
        }
        let word = |r: ArenaRegion| (r.offset / 4) as u32;
        let mut push = [
            word(self.q),
            word(self.new_k),
            word(self.new_v),
            word(self.output),
            word(self.k),
            word(self.v),
            position as u32,
            self.slots as u32,
            rows as u32,
            self.heads as u32,
            self.kv_heads as u32,
            self.dim as u32,
            self.window as u32,
            u32::from(self.packed),
            (1.0 / (self.dim as f32).sqrt()).to_bits(),
            1,
        ];
        let per_word = if self.packed { 2 } else { 1 };
        let limits = &self.ops.context.limits;
        let append_grid = row_dispatch(kv_width.div_ceil(32 * per_word), rows, limits)?;
        let attention_grid = row_dispatch(rows, self.heads, limits)?;

        self.ops.write_f32(self.q, q)?;
        self.ops.write_f32(self.new_k, k)?;
        self.ops.write_f32(self.new_v, v)?;
        let commands = TokenCommands::begin(self.ops.context)?;
        self.ops.record_linear_dispatch(
            &commands,
            CACHED_ATTENTION,
            self.ops.arena_bindings,
            &push,
            append_grid,
        );
        push[15] = 0;
        self.ops.record_linear_dispatch(
            &commands,
            CACHED_ATTENTION,
            self.ops.arena_bindings,
            &push,
            attention_grid,
        );
        commands.submit_and_wait()?;
        output.copy_from_slice(self.ops.read_f32(self.output, output.len())?);
        // A later layer can fail after this layer succeeded. Replaying that batch
        // at last_start is safe because the ring retains its complete visible span.
        self.last_start = position;
        self.position = position + rows;
        Ok(())
    }
}
