/// KV 缓存精度格式。
///
/// - `F32`：单精度，更精确（默认）
/// - `F16`：半精度，节省内存
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum KvFormat {
    #[default]
    F32,
    F16,
}

pub struct ExecutionScratchpad {
    pub x: Vec<f32>,
    pub normed: Vec<f32>,
    pub q: Vec<f32>,
    pub k_new: Vec<f32>,
    pub v_new: Vec<f32>,
    pub attn_out: Vec<f32>,
    pub attn_proj: Vec<f32>,
    pub down_buf: Vec<f32>,
    pub gate_buf: Vec<f32>,
    pub up_buf: Vec<f32>,
    pub logits: Vec<f32>,
    pub q8_buf: Vec<u8>,
    pub scale_buf: Vec<f32>,
    /// Pre-quantized Q8_K blocks for K-quant kernels (Q4_K / Q6_K). One
    /// block holds 256 f32 elements as i8 + scales. Sized for the largest
    /// n_in a K-quant matmul will see (= max(n_embd_q, n_ff)).
    pub q8k_buf: Vec<crate::ops::quant::BlockQ8K>,
    pub score_stride: usize,
    pub scores: Vec<f32>,
}

pub struct KvCacheF16 {
    pub k: Vec<u16>,
    pub v: Vec<u16>,
}

pub struct KvCacheF32 {
    pub k: Vec<f32>,
    pub v: Vec<f32>,
}

pub enum KvCache {
    F16(KvCacheF16),
    F32(KvCacheF32),
}

impl ExecutionScratchpad {
    /// Builds the shared per-step scratchpad.
    ///
    /// ## Buffer sizing invariant
    ///
    /// `q8_buf` / `scale_buf` / `q8k_buf` are sized for the *largest* write
    /// any consumer will perform. Consumers and their widths:
    ///
    /// - attention QKV matmul inputs: `n_embd` (Q8_0 / Q8_K)
    /// - attention output projection input: `n_embd_q`
    /// - FFN gate/up matmul outputs (dense LFM2 / LFM2-MoE): `n_ff`
    /// - FFN Q8_0/Q8_K quantization: `n_ff` bytes / `n_ff/32` scales / `n_ff/256` Q8_K blocks
    /// - shortconv `in_proj` output (the b∥c∥x concat): `3 * n_embd`
    /// - MoE expert gate/up writes share `gate_buf`/`up_buf`, requiring
    ///   `n_ff >= n_expert_used * n_ff_exp` (architectural invariant of LFM2-MoE).
    ///
    /// `max_n_in` is therefore `(n_embd * 3).max(n_embd_q).max(n_ff)`. Every
    /// local `max_n_in` in the model forwards (`forward_layer`,
    /// `forward_attention`, `forward_shortconv`) **must** use the same
    /// expression — keep them in sync. `gate_buf`/`up_buf` are sized
    /// `max(n_ff, n_embd * 3)` to cover both dense-FFN (`n_ff`) and shortconv
    /// in-proj (`3 * n_embd`) writers.
    ///
    /// Failure mode if drift occurs: silent heap corruption that only
    /// surfaces when an adjacent allocation is freed
    /// (STATUS_HEAP_CORRUPTION / STATUS_ACCESS_VIOLATION at process exit).
    pub fn new(
        n_embd: usize,
        n_embd_q: usize,
        n_embd_gqa: usize,
        n_ff: usize,
        vocab: usize,
        n_threads: usize,
        max_ctx: usize,
    ) -> Self {
        let max_n_in = n_embd_q.max(n_ff).max(n_embd * 3);
        let score_stride = max_ctx.div_ceil(256) * 256;
        Self {
            x: vec![0.0f32; n_embd],
            normed: vec![0.0f32; n_embd],
            q: vec![0.0f32; n_embd_q],
            k_new: vec![0.0f32; n_embd_gqa],
            v_new: vec![0.0f32; n_embd_gqa],
            attn_out: vec![0.0f32; n_embd_q],
            attn_proj: vec![0.0f32; n_embd],
            down_buf: vec![0.0f32; n_embd],
            gate_buf: vec![0.0f32; n_ff.max(n_embd * 3)],
            up_buf: vec![0.0f32; n_ff.max(n_embd * 3)],
            logits: vec![0.0f32; vocab],
            q8_buf: vec![0u8; max_n_in],
            scale_buf: vec![0.0f32; max_n_in / 32],
            q8k_buf: vec![
                crate::ops::quant::BlockQ8K {
                    d: 0.0,
                    qs: [0; 256],
                    bsums: [0; 16],
                };
                max_n_in / 256
            ],
            score_stride,
            scores: vec![0.0f32; n_threads * score_stride],
        }
    }
}

impl KvCache {
    pub fn new_f16(n_layer: usize, max_ctx: usize, n_embd_gqa: usize) -> Self {
        let size = n_layer * max_ctx * n_embd_gqa;
        KvCache::F16(KvCacheF16 {
            k: vec![0u16; size],
            v: vec![0u16; size],
        })
    }

    pub fn new_f32(n_layer: usize, max_ctx: usize, n_embd_gqa: usize) -> Self {
        let size = n_layer * max_ctx * n_embd_gqa;
        KvCache::F32(KvCacheF32 {
            k: vec![0.0f32; size],
            v: vec![0.0f32; size],
        })
    }
}

// =============================================================================
// KvState: 标准化 KV 缓存抽象
// =============================================================================
//
// 设计目标（参考 `docs/KV_CACHE_DESIGN.md`）：
// - 让 KV 缓存成为**独立的一等公民**：可创建、传递、复用、共享
// - 自描述架构信息，支持跨模型兼容性检查
// - 显式生命周期策略：Ephemeral / Timed / Persistent
// - 为未来多轮对话、KV 分页、KV 迁移等高级特性打基础
//
// 当前状态：基础类型已就位，迁移进行中。
// 现有 `KvCache` 作为底层存储保留，`KvState` 负责生命周期和兼容性管理。

/// KV 缓存依赖的架构参数（与权重无关）。
///
/// 仅包含影响 KV 布局的维度。两个不同模型若此结构相等，
/// 则它们的 KV 缓存可以相互迁移或复用。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KvArch {
    pub n_layer: usize,
    pub n_head_kv: usize,
    pub n_embd_head_k: usize,
    pub n_embd_head_v: usize,
    pub max_ctx: usize,
}

impl KvArch {
    pub fn new(
        n_layer: usize,
        n_head_kv: usize,
        n_embd_head_k: usize,
        n_embd_head_v: usize,
        max_ctx: usize,
    ) -> Self {
        Self {
            n_layer,
            n_head_kv,
            n_embd_head_k,
            n_embd_head_v,
            max_ctx,
        }
    }

    /// 单 token 占用字节数（K/V head 数量按 max(k,v) 对齐）。
    pub fn bytes_per_token(&self, format: KvFormat) -> usize {
        let stride = self.n_head_kv * self.n_embd_head_k.max(self.n_embd_head_v);
        let elem_size = match format {
            KvFormat::F16 => 2,
            KvFormat::F32 => 4,
        };
        stride * elem_size
    }

    /// 完整缓存字节数（capacity 由调用方提供，因为 KvArch 本身不包含 capacity）。
    pub fn total_bytes(&self, capacity: usize, format: KvFormat) -> usize {
        self.n_layer * capacity * self.bytes_per_token(format)
    }

    /// 当前架构是否与 `other` 兼容（同架构 → KV 可共享）。
    pub fn is_compatible_with(&self, other: &KvArch) -> bool {
        self.n_layer == other.n_layer
            && self.n_head_kv == other.n_head_kv
            && self.n_embd_head_k == other.n_embd_head_k
            && self.n_embd_head_v == other.n_embd_head_v
    }
}

/// KV 缓存生命周期策略。
///
/// - `Ephemeral`：单会话生命周期，随 session 销毁（CLI 单轮对话）
/// - `Timed`：空闲超时后自动销毁（长连接多轮对话）
/// - `Persistent`：显式销毁，永不过期（需要长期上下文保持）
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum KvLifecycle {
    Ephemeral,
    Timed { ttl: std::time::Duration },
    Persistent,
}

/// 标准化 KV 缓存状态。
///
/// 将 KV 缓存从「函数局部变量」提升为「可独立管理的一等公民」：
/// - 自带架构信息，可独立校验兼容性
/// - 显式生命周期，支持 Ephemeral/Timed/Persistent 三种策略
/// - 当前后端使用现有 `KvCache` 存储数据
///
/// 未来扩展点（参考 `docs/KV_CACHE_DESIGN.md`）：
/// - KV 分页（类似 vLLM）
/// - KV 压缩（长时间未访问的 KV 压缩存储）
/// - KV 迁移（跨进程/跨设备）
pub struct KvState {
    pub arch: std::sync::Arc<KvArch>,
    pub format: KvFormat,
    pub lifecycle: KvLifecycle,
    pub cache: KvCache,
    pub capacity: usize,
    pub seq_len: usize,
    pub last_access: std::time::Instant,
}

impl KvState {
    /// 创建新的 KV 状态（默认 `Ephemeral` 生命周期）。
    pub fn new(arch: std::sync::Arc<KvArch>, format: KvFormat, capacity: usize) -> Self {
        let stride = arch.n_head_kv * arch.n_embd_head_k.max(arch.n_embd_head_v);
        let cache = match format {
            KvFormat::F16 => KvCache::new_f16(arch.n_layer, capacity, stride),
            KvFormat::F32 => KvCache::new_f32(arch.n_layer, capacity, stride),
        };
        Self {
            arch,
            format,
            lifecycle: KvLifecycle::Ephemeral,
            cache,
            capacity,
            seq_len: 0,
            last_access: std::time::Instant::now(),
        }
    }

    pub fn with_lifecycle(mut self, lifecycle: KvLifecycle) -> Self {
        self.lifecycle = lifecycle;
        self
    }

    /// 检查是否与给定架构兼容（同架构 → KV 可共享）。
    pub fn is_compatible_with(&self, other_arch: &KvArch) -> bool {
        self.arch.is_compatible_with(other_arch)
    }

    /// 更新最后访问时间（每次推理调用都应该更新）。
    pub fn update_access(&mut self) {
        self.last_access = std::time::Instant::now();
    }

    /// 检查 KV 缓存是否已过期（Timed 模式专属）。
    pub fn is_expired(&self) -> bool {
        match self.lifecycle {
            KvLifecycle::Ephemeral => false,
            KvLifecycle::Persistent => false,
            KvLifecycle::Timed { ttl } => self.last_access.elapsed() > ttl,
        }
    }

    /// 重置 KV 缓存内容（清零 seq_len 和 cache 数据），但保留 capacity 和 lifecycle。
    pub fn reset(&mut self) {
        self.seq_len = 0;
        match &mut self.cache {
            KvCache::F16(c) => {
                for x in c.k.iter_mut() {
                    *x = 0;
                }
                for x in c.v.iter_mut() {
                    *x = 0;
                }
            }
            KvCache::F32(c) => {
                for x in c.k.iter_mut() {
                    *x = 0.0;
                }
                for x in c.v.iter_mut() {
                    *x = 0.0;
                }
            }
        }
        self.update_access();
    }
}

#[cfg(test)]
mod tests {
    use super::{ExecutionScratchpad, KvArch, KvFormat, KvLifecycle, KvState};
    use std::sync::Arc;
    use std::time::Duration;

    #[test]
    fn scores_allocate_padded_non_overlapping_segments_per_thread() {
        let scratch = ExecutionScratchpad::new(1, 1, 1, 1, 1, 2, 257);

        assert_eq!(scratch.scores.len(), 1024);
        let (first_thread, second_thread) = scratch.scores.split_at(512);
        assert_eq!(first_thread.len(), 512);
        assert_eq!(second_thread.len(), 512);
    }

    fn make_arch() -> Arc<KvArch> {
        Arc::new(KvArch::new(4, 8, 128, 128, 1024))
    }

    #[test]
    fn kv_state_basic_allocation() {
        let arch = make_arch();
        let state = KvState::new(arch.clone(), KvFormat::F16, 512);
        assert_eq!(state.capacity, 512);
        assert_eq!(state.seq_len, 0);
        assert!(matches!(state.lifecycle, KvLifecycle::Ephemeral));
        assert_eq!(state.arch.n_layer, 4);
    }

    #[test]
    fn kv_arch_compatibility() {
        let a = KvArch::new(4, 8, 128, 128, 1024);
        let b = KvArch::new(4, 8, 128, 128, 2048);
        let c = KvArch::new(4, 8, 64, 128, 1024); // n_embd_head_k 不同
        let d = KvArch::new(2, 8, 128, 128, 1024); // n_layer 不同

        assert!(a.is_compatible_with(&b)); // max_ctx 不同但兼容（不影响 KV 布局）
        assert!(!a.is_compatible_with(&c)); // head_k 不同
        assert!(!a.is_compatible_with(&d)); // 层数不同
    }

    #[test]
    fn kv_lifecycle_ephemeral_never_expires() {
        let arch = make_arch();
        let mut state = KvState::new(arch, KvFormat::F16, 128);
        state.last_access = std::time::Instant::now() - Duration::from_secs(3600);
        assert!(!state.is_expired());
    }

    #[test]
    fn kv_lifecycle_timed_can_expire() {
        let arch = make_arch();
        let mut state = KvState::new(arch, KvFormat::F16, 128).with_lifecycle(KvLifecycle::Timed {
            ttl: Duration::from_millis(50),
        });
        assert!(!state.is_expired());
        std::thread::sleep(Duration::from_millis(80));
        assert!(state.is_expired());
        state.update_access();
        assert!(!state.is_expired());
    }

    #[test]
    fn kv_lifecycle_persistent_never_expires() {
        let arch = make_arch();
        let mut state =
            KvState::new(arch, KvFormat::F16, 128).with_lifecycle(KvLifecycle::Persistent);
        state.last_access = std::time::Instant::now() - Duration::from_secs(3600);
        assert!(!state.is_expired());
    }

    #[test]
    fn kv_reset_clears_seq_len() {
        let arch = make_arch();
        let mut state = KvState::new(arch, KvFormat::F32, 64);
        state.seq_len = 42;
        state.reset();
        assert_eq!(state.seq_len, 0);
    }

    #[test]
    fn kv_bytes_per_token_format_differs() {
        let arch = make_arch();
        let f16_bytes = arch.bytes_per_token(KvFormat::F16);
        let f32_bytes = arch.bytes_per_token(KvFormat::F32);
        assert_eq!(f32_bytes, f16_bytes * 2);
    }
}
