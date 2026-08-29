# KV Cache 设计文档

## 概述

KV Cache 是 Transformer 模型推理优化的核心机制，通过缓存 Attention 层计算过的 Key-Value 向量，避免重复计算。

## 核心概念

### KV 与模型的绑定关系

KV Cache 依赖的是**模型架构参数**，而非权重值：

| 依赖参数 | 说明 |
|----------|------|
| `n_layer` | 层数 |
| `n_head_kv` | Key/Value head 数量 |
| `n_embd_head_k` | Key 向量维度 |
| `n_embd_head_v` | Value 向量维度 |
| `max_ctx` | 最大上下文长度 |

**与以下无关：**
- 权重数值 (Q4/Q8/F16)
- 量化方式
- 权重来源

### 架构兼容性

```
同架构模型 → KV 可共享
  ├── Qwen3-8B-Q4 + Qwen3-8B-Q8  → ✅ 同架构，量化不影响 KV
  └── Qwen3-8B + Qwen3-4B         → ❌ 不同架构

不同架构模型 → KV 不可共享
  ├── 层数不同
  ├── head 数量不同
  └── head 维度不同
```

## 生命周期管理

### 三种生命周期策略

| 类型 | 创建时机 | 销毁时机 | 适用场景 |
|------|----------|----------|----------|
| **Ephemeral (单会话)** | 每次生成调用 | 调用结束 | stateless API |
| **Timed (定时器)** | 首次请求 | 空闲超时后 | 长连接多轮对话 |
| **Persistent (持久)** | 显式创建 | 显式删除/替换 | 长期上下文保持 |

### 生命周期状态机

```
                    ┌─────────────┐
                    │   Created   │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │ Active   │ │  Idle    │ │ Destroyed │
        └─────┬────┘ └─────┬────┘ └──────────┘
              │            │            ▲
              │   Timer    │            │
              │  Expires    │            │
              └────────────┴────────────┘
```

## 多租户设计

### 独立生命周期

三个维度完全独立，可以自由组合：

| 组件 | 生命周期 | 说明 |
|------|----------|------|
| **Model** | 长期 | mmap 映射，按需加载物理页 |
| **KvState** | 可控 | 按 lifecycle 策略管理 |
| **TenantContext** | 可选 | 身份/配额管理 |

### 组合矩阵

```
┌─────────┬─────────┬─────────┬───────────────────────┐
│  租户   │   KV    │  模型   │         说明          │
├─────────┼─────────┼─────────┼───────────────────────┤
│ 单租户  │ 单 KV   │ 单模型  │ 最简单，服务单一用户   │
│ 多租户  │ 共享 KV │ 单模型  │ KV 结构相同可共享     │
│ 多租户  │ 多 KV   │ 单模型  │ 租户隔离，完全独立    │
│ 单租户  │ 多 KV   │ 多模型  │ 同一用户多会话/多模型  │
│ 多租户  │ 多 KV   │ 多模型  │ 完全隔离，最灵活      │
└─────────┴─────────┴─────────┴───────────────────────┘
```

## 数据结构

### KvState

```rust
pub struct KvState {
    pub arch: Arc<KvArch>,        // 架构信息
    pub format: KvFormat,          // F16 或 F32
    pub lifecycle: KvLifecycle,   // 生命周期策略
    pub k: Vec<u8>,               // Key cache (F16/F32 编码)
    pub v: Vec<u8>,               // Value cache (F16/F32 编码)
    pub capacity: usize,            // 最大 token 数
    pub seq_len: usize,            // 当前序列长度
    pub last_access: Instant,       // 最后访问时间
}
```

### KvFormat

```rust
pub enum KvFormat {
    F16,  // 半精度，节省内存
    F32,  // 单精度，更精确
}
```

### KvLifecycle

```rust
pub enum KvLifecycle {
    Ephemeral,                  // 单会话
    Timed { ttl: Duration },    // 定时器
    Persistent,                 // 持久
}
```

### KvArch

```rust
pub struct KvArch {
    pub n_layer: usize,
    pub n_head_kv: usize,
    pub n_embd_head_k: usize,
    pub n_embd_head_v: usize,
    pub max_ctx: usize,
}
```

## 内存计算

### 单层单 token 内存

```
Bytes_per_token = n_head_kv * (n_embd_head_k + n_embd_head_v) * sizeof(format)

示例 (Qwen3-8B, F16):
  = 8 * (128 + 128) * 2 bytes
  = 2048 bytes/token
```

### 完整模型内存

```
Total_KV = n_layer * max_ctx * n_head_kv * (n_embd_head_k + n_embd_head_v) * sizeof(format)

示例 (Qwen3-8B, 32K ctx, F16):
  = 48 * 32768 * 8 * 256 * 2 bytes
  = 25,165,824 bytes ≈ 24 MB
```

## 实现要点

### 1. 架构兼容性检查

```rust
impl KvState {
    pub fn is_compatible_with(&self, arch: &KvArch) -> bool {
        self.arch.n_layer == arch.n_layer
            && self.arch.n_head_kv == arch.n_head_kv
            && self.arch.n_embd_head_k == arch.n_embd_head_k
            && self.arch.n_embd_head_v == arch.n_embd_head_v
    }
}
```

### 2. 生命周期管理

```rust
impl KvState {
    pub fn update_access(&mut self) {
        self.last_access = Instant::now();
    }

    pub fn is_expired(&self) -> bool {
        match self.lifecycle {
            KvLifecycle::Ephemeral => false, // 永不过期，由外部控制
            KvLifecycle::Timed { ttl } => {
                self.last_access.elapsed() > ttl
            }
            KvLifecycle::Persistent => false, // 永不过期
        }
    }
}
```

### 3. 内存分配

```rust
impl KvState {
    pub fn new(arch: &KvArch, format: KvFormat, capacity: usize) -> Self {
        let stride = arch.n_head_kv * (arch.n_embd_head_k.max(arch.n_embd_head_v));
        let total = arch.n_layer * capacity * stride;
        let elem_size = match format {
            KvFormat::F16 => 2,
            KvFormat::F32 => 4,
        };
        let bytes = total * elem_size;

        Self {
            arch: Arc::new(arch.clone()),
            format,
            lifecycle: KvLifecycle::Ephemeral,
            k: vec![0; bytes],
            v: vec![0; bytes],
            capacity,
            seq_len: 0,
            last_access: Instant::now(),
        }
    }
}
```

## 使用模式

### 模式 1: Session-based (推荐)

```rust
// 用户控制 Session 生命周期
let model = Model::from_source(...)?;
let mut session = model.new_session(capacity, kv_format)?;

loop {
    let result = session.generate(input, options)?;
    // KV 在 session 内保持
}
```

### 模式 2: Stateless API

```rust
// 每次请求独立 KV
fn generate(model: &Model, input: &[u32]) -> Result<String> {
    let mut session = model.new_session(capacity, kv_format)?;
    session.generate(input, options)
} // session drop, KV 释放
```

### 模式 3: 外部 KV 注入

```rust
// 外部管理 KV 生命周期
let kv = KvState::new(arch, format, capacity)?;
let session = model.new_session_with_kv(&mut kv)?;

session.generate(input, options)?;
```

## 与 Model 的关系

```
                    ┌─────────────────┐
                    │      Model      │
                    │  (mmap 权重)    │
                    │                 │
                    │ layers: Vec<...>│
                    │ pool: ComputePool│
                    │ tokenizer: ...  │
                    └────────┬────────┘
                             │
                             │ 引用
                             ▼
┌──────────────────────────────────────────────────┐
│                  InferenceSession                 │
│                                                  │
│  model: &Model                                  │
│  kv: &mut KvState                               │
│  scratch: ExecutionScratchpad                     │
│  capacity: usize                                │
│                                                  │
│  ┌────────────────────────────────────────────┐ │
│  │                KvState                      │ │
│  │  arch: Arc<KvArch>                         │ │
│  │  k: Vec<u8>                               │ │
│  │  v: Vec<u8>                               │ │
│  │  lifecycle: KvLifecycle                      │ │
│  └────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────┘
```

## 实现状态

> 本节记录设计文档与当前代码实现的差距。
> 截至当前，**类型骨架已就位但迁移未完成**：只有 Qwen3 文本路径真正使用 `KvState`，
> 其余 5 个推理路径仍直接使用旧 `KvCache` 枚举，多项 API 尚未落地。

### 已落地

| 设计项 | 状态 | 位置 |
|--------|------|------|
| `KvState` / `KvArch` / `KvFormat` / `KvLifecycle` 类型 | ✅ 已定义 | `src/core/scratchpad.rs` |
| `KvArch::is_compatible_with` | ✅ 已定义 + 单测 | `scratchpad.rs:166` |
| `KvState::new` / `with_lifecycle` / `update_access` / `reset` | ✅ 已定义 | `scratchpad.rs:207-264` |
| Session-based 使用模式 | ✅ 仅 Qwen3 | `src/models/qwen3/base.rs:432` (`new_with_kv_state`) |
| 内存计算公式 `bytes_per_token` / `total_bytes` | ✅ 已定义 + 单测 | `scratchpad.rs:151-163` |
| `KvState::is_compatible_with` 调用 | ✅ 调用方存在 | `qwen3/base.rs` 中 `KvArch` 构造路径 |

### 尚未落地

#### 1. 类型迁移未完成（5/6 路径仍用旧 `KvCache`）

| 路径 | 现状 | 位置 |
|------|------|------|
| `llama/base.rs` | ❌ `KvCache::new_f16` | `models/llama/base.rs:277` |
| `lfm2/base.rs` | ❌ `KvCache::new_f16` | `models/lfm2/base.rs:148` |
| `lfm25/base.rs` | ❌ `KvCache::new_f16` | `models/lfm25/base.rs:117` |
| `qwen35/session.rs` | ❌ `KvCache::new_f32` | `models/qwen35/session.rs:55,85` |
| `bin/server.rs` | ❌ `KvCache::new_f16/f32` 每次请求新建 | `bin/server.rs:433,814` |
| `qwen3/base.rs` | ✅ `KvState` | — |

**影响**：只有 Qwen3 文本生成获得了 `KvState` 带来的兼容性检查、生命周期管理能力；
其余路径（特别是 server 多租户场景）的 KV 仍是裸 `Vec`。

#### 2. 类型重复

`lfm2/base.rs:362` 与 `lfm25/base.rs:321` 分别定义了本地枚举：

```rust
pub enum KvCacheFmt { F16, F32 }
```

与 `core::scratchpad::KvFormat` 功能完全重复，`app/text.rs:60-75` 不得不在两层之间手动转换。
建议合并到 `KvFormat`，删除 `KvCacheFmt`。

#### 3. 生命周期策略形同虚设

- `Timed { ttl }`：**仅在 `scratchpad.rs:321` 的单测中构造**；生产代码 0 处使用。
- `Persistent`：**仅在 `scratchpad.rs:333` 的单测中构造**；生产代码 0 处使用。
- `Ephemeral`：唯一在生产路径使用的策略（`qwen3/text.rs:142`、`qwen3/base.rs:421`）。
- `is_expired()`：**从未被任何生产代码调用**。没有任何后台 reaper / 定时器回收超时的 KV。

#### 4. `TenantContext` 未实现

文档中描述的「租户」维度（身份 / 配额 / KV 与模型自由组合）**完全没有对应类型**。
grep 全仓 `TenantContext` 0 命中。

#### 5. 文档承诺的 API 不存在

| 文档承诺的 API | 实际状态 |
|----------------|----------|
| `Model::new_session(capacity, kv_format)` | ❌ 仅 `Qwen3Model::generate` 内部创建 session，无公开入口 |
| `Model::new_session_with_kv(&mut kv)` | ❌ 「模式 3 外部 KV 注入」无对应 API |
| 模式 2 Stateless API（`fn generate(model, input)`） | ❌ 不存在独立函数，每次都包装在 model 路径 |

#### 6. 跨请求 KV 共享未实现

`bin/server.rs` 每次 `generate_*` 都 `KvCache::new_f16(n_layer, max_ctx, n_embd_gqa)` 全新分配，
无会话池、无跨请求复用、无租户隔离。这与文档「多租户设计」「组合矩阵」的预期严重不符。

### 落地优先级建议

1. **P0**：将 `KvCacheFmt` 合并到 `KvFormat`，消除重复。
2. **P0**：`server.rs` 改造为 `KvState` + 会话池，否则多租户无从谈起。
3. **P1**：`llama` / `lfm2` / `lfm25` / `qwen35` 迁移到 `KvState`。
4. **P1**：提供公开 `Model::new_session` / `Model::new_session_with_kv` 入口。
5. **P2**：实现 `TenantContext` 及组合矩阵。
6. **P2**：实现 `Timed` KV 的 reaper 线程，使 `is_expired()` 在生产路径生效。

---

## 未来优化

1. **KV 压缩**: 对长时间未访问的 KV 进行压缩
2. **KV 分页**: 类似 vLLM 的分页 attention
3. **KV 迁移**: 支持跨模型的 KV 迁移（架构相同时）
4. **分层 KV**: 热 KV 在 GPU，冷 KV 在 CPU
