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

## 未来优化

1. **KV 压缩**: 对长时间未访问的 KV 进行压缩
2. **KV 分页**: 类似 vLLM 的分页 attention
3. **KV 迁移**: 支持跨模型的 KV 迁移（架构相同时）
4. **分层 KV**: 热 KV 在 GPU，冷 KV 在 CPU
