# 异构算力分发：标量为底 + 自动派发

> **文档用途：** 本文定义 `rust-model-inference` 在多后端（AVX2 / NEON / Scalar / Vulkan / wgpu / 未来 CUDA · Metal）共存场景下的设计原则、当前形态评估与目标架构。配套的依赖方向约束见 [`MODEL_ORGANIZATION.md`](MODEL_ORGANIZATION.md)、GPU 模块归属见 [`REFACTOR_PLAN.md`](REFACTOR_PLAN.md) §1。

---

## 1. 核心原则

异构算力的推理引擎只能有一条主航线：

```
   统一推理代码（models/）
            │
            ▼
   Backend Registry（自动派发）
       │     │     │     │
       ▼     ▼     ▼     ▼
   Scalar  AVX2  NEON  Vulkan   (… wgpu / CUDA / Metal)
       ▲
       └── 兜底，所有其他后端的正确性参照
```

**三条不可妥协的规则：**

1. **标量为底。** Scalar 实现是 floor，是其他所有后端的正确性参照。任何 SIMD/GPU 内核必须能与 scalar 在固定容差内对齐；`parity_trace.rs` 是这套对齐的回归门禁。
2. **统一入口。** `models/` 下的 trunk 只能调用 op 级别的统一 API（如 `silu_inplace`、`matmul_q8_0_quantized_parallel_rows`），不直接见到任何 `*_avx2` / `*_neon` / `*_vulkan` 的具名符号。
3. **自动派发。** 选哪个后端是 runtime 决定——`has_avx2_fma()`、`has_neon()`、`get_vulkan_context()`、`is_aarch64_feature_detected!("dotprod")` 等探测在启动时完成；op 调用点不参与选择。

这一形态与 llama.cpp / GGML、oneDNN、PyTorch ATen 是同构的：一份算子语义，多份实现，runtime 选实现，scalar 兜底。

---

## 2. 与现有架构的对齐

引擎已有两条相关硬约束，本原则是它们的延伸：

* **依赖方向**：`models/` → `ops/` → `core/`，**`ops/` 不依赖设备抽象**（见 [`REFACTOR_PLAN.md`](REFACTOR_PLAN.md) §1）。
  这条规则直接否决了「把 `vulkan.rs` 塞进 `ops/`」的方向——ash 依赖、设备生命周期、buffer 持久化不属于 `ops/` 的语义。
* **物理零拷贝 + 静态 Arena**（见 [`ARCHITECTURE.md`](ARCHITECTURE.md) §2）：算子在 L1/L2 cache 内 in-place 接力，dispatch 引入的开销必须为 0 或可忽略——分支预测友好的查表优先于 trait 虚调用。

所以「Backend Registry」不是凭空发明的，而是这两条约束 + 异构算力现实的合并产物。

---

## 3. 当前形态盘点（2026-09）

### 3.1 两个派发维度（核心结论，文档必须先讲清楚）

异构算力派发**不是单一维度**——目前的代码里实际上存在**两个正交的派发轴**，未来 Backend Registry 必须同时承载二者：

| 派发轴 | 现有机制 | 已选定的实现 |
|---|---|---|
| **dtype 轴**：选「哪个 dtype 走哪条 SIMD 路径」 | `Kernel` trait + `Box<dyn Kernel>`（`ops/kernel/trait_.rs`、`weight.rs`、`quantized_tensor.rs`） | 启动期按 `GGMLType` 一次定型；运行期仅在 `dispatch.rs / parallel.rs` 等少数点内做 if-链 |
| **op 轴**：silu / rope / norm 等 element-wise | 直接调用 `ops::silu_inplace`、`ops::rope_neox_inplace` 等函数，每个函数体内部 if-链 | — |

dtype 轴是 **dequant + matmul** 的内层选择；op 轴是 **element-wise + activation** 的选择。Vulkan 的位置更特殊——它**横跨两轴**：dtype 轴上覆盖 Q8_0/Q4_0/Q4_1/Q4_K/Q5_K/Q6_K/F16/BF16/F32 等 9 种权重的整段 matmul；op 轴上覆盖 rms_norm、qk_norm_rope、silu_mul、softmax、attention_values、kv_write 等（见 `src/vulkan/ops.rs` 的 23 个 shader pipeline）。

### 3.2 已经做到的

| 原则 | 落地点 | 状态 |
|---|---|---|
| 标量为底 | `kernel/q8_0/scalar.rs`、`activation/*::scalar_*`、`quant/scalar` 等，每个 dtype kernel 都有 scalar fallback | ✅ |
| 启动期探测 | `has_avx2_fma()` / `has_neon()` / `has_f16c()` / `get_vulkan_context()` 在 `ops/float.rs` 一次性 `init_cpu_features()` 写死 | ✅ |
| GPU 后端可插拔 | `vulkan.rs` 暴露 `VulkanContext`，`wgpu.rs` 暴露 `WgpuContext`；都通过 `OnceLock` 懒初始化 | ✅（仅 vulkan 有生产路径；wgpu 是 operator-check 级别的原型，见 §3.6） |
| 正确性参照 | `parity_trace.rs` 落盘 SIMD/GPU 输出与 scalar 对比；`#[cfg(feature = "parity-trace")]` 同时强制 SIMD 路径退化为 scalar（`ops/quant/q8_0.rs:7` 等） | ✅ |
| dtype 抽象 | `Kernel` trait（`forward_prequantized / forward_prepared`）+ 13 个 dtype 实现 + `QuantizedTensor` 枚举 + `Weight<'a>` 持 `Box<dyn Kernel>` | ✅ |
| GPU 会话 | `Qwen3VulkanSession`（`src/vulkan/qwen3.rs`）、`Qwen35VulkanSession`（`src/vulkan/qwen35.rs`），启动期 `check_eligibility` + 整段 forward 上 GPU | ✅ |
| 模型级 GPU 开关 | `gpu_matmul_active()` / `disable_gpu_matmul_for_scope()` / `GPU_BROKEN` 三层组合：模型决策 + RAII 作用域 + 进程级熔断 | ✅ |

### 3.3 偏离的部分（实际是 ~50% 不是 30%）

**派发复制粘贴的散布**比初版文档说的多得多。现状盘点：

| 类别 | 文件 | 内容 |
|---|---|---|
| op 级 element-wise if-链 | `ops/activation/{silu,gelu}.rs`、`ops/norm.rs`、`ops/softmax.rs`、`ops/rope/neox.rs`、`ops/math/{exp,sigmoid,tanh}.rs`、`ops/ssm.rs` | AVX2 / NEON / scalar 三选一；`ssm.rs` 只有 AVX2，无 NEON |
| op 级工具函数 | `ops/dot.rs` | **~15 处** AVX2/NEON if-链（`dot_f32_*`, `vec_scale_f32_*`, `vec_mad_f32_*`, `sum_*_f32_*` 等）；F16C、`fp16` 子特性在运行时再判一次 |
| dtype 级 matmul 入口 | `ops/quant/q8_0.rs` | `quantize_q8_0_into` / `quantize_q8_0_into_parallel`：AVX2/NEON/scalar |
| **dtype 级 SIMD 派发** | `ops/kernel/{bf16,f16,f32,q4_0,q4_1,q8_0}/mod.rs`（双 SIMD 后端）；`q4_0/mod.rs`、`q4_1/mod.rs`（仅 AVX2）；`q8_0/dispatch.rs`（三选一） | 每个 kernel 文件 `forward_prequantized / forward_prepared` 内独立 if-链 |
| **k-quant / IQ4 内核** | `ops/kernel/{q2_k, q3_k, q4_k, q5_k, q6_k, iq4_nl, iq4_xs}.rs` | `forward_prequantized` 多为 stub（dequant-to-f32 兜底），真实路径在 `forward_prepared` 调 `ops::quant::vec_dot_*_q8k`（Q8K 共享 activation；IQ4_NL / IQ4_XS 走 `avx2_k.rs` AVX2 路径，IQ4_NL 同步有 `neon_k.rs` aarch64 路径） |
| 量化辅助 | `ops/quant/avx2_k.rs`（`vec_dot_q2k_q8k_avx2`、`vec_dot_q3k_q8k_avx2`、`vec_dot_iq4_xs_q8k_avx2` 等） | 一组独立的 AVX2/scalar vec_dot 函数，被 kernel `forward_prepared` 调用——和 dtype if-链正交 |
| **Vulkan per-matmul 入口** | `ops/kernel/q8_0/parallel.rs:28-87` | 全范围 GPU 派发 + 失败回退到全行 CPU 重算（线程 0 独占） |
| **模型级 GPU 会话** | `src/vulkan/qwen3.rs`、`src/vulkan/qwen35.rs` | 整段 forward 在 GPU 上跑，含 eligibility 检查、token-commit 状态机；调用方是 `models/{qwen3,qwen35}/trunk/forward.rs` |
| **模型层 GPU 决策** | `models/{qwen3,qwen35,gemma4}/trunk/{forward,session,prefill}.rs` 与 `models/{llama,lfm2,lfm25}/trunk/forward.rs` | `gpu_matmul_active()` 检查、`disable_gpu_matmul_for_scope()` RAII、`full_model_gpu_failed` 标志 |

后果（修正初版文档的判断）：

* **加 CUDA 后端要改的不是 8 处，而是 30+ 处**——每 op 一处 if-链 + 每 dtype kernel 一处 if-链 + vec_dot 一处。
* **dtype 轴已经有 trait 抽象了**（`Kernel`），只是元素级 op 轴（silu/rope/norm）没有——Backend Registry 应优先收敛 op 轴。
* **GPU 的派发条件**不仅是 shape（`MAX_GPU_N_IN`），还有 model-level 决策（`Qwen3VulkanSession::check_eligibility` 校验 weight format、是否有 MoE、是否 qkv-bias 等）。Backend Registry 的 `Backend::supports()` 必须能承载这种「基于模型元数据」的拒绝条件。
* **模型层有自己的 GPU 开关语义**——`disable_gpu_matmul_for_scope()` 是 RAII 线程局部 flag，独立于 `gpu_broken()` 静态熔断。初版文档没意识到这一层。
* **wgpu 不是 vulkan 的等价兄弟**——`src/wgpu.rs` 仅实现 `WgpuContext::matmul_q8_0`，且从未被生产路径调用；`ops/kernel/q8_0/parallel.rs` 的 GPU 分支只判 vulkan。wgpu 当前是 operator-check 级别的实验后端，不是平行后端。

### 3.4 Vulkan 实现位置的现状

GPU 后端分散在三处：

| 位置 | 内容 |
|---|---|
| `src/vulkan.rs` | `VulkanContext`、`matmul_q8_0`、`GPU_BROKEN`、`get_vulkan_context()` 入口；Q8_0 整段 matmul 的低层封装 |
| `src/vulkan/ops.rs`（~5000 行） | **23 个 shader 算子**：`QUANTIZE_Q8_0 / QUANTIZE_Q8_K / Q8_MATMUL_GROUPED / Q4_0_MATMUL / Q4_1_MATMUL / Q4_K_MATMUL / Q5_K_MATMUL / Q6_K_MATMUL / F16_MATMUL / BF16_MATMUL / F32_MATMUL / RMS_NORM / QK_NORM_ROPE / KV_WRITE / ATTENTION_SCORES / SOFTMAX / ATTENTION_VALUES / SILU_MUL / ADD / QWEN35_DENSE_PREPARE / QWEN35_ATTENTION / QWEN35_RECURRENT_CONV / QWEN35_RECURRENT_SSM`。配套：`ArenaLayout` / `ArenaRegion` / `OperatorBindings` / `TokenCommands` / `Qwen3Ops` / `GpuWeightFormat` |
| `src/vulkan/qwen3.rs`、`src/vulkan/qwen35.rs` | `Qwen3VulkanSession` / `Qwen35VulkanSession`：启动期 `check_eligibility`（架构、weight format、是否 MoE、是否 qkv-bias 等），运行期承担整段 forward 的录制 + 提交 |

初版文档把 GPU 描述为「`vulkan.rs` + `wgpu.rs` 与 ops/` 平级」是不完整的——实际的 GPU 体量在 `src/vulkan/ops.rs` 一个文件就 ~5000 行，远超文档所暗示。

### 3.5 与初版设计相比的漂移（修复项）

初版文档（HETEROGENEOUS_COMPUTE 上一版）的几处判断在事实层面已经不准：

1. **「如果 Vulkan 是单 matmul + 设备抽象」**：错。Vulkan 已演化为「23 算子算子运行时 + 两个模型会话」。
2. **「Backend Registry 没建，是空白」**：部分错。`Kernel` trait 已经是 dtype 轴的事实 Registry；缺的只是元素级 op 轴的统一入口。
3. **「加 CUDA 要改 6 个文件」**：低估约 5 倍（见 §3.3）。
4. **「wgpu 与 vulkan 平级」**：错。wgpu 是实验后端，不在派发路径上。
5. **「派发决策是 `if has_avx2_fma() { avx2 } elif has_neon() { neon } else { scalar }` 三选一」**：部分错。`ssm.rs` 没有 NEON 分支；`q4_0 / q4_1` 没有 NEON 分支；k-quant 走的是 `vec_dot_*_q8k` 路径而不是 Q8_0 SIMD。

### 3.6 wgpu 现状

`src/wgpu.rs` 提供 `WgpuContext::matmul_q8_0`，但：

* 仅实现 Q8_0 matmul 一个算子；
* 没有出现在 `parallel.rs` 的 GPU 分支（只判 vulkan）；
* 没有出现在 `Qwen3VulkanSession` / `Qwen35VulkanSession`；
* 仅被 `vk_*_check` 系列测试与 operator-check 路径使用。

Backend Registry 不应在「wgpu 平等接入」之前提这件事——除非先实现 23 个算子并接入两条 GPU 会话路径。

---

## 4. 目标形态：Backend Registry

### 4.1 形态（修正：两轴并存）

```rust
// ops/backend.rs  进程级单例，由 lib.rs 启动期初始化
//                dtype 轴的 Registry 已经由 Kernel trait 承担（Box<dyn Kernel>）
//                Backend Registry 主要承载 op 轴 + 把 VulkanSession 也收纳进来

pub trait Backend: Send + Sync {
    fn name(&self) -> &'static str;
    /// 数值越小越优先；scalar 必须垫底
    fn priority(&self) -> u8;
    /// 形状 / 算子版本匹配：vulkan 可以按 n_in 上限拒绝；avx2 可以拒绝非 32 对齐
    /// 模型层也支持：vulkan 可以因架构 / MoE / qkv-bias 拒绝
    fn supports(&self, op: OpKind, shape: &Shape) -> bool;
    /// 实际执行；签名稳定，不随后端变化
    unsafe fn run(&self, op: OpKind, args: &OpArgs);
}

/// 进程级单例 + 启动期注册 + 运行期只读
static REGISTRY: OnceLock<BackendRegistry> = OnceLock::new();

pub fn register(b: Box<dyn Backend>) { ... }

/// op 调用点的统一入口：循环 backends，第一个 supports() 的胜出
pub fn dispatch(op: OpKind, shape: &Shape, args: &OpArgs) {
    let registry = REGISTRY.get().expect("backends not registered");
    for b in registry.backends_for(op) {
        if b.supports(op, shape) {
            unsafe { b.run(op, args) };
            return;
        }
    }
    unreachable!("scalar backend must always be registered");
}
```

**与 `Kernel` trait 的关系**（重要）：

* `Backend` 是 **op 轴** 的派发入口：silu / gelu / rope / norm / softmax / dot / vec_scale / vec_mad / quantize_q8 / ssm / matmul。
* `Kernel` 是 **dtype 轴** 的派发入口：每个 dtype 在构造 `Weight<'a>` 时定型，运行期仅在 kernel 自己的 `forward_prequantized` 内做 AVX2/NEON/scalar if-链。
* `OpKind::MatmulQ8_0` 等 matmul 变体在 `Backend` 层只选「matmul 这个 op 用 CPU SIMD 还是 Vulkan Session」；具体走哪个 SIMD 是 `Kernel` 的事。
* Vulkan Session 是**特殊的 backend**：它不只执行一个 op，而是执行一段 forward 序列（Qwen3 的 23 算子链）。它的 `supports` 校验模型元数据 + shape，`run` 提交整段 forward。这是 Backend Registry 必须明确容纳的「序列 backend」，不是简单的「per-op backend」。

### 4.2 注册示例

```rust
// lib.rs 启动期
register(Box::new(ScalarBackend));                          // priority=100, 兜底
#[cfg(target_arch = "x86_64")]
if has_avx2_fma() { register(Box::new(Avx2Backend)); }     // priority=10
#[cfg(target_arch = "aarch64")]
if has_neon()     { register(Box::new(NeonBackend)); }     // priority=10
#[cfg(feature = "vulkan")]
if let Ok(b) = VulkanBackend::try_new() { register(Box::new(b)); }  // priority=0
#[cfg(feature = "wgpu")]
if let Ok(b) = WgpuBackend::try_new()   { register(Box::new(b)); }  // priority=5
```

注意：`Qwen3VulkanSession` / `Qwen35VulkanSession` **不是** `register` 进去的——它们由 `models/{qwen3,qwen35}/trunk/session.rs` 在创建 session 时主动尝试（`try_new_rows`），失败则 `full_model_gpu_failed = true`。这是「序列 backend」和「per-op backend」的差异：前者启动期需要 model-weights 信息，无法在 lib.rs 启动期注册。

### 4.3 op 调用点收敛

每个 op 文件只剩「语义」，不再有 `if has_avx2_fma()`：

```rust
// ops/activation/silu.rs
pub fn silu_inplace(x: &mut [f32]) {
    dispatch(OpKind::Silu, &Shape::from(x.len()), &mut OpArgs::Unary { x });
}

// ops/kernel/q8_0/mod.rs   —— Kernel trait 内部仍走 if-链；这是 dtype 轴
pub fn matmul_q8_0_quantized_parallel_rows(
    weight: &[u8], input: &[u8], scales: &[f32], output: &mut [f32],
    n_in: usize, n_out: usize, ith: usize, nth: usize,
) {
    let (row_start, row_end) = partition(n_out, ith, nth);
    // op 轴：选「per-matmul CPU SIMD 还是 Vulkan full-range dispatch」
    dispatch(OpKind::MatmulQ8_0,
             &Shape::Matmul { n_in, n_out, row_start, row_end },
             &OpArgs::MatmulQ8_0 { weight, input, scales, output });
}
```

**行内不再出现 `*_avx2` / `*_neon` / `ctx.matmul_q8_0`。**

### 4.4 Backend / Kernel 实现的位置

```
src/backend/
├── mod.rs              # Backend trait + Registry + dispatch()
├── scalar.rs           # 纯 Rust，零依赖，永远编译
├── avx2.rs             # x86_64 only，#[target_feature]
├── neon.rs             # aarch64 only，#[target_feature]
├── vulkan.rs           # ash 依赖，仅 feature="vulkan"
└── wgpu.rs             # wgpu 依赖，仅 feature="wgpu"

src/kernel/   # 已经存在；命名不冲突但属同一族
├── mod.rs              # Kernel trait（dtype 轴）
├── bf16/{avx2,avx2_q8,neon,neon_q8,scalar}.rs
├── f16/{avx2,avx2_q8,neon,neon_q8,scalar}.rs
├── f32/{avx2,avx2_q8,neon,neon_q8,scalar}.rs
├── q4_0/{avx2,scalar}.rs
├── q4_1/{avx2,scalar}.rs
├── q8_0/{avx2,dispatch,neon,parallel,scalar}.rs
├── q2_k.rs / q3_k.rs / q4_k.rs / q5_k.rs / q6_k.rs
└── iq4_nl.rs（IQ4_NL kernel 入口；AVX2 实现在 src/ops/quant/avx2_k.rs 共享路径，NEON 实现在 src/ops/quant/neon_k.rs）
└── iq4_xs.rs（IQ4_XS kernel 入口；AVX2 实现在 src/ops/quant/avx2_k.rs 共享路径）

src/vulkan/
├── ops.rs              # 23 个 shader 算子（不搬，作为 Vulkan Session 的内部实现）
├── qwen3.rs            # Qwen3VulkanSession（不搬，模型级 backend）
└── qwen35.rs           # Qwen35VulkanSession（同上）
```

* **`Kernel` 已存在，名字保留**——它处理 dtype 轴；Backend Registry 处理 op 轴 + Vulkan 序列 backend。
* **`ops/kernel/` 不进 `src/backend/`**——`Kernel` 是 op 轴的 dtype 内部抽象，搬动会破坏 `models/ → ops/ → core/` 的依赖方向（`models/` 通过 `Weight<'a>` 持有 `Box<dyn Kernel>`，依赖不能换路径）。
* **`src/vulkan/` 子目录整体不进 `src/backend/`**——它是模型级 GPU Session，不是通用 op 后端；硬塞 `backend/` 会让 `models/{qwen3,qwen35}/trunk/session.rs` 反向依赖 `models/` 的具体类型。
* **Vulkan `matmul_q8_0` 与 `Qwen3VulkanSession` 共存**：前者是 per-matmul 入口（在 `parallel.rs` 被 dispatch）；后者是整段 forward。两者**不互相替代**——per-matmul 路径服务的是非 Qwen3/Qwen3.5 模型（llama、lfm2、lfm25 等）以及 Qwen3 在 GPU Session 不可用时的兜底。

---

## 5. Shader 资产的位置（明确不动）

`shaders/` **保持在 src/ 之外**，是 backend 之间的共享资源。原因：

| 反对把 shader 搬进 `src/ops/` 或 `src/backend/` 的理由 | 说明 |
|---|---|
| 构建管线异构 | `.comp` 走 `glslc → .spv`；`.wgsl` 走 wgpu runtime；`.rs` 走 cargo；混入会污染 rustfmt / rust-analyzer |
| 多 backend 共用 | `shaders/glsl/*.comp` 同时被 vulkan 和 wgpu 引用，跨 backend 共享是天然属性 |
| 真正的「Vulkan 实现」不只是 shader | 是 `matmul_q8_0`（含 buffer 上传、descriptor 绑定、fence）+ shader 两部分；shader 单独搬不能兑现「backend 平级」的对称性 |

`.spv` 通过 `include_bytes!` 嵌入 `src/vulkan/ops.rs` 与 `src/vulkan.rs`，GLSL 源码仍在 `shaders/glsl/`。`shaders/wgsl/matvec_q8_0.wgsl` 由 `src/wgpu.rs` 通过 `include_str!` 嵌入。build 流程不变。

---

## 6. 目标形态对当前代码的差异清单

| 当前 | 目标 | 影响 |
|---|---|---|
| `ops/activation/silu.rs` 含 `if has_avx2_fma() { avx2 } …` | 仅 `dispatch(OpKind::Silu, …)` | 行数 -60% |
| `ops/activation/gelu.rs` 含 6 处 if-链 | 仅 `dispatch(OpKind::Gelu{,Erf,Approx}, …)` | 行数 -50% |
| `ops/ssm.rs` 仅 AVX2 分支 | `dispatch(OpKind::SsmStateDecay, …)`（新增 NEON 后端顺带补齐） | 行数 -40%；aarch64 性能变化 |
| `ops/quant/q8_0.rs` 含 AVX2/NEON if-链 | `dispatch(OpKind::QuantizeQ8_0, …)` | 行数 -50% |
| `ops/norm.rs`、`ops/softmax.rs`、`ops/rope/neox.rs`、`ops/math/{exp,sigmoid,tanh}.rs` 含 if-链 | `dispatch(OpKind::*, …)` | 行数 -40% ~ -60% |
| `ops/dot.rs` ~15 处 if-链 | `dispatch(OpKind::{DotF32, DotF16, VecScale, VecMad, Sum, SumSq}, …)` | 行数 -50% |
| `ops/kernel/q8_0/dispatch.rs` 三选一 if-链 | 删除（并入 registry） | 一个文件消失 |
| `ops/kernel/q8_0/parallel.rs` 内联 vulkan 分支 | 删除 GPU 分支（并入 registry 的 `VulkanBackend::supports` / `run`） | 一个文件消失；保留线程分块逻辑 |
| `ops/kernel/{bf16,f16,f32,q4_0,q4_1}/mod.rs` 等的 `forward_*` 内 if-链 | **保留**——这是 dtype 轴的 Kernel trait 内部逻辑；op 轴的 dispatch 不直接落到这些 | 不变 |
| `ops/kernel/{q2_k, q3_k, q4_k, q5_k, q6_k, iq4_nl, iq4_xs}.rs` 的 `forward_prepared` 内 if-链（间接调 `vec_dot_*_q8k`） | **保留**——同上，dtype 轴不动 | 不变 |
| `src/vulkan.rs` + `src/vulkan/{ops,qwen3,qwen35}.rs` | 物理位置保留；语义封装成 `VulkanBackend`（per-op 入口）+ `Qwen{3,35}VulkanSession`（序列 backend） | 新增 trait 包装，模块位置不变 |
| `src/wgpu.rs` | 物理位置保留；语义封装成 `WgpuBackend`（per-op 入口，仅 Q8_0 matmul 一个 op） | 新增 trait 包装；生产路径仍未启用 |
| `parity_trace.rs` 落盘对比 | 不变；feature gate 与 cfg 不变 | — |
| `has_avx2_fma()` / `has_neon()` 调用点 | 集中到 `lib.rs` 启动期；op 调用点不再出现 | 调用点归一 |
| `core/thread_pool.rs` 的 `GpuMatmulScope` / `GPU_MATMUL_DISABLED` | **保留**——Backend Registry 外的独立机制（线程局部 RAII）；通过 `Backend::supports` 的额外查询间接集成 | 两者并存 |
| `models/{qwen3,qwen35,gemma4}/trunk/{forward,session,prefill}.rs` 的 `gpu_matmul_active()` / `disable_gpu_matmul_for_scope()` | 模型层决策保留——Backend Registry 不接管「模型级 GPU Session」生命周期 | 保持现状 |
| `models/{llama,lfm2,lfm25}/trunk/forward.rs` 的 `gpu_matmul_active()` | 模型层决策保留；通过 `dispatch(OpKind::MatmulQ8_0, …)` 间接消费 Vulkan per-matmul 入口 | 行为不变 |
| `lib.rs` 启动期 `enable_gpu()`（在 `main.rs:177`、`bin/server.rs:798` 调用） | 不变 | — |

---

## 7. 迁移路径（建议顺序）

1. **先抽 `Backend` trait，不搬文件。** 在 `ops/backend.rs`（或新增 `src/backend/mod.rs`）引入 `Backend` trait + `dispatch()`，注册当前已有的 scalar/avx2/neon 三个 impl（impl 体直接调用现有模块）。`Kernel` trait 不动，dtype 轴保持原状。这一步验证 op 轴的 trait 形态是否足够表达「选 backend」语义。
2. **把 op 调用点收敛到 `dispatch()`。** 一个一个 op 迁移：`silu` → `gelu` → `rms_norm` → `rope_neox` → `quantize_q8_0` → `dot` → `vec_scale` → `vec_mad` → `sum_*` → `softmax` → `ssm`。每迁一个，删掉内联的 if-链；`cargo bench` 与 `parity_trace` 比对结果不变。
3. **VulkanBackend 接 per-op 入口。** 把 `parallel.rs:28-87` 的 `gpu_takes_this` 条件搬到 `VulkanBackend::supports()`，把 `ctx.matmul_q8_0` 调用搬到 `VulkanBackend::run()`。`parallel.rs` 退化为「线程分块 + 调 dispatch」。这一步只动 Q8_0 一个 op；其他 dtype 的 Vulkan 入口（`vulkan/ops.rs` 的 23 算子）暂不接 dispatch。
4. **wgpuBackend 接 per-op 入口（仅 Q8_0）。** 把 `src/wgpu.rs:203 matmul_q8_0` 包装为 `WgpuBackend::run`。生产路径仍然默认走 vulkan（priority=0 vs wgpu=5），wgpu 仅作为 fallback 或 operator-check。
5. **Vulkan Session 不进 Registry。** `Qwen3VulkanSession` / `Qwen35VulkanSession` 保留现状——它们是模型级序列 backend，由 `models/{qwen3,qwen35}/trunk/session.rs` 主动创建，**不通过 `Backend::register` 注册**。Backend Registry 处理 per-op backend；模型级 GPU Session 处理整段 forward。两者通过 model-level `gpu_matmul_active()` / `full_model_gpu_failed` 协作。
6. **物理搬迁（可选）。** 仅当 `ops/backend.rs` 文件过大时把 `src/vulkan.rs` 与 `src/vulkan/{ops,qwen3,qwen35}.rs` 物理搬到 `src/backend/{vulkan,ops,qwen3,qwen35}.rs`。`Kernel` trait 与 dtype kernel 文件**不搬**——它们是 op 轴的内部抽象，与 Backend Registry 正交。
7. **加新 backend 验证形态。** 接入 CUDA 时，**只需新增 `CudaBackend` impl + 一行 `register`**（per-op 入口）；23 算子与模型级 Session 是另一份工作量，不应混入本次「op 轴统一入口」的迁移。

---

## 8. 不在本文范围 / 显式不做的事

* **不为 backend dispatch 引入 async runtime。** 同步派发 + 短临界区，与现有 `VulkanContext::mutex` 一致。
* **不做运行期 backend 热切换。** 启动期注册，运行期只读。热切换的收益（按负载切 CPU/GPU）远小于引入状态机的复杂度。
* **不要求 backend 之间的输出 bit-exact。** `parity_trace` 在固定容差内（`|a-b| / (|a|+|b|+ε) < 1e-5`）判等；不同 ISA 的浮点舍入差异是物理事实，不假装对齐。
* **不做 backend 之外的特性探测自动化。** feature flag（`#[target_feature]`）仍是手写，但调用点（`if has_avx2_fma()`）只在 `lib.rs` 启动期出现一次。
* **不接管模型级 GPU Session 的生命周期。** `Qwen3VulkanSession` / `Qwen35VulkanSession` 由 `models/{qwen3,qwen35}/trunk/session.rs` 创建与销毁；Backend Registry 只承载 per-op backend。
* **不强制 wgpu 接入生产路径。** wgpu 当前是 operator-check 级别的实验后端，与 vulkan 不是平级关系。
* **不重写 dtype 轴。** `Kernel` trait + 13 个 dtype kernel + `QuantizedTensor` + `Weight<'a>` 已稳定，是 op 轴的统一依赖面；Backend Registry 是 op 轴的薄包装，不是 `Kernel` 的替代。
* **`parity_trace` 不仅是落盘**——feature 启用时同时强制 SIMD 路径退化为 scalar（`ops/quant/q8_0.rs:7` 等处的 `#[cfg(all(target_arch = "x86_64", not(feature = "parity-trace")))]`）。Backend Registry 收敛后这一行为必须保留。