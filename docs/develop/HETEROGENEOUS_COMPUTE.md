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

### 3.1 已经做到的

| 原则 | 落地点 | 状态 |
|---|---|---|
| 统一入口 | `models/qwen3/trunk/forward.rs` 只调 `silu_inplace`、`matmul_q8_0_quantized_parallel_rows` 等 op | ✅ |
| Scalar 兜底 | `kernel/q8_0/scalar.rs`、`activation/*::scalar_*` 等 | ✅ |
| 启动期探测 | `has_avx2_fma()` / `has_neon()` / `get_vulkan_context()` | ✅ |
| GPU 后端可插拔 | vulkan 与 wgpu 是两套独立 backend，互不耦合 | ✅ |
| 正确性参照 | `parity_trace.rs` 落盘 SIMD/GPU 输出与 scalar 对比 | ✅ |

### 3.2 偏离的部分（30%）

**派发逻辑被复制粘贴 N 份**，每加一个 op 就要复刻一遍：

```
ops/activation/silu.rs        if has_avx2_fma() { avx2 } elif has_neon() { neon } else { scalar }
ops/activation/gelu.rs        if has_avx2_fma() { avx2 } elif has_neon() { neon } else { scalar }
ops/ssm.rs                    if has_avx2_fma() { avx2 } elif has_neon() { neon } else { scalar }
ops/quant/q8_0.rs             if has_avx2_fma() { avx2 } elif has_neon() { neon } else { scalar }
ops/kernel/q8_0/dispatch.rs   同上（已抽出来，但仍是 if-链）
ops/kernel/q8_0/parallel.rs   多塞 gpu 分支，但写成内联 if 而非注册表项
```

后果：

* 加 CUDA 后端要改 6 个文件（每 op 一处 `elif`）。
* Vulkan 的派发条件（`n_in ≤ MAX_GPU_N_IN` 等启发式）和 AVX2/NEON 的派发条件（feature flag）写在一起，语义不对称。
* 「可插拔」只对 GPU 成立——SIMD 的可插拔性是 N 份复制粘贴撑出来的。

### 3.3 Vulkan 实现位置的历史决策

GPU 后端（含 shader 加载、descriptor 绑定、buffer 持久化、`GPU_BROKEN` 标记）住在 `src/vulkan.rs` 与 `src/wgpu.rs`，与 `src/ops/` 平级。这一点在 [`REFACTOR_PLAN.md`](REFACTOR_PLAN.md) §1 已经定调——

> 倾向建 `src/backend/{vulkan.rs, wgpu.rs}`——两者是设备抽象而非计算内核，放 `ops/` 会破坏 `ops 只依赖 core` 的约束。

本原则与该决策一致：后端不进入 `ops/`，但在派发表里和 SIMD 平级。

---

## 4. 目标形态：Backend Registry

### 4.1 形态

```rust
// ops/backend.rs  进程级单例，由 lib.rs 启动期初始化

pub trait Backend: Send + Sync {
    fn name(&self) -> &'static str;
    /// 数值越小越优先；scalar 必须垫底
    fn priority(&self) -> u8;
    /// 形状 / 算子版本匹配：vulkan 可以按 n_in 上限拒绝；avx2 可以拒绝非 32 对齐
    fn supports(&self, op: OpKind, shape: &Shape) -> bool;
    /// 实际执行；签名稳定，不随后端变化
    unsafe fn run(&self, op: OpKind, args: &OpArgs);
}

// 注册顺序无关紧要，priority 决定胜负
static REGISTRY: OnceLock<BackendRegistry> = OnceLock::new();

pub fn register(b: Box<dyn Backend>) {
    REGISTRY.get_or_init(BackendRegistry::default).register(b);
}

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

### 4.3 op 调用点收敛

每个 op 文件只剩「语义」，不再有 `if has_avx2_fma()`：

```rust
// ops/activation/silu.rs
pub fn silu_inplace(x: &mut [f32]) {
    dispatch(OpKind::Silu, &Shape::from(x.len()), &mut OpArgs::Unary { x });
}

// ops/kernel/q8_0/mod.rs
pub fn matmul_q8_0_quantized_parallel_rows(
    weight: &[u8], input: &[u8], scales: &[f32], output: &mut [f32],
    n_in: usize, n_out: usize, ith: usize, nth: usize,
) {
    let (row_start, row_end) = partition(n_out, ith, nth);
    dispatch(OpKind::MatmulQ8_0,
             &Shape::Matmul { n_in, n_out, row_start, row_end },
             &OpArgs::MatmulQ8_0 { weight, input, scales, output });
}
```

**行内不再出现 `*_avx2` / `*_neon` / `ctx.matmul_q8_0`。**

### 4.4 Backend 实现的位置

```
src/backend/
├── mod.rs              # Backend trait + Registry + dispatch()
├── scalar.rs           # 纯 Rust，零依赖，永远编译
├── avx2.rs             # x86_64 only，#[target_feature]
├── neon.rs             # aarch64 only，#[target_feature]
├── vulkan.rs           # ash 依赖，仅 feature="vulkan"
└── wgpu.rs             # wgpu 依赖，仅 feature="wgpu"
```

* **不进 `ops/`**——保持 `ops 只依赖 core`。
* **不进 `src/` 根**——`vulkan.rs` / `wgpu.rs` 目前挂在根目录是历史遗留（见 [`REFACTOR_PLAN.md`](REFACTOR_PLAN.md) §1）。
* **SIMD backend 也搬过来**——既然与 vulkan/wgpu 平级，就物理上住一起，「Backend 平等」才有体感和代码可读性的双重兑现。

---

## 5. Shader 资产的位置（明确不动）

`shaders/` **保持在 src/ 之外**，是 backend 之间的共享资源。原因：

| 反对把 shader 搬进 `src/ops/` 或 `src/backend/` 的理由 | 说明 |
|---|---|
| 构建管线异构 | `.comp` 走 `glslc → .spv`；`.rs` 走 cargo；混入会污染 rustfmt / rust-analyzer |
| 多 backend 共用 | `shaders/glsl/*.comp` 同时被 vulkan 和 wgpu 引用，跨 backend 共享是天然属性 |
| 真正的「Vulkan 实现」不只是 shader | 是 `matmul_q8_0`（含 buffer 上传、descriptor 绑定、fence）+ shader 两部分；shader 单独搬不能兑现「backend 平级」的对称性 |

`.spv` 通过 `include_bytes!` 嵌入 `src/backend/vulkan.rs`（迁移后），GLSL 源码仍在 `shaders/glsl/`。build 流程不变。

---

## 6. 目标形态对当前代码的差异清单

| 当前 | 目标 | 影响 |
|---|---|---|
| `ops/activation/silu.rs` 含 `if has_avx2_fma() { avx2 } …` | 仅 `dispatch(OpKind::Silu, …)` | 行数 -60%，可读性 ↑ |
| `ops/kernel/q8_0/dispatch.rs` 三选一 if-链 | 删除（并入 registry） | 一个文件消失 |
| `ops/kernel/q8_0/parallel.rs` 内联 gpu 分支 | 删除（并入 registry 的 VulkanBackend::supports） | 一个文件消失 |
| `ops/kernel/q8_0/{avx2,neon,scalar}.rs` | 迁至 `src/backend/{avx2,neon,scalar}.rs` | `ops/` 只剩语义 |
| `src/vulkan.rs` + `src/wgpu.rs` | 迁至 `src/backend/{vulkan,wgpu}.rs` | `src/` 根目录只剩入口 |
| `parity_trace.rs` 落盘对比 | 不变 | — |
| `has_avx2_fma()` / `has_neon()` 调用点 | 集中到 `lib.rs` 启动期 | 调用点归一 |

---

## 7. 迁移路径（建议顺序）

1. **先抽 trait，不搬文件。** 在 `ops/backend.rs` 引入 `Backend` trait + `dispatch()`，注册当前已有的 scalar/avx2/neon 三个 impl（impl 体直接引用 `kernel/q8_0/scalar.rs` 等现有模块）。`parallel.rs` 的 GPU 分支保留不动。这一步验证 trait 形态是否足够表达「选 backend」语义。
2. **把 op 调用点收敛到 `dispatch()`。** 一个一个 op 迁移：`silu` → `gelu` → `rms_norm` → `rope` → `quant` → `matmul`。每迁一个，删掉内联的 if-链；`cargo bench` 与 `parity_trace` 比对结果不变。
3. **VulkanBackend 接进来。** 把 `parallel.rs:38-43` 的 `gpu_takes_this` 条件搬到 `VulkanBackend::supports()`，把 `ctx.matmul_q8_0` 调用搬到 `VulkanBackend::run()`。`parallel.rs` 退化为「线程分块 + 调 dispatch」。
4. **物理搬迁。** `ops/kernel/q8_0/{avx2,neon,scalar}.rs` → `src/backend/`，`src/{vulkan,wgpu}.rs` → `src/backend/`。所有 `crate::ops::kernel::q8_0::avx2::*` 引用改为 `crate::backend::avx2::*`。
5. **加新 backend 验证形态。** 接入 wgpu（如果尚未）作为第二轮验证：只需新增 `WgpuBackend` impl + 一行 `register`，不动任何 op 文件。

---

## 8. 不在本文范围 / 显式不做的事

* **不为 backend dispatch 引入 async runtime。** 同步派发 + 短临界区，与现有 `VulkanContext::mutex` 一致。
* **不做运行期 backend 热切换。** 启动期注册，运行期只读。热切换的收益（按负载切 CPU/GPU）远小于引入状态机的复杂度。
* **不要求 backend 之间的输出 bit-exact。** `parity_trace` 在固定容差内（`|a-b| / (|a|+|b|+ε) < 1e-5`）判等；不同 ISA 的浮点舍入差异是物理事实，不假装对齐。
* **不做 backend 之外的特性探测自动化。** feature flag（`#[target_feature]`）仍是手写，但调用点（`if has_avx2_fma()`）只在 `lib.rs` 启动期出现一次。
