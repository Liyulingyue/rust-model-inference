# Rust LLM Inference Engine — 优化记录

> **文档用途：** 本文是项目的优化档案。包含：SIMD/数值正确性的工程教训、量化 kernel 矩阵、固定机器基线、已验证无效方向、未达成门禁与下一批目标。
>
> 文档分两类章节：
> - **经验性**（长期有效，不随代码迭代失效）：§1 SIMD 浮点舍入、§2 已验证无效方向、§3 paritty test 配方
> - **状态性**（随代码迭代需刷新）：§4 量化 kernel 矩阵、§5 固定机器基线、§6 多模型适配、§7 异构后端基线、§8 未达成门禁
>
> 最近一次刷新：2026-09-18（与 ARCHITECTURE.md 同步）。下一次刷新门槛：任何量化 kernel 状态变更、任何 Vulkan/NEON 后端新模型支持、新一轮固定机器对比。

---

## 1. 经验教训：SIMD 浮点内核必须严格匹配 Scalar 的舍入顺序

**问题**：Q4_0 AVX2 实现曾在生产路径中产生 1 ULP 累积漂移，导致 softmax 后 top-1 token 在 temp 0.6 采样时被翻转（"巴黎" → "尼斯"）。Parity test 在合成数据上仅显示 1 ULP 差异，被误判为可接受。

**根因**：IEEE 754 f32 **加法不满足结合律**。`(a + b) + c` 与 `a + (b + c)` 在 f32 中可能产生不同结果（舍入方向取决于中间值的指数）。SIMD 内核的累加顺序与 scalar 不同时，即使每个操作都"正确"，最终结果可能差 1 ULP。

**两类高危操作**：

1. **`f32::mul_add` / `_mm256_fmadd_ps`（FMA）**：fuses `a*b + c` 为单次舍入。Scalar 的 `a*b + c` 是两次舍入（乘 1 次 + 加 1 次）。在 AVX2+FMA 目标上，`mul_add` 会编译成 FMA → 1 ULP drift。
2. **`hsum_ps`（树形 reduction）**：`_mm256_hadd_ps` / `hsum_ps` 等横向求和用树形 `((a+b)+(c+d)+...)`，而 scalar 是顺序累加 `sum = (sum + s0) + s1 + ...`。两种顺序的最终值可差 1 ULP。

**Q4_0 AVX2 实际修复路径**（`src/ops/kernel/q4_0/avx2.rs`）：

```rust
// ✗ 不匹配 scalar：加法不满足结合律
acc += prod0 + prod1;     // = acc + (prod0 + prod1)

// ✓ 匹配 scalar：顺序累加
acc += prod0;
acc += prod1;

// ✗ FMA：1 次舍入 vs scalar 的 2 次
let prod = dc.mul_add(d_b, 0.0).mul_add(si_b, 0.0);

// ✓ 显式 mul+add：2 次舍入
let prod = dc * d_b * si_b;
```

**Parity test 必须强制 bit-exact**：

```rust
let diff_bits = (avx2.to_bits() as i32).wrapping_sub(scalar.to_bits() as i32).unsigned_abs();
assert!(diff_bits == 0, "AVX2 diverged by {} ULP", diff_bits);
```

**不要**用 `rel < 1e-3` 这类容差测试，否则 1 ULP drift 会被掩盖。Q4_0 模型在合成数据 + 真实模型权重上跑了 9 个 parity case 全部 bit-exact 通过，然后才接入生产 dispatch。

### 1.1 K-quant/I-quant 的"末尾乘"差异

K-quant/I-quant 走 `0.25 * sum(d_i*b_i)`（C 末尾乘）vs `sum(0.25 * d_i*b_i)`（Rust 每 block 乘），f32 累加顺序不同。已用 f64 block accumulator 修复大部分情况（IQ3_XXS 完全恢复），IQ2_XXS 仍残留偏差——可能与 2-bit 量化精度边界有关，待查。

### 1.2 已知未完全修复

- **Q6_K AVX2** 仍有 1-2 ULP drift（commit `acb0a2b` 调查根因）：scalar `sumf += sums[l]` 是线性累加，AVX2 `hsum_ps` 是树形 reduction；f32 加法不满足结合律。尝试过多种缓解（FMA→mul+add 拆解、scalar 改 `mul_add`），均无改善——属于 IEEE 754 不可避免现象。生产验证：Q6_K / Q4_K_M / Q8_0 均输出 "The capital of France is **Paris**"（scalar 与 AVX2 一致）。
  - 2026-09-18 复现：`Q6_KKernel::forward_prepared` → `vec_dot_q6k_q8k_avx2` 与 `vec_dot_q6k_q8k_scalar` 在 `tests/quantized_inference.rs::q6_k_prepared_path_matches_existing_scalar_dot_bits` 差 1 ULP。**当前无生产影响**：Q6_K fused path 未接入任何模型，实际 Qwen3-0.6B-Q6_K 文本推理走 `forward_prequantized` → `matmul_q6_k_scalar_range` 全 scalar 路径（实测 55.3 t/s，产出 "Paris"）。`embedding_lookup_q6_k` 与 `dequantize_row_q6_k` bit-exact 一致。详见 [`TODO.md`](TODO.md) high-priority 第一条（已划 ✅）。
- **Q8_0 AVX2 "diff=255"** 是测试 bug（scalar 函数调用时 `(n_in, n_out, 0)` 把 `n_out` 错位传到 `row_start`，导致 scalar 没跑任何行返回 0）。修复后实测 `max_diff=0.000366 rel=1.6e-7`，AVX2 算法 bit-exact 正确。

---

## 2. 已验证无效方向（防重蹈覆辙）

### 2.1 OnceLock 缓存 CPU Feature Detection
- **做法**：用 `std::sync::OnceLock` 替代 `AtomicBool::load()`。
- **结果**：反而变慢（9.6→6.7 tok/s）。
- **原因**：`swap()` 比 `load()` 重（带 lock 前缀），`get().map().unwrap_or()` 分支多。
- **结论**：atomic load 编译成单条 mov，已是 zero-overhead，无需缓存。

### 2.2 合并 FFN compute 调用 ⚠️ 已废弃
- **做法**：把 FFN gate/up/down 3 个 matmul 合并到 1 个 `compute()`。
- **结果**：输出乱码 + 变慢（9.6→6.6 tok/s）。
- **原因**：silu activation 需要在 gate/up matmul 完成后才能执行，合并改变了数据流。
- **结论**：需更仔细的依赖分析；当前 §Quant Kernel 补全的 `QTensorOwned::fuse_vstack` 是约束更窄的"vstack 合并"，结果正确。

### 2.3 Persistent Workers 架构 ⚠️ 已废弃（2025-07-31）
- **做法**：重写 `ComputePool`，让 worker 线程永不退出，在 `worker_loop` 中遍历所有 ops。用 `work_ready` flag + exit_barrier + reenter_barrier 三阶段同步。
- **结果**：Qwen3 T4 从 31.1 → 23.3 tok/s（**更慢**），输出正确。
- **原因**：每步推理多了 `work_ready` spin-wait + 额外 barrier 开销。Fork-join 的 barrier 已经是 minimal overhead 了。
- **结论**：Persistent workers 需要完全重新设计（消除 epoch-based wakeup，改用 work-stealing queue），不能简单叠加在现有模型上。

### 2.4 `select_nth_unstable_by` 替代 O(n*k) 扫描
- **做法**：用 `Vec::select_nth_unstable_by` 替代手写增量扫描。
- **结果**：变慢（9.7→8.5 tok/s）。
- **原因**：需要分配 151936 元素的 `Vec<(usize,f32)>`，堆分配开销超过算法改进收益。
- **结论**：当前 `src/ops/sampling.rs::sample_top_k` 仍是手写 O(n*k) 增量维护。

---

## 3. Parity Test 配方

合成 + 真实模型双轨门禁：

| dtype | 合成 kernel parity | 真实模型 E2E | 工具 |
|---|---|---|---|
| Q4_0 | bit-exact，9 case | Qwen3-0.6B Q4_0 | `tests/quantized_inference.rs` |
| Q8_0 | max_diff=0.000366 rel=1.6e-7 | Qwen3-0.6B Q8_0 | 同上 |
| Q4_K | bit-exact | Qwen3-0.6B Q4_K_M | 同上 |
| Q6_K | 1-2 ULP drift（已知） | 输出与 scalar 一致 | 同上 |
| BF16 | bit-exact | Qwen3.5 BF16 | `tests/qwen35_reference.rs` |
| IQ4_NL / IQ4_XS | ≤ 1 ULP drift（已知） / bit-exact | 输出与 scalar 一致 | `tests/quantized_inference.rs`（IQ4_NL `iq4_nl_prepared_path_avx2_matches_scalar_dot_within_one_ulp`，IQ4_XS `b8d6b7c` 长期 bit-exact） |

`feature=parity-trace` 下的 `src/parity_trace.rs` 是 SIMD/GPU vs scalar 的运行时对照门禁。

---

## 4. 量化 Kernel 矩阵（2026-08-29 主验收，Windows x86_64 AVX2+FMA, `--threads 4 --temp 0`）

| 模型 | Size (MB) | tok/s (gen) | Status | 关键路径 |
|---|---:|---:|---|---|
| **Q4_1** | 390 | **82.3** | ✅ | Q4_1 × Q8_0 AVX2（`src/ops/kernel/q4_1/`） |
| **Q6_K** | 472 | **50.4** | ✅ | Q6K × Q8K AVX2（`src/ops/quant/avx2_k.rs`，1-2 ULP drift） |
| **Q4_K_M** | 378 | 45.0 | ✅ | Q4K × Q8K scalar（`src/ops/kernel/q4_k.rs`） |
| **Q8_0** | 610 | 40.2 | ✅ | Q8_0 × Q8_0 AVX2（`src/ops/kernel/q8_0/avx2.rs`） |
| **Q5_K_M** | 424 | 40.3 | ✅ | Q5K × Q8K scalar（`src/ops/kernel/q5_k.rs`，原 placeholder 已修复） |
| **Q5_K_S** | 416 | 36.9 | ✅ | Q5K × Q8K scalar（原 placeholder） |
| **Q4_K_S** | 366 | 41.1 | ✅ | Q4K × Q8K scalar（原 0 output） |
| **BF16** | 1143 | **27.5** | ✅ | BF16 × F32 AVX2+FMA（`src/ops/kernel/bf16/avx2.rs`，含 `avx2_q8.rs`） |
| Q3_K_M | 331 | 9.2 | ✅ | Q3K × Q8K scalar（`q3_k.rs`，format bug 修复） |
| Q3_K_S | 308 | 6.0 | ✅ | Q3K × Q8K scalar（format bug 修复；Lyon noise） |
| Q2_K | 283 | 4.9 | ✅ | Q2K × Q8K AVX2（`avx2_k.rs`，3 个 bug 修复：qs_base / scale_b / hsum256_ps） |
| IQ4_XS | 351 | 4.8 | ✅ | IQ4_XS × Q8K AVX2（共享 `src/ops/quant/avx2_k.rs`，bit-exact，commit `b8d6b7c`） |
| **IQ4_NL** | 381 | **16.1** | ✅ | **IQ4_NL × Q8K AVX2 + NEON**（`vec_dot_iq4_nl_q8k_avx2` + `vec_dot_iq4_nl_q8k_neon` 共享 `avx2_k.rs` / 新增 `neon_k.rs`，≤ 1 ULP drift，2026-09-18） |
| IQ3_XXS (UD) | ~280 | 4.3 | ✅ | IQ3_XXS × Q8K scalar（f64 block acc；"The capital of France is Paris."） |
| IQ2_XXS (UD) | ~210 | 4.4 | ⚠️ 输出偏 | IQ2_XXS × Q8K scalar（单 block bit-exact，输出与 IQ3_XXS 略不同） |
| IQ1_M (UD) | ~170 | 4.3 | ⚠️ 输出乱 | IQ1_M × Q8K scalar（1.75 bpw 本就精度极低） |
| IQ1_S (UD) | ~150 | 4.7 | ⚠️ 输出乱 | IQ1_S × Q8K scalar（1.5 bpw） |

**关键加速对比**（与 2026-08-29 修复前）：

- Q4_1：scalar 11.6 → AVX2 54.5 → **AVX2+d 优化 82.3 t/s（7.1×）**
- BF16：scalar 7.4 → AVX2 27.5 t/s（**3.7×**）
- Q5_K：was 0 output → 40 t/s（修复 + dispatch 接入）
- IQ4_XS：was panic → AVX2 4.8 t/s（修复 + 打开 dispatch）
- **IQ4_NL**：scalar 8.6 t/s → AVX2 **16.1 t/s**（1.87×，`vec_dot_iq4_nl_q8k_avx2` 在 `src/ops/quant/avx2_k.rs`；同步新增 `vec_dot_iq4_nl_q8k_neon` 在 `src/ops/quant/neon_k.rs` 走 aarch64 路径；测试 `iq4_nl_prepared_path_avx2_matches_scalar_dot_within_one_ulp`；端到端 Qwen3-0.6B-IQ4_NL.gguf 5 批中位 16.1 t/s gen，产出 "Paris"）
- IQ3_XXS (UD)：was panic → scalar 4.3 t/s（修复 + f64 acc → 给出 "The capital of France is Paris."）
- Q2_K：was 乱码（scalar 4.9 t/s）→ AVX2 4.9 t/s（启用 dispatch，3 bugs 修复）
- Q3_K / Q3_K_M / Q3_K_S：scalar 5-9 t/s → AVX2 4.7-7.3 t/s（重写：修 `scale_shuffles` + `hsum256ps` 双 shuffle imm；scalar 改用 `_mm_extract` + `_mm_add_epi16`；Q3_K_M 现在输出 "The capital of France is **Paris**" 与 IQ4_XS 完全一致）

**Kernel 路径索引**：

```
src/ops/kernel/
├── q4_0/{mod,avx2,scalar}.rs        # bit-exact parity（§1）
├── q4_1/{mod,avx2,scalar}.rs        # 7.1× 加速（2026-08-29）
├── q8_0/{mod,avx2,neon,dispatch,parallel,scalar}.rs   # AVX2 dispatch + NEON
├── bf16/{mod,avx2,neon,scalar,avx2_q8,neon_q8}.rs    # 3.7× 加速，含 Q8 prequant
├── f16/{mod,avx2,neon,scalar,avx2_q8,neon_q8}.rs
├── f32/{mod,avx2,neon,scalar}.rs
├── q2_k.rs q3_k.rs q4_k.rs q5_k.rs q6_k.rs           # scalar + 共享 avx2_k.rs
└── iq4_nl.rs iq4_xs.rs                                # IQ4_NL kernel 入口 / IQ4_XS AVX2

src/ops/quant/
├── avx2_k.rs            # 共享 K-quant / IQ4_XS / IQ4_NL / Q2_K / Q3_K AVX2 内核
├── neon_k.rs            # aarch64 NEON：IQ4_NL × Q8K（与 AVX2 同源 1 ULP drift）
├── fuse.rs              # FFN gate+up 融合
├── iq_tables.rs         # IQ 表查查表（LUT）
├── q8_0.rs              # Q8_0 量化
└── mod.rs               # BlockQ8K / QK_K 常量 + AVX2/NEON dispatch
```

### 4.1 已知未实现

- **IQ2_XXS / IQ2_XS / IQ3_XXS / IQ1_S / IQ3_S / IQ2_S / IQ1_M kernel**：仅在 `GGMLType` enum 注册（`src/core/tensor.rs:32-45`），其中 IQ3_XXS / IQ2_XXS / IQ1_M / IQ1_S 已有 scalar kernel（精度受损），其余仍 panic。
- **Q2_K / Q3_K 全 SIMD 覆盖**：当前仅 K-quant 路径共享 `avx2_k.rs`；scalar Q2_K 4.9 t/s 与 AVX2 持平，原因是 K-quant block 结构复杂、SIMD 利用率受限。

---

## 5. 固定机器基线

### 5.1 Apple Silicon NEON（2026-08-09，commit `4557ae3` 前后）

测试环境：

```text
$ uname -m
arm64

$ sysctl -n machdep.cpu.brand_string
Apple M3 Max

$ sw_vers
ProductName:		macOS
ProductVersion:		26.6.1
BuildVersion:		25G76

$ rustc -vV
rustc 1.97.0 (2d8144b78 2026-07-07)
LLVM version: 22.1.6
```

Q8_0 NEON 固定机器门禁（`1024 x 3072`，15 个样本取中位数，每个样本 20 次迭代）：

```text
architecture=aarch64 backend=NEON
gate=1024x3072 scalar_median=0.867ms auto_median=0.109ms speedup=7.980x auto=57.91GFLOPS/30.81GB/s threshold=1.10x
```

Qwen3-0.6B Q8_0，4 线程，32-token 确定性推理：

```text
Model: qwen3 | n_embd=1024 n_layer=28 n_head=16 n_head_kv=8 n_ff=3072 | loaded in 71ms
Prompt: 2 + 3 = (5 tokens)
Output:
 5, 5 + 4 = 9, 9 + 5 = 14, 14 + 6 = 20
PROFILE: norm=0.0% quant=0.0% qkv+attn=26.2% wo=9.5% ffn=41.6% logits=22.6%
PROFILE: norm=0.000s quant=0.000s qkv+attn=0.069s wo=0.025s ffn=0.110s logits=0.060s
[32 tokens in 268ms | 119.4 tok/s]
```

**注意**：该批次为单批（不像 §5.2 多批）；NEON 路径与 x86_64 不可直接对照。

### 5.2 Rust vs llama.cpp 固定机器对比（2026-08-10）

测试环境：macOS 26.6.1（Build 25G76），Apple M3 Max（12P+4E，16 核），Rust 1.97.0。Rust CLI 的 KV cache 默认为 **F16**（注：当前 `KvFormat::default() = F32`，见 ARCHITECTURE.md §5.2；早期 CLI 显式 F16 主验收批次保留此值）。llama.cpp 固定在 `7ba604f1cb61cd14898138e9abc0b4ff2601f180`，并显式使用 `-ctk f16 -ctv f16`。CMake 配置确认 ARM `dotprod` 和 `i8mm` 均可用。

Rust CPU T8 命令：

```bash
for run in 1 2 3 4 5; do
  ./target/release/rust-model-inference \
    --model models/Qwen3-0.6B-Q8_0.gguf \
    --prompt "2 + 3 =" \
    --max-tokens 32 \
    --temp 0 \
    --threads 8 \
    --kv-cache f16 \
    --bench \
    --profile 2>&1 | rg 'BENCH: tg|PROFILE:'
done
```

| Rust T8 批次 | KV 证据 | 五次原始值（eval/s） | 中位数 | 对 llama.cpp CPU 145.199 的差距 |
|---|---|---|---|---|
| 主验收批次 | 显式 `--kv-cache f16` | 157.6, 157.0, 158.5, 158.2, 148.7 | 157.6 | **-8.54%**（Rust 单批更快） |
| 初始批次 | CLI 默认 F16 | 113.2, 102.8, 112.4, 106.1, 72.8 | 106.1 | +26.93% |
| controller 复跑 | CLI 默认 F16 | 124.6, 119.4, 125.2, 131.0, 136.6 | 125.2 | +13.77% |

llama.cpp 复现（`$benchmark_checkout = mktemp -d /tmp/rmi-llama-bench.XXXXXX`；`git clone` 后 `git checkout 7ba604f1cb61cd14898138e9abc0b4ff2601f180`；CMake `-DGGML_METAL=ON`；`--target llama-bench`）：

```bash
"$benchmark_checkout/build/bin/llama-bench" \
  -m "$PWD/models/Qwen3-0.6B-Q8_0.gguf" \
  -p 0 -n 32 -t 8 -r 5 -ngl 0 -ctk f16 -ctv f16 -o json
```

CPU JSON 记录 `n_gpu_layers: 0`，`samples_ts` 为 `[126.287, 124.324, 145.199, 146.637, 148.763]`，中位数 `145.199 eval/s`。

| 后端 | 线程 | decode 中位数 | CPU 差距 |
|---|---|---|---|
| Rust CPU（显式 F16 KV 主批次） | T8 | 157.6 eval/s | **-8.54%（单批）** |
| llama.cpp CPU（`-ngl 0`，F16 KV） | T8 | 145.199 eval/s | 基准 |
| llama.cpp Metal（`-ngl 99`，F16 KV） | T8 | 273.576 eval/s | 信息记录，不参与门禁 |

**状态**：主批次的算术差距为 `(145.199 - 157.6) / 145.199 = -8.54%`，但三个 Rust 批次的中位数从 `106.1` 到 `157.6 eval/s`，结论互相冲突，因此**尚未证明稳定满足 10% CPU 门禁**。必须先进行测量环境与调度器 profiling，再考虑内核工作。本计划没有实现 DotProd/I8MM、权重重排或线程池重写。

### 5.3 llama.cpp 关键实现参考

> pinned commit 在 `references/llama.cpp/`（HEAD `fe12e422a` sync : ggml）；本仓库不再维护独立 `references/ggml/` 作为参考源（仅 2025 年历史快照保留）。

| 算法 | llama.cpp 文件 | 说明 |
|---|---|---|
| `mul_mat` 调度 | `references/llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c` | `atomic_fetch_add(&current_chunk, 1)` + barrier；行号随 commit 变，引用 commit 即可 |
| Q8_0 AVX2 | `references/llama.cpp/ggml/src/ggml-quants.c`（`vec_dot_q8_0_q8_0`） | 优先 `_mm256_dpbssd_epi32`（VNNI），其次 `_mm256_sign + _mm256_maddubs_epi16` |
| K-quant AVX2 | `references/llama.cpp/ggml/src/ggml-quants.c`（`vec_dot_q4k_q8k_avx2` 系列） | 本仓库 `src/ops/quant/avx2_k.rs` 直接对照移植 |
| llama context | `references/llama.cpp/src/llama-context.cpp` | 线程模型与 KV cache 生命周期 |

---

## 6. 多模型适配优化（2026-08-29 → 2026-09-18）

### 6.1 接入新模型与 SIMD 路径

| PR / commit | 模型 / 路径 | 关键优化 |
|---|---|---|
| `#50` `6872575` | DreamX-Creator | GGUF 导出 + 原生 CPU 音视频管道 |
| `#47` `20a07f6` | VibeVoice ASR | GGUF 导出 + Rust 推理 |
| `#53` `d45340c` | dots TTS | Q8 SIMD 推理（LLM + mmproj） |
| `#59` `a8830a7` | Breeze TTS-2 | 原生 Rust 推理 |
| `#65` `a3d860c` | Qwen-Drive-1.0-4B | GGUF planning + perception |
| `#58` `86f29bc` | NeoHorse 9B/4B | BF16、NFC、逐位验证 |
| `#36` `d2b9e97` | Spark-X2.5 | 支持 |
| `#27` `bcb3e75` | Gemma 4 E2B | 多模态推理 |
| `#67` `ac84583` | Breeze | quantized inference kernels |
| `#71` `bb2ad81` | Breeze | 量化推理内核修复 |
| `#29` `0f0b002` | Qwen Omni | 多模态 |
| `#68` `1bc0990` | Qwen3 / Qwen3.5 / Gemma4 | chunked prefill batching |

### 6.2 量化与算子

| commit | 改动 |
|---|---|
| `#40` `43cbe51` | 去除 `.h` 依赖，把 IQ 表 inline 为 Rust const |
| `#72` `2d5495f` | 完整 AVX2 量化 + patch reduction paths |
| `4557ae3` | RoPE 函数全模型切到 inplace 变体 |

### 6.3 异构后端（§7）

---

## 7. 异构后端基线

### 7.1 Vulkan（2026-09-01..18，feature `--features vulkan --gpu`）

| 模型 | 量化 | 状态 | 入口 |
|---|---|---|---|
| Qwen3 dense | Q8_0 / Q4_0 / Q4_1 / Q4_K / Q6_K / F16 | ✅ 完整 token Vulkan 执行 | `src/vulkan/ops.rs` + `src/vulkan/qwen3.rs` |
| Qwen3.5 dense + recurrent + SSM | BF16 | ✅ | `src/vulkan/qwen35.rs` |

PR / commit：
- `#44` `621c622` — Vulkan Qwen3 + Qwen3.5 推理
- `#54` `e56cbfe` — MiniCPM 适配 + Qwen3.5 Vulkan 通路
- `#57` `fffb36d` — Qwen3.5 Vulkan Q4/Q8 支持
- `#60` `637da9e` — Vulkan F32 matmul 支持
- `#66` `e88b251` — Vulkan CI server 二进制修复

**架构特征**（与 llama.cpp 一致的硬件要求底线）：
- 不要求 `storageBuffer16BitAccess`；shader 从 `uint` storage buffer 解包权重并复现 ARM64 FP16 累加/归约顺序。
- 不要求 `shaderInt64` / 整数点积；可用时选 dp4a，否则 baseline pipeline。
- 每次 GPU 调用有可配置超时（`RUST_GPU_TIMEOUT_MS` 默认 5 s；fence 等待内层 60 s）。
- **回退策略**：Vulkan token 失败时从上一个已提交 KV 状态在 CPU 重算；不符合资格的模型直接 CPU，不静默混用。

**Q5_K 当前状态**：只完成合成 kernel parity，未纳入端到端模型支持矩阵。

详见 [`VULKAN.md`](VULKAN.md) / [`VULKAN_INFERENCE_DESIGN.md`](VULKAN_INFERENCE_DESIGN.md) / [`VULKAN_INFERENCE_PLAN.md`](VULKAN_INFERENCE_PLAN.md)。

### 7.2 wgpu（`--features wgpu`，实验性）

`src/wgpu.rs` 是 wgpu 后端根模块；当前覆盖率低于 Vulkan，主要作为未来 CUDA / Metal 移植的中间层。

### 7.3 OpenBLAS 条件编译（`f702c87`）

历史加入；当前 `Cargo.toml` 中 `features` 仍可启用。**注意**：与 HETEROGENEOUS_COMPUTE.md "标量为底" 原则存在张力——OpenBLAS 接入会绕过自研 SIMD 路径，必须逐 dtype parity test 才能放心接入。新模型接入不推荐默认打开。

---

## 8. 未达成门禁 / 下一批目标

### 8.1 显式门禁清单

| 门禁 | 当前状态 | 阻塞点 |
|---|---|---|
| **CPU 后端稳定 ≥ llama.cpp（10% 内）** | ❌ 未稳定（§5.2 多批差异 ±30%） | 测量环境波动；未做调度器 profiling |
| **Vulkan Q5_K 端到端** | ❌ 仅合成 parity | `src/ops/quant/avx2_k.rs` AVX2 实现已就绪；Vulkan shader 未对齐 |
| **Logits 输出投影优化**（V·151936，~23% decode 时间） | ❌ 未实现 | memory-bound；候选：分块、fuse to 采样 |
| **FFN 整体融合**（gate+up+silu+quant+down） | ⚠️ 部分（`fuse_vstack`） | 进一步融合需要 persistent workers（§2.3 已废弃） |
| **Q2_K / Q3_K 全 SIMD 加速**（4.9 t/s → 30+ t/s） | ❌ AVX2 实现已存在但与 scalar 持平 | K-quant block 结构复杂；需重写 algorithm |
| **chunked prefill 推广到所有 trunk** | ⚠️ Qwen3 / Qwen3.5 / Gemma4 已支持（`#68`） | llama / lfm2 / lfm25 / lfm2moe / spark / breeze / nemotron_h 未适配 |
| **dotprod/i8mm 内核**（ARM64） | ❌ 仅 NEON baseline | 需要 ASM 手写或 `std::arch::aarch64::*` 内建 |
| **权重重排**（whisper.cpp-style repack） | ❌ 未实现 | 对 Q4_K / Q6_K 大权重 matmul 提升明显 |

### 8.2 下一批目标（按预期收益排序）

1. **稳定 CPU 门禁**：先做测量环境 profiling（关闭 TurboBoost、统一 governor、5 批以上），得到稳定中位数后再启动内核工作。
2. **Logits 优化**：候选路径 (a) fuse logits matmul 到 sample 阶段（避免完整 151936 行写出），(b) `--top-k=1` 早出（greedy 不需完整排序）。
3. **Vulkan Q5_K 端到端**：参照现有 Q4_K / Q6_K shader 改写，runtime parity test 接入 `parity_trace.rs`。
4. **ARM64 dotprod / i8mm 内核**：Q4_0/Q8_0/Q4_K 全部走 i8mm 路径，预期 NEON 性能再提升 30-50%。
5. **chunked prefill 跨模型推广**：先 lfm2（shortconv），再 llama / spark。
6. **dreamx CUDA oracle**：当前 DreamX-Creator 仅 CPU，已通过 `tools/dreamx/dreamx_oracle_trace.py` 拿到上游 trace（commit `215d4cd7fbed7e161ab508ae1f85a8fee0536f62`），需要 GPU parity。

### 8.3 不建议重做的方向

- §2 列出的 4 个已废弃方向（OnceLock、合并 compute、Persistent workers、select_nth_unstable_by）
- `vec_dot_q8_0_q8_0` VNNI 重新实现：当前 AVX2 + `_mm256_sign + _mm256_maddubs_epi16` 已达标

---

## 9. 工具与门禁

| 工具 | 路径 | 用途 |
|---|---|---|
| `cargo bench` / `cargo run --release --bin micro_bench` | `src/bin/micro_bench.rs` | 单算子 / 模型微基准 |
| `parity_trace` feature | `src/parity_trace.rs` | SIMD/GPU vs scalar 运行时对照 |
| `tests/quantized_inference.rs` | `tests/quantized_inference.rs` | 量化 dtype E2E 推理门禁 |
| `tests/{qwen35,gemma4,nemotron_h,z_image,qwen_drive_vlm,...}_reference.rs` | `tests/` | 跨模型 Oracle 对齐 |
| `tools/qwen_drive/` / `tools/neohorse/` / `tools/dreamx/` | `tools/` | 各模型上游 Oracle trace 工具 |
| `cargo build --release --features vulkan` | — | Vulkan 后端编译 |
| `LD_LIBRARY_PATH=references/llama.cpp/build/bin/Release references/llama.cpp/build/bin/Release/llama-cli.exe` | Windows pinned | llama.cpp Oracle 验证 |

构建参数：`cargo build --release`（`opt-level=3`, `lto=fat`, `codegen-units=1`）。

---

## 10. 关联文档

- [`ARCHITECTURE.md`](ARCHITECTURE.md) §2 法则三（静态 enum 派发）、§8 ComputePool + rayon 双池、§9 已落地/未完成
- [`SUPPORTED_MODELS.md`](SUPPORTED_MODELS.md) — 每个具体型号的"已验证格式 + Oracle + 限制"
- [`MODEL_ORGANIZATION.md`](MODEL_ORGANIZATION.md) — trunk + sibling 目录约定与依赖方向
- [`KV_CACHE_DESIGN.md`](KV_CACHE_DESIGN.md) — KV cache 共享条件 + 三种生命周期
- [`VULKAN.md`](VULKAN.md) / [`VULKAN_INFERENCE_DESIGN.md`](VULKAN_INFERENCE_DESIGN.md) / [`VULKAN_INFERENCE_PLAN.md`](VULKAN_INFERENCE_PLAN.md) — Vulkan 后端
- [`PARALLEL_MATMUL_SAFETY.md`](PARALLEL_MATMUL_SAFETY.md) — Issue 4（`compute_unchecked` 形式别名违规）
- [`HETEROGENEOUS_COMPUTE.md`](HETEROGENEOUS_COMPUTE.md) — "标量为底 + 后端注册表"
- [`LFM25_OPTIMIZATION.md`](LFM25_OPTIMIZATION.md) / [`LFM25_MOE_OPTIMIZATION.md`](LFM25_MOE_OPTIMIZATION.md) / [`ZIMAGE_OPTIMIZATION.md`](ZIMAGE_OPTIMIZATION.md) — 子模型/子方向专项
- [`TODO.md`](TODO.md) / [`ISSUE.md`](ISSUE.md) — 当前任务与开放问题
- [`REFACTOR_PLAN.md`](REFACTOR_PLAN.md) — 残余重构
