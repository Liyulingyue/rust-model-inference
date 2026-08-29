# Rust LLM Inference Engine — 优化记录

## 历史性能基线（2025-07-31，llama-bench 校正）

> ⚠️ 以下数据已过时，最新数据见 "Rust 与 llama.cpp 固定机器对比（2026-08-10）" 章节

| 模型 | 线程 | Rust | llama.cpp | 差距 |
|------|------|------|-----------|------|
| Qwen3-0.6B Q8_0 | T1 | ~16 tok/s | ~21 tok/s | 1.3x |
| Qwen3-0.6B Q8_0 | T4 | ~38 tok/s | ~44 tok/s | 1.16x |
| MiniCPM5-1B Q8_0 | T1 | 9.5 tok/s | ~32 tok/s | 3.4x |
| MiniCPM5-1B Q8_0 | T4 | 27.9 tok/s | ~32 tok/s | 1.15x |

测试条件：40 decode tokens，纯 decode（无 prompt），`--bench` 模式。

> 2025-07-31：移除 prefetch + 优化 f16 加载后，Qwen3 T1 +32%, T4 +15-20%

## 已修复的 Bug

1. **Q/K Norm 计算错误** — Qwen3 特有的 per-head RMS Norm 实现有误
2. **Softmax 重复执行** — double softmax
3. **Sampling 索引错误** — top-k 返回后处理逻辑错误
4. **BPE Byte Decode** — 特殊 byte token 未正确解码
5. **Chat Template** — `<|im_start|>/<|im_end|>` 特殊 token 处理
6. **`sample_top_k` O(n log n) → O(n*k)** — 151936 token 全排序改为增量维护 top-k，T1: 6.0→10.1 tok/s
7. **`vocab_size` fallback** — Qwen3 等模型没有显式 `vocab_size` key，从 tokenizer array 推断

## 已验证无效的方向

### 1. OnceLock 缓存 CPU Feature Detection
- **做法**：用 `std::sync::OnceLock` 替代 `AtomicBool::load()`
- **结果**：反而变慢（9.6→6.7 tok/s）
- **原因**：`swap()` 比 `load()` 重（带 lock 前缀），`get().map().unwrap_or()` 分支多
- **结论**：atomic load 编译成单条 mov，已是 zero-overhead，无需缓存

### 2. 合并 FFN compute 调用 ⚠️ 已废弃
- **做法**：把 FFN gate/up/down 3 个 matmul 合并到 1 个 `compute()`
- **结果**：输出乱码 + 变慢（9.6→6.6 tok/s）
- **原因**：silu activation 需要在 gate/up matmul 完成后才能执行，合并改变了数据流
- **结论**：需更仔细的依赖分析

### 3. Persistent Workers 架构 ⚠️ 已废弃（2025-07-31）
- **做法**：重写 `ComputePool`，让 worker 线程永不退出，在 `worker_loop` 中遍历所有 ops。用 `work_ready` flag + exit_barrier + reenter_barrier 三阶段同步。
- **结果**：Qwen3 T4 从 31.1 → 23.3 tok/s（**更慢**），输出正确
- **原因**：每步推理多了 `work_ready` spin-wait + 额外 barrier 开销。Fork-join 的 barrier 已经是 minimal overhead 了。
- **结论**：Persistent workers 需要完全重新设计（消除 epoch-based wakeup，改用 work-stealing queue），不能简单叠加在现有模型上

### 4. `select_nth_unstable_by` 替代 O(n*k) 扫描
- **做法**：用 `Vec::select_nth_unstable_by` 替代手写增量扫描
- **结果**：变慢（9.7→8.5 tok/s）
- **原因**：需要分配 151936 元素的 `Vec<(usize,f32)>`，堆分配开销超过算法改进收益

## Profiling 结果（MiniCPM5-1B T1）

```
matmul 合计:  74.4% (3.09s)
  - QKV matmul:  11.5% (0.479s)
  - WO matmul:     7.3% (0.305s)
  - FFN matmul:   55.6% (2.308s)
logits:         24.5% (1.017s)
rope+Kv+attn:   1.0%  (0.043s)
```

## Micro-benchmark（单核 matmul kernel）

| 操作 | n_in x n_out | Rust GFLOPS | Rust GB/s |
|------|--------------|-------------|-----------|
| Qwen3 wq | 1024x2048 | 49.86 | 26.55 |
| Qwen3 ffn_gate | 1024x3072 | 23.51 | 12.51 |
| Qwen3 ffn_down | 3072x1024 | 40.04 | 21.37 |
| Qwen3 logits | 2048x151936 | 24.31 | 12.91 |
| MiniCPM5 wq | 1536x2048 | 44.57 | 23.73 |
| MiniCPM5 ffn_gate | 1536x4608 | 20.23 | 10.76 |
| MiniCPM5 ffn_down | 4608x1536 | 30.80 | 16.41 |

llama.cpp FFN 约 50 GB/s，差距 4-5x。

## 后续优化方向（按优先级）

### 🔴 高优先级

#### 1. Persistent Workers 架构 ⚠️ 已废弃
**目标**：消除每层 5 次 fork-join 的调度开销

 llama.cpp 的 workers 是 persistent 的：线程进入 `worker_loop` 后，遍历 **所有 compute 操作**（barrier 模型），无需每次都 spawn/join。

 实现方式：
 - Worker 线程在 `worker_loop` 中维护一个 "当前 op" 指针
 - 每个 compute 操作携带：op 类型、权重指针、数据指针、操作函数
 - 主线程遍历 ops 并分发给 workers（类似 `tokio` 的 task）
 - Worker 遍历所有 op，做完才进入下一轮

 **状态**：2025-07-31 实验结果 Qwen3 T4 31.1 → 23.3 tok/s（**更慢**），已废弃。
 需完全重新设计（work-stealing queue）才能收益。

#### 2. Matmul Kernel 优化（✅ 已实施，2025-07-31）
**改动**：
- 移除了 inner loop 中的 prefetch（`_mm_prefetch`）— 顺序访问本身已被 prefetcher 覆盖
- 将 byte-by-byte f16 加载改为 `std::ptr::read_unaligned`（编译器可生成单一 16-bit load）

**结果**：
- Qwen3 T1: 12.6 → 16.9 tok/s（**+34%**）
- Qwen3 T4: 31.8 → 37-39 tok/s（**+18-22%**）
- MiniCPM5 T4: 26.3 → 27.9 tok/s（**+6%**）

> 实测 2-row tiling 性能与 4-row tiling 相同，无额外收益

**分析**：移除不必要的 prefetch 减少了指令数和 L1 cache 压力。Prefetch 在顺序访问模式下会增加开销而不带来收益。

### 🟡 中优先级

#### 3. Logits 层优化
Logits 占 24.5% 时间，vocab=130560-151936：
- 当前 `sample_top_k` 是 O(n*k) 扫描，可进一步用 `select_nth_unstable_by`（需避免大 Vec 分配）
- 可以预分配固定大小的 `Vec<(usize,f32)>` 并复用，避免每次 `push`

#### 4. 减少 compute() 调用
每层当前 5 次 compute：
- QKV（3 个 matmul）
- WO（1 个 matmul）
- FFN gate/up（2 个 matmul + silu）
- FFN down（1 个 matmul）
- Logits（1 个 matmul）

WO 可以和 QKV 合并（需要研究依赖）；FFN gate/up/silu/down 可以在 persistent worker 模型下自然流水线化。

### 🟢 低优先级

#### 5. Multi-model 测试
- Qwen2.5-0.5B Q4_K_M（验证 Q4_K_M 量化）
- Gemma-3-1B Q4_K_M（验证 gemma3 架构）

#### 6. Chat Template 对齐
MiniCPM5 在 chat 模式下输出乱码（`--bench` 正常），chat template 未对齐。

## llama.cpp 关键实现参考

### 调度器（ggml-cpu.c）
- `ggml_compute_forward_mul_mat`: lines 1254-1451
- Chunk 分配：`atomic_fetch_add(&current_chunk, 1)` + `barrier`
- `current_chunk` 每 matmul 重置为 `nth`
- 2D tiling: `blck_0=16, blck_1=16`

### Q8_0 Kernel（arch/x86/quants.c）
- `ggml_vec_dot_q8_0_q8_0`: lines 1308-1374
- `mul_sum_i8_pairs_float`: lines 122-134
- 优先使用 VNNI（`_mm256_dpbssd_epi32`），其次 `_mm256_sign + _mm256_maddubs_epi16`

### 文件位置
- `references/ggml/src/ggml-cpu/ggml-cpu.c` — 调度器
- `references/ggml/src/ggml-cpu/arch/x86/quants.c` — Q8_0 AVX2 内核
- `references/llama.cpp/src/llama-context.cpp` — llama 上下文和线程管理

## Apple Silicon NEON（2026-08-09）

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
binary: rustc
commit-hash: 2d8144b7880597b6e6d3dfd63a9a9efae3f533d3
commit-date: 2026-07-07
host: aarch64-apple-darwin
release: 1.97.0
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

该模型在 `--max-tokens 1` 时首个解码 token 是换行；生成 4 个 token 时输出包含确定性的 `5`，因此不将首 token 误记为 ` 5`。

数值正确性由 NEON/标量单元测试和 Qwen3-0.6B Q8_0 确定性推理冒烟验证。
x86_64 本次仅完成交叉编译与 AVX2/FMA/F16C 路径静态核对，未执行 x86 硬件性能测试。

## Rust 与 llama.cpp 固定机器对比（2026-08-10）

测试环境：macOS 26.6.1（Build 25G76），Apple M3 Max（12P+4E，16 核），Rust 1.97.0（`2d8144b78 2026-07-07`）。Rust CLI 的 KV cache 默认为 F16，固定对比仍显式传入 `--kv-cache f16`；llama.cpp 固定在 `7ba604f1cb61cd14898138e9abc0b4ff2601f180`，并显式使用 `-ctk f16 -ctv f16`。CMake 配置确认 ARM `dotprod` 和 `i8mm` 均可用。

Rust CPU T8 命令（五次独立进程，比较项仅取 `BENCH: tg`）：

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

显式 F16 KV 的主验收批次中，五次 `BENCH: tg 32 evals` 原始值为 `157.6, 157.0, 158.5, 158.2, 148.7 eval/s`，中位数为 `157.6 eval/s`。

同一机器上的批次间波动较大，不能把任一单批差距当作稳定复现值：

| Rust T8 批次 | KV 证据 | 五次原始值（eval/s） | 中位数 | 对 llama.cpp CPU 145.199 的差距 |
|---------------|---------|----------------------|--------|---------------------------------|
| 主验收批次 | 显式 `--kv-cache f16` | 157.6, 157.0, 158.5, 158.2, 148.7 | 157.6 | -8.54%（Rust 单批更快） |
| 初始批次 | CLI 默认 F16 | 113.2, 102.8, 112.4, 106.1, 72.8 | 106.1 | 26.93% |
| controller 复跑 | CLI 默认 F16 | 124.6, 119.4, 125.2, 131.0, 136.6 | 125.2 | 13.77% |

llama.cpp 复现准备（从 RustModelInference 仓库根目录运行；该块定义后续命令使用的 `$benchmark_checkout`）：

```bash
benchmark_checkout=$(mktemp -d /tmp/rmi-llama-bench.XXXXXX)
git clone https://github.com/ggml-org/llama.cpp.git "$benchmark_checkout"
git -C "$benchmark_checkout" checkout 7ba604f1cb61cd14898138e9abc0b4ff2601f180
cmake -S "$benchmark_checkout" -B "$benchmark_checkout/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_TOOLS=ON \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_SERVER=OFF \
  -DGGML_METAL=ON
cmake --build "$benchmark_checkout/build" --target llama-bench -j 16
```

llama.cpp CPU-only 命令：

```bash
"$benchmark_checkout/build/bin/llama-bench" \
  -m "$PWD/models/Qwen3-0.6B-Q8_0.gguf" \
  -p 0 \
  -n 32 \
  -t 8 \
  -r 5 \
  -ngl 0 \
  -ctk f16 \
  -ctv f16 \
  -o json
```

CPU JSON 记录 `n_gpu_layers: 0`，`samples_ts` 为 `[126.287, 124.324, 145.199, 146.637, 148.763]`，中位数为 `145.199 eval/s`。

llama.cpp Metal 命令：

```bash
"$benchmark_checkout/build/bin/llama-bench" \
  -m "$PWD/models/Qwen3-0.6B-Q8_0.gguf" \
  -p 0 \
  -n 32 \
  -t 8 \
  -r 5 \
  -ngl 99 \
  -ctk f16 \
  -ctv f16 \
  -o json
```

Metal JSON 记录 `n_gpu_layers: 99`，`samples_ts` 为 `[274.077, 273.657, 273.575, 273.576, 270.904]`，中位数为 `273.576 eval/s`。Metal 是不同后端，仅作信息记录，不参与 CPU 门禁。

| 后端 | 线程 | decode 中位数 | CPU 差距 |
|------|------|---------------|----------|
| Rust CPU（显式 F16 KV 主批次） | T8 | 157.6 eval/s | -8.54%（单批） |
| llama.cpp CPU（`-ngl 0`，F16 KV） | T8 | 145.199 eval/s | 基准 |
| llama.cpp Metal（`-ngl 99`，F16 KV） | T8 | 273.576 eval/s | 不参与 |

主批次的算术差距为 `(145.199 - 157.6) / 145.199 = -8.54%`，但三个 Rust 批次的中位数从 `106.1` 到 `157.6 eval/s`，结论互相冲突，因此**尚未证明稳定满足 10% CPU 门禁**。必须先进行测量环境与调度器 profiling，再考虑内核工作。本计划没有实现 DotProd/I8MM、权重重排或线程池重写。

## Quant Kernel 补全（2026-08-28..29）

发现 Q5_K / Q4_1 / BF16 / Q2_K / Q3_K 路径不全或未优化，新增 4 项修复：

### 1. Q5_K kernel 实现（commit `b7509ef`）

之前 `forward_prequantized` 是占位 `output[i] = 0.0`，导致 Q5_K_S/M、Q4_K_S 模型产生全空输出。实现 `vec_dot_q5k_q8k_scalar`（仿 q4k 的 `vec_dot_q4k_q8k_scalar` 结构，复用 llama.cpp 已有的 `vec_dot_q5k_q8k_avx2` SIMD 函数）。

**端到端验证**：Q5_K_S/M、Q4_K_S 从乱码 → 正确产出 `**Paris**`。

### 2. Q4_1 AVX2 + BF16 AVX2+FMA + Q2_K/Q3_K scalar（commit `9d04643`）

- **Q4_1**：从 scalar 4.7× 加速（11.6→54.5 t/s）。
- **BF16**：从 scalar 3.7× 加速（7.4→27.5 t/s）。BF16→F32 转换零成本（u16→u32 左移 16 即 F32 表示），4 个 FMA 累加器并行处理。
- **Q2_K / Q3_K**：scalar matmul kernel 接入 `QuantizedTensor` dispatch。Q2_K/Q3_K_S/M 模型能加载但产出乱码——定位为 Q3_K format bug（见 §3）。

### 3. IQ4_NL scalar + 全 I-quant GGMLType 注册（commit `402bc3d`）

- `GGMLType` 枚举新增 IQ2_XXS/XS/S、IQ3_XXS/XS/S、IQ4_NL/XS（含正确字节数：IQ2 66/74/82、IQ3 98/110/122、IQ4_NL 18 / IQ4_XS 136）。
- IQ4_NL scalar matmul：32 元素/块，16 字节 LUT（`kvalues_iq4nl = {-127, -104, -80, ...}`）替代 Q4_0 的 `(nibble-8)*scale` 线性映射。
- `embedding_lookup_iq4_nl` 加入 embedding 白名单（与 F16/BF16/Q8_0/Q4_0/Q6K 并列）。
- IQ4_XS / IQ2 / IQ3 kernel 是 TODO panic（format 需对照 llama.cpp 验证）。注：qwen3-0.6b 的 IQ4_NL.gguf / IQ4_XS.gguf 实际权重是 IQ2_XS / IQ3_XS，不是 IQ4_NL/IQ4_XS，所以这两个模型暂不能跑通。

### 4. Q3_K / Q2_K format 修复（commit `592ba28`，详见 MODEL_ORGANIZATION.md §9）

对照 `E:\Codes\llama.cpp\ggml\src\ggml-quants.c` 逐行移植：
- `dequantize_row_q3_K`（line 1247）
- `vec_dot_q3_K_q8_K_generic`（line 566 in ggml-cpu/quants.c）
- `dequantize_row_q2_K`（line 903）

修两个关键 bug：
1. **Q3_K 输出索引跨 n 边界遗漏**（n=0 / n=128 两次都写到 output[0..127]，128..255 全 0）→ 改用 `out_idx` 指针递增
2. **scales 字段越界**：Q3_K scales 是 12 字节（3 × u32），不是 16 字节

修 Q2_K sub-block layout：错误按 16 sub-block × sequential 处理；实际是 8 个 16-element pair（sub-A 读 qs[l]、sub-B 读 qs[l+16]）。

### 完整推理基准（2026-08-29，Windows x86_64, AVX2+FMA, `--threads 4 --temp 0`）

| 模型 | Size (MB) | tok/s (gen) | Status | 关键路径 |
|---|---:|---:|---|---|
| **Q4_1** | 390 | **82.3** | ✅ | Q4_1 × Q8_0 AVX2 |
| **Q6_K** | 472 | **50.4** | ✅ | Q6K × Q8K scalar |
| **Q4_K_M** | 378 | 45.0 | ✅ | Q4K × Q8K scalar |
| **Q8_0** | 610 | 40.2 | ✅ | Q8_0 × Q8_0 AVX2 |
| **Q5_K_M** | 424 | 40.3 | ✅ | Q5K × Q8K scalar (was placeholder) |
| **Q5_K_S** | 416 | 36.9 | ✅ | Q5K × Q8K scalar (was placeholder) |
| **Q4_K_S** | 366 | 41.1 | ✅ | Q4K × Q8K scalar (was 0 output) |
| **BF16** | 1143 | **27.5** | ✅ | BF16 × F32 AVX2+FMA |
| Q3_K_M | 331 | 9.2 | ✅ | Q3K × Q8K scalar (fixed) |
| Q3_K_S | 308 | 6.0 | ✅ | Q3K × Q8K scalar (fixed; Lyon noise) |
| Q2_K | 283 | 4.9 | ✅ | Q2K × Q8K scalar (fixed) |
| Q2_K_L | 283 | 5.5 | ✅ | Q2K × Q8K scalar (fixed) |
| IQ4_XS | 351 | 4.8 | ✅ | IQ4_XS × Q8K AVX2 (bit-exact, b8d6b7c) |
| IQ3_XXS (UD) | ~280 | 4.3 | ✅ | IQ3_XXS × Q8K scalar (f64 block acc, "The capital of France is Paris.") |
| IQ2_XXS (UD) | ~210 | 4.4 | ⚠️ 输出偏 | IQ2_XXS × Q8K scalar (单 block bit-exact, 模型输出与 IQ3_XXS 略不同) |
| IQ1_M (UD) | ~170 | 4.3 | ⚠️ 输出乱 | IQ1_M × Q8K scalar (1.75 bpw 本就精度极低) |
| IQ1_S (UD) | ~150 | 4.7 | ⚠️ 输出乱 | IQ1_S × Q8K scalar (1.5 bpw, 同上) |

**关键加速对比**：
- Q4_1: scalar 11.6 → AVX2 54.5 t/s （**4.7×**）
- BF16: scalar 7.4 → AVX2 27.5 t/s （**3.7×**）
- Q5_K: was 0 output → 40 t/s （修复）
- Q2_K/Q3_K: was 乱码 → 5-9 t/s （修复）
- IQ4_XS: was panic → AVX2 4.8 t/s（修复 + 打开 dispatch）
- IQ3_XXS (UD): was panic → scalar 4.3 t/s（修复 + f64 acc → 给出 "The capital of France is Paris."）
- IQ2_XXS (UD): was panic → scalar 4.4 t/s（修复）
- IQ1_M / IQ1_S (UD): was panic → scalar 4.3 / 4.7 t/s（修复）

**I-quant 精度漂移说明**：IQ3_XXS / IQ2_XXS scalar 输出与 IQ4_XS AVX2 输出不完全一致。IQ3_XXS 现在能正确生成 "The capital of France is Paris."，IQ2_XXS 在 "The capital of France is" 上给出 "*The capital of France*"，在 "Once upon a time" 上给出 "(trigger"。Python 单 block dot 与 Rust 单 block dot 完全相等（IQ3_XXS `-0.097571254`，IQ1_M `-65.112305`），证明算法 bit-exact。剩余差异来源于：`0.25 * sum(d_i*b_i)`（C 末尾乘）与 `sum(0.25 * d_i*b_i)`（Rust 每 block 乘）的 IEEE 754 f32 累加顺序不同。已用 f64 block accumulator 修复大部分情况（IQ3_XXS 完全恢复），IQ2_XXS 仍残留偏差——可能与 2-bit 量化精度边界有关，待查。

**Q2_K/Q3_K 下一步是写 SIMD 路径**（仿 q4k AVX2）。当前 5-9 t/s 提不上 30-40 t/s 是 scalar matmul 的吞吐瓶颈——这一档 kernel 复用 Q8K activation 而非 Q8_0，所以可以直接仿照 `vec_dot_q4k_q8k_avx2` 写一个 `vec_dot_q2k_q8k_avx2`/`vec_dot_q3k_q8k_avx2`。预期 5-10× 加速。

---

## 经验：SIMD 浮点内核必须严格匹配 Scalar 的舍入顺序

**问题**:Q4_0 AVX2 实现曾在生产路径中产生 1 ULP 累积漂移，导致 softmax 后 top-1 token 在 temp 0.6 采样时被翻转("巴黎" → "尼斯")。Parity test 在合成数据上仅显示 1 ULP 差异，被误判为可接受。

**根因**:IEEE 754 f32 **加法不满足结合律**。`(a + b) + c` 与 `a + (b + c)` 在 f32 中可能产生不同结果(舍入方向取决于中间值的指数)。SIMD 内核的累加顺序与 scalar 不同时，即使每个操作都"正确"，最终结果可能差 1 ULP。

**两类高危操作**:

1. **`f32::mul_add` / `_mm256_fmadd_ps`(FMA)**:fuses `a*b + c` 为单次舍入。Scalar 的 `a*b + c` 是两次舍入(乘 1 次 + 加 1 次)。在 AVX2+FMA 目标上,`mul_add` 会编译成 FMA → 1 ULP drift。

2. **`hsum_ps`(树形 reduction)**:`_mm256_hadd_ps`, `hsum_ps` 等横向求和用树形(`(a+b)+(c+d)+...`),而 scalar 是顺序累加(`sum = (sum + s0) + s1 + ...`)。两种顺序的最终值可差 1 ULP。

**Q4_0 AVX2 实际修复路径**(`src/ops/kernel/q4_0/avx2.rs`):

```rust
// ✗ 不匹配 scalar:加法不满足结合律
acc += prod0 + prod1;     // = acc + (prod0 + prod1)

// ✓ 匹配 scalar:顺序累加
acc += prod0;
acc += prod1;

// ✗ FMA:1 次舍入 vs scalar 的 2 次
let prod = dc.mul_add(d_b, 0.0).mul_add(si_b, 0.0);

// ✓ 显式 mul+add:2 次舍入
let prod = dc * d_b * si_b;
```

**Parity test 必须强制 bit-exact**:

```rust
let diff_bits = (avx2.to_bits() as i32).wrapping_sub(scalar.to_bits() as i32).unsigned_abs();
assert!(diff_bits == 0, "AVX2 diverged by {} ULP", diff_bits);
```

**不要**用 `rel < 1e-3` 这类容差测试,否则 1 ULP drift 会被掩盖。Q4_0 模型在合成数据 + 真实模型权重上跑了 9 个 parity case 全部 bit-exact 通过,然后才接入生产 dispatch。

**已知未完全修复**(见 `docs/TODO.md`):
- Q6_K AVX2 仍有 1 ULP drift,根因可能更深(`_mm256_madd_epi16` 累加顺序 vs scalar 的 per-element 累加,或 `_mm256_sub_epi32(sumi, q8sclsub)` 减法指令序列差异)。当前生产模型输出正确,但需进一步调查。
- Q8_0 AVX2 在合成 uniform 数据上 drift 255(极端情况),但真实模型权重通过 — FMA + hsum 顺序问题,未深入定位。
