# TODO — RustModelInference Feature Roadmap

## High Priority

- [ ] Prompt 处理速度优化（当前远低于 llama.cpp）
- [ ] **Q6_K embedding_lookup 调试** - 当前实现数值正确但模型挂起
- [ ] **WGPU Buffer Pool 优化** - 当前每次 matmul 调用都重新创建 buffer/bind_group/encoder，导致巨大开销。每个 token 生成需要几十次 matmul。

  **尝试记录（2026-08-19）：**
  - 初步尝试失败，回滚代码
  - 问题1：wgpu buffer 有 mapped/unmapped 状态，不能直接重新写入
  - 问题2：WgpuContext 被多线程同时访问，没有同步机制
  - 问题3：Buffer 大小在调用间会变化（n_out 不同），需要 resize 逻辑
  - 结论：需要深入理解 wgpu 内存模型和线程安全机制后再实现
  - 可能的正确方向：使用 Mutex 保护 WgpuContext，或者每个线程独立的 buffer pool
- [x] **统一 embedding_lookup 函数**
  - [x] 创建统一的 `embedding_lookup(weight, token_id, n_embd, embd_type, out)` 函数
  - [x] qwen3.rs、main.rs 已使用统一函数
  - [x] 保留 token embedding 的类型信息；模型各组件的量化类型应独立管理

## Medium Priority

- [x] **ASR audio encoder 加速 (F16 conv2d 路径)** — 当前 ASR 跑 5 秒音频 ~14.5s，其中 audio_encode 占 12.1s（80%）。瓶颈在 `conv2d_stride2_padding1`（3×/chunk × 8 chunks = 24 次）和 `project_f16`，都调用 `dot_f16_f16_bytes` 但该函数在 x86_64 上是 scalar fallback（每元素 unpack 2 bytes + 2×`f16_to_f32` + FMA），无 SIMD。两条优化路径，独立可分别做：

  - [x] **方案 A: `dot_f16_f16_bytes_avx2`** — 给 `dot_f16_f16_bytes` 加 AVX2+F16C+FMA 实现。参考 `dot_f16_f32_avx2`（已存在），用 `_mm256_cvtph_ps` 把 a/b 的 [u16] 转为 [f32x8]，用 `_mm256_fmadd_ps` 累加。**工作量 ~80 行**，风险低。**实际加速 3-5× audio_encode (12.1s → 1.96s)、6.8× encode_convolution、26.7× project_f16、总 ASR 14.5s → 4.9s (3.0×)**。
  - [ ] **方案 B: Fold conv2d into F16xF16 GEMM** — 重写 `conv2d_stride2_padding1`：用 im2col 把 patch 矩阵化（每个 output pixel 一行），一次 GEMM 算完所有 output pixels × output channels。绕过 50K-200K 次 scalar dot 调用。**工作量 ~250-300 行**，风险中高（要重新设计 outer loop、padding 边界、stride 跳步）。**预期加速 5-10×**，但仅针对 conv2d。**当前已通过 A 拿到 6.8× encode_convolution 加速；B 可进一步优化剩余的 1.74s**。

- [ ] **讨论：MemoryArena 与 BlockAllocator 组合** - BlockAllocator 当前独立管理内存，可考虑改为组合 MemoryArena 的模式，便于统一管理和未来动态扩缩
- [ ] **讨论：GPU 后端架构设计** - 当前 ash Vulkan 实现较为简单。GPU 生态碎片化严重：NVIDIA (CUDA/cuBLAS)、AMD (ROCm)、Intel (OpenCL/oneAPI)、ARM (Mali)、核显（Intel+AMD+ARM）、共享内存（Grace Hopper）等。可考虑：(1) 保留 ash Vulkan 后端 (2) 引入 wgpu 作为跨平台后端 (3) 通过 trait 抽象计算后端，灵活切换
- [ ] **讨论：SIMD 扩展路线** - 当前已有 AVX2+FMA、NEON。后续可考虑：Kleidi (Intel 新加速库)、AVX-512 (高端 CPU)、ARM SVE/NEON 增强等
- [ ] **讨论：两套线程调度统一** — 当前存在两套并行系统：(1) `ComputePool`（项目自研 spin-loop 线程池）用于 LLM prefill/decode；(2) `rayon global pool` 用于 audio conv、vision patch、qwen35 SSM。两者通过 `app::init_rayon_global_pool(n)` 同步线程数，保持与 `--threads N` 一致。**暂不统一**，因为它们从不并发运行（audio conv 结束后 LLM 才启动），且 LLM 是热路径不应轻易改动。**未来统一方向：迁移 audio/vision/qwen35 到 ComputePool**（而非 LLM 迁移到 rayon），因为 LLM 的字节级精确要求 + per-inference 池生命周期是最难验证的风险面。迁移后可在 `Cargo.toml` 中移除 rayon 依赖。详见 `src/core/thread_pool.rs` 顶部的代码注释。
- [ ] **Q8_0 与 Q8_K 量化路径按需量化（消除冗余计算，保留两份 buffer）** — `forward_prepared` 同时接收 Q8_0 (`input_q8` + `input_scales`) 和 Q8_K (`Option<&[BlockQ8K]>`) 两组输入，但**单个 layer 的 kernel 只会消费其中一份**：
  - 默认实现 (`src/ops/kernel/mod.rs:68-71`) 仅用 Q8_0，丢弃 `q8_k`。
  - K-quant 覆写 (`src/ops/kernel/q4_k.rs:71-72`, `q6_k.rs:56-57`) 把 Q8_0 参数标 `_input_q8` / `_input_scales` 丢弃，仅用 `q8_k`。
  - **不要把两份 buffer 合成一份**：模型可以**异构**——某些层用 Q8_0、某些层用 Q4_K / Q6_K——所以 `ExecutionScratchpad` 里 `q8_buf / scale_buf / q8k_buf` 都需要常驻。
  - 真正的浪费：**当前 `qwen3.rs` 在 attention / FFN 的 Q/K/V / gate / up / down 上对同一份输入同时调用 `quantize_q8_0_into` 和 `quantize_row_q8_k_into`**，而每个 layer 只会用一份。多出来的那次量化 pass（Q8_0 或 Q8_K）纯属白做。
  - **优化方向**：(1) 把每次前向的量化入口按**当前 layer 的权重格式** dispatch——参考 `src/models/qwen35.rs:216-232` 的 `match QWeight::{Q4K|Q5K|Q6K|Q8_0}`，它已经是正确模板；(2) 对 Q8_0 权重跳 `quantize_row_q8_k_into`、对 K-quant 权重跳 `quantize_q8_0_into`；(3) 不必改 `forward_prepared` 签名，也不动 `ExecutionScratchpad` 字段（两份 buffer 都保留以兼容异构模型）。
  - **预期收益**：每次前向省一次量化 pass（与权重格式对应的另一份白做的量化）。K-quant-only 模型省 `quantize_q8_0_into`；Q8_0-only 模型省 `quantize_row_q8_k_into`；异构模型按层省一半。Q8_0-only 模型还能省 `q8k_buf` 的写回带宽（≈ `n_embd/256 * 292B`，最大模型 4.6 KB / 推理上下文，写一次前向 ≈ 每 token 一次，可忽略）。
<<<<<<< HEAD
- [ ] **Q6_K AVX2 精度 drift 修复** — `src/ops/quant/mod.rs:963` 的 `vec_dot_q6k_q8k_avx2` 在合成 4-block 测试上仍有 1 ULP drift(漂移在最后 1 bit 位)。Q4_K AVX2 同类问题已通过显式累加顺序修复,Q6_K 修复未完成。可能漂移源:
  - `_mm256_sub_epi32(sumi, q8sclsub)` 减法指令的顺序 vs scalar 的 per-element `aux32[l] += scale * q8 * (weight - 32)` 累加顺序
  - `_mm256_madd_epi16(scale_l, p16l)` 累加 2 个 i16 → 1 个 i32 的顺序 vs scalar 的 2 个独立 mul+add 累加
  - 4 个 `madd` 链 (`p16_0`, `p16_1`, `p16_2`, `p16_3`) 的累加顺序 vs scalar 的 chunk-by-chunk 累加

  **当前状态**:模型 Q4_K_M 推理输出正确("巴黎"),drift 未大到翻转 argmax。但长期应消除以保证 temp 0.6 边缘情况稳定性。
  **修复方向**:参照 Q4_0 修复路径(`docs/OPTIMIZATION.md` "经验: SIMD 浮点内核必须严格匹配 Scalar 的舍入顺序")。可能需要把 sumi 累加顺序与 scalar 的 chunk 顺序对齐;`_mm256_sub_epi32` 后改用 `cvt + mul + add` 链而非 fma;`hsum_ps` 替换为 sequential extraction+sum。
  **验收**:parity test `vec_dot_q6k_q8k_avx2 == vec_dot_q6k_q8k_scalar` bit-exact(diff_bits == 0),含边界 cases(全零/全 max/全 min 输入,real-model Q4_K_M token_embd)。
  **相关文件**:`src/ops/quant/mod.rs:963`(kernel),`src/ops/quant/mod.rs:1076`(`avx2_parity::q6k_avx2_matches_scalar_multi_block` 测试,当前为 `rel < 1e-3` 容差)。
- [ ] **Q8_0 AVX2 精度 drift 调查(合成 uniform 数据上 diff=255,真实模型通过)** — `src/ops/kernel/q8_0/avx2.rs` 在极端 uniform 输入下 max_diff 达 255,但 `blk.0.attn_q.weight` 真实模型权重通过。问题尚未定位到具体指令。可能与 Q4_0 类似(FMA + hsum 顺序)但更深,因为 4-row tile 涉及跨行交叉累加。
  **修复方向**:添加 parity test 用合成数据 + 真实模型权重,对比 `matmul_q8_0_vs_q8_0_avx2` 与 `matmul_q8_0_quantized_scalar_range` 的中间 i32 值,定位漂移源头。
  **验收**:parity test bit-exact。
- [ ] Row 切分支持（tensor parallelism across rows）
- [ ] Layer 切分支持（pipeline parallelism across layers）

### K-quant multi-row tile（vec_dot_q4k_q8k_avx2 / vec_dot_q6k_q8k_avx2）

**目标**：Q4_K_M 从 ~76 t/s → 120-150 t/s，匹配 Q8_0 4-row tile（`src/ops/kernel/q8_0/avx2.rs:71-153`）模式。

**现状**：单行 tile，每次 matmul 调用 reload 一次 Q8_K。vocab=151936 一次 forward 共 reload 151936 × 4 = 607744 次 Q8_K，FMA 流水线未饱和。

**做法**：
- `vec_dot_q4k_q8k_avx2_4x(&[u8]) -> [f32; 4]`：4 行并行 decode nibbles，**共享 Q8_K 加载**（每 super-block 一次 `qs` + `bsums` load，broadcast 到 4 个 acc），4 个独立 acc `fmadd`。
- `vec_dot_q6k_q8k_avx2_4x(&[u8]) -> [f32; 4]`：同结构，复用现有 `_j in 0..2` unroll，4 行 weight 各 decode，Q8_K 共享。
- 调用点 `q4_k.rs:68` 和 `q6_k.rs:53` 的 `forward_prepared` 加 4-row 循环 + 余数单行 fallback。沿用 Q8_0 的 caller 模式（`q8_0/parallel.rs:102`）。
- Parity test：AVX2 4-row 结果 == scalar × 4，1 ULP 以内。参考 `src/ops/kernel/q4_0/avx2.rs::avx2_matches_scalar_on_real_q4_0_weights`。

**风险**：
- 寄存器压力（4 行 × {lo, hi} × Q8_K vector 至少 8+ 个）—— 用 `-C llvm-args=-x86-asm-syntax=intel` 检查。
- L1I 缓存（典型 32 KB）—— 保持 `_j in 0..2` unrolled，参考 Q8_0 结构。
- **Q6_K 已知有微小精度漂移**（temp 0.6 偶发采样分歧）—— 4-row tile 必须保持相同操作序列不引入新误差。

**验收**：
- [ ] `vec_dot_q4k_q8k_avx2_4x` parity vs scalar 1 ULP
- [ ] `vec_dot_q6k_q8k_avx2_4x` parity vs scalar 1 ULP
- [ ] `q4_k.rs` / `q6_k.rs` 调用 tiled path（`end - start >= 4`）
- [ ] Q4_K_M 0.6B benchmark ≥ 120 t/s tg
- [ ] Q4_K_M logits profile 占比下降（vocab matmul 受益最大）
- [ ] Q4_K_M 正确性（temp 0 + temp 0.6 均稳定）

**相关文件**：
- `src/ops/quant/mod.rs:700` — `vec_dot_q4k_q8k_avx2`（单行基线）
- `src/ops/quant/mod.rs:963` — `vec_dot_q6k_q8k_avx2`（单行基线）
- `src/ops/kernel/q4_k.rs:68` — Q4_K 调用方
- `src/ops/kernel/q6_k.rs:53` — Q6_K 调用方
- `src/ops/kernel/q8_0/avx2.rs:53` — `matmul_q8_0_vs_q8_0_avx2`（4-row tile 参考实现）

## Low Priority

- [ ] 更多量化格式支持（Q4_K, Q5_K 等）
- [ ] 完善 GGUfRS 导出功能
- [ ] GGUF 导出支持