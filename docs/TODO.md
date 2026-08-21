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
- [ ] Row 切分支持（tensor parallelism across rows）
- [ ] Layer 切分支持（pipeline parallelism across layers）

## Low Priority
- [ ] 更多量化格式支持（Q4_K, Q5_K 等）
- [ ] 完善 GGUfRS 导出功能
- [ ] GGUF 导出支持
