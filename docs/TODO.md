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

- [ ] **ASR audio encoder 加速 (F16 conv2d 路径)** — 当前 ASR 跑 5 秒音频 ~14.5s，其中 audio_encode 占 12.1s（80%）。瓶颈在 `conv2d_stride2_padding1`（3×/chunk × 8 chunks = 24 次）和 `project_f16`，都调用 `dot_f16_f16_bytes` 但该函数在 x86_64 上是 scalar fallback（每元素 unpack 2 bytes + 2×`f16_to_f32` + FMA），无 SIMD。两条优化路径，独立可分别做：

  - [ ] **方案 A: `dot_f16_f16_bytes_avx2`** — 给 `dot_f16_f16_bytes` 加 AVX2+F16C+FMA 实现。参考 `dot_f16_f32_avx2`（已存在），用 `_mm256_cvtph_ps` 把 a/b 的 [u16] 转为 [f32x8]，用 `_mm256_fmadd_ps` 累加。**工作量 ~80 行**，风险低。**预期加速 3-5×**，audio_encode 12s → 2-3s，总 ASR 14.5s → 4-5s。**通用**：所有调用 `dot_f16_f16_bytes` 的地方都受益（不仅 conv2d，还有 project_f16）。
  - [ ] **方案 B: Fold conv2d into F16xF16 GEMM** — 重写 `conv2d_stride2_padding1`：用 im2col 把 patch 矩阵化（每个 output pixel 一行），一次 GEMM 算完所有 output pixels × output channels。绕过 50K-200K 次 scalar dot 调用。**工作量 ~250-300 行**，风险中高（要重新设计 outer loop、padding 边界、stride 跳步）。**预期加速 5-10×**，但仅针对 conv2d。**推荐路径：先 A 看效果，不够再做 B**。

- [ ] **讨论：MemoryArena 与 BlockAllocator 组合** - BlockAllocator 当前独立管理内存，可考虑改为组合 MemoryArena 的模式，便于统一管理和未来动态扩缩
- [ ] **讨论：GPU 后端架构设计** - 当前 ash Vulkan 实现较为简单。GPU 生态碎片化严重：NVIDIA (CUDA/cuBLAS)、AMD (ROCm)、Intel (OpenCL/oneAPI)、ARM (Mali)、核显（Intel+AMD+ARM）、共享内存（Grace Hopper）等。可考虑：(1) 保留 ash Vulkan 后端 (2) 引入 wgpu 作为跨平台后端 (3) 通过 trait 抽象计算后端，灵活切换
- [ ] **讨论：SIMD 扩展路线** - 当前已有 AVX2+FMA、NEON。后续可考虑：Kleidi (Intel 新加速库)、AVX-512 (高端 CPU)、ARM SVE/NEON 增强等
- [ ] Row 切分支持（tensor parallelism across rows）
- [ ] Layer 切分支持（pipeline parallelism across layers）

## Low Priority
- [ ] 更多量化格式支持（Q4_K, Q5_K 等）
- [ ] 完善 GGUfRS 导出功能
- [ ] GGUF 导出支持
