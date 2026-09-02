# TODO — RustModelInference Feature Roadmap

## High Priority

- [ ] Prompt 处理速度优化（当前远低于 llama.cpp）
- [ ] **Q6_K embedding_lookup 调试** - 当前实现数值正确但模型挂起
- [x] **Vulkan GPU matmul 后端可用（2026-08）** — 权重常驻 + 持久 IO 缓冲 + 单 dispatch 全行覆盖 + 看门狗。正确性（GPU vs CPU rel ≤ 3e-7）与稳定性（不再挂死）已验证，详见 `docs/VULKAN.md`。剩余为 dispatch 开销优化（合并 dispatch / 批量提交 / dp4a），见 VULKAN.md 性能分析。

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
- [ ] **Q2_K / Q3_K SIMD 加速** — 当前 scalar 5-9 t/s。仿 `vec_dot_q4k_q8k_avx2`（q4k/avx2.rs）写 `vec_dot_q2k_q8k_avx2` / `vec_dot_q3k_q8k_avx2`。复用 Q8K activation + AVX2 `_mm256_madd_epi16`。预期 5-10× 加速，目标 30-50 t/s。详见 `docs/OPTIMIZATION.md` § "Quant Kernel 补全"。
- [ ] **IQ4_XS / IQ2_XS / IQ3_XS kernel 实现** — GGMLType 已注册（commit `402bc3d`）但 kernel panic with TODO。IQ4_NL scalar 已实现（kvalues_iq4nl LUT）。IQ2/3 需要查 llama.cpp 参考实现，I-quant 网格 LUT + bit-packed scales 复杂。qwen3-0.6b 的 IQ4_NL/Q4_XS 文件实际权重是 IQ2_XS/IQ3_XS，所以实现 IQ2/3 后这两个 model 就能加载。

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
- [ ] **Qwen3.5：借用权重与 FFN gate/up 输入量化复用的取舍** — 当前 `Qwen35Model::from_source` 将量化权重复制为 `QTensorOwned`，并通过 `fuse_vstack` 生成 `ffn_gate_up`。这使加载期多一次整模型复制；同时原 gate/up 与拼接后的 fused weight 都常驻，FFN 相关权重会额外占用约一份 `gate + up` 大小的内存。
  - **候选基线**：令 Qwen3.5 持有借用 GGUF/mmap 数据的权重；对同一 FFN 输入仅做一次 activation quantization，然后分别执行 `gate * q(x)` 与 `up * q(x)`。Q8_0 复用 `q8_buf + scale_buf`，K-quant 复用 `q8k_buf`。
  - **预期差异**：该方案消除权重复制和重复 activation quantization，但仍有两次权重读取、两套 dot-product 及两次 matmul 调用。`fuse_vstack` 同样只量化一次，潜在收益在于一次调用/线程池调度、更大的行分块及连续布局；不减少总乘加或权重读取。
  - **中期架构方向**：将“权重格式”与“存储所有权”分离，采用模型拥有 source 的混合存储，而非让 `ByteStorage<'a>` 生命周期传播到 `Qwen35Model` 与服务 API。建议形态为 `ByteStorage::{Mmap { backing: Arc<ModelBacking>, offset, len }, Owned(Vec<u8>)}`：普通 GGUF 权重由 `Mmap` 零拷贝读取，只有 `fuse_vstack`、转置、预打包或 GPU 上传等真实变换才创建 `Owned` 数据。这样模型可安全进入缓存、线程与异步任务（`'static`），同时避免为少量可变换权重复制整个模型。
  - **推进条件与风险控制**：这是中期重构，不阻塞局部 FFN 优化。先测量现有 Qwen3.5 的加载时间与 RSS，确认权重复制是实际瓶颈；随后仅迁移 Qwen3.5，完成 logits/token parity、加载时间、峰值/常驻 RSS、prefill/decode 吞吐验证后，再扩展到其他模型。统一设计时 Q8_0 必须显式保存 `n_cols/n_rows`，不能再由总字节数反推 shape。
  - **实施顺序**：(1) 拆分 prepare-activation 与 prepared-matmul API，并使 FFN gate/up 复用 prepared input；(2) 基准比较 borrowed-two-matmul 与 owned-vstack 的加载时间、RSS、decode tok/s、prefill tok/s；(3) 仅当两次调用的调度开销可测量地显著时，再考虑不复制权重的 `matmul_pair_prepared`。
  - **验收**：两条路径 logits/token parity；报告模型加载时间、峰值/常驻 RSS、单 token decode 与 45-token prefill 吞吐。不要仅凭“融合 matmul”假设保留额外权重副本。
- [x] **Q6_K AVX2 精度 drift 调查** — `src/ops/quant/mod.rs:2066` 的 `vec_dot_q6k_q8k_avx2` 在 4-block 合成测试上有 1-2 ULP drift。commit `acb0a2b` 调查确认根因与 Q4_0/Q4_K 同源:**scalar `sumf += sums[l]` 是线性累加,AVX2 `hsum_ps` 是树形 reduction;f32 加法不满足结合律**。尝试过多种缓解(FMA→mul+add 拆解、scalar `f32::mul_add`),均无改善——属于 IEEE 754 不可避免现象。生产验证:Q6_K / Q4_K_M / Q8_0 均输出 "The capital of France is **Paris**"(scalar 与 AVX2 一致);drift 不翻转 argmax。详见 `docs/OPTIMIZATION.md` "经验: SIMD 浮点内核必须严格匹配 Scalar 的舍入顺序"。
- [x] **Q8_0 AVX2 "diff=255" 调查** — `src/ops/kernel/q8_0/avx2.rs` 的 `q8_0_avx2_matches_scalar_uniform` 测试失败(`max_diff=255`)。commit `acb0a2b` 调查确认这是**测试 bug**(scalar 函数调用时 `(n_in, n_out, 0)` 把 `n_out` 错位传到 `row_start`,导致 scalar 没跑任何行返回 0)。修参数顺序 + 改测试数据为 8 行后:`max_diff=0.000366 rel=1.6e-7`,AVX2 算法 bit-exact 正确(实际只含正常的 f32 hsum-tree 1-2 ULP drift)。
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

## MiniCPM5-1B 输出对齐 Llama.cpp (2026-08) — ✅ 已解决

**根因：RoPE 风格用错。** `llama` GGUF 架构（含 MiniCPM5）使用 GGML `ROPE_TYPE_NORM`（**interleaved 相邻对**旋转），HF `rotate_half` 权重由 llama.cpp 转换器 permute 成该布局；而 Rust trunk 之前调用了 `rope_neox`（halves 风格）。

**为什么之前一直没找到**：历史调试全部对比 "L23 step 0"，而 `pos=0` 时两种 RoPE 都是恒等变换——step 0 的逐层 dump 永远一致（残余 0.07 只是量化噪声），差异只在 `pos>0` 出现，且随位置增长（step 16 时 logits 最大差 17+，top-10 排名完全不同）。

**定位过程**（工具已沉淀到 `tools/parity/`）：
1. `minicpm5_reference.py`（models/.venv，gguf+numpy f64 逐层真值）复算整个前向，与 Rust `RUST_LLAMA_DEBUG_OUTFILE` 逐层 trace 对比——误差随层数平滑放大，无单层跳变，指向位置编码。
2. `dump_tokens_oracle.cpp`（按显式 token id 驱动 llama.cpp 逐步 dump logits）：llama.cpp 与 halves-rope numpy 偏差随位置增长到 12+，与 interleaved-rope numpy 贴合（≤1.1，纯量化噪声）→ 确认 RoPE 风格假设。

**修复**：
- `src/ops/rope.rs` 新增 `rope_norm`（interleaved 相邻对旋转，GGML ROPE_TYPE_NORM 风格），带 pinned-bits 单元测试。
- `src/models/llama/trunk/forward.rs` Q/K RoPE 改用 `rope_norm`。

**验证结果**（MiniCPM5-1B-Q8_0，17 token prompt）：
- Rust 逐层激活 vs f64 numpy 真值：相对差从 5–12% 降到 1–3%，逐层 argmax 全部一致。
- Rust vs llama.cpp oracle：8 步贪心生成序列完全一致（122895, 33, 285, 5390, 34609, 559, 316, 2925），top-10 logits 差 ≤ ~1.0（两个 Q8_0 引擎相对 f64 真值各自的量化噪声包络即为 0.5–1.1）。
- 生成质量：think 段中文推理连贯，正确自述"我是 MiniCPM 系列模型，由面壁智能（ModelBest）和 OpenBMB 开发"。

历史排查记录（当时已排除的嫌疑，结论仍有效）：silu 近似 / Q8_0 量化 / rms_norm / matmul 均非主因；step-0 的 0.07 偏差即量化噪声底，无需再追。
## LFM2-8B-A1B (lfm2moe) MoE 支持 (2026-08) — ✅ 已完成

新增 `src/models/lfm2moe/`（对齐 `lfm2` trunk 结构），实现 `lfm2moe` 架构的 Mixture-of-Experts FFN：

- **模型结构**：24 blocks；前 `leading_dense_block_count=2` 个为 dense SwiGLU FFN（n_ff=7168，shortconv 注意力），其余 22 个为 MoE（32 选 4，expert_ffn=1792）；注意力在 shortconv 与 GQA（32 q头/8 kv头，head_dim 64）间交替。GGUF 权重 Q8_0，3-D expert 张量 `[n_in, n_ff_exp, n_expert]` 按 expert 切片为连续 2-D 权重。
- **MoE 数学**（对齐 llama.cpp `build_moe_ffn(gating_op=sigmoid, norm_w=true)`）：`logits = router@x`（F32）→ `probs = sigmoid(logits)` → `select = probs + exp_probs_b`（bias 仅影响 top-k 选择）→ top-4 → `weights = probs[top4] / clamp(sum, 6.1e-5, ∞)`（用无偏 probs 归一化）→ 加权求和 `Σ w_e · down_e(silu(gate_e@x) · up_e@x)`。

### 排查过程中发现并修复的既有 bug

1. **silu 与 matmul 的行分区竞争（全 trunk 模式性 bug）**：闭包内 `rows_per = n_ff / nth`（地板除）与 Q8_0 kernel 的 `(n_out + nth - 1) / nth`（天花板除）分区边界错位——当 `n_ff % nth != 0` 时（本机 nproc=18、n_ff=7168 触发），silu 会读到 matmul 尚未写入/将被覆盖的行，FFN 输出出现 60 量级错误。已统一为 ceil 分区：`llama`、`lfm2`、`lfm25`、`lfm2moe` 四处（`n_ff % nth == 0` 的旧场景行为不变）。
2. **shortconv conv 状态语义**：解码期状态更新原为"全行复制 b×x"，正确语义是滑动窗口（`llama.cpp`: new_conv = (state ‖ bx) 的最后 d_conv 列）；prefill 重建时零填充应在窗口头部（右对齐）而非尾部。`lfm2moe`、`lfm2`、`lfm25` 三处均已修复。
3. **lfm2/lfm25 conv 权重 2-D 形状**：新转换的 GGUF 把 `shortconv.conv.weight` 存为 2-D `[l_cache, n_embd]`，旧加载器只接受 1-D 导致 LFM2.5-1.2B 直接报错（基线即坏，非本次引入）。已兼容两种形状。

### 验证结果（What is the capital of France?，13 token prompt）

- 逐层对比（llama.cpp eval-callback dump vs Rust，step 0）：除 L5 大激活通道（|v|≈62，模型固有 massive activation）上 4.5e-4 相对量化噪声外全部 ≤ 5e-3。
- 逐步 logits：13 步中前 7 步（含 6 步贪心生成）top-1 完全一致，logit 差 0.5~2.8；step 19 在近平局（20.78 vs 20.05）处分叉——MoE 路由近平局对量化噪声敏感，属跨实现固有现象（llama.cpp 自身跨线程 bit-exact，故差异来自算术细节而非不稳定）。
- 生成质量：`"The capital of France is Paris. It is a major global city known for its art, fashion, gastronomy, and landmark attractions like the Eiffel Tower and Louvre Museum."`
- 性能：~4.5 tok/s（MoE 每 token 需读 ~22 层 × 4 expert × 3 矩阵 ≈ 1GB 权重，内存带宽受限；逐 expert 5 次 pool.dispatch 的调度开销可后续优化）。

工具沉淀（`tools/parity/`）：`lfm2moe_reference.py`（numpy f64 逐层真值，惰性反量化防 OOM）、`lfm2moe_layer_cmp.py`（layer-oracle 逐层对比）、`dump_tokens_oracle.cpp` / `dump_layer_oracle.cpp`（按显式 token id 驱动 llama.cpp，后者经 cb_eval dump 全部单 token 中间张量）。


## LFM2.5-1.2B（dense）对齐 llama.cpp (2026-08) — ✅ 已完成

修复 conv 2-D 加载 + silu 分区竞争 + conv 状态滑动窗口后验证：

- LFM2.5-1.2B-Instruct-Q8_0（arch=`lfm2` + basename 含 2.5 → `lfm25` trunk），13 token prompt，`What is the capital of France?`。
- Rust vs llama.cpp oracle：**8/8 步贪心 top-1 完全一致**（1098, 5706, 803, 4481, 856, 5242, 523, EOS=7），EOS 时机一致。
- logit 数值差 0.1~4.8；step 13（首个 decode 步）top-1 值差 11.9（排名不受影响），成因未深挖——如需 bit 级对齐可用 `tools/parity/` 的 layer oracle 逐层排查。
- 架构映射备注：`LFM2-8B-A1B-GGUF` → arch `lfm2moe` → `models::lfm2moe`；`LFM2.5-1.2B-Instruct-GGUF` → arch `lfm2` + basename 2.5 → `models::lfm25`；`models::lfm2`（dense LFM2 v2）当前模型库中没有对应 GGUF，仅为该架构保留的分发路径。

## Ornith-1.5-9B (qwen35 架构, 9B hybrid) 适配 (2026-08) — ✅ 已完成

`models/Ornith-1.5-9B-GGUF`（arch=`qwen35`，33 blocks = 32 主层 + 1 MTP nextn 层；SSM 线性注意力
`full_attention_interval=4`，GQA 16/4 头，head_dim 256 + partial mrope 64/[11,11,10,0]，freq 1e7，
vocab 248320）。现有 qwen35 trunk 逻辑全部适用，适配点有三：

1. **KV cache 按请求预算分配（原为 OOM 根因）**：`app/text.rs` 与 `bin/server.rs` 的 qwen35 路径原来按
   `config.n_ctx`（262144）分配 F32 KV cache——9B 模型 = 33×262144×1024×4B = 35.4GB 直接分配失败。
   改为 `(prompt_len + max_tokens).min(n_ctx)`。
2. **nextn/MTP 层排除**：新增 `config.n_nextn`（读 `qwen35.nextn_predict_layers`，默认 0）与
   `n_layer_impl()`；权重加载与前向只遍历主栈层，MTP 块不参与推理（对齐 llama.cpp
   `n_layer_impl = n_layer_all - n_layer_nextn`，MTP 层 dense attention-only 非 recurrent）。
   recurrent 掩码同步修正：`i < n_layer_impl && (i+1) % interval != 0`。
3. 上述封顶同时惠及 Qwen3.5-2B（原每次运行 KV cache 固定分配 12.9GB）。

验证：llama.cpp token-id oracle 贪心序列 **8/8 步一致**（760→6511→314→9338→369→11751→13→10838），
top-1 logit 差 0.3~0.8；生成 `"The capital of France is Paris. 🇫🇷 …"` 与中文自我介绍均连贯；
Qwen3.5-2B / MiniCPM5 冒烟回归通过。

## Spark-X2.5-1.7B / 4B-GGUF 适配 — ✅ 已完成（功能）+ ⚠️ 性能待优化

`models/Spark-X2.5-{1.7B,4B}-GGUF`（arch=`spark2_5`，Xunfei Spark 2.5 讯飞星火）。模型
结构差异（vs qwen3）需要独立适配：

- **Fused QKV**：`attn_qkv [n_embd, n_embd_qkv]` 单一投影 → split 为 Q/K/V（Q=2048/2560,
  K=V=512/1024）。需在 `forward` 里 `split_at_mut` 而非三次独立 matmul。
- **Per-layer heterogeneous RoPE**：full-attn 层用 `n_rot=64, freq_base=5M`；SWA 层用
  `n_rot=256, freq_base=10K`（sliding_window=512）。需新增 `rope_neox_partial(x, pos,
  head_dim, n_rot, ...)`，只旋转每 head 前 `n_rot` 维；不能用 `rope_neox(x, pos, n_rot, ...)`
  （那会把 `[n_head × head_dim]` 当 `[n_head×n_rot × n_rot]` 处理）。
- **Sliding window mask**：SWA 层 attention 只看最后 `sliding_window=512` 个位置。pattern
  3-swa + 1-full-attn 循环。mask 在 softmax 后清零：`if is_swa && pos+1 > sliding_window { for t in 0..start { scores[t]=0.0; } }`。
- **Per-head gating**：`attn_gate [n_embd, n_head=8/16]` 矩阵乘 pre-attention norm → sigmoid →
  逐 head 乘 attention output（不是 post-output 的 MLP gate，而是 element-wise gate）。
- **GeGLU FFN**：`gelu(gate(x)) * up(x)` 后 down。SwiGLU/GeLU 的差异在 qwen3 里没有，需单独实现。
- **Tied embeddings**：GGUF 无 `output.weight` → 复用 `token_embd.weight`。`load_model`
  里 `output = if source.tensor_info("output.weight").is_some() { ... } else { None }`。
- **Tokenizer pre=`"spark2_5"`**：与 qwen2 正则不同（[punct][A-Za-z]+ 切分 `!hello` 为 `!` + `hello`）。
  新增 `PreTokenizer::Spark2_5` 变体；在 `scan_qwen_ranges` 里对标点+字母边界做切分。

适配后冒烟通过（"`What is the capital of France?` → `capital of France is Paris.`，`你好`
 → `！很高兴与你交流...`，`What is 2+2?` → `+ 2 = **4**`）。1.7B ≈ 1.1 tok/s, 4B ≈ 0.4 tok/s
（4 线程；生产路径未优化）。

### 待优化（性能，正确性已 OK）

- [ ] **Per-token prefill O(N²) 注意力** — `models::spark::trunk::forward::decode_step` 当前对
  每个 prompt token 单独跑一次，每次对前面所有 token 重算 attention。
  100 token prompt → 100×100 = 10K 次。生产路径需 batched prefill（一次 matmul 跑所有
  Q/K/V，按 layer 处理，然后一次性对完整 token 序列做 attention），参考 `llama.cpp`
  `llm_graph_context::build_attn(inp_attn, ..., Qcur, Kcur, Vcur, ...)` 的 batched 路径。

- [ ] **每次 `decode_step` 全量 `Vec::new`** — `hidden / normed / qkv / attn_out /
  attn_proj / gate_raw / gate_proj / up_proj / ffn_hidden / down_out` 每次分配。短期可加
  `ExecutionScratchpad`（参考 qwen3 trunk）；长期与 batched prefill 一起重构。

- [ ] **Generate 阶段没有 KV cache 预热** — 首 token 生成需要 prefill N 个 prompt token
  各自的 K/V 进 cache。当前是 N 次 `decode_step` 串行 → 单线程跑完所有层 + attention。可
  改为 prefilled `Session::new_with_kv_state(...)`，构造时一次性吃进 prompt。

- [ ] **FFN gate 计算输入是 pre-attention norm 还是 post-attention hidden？**
  当前用 `normed`（pre-attention rms_norm 输出），与 XFllama.cpp `attn_inp = cur`
  一致（cur 在 rms_norm 之后立刻）。但需要 llama.cpp 数值对照才能确定精度 — 当前输出
  与 llama.cpp oracle 没逐 token 验证（不像 Ornith-1.5-9B 那次有 8/8 一致）。

- [ ] **GeGLU 用 `0.5 * x * (1 + tanh(0.7978845 * (x + 0.044715 * x³)))` 近似** — 与
  llama.cpp 默认 `LLM_FFN_GELU` 的精确 GELU（`x * Φ(x)`）是否 bit-equivalent 需验证。

- [ ] **chat template 整段手写，未走 `tokenizer.chat_template` 的 Jinja 引擎** —
  当前从 GGUF 读出 template 字符串，用 `format!()` 在 Rust 里"模拟"渲染（手抄
  `{{- ... }}` strip-whitespace 行为）。如果未来模型加 tool_call、message 类型分支、
  reasoning_content 字段，手写版会失效。考虑引入 mini Jinja 引擎（参考 `llama.cpp`
  `common/jinja` 子模块的 PEG-native 方案；保持无依赖优先）。

- [ ] **精度对齐 XFllama.cpp oracle** — `references/XFllama.cpp/src/models/spark2_5.cpp`
  是科大讯飞自家实现的参考实现。当前未做 token-id 级别逐位对齐（不像 qwen3 有
  Q8_0/Q4_K 等的量化 oracle 测试，Spark2_5 是 BF16 没有量化分歧）。建议跑一遍
  同样的"prompt → token id 序列"对照，找差异层（最可能是 rope_partial 的
  theta_scale、GeLUU 近似、per-head gate 输入源）。

### 文件清单

- `src/models/spark/mod.rs`、`src/models/spark/trunk/{mod,config,weights,forward}.rs`
- `src/models/mod.rs`：加 `pub mod spark;`
- `src/app/text.rs`：加 `spark2_5` 分发；`enable_thinking` 从 `--thinking` 传入
- `src/core/tokenizer.rs`：加 `PreTokenizer::Spark2_5` 变体 + `spark2_5 → Spark2_5` 映射
- `src/ops/rope.rs`：加 `rope_neox_partial`
