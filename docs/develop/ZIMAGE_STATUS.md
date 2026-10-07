# Z-Image 当前状态与加速路线（2026-10-06 实测）

本文件是当前 HEAD（`Yue_Xing`，基于合入 #148/#151/#152）的权威性能快照与后续加速路线。
**所有数字都是这台机器（NVIDIA GB10，20 核）同会话实测**，配置：8 步 / 512×512 /
seed 42 / 20 线程 / Q8_0 GGUF。绝对值会随机器状态漂移，**只有同会话 A/B 差值可信**
（见文末"方法论"）。

相关文档：
- `docs/usage/z-image.md` —— 用户向的用法、性能表、画质对比
- `docs/usage/yue2-gpu.md` / `docs/usage/yue2-vae-profiling.md` —— YuE2 侧的同类记录
- `docs/develop/ZIMAGE_OPTIMIZATION.md` —— **已过时**（64×64 baseline、SIMD/CPU 视角）

---

## 1. 速度快照：三条路径

| 路径 | denoise | VAE | 总计 | 每次 forward |
|---|---|---|---|---|
| **本仓库 GPU**（`--gpu`） | 110.3 s | 18.2 s | **130.3 s** | **13.9 s** |
| **本仓库 CPU** | 479.0 s | 19.2 s | **499.8 s** | **59.9 s** |
| **PyTorch 2.11 + CUDA**（同一权重） | 15.06 s 总计 | — | **15.06 s** | **1.88 s** |

- GPU 比 CPU 快 **3.8×**
- GPU 比 PyTorch **慢 8.7×**（这是要缩小的主差距）
- CPU 比 PyTorch 慢 33×

> PyTorch 用 9 步/8 forwards；本仓库 8 步跑 8 次 forward（sigma→0.003 的收尾也算
> 一次）。口径略有差异，但量级结论不变。1024×1024 时 PyTorch 为 6.68 s/forward。

**本轮（2026-10-06）速度提升为 0。** 详见第 4 节。

---

## 2. 每步 13.9 s 的完整归因

`[gpu-block-profile]`（代码内已有分段计时，`GPU_PHASE_LABELS`），8 步合计 107.2 s：

| 阶段 | 8 步 | 占比 | 在哪跑 |
|---|---|---|---|
| ffn: main stack | 50.7 s | 47.3% | GPU（Q8_0 tiled dp4a） |
| out proj | 21.8 s | 20.4% | GPU |
| ffn: refiner | 11.5 s | 10.7% | GPU（F16 tiled） |
| **attention** | **10.9 s** | **10.1%** | **host** |
| norm+adaln+qkv | 10.6 s | 9.9% | GPU |
| **rope** | **1.7 s** | **1.6%** | **host** |
| &nbsp;&nbsp;├ w1 readback | 3.1 s | 2.9% | host |
| &nbsp;&nbsp;└ **host silu** | **12.3 s** | **11.5%** | host |
| modulation | 0.02 s | 0.0% | host |

**26% 的时间在 host 上。** 标签里的 `(host)` 不是笔误：attention 与 FFN 里的 silu
都在 CPU 逐元素跑。旧的单步分解表只列了 9.03 s，遗漏的 4.87 s 主要就是这些 host 段。

### 同步次数是硬约束

`RUST_GPU_SUBMIT_TRACE=1`（3 步）：3308 次提交 = **每步 1103 次 `submit_and_wait`**，
而每步只有 320 次 dispatch —— 平均 3.4 次同步才凑一次有意义的 GPU 工作。
`queue_submit` 本身 0.009 ms，费用全在 `wait_for_fences`（设备真的在等）。

根源：`run_block_gpu` 每 block 有 **5 次 readback**（QKV、out proj、w1/w3/w2），
34 block × 5 ≈ 170 次/步，加上其他同步凑到 1103。

---

## 3. 本轮合入的两个 PR 的效果（同会话交错 A/B）

| | 合 PR 前 | 当前 | 变化 |
|---|---|---|---|
| GPU denoise | 110.6 / 109.1 s | 111.6 / 109.2 s | 无变化 |
| GPU VAE | 75.8 / 73.1 s | 18.7 / 19.1 s | **3.8–4.0×** |
| GPU 总计 | 188.4 / 184.2 s | 132.1 / 130.2 s | 1.42× |
| CPU VAE | 82.5 s | 19.7 s | 4.2× |

- **#151（VAE 空间注意力并行化）**：VAE 4× 提速，两组配对复现，输出 md5 与合入前逐位
  相同（纯并行、零精度代价）。
- **#152（Vulkan 中间值常驻显存，+773 行）**：**denoise 无可测收益**。denoise
  13.9 s/forward 接近此前测到的 Q8 投影流式上限（4088–5320 GOP/s），可能已贴顶。

---

## 4. 本轮（2026-10-06）做了什么 / 没做成

### 交付的 4 个 commit

| commit | 内容 | 速度影响 |
|---|---|---|
| `b2c9b70` | 每步归因：26% 在 host、每步 1103 次同步 | 0（诊断） |
| `724439b` | FFN 合批尝试记录 + 卡点 | 0（失败回退） |
| `9d99ff4` | barrier 修复：quantize↔matmul | 0（正确性） |
| `69db637` | GPU dispatch trace 工具 | 0（工具） |

### 找到一个真 bug（`9d99ff4`）

`record_weight_matmul_tiled_rows` 记录两个 dispatch —— QUANTIZE 把激活写进 arena，
Q8_MATMUL_GROUPED_TILED 再读它 —— **两者之间没有 barrier**。这个依赖是真实的。

它从未暴露，因为每个调用方都在这对之后紧跟一个 `submit_and_wait`，而提交边界是全
刷新。这让缺失的 barrier 对单次投影不可见，但**对任何把多个投影录进同一个 command
buffer 的代码是致命的**（Z-Image 的 FFN 就是三个）。缺它时该层产生全 -inf。

这是独立于 batching 的正确性修复：任何将来合批的代码否则会静默读到过期激活。
修复后 8 步渲染不变（110.3 s），输出与修复前逐位一致（mean|Δ| = 0）。

### FFN 合批：已尝试，未完成（`724439b`）

**目标**：把 `run_block_gpu` 的 w1 → w3 → silu → w2 录进**同一个**
`TokenCommands`，一次 submit 一次 readback，替掉 3 次 `project_scaled` + host silu。
预期收益：消掉 12.3 s host silu + 3.1 s readback + 每 block 3 次 submit→1 次，
13.9 s/步 → 约 11 s/步。

**已验证 silu 上 GPU 数值可行**（`DitGpuSession::record_ffn` 逐步 readback，Z_DEBUG_FFN=1，
512×512，rows=3840）：

```
after w1  : finite 327680/327680  min -1.73e1  max 1.37e1
after w3  : finite 327680/327680  min -3.91e1  max 2.69e1
after silu: finite 327680/327680  min -4.78e2  max 3.44e2
after w2  : finite 122880/122880  min -1.02e3  max 4.08e3
```

silu 区间 [-477.8, 343.8] 与 CPU 参考一致（按 w1/w3 实测极值算 silu(gate)*up 得
[-531.8, 365.8]），所以 `SILU_MUL_SHADER`（pipeline 15）+ `Layout.gate/up` 这条路可行。

**但合批后输出全 -inf，而分开提交正确。** 已排除的解释（都不是原因）：各步之间缺
barrier、quantize↔matmul 缺 barrier、descriptor set 混叠、arena 区域重叠、形状/格式、
q8 暂存脏数据。`record_ffn` 单独跑三种组合（w1+w2 / w1+w3+w2 / w1+w3+silu+w2）都
finite，值域与原路径一致 —— **差别只在于真实序列前面跑过 QKV/attention/out proj**。

改动已回退。`69db637` 的 trace 工具就是为此造的：应能一次定位失败序列里第几个
dispatch 出问题。**顺序上本该先写 trace 再改，这里搞反了。**

---

## 5. 加速路线（按性价比）

### 已有的工具

- `RUST_GPU_DISPATCH_TRACE=1` —— 每 pipeline 的 dispatch 计数
- `RUST_GPU_DISPATCH_TRACE=2` —— **有序 dispatch 日志**（pipeline / descriptor set /
  workgroup 网格，按 submit 分组）。诊断用，见 `69db637`
- `RUST_GPU_SUBMIT_TRACE=1` —— 提交三段计时（record / queue_submit / wait_for_fences）
- `RUST_GPU_DIAG=1` —— 每步 `blocks_on_gpu` / `projections` 计数
- `cargo run --release --features vulkan --example gpu_ceiling` —— 设备算力上限

### 优先级

1. **定位并完成 FFN 合批**（用 `69db637` 的 trace）
   预期 13.9 → 约 11 s/步。**明天第一件事。**
2. **out proj 查因**（20.4%，277 ms/call）
   远超此前测到的 Q8 投影流式上限（5320 GOP/s 对应约 55 ms），同样值得查。
3. **attention + RoPE 彻底上 GPU**（10.1% + 1.6%，全在 host）
   QKV 每 block 往返 177 MB，8 步合计约 80 GB PCIe。`qk_norm_rope.comp` shader 已存在
   （YuE2 AR 在用）。RoPE 上 GPU 可消掉一次同步 + 一次大往返。
4. **减少每 block 同步次数**（5 readback → 尽量 1）
   需要重构 `run_block_gpu` 数据流，工程量最大，但天花板最高（6.0 s/步里大部分是
   同步等待）。
5. **FFN 主栈（47.3%）暂不动** —— 文档称"105% 理论地板"，但理论算过（我曾漏乘一个
   维度导致误判）。做 1 之后重测该项 FLOPS 再判断。

### 关于追平 PyTorch 的现实预期

PyTorch 用 int8 tensor core（mma），而本机 `GL_KHR_cooperative_matrix` glslang 16
**不实现**（记录在 `c0c1db4`），手写 SPIR-V 是唯一路径但工程量巨大。
**现实短期目标是吃掉那 26% host 段和同步开销，把 13.9 s 推到 9–10 s 区间
（≈4.7–5.3× PyTorch），而不是追到 1.88 s。**

---

## 6. 方法论（本轮最大的教训）

这台机器的吞吐漂移大到**单次前后对比不可用**：

- 同一份未改动代码、同一 seed，两次运行的 ABC 波动 43%（24.1 s vs 34.4 s）、
  semantic 波动 50%
- 纯 CPU 的 VAE 解码在重复运行里给出 34.5 / 44.8 / 77.2 s（2.2× 噪声）
- 我曾用**跨运行的输出**做减法，算出两个"黑洞"（QKV norms 1.55 s、VAE 157 s），
  两次都是把不同运行的时间戳混着减 —— 都是假的，已在各自文档更正

**必须遵守的顺序**：

1. 先测（分段计时 + 流量 + FLOPS 核算），**再**动代码
2. 改完**先验证正确性**（图像 PSNR / 数值对拍），**再**测速度。
   本轮两次"改完直接测速、发现 -inf 才发现错了"，其中一次其实前 3 步都对
3. 单次前后对比一律不可信；用**同会话交错 A/B**（after/before/after/before）
4. 排除法猜测有上限；造工具（trace）比猜第 7 轮快

---

## 附：与本轮相关但属于其它模型的结论

- **YuE2 GPU**：AR session 已实现（`362d5fa`）但比 CPU 慢 20×，根因是 k-quant
  shader 只用 64 个 workgroup lane 中的 1 个（`a84635a` / `docs/usage/yue2-gpu.md`）
- **YuE2 VAE**：155 s 里的 5 条优化路径已全部否证（`bc5eae4` /
  `docs/usage/yue2-vae-profiling.md`）
- **Edge0**：35B MoE + 30/40 层 Mamba/SSM + 620 个 LoRA 张量；稠密:稀疏算力
  ≈ 6:1，最大单项是 LoRA 而非路由专家。GPU forward 未做
