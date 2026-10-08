# Z-Image Turbo 用法

Z-Image Turbo 是文生图模型，分三个独立 GGUF 组件：

- DiT：`--model`
- Qwen3 文本编码器：`--text-encoder`
- Flux VAE：`--vae`

GGUF `general.architecture = pig`，对应 `src/models/diffusion/pig.rs` 与
`src/models/diffusion/z_image/`。CLI 入口 `src/app/diffusion.rs::run_z_image_cli`。

> 共用前置：构建 `cargo build --release --features vulkan --bin rust-model-inference`。
> CPU 的 pinned Oracle 验证范围为 **512×512、txt2img**。
> Vulkan 为实验性后端，运行加 `--gpu`。`default = []` 不含 Vulkan，省略构建 feature 时仍走 CPU。
> img2img、Z-Image Base 仍为 `Unsupported`；设备验证见 [RADV 记录](../develop/VULKAN_RADV_VALIDATION_2026-10-07.md)。

## 1. 三组件

```bash
cargo run --release --bin rust-model-inference -- \
  --model models/z-image-gguf/z-image-turbo-q8_0.gguf \
  --text-encoder models/z-image-gguf/qwen3_4b_f32-q8_0.gguf \
  --vae models/z-image-gguf/pig_flux_vae_fp32-f16.gguf \
  --prompt "A red fox sleeping beneath a pine tree" \
  --steps 8 --resolution 512 --seed 42 --threads 1 --out fox.png
```

三个 GGUF 缺任一都会立即拒绝。加载 `pig` 架构时先过一道聚合检查
（`reject_incomplete_z_image_architecture`，`src/app/mod.rs:187-192`）：

```
Z-Image model requires --text-encoder, --vae, --prompt, and --out
```

过了这道之后，参数解析阶段还有逐项检查
（`z_image_cli_options`，`src/app/cli/options.rs:608-637`），分别报
`Z-Image requires --text-encoder` / `--vae` / `--out` / `--prompt`。

`--prompt` / `--out` 同样必填。

### 实测耗时（20 核 DGX Spark，512×512，8 步）

同机三条路径，prompt 为 "a red fox sleeping beneath a pine tree"，seed 42。

| 路径 | 步数 | DiT forwards | 总耗时 | 每次 forward |
|---|---|---|---|---|
| 本仓库 CPU | 8 | 7 | 466 s | 62.4 s |
| **本仓库 GPU（`--gpu`）** | 8 | 7 | **150 s** | **16.8 s** |
| PyTorch 2.11 + CUDA（同一权重） | 9 | 8 | 15.1 s | 1.89 s |

当前 `--steps N` 跑 **N 次** forward：8 步在 `RUST_GPU_DIAG=1` 下打出 8 行 sigma，
`dispatches` 每步 +300（300→2400）。表中"DiT forwards = N-1"是当时的旧语义，
最后一步现已真正执行 forward，算单步耗时时不要沿用。

⚠️ **这一张表的绝对值已经失效**，本机今天实测纯 CPU 就要 496.6 s（denoise
462.4 s），远高于表里的 466 s / 437 s；GPU 侧同理。差异不是编译 profile
（`release` 与 `release-fast` 只差 0.1 s），而是机器状态漂移——同一份未改动的
VAE 解码代码在重复运行里就给出 34.5 / 44.8 / 77.2 s。**请只使用下方"绝对值
不可信"一节里的同会话 A/B 差值**，并重新测量 PyTorch 之后再谈比值：表里的
1.89 s/forward 与 8.9× 同样是历史数字。

单步分解（GPU；attention 已上设备，见下表与下方 A/B）：

| 阶段 | 单步 | 备注 |
|---|---|---|
| FFN 主栈 | 3.86 s | Q8_0 tiled，105% 理论地板，别碰 |
| rms_norm + AdaLN + QKV | 2.09 s | 单 command buffer；其中约 1.4 s 去向未查明 |
| 输出投影 | 1.24 s | |
| FFN refiner | 1.42 s | F16 tiled，原生解码 + packed staging 后 2.54× |
| attention | 0.31 s | GPU 整链：tiled scores + softmax + value reduction |
| RoPE + 调制 | 0.11 s | 仍在 CPU |

> **线程数也非越多越好**：16 线程首步 269.3 s，反而慢于 8 线程的 140.5 s。

### ⚠️ 绝对值不可信，只有 A/B 差值可信

历史文档写 14.7 s/步。本机今天实测**纯净 HEAD（9a831ee）已经是 23.07 s/步**，
`release`（fat LTO）与 `release-fast` 相差 0.1 s，所以**不是编译 profile**。旁证
VAE 解码——只跑一次、与步数无关、代码未改——在多次相同运行里分别是
34.5 / 44.8 / 77.2 s，本身带 2.2× 噪声，**不能用于任何归因**。

同会话前后对照（8 步 / 512 / seed 42 / 20 线程 / release-fast）：

| | 纯净 HEAD | +F16 改造 | +attention 上 GPU |
|---|---|---|---|
| denoise | 184.5 s | 158.8 s | **109.3 s** |
| 单步 | 23.07 s | 19.85 s | **13.35 s** |
| refiner/步 | 3608 ms | 1421 ms | 1421 ms |
| attention/步 | ~3.4 s（CPU） | ~3.4 s（CPU） | **~0.31 s（GPU）** |
| 总计 | 231.5 s | 211.7 s | 145.4 s |

前两列输出逐字节一致。attention 那一列**不逐字节一致**，见下节。

#### 合并 FFN 到单个 command buffer：已尝试，未能完成

目标是把 `run_block_gpu` 里的 w1 → w3 → silu → w2 录进**同一个**
`TokenCommands`，一次 submit 一次 readback，替掉现在的 3 次
`project_scaled`（各自 `submit_and_wait`）+ host silu。

`DitGpuSession::record_ffn` 已写出来，逻辑本身在**分开提交**时经逐步
readback 验证是正确的（Z_DEBUG_FFN=1，512×512，rows=3840）：

```
after w1  : finite 327680/327680  min -1.73e1  max 1.37e1
after w3  : finite 327680/327680  min -3.91e1  max 2.69e1
after silu: finite 327680/327680  min -4.78e2  max 3.44e2
after w2  : finite 122880/122880  min -1.02e3  max 4.08e3
```

silu 的输出区间与 CPU 参考一致（按 w1/w3 的实测极值算 silu(gate)*up
得 [-531.8, 365.8]，设备给出 [-477.8, 343.8]，符号与量级吻合），所以
**silu shader 本身是对的**，`SILU_MUL_SHADER` + `Layout.gate/up` 这条路
可行。

**但合成一个 command buffer 后输出全 -inf**，而逐步 submit 的版本正确。
已排除的解释（都不是原因）：

- 缺 barrier。`compute_barrier` 是 SHADER_WRITE → SHADER_READ | SHADER_WRITE、
  COMPUTE → COMPUTE，语义正确；在 w1/w3/silu/w2 每步前后都加，仍然全 -inf。
- descriptor set 混叠。silu 绑 `self.arena_bindings`，投影绑各自的
  `bindings.descriptor_set`，两者来自同一个 pool 的不同 set。
- 区域重叠。`Layout` 用 bump allocator，`x/normed/out/qkv/gate/up/q8/
  q8_scales/q4_1_input_sums/q8k/q8k_scales` 依次 `take()`，尺寸算下来
  `rows_ffn == rows*FFN_WIDTH`，同为大小但偏移不同。
- 形状。W1/W3 n_in=HIDDEN n_out=FFN_WIDTH，W2 n_in=FFN_WIDTH n_out=HIDDEN，
  与 GGUF 里 `(3840,10240)/(10240,3840)/(3840,10240)` 一致；W1/W2/W3 都是
  Q8_0（GGML type 8），走 tiled Q8_0 + 32 元素量化。

剩下的差异只有"三个 dispatch 挤在一个 buffer 里 vs 各自 submit"。
`TokenCommands::begin` 复用**同一个** `command_buffer` 并先调
`recover_commands` + `reset_command_buffer`（`vulkan.rs:420`），而
`begin` 全程持有 `context.mutex`。分开提交时每次 `read_f32_into` 都会把
host 读钉在 pipeline 上，等于强制刷新；合批后所有 GPU 阶段只能靠
barrier 互相可见——而 barrier 已证明加对了。**未定位的可能是
descriptor set 与 pipeline 布局在跨 dispatch 复用时的状态**，需要
更细的 trace（例如逐 dispatch 的 `RUST_GPU_SUBMIT_TRACE` + dispatch 计数）
才能继续。

改动已回退。留在这里是为了让下一个人不必重走：silu 数值已验证正确，
卡的是合批机制本身。

> 工程向的完整快照与加速路线见 `docs/develop/ZIMAGE_STATUS.md`。

### 2026-10-06 在当前 HEAD（合入 #151/#152 后）复测的分解

`[gpu-block-profile]` 是代码里已有的分段计时（`GPU_PHASE_LABELS`），8 步
512×512 seed 42，20 线程，合计 107.2 s：

| 阶段 | 8 步 | 占比 |
|---|---|---|
| ffn: main stack (gpu) | 50.7 s | 47.3% |
| out proj (gpu) | 21.8 s | 20.4% |
| ffn: refiner | 11.5 s | 10.7% |
| **attention (host)** | 10.9 s | 10.1% |
| norm+adaln+qkv (gpu) | 10.6 s | 9.9% |
| **rope (host)** | 1.7 s | 1.6% |
| modulation (host) | 0.02 s | 0.0% |
| &nbsp;&nbsp;其中 w1 readback | 3.1 s | 2.9% |
| &nbsp;&nbsp;其中 **host silu** | **12.3 s** | **11.5%** |

**26% 的时间在 host 上**：attention 10.9 s + rope 1.7 s + w1 readback 3.1 s +
host silu 12.3 s。标签里的 `(host)` 不是笔误。

同步次数是硬约束：`RUST_GPU_SUBMIT_TRACE=1` 在 3 步上给出 3308 次提交，
即**每步 1103 次** `submit_and_wait`，而每步只有 320 次 dispatch——平均每
3.4 次同步才凑一次有意义的 GPU 工作。`queue_submit` 本身 0.009 ms，费用
全在 `wait_for_fences`。根源是 `run_block_gpu` 每 block 有 5 次 readback
（QKV、out proj、w1/w3/w2），34 block × 5 ≈ 170 次/步。

`run_block_gpu` 里 w1/w3 投影后在 host 做 silu 再传回给 w2，是纯粹的
CPU/GPU 边界错配：`SILU_MUL_SHADER` 已经是 pipeline 16（YuE2 AR 在用），
`Layout` 也已有 `gate`/`up` 区域，所以这段可以整段留在设备上。

⚠️ **但改的时候必须把 w1+w3+silu+w2 录进同一个 command buffer**。
`project_scaled` 自带 `submit_and_wait`（`dit_gpu.rs:434`），
`record_projection` 才是"录到在途 buffer、不回读"的那个变体。用
`session.begin()` 单独录 silu 会让这个 `TokenCommands` 在语句结束时被丢弃
（drop 不提交），w2 于是读到未激活的 gate——实测 PSNR 掉到 5.11 dB，整张图
报废。该改动已回退。

## 画质：attention 上 GPU 没有抬高偏差

拿同 seed 的三张图对比（512×512，8 步）：A = 纯 CPU 渲染，B = GPU 路径 +
CPU attention，C = GPU 路径 + GPU attention。

| 对比 | mean\|Δ\| | PSNR |
|---|---|---|
| A↔B（CPU vs 旧 GPU 路径） | 4.93 | 28.43 dB |
| A↔C（CPU vs 新 GPU 路径） | **5.01** | **28.96 dB** |
| B↔C（两种 attention 互比） | 3.45 | 31.42 dB |

**A↔C 没有比 A↔B 更差**，所以设备 attention 落在 GPU 路径本来就有偏差之内，
没有新增代价。整条 GPU 路径与纯 CPU 的 28–29 dB 差距早于本次改动（f16 舍入贯穿
Q8 主栈、F16 refiner 与 QKV），**GPU 路径从来就不是逐位精确的**，需要与 CPU 输出
对拍时用 `RUST_GPU_ATTENTION=0` 把它关掉。

差异集中在高频：把图模糊一下 PSNR 涨到 32.6 dB，降采样 /8 涨到 35.2 dB，
说明结构与构图没有变，差的是细纹理与像素级噪声——肉眼"一眼看过去一样、近看有
差异"就是这个 PSNR 区间的典型表现。

### 环境变量

| 变量 | 默认 | 作用 |
|---|---|---|
| `RUST_GPU_ATTENTION` | `1` | `0` 把 DiT attention 放回 CPU 逐行路径 |
| `RUST_GPU_F16_REF` | `1` | `0` 把 refiner 栈放回 CPU |
| `RUST_GPU_TILED` | `1` | `0` 关掉 Q8_0 tiled matmul |
| `RUST_GPU_VAE` | 关 | 仅 `1` 显式启用 F16 VAE 卷积 offload；需 `--gpu`，可能显著变慢 |
| `RUST_GPU_DIAG` | 关 | 打印每步 GPU 上的 block / dispatch 数 |

在重建绝对基线之前，不要用这些数字与 PyTorch 的 1.89 s/forward 算比值。

### 让 GPU 路径快起来的四件事

1. **Q8_0 grouped matmul 的 tiling。** 旧 kernel 每个 workgroup 只算**一个**
   输出元素：64 个 lane 切分 K 维再做树形归约，于是每字节权重只服务**一个**
   token，权重复用为 1。实测 W2 形状 757.6 ms / 110 GOP/s，而同一形状 PyTorch
   约 5900 GOP/s。`shaders/glsl/q8_matmul_tiled_dp4a.comp` 让 lane 持有输出列、
   一次权重加载复用于一个 token tile 的寄存器累加器。token tile 从 8 加到 32
   后（W2 形状 64.4 → 20.6 ms，1289 → 4035 GOP/s；64 会超出寄存器而回落）
   合计 **37×**，与旧 kernel 相对误差 9.5e-7。

2. **attention 在 host 上并行，并按 query 分块。** 它一度占单步 75.9%，
   128 GMAC 却要搬 514 GB——每个 K 向量 528 KiB，每个 query 重读一遍整套。
   按 ~46 GB/s 的 STREAM 带宽正好是那 10 s。分成 8 个 query 一块后每个 K
   向量只读一次，10.0 s → 3.4 s。分块 8 最优（16 → 16.6 s，32 → 16.5 s，
   再大 score 行撑爆 L1）。由于每个 (query, head) 只读 `qkv`、只写 `output`
   的一段连续区间，这只是重排循环而非近似，测试要求与单线程**逐位相同**。

3. **host 写 → device 读的 barrier。** arena 是 CPU 直接写、shader 直接读的
   映射内存，缺失 HOST→COMPUTE barrier 时能否看见写入只取决于速度：旧 kernel
   每次 dispatch 约 700 ms，写入早被"看见"；tiled kernel 约 64 ms，同种子两次
   渲染就会差 1.9–5.8/255，且只有 arena 大到装不进缓存的 512×512 才复现
   （256×256 逐位一致）。`VulkanContext::host_write_barrier` 在每次 submit 前
   补上它。

4. **F16 refiner 上 GPU。** 两层 refiner 栈在 GGUF 里是 F16，而绑定只接受
   Q8_0，所以 4 个 block 曾整个走 CPU 逐行路径，比 30 层主栈在 GPU上还贵。
   `shaders/glsl/f16_matmul_tiled.comp` 是 Q8_0 tiled kernel 的浮点版。

5. **F16 kernel 的两处改写，refiner 再快 2.54×。** 权重解码原本是手写位运算：
   每个值一次分支加一次 `exp2()`，而 Q8_0 的 dequant 只有一次乘法。权重本来
   就成对塞在一个 32-bit word 里，改用原生 `unpackHalf2x16` 一次取两个、零
   分支（顺带把 inf/nan 与 subnormal 也算对了）。第二处是共享内存：激活在入
   shared 之前已经 `round_f16_rte` 过，却仍以 f32 存放，占 32 KB，正好是每 SM
   预算的一半，只塞得下一个 workgroup；Q8_0 kernel 同一块 chunk 只占 16 KB。
   改成按 f16 成对打包后同样是 16 KB。refiner 3610 ms → 1423 ms/步，同 seed
   输出与旧 kernel 逐字节一致。**refiner 是解码受限而不是字节受限**，所以把
   权重换成更小的量化格式没有意义——这一点先测出来省掉了一次错误方向的重做。

> **`examples/dit_vk_bench.rs` 的 0.4×–1.36× 是误导性的**：它测的是 GEMV
> （`gpu_out` 只有一行长），权重复用同样为 1，于是读带宽看起来正常，却完全
> 没有测到真实形状。判断 kernel 速度必须用
> `zimage_tiled_matmul_beats_the_one_token_per_weight_kernel`，它轮流读取 8 个
> 不同权重矩阵以绕开 L2，单矩阵热缓存下会报出 10 万 GOP/s 的假数字。

### 还没搬上 GPU 的

attention 仍占单步 23%，在 CPU 上跑。它比 GEMM 难搬：DiT 没有 KV cache——每个
token 只看自己 block 里的每个 token 一次——而现成的 scores/values operator
按 decoder 的 `[layer][position][head][dim]` 缓存布局索引，需要给
`kv_write` 和 `attention_scores` 加源 stride，并且要确认 softmax 归约的轴。
另外它有和 GEMM 同样的 514 GB K 重读，必须一并做 token 分块才有意义。

> `RUST_GPU_DIAG=1` 会逐步打印 `blocks_on_gpu` 与 `projections`。这个计数器
> 抓到过一次静默故障：block 0 报错后 `mark_gpu_broken` 把设备锁死，而 block
> profile 仍报出看似合理的 14 s/步 GPU 时间——只有"没有 block 进入
> `run_block_gpu`"能说明问题。

## 2. 参数约束

来自 `src/app/cli/options.rs:635-637`（`src/app/cli.rs` 已拆分，旧路径失效）：

- `--steps` 必须正整数
- `--resolution` 必须正整数且 **divisible by 16**

## 3. 从 PyTorch 权重转换

上游 `gguf-org/z-image-gguf` 只提供 Q8_0 一档 DiT。仓库自带的转换器可以从
原始 safetensors 重新导出：

```bash
models/.venv/bin/python tools/converter/z_image/convert_z_image.py \
  models/Z-Image-Turbo --out-dir models/z-image-gguf-mine --outtype q8_0
```

参数：

| 选项 | 说明 |
|---|---|
| `--outtype` | `q8_0`（默认）/ `f16` / `f32` |
| `--components` | `all`（默认）/ `dit` / `vae` / `text` |
| `--overwrite` | 覆盖已有输出 |

| 档位 | DiT 体积 | 每步耗时（8 线程） | 与上游权重差异 |
|---|---|---|---|
| `q8_0` | 6.73 GB | 140.5 s | 3.75 / 255 |
| `f16` | 11.47 GB | 138.8 s | **3.44 / 255** |
| `f32` | 22.93 GB | — | — |

`q8_0` 模式下两个 refiner 栈保持 F16（每步只跑两次，量化收益不抵误差），
30 个主层按 `--outtype` 走。

> **F16 不更快。** 权重流量翻倍（10.6 GB vs 5.64 GB）正好抵消了省掉输入量化
> 的收益，两者落在噪声范围内。它的价值是精度更接近原始权重。

VAE 只导出 `decoder.*`（txt2img 是 latent → 像素，编码器用不上）。

## 4. 已知范围

| 范围 | 状态 |
|---|---|
| Z-Image Turbo（CPU、512×512、txt2img） | `Verified`（[tests/z_image_reference.rs](../../tests/z_image_reference.rs) 覆盖 pinned Oracle） |
| Z-Image Base | `Unsupported` |
| img2img | `Unsupported` |
| GPU 后端 | `Experimental`（DiT offload；F16 VAE 卷积需 `RUST_GPU_VAE=1`；[RADV 记录](../develop/VULKAN_RADV_VALIDATION_2026-10-07.md)） |

GPU 一列的判定依据与已知缺口：

- **已跑通**：NVIDIA GB10（`--features vulkan` 构建）上 8 步 512×512 端到端出图，
  132 s；`RUST_GPU_DIAG=1` 报 `blocks_on_gpu=34`（全部 34 层进入 `run_block_gpu`），
  每步 170 次 projection。分阶段耗时与本文上方 2026-10-06 的基线吻合
  （denoise 109.3 s vs 107.2 s，占比一致）。
- **画质未回归**：同 seed 256×256 2 步，GPU 与 CPU 输出 PSNR **41.9 dB**，
  mean\|Δ\| 1.46/255。
- **单测覆盖**：`cargo test --features vulkan --lib` 下 5 个 GPU 正确性测试通过
  （AdaLN 调制、融合 norm+modulate、batched QKV、W2 scale 抵消、attention 分块逐位一致）。
- **新增 RADV 记录**：512×512 八步出图 402.758633 s；128×128 两步 CPU/GPU
  为 359.780291 / 26.349318 s，PSNR 39.3589 dB。权重与执行范围见链接，和 GB10 记录分开看待。
- **验证边界**：`tests/z_image_reference.rs` 是纯 CPU 且 `#[ignore]`。
  GPU 成图来自实机运行；像素不逐位一致，512×512 CPU 成品对比和整条浮点管线 parity 尚未完成。
- **已知缺口**：attention 与 RoPE 仍在 host；FFN 的 w1/w3 激活要经 host silu
  （合批到单 command buffer 的尝试已回退，见第 1 节）。

## 5. 与 Oracle 的对齐

Pinned Oracle：[leejet/stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)
@ `97d2990807fe6d558e395f8764198d7c7e7b411c`

构建 / 测试入口：

- `tools/oracle/z_image/build_stable_diffusion_oracle.sh`
- `tools/oracle/z_image/stable-diffusion-z-image-trace.patch`
- `tests/z_image_reference.rs`

> 仓库口头习惯说的 `Dif.cpp` 实际指这个 stable-diffusion.cpp fork，
> `docs/REFERENCE_IMPLEMENTATIONS.md` 顶端有说明。

## 6. CLI 选项速查

| 选项 | 用途 | 必填 | 默认 |
|---|---|---|---|
| `--model` | DiT GGUF | ✓ | — |
| `--text-encoder` | Qwen3 文本编码器 GGUF | ✓ | — |
| `--vae` | Flux VAE GGUF | ✓ | — |
| `--prompt` | 文生图 prompt | ✓ | — |
| `--out` | 输出 PNG 路径 | ✓ | — |
| `--steps` | Euler 步数（NFE） | — | 8 |
| `--resolution` | 输出分辨率（divisible by 16） | — | 512 |
| `--seed` | RNG 种子 | — | 0 |
| `--threads` | ComputePool 线程数 | — | 自动 |
| `--gpu` | DiT 投影走 Vulkan 后端，VAE 默认 CPU | — | 关 |

只有前五项必填（聚合检查在 `src/app/mod.rs:187-192`，逐项检查在
`src/app/cli/options.rs:608-632`）；`--steps` / `--resolution` 取 `unwrap_or`
默认值（`:633-634`），旧版本文档把这两个标成必填是错的。

`--gpu` 刻意不在互斥表里（`options.rs:600-604`）：DiT 投影经
`matmul_q8_0_quantized_parallel_rows` 交给 Vulkan backend，不支持的形状逐个回退，
所以这个开关在这里是有意义的。VAE 默认使用 CPU：F16 卷积 offload 每 64 个像素
就提交并等待一次，512×512 的单层卷积会产生 4096 次同步，可能抵消 DiT 的加速。
仅在显式设置 `RUST_GPU_VAE=1` 时，F16 VAE 的 1×1 / 3×3 卷积才进入 Vulkan；
VAE attention、norms 与 BF16/F32 分支保持 CPU 执行。

## 7. 相关源码索引

- `src/app/diffusion.rs` — `run_z_image_cli` 主入口
- `src/app/mod.rs:187-192` — `reject_incomplete_z_image_architecture`（pig 三组件聚合检查）
- `src/app/cli/options.rs:608-637` — 逐项必填 + `--steps`/`--resolution` 校验
- `src/ops/float.rs:31-43` — `gpu_matmul_active()`；无 vulkan feature 时恒为 false
- `src/main.rs:419-424` — `ops::enable_gpu()`（须在 Z-Image 分支内提前调用）
- `src/models/diffusion/pig.rs` — DiT 模型
- `src/models/diffusion/z_image/dit.rs` — DiT forward（`run_block_gpu` 在 `:2312-2688`）
- `src/models/diffusion/z_image/dit_gpu.rs` — GPU DiT 会话（`DitGpuSession`）
- `src/models/diffusion/z_image/text.rs` — Qwen3 文本编码器
- `src/models/diffusion/z_image/vae.rs` — Flux VAE
- `tests/z_image_reference.rs` — pinned Oracle 对齐（纯 CPU，`#[ignore]`）
- `docs/REFERENCE_IMPLEMENTATIONS.md` — Oracle pin 与构建脚本
- `tools/converter/z_image/convert_z_image.py` — safetensors → 三组件 GGUF
- `tools/converter/z_image/probe_layout.py` — 只读 header，打印张量名/维度分布

> `src/models/diffusion/z_image/dit_gpu_block.rs` 是被取代的旧 GPU 实现，**未在
> `mod.rs` 里声明**，无任何引用。改动前先确认自己不是在读死代码。
