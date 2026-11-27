# YuE2 用法

本仓库对 `m-a-p/YuE2-3B` + `m-a-p/YuE2-Vae`（ModelScope 上的规范 repo id）
的端到端命令行示例。

> 通用前置：`cargo build --profile release-fast --bin rust-model-inference`。
> 推理路径在 `src/models/yue2/`，AR（自回归）与 NAR（非自回归扩散）双流共用一套
> 28 层权重。

实验性 Vulkan 投影可用 `cargo build --profile release-fast --features vulkan --bin rust-model-inference`
构建，并在现有生成命令上加 `--gpu`。AR/NAR 的 BF16 投影仍在 bias 后舍入为 BF16，
attention、音频 VAE 与 NAR F16 的 Q8 activation 路径留在 CPU。
RADV 固定样例的 AR/NAR 和完整波形已与 CPU 逐位对齐；其它设备和成品质量仍需验证，
详见 [Vulkan 支持范围](../develop/VULKAN.md)。

## 1. 权重来源

| 用途 | ModelScope | 说明 |
| --- | --- | --- |
| 主模型 | `m-a-p/YuE2-3B` | `YuE2ForCausalLM`，28 层，hidden 2048，vocab 184704 |
| VAE | `m-a-p/YuE2-Vae` | `YuE2VAE`，48 kHz 立体声，`downsampling_ratio=1920` |

**不要用 `m-a-p/YuE-s1-7B-anneal-en-cot`。** 那是 7B 的 Llama 架构检查点，与本仓库
`YuE2ForCausalLM` 的 AR+NAR 双流实现不兼容，metadata 校验会直接失败。

## 2. 转换

VAE 固定为 F32，只有主模型可以量化：

```bash
./.venv/bin/python -m tools.converter.yue2.convert_yue2 \
  --model-dir models/YuE2-3B \
  --vae-dir  models/YuE2-Vae \
  --main-out models/YuE2-gguf/yue2-q8_0.gguf \
  --vae-out  models/YuE2-gguf/yue2_vae.gguf \
  --quant q8_0
```

`--quant` 取值：

| 值 | 结果 | 备注 |
| --- | --- | --- |
| `bf16` | 7.26 GB，628 个 BF16 张量 | 默认，零拷贝直传，不做任何算术 |
| `f32` | 12.91 GB | 便于排查量化误差 |
| `q8_0` | 4.61 GB，394 Q8_0 + 234 BF16，约 25 s | ⚠ **不要用于完整生成**，见下 |
| `q4_0` | 3.20 GB，约 50 s | ⚠ 同上，误差更大 |
| `ar_q8_0` | 5.94 GB（196 AR + 198 BF16），约 100 s | **AR 专用量化**，见 §2.2 |
| `q4_k_m` | ~3.2 GB | 类型链路已通，但导出需数小时 |
| `q6_k` | 3.93 GB | 同上 |

### 2.1 ⚠ 量化会毁掉 NAR，完整生成必须用 BF16

Q8_0 的量化质量本身是正常的：逐张量反量化后 SNR 约 **45 dB**（相对误差 0.54%，
每 32 元素一个 scale 的 8-bit 量化的应有水平），转换器没有 bug。

问题在于 **NAR 声学流对这点噪声极度敏感**。NAR 是 flow-matching 扩散求解，
每个 step 对全部 latent 位置做两次全模型前向，误差会随迭代步数放大。
实测（Q8_0，seed 12300，同一歌词）：

| NAR steps | 结果 |
| --- | --- |
| 1 | 可听 |
| 4 | **一团浆糊** |
| 32 | **一团浆糊** |

换成 BF16 权重后同样的 4 步完全正常。**这不是注意力实现的问题** —— 用
`RMI_YUE2_LEGACY_ATTN=1` 恢复原始标量注意力，Q8_0 一样是浆糊。

结论：`q8_0` / `q4_0` 只适合测速和 AR 阶段单点验证，**要出成品音频必须用默认的
`bf16`**。这也是为什么 `--quant` 默认值保持 `bf16`。

只有 338 个 transformer 投影 + 2 个 `time_embedder` 矩阵参与量化。1-D norm、
两个 184704 宽的词表矩阵（`embed_tokens` / `lm_head`）以及 position / bridge 查表
保持 BF16 —— loader 通过 `load_f32_tensor` 读取它们，且这些张量对量化误差最敏感。

> **K-quant 目前不可用。** `quantize_q4_k` 量化单个 6144×2048 矩阵需要约 97 s，
> 完整导出会耗时数小时。类型与解码链路都已打通并可往返校验，但 block search
> 需要批量化之后才能实际使用。

### 2.2 `ar_q8_0`：只量化自回归那一半

这是唯一在音质与速度之间取得平衡的量化模式。AR 和 NAR 读的是两组不同的
safetensors 张量（`self_attn` / `mlp` vs `nar_self_attn` / `nar_mlp`），
所以可以只压前者：

```bash
./.venv/bin/python -m tools.converter.yue2.convert_yue2 \
  --model-dir models/YuE2-3B --vae-dir models/YuE2-Vae \
  --main-out models/YuE2-gguf/yue2-ar_q8_0.gguf \
  --vae-out  models/YuE2-gguf/yue2_vae.gguf \
  --quant ar_q8_0
```

实测（960 latent 帧，BF16 之外全部如此）：

| | 效果 |
|---|---|
| AR 阶段 | 261.4s → 206.2s（**-21%**） |
| 端到端 | 846.9s → 740.3s（**-13%**） |
| 体积 | 7.26 GB → 5.94 GB |

**输出是另一首曲子，不是劣化版。** AR 的 token 轨迹会变，旋律保留但音色变化。
NAR 权重是纯 BF16，所以扩散流完全不受影响 —— 这正是与 `q8_0` 全量量化的区别。

注：`time_embedder.mlp.{0,2}` 名字里没有 `nar_` 前缀，但只在
`src/models/yue2/nar.rs` 的 `time_embedding()` 里被使用，因此按 NAR 处理，
`ar_q8_0` 确实把整个非自回归半边留在了 BF16。

## 3. 推理

**风格描述走 `--prompt`，不是 `--style`**（没有 `--style` 这个参数）。写错会被
静默忽略，然后报一个完全不相干的冲突错误：

```
$ ... --style "City Pop" --lyrics "..."
YuE2 --yue2 cannot be used with --audio
```

真正的缺失参数是 `--prompt`。

```bash
./target/release-fast/rust-model-inference \
  --yue2 \
  --model models/YuE2-gguf/yue2-q8_0.gguf \
  --vae   models/YuE2-gguf/yue2_vae.gguf \
  --prompt "City Pop, upbeat, danceable, groovy bass" \
  --lyrics  "[Verse]
路灯眨着眼睛 偷看谁的身影

[Chorus]
今晚不眠 快乐无限" \
  --out out.wav \
  --seed 12300 --steps 32 --max-tokens 600
```

### 3.1 参数

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--prompt` | 必填 | 风格 / 编曲描述 |
| `--lyrics` | 必填 | 带 `[Verse]` / `[Chorus]` 等 section 标记的歌词 |
| `--out` | 必填 | 输出路径，**必须以 `.wav` 结尾**，否则报错 |
| `--steps` | 32 | NAR 扩散步数。1～2 仅用于测速，音质明显劣化 |
| `--max-tokens` | 见下 | 限制 **semantic** 阶段上限；ABC 阶段仍用 metadata 里的 4096 |
| `--seed` | 831001 | |
| `--temperature` / `--top-k` / `--top-p` | 见 metadata | 同时覆盖 semantic 阶段 |

`--max-tokens` 存在一个坑：它只改 semantic 的 `max_tokens`，ABC 阶段的
`yue2.abc.max_tokens`（4096）不受影响。快速冒烟测试时 ABC 仍会跑满上千 token。

### 3.2 四个阶段与耗时

CPU（aarch64 20 核，Q8_0）实测：

| 阶段 | `--steps 1` | `--steps 32` |
| --- | --- | --- |
| ABC 采样 | 75 s | 75 s |
| semantic 采样 | 103 s | 103 s |
| prefix_kv prefill | 13 s | 13 s |
| NAR 扩散 | 6 s | 537 s |
| VAE 解码 | 52 s | 52 s |
| **合计** | **249 s** | **780 s** |

NAR 是绝对瓶颈：每步对全部 latent 位置做两次全模型前向。VAE 解码与 NAR 步数
无关，是固定开销。

`--threads` 不要超过 8。`ComputePool` 是纯自旋屏障线程池，线程数超过 8 后同一
cache line 的乒乓开销会压倒收益：实测 `--threads 20` 比默认慢 **16 倍**
（1.45 tok/s vs 23.96 tok/s）。默认值 `DEFAULT_THREAD_CAP = 8` 对本机是最优的。

### 3.3 BF16 性能路径

BF16 批量投影在一次 `ComputePool` 调度中处理全部输入帧，避免逐帧同步。
ARM64 NEON 路径每组四帧共享权重读取和 BF16 展开，但每个输出的累加顺序、
bias 加法和 BF16 舍入保持不变；不足四帧的尾部和单 token 解码沿用原算子。
其它架构及标量模式复用各自已有的单行 dot。

AR prefill 只计算最后一个输入 token 的最终 norm 和词表 logits，中间 token
仍完整更新 KV cache。`parity-trace` 构建保留所有中间 logits/checkpoint，
因此正常推理测速不要开启该 feature。

Vulkan BF16 dot 路径在 64 线程组中打包独立输出行，并让四帧共享权重读取，
每个输出仍保留 CPU 的 FMA 流、归约顺序和标量尾部。RADV NAVI31 上，256 帧
NAR velocity 的重复调用从 33.60 降到 12.42 秒，输出原始位一致；AR 耗时基本不变，
该 NAR 样例仍略慢于四线程 CPU。计时条件与完整生成检查见
[YuE2 Vulkan 优化记录](../develop/VULKAN_RADV_VALIDATIONS.md)。

NAR 的 q/k/v 三个投影由一次 `LinearExecutor::run_group` 提交。设备端的分组投影默认
走 prepared kernel：它把 activation 先舍入到 BF16（相对误差约 4e-3），并且交回未舍入
的 F32；CPU 的 `YuE2Weight::matmul_bf16_cpu` 则用原始 F32 输入、四帧共享一次权重、
最后把结果回舍入到 BF16。为了让设备与 CPU 逐位一致，q/k/v 的 executor 用
`with_cpu_dot_contract()` 走 `GpuMatmulMode::Dot`（shader 的 FMA stream 与
`dot_bf16_f32_4` 的归约相同），写回后再按 CPU 契约回舍入 BF16。只补回舍入、不改
dot 契约，整链仍有 98.4% 采样不同；两者缺一不可。回归：
`grouped_bf16_dot_contract_matches_the_cpu_kernel_bit_for_bit`。

这些优化不更换权重、不减少扩散步数，也不修改采样参数。可用以下回归检查
逐位一致性及中间 logits 的省略行为（不需要下载模型）。

### 3.4 VAE 卷积的设备路径

`--compute auto|vulkan` 下，VAE 的**普通卷积**（patch embedding、残差块、输出卷积）
走共享 `try_conv3d` 上设备；**转置卷积没有设备内核**，始终留在 CPU。CPU 路径
（`--compute cpu`，也是默认）完全不变 —— 它有多条按规模选择的实现，并且是
逐位对齐 reference 的那一条。

设备路径不经过 `ops::nn::conv3d_with_options` 的**默认** CPU 复核（那一层会重算一遍
卷积，等于把设备收益吃掉），而是直接调用 `try_conv3d`，失败时回落到 CPU。复核现在是
门控的：`RMI_GPU_CPU_CHECK=1` 或严格档 `--compute vulkan` 打开它，`--gpu`（Auto）
默认信任设备。打开时 VAE 用自身钉住 torch 的 CPU 路径核对设备卷积（容差
`1e-4 * scale`），不符就回退 CPU；共享的 `conv3d_with_options` 走同一个开关
（`ops::nn::conv_cpu_check_enabled`）。

实测（2026-11-12，本机 Linux x86_64 + NVIDIA L20，`--steps 1`，256 latent 帧 /
10.24 s 音频，8 线程）：

| 阶段 | CPU | Vulkan |
|---|---:|---:|
| ABC 采样 | 76.3 s | 74.8 s |
| semantic 采样 | 116.1 s | 113.6 s |
| NAR（含 prefix prefill） | 46.2 s | 63.9 s |
| VAE 解码 | 77.0 s | 25.6 s（3.0x） |
| **合计** | **315.6 s** | **277.9 s（-12%）** |

VAE 快 3 倍，AR/NAR 回到 CPU 路径（见下），所以**整链现在比 CPU 快 12%**。
设备路径的输出与 CPU 的差异为**最多 1 个 PCM16 档位**（982,912 个采样中
6,573 个不同，0.67%，最大绝对差 1/32768）—— 不是逐位一致，但对音频是舍入量级。

2026-11-19 在本机 aarch64 上用真实权重复测，
确认这条残余差异**只来自 VAE**：修好 NAR q/k/v 的 BF16 回舍入与 dot 契约后（见
§3.3），`yue2.nar.latents` 的 16384 个 F32 在 `--compute cpu` 与 `--gpu` 下逐位相同；
整链 wav 只有 0.566% 采样不同、最大 1 个 PCM16 档位（rms 0.075）。parity-trace 显示
差异从 VAE 的第一个设备卷积开始（`yue2.vae.decoder` layer 0 最大 1.7e-6，逐层放大到
输出前约 1e-5）。验收口径因此是：**AR + NAR 逐位一致，VAE 设备卷积 ≤1 个 PCM16
档位**。

NAR 的 q/k/v 三个投影共享同一个输入，现在由一次 `LinearExecutor::run_group`
提交（而不是三次）。实测这一步只省下 ~0.5 s（63.9 s → 63.4 s），因为设备侧的
成本随**行数**走（每次调用都要读对应权重），合并只减少提交次数；输出与三次分别
提交**逐字节相同**。

### 3.5 单行投影不上设备

`GpuLinear::try_matmul` 现在要求 **rows >= 4**：单行是 GEMV，设备侧每个 workgroup
的派发开销远超 CPU kernel。实测 YuE2 的 lm_head（2048x184704）在设备上
**~70 ms/次**（rows=1），而 CPU 只需 ~8 ms —— 所以 AR 的逐 token lm_head 现在
留在 CPU（这也是上面 AR 阶段追平 CPU 的原因）。rows>=4 的批量形状（例如 NAR 的
256 帧 velocity）仍然在设备上。

实机 GPU 回归 `vulkan_yue2_bf16_rounds_after_bias_across_tiles` 用的规模小于
`RUST_GPU_MIN_OUTPUTS` 默认门槛（16K 输出），设备会拒绝这些投影，所以这个测试自己把
门槛设成 0（进程级变量，因此它必须单独跑，别和别的 GPU 测试并进同一进程）。

```bash
cargo test --profile release-fast --lib models::yue2::ar::performance_tests
cargo test --profile release-fast --lib ops::dot::tests
RMI_SCALAR=1 cargo test --profile release-fast --features parity-trace --lib \
  bf16_batches_match_single_row_bits_with_bias_tails_and_partitions
```

## 4. 样例音频

本机在 `models/YuE2-gguf/samples/` 下留了真权重跑出的成品音频与说明
（`README.md` 内含各文件的权重精度 / 步数 / 耗时对照与复现命令）。

注意 `/models/` 在 `.gitignore` 中，这批样例只存在于本地，不随仓库分发。

## 5. Vulkan 设备路径的历史调查

当前 `--gpu --yue2` 的行为、实测收益与已知边界见 §3.3–§3.5：AR/NAR 只把
矩阵投影放到设备上，attention、状态操作和 BF16 舍入边界留在 CPU。下面保留
的是 2026-10/11 对**旧版「整个 AR 执行器上设备」**那条路径的 profile 记录，
用来解释它为什么被关掉、k-quant shader 为什么慢；它不代表当前投影路径的
正确性或性能。

The legacy whole-AR executor (`src/vulkan/yue2.rs`) is disabled in normal
sessions: its delayed initialization dropped the prefix KV, and after fixing
initialization a real 56-token prefix triggered RADV device loss. Its F32
normalization and residual operations also lack YuE2's BF16 rounding contract.
The initialization regression covers an isolated four-token prefix; it does
not establish numerical parity or make the whole executor eligible.

The measurements below are historical profiles of that executor. They do not
establish correctness or performance of the current projection path.

### The measurement

Same binary, same shapes, same seed; only the weight format differs.

| weights | per token | 28 layers | per layer | achieved |
| --- | --- | --- | --- | --- |
| BF16 (`yue2.gguf`) | 223 ms | 222.5 ms | 7.9 ms | 13.8 GOP/s |
| Q4_K_M (`yue2-ar_q4_k_m.gguf`) | 936 ms | 935.9 ms | 33.4 ms | 3.3 GOP/s |

Q4_K_M reads a quarter of the bytes and takes **4.2x longer**.

### Where the time is

`YUE2_GPU_PROFILE=1` splits the chunk into record / layers / head / submit /
read. For the Q4_K_M model:

```
record 0.5ms   layers 0.5ms   head 0.0ms   submit 936ms   read 0.0ms
```

Recording is free, readback is free, and `queue_submit` is 0.02 ms
(`RUST_GPU_SUBMIT_TRACE=1`). All of it is `wait_for_fences`, i.e. the device
is genuinely busy. `RUST_GPU_DISPATCH_TRACE=1` attributes it to two
pipelines: 168 quantize + 112 Q6_K + 56 Q4_K + 1 BF16 (`lm_head`) per token.

Two of those measurements had to be taken to get here:

- **Not the lm_head.** 184704 x 2048 is 20% of the FLOPs and looks like the
  obvious suspect. Skipping it changes 888 ms to 904 ms, i.e. nothing.
- **Not per-layer bookkeeping.** Truncating the layer loop gives 0.1 / 32.2 /
  66.4 / 129.4 / 258.4 / 904 ms for 0 / 1 / 2 / 4 / 8 / 28 layers: a perfectly
  linear 32 ms per layer, so nothing is amortizing badly and no single layer
  is special.

### The cause

Every matmul shader guards on `gl_LocalInvocationID.x != 0u`:

```glsl
layout(local_size_x = 64) in;
...
if (slot >= pc.group_count || row >= rows || gl_LocalInvocationID.x != 0u) return;
```

So each 64-thread workgroup does its work on lane 0 alone and the other 63
threads exit immediately. That is fine for the BF16 and F16 kernels, whose
inner loop is a straight `n_in` walk. It is expensive for the k-quants, whose
inner loop is 256 iterations of per-element unpacking:

```glsl
for (uint block = 0u; block < pc.blocks_per_row; ++block) {
    ...
    for (uint local = 0u; local < 256u; ++local) {
        uint packed = weight_byte(slot, offset + 16u + (group >> 1u) * 32u + (local & 31u));
        int value = int((group & 1u) == 0u ? packed & 15u : packed >> 4u);
        int activation = q8(block * 256u + local);
        partial[(local & 31u) >> 2u] += int(scale(slot, offset, group)) * activation * value;
        minimum_partial[group >> 1u] += int(minimum(slot, offset, group)) * activation;
    }
}
```

That whole 256-element block loop, plus the 6-bit unpack and the min-term
accumulator, runs on one thread of a 64-thread workgroup. The k-quants are the
only shaders with this much serial work per invocation, which is why they lose
4x while the dense shaders lose nothing.

The same guard is in `q4_0`, `q4_1`, `q5_k`, `q6_k`, `bf16`, `f16` and
`f32` matmul; the k-quants and `q4_0`/`q4_1` are the ones where it will
matter.

### What the device can actually do

`cargo run --release --features vulkan --example gpu_ceiling` on this aarch64 box:

```
  calls    total   us/call    GOP/s
      1   0.41 ms    412.4      215
    256  94.93 ms    370.8      239
```

215-239 GOP/s, against a CPU reference of 201 GOP/s for the same kernel. So
the hardware, the driver and the dispatch path are all healthy: 3.3 GOP/s for
a k-quant matmul is the shader, not the device. The same table shows the fixed
submit round trip is about 370 us, which matters for a per-token decode but
does not explain this -- the whole chunk is one submission.

### 继续收窄：分歧起于「第一个读历史的 attention 输出」

按 row 逐点 dump 后（`YUE2_DUMP_XIN`，临时插桩已回退）：

| pos | CPU embedding | GPU embedding |
|---|---|---|
| 36 | 0.028076172, -0.05908203 | **同左，逐位一致** |
| 37 | -0.021118164, 0.007019043 | -0.023925781, 0.003967285 ← **不同** |
| 38 | -0.0021209717, 0.013000488 | -0.023925781, 0.003967285 ← **GPU 重复了 37 的值** |

pos=38 的 GPU 值等于 pos=37 的 GPU 值，不是算错而是**下游发散**：pos=37 的输入
取决于 pos=36 的**28 层输出**，而 pos=36 的 embedding 本身一致。也就是说分歧**不在
embedding、不在 rms_norm 的输入**，而在 **pos=36 这一行经过 attention 之后的残差流**。

为什么之前测「prefill 的 attention 输出逐位一致」？因为那个 dump 读的是
`layout.attn` 的 region 起点（第 0 行 = prefix 的 token），不是当前 chunk 的第 36 行。
所以它一直「一致」，是**读错了行**。同一条记录里 pos=38 重复 pos=37 的值，也是同一个
坑：region 起点 vs 当前 row。

逐层 trace（`YUE2_TRACE_LAYER`，每层 submit+回读）证实 n=35（prefill 中间）全部层
逐位一致 —— 但因为每层都同步，GPU trace 只能跑到 n=35，到不了出问题的 n=37，所以
「逐层第一个偏离点」还没拿到。

⚠️ **这是今晚在这上面踩的第三个读法坑**，值得单独记：arena region 的 `read_f32(region)`
从 region 起点读，region 大小是 `max_rows * width`，**不是**整条序列。所以
1) prefill(36行) 时读 region 起点得到 token 0，不是 token 35
2) decode(1行) 时读 region 起点得到本 chunk 那一行（这个对）
3) 要看第 k 行必须按 `(k * width)` 偏移自己切片

### 分歧的精确范围（逐段 dump 实测）

在真实 GGUF 上用临时插桩逐段对比 CPU/GPU（插桩已回退）：

| 段 | n=36 (prefill, rows=36) | n=37 (decode, rows=1) |
|---|---|---|
| embedding (x 入口) | 一致 | 一致 |
| normed = rms_norm(x) | **一致** | **不同** |
| q = qk_norm_rope(normed) | **一致** | **不同** |
| KV cache (k,v) | **一致** | 一致 |
| attention 输出 | **一致** | **不同** |
| logits | **一致** | **不同** |

即：**分歧出在 rms_norm 之后、qkv 之前的 normed，而且只在 rows=1 的 decode
chunk 里出现**。因为 normed = rms_norm(x)，KV 和 embedding 又都一致，所以 x
（28 层残差累积）在第一个 decode chunk 内就已经错了 —— 而 prefill（rows=36）
整条 x 流是对的。

也就是说：rows>1 时残差流正确，rows=1 时错。最可能的原因是某个 recorder 在
rows=1 时依赖了跨 chunk 残留的 region 内容（例如 `layout.x` 只写了 rows 行，
而某个 in-place 的 residual add / 投影读到了上一 chunk 的残留，或 attention/scores
region 的有效范围在 rows=1 时算错）。

⚠️ 注意：我尝试逐层 dump `x` 时，读 `layout.x` region 起始位置拿到的是 prefix
残留（第 0 行）而非当前 chunk 的 row，导致一度误读成"GPU base_position 跑偏"。
逐层 dump 需要按 row 偏移读，不是按 region 起点。这是排查时的一个坑。

下一步应当：在 forward_chunk 内按 row 偏移 dump 每层后的 x（rows=1 时取
`layout.x + base_position*hidden`），定位第几层的残差开始偏离。

### 分歧定位：多 token prefill 逐位一致，单 token decode 全错

用 `YUE2_DUMP_LOGITS`（临时插桩，已回退）在真实模型上对比 CPU/GPU 的 AR logits：

```
prefill#0 (prefix, 多 token)   SAME   逐位一致
prefill#1 (第 1 个单 token)     DIFF
prefill#2+                      DIFF
```

**#0 完整一致，#1 起全错。** #0 与 #1 的唯一区别是 rows=N vs rows=1，以及 #1 依赖
#0 写进 KV cache 的内容。由于 #0 逐位一致，权重上传、host 写、量化、28 层前向本身
都是对的 —— 错的只发生在**单 token decode** 这条路径上（KV carryover 或 rows=1 的
某个 recorder 分支）。

已排除：

- **host-write 缺失**：forward_chunk 里 `write_f32` 后没有 host→compute 屏障。
  补上 `host_write_barrier` 后仍然 #1 分歧（逐位一样），所以不是它。
- **AR 权重本身**：#0 逐位一致说明上传/量化/前向都对。
- **token 流不同**：同 seed 同 logits(#0) 下采样出同样的 token，#1 的输入相同。

结论：这是 AR **decode（rows=1）** 路径的独立 bug，与 k-quant shader 的 1/64 lane
问题**无关**（那条是纯性能）。修好 decode 正确性之后，k-quant 的 5× 才敢合。

下一步应直接对拍 rows=1 的单步：用一个测试在 GPU 上只跑一次 rows=1 的
record_attention_rows / record_kv_write_rows / record_qk_norm_rope_rows，与 CPU 的
单步 forward 逐层比对，定位第一个分歧的算子。现有 tiny_for_test 走不了 GPU
（没有 tensor source），所以需要一个真实 GGUF 的对拍 harness。

### 修 k-quant shader 的 1/64 lane：5.05× 但正确性未确认（已回退）

上面的诊断说 Q4_K/Q6_K shader 慢 4.2× 的原因是
`gl_LocalInvocationID.x != 0u` 让 63/64 个 lane 立即返回，整条 256 元素解包循环
压在 lane 0。`q8_matmul_tiled_dp4a`（Z-Image 用，跑到 4088–5320 GOP/s）是**每 lane
一个输出、归约不跨 lane** 的范式，所以照抄：让每个 workgroup 处理 64 个输出行、
一行一个 lane，host 侧 k-quant 的 dispatch 从 `x=n_out` 改成 `x=ceil(n_out/64)`。

每个 lane 算自己那行的完全相同的求和顺序，**数学上逐位一致**。

**实测：AR 单 token 930 ms → 184 ms，5.05×。**

但**输出验证没通过**（PSNR 16 dB，GPU 与 CPU 音频差异明显），而且用 BF16 GPU
（不经过 k-quant shader）去比 CPU 也是 PSNR 16 dB、且两次 mean|Δ| 都恰好 ~3763，
**说明这个差异可能与 k-quant shader 无关，而是 AR 权重张量在 GPU/CPU 两条路径
本身就有别的分歧**（BF16 GPU 本该最接近 CPU）。

已回退，不留未验证的数值改动。留下的结论：

- k-quant shader 确实有 1/64 lane 的问题，且**按 tiled-dp4a 范式改成每 lane 一行
  能拿到 5×** —— 这个方向是对的，dispatch 网格推导也自洽
- 但 YuE2 AR 的 GPU 路径**在 k-quant 之外还有一个独立的正确性问题**，需要先定位它
  才能放心用 shader 修复。线索：BF16 GPU 与 CPU 也是 16 dB，且 mean|Δ| 在
  Q4KM-GPU / Q4KM-CPU / BF16-GPU 三者之间都接近 3763，提示差异可能来自 VAE/NAR
  （仍在 CPU）或 attention kernel 的 CPU/GPU 实现不同，而非 AR 权重本身
- 下一步应是：把 VAE/NAR 关掉、只跑 AR 单步，逐段对拍 BF16-GPU vs BF16-CPU 的
  hidden state，定位第一个分歧点，再回来收 shader 修复

### What would fix it

Split the 256-element inner loop across the workgroup and reduce once, instead
of returning 63 lanes. The accumulators already have the right shape for it:
`partial[8]` indexed by `(local & 31u) >> 2u` and `minimum_partial[4]` by
`group >> 1u` are both partial sums over disjoint subsets of the 256 values,
so a 64-thread version reduces `partial[8]` and `minimum_partial[4]` across
lanes and then does the same final `fma` chain on lane 0. The scale/minimum
terms are per-`group` constants, so they can be loaded once per group rather
than re-fetched 256 times.

Expectation: the k-quant matmul should approach the dense shader's
per-invocation throughput, which is the 4.2x between the two rows of the
table above. Even at parity with BF16 the device is 8x slower than the CPU's
21 tok/s for this stage, so the shaders are necessary but not sufficient --
single-token decode on a 20-core CPU is a hard target to beat.

### 旧执行器的阶段耗时参考

These stage costs are from the legacy whole-AR executor, whose delayed
initialization and F32 norm/residual operations are what got it disabled.
They are not the current `--gpu` projection-offload numbers (see §3.4).
Measured stage cost for the 8-step / 960-frame config, for reference:

| stage | time |
| --- | --- |
| ABC | 29.8 s |
| semantic | 139.2 s |
| NAR prefill | 24.1 s |
| NAR 4 steps | 181.2 s |
| VAE decode | 159.9 s |

The AR half is 169 s of a 537 s render, which is the ceiling on what this
session can win.
