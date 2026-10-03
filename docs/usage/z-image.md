# Z-Image Turbo 用法

Z-Image Turbo 是文生图模型，分三个独立 GGUF 组件：

- DiT：`--model`
- Qwen3 文本编码器：`--text-encoder`
- Flux VAE：`--vae`

GGUF `general.architecture = pig`，对应 `src/models/diffusion/pig.rs` 与
`src/models/diffusion/z_image/`。CLI 入口 `src/app/diffusion.rs::run_z_image_cli`。

> 共用前置：构建 `cargo build --release --bin rust-model-inference`。
> 当前**仅 CPU**，**仅 512×512**，**仅 txt2img**。
> img2img、GPU、Z-Image Base 列为 `Unsupported`（SUPPORTED_MODELS.md）。

## 1. 三组件

```bash
cargo run --release --bin rust-model-inference -- \
  --model models/z-image-gguf/z-image-turbo-q8_0.gguf \
  --text-encoder models/z-image-gguf/qwen3_4b_f32-q8_0.gguf \
  --vae models/z-image-gguf/pig_flux_vae_fp32-f16.gguf \
  --prompt "A red fox sleeping beneath a pine tree" \
  --steps 8 --resolution 512 --seed 42 --threads 1 --out fox.png
```

三个 GGUF 缺任一都会立即拒绝（`src/app/mod.rs:32-37`）：

```
Z-Image model requires --text-encoder, --vae, --prompt, and --out
```

`--prompt` / `--out` 同样必填。

### 实测耗时（20 核 DGX Spark，512×512，8 步）

同机三条路径，prompt 为 "a red fox sleeping beneath a pine tree"，seed 42。

| 路径 | 步数 | DiT forwards | 总耗时 | 每次 forward |
|---|---|---|---|---|
| 本仓库 CPU | 8 | 7 | 466 s | 62.4 s |
| **本仓库 GPU（`--gpu`）** | 8 | 7 | **150 s** | **16.8 s** |
| PyTorch 2.11 + CUDA（同一权重） | 9 | 8 | 15.1 s | 1.89 s |

`--steps N` 跑 N-1 次 forward（最后一步是 sigma→0 的收尾）。本仓库两条路径步数
相同，**GPU 快 3.12×**；PyTorch 多跑一次 forward，按每次 forward 折算仍快
**8.9×**。三条路径与 CPU 参考图的差异都在 2.5/255（40 dB）以内，本仓库两条
路径同种子逐位可复现。

去噪阶段（不含文本编码与 VAE 解码）：CPU 437 s，GPU 118 s，PyTorch 约 13 s。

单步分解（GPU，历史基线 14.7 s/步；当前机器已失效，见下方 A/B）：

| 阶段 | 单步 | 占比 |
|---|---|---|
| FFN 主栈（GPU，Q8_0 tiled） | 3.86 s | 26% |
| FFN refiner（GPU，F16 tiled） | 1.42 s | 7% |
| attention（CPU，pool 并行 + query 分块） | 3.41 s | 23% |
| rms_norm + AdaLN + QKV（GPU，单 command buffer） | 2.09 s | 14% |
| 输出投影（GPU） | 1.24 s | 8% |
| RoPE + 调制（CPU） | 0.11 s | 1% |

> **线程数也非越多越好**：16 线程首步 269.3 s，反而慢于 8 线程的 140.5 s。

> ⚠️ **历史绝对值已失效。** 下表的 14.7 s/步 记录于本机更早的状态；当前机器上
> 纯净 HEAD（9a831ee）实测已是 23.07 s/步，`release` 与 `release-fast` 几乎无差
> （19.5 vs 19.4 s），所以差异**不是** LTO。旁证 VAE 解码——只跑一次、与步数无关、
> 代码未改——在多次相同运行里分别为 34.5 / 44.8 / 77.2 s，本身就有 2.2× 噪声，
> 不能用于任何归因。**唯一可信的是同一会话内前后对照的差值**，故下表改为 A/B 实测：
>
> | 8 步 / 512 / seed 42 / 20 线程 | 纯净 HEAD | 现状 | Δ |
> |---|---|---|---|
> | denoise | 184.5 s | 158.8 s | **−13.9%** |
> | 单步 | 23.07 s | 19.85 s | −3.22 s |
> | refiner/步 | 3608 ms | 1421 ms | −60.6% |
> | 总计 | 231.5 s | 211.7 s | −8.6% |
>
> 两次输出逐字节一致。总计只降 8.6% 是因为 45–51 s 的 VAE 解码占了尾段且噪声大。
> 在重建绝对基线之前，不要用这些数字与 PyTorch 的 1.89 s/forward 算比值。

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

来自 `src/app/cli.rs:447-449`：

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
| GPU 后端 | `Unsupported`（仓库 CPU-only） |

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

| 选项 | 用途 | 必填 |
|---|---|---|
| `--model` | DiT GGUF | ✓ |
| `--text-encoder` | Qwen3 文本编码器 GGUF | ✓ |
| `--vae` | Flux VAE GGUF | ✓ |
| `--prompt` | 文生图 prompt | ✓ |
| `--out` | 输出 PNG 路径 | ✓ |
| `--steps` | Euler 步数（NFE） | ✓ |
| `--resolution` | 输出分辨率（divisible by 16） | ✓ |
| `--seed` | RNG 种子 | — |
| `--threads` | ComputePool 线程数 | — |

## 7. 相关源码索引

- `src/app/diffusion.rs` — `run_z_image_cli` 主入口
- `src/app/mod.rs:32-37` — pig arch 三组件必填检查
- `src/app/cli.rs:447-449` — `--steps` / `--resolution` 校验
- `src/models/diffusion/pig.rs` — DiT 模型
- `src/models/diffusion/z_image/dit.rs` — DiT forward
- `src/models/diffusion/z_image/text.rs` — Qwen3 文本编码器
- `src/models/diffusion/z_image/vae.rs` — Flux VAE
- `tests/z_image_reference.rs` — pinned Oracle 对齐
- `docs/REFERENCE_IMPLEMENTATIONS.md` — Oracle pin 与构建脚本
- `tools/converter/z_image/convert_z_image.py` — safetensors → 三组件 GGUF
- `tools/converter/z_image/probe_layout.py` — 只读 header，打印张量名/维度分布