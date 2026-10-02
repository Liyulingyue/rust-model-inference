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

| 阶段 | 耗时 | 占比 |
|---|---|---|
| 三组件加载 | 73 ms | — |
| 文本编码 | 677 ms | 0.06% |
| **去噪（8 步）** | **1133 s** | **97.2%** |
| VAE 解码 | 30 s | 2.6% |
| **总计** | **1166 s ≈ 19.4 min** | |

每步约 140 s，其中 `attention_into` 占 50%、`linear ffn` 36%、`linear qkv` 11%。

> **线程数也非越多越好**：16 线程首步 269.3 s，反而慢于 8 线程的 140.5 s。

### GPU 路径（`--gpu`，同机 512×512，8 步）

| | 耗时 | 每步去噪 |
|---|---|---|
| CPU | 1166 s | 约 140 s |
| GPU（Vulkan） | **718 s** | 约 86 s |

整体 **1.62×**，去噪阶段 **1.65×**。主层 30 层的 Q8_0 投影与
`rms_norm` + AdaLN 调制在 GPU 上；attention 与 4 层 F16 refiner 仍在 CPU，
所以 CPU attention（约 70 s/步）是目前的天花板。

关键改动是 Q8_0 grouped matmul 的 tiling。旧 kernel 每个 workgroup 只算
**一个**输出元素：64 个 lane 切分 K 维再做树形归约，于是每字节权重只服务
**一个** token，权重复用为 1。实测 W2 形状 757.6 ms / 110 GOP/s，而同一形状
PyTorch 约 5900 GOP/s。`shaders/glsl/q8_matmul_tiled_dp4a.comp` 让 lane 持有
输出列、一次权重加载复用于 8 个 token 的寄存器累加器，**11.8×**（64.4 ms /
1289 GOP/s），与旧 kernel 相对误差 9.5e-7。

> **`examples/dit_vk_bench.rs` 的 0.4×–1.36× 是误导性的**：它测的是 GEMV
> （`gpu_out` 只有一行长），权重复用同样为 1，于是读带宽看起来正常，却完全
> 没有测到真实形状。判断 kernel 速度必须用
> `zimage_tiled_matmul_beats_the_one_token_per_weight_kernel`，它轮流读取 8 个
> 不同权重矩阵以绕开 L2，单矩阵热缓存下会报出 10 万 GOP/s 的假数字。

> **host 写 → device 读的 barrier 不能省。** arena 是 CPU 直接写、shader 直接
> 读的映射内存，缺失 HOST→COMPUTE barrier 时能否看见写入只取决于速度：旧
> kernel 每次 dispatch 约 700 ms，写入早被"看见"；tiled kernel 约 64 ms，
> 同种子两次渲染就会差 1.9–5.8/255，且只有 arena 大到装不进缓存的 512×512
> 才复现（256×256 逐位一致）。`VulkanContext::host_write_barrier` 在每次
> submit 前补上它。

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