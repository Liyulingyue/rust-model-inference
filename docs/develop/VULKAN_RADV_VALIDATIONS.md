# Vulkan RADV 实机验证记录

> **本文档（2026-11-19）合并自三份带日期的验证快照**：
> `VULKAN_RADV_VALIDATION_2026-10-07.md`（多模型验证）、
> `VULKAN_AUK_RADV_VALIDATION_2026-10-08.md`（AuK TTS 投影）、
> `VULKAN_YUE2_OPTIMIZATION_2026-10-08.md`（YuE2 BF16 投影优化）。
> 三轮共用同一台 RADV NAVI31 验证机；内容按时间顺序原样收录，
> 各自的验证边界声明保持不变。
>
> 目录：**一、2026-10-07 多模型验证**（embedding / dense LFM / YuE2 / LFM MoE /
> Edge / Z-Image）；**二、2026-10-08 AuK TTS 投影验证**；**三、2026-10-08
> YuE2 BF16 投影优化**。

## 一、Vulkan 实机验证：RADV NAVI31，2026-10-07

验证分支 `codex/vulkan-multi-model`，基础提交 `eccce00`，已合入 main `50fd20e`。
下列结果包含本次实机发现的修复，执行的是原生 Rust 推理。

五类路径均完成真实 GPU 执行。Embedding、dense LFM 和 YuE2 固定样例逐位一致；
Edge/LFM MoE 通过原有数值门槛且 greedy IDs 一致。Z-Image 成图成功，但像素有差异，
没有建立整条图像管线的浮点 parity。GPU 不保证每个模型都更快，YuE2 样例仍较慢。

### 环境与证据

- Ubuntu 24.04，内核 `6.8.0-79-generic`，Rust 1.98.1，`release-fast`。
- CPU 为 AMD EPYC 9334；容器配额 16 CPU、55 GiB RAM，推理线程数 4。
- Vulkan 设备为 `AMD Radeon Graphics (RADV NAVI31)`，PCI `1002:744b`，48 GiB VRAM。
  Mesa 25.2.8，设备 API 1.4.318；shader F16、int8 dot product 均可用。
- ICD：`/usr/share/vulkan/icd.d/radeon_icd.json`。
- 远端工作目录 `/workspace/rmi-vulkan-eccce00`；日志和验证源码在 `target/validation/`。
  本地小体积证据在 `target/vulkan-radv-validation/`；不包含模型权重。
- 本地证据归档 `target/vulkan-radv-validation-evidence.tar.gz` 约 10 MiB，包含日志、
  helper 源码、JSON、PNG、WAV 与短片段原始 F32；`evidence-sha256.json` 记录文件身份。
  689 个 Rust 源码、shader、Cargo 文件及 helper 的本地/远端 SHA-256 核对无差异，
  清单为 `source-sha256.json`。
- 远端无法联网。权重从 ModelScope 分块读取，直接通过 SSH 写入远端 `.part`；
  SHA-256 校验成功后才原子改名。完整权重及其归档未落到本地磁盘。

冷启动/预热是同一进程中的两次 GPU 检查；embedding 与 dense LFM 的这些便捷 API 每次重新加载权重，
不能据此推断持久缓存的预热收益。计时是功能验证样本，部分期间有 CPU 推理和下载并行，
没有统计中位数，也未与其他推理框架作性能比较。

### 数值与执行结果

统一使用现有模型检查门槛 `abs(gpu-cpu) <= 2e-3 + 2e-3*abs(cpu)`。
同时记录 F32 原始位差异、有限值、greedy IDs，以及真实 GPU submission；
GPU broken 或零 submission 都会让检查失败。未启用会关闭 GPU 投影的 `RMI_PARITY_TRACE`。

| 模型与用例 | 对比 F32 值数 | CPU 秒 | GPU 冷/预热秒 | submission/次 | 结果 |
|---|---:|---:|---:|---:|---|
| BERT base Q8_0，4 文本 × 768 维 | 3,072 | 1.007585 | 0.916882 / 0.891856 | 2,736 | 原始位一致，排序一致 |
| EmbeddingGemma 300M Q8_0，同上 | 3,072 | 1.144942 | 1.447914 / 1.461246 | 4,880 | 原始位一致，排序一致 |
| LFM2-350M Q8_0，prefill logits | 65,536 | 1.866565 | 0.891754 / 0.931654 | 1,581 | 原始位一致，首 token 一致 |
| LFM2.5-1.2B Instruct Q8_0，同上 | 65,536 | 5.739302 | 2.424361 / 2.430642 | 1,581 | 原始位一致，首 token 一致 |
| YuE2 3B BF16，56-token 前缀 + 32 greedy | 369,408 | 6.202244 | 15.861734 / 14.275351 | 17,281 | 首尾 logits 原始位一致，32 IDs 一致 |
| LFM2.5-8B-A1B Q8_0，prefill + 32 greedy | 256,000 | 45.919748 | 18.965203 / 6.666924 | 15,888 | max abs 3.815e-6，门槛违规 0，32 IDs 一致 |
| Edge0-35B-A3B，lossless MLX affine，prefill + 32 greedy | 496,640 | 53.214280 | 21.823901 / 12.366624 | 37,143 | max abs 1.049e-5，门槛违规 0，32 IDs 一致 |

前五行的冷/预热检查均为 `max_abs=0`、原始位差异 0、门槛违规 0。
LFM2.5 MoE 两次都有 149,297 个原始位差异，未达到逐位一致；已有误差门槛和 token 对比均通过。
Edge0 两次都有 463,277 个原始位差异；同样只通过误差门槛和 token 对比。
Edge GGUF 用仓库 lossless converter 从校验过的四个原始分片转换，验证了 2,377 个张量。
该文件的 CPU/GPU 对比没有重新建立官方 MLX Oracle；原 CPU Oracle 记录的 GGUF 身份另见
`tools/oracle/edge0/README.md`。
Edge 主 CLI 的 CPU/`--gpu` 都生成 `5` 并正常结束；GPU 记录 16,862 次 MLX dispatch，
不再输出过时的“忽略 --gpu”提示。该 CLI 用例使用 chat template，遇到 EOS 提前结束；
32 IDs 的对比来自上表的 raw-prompt session 用例。
BERT、EmbeddingGemma 的相关性排序都是 `[0,1,2]`；非相关文档排在最后。
上述 embedding 与 dense LFM 还通过了 `RUST_GPU_DP4A=0` 的真实模型对比。

Dense LFM 另跑主 CLI，prompt 为 `Continue the sequence: 1, 2, 3, 4,`，
`--temp 0 --kv-cache f32 --max-context 256 --threads 4 --max-tokens 32`。
两种模型 CPU/GPU 的全部 32 个输出 IDs 相同，GPU 每次 6,232 次 submission：

| 模型 | CPU/GPU 总耗时毫秒 | CPU/GPU 生成 tok/s |
|---|---:|---:|
| LFM2-350M | 6,995 / 2,741 | 9.6 / 26.4 |
| LFM2.5-1.2B | 22,223 / 7,615 | 3.0 / 9.7 |

CLI IDs 通过临时输出记录验证，记录后已恢复源文件；未加入生产调试开关。
日志为 `lfm2-cli-{cpu,gpu}-final.log`、`lfm25-cli-{cpu,gpu}-final.log`。

### 图像与音频

#### Z-Image Turbo

固定三组件 Q8_0 DiT、Q8_0 Qwen3-4B、F16 VAE，prompt
`A red fox sleeping beneath a pine tree`，seed 42、4 线程。
CPU 八步尝试的首个 DiT 前向耗时 1,825.2156 秒，随后主动停止。
512×512 的两步 CPU 仍要执行两个完整前向，该尝试也主动停止。
数值对比使用 128×128 两步；512×512 两步/八步 GPU 成图单独检查。

| GPU 成图 | 总秒数 | text / DiT / VAE 毫秒 | submission | F16 卷积 dispatch |
|---|---:|---|---:|---:|
| 512×512，两步 | 323.393920 | 2,510.5 / 30,193.5 / 290,194.8 | 53,185 | 47,808 |
| 512×512，八步 | 402.758633 | 2,453.6 / 108,009.6 / 291,828.7 | 57,037 | 47,808 |

两次均为有限值检查后的 512×512 PNG，GPU 未 broken；实际使用 Q8 tiled、F16 tiled
DiT 与 F16 VAE shader。VAE 是该样本的主要耗时，尚未完成性能优化。
八步图像已查看，可辨认出树下卧着的狐狸；该观察不是模型质量 Oracle。
未完成 512×512 CPU 成品对比，因此不声明该尺寸的数值 parity 或加速比。

128×128 两步，CPU 总耗时 359.780291 秒、GPU 26.349318 秒（约 13.65 倍）；
GPU 8,365 次 submission，其中 F16 卷积 2,988 次。
CPU text / DiT / VAE 为 16,083.1 / 312,816.1 / 30,213.5 毫秒；
GPU 为 2,461.6 / 6,139.5 / 17,106.4 毫秒。

49,152 个 RGB channels 中 38,254 个不同，最大绝对差 39（uint8），MAE 1.7981，
RMSE 2.7453，PSNR 39.3589 dB；PNG **不逐位一致**。
这是成品差异测量，没有借此放宽已有算子/模型门槛，也未验证整条图像管线的逐层浮点值。
详情为 `zimage-output-comparison.json`；CPU/GPU PNG 和 512 成图均已留存。

#### YuE2

主模型 BF16，音频 VAE F32，seed 12300，4 线程；ABC 最大 1,024 tokens，
semantic 最大 256 tokens，NAR 4 步。style/lyrics 与 `target/validation/yue-check.rs` 固定一致。
CPU 基线生成 302 个 ABC tokens、256 个 semantic tokens、256 帧 latent，
48 kHz 双声道，每声道 491,456 samples（约 10.239 秒），耗时 516.568676 秒。
波形全为有限值，峰值 0.5436629653、RMS 0.0885989902，无超出 ±1 的样本。

GPU 耗时 852.855718 秒，实际 201,163 次 BF16 projection submission。
全部 ABC/semantic IDs、维度、16,384 个 latent F32 和 982,912 个双声道波形 F32
均与 CPU 原始位一致，max abs/RMSE/门槛违规均为 0，RMS/峰值/无 clipping 与 CPU 一致。
留存 `yue-output-comparison.json`、原始 F32 和可播放的 `yue-gpu.wav`。
该样本 GPU 比 CPU 慢约 1.65 倍；投影同步/上传和 NAR 开销尚需优化。
256-token 上限使该用例只验证短片段；不代表完整歌曲质量，未做主观听测。

### 实机发现的修复

1. **MLX 上传尾部 padding**：静态 buffer 分配包含 16-byte guard 和 uint 对齐，
   shape 验证曾将它误判为额外矩阵行。现在接受逻辑长度或实际 padding 长度，
   仍拒绝错误行数；真实 MLX 4/8-bit、LoRA 和 65-row tile 尾部检查通过。
2. **Q8_0 分组求和**：原 shader 将 block 分到 64 个累加器，改变了 CPU 的求和顺序，
   真实 embedding/LFM 未达到原有门槛。两种 grouped shader 都保留八条 FMA 流及
   原来的最终归约顺序。旧 shader 下回归失败，修复后 dp4a 和 baseline 都通过。
   64-lane workgroup 中仅八个 lane 计算是当前性能上限，后续优化需保留数值合约。
3. **YuE2 BF16 投影**：沿用 `GpuLinear`，新增同一 BF16 存储上的 dot 归约语义，
   匹配 CPU AVX2 的八条 FMA 流、NEON 的四条 FMA 流及 scalar tail；bias 后才舍入 BF16。
   通用 BF16 kernel 的 GPU 语义保持原样。1024/1027 宽度、65 行的 cancellation 回归
   在旧 shader 下失败，修复后通过；真实 AR logits 从 `max_abs=0.125`、315,112 个
   门槛违规变为原始位一致。当前机器实际执行的是 AVX2 对应模式。
4. **YuE2 旧整图 AR 执行器**：首次初始化 guard 曾跳过前缀，decode 才创建空 KV。
   修复生命周期后，真实 56-token 前缀仍触发 RADV device lost；该执行器也缺少
   BF16 norm/residual/activation/RoPE 舍入，以及 KV delta 回传。
   正常会话因此使用已验证的投影路径。隔离四-token 生命周期检查不构成整图数值验收。

### 算子、静态检查与已知失败

- `vk_check` 的五组 shape 通过；batched 11 格式检查通过。
- `vk_ops_check --formats q8_0,q4_0,q4_1,q4_k,q5_k,q6_k,mlx4,mlx8,bf16,f16 --rows 3` 通过。
- GPU 回归覆盖 Q8 canonical reduction（dp4a/baseline）、MLX 4/8 + LoRA、
  VAE F16 卷积，以及 YuE2 BF16 bias、65-row 尾部和 cancellation。
- default、vulkan、vulkan+parity-trace 的 lib/CLI/`rust-model-server` 编译，
  CPU BF16 batched 原始位回归、格式检查通过。
- 修改的三个 shader 通过 SPIR-V validator、重编译字节一致性；完整 manifest hash 校验通过。
  全量 shader checker 随后停在**未修改的 `softmax.spv` 重编译字节差异**，不能声称全量检查通过。
- 旧单行 F32 的零误差门槛仍失败：max abs `3.624e-5`，已在隔离 main `50fd20e`
  原样复现。VAE SiLU 的 CPU 逐位断言也在该基线复现；没有放宽断言。
  不据此声称全量 CPU 或全格式算子套件通过。

### 复现

在远端工作目录中使用仓库内工具链和离线 cache：

```bash
export PATH="$PWD/target/toolchain/bin:$PATH"
export CARGO_HOME="$PWD/.cargo-home"
export CARGO_BUILD_JOBS=8
export RUSTFLAGS=-Awarnings
export VK_DRIVER_FILES=/usr/share/vulkan/icd.d/radeon_icd.json
cargo build --profile release-fast --locked --offline --features vulkan --lib

# 用现有库构建留存的验证 helper，无第三方推理引擎。
python3 - <<'PY'
from pathlib import Path
import subprocess
libs = list(Path('target/release-fast/deps').glob('librust_model_inference-*.rlib'))
assert len(libs) == 1, libs
for name in ['model-check', 'yue-check', 'zimage-check']:
    subprocess.run(['target/toolchain/bin/rustc', '--edition=2021', '-C', 'opt-level=3',
                    f'target/validation/{name}.rs', '--extern',
                    f'rust_model_inference={libs[0]}', '-L', 'dependency=target/release-fast/deps',
                    '-o', f'target/validation/{name}'], check=True)
PY

export RUST_GPU_DISPATCH_TRACE=1
target/validation/model-check embedding models/ggml-org--bert-base-uncased/bert-base-uncased-Q8_0.gguf
target/validation/model-check embedding models/ggml-org--embeddinggemma-300m-GGUF/embeddinggemma-300M-Q8_0.gguf
target/validation/model-check lfm models/unsloth--LFM2-350M-GGUF/LFM2-350M-Q8_0.gguf
target/validation/model-check lfm models/unsloth--LFM2.5-1.2B-Instruct-GGUF/LFM2.5-1.2B-Instruct-Q8_0.gguf
target/validation/model-check yue models/YuE2-3B/yue2-bf16.gguf
target/validation/model-check moe models/unsloth--LFM2.5-8B-A1B-GGUF/LFM2.5-8B-A1B-Q8_0.gguf
target/validation/model-check edge models/edge0--Edge0-35B-A3B-preview/Edge0-35B-A3B-preview-lossless.gguf

RAYON_NUM_THREADS=4 target/validation/yue-check cpu target/validation/yue-cpu
RAYON_NUM_THREADS=4 target/validation/yue-check gpu target/validation/yue-gpu
RAYON_NUM_THREADS=4 target/validation/zimage-check cpu 2 target/validation/zimage-cpu-128.png 128
RAYON_NUM_THREADS=4 target/validation/zimage-check gpu 2 target/validation/zimage-gpu-128.png 128
RAYON_NUM_THREADS=4 target/validation/zimage-check gpu 2 target/validation/zimage-gpu-2.png
RAYON_NUM_THREADS=4 target/validation/zimage-check gpu 8 target/validation/zimage-gpu-8.png

cargo test --profile release-fast --locked --offline --features vulkan --lib \
  vulkan_yue2_bf16_rounds_after_bias_across_tiles -- --ignored --nocapture
cargo test --profile release-fast --locked --offline --features vulkan --lib \
  vulkan_q8_grouped_preserves_eight_stream_reduction -- --ignored --nocapture
RUST_GPU_DP4A=0 cargo test --profile release-fast --locked --offline --features vulkan --lib \
  vulkan_q8_grouped_preserves_eight_stream_reduction -- --ignored --nocapture
```

### 权重身份

| 文件 | bytes | SHA-256 |
|---|---:|---|
| bert-base-uncased-Q8_0.gguf | 117852384 | `d03fa752e9930130ce03358a8e22ecd4f3f1dba4ec37149218ef1e310ce72460` |
| embeddinggemma-300M-Q8_0.gguf | 333590944 | `b5ce9d77a3fc4b3b39ccb5643c36777911cc4eb46a66962eadfa3f5f60490d63` |
| LFM2-350M-Q8_0.gguf | 379214560 | `cd29222147b1f62b4bb739f2fc575f0883ea9a5c05f5ac49b28d101ee9afbab7` |
| LFM2.5-1.2B-Instruct-Q8_0.gguf | 1246254304 | `b808eead8d6061f71990ab7b144c5ca6650fad219aa7acc407fe5fb26abf3dc2` |
| LFM2.5-8B-A1B-Q8_0.gguf | 9010196064 | `ec11666b6129f0b4fe893760b66797f22e1c478a561b40e365f2b6930729b8d2` |
| Edge0-35B-A3B-preview-lossless.gguf | 19562215072 | `068c996c2d50d9332a3f44b35b82430837144110b1bdb6e80f16300a77086efe` |
| qwen3_4b_f32-q8_0.gguf | 4274478528 | `aeaeb1222b858fc98fa01f31bdfdceb8a5b1719a9b91a3909774a85ecf8930fe` |
| z-image-turbo-q8_0.gguf | 7224691776 | `39674cec3b98e737276443cbf02f29b0aea616164e465c731f19903b00363cfd` |
| pig_flux_vae_fp32-f16.gguf | 167788736 | `7e9b2072ef8d8bde202804362b273a96233e54e4b52c820662cdc70b3e08b27a` |
| YuE2-3B/yue2-bf16.gguf | 7268395360 | `e2d202b4d16431159338337b6b2ced185400224d9d77eeef770c013efeb12889` |
| YuE2-Vae/yue2-vae-f32.gguf | 265515360 | `b6f8c4da9e5281546d3e76787ba6d75d2163cd8611a0a9fd31d6510df6a3bb8b` |

Edge 原始权重的 SHA-256：

| 文件 | SHA-256 |
|---|---|
| model-00001-of-00004.safetensors | `f7fff28fecb8221e53787c6339c938b113bcb94c7bbc60d09a8826eba08ef1ab` |
| model-00002-of-00004.safetensors | `31dcdb1c49eebdb1505bd14e3cb33f9cf900bd2546b638f2464694ae763a033f` |
| model-00003-of-00004.safetensors | `3e66de06a1f03dade16a612a368cfce4a4c9caa4efd7d28185454384082cec03` |
| model-00004-of-00004.safetensors | `a5d0cf03519c26f8b506df6b0ba60526e5c08c8cea22d0c21ce92950e58a5422` |
| lora_edge0_35b.safetensors | `a407f6d27bb55e39e6461bd4bce1fd90489e70c7582b68a796fb764bdb441ec1` |

范围只限上述文件、fixture、RADV 设备和所执行模式；其他量化、图像尺寸、长音频、
跨驱动/跨 CPU 的逐位一致性及整体性能需要各自验证。

## 二、AuK Vulkan 实机验证：RADV NAVI31，2026-10-08

延续 `codex/vulkan-multi-model`，基础提交 `65fb749`，main `50fd20e` 已合入。
本次验证原生 Rust 的 AuK TTS 投影 offload；模型支持等级仍为 **Experimental**。

### 环境和权重

Ubuntu 24.04，Rust 1.98.1，`release-fast`。AMD EPYC 9334，容器配额 16 CPU、55 GiB RAM，
本轮推理使用 8 线程。GPU 为 AMD Radeon Graphics（RADV NAVI31），48 GiB VRAM；
Mesa 25.2.8，Vulkan API 1.4.318。远端目录 `/workspace/rmi-vulkan-eccce00`。

权重来自 [ModelScope audio-cpp/AuK-Base-and-Flash-GGUF](https://www.modelscope.cn/models/audio-cpp/AuK-Base-and-Flash-GGUF)。
按 32 MiB range 读取，经 SSH 直接写入远端，单次读写块 1 MiB；完整权重未落在本地磁盘。
远端完整 SHA-256 校验后才原子改名。

| 文件 | bytes | SHA-256 |
|---|---:|---|
| `auk-flash-f16.gguf` | 3,061,149,120 | `f9ad537dbe1f207943a42ee48dad070fb91b62c7b812e0a124f16dfba72dc092` |
| `auk-base-f16.gguf` | 3,061,149,120 | `d7efc034ef18aa6d5080cb8a3a67731df31aeae7f73d68b19400503db89bd586` |
| `auk-base-q8_0.gguf` | 1,639,358,080 | `dcf96485cf4d44f8dd84eb190348aa3354117380c95e9b3d90b1f4c797503a39` |
| `auk-vae-f32.gguf` | 637,364,224 | `401e96baa1ac958d66fbe2c8cf5d25105c995c46dd46247c879e6114ef503abd` |
| `qwen2.5-omni-3b-q8_0.gguf` | 4,253,964,480 | `732861a7d938d08770f98d4732ee32982ee042c6c7161eed48db5569a8a620c3` |

### 修复与算子检查

- 发布的原生 Qwen GGUF 使用 `audiocpp` 架构和 `thinker.model.*` 张量名。AuK 在
  校验 36 层、2048 hidden、16 Q heads/2 KV heads、11008 FFN 的完整签名后，
  用零复制视图复用现有 Qwen2.5 trunk；配置来自 `Qwen/Qwen2.5-Omni-3B` 的
  `thinker_config.text_config`。编码器 GGUF 不含 LM head，文本 hidden 路径不计算 logits。
- audio.cpp 对长于 GGML 名称限制的张量使用 `_audiocpp.N` 物理名，完整原名保存在
  `audiocpp.tensor_names`。公共 GGUF 加载器现在验证这个数组的类型、数量和唯一性，
  恢复原名，保留原始数据偏移与字节。此前第 10–19 个 single block 的 AdaLN
  weight 因名称过长无法被查到，代码只执行了前 10 个；现在加载并执行全部 20 个。
  此规则按 [audio.cpp 固定提交](https://github.com/0xShug0/audio.cpp/blob/129e93ed623c250da1ce762bc14583a81baf74a4/src/framework/assets/tensor_source.cpp)
  核对；没有猜测共享矩阵，也没有补造权重。
- 旧 F16 shader 按顺序累计 F32，AuK CPU 使用八路 FMA 后归约。实机消去样例
  CPU 得到 `2.0`，GPU 得到 `1.125`。新增 `F16Dot` 模式复现 CPU 输入 F16 舍入、
  八路累加、四元素尾部和标量尾部；普通 F16 模式保持原计算。
- 旧进程级 F16 指针缓存不随模型释放。AuK 改用已有 `GpuLinear`，F16/Q8_0 上传缓存
  由 denoise scratch 持有；GPU 拒绝或失败时 CPU 重算全部输出。
- F16 缩放遵循 CPU 的“先缩放输入、F16 舍入、点积、逆缩放输出”。舍入可能溢出的输入
  保留 CPU 计算。CPU scope、`RMI_PARITY_TRACE`、scalar 模式均禁止投影 offload。
- Qwen 文本 trunk 的 prepared 投影进入旧单矩阵 Q8 dp4a shader。此前该 shader
  按 64 个 block 累加后树归约，CPU 按八路 FMA 累加；真实 2048 维 hidden 全部有
  位差，max abs `1.0031738`。旧 dp4a shader 现在也复现八路归约，实际 int8 dot 指令
  保留。首轮 Flash 波形 max abs `1.04155`、RMSE `0.23427`、cosine `0.51623`，
  未通过原门槛；这是修复前的失败证据，不能当作有效加速结果。
- Base Q8_0 首轮 GPU 在 32 步 CFG 2 下也失败：max abs `1.027614266`、RMSE
  `0.239731853`，23,819 个波形样本违反原门槛。逐投影诊断捕获 1536 维输入的
  第 11 个 block：`amax=7.987884998321533`（原始位 `0x40ff9cc1`）。CPU 的
  `amax/127` 为 `0x3d80d001`，GPU 乘倒数为 `0x3d80d000`；1 个 F32 ULP
  恰好跨越 F16 舍入边界，scale 分别为 `0x3d80e000` 和 `0x3d80c000`。
  Q8 shader 现在复用已有的 `divide_finite_normal` 计算 scale。现有量化检查
  增加这个独立 amax 样例和 scale 位比较，修复前稳定失败；临时逐投影诊断没有保留在源码中。

`vulkan_auk_projections_preserve_cpu_contract_and_cache_lifetime` 在实机通过：
F16 宽度 1024/1027/1031、65 输出，缩放 1/0.125；Q8_0 宽度 1024。
覆盖同一地址重新使用后的缓存重建、实际 GPU submission、原始 F32 位比较、trace 回退。
修改前 F16 位比较失败，修改后通过。共享普通 F16 的三行算子检查也通过。

真实权重检查确认 Qwen 2048 个 hidden 值、多 token 文本（8 token）的 16,384 个 hidden 值、
固定 conditioning 的四步 DiT 3200 个 latent 值均与 CPU 原始位一致，覆盖 Flash F16
的 CFG 0 和 Base Q8_0 的 CFG 2。四项 AuK ignored 回归通过，另用 Base Q8_0
重复真实组件回归通过。公共 GGUF loader 的 17 项检查通过；三份 DiT
各自的六项结构检查通过，覆盖 10 个 double + 20 个 single block 的真实维度。

默认、`vulkan`、`vulkan,parity-trace` 的 lib/CLI/server 编译检查和格式检查通过。
修改的 F16、旧 Q8 dp4a 和 Q8 量化 shader 均通过 SPIR-V validator，重编译字节与提交的 SPV 一致；
完整 manifest hash 匹配。
旧 Q8 单矩阵的五组 shape 检查，以及所有格式的三行共享算子检查通过，包含上述 scale 原始位回归。
既有全量 shader checker 的 `softmax.spv` 重编译字节差异，以及单行 F32 的旧零误差失败，
见上文**一、2026-10-07 多模型验证**的「算子、静态检查与已知失败」，本次不据此声称全量通过。

### 真实模型结果

固定 prompt `Hi`（1 token）、seed 42、8 线程、1 秒、24 kHz mono，配套 Qwen Q8_0、
F32 VAE。CPU/GPU 顺序执行；每份 GPU 第一轮包括投影权重上传，设备 warmup 在生成前完成。
系统文件缓存没有清空。Flash 第二次在同一进程复用文本模型，但 DiT scratch 重新上传。

| DiT / 调度 | CPU 秒 | GPU 秒 | submission / 次 | 波形比较 |
|---|---:|---:|---:|---|
| Flash F16，4 步，CFG 0 | 46.152569 | 34.225369 / 32.098509 | 25,308 | 两次的 24,000 个样本原始位一致 |
| Base F16，32 步，CFG 2 | 500.704428 | 283.727838 | 401,148 | 24,000 个样本原始位一致 |
| Base Q8_0，32 步，CFG 2 | 368.387956 | 269.289640 | 401,148 | 24,000 个样本原始位一致 |

Flash CPU text / DiT / VAE 为 3,425.3 / 30,100.6 / 12,626.6 ms；GPU 首次为
1,617.6 / 19,782.1 / 12,825.5 ms，第二次为 132.5 / 19,238.0 / 12,727.9 ms。
两次 GPU 的 max abs、RMSE、原门槛违规数均为 0；峰值 `0.950000048`，RMS `0.231008116`。
每次 25,056 个 DiT F16 dispatch，另有 252 个文本 Q8 投影 submission；GPU 保持健康。

Base F16 CPU text / DiT / VAE 为 3,372.9 / 484,171.0 / 13,160.4 ms；GPU 为
1,555.6 / 269,546.2 / 12,626.0 ms。整体约 1.76 倍；max abs、RMSE 和原门槛违规数均为 0，
峰值 `0.949999988`，RMS `0.241932970`。400,896 个 DiT F16 dispatch，另有 252 个文本 Q8 submission。

Base Q8_0 CPU text / DiT / VAE 为 3,326.1 / 352,192.1 / 12,869.7 ms；GPU 为
1,631.6 / 254,624.2 / 13,033.8 ms。整体约 1.37 倍；max abs、RMSE 和原门槛违规数均为 0，
峰值 `0.949999988`，RMS `0.243952514`。每次包含 397,568 个 Q8_0 DiT 投影
（各录制量化和 matmul 两个 dispatch）、3,328 个 DiT F16 dispatch，另有 252 个文本 Q8 submission。
三组 GPU 均保持健康，没有通过 CPU 回退来掩盖整次生成失败。

最终 CLI 另用 Flash F16、相同配套 Qwen/VAE 与参数生成 24 kHz mono WAV，
24,000 个 PCM16 样本全部与已验证的原始波形按 CLI 的转换规则一致。生成耗时
34,255.4 ms（含加载的 CLI 总耗时 34,764 ms）；这项检查证明 CLI 参数和原生文本权重入口走通。

### 复现与证据

远端在上述目录内使用仓库工具链和离线缓存：

```bash
export PATH="$PWD/target/toolchain/bin:$PATH"
export CARGO_HOME="$PWD/.cargo-home" CARGO_BUILD_JOBS=8 RUSTFLAGS=-Awarnings
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json
export RMI_AUK_NATIVE_TEXT="$PWD/models/audio-cpp--AuK-Base-and-Flash-GGUF/qwen2.5-omni-3b-q8_0.gguf"
export RMI_AUK_GGUF="$PWD/models/audio-cpp--AuK-Base-and-Flash-GGUF/auk-flash-f16.gguf"
cargo test --profile release-fast --locked --offline --features vulkan --lib \
  models::diffusion::auk -- --ignored --nocapture --test-threads=1
cargo test --profile release-fast --locked --offline --features vulkan \
  --test auk_di_t_q4_k_m -- --nocapture --test-threads=1
```

第二条结构检查分别设置 Flash F16、Base F16、Base Q8_0 的 `RMI_AUK_GGUF`。
计时 harness `target/validation/auk-check.rs` 通过当前库的 `AukPipeline` 调用生成，
校验实际 submission、GPU 健康状态、24 kHz mono/24,000 样本、有限且非静音的输出，
保存原始 F64 位供比较。其参数为：

```bash
target/validation/auk-check cpu flash-f16 4 target/validation/auk-flash-cpu
RUST_GPU_DISPATCH_TRACE=1 target/validation/auk-check gpu flash-f16 4 target/validation/auk-flash-gpu-fixed 2
target/validation/auk-check cpu base-f16 32 target/validation/auk-base-f16-cpu
RUST_GPU_DISPATCH_TRACE=1 target/validation/auk-check gpu base-f16 32 target/validation/auk-base-f16-gpu
target/validation/auk-check cpu base-q8_0 32 target/validation/auk-base-q8_0-cpu
RUST_GPU_DISPATCH_TRACE=1 target/validation/auk-check gpu base-q8_0 32 target/validation/auk-base-q8_0-gpu
```

CLI 冒烟命令：

```bash
target/release-fast/rust-model-inference --gpu \
  --model models/audio-cpp--AuK-Base-and-Flash-GGUF/auk-flash-f16.gguf \
  --text-encoder models/audio-cpp--AuK-Base-and-Flash-GGUF/qwen2.5-omni-3b-q8_0.gguf \
  --vae models/audio-cpp--AuK-Base-and-Flash-GGUF/auk-vae-f32.gguf \
  --prompt Hi --out target/validation/auk-flash-cli-final.wav \
  --steps 4 --cfg-scale 0 --seed 42 --resolution 24000 --duration-seconds 1 --threads 8
```

比较要求 shape 相同、所有值有限，记录原始位差、max abs、RMSE，原误差门槛为
`abs(gpu-cpu) <= 2e-3 + 2e-3*abs(cpu)`；修复后直接达到逐位一致，没有放宽门槛。

最终 721 份 Rust/Cargo/shader/测试文件和 harness 的本地、远端 SHA-256 全部一致，
清单摘要为 `ca706196949573f51e9237a4a4374363edf4f32ce99f07b50c2de64db2e87cf0`。
本地证据目录 `target/vulkan-auk-validation/` 保存小日志、比较 JSON、原始波形、PCM16 WAV、
harness 源码、`reproduce.sh`、配置和固定版本参考源码；不包含模型权重。
远端原始证据在 `target/validation/auk*`。

### 验证边界

DiT F16/Q8_0 与文本编码投影进入 GPU，attention、Euler/CFG 和 VAE 继续在 CPU。
AuK VAE 内部为 F32，最后转为 F64 波形；未降精度，也未以 F16/Q8 替换 VAE 权重。
F16Dot 对其他 CPU SIMD 后端保留 CPU 回退，本轮只验证上述 x86_64 + RADV 设备。

当前 CPU AuK 实现仍含 velocity 裁剪、简化的 VAE dilation/latent normalization、
峰值归一化等既有逻辑，尚未建立 audio.cpp/官方模型的完整数值 Oracle。
CPU/GPU 波形比较只证明此次 offload 的一致性，不证明语音质量或官方等价。
参考 WAV CFMEdit、独立 BF16 audio tower、长音频和其他 GPU 尚未验证。
配套 Qwen GGUF 包含 audio tower，但本次只调用文本 trunk；该 trunk 仍使用既有进程级 Q8
缓存，本次 DiT 缓存修复没有证明服务热换文本模型的安全性。Base 每组仅一次 CPU/GPU 运行，
这些数字是功能验证的单次计时，不是统计 benchmark。
前一轮五类模型的计时和误差记录对应 `65fb749`；本次公共 Q8 shader 修改后重跑了算子检查，
没有把旧整模型数字作为最终 AuK 提交上的新测量。

## 三、YuE2 Vulkan BF16 投影优化（2026-10-08）

在 `9189604`（已合入 main `ce17cd5`）上优化共享 BF16 dot shader，
没有修改 YuE2 的采样、attention、扩散步数或 CPU VAE。RADV 上 256 帧 NAR
velocity 的重复调用从 33.600249 降到 12.419976 秒，约 2.71 倍，减少 63.0%；
32 帧从 3.555778 降到 0.720680 秒。所有测量输出与 CPU 原始位一致。
AR 耗时基本不变；256 帧 NAR 仍略慢于四线程 CPU，不能据此承诺 GPU 全链更快。

### 实现与数值合约

旧 BF16 dot shader 的 64 线程组只使用 4/8 个线程计算一个输出。
现在线程组同时计算 16 个 NEON 输出或 8 个 AVX2 输出，填满组内线程。
输入至少四帧时，每次展开一个 BF16 权重后服务四帧；不足四帧的最后一组屏蔽无效帧。
主机同步减少 X/Y 输出组数和 Z token 组数，三组投影的 slot 布局保持不变。

每帧仍分别保留 CPU 的四/八条 FMA 流、原水平归约和非 FMA 标量尾部，
bias 后才舍入 BF16。无效输出行也参与 barrier，避免末组死锁。
共享存储为 64 个 `vec4`（1 KiB）。单 token、普通 BF16 和 scalar 模式不做四帧批处理。
这是投影级优化；GPU 提交次数、CPU attention/state/采样和 VAE 路径没有改变。

### 实机与输入

- AMD EPYC 9334，容器 16 CPU / 55 GiB RAM；ComputePool 与 Rayon 均为四线程。
- RADV NAVI31，48 GiB VRAM；Mesa 25.2.8、Vulkan 1.4.318、Rust 1.98.1。
- `release-fast`、`--features vulkan`；工具链和 Cargo 缓存位于仓库内，离线构建。
- 主模型：`models/YuE2-3B/yue2-bf16.gguf`，7,268,395,360 bytes，
  SHA-256 `e2d202b4d16431159338337b6b2ced185400224d9d77eeef770c013efeb12889`。
- VAE：`models/YuE2-Vae/yue2-vae-f32.gguf`，265,515,360 bytes，
  SHA-256 `b6f8c4da9e5281546d3e76787ba6d75d2163cd8611a0a9fd31d6510df6a3bb8b`。

沿用远端已校验的权重，不再下载；只通过 tar/SSH 流式同步小文件。
五个修改的运行时源文件、shader 和 manifest 已核对本地/远端 SHA-256。
每组 CPU/GPU 测量串行运行，GPU 提交和健康状态均显式检查。
这些是固定输入的单次功能计时，不是统计基准；“首次”指该 shape 的第一次调用，
32 帧和 256 帧在同一进程中依次测试，不能将所有首次调用视作全冷启动。

### AR 与 NAR 计时

AR 使用 56-token 前缀、32 次 greedy decode、F32 KV、capacity 256。
NAR 使用 prefix `[55,25,16,198]`、codec IDs `0..frames`、seed 12300，
由 `song_chunks` 创建 noise；velocity 输入为该 noise，时间参数为 0.875。
32/256 帧分别有 37/261 个 AR prefix tokens。

单位：秒。中间版只打包输出行；最终版同时共享四帧权重读取。

| 测量 | CPU | 原 GPU | 仅打包输出行 | 最终 GPU |
|---|---:|---:|---:|---:|
| AR，冷 / 预热 | 6.121840 | 15.509254 / 14.139167 | 15.454580 / 13.856802 | 15.763981 / 14.121409 |
| NAR 32，prefix | 0.793243 | 5.412525 | 4.175989 | 2.105068 |
| NAR 32，velocity 首次 / 重复 | 0.937612 / 0.779209 | 5.483484 / 3.555778 | 4.592119 / 2.398540 | 2.417971 / 0.720680 |
| NAR 256，prefix | 8.869344 | 31.726256 | 24.868826 | 10.269740 |
| NAR 256，velocity 首次 / 重复 | 11.710974 / 11.788918 | 35.230863 / 33.600249 | 28.422495 / 26.824717 | 14.072444 / 12.419976 |

最终 AR 同次 CPU 计时为 6.278603 秒；冷/预热各比较 369,408 个首尾 logits 值，
原始位差异为 0，32 个 greedy IDs 一致，每次有 17,281 次 GPU submissions。
NAR 两次 velocity 都直接逐字节比较优化前保存的 CPU F32 参考：
32 帧 2,048 个值、256 帧 16,384 个值，原始位差异为 0。
prefix submissions 为 196/980，velocity 为 200/992；优化没有以减少投影为代价。

### 完整生成

style 为 `Acoustic pop, gentle female vocal, guitar`，lyrics 为：

```text
[Verse]
The morning light is shining bright
We walk together into the light
```

seed 12300，ABC 上限 1024、semantic 上限 256、NAR 4 步，BF16 主模型 / F32 VAE。
最终 GPU 生成耗时 **608.556388 秒**，实际 201,163 次 submissions，GPU 健康。
302 个 ABC IDs、256 个 semantic IDs 和维度 JSON 与保存的 CPU 参考逐字节一致；
256 帧的 16,384 个 latent F32、982,912 个波形 F32 全部原始位一致。
输出为 48 kHz 双声道、每声道 491,456 samples，约 10.239 秒。

此前 `65fb749` 的同输入实机记录为 CPU 516.568676 秒、GPU 852.855718 秒，
见上文**一、2026-10-07 多模型验证**的「YuE2」小节。
当前 GPU 比该旧 GPU 记录少 28.6%，但仍比该 CPU 记录慢约 17.8%。
这不是在 `9189604` 上重新跑出的完整生成基线；本次同提交重建的性能对比是上表 AR/NAR。
完整生成复用了原 CPU 输出参考，VAE 仍在 CPU 执行，没有改进成品质量或 VAE 性能的结论。

### 回归与边界

- 调度单测覆盖普通 BF16 / BF16 dot、3/5 token rows、三组 65/28/8 输出，以及设备 X/Y/Z 限制。
  第二轮实现前的预期失败为 `[8,2,15] != [8,2,6]`，修改后实机通过。
- 实机 ignored 回归覆盖 bias 后舍入、65/69 输入帧、65 个输出、1024/1027 宽度，
  包括 64+5 帧的四帧尾组屏蔽、输出组尾部和标量 tail；最终版本通过。
- 本地 YuE2：46 passed、8 ignored；实机显式运行上述 ignored GPU 回归通过。
- `--features vulkan --lib` 全库：1157 passed、34 failed、119 ignored。
  隔离未修改 `9189604` 快照：1156 passed、同样 34 failed、119 ignored，失败名称集合一致。
  失败涉及既有模型 fixture、NEON 数值断言、argmax 与本地 Vulkan 驱动，未放宽断言。
- 修改 shader 通过 `spirv-val --target-env vulkan1.1` 与重编译逐字节比较，完整 manifest 校验通过。
  之前全量 shader checker 的未修改 `softmax.spv` 重编译差异仍单独记录，未声明全量 shader 检查通过。
- default、`vulkan`、`vulkan,parity-trace` 的 lib/CLI/server `cargo check` 通过。
- dot 测试 9/9，`RMI_SCALAR=1` + `vulkan,parity-trace` 的 BF16 批处理逐位回归通过。
- `cargo fmt --all -- --check` 与 `git diff --check` 通过。
- 实机共享 BF16 算子检查 `vk_ops_check --formats bf16 --rows 3/5` 均通过现有门槛，
  普通 BF16 路径与 BF16 dot 同一 shader 的变更已一起检查。

其他设备、NEON 四流的实机 Vulkan 执行、长歌曲和更高扩散步数尚未验证。
CPU/GPU 固定输出一致性不构成新的官方模型质量评估；VAE 没有加速。

### 复现入口与证据

远端 `/workspace/rmi-vulkan-eccce00/target/validation/` 保存：

- `model-check.rs`、`yue-ar-perf-before.log`、`yue-ar-perf-after.log`、`yue-ar-perf-final.log`。
- `yue-nar-perf.rs`、`yue-nar-perf-{cpu,gpu}-before.log`、
  `yue-nar-perf-gpu-after.log`、`yue-nar-perf-gpu-final.log`、CPU 原始 F32 参考。
- `yue-check.rs`、`yue-e2e-perf-final.log`、`yue-final-device-status.json`，
  `yue-gpu-optimized.{json,latents.f32,audio.f32}` 与 `yue-cpu.*` 原始参考。
- `yue-output-optimized-comparison.json`：输出 SHA-256、原始位对比、波形统计与源文件 SHA-256。
- `yue-four-token-static-red.log`、`yue-four-token-device-build.log`、
  `yue-final-device-regression.log`、`yue-final-bf16-operators-{3,5}.log`。

可重复运行的仓库回归：

```bash
cargo test --profile release-fast --locked --features vulkan --lib \
  bf16_dot_dispatch_packs_independent_output_rows
cargo test --profile release-fast --locked --features vulkan --lib \
  vulkan_yue2_bf16_rounds_after_bias_across_tiles -- --ignored --nocapture --test-threads=1
cargo test --profile release-fast --locked --features vulkan --lib yue2
```

在上述远端目录，使用当前源码重建库并重链接 helper 后运行固定输入：

```bash
export PATH="$PWD/target/toolchain/bin:$PATH" CARGO_HOME="$PWD/.cargo-home"
export VK_DRIVER_FILES=/usr/share/vulkan/icd.d/radeon_icd.json RAYON_NUM_THREADS=4
RUSTFLAGS=-Awarnings cargo build --profile release-fast --locked --offline --features vulkan --lib
RMI_YUE_LIB=$(python3 -c 'from pathlib import Path; p=list(Path("target/release-fast/deps").glob("librust_model_inference-*.rlib")); assert len(p)==1; print(p[0])')
rustc --edition=2021 -C opt-level=3 target/validation/yue-nar-perf.rs \
  --extern "rust_model_inference=$RMI_YUE_LIB" -L dependency=target/release-fast/deps \
  -o target/validation/yue-nar-final
target/validation/yue-nar-final cpu target/validation/yue-nar-perf-cpu-recheck
target/validation/yue-nar-final gpu target/validation/yue-nar-perf-gpu-recheck
```

本地静态检查、测试及失败集合比对保存在 `target/vulkan-radv-validation/yue-final-*`，
模型和临时验证产物不提交到 Git。
