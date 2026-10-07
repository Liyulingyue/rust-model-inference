# Vulkan 实机验证：RADV NAVI31，2026-10-07

验证分支 `codex/vulkan-multi-model`，基础提交 `eccce00`，已合入 main `50fd20e`。
下列结果包含本次实机发现的修复，执行的是原生 Rust 推理。

五类路径均完成真实 GPU 执行。Embedding、dense LFM 和 YuE2 固定样例逐位一致；
Edge/LFM MoE 通过原有数值门槛且 greedy IDs 一致。Z-Image 成图成功，但像素有差异，
没有建立整条图像管线的浮点 parity。GPU 不保证每个模型都更快，YuE2 样例仍较慢。

## 环境与证据

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

## 数值与执行结果

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

## 图像与音频

### Z-Image Turbo

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

### YuE2

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

## 实机发现的修复

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

## 算子、静态检查与已知失败

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

## 复现

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

## 权重身份

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
