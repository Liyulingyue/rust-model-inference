# YuE2 Vulkan BF16 投影优化（2026-10-08）

在 `9189604`（已合入 main `ce17cd5`）上优化共享 BF16 dot shader，
没有修改 YuE2 的采样、attention、扩散步数或 CPU VAE。RADV 上 256 帧 NAR
velocity 的重复调用从 33.600249 降到 12.419976 秒，约 2.71 倍，减少 63.0%；
32 帧从 3.555778 降到 0.720680 秒。所有测量输出与 CPU 原始位一致。
AR 耗时基本不变；256 帧 NAR 仍略慢于四线程 CPU，不能据此承诺 GPU 全链更快。

## 实现与数值合约

旧 BF16 dot shader 的 64 线程组只使用 4/8 个线程计算一个输出。
现在线程组同时计算 16 个 NEON 输出或 8 个 AVX2 输出，填满组内线程。
输入至少四帧时，每次展开一个 BF16 权重后服务四帧；不足四帧的最后一组屏蔽无效帧。
主机同步减少 X/Y 输出组数和 Z token 组数，三组投影的 slot 布局保持不变。

每帧仍分别保留 CPU 的四/八条 FMA 流、原水平归约和非 FMA 标量尾部，
bias 后才舍入 BF16。无效输出行也参与 barrier，避免末组死锁。
共享存储为 64 个 `vec4`（1 KiB）。单 token、普通 BF16 和 scalar 模式不做四帧批处理。
这是投影级优化；GPU 提交次数、CPU attention/state/采样和 VAE 路径没有改变。

## 实机与输入

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

## AR 与 NAR 计时

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

## 完整生成

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
见 [原验证记录](VULKAN_RADV_VALIDATION_2026-10-07.md#yue2)。
当前 GPU 比该旧 GPU 记录少 28.6%，但仍比该 CPU 记录慢约 17.8%。
这不是在 `9189604` 上重新跑出的完整生成基线；本次同提交重建的性能对比是上表 AR/NAR。
完整生成复用了原 CPU 输出参考，VAE 仍在 CPU 执行，没有改进成品质量或 VAE 性能的结论。

## 回归与边界

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

## 复现入口与证据

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
