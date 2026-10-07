# AuK Vulkan 实机验证：RADV NAVI31，2026-10-08

延续 `codex/vulkan-multi-model`，基础提交 `65fb749`，main `50fd20e` 已合入。
本次验证原生 Rust 的 AuK TTS 投影 offload；模型支持等级仍为 **Experimental**。

## 环境和权重

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

## 修复与算子检查

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
见 [前一轮记录](VULKAN_RADV_VALIDATION_2026-10-07.md)，本次不据此声称全量通过。

## 真实模型结果

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

## 复现与证据

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

## 验证边界

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
