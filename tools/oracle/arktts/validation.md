# Audio8-TTS-Realistic-Air 验证记录

2026-11-04，macOS 27.0.1 arm64，CPU 4 线程，`release-fast`。
官方 ModelScope 仓库：`Edge0/Audio8-TTS-Realistic-Air`，记录的 HEAD 为
`2af90384f7c0d38070f688483dd5344265601e2b`。
模型目录：`/Users/gouzi/Documents/git/rust-model-inference/models/Audio8-TTS-Realistic-Air`。
Python 3.12.13，torch 2.9.1，transformers 4.57.3，numpy 2.5.4，safetensors 0.8.0，soundfile 0.14.0。
Rust 1.99.1 工具链；原始 BF16 权重提升为 F32 计算，对照官方 CPU FP32 路径。

## 文件身份

| 文件 | 字节数 | SHA-256 |
|---|---:|---|
| config.json | 1692 | `a6a2189cb6938b117ad388bad1e4ad786df416170046a0082f07fad28438de0c` |
| model.safetensors | 1202342528 | `f590053e980a45ebcbb2e787cff4f2dead932b2b61e5cf54d05664d9764e12d8` |
| codec.pth | 1349857559 | `c310505aa11fe2f6cc63b8d3130dc7e77e73227774f5c62575769b1f47a8d048` |
| tokenizer.json | 12217872 | `f24e08099d45a8adf3f52f5f0b03276e433bb9d689bb15fcbcc48ce58744588b` |
| configuration_arktts.py | 4554 | `27fa4fd4073ef1983a48919a4562300cace84687335035a9b9a077966e1ac04e` |
| modeling_arktts.py | 34737 | `b320b6c875e6a57afb9745cc98fc72c116d9df6b3bf20d092a5ad3095026911f` |
| modeling_arktts_codec.py | 19838 | `e1d6b0e1ef1e9ad303dc01d45d86229cf7f2e98d06a7b426234a7e4d7e3ede68` |
| processing_arktts.py | 11185 | `eba6ecb6b7f0e221bcd19c5e3fe5978e677faf6b362912ff4febc08c3dc7f9df` |
| Audio8-TTS-Realistic-Air-BF16.gguf | 1214553760 | `d92fed3ab6875103ca153109ed11bc0ecb224670d3b680c4873978c9c3dc6d2b` |
| Audio8-TTS-Realistic-Air-codec-F32.gguf | 1349626816 | `e26915dcec4b11390965404fe5b9fd15eafd2ba4a94b82261c29e6d86a901198` |

转换器检查 226 个主模型 BF16 payload 与原 safetensors 字节一致、455 个 F32 codec payload
与 PyTorch 展开后的权重一致。运行时检查每个所需张量的 shape、类型、字节长度，以及组件 bundle ID。

## 数值和入口结果

`official.py --frames 96` 直接加载上述官方代码和原始权重。hooks 只收集输出，不改写算子或权重。
官方在 EOS 上计算后丢弃的 FastAR 输出不参与 trace；Rust 在 EOS 上提前结束该部分计算。
原生实现保留官方 FP32 CPU 路径使用的 BF16 RoPE 系数表语义。

| 验证 | 结果 |
|---|---|
| 英文 `Hello world.` | 23 个 prompt columns；22 帧完整 greedy 到 EOS，所有 220 个 codec IDs 一致 |
| 中文 / Unicode / 组合字符 / 日文字符 / 空白 | 30 个 prompt columns；68 帧到 EOS，所有 680 个 IDs 一致 |
| 参考音频 codes + transcript | 49 个 prompt columns；40 帧到 EOS，所有 400 个 IDs 一致 |
| 参考编码 | 4097 个 44100 Hz 混合正弦采样点，3 帧 × 10 个离散码完全一致 |
| 默认 ARM64 CPU SIMD | 11,161 个浮点检查点、38,236,288 个值，完整顺序/形状与容差检查通过 |
| `RMI_SCALAR=1` | 三组各 4 帧，1,053 个浮点检查点通过，prompt/码流/参考编码一致 |
| CLI 普通生成 | 22 帧、45056 个采样点、44100 Hz 单声道 PCM16 WAV，有限且峰值 0.78052 |
| CLI 克隆 | 以英文生成 WAV 和原文作参考，中文输出 39 帧、79872 个采样点，有限且峰值 0.78391 |
| CPU / Auto | `Hello world.` 的 WAV 相同；Auto 明确报告 CPU fallback |
| HTTP | WAV/PCM/参考克隆成功；普通和克隆 WAV 与 CLI 一致，PCM 为 WAV 的裸 PCM16 payload |
| HTTP 取消 / 并发 | 关闭正在推理的连接后，新生成请求返回 429，原后台任务结束后恢复成功 |
| HTTP 文件 / 重采样边界 | 普通文件参考成功；设备、目录和超 16 MiB 文件为 400；44101 Hz WAV 为 422 |
| 拒绝入口 | CLI/server 的文本与 JEV 启动明确拒绝；TTS server 的聊天/JEV 路由为 404 |
| 拒绝参数 | 错误 codec、零 top_k、缺参考 transcript、指令、不支持的调节参数、已有输出均拒绝；HTTP format/speed/参考错误返回 400/422，未知字段为 422 |

默认 SIMD 完整 trace 的误差：

| 指标 | 观测 |
|---|---:|
| 最大 relative RMSE | `6.71361e-5`（Fast logits） |
| 最大 normalized max_abs | `2.31540e-5` |
| 最小 cosine | `0.999999997758` |
| Slow logits 最大绝对误差 | `6.94275e-4` |
| Fast logits 最大绝对误差 | `6.99520e-4` |
| 最大中间层绝对误差 | `0.0764160`（reference prompt Slow layer 23，参考峰值约 11775；relative RMSE `6.72542e-6`） |
| 固定 3 帧解码波形 max_abs / RMSE | `8.81031e-7` / `1.39681e-7`，6144 个采样点 |
| 标量固定波形 max_abs / RMSE | `8.75443e-7` / `1.46365e-7` |
| 完整英文 PCM16 WAV 对官方完整浮点解码 max_abs / RMSE | `5.36442e-5` / `2.08817e-5`，含 PCM16 量化误差；检查门槛 `1e-4` / `5e-5` |

浮点门槛和复现步骤见 [README](README.md#官方对齐)。整数 token/码流精确比较；有损浮点算子只使用数值容差。
固定解码输入还包含超出 residual codebook 的 ID，覆盖官方 clamp 到 1023 的行为。
标量路径不保证所有现有 SIMD 算子都关闭，仅声明显式 `RMI_SCALAR=1` 请求下的结果。

## 自动检查与自查

- `cargo test --locked --profile release-fast --features vulkan,parity-trace --lib arktts`：8 passed，2 ignored。
- `cargo test --locked --profile release-fast --lib audio_speech`：1 passed。
- Python same-weight/converter/comparator 单测：9 passed。
- `cargo build --locked --profile release-fast --features parity-trace --bin rust-model-inference --bin rust-model-server`：通过。
- `cargo check --locked --profile release-fast --features vulkan --lib --bin rust-model-inference --bin rust-model-server`：通过；真实 GPU 运行另由下方 Vulkan 矩阵和 4308 次 dense device calls 证明。
- 真实权重集成测试：原始 BF16/F32 CPU 三组短历史通过；九档 held-out CPU/Auto/Vulkan 全部通过。
- `git diff --check`：通过。
- 全库 lib 回归（不启用 Vulkan feature）：1553 passed / 13 failed / 100 ignored；当次基线为 1552 / 14 / 99，失败名单仍是原有平台/合同用例，没有新增 ArkTTS 失败。

既有失败位于：Breeze 的 3 个 dtype 契约用例、CLM 的 2 个 golden/embedding 用例、
Z-Image 的 3 个 NEON 逐位用例、FunASR 的 2 个 NEON 逐位用例、Gemma4 的 1 个 trunk 用例、
app media 的 1 个平台用例和 matmul 的 1 个 NEON softmax 用例。本次未修改这些计算行为。

PR #200 的初次 CI：`x86 dispatch regressions` 通过；Rustfmt 与 Vulkan job 失败。
后者在 `bash scripts/vulkan-shaders.sh check` 的 8 个 GLSL/SPIR-V checksum 上失败，尚未进入 Rust 编译。
已在未修改的 `afed6d48b` 临时快照与本分支分别执行 `cargo fmt --check` 和 shader check：
shader 输出逐字节相同，失败文件为 q4_k/q5_k/q6_k/f16 matmul 的源码与二进制。
本次新增的 module 顺序和 backend probe 换行问题已修正；余下 32 个 Rustfmt diff blocks 均出现在未修改基线中。
未改写无关模型/算子的既有格式或 shader 校验和；上述两项 CI 仍受基线问题影响。

自查已修复：

1. P1，`src/app/server/audio_speech.rs:168`：生成锁移入 blocking worker，防止 HTTP 取消后旧推理继续而新请求并发加载 codec；以连接取消后的 429 验证。
2. P1，`src/app/tts/mod.rs:230`、`src/app/server/audio_speech.rs:133`：在 sinc resampler 分配前验证采样率和输出长度，HTTP 文件参考限制为最多 16 MiB 的普通文件；单测与真实 HTTP 边界请求通过。
3. P1，`src/models/arktts/mod.rs:139`：main/codec bundle ID 校验，防止混配不同 checkpoint；精确识别架构，不按相近 Qwen 模型路由；混配单测通过。
4. P2，`src/app/tts/mod.rs:214`、`src/main.rs:592`：对不支持的 CLI/HTTP 控制和文本/JEV 模式明确报错；修正文本后处理桥的固定 Qwen3/24 kHz 描述；两种二进制的拒绝入口已实测。
5. P2，`src/models/arktts/mod.rs:461`：文本清洗补齐 Python 视为空白的 U+001C–U+001F；温度下限与官方 `1e-5` 一致；单测与 Oracle 通过。

## GPU 与多精度复验（2026-11-11）

上面的 2026-11-04 记录保留为原始 BF16/F32 基线；它当时写着「未实现量化和 Vulkan」，
随着本次 PR 的 dense executor 接入已被下面实测结果替代。模型文件和逐项 JSON 证据在
`/Users/gouzi/Documents/git/rust-model-inference/models/Audio8-TTS-Realistic-Air/multiq/`。

| 格式 | CPU / Auto / Vulkan | 最差 relative RMSE | 固定码流 waveform max_abs | Vulkan device calls |
|---|---|---:|---:|---:|
| F32 | 通过 / 通过 / 通过 | 3.27e-5 | 1.58e-6 | 4308 |
| F16 | 通过 / 通过 / 通过 | 5.00e-3 | 2.06e-4 | 4308 |
| BF16 | 通过 / 通过 / 通过 | 2.97e-5 | 1.46e-6 | 4308 |
| Q8_0 | 通过 / 通过 / 通过 | 5.85e-2 | 1.50e-3 | 4308 |
| Q4_0 | 通过 / 通过 / 通过 | 7.87e-2 | 5.99e-3 | 4308 |
| Q4_1 | 通过 / 通过 / 通过 | 8.69e-2 | 2.93e-3 | 4308 |
| Q4_K | 通过 / 通过 / 通过 | 5.85e-2 | 1.55e-6 | 4308 |
| Q5_K | 通过 / 通过 / 通过 | 6.28e-2 | 1.35e-6 | 4308 |
| Q6_K | 通过 / 通过 / 通过 | 6.60e-2 | 8.93e-4 | 4308 |

held-out suite 前冻结了分精度门槛：量化为 relative ≤ 0.2、normalized ≤ 0.15、logits
max_abs ≤ 5、waveform max_abs < 0.006/RMSE < 0.001；F16 为 0.01/0.01/0.25/
0.001/0.0001；F32/BF16 使用上面的浮点门槛。量化和半精度不做逐位验收：激活量化
bin 的边界舍入会放大为中间层差异，固定 greedy history 用于把实现误差与离散码流分叉分开。

实现是混合 Vulkan 路径：所有二维 dense projection 有真实 device submission；norm、RoPE、
attention/KV、卷积、codebook、采样仍在主机。每个 Auto 组合记录 1188 次 device dense
calls + 7548 次 host linear calls；严格 Vulkan 记录 8736 次 device calls、0 次 host
linear calls。CPU/Auto/严格 Vulkan 的每个格式都分别通过 CLI 普通、CLI 克隆、HTTP WAV 和
HTTP PCM，HTTP 字节与 CLI 字节相等；对应入口证据在 `multiq/entries-v2-results.json`。

## 仍未覆盖

未验证完整发布构建的性能峰值、其他 CPU 架构的真实模型推理、官方原生 BF16 激活语义、
2048 位置上限与跨 codec 滑动窗口的长音频、全部 23 种语言、主观音质/声音相似度指标，
以及完整多模态文本后处理桥。参考 WAV 重采样沿用现有 sinc 实现，未声称和官方前端一致；
采样 RNG 与 PyTorch 不同，固定 seed 只保证 Rust 自身复现。
