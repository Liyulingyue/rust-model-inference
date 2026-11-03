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

- `cargo test --locked --profile release-fast --lib arktts`：7 passed。
- `cargo test --locked --profile release-fast --lib audio_speech`：1 passed。
- Python converter/comparator 单测：5 passed。
- `cargo build --locked --profile release-fast --features parity-trace --bin rust-model-inference --bin rust-model-server`：通过。
- `cargo check --locked --profile release-fast --features vulkan --lib --bin rust-model-inference --bin rust-model-server`：通过；仅证明 feature 编译，不代表 GPU 运行或对齐。
- 真实权重集成测试：默认 SIMD 完整三组和标量短三组均通过。
- `git diff --check`：通过。
- 全库 lib 回归：1338 passed / 20 failed / 88 ignored；变更前为 1330 / 20 / 88，20 个失败名称完全相同，无新增失败。

既有失败位于：Breeze 的 3 个 dtype 契约用例、CLM 的 2 个 golden/embedding 用例、
Z-Image 的 3 个 NEON 逐位用例、FunASR 的 2 个 NEON 逐位用例、Gemma4 的 4 个 trunk 用例、
argmax 的 5 个用例和 matmul 的 1 个 NEON softmax 用例。本次未修改这些计算行为；
Breeze 仅开放已有 causal convolution 的 crate 内可见性。

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

## 未覆盖范围

仅验证上述官方发布版本及 BF16 主模型 / F32 codec。未实现 Q4/Q8 等量化导出和已校准的 Vulkan 路径；
未验证 Vulkan 设备运行/数值及完整发布构建、其他 CPU 架构的真实模型推理、官方原生 BF16 激活、2048 位置上限、跨 codec 滑动窗口的长音频、
全部 23 种语言、主观音质、声音相似度指标、完整多模态文本后处理桥或性能基准。参考 WAV 重采样沿用现有 sinc 实现，未声称与官方前端一致。
采样 RNG 与 PyTorch 不同；随机生成的 seed 不保证跨实现复现。
