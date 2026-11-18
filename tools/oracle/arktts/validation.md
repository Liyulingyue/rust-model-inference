# Audio8-TTS-Realistic-Air 验证记录

本文件保留各阶段历史证据；当前设备驻留实现与最终矩阵见下方「设备驻留数值图」记录。
历史 dense-only 表格中的主机 attention/codec 分工已被替代，不代表当前 GPU 支持范围。

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
- Python same-weight/converter/comparator 单测：10 passed（含后续 Q4_1 修复用例，ARM64/x86_64 均通过）。
- `cargo build --locked --profile release-fast --features parity-trace --bin rust-model-inference --bin rust-model-server`：通过。
- `cargo check --locked --profile release-fast --features vulkan --lib --bin rust-model-inference --bin rust-model-server`：通过；真实 GPU 运行另由下方 Vulkan 矩阵和 4308 次 dense device calls 证明。
- 真实权重集成测试：原始 BF16/F32 CPU 三组短历史通过；九档 held-out CPU/Auto/Vulkan 全部通过。
- `git diff --check`：通过。
- 全库 lib 回归（不启用 Vulkan feature）：1553 passed / 13 failed / 100 ignored；未修改的 `upstream/main@d16dc16d9` 当次基线为 1544 / 13 / 99，13 个失败名称完全一致。此前 1552 / 14 / 99 的记录包含新增用例的 red 阶段，不能当作未修改主线基线，现予以更正。

既有失败位于：Breeze 的 3 个 dtype 契约用例、CLM 的 2 个 golden/embedding 用例、
Z-Image 的 3 个 NEON 逐位用例、FunASR 的 2 个 NEON 逐位用例、Gemma4 的 1 个 trunk 用例、
app media 的 1 个平台用例和 matmul 的 1 个 NEON softmax 用例。本次未修改这些计算行为。

PR #200 的初次 CI：`x86 dispatch regressions` 通过；Rustfmt 与 Vulkan job 失败。
后者在 `bash scripts/vulkan-shaders.sh check` 的 8 个 GLSL/SPIR-V checksum 上失败，尚未进入 Rust 编译。
已在未修改的 `afed6d48b` 临时快照与本分支分别执行 `cargo fmt --check` 和 shader check：
shader 输出逐字节相同，失败文件为 q4_k/q5_k/q6_k/f16 matmul 的源码与二进制。
本次新增的 module 顺序和 backend probe 换行问题已修正；余下 32 个 Rustfmt diff blocks 均出现在未修改基线中。
未改写无关模型/算子的既有格式或 shader 校验和；当时这两项 CI 受基线问题影响，最新状态见下方合并记录。

2026-11-11 合并 `main@d16dc16d9` 后复验：`cargo fmt --all --check` 已通过，原来的格式
失败结论不再适用于当前树。`scripts/vulkan-shaders.sh check` 的所有 checksum 检查通过，
随后本机重编译 `nemotron_moe_route.spv` 与提交产物在 char 13 不同；未修改主线出现同样结果。
这与 2026-11-04 的八个 checksum 失败不同，本次没有改动任何 GLSL/SPIR-V。

随后合并 `main@f553db2a4d88ea846c0c7a9f6438a96bfeb3169a`（合并提交 `154ec0c33`），
PR 的 Rustfmt、x86 dispatch regressions、Vulkan compile and shaders 三项 CI 全部通过；
本机 ArkTTS 8 项、System One CLI 7 项复验通过。上面的全库失败名单比较针对 `d16dc16d9`，
没有把它当成 `f553db2a4` 的全库复验。

自查已修复：

1. P1，`src/app/server/audio_speech.rs:168`：生成锁移入 blocking worker，防止 HTTP 取消后旧推理继续而新请求并发加载 codec；以连接取消后的 429 验证。
2. P1，`src/app/tts/mod.rs:230`、`src/app/server/audio_speech.rs:133`：在 sinc resampler 分配前验证采样率和输出长度，HTTP 文件参考限制为最多 16 MiB 的普通文件；单测与真实 HTTP 边界请求通过。
3. P1，`src/models/arktts/mod.rs:139`：main/codec bundle ID 校验，防止混配不同 checkpoint；精确识别架构，不按相近 Qwen 模型路由；混配单测通过。
4. P2，`src/app/tts/mod.rs:214`、`src/main.rs:592`：对不支持的 CLI/HTTP 控制和文本/JEV 模式明确报错；修正文本后处理桥的固定 Qwen3/24 kHz 描述；两种二进制的拒绝入口已实测。
5. P2，`src/models/arktts/mod.rs:461`：文本清洗补齐 Python 视为空白的 U+001C–U+001F；温度下限与官方 `1e-5` 一致；单测与 Oracle 通过。
6. P1，发布树校验：之前 API 发布遗漏主线删除的两份旧兼容文件；合并已发布分支后保留本地正确树，发布前比较树 SHA，并保留 main 的真实合并祖先。
7. P2，Oracle 根据 GGUF 实际张量类型选择波形门槛；入口证据对严格 Vulkan 额外断言 host linear 为零，测试的拒绝入口使用所选主模型文件，避免依赖另一档 BF16 文件。
8. P2，性能分开记录含加载的请求观察与同口径 generate/codec 计时；新增自由 greedy 分叉、量化相对原精度的损失记录，并补完整矩阵复现命令。
9. P1，x86 Q4_1 codec 在冻结波形门槛失败；定位到解码 Transformer 的 activation QDQ 分叉，
   仅保留 `quantizer.post_module.*` 的 F32 源精度，新增行为用例并在两台机器完成三后端数值和入口复验，详见下方 Linux 记录。

## GPU 与多精度复验（2026-11-11，历史 dense-only 基线）

上面的 2026-11-04 记录保留为原始 BF16/F32 基线；它当时写着「未实现量化和 Vulkan」，
随着本次 PR 的 dense executor 接入已被下面实测结果替代。模型文件和逐项 JSON 证据在
`/Users/gouzi/Documents/git/rust-model-inference/models/Audio8-TTS-Realistic-Air/multiq/`。
本节指标来自 M3 Max；Q4_1 已改用 `multiq-reviewed/q4_1/` 的修订导出和复验结果，
原 `multiq/q4_1/` 仅保留历史证据。Linux 结果及两平台文件差异见下方独立记录。

| 格式 | CPU / Auto / Vulkan | 最差 relative RMSE | 固定码流 waveform max_abs | Vulkan device calls |
|---|---|---:|---:|---:|
| F32 | 通过 / 通过 / 通过 | 3.27e-5 | 1.58e-6 | 4308 |
| F16 | 通过 / 通过 / 通过 | 5.00e-3 | 2.06e-4 | 4308 |
| BF16 | 通过 / 通过 / 通过 | 2.97e-5 | 1.46e-6 | 4308 |
| Q8_0 | 通过 / 通过 / 通过 | 5.85e-2 | 1.50e-3 | 4308 |
| Q4_0 | 通过 / 通过 / 通过 | 7.87e-2 | 5.99e-3 | 4308 |
| Q4_1 | 通过 / 通过 / 通过 | 8.69e-2 | 1.58e-6 | 4308 |
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

### 自由生成与性能复验

固定历史验收通过不代表自由码流相等。新增不带 trace I/O 的自由 greedy 检查：每档三个文本、各 4 帧，
共 120 个 ID；原始 ID/首个分叉及逐阶段计时保存在 `multiq/review-20261111/`。

| 格式 | CPU / Auto / Vulkan 与同权重 Oracle 相同的 ID 数（各 120 个） |
| --- | --- |
| F32 | 120 / 120 / 120 |
| BF16 | 120 / 120 / 120 |
| F16 | 119 / 119 / 119 |
| Q8_0 | 62 / 62 / 62 |
| Q4_0 | 31 / 31 / 31 |
| Q4_1 | 87 / 87 / 87 |
| Q4_K | 77 / 67 / 47 |
| Q5_K | 72 / 72 / 70 |
| Q6_K | 95 / 87 / 64 |

F16 和块量化在多步 greedy 中出现离散分叉，K 格式在后端间也可能分叉。独立同输入 projection 的
误差满足 5e-5 relative RMSE 门槛，固定历史中间层误差另按上面的分精度门槛验收；
归约/RMS/RoPE 的舍入可跨越 activation QDQ 的 bin 边界，之后自回归历史会放大差异。
这不能替代主观音质、说话人相似度或长音频任务质量评估；F16/块量化保留 Experimental 标记。

性能 JSON 已将含加载的 CLI/HTTP 观察与不含加载、无 trace I/O 的 generate/codec 分阶段时间分开。
同机四线程数据见 `docs/develop/ARKTTS.md`；资源未隔离、部分 Oracle 计时与编译重叠，
属于单次观察值，不作速度排名。GPU 仍有逐投影同步/上传和主机 attention/codec 开销。

量化损失与实现误差分别记录：`multiq/review-20261111/quantization-reference.json` 比较原始
官方权重和同档 GGUF 的官方 Oracle（相同 prompt codes、固定解码 codes），而上面的
1053 检查点表比较同一 GGUF 的 Oracle 与 Rust。固定码流波形的前一项损失为：

| 格式 | Oracle 自由 ID 对原精度相同数 / 120 | 波形 max_abs | 波形 RMSE |
| --- | ---: | ---: | ---: |
| F32 | 120 | 0 | 0 |
| BF16 | 120 | 0.002624 | 0.0003523 |
| F16 | 120 | 0.0003250 | 0.00003974 |
| Q8_0 | 62 | 0.04056 | 0.004473 |
| Q4_0 | 6 | 0.03744 | 0.006020 |
| Q4_1 | 18 | 0 | 0 |
| Q4_K | 23 | 0.08470 | 0.01115 |
| Q5_K | 42 | 0.04889 | 0.006544 |
| Q6_K | 44 | 0.04027 | 0.004577 |

这张表描述权重/激活精度改变后的损失，没有将其混入同权重实现误差；它也不等于语音质量评分。
Q4_1 行已更新为解码 Transformer 保留 F32 的修订导出：原 max_abs/RMSE 为 0.1290/0.01501，
修订后固定码流波形与原精度 Oracle 相同，但主模型的自由生成仍有精度损失。
新证据在 `remote-20261111/macos-reviewed-results.json`，旧记录保留在原目录。

## 历史 dense-only 覆盖矩阵与改动边界

以下每行同时覆盖 `arktts` 主模型、`arktts_codec` 的 encode/decode，以及普通 TTS 和参考声音克隆。
CLI 为普通/克隆两条命令，HTTP 为普通/克隆各自的 WAV/PCM 请求；命令和请求体由 `entries.py` 写入 JSON。
KV、attention 和输出缓冲为 F32；矩阵乘使用共享 Prepared 契约，F16 激活/累加及块量化 QDQ 遵循共享 kernel。
这是每档 4 帧、三种文本的数值/入口验收，不是长音频质量或全图 GPU 验收。
下表的 27 行已在 M3 Max ARM64 和 Linux x86_64/RADV NAVI31 两台机器分别验证通过，
两台机器记录的 dense call 数量一致；Linux 原始证据路径见下方复验记录。

| 主模型 + codec 格式 | 请求后端 | CLI 普通/克隆 | HTTP WAV/PCM 普通/克隆 | 实际 dense device / host calls | 实现状态 | 验证状态 |
| --- | --- | --- | --- | ---: | --- | --- |
| F32 | cpu | 通过 / 通过 | 通过（4 请求） | 0 / 8736 | 已实现 | 通过 |
| F32 | auto | 通过 / 通过 | 通过（4 请求） | 1188 / 7548 | 已实现 | 通过 |
| F32 | vulkan | 通过 / 通过 | 通过（4 请求） | 8736 / 0 | 已实现 | 通过 |
| F16 | cpu | 通过 / 通过 | 通过（4 请求） | 0 / 8736 | 已实现 | 通过 |
| F16 | auto | 通过 / 通过 | 通过（4 请求） | 1188 / 7548 | 已实现 | 通过 |
| F16 | vulkan | 通过 / 通过 | 通过（4 请求） | 8736 / 0 | 已实现 | 通过 |
| BF16 | cpu | 通过 / 通过 | 通过（4 请求） | 0 / 8736 | 已实现 | 通过 |
| BF16 | auto | 通过 / 通过 | 通过（4 请求） | 1188 / 7548 | 已实现 | 通过 |
| BF16 | vulkan | 通过 / 通过 | 通过（4 请求） | 8736 / 0 | 已实现 | 通过 |
| Q8_0 | cpu | 通过 / 通过 | 通过（4 请求） | 0 / 8736 | 已实现 | 通过 |
| Q8_0 | auto | 通过 / 通过 | 通过（4 请求） | 1188 / 7548 | 已实现 | 通过 |
| Q8_0 | vulkan | 通过 / 通过 | 通过（4 请求） | 8736 / 0 | 已实现 | 通过 |
| Q4_0 | cpu | 通过 / 通过 | 通过（4 请求） | 0 / 8736 | 已实现 | 通过 |
| Q4_0 | auto | 通过 / 通过 | 通过（4 请求） | 1188 / 7548 | 已实现 | 通过 |
| Q4_0 | vulkan | 通过 / 通过 | 通过（4 请求） | 8736 / 0 | 已实现 | 通过 |
| Q4_1 | cpu | 通过 / 通过 | 通过（4 请求） | 0 / 8736 | 已实现 | 通过 |
| Q4_1 | auto | 通过 / 通过 | 通过（4 请求） | 1188 / 7548 | 已实现 | 通过 |
| Q4_1 | vulkan | 通过 / 通过 | 通过（4 请求） | 8736 / 0 | 已实现 | 通过 |
| Q4_K | cpu | 通过 / 通过 | 通过（4 请求） | 0 / 8736 | 已实现 | 通过 |
| Q4_K | auto | 通过 / 通过 | 通过（4 请求） | 1188 / 7548 | 已实现 | 通过 |
| Q4_K | vulkan | 通过 / 通过 | 通过（4 请求） | 8736 / 0 | 已实现 | 通过 |
| Q5_K | cpu | 通过 / 通过 | 通过（4 请求） | 0 / 8736 | 已实现 | 通过 |
| Q5_K | auto | 通过 / 通过 | 通过（4 请求） | 1188 / 7548 | 已实现 | 通过 |
| Q5_K | vulkan | 通过 / 通过 | 通过（4 请求） | 8736 / 0 | 已实现 | 通过 |
| Q6_K | cpu | 通过 / 通过 | 通过（4 请求） | 0 / 8736 | 已实现 | 通过 |
| Q6_K | auto | 通过 / 通过 | 通过（4 请求） | 1188 / 7548 | 已实现 | 通过 |
| Q6_K | vulkan | 通过 / 通过 | 通过（4 请求） | 8736 / 0 | 已实现 | 通过 |

证据根目录为上面的 `multiq/`：入口矩阵是 `entries-v2-results.json`，逐条原始日志/WAV 在
`entries-v2/<format>/<format>-<backend>/`；固定历史矩阵为 `validation-v2-results.json`，
逐检查点指标/日志在 `<format>/validation-v2-<backend>/`，同输入算子证据为 `projection-results.json`。
`inventory-v2.json` 保存初版 18 个导出文件的大小、SHA-256、bundle ID 和逐类型张量数；
最终 Q4_1 文件身份以 `final-inventory-20261111.json` 为准，修订数值/入口证据在
`multiq-reviewed/q4_1/`、`entries-q4_1-reviewed/` 和 `remote-20261111/macos-reviewed-results.json`。

转换器复用共享导出格式集合：F32 + `quantize_gguf.BLOCK_FORMATS`。Q5_0/Q5_1 虽有共享运行内核，
当前共享 Python exporter 没有编码器；IQ 也没有匹配的共享 LinearExecutor Vulkan kernel，
本转换器没有新增这些量化算法，也不把它们宣称为已支持。

九档文件都是混合精度。主模型 norm/bias 保留 BF16；K 格式只量化 28 个 4864 宽的 down projection，
其余 198 个张量保留 BF16。codec 的卷积/norm/codebook/重采样 ConvNeXt 保留 F32；
K 格式还保留 1216 宽 down projection，因此 codec 只有 92 个 K 格式张量；
F16/BF16/Q8_0/Q4_0 转换 100 个 dense 张量。Q4_1 另外保留 40 个解码 Transformer 投影为 F32，
只有 60 个编码器投影为 Q4_1（395 个 F32 张量）；主模型的格式策略不变。
F16/BF16/Q8_0/Q4_0/Q4_1 的主模型有 144 个所选格式二维张量（BF16 总共 226 个），
F32 主模型也保留 82 个 BF16 向量，不能解读成所有张量均为 F32。

改动范围：模型/codec/Transformer 位于 `src/models/arktts/`，转换/Oracle 工具位于各自 arktts 目录；
必要接线为 models 注册、既有 CLI `--tts`、按能力命名的 HTTP `/v1/audio/speech` 及文本后处理桥。
纯 TTS 不提供文本评分/JEV，也没有流式协议；CLI/HTTP 对这些请求明确拒绝。Cargo metadata 与
`upstream/main@d16dc16d9` 对比，9 个 binary targets 的名称/源码路径一致。

共享变更仅有：Breeze causal Conv 的 crate 内可见性放开以直接复用其实现；以及 Vulkan 的
`build_operator_pipelines` 在无 shaderFloat64 设备上为 Nemotron MoE router 使用占位 pipeline，
并由 `record_nemotron_moe_route` 在记录前明确拒绝。这修复了 M3 上所有 dense executor 的全局初始化失败；
FP64 设备仍使用原 shader，Qwen/Nemotron 的模型配置、权重和默认计算语义未改。
受影响路径是 `VulkanContext::operator_pipelines` 的所有调用方和 Nemotron MoE router；
真实 M3 dense 提交及共享 compute 测试覆盖了可用设备路径。没有 Nemotron 30B 权重/FP64 设备，
该旧模型真实推理回归未运行，不以 ArkTTS 通过代替其回归。


### 本次导出文件身份

文件名为 `Audio8-TTS-Realistic-Air-<FORMAT>.gguf` 与 `Audio8-TTS-Realistic-Air-codec-<FORMAT>.gguf`，
位于 `multiq/<format>/`，修订的 Q4_1 位于 `multiq-reviewed/q4_1/`。共同 bundle ID 为 `8ff0085d1ab4bbe1fbf8f68167ca44d091d88c7beb2c363e2563bde0fce0a391`。
下表从 `final-inventory-20261111.json` 保留最终身份；`inventory-v2.json` 和旧 Q4_1 记录保留历史证据。
原始 2026-11-04 文件另见上方表，新增 `general.file_type` 会改变 GGUF 哈希。

| 格式 | 主模型字节数 / SHA-256 | codec 字节数 / SHA-256 |
| --- | --- | --- |
| F32 | 2416713408 / `508acc5ecaa94303d64e8a53cddc4f48a842ab27ddf2d39f3a5b61b9e168dbf9` | 1349626848 / `f50e1f52b878518f8ab012daa52c7b528053632c66e260a5553bfab0e46eddf3` |
| BF16 | 1214553792 / `c6a26e3ddeb6b8b7e25ac6e8fa26891929d3fe236c4e573f959f2ac30a6aef19` | 912370656 / `5c05c5017c0f13096a93a807e9a940e2b7676a473774ef9b256bdc8b2eafe033` |
| F16 | 1214553792 / `e1a152a5e7f6aeb6a40899e1d2407dbe3afd5e2608ddccda4d0d995f4b09691e` | 912370656 / `0eebc4a94ecae9c412fa18871ac69bfe8e44c7493e2961f9b780a0f453e8b8af` |
| Q8_0 | 651041472 / `dc078f273da83f219596ac89046905544eaa7f077d388618ecc4d50d81c85599` | 707406816 / `a15704063f14d23bad396b768dde732534a74935e0a5c18cc4f9d085b86bf589` |
| Q4_0 | 350501568 / `9aa78f08b6a5145aa14b03473346659b41131ffea2c00a108a5246b2c9b511d9` | 598092768 / `3e5079beb89dfc2d14ba83b8ea84264bb83e237f8e401d4a8e7b66019b523a4c` |
| Q4_1 | 388069056 / `78c148d1329c8338a97691b1bf83c7fc2a0b38f396bd9a5f7817f20fa53f2bb6` | 797551584 / `d70ea4b1aaf1a8fdeaea83510ddb0b3fd2978c38aa799fdc7bd4c771c05659ad` |
| Q4_K | 1039138496 / `a2d5bad578ebe5b044a6264dec9249881ef5a7191355d3219cd35dc4bd63cdf1` | 632335328 / `bcf725d253b65861320b5fdc107d56085a00425fa9b6dad3364fae2dd7d130e4` |
| Q5_K | 1054392000 / `02fa16a3dd9b8f131f3637f84fd36d7a8a037b44ebe2f85449d6d5756170d702` | 658418656 / `c3f603b8bc6cfa913b96a03a44b458ef329d8145bcf14bc48bfeae185ec3a6d0` |
| Q6_K | 1070598848 / `462ae46b6e184ba47e4fdac8cd42c4b02cbcb659049c2e51da3645035b6b6d49` | 686132192 / `da197f0fdee862bfb19bbc7605e6314b88c1c163dfb0a6cf58810517f92281d4` |

新增复验：6 个真实设备 compute::linear 用例通过，标量门禁用例按其要求在独立
`RMI_SCALAR=1` 进程通过（批量 include-ignored 未满足这个前提的那次失败不算产品回归）。
Q4_0 的 CPU/Auto/Vulkan CLI/HTTP 普通与克隆重新运行通过；禁用 ICD 后，CLI/HTTP Auto
回退输出均与 CPU WAV 相同，两个入口的严格 Vulkan 均拒绝启动。证据在
`multiq/review-20261111/{entries,fallback}/`。新 Oracle 与既有 27 份固定历史 trace 再比较，
每份 1053 个检查点仍全部通过；记录在 `review-20261111/fixed-history-results.json`。

## Linux x86_64 / RADV NAVI31 复验（2026-11-11，历史 dense-only 基线）

使用独立目录 `/workspace/rust-model-inference-arktts-20261111`，Linux 6.8.0-79-generic x86_64，
AMD EPYC 9334、AMD Radeon Graphics（RADV NAVI31，VRAM 51,522,830,336 字节）。
容器 CPU quota 16、内存上限 59,055,800,320 字节；推理和 PyTorch 均为 4 线程，
`release-fast` + `vulkan,parity-trace`，ICD 为 `/usr/share/vulkan/icd.d/radeon_icd.json`。
Python 3.12.3、torch 2.9.1+gitff65f5b、transformers 4.57.3、numpy 2.5.4、
safetensors 0.8.0、soundfile 0.14.0、gguf 0.17.1。新装 Python 依赖、Cargo 缓存和临时文件均在
测试仓库内；只读复用既有 `/opt/venv` 的 torch 和已有 Rust 工具链，未改系统环境。

14 个原始模型文件的 SHA-256 与本机固定清单一致。远端 1656 个 tracked source 文件按 SHA-256
与 `9f607fbcbd7fb03bcc6361b3b903740e3748680f` 对照；Rust binary 构建于 `154ec0c33`，
这两个提交间仅修改 Python converter/测试，Rust 源码相同。`host/versions.json`、
`host/source-reviewed.json` 和 `host/host.txt` 保存身份及环境。

最终九档 × 三后端的 27 组均通过：每组 1053 个固定历史检查点、6 个独立同输入 projection、
CLI 普通/克隆和 HTTP 普通/克隆各自 WAV/PCM。HTTP 的 108 个响应均与对应 CLI 字节相同。
同输入 projection 的最差 relative RMSE 为 `1.382e-06`，低于冻结的 `5e-5`。
固定历史严格 Vulkan 每组 4308 次 device / 0 次 host linear，Auto 为 468 / 3840；
入口矩阵严格 Vulkan 为 8736 / 0，Auto 为 1188 / 7548。计数仅指 dense linear，
不包含仍在主机执行的 attention/KV、norm/RoPE、卷积、codebook 和采样。

下表取每档三个后端的最差指标；自由 greedy 一列分别为 CPU/Auto/Vulkan 对同权重 Oracle
相同的 ID 数，每个后端各 120 个。检查门槛沿用前述冻结 profile，没有按远端结果放宽。

| 格式 | 最差 relative RMSE | 最差 normalized max_abs | 波形 max_abs | 波形 RMSE | 自由 greedy 相同 ID / 120（CPU/Auto/Vulkan） |
| --- | ---: | ---: | ---: | ---: | --- |
| F32 | 9.353e-06 | 1.523e-05 | 9.015e-07 | 1.358e-07 | 120 / 120 / 120 |
| BF16 | 1.311e-05 | 1.417e-05 | 1.128e-06 | 1.671e-07 | 120 / 120 / 120 |
| F16 | 2.600e-03 | 2.619e-03 | 2.424e-04 | 3.364e-05 | 120 / 120 / 120 |
| Q8_0 | 1.136e-01 | 7.106e-02 | 3.593e-03 | 3.932e-04 | 56 / 56 / 56 |
| Q4_0 | 8.495e-02 | 6.566e-02 | 3.689e-03 | 4.576e-04 | 62 / 39 / 43 |
| Q4_1 | 7.662e-02 | 6.898e-02 | 9.015e-07 | 1.358e-07 | 35 / 35 / 35 |
| Q4_K | 5.585e-02 | 4.514e-02 | 8.494e-07 | 1.396e-07 | 55 / 40 / 73 |
| Q5_K | 1.326e-01 | 6.045e-02 | 1.064e-06 | 1.366e-07 | 81 / 44 / 51 |
| Q6_K | 4.861e-02 | 4.833e-02 | 1.360e-06 | 1.831e-07 | 74 / 83 / 63 |

F32/BF16 在两机均为 120/120；F16 本机为 119/120，远端为 120/120。块量化的自由码流仍可
随 CPU 架构、归约和后端发生分叉，因此量化结果保留 Experimental，固定历史通过不是音质证明。

### Q4_1 波形门槛失败与修复

初版 Q4_1 codec 在 x86 CPU 的固定波形 `max_abs=0.0065242685`、`RMSE=0.0007709410`；
前者超过预先冻结的 `<0.006`，不能算通过。独立同输入 projection 检查通过，
`quantizer.post_module` 输入差约 1e-7，解码 Transformer 第 2 层开始 activation QDQ bin 分叉，
后续重采样/decoder 将差异放大。没有修改共享 kernel 或验收门槛。

修复 `9f607fbcb` 仅让 Q4_1 的 `quantizer.post_module.*` 保留 F32；主模型与 codec 编码器
继续 Q4_1，codec 最终为 395 个 F32 + 60 个 Q4_1 张量。先运行新增行为用例观察失败，再修正转换器。
新文件在两台机器各自的 `multiq-reviewed/q4_1/`，两机 CPU/Auto/Vulkan 的全部数值/入口复验通过。
修复后的波形 max_abs 分别为：

| 主机 | CPU | Auto | Vulkan |
| --- | ---: | ---: | ---: |
| M3 Max | 1.457e-06 | 1.576e-06 | 1.517e-06 |
| Linux x86_64 | 8.028e-07 | 8.773e-07 | 9.015e-07 |

原失败资产和日志留在 `multiq/q4_1/`，不作为最终通过证据。Mac 的修订同权重 Oracle 固定波形
相对原精度损失为零，主模型自由码流仍有量化差异；不能将此解释成整套 Q4_1 无损。

### 跨平台 codec 文件身份

九档主模型文件的 SHA-256 与上面的 Mac 清单全部一致；codec 字节数相同、SHA-256 不同。
对 F32 codec 的 455 个张量逐项检查，69 个差异全部来自 80 个 weight-norm 展开张量，
非 weight-norm 张量全部相同。两机分别与独立 FP64 weight-norm 公式比较，max_abs
为 Mac `3.402925773e-7`、Linux `1.731922971e-7`。该浮点展开步骤使用容差，不强制跨平台逐位一致。
`host/{macos-weight-norm,linux-weight-norm,codec-export-difference}.json` 保留逐张量指标。

以下为最终 Linux codec SHA-256（Q4_1 为修订文件），大小见前述相同格式的文件表：

| 格式 | Linux codec SHA-256 |
| --- | --- |
| F32 | `cc5eae7c38aaa81997eef6b67a1073b6d222d0208fad295cde8e93a78c59ed3c` |
| BF16 | `002d886c693b7e64a9ef0771d5eceae7ba0174bb9962e4a463715b5553386245` |
| F16 | `dab733738891f581c7a09d54b8382ba3369fc007e42c2119053008973fe40861` |
| Q8_0 | `14ae264377dbfcc27f1f4184a7923a1b7c3eb30824729893fd85764e289778ca` |
| Q4_0 | `82ea13588c3115629c33e5f04d0e25b12069a4169c0e8dec8a62243260fd4c56` |
| Q4_1 | `320117073a9e5ebbec4bcb9f914465745d403de49df5bd9415391822645807e2` |
| Q4_K | `b0437acf809ce26070ff5ec03bc7b9a933860e51992564c5a45130ecc8f586f9` |
| Q5_K | `fa73e9e8ab017d48aaa1421f692839c081c78d5e15d9ba8f64b832f44e06bbfa` |
| Q6_K | `72c311768c4eff2d4c41e87db2e43a4d20d45d29588dd112f3bc274507ae29dd` |

### 入口、回归与证据

- ArkTTS：8 passed / 2 ignored；共享 `compute::linear`：12 passed，含 6 个真实设备检查。
- `RMI_SCALAR=1` 独立门禁：1 passed；ArkTTS GPU projection：1 passed；`audio_speech`：1 passed；System One CLI：7 passed。
- 修复后的 Python same-weight/converter/comparator：10 passed；两平台均通过。
- 禁用 ICD 的 Q4_0 CLI/HTTP 普通/克隆 Auto 均与 CPU 输出逐字节相同、GPU calls=0；两个入口严格 Vulkan 均拒绝启动。
- Linux 同机 CPU/Auto/Vulkan/同权重 Oracle 的 generate/decode/encode 阶段时间已写入 `performance.json` 的 `linux`，并在 `docs/develop/ARKTTS.md` 给出生成时间表。

所有最终远端证据已归档到本机 `MODEL_DIR/remote-20261111/`：

| 文件/目录 | 内容 |
| --- | --- |
| `remote-results.json` | 最终 27 组指标、自由 ID/首个分叉、阶段耗时、27 组入口、18 个 GGUF 身份和检查汇总 |
| `multiq/<format>/` | 原始 Oracle/trace/指标和 projection/replay/compare/greedy 日志（最终排除旧 Q4_1） |
| `multiq-reviewed/q4_1/` | 修订 Q4_1 的三后端完整数值证据 |
| `entries-remote/`、`entries-q4_1-reviewed/` | 两个 CLI 命令、四个 HTTP 请求及字节比较；最终以原八档 + 修订 Q4_1 三组为准 |
| `fallback-remote/` | 无 ICD 的 CPU/Auto 字节比较及严格 Vulkan 拒绝 |
| `host/` | 构建/测试日志、源码清单、依赖版本、机器快照和复现脚本 |

JSON 的 `binary_path` 保留远端绝对地址，在原远端目录可直接复现；使用本机归档时需映射到对应文件路径。
归档不重复下载远端 GGUF；本机最终模型文件仍位于用户指定的 `MODEL_DIR`。
计时来自共享机器、并发验证中的单次 `release-fast` 观察，未作隔离吞吐 benchmark，也未测完整 release 性能。

## 设备驻留数值图（2026-11-11，当前最终实现）

此前 `c4b07899e` 只卸载 dense projection，attention/KV、norm/RoPE、卷积、码本和采样还在 CPU。
本轮补齐这些 GPU 算子：生成、codec encode/decode 各有设备驻留图，中间激活/KV 不逐算子往返主机。
RMS 和原生格式矩阵乘复用共享设备算子，其余差异放在 ArkTTS 私有 shader/recorder；没有新二进制或外部加速依赖。
Host 保留 tokenizer、输入校验、seeded RNG、控制循环和 WAV 序列化；只读 embedding 表初始化时展开 F32，
投影保持原生 GGUF 存储与 Prepared 激活契约。`auto` 失败时从头重跑该阶段的 CPU 图，严格 `vulkan` 返回错误。
设备 storage-buffer/显存和 checked arena 容量不足也明确失败。

同一模型/revision、三文本、四帧、固定历史和线程沿用前述 pin，误差门槛未改。
每机九档 × CPU/Auto/Vulkan 共 27 组，各 1053 检查点；每组还分别执行两个 CLI 和四个 HTTP 请求。
最终缩短 GPU 提交批次后，18 个 Auto/Vulkan 组合全部重跑；CPU 实现未变，沿用本轮此前的九组 CPU 记录，
最终目录的 CPU 子目录链接到其原始证据。旧 Mac lossy fixture 只补当前 Oracle 的 `exact_generated_codes:false` 元数据，
浮点 payload 与波形门槛未改；F32/BF16 的生成 IDs 仍精确检查，固定历史比较器逐档执行更严格的冻结门槛。

### 当前入口与后端矩阵

每行同时覆盖主模型、codec encode/decode、普通 TTS 和参考声音克隆；CLI/HTTP 两列为 Mac / Linux。

| 格式 | 请求后端 | CLI 普通/克隆 | HTTP 普通/克隆 WAV/PCM | 实际数值后端 | 实现 / 验证 |
| --- | --- | --- | --- | --- | --- |
| F32 | cpu | 通过 / 通过 | 通过 / 通过 | CPU SIMD | 已实现 / 通过 |
| F32 | auto | 通过 / 通过 | 通过 / 通过 | Vulkan 驻留图；host linear=0 | 已实现 / 通过 |
| F32 | vulkan | 通过 / 通过 | 通过 / 通过 | Vulkan 驻留图；host linear=0 | 已实现 / 通过 |
| BF16 | cpu | 通过 / 通过 | 通过 / 通过 | CPU SIMD | 已实现 / 通过 |
| BF16 | auto | 通过 / 通过 | 通过 / 通过 | Vulkan 驻留图；host linear=0 | 已实现 / 通过 |
| BF16 | vulkan | 通过 / 通过 | 通过 / 通过 | Vulkan 驻留图；host linear=0 | 已实现 / 通过 |
| F16 | cpu | 通过 / 通过 | 通过 / 通过 | CPU SIMD | 已实现 / 通过 |
| F16 | auto | 通过 / 通过 | 通过 / 通过 | Vulkan 驻留图；host linear=0 | 已实现 / 通过 |
| F16 | vulkan | 通过 / 通过 | 通过 / 通过 | Vulkan 驻留图；host linear=0 | 已实现 / 通过 |
| Q8_0 | cpu | 通过 / 通过 | 通过 / 通过 | CPU SIMD | 已实现 / 通过 |
| Q8_0 | auto | 通过 / 通过 | 通过 / 通过 | Vulkan 驻留图；host linear=0 | 已实现 / 通过 |
| Q8_0 | vulkan | 通过 / 通过 | 通过 / 通过 | Vulkan 驻留图；host linear=0 | 已实现 / 通过 |
| Q4_0 | cpu | 通过 / 通过 | 通过 / 通过 | CPU SIMD | 已实现 / 通过 |
| Q4_0 | auto | 通过 / 通过 | 通过 / 通过 | Vulkan 驻留图；host linear=0 | 已实现 / 通过 |
| Q4_0 | vulkan | 通过 / 通过 | 通过 / 通过 | Vulkan 驻留图；host linear=0 | 已实现 / 通过 |
| Q4_1 | cpu | 通过 / 通过 | 通过 / 通过 | CPU SIMD | 已实现 / 通过 |
| Q4_1 | auto | 通过 / 通过 | 通过 / 通过 | Vulkan 驻留图；host linear=0 | 已实现 / 通过 |
| Q4_1 | vulkan | 通过 / 通过 | 通过 / 通过 | Vulkan 驻留图；host linear=0 | 已实现 / 通过 |
| Q4_K | cpu | 通过 / 通过 | 通过 / 通过 | CPU SIMD | 已实现 / 通过 |
| Q4_K | auto | 通过 / 通过 | 通过 / 通过 | Vulkan 驻留图；host linear=0 | 已实现 / 通过 |
| Q4_K | vulkan | 通过 / 通过 | 通过 / 通过 | Vulkan 驻留图；host linear=0 | 已实现 / 通过 |
| Q5_K | cpu | 通过 / 通过 | 通过 / 通过 | CPU SIMD | 已实现 / 通过 |
| Q5_K | auto | 通过 / 通过 | 通过 / 通过 | Vulkan 驻留图；host linear=0 | 已实现 / 通过 |
| Q5_K | vulkan | 通过 / 通过 | 通过 / 通过 | Vulkan 驻留图；host linear=0 | 已实现 / 通过 |
| Q6_K | cpu | 通过 / 通过 | 通过 / 通过 | CPU SIMD | 已实现 / 通过 |
| Q6_K | auto | 通过 / 通过 | 通过 / 通过 | Vulkan 驻留图；host linear=0 | 已实现 / 通过 |
| Q6_K | vulkan | 通过 / 通过 | 通过 / 通过 | Vulkan 驻留图；host linear=0 | 已实现 / 通过 |

每机 54 个 CLI 输出和 108 个 HTTP 响应；所有 HTTP WAV/PCM 与同后端对应 CLI 字节相同。
`entries.py` 要求 generate/encode/decode 图都存在、attention/norm/RoPE/卷积等计数为正、encode 有 10 次码本搜索、
无 CPU fallback、无 trace 读回，并检查实际 `ComputeStats` submissions/上传/读回均非零。

### 数值结果

下表每项取该格式三个后端的最差值；max_abs/RMSE 为固定码流的同权重实现误差，未混入量化相对原始权重的损失。

| 格式 | Mac relative RMSE | Linux relative RMSE | Mac 波形 max / RMSE | Linux 波形 max / RMSE |
| --- | ---: | ---: | --- | --- |
| F32 | 3.269e-05 | 9.353e-06 | 1.606e-06 / 2.005e-07 | 9.751e-07 / 1.375e-07 |
| BF16 | 2.796e-05 | 9.353e-06 | 1.654e-06 / 2.231e-07 | 1.095e-06 / 1.587e-07 |
| F16 | 5.208e-03 | 2.015e-03 | 2.115e-04 / 2.983e-05 | 2.132e-04 / 2.934e-05 |
| Q8_0 | 6.544e-02 | 1.136e-01 | 1.501e-03 / 1.730e-04 | 3.593e-03 / 3.932e-04 |
| Q4_0 | 7.871e-02 | 8.982e-02 | 1.606e-06 / 2.005e-07 | 9.751e-07 / 1.375e-07 |
| Q4_1 | 8.691e-02 | 8.803e-02 | 1.606e-06 / 2.005e-07 | 9.751e-07 / 1.375e-07 |
| Q4_K | 5.854e-02 | 5.585e-02 | 1.846e-06 / 2.720e-07 | 1.113e-06 / 1.739e-07 |
| Q5_K | 6.386e-02 | 1.004e-01 | 1.345e-06 / 1.864e-07 | 9.127e-07 / 1.400e-07 |
| Q6_K | 7.045e-02 | 5.282e-02 | 1.253e-03 / 1.822e-04 | 1.360e-06 / 1.903e-07 |

| 格式 | Mac 自由 greedy 相同 ID / 120（CPU / Auto / Vulkan） | Linux（同左） |
| --- | --- | --- |
| F32 | 120 / 120 / 120 | 120 / 120 / 120 |
| BF16 | 120 / 120 / 120 | 120 / 120 / 120 |
| F16 | 119 / 120 / 120 | 120 / 120 / 120 |
| Q8_0 | 62 / 52 / 52 | 56 / 98 / 98 |
| Q4_0 | 31 / 46 / 46 | 62 / 53 / 53 |
| Q4_1 | 87 / 72 / 72 | 35 / 39 / 39 |
| Q4_K | 77 / 43 / 43 | 55 / 43 / 43 |
| Q5_K | 72 / 84 / 84 | 81 / 54 / 54 |
| Q6_K | 95 / 57 / 57 | 74 / 86 / 86 |

自由 greedy 的原始 ID 和首个分叉在逐组 `greedy.log` 及汇总 JSON；低精度分叉仍为已知限制，
固定历史通过不等于主观音质/说话人相似度已验证。性能按无 trace 的同机阶段记录，见 `performance.json` 的
`resident_graph`；`historical_projection_only` 保留旧混合实现的数据，不能用它描述当前计算分工。

### 驻留与传输证据

以 F32 无 trace 的四帧生成、三帧解码和 6145 点参考编码为例，两机记录相同逻辑算子数：

| 阶段 | 图操作 | attention / norm / RoPE | 卷积 | 码本搜索 / 采样 | host 输入 / 输出字节 |
| --- | ---: | --- | ---: | --- | --- |
| generate（英文） | 3368 | 256 / 552 / 256 | 0 | 0 / 40 | 1420 / 480 |
| codec_decode | 230 | 8 / 19 / 8 | 44 | 0 / 0 | 120 / 24576 |
| codec_encode（含补零） | 287 | 12 / 28 / 12 | 54 | 10 / 0 | 32768 / 160 |

表中不含初始化权重上传；完整上传/分配/提交计数保存在同一日志的 `ComputeStats`，其值是设备上下文累计值。
GPU parity trace 会按检查点额外读回，性能与传输表使用 `trace_read_bytes=0` 的独立运行。

### 本轮自查修复

1. M3 的 64 线程 attention 跨两个 subgroup：softmax 最大值尚未被后一组读取就被前一组覆盖。
   加 workgroup barrier 后，F32 的首次超限检查点恢复通过；真实 GPU 测试覆盖 31/32/33 和 128/512 窗口边界。
2. 新驻留路径下旧 Q4_0 codec 在 M3 波形 max_abs=0.0086856708、RMSE=0.0009986057，超过冻结 max<0.006。
   投影输入约 1e-7 的差异跨越 decoder 的 activation QDQ bin，逐层放大；没有改共享 kernel 或门槛。
   让 Q4_0 与此前 Q4_1 一样，仅将 `quantizer.post_module.*` 保留 F32；主模型和 60 个 codec encoder 投影仍量化，
   codec 为 395 F32 + 60 Q4_0。新增 red/green 转换用例，重新导出、建立同权重 Oracle 并在两机三后端复验。
3. 129 帧长音频在 RADV 的大批次提交触发 guilty hard recovery / device-lost；插入 trace 同步后 decode 可通过，
   encode 的长批次仍失败。将每批图操作上限由 64 降至 8 后，无 trace 的 CPU/Vulkan/Auto 长用例通过，
   保留原失败日志；不以 Auto 回退掩盖严格设备失败。

Q4_0 修订主模型与原文件相同；codec 为 787327968 字节，当前身份：

| 主机 | codec SHA-256 |
| --- | --- |
| macos | `9dcef55c16220f2e31f8ad161e8587bead2fec796672990fb5d498aa377c5504` |
| linux | `6fa66811537a21f914579f6248883680c66819054cdf62f5ce3ba61f524ea596` |

Mac 修订资产为 `MODEL_DIR/multiq-device/q4_0/`，Q4_1 仍为 `multiq-reviewed/q4_1/`；
其余格式复用 `multiq/<format>/`。两机主模型哈希相同，codec 的跨平台 weight-norm 展开差异沿用前述容差解释。

### 长音频、随机采样、回退与回归

独立官方 F32 fixture：129 帧固定解码、264192 点参考编码（约 6 秒），跨 decoder 128 和 encoder 512 窗口；
三个后端均检查完整输出、冻结 float 波形门槛与参考编码的精确 IDs。结果：

| 主机 | 后端 | 波形 max_abs | 波形 RMSE |
| --- | --- | ---: | ---: |
| macos | cpu | 4.923e-06 | 2.291e-07 |
| macos | auto | 1.022e-05 | 3.001e-07 |
| macos | vulkan | 1.022e-05 | 3.001e-07 |
| linux | cpu | 3.967e-06 | 1.492e-07 |
| linux | auto | 3.964e-06 | 1.673e-07 |
| linux | vulkan | 3.964e-06 | 1.673e-07 |

两机各自的 BF16 默认 temperature=0.8、seed=42，在三后端重复运行 CLI/HTTP 普通/克隆 WAV/PCM，
同后端字节相同；真实 GPU 原语测试还检查与 CPU 的 top-k/top-p/Gumbel 结果及 RNG 消耗。
两机禁用 ICD 后，Q4_0 CLI/HTTP Auto 与 CPU WAV/PCM 字节相同，严格 CLI/server 拒绝；没有设备提交。

本轮检查：ArkTTS 9 passed / 4 ignored；两个 ignored GPU 原语用例显式运行通过，连同 arena 为 3 passed（两机）。
Python 10 项通过（两机），`cargo check --features vulkan,parity-trace`、Rustfmt、diff/checksum 检查通过。
当次全库 lib 前后均为 1555 passed / 13 failed / 100 ignored，13 个失败名逐条相同；保留完整基线和结果，未当作全绿。
Breeze Conv 仅新增只读 view，旧 forward 与本轮改动前源文本逐字节相同；ComputePolicy 清单只增加 ArkTTS 两个架构。
旧模型没有新的数值分支；此前 FP64 Nemotron router 的实模回归仍未运行，不以本模型通过代替。

当前原始证据：

- `device-20261111-final-macos/`：27 组 done/metrics/replay/greedy；CPU 链接到本轮原始 CPU 目录。
- `entries-device-final-macos/`：27 组真实 CLI/HTTP 请求、日志和 WAV；`stochastic-device-final-macos/`、`fallback-device-final-macos/` 是额外检查。
- `device-20261111-remote/`：相同 Linux 目录及 `host/` 的脚本、失败/通过日志、源码与资产身份；归档不下载远端 GGUF 或 trace 浮点 payload。
- `device-20261111-results.json`：两机完整指标、首个分叉、命令/请求、提交计数与资产身份；远端绝对路径保留，使用本机归档需映射路径。

没有扩展流式或 JEV：该纯 TTS 契约不提供这些能力；默认构建产物仍为原 9 个 binary targets。


## 仍未覆盖

未验证完整发布构建的性能峰值、除已记录 ARM64/x86_64 配置外的真实模型推理、官方原生 BF16 激活语义、
2048 位置上限、F16/量化长音频、全部 23 种语言、主观音质/声音相似度指标，
以及完整多模态文本后处理桥。参考 WAV 重采样沿用现有 sinc 实现，未声称和官方前端一致；
采样 RNG 与 PyTorch 不同，固定 seed 只保证 Rust 自身复现。
