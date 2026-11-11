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
- 全库 lib 回归（不启用 Vulkan feature）：1553 passed / 13 failed / 100 ignored；未修改的 `upstream/main@d16dc16d9` 当次基线为 1544 / 13 / 99，13 个失败名称完全一致。此前 1552 / 14 / 99 的记录包含新增用例的 red 阶段，不能当作未修改主线基线，现予以更正。

既有失败位于：Breeze 的 3 个 dtype 契约用例、CLM 的 2 个 golden/embedding 用例、
Z-Image 的 3 个 NEON 逐位用例、FunASR 的 2 个 NEON 逐位用例、Gemma4 的 1 个 trunk 用例、
app media 的 1 个平台用例和 matmul 的 1 个 NEON softmax 用例。本次未修改这些计算行为。

PR #200 的初次 CI：`x86 dispatch regressions` 通过；Rustfmt 与 Vulkan job 失败。
后者在 `bash scripts/vulkan-shaders.sh check` 的 8 个 GLSL/SPIR-V checksum 上失败，尚未进入 Rust 编译。
已在未修改的 `afed6d48b` 临时快照与本分支分别执行 `cargo fmt --check` 和 shader check：
shader 输出逐字节相同，失败文件为 q4_k/q5_k/q6_k/f16 matmul 的源码与二进制。
本次新增的 module 顺序和 backend probe 换行问题已修正；余下 32 个 Rustfmt diff blocks 均出现在未修改基线中。
未改写无关模型/算子的既有格式或 shader 校验和；上述两项 CI 仍受基线问题影响。

2026-11-11 合并 `main@d16dc16d9` 后复验：`cargo fmt --all --check` 已通过，原来的格式
失败结论不再适用于当前树。`scripts/vulkan-shaders.sh check` 的所有 checksum 检查通过，
随后本机重编译 `nemotron_moe_route.spv` 与提交产物在 char 13 不同；未修改主线出现同样结果。
这与 2026-11-04 的八个 checksum 失败不同，本次没有改动任何 GLSL/SPIR-V。

自查已修复：

1. P1，`src/app/server/audio_speech.rs:168`：生成锁移入 blocking worker，防止 HTTP 取消后旧推理继续而新请求并发加载 codec；以连接取消后的 429 验证。
2. P1，`src/app/tts/mod.rs:230`、`src/app/server/audio_speech.rs:133`：在 sinc resampler 分配前验证采样率和输出长度，HTTP 文件参考限制为最多 16 MiB 的普通文件；单测与真实 HTTP 边界请求通过。
3. P1，`src/models/arktts/mod.rs:139`：main/codec bundle ID 校验，防止混配不同 checkpoint；精确识别架构，不按相近 Qwen 模型路由；混配单测通过。
4. P2，`src/app/tts/mod.rs:214`、`src/main.rs:592`：对不支持的 CLI/HTTP 控制和文本/JEV 模式明确报错；修正文本后处理桥的固定 Qwen3/24 kHz 描述；两种二进制的拒绝入口已实测。
5. P2，`src/models/arktts/mod.rs:461`：文本清洗补齐 Python 视为空白的 U+001C–U+001F；温度下限与官方 `1e-5` 一致；单测与 Oracle 通过。
6. P1，发布树校验：之前 API 发布遗漏主线删除的两份旧兼容文件；合并已发布分支后保留本地正确树，发布前比较树 SHA，并保留 main 的真实合并祖先。
7. P2，Oracle 根据 GGUF 实际张量类型选择波形门槛；入口证据对严格 Vulkan 额外断言 host linear 为零，测试的拒绝入口使用所选主模型文件，避免依赖另一档 BF16 文件。
8. P2，性能分开记录含加载的请求观察与同口径 generate/codec 计时；新增自由 greedy 分叉、量化相对原精度的损失记录，并补完整矩阵复现命令。

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
| Q4_1 | 18 | 0.1290 | 0.01501 |
| Q4_K | 23 | 0.08470 | 0.01115 |
| Q5_K | 42 | 0.04889 | 0.006544 |
| Q6_K | 44 | 0.04027 | 0.004577 |

这张表描述权重/激活精度改变后的损失，没有将其混入同权重实现误差；它也不等于语音质量评分。

## 覆盖矩阵与改动边界

以下每行同时覆盖 `arktts` 主模型、`arktts_codec` 的 encode/decode，以及普通 TTS 和参考声音克隆。
CLI 为普通/克隆两条命令，HTTP 为普通/克隆各自的 WAV/PCM 请求；命令和请求体由 `entries.py` 写入 JSON。
KV、attention 和输出缓冲为 F32；矩阵乘使用共享 Prepared 契约，F16 激活/累加及块量化 QDQ 遵循共享 kernel。
这是每档 4 帧、三种文本的数值/入口验收，不是长音频质量或全图 GPU 验收。

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
`inventory-v2.json` 保存全部 18 个导出文件的大小、SHA-256、bundle ID 和逐类型张量数。

转换器复用共享导出格式集合：F32 + `quantize_gguf.BLOCK_FORMATS`。Q5_0/Q5_1 虽有共享运行内核，
当前共享 Python exporter 没有编码器；IQ 也没有匹配的共享 LinearExecutor Vulkan kernel，
本转换器没有新增这些量化算法，也不把它们宣称为已支持。

九档文件都是混合精度。主模型 norm/bias 保留 BF16；K 格式只量化 28 个 4864 宽的 down projection，
其余 198 个张量保留 BF16。codec 的卷积/norm/codebook/重采样 ConvNeXt 保留 F32；
K 格式还保留 1216 宽 down projection，因此 codec 只有 92 个 K 格式张量；非 K 格式转换 100 个 dense 张量。
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
位于 `multiq/<format>/`。共同 bundle ID 为 `8ff0085d1ab4bbe1fbf8f68167ca44d091d88c7beb2c363e2563bde0fce0a391`。
下表从 `inventory-v2.json` 保留完整身份；原始 2026-11-04 文件另见上方表，新增 `general.file_type` 会改变 GGUF 哈希。

| 格式 | 主模型字节数 / SHA-256 | codec 字节数 / SHA-256 |
| --- | --- | --- |
| F32 | 2416713408 / `508acc5ecaa94303d64e8a53cddc4f48a842ab27ddf2d39f3a5b61b9e168dbf9` | 1349626848 / `f50e1f52b878518f8ab012daa52c7b528053632c66e260a5553bfab0e46eddf3` |
| BF16 | 1214553792 / `c6a26e3ddeb6b8b7e25ac6e8fa26891929d3fe236c4e573f959f2ac30a6aef19` | 912370656 / `5c05c5017c0f13096a93a807e9a940e2b7676a473774ef9b256bdc8b2eafe033` |
| F16 | 1214553792 / `e1a152a5e7f6aeb6a40899e1d2407dbe3afd5e2608ddccda4d0d995f4b09691e` | 912370656 / `0eebc4a94ecae9c412fa18871ac69bfe8e44c7493e2961f9b780a0f453e8b8af` |
| Q8_0 | 651041472 / `dc078f273da83f219596ac89046905544eaa7f077d388618ecc4d50d81c85599` | 707406816 / `a15704063f14d23bad396b768dde732534a74935e0a5c18cc4f9d085b86bf589` |
| Q4_0 | 350501568 / `9aa78f08b6a5145aa14b03473346659b41131ffea2c00a108a5246b2c9b511d9` | 598092768 / `3e5079beb89dfc2d14ba83b8ea84264bb83e237f8e401d4a8e7b66019b523a4c` |
| Q4_1 | 388069056 / `78c148d1329c8338a97691b1bf83c7fc2a0b38f396bd9a5f7817f20fa53f2bb6` | 611757024 / `66bcc58d321fa2417d72998623603a4a86a488f857ee11fdffc9c542d43f8e6d` |
| Q4_K | 1039138496 / `a2d5bad578ebe5b044a6264dec9249881ef5a7191355d3219cd35dc4bd63cdf1` | 632335328 / `bcf725d253b65861320b5fdc107d56085a00425fa9b6dad3364fae2dd7d130e4` |
| Q5_K | 1054392000 / `02fa16a3dd9b8f131f3637f84fd36d7a8a037b44ebe2f85449d6d5756170d702` | 658418656 / `c3f603b8bc6cfa913b96a03a44b458ef329d8145bcf14bc48bfeae185ec3a6d0` |
| Q6_K | 1070598848 / `462ae46b6e184ba47e4fdac8cd42c4b02cbcb659049c2e51da3645035b6b6d49` | 686132192 / `da197f0fdee862bfb19bbc7605e6314b88c1c163dfb0a6cf58810517f92281d4` |

新增复验：6 个真实设备 compute::linear 用例通过，标量门禁用例按其要求在独立
`RMI_SCALAR=1` 进程通过（批量 include-ignored 未满足这个前提的那次失败不算产品回归）。
Q4_0 的 CPU/Auto/Vulkan CLI/HTTP 普通与克隆重新运行通过；禁用 ICD 后，CLI/HTTP Auto
回退输出均与 CPU WAV 相同，两个入口的严格 Vulkan 均拒绝启动。证据在
`multiq/review-20261111/{entries,fallback}/`。新 Oracle 与既有 27 份固定历史 trace 再比较，
每份 1053 个检查点仍全部通过；记录在 `review-20261111/fixed-history-results.json`。

## 仍未覆盖

未验证完整发布构建的性能峰值、其他 CPU 架构的真实模型推理、官方原生 BF16 激活语义、
2048 位置上限与跨 codec 滑动窗口的长音频、全部 23 种语言、主观音质/声音相似度指标，
以及完整多模态文本后处理桥。参考 WAV 重采样沿用现有 sinc 实现，未声称和官方前端一致；
采样 RNG 与 PyTorch 不同，固定 seed 只保证 Rust 自身复现。
