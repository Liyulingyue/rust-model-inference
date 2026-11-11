# Sophon-Realtime-32B-Tag

状态：**Experimental，部分适配**。F32/F16/BF16 thinker 八步、F32 vision encoder 的 32 个 checkpoint、F32/BF16 音频边界与短语音组件有独立数值证据，F32 CLI 已输出有效 WAV；新 lossless 图像 CLI/HTTP 的 9 项入口检查通过。量化失败和未验证入口见 [逐项记录](../../tools/oracle/sophon_realtime/validation.md) 与 [550 行覆盖矩阵](../../tools/oracle/sophon_realtime/coverage.json)。这不代表完整双工、原生 FP8 或完整 GPU 支持。

## 远端准备

源模型固定为 ModelScope `AdventistsAI/Sophon-Realtime-32B-Tag` revision `990d6a74bbf15c1765a6f7736fa839f423fd5b59`。直接在推理机下载，按 [manifest](../../tools/oracle/sophon_realtime/modelscope-files.json) 校验 size/SHA256。源文件 64 个、42,318,478,677 bytes。检查 `models/` 实际所在文件系统的空间，F32 thinker 导出本身约 131 GB。

验证使用授权 Linux x86_64 机器，SSH 端口以当前会话为准。模型、GGUF 与二进制 checkpoint 全程留在远端。远端能访问 ModelScope 时不经过 Mac；只有 laptop 能访问时用 `curl --fail --location ... | ssh ... 'cat > ...partial'` 转发，启用 pipefail，在远端校验后再改名。不要在 Mac 建模型缓存。

转换使用 [现有转换器](../../tools/converter/sophon_realtime/README.md)，支持 F32/F16/BF16/Q8_0/Q4_0/Q4_1/Q4_K/Q5_K/Q6_K。原始 thinker 是带 channel scale 的 FP8 权重与动态 FP8 激活；GGUF 是 weight-only 路径，不能把它称为原生 FP8 加速。HiFT 在九种 requested 格式中均实际保存为 F32。

## 模型与组件

| 文件 | 内容 | 使用 |
| --- | --- | --- |
| thinker-*.gguf | 显式 sophon_realtime 契约，复用 Qwen3-VL decoder | 文本、视觉/音频理解及 JEV |
| mmproj-F32.gguf；本轮新资产另存 vision-lossless-mmproj-F32.gguf | Qwen3-VL 视觉塔、merger、DeepStack；纯 F32 计算由 metadata 显式 opt-in | thinker --image；新 lossless 入口待复测 |
| audio-*.gguf | 18 层 Qwen3-ASR encoder + stack-2 Ultravox projector | thinker --audio |
| talker-*.gguf | 带标签的 CosyVoice3 Qwen2 speech decoder | --tts 或 --tts-model |
| code2wav-*.gguf | Flow + F32 HiFT + 六个固定 voice prompt | --mmproj 或 --tts-mmproj |

当前 `--mmproj` 选择视觉或音频组件；同次请求组合音频与图像/视频尚未接入。组件版本、architecture、tensor shape/type 和 voice inventory 都会校验，不能拿其他模型的 mmproj 替代。

视觉转换器现写入布尔 `clip.vision.preserve_f32=true`，复用共享 encoder 已有的 lossless 路径，保留 F32 线性权重与激活。历史 `mmproj-F32.gguf` 只有 F32 存储语义，实际对齐的线性层会预打包 Q8 权重并动态量化输入，4304 宽的 FFN-down 输入走 F32 fallback；缺省/false metadata 的行为保留。新资产 `vision-lossless-mmproj-F32.gguf`（SHA256 `38c73644…`）与历史资产（`75a9be7a…`）的 352 个 tensor payload 相同，只新增 opt-in metadata；没有覆盖历史文件。

将 Flow、HiFT 和 voices 打包：

```sh
.venv/bin/python -m tools.converter.sophon_realtime.bundle \
  models/Sophon-Realtime-32B-Tag \
  models/Sophon-Realtime-32B-Tag-GGUF/flow-F32.gguf \
  models/Sophon-Realtime-32B-Tag-GGUF/hift-F32.gguf \
  models/Sophon-Realtime-32B-Tag-GGUF/code2wav-F32.gguf
```

## CLI

使用仓库原有 `rust-model-inference`，不新增 executable。以下路径均位于远端 checkout；F32 thinker 需要充足内存或 mmap 文件缓存，否则分页会很慢。

```sh
BIN=target/release-fast/rust-model-inference
MODELS=models/Sophon-Realtime-32B-Tag-GGUF
$BIN --model "$MODELS/thinker-F32.gguf" --compute cpu --kv-cache f32 \
  --prompt '你好，请简短介绍自己。' --temperature 0 --max-tokens 8 --threads 4
$BIN --model "$MODELS/thinker-F32.gguf" --mmproj "$MODELS/audio-F32.gguf" \
  --audio input.wav --prompt '请描述听到的内容。' --compute cpu --temperature 0 --threads 4
$BIN --tts --model "$MODELS/talker-F32.gguf" --mmproj "$MODELS/code2wav-F32.gguf" \
  --voice brand_zh --prompt '你好，欢迎使用语音助手。[breath]' \
  --out output.wav --compute cpu --temperature 0 --max-tokens 128 --threads 4
$BIN --model "$MODELS/thinker-F32.gguf" --mmproj "$MODELS/audio-F32.gguf" \
  --audio input.wav --prompt '请简短回答。' --tts-model "$MODELS/talker-F32.gguf" \
  --tts-mmproj "$MODELS/code2wav-F32.gguf" --voice brand_zh --out answer.wav \
  --compute cpu --temperature 0 --max-tokens 32 --threads 4
$BIN --model "$MODELS/thinker-F32.gguf" --mmproj "$MODELS/audio-F32.gguf" \
  --audio input.wav --jev --jev-context '判断音频。' --jev-question '是否有人讲话？' \
  --jev-option 是 --jev-option 否 --jev-output json --compute cpu --threads 4
```

Sophon 默认启用其固定 Jinja 模板。文本、图像、音频 CLI JEV 均支持 `--jev-multi` / `--jev-block`，并复用已配置的 resident runtime、KV 和上下文限制；HTTP 音频 JEV 保持单模式协议。`--tts-model` 与 `--tts-mmproj` 必须成对。语音理解输入统一为 mono 16 kHz；输出为 mono 24 kHz WAV/PCM。speech-to-speech 当前先完成文本生成，再合成语音，尚无双工会话调度。流式 TTS 每十个 speech token 推进，固定提交旧 mel，EOS 释放尾部。

## HTTP

原有 `rust-model-server` 使用相同加载参数。thinker 可带 `--tts-model` 和 `--tts-mmproj` 挂接语音输出；独立 talker 使用 `--tts --model talker-*.gguf --mmproj code2wav-*.gguf`。路由与 payload：

| 路由 | 输入 | 输出 |
| --- | --- | --- |
| /v1/chat/completions、/v1/messages、/v1/responses | 现有文本/图像协议 | 文本 JSON 或 SSE |
| /v1/audio/chat | {audio: raw-base64-audio, prompt, max_tokens, temperature, stream, output_audio} | 文本/token_ids；output_audio 时 batch 带 base64 WAV，stream 增加 audio 事件 |
| /v1/jev/score、/v1/jev/image | 现有 JEV 协议 | 候选概率 |
| /v1/jev/grouped、/v1/jev/image_grouped | 现有 groups 协议；图像加 image_url | 每组独立归一化的候选概率 |
| /v1/jev/audio | {audio: raw-base64-audio, context, questions:[{text,options}], positive?} | 与 CLI 相同的单模式候选概率 |
| /v1/audio/speech | {input, voice?, response_format: wav/pcm, stream?} | WAV/raw PCM；stream 为 PCM16 base64 SSE |

音频不接受 URL 或本机文件路径。先取得并发许可，再按上下文和十分钟绝对上限限制解码后的样本；无 ffmpeg 时仅接受 16 kHz mono PCM16 WAV，同样先检查预算。输入错误在开始 SSE 前返回 400；音频对话/JEV 的运行失败返回 500，独立 speech batch 延用 422，已开始流的失败使用 error 事件。batch 与 stream 断连均停止 text sink，随后每个 speech token、Flow block/Euler step 和 HiFT stage/resblock 检查取消；音频 encoder/prefill 内部尚未新增逐层取消。

## 对齐、后端与限制

Tokenizer、shape、控制 token、版本和 stream ordering 精确检查。F32/SIMD/F16/BF16/量化以事先固定的 rRMSE、cosine、max_abs 等指标比较，不要求有损路径逐位相同。原始 FP8 激活损失、导出精度损失和同文件实现误差分别报告。

九种 thinker 导出均有 CPU/F32-KV 数值记录，分别绑定其实际二进制和主机：F32/F16/BF16 通过，Q8_0/Q4_0/Q4_1/Q4_K/Q5_K/Q6_K 均未过原块量化门槛。替换机 31344 上的新 Q6_K 比较有 7/8 行失败，最大 rRMSE `0.1524552110`、最低 cosine `0.9883425540`；同文件 greedy 8/8 一致。旧机 33367 的最后一项 Q6 结果仍为未知，不能用新运行倒推旧进程结果。CPU 数值记录不代替 HTTP、F16 KV 或 GPU 验证，详见 [thinker-results.json](../../tools/oracle/sophon_realtime/thinker-results.json)。

CPU 走仓库线程池与 SIMD/量化算子。Auto 复用现有分发，须按真实 backend/submission 证据判断；回退 CPU 不代表 GPU 通过。现有 Qwen3-VL resident decoder 已实现 interleaved M-RoPE/DeepStack，设备路径要求 F16 KV 和可用权重格式。最新 Q8_0 文本实测有 714 次设备提交，但未过现有 CPU/GPU 数值门槛；不能宣称完整 GPU 支持。音频 block attention/frontend、causal convolution 和 HiFT spectral 的严格设备路径尚缺共享算子，明确拒绝。后续共享层及顺序见 validation；未以没 GPU 硬件掩盖缺 kernel。

Q8_0 thinker 配 F16 KV 的 CPU 文本、图像及单/分组 JEV 已跑通，CLI/HTTP 概率差为 0；图像记录使用历史 F32 存储、内部 Q8 预打包视觉路径，不能冒充纯 F32 数值对齐。量化音频/Flow 存在未过原门槛的结果；F16 talker 有极小 margin 的 greedy 分歧，部分 K talker 在 128-token 边界未 EOS。长 HiFT 的中间检查点仍有失败，尽管最终 waveform 门槛通过。没有广泛听感、任务准确率或真实 full-duplex 质量结论。

新 lossless F32 vision 已用官方 Transformers 4.57.6 `Qwen3VLVisionModel`、相同导出权重、CPU 四线程独立验证，不加载 thinker。固定 32×32 红图按源配置放大到 256×256，得到 64 行、每行 5120 维；1152 宽、27 层视觉塔的 input、27 个 block、三层 DeepStack（8/16/24）及最终投影共 **32/32 通过**，最差 rRMSE `1.26997018e-5`、最低 cosine `0.9999999999195905`，无首分叉；原门槛仍为 `0.001 / 0.999999`。历史默认资产的投影与三层 DeepStack 位哈希也在改动前后完全一致。证据见 [vision-results.json](../../tools/oracle/sophon_realtime/vision-results.json)；其他模型的真实 vision 资产缺失，未声称其真实回归通过。新 lossless 图像组件已配合 Q8_0 thinker/F16 KV，在最新构建通过 CLI/HTTP 的 9 项检查：图像 chat、单/分组 JEV、batch/SSE 与上下文预检；两项 CLI/HTTP 概率差均为 0。Auto/Vulkan vision 数值路径仍待验证；独立 encoder 数值和组合入口功能分别留证。

[performance.json](../../tools/oracle/sophon_realtime/performance.json) 保留同机实测。F32 thinker 八步曾耗时 802 秒，发生分页和资源竞争，不能用作常规吞吐或容量。初始独立 TTS 77 speech tokens / 73,920 samples 用时约 78 秒；release-fast 数字不代表 release 部署性能。严格 Q8_0/F16-KV 设备检查耗时 54.55 秒后因数值门槛失败，含上传与 CPU 校验开销，不能作为吞吐；同 KV 参考性能对比仍待完成。
