# Sophon-Realtime-32B-Tag

状态：**Experimental，部分适配**。F32 thinker 八步、F32/BF16 音频边界与短语音组件有独立数值证据，F32 CLI 已输出有效 WAV；量化失败和未验证入口见 [逐项记录](../../tools/oracle/sophon_realtime/validation.md) 与 [543 行覆盖矩阵](../../tools/oracle/sophon_realtime/coverage.json)。这不代表完整双工、原生 FP8 或完整 GPU 支持。

## 远端准备

源模型固定为 ModelScope `AdventistsAI/Sophon-Realtime-32B-Tag` revision `990d6a74bbf15c1765a6f7736fa839f423fd5b59`。直接在推理机下载，按 [manifest](../../tools/oracle/sophon_realtime/modelscope-files.json) 校验 size/SHA256。源文件 64 个、42,318,478,677 bytes。检查 `models/` 实际所在文件系统的空间，F32 thinker 导出本身约 131 GB。

验证使用授权 Linux x86_64 机器，SSH 端口以当前会话为准。模型、GGUF 与二进制 checkpoint 全程留在远端。远端能访问 ModelScope 时不经过 Mac；只有 laptop 能访问时用 `curl --fail --location ... | ssh ... 'cat > ...partial'` 转发，启用 pipefail，在远端校验后再改名。不要在 Mac 建模型缓存。

转换使用 [现有转换器](../../tools/converter/sophon_realtime/README.md)，支持 F32/F16/BF16/Q8_0/Q4_0/Q4_1/Q4_K/Q5_K/Q6_K。原始 thinker 是带 channel scale 的 FP8 权重与动态 FP8 激活；GGUF 是 weight-only 路径，不能把它称为原生 FP8 加速。HiFT 在九种 requested 格式中均实际保存为 F32。

## 模型与组件

| 文件 | 内容 | 使用 |
| --- | --- | --- |
| thinker-*.gguf | 显式 sophon_realtime 契约，复用 Qwen3-VL decoder | 文本、视觉/音频理解及 JEV |
| mmproj-F32.gguf | Qwen3-VL 视觉塔、merger、DeepStack | thinker --image |
| audio-*.gguf | 18 层 Qwen3-ASR encoder + stack-2 Ultravox projector | thinker --audio |
| talker-*.gguf | 带标签的 CosyVoice3 Qwen2 speech decoder | --tts 或 --tts-model |
| code2wav-*.gguf | Flow + F32 HiFT + 六个固定 voice prompt | --mmproj 或 --tts-mmproj |

当前 `--mmproj` 选择视觉或音频组件；同次请求组合音频与图像/视频尚未接入。组件版本、architecture、tensor shape/type 和 voice inventory 都会校验，不能拿其他模型的 mmproj 替代。

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
  --jev-option 是 --jev-option 否 --jev-output-json --compute cpu --threads 4
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

CPU 走仓库线程池与 SIMD/量化算子。Auto 复用现有分发，须按真实 backend/submission 证据判断；回退 CPU 不代表 GPU 通过。现有 Qwen3-VL resident decoder 已实现 interleaved M-RoPE/DeepStack，设备路径要求 F16 KV 和可用权重格式，仍需本模型实测。音频 block attention/frontend、causal convolution 和 HiFT spectral 的严格设备路径尚缺共享算子，明确拒绝。后续共享层及顺序见 validation；未以没 GPU 硬件掩盖缺 kernel。

量化音频/Flow 存在未过原门槛的结果；F16 talker 有极小 margin 的 greedy 分歧，部分 K talker 在 128-token 边界未 EOS。长 HiFT 的中间检查点仍有失败，尽管最终 waveform 门槛通过。没有广泛听感、任务准确率或真实 full-duplex 质量结论。

[performance.json](../../tools/oracle/sophon_realtime/performance.json) 保留同机实测。F32 thinker 八步曾耗时 802 秒，发生分页和资源竞争，不能用作常规吞吐或容量。初始独立 TTS 77 speech tokens / 73,920 samples 用时约 78 秒；release-fast 数字不代表 release 部署性能。完整 GPU 与同机参考性能对比尚未运行。
