# Audio8-TTS-Realistic-Air

原生 Rust 支持官方 ArkTTS DualAR 主模型、参考音频编码和 44.1 kHz 波形解码。
模型计算位于 `src/models/arktts/`；CLI 和 HTTP 负责加载、请求校验和 WAV/PCM 序列化。
主模型复用 `Weight` 和 CPU SIMD 算子，codec 复用已有 causal Conv、transpose Conv 和 Snake。
运行时不调用 Python。原始权重及导出的 GGUF 位于用户指定的 `models/Audio8-TTS-Realistic-Air/`。

## 转换与 CLI

先按 [转换说明](../../converter/arktts/README.md) 导出配套 GGUF，再执行：

```sh
cargo build --locked --profile release-fast --bin rust-model-inference --bin rust-model-server
MODEL_DIR=/Users/gouzi/Documents/git/rust-model-inference/models/Audio8-TTS-Realistic-Air
MODEL="$MODEL_DIR/Audio8-TTS-Realistic-Air-BF16.gguf"
CODEC="$MODEL_DIR/Audio8-TTS-Realistic-Air-codec-F32.gguf"
BIN=target/release-fast/rust-model-inference
"$BIN" --tts --model "$MODEL" --mmproj "$CODEC" \
  --prompt 'Hello world.' --temperature 0 --max-tokens 96 \
  --threads 4 --compute cpu --out hello.wav
"$BIN" --tts --model "$MODEL" --mmproj "$CODEC" \
  --prompt '欢迎使用语音合成。' --ref-audio hello.wav --ref-text 'Hello world.' \
  --temperature 0 --max-tokens 96 --threads 4 --out clone.wav
```

输出为 44100 Hz、单声道 PCM16 WAV，拒绝覆盖已有输出。`--max-tokens` 表示 codec 帧预算，
每帧 2048 个采样点，遇到 EOS 提前结束；prompt 加帧预算必须在 2048 个位置内。
声音克隆必须同时提供参考 WAV 和对应 transcript。标准 PCM16 采样率
8000/11025/16000/22050/24000/32000/44100/48000/88200/96000/176400/192000 Hz
会混为单声道并重采样到 44100 Hz；输入时长最多 2048 个 codec 帧，实际还受 prompt 预算限制。
官方数值对齐使用原生 44100 Hz，其他采样率复用现有 sinc resampler，未对齐官方的重采样前端。

默认采样 `temperature=0.8, top_k=50, top_p=0.95, seed=42`，匹配官方先过滤后温度缩放及 RAS 策略；
`--temperature 0` 为 greedy。随机数实现不同，固定 seed 只保证 Rust 自身复现，不保证与 PyTorch 抽样序列相同。
语言由文本决定，`--language`、指令、CFG、编辑、JEV、聊天模板和文本/KV 调节参数明确拒绝。
该模型的 `--tts-model/--tts-mmproj` 文本后处理桥也可使用，参考声音克隆请使用独立 `--tts` 入口。

`--compute cpu` 使用 CPU；`--compute auto` / `--gpu` 明确报告并使用 CPU fallback。
强制 `--compute vulkan` 拒绝运行；未宣称 GPU 加速或 GPU 数值通过。

## HTTP

```sh
target/release-fast/rust-model-server --tts --model "$MODEL" --mmproj "$CODEC" \
  --host 127.0.0.1 --port 18081 --temperature 0 --max-tokens 96 --threads 4 --compute cpu
curl --fail http://127.0.0.1:18081/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"audio8","input":"Hello world.","response_format":"wav"}' \
  --output http.wav
```

`response_format` 仅支持 `wav` 和 `pcm`，后者为无头的 44100 Hz 单声道 signed PCM16 little-endian。
`speed` 仅支持 1；sampling 和帧预算由服务器启动参数决定，未知请求字段拒绝反序列化。
`model` 为兼容字段，服务器使用启动时加载的模型。
克隆请求同时传入 `voice`（`file:///absolute/reference.wav` 或 `data:audio/wav;base64,...`）和
`reference_text`。一次只允许一个生成任务，并发返回 429；客户端取消后仍保留生成锁直到后台工作结束。
请求体最多 16 MiB，`file://` 参考音频必须是最多 16 MiB 的普通文件。
此后端只注册语音和通用健康检查路由，文本生成/JEV 接口不可用。

## 官方对齐

Python Oracle 使用下载的原始 `configuration_arktts.py`、`modeling_arktts.py`、
`modeling_arktts_codec.py`、`processing_arktts.py`，不修改官方计算逻辑。
固定版本、哈希、实际结果和覆盖限制见 [validation.md](validation.md)。

```sh
# 仓库内 .venv：Python 3.12，torch 2.9.1，transformers 4.57.3，numpy，safetensors，soundfile
.venv/bin/python tools/oracle/arktts/official.py "$MODEL_DIR" "$MODEL_DIR/oracle" --frames 96
cargo test --locked --profile release-fast --lib arktts
cargo test --locked --profile release-fast --lib audio_speech
# 使用全新的 trace 路径；检查点文件会随 trace 一起写入磁盘。
RAYON_NUM_THREADS=4 RMI_ARKTTS_MODEL_DIR="$MODEL_DIR" \
  RMI_ARKTTS_ORACLE_DIR="$MODEL_DIR/oracle" RMI_PARITY_TRACE=/tmp/arktts-rust.jsonl \
  cargo test --locked --profile release-fast --features parity-trace \
  --test arktts_reference -- --ignored --nocapture
OPENBLAS_NUM_THREADS=1 .venv/bin/python tools/oracle/arktts/compare.py \
  "$MODEL_DIR/oracle/trace.json" /tmp/arktts-rust.jsonl --output "$MODEL_DIR/comparison.json"
.venv/bin/python -m unittest tools.oracle.arktts.test_compare tools.converter.arktts.test_convert_arktts
```

Oracle 输出目录必须不存在，避免混入旧检查点。真实权重测试默认 ignored，需要显式执行。
token IDs、prompt packing、codec 离散码和 greedy 帧精确比较；浮点计算使用数值容差，
不要求矩阵乘、归一化、激活、注意力或 codec 逐位一致。

| 浮点检查 | 门槛 |
|---|---|
| 全部检查点相对 RMSE | ≤ `1e-4` |
| 全部检查点 max_abs / max(1, reference peak) | ≤ `5e-5` |
| Slow/Fast logits 额外 max_abs | ≤ `2e-3` |
| 最终固定码流波形 | max_abs < `2e-4` 且 RMSE < `2e-5` |

比较器还检查完整检查点数量、顺序、name/layer/step/shape、数据长度和有限值，并报告 cosine。
大幅残差激活可超过 10000，统一的固定绝对阈值不适合中间层；先前 `5e-3` 的绝对检查会在
相对 RMSE 为 `6.73e-6` 的层误报，因而改用上述尺度相关检查，并保留 logits/波形的绝对约束。
