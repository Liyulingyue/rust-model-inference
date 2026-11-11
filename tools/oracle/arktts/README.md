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

`--compute cpu` 使用 CPU；`--compute auto` 在本机有 Vulkan 时实际提交可卸载的 dense
projection，无设备时回退 CPU；`--compute vulkan` 要求可用设备并严格拒绝静默回退。
ArkTTS 的设备路径是混合执行：dense projection 在 GPU，attention/KV、norm、RoPE、卷积、
codebook 和采样仍在 CPU。

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
.venv/bin/python -m tools.oracle.arktts.official "$MODEL_DIR" "$MODEL_DIR/oracle" --frames 96
cargo test --locked --profile release-fast --lib arktts
cargo test --locked --profile release-fast --lib audio_speech
# 使用全新的 trace 路径；检查点文件会随 trace 一起写入磁盘。
RAYON_NUM_THREADS=4 RMI_ARKTTS_MODEL_DIR="$MODEL_DIR" \
  RMI_ARKTTS_ORACLE_DIR="$MODEL_DIR/oracle" RMI_PARITY_TRACE=/tmp/arktts-rust.jsonl \
  cargo test --locked --profile release-fast --features parity-trace \
  --test arktts_reference -- --ignored --nocapture
.venv/bin/python -m tools.oracle.arktts.compare \
  "$MODEL_DIR/oracle/trace.json" /tmp/arktts-rust.jsonl --output "$MODEL_DIR/comparison.json"
.venv/bin/python -m unittest tools.oracle.arktts.test_compare tools.converter.arktts.test_convert_arktts
```

Oracle 输出目录必须不存在，避免混入旧检查点。真实权重测试默认 ignored，需要显式执行。
Tokenizer/prompt IDs 与形状契约精确比较；原精度基线核对 codec 离散码和 greedy 帧，
半精度/量化另记录码流分歧。浮点计算使用数值容差，
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

## GPU 与多精度矩阵（2026-11-11）

转换器和 loader 覆盖九档配对格式：`F32`、`F16`、`BF16`、`Q8_0`、`Q4_0`、`Q4_1`、
`Q4_K`、`Q5_K`、`Q6_K`。每档均用同一量化 GGUF 通过独立 gguf-py 反量化 + 官方模型
进行 1053 个固定历史检查点，再在 CPU/Auto/严格 Vulkan 各跑一次。以下为 M3 Max ARM64 结果，目录为
`/Users/gouzi/Documents/git/rust-model-inference/models/Audio8-TTS-Realistic-Air/multiq/`。

| 格式 | CPU / Auto / Vulkan | 最差 checkpoint relative RMSE | 固定码流 waveform max_abs | 严格 Vulkan dense calls |
| --- | --- | ---: | ---: | ---: |
| F32 | 通过 / 通过 / 通过 | 3.27e-5 | 1.58e-6 | 4308 |
| F16 | 通过 / 通过 / 通过 | 5.00e-3 | 2.06e-4 | 4308 |
| BF16 | 通过 / 通过 / 通过 | 2.97e-5 | 1.46e-6 | 4308 |
| Q8_0 | 通过 / 通过 / 通过 | 5.85e-2 | 1.50e-3 | 4308 |
| Q4_0 | 通过 / 通过 / 通过 | 7.87e-2 | 5.99e-3 | 4308 |
| Q4_1 | 通过 / 通过 / 通过 | 8.69e-2 | 1.58e-6 | 4308 |
| Q4_K | 通过 / 通过 / 通过 | 5.85e-2 | 1.55e-6 | 4308 |
| Q5_K | 通过 / 通过 / 通过 | 6.28e-2 | 1.35e-6 | 4308 |
| Q6_K | 通过 / 通过 / 通过 | 6.60e-2 | 8.93e-4 | 4308 |

量化门槛在 held-out suite 前冻结为 relative RMSE ≤ 0.2、normalized max_abs ≤ 0.15、
logits max_abs ≤ 5、waveform max_abs < 0.006 且 RMSE < 0.001；F16 使用
0.01 / 0.01 / 0.25 / 0.001 / 0.0001；F32/BF16 使用上表前的浮点门槛。它们是数值
容差，不是逐位比较；块量化和半精度的舍入会改变激活量化 bin，greedy 码流也可能分叉。

真实入口矩阵也已完成：九档 × `cpu/auto/vulkan` 均分别通过 CLI 普通生成、CLI 参考音频
克隆、HTTP WAV、HTTP 裸 PCM；每个 HTTP WAV/PCM 与对应 CLI 输出逐字节相等。每个 Auto
组合记录 1188 次 Vulkan dense calls + 7548 次主机 linear calls，严格 Vulkan 记录 8736
次设备 calls、0 次主机 linear calls（卷积等非 linear host 操作仍存在）。证据由
`tools/oracle/arktts/entries.py`、`projections.py`、`same_weight.py` 生成。

完整的逐组合状态与证据路径见 [validation.md](validation.md#覆盖矩阵与改动边界)。
导出格式集合来自共享 `quantize_gguf.BLOCK_FORMATS` 加 F32；没有为 ArkTTS 新增量化算法。
K 格式保留不满足 256 元素块宽的主模型行和 codec 1216 宽行，不能把这些混合文件描述为全图低比特。
Q4_1 的最终导出另外保留 codec `quantizer.post_module.*` 解码 Transformer 为 F32，
其主模型和 60 个 codec 编码器投影仍为 Q4_1。初版在 x86 固定码流波形超限，未放宽门槛；
修正后已在 M3 Max 和 RADV NAVI31 的 CPU/Auto/Vulkan 数值及 CLI/HTTP 矩阵复验通过。
Q4_1 新资产与原始失败证据分别保存在 `multiq-reviewed/q4_1/` 和原 `multiq/q4_1/`；
最终文件身份见 `final-inventory-20261111.json` 和 [validation.md](validation.md)。

同日 Linux x86_64 的 AMD EPYC 9334 / RADV NAVI31 也完成了最终九档 × 三后端矩阵：
27 组各 1053 个固定历史检查点、同输入 projection，以及 CLI/HTTP 普通/克隆 WAV/PCM
全部通过。禁用 ICD 时 Q4_0 Auto 输出与 CPU 相同，严格 Vulkan 的两个入口均拒绝启动。
远端证据已归档到 `MODEL_DIR/remote-20261111/`；详细指标、环境和 codec 哈希差异见
[Linux 复验记录](validation.md#linux-x86_64--radv-navi31-复验2026-11-11)。

### 完整矩阵复现

在仓库根 `.venv` 安装上述 Oracle 依赖及 `gguf==0.17.1`，使用全新的 `RUN_DIR`，避免覆盖
现有权重和 trace。CPU/Auto/Vulkan 共用每档相同的 GGUF；以下命令还分别运行真实 CLI/HTTP：

```sh
set -eu
MODEL_DIR=/Users/gouzi/Documents/git/rust-model-inference/models/Audio8-TTS-Realistic-Air
RUN_DIR="$MODEL_DIR/reproduce-$(date +%Y%m%dT%H%M%S)"
MATRIX_DIR="$RUN_DIR/multiq"
mkdir -p "$MATRIX_DIR"
cargo build --locked --profile release-fast --features vulkan,parity-trace \
  --bin rust-model-inference --bin rust-model-server
.venv/bin/python -m tools.oracle.arktts.official "$MODEL_DIR" \
  "$RUN_DIR/oracle-validation-base" --suite validation --frames 4
export RAYON_NUM_THREADS=4 RMI_ARKTTS_MODEL_DIR="$MODEL_DIR"
for fmt in f32 f16 bf16 q8_0 q4_0 q4_1 q4_k q5_k q6_k; do
  .venv/bin/python -m tools.converter.arktts.convert_arktts "$MODEL_DIR" \
    --out-dir "$MATRIX_DIR/$fmt" --format "$fmt" --codec-format "$fmt"
  upper=$(printf '%s' "$fmt" | tr '[:lower:]' '[:upper:]')
  export RMI_ARKTTS_MAIN="$MATRIX_DIR/$fmt/Audio8-TTS-Realistic-Air-$upper.gguf"
  export RMI_ARKTTS_CODEC="$MATRIX_DIR/$fmt/Audio8-TTS-Realistic-Air-codec-$upper.gguf"
  export RMI_ARKTTS_ORACLE_DIR="$MATRIX_DIR/$fmt/oracle"
  .venv/bin/python -m tools.oracle.arktts.official "$MODEL_DIR" "$RMI_ARKTTS_ORACLE_DIR" \
    --suite validation --frames 4 --history "$RUN_DIR/oracle-validation-base/cases.json" \
    --main-gguf "$RMI_ARKTTS_MAIN" --codec-gguf "$RMI_ARKTTS_CODEC"
  profile=quantized
  case "$fmt" in f32|bf16) profile=float ;; f16) profile=f16 ;; esac
  for backend in cpu auto vulkan; do
    export RMI_ARKTTS_COMPUTE="$backend"
    trace="$MATRIX_DIR/$fmt/$backend.jsonl"
    RMI_ARKTTS_REPLAY=1 RMI_PARITY_TRACE="$trace" RMI_COMPUTE_TRACE=1 \
      cargo test --locked --profile release-fast --features vulkan,parity-trace \
      --test arktts_reference -- --ignored --nocapture > "$MATRIX_DIR/$fmt/$backend.log" 2>&1
    .venv/bin/python -m tools.oracle.arktts.compare "$RMI_ARKTTS_ORACLE_DIR/trace.json" \
      "$trace" --profile "$profile" --output "$MATRIX_DIR/$fmt/$backend-metrics.json"
    # No trace I/O: report free-running greedy differences and per-phase wall time.
    cargo test --locked --profile release-fast --features vulkan,parity-trace \
      --test arktts_reference -- --ignored --nocapture > "$MATRIX_DIR/$fmt/$backend-greedy.log" 2>&1
  done
done
.venv/bin/python -m tools.oracle.arktts.projections "$MATRIX_DIR"
for fmt in f32 f16 bf16 q8_0 q4_0 q4_1 q4_k q5_k q6_k; do
  for backend in cpu auto vulkan; do
    RMI_ARKTTS_PROJECTIONS="$MATRIX_DIR/$fmt/projections.json" RMI_ARKTTS_COMPUTE="$backend" \
      cargo test --locked --profile release-fast --features vulkan,parity-trace --lib \
      same_input_projections_match_independent_oracle -- --ignored --nocapture
  done
done
.venv/bin/python -m tools.oracle.arktts.entries "$MATRIX_DIR" --output-dir "$RUN_DIR/entries"
```

量化的同权重 Oracle 通过独立 gguf-py 反量化，并适配共享算子的 activation QDQ；官方
模型源码保持只读。固定历史检查计算误差，自由 greedy 报告离散分叉，两者分别留证。
性能记录见 [`performance.json`](performance.json) 和 `docs/develop/ARKTTS.md`；它是同机
`release-fast`、4 线程、4 帧参考，不能当作发布构建吞吐或长音频质量结论。
