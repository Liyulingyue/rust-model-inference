# 参考实现与 Oracle 清单

> 更新于 2026-09-08。这里记录本项目实际参考过的外部实现，以及用于回归对齐的固定 Oracle。

“参考源码”只表示实现时对照过其格式、算子或模型逻辑；“Pinned Oracle”则表示仓库提交已固定，并有构建脚本、补丁或回归测试。二者不能混用：没有 pin 和可执行测试的仓库，不能作为当前结果已经对齐的证据。

## 仓库清单

| 仓库 | 对应范围 | 用途 | 固定提交 | 本地入口 / 证据 | 状态 |
|---|---|---|---|---|---|
| [ggml-org/ggml](https://github.com/ggml-org/ggml) | 通用 GGUF、量化和 CPU kernel | 对照量化格式、NEON/AVX kernel、RoPE 和 reduction 顺序 | 未固定 | [`docs/OPTIMIZATION.md`](docs/OPTIMIZATION.md) | `Reference only` |
| [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) | 文本、Embedding、Qwen3.5/Qwen3.8、Gemma 4、Qwen3-TTS 和性能基准 | 生成 token、checkpoint、logits、音频及性能对照 | 按用途固定，见下表 | `tools/parity/`、`tools/gemma4/`、`tools/tts/` 和对应 `tests/*_reference.rs` | `Pinned Oracle` |
| [hqu-little-boy/asr.cpp](https://github.com/hqu-little-boy/asr.cpp) | Qwen3-ASR | C++/GGML 行为参考 | 未固定 | 当前没有 checkout、builder 或外部 Oracle 测试；现有回归见 [`src/format/ggufrs.rs`](src/format/ggufrs.rs) | `Reference only` |
| [leejet/stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) | Z-Image Turbo | 文生图 checkpoint 与最终图像 Oracle | `97d2990807fe6d558e395f8764198d7c7e7b411c` | [`tools/z_image/build_stable_diffusion_oracle.sh`](tools/z_image/build_stable_diffusion_oracle.sh)、[`tools/z_image/stable-diffusion-z-image-trace.patch`](tools/z_image/stable-diffusion-z-image-trace.patch)、[`tests/z_image_reference.rs`](tests/z_image_reference.rs) | `Pinned Oracle` |
| [XHToken/llama.cpp](https://github.com/XHToken/llama.cpp) | Spark-X2.5 | Spark2.5 模型实现参考；仓库内旧称 `XFllama.cpp` | 未固定 | [`src/models/spark/trunk/config.rs`](src/models/spark/trunk/config.rs)、[`docs/TODO.md`](docs/TODO.md)；尚无可执行 Oracle | `Pending pin` |
| [studio-dots-ai/dots.tts](https://github.com/studio-dots-ai/dots.tts) | dots.tts-base、dots.tts.edit | 官方 Python 推理链路与逐 checkpoint/PCM Oracle | `32407a55228630475c48ecdb2c4e2c0f9c09e030` | [`tools/dots/build_dots_tts_oracle.sh`](tools/dots/build_dots_tts_oracle.sh)、[`tools/dots/run_dots_tts_oracle.py`](tools/dots/run_dots_tts_oracle.py)、[`tools/dots/dots-tts-oracle-trace.patch`](tools/dots/dots-tts-oracle-trace.patch)、[`tests/dots_tts_reference.rs`](tests/dots_tts_reference.rs) | `Pinned Oracle` |
| [AMAP-ML/DreamX-Creator](https://github.com/AMAP-ML/DreamX-Creator) / [ModelScope weights](https://modelscope.cn/models/GD-ML/DreamX-Creator) | DreamX-Creator base audio-video pipeline、2K refiner | 导出映射、Creator/UMT5/VAE/DAC/SR-DiT/upsampler/LightVAE 行为参考及 CUDA trace 入口 | `215d4cd7fbed7e161ab508ae1f85a8fee0536f62` | [`tools/dreamx/dreamx_oracle_trace.py`](../../tools/dreamx/dreamx_oracle_trace.py)、[`tests/dreamx_reference.rs`](../../tests/dreamx_reference.rs) | `Pinned reference; Oracle unrun` |

用户口头所称的 `Dif.cpp`，本项目实际使用的是 `stable-diffusion.cpp`。`XFllama.cpp` 则是仓库内沿用的本地目录名，对应的上游是 `XHToken/llama.cpp`。

## llama.cpp 固定版本

`llama.cpp` 没有一个适用于所有回归的统一版本。每个 Oracle 构建脚本中的 pin 才是对应测试的权威版本：

| 用途 | 固定提交 | 构建 / 测试入口 |
|---|---|---|
| 通用 scalar 位级回归 | `749f688fcaa4c472ec034b08cb8a907c45cfaa02` | [`tools/parity/build_llama_oracle.sh`](tools/parity/build_llama_oracle.sh)、[`tests/inference_parity.rs`](tests/inference_parity.rs) |
| Qwen3.5 / Qwen3.8-27B | `b96806d96061049a5b574269b049bf6241d63d46` | [`tools/parity/build_qwen35_oracle.sh`](tools/parity/build_qwen35_oracle.sh)、[`tests/qwen35_reference.rs`](tests/qwen35_reference.rs) |
| Gemma 4 E2B | `3173a56471c1753650cd806694145ffd6dcace67` | [`tools/gemma4/build_oracle.sh`](tools/gemma4/build_oracle.sh)、[`tests/gemma4_reference.rs`](tests/gemma4_reference.rs) |
| Qwen3-TTS Base | `201e50cc2076a20adc460c41598593c7cd7b0813` | [`tools/tts/build_qwen3_tts_oracle.sh`](tools/tts/build_qwen3_tts_oracle.sh)、[`tests/qwen3_tts_reference.rs`](tests/qwen3_tts_reference.rs) |
| Apple Silicon 性能基准（2026-08-10） | `7ba604f1cb61cd14898138e9abc0b4ff2601f180` | [`docs/OPTIMIZATION.md`](docs/OPTIMIZATION.md#rust-与-llamacpp-固定机器对比2026-08-10)；这是性能基准 pin，不是通用正确性 Oracle |
| Nemotron-3 Nano 4B Mamba2 | `96013c511b8e2dc5b6a5dbcf6bf4ad9c10d2bf77`（2026-09-11） | 本地 `references/llama.cpp/build-release/bin/llama-cli`（version 10120），或重新 build；对应 [`tests/nemotron_h_parity.rs`](tests/nemotron_h_parity.rs) 和 [`docs/parity_fixtures/nemotron_h_4b/`](docs/parity_fixtures/nemotron_h_4b/README.md) |

## 本地 checkout 约定

构建脚本接收外部 checkout 路径，并在运行前校验提交；除通用 scalar 脚本外，脚本会复制到临时目录后再打 trace patch，避免修改输入 checkout。

`references/` 只是可选的本地便利目录，不是完整依赖清单，也不应据此判断某个 Oracle 是否存在。当前仓库不会自动下载这些上游；需要执行回归时，应按上表准备对应仓库和固定提交。

## DreamX-Creator trace

`tools/dreamx/dreamx_oracle_trace.py` 直接包装固定上游的两个官方入口，不修改 checkout。`creator` 子命令记录 tokenizer IDs、text contexts、first-frame latent、joint layers 0/15/29、最终 video/audio latents 和两个 decoder 输出；`refiner` 子命令记录首尾 SR-DiT blocks、最终 SR latent 和 decoder 输入/输出。每个 tensor 转为 little-endian F32 原始数组，token IDs/attention mask 保存为 little-endian I64，shape 和原始 dtype 写入 `manifest.json`。

```bash
python3 tools/dreamx/dreamx_oracle_trace.py creator \
  --upstream-root /path/to/DreamX-Creator --out-dir /tmp/dreamx-creator-trace -- \
  <audio_video_generation/inference.py arguments>

python3 tools/dreamx/dreamx_oracle_trace.py refiner \
  --upstream-root /path/to/DreamX-Creator --out-dir /tmp/dreamx-refiner-trace -- \
  <video_refiner/inference_sr.py arguments>
```

两个官方入口都要求 CUDA。当前 Apple Silicon 验证机只执行了脚本的帮助、静态编译和无 CUDA 明确退出，没有生成 trace 或与 Rust checkpoint 比较，因此 DreamX-Creator 目前不是 `Pinned Oracle` 数值对齐状态。
