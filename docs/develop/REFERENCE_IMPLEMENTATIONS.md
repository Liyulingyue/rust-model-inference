# 参考实现与 Oracle 清单

> 更新于 2026-09-22。这里记录本项目实际参考过的外部实现，以及用于回归对齐的固定 Oracle。

“参考源码”只表示实现时对照过其格式、算子或模型逻辑；“Pinned Oracle”则表示仓库提交已固定，并有构建脚本、补丁或回归测试。二者不能混用：没有 pin 和可执行测试的仓库，不能作为当前结果已经对齐的证据。

## 状态分级

| 状态 | 含义 |
|---|---|
| `Reference only` | 实现时翻阅过源码/文档但未固定 commit，无可执行回归 |
| `Pinned Oracle` | 上游 commit 固定，有构建脚本 + trace patch + `tests/*_reference.rs`，可端到端跑通并 bit-exact 对照 |
| `Pinned numpy oracle` | 不依赖 C++ fork，由 `numpy` / `gguf` / `torch` 直接 replay GGUF，按 byte / f32 bits 对比（如 Breeze TTS、MiniCPM5、LFM2-MoE）。构建成本远低于 C++ Oracle，但覆盖不到 CPU 算子 bug；与 C++ Oracle 共存时视为补充 |
| `Pending pin` | 已选型（fork 已知或代码路径已确定），但尚未固定 commit 或尚无构建脚本 |
| `Pinned reference; Oracle unrun` | 上游 commit 已固定 + trace patch 已就位 + `tests/*_reference.rs` 已存在，但当前验证机没有跑过端到端 trace 收集（缺少 CUDA / 模型文件等） |

`Pinned Oracle` / `Pinned numpy oracle` 不等于「全部量化格式都已对齐」；只保证「已验证格式」列出的组合。

## 仓库清单

| 仓库 | 对应范围 | 用途 | 固定提交 | 本地入口 / 证据 | 状态 |
|---|---|---|---|---|---|
| [ggml-org/ggml](https://github.com/ggml-org/ggml) | 通用 GGUF、量化和 CPU kernel | 对照量化格式、NEON/AVX kernel、RoPE 和 reduction 顺序 | 未固定 | [`docs/develop/OPTIMIZATION.md`](OPTIMIZATION.md) | `Reference only` |
| [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) | 文本、Embedding、Jina Embeddings v5 Omni、Qwen3.5/Qwen3.8、Gemma 4、Qwen3-TTS、Qwen-Drive、Nemotron-3 Nano 4B 和性能基准 | 生成 token、checkpoint、logits、Embedding、音频及性能对照 | 按用途固定，见 §「llama.cpp 固定版本」 | `tools/oracle/shared/`、`tools/oracle/gemma4/`、`tools/oracle/qwen3_tts/`、`tools/oracle/qwen_drive/` 和对应 `tests/*_reference.rs` | `Pinned Oracle` |
| [hqu-little-boy/asr.cpp](https://github.com/hqu-little-boy/asr.cpp) | Qwen3-ASR | C++/GGML 行为参考 | 未固定 | 当前没有 checkout、builder 或外部 Oracle 测试；现有回归见 [`src/format/ggufrs.rs`](../../src/format/ggufrs.rs) | `Reference only` |
| [leejet/stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) | Z-Image Turbo | 文生图 checkpoint 与最终图像 Oracle | `97d2990807fe6d558e395f8764198d7c7e7b411c` | [`tools/oracle/z_image/build_stable_diffusion_oracle.sh`](../../tools/oracle/z_image/build_stable_diffusion_oracle.sh)、[`tools/oracle/z_image/stable-diffusion-z-image-trace.patch`](../../tools/oracle/z_image/stable-diffusion-z-image-trace.patch)、[`tests/z_image_reference.rs`](../../tests/z_image_reference.rs) | `Pinned Oracle` |
| [XHToken/llama.cpp](https://github.com/XHToken/llama.cpp) | Spark-X2.5 | Spark2.5 模型实现参考（仓库历史沿用名 `XFllama.cpp`，对应上游即 XHToken fork） | 未固定 | [`src/models/spark/trunk/config.rs`](../../src/models/spark/trunk/config.rs)、[`docs/TODO.md`](../TODO.md)；尚无可执行 Oracle | `Pending pin` |
| [studio-dots-ai/dots.tts](https://github.com/studio-dots-ai/dots.tts) | dots.tts-base、dots.tts.edit | 官方 Python 推理链路与逐 checkpoint/PCM Oracle；LLM 部分由 `arch=qwen2` GGUF 驱动，mmproj 由 `arch=dotstts` 驱动 | `32407a55228630475c48ecdb2c4e2c0f9c09e030` | [`tools/oracle/dots/build_dots_tts_oracle.sh`](../../tools/oracle/dots/build_dots_tts_oracle.sh)、[`tools/oracle/dots/run_dots_tts_oracle.py`](../../tools/oracle/dots/run_dots_tts_oracle.py)、[`tools/oracle/dots/dots-tts-oracle-trace.patch`](../../tools/oracle/dots/dots-tts-oracle-trace.patch)、[`tests/dots_tts_reference.rs`](../../tests/dots_tts_reference.rs) | `Pinned Oracle` |
| [AMAP-ML/DreamX-Creator](https://github.com/AMAP-ML/DreamX-Creator) / [ModelScope weights](https://modelscope.cn/models/GD-ML/DreamX-Creator) | DreamX-Creator base audio-video pipeline、2K refiner | 导出映射、Creator/UMT5/VAE/DAC/SR-DiT/upsampler/LightVAE 行为参考及 CUDA trace 入口 | `215d4cd7fbed7e161ab508ae1f85a8fee0536f62` | [`tools/oracle/dreamx/dreamx_oracle_trace.py`](../../tools/oracle/dreamx/dreamx_oracle_trace.py)、[`tests/dreamx_reference.rs`](../../tests/dreamx_reference.rs) | `Pinned reference; Oracle unrun` |
| Qwen-Drive-1.0-4B 原始权重（仓库闭源，按 `tools/oracle/qwen_drive/README.md` 表里的 SHA256 + size 锁定；VLM 部分仍是 `arch=qwen2vl/qwen3vl`，planner 是独立 head） | Qwen-Drive-1.0-4B VLM + planner-SFT + planner-RL + perception | VLM（mtmd qwen2vl/qwen3vl mtmd-debug）+ 规划器（PyTorch oracle） | `b96806d96061049a5b574269b049bf6241d63d46`（VLM 同 Qwen3.5） | [`tools/oracle/qwen_drive/build_qwen_drive_vlm_oracle.sh`](../../tools/oracle/qwen_drive/build_qwen_drive_vlm_oracle.sh)、[`tools/oracle/qwen_drive/qwen_drive_vlm_trace.patch`](../../tools/oracle/qwen_drive/qwen_drive_vlm_trace.patch)、[`tools/oracle/qwen_drive/qwen_drive_oracle.py`](../../tools/oracle/qwen_drive/qwen_drive_oracle.py)、[`tools/oracle/qwen_drive/README.md`](../../tools/oracle/qwen_drive/README.md)、[`tests/qwen_drive_vlm_reference.rs`](../../tests/qwen_drive_vlm_reference.rs)、[`tests/qwen_drive_planner_reference.rs`](../../tests/qwen_drive_planner_reference.rs) | `Pinned Oracle` |
| [microsoft/VibeVoice-ASR-Streaming-7B](https://huggingface.co/microsoft/VibeVoice-ASR-Streaming-7B) 原始 BF16 safetensors | VibeVoice ASR（arch=`qwen2` LLM + `clip` mmproj + `vibevoice_asr` projector） | 语音前端（ConvNeXt + speech connector）+ Qwen2.5 decoder | 未固定 pin；按 SHA 锁定原始权重 | [`tools/oracle/vibevoice/vibevoice_oracle.py`](../../tools/oracle/vibevoice/vibevoice_oracle.py)（speech-frontend）、[`tools/oracle/vibevoice/vibevoice_llm_oracle.py`](../../tools/oracle/vibevoice/vibevoice_llm_oracle.py)（Qwen2.5 decoder）、[`tests/vibevoice_encoder_reference.rs`](../../tests/vibevoice_encoder_reference.rs)、[`tests/vibevoice_llm_reference.rs`](../../tests/vibevoice_llm_reference.rs) | `Pinned numpy oracle` |
| LFM2-8B-A1B Q8_0 GGUF + llama.cpp layer dump | LFM2-MoE 8B-A1B | 与 llama.cpp 前 6 token 对齐（MoE 平局处分叉） | 跟随 llama.cpp 通用 pin `749f688...02`（layer dump） | [`tools/oracle/shared/lfm2moe_reference.py`](../../tools/oracle/shared/lfm2moe_reference.py)（numpy step-by-step F64 ground truth）、[`tools/oracle/shared/lfm2moe_layer_cmp.py`](../../tools/oracle/shared/lfm2moe_layer_cmp.py)（与 llama.cpp layer dump 对比） | `Pinned numpy oracle` |
| MiniCPM5-1B Q8_0 GGUF（arch=`llama`）+ llama.cpp replay | MiniCPM5-1B | 8/8 greedy token 与 llama.cpp 一致（见 `SUPPORTED_MODELS.md`） | `749f688fcaa4c472ec034b08cb8a907c45cfaa02`（同通用 scalar pin，via `dump_tokens_oracle`） | [`tools/oracle/shared/minicpm5_reference.py`](../../tools/oracle/shared/minicpm5_reference.py)（numpy）、[`tools/oracle/shared/dump_tokens_oracle.cpp`](../../tools/oracle/shared/dump_tokens_oracle.cpp)（llama.cpp replay） | `Pinned numpy oracle` |
| Breeze-TTS-2 原始 BF16 checkpoint（仓库闭源，按 `tools/oracle/breeze/README.md` SHA + size 锁定） | Breeze-TTS-2（`breeze` + `breeze_audio`） | 基于原 checkpoint 的逐层 F32 bit-level 对比 | 未固定 commit；按 SHA 锁定原始 checkpoint | [`tools/oracle/breeze/compare_breeze_trace.py`](../../tools/oracle/breeze/compare_breeze_trace.py)（逐层 little-endian F32 bits 对比）、[`tools/oracle/breeze/README.md`](../../tools/oracle/breeze/README.md) | `Pinned numpy oracle` |
| NeoHorse-1-4B / 1-9B 原始 BF16 safetensors（仓库闭源，按 `tools/converter/neohorse/README.md` 锁定；架构 `Qwen3_5ForCausalLM`，复用 qwen35 入口） | NeoHorse | 转换 + qwen35 文本推理；LLM 部分走 Qwen3.5 pin，无独立 Oracle | 跟随 qwen35 pin `b96806d...46`（converter） | [`tools/converter/neohorse/convert_neohorse.py`](../../tools/converter/neohorse/convert_neohorse.py)、[`tools/converter/neohorse/README.md`](../../tools/converter/neohorse/README.md) | `Reference only`（依赖 Qwen3.5 oracle） |

> 注：「XFllama.cpp」是仓库历史沿用的本地目录名（与 Spark-X2.5 实现参考的 fork 路径相关），对应上游即 `XHToken/llama.cpp`，仅在 XHToken 行备注里出现一次；本项目主 Oracle 全部走 `ggml-org/llama.cpp` 固定 commit，不使用 XHToken 的 binary。

## llama.cpp 固定版本

`llama.cpp` 没有一个适用于所有回归的统一版本。每个 Oracle 构建脚本中的 pin 才是对应测试的权威版本：

| 用途 | 固定提交 | 构建 / 测试入口 |
|---|---|---|
| 通用 scalar 位级回归（Q4_0、Q4_K_M 等基础回归） | `749f688fcaa4c472ec034b08cb8a907c45cfaa02` | [`tools/oracle/shared/build_llama_oracle.sh`](../../tools/oracle/shared/build_llama_oracle.sh)、[`tests/inference_parity.rs`](../../tests/inference_parity.rs)；亦供 MiniCPM5 `dump_tokens_oracle`、LFM2-MoE layer dump 使用 |
| Qwen3.5 / Qwen3.8-27B | `b96806d96061049a5b574269b049bf6241d63d46` | [`tools/oracle/qwen35/build_qwen35_oracle.sh`](../../tools/oracle/qwen35/build_qwen35_oracle.sh)、[`tests/qwen35_reference.rs`](../../tests/qwen35_reference.rs) |
| Qwen-Drive-1.0-4B VLM（qwen2vl/qwen3vl mtmd-debug） | `b96806d96061049a5b574269b049bf6241d63d46`（同 Qwen3.5） | [`tools/oracle/qwen_drive/build_qwen_drive_vlm_oracle.sh`](../../tools/oracle/qwen_drive/build_qwen_drive_vlm_oracle.sh)（复用 Qwen3.5 pin，patch 替换为 `qwen_drive_vlm_trace.patch`，target `mtmd-debug`）、[`tests/qwen_drive_vlm_reference.rs`](../../tests/qwen_drive_vlm_reference.rs) |
| Jina Embeddings v5 Omni Small Retrieval（Q8_0 text + F16 vision） | `b96806d96061049a5b574269b049bf6241d63d46` | [`tools/oracle/qwen_drive/build_qwen_drive_vlm_oracle.sh`](../../tools/oracle/qwen_drive/build_qwen_drive_vlm_oracle.sh) 构建固定 `llama-debug` / vision Oracle；[`tests/embedding_parity.rs`](../../tests/embedding_parity.rs) 对照 token IDs 与 pooled/final F32 bits，[`tests/qwen_drive_vlm_reference.rs`](../../tests/qwen_drive_vlm_reference.rs) 对照 vision checkpoints |
| Gemma 4 E2B / 12B 文本 | `3173a56471c1753650cd806694145ffd6dcace67` | [`tools/oracle/gemma4/build_oracle.sh`](../../tools/oracle/gemma4/build_oracle.sh)、[`tests/gemma4_reference.rs`](../../tests/gemma4_reference.rs) |
| Gemma 4 12B `gemma4ua` 音频 | `b96806d96061049a5b574269b049bf6241d63d46` | [`tools/oracle/gemma4/build_audio_oracle.sh`](../../tools/oracle/gemma4/build_audio_oracle.sh)、[`tools/oracle/gemma4/gemma4ua-trace.patch`](../../tools/oracle/gemma4/gemma4ua-trace.patch)、[`tests/gemma4_reference.rs`](../../tests/gemma4_reference.rs)（AVX2+FMA+F16C x86_64 CPU RMSNorm + F16 projector raw-bit parity） |
| Qwen3-TTS Base | `201e50cc2076a20adc460c41598593c7cd7b0813` | [`tools/oracle/qwen3_tts/build_qwen3_tts_oracle.sh`](../../tools/oracle/qwen3_tts/build_qwen3_tts_oracle.sh)、[`tests/qwen3_tts_reference.rs`](../../tests/qwen3_tts_reference.rs) |
| Apple Silicon 性能基准（一次性快照 2026-08-10） | `7ba604f1cb61cd14898138e9abc0b4ff2601f180` | [`docs/develop/OPTIMIZATION.md`](OPTIMIZATION.md#rust-与-llamacpp-固定机器对比2026-08-10)；这是性能基准 pin，**不是通用正确性 Oracle**，不随新回归自动刷新 |
| Nemotron-3 Nano 4B Mamba2 | `96013c511b8e2dc5b6a5dbcf6bf4ad9c10d2bf77`（2026-09-11） | 本地 `references/llama.cpp/build-release/bin/llama-cli`（version 10120），或重新 build；对应 [`tests/nemotron_h_parity.rs`](../../tests/nemotron_h_parity.rs)。**fixture 状态**：`tests/nemotron_h_parity.rs` 期望 `docs/parity_fixtures/nemotron_h_4b/`（含 `README.md`）提供 oracle 文本/token dump — 该目录**当前未随仓库提交**，需手动 drop oracle trace 后再跑，详见 `SUPPORTED_MODELS.md` 对 nemotron_h 的「Experimental」备注与该测试文件头注释 |
| Falcon-H1 1.5B Instruct | `171e8846b4af9766c354064cb776cb34a50f053f`（本地 checkout 当前 HEAD） | 本地 `references/llama.cpp/build-rmi-falconh1/bin/{llama-cli,llama-tokenize,llama-completion}`（scalar-no-OpenMP 构建，遵循 [`tools/oracle/shared/build_llama_oracle.sh`](shared/build_llama_oracle.sh) 标志，额外 `-DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TOOLS=ON` 以触发 llama-cli 构建）；对应 [`tests/falcon_h1_q8.rs`](../../tests/falcon_h1_q8.rs)。**状态**：Q8_0 仅在 `RMI_FALCON_H1_Q8_MODEL` 指向本地 `models/Falcon-H1-1.5B-Instruct-GGUF/Falcon-H1-1.5B-Instruct-Q8_0.gguf` 时运行；分词器逐位对齐 oracle；张量级 Q8 matmul 与 `hsum_float_8` 归约顺序仍存在 ulp 级差异（受全局 `hsum_ps` 复用约束，后续可提本地 matmul） |

## 本地 checkout 约定

构建脚本接收外部 checkout 路径，并在运行前校验提交；除通用 scalar 脚本外，脚本会复制到临时目录后再打 trace patch，避免修改输入 checkout。

`references/` 只是可选的本地便利目录，不是完整依赖清单，也不应据此判断某个 Oracle 是否存在。当前仓库不会自动下载这些上游；需要执行回归时，应按上表准备对应仓库和固定提交。

## DreamX-Creator trace

`tools/oracle/dreamx/dreamx_oracle_trace.py` 直接包装固定上游的两个官方入口，不修改 checkout。`creator` 子命令记录 tokenizer IDs、text contexts、first-frame latent、joint layers 0/15/29、最终 video/audio latents 和两个 decoder 输出；`refiner` 子命令记录首尾 SR-DiT blocks、最终 SR latent 和 decoder 输入/输出。每个 tensor 转为 little-endian F32 原始数组，token IDs/attention mask 保存为 little-endian I64，shape 和原始 dtype 写入 `manifest.json`。

```bash
python3 tools/oracle/dreamx/dreamx_oracle_trace.py creator \
  --upstream-root /path/to/DreamX-Creator --out-dir /tmp/dreamx-creator-trace -- \
  <audio_video_generation/inference.py arguments>

python3 tools/oracle/dreamx/dreamx_oracle_trace.py refiner \
  --upstream-root /path/to/DreamX-Creator --out-dir /tmp/dreamx-refiner-trace -- \
  <video_refiner/inference_sr.py arguments>
```

两个官方入口都要求 CUDA。当前 Apple Silicon 验证机只执行了脚本的帮助、静态编译和无 CUDA 明确退出，没有生成 trace 或与 Rust checkpoint 比较，因此 DreamX-Creator 目前不是 `Pinned Oracle` 数值对齐状态。
