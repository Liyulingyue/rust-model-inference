# RustModelInference

> 100% 纯 Rust · mmap 零拷贝 · 多模态 LLM 推理引擎

## 概述

从零构建的 LLM 推理引擎，通过 mmap 加载 GGUF 文件，支持文本/图像/音频生成。遵循五大原则：

1. **热路径零堆分配** — `forward()` 写入预分配的 `&mut [f32]`
2. **mmap 零拷贝** — 权重是通过 `memmap2` 区域借用的 `&'a [u8]` 切片
3. **显式内存生命周期** — 所有缓冲区均由调用方提供
4. **Trait 架构** — 算子和内存通过 trait 解耦
5. **无 C/C++ FFI** — 100% 纯 Rust，包括量化 kernel

**支持的模型**：Qwen3-0.6B、Qwen3-Embedding、Qwen3-ASR、Qwen3-TTS、Qwen3.5-VL、MiniCPM5-1B、Hunyuan-MT2、Nanbeige

## 快速开始

```bash
# 构建
cargo build --release

# 文本推理
cargo run --release --bin rust-model-inference -- \
  --model models/Qwen3-0.6B-Q8_0.gguf \
  --prompt "法国的首都是" --max-tokens 30

# Embedding 模式
cargo run --release --bin rust-model-inference -- \
  --model models/Qwen3-Embedding-0.6B-Q8_0.gguf \
  --prompt "Hello, 世界! 123" --embedding --embedding-output raw --threads 1

# 多模态（Qwen3.5-VL）
cargo run --release --bin rust-model-inference -- \
  --model models/Qwen3.5-0.8B-Q8_0.gguf \
  --mmproj models/Qwen3.5-0.8B-mmproj-F16.gguf \
  --image path/to/image.jpg \
  --prompt "描述这张图片"

# ASR（音频）
cargo run --release --bin rust-model-inference -- \
  --model models/Qwen3-ASR-0.6B-Q8_0.gguf \
  --mmproj models/mmproj-Qwen3-ASR-0.6B-Q8_0.gguf \
  --audio models/001_16k.wav \
  --language en
```

### Gemma 4 E2B 多模态

Gemma 4 E2B 使用 `models/gemma-4-e2b/gemma-4-E2B-it-Q8_0.gguf`。文本可直接运行；任一媒体输入都必须同时传入 `models/gemma-4-e2b/mmproj-F16.gguf`：

```bash
# 文本
cargo run --release --bin rust-model-inference -- \
  --model models/gemma-4-e2b/gemma-4-E2B-it-Q8_0.gguf \
  --prompt "Explain the result." --max-tokens 32 --temp 0

# 图像 + 文本
cargo run --release --bin rust-model-inference -- \
  --model models/gemma-4-e2b/gemma-4-E2B-it-Q8_0.gguf \
  --mmproj models/gemma-4-e2b/mmproj-F16.gguf \
  --image path/to/image.png \
  --prompt "Describe this image." --max-tokens 32 --temp 0

# 音频 + 文本
cargo run --release --bin rust-model-inference -- \
  --model models/gemma-4-e2b/gemma-4-E2B-it-Q8_0.gguf \
  --mmproj models/gemma-4-e2b/mmproj-F16.gguf \
  --audio path/to/audio.wav \
  --prompt "Transcribe the audio." --max-tokens 32 --temp 0

# 图像 + 音频 + 文本
cargo run --release --bin rust-model-inference -- \
  --model models/gemma-4-e2b/gemma-4-E2B-it-Q8_0.gguf \
  --mmproj models/gemma-4-e2b/mmproj-F16.gguf \
  --image path/to/image.png \
  --audio path/to/audio.wav \
  --prompt "Describe the image and audio." --max-tokens 32 --temp 0
```

当前仅支持一个用户轮次、每种媒体各一个文件；当同时提供时，输入顺序固定为图像、音频、提示词。音频输入为 PCM16 WAV。CPU 对齐覆盖 attention softmax 前的 token IDs，以及图像 `gemma4.vision.preprocessed`、音频 `gemma4.audio.mel` 的原始 F32 `u32` 位。attention 统一使用准确、稳定的标量 softmax，因此其后的 checkpoint、logits 与 greedy token IDs 不承诺和 llama.cpp 逐位一致；`--gpu` 可以运行，但当前不提供 GPU 位级对比保证。

### Qwen3-TTS Base 声音克隆

```bash
cargo run --release --bin rust-model-inference -- \
  --model models/Qwen3-TTS/Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf \
  --mmproj models/Qwen3-TTS/mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf \
  --tts --prompt "你好，这是一个语音合成测试。" --language cn \
  --ref-audio reference.wav --out output.wav
```

参考音频必须是 PCM16 WAV；Base 模型不支持 `--ref-text`。输出固定为单声道 24 kHz PCM16 WAV。语言支持 `cn/en/ge/it/po/sp/ja/ko/fr/ru`、对应英文全名，以及 `zh/de/pt/es` 别名。

### Z-Image Turbo

```bash
cargo run --release --bin rust-model-inference -- \
  --model models/z-image-gguf/z-image-turbo-q8_0.gguf \
  --text-encoder models/z-image-gguf/qwen3_4b_f32-q8_0.gguf \
  --vae models/z-image-gguf/pig_flux_vae_fp32-f16.gguf \
  --prompt "A red fox sleeping beneath a pine tree" \
  --steps 8 --resolution 512 --seed 42 --threads 1 --out fox.png
```

当前范围是原生 Rust、CPU、512×512 的 Z-Image Turbo 文生图；暂不支持 Z-Image Base、GPU 或 img2img。

### Apple Silicon (ARM64)

Apple Silicon 原生构建，自动选择稳定的 Rust NEON kernel；无需 Rosetta 或外部 C/C++ 库。无 ARM SIMD 路径的算子保留标量回退。

```bash
cargo check --all-targets
cargo test --all-targets
cargo build --release --all-targets
cargo run --release --bin rust-model-inference -- --model models/Qwen3-0.6B-Q8_0.gguf --prompt "2 + 3 =" --max-tokens 4 --temp 0 --threads 8 --kv-cache f16 --bench
cargo run --release --bin micro-bench
```

文本推理和 embedding 的 `--threads` 默认为 `min(available_parallelism, 8)`，可显式设置。KV cache 默认为 F16；固定对比时显式传入 `--kv-cache f16` 以匹配 llama.cpp 的 `-ctk f16 -ctv f16`。`--bench` 分别报告 prompt 处理（`BENCH: pp`）和 token 生成（`BENCH: tg`）的评估速率。公平 CPU 对比时，用 `llama-bench -ngl 0 -t 8` 运行 llama.cpp；`-ngl 99` 使用 Metal 后端，需单独报告。固定的、自包含的 llama.cpp 复现步骤见 [OPTIMIZATION.md](./OPTIMIZATION.md#rust-与-llamacpp-固定机器对比2026-08-10)。

在固定的 Apple Silicon 性能机器上，显式强制 Q8_0 NEON 验证：

```bash
cargo run --release --bin micro-bench -- --check
```

### GPU 后端 (Vulkan)

启用 GPU 加速推理（需要支持 Vulkan 的 GPU）：

**1. 检查 Vulkan runtime 和设备**

```bash
vulkaninfo --summary
```

程序会先使用系统 Vulkan Loader 的标准发现机制；macOS 还会自动尝试 Homebrew 的
`/opt/homebrew/lib/libvulkan.dylib` 和 `/usr/local/lib/libvulkan.dylib`，并自动启用
MoltenVK 需要的 portability 扩展。正常使用不需要设置 `DYLD_*`、
`VK_ICD_FILENAMES` 或 `VK_DRIVER_FILES`。

macOS 可通过 `brew install vulkan-tools molten-vk glslang spirv-tools` 安装运行时和工具；
Debian/Ubuntu 可安装 `vulkan-tools glslang-tools spirv-tools` 以及对应显卡驱动。

**2. 校验或重新生成 shader（仅开发时需要）**

```bash
glslangValidator -V shaders/glsl/q8_matmul.comp -o shaders/bin/q8_matmul.spv
spirv-val shaders/bin/q8_matmul.spv
```

**3. 启用 GPU 推理**

```bash
cargo run --release --features vulkan -- \
  --gpu \
  --model /Users/gouzi/Documents/git/rust-model-inference/models/Qwen3-0.6B-Q8_0/Qwen3-0.6B-Q8_0.gguf \
  --prompt "法国的首都是"
```

server 使用同一个 `--gpu` 开关：

```bash
cargo run --release --features vulkan --bin server -- \
  --gpu --model /path/to/model.gguf --port 8080
```

只有在标准发现选错 ICD 或调试多驱动机器时才需要覆盖 Loader，例如：

```bash
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/intel_icd.json vulkaninfo --summary
```

**注意**：当前仍是实验性 Q8_0 matmul offload；完整模型算子和更多权重格式的 Vulkan
覆盖见 [VULKAN.md](./docs/VULKAN.md)。未传 `--gpu` 时保持纯 CPU 路径。

### CLI 选项

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `--model` | — | GGUF 文件路径（LLM） |
| `--mmproj` | — | mmproj GGUF 路径（视觉/音频投影器） |
| `--audio` | — | 音频文件路径（WAV，16kHz，用于 ASR） |
| `--image` | — | 图像文件路径（多模态输入） |
| `--prompt` | — | 输入 prompt（省略则进入交互模式） |
| `--language` | — | ASR 语言代码（如 `en`、`zh`） |
| `--max-tokens` | 128 | 最大生成 token 数 |
| `--temp` | 0.6 | 采样温度 |
| `--threads` | `min(available_parallelism, 8)` | 计算线程数 |
| `--kv-cache` | `f16` | KV cache 类型：`f16` 或 `f32` |
| `--bench` | off | 打印 `BENCH: pp`（prompt）和 `BENCH: tg`（生成）评估速率 |
| `--profile` | off | 打印逐层耗时分解 |
| `--embedding` | off | 启用 embedding 模式 |
| `--embedding-output` | `summary` | Embedding 显示：`summary` 或机器可读的 `raw` |
| `--thinking` | off | 启用链式思维推理（Qwen3） |
| `--dump-logits` | off | 将 logits 写入 `/tmp/rust_logits.bin` 用于精度验证 |

### 调试标志（环境变量）

| 变量 | 描述 |
|-----|------|
| `VERBOSE=1` | 显示 top-10 token 和 logit 统计 |
| `DEBUG_LAYER=N` | 转储第 N 层的逐层中间值 |
| `DEBUG_POS=N` | 在第 N 位置转储 |

### 精度验证

验证与 llama.cpp 的数值对齐：

```bash
cargo build --release --features parity-trace
cargo run --release --features parity-trace --bin rust-model-inference -- \
  --model models/Qwen3-0.6B-Q8_0.gguf \
  --prompt "法国的首都是" \
  --dump-logits
```

## 示例输出

```
$ cargo run --release --bin rust-model-inference -- --model models/Qwen3-0.6B-Q8_0.gguf --prompt "法国的首都是"
Output:  Paris. The capital of France is located in the southern part of France...

$ cargo run --release --bin rust-model-inference -- --model models/Qwen3-0.6B-Q8_0.gguf --prompt "2 + 3 ="
Output:  5, 3 + 4 =
```

## 项目结构

```
src/
├── lib.rs          # 包根目录，公开重导出
├── traits.rs       # Layer trait, ExecContext, ModelConfig
├── memory.rs       # PagedKVBlock, BlockAllocator, MemoryArena, KVCache
├── quant.rs        # Q4_K_M 块结构体 + 反量化 kernel
├── model.rs        # GGUF V2/V3 mmap 加载器, QuantizedLinear<'a>, ModelGraph
├── ops.rs          # rms_norm, rope_neox, silu, softmax, matmul_q8_0, sampling
├── tokenizer.rs    # GPT-2 BPE tokenizer，带 byte-encoder/decoder
├── scratchpad.rs   # ExecutionScratchpad, KvCache (F16/F32)
├── prompt.rs       # Chat 模板构建器（Qwen, Hunyuan）
├── load_plan.rs    # NUMA 感知加载规划，层放置
├── ggufrs.rs       # GGUFRS 文件格式导出/加载
├── vision.rs       # VisionEncoder trait, VisionGrid, VisionScratchpad
├── asr.rs          # ASR 音频预处理
├── clip_config.rs  # CLIP/Vision 配置解析
├── qwen3.rs        # Qwen3 模型实现
├── qwen35.rs       # Qwen3.5/VL 模型实现
├── qwen3a.rs       # Qwen3-ASR 模型实现
├── parity_trace.rs # llama.cpp 逐层精度对比（parity-trace feature）
├── vulkan.rs       # Vulkan GPU 后端（vulkan feature）
└── main.rs         # CLI + 推理循环

shaders/
├── src/q8_matmul.comp   # GLSL 计算着色器（Q8_0 matmul）
└── bin/q8_matmul.spv    # 预编译的 SPIR-V
```

## Qwen3-0.6B 参数

| 参数 | 值 |
|------|-----|
| 架构 | qwen3 |
| Embedding 维度 | 1024 |
| 层数 | 28 |
| Attention Q 头数 | 16 |
| Attention KV 头数 | 8 (GQA) |
| Head dim (K/V) | 128 |
| Q 维度 | 2048 |
| FFN 维度 | 3072 |
| 上下文长度 | 40960 |
| 词表大小 | 151,936 |
| RoPE freq base | 1,000,000 |
| Norm epsilon | 1e-6 |
| Q/K Norm | 是（逐 head RMSNorm） |

## 支持的 GGUF 特性

- GGUF V2/V3 格式解析
- Q8_0 量化（反量化 + matmul）
- Q4_0 / Q4_K_M 量化（反量化 + matmul）
- F32 tensor（norm 权重等）
- mmap 零拷贝权重加载
- 多模态：Vision（Qwen3-VL）+ mmproj 投影器
- ASR：音频预处理 + Qwen3-ASR
- 分页 KV cache（F16/F32）
- NUMA 感知层放置
- GGUFRS 单文件打包（`.ggufrs` 导出/加载）

## 架构

完整设计文档见 [ARCHITECTURE.md](./docs/ARCHITECTURE.md)。

## 依赖

| 包 | 版本 | 用途 |
|----|------|------|
| `memmap2` | 0.9 | mmap 零拷贝文件加载 |
| `half` | 2.4 | Q8_0 scale factor 的 f16 |
| `rayon` | 1.10 | 数据级并行 |
| `image` | 0.25 | 多模态图像解码 |
| `tokio` | 1 | 服务端模式异步运行时 |
| `serde` | 1 | 序列化 |
| `rand` | 0.8 | 采样工具 |
| `ash` | 0.37 | Vulkan API 绑定（vulkan feature） |
| `bytemuck` | 1.0 | 类型转换（vulkan feature） |

## 路线图

- [x] SIMD 反量化（AVX2 / NEON）
- [x] Chat 模板支持（Qwen, Hunyuan）
- [x] 分页 KV cache（F16/F32）
- [x] 逐层数值对齐（parity-trace feature）
- [x] Q4_K_M matmul kernel
- [x] 多模态（Vision + mmproj）
- [x] ASR（音频）
- [ ] 连续 batching / 多序列
- [ ] 更多量化格式（Q5_K, Q6_K）
- [x] Vulkan GPU 后端（实验性，Q8_0 matmul）

## 服务端模式

`server` 二进制与主 CLI 共用同一份 `app::parse_cli_options`，因此所有模式
（文本、视觉、音频、ASR、TTS、Embedding、Z-Image 以外的 `qwen3/qwen35/gemma4/hunyuan/lfm2/lfm2moe/llama` 架构）
都可以通过同一组 OpenAI 兼容 HTTP 端点暴露。`--host/--port` 是唯一额外的 server 端参数。

```bash
# 文本（qwen3、qwen35 等）
cargo run --release --bin server -- \
  --model models/Qwen3-0.6B-Q8_0.gguf --host 0.0.0.0 --port 8080 --threads 4

# Embedding（Qwen3-Embedding）
cargo run --release --bin server -- \
  --model models/Qwen3-Embedding-0.6B-Q8_0.gguf --embedding

# ASR（Qwen3-ASR + mmproj）
cargo run --release --bin server -- \
  --model models/Qwen3-ASR-0.6B-Q8_0.gguf \
  --mmproj models/mmproj-Qwen3-ASR-0.6B-Q8_0.gguf \
  --audio models/001_16k.wav --language en

# TTS（Qwen3-TTS + mmproj）
cargo run --release --bin server -- \
  --model models/Qwen3-TTS/Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf \
  --mmproj models/Qwen3-TTS/mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf \
  --tts --language cn
```

### 端点

| 路径 | 方法 | 模式 | 说明 |
|------|------|------|------|
| `/health` | GET | any | liveness probe |
| `/v1/models` | GET | any | 模型列表 |
| `/v1/chat/completions` | POST | text | OpenAI Chat Completions，支持 `stream: true` (SSE) |
| `/v1/embeddings` | POST | embedding | OpenAI Embeddings（字符串或字符串数组） |
| `/v1/audio/transcriptions` | POST | asr | OpenAI Audio Transcriptions（multipart：`file`、`language`、`prompt`） |
| `/v1/audio/transcriptions_json` | POST | asr | 同上，但用 JSON 体，`input` 字段为 base64 WAV |
| `/v1/audio/speech` | POST | tts | OpenAI Audio Speech：`input` 是文本，`voice` 接受 `file://path` 或 `data:audio/wav;base64,...` 形式的参考音频 |

### 示例

```bash
# Chat Completions (stream)
curl -N http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"1+1="}],"max_tokens":10,"temperature":0,"stream":true}'

# Embeddings
curl http://localhost:8080/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"input":"hello world"}'

# ASR (multipart)
curl http://localhost:8080/v1/audio/transcriptions \
  -F file=@models/001_16k.wav -F language=en

# TTS
curl http://localhost:8080/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-tts","input":"你好，这是一个测试。"}' \
  --output out.wav
```

**当前未覆盖：** 单次请求级别传入图片/音频的视觉多模态（VL 模型的图像通过 CLI
`--image` 在 server 启动时固定传入）；`gemma4 / hunyuan / lfm2 / llama` 等文本架构
的服务端流式输出（CLI 仍可用，详见 `cargo run --bin rust-model-inference -- --help`）。

## License

MIT
