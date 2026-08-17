# RustModelInference

> 100% 纯 Rust · mmap 零拷贝 · 多模态 LLM 推理引擎

## 概述

从零构建的 LLM 推理引擎，通过 mmap 加载 GGUF 文件，支持文本/图像/音频生成。遵循五大原则：

1. **热路径零堆分配** — `forward()` 写入预分配的 `&mut [f32]`
2. **mmap 零拷贝** — 权重是通过 `memmap2` 区域借用的 `&'a [u8]` 切片
3. **显式内存生命周期** — 所有缓冲区均由调用方提供
4. **Trait 架构** — 算子和内存通过 trait 解耦
5. **无 C/C++ FFI** — 100% 纯 Rust，包括量化 kernel

**支持的模型**：Qwen3-0.6B、Qwen3-Embedding、Qwen3-ASR、Qwen3-VL、MiniCPM5-1B、Hunyuan-MT2、Nanbeige

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

# 多模态（Qwen3-VL）
cargo run --release --bin rust-model-inference -- \
  --model models/Qwen3-0.6B-Q8_0.gguf \
  --mmproj models/mmproj-Qwen3-ASR-0.6B-Q8_0.gguf \
  --image path/to/image.jpg \
  --prompt "描述这张图片"

# ASR（音频）
cargo run --release --bin rust-model-inference -- \
  --model models/Qwen3-ASR-0.6B-Q8_0.gguf \
  --mmproj models/mmproj-Qwen3-ASR-0.6B-Q8_0.gguf \
  --audio models/001_16k.wav \
  --language en
```

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

**1. 安装 glslangValidator（如需重新编译 shader）**
```bash
sudo apt install glslang-tools
```

**2. 编译 shader**
```bash
glslangValidator -V shaders/src/q8_matmul.comp -o shaders/bin/q8_matmul.spv
```

**3. 配置 Vulkan ICD**
```bash
# Intel GPU
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/intel_icd.json

# NVIDIA GPU
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json

# AMD GPU
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/amd_icd.json
```

查看可用设备：
```bash
ls /usr/share/vulkan/icd.d/
ls -la /dev/dri/  # 查看 GPU 设备
```

**4. 启用 GPU 推理**
```bash
export USE_GPU=1
cargo run --release --features vulkan -- --model models/Qwen3-0.6B-Q8_0.gguf --prompt "法国的首都是"
```

**注意**：GPU 输出结果可能有乱码，当前为实验性支持。

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

提供 REST API 服务：

```bash
cargo run --release --bin server
```

## License

MIT
