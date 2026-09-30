# tools/converter

所有 GGUF 转换器、转换器测试和转换必需的数据都集中在这里。
模型专用的 Oracle、trace、README 和构建脚本仍留在 `tools/<model>/`。

## 目录结构

```
converter/
├── breeze/       ← 原版与扩展精度转换器
├── clm/          ← Contrastive-LM 投影头转换器和测试
├── dots/         ← dots writer、转换器和测试
├── dreamx/       ← DreamX 转换器和测试
├── edge0/        ← Edge0-35B MLX-affine 转换器、反量化与测试
├── gliner/       ← GLiNER 转换器和测试
├── neohorse/     ← NeoHorse 转换器和测试
├── qwen_drive/   ← Qwen-Drive 转换器、测试和 source-tensors.json
├── vibevoice/    ← 原版与扩展精度转换器
├── yue2/         ← YuE2 主模型与 decoder-only VAE 转换器
└── utils/        ← 共享 GGUF reader/writer、dtype 和量化工具
```

## 实现边界

| 路径 | 备注 |
|---|---|
| `breeze/convert_breeze_plain.py` | 原版未量化导出 |
| `breeze/convert_breeze.py` | 支持 `--quant bf16/f16/f32/q8_0/q4_0/q4_mixed` 和 `--codec-quant f32/q8_0` |
| `clm/convert_clm.py` | CLM 投影头 F32 导出 |
| `dots/convert_dots_tts.py` | dots 专用 writer 与 BF16/Q8_0 导出 |
| `dreamx/convert_dreamx_creator.py` | DreamX 主模型/mmproj 配对导出 |
| `edge0/convert_edge0.py` | Edge0-35B MLX-affine；支持 `--quant lossless/f32/f16/q8_0/q4_0` |
| `edge0/mlx_affine.py` | MLX-affine 反量化（与 `MlxAffineKernel::value` 对齐） |
| `neohorse/convert_neohorse.py` | 使用固定 llama.cpp 版本导出 |
| `qwen_drive/convert_qwen_drive.py` | `inspect`、`export`、`verify` |
| `vibevoice/convert_vibevoice_asr_original.py` | 原版导出 |
| `vibevoice/convert_vibevoice_asr.py` | 扩展精度导出 |
| `yue2/convert_yue2.py` | 流式导出 `yue2` BF16 主模型和 `yue2_vae` F32 decoder；固定协议/tokenizer metadata，原子写入并读回校验 |

## breeze 量化支持现状

| `--quant` | 输出文件 | 用途 | Rust 端加载 | 验证 |
|---|---|---|---|---|
| `bf16` (default) | `breeze-tts-2-BF16.gguf` | 与原始 checkpoint 字节一致 | ✅ | 与原 `plain.wav` md5 一致 |
| `f16` | `breeze-tts-2-F16.gguf` | BF16 → F16 重编码 | ✅ | 59 frames / 222K wav |
| `f32` | `breeze-tts-2-F32.gguf` | BF16 → F32 重编码 | ✅ | 44 frames / 166K wav |
| `q8_0` | `breeze-tts-2-Q8_0.gguf` | learned 2D weight 矩阵量化到 Q8_0 | ✅ | 29 frames / 17s **3.6× speedup**（v3 baseline；md5 `b051f3c1...`） |
| `q4_0` | `breeze-tts-2-Q4_0.gguf` | learned 2D weight 矩阵量化到 Q4_0 | ⚠️ | 128 frames / 481K wav — **输出退化**；`docs/TODO.md#todo-004` |

`--codec-quant` 默认 `f32`；可设 `q8_0` 把 Mimi codec 的 learned 2D weight 也量化。

**保持源精度的张量**（不受 `--quant` 影响）：
- `codec_model.*` — Mimi codec 快照（BF16 conv 权重、F32 codebook 标志）
- `depth_decoder.codebooks_head.weight`、`text_encoder.embed_tokens.eoi_embedding`
- 全部 norm weight（`*.norm.weight`、`*.layers.{i}.{input_layernorm,...}.weight`）

这些张量走 `core::tensor::load_f32_tensor`（仅 F32/BF16）。量化它们会破坏
loader；除非 `load_f32_tensor` 也扩到接受 Q 类型，否则永远保留源 dtype。

## edge0 量化支持现状

Edge0-35B 的源 checkpoint **已经是 MLX 4-bit affine 量化**（`group_size=64`，
router 为 8-bit），不是原始 BF16。所以这里的 `--quant` 语义与 breeze 相反：
除 `lossless` 外都是**第二次**有重量化。

| `--quant` | 输出张量类型 | payload | 用途 | Rust 端加载 |
|---|---|---|---|---|
| `lossless` (default) | packed U32 → GGUF I32 | 19.0 GB | 架构对齐，oracle 逐位验证 | ✅ `MlxAffineKernel` |
| `f32` | 全部矩阵 → F32 | 129 GiB | affine 展开的最高保真参考 | ✅ 但内存不足时不可用 |
| `f16` | 全部矩阵 → F16 | 64.6 GiB | 通用 GGML 消费者 | ✅ |
| `q8_0` | 全部矩阵 → Q8_0 | 34.3 GiB | 通用 GGML 消费者 | ✅ |
| `q4_0` | 全部矩阵 → Q4_0 | 18.2 GiB | 通用 GGML 消费者 | ✅ |

`lossless` 保持字节完全一致，因此是唯一能跑 scalar oracle 的格式。其他模式把
affine group 展开成 F32 后重新编码，`scales`/`biases` 随之合并进矩阵、不再单独
输出；norm、SSM 参数和 LoRA 仍保留源精度。metadata 记录 `edge0.quant.mode`，
`edge0.quant.group_size` 只在 `lossless` 下出现。

`load_affine` 按张量类型分派：I32 走 `load_packed`（`MlxAffineKernel`），F32/F16/
Q8_0/Q4_0 走 `load_expanded`（通用量化 kernel，按 expert 步长切片）。所以除
`lossless` 外的所有模式都能被 Rust 加载推理。

**实测（`Hello`，greedy，官方参考 `[9419, 0, 2500, 628]`）：**

| 模式 | 文件大小 | 生成速度（8 线程） | token IDs |
|---|---|---|---|
| `lossless` | 19 GB | 0.1 t/s | ✅ 逐位一致 |
| `q4_0` | 18.2 GB | **22.9 t/s** | ✅ 逐位一致 |
| `q8_0` | 34.3 GB | 16.8 t/s | ✅ 逐位一致 |
| `f16` | 64.6 GB | 8.7 t/s | ✅ 逐位一致 |

`q4_0` 与 `lossless` 体积相同，但因 lossless 需要逐元素反量化 affine group，
实测快 **229×**。这是推荐的生产格式。`f32` 需要 129 GiB，本机内存不足。

反量化公式（`mlx_affine.py` 与 `src/ops/kernel/mlx_affine.rs` 必须一致）：

```
value(row, col) = bf16(scales[group]) * q + bf16(biases[group])
group           = row * (n_in // 64) + col // 64
```

## utils 现状（`converter/utils/gguf.py`）

公开符号：

```python
GGML_F32, GGML_F16, GGML_Q4_0, GGML_Q8_0, GGML_I64, GGML_BF16
GGUF_ALIGNMENT (= 32)
Q8_0_BLOCK, Q8_0_BLOCK_BYTES
Tensor, Safetensors, open_safetensors, validated_dir
bf16_to_f32, f16_to_f32, bf16_to_f16, f32_to_bf16, f32_to_f16, f32_values
quantize_q8_0, quantize_q4_0, bf16_bytes_to_q8_0
GgufWriter, TensorPayload, gguf_dims, emit_tensor
read_gguf_directory, read_gguf_tensor_bytes
load_latent_stats
```

`utils` 的 writer 与 dots writer 在原子写入、覆盖及读回校验上行为不同，
不可直接互换；dots 的 Q8 量化也使用不同的缩放计算方式，故未合并。

## 跑测试

```bash
PYTHONPATH=. python3 -m unittest \
  tools.converter.breeze.test_convert_breeze_plain \
  tools.converter.breeze.test_convert_breeze \
  tools.converter.dots.test_convert_dots_tts \
  tools.converter.dreamx.test_convert_dreamx_creator \
  tools.converter.neohorse.test_convert_neohorse \
  tools.converter.qwen_drive.test_convert_qwen_drive \
  tools.converter.vibevoice.test_convert_vibevoice_asr_original \
  tools.converter.vibevoice.test_convert_vibevoice_asr \
  tools.converter.yue2.test_convert_yue2

# edge0 的反量化测试是 pytest 风格（用 parametrize），单独跑
PYTHONPATH=. python3 -m pytest tools/converter/edge0/test_convert_edge0.py
```

## 不变原则

- 转换代码只放在 `tools/converter/<model>/`，旧路径不保留包装或软链接。
- Oracle、trace 和构建脚本继续放在 `tools/<model>/`。
- 更改已使用的 utils API 时，保持现有输出的字节兼容。
