# tools/converter

共享 GGUF 工具与有独立功能的转换器。原 `tools/<model>/` 仍是对外入口；
完全相同的 dots 转换器、Breeze trace 工具和 VibeVoice Oracle 不在此重复存放。

## 目录结构

```
converter/
├── README.md          ← 本文件
├── __init__.py
├── utils/
│   ├── __init__.py
│   └── gguf.py        ← 共享 GGUF writer/reader、dtype 转换、quantize_q8_0
├── breeze/            ← 扩展精度/量化的转换器及测试
└── vibevoice/         ← 扩展精度/量化的转换器及测试；复用原 dots writer
```

## 实现边界

| 子目录 | 状态 | 备注 |
|---|---|---|
| `breeze/convert_breeze.py` | 使用 `converter.utils.gguf` | 支持 `--quant bf16/f16/f32/q8_0/q4_0/q4_mixed` 和 `--codec-quant f32/q8_0` |
| `vibevoice/convert_vibevoice_asr.py` | Q8/Q4 复用 utils；writer 复用 `tools/dots/convert_dots_tts.py` | 保留本目录独有的精度选项 |

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
# 独有转换器
PYTHONPATH=tools python3 tools/converter/breeze/test_convert_breeze.py
PYTHONPATH=tools python3 tools/converter/vibevoice/test_convert_vibevoice_asr.py

# 原目录保留的转换器 / trace / Oracle 测试
python3 tools/dots/test_convert_dots_tts.py
python3 tools/breeze/test_compare_breeze_trace.py
PYTHONPATH=.:tools python3 tools/vibevoice/test_convert_vibevoice_asr.py
```

## 不变原则

- `tools/<model>/` 原目录**永远不动**（用户明确指示）。
- 本目录**不是**新对外约定，仓库其他子系统（CI、文档、脚本）仍以
  `tools/<model>/` 为准。
- 更改已使用的 utils API 时，保持现有输出的字节兼容。
