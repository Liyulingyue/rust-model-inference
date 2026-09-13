# tools/converter

镜像 `tools/{breeze,dots,vibevoice}/` 的 GGUF 转换器副本，作为模块化
重构的起点。原 `tools/<model>/` 目录**完全保留**，所有现有脚本、CI、
文档路径不变；本目录只用于：

1. 演进共享 GGUF / 量化 / safetensors 工具层（`utils/`）。
2. 在不影响原路径的前提下，逐步把副本切换到 `utils` 共享层。
3. 作为新转换器的入口（后续 `dreamx`、自定义模型优先放这里）。

## 目录结构

```
converter/
├── README.md          ← 本文件
├── __init__.py        ← 空，让 `converter.utils.gguf` 成为 canonical import
├── utils/
│   ├── __init__.py
│   └── gguf.py        ← 共享 GGUF writer/reader、dtype 转换、quantize_q8_0
├── breeze/            ← 1:1 副本；已切换 import 到 converter.utils.gguf
├── dots/              ← 1:1 副本；自包含（含 GgufWriter 等定义），下一步切 utils
└── vibevoice/         ← 1:1 副本；通过 sys.path 引用副本 dots
```

## 切换进度

| 子目录 | 状态 | 备注 |
|---|---|---|
| `breeze/convert_breeze.py` | ✅ 已切到 `converter.utils.gguf` | 支持 `--quant bf16/f16/f32/q8_0/q4_0` 和 `--codec-quant f32/q8_0` |
| `dots/convert_dots_tts.py` | ⏳ 自包含副本 | 仍带本地 `GgufWriter` 等；下一步抽 |
| `vibevoice/convert_vibevoice_asr.py` | ⏳ 自包含副本 | 通过 `sys.path` 引用副本 `dots` |

## breeze 量化支持现状

| `--quant` | 输出文件 | 用途 | Rust 端加载 |
|---|---|---|---|
| `bf16` (default) | `breeze-tts-2-BF16.gguf` | 与原始 checkpoint 字节一致 | ✅ |
| `f16` | `breeze-tts-2-F16.gguf` | BF16 → F16 重编码；常用于 GPU 推理 | ⏳ |
| `f32` | `breeze-tts-2-F32.gguf` | BF16 → F32 重编码；CPU 数值稳定 | ⏳ |
| `q8_0` | `breeze-tts-2-Q8_0.gguf` | learned 2D weight 矩阵量化到 Q8_0 | ⏳（需 dequant） |
| `q4_0` | `breeze-tts-2-Q4_0.gguf` | learned 2D weight 矩阵量化到 Q4_0 | ⏳（需 dequant） |

`--codec-quant` 默认 `f32`；可设 `q8_0` 把 Mimi codec 的 learned 2D weight 也量化。

**已知限制**：Breeze 的 Rust 端 `BreezeModel::matrix()` 与 `require_bf16()` 目前只
接受 `GGMLType::BF16`（`src/models/breeze/mod.rs:42-88`、`src/models/breeze/transformer.rs:8-18`）。
因此 `f16`/`f32`/`q8_0`/`q4_0` 输出的 GGUF 加载时会立即报错
`Breeze requires original BF16 weights`，即使文件本身字节合法。要让这些
精度生效，必须扩展 `matrix()` / `Transformer::load` 接受多 GGML 类型并走
dequant 路径（后续 PR）。

codec（`breeze-tts-2-mmproj-*.gguf`）当前始终 F32 加载，无需此变更。

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

`GgufWriter` 输出的 GGUF 文件与原 `tools/dots/convert_dots_tts.GgufWriter`
**字节一致**（相同的 dialect：u64 字符串长度、tensor nbytes 字段存
aligned 相对偏移、32-byte 对齐）。

## 跑测试

```bash
# breeze 副本
.venv/bin/python tools/converter/breeze/test_convert_breeze.py

# dots 副本（自包含）
.venv/bin/python tools/converter/dots/test_convert_dots_tts.py

# vibevoice 副本（sys.path 引用副本 dots）
PYTHONPATH=tools .venv/bin/python tools/converter/vibevoice/test_convert_vibevoice_asr.py
```

## 渐进迁移步骤

1. ✅ 抽 `converter/utils/gguf.py`（含 GgufWriter / reader / dtype / quant）。
2. ✅ breeze 副本切换到 utils。
3. ⏳ dots 副本去重：把 `convert_dots_tts.py` 内嵌的 GgufWriter / Tensor / 量化
   删掉，改 import 自 `converter.utils.gguf`；保证 20/20 测试通过。
4. ⏳ vibevoice 副本去重：把 `quantize_q8_0` 等本地定义删掉，改 import 自
   `converter.utils.gguf`。
5. ⏳ dreamx / 未来新转换器直接放 `converter/<model>/`，import 自 utils。

## 不变原则

- `tools/<model>/` 原目录**永远不动**（用户明确指示）。
- 本目录**不是**新对外约定，仓库其他子系统（CI、文档、脚本）仍以
  `tools/<model>/` 为准。
- utils API 变更必须保持字节兼容（与原 `convert_dots_tts.GgufWriter`
  输出可逐字节对齐）。
