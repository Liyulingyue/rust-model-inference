# BitNet Embeddings 用法（实验性）

`Microsoft/bitnet-embedding-0.6b`（Qwen3 backbone）和
`Microsoft/bitnet-embedding-270m`（Gemma3 backbone）是 BitNet
b1.58 改造的 LLM-embedding 系列：1.58-bit 三值权重（`{-1, 0, +1}`）+
8-bit per-token absmax 激活量化（W1.58A8）+ per-projection RMSNorm
预归一化（BitLinear pattern）。**decoder-only + last-token pooling +
L2 normalization** 输出 dense text embedding。

> ⚠️ **当前状态：仅 0.6B 的 metadata + tensor contract 已锁。
> Forward 集成尚未接入。**本仓库到本 commit（`I2_S` GGMLType 占位 +
> i2_s dequant + BitLinear ops + contract test）已确认引擎能正确
> 读出 BitNet GGUF 的字节，但**任何 --prompt / --embed 调用 0.6B 都
> 会进到 qwen3 trunk 的"无 BitLinear forward"路径，silently 错。
> Full forward 集成（约 1500-2000 行新代码 + qwen3 trunk 修改）
> 见 §3。

## 1. 已下载并验证的 GGUF

| 模型 | 大小 | arch | 来源 | tensor 总数 |
|---|---|---|---|---|
| `bitnet-embedding-0.6b` | 408 MB（i2_s BF16 混合） | `qwen3` | 官方 HF（mistralai 上 ModelScope 同步） | 506（310 F16 + 196 I2_S） |
| `bitnet-embedding-270m` | 351 MB | `gemma3`（**未支持**） | 官方 HF | 362（236 F16 + 126 I2_S） |

GGUF 文件路径：
- `models/bitnet-embedding-0.6b-GGUF/bitnet-embeddings-0.6b-bf16-i2_s.gguf`
- `models/bitnet-embedding-270m-GGUF/bitnet-embeddings-270m-bf16-i2_s.gguf`

## 2. 引擎层支持现状

### 2.1 已就绪 ✓

- **GGML type 36 (`I2_S`) 已注册**：`src/core/tensor.rs::GGMLType::I2_S = 36`，
  block layout `(128, 32)` = `QK_I2_S = 128` elements × 2 bits/element / 8 bits/byte。
- **i2_s dequant kernel** (`src/ops/kernel/i2_s.rs`):
  - `dequant_i2_s_block(&[u8; 32], &mut [f32; 128])` — 单 block 标量 dequant
  - `dequant_i2_s_row(bytes, n_elements, &mut [f32])` — 整行 dequant
  - `is_i2_s_aligned(n_elements)` / `i2_s_row_bytes(n_elements)`
  - 7 个单测覆盖：所有 2-bit code (0b00=-1, 0b01=0, 0b10=+1, 0b11=reserved→0)、
    多 block 对齐、**真实 GGUF block 读取**（确认从 Mistral 官方转换出来
    的字节落到 `{-1.0, 0.0, +1.0}` 而不是被误读 scale 字节弄出垃圾值）
- **BitLinear forward ops** (`src/ops/bitlinear.rs`):
  - `quantize_activation_per_token(&[f32]) -> (Vec<i8>, f32)` — per-row absmax → int8
  - `bitlinear_forward(weights_i2s, x_q, absmax, n_in, n_out, &mut [f32])` — 标量 reference matmul
  - `bitlinear_forward_from_f32(weights_i2s, &x, n_in, n_out, &mut [f32])` — 端到端（量化 + matmul）
  - 4 个单测：zero-sum / constant-input / sparse-weight / dequant-vs-quant 一致性
- **GGUF loader 兼容**：测试 `tests/bitnet_embedding_0_6b_q4_k_m.rs` 6/6 通过，
  锁住 0.6B 的 arch、file_type=40、Qwen3 dims、plain RoPE（无 YaRN）、
  tokenizer、506 tensor inventory、I2_S row layout、per-projection `*_norm_in`
  RMSNorm 存在性。

### 2.2 待补 ☐

- **BitLinear forward 接入 qwen3 trunk**：每个 BitLinear（attn_q/k/v/o、
  ffn_gate/up/down）需要先做 RMSNorm（用对应 `*_norm_in.weight`）再做
  `bitlinear_forward_from_f32`；qwen3 trunk 目前是直接调 Q8_0 matmul。
- **Pooling 修复**：`qwen3.pooling_type=1` 在 BitNet 是 last-token，
  现有 `src/models/qwen3/embedding.rs:80` 把它映射成 Mean（Qwen3-Embedding
  约定）。需要加一个 `EmbeddingPooling::BitNetLast` variant 或在
  `embedding_config` 里加 arch=`bitnet` / 文件名 heuristic 分支。
- **L2 normalization**：embedding 抽取后做 `x /= x.norm()`，没实现。
- **arch 路由**：file_type=40 是 BitNet I2_S 标记，但 GGUF 里
  `general.architecture` 还是 `qwen3`（or `gemma3`），所以需要按 file_type +
  `*_norm_in` tensor 存在性来 dispatch。
- **Gemma3 trunk**：270M 的 arch 是 `gemma3`，本仓库**没有 gemma3 trunk**。
  270M 当前**硬阻塞**——除非新增整个 Gemma3 trunk 实现，否则无法适配。
  i2_s / BitLinear 本身是 arch-agnostic，对 270M 同样适用。

## 3. Forward 集成所需的工作（未实施）

适配 `bitnet-embedding-0.6b` 到能 `--embed` 跑通，估计需要：

1. **`src/models/qwen3/trunk/bitlinear.rs`**（新文件）：BitLinear forward
   的 qwen3 适配，~300 行。每层每个 BitLinear 投影前：
   ```
   let x = rms_norm(&hidden, &norm_in_weight, eps)?;
   let x = quantize_activation_per_token(&x);  // -> (Vec<i8>, f32)
   bitlinear_forward(&weight_i2s, &x_q, absmax, n_in, n_out, &mut y_out)?;
   ```
2. **`src/models/qwen3/trunk/forward.rs`**：把 attn_q/k/v/o + ffn_gate/up/down
   7 个 matmul 全部切到 BitLinear forward；FFN 的 silu/gate 仍然走
   标准路径（在 BitLinear 之后做 silu(ffn_gate) * ffn_up）；attn output
   projection 也是 BitLinear（不是 Linear）。
3. **`src/models/qwen3/embedding.rs`**：加 `EmbeddingPooling::BitNetLast`
   variant 或在 `embedding_config` 里 detect file_type=40 → 强制 LAST pooling；
   最后做 L2 normalization（`x /= x.norm()`）。
4. **CLI 入口**：`rust-model-inference --model bitnet-0.6b.gguf --embed --prompt "text"`
   走新的 BitNet path（或者复用 `--embedding` 子命令，dispatch on file_type=40）。
5. **对齐测试**：写一个跟 bitnet.cpp `run_inference.py` 完全相同输入的测试
   （或写一个等价 Python 脚本从 safetensors 跑 BitLinear），把两边的
   per-token embedding last 1024-dim vector 做 bit-exact 对比（容差 ≤ 1 ULP
   或 2-bit 量化的 round-trip 误差范围）。

总估 ~1500-2000 行新代码 + 对齐测试。当前会话交付的是：

- ✓ `I2_S` GGMLType 占位（commit `2d08dba`）
- ✓ `i2_s` dequant kernel + 7 个单测（commit 后续）
- ✓ `BitLinear` ops 模块 + 4 个单测（commit 后续）
- ✓ `tests/bitnet_embedding_0_6b_q4_k_m.rs` 6/6 锁住 metadata contract（commit 后续）

未实施：1-5 项，按 §2.2 列出的依赖关系排优先级。

## 4. 参考实现（Oracle）

- `microsoft/BitNet`（已 clone 到 `/tmp/oracle-bitnet-cpp/BitNet/`）：
  - `src/ggml-bitnet-mad.cpp::quantize_i2_s` — i2_s GGUF packing 的
    MAD-path reference
  - `include/bitnet-lut-kernels.h` — 1170 行的 AVX2/NEON LUT kernel
    （BitNet b1.58 的真正高效 kernel，本机 4 核 + 7.5 GiB 没 AVX-512
    也跑不了完整 LUT 实现）
  - `docs/bitnet-embeddings-i2s-guide.md` — 完整的 I2_S 转换 + BitLinear
    forward + per-projection RMSNorm 流程
  - `utils/convert-bitnet-embedding-to-gguf.py` — safetensors → GGUF 转换脚本

## 5. 已知限制

- 270M（gemma3 backbone）**完全硬阻塞**——需要新写整个 gemma3 trunk
- 8B/22B Shieldstral 也走 BitLinear，但 arch 不同（mistral3 / qwen3-22B），
  同样需要对应的 trunk + 0.6B-style BitLinear 适配
- SIMD / LUT kernel（bitnet.cpp 的 1170 行 AVX2）性能远高于本仓库标量
  reference——若要在生产负载用 BitNet-Embedding，应在跑通标量路径后
  优先集成 `bitnet-lut-kernels.h` 的 TL1/TL2 分支（视本机 ISA）
