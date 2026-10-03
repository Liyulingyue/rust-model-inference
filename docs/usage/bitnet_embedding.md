# BitNet Embeddings 用法

`Microsoft/bitnet-embedding-0.6b`（Qwen3 backbone）和
`Microsoft/bitnet-embedding-270m`（Gemma3 backbone）是 BitNet
b1.58 改造的 LLM-embedding 系列：1.58-bit 三值权重（`{-1, 0, +1}`）+
8-bit per-token absmax 激活量化（W1.58A8）+ per-projection RMSNorm
预归一化（BitLinear pattern）。**decoder-only + last-token pooling +
L2 normalization** 输出 dense text embedding。

## 1. 已下载并验证的 GGUF

| 模型 | 大小 | arch | 来源 | tensor 总数 |
|---|---|---|---|---|
| `bitnet-embedding-0.6b` | 408 MB（i2_s BF16 混合） | `qwen3` | 官方 HF（mistralai 上 ModelScope 同步） | 506（310 F16 + 196 I2_S） |
| `bitnet-embedding-270m` | 351 MB | `gemma3`（**未支持**） | 官方 HF | 362（236 F16 + 126 I2_S） |

GGUF 文件路径：
- `models/bitnet-embedding-0.6b-GGUF/bitnet-embeddings-0.6b-bf16-i2_s.gguf`
- `models/bitnet-embedding-270m-GGUF/bitnet-embeddings-270m-bf16-i2_s.gguf`

## 2. 引擎层支持现状

### 2.1 已就绪 ✓（0.6B，Qwen3 backbone）

- **GGML type 36 (`I2_S`) 已注册**：`src/core/tensor.rs::GGMLType::I2_S = 36`，
  block layout `(128, 32)` = `QK_I2_S = 128` elements × 2 bits/element / 8 bits/byte。
- **i2_s dequant kernel** (`src/ops/kernel/i2_s.rs`):
  - `dequant_i2_s_block(&[u8; 32], &mut [f32; 128])` — 单 block 标量 dequant
  - `dequant_i2_s_row(bytes, n_elements, &mut [f32])` — 整行 dequant
  - `is_i2_s_aligned(n_elements)` / `i2_s_row_bytes(n_elements)`
  - 3 个单测覆盖：所有 2-bit code (0b00=-1, 0b01=0, 0b10=+1, 0b11=reserved→0)、
    多 block 对齐、**真实 GGUF block 读取**（确认从 Mistral 官方转换出来
    的字节落到 `{-1.0, 0.0, +1.0}` 而不是被误读 scale 字节弄出垃圾值）
- **BitLinear forward ops** (`src/ops/bitlinear.rs`):
  - `quantize_activation_per_token(&[f32]) -> (Vec<i8>, f32)` — per-row absmax → int8
  - `bitlinear_forward(weights_i2s, x_q, absmax, n_in, n_out, &mut [f32])` — 标量 reference matmul
  - 4 个单测：zero-sum / constant-input / sparse-weight / dequant-vs-quant 一致性
- **qwen3 trunk BitLinear 接入**：`src/models/qwen3/trunk/forward.rs::bitlinear_projection`
  作为统一的 per-projection RMSNorm + 量化 + ternary matmul helper；
  `text_encode` 的 7 个 matmul 站点（attn_q/k/v/o + ffn_gate/up/down）在
  `cfg.is_bitnet` 时全部切到 BitLinear forward；attn_score / FFN
  silu(gate)·up / residual sum 路径不变。FFN 的 `silu(gate) * up` 在
  BitLinear 之后仍按标准路径执行（BitLinear 替代的是 Linear，不是 activation）。
- **embedding.rs 接入**：`src/models/qwen3/embedding.rs::embed_bitlinear`
  复用同一 helper；7 个站点同样分支。Pooling: BitNet 模式下 `qwen3.pooling_type=1`
  走 `EmbeddingPooling::Last`（Qwen3-Embedding 走 Mean，BitNet 走 Last，靠
  `file_type==40 || has_norm_in` 区分）；不做 L2 normalization（保留 raw，
  让上层 cosine 时自己归一化）。
- **Qwen3Config::is_bitnet 自动检测**：从 `general.file_type == 40` 或
  存在 `*_norm_in` tensor 推导；`load_layers_static` 和 `load_layers` 都
  接受 `is_bitnet` 参数；BitLinear 权重放进 `Qwen3LayerWeights.bitlinear`
  字段（`BitLinearSlot { attn_q/k/v/output, ffn_gate/up/down: Option<BitLinearWeights> }`）。
- **关键 bug 修复**：`get_f32_tensor` 旧实现只接受 F32/BF16，F16 的
  `*_norm_in` RMSNorm 权重被 silently 初始化为 0，导致 RMSNorm(0) = 0
  → 全部 BitLinear 输出 0 → embedding 全 0。修复后 end-to-end 跑出
  非零 1024-dim embedding。
- **GGUF loader 兼容**：测试 `tests/bitnet_embedding_0_6b_q4_k_m.rs` 6/6 通过，
  锁住 0.6B 的 arch、file_type=40、Qwen3 dims、plain RoPE（无 YaRN）、
  tokenizer、506 tensor inventory、I2_S row layout、per-projection `*_norm_in`
  RMSNorm 存在性。
- **End-to-end smoke**：
  ```
  ./target/release-fast/rust-model-inference \
      --model models/bitnet-embedding-0.6b-GGUF/bitnet-embeddings-0.6b-bf16-i2_s.gguf \
      --embedding --prompt "Hello, world!" --threads 4
  # → Embedding (1024 dims, 28 layers, arch=qwen3 ~7s)
  #   -0.005512016 -0.177880913 -0.000000001 -0.168598458 0.077839665 ...
  ```
  不同 prompt 产生不同 embedding；embedding_raw 模式下输出完整 1024-dim
  float32 向量。

### 2.2 未支持 ☐

- **270M（gemma3 backbone）**：本仓库**没有 gemma3 trunk**。270M GGUF 字节
  能正确读出（`dump_meta` + `dump_tensors` 都过），i2_s dequant 同样适用
  （arch-agnostic），但 forward 必须等 gemma3 trunk 写出来——这是一个
  独立的大型工作，**当前硬阻塞**。
- **L2 normalization**：BitLinear 输出本身就是 rough scale（int8 × ternary，
  absmax rescale），上层的 cosine 用户通常自己 `x / x.norm()`。如果需要
  CLI 自动归一化，在 `--embedding-output raw` 之外的 mode 里加一行即可。
- **SIMD / LUT kernel**：标量 reference matmul 是 correctness-only。
  bitnet.cpp 的 `bitnet-lut-kernels.h`（1170 行 AVX2/NEON LUT）给出
  1.4–2.3× 的 paper 报告加速比；本机 4 核只有 AVX2 + AVX-VNNI，没 AVX-512，
  完整 LUT 实现不可达。**生产负载用 BitNet-Embedding 时优先集成 LUT kernel**。

## 3. Forward 集成（已实施）

上一版文档原本估 1500-2000 行新代码；最终实际改动 ≈849 行
（8 文件修改），其中绝大部分是 BitLinear forward 复用 qwen3 trunk 现有
matmul 站点而不是新写一份：

1. **`src/core/tensor.rs::GGMLType::I2_S = 36`**（先前 commit `2d08dba`）
2. **`src/ops/kernel/i2_s.rs`**：i2_s dequant kernel + 3 个单测
3. **`src/ops/bitlinear.rs`**：W1.58A8 BitLinear ops + 4 个单测
4. **`src/ops/kernel/quantized_tensor.rs::I2S`** variant +
   noop kernel 让 `Weight::from_quantized` 不 panic（实际 BitLinear 走
   `BitLinearWeights::weight` 原始字节，不走 kernel）
5. **`src/models/qwen3/trunk/config.rs::is_bitnet`** 字段 +
   `src/models/qwen3/trunk/weights.rs::BitLinearWeights` + `BitLinearSlot`
   + `load_bitlinear_layer`（含 `_static` / `_borrowed` 两个入口）
6. **`src/models/qwen3/trunk/forward.rs::bitlinear_projection`** helper +
   `text_encode` 7 个站点分支
7. **`src/models/qwen3/embedding.rs::embed_bitlinear`** helper +
   `run_embedding_tokens` 7 个站点分支 + pooling type 1 → Last（BitNet）
8. **get_f32_tensor F16 arm 修复**：`models/qwen3/trunk/weights.rs::get_f32_tensor`
   之前只匹配 F32/BF16，F16 tensor 静默落到 `vec![0.0; expected_len]`，
   BitNet 的所有 `*_norm_in` RMSNorm 权重因此被 zero-init，cascade 到
   所有 BitLinear 输出 0 → embedding 全 0。修复后 smoke 跑出真实非零
   1024-dim embedding。

未实施的对齐测试：
- 跟 bitnet.cpp `run_inference.py` 完全相同输入的 bit-exact 对比。本机
  4 核没有 cmake / PyTorch / bitnet.cpp build 环境，**无法做 oracle 对比**。
  当前可用的对齐指标是：(a) dequant kernel 在真实 GGUF 字节上只产出
  `{-1.0, 0.0, +1.0}`；(b) 不同 prompt 产生不同 embedding；(c) embedding
  全 finite、非零、L2-norm 合理范围。

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
- 其他 BitNet 改造模型（如 bitnet.cpp 的 8B/22B 文本生成）arch 是
  `llama` / `qwen3-22B` 等，需要各自的 BitLinear forward 适配；本工作
  只覆盖 qwen3 trunk 的 embedding-style 路径
- SIMD / LUT kernel（bitnet.cpp 的 1170 行 AVX2）性能远高于本仓库标量
  reference——若要在生产负载用 BitNet-Embedding，应在跑通标量路径后
  优先集成 `bitnet-lut-kernels.h` 的 TL1/TL2 分支（视本机 ISA）
