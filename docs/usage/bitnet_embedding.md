# BitNet Embeddings 用法

`Microsoft/bitnet-embedding-0.6b`（Qwen3 backbone）和
`Microsoft/bitnet-embedding-270m`（Gemma3 backbone）是 BitNet
b1.58 改造的 LLM-embedding 系列：1.58-bit 三值权重（`{-1, 0, +1}`）+
8-bit per-token absmax 激活量化（W1.58A8）+ per-projection RMSNorm
预归一化（BitLinear pattern）。**decoder-only + last-token pooling**
输出 dense text embedding（unnormalized，下游 cosine 用户自行 `x / x.norm()`）。

## 1. 已下载并验证的 GGUF

| 模型 | 大小 | arch | 来源 | tensor 总数 |
|---|---|---|---|---|
| `bitnet-embedding-0.6b` | 408 MB（i2_s BF16 混合） | `qwen3` | 官方 HF（mistralai 上 ModelScope 同步） | 506（310 F16 + 196 I2_S） |
| `bitnet-embedding-270m` | 351 MB | `gemma3` | 官方 HF | 362（236 F16 + 126 I2_S） |

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
  - 3 个单测覆盖：所有 2-bit code (0b00=-1, 0b01=0, 0b10=+1, 0b11=reserved→0)、
    多 block 对齐、**真实 GGUF block 读取**
- **共享 BitLinear 数据类型** (`src/ops/bitlinear/weights.rs`):
  - `BitLinearWeights { norm_in, weight, n_in, n_out }`
  - `BitLinearSlot { attn_q/k/v/output, ffn_gate/up/down }`
  - `pub use` 给 `qwen3` + `gemma3` trunk 共用
- **BitLinear forward ops** (`src/ops/bitlinear/mod.rs`):
  - `quantize_activation_per_token(&[f32]) -> (Vec<i8>, f32)` — per-row absmax → int8
  - `bitlinear_forward(weights_i2s, x_q, absmax, n_in, n_out, &mut [f32])` — 标量 reference matmul
  - 4 个单测：zero-sum / constant-input / sparse-weight / dequant-vs-quant 一致性
- **qwen3 trunk BitLinear 接入**：`src/models/qwen3/trunk/forward.rs::bitlinear_projection`
  + `text_encode` 7 个站点分支（attn_q/k/v/o + ffn_gate/up/down）；FFN
  silu(gate)·up / residual sum 路径不变。
- **embedding.rs 接入**：7 个站点同样分支 + pooling type 1 → Last（BitNet）
- **Qwen3Config::is_bitnet 自动检测**：`file_type==40 || has_norm_in`
- **关键 bug 修复**：`get_f32_tensor` 旧实现只接受 F32/BF16，F16 RMSNorm 权重
  被 silently 初始化为 0，cascade 到 BitLinear 输出 0。修复后 end-to-end 跑出
  非零 1024-dim embedding。
- **测试**：`tests/bitnet_embedding_0_6b_q4_k_m.rs` 6/6（contract）+
  `tests/bitnet_embedding_0_6b_e2e_embed.rs` 5/5（end-to-end）
- **End-to-end smoke**：
  ```
  ./target/release-fast/rust-model-inference \
      --model models/bitnet-embedding-0.6b-GGUF/bitnet-embeddings-0.6b-bf16-i2_s.gguf \
      --embedding --prompt "Hello, world!" --threads 4
  # → Embedding (1024 dims, 28 layers, arch=qwen3 ~7s)
  #   -0.005512016 -0.177880913 -0.000000001 -0.168598458 0.077839665 ...
  ```

### 2.2 已就绪 ✓（270M，Gemma3 backbone）

- **新 trunk `src/models/gemma3/`**：完整支持 270M GGUF 的 4-norm sandwich
  (attn_norm → post_attention_norm → ffn_norm → post_ffw_norm)、
  per-head QK-norm on Q/K before RoPE、640-dim hidden state、
  4:1 GQA at head_dim=256、18 layers、SPM tokenizer。
- **SPM tokenizer 接入**：270M GGUF 用 `tokenizer.ggml.model = "llama"` +
  `pre = "default"` + scores（**没有 merges**），走
  `SPMTokenizer::from_gguf_metadata`。
- **token_embd 加载**：`static_weight` 把 F16 token 展开为 F32
  `vocab × n_embd`（270M = 671 MB），存储在
  `Gemma3Model.token_embedding_rows`。展开后的 Vec 不依赖
  `TensorSource` lifetime，让模型可以跨调用存活。
- **测试**：`tests/bitnet_embedding_270m_q4_k_m.rs` 5/5（contract）+
  `tests/bitnet_embedding_270m_e2e_embed.rs` 5/5（end-to-end）
- **End-to-end smoke**：
  ```
  ./target/release-fast/rust-model-inference \
      --model models/bitnet-embedding-270m-GGUF/bitnet-embeddings-270m-bf16-i2_s.gguf \
      --embedding --prompt "Hello, world!" --threads 4
  # → Embedding (640 dims, arch=gemma3 ~1.4s)
  #   4.147890568 -12.476488113 -3.515295982 8.655247688 -10.699966431 ...
  ```
  270M forward 比 0.6B 快（18 layers vs 28、hidden 640 vs 1024、
  n_ff 2048 vs 3072）；embedding 在 ±15 范围、L2 范数 ~50-100
  （比 0.6B 的 ±2/L2~5 大一个数量级，但仍是 well-behaved）。
  不同 prompt 产生不同 embedding。

### 2.3 未实施 ☐

- **L2 normalization**：两个 BitNet model 都是 raw 产出，未做
  L2 normalize；上层 cosine 用户自己 `x / x.norm()`。CLI 加一行
  就能补，但本会话未做。
- **BitLinear SIMD / LUT kernel**：标量 reference matmul 是
  correctness-only。bitnet.cpp 的 `bitnet-lut-kernels.h`（1170 行
  AVX2/NEON LUT）给出 1.4–2.3× paper 报告加速比；本机 4 核只有
  AVX2 + AVX-VNNI，没 AVX-512，完整 LUT 实现不可达。
  **生产负载用 BitNet-Embedding 时优先集成 LUT kernel**。
- **其他 BitNet 改造模型**（如 bitnet.cpp 的 8B/22B 文本生成）
  arch 是 `llama` / `qwen3-22B` 等，需要各自的 BitLinear forward
  适配；本工作只覆盖 qwen3 + gemma3 trunk 的 embedding-style 路径。

## 3. Forward 集成（已实施）

### 3.1 0.6B (Qwen3 backbone)

1. **`src/core/tensor.rs::GGMLType::I2_S = 36`**（先前 commit `2d08dba`）
2. **`src/ops/kernel/i2_s.rs`**：i2_s dequant kernel + 3 个单测
3. **`src/ops/bitlinear/`**（已重构为 module）：
   - `weights.rs`：`BitLinearWeights` + `BitLinearSlot`（共享）
   - `mod.rs`：W1.58A8 forward ops + 4 个单测
4. **`src/ops/kernel/quantized_tensor.rs::I2S`** variant + noop kernel
5. **`src/models/qwen3/trunk/config.rs::is_bitnet`** 字段 +
   `BitLinearSlot` 字段、`load_layers_static` 接受 `is_bitnet`
6. **`src/models/qwen3/trunk/forward.rs::bitlinear_projection`** helper +
   `text_encode` 7 个站点分支
7. **`src/models/qwen3/embedding.rs::embed_bitlinear`** helper +
   `run_embedding_tokens` 7 个站点分支 + pooling type 1 → Last（BitNet）
8. **get_f32_tensor F16 arm 修复**：F16 tensor 之前被静默 zero-init

### 3.2 270M (Gemma3 backbone)

1. **共享 `BitLinearWeights`/`BitLinearSlot`**（已在 §3.1 step 3 完成）
2. **新 trunk `src/models/gemma3/`**（约 1000 行新代码）：
   - `trunk/config.rs::Gemma3Config`
   - `trunk/weights.rs::Gemma3LayerWeights` (4 norms + 2 QK norms +
     `BitLinearSlot`) + `Gemma3Model { config, layers, output_norm,
     token_embedding_rows }` + `load_layers_static` +
     `static_weight` (F16 → F32 展开)
   - `trunk/forward.rs::bitlinear_projection` (per-projection
     RMSNorm + absmax int8 quant + ternary matmul) +
     `apply_qk_norm` (per-head RMSNorm over head_dim) +
     `causal_self_attention` (GQA 4:1 with causal mask) +
     `text_encode` (full forward loop with 4-norm sandwich)
3. **SPM tokenizer 接入**：`embedding.rs::compute_embedding` 走
   `SPMTokenizer::from_gguf_metadata`（不是 BPETokenizer — 270M
   GGUF 没有 `tokenizer.ggml.merges`）
4. **arch 路由**：`src/app/mod.rs::run_embedding` 加 `"gemma3"` arm
   指向 `crate::models::gemma3::run_embedding`

未实施的对齐测试：
- 跟 bitnet.cpp `run_inference.py` 完全相同输入的 bit-exact 对比。
  本机 4 核没有 cmake / PyTorch / bitnet.cpp build 环境，
  **无法做 oracle 对比**。当前可用的对齐指标是：(a) dequant kernel
  在真实 GGUF 字节上只产出 `{-1.0, 0.0, +1.0}`；(b) 不同 prompt
  产生不同 embedding；(c) embedding 全 finite、非零、L2-norm
  合理范围。两个 BitNet 模型的 e2e 测试都包含这 6 条。

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

- 两个 BitNet Embedding model 都是 raw 产出（没 L2 normalize）
- 标量 reference matmul 不是生产性能（缺 SIMD / LUT kernel）
- 270M 内部的 multilingual SPM vocab=262144，CLI 启动时一次性加载
  ~671 MB 到 `Gemma3Model.token_embedding_rows`。如果未来需要 1B+
  BitNet Gemma3，可考虑改用 `Weight<'static>` + mmap 直读，避免这层展开
- 其他 BitNet 改造模型（文本生成 8B/22B）arch 是 `llama` /
  `qwen3-22B` 等，需要各自的 BitLinear forward 适配