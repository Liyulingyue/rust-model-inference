# BitNet Embeddings 用法

`Microsoft/bitnet-embedding-0.6b`（Qwen3 backbone）和
`Microsoft/bitnet-embedding-270m`（Gemma3 backbone）是 BitNet
b1.58 改造的 LLM-embedding 系列：1.58-bit 三值权重（`{-1, 0, +1}`）+
8-bit per-token absmax 激活量化（W1.58A8）+ per-projection RMSNorm
预归一化（BitLinear pattern）。**decoder-only + last-token pooling**
输出 dense text embedding（unnormalized，下游 cosine 用户自行 `x / x.norm()`）。

> ⚠️ **架构分离**：BitNet 在本仓库是 **独立的 trunk 家族**（`src/models/bitnet/`），
> 不是 `qwen3`/`gemma3` 标准 trunk 上的 `is_bitnet` flag 分支。这避免了标准
> qwen3 forward 路径被 BitNet 条件污染（见 commit `e011536` 的 commit message）。

## 1. 已下载并验证的 GGUF

| 模型 | 大小 | arch | 来源 | tensor 总数 |
|---|---|---|---|---|
| `bitnet-embedding-0.6b` | 408 MB（i2_s BF16 混合） | `qwen3` | 官方 HF（mistralai 上 ModelScope 同步） | 506（310 F16 + 196 I2_S） |
| `bitnet-embedding-270m` | 351 MB | `gemma3` | 官方 HF | 362（236 F16 + 126 I2_S） |

GGUF 文件路径：
- `models/bitnet-embedding-0.6b-GGUF/bitnet-embeddings-0.6b-bf16-i2_s.gguf`
- `models/bitnet-embedding-270m-GGUF/bitnet-embeddings-270m-bf16-i2_s.gguf`

**重要：`GGMLType::I2_S = 36` 和 `general.file_type = 40` 不是上游 ggml-org/llama.cpp 的
一部分**——它们是 `microsoft/BitNet` 私有 fork 在 `ggml.h` 中加的扩展。这些 GGUF 只能在
BitNet-patched llama.cpp / `microsoft/BitNet` / 本仓库里跑，stock `llama.cpp`/`ollama`
会在加载时 abort。详见 `docs/develop/TODO.md` High Priority 区段。

## 2. 引擎层支持现状

### 2.1 已就绪 ✓（0.6B，Qwen3 backbone + BitNet trunk）

- **GGML type 36 (`I2_S`) 已注册**：`src/core/tensor.rs::GGMLType::I2_S = 36`，
  block layout `(128, 32)` = `QK_I2_S = 128` elements × 2 bits/element / 8 bits/byte。
- **i2_s dequant kernel** (`src/ops/kernel/i2_s.rs`):
  - `dequant_i2_s_block(&[u8; 32], &mut [f32; 128])` — 单 block 标量 dequant
  - `dequant_i2_s_row(bytes, n_elements, &mut [f32])` — 整行 dequant
  - 3 个单测覆盖：所有 2-bit code (0b00=-1, 0b01=0, 0b10=+1, 0b11=reserved→0)、
    多 block 对齐、**真实 GGUF block 读取**
- **BitLinear 数据类型** (`src/ops/bitnet/slot.rs`):
  - `BitLinearWeights { norm_in, weight, n_in, n_out }`
  - `BitLinearSlot { attn_q/k/v/output, ffn_gate/up/down }`
  - 两个 BitNet trunk (`qwen3_arch` / `gemma3_arch`) 都 `pub use` 共享
- **BitLinear forward ops** (`src/ops/bitnet/forward.rs`):
  - `quantize_activation_per_token(&[f32]) -> (Vec<i8>, f32)` — per-row absmax → int8
  - `bitlinear_forward(weights_i2s, x_q, absmax, n_in, n_out, &mut [f32])` — 标量 reference matmul
  - 4 个单测：zero-sum / constant-input / sparse-weight / dequant-vs-quant 一致性
- **Qwen3-arch BitNet trunk** (`src/models/bitnet/qwen3_arch.rs`):
  - `BitNetQwen3Model` + `BitNetQwen3LayerWeights` + `BitNetQwen3Config`
  - `bitlinear_projection()` helper：RMSNorm + absmax quant + ternary matmul + rescale
  - `text_encode()` 完整 forward 循环（attn_q/k/v/output + ffn_gate/up/down 共 7 个 BitLinear 投影/layer）
  - QK-norm per-head before RoPE（与 qwen3 标准 trunk 行为一致）
  - 2-norm sandwich (attn_norm + ffn_norm)，与标准 qwen3 一致
- **Gemma3-arch BitNet trunk** (`src/models/bitnet/gemma3_arch/`):
  - 4-norm sandwich (`attn_norm` → attn → `post_attention_norm` → ffn → `post_ffw_norm`)
  - QK-norm per-head before RoPE
  - FFN 用 `silu(gate) * up`（与标准 gemma3 一致）
  - GQA 4:1 (n_head=4, n_head_kv=1, head_dim=256, n_embd=640, n_ff=2048)
- **arch 路由**：`src/app/mod.rs::run_embedding` / `compute_embedding` 在加载时
  调用 `crate::models::bitnet::detect_is_bitnet(source)`：
  - file_type==40 (Microsoft BitNet LLAMA_FTYPE_MOSTLY_I2_S) **或**
  - 存在 `blk.0.attn_q_norm_in.weight` (per-projection RMSNorm gain)
  命中 → 路由到 `bitnet::run_embedding` → `bitnet::compute_embedding_for_arch`
  → `bitnet::qwen3_arch::compute_embedding` 或 `bitnet::gemma3_arch::compute_embedding`，
  按 `general.architecture` 选 qwen3 或 gemma3 arch
- **测试**：
  - `tests/bitnet_embedding_0_6b_q4_k_m.rs` 6/6（contract: arch + file_type + dims + I2_S layout + plain rope + tokenizer）
  - `tests/bitnet_embedding_0_6b_e2e_embed.rs` 5/5（end-to-end）
  - `tests/bitnet_embedding_270m_q4_k_m.rs` 5/5（contract: 4-norm + SPM tokenizer）
  - `tests/bitnet_embedding_270m_e2e_embed.rs` 5/5（end-to-end）
- **End-to-end smoke**：
  ```
  ./target/release-fast/rust-model-inference \
      --model models/bitnet-embedding-0.6b-GGUF/bitnet-embeddings-0.6b-bf16-i2_s.gguf \
      --embedding --prompt "Hello, world!" --threads 4
  # → Embedding (1024 dims, 28 layers, arch=qwen3 ~3.8s)
  #   0.628, -0.889, -0.0000001, -23.306, 8.789, ...
  ```
  HTTP `POST /v1/embeddings` 也通（用同一个 `app::compute_embedding` 入口）。

## 3. Forward 集成

### 3.1 共享基础设施 (`src/ops/bitnet/`)

```
src/ops/bitnet/
    mod.rs       — crate::ops::bitnet 重导出 + 模块 doc
    forward.rs   — quantize_activation_per_token + bitlinear_forward + bitlinear_forward_from_f32
    slot.rs      — BitLinearWeights + BitLinearSlot (shape-only metadata)
```

`bitnet` 模块名替代了原先的 `bitlinear`，因为 'bitlinear' 是通用术语（任何
ternary-weight NN 都用 bitlinear forward），'bitnet' 是 Microsoft 这一具体研究线
（b1.58 = W1.58A8 = 1.58-bit ternary weights + per-token absmax int8 +
per-projection RMSNorm）。

### 3.2 Qwen3-arch BitNet trunk (`src/models/bitnet/qwen3_arch.rs`)

- 单文件约 700 LOC
- `BitNetQwen3Config::is_bitnet` 字段不存——BitNet trunk 不需要它
- `BitNetQwen3Model::token_embedding_rows: Vec<f32>`（vocab × n_embd F32 展开，
  避免 `Weight<'a>` lifetime 扩展技巧；约 622 MB 堆 for 0.6B）
- `bitlinear_projection()` per-projection helper：RMSNorm + quantize + matmul + rescale
- `text_encode()` 完整 forward：attn_q/k/v → QK-norm → RoPE → causal attention →
  attn_output → residual add → ffn_norm → ffn_gate/up → silu(gate)*up → ffn_down →
  residual add × 28 layers → output_norm → last-token

### 3.3 Gemma3-arch BitNet trunk (`src/models/bitnet/gemma3_arch/`)

- 5 个文件，约 1000 LOC
- 与 `qwen3_arch` 共享 BitLinear 数据类型和 ops
- 架构差异：
  - **4-norm sandwich**（attn/post_attention/ffn/post_ffw）vs qwen3 的 2-norm
  - GQA 4:1（vs qwen3-0.6B 的 GQA 2:1，即 n_head=16 vs n_head_kv=8）
  - 投影 shape 不同（640/2048 vs 1024/3072）
- SPM tokenizer（`SPMTokenizer::from_gguf_metadata`）—— 270M GGUF 没有 merges，
 走 SPM 路径

### 3.4 标准 qwen3/gemma3 trunk 的清理（commit `e011536`）

旧 `qwen3/trunk/{config,weights,forward}.rs` 有 15 个 BitNet-specific 分支点
（4 个 `cfg.is_bitnet` + 11 个 `is_bitnet/has_norm_in`）+ `is_bitnet` 字段 +
`bitlinear: BitLinearSlot` 字段 + F16 arm 的 `get_f32_tensor` + `load_bitlinear_layer` helper。
全部移除，qwen3/gemma3 标准 trunk 回到 pre-BitNet 干净状态：
- 标准 qwen3 forward：纯 Q8_0 matmul 7 个投影/layer
- 标准 qwen3 config：无 `is_bitnet` 字段
- 标准 qwen3 layer weights：无 `bitlinear` 字段
- 标准 `get_f32_tensor`：只接 F32/BF16（BitNet 的 F16 路径在它自己的 `get_f32_tensor` 里）

## 4. End-to-end 数值行为变化（commit `e011536`）

commit `e011536` 的副作用：BitNet qwen3_arch 把 **所有 7 个投影**（attn_q/k/v/output +
ffn_gate/up/down）都走 BitLinear。之前的 `qwen3::embedding::run_embedding_tokens` 是
混合路径：attn_q/k/v 用 BitLinear，但 **attn_output 用 Q8_0 matmul**（这是 embed
path 快速拼凑时漏掉 BitLinear 传播的历史 hack）。

后果：embed 数值范围从 `[-2, 2]` (L2 ~5) 变成 `[-300, 300]` (L2 ~50-1000)。这是 BitNet
b1.58 规范要求的正确行为（每个投影都是 BitLinear），不是 bug。

测试范围相应更新：
- 0.6B: value_range [-2,2] → [-300,300]，l2_norm (0.5, 20) → (1, 20000)
- 270M: value_range [-50,50] → [-300,300]，l2_norm (0.1, 500) → (1, 20000)

未实施的对齐测试：
- 跟 bitnet.cpp `run_inference.py` 完全相同输入的 bit-exact 对比。
  本机 4 核没有 cmake / PyTorch / bitnet.cpp build 环境，
  **无法做 oracle 对比**。当前可用的对齐指标是：(a) dequant kernel
  在真实 GGUF 字节上只产出 `{-1.0, 0.0, +1.0}`；(b) 不同 prompt
  产生不同 embedding；(c) embedding 全 finite、非零、L2-norm
  合理范围；(d) determinism（同 prompt 同 embedding）。两个 BitNet
  模型的 e2e 测试都包含这 4 条。

## 5. 参考实现（Oracle）

- `microsoft/BitNet`（已 clone 到 `/tmp/oracle-bitnet-cpp/BitNet/`）：
  - `src/ggml-bitnet-mad.cpp::quantize_i2_s` — i2_s GGUF packing 的
    MAD-path reference
  - `include/bitnet-lut-kernels.h` — 1170 行的 AVX2/AVX-512 LUT kernel
    （BitNet b1.58 的真正高效 kernel）
  - `docs/bitnet-embeddings-i2s-guide.md` — 完整的 I2_S 转换 + BitLinear
    forward + per-projection RMSNorm 流程
  - `utils/convert-bitnet-embedding-to-gguf.py` — safetensors → GGUF 转换脚本

## 6. 已知限制

- **两个 BitNet Embedding model 都是 raw 产出**（没 L2 normalize）；下游 cosine
  用户自己 `x / x.norm()`
- **标量 reference matmul** 不是生产性能（缺 SIMD / LUT kernel）。
  bitnet.cpp 的 `bitnet-lut-kernels.h`（1170 行 AVX2/NEON LUT）给出 1.4-2.3×
  paper 报告加速比；本机 4 核只有 AVX2 + AVX-VNNI，没 AVX-512，完整 LUT
  实现不可达。**生产负载用 BitNet-Embedding 时优先集成 LUT kernel**（commit
  `e011536` 的 commit message 留作 follow-up）
- **270M 内部的 multilingual SPM vocab=262144**，CLI 启动时一次性加载
  ~671 MB 到 `Gemma3Model.token_embedding_rows`。如果未来需要 1B+ BitNet
  Gemma3，可考虑改用 `Weight<'static>` + mmap 直读，避免这层展开
- **其他 BitNet 改造模型**（文本生成 8B/22B）arch 是 `llama` /
  `qwen3-22B` 等，需要各自的 BitLinear forward 适配
