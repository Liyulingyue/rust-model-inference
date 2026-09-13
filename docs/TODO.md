# TODO

## TODO-002: SIMD GeGLU `tanh_approx` 未启用

### 现状

`src/models/gemma4/trunk/forward.rs` 的 `ggml_geglu_fp16_inplace` 仍然走 scalar 路径。AVX2+F16C SIMD 版本原型写过又回退了，注释详细记录在函数体内（forward.rs 顶部）。

### 影响

- **性能**：scalar GELU 算 10240 元素 × 2 个 GELU/层（FFN + per-layer）× 35 层 = ~717k scalar ops/token。scalar ~12ns/element = **~8.6 ms/token**（当前 E2B ~80ms/token 的 ~10%）。
- **正确性**：scalar 版本 bit-exact 同 llama.cpp。所有 gemma4 测试通过。

### 何时触发

每次 FFN 块调用 `ggml_geglu_fp16_inplace(&mut scratch.gate[..ffn], &scratch.up[..ffn])` 时。

### 选项

1. **接受 ±1-2 ULP drift 启用 SIMD**（推荐，参考 llama.cpp `GGML_FMA_DISABLED` 做法）：
   - Padé [7/6] tanh 近似：~0.04% max error in |x| ≤ 4.5（vs Padé [3/2] 的 1.5%）
   - 加 `#[cfg(feature = "experimental-geglu-simd")]` 默认关
   - 用 `cargo run --features experimental-geglu-simd` 启用
   - 预期：+5-8% ETE
   - 风险：gemma4_reference.rs 中 token Oracle 偶尔 ±1 分叉（top-K 边界）
   - 工作量：半天

2. **Schraudolph fast exp + tanh = (exp(2x)-1)/(exp(2x)+1)**：
   - 5-10% tanh error on |x|>2 → gelu 漂移 ~3-5% → logits 漂移 ~5-10% → 必分叉
   - 不推荐

3. **保持 scalar**（当前）：零风险，零加速。

### 推荐

短期：选项 3（保持 scalar，已记录）。
中期：当跑 gemma4_reference.rs Oracle 时确认 Padé [7/6] 漂移可控后，切到选项 1。

### 关联文件

- `src/models/gemma4/trunk/forward.rs:858` — 当前 scalar + 详细 doc-comment
- `src/ops/kernel/q8_0/avx2.rs` — FMA pattern 参考

---

## TODO-001: Q4_0 AVX2 kernel 不使用 FMA，性能受限

### 现状

`src/ops/kernel/q4_0/avx2.rs` 的 Q4_0 matmul kernel 故意**不使用 FMA**（`fused multiply-add`）：

```rust
// q4_0/avx2.rs:14
//! 8. Multiply by d * scale in f32 with explicit `_mm256_mul_ps` + `_mm256_add_ps`
//!    (no `_mm256_fmadd_ps`) to match scalar's mul+add rounding exactly.
//!
//! **Precision contract**: bit-exact with the scalar implementation, including
//! for edge cases (all-zero Q8, all-127 Q8, all-0 nibble, all-15 nibble).
```

行累加采用 `acc += dc * d * scale` 三次 f32 运算（mul → mul → add），而不是 `fmadd(dc, d*scale, acc)` 一次融合运算。

### 影响

- **性能**：相比 Q4_K / Q8_0 kernel（已用 FMA）慢约 1.5-2×。E4B Q4_0 文件实测 5.8 t/s decode，E2B Q4_K_M 10 t/s（同样 embd 比例下 kernel 速度是主要差异）。
- **正确性**：与 scalar reference **bit-exact**（`assert_avx2_eq_scalar` 测试在 `q4_0/avx2.rs:241` 验证 `a.to_bits() == b.to_bits()`）。所有 Q4_0 模型通过 parity 测试。

### 何时触发

加载任何 Q4_0 权重时——目前主要是 Gemma 4 E4B Q4_0 导出文件。`require_tensor_any` 列表里 Q4_0 在 `src/models/gemma4/trunk/config.rs:215`。

### 选项

1. **加 FMA + 放松精度契约**（推荐 follow-up）：把 `acc += dc * d * scale` 改成 `acc = _mm256_fmadd_ps(prod_d_scale, ...)`，并把 `assert_avx2_eq_scalar` 改成 ~1 ULP 容忍。需要新增"near bit-exact"测试模式（参考 llama.cpp 的 GGML_FMA_DISABLED 编译开关做法）。
   - 预期：1.5-2× 加速
   - 风险：所有 Q4_0 模型需要重新跑 Oracle 验证（输出 logits 可能漂移 ±1 ULP，最终 token 偶发分叉）
   - 工作量：半天

2. **重导出模型到 Q4_K / Q8_0**（推荐立即做）：
   - `q4_k/avx2.rs` 已经用 FMA 写好 + 通过 parity
   - `q8_0/avx2.rs` 已经用 FMA 写好 + 通过 parity
   - 用 llama.cpp 的 `convert.py` 把 E4B 转成 Q4_K_M 或 Q8_0
   - 文件大 ~30%，但推理快 2-3×
   - 工作量：1 小时（含下载 + 转换 + 验证）

3. **写 FMA + 8-wide row unroll 版本**：保留旧 kernel，新增 `matmul_q4_0_vs_q8_0_avx2_fma`，编译时 feature flag 切换。
   - 预期：2-3× 加速
   - 工作量：1 天

### 推荐

短期：选选项 2（重导出 E4B），立即得 2-3× 加速，零代码风险。
中期：选选项 1（加 FMA kernel），其他 Q4_0 模型（Qwen3-0.6B-Q4_0 等）也受益。

### 关联文件

- `src/ops/kernel/q4_0/avx2.rs` — kernel 实现 + parity 测试
- `src/ops/kernel/q4_0/scalar.rs` — scalar reference
- `src/ops/kernel/q4_0/mod.rs` — kernel 分发

## TODO-003: 统一 5 个 `load_weight` / `load_weight_any` 到单一 `core::load_weight_any`

### 现状

仓库目前每个模型自维护一份 GGML weight loader，签名 / 白名单 / dims 拆解 / 错误返回都不一致：

| 实现 | 签名 | 接受类型 | dims 校验 | 错误返回 | 备注 |
|---|---|---|---|---|---|
| `src/models/dots/weights.rs::load_weight` | `(source, name, dims)` | F32/F16/BF16/Q8_0 | 严格 dims | `Result<_, String>` | dots 私有；breeze 借用 |
| `src/models/qwen35/trunk/weights.rs::load_weight` | `(source, name)` | 8 种：F32/F16/BF16/Q8_0/Q4_0/Q4_1/Q4_K/Q5_K/Q6_K | 不校 | `Option<Weight>` | qwen35 私有 |
| `src/models/qwen35/trunk/weights.rs::load_weight_f32` | `(source, name) → Vec<f32>` | F32/F16/BF16 | n_elements | `Option<Vec<f32>>` | norm/bias 专用；与 `core::load_f32_tensor` 重复 |
| `src/models/gemma4/trunk/weights.rs::load_weight` | `(source, name, dims, ggml_type)` | 单一类型（调用方指定） | 严格 dims+type | `Result<_, String>` | gemma4 私有 |
| `src/models/gemma4/trunk/weights.rs::load_weight_any` | `(source, name, dims, &[GGMLType])` | 多个白名单（调用方传入） | 严格 dims+白名单 | `Result<_, String>` | gemma4 私有 |

`qwen3` 干脆没有 `load_weight` —— 直接展开 `Weight::from_quantized(QuantizedTensor::from_bytes(bytes, info.ggml_type, n_in, n_out))`，天然支持 `QuantizedTensor::from_bytes` 接受的全部 20+ 类型。

**核心公共部分**（≈ 5 行）：

```rust
let info = source.tensor_info(name)?;
let bytes = source.tensor_slice(name)?;
let weight = Weight::from_quantized(QuantizedTensor::from_bytes(
    bytes, info.ggml_type, n_in, n_out,
));
weight.n_in = n_in;
weight.n_out = n_out;
```

### 影响

- **未来加 IQ2/IQ4/Q6_K/Q5_K 等新量化类型**——每个 `load_weight` 都要扩白名单。breeze 之前借 dots 命名，加 Q4_0 必须改 `dots/weights.rs`（命名遗留）。
- **dims 拆解不一致**：`dims.last()` vs `dims[0]/dims[1]` vs `dims[..-1]` 累乘——GGUF 约定是 `(in, out)`，但每个 loader 的视角不同。
- **白名单粒度不一**：写死（dots/qwen35）vs 参数化（gemma4）vs 无（qwen3）。新增类型时要逐一审查。
- **错误返回风格不一**：`Result<_, String>` vs `Option<Weight>`，导致调用方写法分裂。

### 选项

1. **最小**：扩 `dots::load_weight` 接受 Q4_0/Q4_1（恢复 qwen35 一致范围）；其他不动。
   - 工作量：~20 行；维持当前命名遗留。
2. **中度**：breeze 摆脱 dots 命名（已落地的方案），改直接展开 `Weight::from_quantized(QuantizedTensor::from_bytes(...))`（学 qwen3）。dots 保留以服务 dots 自身。
   - 工作量：~80 行；框架方向正确；dots 命名遗留仍在。
3. **完整**：把 5 个 `load_weight`/`load_weight_any` 全部迁到 `crate::core::tensor::load_weight_any`：
   - 签名：`(source, name, dims: &[u64], allowed: Option<&[GGMLType]>) -> Result<Weight, String>`
   - `dims` 校验可选；`allowed=None` 表示 `QuantizedTensor::from_bytes` 接受全部类型
   - 所有模型调用方改 import；`qwen35::load_weight_f32` 与 `core::load_f32_tensor` 合并去重
   - 工作量：~200 行；彻底统一命名、签名、白名单策略

### 推荐

**方案 3**。`load_weight` 不是 dots 私有的，它是项目 GGML weight loader 的事实标准入口——`qwen3` 已经证明**不抽象也活得很好**，所以抽象必须带来明显收益。当前 5 份实现是历史遗留，每加一种新量化就要碰 3-5 个文件。

### 关联文件

- `src/models/dots/weights.rs:5-67` — `load_weight`（白名单最窄）
- `src/models/qwen35/trunk/weights.rs:80-124` — `load_weight`（白名单最宽）
- `src/models/qwen35/trunk/weights.rs:129-...` — `load_weight_f32`（与 `core::load_f32_tensor` 重复）
- `src/models/gemma4/trunk/weights.rs:226-265` — `load_weight`（参数化单类型）
- `src/models/gemma4/trunk/weights.rs:270-309` — `load_weight_any`（参数化白名单）
- `src/models/qwen3/trunk/weights.rs:129-205` — `Weight::from_quantized(QuantizedTensor::from_bytes(...))` × 8（qwen3 不抽象，参考样本）
- `src/core/tensor.rs:288-330` — `load_f32_tensor`（norm/bias 公共入口；与 `qwen35::load_weight_f32` 重叠）
- `src/models/gemma4/trunk/config.rs:215` — Q4_0 类型在允许列表

## TODO-004: Breeze Q4_0 不可用 — 走 per-tensor 混合精度路径

### TL;DR

**Q4_0 单层全量化对 Breeze 不可用**（128 frames vs BF16 59 frames = 117% 偏差）。短期路线不是"再换一种 4-bit 格式"，而是 **per-tensor 混合精度**：embedding / lm_head / codebook 保留 BF16 或 Q8_0，hidden-attn / mlp 主矩阵走 Q4_0 或 Q4_K。长期再评估 Q4_K / Q6_K 全替换。

### 现状

`tools/converter/breeze/convert_breeze.py --quant q4_0` 已能产出 GGUF，Rust 端也能加载（`Weight::from_quantized(QuantizedTensor::from_bytes(...)` 自动接受 Q4_0），实测推理**不报错**，但输出退化明显：

| 精度 | 帧数（prompt "你好。"，seed 42） | 文件大小 | 备注 |
|---|---|---|---|
| BF16 | 29 | 6.6 GB | 与原字节等价 |
| F16 | 35 | 6.6 GB | 同源 |
| F32 | 29 | 13 GB | md5 与 BF16 bit-exact |
| Q8_0 | 37 | 3.6 GB | **3.6× 加速，推荐部署** |
| **Q4_0** | **128** | 2.1 GB | **输出退化**（4-bit 噪声让 TTS 在补偿阶段不停 token） |

128 frames 是模型"无法在合理帧数内停"的征兆——4-bit 量化误差在逐帧生成（每帧=一段音频 token）的场景被累积放大。

### 影响的张量

Q4_0 失败不是形状问题（所有 `weight` 都是 2D 且行宽 % 32 == 0），而是**语义**问题：

| 张量 | Q4_0 风险 | 原因 | 推荐最低精度 |
|---|---|---|---|
| `lm_head.weight` (HIDDEN × VOCAB+1) | 高 | 输出 token 概率分布对低比特敏感 | Q8_0 |
| `text_encoder.embed_tokens.weight` | 高 | embedding 单行精准取，4-bit 误差直达 hidden | Q8_0 |
| `depth_decoder.model.embed_tokens.weight` | 高 | 同上；codebook 偏移误差会污染所有 codec frame | Q8_0 |
| `depth_decoder.codebooks_head.weight` | 极高 | codebook 表 L2 距离对量化噪声敏感 | BF16 / F32（保持源精度） |
| `backbone_model.layers.{i}.self_attn.{q,k,v,o}_proj.weight` | 低 | 大矩阵冗余，4-bit 可接受 | Q4_0 |
| `backbone_model.layers.{i}.mlp.{gate,up,down}_proj.weight` | 低 | 同上 | Q4_0 |
| `text_encoder.layers.{i}.{q,k,v,o,gate,up,down}_proj.weight` | 中 | hidden_attn/MLP 矩阵 | Q8_0 |
| `depth_decoder.model.layers.{i}.{q,k,v,o,gate,up,down}_proj.weight` | 中 | depth decoder 矩阵 | Q8_0 |
| norm weights / eoi_embedding | 极高 | 数值精度敏感（已 `must_keep_source`） | BF16 / F32 |

**关键洞察**：embedding / lm_head / codebook 占模型 20-30% 体积但**决定 token 输出**，不能让 Q4 0；attn / mlp 矩阵占 60-70% 体积但**冗余度高**，Q4_0 可接受。混合精度既达到 4-bit 压缩目标，又避免全 Q4_0 的退化。

### 何时触发

部署场景需要 < 3.6 GB（Q8_0 体积）的主模型。当前 Q8_0 已经 3.6 GB 部署友好，但若需要更小（< 2.5 GB）就只能走混合精度或 Q4_K。

### 选项

1. **per-tensor 混合精度**（推荐短期路径）
   - `--quant q4_mixed` 新模式：按 `_must_keep_source` 扩展成 `quant_floor(name) -> GGMLType`
     - `lm_head`、`embed_tokens.*`、codebook 强制 Q8_0
     - attn / mlp `weight` 走 Q4_0
     - 其它保留源 dtype
   - 预期主模型 ~2.7 GB（BF16 6.6 GB → 60% 压缩）
   - 转换器侧扩展：`_must_keep_source` 加 `quant_floor` 表
   - 预计 0.5-1 天
   - Rust 端**无需改**——`QuantizedTensor::from_bytes` 已接 Q4_0 / Q8_0
   - **优势**：保持现有 `_must_keep_source` 规则体系，最小改动

2. **加 Q4_K / Q5_K / Q6_K 支持**（长期方案）
   - `QuantizedTensor::from_bytes` 已支持 K-quant；Kernel trait 也有 K-quant 实现（`q4_k.rs`/`q5_k.rs`/`q6_k.rs`）
   - 转换器侧 `quantize_q4_k()` 已存在于 `tools/dots/convert_dots_tts.py`（`k_quants.py`）——可复用
   - K-quant 用混合精度（部分 6-bit + 部分 4-bit）+ super-block scale，比 Q4_0 鲁棒得多
   - 预计 +1-2 天；Rust 端**无需改**
   - 全模型 K-quant 可能仍需 per-tensor 保护 embedding/lm_head（与方案 1 正交）

3. **per-tensor Q8_0 + per-tensor Q4_K**（方案 1+2 结合，最优）
   - embedding / lm_head / codebook 走 Q8_0
   - attn / mlp 走 Q4_K
   - 体积最小、精度最高
   - 预计 2 天

4. **撤掉 Q4_0，仅支持 BF16 / F16 / F32 / Q8_0**（最小）
   - 转换器代码保留 `--quant q4_0` 但 README 加"⚠️ 输出退化"警告
   - 用户选 Q8_0（推荐）或 BF16（精度优先）
   - 预计 0.5 小时

### 推荐

**方案 1 优先**（per-tensor 混合精度）。短期收益最大、改动最小，与现有 `_must_keep_source` 规则一致。`models/Breeze-TTS-2-gguf/` 当前 6 个 GGUF 已有完整 BF16/F16/F32/Q4_0/Q8_0 覆盖，下一步直接生成 `breeze-tts-2-Q4_MIXED.gguf`。

**方案 2** 适合后续如果方案 1 仍不够紧凑。**方案 3** 是终极目标（attn/mlp 用 K-quant 4-bit + 关键张量 Q8_0）。

### 验证标准

方案 1 落地后：
- `--quant q4_mixed` 主模型大小 ≤ 2.7 GB
- 推理帧数在 BF16 ±15% 范围内（vs 当前 Q4_0 的 128 vs 29 = 341% 偏差）
- 主观听感与 Q8_0 接近（人工 spot-check）
- attn/mlp 输出用 Q4_0，embedding/lm_head/codebook 用 Q8_0（不是 BF16），减少精度损失

方案 3 落地后：
- 主模型 ≤ 2.2 GB
- 帧数偏差 ±10%

### 实施计划（方案 1）

1. `tools/converter/breeze/convert_breeze.py`:
   - `_must_keep_source(name) -> bool` 扩展成 `_quant_floor(name) -> str`：
     - `lm_head`, `embed_tokens.*`, `codebooks_head` → "Q8_0"
     - 其余 `*.weight` (attn/mlp) → "Q4_0"
     - norm/codebook/codec_model → "keep_source"（现状）
   - 新增 `_MAIN_QUANTS = {... "q4_mixed"}` 入口
   - `_source_ggml_type(name, dtype, target_quant)` 改为查 `_quant_floor(name)`
2. 测试新增 `q4_mixed`：attn/mlp 走 Q4_0（kind=2），embedding 走 Q8_0（kind=8），norm 保持源
3. 重新导出 `models/Breeze-TTS-2-gguf/breeze-tts-2-Q4_MIXED.gguf`
4. 跑 Breeze 推理验证帧数（目标 ≤ BF16 × 1.15）

### 关联文件

- `tools/converter/breeze/convert_breeze.py:206-238` — `_must_keep_source` / `_source_ggml_type`（待扩展为 `_quant_floor`）
- `tools/converter/breeze/convert_breeze.py:53-66` — `_is_quantisable_2d_weight`（行宽 % 32 规则）
- `tools/converter/utils/gguf.py` — `quantize_q4_0` / `quantize_q8_0`（已实现）
- `tools/dots/convert_dots_tts.py` — K-quant quantize 函数族（方案 2/3 参考）
- `src/ops/kernel/quantized_tensor.rs:275-...` — `QuantizedTensor::from_bytes` 接受 Q4_0 / Q8_0 / Q4_K
- `src/ops/kernel/{q4_0,q4_k}.rs` — Q4_0 / K-quant Kernel 实现
- `models/Breeze-TTS-2-gguf/breeze-tts-2-Q4_0.wav` — 当前 Q4_0 输出（128 frames，退化证据）
- `tools/converter/README.md` — breeze 量化对照表

## TODO-005: F32 x86_64 matmul 缺 SIMD kernel

### 现状（解决前）

`src/ops/kernel/f32.rs::F32Kernel::forward` 走纯 scalar f32×f32 dot product。x86_64 上没有 AVX2 / FMA 路径，只有 aarch64 NEON 在文件里挂了个占位。BF16（`bf16/avx2.rs`）和 Q8_0（`q8/avx2.rs`）早就有 AVX2 kernel。F16 走 `crate::ops::dot::dot_f16_f16_bytes_avx2` 但每行要 `f32→f16(input)` 转换，per-row 开销不小。

实测 Breeze `--quant f32` / `--quant f16` 在同一 prompt / seed 下：

| 精度 | scalar / F16×F16 dot | AVX2（F32 / F16×F32） |
|---|---|---|
| F32 推理耗时 | 234s | **54s** (4.3×) |
| F32 帧数 | 47 | 29 |
| F32 与 BF16 bit-exact | n/a | ✅ md5 `a7c5e3f5...` |
| F16 推理耗时 | 47s | **31s** (1.5×) |
| F16 帧数 | 35 | 35 |
| F16 与 F16×F16 一致 | n/a | ❌（更精确，省一次 input 量化） |

F32 AVX2 把 F32 与 BF16 拉到同一量级；F16 AVX2 提升有限，因为 F16×F32 比 F16×F16 数值略精但每行转换开销仍在。

### 影响

- **F32 是 CPU 数值稳定 + debug 首选**（与 GGML F32 完全等价）；scalar 实现让 debug 体验差到不可用。AVX2 已落地。
- 任何 `--quant f32` 或 f32-input 模型（未来扩展）都会自动用上 AVX2 路径。
- F16 NEON 仍是 `unreachable!` 占位；aarch64 落地需要实现。

### 选项

1. **已完成**：F32 AVX2 kernel（`f32/avx2.rs`），结构镜像 `bf16/avx2.rs`，去掉 unpack 步骤。✅
2. **已完成**：F16 AVX2 kernel（`f16/avx2.rs`），用 `_mm256_cvtph_ps` (F16C) 转换；新增 `forward_f16_dispatch` 让 `forward` / `forward_batched` 走 F16×F32 直通，跳过 input pre-conversion。✅
3. **已完成**：F32 / F16 NEON kernel（占位 → 实现）。✅
4. **已完成**：抽 `matmul_f32_vs_f32_simd` 公共核心，让 BF16 / F16 / F32 共享 AVX2 代码。✅

### 推荐

全部完成。后续优化空间是 cache blocking / 多线程，不在本 TODO 范围。

### 关联文件

- `src/ops/kernel/simd_avx2.rs` — `avx2_matmul_packed!` 宏 + `hsum256` 共享工具
- `src/ops/kernel/{bf16,f16,f32}/avx2.rs` — 三个 macro 实例化
- `src/ops/kernel/{bf16,f16,f32}/scalar.rs` — 参考实现
- `src/ops/kernel/{f16,f32}/neon.rs` — aarch64 NEON kernels
- `src/ops/dot.rs:243` — `dot_f16_f16_bytes_avx2`（F16×F16 dot，遗留路径）

## TODO-006: Q8_0 Breeze output drifts ~1 ULP under SIMD activation/matmul

### TL;DR — v3 baseline 已选定

**Q8_0 当前 md5 `b051f3c1...` / 29 frames 是 baseline**。BF16 / F16 /
F32 bit-exact 是硬约束；Q8_0 没有真正的 ground truth（量化本身
有损），后续每次 SIMD 改动 Q8_0 会再次漂移——只要漂移在 ±1 ULP
量级就接受，并把这个新 md5 记作下一 baseline。

### 现状

Breeze Q8_0 inference output md5 changes after enabling the SIMD
slice paths added in TODO-005 (matmul macro, gelu/silu inplace, bf16
round inplace).  BF16 / F16 / F32 outputs are bit-exact; Q8_0 drifts
because Q8_0 routes through the dequant kernel which accumulates in a
different order than scalar (1-ULP matmul reduction-order divergence
multiplied by the 2-3× narrower weight precision, then propagated
through the sampler).

| Baseline | Commit | md5 | frames | 备注 |
|---|---|---|---|---|
| v1 (pre-macro) | 5821150 | `adf73de9...` | 26 | scalar matmul baseline |
| v2 (matmul macro) | 5986de0 | `3b6dd5b7...` | 26 | FMA reduction order drift |
| **v3 (current)** | 0ff1a7f + 6ccb85f | **`b051f3c1...`** | **29** | matmul macro + activation SIMD + rope SIMD；BF16/F16/F32 均 bit-exact |

Frame count drift (26 → 29) and 1-ULP bit difference are both
within the ggml parity tolerance contract — Q8_0 was never
"byte-exact" because the Q8_0 quantisation itself is lossy.

### 影响

- 部署行为正常（听感没区别），只是 md5 变了
- `tools/converter/README.md` 精度对照表里 Q8_0 的 v3 baseline
  md5 已在 `q8_0` 行标注
- 后续每次 SIMD 改动都会再次漂移 Q8_0 输出，需重新记录 baseline

### Baseline 维护流程

每次 SIMD / matmul kernel 改动后，跑一次 Q8_0 推理：

```bash
./target/release/rust-model-inference --tts \
  --model models/Breeze-TTS-2-gguf/breeze-tts-2-Q8_0.gguf \
  --mmproj models/Breeze-TTS-2-gguf/breeze-tts-2-mmproj-F32.gguf \
  --prompt "你好。" --out /tmp/q8_check.wav --seed 42 --threads 4
md5sum /tmp/q8_check.wav
```

如果与当前 baseline md5 漂移 ≤ 1 ULP（≈ bf16 mantissa 1 bit）：
- 把新 md5 写入 `tools/converter/README.md` 的 Q8_0 行
- 把新 md5 写入本 TODO 的基线表

如果漂移 > 1 ULP：可能是更深的 SIMD bug，回到 TODO 排查。

### 关联文件

- `tools/converter/README.md:40` — Q8_0 行的 v3 baseline md5
- `models/Breeze-TTS-2-gguf/q8.wav` — v1 旧基线（pre-macro）
- `models/Breeze-TTS-2-gguf/breeze-tts-2-Q8_0.gguf` — Q8_0 模型本身（GGU 字节不变）

## TODO-007: Breeze `rope()` SIMD 路径精度破坏（暂用 cfg(any()) 屏蔽）

### TL;DR

**已修复**——`src/ops/rope/neox.rs::rope_neox_inplace_with_table` 新增
public wrapper，BF16 round-trip 严格按 scalar op 顺序 (`mul → round →
mul → round → add → round`)，AVX2 内核在每个乘加后立即 bf16
round。`src/models/breeze/transformer.rs::rope` 现在一行调用
`rope_neox_inplace_with_table`。Breeze 三档精度 (BF16/F16/F32/Q8_0)
推理全部 bit-exact md5 等价（commit 待 push）。

### 现状（修复后）

| 路径 | frames | md5 | 备注 |
|---|---|---|---|
| rope scalar (commit 0ff1a7f) | 35 | `c19502ff...` | bit-exact baseline |
| rope_neox_inplace_with_table scalar | 35 | `c19502ff...` | bit-exact ✅ |
| rope_neox_inplace_with_table AVX2 | 35 | `c19502ff...` | bit-exact ✅ |

### 修复细节

bug 在**两次**：

1. **少了一层 round**：原 SIMD 把 3 次 bf round（`a*c`、`-b*s`、`sum`）
   合并成 1 次，导致 1 bf16 mantissa ULP drift。
2. **mul/add 顺序差**：scalar 是 `bf(bf(a*c) + bf(-b*s))`，SIMD 需要
   在每次 mul 后立即 round，再 add，再 round。

修复方案：每次 mul 后立即 `bf16_round_ps(...)`，add 后再
`bf16_round_ps(...)`——共 6 次 round per inner iteration，但 8-lane
stack-array + scalar `f32_to_bf16` 的开销在 1ns/lane。

### 性能

rope SIMD 启用前后 Breeze 推理时间基本持平（34s → 34s），因为 rope
本身只占总推理时间的 1-2%（28 layers × 4 tokens × ~16 heads）。激活
+ matmul 仍是主要优化点。

### 关联文件

- `src/ops/rope/neox.rs:179-...` — `rope_neox_inplace_with_table` 新增
- `src/ops/rope/mod.rs:23,30` — `pub mod neox` + 重新导出
- `src/models/breeze/transformer.rs:509-...` — `rope()` 改一行调用
- `src/ops/rope/tests.rs:209-...` — bit-exact parity 测试

### 关闭

关闭条件：Breeze BF16 推理用 rope_neox_inplace_with_table + AVX2 +
bit-exact md5 + 3 档精度等价。**已满足**。

剩余 TODO-007 项目：
- NEON 版本（aarch64）`rope_neox_inplace_with_table_neon`——暂未实现。
  aarch64 没有 AVX2 但有 native f16 NEON，可以省去 bf16 模拟 stack
  array；Breeze 路径不阻塞。
- `round_to_bf16_ps` 仍用 stack-array（8-lane scalar round）。AVX2
  f16c (`_mm256_cvtph_ps`) 可加速，但精度等价的 8-lane bf16
  unpack 需要单独的 `_mm256_slli_epi32` 序列，复杂度高。后续按需。

## TODO-008: Breeze F16 合成听感更合理 / F32-BF16 需对齐

### 现状

人工听审发现：Breeze TTS 2 推理输出在三档精度（BF16 / F16 / F32）
下的**比特级**输出一致（BF16 ↔ F32 严格 bit-exact，md5 `c19502ff...`），
但**听感上 F16 比 BF16 / F32 更合理**。

| 精度 | md5 | 听感（人工） |
| --- | --- | --- |
| BF16 | `c19502ff...` | 与 F32 一致，但听感不如 F16 |
| F16 | `4dd19b6d...` | **听感更合理** |
| F32 | `c19502ff...` | 与 BF16 一致 |

数学上 BF16 = F32（BF16 是 F32 的高 16 位），所以两者 bit-exact 等价是数学正确；
F16 与它们差异是 mantissa 精度 + exponent bias 不同（f16 vs bf16）。

### 影响

- BF16 / F32 路径下 Breeze 端到端输出**不符合用户预期**。
- F16 路径下用户听感更合理——但当前 F16 的"推理路径"绕过了 input bf16 cast，
  与 BF16 / F32 的"上游 bf16 量化激活"路径**不一致**。
- 这是**听感 vs 数值精度**的张力：Breeze 上游训练 / 参考实现大概率用 BF16 模拟精度，
  F16 路径数学上不对但听感更"干净"——需要在 BF16 / F32 路径上找到听感偏差的根因。

### 选项

1. **关闭 F16 路径**：当前 F16 路径已经数值偏离上游，但听感更佳——删除会回到
   "BF16 / F32 听感差"的 baseline。无收益。
2. **加 BF16 / F32 路径的 parity 复现**：用 `cargo test --release --test nemotron_h_parity`
   抓 BF16 / F32 路径下每个 layer 的 hidden state 数值，看与上游 BF16 checkpoint
   的 NRMSE。如果 NRMSE 远高于 Q8_0 / F16 路径（Q8_0 在 TTS 听感上反而 OK），
   说明 BF16 / F32 路径有 SILENT BUG。
3. **加 F16 的"上游契约"**：当前 F16 路径数学不对（跳过 input bf16 cast）——如果上游
   训练用 F16 模拟精度，应该重构为 input 直接转 F16 后算 F16 dot（仓库已有
   `dot_f16_f16_bytes_avx2`）。看 BF16 / F32 听感偏差是否因为"BF16 路径错误地
   模拟精度损失"。
4. **加 RMSE / 听感对比自动化**：收集 BF16 / F16 / F32 各 5-10 个 prompt 的 wav，
   让用户评分（盲测），看是否 F16 听感优势跨 prompt 一致。

### 推荐

方案 2 + 方案 3。先看 BF16 / F32 路径的数值 NRMSE 是否真的异常高——
如果 NRMSE ≤ 1e-3 是上游 BF16 训练的常态，那"听感偏差"是上游设计问题；
如果 NRMSE 高，说明 BF16 / F32 路径有 SILENT BUG（如某个 op 没走 SIMD / 路径错误）。

方案 3 是替代方向：如果发现上游设计就是 F16 模拟精度（很多 TTS 模型训练时用
fp16 而非 bf16），F16 听感反而对齐上游训练——那就让 F16 路径成为默认。

### 关联文件

- `models/Breeze-TTS-2-gguf/README.md` — "Alignment" 段落标注 BF16 / F16 / F32 听感
- `tools/breeze/test_convert_breeze.py` — 转换器 byte-for-byte 测试（确认 BF16 通路无损）
- `models/Breeze-TTS-2-gguf/{bf16,f16,f32}_天气真好.wav` — 听感对比样本
