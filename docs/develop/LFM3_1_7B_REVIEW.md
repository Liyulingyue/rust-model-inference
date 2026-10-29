# LFM3-1.7B 分支审阅报告

> §1–9 保留原始 `0d6cc2a0` 审阅结论。后续修复及真实权重复验见 §10；其中的
> “未做 HTTP/JEV/自动分发”和旧误差数值不再代表当前实现。

审阅对象:`gouzil/codex/lfm3-1-7b`
分支 tip:`0d6cc2a0`("Support LFM3-1.7B GGUF with CPU alignment")
基线:`origin/main` = `92d2eca0`(`[codex] Support Sage-A1.5 Q8_0 with aligned Vulkan MoE decoding (#187)`)
本地评审分支:`codex/lfm3-1-7b-review`
- `0d6cc2a0` — 上游原始提交
- `25d644f6` — 本次审阅附带的 RoPE 改进(见 §5)

审阅日期:2026-10-29
审阅性质:代码 + 文档审阅;**未在真实权重上复现数值**(见 §9)

---

## 1. 结论摘要

| 维度 | 结论 | 说明 |
| --- | --- | --- |
| 分支可合并性 | ✅ 可合并 | 已基于最新 main,无需合并,无冲突 |
| 架构定位 | ✅ 独立新架构 | `src/models/lfm3/`,24 层 hybrid(shortconv + gated full/SWA GQA) |
| 改动规模 | 小而集中 | 22 文件 +1969/−7,**唯一 commit** |
| 多 weight 分发 | ❌ **未做** | `src/models/lfm3/` 内 `LinearExecutor`/`run_group`/`execute_with_linear` 零命中 |
| GPU/CPU 自动分发 | ❌ **未做** | `ComputePolicy` 零命中;`run_inference` 无 policy 参数;`--gpu` 是静默 no-op |
| JEV | ❌ **未做** | `src/app/jev/` 内 `lfm3` 零命中 |
| HTTP | ❌ **未做** | `build_text_runtime` 无 `lfm3` 分支 → server 启动报错 |
| CLI | ✅ **可用** | `src/app/text/generation.rs` 分派 + `src/core/loader.rs` 归档识别 |
| SIMD 加速适配 | ⚠️ **复用既有,无新增** | RoPE / 量化投影 / dot / softmax 走既有 SIMD;shortconv 与 value gather 是标量热点 |
| 数值对齐 | ⚠️ **未达位级** | max abs `1.115660` / RMSE `0.171233`,门槛 `1.25 / 0.20`(余量 89%) |
| 误差归因 | ❌ **未做** | 见 §6,已排除 2 个嫌疑,剩 5 个候选,均可用现成开关逐个测 |

**总体判断:可以合并,定位应保持"CPU-only 的实验性支持"。** 五问里只有 CLI 可用;其余四项(多 weight 分发、GPU/CPU 自动分发、JEV、HTTP)均未实现 —— 而从分支标题("with CPU alignment")与 `SUPPORTED_MODELS.md` 的措辞("仅 CPU CLI/model API、F32 KV")看,这是**有意的范围划定**,不是疏漏。

**真正需要作者补的只有一件事:归因那 1.1157 的 logits 误差**(§6)。其余能力缺口是产品排期问题,不阻塞合并。

---

## 2. 分支与模型

### 2.1 提交与基线

| Commit | 日期 | 标题 |
| --- | --- | --- |
| `0d6cc2a0` | 2026-10-29 | Support LFM3-1.7B GGUF with CPU alignment |

merge-base 与 `origin/main` 相同(`92d2eca0`)→ **分支已同步最新 main,合并零冲突**。

### 2.2 文件分布

| 区域 | 文件 | 说明 |
| --- | --- | --- |
| 新架构 | `src/models/lfm3/trunk/{config,forward,mod,session,trace,weights}.rs` | +1049,独立 trunk,不复用 qwen3/llama |
| 模型注册 | `src/models/lfm3/mod.rs`、`src/models/mod.rs` | +5 |
| 入口 | `src/app/text/generation.rs` | +14,`arch == "lfm3"` 分派 |
| 归档识别 | `src/core/loader.rs` | +15,arch 前缀列表 + per-layer `head_count_kv` 解析 |
| **全局算子** | `src/ops/rope/{mod,neox}.rs`、`src/ops/rope/tests.rs` | +46,**唯一触及全局模块的地方** |
| 测试 | `tests/lfm3_reference.rs`(+302)、`tests/lfm3_synthetic.rs`(+232) | 合成测试 3 个 + 真模型测试 3 个 |
| Oracle | `tools/oracle/lfm3/` | README、oracle.cpp、compare_checkpoints.py、checkpoints.patch |

### 2.3 模型契约

| 项 | 值 |
| --- | --- |
| 权重 | `atomic.chat/LFM3-1.7B-GGUF` 的 `LFM3-1.7B-Q8_0.gguf` |
| 大小 / SHA-256 | 1,846,238,336 B / `4dc04caea646076de9502b41b6c6db52203945374246d928d91e70ef5ec8fdcb` |
| 架构 | `general.architecture=lfm3` |
| 结构 | 24 层 / 2048 hidden / 16 query head / 128 head_dim / 7680 FFN;vocab 由 `lfm3.vocab_size` 元数据给出,缺失时取 tokenizer 长度 |
| KV heads | `[0,0,4,0,4,0,4,…]` —— 0 表示 shortconv(kernel 7 / 历史 6);注意力层用 sigmoid head gate |
| SWA | 0-based 层 `[4,8,12,16,20]`,窗口 256,RoPE base `10000` |
| 全局层 | RoPE base `1000000` + `rope_freqs.weight` per-pair 因子(前 32 = 1,后 32 ≈ `1e30`) |
| 其他 | `post_attention_norm`/`post_ffw_norm` 成对出现;缺 `output.weight`(输出头与 embedding 绑定);GPT2 BPE、`tokenizer.ggml.pre=lfm2`;BOS `124894` / EOS `124900`,均不自动添加 |

Oracle 固定为 llama.cpp `19cce6919992fdffd92e0ac152caaf1e2b8f254e`(`src/models/lfm3.cpp` / `src/models/lfm-base.cpp`),作者只加了一层 Q/K RoPE 检查点补丁(`tools/oracle/lfm3/checkpoints.patch`),不改计算。

---

## 3. 五个专项问题

### 3.1 多 weight 分发 —— ❌ 未做

`src/models/lfm3/` 内 `LinearExecutor` / `run_group` / `execute_with_linear` **零命中**。

投影走的是既有量化 matmul 入口(`src/models/lfm3/trunk/forward.rs:47`):

```rust
let mut project = |w: &Weight<'_>, input: &[f32], output: &mut [f32]| {
    w.quantize_and_matmul_with_scratch(input, q8k_buf, q8_buf, scale_buf, output, pool);
};
```

即**每个投影单独一次激活量化 + 一次 matmul**,经 `ComputePool` 的 CPU 路径执行。没有多投影分组(`LinearExecutor::run_group` 的 QKV/gate-up 合并),也没有 GPU 侧的权重常驻与复用。

对比:仓库里已有两套可复用的机制,本分支都没接:

| 机制 | 位置 | 能力 |
| --- | --- | --- |
| `LinearExecutor::run_group` | `src/compute/linear.rs:186` | CPU+Vulkan 双端共享的多投影执行器(PR #166 起) |
| `DenseCpu::execute_with_linear` | `src/compute/dense.rs:167` | 把 `DenseOp::Linear{projections}` 组路由到 `run_group` |

> **建议**:若后续要给 LFM3 上 GPU,`DenseCpu::execute_with_linear` 那条路是现成的 —— 但 LFM3 的投影是 `Weight` 上的方法而不是 `DenseOp`,需要先确认两者的权重表示能否对接。

### 3.2 GPU/CPU 自动分发 —— ❌ 未做

`src/models/lfm3/` 内 `ComputePolicy` **零命中**;`run_inference(...)` 的签名里没有 policy 参数;`Lfm3Session::from_source` 也不感知设备策略。整条链是纯 CPU。

两个具体后果:

**① `--gpu` 是静默 no-op。** CLI 把 `--gpu` 映射到 `ComputePolicy::Auto`,而 LFM3 完全不查这个 policy,于是 `--gpu` 只是照常跑 CPU —— 用户不会收到任何提示。建议要么显式报错,要么至少打印一条 "lfm3 has no device path; running on CPU"。

**② `--compute vulkan` 会干净地失败,这是对的。** `ComputePolicy::validate_text_arch`(`src/compute/mod.rs`)的白名单是 `llama | qwen3 | qwen35 | qwen35moe | dory`,不含 `lfm3`,所以会返回 "full Vulkan text decoding is not supported for lfm3"。与 holo4/turbo 分支给 `qwen35moe`/`dory` 放开白名单的做法不同 —— LFM3 没有放开,这是合理的(没有 Vulkan 实现)。

对比 holo4 的做法:`src/models/qwen35/trunk/session.rs:268-298` 有一条完整的降级链(Auto 试 Vulkan → 失败打印 "Falling back to CPU" → 降级;Vulkan 强制 → 报错)。LFM3 缺的就是这一层。

### 3.3 JEV —— ❌ 未做

`src/app/jev/` 内 `lfm3` **零命中**;`/v1/jev/score`、`/v1/jev/grouped`、`/v1/jev/image` 均无 `lfm3` 分支。

LFM3 是**语言类模型**(文本生成),按常规口径 JEV 是适用的,所以这是一个**能力缺口**。不过它是自洽的:作者在 `docs/develop/SUPPORTED_MODELS.md` 里把 LFM3 的状态写为"**仅 CPU CLI/model API、F32 KV**",明确划出了入口边界。

若要补 JEV,现成的接入点是 `src/app/jev/single/qwen35.rs` 那套 —— 但 LFM3 的 forward 不暴露 JEV 需要的 logits-only 前向接口(`run_forward_logits_lfm3_with_batch` 存在,需要确认其签名能否直接给 JEV 用)。

### 3.4 HTTP / CLI

**CLI ✅ 可用**

- `src/app/text/generation.rs:135` 新增分派:
  ```rust
  } else if arch == "lfm3" {
      crate::models::lfm3::run_inference(
          source.as_ref(), prompt, max_tokens, temperature, n_threads_arg,
          profile, kv_format, max_context, repetition_penalty, thinking, jinja,
      )
  ```
- `src/core/loader.rs`:`"lfm3"` 加入 arch 前缀列表(拿到专用 `ModelConfig` 而非通用 config),并为它单独解析 `lfm3.attention.head_count_kv`(per-layer 数组,取首个非零值给通用 config;完整数组留给 LFM3 trunk 自己分派 recurrent/attention)。
- 支持 `--thinking`(LFM3 的 thinking 模板)、`--jinja`、F32 KV、session 续写与 reset。

**HTTP ❌ 不可用**

`src/app/text/runtime.rs::build_text_runtime` 没有 `lfm3` 分支 → server 启动即返回 `No TextRuntime adapter for architecture lfm3`。`/v1/chat/completions`、`/v1/jev/*` 全部不可用。

这是 server 侧**统一适配器机制**的固有约束:每个 arch 都要实现 `TextRuntime`。LFM3 若要上 HTTP,需要写一个 `Lfm3TextRuntime`(`src/app/text/runtime.rs` 里已有 `LlamaTextRuntime` / `Qwen3TextRuntime` / `HybridTextRuntime` / `Lfm2MoeTextRuntime` 等同形实现可参考)。

### 3.5 SIMD 加速适配 —— ⚠️ 复用既有,无新增

分支**没有新增任何 SIMD kernel**;它对 SIMD 层的唯一改动是 RoPE 的全局改动(§5)。各计算环节:

| 环节 | SIMD | 备注 |
| --- | --- | --- |
| RoPE(含新的 `rope_neox_inplace_with_freqs`) | ✅ AVX2 / NEON | 走 `src/ops/rope/neox.rs` 的 `rope_neox_inplace_avx2` / `_neon` |
| 量化投影 | ✅ | `quantize_and_matmul_with_scratch` → Q8_0 / Q8_K 既有 kernel |
| `dot_f32`(QK 与 attn_out) | ✅ AVX2 / NEON | `src/ops/dot.rs:894` |
| `softmax_inplace`(`src/ops/softmax.rs:4`) | ⚠️ **无 SIMD 但是精确的** | 标量 `f32::exp` + **F64 累加** + `(1.0/sum) as f32` |
| `sigmoid_inplace`(head gate) | ✅ AVX2 / NEON | 但内部是**精确 libm exp**,只是 SIMD 载/存 + 标量 expf 取 ILP |
| `vec_scale_f32`(gate 缩放) | ✅ AVX2 / NEON | |
| **shortconv(taps = 7)** | ❌ **手写标量** | `forward.rs:196`,用 `mul_add` 显式 FMA 对齐 ggml |
| **attention 的 value gather** | ❌ **内层标量** | `forward.rs:127-135`,每个 (d, i) 一次 `match kv { F32/F16 }` 分支 + `f16_to_f32` 转换 |

两处标量热点都是 LFM3 特有结构带来的:

- **shortconv**:kernel 7、历史 6、宽度 2048,每层每 token 一次 2048×7 的乘加。`mul_add` 是为了对齐 ggml 的 FMA 累加顺序,这个选择本身是对的,但没有 SIMD 版本。
- **value gather**:F16 KV 时,内层循环对每个元素做一次分支判断 + `f16_to_f32` 转换。复杂度 O(dim × seq) per head,24 层 × 16 head 全注意力层都要跑。F32 KV 时只是拷贝,但仍是非连续访问。

> 这两处是**性能优化点**,不是正确性问题。文档里的计时(见 §7)没有把 shortconv / gather 单独拆出来,所以无法判断它们占比多少。

---

## 4. 与 holo4 / turbo 分支的横向对比

| | LFM3 | holo4-35B-A3B | Qwen-Image-2.1-Turbo |
| --- | --- | --- | --- |
| 多 weight 分发 | ❌ | ⚠️ Vulkan 侧分组绑定 | ✅ `execute_with_linear` → `run_group` |
| GPU/CPU 自动分发 | ❌ | ✅ base 既有降级链 | ✅ `context_with` Auto→CPU |
| JEV | ❌ | ✅(语言类,base 既有) | N/A(图像类) |
| HTTP | ❌ | ✅(语言类) | ❌(图像类) |
| CLI | ✅ | ✅ | ✅ |
| SIMD 新增 | ❌ | ❌(GPU shader 为主) | ✅ 修了 3 个全局 SIMD bug |
| 数值证据 | ⚠️ 经验界(1.25/0.20) | ⚠️ 整模未对齐(0.02 门槛) | ✅ 标量路径位级精确 |

**结论:LFM3 是三条分支里能力面最窄的一条**,但它的"窄"是显式声明过的,而 turbo 分支的默认 SIMD 路径反而是数值上最弱的(默认路径未过 1% 门槛)。若要排优先级,给 LFM3 补**误差归因**(§6)比补 GPU/JEV 更划算。

---

## 5. 附带的 RoPE 改进(`25d644f6`)

审阅中发现 `src/ops/rope/neox.rs` 的分支改动有几处可改进,已在本地评审分支上实现(+141/−6,4 文件)。

### 5.1 改了什么

| 项 | 改动 |
| --- | --- |
| ① 前置条件 | `assert_eq!(freqs.len(), head_dim/2)` → `debug_assert_eq!`,与同文件兄弟 API `rope_neox_inplace_with_table`(`neox.rs:264`)一致;同时把循环里的 `freqs[i]` 改成 `freqs.get(i).copied().unwrap_or(1.0)`,release 下短切片退化为"未缩放"而非在循环里 panic。参数名 `freqs` → `divisors`,与 GGUF 语义对齐 |
| ② 缺 precise 变体 | 新增 `rope_neox_inplace_with_freqs_precise`,用 `sin_cos_precise`(分离的 libm sin/cos)。默认路径仍是融合的 `sin_cos()`(更便宜)。**未把 LFM3 切到 precise** —— 那会改变数值,需要真实权重 + 自建 oracle 才能验证 |
| ③ 文档 | 补齐语义:公式 `angle_i = pos·freq_base^(-2i/d)/divisor[i]`、ggml 的乘数约定(llama.cpp 存除数、建图时转 `1.0f/freq` 乘数)、**除数 1.0 = 普通 RoPE**、**极大除数(1e30) = 冻结该对**、长度契约 |
| ④ 与 gemma4 的重复 | **没有合并**,只在两侧加交叉注释。原因:gemma4 的 `apply_rope_ggml` 旋转用 `mul_add`(显式 FMA),而共享路径的 scalar/AVX2/NEON **刻意不用 FMA**(注释明说要匹配 pinned ggml reference)→ 换成共享 API 会改掉 gemma4 每个旋转通道的最后一位,而 gemma4 是 `Verified` + raw-bit parity。**这必须单独立项并重跑 gemma4 的证据** |

### 5.2 为什么这个公式是对的

仓库内有**独立佐证**:`src/models/gemma4/trunk/forward.rs::apply_rope_ggml` 早就用同一公式(`angle = theta / factors[i]`,且 `theta *= theta_scale` 是**乘法**推进)。乘法推进这点关键 —— 若写成加法就与 ggml 的 `p·theta_scale^i` 不符了。

与模型数据也自洽:LFM3 的 `rope_freqs.weight` 前 32 项 = 1.0、后 32 项 ≈ 1e30。只有"乘法"解释成立(`angle_i = p·base^(-2i/d)/freq_i`):前 32 对保持标准 RoPE,后 32 对角度≈0 → 冻结。若按"替换"解释,前 32 对会全部用同一角度 `p`,模型根本不成立。

### 5.3 测试(+3)

| 测试 | 覆盖 |
| --- | --- |
| `neox_frequency_divisors_scale_each_pair_independently` | per-pair 独立缩放:pos=4 + divisor 2 ≡ 普通 pos=2 |
| `neox_frequency_divisors_precise_agrees_with_the_fused_variant` | precise 与 fused 容差内一致 + 冻结行为 |
| `neox_frequency_divisors_without_a_per_pair_list_match_the_plain_path` | `None` 路径与普通 RoPE 逐位相同 |

写测试时发现两件值得记的事:

- **per-pair 语义**:divisor=2 时要用 **pos=4** 才等于普通路径的 pos=2(除数是作用在**已含 `theta_scale^i`** 的角度上)
- **冻结不是精确的 0**:除数 1e30 让角度 ≈ 1e-33 而非 0,`sin` 留下 ~1e-33 残差。原有的 `preserve_unscaled_and_frozen_pairs` 测试能断言精确相等,是因为它的输入恰好不含 0 元素;输入含 0 时残差就显现。新测试因此用容差断言

### 5.4 验证

| 项 | 结果 |
| --- | --- |
| `cargo check --features vulkan --all-targets` | 0 error |
| `cargo fmt --check` | ✓ |
| `cargo test --lib rope` | 38 passed / 2 failed —— 两个都是 **pre-existing**(在未改动的上游 tip 上同样失败:`vision_rope_matches_ggml_libc_trig_bits`、`neon_tests::rope_neox_inplace_matches_pinned_ggml_recurrence_and_fused_rotation`,都是 x86 上的 libc / NEON 位级 pin) |
| `cargo test --lib gemma4` | 64 passed / 4 failed —— 全是 BF16 投影 / softcap 相关,与 RoPE 无关;本次对 gemma4 的改动是 11 行**纯注释** |
| `cargo test --lib lfm3` | 3 passed |
| `cargo test --lib ops::` | 238 passed / 4 failed(同上那批 pre-existing) |

---

## 6. 误差归因:那 1.1157 的 logits 误差从哪来

### 6.1 现状

`tests/lfm3_reference.rs` 的判据是**经验回归界**,不是位级对齐:

```rust
// Empirical Q8_0/SIMD regression bounds for this pinned CPU Oracle.
assert!(error <= 1.25 && rmse <= 0.20, "{name}/{step}: max={error}, rmse={rmse}");
```

| 输入 | prompt token | greedy | 最大 logits abs error | 最大逐步 RMSE |
| --- | ---: | ---: | ---: | ---: |
| `Hello` | 1 | 8/8 | 0.243883 | 0.048465 |
| Jinja thinking chat | 17 | 8/8 | 1.115660 | 0.171233 |
| `你好,LFM3!\n\t café 🦀` | 13 | 8/8 | 0.533533 | 0.095196 |
| 混合英文重复文本(跨 SWA 窗口) | 265 | 8/8 | 0.452057 | 0.068492 |

prompt IDs 四组全部精确一致;5,176 个共同检查点的名称/shape/次数核对通过;24 次 embedding 逐位一致。

**但 1.115660 已占上限 1.25 的 89%**,余量很薄,且**误差来源完全没有被拆解** —— 文档只笼统归因为 "Q8_0/SIMD regression bounds"。这既让阈值失去意义(它是拟合出来的,不是推导出来的),也让"这个模型能不能用"无法回答。

### 6.2 已排除的嫌疑

审阅中逐个查过源码,以下两个**不是**误差来源:

| 嫌疑 | 结论 | 证据 |
| --- | --- | --- |
| softmax 用了近似实现 | ❌ 排除 | `softmax_inplace`(`src/ops/softmax.rs:4`)是**精确**实现:`f32::exp` + **F64 累加** + `(1.0/sum) as f32`,没有 SIMD 近似。近似版是另一个函数 `softmax_approx_inplace`,LFM3 没用它 |
| sigmoid head gate 用了近似 exp | ❌ 排除 | `sigmoid_inplace`(`src/ops/math/sigmoid.rs:8`)的 AVX2 / NEON 路径内部是**精确 libm exp**,只是 SIMD 载/存 + 标量 expf 取 ILP |

### 6.3 剩余候选与逐个实验

按"最可能 → 最不可能"排列。每个都给出**可立刻执行的实验**。

#### 候选 1:RoPE 的 fused `sin_cos` vs llama.cpp 的分离 `sinf`/`cosf` —— 最可能

`rope_neox_inplace_with_freqs` 走 `rope_sin_cos` → `mrope::sin_cos` → `value.sin_cos()`(f32 融合调用)。而仓库自己有一条注释说明这个差异真实存在:

> `// Separate libm calls: LLVM's sincos fusion changes scalar rounding on macOS.`
> —— `src/ops/rope/mrope.rs:37`

而 LFM3 的 oracle 正是在 **macOS ARM64** 上构建的(`README`:验证环境为 macOS ARM64 单线程)。ggml 的 rope kernel 用分离的 `sinf`/`cosf`。每次 rope 调用有 `head_dim/2` = 64 个角度,24 层 × 2(Q/K)× 每 token,1 ulp 的差异经过 24 层放大到 logits 上 0.1~1.0 是完全可能的。

> **实验**:把 `src/models/lfm3/trunk/forward.rs` 的 `rope_neox_inplace_with_freqs` 换成 `rope_neox_inplace_with_freqs_precise`(§5 已提供),重跑 `lfm3_oracle_alignment`。若误差显著下降,这一项即被证实。
> **前置**:本次改进只提供了 API,**没有切换 LFM3 的调用**,因为那需要真实权重 + 自建 oracle 才能验证。

#### 候选 2:attention 的 dot 归约顺序

Rust `dot_f32` 在 SIMD 下是 AVX2 / NEON 的多累加器 FMA;ggml 的 `ggml_vec_dot_f32` 在 ARM NEON 上是另一种分块顺序。两者数学等价、浮点不等价。

> **实验**:`RMI_SCALAR=1` 跑一次。scalar 路径走 `dot_f32_f64_acc`(F64 累加,顺序又不同)。若误差明显变化 → 归约顺序是主因之一;若几乎不变 → 排除。

#### 候选 3:softmax 的归一化顺序

Rust:`sum` 用 **F64** 累加,再 `(1.0/sum) as f32`;ggml 的 CPU softmax 用 **F32** 累加。这是一处**结构性差异**,不是位级差异。

> **实验**:写一个 F32 累加版的 softmax 临时替换,重跑对比。差异应当是系统性的(整体缩放偏差)而非随机。

#### 候选 4:激活量化的舍入规则

Rust 的 `quantize_row_q8_k_into` / `quantize_q8_0_into` 与 ggml 的量化舍入规则在**边界值**上可能不同 —— turbo 分支的作者就抓到过同类问题(NEON 与标量对 `0.5/1.5/2.5` 的 tie 处理不同,而 "pinned ggml ARM and generic kernels use these respective rules as well")。LFM3 的权重是 Q8_0,激活也被量化到 Q8_0/Q8_K。

> **实验**:用 turbo 分支已有的 `tools/oracle/qwen_image_2_1/` 比较器思路,写一个 Q8_0 量化探针,对比 Rust 与 ggml 在 tie 值上的舍入。

#### 候选 5:shortconv 的 FMA 累加顺序

`shortconv_step`(`forward.rs:196`)显式用 `mul_add` 对齐 ggml 的 FMA 累加:

```rust
for t in 0..taps - 1 {
    out[channel] = state[t * width + channel].mul_add(kernel[channel * taps + t], out[channel]);
}
out[channel] = bx[channel].mul_add(kernel[channel * taps + taps - 1], out[channel]);
```

但**累加顺序**是 t = 0..taps-2 然后最后一项。ggml 的 `ggml_conv_1d` 深度卷积顺序是否一致,需要核对。24 层 shortconv 各一次,每次 2048 通道。

> **实验**:对照 ggml 的 `ggml_conv_1d` / `build_shortconv` 实现核对顺序。

### 6.4 建议的归因流程

按"改动量 ÷ 信息量"排序,每步只动一个变量:

1. **先测候选 1**(只换 rope 的 precise 变体,一行改动)—— 最可能且代价最小
2. **再测候选 2**(`RMI_SCALAR=1`,零改动)
3. **再做候选 3 / 4 / 5**(需要写探针)
4. 每一步记录 `max_abs` / `rmse` 的变化,找到主导项后**再决定阈值**

找到主导项之前,`1.25 / 0.20` 这个阈值不应被当作"通过"的证明 —— 它是拟合出来的经验界,换一组权重或换一台机器就可能失效。**归因完成后,阈值应当重新推导(例如从各成分的误差上界合成),而不是继续沿用。**

---

## 7. 性能观察

分支文档只给了端到端计时,没有拆分,信息量有限:

- 4 组 prompt 的 greedy 对齐在**单线程** CPU 上完成(macOS ARM64)
- 256×256 / 512×512 之类的图像尺寸指标不适用(这是文本模型)
- 文档未给出 tokens/s,只有 wall-clock 与对齐耗时

已知的两处标量热点(§3.5 的 shortconv 与 value gather)在端到端计时里无法分离。若要给出可比较的性能结论,建议在 `tools/oracle/lfm3/benchmark.rs` 的形式下补一个分项计时。

---

## 8. 风险清单

### R1 ★★★ 误差未归因,阈值是拟合出来的

`1.25 / 0.20` 的经验界让 1.1157 勉强通过,但没有任何机制解释误差从哪来。换个权重 / 换个平台 / 换个编译器版本都可能越界。**§6 给了逐项实验方案。**

### R2 ★★★ `--gpu` 静默 no-op

用户传 `--gpu` 期望 GPU,实际静默跑 CPU,没有任何提示。建议显式报错或打印说明。

### R3 ★★ JEV 与 HTTP 缺口

LFM3 是语言类模型,JEV 天然适用,却未实现;HTTP server 也无适配器。这两项是能力缺口,不是缺陷,但文档应明确"当前只支持 CLI"。

### R4 ★★ 验证环境窄

全部对齐证据来自 **macOS ARM64 单线程 + 一个固定的 Q8_0 权重**。x86_64、Windows、不同量化形态(作者在 README 里明确说"其它 GGUF 量化不在验证范围内")都没有覆盖。CI 上只有 3 个合成测试,真模型对齐测试需要 `RMI_LFM3_ORACLE` 指向自建 llama.cpp,常规 CI 跑不到。

### R5 ★★ `Verified` 状态与证据强度不匹配

`SUPPORTED_MODELS.md` 里 LFM3 标 `Verified`,同行又写"有损路径按数值验收,实测最大 logits abs error 1.115660 / RMSE 0.171233,不声明完整浮点逐位一致"。对比 MiniCPM-V 4.7 35B-A3B 那行的写法("唯一残差来自 llama.cpp 的 F16 KV cache,本仓库保留 F32,是更精确的一方"),LFM3 的 1.1 绝对误差没有任何解释性说明。建议在归因完成前标 `Supported`,或补上误差构成说明。

### R6 ★ RoPE 全局改动的连带面

分支对 `src/ops/rope/neox.rs` 的改动触及所有模型的 RoPE。**所幸所有既有调用者都传 `None`,`map_or` 短路 → 行为逐位不变**,风险被隔离在新增 API 后面。但这是全局模块,值得在 review 时确认一遍既有调用点。

### R7 ★ shortconv / value gather 的标量热点

不影响正确性,但 LFM3 特有的结构带来两处 O(dim×seq) 的标量循环(§3.5)。

---

## 9. 审阅方法与边界

**已做**

- `git fetch origin main` / `git fetch gouzil codex/qwen-image-2-1-turbo` → `0d6cc2a0`;确认分支已基于最新 main(merge-base = `92d2eca0`),无冲突、无需合并
- 逐文件阅读 `src/models/lfm3/**`、`src/app/text/generation.rs`、`src/core/loader.rs`、`src/ops/rope/{mod,neox,tests}.rs`、`tests/lfm3_{reference,synthetic}.rs`、`tools/oracle/lfm3/**`
- 逐条核对作者文档(`tools/oracle/lfm3/README.md`、`docs/develop/SUPPORTED_MODELS.md`)与代码实现
- 追查 `LinearExecutor` / `run_group` / `ComputePolicy` 在仓库中的真实来源与可用性,给出 LFM3 未接入的具体影响
- 在 `src/ops/rope/neox.rs` 上实现 4 项改进并补 3 个测试(§5),`cargo check` / `cargo fmt` / 定向测试组复验
- 排查误差来源:读 `softmax.rs`、`math/sigmoid.rs`、`math/exp.rs`、`dot.rs`、`ops/rope/neox.rs` 的实现,排除 2 个嫌疑,列出 5 个候选与逐个实验方案(§6)

**未做**

- **未在真实权重上复现数值** —— 需要 `LFM3-1.7B-Q8_0.gguf`(1.85 GB)与自建 llama.cpp `19cce691` oracle,环境不具备
- **未复现 oracle 对齐测试**,因此 §6 的 5 个候选**未做实验验证**,只做了源码级排除与实验设计
- **未跑全库测试套件**,只跑了 rope / gemma4 / lfm3 / ops 四个定向组;§5.4 记录了每个组的失败项及其 pre-existing 判定依据
- **未测性能** —— §7 只引用作者的数据

**关键文件索引**

| 关注点 | 文件 |
| --- | --- |
| 模型 trunk | `src/models/lfm3/trunk/{config,forward,session,weights}.rs` |
| CLI 分派 | `src/app/text/generation.rs:135` |
| 归档识别 | `src/core/loader.rs`(`"lfm3"` arch 列表 + `lfm3.attention.head_count_kv`) |
| RoPE 改动 | `src/ops/rope/neox.rs` |
| 共享多投影执行器(未接入) | `src/compute/linear.rs::run_group`、`src/compute/dense.rs::execute_with_linear` |
| GPU/CPU 降级链(未接入) | `src/models/qwen35/trunk/session.rs:248-309` |
| server 适配器注册(未接入) | `src/app/text/runtime.rs::build_text_runtime` |
| JEV 分派(未接入) | `src/app/jev/single.rs`、`src/app/jev/grouped.rs` |
| 数值判据 | `tests/lfm3_reference.rs::lfm3_oracle_alignment` |
| Oracle | `tools/oracle/lfm3/{README.md,oracle.cpp,checkpoints.patch,compare_checkpoints.py}` |
| 声明状态 | `docs/develop/SUPPORTED_MODELS.md` |

## 10. 修复与复验（2026-10-29）

已拉取并合入上游 main `8d18565b`；保留 §5 的 RoPE 改进。

| 原审阅项 | 当前结果 |
| --- | --- |
| 多 weight 分发 | 所有投影复用 `LinearExecutor`；Q/K/V/head gate、FFN gate/up 使用 `run_group`，共享激活准备 |
| CPU 多精度 | F32/F16/BF16/Q8_0/Q4_0/Q4_1/Q4_K/Q5_K/Q6_K 及混合投影的合成会话测试通过，检查有限值、非零输出、分块/续写/reset 一致性；Q4_1 embedding 仍明确拒绝 |
| GPU/CPU 自动分发 | CLI、HTTP、JEV 共用 ComputePolicy；Auto 在 M3 Max 上实际执行 Vulkan 投影，缺驱动时显式回退 CPU；全图 Vulkan 明确拒绝 |
| HTTP | 新增 LFM3 TextRuntime、默认 GGUF Jinja 和输出解析器注册；真实 chat/thinking/SSE 请求通过，超上下文返回 400 |
| JEV | CLI 与 HTTP 单项/分组入口接通；真实权重下选出红色选项，CPU/Auto 概率一致 |
| 数值 | 独立 sin/cos 后聊天 max abs/RMSE 为 `1.007810/0.154821`；四组 CPU prompt 各 8 步 greedy 仍与固定 Oracle 一致；聊天 Auto 同样通过有损路径检查 |
| 误差归因 | 原 fused、precise、precise+scalar 的对照，以及同一 Oracle Q8 量化器处理双方 norm 输入的探针，见下方记录 |

真实聊天检查点仍为 5,176 个，24 次 embedding 原始位一致；当前首个位差在位置 1、
layer 2 的 Q RoPE（`4.7683716e-7`）。位置 4、layer 0 的短卷积还存在
`3.7252903e-9` 的累加差异。后续位置 4、layer 13 的 operator norm 输入差
`0.00017584115` 已让同一 Oracle Q8 量化器产生 10 个不同字节，证实了重新量化放大。
这是首个被探针捕获的 **norm 输入**量化差异，不是全图首次量化分叉。

§6 的 softmax 推测需修正：固定 Oracle 在 NEON 下使用向量 exp 近似和四路 F32
归约，再累加进 `ggml_float`；Rust 为精确 exp 与逐值 F64 累加。两侧原生 NEON
Q8_0 都采用 ties-to-even。scalar 控制会同时改变多个算子，不能作为 dot 单项归因。
RoPE/scalar 对照不是可相加的误差分解，也未得到通用全图误差上界。

因此保留 `1.25/0.20` 作为固定模型/样例的经验报警线，并将清单状态细分为 CPU/Q8_0
运行已验证、Vulkan 投影实验性支持；`Verified` 按清单定义表示真实运行，不代表浮点
逐位一致。没有为了对齐而替换共享算子或关闭默认 SIMD。

复现命令和机器可读证据见 [README](../../tools/oracle/lfm3/README.md) 与
[review-fixes.json](../../tools/oracle/lfm3/review-fixes.json)。未运行全仓测试，其他真实
权重量化档、其他硬件、全图 GPU、多模态和工具调用不在本次验证范围内。

---

## 11. 对齐原则与二轮审阅（2026-10-29）

### 11.1 仓库的对齐原则

审阅过程中确立了一条应当适用于**所有**模型适配的原则,先记在这里,因为它
改变了本报告 §6 与 §8 的建议优先级:

> **近似会带来精度损失,并不意味着 llama.cpp 的近似是最好的。我们永远对齐架构、
> 对齐计算逻辑,但没必要吹毛求疵追求完全比特对齐 —— 因为量化本身就是比特损失的事。**

这条原则的含义:

| 该做 | 不该做 |
| --- | --- |
| 对齐架构:层数、注意力形态、归一化位置、门控语义、RoPE 的角度公式 | 对齐近似的实现细节:ggml 的 NEON 向量 exp 近似、四路 F32 横向求和 |
| 对齐计算逻辑:算子的数学定义、累加顺序、量化发生在哪一步 | 为了位级一致而关闭默认 SIMD、替换共享算子或加特例分支 |
| 记录残差落在哪个量级、是否可接受 | 把残差拆成逐算子的误差预算并追求归零 |

**按这条原则,§6 的候选 3(在 Rust 侧复刻 Oracle 风格的 softmax)不应该做。**
Rust 的精确 exp 不劣于 ggml 的向量近似,把它换成近似只会降低精度、换不来一致性。
§6 候选 1(精确 sin/cos)已经落地,因为它不是"模仿近似",而是**去掉了一个与
参考无关的舍入差异源**;候选 5(短卷积累加顺序)同理,那是计算逻辑而不是近似选择。

### 11.2 二轮审阅结论

`fa10afc8`("Complete LFM3 entry points and shared compute dispatch")回应了
§3 的四项缺口。逐条核对实现:

| 原审阅项 | 实现 | 核对 |
| --- | --- | --- |
| 多 weight 分发 | 全部投影走 `LinearExecutor`;**Q/K/V/head gate(4 个)与 FFN gate/up(2 个)走 `run_group`**,共享 `PreparedRows` 激活量化 | `forward.rs` 的 `run_group` 两处 ✓ |
| GPU/CPU 自动分发 | CLI/HTTP/JEV 共用 `ComputePolicy`;Auto 在 M3 Max 上真跑 Vulkan 投影(2,312 次 dispatch);缺驱动回退实测过(`VK_DRIVER_FILES=/nonexistent/...`);`--compute vulkan` 全图拒绝 | `session.rs` 的 `new_with_compute` ✓ |
| HTTP | `Lfm3TextRuntime` 注册进 `build_text_runtime`;强制 GGUF Jinja;拒图片/工具调用;超上下文 400 | `src/models/lfm3/runtime.rs` ✓ |
| JEV | CLI + HTTP 单项/分组,用共享 `JevScorer` trait 与 `run_jev_decision_core` | `src/app/jev/{single,grouped}/lfm3.rs` ✓ |

另外两处值得记:

- **Auto 时打印 backend 诊断**(`LFM3 projections: requested=… backend=…; attention/shortconv/KV=Cpu`)
  —— 正面回应了 §8 R2 的"`--gpu` 静默 no-op"
- **清单状态细分**为 `Verified`(CPU/Q8_0 运行)+ `Experimental`(Vulkan 投影)
  —— 回应了 §8 R5 的"状态与证据强度不匹配"

### 11.3 对 §6 误差归因的修正

`fa10afc8` 修正了本报告 §6 的两处推测,方向都是本报告写得太乐观:

1. **softmax**:固定 Oracle 的 `ggml_vec_soft_max_f32` 在 NEON 上使用**向量 exp
   近似**与**四路 F32 横向求和**,再累加进 `ggml_float`;Rust 是精确 exp 与逐值
   F64 求和。差异在 **Oracle 一侧**,不是 Rust 一侧。§6 原先把 softmax 列为
   "已排除",结论对、理由错。
2. **Q8_0 tie 规则**:两侧原生 NEON Q8_0 激活量化**都**用 ties-to-even,不能把
   原生路径的差异解释成 tie 规则错误。§6 候选 4 的机理成立,但归因方向错了。

同时接受了本报告的两条方法论意见,并写进了机器可读证据:

- `control_scope`:`scalar changes SIMD reductions and activation quantization
  together; controls are not an additive decomposition` —— §6 候选 2 的实验设计
  有缺陷,scalar 控制不能当 dot 单项归因
- `acceptance.scope`:`empirical regression alarms for these Q8_0 fixtures; not a
  proven graph-wide bound` —— §6 提醒的"阈值是拟合出来的"已被如实标注

### 11.4 按原则重新分级:还剩什么

模型**工作正常**:四组 prompt 的 token ID 精确一致、各 8 步 greedy 与固定 Oracle
一致、24 次 embedding 逐位一致、5,176 个检查点的名称/shape/次数核对通过、
Tokenizer 与嵌入精确检查、有损路径检查有限值与完整 logits 误差。按 11.1 的原则,
这已经是可接受的状态,`1.25/0.20` 作为**回归报警线**(防止数值变差)而非
"精确对齐"证明。

于是本报告 §6 剩余建议的优先级变了:

| 原建议 | 重新分级 | 理由 |
| --- | --- | --- |
| 误差预算(把 ~1.0 分解到各算子) | ❌ **撤销** | 归零不是目标;11.1 明确不追比特对齐。记录量级与"是否可接受"已足够 |
| 在 Rust 侧复刻 Oracle 风格 softmax 做 A/B | ❌ **撤销** | 那是把精确换成近似,方向相反 |
| 探针覆盖 sigmoid head gate 路径 | ✅ **保留并已做** | 量的不是对齐,是**数值裕度**:哪里的输入差异容易被量化放大。这是鲁棒性信息,不是对齐债 |
| 阈值余量(1.0078/1.25 = 81%) | ⚠️ **降级为记录项** | 作为回归报警线够用;换权重/换平台仍可能越界,故保留在证据里而不是继续收紧 |

### 11.5 本轮随审阅完成的改动

| 文件 | 改动 |
| --- | --- |
| `tools/oracle/lfm3/checkpoints.patch` | 增加 `cb(attn_out, "lfm3_attn_gated", il)`,让 **sigmoid head gate 缩放后**的输出(输出投影的输入)能被探针覆盖。已对 pinned `19cce691` 的 `src/models/lfm3.cpp` 实测 `git apply` 通过 |
| `tools/oracle/lfm3/compare_checkpoints.py` | 探针从只查两个 norm 扩展到**每一个量化投影的输入**;报告 `q8_amplification_points` 全量清单、`projection_inputs_available` / `projection_inputs_without_oracle_checkpoint`;`first_norm_q8_change` 保留以兼容既有证据;量化器改为可注入(`quantize=`),便于测试 |
| `tools/oracle/lfm3/test_compare.py` | 新增:合成 fixture + 假 Q8_0 量化器,覆盖"全部投影输入可探测""漂移后被报告为放大""Oracle 暴露 gated 输出后即可探测""Oracle 缺失时跳过而非误判"四条路径 |
| `tools/oracle/lfm3/README.md` | 记录新的探针覆盖面、字段含义,以及"探针量的是数值裕度而非比特对齐" |

**本机条件限制**:LFM3-1.7B 的 Q8_0 权重(1.85 GB)不在本机,也没有构建好的 pinned
llama.cpp oracle,所以本轮改动**未经真实权重复验**;补丁与探针的行为分别用
`git apply`(对真实 `lfm3.cpp`)与合成 fixture 验证。真实复验命令见
`tools/oracle/lfm3/README.md`。

### 11.6 二轮审阅的风险

| 项 | 说明 |
| --- | --- |
| 未复验 | 本轮探针改动没在真实 trace 上跑过;`q8_amplification_points` 的实际内容未知 |
| 补丁未在真 oracle 上编译 | `checkpoints.patch` 的新 hunk 只验证了 `git apply`,没有编译 llama.cpp |
| 全库测试未跑 | 只跑了 `cargo check --all-targets`(0 error)、`cargo fmt --check`、定向测试组 |

### 11.7 本机复验（2026-10-29）

本机补齐了条件并做了一次完整复验。

**权重的来源变化**:作者 pin 的 `atomic.chat/LFM3-1.7B-GGUF` 已不可公开下载（ModelScope
API 返回 404）。改用官方 `LiquidAI/LFM3-1.7B-GGUF` 的 `LFM3-1.7B-Q8_0.gguf`
（1,846,238,496 B，SHA-256 `8f6b24182258205e9e01e21f1f24a1f208c1e8115578825d202ce39976af20b6`），
与作者 pin 的 `4dc04cae…` **不同**。因此下面是一次**全新自洽的验证**，不是对
`review-fixes.json` 的复现 —— 数值不可直接比较，但机制可以。

**Oracle**:按 `tools/oracle/lfm3/README.md` 的流程构建了 pinned
`19cce6919992fdffd92e0ac152caaf1e2b8f254e`（CPU only，禁 Accelerate/BLAS/Metal/
llamafile/OpenMP），打上 `checkpoints.patch`。

**对齐结果**:

| 项 | 结果 |
| --- | --- |
| prompt token IDs | 17/17 精确一致 |
| 8 步 greedy IDs | `[597,4695,20589,34,496,57922,415,8043]`，与 Oracle 一致，**与作者记录的那一组完全相同** |
| 最大 logits abs error | `0.79560184`（报警线 1.25） |
| 最大逐步 RMSE | `0.15557143`（报警线 0.20） |
| 检查点 | 6,280 个比较，24 次 embedding 逐位一致 |
| 首个浮点位差 | 位置 0、layer 0 的**短卷积输入投影**，`1.1920928955078125e-07` |

**探针的实际结果**（2,288 个放大点，六个投影输入全部可探测）:

| 投影输入 | 出现 | 最大 Q8 字节差异 | 该处 input_max_abs |
| --- | ---: | ---: | ---: |
| `ffn_swiglu`（down 投影输入） | 576 | **6,678** | `1.654e-02` |
| `result_norm`（输出头输入） | 8 | 1,907 | `6.002e-01` |
| `attn_norm`（q/k/v/head-gate 输入） | 552 | 1,876 | `4.567e-02` |
| `ffn_norm`（gate/up 输入） | 576 | 1,868 | `1.388e-01` |
| `attn_gated`（输出投影输入） | 264 | 1,667 | `1.950e-03` |
| `conv_conv`（短卷积输出投影输入） | 312 | 1,610 | `2.384e-03` |

这是本次复验最有价值的一条结论：**放大机制存在于每一个量化投影的输入上**，而不是
作者此前记录的那一处。`ffn_swiglu` 一处 `1.65e-02` 的输入差就让 2,048 维张量的
Q8_0 量化字节几乎整排改写（6,678 字节不同）；`attn_gated` 与 `conv_conv` 的输入差
只有 `2e-03` 量级，却同样造成 1,600 余字节变化 —— 说明这两个位置的 Q8_0 量化步长
与两侧的舍入差异同量级。**这正是 §11.1 所说的"量化本身就是比特损失"**：
最终 logits 能差到 0.8，而两边的 greedy 序列始终一致。

机器可读记录：[`tools/oracle/lfm3/probe-linux.json`](../../tools/oracle/lfm3/probe-linux.json)。

**本轮为支持这次复验而做的改动**:

| 文件 | 改动 |
| --- | --- |
| `src/models/lfm3/trunk/forward.rs` | 在 head gate 缩放之后增加 `attn_gated` 检查点 —— 它才是输出投影真正量化的输入（此前的 `attn_values` 是 gate **之前**的） |
| `tools/oracle/lfm3/checkpoints.patch` | 增加 `cb(attn_out, "lfm3_attn_gated", il)`，让 Oracle 暴露同一个张量 |
| `tools/oracle/lfm3/oracle.cpp` | capture 过滤器增加 `attn_out` 与 `ffn_swiglu` —— llama.cpp 的 `build_ffn()` **本来就有** `cb(cur, "ffn_swiglu", il)`，但过滤器把无前缀的名字挡掉了 |
| `tools/oracle/lfm3/compare_checkpoints.py` | 映射修正：`attn_values` → `attn_out`、`ffn_swiglu` → `ffn_swiglu`、`attn_gated` → `lfm3_attn_gated`；量化器改用 `struct.pack` 传入（`ctypes.from_buffer_copy` 不接受 list） |
| `tools/oracle/lfm3/probe-linux.json` | 新增，本机复验的机器可读证据 |

**对 §11.4 的修正**:§11.4 说"误差预算撤销、探针覆盖已做"。复验之后要补一句:
探针不但覆盖了全部六个投影输入，而且**实测表明放大点是普遍现象而非个案** ——
所以它作为回归监控项的价值比原先估计的高，而"误差预算"依然不必要。

**权重契约的修正**：作者 pin 的 `atomic.chat` 权重已不可下载，而 README、
`docs/MODEL_LIST.md`、`docs/develop/SUPPORTED_MODELS.md` 与 `review-fixes.json`
全都指向它 —— 复现链是断的。已把可获得的官方 `LiquidAI/LFM3-1.7B-GGUF` 写进这四处
（连带下载命令与 SHA-256），并在 `review-fixes.json` 里注明旧数字不可复现、
可复现的那一份在 `probe-linux.json`。

**Auto/Vulkan 投影路径的实测**（本机 Intel 集显）:

`--gpu` 确实把投影放到了 Vulkan（诊断打印 `requested=Auto backend=Vulkan;
attention/shortconv/KV=Cpu`，`Q8_0 matmul: packed int8 dot product`），输出与 CPU 路径
**逐字节相同**（32 步 greedy 与 216 token prompt + 8 步 decode 两种输入）。但更慢：

| 输入 | CPU | Auto | 比值 |
| --- | ---: | ---: | ---: |
| 32 步 greedy | 2.65 s | 22.34 s | 慢 8.4× |
| 216 token prompt + 8 步 decode | 13.80 s | 127.63 s | 慢 9.25× |

这是集显的必然结果：设备内存就是系统内存，1.85 GB 权重"放进显存"后每次投影都要过
内存总线，比 CPU 直接读 RAM 更慢。它印证了 §11.4 里"Auto 投影实验性支持"的定位 ——
在集显上它既不是加速路径，也不能默认开启。
