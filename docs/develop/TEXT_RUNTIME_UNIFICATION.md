# 文本推理运行时统一：HTTP 与 CLI 共用一条生成路径

> **状态**：设计已锁定，待实施。四个决策由维护者确认（见 §5）。
> **动机**：HTTP 层当前不是 CLI 的薄封装，而是一套平行的 arch dispatcher——
> CLI 文本入口 `app::text::run_inference` 从未被 HTTP 复用，导致每个 arch 在
> HTTP 侧都要单独写分支 + 手写 decode 循环，且大量 arch 加载后只能返回 501。

## 1. 现状证据

### 1.1 CLI 侧：一个入口 + arch if/else

```
src/main.rs:14-32        DispatchMode 预占（dreamx / yue2 / tts / qwen-drive）
src/main.rs:388-463      DispatchMode::Model → app::text::run_inference
src/app/text/generation.rs:21-164
                         arch 链：hunyuan-dense / lfm2 / lfm2moe /
                         uses_llama_trunk → llama::run_inference（:14-19, :100）/
                         spark2_5 / nemotron_h / falcon-h1 / catch-all = qwen3（:148）
```

新增一个 stock llama 系 arch 只需 2 处：`src/core/loader.rs:392-414` 的 allowlist
+ `src/app/text/generation.rs:14-19` 的 `uses_llama_trunk`。

### 1.2 HTTP 侧：3 个变体 + 每变体手写 decode

| 位置 | 内容 |
|---|---|
| `src/app/server/mod.rs:96-114` | `TextInner { Qwen3, Qwen35, Lfm2Moe, Fallback{arch} }` |
| `src/app/server/mod.rs:803-840` | `build_text`：`match arch` 只认 `qwen3\|qwen3vl` / `qwen35` / `lfm2moe` |
| `src/app/server/api.rs:694-811` | `generate()`：Qwen3 调 `generate_streaming_until`；Qwen35 手写 `step_with_tokens` + `sample_token_from_logits`；Lfm2Moe 手写 `forward_token`；`Fallback` → 501 |
| `src/app/server/api.rs:303-311` | Fallback 在请求期返回 HTTP 501 |
| `src/app/server/api/tools.rs:29-37` | `is_qwen35` arch 白名单，其余 arch 直接 `Err("Tool/chat template is unsupported for architecture {arch}")` |

`llama::run_inference_tokens`（llama 家族 token 级入口）在 HTTP 层**零调用点**，
且上层 `run_inference_tokens` 直接 `println!` / `print!("Output: ")` +
`io::stdout().flush()`（`src/models/llama/trunk/forward.rs:421,443,473-474`）——
这是 CLI 入口无法被 HTTP 复用的直接原因。

**后果**：llama / nanbeige / exaone / k2-horizon / granite、gemma4、
hunyuan-dense、lfm2 dense、lfm25（非 MoE）、spark2_5、nemotron_h、falcon-h1、
qwen2 全部能加载但请求 501，与 `README.md:414-416` 的宣称矛盾。

### 1.3 其他重复面

- **prompt 三条路径**：`src/prompt.rs` 7 个 builder（CLI 各 trunk 自选）、
  `src/models/chat_template.rs` 5 个 preset（无 Jinja 引擎，`chat_template.rs:3-5`
  明说无法渲染 GGUF 里的 Jinja 源）、HTTP 自己的 `api/tools.rs::build_prompt`
  （硬编码 `enable_thinking=false`）。
- **采样三套实现**：CLI 的 `src/ops/sampling.rs` 链、qwen3 trunk 内部、
  server 自己的 `sample_token_from_logits(logits, temperature)`
  （`src/app/server/mod.rs:228`，只有温度）。
- **KV format 分歧**：server 对 qwen3 硬编码 `KvFormat::F16`
  （`src/app/server/api.rs:699`）、lfm2moe 同样 F16
  （`src/app/server/mod.rs:825`）；CLI 默认 F32（`KvFormat` 的 `#[default]`）。

## 2. 目标架构

```
                ┌──────────────────────────────────────┐
  CLI ──────────┤  build_text_runtime(arch, source, …)  ├────────── HTTP (/v1/chat/*)
                │  唯一 arch→runtime 分发点              │
                └───────────────┬──────────────────────┘
                                │ Box<dyn TextRuntime>
   ┌────────────────────────────┼────────────────────────────┐
LlamaAdapter              Qwen3Adapter                Lfm2MoeAdapter …（每 arch 一个薄适配器）
（调 trunk session）      （调 generate_streaming_until）  （调 session.forward_token）
   └────────────────────────────┴────────────────────────────┘
                 GenerationRequest + TokenSink
        CLI: StdoutSink   HTTP: SseSink + StopSink(装饰器)
```

### 2.1 四个新原语（`src/core/generation.rs`）

```rust
pub struct SamplingParams { temperature, top_k, top_p, repetition_penalty, seed }
pub struct GenerationRequest { token_ids, max_new_tokens, sampling, prefill_batch_size, stop }
pub trait TokenSink { fn push_text(&mut self, chunk: &str) -> Flow }   // Flow::{Continue, Stop}
pub trait TextRuntime: Send {
    fn arch(&self) -> &str;
    fn generate(&mut self, req: &GenerationRequest, sink: &mut dyn TokenSink)
        -> Result<GeneratedText, GenError>;
}
```

设计要点：

- **停止序列用装饰器**：`StopSink` 包住底层 sink，截断 + 前缀保留（直接复用
  `src/app/server/api/stop.rs::StopTracker` 的逻辑），arch 无感。CLI 无 stop 序列。
- **采样默认值单一来源**：`resolve_sampling(arch, source, opts)` 读 GGUF
  `general.sampling.*`（llama 家族）+ per-arch 规则（ASR/gemma4 greedy）。
  HTTP 侧**只填 temperature**，`top_k/top_p/penalty` 继续 warn-ignore（决策 3）。
- **RuntimeOptions per-arch 显式化**：`kv_format / prefill_batch_size /
  sampling_profile` 由 `build_text_runtime` 按 arch 显式给出，把现在散落的
  硬编码（`api.rs:699` F16、`mod.rs:825` F16）集中到一张表。**迁移期 HTTP 行为
  不变**：qwen3/qwen35/lfm2moe 继续 F16，CLI 路径继续各自默认（决策 1）。

### 2.2 prompt 收敛

`src/prompt.rs` 新增分发器 `build_chat_prompt(tokenizer, arch, messages,
thinking) -> Result<Vec<u32>, String>`，内部按 arch 调既有 builder；llama trunk
内联的 arch+chat_template 嗅探（`src/models/llama/trunk/forward.rs:240-268`）
改为调用它。HTTP `tools.rs::build_prompt` 改为薄包装：tool 注入段保留（Qwen 系），
base 走共享 builder，`enable_thinking` 从请求带下来。

## 3. 验证策略（两层，均不需要真实权重）

- **Tier A（常驻，零权重）**：prompt builder 黄金测试——每个 builder 的输出
  token 序列钉死在 `src/prompt.rs` 的 `#[cfg(test)] mod tests`（沿用现有
  `prompt_tokenizer()` 合成 tokenizer 套路）。覆盖 Phase 2 与每个 arch 迁移的
  prompt 部分。
- **Tier B（env-gated，符合仓库约定）**：CLI vs trait 字节级一致测试，仿
  `tests/nanbeige.rs` / `tests/exaone_q8.rs` 的 `RMI_*` 门控风格，有权重时本地
  跑、CI 自动跳过。
- Tier C（可选，默认不做）：n_embd=32 / 1 层的合成 llama GGUF 端到端黄金测试，
  约 200~300 行脚手架；Tier A+B 不足时再上。

## 4. Commit 序列（不 push，每步可回滚）

| # | 内容 | 行为变化 |
| --- | --- | --- |
| 0 | 本文档 + `TODO.md` 指针 | 无 |
| 1 | Phase 0：Tier A 黄金基线（现有 builder 输出全部钉死） | 无 |
| 2 | Phase 1：`core/generation.rs` types + trait + `StopSink` 装饰器 | 无（纯新增） |
| 3 | Phase 2：prompt 收敛（`build_chat_prompt`、trunk 改调、HTTP 委派） | HTTP 模板变正确（non-thinking 保持）；PR 显式说明 |
| 4~10 | Phase 3：每 arch 一个 commit（llama 家族 → qwen3 → qwen35 → lfm2moe → lfm2/lfm25 → spark → nemotron_h/falcon_h1 → hunyuan） | CLI 逐字节不变（Tier A/B 证明）；HTTP 逐个 arch 切换 |
| 11 | Phase 4：HTTP 收尾，删 `TextInner` / 三段 decode 循环 / tools 白名单 | 501 集合清零（未适配 arch 保留 Fallback） |
| 12 | Phase 5：清理第三套采样实现、KV/采样差异表入档、ARCHITECTURE 更新 | 无（HTTP top_k/top_p 仍 warn） |

## 5. 已锁定决策（维护者确认）

1. **CLI 现有行为逐字节不变**——黄金测试锁死；HTTP 迁移期保持各 arch 现有
   HTTP 行为，不借机统一默认值。语义差异集中记录，不静默修改。
2. **每 arch 一个 commit/PR**，各自带 Tier A/B 绿灯。
3. **采样统一后 HTTP 的 top_k/top_p/penalty 暂不启用**，保持 warn-ignore，
   待单独 PR 评估。
4. **保留 501 Fallback**：未适配 arch 仍可加载、请求返回 501；新 arch 只要进
   loader allowlist + arch 链即自动获得 HTTP 能力，`src/app/server/` 零改动。

## 6. 风险清单

| 风险 | 缓解 |
| --- | --- |
| 每 arch 的采样默认值不同（ASR/gemma4 greedy、llama GGUF 默认、audio temp 0） | 全部进 `resolve_sampling`，迁一个 arch 用 Tier A/B 抓一次 |
| qwen35 forward 需 `&mut self` | `Mutex<Box<dyn TextRuntime>>`，`server/mod.rs:103` 已有先例 |
| lfm2moe 需 `reset()`、各 trunk KV 生命周期不同 | adapter 内保留各自语义，`GenerationRequest` 不假设 |
| HTTP 现有 F16 硬编码 vs CLI F32 | 迁移动作 = 原样搬进 `RuntimeOptions`，PR 点名，不改数值 |
| hunyuan v1 依赖 `hy_*` semantic tokens（仅 `pre="hunyuan-dense"` 有） | Tier A 用合成 tokenizer 覆盖 v1/v2 两分支 |

## 7. 实施进度与验证矩阵

分支 `msi-new`，截至 2026-09-27：

| commit | 内容 |
| --- | --- |
| `2e62e8a` | 本计划文档 + `TODO.md` 指针 |
| `6b8d700` | llama 家族解封 501（`build_prompt_tokens` 抽出 + `TextInner::LlamaTrunk`） |
| `f23a69b` | 删除 `TextInner` 四变体，改为 `build_text_runtime` + 四个 adapter；`ops::generation_runtime` 类型层 |
| `491249e` | llama 家族 CLI/HTTP 共用 `LlamaSampler` + `stop_after_sample` |
| `87c199c` | CLI/HTTP 一致性哨兵 `tests/cli_http_agreement.rs`；lfm2moe 共用 `Lfm2MoeSampler` + 停止规则 |

### 一致性哨兵（永久守卫）

```bash
RMI_AGREEMENT_MODEL=models/K2-Horizon-GGUF/K2-Horizon-1B-BF16.gguf \
  cargo test --test cli_http_agreement
```

跑同一 prompt 的 CLI 二进制与 `build_text_runtime` 产物，断言解码文本一致；
按 arch 分流 prompt builder（llama trunk / lfm2moe / qwen ChatML）。
无 `RMI_AGREEMENT_MODEL` 时自动跳过。

| arch 家族 | 状态 | 证据 |
| --- | --- | --- |
| llama / nanbeige / exaone / k2-horizon / granite | ✅ 一致 | K2-Horizon-1B-BF16，temp=0 与 0.8 均逐字节一致 |
| lfm2moe | 🟡 forward 分裂**已修**，余 1-ULP 贪心翻转 | 修复：`Lfm2MoeSession::prompt_len` + `is_prefill = pos < prompt_len`（原先硬编码 `true`，decode 阶段把整段 b*x 历史塞回短卷积窗口）。剩余差异：第 ~30 字符处 `says "` vs `says: "`，greedy 下 top-2 logits 极接近时的 1-ULP 翻转，两侧首 token 与前 30 字符完全一致；精确定位需逐层 logits 对比（独立任务） |
| qwen3 | ✅ greedy 一致 | `Qwen3-0.6B-Q8_0`。两处真实因素：session capacity 须为 `min(n_ctx, max_context)`（原先 `prompt+max_new`，改变 KV 布局与 chunked prefill 归约），prefill batch 须为默认 64 |
| qwen35 | ✅ greedy 一致 | `Qwen3.5-0.8B-Q8_0`，同 capacity/batch 口径 |
| 任意 arch，temp>0 | ⛔ 不适用 | 两端 RNG 独立且 CLI 侧无种子，字节一致不可达；哨兵只断言 greedy |

### 采样器统一状态

`ops::sampling::sample_greedy_or_temperature` 是唯一的 greedy/temperature
采样器，qwen3 CLI（`trunk::util::sample_token`）、qwen35 CLI
（`app::text::generation::sample_token`）与四个 HTTP adapter 全部委托它。
替换掉的三个副本语义并不相同，统一时按"已验证那份"裁决：

| 曾存在的副本 | 差异 | 处置 |
| --- | --- | --- |
| `qwen3::trunk::util` | 校验空/非有限 logits；严格 `>`（平局取**首个**） | 作为规范实现 |
| `app::text::generation` | **硬编码 `r = 0.5`**（永远取分布中位数），无校验 | 委托规范实现；**qwen35 CLI 在 `--temp > 0` 下行为改变**（stub → 真随机），greedy 不变 |
| `ops::sampling`（源自旧 server） | `max_by(partial_cmp)` 平局取**末个**；NaN 会 panic；空 logits 静默返回 0 | 删除 |
| `spark::trunk::forward` | 用 `softmax_approx_inplace`（近似 exp），与 arch 的既有精度路线一致 | 保留不动 |

### 仍未做（有意保留）

- qwen3 / qwen35 的采样与各自 CLI 对齐（各自的 CLI 路径与 adapter 用的函数不同）。
- CLI 全量委托 `TextRuntime`（llama trunk 的 600 行 forward + `RUST_LLAMA_DEBUG_*` hooks
  内联在循环里，抽骨架风险高于收益）。
- HTTP 侧 `top_k` / `top_p` / `repetition_penalty` 仍只 warn（决策 3）；注意 llama adapter
  已因复用 `LlamaSampler` 而**开始读取** GGUF `general.sampling.*` 默认值。
