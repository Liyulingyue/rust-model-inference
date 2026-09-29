# Server backend 选择：flag 驱动 vs 元数据探测

一个 `rust-model-server` 进程 = 一个 `Backend` 变体，启动时定死。选择发生在
`src/app/server/mod.rs::build_backend()`，实际顺序（以源码为准）：

```rust
fn build_backend(options: &CliOptions) -> Result<Arc<Backend>, String> {
    if options.tts                  { return Ok(Arc::new(Backend::Tts(build_tts(options)?))); }
    if options.audio.is_some()      { return Ok(Arc::new(Backend::Asr(build_asr(options)?))); }
    if options.embedding            { ... Backend::Embedding ... }
    if is_rerank_gguf(&options.model) { ... Backend::Rerank ... }   // ← 唯一的元数据探测
    if options.clm_head.is_some()   { return ... Backend::Clm ... }
    if options.gliner2_decide       { return ... Backend::Gliner2 ... }
    Ok(Arc::new(Backend::Text(build_text(options)?)))                // 兜底
}
```

`Backend` 决定两件事：**能加载哪些权重**，以及**挂哪些路由**。所以选错不是
"功能少一点"，而是启动就用错误的方式开文件、或者把请求路由到一个不可能成功的
handler。

## 选择依据只有两个来源

| 来源 | 含义 | 适用条件 |
|---|---|---|
| **flag** | 判别信息在命令行上 | 命令行能唯一确定后端 |
| **元数据探测** | 判别信息只在 GGUF 里 | **同一个 `general.architecture` 对应两种语义不同的模型**，命令行分不出 |

## 各后端用的是哪种，以及为什么

| 后端 | 依据 | 理由 |
|---|---|---|
| `Text` | 兜底 | 生成模型的默认形态 |
| `Embedding` | `--embedding` | 判别信息在 flag：同一个 qwen3 GGUF 既能当聊天模型也能当 embedding |
| `Asr` | `--mmproj`（+ arch） | 同上，投影仪是第二个文件，命令行携带语义 |
| `Tts` | `--tts` | 同上 |
| `Clm` | `--clm-head` | **头在第二个 GGUF 里**，`--model` 只是个普通 qwen3，判别信息不可能来自权重 |
| `Rerank` | **元数据探测** `is_rerank_gguf()` | reranker 和普通 qwen3 聊天的 `general.architecture` **都是 `"qwen3"`**，命令行无从区分 |
| `Gliner2` | `--gliner2-decide` | arch 是 `gliner2`，`--model` 本身就唯一；用 flag 是为了和 CLM 对齐、避免再造一个探测函数 |

### 为什么 rerank 必须探测而 gliner2 不必

这是唯一一处不对称，值得写清楚，因为看起来像疏漏：

```rust
// is_rerank_gguf：三个条件同时成立
arch != "qwen3"                          → false   // 且 chat 模型也过这关
loader.metadata("qwen3.pooling_type") != 4 → false // 4 = llama.cpp 的 rank pooling
!loader.tensor_info("cls.output.weight")   → false // 真正打分的分类头
```

rerank 的困境是**同一个 arch 名下有两种模型**。GLiNER 的 GGUF 里
`general.architecture = gliner2`，和任何其他 arch 都不冲突，`--model` 一次性说清，
所以不需要第二个机制。

## 探测函数自身的要求

`is_rerank_gguf` 是仓库里唯一的探测实现，它立下的规矩：

1. **只读 metadata 和 tensor 目录，不加载权重。** `GGUFLoader::from_file` 在函数
   结束时 drop，文件句柄随即关闭。它排在完整 `build_text` 之前，早退能避免白加载
   几个 GB。
2. **读失败返回 `false`**，不是 panic。文件打不开应该由后续真正的加载给出有意义的
   报错，而不是在探测阶段吐一个无关的错误。
3. **多重条件互相印证。** `pooling_type` 是 llama.cpp 侧的约定，`cls.output.weight`
   是权重侧的事实；只有一方可能是转换器的疏漏，两边都在才下结论。

## 已知隐患：探测排在 flag 之前

```rust
if is_rerank_gguf(&options.model) { ... }   // 先跑
if options.clm_head.is_some() { ... }       // 后跑
```

所以 `--clm-head some-reranker.gguf` 会静默落进 Rerank，**头文件被丢掉**，服务起
来但打分用的是 rerank 而不是 CLM 的 cosine。这是 `TODO(clm)` 记录的问题。

顺序本身难倒置（探测必须早于 `build_text`，否则白加载几个 GB），但"静默"是可以修的：
探测命中且用户同时给了 flag 时应该显式报错。

（附注：`--tts` / `--audio` / `--embedding` 排在探测**之前**。它们都是 flag 且
互不重叠，先返回不影响探测；但如果用户同时传了这些 flag 和一个 reranker GGUF，
同样会静默丢掉前面的 flag。）

## 新增后端时的决策树

按顺序问自己：

1. **`--model` 的 arch 能唯一确定这个后端吗？** 不能（同 arch 多语义）→ 写探测
   函数，规矩见上一节。能 → 继续。
2. **判别信息是否只有在第二个文件里？** 是（像 CLM 的头、ASR 的 mmproj）→ 用 flag。
3. **否则用 flag。** 别为了"和某个后端对称"去改既有后端的驱动方式——rerank 的
   全 flag 化是 breaking change（`rust-model-server --model Qwen3-Reranker.gguf`
   今天不写 flag 就能跑），和任何新后端都是独立决策。

## 路由约束

后端定了，路由就是一张白名单表。当前：

| 路由 | Text | Embedding | Asr | Tts | Rerank | Clm | Gliner2 |
|---|---|---|---|---|---|---|---|
| `/v1/chat/completions` 等 | ✓ | | | | | | |
| `/v1/embeddings` | | ✓ | | | | | |
| `/v1/audio/*` | | | ✓ | | | | |
| `/v1/audio/speech` | | | | ✓ | | | |
| `/v1/rerank` | | | | | ✓ | | |
| `/v1/jev/score` | ✓ | | | | | ✓ | ✓ |
| `/v1/jev/grouped` | ✓ | | | | | | |
| `/v1/jev/image*` | ✓（需 `--mmproj`） | | | | | | |

未列出的路由由 axum 直接 404。**这是模型能力边界，不是接口缺功能**——例如
`/v1/jev/grouped` 对 Clm/Gliner2 返回 404，因为 per-group softmax 在 cosine 打分
和"每头独立归一化"里没有对应物。
