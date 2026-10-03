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
| `Gliner2Boundary` | `--gliner2-boundary` | 同上，**且** `gliner2.variant = "boundary"`；flag 选模式，权重确认变体 |

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

## 已知隐患（已修）：探测排在 flag 之前

```rust
if is_rerank_gguf(&options.model) { ... }   // 先跑
if options.clm_head.is_some() { ... }       // 后跑
```

顺序难倒置（探测必须早于 `build_text`，否则白加载几个 GB），所以探测命中时
**显式报错**而不是让 flag 被静默丢掉：

| 同时给的 flag | 行为 |
|---|---|
| `--clm-head h.gguf` | 400：`--clm-head` 选 CLM，但 model 是 reranker，`h.gguf` 会被丢 |
| `--gliner2-decide` | 400：同上，且 GLiNER2 需要 DeBERTa GGUF 而非 Qwen3 reranker |
| `--gliner2-boundary` | 400：同上，且 boundary 变体必须是 boundary-variant DeBERTa |
| `--mmproj m.gguf` | 400：`--mmproj` 意味多模态聊天，reranker 上它会被丢 |
| 无其它 backend flag | 正常进 Rerank（探测的本职） |

`--tts` / `--audio` / `--embedding` 排在探测**之前**且各自提前 return，所以它们
天然优先，不存在"丢掉"的问题——用户显式选了那个后端。

## GLiNER2 boundary 后端：flag 选模式，权重定变体

`--gliner2-decide` 和 `--gliner2-boundary` 指向**同一个 arch**，权重里靠
`gliner2.variant` 区分（`classification` vs `boundary`）。所以：

- flag 决定**跑哪个 head**，因此决定挂哪条路由；
- `is_boundary_gguf()` 确认**权重确实是 boundary 变体**，启动时校验一次；
- 两者不匹配时报错而不是回退。`--gliner2-decide` 吃到 boundary GGUF 时，
  分类头根本不存在，回退会得到"加载成功、输出全空"的静默错误。

模型不缓存：mapping 留在后端里，**每个请求**从 `&dyn TensorSource` 现建一个
`BoundaryModel`（零拷贝视图 + 一次 settings 解析），和 Decide 后端每请求重建
tokenizer 的做法一致，因此不需要 `'static` 泄漏。

### 为什么 boundary 不挂在 `/v1/jev/score` 上

`JevResult` 装不下 span：它只有 per-question 的分类概率，没有 word offset、没有
重叠消解后的顺序、也没有多组 classification。所以 boundary 走自己的
`POST /v1/jev/boundary`，body 直接吃 **reference 自己的 schema 形状**
（`{"entities": [...], "entity_descriptions": {...}, "classifications": [...]}`）。

刻意**不**把 `/v1/jev/score` alias 到它：JEV 形状的 body 在那里会因为缺 `schema`
字段而报反序列化错误，读起来像"请求写错了"，而不是"这个 server 不提供这个路由"。
404 才是诚实的答案。

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
