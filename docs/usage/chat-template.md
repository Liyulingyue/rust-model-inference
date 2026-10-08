# Chat 模板：Jinja2 渲染

GGUF 在 `tokenizer.chat_template` 里自带模型的 chat 模板。它是一段 **Jinja2
源码**，不是字符串拼接。仓库里有两套更早的实现：

- `src/prompt.rs` —— 14 个按模型手写的 builder
- `src/prompt/legacy.rs` —— 7 个 `ChatTemplate` 预设
  （`chatml` / `llama3` / `gemma` / `lfm2` / `glm4` / `exaone` / `phi4`）

`chat_template.rs:36-40` 当初明确写了"不引入 minijinja / tera"，理由是
"base model 没有 instruct 微调，不需要 thinking 和 tool call"。**这个前提
今天已经不成立** —— 仓库里跑的就是 Qwen3（输出 `<think>`）、LFM2.5-8B-A1B、
GLM-4、GLiNER 这类指令模型，而真实模板的复杂度也远超手写 builder 能覆盖的
范围（Qwen3-0.6B 4905 字符，LFM2.5-8B-A1B 4621 字符，含 `{% macro %}`、
`namespace()`、`|tojson`、切片、`is mapping/string/defined/none`、空白控制）。

因此新增 `src/prompt/jinja.rs`，用 **minijinja 2.24**（Rust 的
Jinja2 协议实现）渲染模型自带的模板。

> Jinja2 是模板语言规范，minijinja 是它在 Rust 的一个实现。规范与实现要分开
> 看：换 crate 不改变协议，GGUF 里存的那份源码语法也不由我们决定。

## 用法

```bash
# 渲染 GGUF 自带的 tokenizer.chat_template
cargo run --release --bin rust-model-inference -- \
  --model Qwen3-0.6B-Q8_0.gguf --prompt "Capital of France?" --jinja

# 用外部 Jinja2 文件覆盖（隐含 --jinja）
cargo run --release --bin rust-model-inference -- \
  --model Qwen3-0.6B-Q8_0.gguf --prompt "hi" --chat-template-file my.jinja
```

优先级：**`--chat-template-file` > GGUF `tokenizer.chat_template` > 内置 builder**。

两个开关**默认关闭**。不开时行为与之前逐字节一致 —— 这不是保守，是因为打开
默认值会让所有"模板与我们 builder 不一致"的模型 token ID 悄悄改变，而这正是
A/B 测试要抓的东西。

`--chat-template <name>` 保持原义，仍然选内置预设，与 `--jinja` 互不影响。

## 验证过的等价性

`tests/jinja_chat_template_ab.rs`（`#[ignore]`，需真实 GGUF）在同一条 prompt
上分别走两条路径并**逐 token 比对，不接受任何容差**：

```sh
JINJA_AB_QWEN3=models/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf \
JINJA_AB_LFM25=models/LFM2.5-8B-A1B-GGUF/LFM2.5-8B-A1B-Q8_0.gguf \
  cargo test --profile release-fast --test jinja_chat_template_ab -- --ignored --nocapture
```

实测结果（2026-10-07）：

| 模型 | 结果 |
|---|---|
| Qwen3-0.6B (Q8_0) | **逐 token 完全一致**，`thinking=true` 12 token / `thinking=false` 16 token，零分歧 |
| LFM2.5-8B-A1B (Q8_0) | 渲染并正确分词；与手写 builder **存在已知分歧**，见下节 |

CLI 层同样验证过：Qwen3-0.6B 加 `--jinja` 与不加，prompt 均为 16 token，
输出同为 `The capital of France is **Paris**.`

## 两个容易踩的坑（都已在代码里处理）

**1. special token 必须整体成词。** 渲染出的字符串用
`EncodeOptions { add_special: false, parse_special: true }` 编码，否则
`<|im_start|>` 会被拆成 `<`、`|`、`im_start`…… 一串普通 token。这是实现里
最容易静默出错的地方，A/B 测试直接断言 `<|im_start|>` / `<|im_end|>` 在结果
里是**单个 id**。

**2. 不能重复加 BOS。** 模板自己会输出 `{{ bos_token }}`（LFM2.5 就有），所以
`add_special` 必须是 `false`，否则 BOS 出现两次。

## 真实模板里的非标准构造

扫描了本地全部 GGUF 的 `tokenizer.chat_template`，跨模型只有两类非标准写法：

| 构造 | 出现处 | 处理 |
|---|---|---|
| `{% generation %}` / `{% endgeneration %}` | 仅 LFM2.5 | llama.cpp 扩展，标记"模型该生成哪一段"，单次渲染不产生输出。改写成同 trim 标记的注释 `{#- -#}` —— 若直接删除会丢掉 `{%- -%}` 的空白控制，模板每轮之间会多出空行。 |
| `x.get("k")` | 仅 LFM2.5（2 处） | Python `dict.get`，minijinja 的 map 没有这个方法。改写成 `x["k"]`，仅限单字符串字面量参数形式。 |

两点都有单测钉住（`generation_tag_keeps_whitespace_control`、
`map_get_rewrites_match_truthiness`、`two_arg_get_is_left_alone`）。后者验证了
`x.get("k")` 与 `x["k"]` 在"存在 / 缺失 / 空串"三种情况下真值判断一致 —— 这个
等价关系是被测出来的，不是推断的。

**其他未识别的语句一律不动**，交给 minijinja 报错，避免我们静默改写一个没看懂
的模板（`unknown_statement_is_not_mangled` 守住这条）。

## 已知分歧：LFM2.5 手写 builder 很可能是错的

`prompt.rs:213` 的 `build_lfm25_chat_prompt_with_thinking` 输出：

```
<|im_start|>user\n{content}\n<|im_start|>assistant\n
```

模型自带模板输出：

```
{bos_token}<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n
```

差两点：**手写版完全没有 `<|im_end|>`**，且在 content 后多一个 `\n`。
实测 token 数 12 vs 14，**首个分歧在第 8 个 token**。

**没有顺手改手写 builder。** 原因：`SUPPORTED_MODELS.md:57` 的
LFM2.5-1.2B-Instruct 是 `Verified`，证据是"与 llama.cpp 一致 8/8 greedy
token"，而它走的就是这条手写路径。改 builder 会作废那份逐位证据，需要重跑
parity 并重新定级。

**但 `--jinja` 已经是可用的正确出口**，而且差别不只是格式：

同一条中文 prompt，8B-A1B Q8_0：

| | prompt token 数 | 思考语言 | 表现 |
|---|---|---|---|
| 默认（手写） | 24，**缺 `<|im_start|>`(124899) 与 `<|im_end|>`(124900)** | 英文 | 英文复述中文问题，并凭空捏造了一条 "You are a helpful assistant." 系统指令 |
| `--jinja` | 14，控制 token 完整 | 中文 | `嗯，用户问的是"粒子群算法是什么"…` 本地化推理 |

也就是说，残缺的 prompt 正在**实际削弱指令遵循**，不只是"不够标准"。
该分歧由 `lfm25_jinja_known_divergence_from_hardcoded_builder` 记录成一条会
失败的信号：一旦手写版也发出 `<|im_end|>`，这条测试就会失败，提醒去重跑
parity 并更新 `SUPPORTED_MODELS.md`。

## 当前范围

### 已接入（10 条路径）

| 路径 | 覆盖 arch | 位置 |
|---|---|---|
| Qwen3 / Hunyuan | `qwen3`、`hunyuan-dense` | `src/app/text/qwen3.rs` |
| llama trunk | `llama` / `exaone` / `k2-horizon` / `granite` / `nanbeige` / `phi3` / `glm4` / `mistral3` / `gemma2` / `gemma4` | `src/models/llama/trunk/forward.rs`（直接短路整条 arch 启发式链） |
| LFM 家族 | `lfm2` / `lfm2.5` / `lfm2moe` | `lfm2/`、`lfm25/`、`lfm2moe/trunk/forward.rs` |
| xing4_0 | `xing4_0` | `src/models/xing4_0/trunk/run.rs` |
| spark2_5 | `spark2_5` | `src/models/spark/trunk/forward.rs` |
| nemotron_h | `nemotron_h` | `src/models/nemotron_h/trunk/forward.rs` |
| falcon-h1 | `falcon-h1` | `src/models/falcon_h1/trunk/forward.rs` |
| server 多轮 | `/v1/chat/completions` | `src/app/server/api/tools.rs` + `TextBackend.chat_template` |

> server 侧带 tools 或带图片的请求**不走 Jinja**（见下方"其他边界"）。

llama trunk 那条最有价值：`--jinja` 一开就**整段跳过**原先的 MiniCPM5 /
Mistral / Zephyr / Granite / Nanbeige 判定链 —— 模板本身就是规范。

| 多模态 qwen3vl | `qwen2vl` / `qwen3vl` / `qwen3vlmoe` | `src/app/text/multimodal.rs`（`run_qwen3_family_multimodal`） |
| 多模态 qwen35 | `qwen35` | `src/app/text/multimodal.rs`（logits 变体，同一契约） |
| JEV grouped | 9 个 grouped wrapper | `src/app/jev/grouped.rs::grouped_prompt_via_jinja` |
| JEV qwen35 带图 | `qwen35` | `src/app/jev/single/qwen35.rs::plan_prompt` |

### 多模态：模板已经知道怎么写，我们之前只是没用

Qwen3-VL-2B 的模板（5292 字符）本身就描述了多模态 prompt：

```
<|vision_start|><|image_pad|><|vision_end|>   图片
<|vision_start|><|video_pad|><|vision_end|>   视频
add_vision_id → "Picture 1: " / "Video 1: "   自动编号
```

它还遍历 content parts 并判断 `c.type == 'image'`，所以把 content 传成
`[{type:image},{type:text}]` 数组就得到正确结果。

**关键的一步是占位符展开。** 模板每个媒体只写**一个** `<|image_pad|>`，但
视觉编码器为这张图产出了 `rows` 个 grid token。两个位置函数
（`build_qwen3_media_positions` / `build_qwen35_positions`）的契约是一致的：
遇到 `placeholder_id` 就按 `grid_shapes[i]` 取**连续 N 个**，且必须是同一 id。

于是顺序是：模板渲染 → `parse_special` 分词 → 把每个 `<|image_pad|>` **展开成
连续 N 个** → 位置计算。展开后原有契约完全满足，所以位置数学一行没改。
`expand_vision_placeholders` 在占位符数与 grid 数不一致时**直接报错** ——
不一致意味着模板与投影器对"有多少媒体"的理解不同，放过去会让每个位置 id 静默错位。

实测（Qwen3-VL-2B-Instruct Q8_0 + mmproj Q8_0，448×448 红椭圆测试图）：

| 路径 | 输出 |
|---|---|
| 手写 | "The image is a simple, stylized logo featuring a red, symmetrical, oval-shaped design..." |
| `--jinja` | "The image is a simple, stylized logo featuring a red, symmetrical, oval shape with a smooth, curved design. The logo is centered within a white background..." |

两者都是视觉接地的准确描述，质量相当。

### JEV 打分：已接入 LFM2 / LFM2.5 / LFM2-MoE，且修掉一个真 bug

JEV 的形状其实和非 JEV 一样：`[system?, user]` + **打开** assistant 轮
（下一个 token 就是要打的分，所以 `add_generation_prompt = true`，和生成一样，
`thinking` 关掉否则打分位置会跑）。差别只是**内容由谁撰写** —— JEV 里
system 是 `jev_system_prompt(mode)`、user 是 `jev_payload_json(...)`，都由引擎
生成。渲染层不需要为它另开一套。

原来的 `src/app/jev/single/lfm2.rs` 直接拼裸文本
`"system\n{system}\n" / "user\n{payload}\n" / "assistant\n"`，且
`parse_special: false` —— **一个控制 token 都没有**。注释里还把这个猜测写成了
"the LFM2 convention shared with LFM2.5 / LFM2-MoE"，但 LFM2.5-8B-A1B 的真模板
发的是 ChatML。

实测后果（LFM2.5-8B-A1B Q8_0，20 线程，选项 Paris / London / Tokyo）：

| | 选择 | confidence | entropy | margin |
|---|---|---|---|---|
| 默认（错格式） | **C: Tokyo** ❌ | 0.4961 | 1.0074 | 0.1512 |
| `--jinja`（真模板） | **A: Paris** ✅ | **0.8258** | **0.5248** | **0.6719** |

上下文明写 "The Eiffel Tower is located in Paris"，默认路径**答错了**；`--jinja`
答对，且 confidence 从 0.50（几乎无区分度）升到 0.83，margin 大 4.4 倍。

### 其余 8 个 arch 也已接入

**11/11 arch 全部接入**：`gemma4` / `nemotron_h` / `spark` / `llama` /
`qwen35` / `qwen3` / `hunyuan` / `falcon_h1` / `lfm2` / `lfm2.5` / `lfm2moe`。
模板在 `new()` 里解析一次存成 `Option<JinjaChatTemplate>`（那里才有
`TensorSource`），`build_prompt` 只负责渲染 —— 因此每个 scorer 的改动是
「一个字段 + 一个构造参数 + 一个分支」。

**注意：grouped 不委托 single 的 `build_prompt`。** grouped 在
`build_grouped_prompt` 里自己出 prompt，走自由函数
`grouped.rs::grouped_prompt_via_jinja`（`hunyuan` 另有分支）。早期版本把模板
字段存进 scorer 后**再没读过**，于是 9 个 grouped 实现里 `--jinja` 全是 no-op；
现在每个 grouped wrapper 都把 `self.inner.jinja` 传进去。
`qwen3` 走共享的 `build_jev_prompt`（`single.rs:461`）。

**未接入**：JEV 的 HTTP 端点（`/v1/jev/score`、`/v1/jev/grouped`）显式传
`Options::default()`，与 CLI 行为分开 —— 它们没有 chat-template 开关。

**`--jev --image --jinja` 必须走多模态 forward。** 带图时 `build_prompt` 不再
渲染文本，而是设 `pending` 并返回空 id，由 `forward_logits` 路由到
`run_qwen35_family_multimodal_logits`（占位符展开 + 模板都在那里）。早期版本
的 Jinja 分支先返回纯文本 id，导致 `pending` 没被设置、forward 走了文本路径，
**图片被静默忽略**。判定收敛到 `plan_prompt()`，`has_image` 优先于模板分支。

### Python 兼容层：方法回调，不是源码改写

真实模板会调 minijinja 没有的 Python 方法。minijinja 2.24 没有 `add_method`，
但有 `Environment::set_unknown_method_callback` —— 它把 receiver 作为**已求值的
Value** 交出来，所以不需要解析源码。

实现见 `src/prompt/jinja_compat.rs` 的 `python_method`：

| 方法 | minijinja 原生 | 处理 |
|---|---|---|
| `x.get(k)` / `x.get(k, default)` | `UnknownMethod`（map 没有 `get`） | 命中返回值；缺失返回真正的 `None` |
| `x.startswith(p)` / `x.endswith(p)` | 同上 | bool |
| `x.lstrip(s?)` / `x.rstrip(s?)` | 同上 | Python 的可选字符集语义 |
| `x.split(p?)` | 同上（内置 filter 是惰性序列，`[-1]` 会拿到**第一个**元素） | 真 list |
| `x.lower/upper/strip/replace/count` | 部分有 | 补齐；`replace` 需两个参数 |

**`is none` 语义是关键。** Jinja2 对缺失 key 返回 `None`，`x.get("k") is none`
为真；minijinja 的 `x["k"]` 是 Undefined，`is none` 为**假**。所以缺失时返回
`Value::from(())`（真 None）—— minijinja 的 `none` 是个 *test*，不是值，不能用。
`.get(k, default)` 只在 miss 时用 default，命中时即使值为 falsy 也返回原值。

未实现的方法照旧返回 `UnknownMethod`，不静默兜底 —— 真实缺口要暴露出来。

**为什么不用改写源码。** 之前 `0c953d3` 是改写源码（`x.get("k")` → `x["k"]|...`）。
所有版本都在某处出错：按字节切片会落在多字节字符中间（`你好{{ ... }}` 直接
panic）、receiver 提取吃掉 `for part in` 的空格、prose 里的 `dict.get("key")` 也被
改、链式调用产出括号不匹配的语法错误。回调让这些整类问题从根上消失 —— 源码原样
交给 minijinja。**当前只改写一处**：llama.cpp 的 `{% generation %}` 标签转成注释
（保留 trim 标记），因为那是标签而非表达式，没有可恢复的语义。

多模态 content part 带 `type`（image/video/audio），此前一律硬编码 `"image"`，
视频会与 `video_pad` 配不上。只要有 media 就一定是数组：早期版本会把单个 part
折回裸字符串，于是"只有图片、没有文字"的 turn 里图片被整个丢掉。占位符数量取
`rows`（投影 token 数 / width），三种媒体都适用 —— 取 vision grid 的话 Omni
音频（不产生 grid）会拿到空列表。

**BOS。** `add_special` 恒为 `false`，BOS 由模板负责。但模板有时压根不提 BOS
（Ministral3 的 `[INST]` 模板就是如此，靠 `add_bos_token=true` 拿 `<s>`）。所以
tokenizer 要求 `add_bos` 时，若结果开头不是 BOS 就补一个 —— 模板自己渲染过就不补。

### 其他边界

- **不做 tool calling。** 模板的 `{% if tools %}` 因为 `tools` 恒为空数组而
  自动走无工具路径。server 侧额外加了闸门：**带 tools 或带图片的请求不走
  Jinja**，直接回落到手写渲染器 —— 否则会静默丢掉 `attached_images`。
- CLI 仍只发单轮 user 消息；`add_generation_prompt` 与多轮渲染已实现，server
  多轮已在用。

## 源码索引

- `src/prompt/jinja.rs` —— 渲染器、`Options`、`MediaPart`、BOS 处理
- `src/prompt/jinja_compat.rs` —— `set_unknown_method_callback` 的 `python_method`，以及 `{% generation %}` 标签改写
- `src/prompt/legacy.rs` —— falcon_h1 / nemotron_h 在 Jinja 关闭时的手写 formatter
- `src/prompt/mod.rs` —— 手写 prompt builder
- `src/app/text/{qwen3,generation,multimodal}.rs` —— 文本 / REPL / 多模态接入点
- `src/app/jev/single/*.rs` + `src/app/jev/grouped.rs` —— 11/11 arch 的 JEV 接入
- `src/app/server/api/tools.rs` —— server 多轮接入（带 tools / 带图片时回落手写）
- `src/app/cli/{types,parse}.rs` —— `--jinja` / `--chat-template-file`
- `tests/jinja_chat_template_ab.rs` —— 真实 GGUF 的 A/B 比对（Qwen3 逐 token 一致；LFM2.5 记录**已知分歧**）
