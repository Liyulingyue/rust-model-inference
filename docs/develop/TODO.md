# TODO — RustModelInference Roadmap

This document merges the legacy `docs/TODO.md` (deep-dive format with
TODO-001…TODO-018) and the roadmap-style `docs/develop/TODO.md`
(checklist of upcoming work). The bottom half carries the detailed
investigation notes; the top half carries the at-a-glance priority list.

CPU SIMD capability assumed throughout: see [SIMD extension inventory](#simd-extension-inventory)
at the bottom of this file (AVX2 + FMA + F16C + AVX-VNNI on Intel Core Ultra 5 125H;
NEON on aarch64; no AVX-512).

---

## High Priority

- [x] **HTTP 文本运行时统一（TextRuntime）** — 已完成：`build_text_runtime(arch)` 是唯一
      arch 分发点，`TextInner` 四变体与 server 内手写 decode 循环全部删除；prompt 构造、greedy/
      温度采样、llama 家族的 rep-penalty+停止规则均单一来源（CLI/HTTP 共用）；
      `tests/cli_http_agreement.rs` 哨兵对 llama/qwen3/qwen35 三家族 greedy 输出逐字节一致
      （本地有权重时可复跑）。设计、验证矩阵与 8 条**有意冻结的剩余差异**（KV format、temp>0 RNG、
      多轮消息、CLI 诊断、请求字段 warn 等）见
      [`TEXT_RUNTIME_UNIFICATION.md`](TEXT_RUNTIME_UNIFICATION.md) §7。
- [ ] **HTTP `enable_thinking` 默认值与文档契约矛盾（thinking 模型默认被关掉）** —
      `src/app/server/api/protocol.rs:52-58` 的字段注释写的是「`None` 表示用模型训练默认值，
      thinking-tuned 变体通常为开启」，但 `src/app/server/api/tools.rs:130` 与 `:148` 的 jinja
      渲染分支都写成 `enable_thinking.unwrap_or(false)`。于是**客户端不传该字段**时，所有
      jinja 渲染的 thinking 模型都会被注入一个空的 `<think>\n\n</think>\n\n` 块；同一文件
      `:179` 的 lfm2 分支用的是 `unwrap_or(true)`，自相矛盾。
      引入于 `e20d2c1`（"add miniJinja"），影响 qwen3-thinking / k2-horizon /
      MiniCPM5-thinking / lfm25 / mellum 等所有 thinking 变体，**不是某一个模型的 bug**。
      实测（mellum-12b-a2.5b-thinking，greedy，`max_tokens=220`）：默认 `None` 时模型在空 think
      块后开始输出元评论并陷入重复，`content` 被 `think.rs` 的 `ThinkFilter` 全部缓冲后为空；
      显式传 `"enable_thinking": true` 同一 prompt 立刻正常返回 `2, 3, 5`（159 tokens，与 CLI
      逐 token 一致）。
      修复方向：不要在调用点硬编码 `unwrap_or`，改为按 arch 查「训练默认」，或让 `None` 保持
      `enable_thinking` **未定义**状态传给模板——GGUF 模板自身写的是
      `{%- if enable_thinking is defined and enable_thinking is false %}`，未定义即不注入空块，
      与 `unwrap_or(true)` 等价。**改动会翻转现有 thinking 模型的 HTTP 默认输出**，需先确认
      是否有依赖当前行为的调用方。
- [x] **lfm2moe 残余 1-ULP 贪心翻转** — ✅ **已定位根因并修复**（2026-10-15，LFM2.5-24B-A2B 适配期间）
      **先前的判断是错的**：它不是「greedy 下 top-2 logits 极接近时的 1-ULP 翻转」，而是一个
      **确定性的 shortconv 状态 bug**。完整 oracle 逐位对齐后，首个分歧 checkpoint 精确落在
      layer 0 的 `conv_conv`（step 8 = 首个 decode 步），其输入 `conv_in_proj` 逐位一致 ——
      即卷积**输入**没错，**状态窗口**错了。
      根因：`forward_shortconv` 只在 `!is_prefill` 时滑动 conv 状态。最后一个 prefill token 之后
      没有任何调用方重建状态，调用方的重建发生在**每个 prefill token 之前**，所以状态停留在
      `[bx_{t-2}, bx_{t-1}]`，缺少最后一步的 `bx_t`，第一个 decode 步的窗口整体错位一格。
      修复：状态滑动**每步都执行**（prefill 期间的滑动会被下一步的重建覆盖，无副作用）。
      顺带收益：decode 4.7 → **18.0 t/s**（3.8×，此前 decode 一直在用过期的窗口做无用功）。
      影响面：**所有 lfm2 / lfm2moe / lfm25 家族**的首次生成 token。
      证据：12 步 9 912 个 checkpoint 全部逐位一致，见
      [`tools/oracle/lfm2moe/README.md`](../../tools/oracle/lfm2moe/README.md)。
      守卫：`tests/lfm2moe_reference.rs`（真实权重 + 逐位 oracle）。
- [x] **`lfm2` / `lfm25` 的 trunk 选择靠 `general.basename` 判断——已删除** —
      原实现：`src/app/text/generation.rs` 用 `source.metadata("general.basename").contains("2.5")`
      决定走哪个 trunk。这是**来源/命名字段**而非架构字段：Liquid 的 QAD 量化管线写它，
      llama.cpp 的 `convert_hf_to_gguf.py` 不写。实测 15 个稠密 GGUF：只有 QAD 档带这个 key，
      其余八个档全部静默走 lfm2 trunk。
      llama.cpp 根本没有这个选择：`llama-arch.cpp` 只定义 `LLM_ARCH_LFM2` 与
      `LLM_ARCH_LFM2MOE`，所有稠密生成都走唯一的 `src/models/lfm2.cpp`，**无 2.5 分支**。
      **当时不会算错**：`dense_trunks_produce_identical_logits` 实测两个 trunk 在同一 GGUF 上
      输入同一批 token ids 后 logits 逐位相同（65536 项零差异），两份 `config.rs`
      只差一个 struct 名。但这恰好是“第二份 transformer forward 在任何一侧被修改时静默漂移”
      的典型场景，而路由之间只有一个可选元数据键在撑着。
      **已修（同一 commit）**：`src/models/lfm25/` 改为 lfm2 trunk 的**别名层**，1.5k 行重复实现删除；
      `generation.rs` 不再读 `general.basename`。JEV single/grouped 两套 scorer 保持可用（走别名）。
      已核对两个 `run_inference` 原来都调用 `build_lfm2_chat_prompt_with_thinking`，非 ChatML 的
      `build_lfm25_chat_prompt_with_thinking`。后者实为**不同**而非重复，且被 `tests/jinja_chat_template_ab.rs`
      锁定（记录它与 GGUF 自带 Jinja 模板的已知分歧），保留。
- [x] **lfm2moe 8B-A1B 逐位对齐 + shortconv 符号零累加顺序修复** — ✅ 已完成（2026-10-16）。
      在 24B 之外又验了 `LFM2.5-8B-A1B-Q4_0` 与 `Q4_K_M`，各 **7 840 个 checkpoint 零失配**（
      Q4_K_M 顶外覆盖了 MoE trunk 的 Q8_K/Q4_K 路径）。
      此体暴露了一个真实缺陷：`forward_shortconv` 的 `l_cache == 3` 展开分支写成
      `a0*w0 + a1*w1 + a2*w2`，把第一个乘积并进了初始值；而通用分支与 ggml 都是**从 0.0 开始逐项累加**。
      两者数值相同，但当三个乘积全为±0.0 时产生 **-0.0 与 +0.0 的符号位差异**
      （`0.0 + (-0.0) == +0.0`）。
      触发条件是某通道的 `b*x` 恰好为零，因此 24B 的 9 912 个 checkpoint 一次都没踩到，
      8B 却在 layer 0 就有 104 / 2048 个通道中中。修复后三个 checkpoint 全部零失配，24B 无回归。
      说明：需要真实 GGUF 才能触发，纯单元测试没有足够长观测，因此这里只记录现象与修复。
- [x] **把同款 oracle 对齐扩展到 lfm2 / lfm25 稠密档** — ✅ **已完成**（2026-10-16）。
      15 个稠密 LFM2 GGUF 全部与固定 oracle（llama.cpp `749f688`，`-n 24 --temp 0 -c 512`，
      `RMI_SCALAR=1`）的 greedy token ids **逐个一致**：
      `LFM2.5-230M` 的 QAD-Q4_0 / Q4_K_M / Q5_K_M / Q6_K / Q8_0 / F16 / BF16（7）、
      `LFM2.5-1.2B-Instruct` 的 QAD-Q4_0 / Q4_K_M / Q6_K / Q8_0（4）、
      `LFM2.5-2.6B` 的 QAD-Q4_0 / Q4_K_M / Q6_K / Q8_0（4）。
      另外 `LFM2.5-2.6B-Q4_K_M` 做了逐层逐位对比：35 步、**75 240 个 checkpoint 零失配**，
      包含每步完整的 128 000 项 `result_output`。
      稠密 trunk 新增 `src/models/lfm2/trunk/trace.rs`（24 个检查点，命名与 oracle 对齐）。
      工具链 `tests/lfm2_dense_trunk_agreement.rs`（验证两个 trunk 逐位一致 + greedy 对拍）。
      遗留：CLI 的增量 decode 路径已由 `tests/lfm2_incremental_decode.rs` 单独覆盖；`lfm2moe` 的 8B-A1B 逐位对齐已完成（Q4_0 与 Q4_K_M 各 7 840 个 checkpoint 零失配）。
      会课到的三个临时脚本陷阱均已写进 commit message（重新选取二进制、按
      mtime 选二进制、缺记录当作匹配）。
- [ ] **HTTP 多模态图片输入（OpenAI 兼容）** — `/v1/chat/completions` 现在**完全不支持图片**：
      `protocol.rs::text_content` 只接受 string / `type:"text"` block，带 `image_url` 的请求实测
      返回 400 `unsupported content block`。CLI 侧 `--mmproj --image` 能力完备（`app/text/
      multimodal.rs` + `inject_vision_embeddings`），HTTP 侧 `--mmproj` 只为 `/v1/jev/image*`
      决策打分端点加载，**没有通用带图对话端点**。
      对齐目标（已查证 `references/llama.cpp/tools/server/server-common.cpp:1060` 的 `handle_media`）
      接受四种形式：`http(s)://`（真下载，10MB 上限 + 10s 超时）、`file://`（llama.cpp 要求
      `--media-path` 白名单）、`data:image/...;base64,...`、裸 base64。OpenAI 官方 API 同样
      支持 URL 与 base64 两种，故 URL 不是可选装饰而是兼容性的一部分。
      建议实现（暂缓，等维护者启动）：
      1. `protocol.rs` 接三种 block：Chat `{"type":"image_url","image_url":{"url":...}}`、
         Responses `{"type":"input_image","image_url":"..."}`、Anthropic
         `{"type":"image","source":{"type":"base64","media_type":...,"data":...}}`（**裸 base64，
         字段名不同，别漏**）。
      2. base64 → `image::DynamicImage`：复用 TTS 的 data-URI 解析与 base64 报错模式
         （`server/mod.rs:447,522-540`）+ `media.rs:261` 的 `image::load_from_memory`（吃的就是
         `&[u8]`）。
      3. **`api.rs:164` 的 chat 路由 `DefaultBodyLimit` 是 4MB**，一张 3MB PNG base64 后必 413；
         需抬到 32~64MB（ASR 端点已是 64MB，`server/mod.rs:987`）。
      4. `RuntimeOptions` 加 `images` 字段，`Qwen3TextRuntime` / `Qwen35TextRuntime` 建 vision
         encoder 并在首步注入 embedding（复用 CLI 已验证路径）；`build_text_runtime` 已是为此
         留好的扩展点，server 结构无需再改。
      5. 哨兵扩一档：带图 prompt 的 CLI vs HTTP 对比。
      安全决策（建议）：远程 URL 需要新增 HTTP client 依赖（仓库现无 reqwest），且给推理服务开
       SSRF 面（内网探测 / 超时消耗）。建议加 `--allow-remote-images`，**默认关闭**（只允许
      data URI / 裸 base64 / 本地 `file://`），开启时带 10MB + 10s + 重定向上限护栏。
      `file://` 是否限制目录需明示：本服务与本机同用户（模型文件即本地路径），但应文档化该姿态。
- [ ] **Q2_K / Q3_K SIMD 加速** — 当前 scalar 5-9 t/s。仿 `vec_dot_q4k_q8k_avx2` 写 `_avx2` AVX2 kernel。
      预期 5-10× 加速，目标 30-50 t/s。详见 `docs/OPTIMIZATION.md` § "Quant Kernel 补全"。
- [ ] **IQ2_XS / IQ3_S / IQ2_S scalar forward_prequantized stub 修复** — 现状：`src/ops/kernel/iq4_xs.rs`
      的 `iq_kernel_impl!` macro 给 IQ2_XXS / IQ2_S / IQ3_XXS / IQ3_S / IQ1_M / IQ1_S 生成了
      Kernel trait impl，`forward_prepared` 正确调 `vec_dot_*_q8k` scalar（生产走 Q8K 路径无 panic）；
      但 `forward_prequantized` 是 stub（写 0）。`uses_q8_k()` 包含所有 IQ 类型 → 生产永远走
      `forward_prepared` → stub 永不触发，但接口完整性差。
      选项 1：把 macro 里的 stub 改成完整 dequant-to-f32 + dot 路径（与 IQ4_NL / Q4_K 同款），低成本。
      选项 2：直接写 SIMD kernel（需要 IQ2_XS / IQ3_S 的 GGUF 测试模型 + llama.cpp Oracle；当前
      `models/qwen3-0.6b-gguf/` 没有这些格式，按 `adapting-new-models` skill 暂无法做精度验证）。
- [ ] **AVX-VNNI int8 dot 加速** — 见 [TODO-AVX-VNNI](#todo-avx-vnni-int8-dot-加速) 详细说明。
      当前 Q8_0 × Q8_0 matmul 已用 `_mm256_maddubs_epi16` (AVX2)，但 AVX-VNNI 的
      `_mm256_dpbssd_epi32`（带 saturate 的三操作数 int8 dot）在
      K2-Horizon / Breeze 等 Q8_0 路径上可省一次 saturate pass。
- [x] ✅ **Vulkan SiLU 的 f32 次正规数缺陷** (2026-10-12) — `exp_approx` clamp 在 exp
      溢出点导致 `1/den` 落进次正规数、silu 修正项产出 NaN，Z-Image FFN 43% 激活损坏。
      **算子从未在生产规模验证过**（自检只覆盖 96 个元素，真实是 3800 万）。
      clamp 改 `±80` + 新增生产规模门禁。⚠️ **clamp 值不可调回 88.376**，且
      GPU SiLU 已与 CPU 不再逐位一致（PSNR 33.34 → 31.52 dB）。
      详见 [TODO-020](#todo-020-vulkan-silu-的-f32-次正规数缺陷--已修复但饱和边界不可回退)。
- [ ] **Q4_0 kernel 加 FMA + tiling** — 见 [TODO-001](#todo-001-q4_0-avx2-kernel-不使用-fma性能受限)。
- [ ] **token 维寄存器分块：消除 per-token 权重重读** — 详见 [TODO-019](#todo-019-token-维寄存器分块消除-per-token-权重重读)。
      文本编码只跑到约 11% 的 NEON 峰值，瓶颈是每个 token 都从 DRAM 重读整个权重矩阵。
      预期 1.5-2× 加速。

- [x] **LFM2.5-24B-A2B 适配 + 标量逐位对齐** — ✅ 已完成 (2026-10-15)。
      24B 总 / 2B 激活的 MoE，`lfm2moe` 架构：40 层 × 2048 宽 × 32 头（head_dim 64），注意力层
      `[2,6,…,38]`（每 4 层一个），前 2 层 dense SwiGLU（`feed_forward_length=11776`），其余 38 层
      MoE（64 选 4，`expert_feed_forward_length=1536`，sigmoid 路由），`l_cache=3` shortconv，
      `rope.freq_base=1e7`，ctx 131072，词表 128000，Q6_K `token_embd` + Q4_0 其余。
      **未新增模型代码** —— 现有 `src/models/lfm2moe/` 直接跑通真实权重（QAD-Q4_0，SHA-256
      `d1c03f73…f8a5`）。对固定 llama.cpp `749f688` 的标量 oracle：12 步（8 prompt + 4 greedy）
      token IDs 一致、**9 912 / 9 912 个 checkpoint 原始 F32 位完全相同（0 mismatch）**、
      `generated_ids` 一致。CLI 实测 `"What is 2+2?"` → `"2+2 equals 4."`。
      ⚠ 未覆盖：Q4_K_M/Q5_K_M/Q6_K/Q8_0/F16/BF16 档；SIMD 路径；批量 prefill；F16 KV；
      JEV / embedding 入口。
      [oracle 说明、复现命令与边界](../../tools/oracle/lfm2moe/README.md)。
- [ ] **oracle 构建脚本有一处静默陷阱，`GGML_NATIVE=OFF` 不等于关掉 SIMD 变体** — 本次对齐初期
      83% 的 checkpoint 全错，最根本的原因是 **oracle 侧**：`GGML_NATIVE=OFF` 只去掉
      `-march=native`，这一版 llama.cpp 的 `GGML_AVX2` / `GGML_FMA` / `GGML_F16C` 仍默认 `ON`，
      于是 `arch/x86/quants.c` 的 `#if defined(__AVX2__)` 分支被编译进去，Q4_0 点积用
      `_mm256_fmadd_ps` 累加，与参考实现差 ~2 ULP。**逐位对齐必须显式
      `-DGGML_AVX2=OFF -DGGML_FMA=OFF -DGGML_F16C=OFF -DGGML_CPU_ALL_VARIANTS=OFF`**，
      否则得到的是一个「看起来能跑」的 SIMD oracle。
      `tools/oracle/shared/build_llama_oracle.sh`（既有 mellum 等用的那份）**没有**这些开关，
      只有 `GGML_NATIVE=OFF` —— 已在本次的新脚本 `tools/oracle/lfm2moe/build_lfm2moe_oracle.sh`
      补齐，**但共享脚本尚未同步**，其中所有 arch 的既有「逐位对齐」结论都应复核一遍是否真的
      跑在标量路径上。⚠ 优先级取决于复核结果：若 mellum 等当时确实跑在 AVX2 上，它们的
      0-mismatch 结论就与本次的构造方式不一致。
- [ ] **标量 Q8_K 激活量化器与 ggml 契约不一致（已修，但调用面比想象的大）** — ✅ 已修（2026-10-15）：
      `quantize_row_q8_k_scalar_into` 用的是对称 `d = amax/127` + `f32::round()`，
      而 ggml 的 `quantize_row_q8_K_ref` 是 `iscale = -127/max`（`max` 是**带符号**的最大幅值元素）、
      `d = 1/iscale`、取整用 `nearest_int()`（12582912.0 技巧，**ties-to-even**），钳位是
      `min(127, v)`（只有上界）。负值主导的 block 因此符号相反、`d` 反号，Q6_K 的每个点积都差
      ~2 ULP。仓库里其实**已经**有一份正确实现 `quantize_row_q8_k_ggml_into`（ERNIE 在用），
      标量路径只是没跟上 —— 说明同一契约存在两份实现且会漂移。
      **待办：收敛成单一实现**，让 `quantize_row_q8_k_into` 的标量分支直接走 ggml 版本，
      删掉重复代码；否则下次改动仍可能只改一处。
      守卫：`ops::quant::q8k_padding_tests::scalar_quantizer_matches_the_signed_ggml_contract`。
- [ ] **MoE 权重归一化的累加精度** — ✅ 已修（2026-10-15）：归一化用的 sum 必须像
      `ggml_sum_rows` 一样在 **f64** 中累加，然后转 f32 再做 f32 除法。用 f32 累加会让专家权重
      差 2 ULP，在近平局处直接换专家。`src/models/lfm2moe/trunk/forward.rs` 已修；
      **待查** `mellum`、`qwen3vlmoe` 等其它 MoE trunk 是否也用 f32 累加。

- [x] **`tools/oracle/shared/build_llama_oracle.sh` 只有 `GGML_NATIVE=OFF`，并非标量 oracle ——已修，并已实测定量** —
      `GGML_NATIVE=OFF` 只是不传 `-march=native`，**并不改 ggml 自身的默认值**：该 pin 下
      `GGML_CPU_ALL_VARIANTS` 仍为 ON（会为每个 kernel 额外编译一份 AVX2 / AVX512 / FMA 副本并运行时分发），
      `GGML_AVX2` / `GGML_FMA` / `GGML_F16C` / `GGML_AVX512*` 也仍为 ON。
      **实测定量**（同一 llama.cpp commit、同一模型、同一 prompt，仅差那 10 个 ISA 开关，
      `LFM2.5-230M-Q4_K_M` 单线程）：
      - **3 468 个 checkpoint 中 3 445 个逐位不同（99.3%）**；
      - `result_output` 每步 65 536 个 logits 中 65 535 个不同，max|delta| 0.218（logit 量程 18.79，约 1.2%）；
      - **token ids 相同**（11 个 prompt + 6 个 greedy 完全一致）。
      即：**逐位结论失效，token 级结论通常存活**，但会在近平局处翻转。
      已在共享脚本补齐 `GGML_AVX/AVX2/FMA/F16C/AVX512*/AVX_VNNI/AMX=OFF` 与 `GGML_CPU_ALL_VARIANTS=OFF`。
      **影响面（需逐一复验）**：
      - `tests/inference_parity.rs` 的 `q4_0_matches_pinned_scalar_oracle_bit_for_bit` /
        `q4_k_m_matches_pinned_scalar_oracle_bit_for_bit` 用 `f32::from_bits` 原始位比较，
        且 `RMI_LLAMA_ORACLE` 指向的二进制若由共享脚本构建，则这两个结论需重跑（两者均 `#[ignore]`，需手动运行）。
      - MiniCPM5 `dump_tokens_oracle`、Falcon-H1“遵循共享脚本标志”构建。
      - **不受影响**：`inference_parity.rs` 里的 schema 测试（只校验 record 名/顺序/shape）、
        LFM2-MoE（自己的构建脚本本就带全部开关）。
      **同类脚本仍存在 10 个**：`mellum` / `qwen35` / `qwen_drive` / `gemma4`(两个）/ `ernie_image` /
      `nanbeige` / `longcat` / `jina_bert_v2` / `qwen_image_2_1`。它们同样只有 `GGML_NATIVE=OFF`。
      未批量修改的原因：部分归档在 **ARM** 上取得结论（那里 `-U__ARM_NEON` 已经在 CMAKE_C_FLAGS 里），
      x86 的陷阱不适用；而且改脚本会改变已有验证产物的构建条件。
      应当逐个 arch 确认其结论是否依赖浮点位级一致性，再决定是否重建 oracle。
      **复现**：用只带 `GGML_NATIVE=OFF` 的标志配置构建一份 oracle，与全标量那份对同一模型跑同一 prompt，
      再用 `tools/oracle/lfm2moe/check_scalar.py` 对比两份 manifest。

- [ ] **AVX2 kernel 与 ggml 参考实现不逐位一致（已定位到入口，未定位到具体 kernel）** —
      测量：15 个稠密 LFM2 GGUF，Rust 不设 `RMI_SCALAR`（走 AVX2）vs 标量 oracle，**12/15 greedy 一致**；
      `RMI_SCALAR=1` 下是 **15/15**。失配三个的 top-2 间距为 **6.6e-3 ~ 9.6e-3 相对**（比 f32 epsilon 高三四个量级），
      属真实算术差异而非近平局；其中 `LFM2.5-2.6B-Q4_K_M` **第 0 个生成 token 就分歧**。
      **入口已确认**：逐层 trace 显示 `LFM2.5-2.6B-Q4_K_M` step 0 下，`RMI_SCALAR=1` 时
      **396 个 checkpoint 全部零失配**，AVX2 时第一个失配就是 `conv_in_proj`（layer 0，Q4_K × Q8_K 矩阵乘，
      max|delta| 2.5e-07），之后沿层级放大。`attn_norm` 仍然匹配，说明 RMS norm 未受影响。
      **性质**：2.5e-07 是求和顺序不同的 ULP 级差异，不是算法不同。
      **注意**：仓库自带的 `ops::quant::avx2_parity` 单元测试（q4k/q6k AVX2 vs scalar）**全部通过**，
      但那些只用合成的小 block，覆盖不了真实多行 / 真实权重的累加顺序。
      **未办**：尚未定位到具体 kernel。下一步是用真实 GGUF 权重构一个
      针对 `conv_in_proj` 矩阵乘的 scalar/AVX2 对照测试（而不是合成 block），该测试应能直接拍住求和顺序。
      **机制已看懂，但未定位到具体 kernel**：`vec_dot_q4k_q8k_avx2` 每个 block 用
      `_mm256_fmadd_ps(vd, sumi, acc)` 累加（**FMA，单次舍入**），最后走 `hsum_ps` 树形归约；
      标量路径是逐 block `sumf += d * sumi`（**两次舍入**）。两者算的是同一个和，
      但添加顺序不同 —— 与 2.5e-07 的偏差尺度一致。
      **这是性能与精度的取舍，不是 bug**：`LFM2.5-2.6B-Q4_K_M` 同一 prompt 下
      AVX2 输出 `"2"`（25.1 t/s）、scalar 输出 `"2."`（2.0 t/s）——两者都合理，
      但**AVX2 快 12.5 倍**。要让 AVX2 与标量 oracle 逐位一致，就要拉掉 FMA 和树形归约，
      代价是吞吐掉一个数量级。**不建议改**；应该把「逐位对齐」明确限定在
      `RMI_SCALAR=1`，并在文档里写清“默认 AVX2 路径不在逐位契约内”。
      **影响范围**：当前所有逐位结论都只在标量下成立。真实用户在有 AVX2 的机器上跑的是
      这条路径，**与 oracle 存在 ULP 级差异**，并会在近平局处翻转为 token 级分歧。

### JEV 决策评分 follow-ups

`--jev` 已在 `src/app/jev/single.rs::run_jev_decision` 中按 `general.architecture`
自动路由到 10 个 per-arch scorer（Qwen3 / Qwen3.5 / Llama 家族 / Gemma4 /
LFM2 / LFM2.5 / Spark / Nemotron-H / Hunyuan / LFM2-MoE），每个 scorer 实现
`JevScorer` trait。Grouped 模式（MultiSelect + BlockChoice）在
`src/app/jev/grouped.rs::run_jev_grouped_decision` 通过 `JevGroupedScorer` trait。
通用协议与 per-arch chat template 见 [`docs/usage/qwen3.md` §10](../usage/qwen3.md) 与
[`docs/develop/jev.md`](jev.md) §14（源码索引）+ §15（trait 重构）。

- [ ] **JEV Multi-question KV 共享** — 当前多 question 模式（`--jev-question × N`）
      每个 question 都会新建 session 重新 prefill 一遍 context + question 文本，
      耗时 N × prefill_time。KV 共享方案：让 session 支持 `reset_kv(seq_len)`
      把 KV cache 截到 context 之后的长度，对每个 question 只 prefill
      `{"question": ..., "candidates": ...}` 部分复用同一份 context KV。
      预期 N 个 question 总耗时从 `N × t` 降到 `t + (N-1) × t'`，其中 `t'`
      是 question-only 部分。
      适用 trunk：qwen3 / qwen35 / gemma4 / spark（有 Session API）；
      llama / lfm2 / lfm25 / nemotron_h / hunyuan 需要先把 monolith
      拆出 session API。
- [ ] **JEV Shared prefix batching** — 多个 `--jev-question` 不仅共享 prefix，
      还可以把 K 个不同 question 的最后 token batch 成一次前向（拼接
      成 `[prefix; q1_suffix; q2_suffix; ...; qK_suffix]`，attention 阶段
      mask 让 K 个问题互不干扰）。当 K 个 question 文本相似度高时（典型
      客服路由、FAQ 分类场景），可以把 prefill 提速近 K×。
      需要先解决：不同 question 的 last-token 位置索引；padding 到统一
      长度；attention mask 的构造。当前架构（每 question 一个独立 session）
      完全不支持此模式，需要结构化改造。
- [ ] **JEV instruct-tuned GGUF 文档** — 现状：本地 `models/qwen3-0.6b-gguf/`
      只有 base model Qwen3-0.6B（IQ4_NL / Q4_0 / Q8_0 等量化），不是
      Instruct 变体。OpenJEV 锁定 Qwen3-4B-Instruct-2507。要让 JEV 输出
      真正准确的 label，需要：
      1. 找/下载 `Qwen3-0.6B-Instruct` 的 GGUF（huggingface 上有）
      2. 同 §1（用 base model + IQ4_NL 跑通作为 baseline）
      3. 加 `tests/qwen3_instruct_jev_reference.rs` 比对 base vs instruct
- [ ] **JEV 与生成模式共享 KV（暂留）** — 当 `--prompt` 后面接 `--jev` 时，
      可以复用生成模式 prefill 出来的 KV cache，避免重复编码同一段 context。
      当前架构下两者用不同的 session path，需要在 dispatch 入口统一
      KV lifecycle（`KvLifecycle::Shared`）。
- [ ] **JEV Cardinality 扩展（26 → 255）** — 当前 label 固定 A-Z（最多 26 候选）。
      TypeSafe Jev 官方支持最高 255 cardinality。扩展路径：
      1. 候选 ≤ 26 时继续用 A-Z label（现有逻辑不变）
      2. 候选 > 26 时自动切换到数字 label（1, 2, 3, ...），验证每个数字 token
         在 tokenizer 中是单 token（0-9 在所有 BPE tokenizer 中都是单 token；
         10-99 大部分也是；100-255 需要逐个验证）
      3. 更高基数时参考 TypeSafe 的 2-stage 方案：先 score 所有候选（并行），
         再 top-K choose（第二次 forward pass）
      注意：当前 `verify_label_tokens_single` 只校验 A-Z；扩展后需要校验
      实际使用的 label set（数字或字母）。
- [x] ✅ **JEV Qwen3.5 不走 `JevScorer` trait 的 follow-up** (2026-09-22,
      commit `53581f3`) — 9/9 trunks 现在统一走 trait，Qwen3.5 通过新增的
      `run_forward_logits_qwen35_with_batch` free function + Llama-style
      `JevScorer` / `JevGroupedScorer` impl 加入。`Qwen35Model<'a>` 不动，
      zero-copy 完全保留。JEV 单 + group smoke test 在 Qwen3.5-0.8B Q8_0
      真实模型上跑通（Paris = 0.9141、pair_1 = A 0.7158）。图像推理路径
      （`--image` + mmproj）确认不受影响（apple.png 描述正确）。

- [x] **MiniCPM-V 4.7 1B（`qwen35` 语言塔）多模态接入** — 已完成，`20c247d`…`1bd1165`。
      视觉塔逐位对齐：602,112 个 normalized pixel F32 与 pinned oracle
      `69f201a2` 完全一致；`conv_output_silu` 逐位相同（2158/8192 分量一致，L1 6e-5）；
      `pos_embed` 约 `1e-6` 相对差异，按 SIMD/FMA 标准接受。切片、CANVAS M-RoPE、
      tile separator 的 token-free embedding span 已接入；`<image_id>0</image_id>`
      由 mtmd 层注入。复现命令见 [`tools/oracle/minicpmv/README.md`](../../tools/oracle/minicpmv/README.md)。
- [x] **MiniCPM-V 4.7 35B-A3B（`qwen35moe`）语言塔** — 已完成，`032ff5c`…`385d6cf`。
      SSM 全链路对齐：输入 embedding / conv / `q_conv_predelta` / `k_conv_predelta` /
      `v_conv_predelta` / `beta_sigmoid` / `gate` 全部逐位一致或 ~1e-7；递推本身与
      oracle 的 chunked WY 输出在**同一输入、同一初始 state**（用 oracle 自带的
      `state_predelta` 锚定）下相差 avg 0.0063%。多模态端到端可用，`--jinja` 下
      512×512 出 overview + 2 slices 共 202 个视觉 token。
      ⚠️ 途中**两次误判**：先是 oracle eval callback 把多 ubatch 写进同一文件（后写覆盖先写），
      导致「最佳匹配 L1 923 vs 我方相邻 token 间 L1 中位数 956」——结论**不可证伪**；
      再是把「我方 token *i*」对「oracle row *i*」，而 llama.cpp 的 prefill ubatch
      首尾各带 3 个 padding row，3-token prompt 的 ubatch 是
      `[pad,pad,pad,t1,t2,t3,pad,pad,pad]`。据此曾误报「~14% 差异」，`385d6cf` 已撤回。
      **教训：oracle dump 必须先解决 ubatch 对齐与 padding，否则任何归因都是错的。**
      **2026-10-20 追到底**：修掉 `l2_norm` 把 eps 加在根号内（应为分母下限
      `1/max(sqrt(sum), eps)`，见 `ggml/src/ggml-cpu/ops.cpp:4333`；且 `sqrtf`
      会先把 f64 累加器收窄到 f32）后，layer 1 的 `conv_output_raw` / `q_conv` /
      `q_conv_predelta` / `k_conv_predelta` / `state_predelta` / `final_output`
      **全链逐位精确**，0-2 层 `l_out` 从 1e-5 降到 1e-9 量级。此前
      `qwen35_l2_norm_matches_pinned_llama_cpp_bits` 没能拦住，是因为它是从它要
      检查的 Rust 代码生成的（对旧公式 128/128、对 ggml 公式 0/128），已按 kernel 重生成。
- [x] **35B 的 MoE（routed + shared expert）数值对照** — 已完成，2026-10-20。
      `src/models/qwen35/trunk/moe.rs` 是**逐算子照抄** oracle 的公式
      （`LLM_FFN_PAR` 是 gate 模式而非残差；`routed = Σ score_e down_e(silu(gate_e·x)·up_e·x)`；
      `out = routed + sigmoid(gate_inp_shexp·x) · shared`；256 专家 softmax → top-8 → 归一），
      张量 shape 也对得上，输出质量正确——但这是**构造性论证 + 行为观察**，
      不是数值验证。早先「路由选中同一批专家（67,140,206）」的观测**不可信**：
      用了「首个精确匹配」做 token 映射，重复 token 全部塌到同一下标，且发生在搞懂 padding 之前。
      ⚠️ **本文原先写「oracle 侧 `ffn_moe_logits` / `ffn_moe_out` / `ffn_shexp_gated`
      都已能 dump」——这是错的**，属于此前乐观记录的残留。qwen35 的 oracle patch
      （`tools/oracle/qwen35/qwen35-llama-trace.patch`）白名单里**一个 MoE 节点都没有**，
      并且 `architecture != "qwen35"` 直接 throw，`qwen35moe` 连加载都做不到。
      那几个 `ffn_moe_*` dump 属于 **lfm2moe / mellum** 的 patch，是别的模型。
      即「oracle 从未为本架构产出过这些数字」。
      Rust 侧的 `RMI_MOE_DUMP` append 式 dumper（6 个文件、无 layer/step、无长度头，
      40 层 × N token 全部交错，无法归因）已删除，改为
      `src/models/qwen35/trunk/trace.rs` 的 `(name, layer, step)` 定点 checkpoint，
      对齐方式定为**两侧各跑单 token**（ubatch padding 只在多 token prefill 出现，
      单 token 下 `step == 0` 即同一个真 token，无需穿热路径插桩）。
      oracle 侧也已补齐：`tools/oracle/qwen35/qwen35moe-ffn-dump.patch` 让
      `llama-eval-callback` 接受 `qwen35moe`，产出 10 个 MoE 节点 + 逐层边界；
      `tools/oracle/qwen35/build_qwen35_oracle.sh` 从 tarball 构建并做 blob 校验、
      关掉全部 x86 ISA 变体以保证标量路径。35B Q8_0 权重已下载并校验
      （`MiniCPM-V-4_7-35B-A3B-Q8_0.gguf`，36,902,368,928 bytes，
      SHA256 `8543ee5b0f83152066f2fec58b8d9757d56bb7c4233b4fdd26ae364f0107d16b`）。
      **结果**：`ffn_moe_topk` 在 layer 0 / 3 / 39 逐位一致（id 与顺序都对上），
      layer 0 的 `ffn_moe_logits` 与 shared expert 全链亦逐位一致。
      节点清单与对齐规则见
      [`tools/oracle/qwen35/qwen35moe-ffn-dump.patch.README.md`](../../tools/oracle/qwen35/qwen35moe-ffn-dump.patch.README.md)。
      ⚠️ 教训：capture / selection / writing 是**三份独立名单**。「节点不在 manifest 里」
      在三份都查过之前什么都不能说明；`RMI_CB_TRACE=1` 会打印调度器实际问到的每个名字，
      一次运行即可分辨「从未提供」与「提供了但被拒」。
- [ ] **35B 视觉塔未与 oracle 做数值对照** — 1B 的 `pos_embed` 逐位比过，35B 完全没有。
      projector 代码零改动（两个尺寸共用同一 `minicpmv4_7` 实现，仅 `projection_dim`
      1024 → 2048），但 2048 维输出、`attn_post_norm` 全层 checkpoint 都没验证过。
- [x] **canvas 路径缺 `--jinja` 时报错信息误导** — 已修。
      根因确认为 `jinja.resolve()` 在未启用时返回 `None`，`text_conversation_tokens`
      因此返回 `None`，而旧文案说的是「投影器已实现但接线缺失」——那个缺口早已不存在
      （canvas 路径就是把 media span 拼到 checkpoint 自己模板产出的那一个
      `<|image_pad|>` 上）。同一个常量被 6 处以两种语义复用，已拆成两条：
      `MINICPMV47_CANVAS_ONLY_MSG`（anyres helper 根本处理不了这个家族）与
      `MINICPMV47_JINJA_MSG`（canvas 路径 + `--jinja` 关闭，指明 `--jinja` /
      `--chat-template-file`）。1B 一直需要 `--jinja`，不是新引入的限制。
- [ ] **`qwen35moe` 的 Vulkan 路径未接** — `src/vulkan/qwen35.rs:17` 仍 gate 在
      `architecture == "qwen35"`，35B 目前只能走标量/SIMD CPU 路径。

## 入口面覆盖（2026-10-19 审计）

MiniCPM-V 4.7 的 `qwen35moe`（35B-A3B）此前只在**生成面**注册，多模态对话可用、
但 JEV 打分全线不通——而它与 `qwen35` 共用同一个 `minicpmv4_7` projector 和同一个
`Qwen35Model`，只有 arch 字符串不同，属漏改而非有意限制：

| 入口 | `qwen35` | `qwen35moe` |
| --- | --- | --- |
| CLI `--image` / HTTP `/v1/chat/completions` | ✅ | ✅ |
| CLI `--jev --image` | ✅ | ✅（本次补） |
| HTTP `/v1/jev/image`(`/v1/jev/image_grouped`) | ✅ | ✅（本次补） |

改动点：`src/app/jev/single.rs` 的 `IMAGE_JEV_ARCHES` 与 dispatch、
`src/app/jev/grouped.rs` 的 `build_jev_token_ids_for_arch` 与 grouped dispatch、
`src/app/server/api.rs` 的 `run_multimodal_text_only`（`/v1/jev/image` 与
`image_grouped` 共用这一个闸门，故两侧同时生效）。
回归测试 `app::jev::single::image_arch_tests`。

顺带修掉两处会把排查带偏的文案：HTTP 侧原写「only supports qwen3/qwen3vl/qwen35」
却漏了实际支持的 `qwen3vlmoe`；生成侧原写「Only qwen35, qwen3vl, and edge0」
却漏了条件里本来就有的 `qwen35moe`。

- [x] **量化 mmproj 无法加载** — 已修。`src/models/minicpmv/weights.rs` 的
      `decode_matrix` 原本只认 F32/F16/BF16，其余一律
      `unsupported ggml type`，于是 Q8_0 / Q4_K / Q6_K 的 mmproj 直接加载失败，
      而 `ops::quant::dequantize_tensor` 恰好覆盖这三种。现在委托下去。
      投影器权重仍是 eager 的 `Vec<f32>`：它的 forward 已 pin 在 oracle trace 上，
      改成惰性的 `Weight` / `QuantizedTensor` 会动到已验证的数值路径，只放宽解码面。
      回归测试 `minicpmv::weights::decode_matrix_tests`（含 F32/F16 不变、
      不支持的类型仍报出类型名）。


## Medium Priority

- [ ] **讨论：MemoryArena 与 BlockAllocator 组合**
- [ ] **讨论：GPU 后端架构设计** — Vulkan / wgpu / CUDA 等多后端抽象
- [ ] **讨论：SIMD 扩展路线** — 当前 AVX2+FMA、NEON。后续可考虑 AVX-512 (高端 CPU)、ARM SVE、AVX-VNNI (int8 dot)
- [ ] **GLiNER 特性层补齐** — 12 个模型已 byte-exact，但用户可见**特性**未齐。
      详细清单、划界标准（模型家族特性 vs 框架层）、以及本轮抓到的三个真 bug
      见 [`GLINER_ADAPT_PLAN.md`](GLINER_ADAPT_PLAN.md) 的「特性层待办」一节。
      当前半成品：F-1 `choices` 解码侧（prefix 已进 prompt 流但无人消费，**优先收尾**）、
      F-2 `entity_attributes`（两套机制、零 ground truth）、
      F-3 长文本 `*_long` chunk+merge、F-4 relation 4 阶段 dedup。
- [ ] **讨论：两套线程调度统一** — ComputePool vs rayon。暂不统一（LLM 热路径不应轻易改动）
- [ ] **`kquants.py` 的 Q6_K 批大小估算偏低** — `--format q6_k` 量化 783 MB 的
      `gliner2.5-base-v1` 跑 15 分钟未完成（Q4_K 同输入 4.5 分钟）。`_Q6K_BYTES_PER_BLOCK`
      记的是 `16 × 19 × 8 × 3`（每个 256 元素 super-block 的 19-candidate scale 搜索），
      但 `_q6k_batch` 实际还要为整个 `(n, 256)` batch 分配 F32 输入、副本与打包中间量，
      远不止这个数，于是 `K_SEARCH_BUDGET_BYTES` 形同虚设、批大到吃满内存。修法是让
      `_Q6K_BYTES_PER_BLOCK` 覆盖真实的瞬时占用（或直接给 `_q6k_batch` 一个独立的
      批上限），Q4_K 侧同样值得复核。`q4_k` 与 `q8_0` 两条路径已验证，不受此影响。
- [ ] **Q8_0 与 Q8_K 量化路径按需量化（消除冗余计算，保留两份 buffer）** — dispatch 按 layer 权重格式，省一次量化 pass
- [ ] **Qwen3.5：借用权重与 FFN gate/up 输入量化复用的取舍** — 中期重构，不阻塞局部 FFN 优化
- [ ] **删掉三个纯转发的无参 `new()`** — `CollectSink`（`src/ops/generation_runtime.rs:158`）、
      `DacState`（`src/models/qwen3/tts/codec/dac.rs:115`）、
      `Lfm2MoeSampler`（`src/ops/sampling.rs:376`）都是
      `pub fn new() -> Self { Self::default() }`，而类型本身 `#[derive(Default)]`。
      同一个零值因此有两个名字，调用点又只用其中一个（`new()` 1/3/2 处，
      `default()` 0 处），`Default` derive 只是为了给 `new()` 转发而留。
      `Lfm2MoeSampler` 的动机可见：同族 `LlamaSampler::new(40, 0.95)` 必须带参，
      于是给无参版本也补了个 `new()` 让两行看起来一致 —— 代价是多一个名字。
      仓库分工应是 `new(args)` = 真构造函数（可带参、可 `Result`，如 `VulkanContext::new()`），
      `::default()` = derive-`Default` 类型的零值（`CliOptions` 17 处、
      `ExportOptions` 16 处均如此），两者不互换。已核实：三者均无
      `Default::default()` / `..Default::default()` 残留，也没有任何外层结构体
      依赖它们的 `Default`，故清理是每处 3 行（删 `new()` + 删 derive 里的
      `Default`）。
      反向教训见 `docs/develop/OPTIMIZATION.md:215` 记的 `KvFormat::default() = F32`
      （与 CLI 实际想要的 F16 相反）：**`Default` 的零值不成立时才真该用 `new()`**
      或干脆不给 `Default`。

### K-quant multi-row tile（vec_dot_q4k_q8k_avx2 / vec_dot_q6k_q8k_avx2）

目标：Q4_K_M 从 ~76 t/s → 120-150 t/s。详见 TODO-001 关联。

### BitNet `I2_S` GGML type 的 provenance 与上游兼容性

`GGMLType::I2_S = 36` 和 `general.file_type = 40` (LLAMA_FTYPE_MOSTLY_I2_S) **不是上游
ggml-org/llama.cpp 的一部分**——它们是 `microsoft/BitNet` 私有 fork 在 `ggml.h` 中加的扩展，
转换脚本 `utils/convert-hf-to-gguf-bitnet.py` 用 `gguf.GGMLQuantizationType.I2_S`。后果：

- 这两种 BitNet GGUF **只能**在 BitNet-patched llama.cpp / `microsoft/BitNet` build / 本仓库 (`I2_S` 已注册) 里
  跑，stock `llama.cpp`、`ollama`、`koboldcpp` 等社区 GGML 引擎会在加载时报
  "unknown tensor type" 直接 abort。
- 跨生态分发 BitNet 模型时需要在 README / GGUF 描述里明确这一点，避免用户误以为
  `ollama run bitnet-embedding-0.6b` 能跑（实际上 GGUF 类型会被拒）。
- 长期：上游 PR 把 `I2_S` 合并进 ggml-org/ggml（需要 bitnet.cpp 团队与 ggml 维护者协调），
  或者在 GGUF 规范里加一个 `quantization_version` 字段，让 stock llama.cpp 把
  `file_type=40 + GGMLType=36` 解释成 "需要 BitNet-patched build" 错误并清晰报错。

当前会话的处理：commit message + `src/core/tensor.rs::GGMLType::I2_S = 36` 旁的注释明确写了
"Microsoft BitNet b1.58 I2_S extension; not upstream ggml"。`docs/usage/bitnet_embedding.md` 第一节
"已下载并验证的 GGUF" 提示兼容性受限。**没有自动对齐上游的工作；等上游 merge 或社区分流后再调整。**

## Low Priority

- [ ] 更多量化格式支持（Q4_K, Q5_K 等）
- [ ] 完善 GGUfRS 导出功能
- [ ] GGUF 导出支持

---

## Detailed TODOs (merged from legacy docs/TODO.md)

### TODO-001: Q4_0 AVX2 kernel 不使用 FMA，性能受限

#### 现状

`src/ops/kernel/q4_0/avx2.rs` 的 Q4_0 matmul kernel 故意**不使用 FMA**：

```rust
//! 8. Multiply by d * scale in f32 with explicit `_mm256_mul_ps` + `_mm256_add_ps`
//!    (no `_mm256_fmadd_ps`) to match scalar's mul+add rounding exactly.
//!
//! **Precision contract**: bit-exact with the scalar implementation.
```

行累加采用 `acc += dc * d * scale` 三次 f32 运算（mul → mul → add），
而不是 `fmadd(d, d*scale, acc)` 一次融合运算。

#### 影响

- **性能**：相比 Q4_K / Q8_0 kernel（已用 FMA）慢约 1.5-2×。
  E4B Q4_0 文件实测 5.8 t/s decode，E2B Q4_K_M 10 t/s（同 embd 比例下 kernel 速度是主要差异）。
- **正确性**：与 scalar reference **bit-exact**（`assert_avx2_eq_scalar` 测试在 `q4_0/avx2.rs:241`）。
  所有 Q4_0 模型通过 parity 测试。

#### 何时触发

加载任何 Q4_0 权重时——目前主要是 Gemma 4 E4B Q4_0 导出文件。

#### 选项

1. **加 FMA + 放松精度契约**（推荐 follow-up）：把 `acc += dc * d * scale` 改成
   `acc = _mm256_fmadd_ps(prod_d_scale, ...)`。需要新增"near bit-exact"测试模式
   （参考 llama.cpp 的 GGML_FMA_DISABLED 编译开关做法）。
   - 预期：1.5-2× 加速
   - 风险：所有 Q4_0 模型需要重新跑 Oracle 验证（logits 可能漂移 ±1 ULP，最终 token 偶发分叉）
   - 工作量：半天
2. **重导出模型到 Q4_K / Q8_0**（推荐立即做）：
   - `q4_k/avx2.rs` 已用 FMA 写好 + parity 通过
   - `q8_0/avx2.rs` 已用 FMA 写好 + parity 通过
   - 用 llama.cpp 的 `convert.py` 把 E4B 转成 Q4_K_M 或 Q8_0
   - 文件大 ~30%，但推理快 2-3×
   - 工作量：1 小时
3. **写 FMA + 8-wide row unroll 版本** — 保留旧 kernel，新增 `matmul_q4_0_vs_q8_0_avx2_fma`，编译时 feature flag 切换。
   - 预期：2-3× 加速
   - 工作量：1 天

#### 推荐

短期：选选项 2（重导出 E4B），立即得 2-3× 加速，零代码风险。
中期：选选项 1（加 FMA kernel），其他 Q4_0 模型（Qwen3-0.6B-Q4_0 等）也受益。

#### 关联文件

- `src/ops/kernel/q4_0/avx2.rs` — kernel 实现 + parity 测试
- `src/ops/kernel/q4_0/scalar.rs` — scalar reference
- `src/ops/kernel/q4_0/mod.rs` — kernel 分发

---

### TODO-002: SIMD GeGLU `tanh_approx` 未启用

#### 现状

`src/models/gemma4/trunk/forward.rs` 的 `ggml_geglu_fp16_inplace` 仍然走 scalar 路径。
AVX2+F16C SIMD 版本原型写过又回退了，注释详细记录在函数体内。

#### 影响

- **性能**：scalar GELU 算 10240 元素 × 2 个 GELU/层（FFN + per-layer）× 35 层 = ~717k scalar ops/token。
  scalar ~12ns/element = **~8.6 ms/token**（当前 E2B ~80ms/token 的 ~10%）。
- **正确性**：scalar 版本 bit-exact 同 llama.cpp。所有 gemma4 测试通过。

#### 选项

1. **接受 ±1-2 ULP drift 启用 SIMD**（推荐）：Padé [7/6] tanh 近似（~0.04% max error in |x| ≤ 4.5）；
   加 `#[cfg(feature = "experimental-geglu-simd")]` 默认关；
   用 `cargo run --features experimental-geglu-simd` 启用。
   预期：+5-8% ETE。风险：gemma4_reference.rs 中 token Oracle 偶尔 ±1 分叉。
2. **Schraudolph fast exp + tanh = (exp(2x)-1)/(exp(2x)+1)**：5-10% tanh error on |x|>2 → 必分叉。不推荐。
3. **保持 scalar**：零风险，零加速。

#### 推荐

短期：选项 3。中期：跑 gemma4_reference.rs Oracle 确认 Padé [7/6] 漂移可控后，切到选项 1。

#### 关联文件

- `src/models/gemma4/trunk/forward.rs:858` — 当前 scalar + 详细 doc-comment

---

### TODO-003: 统一 5 个 `load_weight` / `load_weight_any` 到单一 `core::load_weight_any`

#### 现状

仓库目前每个模型自维护一份 GGML weight loader，签名 / 白名单 / dims 拆解 / 错误返回都不一致：

| 实现 | 签名 | 接受类型 | dims 校验 | 错误返回 |
|---|---|---|---|---|
| `dots/weights.rs::load_weight` | `(source, name, dims)` | F32/F16/BF16/Q8_0 | 严格 | `Result<_, String>` |
| `qwen35/trunk/weights.rs::load_weight` | `(source, name)` | 8 种 | 不校 | `Option<Weight>` |
| `qwen35/trunk/weights.rs::load_weight_f32` | `(source, name) → Vec<f32>` | F32/F16/BF16 | n_elements | `Option<Vec<f32>>` |
| `gemma4/trunk/weights.rs::load_weight` | `(source, name, dims, ggml_type)` | 单一类型 | 严格 dims+type | `Result<_, String>` |
| `gemma4/trunk/weights.rs::load_weight_any` | `(source, name, dims, &[GGMLType])` | 白名单 | 严格 dims+白名单 | `Result<_, String>` |

`qwen3` 干脆没有 `load_weight` —— 直接展开 `Weight::from_quantized(QuantizedTensor::from_bytes(...))`，
天然支持 `QuantizedTensor::from_bytes` 接受的全部 20+ 类型。

**核心公共部分**（≈ 5 行）：

```rust
let info = source.tensor_info(name)?;
let bytes = source.tensor_slice(name)?;
let weight = Weight::from_quantized(QuantizedTensor::from_bytes(
    bytes, info.ggml_type, n_in, n_out,
));
```

#### 影响

- **未来加 IQ2/IQ4/Q6_K/Q5_K 等新量化类型**——每个 `load_weight` 都要扩白名单。
- **dims 拆解不一致**：`dims.last()` vs `dims[0]/dims[1]` vs `dims[..-1]` 累乘
- **白名单粒度不一**：写死 vs 参数化 vs 无
- **错误返回风格不一**：`Result<_, String>` vs `Option<Weight>`

#### 选项

1. **最小**：扩 `dots::load_weight` 接受 Q4_0/Q4_1。维持命名遗留。
2. **中度**：breeze 摆脱 dots 命名，改直接展开 `Weight::from_quantized(...)`（学 qwen3）。
3. **完整**：5 个 `load_weight` 全部迁到 `crate::core::tensor::load_weight_any`：
   - 签名：`(source, name, dims: &[u64], allowed: Option<&[GGMLType]>) -> Result<Weight, String>`
   - `dims` 校验可选；`allowed=None` 表示 `QuantizedTensor::from_bytes` 接受全部类型
   - 所有模型调用方改 import；`qwen35::load_weight_f32` 与 `core::load_f32_tensor` 合并去重
   - 工作量：~200 行；彻底统一

#### 推荐

**方案 3**。`load_weight` 不是 dots 私有的，它是项目 GGML weight loader 的事实标准入口——
`qwen3` 已经证明**不抽象也活得很好**，所以抽象必须带来明显收益。

#### 关联文件

- `src/models/dots/weights.rs:5-67`、`qwen35/trunk/weights.rs:80-124`、`gemma4/trunk/weights.rs:226-309`
- `src/core/tensor.rs:288-330` — `load_f32_tensor`

---

### TODO-004: Breeze Q4_0 不可用 — 走 per-tensor 混合精度路径

#### TL;DR

**Q4_0 单层全量化对 Breeze 不可用**（128 frames vs BF16 59 frames = 117% 偏差）。
短期路线不是"再换一种 4-bit 格式"，而是 **per-tensor 混合精度**：embedding / lm_head / codebook 保留 BF16 或 Q8_0，
hidden-attn / mlp 主矩阵走 Q4_0 或 Q4_K。长期再评估 Q4_K / Q6_K 全替换。

#### 现状

实测推理结果：

| 精度 | 帧数（prompt "你好。"，seed 42） | 文件大小 | 备注 |
|---|---|---|---|
| BF16 | 29 | 6.6 GB | 与原字节等价 |
| F16 | 35 | 6.6 GB | 同源 |
| F32 | 29 | 13 GB | md5 与 BF16 bit-exact |
| Q8_0 | 37 | 3.6 GB | **3.6× 加速，推荐部署** |
| **Q4_0** | **128** | 2.1 GB | **输出退化** |

#### 关联文件

- `tools/converter/breeze/convert_breeze.py` — `_must_keep_source` / `_source_ggml_type`
- `src/ops/kernel/quantized_tensor.rs:275-...` — `QuantizedTensor::from_bytes`
- `models/Breeze-TTS-2-gguf/breeze-tts-2-Q4_0.wav` — 当前 Q4_0 输出（退化证据）

详见原 `docs/TODO.md` §TODO-004 完整分析（256 行），含 per-tensor 风险表、推荐方案、实施步骤。

---

### TODO-005: F32 x86_64 matmul 缺 SIMD kernel — ✅ 已完成

`src/ops/kernel/f32/avx2.rs` 实现完成，F32 / F16 NEON kernel 完成，AVX2 matmul macro 共享代码。
BF16 / F16 / F32 三档精度 matmul 共享 `avx2_matmul_packed!` 宏。

实测：Breeze `--quant f32` scalar 234s → AVX2 54s (4.3×)，`--quant f16` 47s → 31s (1.5×)。
F32 与 BF16 md5 bit-exact (a7c5e3f5...)，三档精度 md5 都已记录 baseline。

---

### TODO-006: Q8_0 Breeze output drifts ~1 ULP under SIMD activation/matmul

**v3 baseline `b051f3c1...` / 29 frames 已选定**。BF16 / F16 / F32 bit-exact 是硬约束；
Q8_0 没有 ground truth（量化本身有损），后续每次 SIMD 改动会再次漂移——只要 ≤ 1 ULP 就接受，记录新 baseline。

详见 `tools/converter/README.md:40` Q8_0 行。

---

### TODO-007: Breeze `rope()` SIMD 路径精度破坏 — ✅ 已修复

`src/ops/rope/neox.rs::rope_neox_inplace_with_table` 新增，BF16 round-trip 严格按 scalar op 顺序
（mul → round → mul → round → add → round），AVX2 内核在每个乘加后立即 bf16 round。
`src/models/breeze/transformer.rs::rope` 一行调用。Breeze 三档精度 (BF16/F16/F32/Q8_0) bit-exact。

剩余：NEON 版本（aarch64）暂未实现，不阻塞 Breeze 路径。

---

### TODO-008: Breeze F16 合成听感更合理 / F32-BF16 需对齐

人工听审发现：Breeze TTS 2 在 BF16 / F16 / F32 下比特级输出一致（BF16 ↔ F32 md5 `c19502ff...`），
但听感上 F16 更合理（F16 md5 `4dd19b6d...`）。

数学上 BF16 = F32（高 16 位），两者 bit-exact 等价是数学正确；F16 与它们差异是 mantissa 精度 + exponent bias 不同。
这是**听感 vs 数值精度**的张力：Breeze 上游训练 / 参考实现大概率用 BF16 模拟精度，
F16 路径数学上不对但听感更"干净"——需要在 BF16 / F32 路径上找到听感偏差的根因。

详见 `models/Breeze-TTS-2-gguf/README.md` Alignment 段落。

---

### TODO-009: VibeVoice F16/F32 kernel panic + F32 matmul placeholder — ✅ 已修复

`f16::forward_prepared` / `f32::forward_prepared` 加 `input_f32.len() >= n_in` fallback；
`f32::forward_prequantized` 真实实现替换 `row_dot_range` 占位符；`QuantizedTensor::F32(Vec<f32>)` 改为 struct variant
存 `n_in / n_out`。全 4 个 VibeVoice LLM 精度 (BF16/F16/F32/Q8_0) zh.wav 2 chunks 输出 bit-equivalent
"Speaker 0: 我认为跑步最重要的就是给我带来了身体健康。"

---

### TODO-010: Prefill runtime 通用化，避免每个模型重复实现 chunk 调度

#### 现状

`src/core/prefill.rs` 目前只提供 `prefill_chunks()` 和 batch size 校验。
Qwen3、Qwen3.5、Gemma4 仍分别维护自己的 chunk loop、CPU/Vulkan fallback、KV/state snapshot、commit/rollback 和 logits 输出处理。

#### 目标

参考 llama.cpp 的 `llama_batch_allocr`、`llama_memory_context_i` 和 `process_ubatch` 分层：

- 公共 `PrefillRunner` 负责输入校验、chunk 切分、`base_position`、输出顺序、失败重试、backend fallback 和 chunk 级事务；
- 公共 `PrefillContext` / `ChunkTxn` 统一 row shape、状态快照、commit/rollback；
- 模型只实现 `PrefillKernel::forward_chunk`，保留各自的 attention、SSM/conv、MoE 和多模态 position 语义；
- 不支持多行 batch 的模型声明 singleton fallback，不阻塞模型接入。

#### 实施顺序

1. 抽 `PrefillRunner` 包裹现有 Qwen3/Qwen3.5/Gemma4 loop，不改变数值行为。
2. 抽 `PrefillContext` / `ChunkTxn`，统一 KV/state commit、rollback 和 CPU/GPU fallback。
3. 模型接口收敛为 `PrefillKernel` + capability；无 batch 能力时自动走 batch=1。
4. 复用 shape 兼容的 scratch/graph，补充 batch=1、跨 chunk、失败回滚和真实 GGUF parity/benchmark。

#### 关联文件

- llama.cpp：`src/llama-batch.h`、`src/llama-memory.h`、`src/llama-context.cpp`、`src/llama-graph.h`
- 当前 Rust：`src/core/prefill.rs`、`src/ops/kernel/mod.rs::PreparedRows`、`src/models/qwen3/trunk/prefill.rs`、`src/models/qwen35/trunk/session.rs`、`src/models/gemma4/trunk/forward.rs`

---

### TODO-011: GLM-4 commit `65f8c8b` follow-up cleanups — ✅ 已完成 (2026-09-29)

来源：merge `origin/main` (62ab751) → `msi-new2` (daed1ec) 后评审 6 个非 PR commit 时记录。
GLM-4 (`arch="glm4"`) 在 llama trunk 上跑通（Paris、中文 ML 题本地 IQ4_NL 验证通过），
这些都是 cosmetic / dead-code。在 `msi-new3`（PR #120 合并后的新起点）上**全部处理完毕**。

1. ~~**删除 dead scratchpad 字段 `x_after_attn` / `x_after_mlp`**~~ ✅ 已删。
   原 `src/core/scratchpad.rs` 的 `Default::default` / `new_batched` +
   `src/models/qwen3/trunk/session.rs` 共 6 处分配，forward.rs 无任何读取。每
   batch row 多 ~8KB（n_embd × 2 × 4B）释放。
2. ~~**清理 `quantize_row_q8_k_scalar_into` 死代码**~~ ✅ 已清理，**并发现一个真 bug**：
   `max_val` 的累加器 + `sum_q` 都已丢弃（`let _ = max_val; let _ = sum_q;`），
   dead code 已删。**但清理过程中发现 `padded` buffer 尾部未清零** —
   GLM-4 n_ff=13696 = 54 blocks（53×256 + 1×128），最后 partial block 的
   `padded[128..256]` 保留前一个 full block 的脏值并参与量化，quantize 结果错。
   修复：`padded[len..].fill(0.0)`。由新增的 4 个单元测试验证（删掉 fix 后 2 个
   测试立即 FAILED，确认测试能抓住 bug）。
3. **`tools.rs` 让 glm4 + tools 显式 Err — 误报，无需修**。
   当前 `tools.rs` 的 `llama_family` 分支在 `if !tools.is_empty()` 时已经统一
   `return Err("Function tools are unsupported for llama-family architectures")`
   （由 PR `b018708` #112 "HTTP CLI 公共逻辑抽取" 引入），glm4 + tools 已被拒。
   **修 TODO 说明此条为误报**，不是真实 bug。
4. ~~**修复 `docs/MODEL_LIST.md` 孤行 `| `**~~ ✅ 已删。GLM-4 row 末尾原本
   缺失闭合 `）`，一并补上。
5. ~~**squash 掉 cix3 docs stash dance**~~ ✅ 已由 PR #120 的 squash merge 处理，
   main 历史已干净（无需单独 rebase）。
6. ~~**补 Q8_K zero-pad 单元测试**~~ ✅ 已加。
   `src/ops/quant/mod.rs::q8k_padding_tests` 4 个测试：
   - `partial_block_tail_roundtrips_to_zero`：GLM-4 n_ff=13696 partial block
     尾部 roundtrip 到 0
   - `partial_block_valid_values_roundtrip_within_quantum`：有效值误夽
     ≤ 半个 quantum（`d/2`）
   - `partial_block_tail_is_independent_of_previous_block`：直接验证字段 2 的
     脏 buffer bug — 删掉 fix 后 FAILED
   - `undersized_buffer_panics`：buf 长度合约 `len.div_ceil(QK_K)` 强制执行

关联文件：
- `src/core/scratchpad.rs` — 字段 1（已删）
- `src/models/qwen3/trunk/session.rs` — 字段 1（已删）
- `src/ops/quant/mod.rs:104-155` + `q8k_padding_tests` — 字段 2 + 6（已修 + 已测）
- `docs/MODEL_LIST.md` — 字段 4（已删孤行 + 补 `）`）

---

### TODO-012: minijinja-based user-GGUF chat_template support

#### 动机

当前 prompt builder 在 `src/prompt.rs` 和 `src/models/llama/trunk/forward.rs::llama_turn_text`
里 hardcoded 一组已知 arch 的模板（`lfm2`/`lfm2moe`/`qwen3`/`qwen35`/llama/nanbeige/
granite/glm4/minicpm5/phi3/exaone/k2-horizon）。当用户用 HF 微调了一个模型
（典型场景：微调 Qwen3-0.6B Instruct 改 chat_template），GGUF 里
`tokenizer.chat_template` 会变 — 我们的 hardcoded path 渲染错，模型输出
退化，但代码完全无感。要支持这个场景，必须把 chat template render 委托
给 GGUF 自带的 Jinja。

#### 设计：共存，不是替代

minijinja 与现有 hardcoded builder **共存**，通过 `build_prompt` dispatch
**选择**走哪条：

```
build_prompt() dispatch
 ├─ 优先级 1: 用户显式 chat_template override（HTTP field / CLI flag / env var）
 ├─ 优先级 2: 已知 arch fast path（hardcoded builder，零 runtime 开销）
 ├─ 优先级 3: GGUF 自带 chat_template（minijinja render）
 └─ 优先级 4: 兜底（raw prompt）
```

理由：
- 手写 builder 是 **fast path**：编译期已知、0 开销、当前所有 model list
  都 byte-aligned 过官方模板（PR #118 + da6bbcb + 后续实测验证）。
  **不删**。
- minijinja 是 **fallback / escape hatch**：用户传任意 HF GGUF / 微调模型
  / 自定义 jinja 时自动走通。**不动现有 fast path**。
- 两条 path 产物等价 — 都是 `Vec<u32>` tokens，下游 runtime 完全无感。
- 跨验证：用 minijinja render 一个等价 jinja（展开我们手写 builder 的逻辑），
  验证 byte-equal — 这给我们一个 regression net，防 hardcoded 漂移。

#### 触发入口（用户可控的 4 种）

| 入口 | 用例 |
|---|---|
| GGUF 自带 `tokenizer.chat_template` + 未知 arch | 用户上传任意 HF 模型，自动 render |
| GGUF 自带 `tokenizer.chat_template` + 已知 arch 但与 hardcoded 不一致 | 检测 diff，触发 jinja 兜底 |
| HTTP `/v1/chat/completions` 的 `chat_template` 字段 | 想精确控制的 OpenAI 兼容客户端 |
| CLI `--chat-template-file path.j2` 或 env `RMI_CHAT_TEMPLATE_FILE` | 本地测试 / 容器化部署 |

#### 实施 cost（粗算）

| 步骤 | LOC | 风险 |
|---|---|---|
| 加 `minijinja = "2"` 依赖 + json/macros features | 1 | 编译时间 +5s |
| 实现 `{% generation %}` block（minja 兼容） | ~50 | 中：custom parser extension via `unstable_machinery` |
| 实现 `raise` statement（minja 兼容） | ~30 | 低：minijinja 自带 syntax hook |
| `JinjaChatTemplate::new()` + cache | ~30 | 低 |
| `build_prompt` 集成 + 分发 | ~50 | 低（向后兼容 fallback） |
| HTTP `chat_template` 字段 | ~20 | 低 |
| CLI `--chat-template-file` flag + env var | ~30 | 低 |
| 测试（minja `tests_files/` 借用为 golden） | ~100 | 中（要选几个真实 GGUF 跑 byte-equal） |
| 跨验证（jinja-rendered == hardcoded for 已知 arch） | ~50 | 低 |
| **总计** | **~360 LOC + 1 dep** | **新功能 PR，不动现有 fast path** |

#### 验证策略

1. **已知 arch 不变**：跑现有 `tests/cli_http_agreement.rs` + `prompt.rs::tests`
   byte-equal 测试，确认 jinja 引入后 hardcoded path 行为零漂移。
2. **minja `tests_files/`**：借用其 ~80 个真实模型的 `tokenizer.chat_template`
   + 期望输出，做 golden test（不需要权重，只需 template + 合成 messages）。
3. **LFM2.5-8B-A1B 真实 GGUF**：用 `RUST_LFM2MOE_DEBUG_LOGITS` 验证 jinja render
   vs hardcoded render 输出的 token 序列一致。

#### ROI 时间线

- 现在：**0%** — 项目当前 model list 完备，plain text 全部对齐官方 jinja 输出
  （已 byte-equal 验证），用户微调场景没出现。
- 3-6 个月：**30%** — 如果项目推"通用 GGUF 推理"产品定位。
- 12+ 个月：**100%** — 任何 HF 推理产品最终都要支持任意 jinja 模板，这是行业
  标准做法（llama.cpp / vLLM / TGI 都已支持）。

#### 不立即实施的核心理由

1. **零增量价值**：当前所有 model list 的 plain text 输出已 byte-equal 官方 jinja。
2. **不解决"thinking-tuned 模型仍 emit think block"问题** — 这是模型训练特性，
   jinja 也救不了（已经验证）。
3. **不解决 byte-slice panic 等已修 bug**（这些是 hardcoded path 内部的 bug，
   跟 jinja 正交）。
4. **避免 scope creep — TODO-011 的 6 条死代码提醒我们**：不必要的代码会带来
   维护负担。jinja 是 ~360 LOC + 1 dep，没用户报"prompt 不对"前不值。

#### 触发条件（何时启用）

满足下列**任一**条件即应启动实施：

- 用户公开 issue / Discord 报告"我的微调模型 prompt 渲染错"（≥3 次报告后启动）
- 项目决定支持"用户上传任意 HF GGUF"产品定位（一次性启动）
- 仓库添加 ≥5 个新 arch 而其中 ≥2 个 GGUF 带非标准 chat_template（渐进启动）

#### 关联文件（实施时）

- `Cargo.toml` — 加 `minijinja = "2"` 依赖
- `src/prompt/jinja.rs` (新) — JinjaChatTemplate 封装
- `src/app/server/api/tools.rs::build_prompt` — 加 dispatch 分支（不删 fast path）
- `src/app/server/api/protocol.rs::Request` — 加 `chat_template` 字段
- `src/app/cli/parse.rs` — 加 `--chat-template-file` flag
- `src/prompt.rs` (现有) — 不动，保留 hardcoded builder 作为 fast path
- `docs/usage/*.md` — 加新章节解释"用户微调模型怎么用"

---

### TODO-013: CLI interactive mode — multi-turn history + slash commands

#### 现状

CLI interactive mode **已存在**（`src/app/text/generation.rs:218` 的
`run_interactive` 和 `:267` 的 `run_interactive_qwen35`），触发条件是
`options.prompt.is_empty()`（`src/main.rs:495` 的 else 分支）。REPL
loop 完整，从 stdin 读行、Ctrl+C / EOF 退出。

但**核心功能缺失**：

| 功能 | 状态 |
|---|---|
| REPL loop（stdin → stdout） | ✅ |
| Ctrl+C / EOF 退出 | ✅ |
| 每次 input 独立生成 reply | ✅ |
| **multi-turn history** | ❌ 每行独立，模型看不到前文 |
| **system turn（CLI 配置）** | ❌ 无 CLI 接口；模型不知道角色 |
| **`/clear` / `/system "..."` / `/exit` 命令** | ❌ |
| **streaming token-by-token 输出** | ❌ 整段生成完才 print |
| **`--thinking` / `--no-thinking` 生效** | ❌ `run_inference` 在 interactive 路径 hardcode `thinking=false` |
| **KV cache 增量（不重 prefill）** | ❌ 每轮从 0 开始 |

**最大痛点**：
```bash
> Hi, I'm Alice.
Hello Alice!

> What is my name?
I don't know your name.  # ← 不记得 "Alice"
```

根因：`lfm2::run_inference` 和 `lfm25::run_inference` 都签名是
`(..., prompt: &str, ...)`，内部 `build_lfm2_chat_prompt_with_thinking`
的 `&[Lfm2Message]` 被 hardcode 为 `[Lfm2Message { role: "user",
content: prompt }]`，没有 history / system 概念。

HTTP 路径**已经完整**（PR #118 + `da6bbcb` + 后续）—— 缺的是 CLI。

#### 实施 cost（粗算）

| 步骤 | LOC | 风险 |
|---|---|---|
| `lfm2::run_inference` 加 `turns: &[Lfm2Message]` 替代 `prompt: &str` | ~15 | 低（纯重命名 + 调整） |
| `lfm25::run_inference` 同上 | ~15 | 低 |
| `run_interactive` 重写：维护 `Vec<(role, content)>` history + system + slash commands | ~80 | 低 |
| `run_interactive_qwen35` 同上（multimodal path） | ~80 | 低 |
| Streaming token 输出（可选） | ~50 | 中（要 token sink 接口） |
| 思考 flag 透传 | ~5 | 低 |
| 测试（stdin mock + 历史一致性） | ~80 | 中 |
| **总计** | **~325 LOC** | **触及多个 model forward 入口；可能影响 Oracle 测试 baseline** |

#### 为什么不立即实施

1. **interactive mode 是 partial feature，不是 bug**。当前 1-shot-per-line
   满足 smoke test / benchmark / debug 三大主要 CLI 场景。
2. **HTTP 路径已完整**（multi-turn, system, thinking, streaming）。
   真正需要 multi-turn 对话的用户走 HTTP。
3. **改 `lfm2::run_inference` 签名**会触动 CLI / benchmark / Oracle
   test 的多个调用点，需要回归测试全套 llm 模型的 greedy baseline。
4. **ROI 偏低**：LFM2.5-8B-A1B 等思考模型在 CLI 上调试时确实希望有 history，
   但生产场景主要是 HTTP API 调用方。

#### 触发条件

满足下列**任一**条件即应启动：

- 用户公开 issue / Discord 报告"interactive mode 不支持 multi-turn"（≥3 次）
- LFM2.5-Thinking 系列在 CLI 上成为常见调试 / 微调工作流
- 决定给仓库加一个"开发者本地 chat loop"功能，作为测试 / 微调工具

#### 关联文件（实施时）

- `src/models/lfm2/trunk/forward.rs` — `run_inference` 接受 turns
- `src/models/lfm25/trunk/forward.rs` — 同上
- `src/app/text/generation.rs` — `run_interactive` 重写 + 加 streaming 接口
- `src/app/cli/parse.rs` — 加 `--system` CLI flag
- `src/main.rs` — interactive dispatch 把 `options.thinking` 透传
- `tests/cli_history.rs`（新）— 多轮 history 一致性测试

---

### TODO-014: Falcon-H1 Q4_K_M 加载 panic — ✅ 已修复 (2026-09-29)

#### 现象

`unsloth/Falcon-H1-1.5B-Instruct-GGUF` 的 **Q4_K_M**（以及任何把 `attn_v` /
`ffn_down` 打成 Q6_K 的量化：Q5_K_M / Q6_K / UD-* 等）在 prefill 阶段 panic：

```
thread 'main' panicked at src/ops/kernel/q6_k.rs:79
range start index 1290240 out of range for slice of length 430080
```

`430080 = 256 rows × 1680 B` 正是 `blk.0.attn_v.weight`（dims `[2048, 256]` Q6_K）
的真实体积，而 kernel 却按 `n_out = 1024` 切分行索引（`out_idx = 768` × 4 线程）。
Q8_0 模型不受影响，因为 unsloth 在该量化下把所有 attn 张量都打成 Q8_0。

#### 根因

Falcon-H1 的 attention 是 **GQA**，三个维度的 head 数不同：

| 张量 | GGUF dims | 语义 | 代码里当时的值 |
|---|---|---|---|
| `attn_q.weight` | [2048, **1024**] | `n_head * head_dim_k` (8×128) | 1024 ✓ |
| `attn_k.weight` | [2048, **256**] | `n_head_kv * head_dim_k` (2×128) | 256 ✓ |
| `attn_v.weight` | [2048, **256**] | `n_head_kv * head_dim_v` (2×128) | **1024 ✗** |
| `attn_output.weight` | [**1024**, 2048] | `n_head * head_dim_v` (concat 后) | **256 ✗** |

`forward.rs` 当时把 `n_attn_v` 定义成 `n_head * head_dim_v`（=1024，
"concat 输出"维度），并用它同时驱动：

1. V matmul 的 `n_out`（应为 256 → 越界 panic），
2. V KV-cache / f16 cache 的行 stride（应为 256），
3. `w_o` matmul 的 `n_in`（应为 1024）。

`weights.rs::load_weight` 用的是 `n_head_kv * n_embd_head_v`（=256，正确），
所以维度校验通过、加载不报错，直到 forward 时 kernel 才越界。

#### 修复

`src/models/falcon_h1/trunk/forward.rs` 把 `n_attn_v` 拆成两个名字：

- `n_attn_v_kv = n_head_kv * head_dim_v` — V projection 输出、V cache、
  `v_f16_storage` stride、`v_col` 索引；
- `n_attn_o = n_head * head_dim_v` — `attn_out` 缓冲区、`w_o` matmul 的 `n_in`。

`n_attn_q` / `n_attn_kv` 不变。`FalconH1Scratch::new` 的 `v` / `v_buf` /
`attn_out` 分配分别改用 `n_attn_v_kv` 与 `n_attn_o`。

#### 验证

本地 `Falcon-H1-1.5B-Instruct-Q4_K_M.gguf`，8 线程，`--max-context 512`，greedy：

| prompt | 输出 |
|---|---|
| `What is the capital of France? Answer with just the city name.` | `Paris` |
| `What is 2+2? Answer with just the number.` | `4` |
| `Name one color in the rainbow.` | `Red` |
| `用中文回答：什么是机器学习？` | 中文 ML 解释（流畅，未截断） |

速度 26–28 t/s prompt、24–25 t/s generation。lib 测试 934 passed /
19 failed（= 之前修完 3 条 baseline 后的水平，无新增）。

#### 遗留 / 后续

- **未做** bitwise llama.cpp oracle 对比（Q4_K_M 精度基线）。`tests/falcon_h1_q8.rs`
  目前只 gate 在 Q8_0 模型上；Q4_K_M 若要纳入 sentinel，需要新增
  `RMI_FALCON_H1_Q4K_M_MODEL` env 并记录 golden。
- Q4_K_M 的 `ffn_down` 也是 Q6_K，已随本次修复一同跑通。
- **3B 变体已实测**（2026-09-29，`Falcon-H1-3B-Instruct-Q4_K_M.gguf`）：config
   `n_embd=2560 / n_head=10 / n_head_kv=2 / head_dim=128` → `group_size=5`
   （1.5B 是 4），`attn_v`/`ffn_down` 同为 Q6_K。同一修复代码零改动跑通，
   Paris / 4 / red / 中文 ML 四例全对，13-14 t/s @ 8 线程。

---

### TODO-015: arch 接入的"公共层报到"问题 — 设计记录，暂不实施

#### 起因

`Falcon-H1` 适配 CLI 后用 `/v1/chat/completions` 失败：

```
Architecture "falcon-h1" is not supported by the server text endpoints
```

排查发现一个 arch 要为**三个公共入口各报到一次**，且漏了不报错：

| 入口 | 位置 | falcon 状态 |
|---|---|---|
| CLI | `src/app/text/generation.rs` `} else if arch ==` | ✅ 已接 |
| HTTP runtime | `src/app/text/runtime.rs::build_text_runtime` | ❌ 无 adapter |
| HTTP prompt | `src/app/server/api/tools.rs::is_qwen35` / `llama_family` | ❌ 无分支 |
| JEV | `src/app/jev/single.rs` `match &*arch` | ❌ 落进 `other => Err` |

维护者的比喻：**公共层是"插口/门牌"，用户只面对任务（传文本/图片 → 拿结果），不关心里面是谁**。而当前 registry 全是 `match arch`——**按施工队组织，不按门牌组织**，导致：

1. 新 arch 不知道该去哪几处报到（falcon 漏了 2 处，且静默）；
2. 不支持某能力的 arch（如 TTS）要"证明自己不该在 JEV 里"，需要维护
   "故意不支持"白名单；
3. 同一个能力在三个入口重复出现。

#### 三层拆解（讨论结论）

| 层 | 现状 | 是否要动 |
|---|---|---|
| **执行接口** | `TextRuntime` / `JevScorer` trait 已存在且健康（JEV 9/9 trunks 统一走 trait，TODO-014 之后仍成立） | 不用动 |
| **公共 API** | HTTP 的 `/v1/chat/completions` / `/v1/audio/speech` 本来就是按任务分、不关心背后 arch | 已经对了 |
| **登记处** | 3 个手写 `match arch`，新 arch 报到 3 次、漏了不报错 | **唯一真痛点** |

关键认知：用户问"falcon 为什么 HTTP 不能用"，答案是 **adapter 没写**（工作量问题），
不是"架构缺插口"——接口在，只是没插。

#### 评估过的方案

**A. distributed slice（`inventory` crate）+ capability 门牌**

```rust
pub mod capability {
    #[distributed_slice] pub static TEXT_GENERATE: [CapabilityFactory];
    #[distributed_slice] pub static JEV: [CapabilityFactory];
    #[distributed_slice] pub static TTS: [CapabilityFactory];
    // ...
}
// trunk 内自己挂门；不写就是不支持 → TTS 天然不进 JEV，无需白名单
// 公共层写完永久冻结
```

- 语义最干净："不声明 = 不支持"，新 arch 零公共层改动。
- **否决（现在）**：为"省 30 行 registry"引入 linker-magic 依赖不值；且
  I/O 组合仍在快速扩张（chat: 纯文本 → +图 → +音频/视频；omni 会把
  chat/score/embed 的边界糊掉），**预定义 capability slot 是在流沙上钉钉子**——
  真到 omni 那天，是 I/O 抽象重塑 capability，不是反过来。

**B. Plugin trait + 单注册表（~500 LOC 重构）**
- 否决：公共层仍要每 arch 加一行；且 TTS 不支持 JEV 要靠 `default fn` 或按
  capability 拆 4 个 trait，比 A 更脏。

**C. loader arch 列表提成 `SUPPORTED_ARCHES` const + coverage 测试（~80 LOC）**
- **设计已定稿，暂不实施**。见下。

#### TODO-015 的落地设计（将来启用时直接用）

1. `src/core/loader.rs`：把 `model_config_from_source` 里的 `matches!(...)`
   改成 `pub const SUPPORTED_ARCHES: &[&str]` + `contains`。新 arch 必须先加进
   const → 测试自动覆盖它。
2. `tests/arch_coverage.rs`（或 lib test `arch_coverage`）：遍历 `SUPPORTED_ARCHES`，
   断言每个 arch 在 CLI / HTTP / JEV 三个 dispatch 都有**显式**答案
   （`Supported` 或在带原因注释的"故意不支持"白名单里）。
3. CI 加**独立 job**（不能并进 `cargo test --lib`——那 19 个 pre-existing
   failures 会把它染红）：
   ```yaml
   arch-coverage:
     steps:
       - run: cargo test --profile release-fast --lib arch_coverage
   ```

效果：falcon 这类"静默漏注册"以后 CI 立刻红；本地
`cargo test --lib arch_coverage` 秒级回报。零新依赖、零架构改动。

#### 触发条件（满足任一才启动实施）

- arch 数量 > 25，或
- 新 arch 接入频率 > 每月 1 个（"报到 3 次"成为真瓶颈），或
- omni 模型（原生统一多模态 I/O）落地——届时 I/O 抽象自然重塑 capability 定义，
  **那时做 A 才是顺势而为**，今天钉死的 slot 只会碍事。

在触发前，处理方式是：**新 arch 接入时手工检查三处 dispatch + 补实现**
（正如本次给 falcon 补 HTTP/JEV）。

---

### TODO-016: Jina v5 Omni audio encoder — oracle 验证待补

本仓库 `encode_audio()` 已按 jina-ai `feat-v5-omni` llama.cpp fork 设计切 30 s Whisper 块（见 commit `9717d17`），但**未跑过 llama.cpp 端的逐位对照**：

1. **per-chunk byte-equal 对齐 llama.cpp `b96806d`**
   - 现存 `tests/jina_audio_projection_matches_llama_cpp_bits`（#[ignore]）已经覆盖单 30 s 块的 750 × 1024 投影 F32 对照，但每次跑需要：
     - `LLAMA_DIR` 是固定 commit `b96806d96061049a5b574269b049bf6241d63d46` 的独立副本
     - 应用 `tools/oracle/jina_audio/mtmd-audio-projection.patch` + `tools/oracle/qwen35/qwen35-scalar-softmax.patch`
     - `cmake -B build-rmi-jina-audio -DGGML_ACCELERATE=OFF -DGGML_METAL=OFF -DCMAKE_CXX_FLAGS=-DRMI_QWEN35_SCALAR_SOFTMAX`
     - 跑 `llama-mtmd-cli` dump oracle F32，再跑 `cargo test ... -- --ignored`
   - macOS ARM CPU / 单线程 / 标量 softmax / 关 Flash Attention 的严格 CPU 路径
   - 触发条件：新增 jina v5 audio encoder 任何改动时必跑

2. **跨块拼接 oracle 对齐**
   - 上一步只覆盖单个 30 s 块。我们的 `encode_audio()` 把多个 30 s 块的 post-conv tokens 拼给 LLM，让 LLM cross-attend 跨块融合
   - 上游 `feat-v5-omni` fork 也是同样设计（mtmd split 30s + LLM cross-attend），所以两边的最终 LLM embedding 应等价
   - 当前只在仓库端跑过 30s / 60s / 90s 端到端语义验证（cos 关系正确），**未跟 llama.cpp 跨块拼接对比**
   - 验证方法：取 jina-omni + llama.cpp fork + 一段 >30s 音频（e.g. 60s），分别跑两端的最终 embedding，断言 cos ≥ 0.99 + max abs diff 在量化噪声内（Q8_0 文本 + F16 mmproj 路径下应该是 ~1e-3）
   - 触发条件：commit `9717d17` 之后任何 jina v5 改动、Q4_K / Q5_K / Q6_K 等不同量化、新 mmproj 版本

**2026-10-01 per-chunk oracle 实跑结果**（commit `fe4b99c` 之前 / 之后）：
- **mel extraction**：✅ 与 llama.cpp bitwise-aligned（max abs diff 0.378 主要来自帧 2999 末尾 40 个 reflect-vs-zero 反射差异，符合预期）
- **位置嵌入**：❌→✅ commit `1201166` 修了 `token % self.config.window` bug，固定到 `token`（1500 个唯一 pos）。见 commit message
- **conv1d 输出**：✅ **Rust 端正确**——numpy 参考实现和 Rust `conv2_only` 在 ULP 级吻合（max abs 3.4e-3，AVX2 vs scalar 重排差），意味着 Rust 的 `conv1d_same_f16` patch 构造 + `dot_f16_f16_bytes` 实现都没问题
- **oracle dump 不可信**：之前以为是 Rust conv1d 有大 bug（cos=0.18 with oracle），但 2026-10-01 bisect 时发现 llama.cpp 的 `inp_raw` dump 两次值不一样（v0[0]=4.92, v1[0]=1.35），且 dump 出的值和我们 mel 也对不上——说明 ggml 的 input tensor 在 `ggml_set_input` 之后 buffer 被 allocator 复用了，`ggml_backend_tensor_get` 读到的是被覆盖的脏数据。同理 `after_conv1d` dump 也不可信
- **结论**：当前 oracle 对比无法作为 ground truth。Rust 端的 correctness 由 numpy 参考独立验证了。TODO-016 暂时挂起，等 oracle dump 机制修好后重新跑

**当前不做**：
- 触发条件没到：jina v5 audio encoder 改动少（一次 30s 切块重构 + 一次位置嵌入修复），且端到端语义验证已经覆盖了跨块拼接的功能正确性
- 仓库的 cargo test 已经能在没有 llama.cpp 副本的 CI 上跑出 934/19/67 baseline，oracle 失败不会阻塞
- **触发再做**：
  1. llama.cpp dump 机制修了之后（用 `ggml_backend_tensor_get_and_copy` 或者 `ggml_set_input` 后立即读），重新跑 per-chunk 对齐
  2. 用户加新的 jina-omni 量化、或者上游 llama.cpp fork 有新 audio 相关改动需要重新对齐

调试现场保留在 `/tmp/audio-fix-test/`（`our-trace-conv1.jsonl.*`、numpy 参考脚本 `conv2_only=0.054`/`-0.008` 验算过程）和 `/tmp/rmi-jina-audio-*-oracle*.f32`。
下次继续走：先修 oracle dump（直接读 `t->data` 改用 `ggml_backend_tensor_copy` 或者改在 `set_input_f32` 之后立刻 dump），再用可信的 oracle 验证 Rust 端对齐。

**实施步骤**（任一触发条件满足时）：
1. `cd $LLAMA_DIR && git checkout b96806d` 起固定 commit
2. apply 现有两个 patch
3. cmake + build（macOS ARM / Linux x86_64 各一次）
4. 跑现有 `jina_audio_projection_matches_llama_cpp_bits` 对 30s 单块
5. 新增 `jina_audio_projection_matches_llama_cpp_bits_concat`（#[ignore]）跑 60s 跨块拼接
6. 把对照数据 commit 到 `tools/oracle/jina_audio/`（或外置 datum）

### TODO-017: BERT 家族 + EmbeddingGemma encoder — llama.cpp 位级 oracle 未做

`src/models/bert_family/`（bert / jina-bert-v2 / jina-bert-v3 / nomic-bert / nomic-bert-moe）
与 `src/models/gemma_embedding/` 共 7 个 arch，全部**只做了语义验证，没做 llama.cpp 位级 oracle**。

每个测试文件自己都写明了这一点，例如 `tests/jina_v2_base_en.rs`：

> Oracle: local read-only `references/llama.cpp/src/models/bert.cpp`.
> bit-level parity with llama.cpp requires the oracle binary and is tracked separately.

而 `tools/oracle/` 下**没有** `bert/` 也没有 `gemma_embedding/` 目录——对比 gliner2 / clm / laya /
jina_audio / qwen3_tts 都有完整的 `dump_*.py` + `fixtures/` + `compare.py`。

| arch | 上游源文件（本地 `references/llama.cpp` 存在） | 现有验证 |
|---|---|---|
| `bert` | `src/models/bert.cpp` | 语义 + 元数据 + 张量清单 |
| `jina-bert-v2` | `src/models/bert.cpp` + `jina-bert-v2.cpp` | 同上 + WPM pinned ids + 语义排序 |
| `nomic-bert` | `src/models/bert.cpp` | 同上 |
| `nomic-bert-moe` | `src/models/bert.cpp` + `nomic-bert-moe.cpp` | 同上 + MoE 张量清单 |
| `bge-m3` | `src/models/bert.cpp` | 同上 |
| `gemma-embedding` | `src/models/gemma-embedding.cpp` | 同上 + 语义排序 |

已验证的部分（2026-09-30 对 `jina-embeddings-v2-base-en` 复核）：4/4 集成测试通过、CLI 与 HTTP
输出 768/768 元素 `%.9f` 文本完全一致、线程数 1/4/8/0 输出确定、L2 范数 1.000000、语义排序正确。
**这些都不等价于"与 llama.cpp 逐位一致"。**

#### 为什么值得做

这个区域已经出过两个只有 oracle 能早发现的真 bug，都是 port 完成后靠别暴露的：

1. **`attn_proj` 从未加回 `hidden`**（PR #125 记）：每层对 `inpL` 都是恒等映射，attention stack
   整体空转。jina-bert-v2 / nomic-bert / bert 三个变体全中，只是前两个 ERSS 相似度够高掩盖了；
   bge-small 因 CLS pooling 直接输出退化（相似度全 ~1.0）才暴露。修复后 rel→unrel 差距
   jina 0.170→0.417、nomic 0.258→0.443。
2. **MoE `per_expert_bytes` 算成 44 应为 34**（nomic-embed-text-v2-moe）：每个 expert 多读
   0.73 MB，第二个 expert 之后产出 NaN。

共同点：都能被一个位级 oracle 在 port 当天抓住，而不是等用户撞上。

#### 需要的产出

1. `tools/oracle/bert/` 与 `tools/oracle/gemma_embedding/`：`dump_golden.py`（从 llama.cpp
   `llama-embedding` dump 向量 + token ids）、`fixtures/*.json`（覆盖每个变体的开关组合：
   ALiBi vs pos_embd、GELU vs geglu vs SwiGLU、mean vs CLS vs last pooling、fused vs split QKV、
   MoE 偶奇层）、`compare.py`（严格查 token ids / 形状 / F32 原始位，报告首个分叉）。
2. 复用 `tools/oracle/shared/` 的 llama.cpp 构建 recipe（标量、单线程、关 Accelerate/
   Flash Attention），与 gliner2 的 `scalar.c` 模式一致。
3. 每个 arch 至少一条 `#[ignore]` 的 `matches_llama_cpp_bits` 测试，`RMI_*` 环境变量 gating。

#### 触发条件（满足任一才启动）

- 改动 `src/models/bert_family/` 或 `src/models/gemma_embedding/` 的任何数值路径；
- 换量化档（目前 7 个 arch 只有 Q8_0 验证过）；
- 用户报告 embedding 与官方实现不一致；
- 新接 BERT 家族变体（如 `jina-bert-v3`，源文件已在但没接）。

在触发前，`docs/MODEL_LIST.md` 对应行的口径保持"语义验证 + CLI/HTTP bit 一致"，**不写**
"逐位对齐 llama.cpp"。

### TODO-018: server 的两池超售（vision/audio 仍走 rayron）

核对 jina-embeddings-v5-omni 的 image 路径时发现两件事，都还没修。共同背景是
`src/core/thread_pool.rs:45-72` 记录的双池设计：LLM 走 `ComputePool`（显式分区、
无 work-stealing），vision/audio/qwen35 走 **rayron 全局池**（`into_par_iter` / `par_chunks_mut`）。
文档说两池靠 `src/main.rs:130` 的 `init_rayon_global_pool(n)` 对齐线程数，且
"never run concurrently ... so oversubscription is not an issue"。

#### 1. server 从不调用 `init_rayon_global_pool` → 线程数不可控 + 与 ComputePool 超售

CLI 在 main.rs 解析 `--threads` 后就建好 rayron 池；**`src/app/server/mod.rs::run_server`
完全没有这一步**。于是 HTTP 进程里第一个用 rayron 的请求会懒建池，线程数取 rayron
自己的默认（本机 `num_cpus`=18），而 ComputePool 是 `resolve_thread_count(--threads, 18)`
= clamp 到 `DEFAULT_THREAD_CAP=8`。

在 CLI 里这无害（vision 与 LLM 不并发）。在 server 里并发请求可以让 vision（rayron）
与 LLM（ComputePool）真正重叠，于是 18+8 个线程抢 8 核。

曾试过在 server 里补 `init_rayon_global_pool(8)`（2 行），**已撤回**，原因是：
  (a) 它不解决第 2 条的分歧（实测差异数仍是 153/1024，一点没变）；
  (b) 它只把超售从 18+8=26 降到 8+8=16，没有消掉超售本身；
  (c) `rayon::ThreadPoolBuilder::build_global` 只能成功一次，若别处先建了池则我的
      调用静默失效，属于脆弱写法。
根治手段是第 3 条。

#### 2. 〔已结案：误报〕jina-v5-omni image 路径 CLI/HTTP "1 ULP 分歧"是测量假象

一度记录为未决 bug，**实为比对方法错误，两侧逐位一致**。保留此节是因为那个坑
很容易再踩，且我踩了两次。

现象：CLI 打印（`print_embedding` 用 `{:.9}`）与 HTTP JSON（serde_json 对 f32 用
ryu 最短往返）逐元素比，**153/1024 个 `%.9f` 文本不同**，max abs diff 5.0e-10。

错在比法：把 HTTP 的十进制字符串解析成 **f64** 之后直接 format。该字符串是
"能 round-trip 回同一个 f32 的最短表示"，但它对应的 f64 落在 f32 精确值之上，
二次取整后末位进一。元素 0 实证：

    f32 精确值 = 0.0010754974791780114   (bits 3a8cf7b5)
    Rust {:.9}            -> 0.001075497   (正确：第 10 位是 4，舍)
    JSON 最短往返字符串    -> 0.0010754975
    Python format(f64)    -> 0.001075498   (错误：f64 落在 .5 上，进)

正确比法：先 `struct.pack('<f', ...)` 还原成 f32，再 format。改用后
**0/1024 不一致**。旁证：`embedding.final` trace 二进制两份 md5 相同、
HTTP JSON 与其 trace 1024/1024 位一致、CLI 文本与其 trace 1024/1024 一致。

教训（写给你我他）：**跨 `%.9f` 文本与 serde_json 浮点比较，必须先过 f32。**
本轮先是在 BERT/Gemma encoder 上正确绕过（0/1024），随后在 jina image 路径上
又直接 format f64，顺着错方向排查了很多轮（rayron 线程数 / `--threads` /
`--max-context` / 图片路径 / vision 逐位比对都做了，全是白工）。

#### 3. vision 迁到 ComputePool（`thread_pool.rs` 已登记的优选方向）

`src/core/thread_pool.rs:66-72`：未来统一方向是"把 audio/vision/qwen35 迁到
ComputePool，不是把 LLM 迁去 rayron"。迁移面：
- rayron 用点：`src/models/qwen3/vision/mod.rs` 3 处（1094 attention 的
  `into_par_iter`、213/278 两个 `par_chunks_mut`）、`src/models/qwen35/vision/mod.rs` 1 处；
- 需要穿 `pool` 的公共 API：`encode_pair` / `encode_image`（13 个调用点，分布在
  `app/qwen_drive.rs`、`app/omni.rs`、`app/text/multimodal.rs`、`app/text/vision.rs`）
  与 `VisionEncoder::from_source` / `VisionEncoder35::from_source`（10 个构造点）；
- 迁移后 rayron 依赖可整体从 Cargo.toml 移除，第 1 条的"两池对齐"问题随之消失。

注意：**这一条不修第 2 条**（vision 输出已证逐位相同）。它是第 1 条的根治手段。

### TODO-019: token 维寄存器分块，消除 per-token 权重重读

**目标**：让一次权重载入服务多个 token，把权重流量除以分块宽度 NR。图像编辑的文本
编码与去噪合计占端到端 ~95% 的时间，而算力利用率只有峰值的 ~11%——不是算不动，是
权重搬不过来。

#### 现状

`BF16Kernel::forward_batched`（`src/ops/kernel/bf16/mod.rs`）与
`F16Kernel::forward_batched`（`src/ops/kernel/f16/mod.rs`）都是：

```rust
for token in 0..n_tokens {
    self.forward(&input[token * n_in..], &mut output[token * n_out..], ...);
}
```

每个 token 独立做一次 matmul，**整个权重矩阵每个 token 从 DRAM 重读一遍**。
`qwen3::trunk::text_encode_forward` 的 `project_tokens`（本分支新加，按 token 分区到
`ComputePool`）虽然把并行度做上去了，但每个 token 内部仍是
`forward_prepared(..., 0, 1)` 的单行调用，权重流量一分没省。
`longcat::dit::Layer::forward` 是同样的 `chunks_exact` 逐 token 结构。
`qwen35::vision::matmul_weight_batch_pooled` 同理——分区解决并行度，不解决流量。

以 LongCat 文本编码（Qwen2.5-VL-7B，512 token）估算：每层 7 个投影、单投影权重
3584×3584×2 B ≈ 25.7 MB，512 token × 7 × 25.7 MB ≈ **2.6 TB** 权重流量。

#### 影响

本机（20 核 aarch64，`target-cpu=native`，release-fast）Turbo 32×32 / seed 42 / 20 线程：

| 阶段 | 耗时 |
|---|---|
| 文本编码（512 tok） | 158 s |
| 去噪 | 291 s/步 |
| vision | 22.6 s |
| VAE | 0.10 s |

文本编码 158 s 跑约 7.3 TFLOP ≈ **46 GFLOP/s**，相对 20 核 NEON 峰值（~400 GFLOP/s）
只有 **~11%**。分块宽度 NR=4 理论上把权重流量除以 4。

#### 何时触发

任何 prefill / 多 token 场景（图像编辑的文本编码与去噪、embedding、batched prefill）。
**autoregressive 单 token decode 不受影响**（n_tokens=1，没有可复用的维度），
所以这条不影响生成速度，只影响 prefill。

#### 选项

1. **在 kernel 内加 token 分块入口**（推荐）。新增
   `forward_batched_tiled(&self, input, output, n_in, n_out, ith, nth)`：对每个权重行，
   一次载入后用 NR 组累加器算 NR 个 token，然后 `ith/nth` 按**输出行**分区。
   调用方把现有的"按 token 分区"改成"按行分区 + kernel 内部分块"。
2. **调用方自建分块缓冲**。不改 kernel，在模型层把 NR 个 token 的输入打包成一个
   `n_in*NR` 的行——但这会改变 matmul 的数学形状（N 那维变成假的），权重布局不匹配，
   实际等于把权重复制 NR 份。**不可行**。
3. **转置权重为 token-major**。一次性重排权重让 token 维连续。改动面大、加载变慢，
   且对 mmap 的 GGUF 不友好。

#### 推荐

选项 1。理由：分块是纯粹的访存优化，不改权重精度、不改输入精度、不改累加顺序之外的
任何数学，因此**不触碰任何逐位对齐契约**——与本分支已建立的"重结合可以、近似不行"
原则相容。

实施注意：
- `project_tokens` 当前按 token 分区，改成按行分区后 dispatch 次数会变多
  （QKV 从 1 次变 3 次）。需要权衡 dispatch 开销与流量节省，NR=4 时预计净收益为正。
- Q8_0 路径（`q8_0::dispatch`）已经按 NRC4 一次算 4 个**输出行**，思路相同；
  缺的是复用同一权重的**多个输入行**，两者是正交的分块维度。
- 与 [TODO-018](#todo-018-server-的两池超售visionaudio-仍走-rayron) 第 3 条
  （vision 迁到 ComputePool）相关但**不同**：那条解决线程池归属，这条解决权重流量。
  建议先做这条——它不依赖线程池迁移。
- AuK 那边已记了同类问题（"Per-token matmul batching（`linear_into` 一调一次 token，
  瓶颈）"），两条应共用同一个 kernel 侧分块实现。

#### 关联文件

- `src/ops/kernel/bf16/mod.rs`（`forward_batched`）
- `src/ops/kernel/f16/mod.rs`（`forward_batched`）
- `src/ops/kernel/q8_0/dispatch.rs`（已有的 NRC4 输出行分块，可作参照）
- `src/models/qwen3/trunk/forward.rs`（`project_tokens`、`RowsWrap`、`TokenProjection`）
- `src/models/diffusion/longcat/dit.rs`（`Layer::forward` 逐 token 结构）
- `src/models/qwen35/vision/mod.rs`（`matmul_weight_batch` / `..._pooled`）
- `src/core/thread_pool.rs`（`compute` / `row_range`）

---

## Archived / completed work

### MiniCPM5-1B 输出对齐 Llama.cpp (2026-08) — ✅ 已解决

**根因：RoPE 风格用错。** `llama` GGUF 架构（含 MiniCPM5）使用 GGML `ROPE_TYPE_NORM`（interleaved 相邻对旋转），
而 Rust trunk 之前调用了 `rope_neox`（halves 风格）。

**修复**：`src/ops/rope.rs` 新增 `rope_norm`（interleaved 相邻对旋转），带 pinned-bits 单元测试；
`src/models/llama/trunk/forward.rs` Q/K RoPE 改用 `rope_norm`。

**验证**：MiniCPM5-1B-Q8_0 17 token prompt，Rust vs llama.cpp oracle 8 步贪心 top-1 完全一致。

### LFM2-8B-A1B (lfm2moe) MoE 支持 (2026-08) — ✅ 已完成

新增 `src/models/lfm2moe/`，实现 `lfm2moe` 架构的 Mixture-of-Experts FFN。
对齐 llama.cpp `build_moe_ffn(gating_op=sigmoid, norm_w=true)`：sigmoid 路由 + 偏置仅影响 top-k 选择 +
无偏 probs 归一化 + 加权求和。

排查过程发现并修复的既有 bug（**模式性 bug**，影响所有 trunk）：
1. **silu 与 matmul 的行分区竞争** — `n_ff % nth != 0` 时 silu 读到 matmul 尚未写入的行。
   闭包内统一为 ceil 分区：`llama`、`lfm2`、`lfm25`、`lfm2moe` 四处。
2. **shortconv conv 状态语义** — 解码期状态更新应为滑动窗口而非全行复制。
3. **lfm2/lfm25 conv 权重 2-D 形状** — 兼容 1-D 与 2-D 两种存储。

### LFM2.5-1.2B（dense）对齐 llama.cpp (2026-08) — ✅ 已完成

修复 conv 2-D 加载 + silu 分区竞争 + conv 状态滑动窗口后验证：
`What is the capital of France?` 13 token prompt，**8/8 步贪心 top-1 完全一致**（1098, 5706, 803, 4481, 856, 5242, 523, EOS=7），EOS 时机一致。

### Ornith-1.5-9B (qwen35 架构, 9B hybrid) 适配 (2026-08) — ✅ 已完成

33 blocks = 32 主层 + 1 MTP nextn 层；SSM 线性注意力 `full_attention_interval=4`，GQA 16/4 头，head_dim 256 + partial mrope 64/[11,11,10,0]。
适配点：
1. KV cache 按请求预算分配（原为 OOM 根因）：`(prompt_len + max_tokens).min(n_ctx)`。
2. nextn/MTP 层排除：新增 `config.n_nextn`，主栈层只遍历 n_layer_impl。
3. 上述封顶同时惠及 Qwen3.5-2B。

验证：llama.cpp oracle 8/8 步一致。封顶同时惠及 Qwen3.5-2B（原每次运行 KV cache 固定分配 12.9GB）。

### ERNIE-Image / Turbo (8B, single-stream DiT) — CPU 实验支持 (2026-10-05)

本次真实文件为 `unsloth/ERNIE-Image-GGUF/ernie-image-Q4_K_M.gguf`，
5,019,124,416 bytes，SHA256 `ed43d36ab45df0ef24e55d88e72bea241d0d54e4006ce0442386aba071b04c2a`。
该 GGUF 错标 `general.architecture=wan`，入口根据完整 ERNIE 张量签名识别。
参考固定在 stable-diffusion.cpp `3f8527a46c54ecf4cb4ed6003da8e8982283c73c`，
ggml `89c4413f5da6fb20cc796f16033d37f129be81fd`。

- DiT：36 层，hidden=4096，32×128 heads，FFN=12288，`GELU(gate) * up`。
  共享 6 路 AdaLN、RMSNorm、image-first/text-last 联合注意力、3D RoPE [32,48,48] / theta=256。
- 文本：Ministral-3-3B-Instruct-2512 Q4_K_M，26 层 / 3072 hidden，Tekken，原始 prompt + BOS；取 hidden_states[-2]，只执行前 25 blocks。
  参考图使用普通 full-width NeoX RoPE / base=1e6、causal attention，输出倒数第二层且不执行 final RMSNorm。
- VAE：`Comfy-Org/ERNIE-Image/vae/flux2-vae.safetensors`，转换入口
  `python -m tools.converter.ernie_image.convert_vae INPUT OUTPUT`。
  Conv 权重 F16，Attention 的四个 Linear 与向量保持 F32。
  128 packed channels 反归一化 → pixel shuffle → 32 channels → post_quant_conv → 共享 Flux decoder。
  latent 边长为输出分辨率 / 16；1024×1024 对应 4096 个 image tokens。
- 采样：discrete flow schedule / shift=4，Euler 积分 velocity；普通模型默认 32 steps / CFG 5，Turbo 默认 8 / CFG 1。
- 已有主 CLI 入口、真实组件契约测试、RoPE/调度/残差/Flux2 unpack/CFG 检查。
  64×64、1 步、CFG 1、seed 42、单线程的完整生成链路 93 个检查点 / 6,440,960 个 F32 原始位一致，
  包含文本编码、噪声、36 层 DiT、Euler latent 和完整 VAE RGB F32。
- 2 步 / CFG 5 的条件/空提示词两路、完整 DiT、采样和 VAE：261 个检查点 / 16,132,096 个 F32 原始位一致。
- 远端 256×256 / 32 步 / CFG 5 / seed 42 / 16 线程运行成功，1027.489 s；样图有桌面和窗景但漏猫，质量验证未通过。
- 可复现工具：`tools/oracle/ernie_image/`，固定版本、隔离插桩、禁用 SIMD/FMA/融合、严格 u32 比较。
- 待完成：Turbo 独立权重、SIMD/FMA 数值对齐及 512/1024 生成质量验证。
  CPU 路径；尚无 ERNIE Vulkan/GPU 支持验证，不能继承 Z-Image 的加速结论。

### AuK-Base (1.5B, Flux2Edit 音频 DiT) 适配 — 🚧 进行中 (2026-10)

`general.architecture = auk`（audio.cpp 的 community model；GGUF 通过 `transformer.transformer_blocks.0.img_attn.qkv.weight` 这类 tensor 命名探测，参考 `references/audio.cpp/src/community_models/auk/{flow,conditioning,vae,audio_conditioning,session}.cpp` 116 KB 参考实现；HuggingFace 镜像源 `audio-cpp/AuK-Base-and-Flash-GGUF`）

核心架构（`audio.cpp/docs/community_models/auk.md` + `config/auk-base.yaml`）：
- Backbone = **Flux2Edit**（Flux 风格双流 DiT，跟 `pig` (Z-Image) 同一家族）
- `dim=1536, heads=24, head_dim=64`，`ff_mult=2 → ffn=3072`
- 10 个 double blocks（并行 img/txt 注意力 + MLP）+ 20 个 single blocks（仅 img 注意力）
- `text_hidden_dim=2048`（Qwen2.5-Omni-3B n_embd）
- VAE = `BigVGANFlowVAE`，64-dim latent，480× 下采样，24 kHz 输出
- CFMEdit = flow-matching 编辑变体，logistic_normal schedule（P_mean=-0.8, P_std=0.8）
- AuK-Base：32 Euler 步 + guidance 2.0
- AuK-Flash：固定 4 步 + guidance 0（蒸馏版；本轮暂不 port）

GGUF 重量（从 `audio-cpp/AuK-Base-and-Flash-GGUF`）：
- `auk-base-f16.gguf` 2.9 GB（F16 DiT，本轮首选）
- `auk-base-q8_0.gguf` 1.5 GB（Q8_0 量化）
- `auk-vae-f32.gguf` 608 MB
- `qwen2.5-omni-3b-q8_0.gguf` 4.0 GB（**本地已有** `models/Qwen2.5-Omni-3B-GGUF/Qwen2.5-Omni-3B-Q8_0.gguf`）

进度：
1. ✅ MODEL_LIST.md 加行 + TODO.md 占位
2. ✅ 下 AuK GGUF + inspect tensor 命名（vs audio.cpp 期望），自校
3. ✅ `src/models/diffusion/auk/{mod,dit,vae,text}.rs` scaffold（参照 pig 的 Flux 双流 + AuK 专有 CFMEdit 逻辑）
4. ✅ CLI dispatch + arch detection + 测试钉 contract（6/6 auk_di_t_q4_k_m 通过）
5. ✅ 端到端冒烟：`--model auk-base-f16.gguf --text-encoder Qwen2.5-Omni-3B-Q8_0.gguf --vae auk-vae-f32.gguf --prompt "..." --out speech.wav`（4 步 ~8.5 min，生成 0.13s 24 kHz mono WAV）

实现要点（commit `899602c..dfd74b9` on `Auk` branch，共 12 commits ~3600 LoC）：
- DiT：Flux2Edit 10 double + 10 single blocks，6-way AdaLN modulation，joint img+text attention，per-head RMS QK-norm + RoPE，packed gate+up SwiGLU FF
- Text encoder：Qwen2.5-Omni-3B（直接复用 `qwen3::trunk::Qwen3Model`）
- Diffusion：linear sigma schedule，Euler step `x_next = x + v * dt`，velocity clipping (band-aid)，CFG with zero-text unconditional pass
- VAE：BigVGANFlow decoder — conv_pre (64→1536) + 6× upsample stages + 18 resblocks (3/stage) + conv_post (24→1)；weight-norm decomposition；SnakeBeta with GGUF-stored alpha/beta
- WAV：16-bit PCM mono writer

待优化（follow-up，no rush）：
- [ ] 数值校准 oracle-diff vs `references/audio.cpp/src/community_models/auk/flow.cpp` — block forward magnitudes drift without the velocity clip
- [ ] Per-token matmul batching（`linear_into` 一调一次 token，瓶颈；需新增 batched Q8_0 matmul helper 或 thread-local Q8Scratch per rayon worker）
- [x] **F16 matmul Vulkan path**（done as A：A 任务）— `auk_f16_gpu_runtime`（max_rows=1, max_n_in=4096, max_n_out=9216, descriptor_capacity=4096）+ `f16_gpu_linear_into`（单行 `BatchedLinearRuntime::matmul_rows(format=F16)`）接入 `linear_into_scaled_impl` 的 F16 分支。**pre_quantize F16→Q8_0 workaround（`f4e7879`）已删除**，主路径现在按 GGMLType dispatch——F16 走 F16 GPU，Q8_0 走 Q8 GPU matmul，BF16/Q*_K 走 `QTensorOwned` CPU。消除了 21s 一次性 F16→Q8 load + 1.5 GB cache RAM + post-training quant noise。end-to-end dev 335s（vs 311s baseline Q8 path，持平）。
- [x] **修掉 `linear_into_dispatched` + `linear_into_dispatch` 两处无限递归** —— 之前在 Q8 miss 时调用 `self.linear_into_dispatched` / `linear_into_dispatch`（实际是无限递归，被 Q8 cache 100% 命中掩盖）。两者现在都改为 `super::linear_into_scaled_impl`。拆掉 pre_quantize 后两个 recursion 都会 stack overflow，所以必须一起改。
- [ ] VAE resblocks 的 causal padding（当前用 symmetric，lossy 一点；low impact 因 upsample 已用 linear interp）
- [ ] VAE SnakeBeta 的 FIR up/down filters（audio.cpp 用 `build_activation`，我们 skip；lossy 一点，minor impact）
- [x] **CFMEdit reference-audio conditioning path**（done as C：C 任务）— `AukPipeline::generate_audio_with_audio(prompt, audio_embeddings, audio_tokens, options)` 接入。`audio_embeddings` 是 `audio_tokens * 2048` 的 f32 buffer（Qwen2.5-Omni audio tower 的隐藏空间），与 text embeddings 拼成 `[img, audio, text]` joint sequence 走 DiT forward。`dit.denoise` 新增 `audio_conditioning` + `audio_tokens` 参数，`predict_velocity_inner` 改用 `cond_tokens = audio_tokens + text_tokens`；joint layout、txt_proj、scratch.prepare 全部按 cond_tokens 处理。零向量 smoke 337ms OK（max_abs=0.95，finite audio）。**Qwen audio tower encoder 还没 port**——端到端 CFMEdit 需要外部提供 audio embeddings（Python 跑 Qwen audio tower 导出 f32 binary 即可）。这是 follow-up 工作，涉及 32 transformer encoder layers + 2 Conv1d + ln_post + proj（1280→2048）。
- [x] **Qwen2.5-Omni audio tower encoder**（done as audio tower 任务：commits f8accde / c64f1ea）— 新模块 `src/models/qwen3/omni_audio/`（~870 LoC）实现完整 conv1/conv2 + 32 transformer encoder layers + ln_post + proj 1280→2048。Whisper Kokoro mel 提取（复用 qwen3/asr/audio_processor 已有 helpers，16kHz/400/160/128），F16/BF16→F32 反量化在 load 时做。`AukPipeline::load_with_audio_tower` + `AukPipeline::generate_audio_with_reference_wav` 接入。Smoke 测试：0.5s 440Hz 正弦波 → 26 audio tokens, max_abs=3.54, finite。**完整 CFMEdit 路径**（done as commit 03f71b9）：Qwen3Model 新增 `text_encode_with_audio`，在 embedding lookup 阶段把 audio tower 输出替换进 `<|AUDIO|>` token 位置（id 151646），让 Qwen 36 层 transformer 跑整条 sequence，text 和 audio 通过 self-attention 交互。BPETokenizer 新增 `from_qwen25_omni_embedded_merges` 加载 22 个 Qwen2.5-Omni special tokens（含 `<|AUDIO|>=151646`、`<|audio_bos|>=151647`、`<|audio_eos|>=151648`），AukTextEncoder 新增 `encode_with_audio` 构造 CFMEdit 提示模板 + 调 text_encode_with_audio。端到端 smoke 670s dev mode，48 cond_tokens (vs 简化版 27)，3200 samples @ 24kHz finite。
- [x] **AuK-Flash 蒸馏版（4 步 + guidance 0）**（done as D：D 任务）— **零代码改动**。Flash 跟 AuK-Base 共享同一份架构（Flux2Edit, dim=1536, 24 heads, ff_inner=3072, num_layers=10, num_single_layers=20）+ tensor layout（420 个 tensor，一一对应）。下载 `auk-flash-f16.gguf`（2.9 GB F16 from `audio-cpp/AuK-Base-and-Flash-GGUF`），通过同一份 `AukPipeline::load` 加载；调度切换为 `steps=4, guidance_scale=0.0`（跳过 CFG unconditional run）。Smoke 测试：`tests/auk_flash_smoke.rs`（`#[ignore]`）；dev mode 4 步 323s，3200 samples @ 24 kHz，max_abs=0.95，finite。
- [x] **16/16 audio.cpp 任务覆盖**（commit `pending`）— 加 `AukAudio::pitch_shift(semitones)` / `speed_change(rate)` / `volume_change(db)` 三个 DSP 编辑（线性重采样，pitch_shift 同时改音高和时长，**真正"音高↑ 时长不变"需要 phase vocoder，是 follow-up**）+ `AukOptions::instruct` 字段 + CFMEdit 提示模板里加 `<instruct=...>\n` 前缀（对齐 audio.cpp 的 instruct-TTS 格式）。audio.cpp 16 任务验证表里的 12 个 reference-audio 任务（zero-shot TTS / speech editing / timbre / emotion / de-accent / whisper / enhance / separate / target speaker 等）和 4 个 text-only / DSP 任务（instruct TTS / pitch / speed / volume）都有 Rust 路径。7 个新单元测试覆盖 DSP 方法（`tests/auk_dsp_unit.rs`）。

性能现状（4 steps + CFG=2.0 + 4 threads + Intel MTL + `--gpu`，**release 模式**）：
- text_encode: 1.6s（CPU；Qwen2.5-Omni qwen2vl arch 还没 Vulkan）
- denoise: 40s（F16→Q8_0 pre-quant + Vulkan Q8 GPU matmul；4.1x vs 168s CPU F16）
- vae_decode: 50s（ComputePool par_chunks over resblocks，已 2.86x vs 166s serial）
- total: 93s（5.5x vs 515s baseline）

vs 起点（515s/0.13s）：5.5x end-to-end 加速。

注：audio.cpp 的 C++ 代码是 **参考** 而非金标准 — tensor 命名、参数化约定可能跟实际 GGUF 有微小差异（参考 ERNIE-Image 跟 unsloth GGUF 的踩坑先例），以 **GGUF 实际 tensor 名 + shape** 为准。

### Spark-X2.5-1.7B / 4B-GGUF 适配 — ✅ 已完成（功能）+ ⚠️ 性能待优化

`arch=spark2_5`，Xunfei Spark 2.5 讯飞星火。适配点：
- **Fused QKV**：`attn_qkv [n_embd, n_embd_qkv]` 单一投影 → split 为 Q/K/V。
- **Per-layer heterogeneous RoPE**：full-attn `n_rot=64, freq_base=5M`；SWA `n_rot=256, freq_base=10K`。新增 `rope_neox_partial`。
- **Sliding window mask**：SWA 层只看最后 `sliding_window=512` 个位置。
- **Per-head gating**：`attn_gate` pre-attention norm → sigmoid → 逐 head 乘 attention output。
- **GeGLU FFN**：`gelu(gate(x)) * up(x)` 后 down。
- **Tied embeddings**：GGUF 无 `output.weight` → 复用 `token_embd.weight`。
- **Tokenizer pre=`spark2_5`**：与 qwen2 正则不同，新增 `PreTokenizer::Spark2_5`。

冒烟通过。1.7B ≈ 1.1 tok/s, 4B ≈ 0.4 tok/s（4 线程；生产路径未优化）。

待优化：
- [ ] Per-token prefill O(N²) 注意力 → batched prefill
- [ ] 每次 `decode_step` 全量 `Vec::new` → ExecutionScratchpad
- [ ] Generate 阶段没有 KV cache 预热
- [ ] FFN gate 输入源（pre-attention norm vs post-attention hidden）
- [ ] GeGLU 用 `0.5 * x * (1 + tanh(...))` 近似 vs llama.cpp 默认精确 GELU
- [ ] chat template 整段手写，未走 `tokenizer.chat_template` 的 Jinja 引擎
- [ ] 精度对齐 XFllama.cpp oracle
- [ ] `get_f32_tensor` 重复 6 处（暂不抽，等待 `RequiredTensor` trait 落地）

---

## SIMD extension inventory

This project targets the following SIMD instruction sets:

| Architecture | Extensions used | Detection | Files |
|---|---|---|---|
| **x86_64** (Intel Core Ultra 5 125H, AMD Zen 4, etc.) | SSE2 / SSE4.2 / AVX / **AVX2** / **FMA** / **F16C** / **BMI2** | `is_x86_feature_detected!` | `src/ops/simd_avx2.rs`, `src/ops/kernel/*/avx2*.rs` |
| x86_64 (Skylake-X, Ice Lake, Sapphire Rapids, Zen 5) | + **AVX-512F** / **AVX-512BW** / **AVX-512VNNI** | runtime gate | (not yet wired) |
| x86_64 (Intel Alder Lake+, Zen 5) | + **AVX-VNNI** (`vpdpbusd`/`vpdpwssd`) | CPUID `avx_vnni` | (see TODO-AVX-VNNI below) |
| **aarch64** | NEON / NEON-F16 / NEON-F32 | `is_aarch64_feature_detected!` | `src/ops/kernel/*/neon*.rs` |
| aarch64 (Apple M1+) | + NEON-DOT / SME | runtime | (not yet wired) |

### AVX-VNNI / AVX-512 background

**AVX-VNNI** (`vpdpbusd`/`vpdpwssd`, also written as `vpdpbusd_yx_yx` or `_mm256_dpbssd_epi32`)
is an Intel/AMD x86 extension that performs **saturating int8 dot-product with a
single FMA-like instruction**. First shipped on Intel Tiger Lake (2020) and AMD Zen 5;
the current dev box (Core Ultra 5 125H, Meteor Lake) supports it via the `avx_vnni`
CPUID bit. It computes `acc += (a - 128) * b` over 16 int8 lanes with saturating accumulation
to int32 — exactly the operation our Q8_0 × Q8_0 matmul does via three separate
AVX2 instructions (`_mm256_maddubs_epi16` → unpack → `_mm256_madd_epi16` → CVTPS).

**AVX-512 VNNI** (`vpdpbusd` with 64-lane / 32-int32 ops on zmm registers) is the server-class
extension of the same idea: 2× the throughput per instruction and the only way to reach
> 50 tok/s on Qwen3-0.6B decode. Sapphire Rapids, Zen 5, and Zen 6 (Medusa Ridge) support it;
our dev CPU does not.

### TODO-AVX-VNNI: int8 dot 加速

**目标**：在已有 `q8_0/avx2.rs` 的 3-instruction int8 dot (`_mm256_maddubs_epi16` →
unpack → `_mm256_madd_epi16` → CVTPS) 基础上加一条 AVX-VNNI 路径
(`_mm256_dpbsud_epi32` / `_mm256_dpbssd_epi32`)，把 saturating int8 dot
缩成 1 条指令。预期 1.3-1.5× Q8_0 matmul 加速。

**实现计划**：
1. 加 `#[target_feature(enable = "avx2", enable = "avxvnni")]` 的 kernel
   `matmul_q8_0_vs_q8_0_avxvnni(weight, q8, scales, output, n_in, n_out, start, end)`，
   内部用 `_mm256_dpbsud_epi32` (unsigned×signed saturating dot)。
2. 在 `q8_0/dispatch.rs::matmul_q8_0_quantized_range` 加 `is_x86_feature_detected!("avxvnni")` 分支。
3. 加 parity 测试 `q8_0_avxvnni_matches_scalar_q8_dots`，相对 scalar 1 ULP 内。
4. 在 `src/ops/kernel/mod.rs::Kernel` dispatch 表里 KV 缓存值的 int8 dot 也走 AVX-VNNI。

**风险**：
- `_mm256_dpbssd_epi32` saturate 到 int32 → 16 lanes per dot。K2-Horizon-4B / Breeze 等
  Q8_0 权重 dot 输出通常在 ±10⁴ 量级，不会触发 saturate，所以等价于无 saturate 的 int32 dot。
- AVX-VNNI 在 Meteor Lake 上只有 256-bit 寄存器（不是 512-bit）。所以加速上限是"少 2 条 unpack 指令"，
  不是 2× lane 宽。
- 加 `target_feature("avxvnni")` 意味着老的 CPU 不能跑这条路径（fallback 到 AVX2）。

**验收**：
- Q8_0 Q4_0 matmul parity 测试通过（1 ULP 内）
- Qwen3-0.6B-Q8_0 decode ≥ 36 t/s（vs 当前 ~28 t/s），用 `--max-context 8192` 默认 cap
- K2-Horizon-4B-Q8_0 decode ≥ 8 t/s（vs BF16 当前 4 t/s），关键前置：**必须** `--max-context 8192`
  否则默认走 524288 → 77 GB OOM（见 TODO-MAX-CONTEXT-CLI）

### TODO-LLAMA-PER-TOKEN-SIMD化

**目标**：扫描 `src/models/llama/trunk/forward.rs` 仍为 scalar 的 per-token 小循环，
逐个替换为 SIMD 版本。AVX2/FMA/NEON 已覆盖 `vec_scale_f32` / `vec_add_into` / `vec_mad_f32` 等基础算子。

**已审计的热点**（per token × per layer）：

| 操作 | 当前 | 行号 | 优化 |
|---|---|---|---|
| RMS norm grouped | ✅ SIMD via `rms_norm_grouped` → `scale_mul_avx2` | 462, 750 | - |
| Q8_0 quantize | ✅ SIMD via `quantize_q8_0_into_avx2` | 464, 766 | - |
| Q8_K quantize | ✅ SIMD via `quantize_row_q8_k_avx2` | 472, 772 | - |
| BF16 × Q8 matmul | ✅ AVX2 (`matmul_bf16_vs_q8_avx2`) | 484, 702, 804 | - |
| **Embedding scale** | ❌ scalar `for v in x.iter_mut() { *v *= scale; }` | 419-422 | ✅ **已替换为 `vec_scale_f32`**（commit 见 K2check 分支） |
| RoPE (k2-horizon neox) | ✅ SIMD `rope_neox_inplace_avx2` | 528-544 | - |
| KV cache write (F32) | ✅ `copy_from_slice` (memcpy → SIMD 路径) | 577-583 | - |
| KV cache write (F16) | ✅ `f32_slice_to_f16_avx2` (F16C) | 559-568 | - |
| Attention (F16) | ✅ SIMD `dot_f16_f32`, `vec_mad_f16_f32`, `vec_scale_f32` | 609-630 | - |
| Attention (F32) | ✅ SIMD `dot_f32`, `vec_mad_f32` | 636+ | - |
| Residual add | ✅ `vec_add_into`, `vec_mad_f32` | 720-724, 910-914 | - |
| silu_mul | ✅ SIMD `silu_mul_approx_inplace` | 831, 840 | - |
| **KV cache / scores buffer size** | ❌ hardcoded `512.min(cfg.n_ctx)` + `[0.0f32; 512]` | 117, 941 | ✅ **已替换为 `cfg.n_ctx.min(max_context)` + `vec![0.0f32; max_ctx]`**（见 TODO-MAX-CONTEXT-CLI） |

**结论**：除 `embedding_scale` 外 llama trunk 已是 SIMD 化。`embedding_scale` 修复已落地 K2check 分支
（每个 token 节省约 1.3 µs @ n_embd=1536，对总推理时间影响 < 0.1%）。

剩余 per-row attention 在 autoregressive decode 时是 sequential（依赖前面 token 的 KV），
SIMD 不能跨 token 加速——这部分要看 [TODO-010](#todo-010-prefill-runtime-通用化避免每个模型重复实现-chunk-调度) batched prefill。

### LLama trunk matmul 真实瓶颈（per-token）

解码时 n_out=1，所有 matmul kernel 的内并行（按 n_out 切 rows）失效——线程利用率 12.5%（1/8）。
**这是 autoregressive decode 慢的真凶**，不是 SIMD 缺位。

修复路径见 TODO-010（batched prefill）+ 自动接受单 token decode 慢的现实（llama.cpp 同样慢）。

### TODO-MAX-CONTEXT-CLI: `--max-context` 用户可调 KV cache 上限

**背景**：`src/models/llama/trunk/forward.rs` 和 `src/models/lfm2/trunk/forward.rs` 历史上都把
KV cache 上限硬编码到 512：

```rust
// llama trunk 历史硬编码
let max_ctx = 512usize.min(config.n_ctx);
// lfm2 trunk 历史硬编码
let max_ctx = 512usize.min(cfg.n_ctx);
// lfm2 attention `values` 数组
let mut values = [0.0f32; 512];  // attention 长生成时越界 panic
```

后果：512 token 以上的生成直接越界 panic，或更长生成被静默截断。多个用户的报告
（VibeVoice ASR 1.5B 长音频、K2-Horizon 中文翻译模型 chat 段、K2-Horizon-4B 192K 长输出）
都触及过同一类 bug。

**修复**（已落地 K2check 分支）：

- llama trunk / lfm2 trunk：`max_ctx = config.n_ctx.min(max_context)`，配套
  `run_inference[_stream]` 加 `max_context: usize` 参数
- lfm2 attention `values`：`vec![0.0f32; max_ctx]`
- 新 CLI 参数 `--max-context N (default 8192)`（`CliOptions::DEFAULT_MAX_CONTEXT`），
  `effective_max_context()` helper
- 全链路贯通：`src/app/cli.rs` / `src/app/text.rs` / `src/main.rs` /
  `src/models/lfm2/vision.rs`（multimodal LFM2-VL 也接 max_context）

**为什么用 CLI 而不是无脑用 model claim**：部分模型 GGUF `context_length` 字段是
"该 GGUF 可接受的 max"，而不是该模型实际有意义的长度。例如：

| 模型 | claim `context_length` | 512-tok panic | CLI `--max-context 8192` |
|------|------------------------|---------------|---------------------------|
| K2-Horizon-1B | 131072 | 是 | KV cache 0.29 GB |
| K2-Horizon-4B | **524288** | **77 GB OOM** | KV cache 0.86 GB |
| LFM2-1.2B | 32768 | 是 | KV cache 0.5 GB |
| Qwen3-0.6B | 32768 | 是 | KV cache 0.4 GB |

用户按需 `--max-context` 即可（绝大多数 chat 8K 够，长上下文特殊任务手动加）。

**后续**：

- `docs/usage/llama.md` / `docs/usage/qwen3.md` 等可加一行 "默认 KV cache cap 是 8K，可用
  `--max-context N` 覆盖"。
- TODO-AVX-VNNI / TODO-LLAMA-PER-TOKEN-SIMD 实现后，再跑一遍 K2-Horizon-4B
  对比 `--max-context 8192` vs `--max-context 32768` 的 prefill 时间（验证大 context
  不会因为 KV 随机访问模式变慢）。
### Standard gemma3 trunk (`src/models/gemma3/trunk/forward.rs`) — DONE for 270M-it

`src/models/gemma3/` 现在服务 **BitNet 270M** (file_type=40) **和**
**standard gemma3-270m-it** (file_type=15, Q4_K_M mixed) — 两条 forward
path 共享同一个 `text_encode` 函数体，`cfg.is_bitnet` 在 7 个投影
（attn_q/k/v/output + ffn_gate/up/down）处分发：

- `is_bitnet=true`  → `bitlinear_projection(...)` (BitNet path，SIMD hot path)
- `is_bitnet=false` → `standard_projection(...)` (Q4_K/Q5_0/Q6K/Q8_0 matmul，
  Q8_0-quantized 激活 × mixed-quant 权重，跟 qwen3 trunk 同款)

新增 `Gemma3Config.sliding_window: usize` 字段（默认 0 = full causal），
`causal_self_attention` 在 mask `j ∈ [0, i + 1 - sliding_window)`。
Standard gemma3-270m-it 声明 `sliding_window=512`（hybrid local/global attn）。

`Gemma3LayerWeights` 同时存 `bitlinear: BitLinearSlot` 和
`wq/wk/.../w_down: Option<StdProjection>`（owned `Vec<u8>` + ggml type + shape）。
Forward 路径在调用 `standard_projection` 时按需构造 `Weight` kernel
（cheap; kernel 只是 byte buffer 的 view + 维度）。

`static_weight` (`src/models/gemma3/trunk/weights.rs`) 现支持 F16 (BitNet 270M)
和 Q8_0 (standard gemma3-270m-it，mixed-quant conversion) 两种 token
embedding。Q8_0 dequant 用 34 bytes / 32-element block layout
(2-byte F16 scale + 32 signed int8)。

`build_config` (`src/models/gemma3/embedding.rs`) 改：
- vocab 从 `tokenizer.ggml.tokens` array length 读（standard GGUF 没有
  `gemma3.vocab_size`；BitNet 仍用 `gemma3.vocab_size` fallback）。
- `gemma3.context_length` 改 unwrap_or(0)（standard GGUF 总是有，BitNet 也总有；fallback
  仅为防御）。
- `gemma3.pooling_type` 改 unwrap_or(1)（standard GGUF 不设，BitNet 设 1）。
- `gemma3.attention.sliding_window` 新字段。

`app/mod.rs::compute_embedding` / `run_embedding` dispatch 加 `"gemma3"` case
→ `crate::models::gemma3::compute_embedding` / `run_embedding`（之前默认
fall through 到 qwen3，不识别 arch）。`models::gemma3` 现在 pub mod。

`models::gemma3::trunk::BitLinearWeights` 引用从 `crate::ops::bitlinear` 改到
`crate::ops::bitnet`（旧的 `bitlinear/` 模块已不再 pub；commit e011536
之后一直未更新 gemma3 trunk 的 import，这是 e011536 的 holdout）。

**e2e** (`tests/gemma3_270m_it_e2e_embed.rs`) 5/5：
shape=640、finite、value range `[-300, 300]`、L2 norm `∈ (0.1, 1000)`、
byte-deterministic、longer prompt 32 tokens 仍 finite、cooking vs
software cosine sim < 0.99（IT 模型 + last-token pooling discrimination
较弱；0.99 threshold 是宽口径，只 catch 完全相同 embedding）。

**性能**：standard gemma3-270m-it Q4_K_M 单 token 4 线程 ~1.0 s；BitNet-270M
packed SIMD 路径 ~480 ms。两者 forward 时间在同一数量级 — Q4_K matmul
是 AVX2 + Q8_0-quantized activation，跟 SIMD int8 × int8 dot 在内存带宽上
类似。

**未做**（4B/12B/27B standard gemma3 的前置）：

1. KV cache + decode loop（生成模式；当前只有 embedding extraction = last-token）。
2. IT chat template（`tokenizer.chat_template` 已在 contract test 里锁定含
   `<start_of_turn>{role}
...<end_of_turn>
`，但 `run_shared_inference` 仍
   返回 error）。
3. **Bit-exact 对照 oracle**：本机没有 cmake/PyTorch/bitnet.cpp build；任何
   "对得上 llama.cpp b96806d --temp 0 --top-k 1 golden log" 的工作需 cmake
   工具链，目前不可达。Forward 通过有限 golden test 验证（shape / finite /
   range / determinism / discrimination）— 跟 BitNet-270M 的 e2e 套件同级别。

针对 gemma-3-270m-it 的 contract test (`tests/gemma3_270m_it_q4_k_m.rs`)
已 8/8：metadata (arch, file_type=15, dims)、sliding_window=512、
SPM tokenizer (`model=llama`, no merges, EOS=106)、mixed-quant tensor
inventory (109 F32 + 81 Q5_0 + 27 Q4K + 9 Q6K + 10 Q8_0 = 236 张量, no I2_S, no norm_in)。

历史：

1. `causal_self_attention` 加 sliding window mask（`gemma3.attention.sliding_window=512`，
   屏蔽 `|i - j| > sliding_window` 的 K/V）。
2. 在 `gemma3/trunk/weights.rs` 加非 BitLinear 路径：`attn_q/k/v/output` +
   `ffn_gate/up/down` 用 `Weight::from_quantized`（Q4_K/Q5_0/Q6K/Q8_0 混合，按张量名匹配），
   走 `matmul_q8_0_quantized_parallel_rows`（现有的 standard matmul，已在 qwen3 trunk 用过）。
3. `Gemma3Config::is_bitnet` field 保留作 dispatcher，但加一个
   `pub fn new_from_source(&self) -> ...` 的 standard gemma3 入口（`text_encode_v2` 或
   重命名）；`embedding.rs::compute_embedding` 走 `cfg.is_bitnet` 分发两个 path。
4. IT tokenizer：`tokenizer.chat_template`（含 `<start_of_turn>{role}\n…<end_of_turn>\n`）
   + EOS=106（不是 BitNet 的 EOS=1）已经在 contract test 锁定；`BPETokenizer` 已经能处理
   SPM `tokens + scores + token_type`（这是 BitNet-270M 复用过的），IT 模型跟 base 模型
   tokenizer layout 完全一致，理论上零增量工作量。
5. KV cache + decode loop（用于 IT generation 而非仅 embedding extraction）。

rough scope: ~400-500 LOC（取决于要不要 KV cache / generation），
与 `tests/gemma3_270m_it_q4_k_m.rs` 的 8 条 contract test + 端到端 forward parity test 配对
（参 llama.cpp `--temp 0 --top-k 1` greedy 输出做 32→1 字符串对齐；oracle 见 llama.cpp `b96806d`
在 `unsloth/gemma-3-270m-it-GGUF` 上的 golden log `main-0319c65`）。

---

### TODO-020: Vulkan SiLU 的 f32 次正规数缺陷 — ✅ 已修复，但饱和边界不可回退

#### 现象

`record_silu_mul` 一旦被写进同一 command buffer 的 dispatch 链（Z-Image DiT 的 FFN
合批），整幅渲染产出全 `-inf`：

```
Inference error: Non-finite Z-Image predicted flow: len=65536, nan=65536, -inf=65536
```

而同样的算子在**分开提交**时完全正确。历史上这个失败被归因为「descriptor set 与
pipeline 布局在跨 dispatch 复用时的状态」，并标注「根因未定位」——**那个归因是错的**。

#### 根因

`shaders/glsl/silu_mul.comp` 的 `exp_approx` 把 exp 输入 clamp 在 `±88.376`
（exp 的溢出点）。但 silu 的补偿算法需要 `1/denominator` 落在 f32 **正规数**范围：

```glsl
den = 1 + exp(-gate);
act = gate / den;
res = fma(-den, act, gate);     // 灾难性抵消，res 是残差
act = fma(res, 1/den, act);     // 用 1/den 修正
```

| clamp | `den` | `1/den` | 状态 |
|---|---|---|---|
| 88.376 | 2.406e38 | **4.157e-39** | **低于正规数下限 1.175e-38** → 修正项退化 |
| 80 | 5.541e34 | 1.805e-35 | 正规数 → 精确 |

`gate ≤ -88` 时修正项产生 NaN。Z-Image FFN 的 gate 实测 amax 到 962、
min 到 -197，所以必然踩中。

**为什么藏了这么久**：自检 `check_operator_chain` 用 `N_FF = 96` 覆盖这个 shader，
而真实 FFN 是 `3840 × 9728 = 37 324 800`——**38.9 万倍**，且小 fixture 的输入从未
接近边界。**这个算子在生产规模下从未被验证过**，任何模型在 GPU 上调用它都会中招
（`record_silu_mul` 是共享算子，不只 Z-Image 用）。

#### 修复

clamp 从 `±88.376` 改为 `±80`，让 `1/den` 保持正规数。新增
`silu_ffn_scale_probe`（`src/vulkan/ops.rs`）用 FFN 真实规模跑，断言零非有限，
并挂进 `run_qwen3_operator_check` 作为常规门禁。

修复前 `16 009 509 / 37 324 800` 非有限（43% 损坏），修复后为 0。

```bash
SILU_ONLY=1 ./target/release/examples/vk_ops_check   # 只跑这一项
```

#### ⚠️ 风险 1：与 CPU SiLU 不再逐位一致（已发生）

分两个区间，不是统一说法：

| 区间 | CPU | GPU（clamp=80）| 一致性 |
|---|---|---|---|
| `\|gate\| < 80`（绝大多数激活）| `x/(1+exp(-x))` | 同式，无 clamp | **逐位一致**，相对差 0 |
| `gate ≤ -88`（极少数）| `1+exp(103)` 溢出 f32 → `x/inf` = **-0** | 饱和在 `exp(80)` → **≈ -1.9e-33** | **绝对差 ~1e-33，相对差不成立** |

**CPU 在这个区间自己也是错的**（数学上 `silu(gate) → 0`，`-0` 是"意外对"），两边只是以
不同方式失效。1e-33 在 f32 里是噪声底，不会污染下游 matmul，但**任何拿 GPU 与 CPU
比对这个区间的测试必须用绝对容差**，用相对容差会误报。

实测代价（全部来自边界区间）：Z-Image 512×512 与 CPU 路径的 PSNR 从
**33.34 dB 降到 31.52 dB**（mean|Δ| 2.81 → 3.61）。人眼核查差异不可见。

**若要两边数学同解**：对 `gate < -80` 直接返回 0（真值就是 0），`|gate| < 80` 的
精确路径不动，这样边界区变成逐位的 `-0`。**未做**——当前状态已足够，
且这是额外改动而非修 bug。

#### ⚠️ 风险 2：clamp 值不可随手调高

看到「88.376 是 exp 溢出点，调回去更精确」是很自然的直觉——**那正是这个 bug**。
`1/den` 必须是正规数，这是硬约束，不是保守取值。

#### ⚠️ 风险 3：任何模型在 GPU 上用 SiLU 都继承这个饱和

`record_silu_mul` 是共享算子。新增模型或新路径调用它时，**不能假设它与 CPU SiLU
等价**。代码注释（`silu_mul.comp` 的 PARITY NOTE）已写明。

#### 关联文件

- `shaders/glsl/silu_mul.comp` — `exp_approx` 的 clamp + SATURATION BOUNDARY 注释
- `src/vulkan/ops.rs` — `silu_ffn_scale_probe`（生产规模门禁）
- `src/models/diffusion/z_image/dit_gpu.rs` — `record_ffn`（FFN 合批，依赖本修复）
- `docs/usage/z-image.md` — 「FFN 合批的 `-inf`：已定位」记录排查全过程

---

### TODO-021: Z-Image FFN 合批 `record_ffn` — 已合入，依赖 TODO-020

一次 command buffer 录 w1 → w3 → SiLU → w2，取代原来「3 次 submit + 2 次 readback +
host 侧 3700 万次 exp」。denoise 52.7 s → 42.5 s，端到端 59.9 s → 49.7 s
（512×512，8 步，seed 42，NVIDIA GB10）。

**与 TODO-020 是同一件事的两端**：旧 clamp + 合批必然 `-inf`，两者必须一起看。

排查时已排除（别重复）：barrier 覆盖范围、提交粒度、Q8 暂存别名（w3 已加独立
`q8_up`）、region 重叠（bump allocator 不重叠）、validation layer（零警告，代码合规）。
真正的根因在 silu 的数值，不在调度。

`FFN_BATCH_MAX=<n>` 可限制合批生效的层数（默认全部），用于二分定位。

**遗留**：合批后与 CPU 基线 PSNR 31.61 dB（改动前 33.34 dB），差异来自 TODO-020
风险 1 的 silu 饱和。TODO-023 的 quantize reciprocal 修复已把该值从 31.52 拉回
32.21 dB（mean|Δ| 3.44 → 3.01），剩余差距仍主要来自 silu 饱和边界。

### TODO-022: Z-Image Q/K norm + 交错 RoPE 上设备 — ✅ 已合入

`qk_norm_rope.comp` 是 **NeoX** 布局（`lane` 与 `lane + head_dim/2` 配对），DiT 的
`rotate_interleaved_inplace` 是**交错**相邻对（`chunks_exact(2)`），两者索引不同，
不能复用。新增 `shaders/glsl/qk_norm_rope_interleaved.comp`：norm 段逐行照抄
NeoX 版（含保持 CPU 逐位一致的串行 Kahan 归约），只把旋转索引换成 `2*lane` / `2*lane+1`。

管线索引 `QK_NORM_ROPE_INTERLEAVED = 28`，**追加**在数组末尾——`OPERATOR_SHADERS`
每个位置都是硬编码常量，插入会静默改绑别人的 SPIR-V。

**收益**：QKV 不再往返。`rope (host)` 893.8 ms → 0.3 ms，denoise 43.1 → 42.0 s。
比先前「传输值 1.9–3.9 s」的估计低：177 MB 单向传输在这台机器上约 0.4 s。

**风险 1（已修）：跨块行步长 ≠ 行内 span。**
DiT 的 QKV 是**逐行交错** `[Q|K|V]`（`QKV_WIDTH`），而 Qwen3 是**分块** `[整块Q|整块K]`。
照搬分块语义把 `q_heads*head_dim` 当成行跨度传进去，着色器按 `q_stride=3840` 寻址，
而真实间距是 `QKV_WIDTH=11520`——于是**只有第 0 行落在正确位置**，其余行整体错位。
因为行跨度整除缓冲长度，**不会越界报错**，只会静默写坏 K 和 V。
修复：record 函数新增显式 `q_row_stride` / `k_row_stride` 参数。

**风险 2（已修）：行内 Q→K 跨度。**
Q 与 K 同属一个 region，仅靠 `k_word` 无法区分，K 每行基址还缺 `HIDDEN` 的偏移。
修复：push constant 新增 `k_base`。

两条都是自检抓到的，不是渲染对比发现的。**若只做「整图看起来一样」的核对，
这两个 bug 都可能以「V 被污染但肉眼看不出」的形式溜过去。**

`gpu_interleaved_qk_norm_rope_matches_the_host_chain` 钉住：
生产规模（3840 token × 30 head × 128）下与 host 的 `rms_norm_inplace` +
`rotate_interleaved_inplace` **逐位一致**，并额外断言 **V 与输入逐位相同**
（V 未被触碰是 K 基址算错的唯一可观测信号）。

### TODO-023: Vulkan Q8_0 量化倒数与 CPU 参考差 1 ulp — ✅ 已修复

`vk_ops_check` 的 `check_quantize_tie_even` 一直失败：
`mismatch at 0: gpu=-21 cpu=-20`。与 RoPE 那次不同，这次**不是新引入的**，
自 2026-09-08 的 `621c622` 起就在失败。

**根因**：倒数算的形式不同。

| | scale | inverse |
|---|---|---|
| CPU (`quantize_q8_0_into`) | `amax / 127.0` | `1.0 / scale` |
| GPU (`quantize_q8_0.comp`) | `divide(amax, 127.0)` | `divide(127.0, amax)` |

两者数学上相等，但 `1/(amax/127)` 与 `127/amax` 差 1 ulp。对测试输入
`0xbdbf0aec / 0x3f13f10a`，这个 ulp 让乘积从**恰好 -20.5** 变成 -20.500002——
本来是个 tie，round-to-even 应给 -20（与 CPU 一致），现在不再是 tie，直接给 -21。

**修复**：inverse 改为从已算出的 `scale` 出发，`divide_finite_normal(1.0, scale)`。
`divide_finite_normal` 本身是精确舍入除法，与 CPU 的硬件除法一致。

**代价为零**：两种写法都恰好做两次精确除法（算 scale 一次、算 inverse 一次），
只是第二次除法的形式变了。

**收益**：GPU 量化与 CPU 参考逐位一致，且**整体画质变好**——512×512 8 步同 seed
对比 CPU 渲染，PSNR **31.61 → 32.21 dB**，mean|Δ| **3.44 → 3.01**（降 12%）。
注意 GPU 修复前后两图之间 PSNR 只有 30.01 dB：扩散 8 步会把任何微小差异放大成
完全不同的轨迹，**不能用「与上一版 GPU 比」判断好坏，只能对 CPU 参考比**。

### TODO-024: attention 相关自检的容差是照 x86 标定的 — ✅ 由 TODO-025 一并解决

**根因**：`cpu_attention_scores` / `cpu_attention_values` /
`check_attention_scores_match_cpu_reduction` 拿 `dot_f16` 当 oracle，而

| 架构 | `dot_f16` | 累加 |
|---|---|---|
| x86_64 | `_mm256_cvtph_ps` + `_mm256_fmadd_ps` | **f32** |
| aarch64 | `fmla .8h` | **f16** |

容差是照 **x86 的 f32 参考**标定的（`attention_scores` 用 3e-5），在 ARM 上参考自带
`sqrt(head_dim) * eps_f16` 误差（head_dim=128 时约 5e-3），与实测 6.6e-3 吻合——
**原理上过不了**。

**解法不在放宽容差，在换 oracle**：统一改用 `reference_dot_f16`（f64 精确求和）。
oracle 是精确算术后，容差自然回到 3e-5 并且真的成立。详见 TODO-025。

**教训**：一个有损的参考值，会让「校验 GPU」这件事悄悄变成「比较两个有损实现」。
前者能发现 bug，后者只能发现巧合——而且它同时把容差推大到连真正的 GPU 回归都测不出来。

### TODO-025: `dot_f16` 在 aarch64 上用 f16 累加 — ✅ 已改，`ggml-f16-dot-parity` 保留旧行为

新增 Cargo feature `ggml-f16-dot-parity`（默认关闭）：

| | 累加 | 指令 | 吞吐(n=4096) | 相对误差 |
|---|---|---|---|---|
| 默认 | **f32** | `FMLAL` | 31 GFLOP/s | **8.5e-08** |
| `--features ggml-f16-dot-parity` | f16 | `fmla .8h` | **122 GFLOP/s** | 2.3e-03 ~ 3.4e-03 |

**吞吐代价实测约 4 倍**，这台机器的 FP16 单元确实比 f32 宽。所以这不是免费的
正确性提升，而是一个需要按调用点分别权衡的取舍——feature 就是为此留的：
校验/参考路径用默认档，吞吐敏感的推理路径可以显式打开旧行为。

（测量用的是逐点测试，n=4096、2 万次迭代；FMLAL 档每 32 元素用 8 次 `.4h`
装载而非 4 次 `.8h`，装载指令是 2 倍，因此其中一部分差距可能来自装载而非
FMLAL 本身吞吐。）

#### 「保留 16 精度」不等于「保留 16 位累加」

现状把两件事捆在一起了，但它们可以分开：

- **f16 存储**：模型格式与 ggml 对齐的要求，**任何配置下都不变**，带宽与显存
  收益完全保留。
- **f16 累加**：实现细节，**不是对齐要求**。决定性证据是 x86 路径
  （`_mm256_cvtph_ps` + `_mm256_fmadd_ps`）**从来不**做 f16 累加。同一份 ggml
  语义，两个架构行为不一致，说明这从来不是「对齐 ggml」，而是各自挑了个实现，
  再被一个逐位测试冻住。把 `fmla .8h` 当作对齐的代价保留下来，等于主动接受一个
  和 x86、和 GPU shader 都不一致的数值行为，只因为它先被写下来了。

`FMLAL` 是 Armv8.2 里「f16 乘、f32 累加」的正规指令，与 `fmla .8h` 同属 FEAT_FP16，
**不引入新的架构要求**。

#### 顺带修好的：自检参考不再依赖 build flag

原先 `cpu_attention_scores` / `cpu_attention_values` / `check_attention_scores_match_cpu_reduction`
拿 `dot_f16` 当 oracle——也就是拿一个「快速实现」当标准答案。累加器宽度一旦变成
构建选项，**打开 flag 就会让检查自动变弱**，而且它问的本来就不是「shader 对不对」，
而是「两个有损实现会不会碰巧对上」。

现在统一改用 `reference_dot_f16`（f64 精确求和）。结果：

| 检查 | 改前 | 改后 |
|---|---|---|
| attention_score_reduction | 容差被迫放大到 2e-3（现收回 3e-5）| **max_rel 0.000e0** |
| attention_scores | 容差 3e-5，实测 6.6e-3 失败 | **max_rel 3.4e-07** |
| attention_values | — | **max_rel 0.000e0** |
| quantize_tie_even | 自 621c622 起一直红 | exact=true |

`vk_ops_check` 现在 **exit 0、25 个算子全过**（连跑多次稳定）。

#### 被推翻的 pinned 测试

以下测试原本钉死 ggml 的 f16 累加结果，现改为按 feature 钉两套值，并补上
**对 f64 参考的精度断言**（只钉 bit 只能证明可复现，证明不了正确）：

- `neon_f16_attention_dot_matches_ggml_four_accumulator_reduction`
  → 仅在 `ggml-f16-dot-parity` 下编译；新增
  `neon_f16_attention_dot_accumulates_in_f32`（默认档，对 f64 参考 rel < 1e-6）
- `f16_dot_dispatch_matches_platform_reduction_contract` → 两套值 + f64 精度断言
- `f16_kernel_matches_ggml_scaled_input_contract`、
  `f16_kernel_scaled_path_matches_ggml_f16_input_and_native_accumulation`
  → 两套值 + 新增 `assert_f32_accumulated` helper
- `z_image::vae::conv_f16_matches_pinned_ggml_im2col_dot_and_post_bias`
  → 两套值（该测试**只**钉了值未补 f64 断言；精度由上一层的 dot 级测试保证）

#### 回归

`cargo test --lib`：默认与 `ggml-f16-dot-parity` 两个配置均为
**1137 passed / 29 failed**，与父提交**完全一致**，零新增失败。

#### 端到端代价（512×512 8 步，Z-Image）

| 阶段 | 改动前 | 改动后 |
|---|---|---|
| CPU vae_decode | 11.6 s | **15.2 / 15.9 s（+34%）** |
| CPU denoise | 波动极大 | 无关（本就不走 `dot_f16`）|
| GPU 端到端 | 48.99 s | 48.59 s（不变）|

**VAE 的 +34% 是真实回退**，两次复测都稳定复现：VAE 的 f16 conv 走
`F16Kernel::forward_scaled` → `dot_f16`，正好落在 FMLAL 慢 4 倍的那一档。
GPU 端到端不受影响，因为 DiT 走 Q8_0 matmul。

同时 **GPU/CPU 一致性改善**：PSNR 32.21 → **33.50 dB**，mean|Δ| 3.01 → **2.87**。
CPU 侧少了一个 f16 累加的误差源，自然更靠近 GPU。

所以这一档的取舍是明确的：**CPU VAE 多花 34% 时间，换 CPU/GPU 差 1.3 dB**。
两次 CPU 渲染逐字节一致，结果可复现。

#### 尚未验证

- ASR（gemma4 / qwen3 mel encoder）与 MLX LoRA 的端到端数值未跑（缺模型）。
  这些是 `dot_f16` 的真实调用方，其中 ASR 的 mel encoder 可能落在算力敏感区，
  4 倍差距是否可接受需要实测决定。
- FMLAL 档的装载指令数是否为瓶颈（见上）。

### TODO-026: 测试套件在满负载下有 flaky，且 CI 根本不跑测试

#### 现象

同一个未改动的提交，重复跑全量 `cargo test --lib --features vulkan`，失败集合会变。
2026-10-13 在 `ZImage3`（`8297e19`）上实测三次：

| 轮次 | 相对首次基线多出的失败 |
|---|---|
| 1 | 无 |
| 2 | `models::dots::vocoder::q8_convolution_matches_scalar_layout_bias_stride_dilation_and_batch_tail` |
| 3 | `models::qwen3::trunk::qwen3_cpu_prefill_matches_batch_one_at_chunk_boundaries` |

两个测试**单独跑时 13/13 全过**（改动前后各 8 次与 5 次），只有全量并行时才偶发。
已确认至少这两个，**完整 flaky 集合尚未枚举**。

#### 为什么这条比看起来严重

全量回归目前只能给出「与基线比对」这种相对结论，而这个基线本身在漂。于是：

- 任何一次单轮全量回归的结论都不可靠 —— 改坏了分不清是真回归还是 flaky，没坏也可能是 flaky 抵消了真回归
- 已经在实践中造成误判：本次差点把 `--features vulkan` 的结果（1220 passed / 39 failed）
  去比**不带** vulkan 的基线（1137 / 29），把 10 个正常差异当成回归；之后又差点把
  `qwen3_cpu_prefill` 的偶发失败归到自己的改动上，实际它在基线上同样会飘
- 新增测试的通过率也不可信：`silu_probe_uses_the_production_ffn_width` 这类断言，
  理论上「一次通过」不构成证据

#### 为什么一直没被发现

`.github/workflows/ci.yml` 只有两个 job：`fmt`（`cargo fmt --check`）和
`vulkan`（shader 校验 + `cargo check` 若干 target）。**没有任何 job 跑测试。**
所以这类问题只能靠本地人工回归撞上，而人工回归恰恰是最容易漏判的场景。

#### 推测成因（未验证）

两者都涉及线程池与分块边界：一个是 `ComputePool` 下的分块量化卷积，一个是
chunk 边界的 prefill。满负载时线程调度/分块顺序变化可能触发只在特定顺序下
才成立的假设。需要看代码确认，暂不下结论。

#### 待办

1. **枚举 flaky 集合**：对全量套件连续跑 N 轮（如 10），统计每个测试的失败频次，
   产出「偶发失败」名单
2. **定位根因**：逐个看名单上的测试是否依赖线程调度/分块顺序；能修则修
3. **暂时隔离**：修不了的先用 `#[ignore]` 或加显式注释说明已知偶发，
   并在名字/注释里写清触发条件，避免下一个人当回归查
4. **CI 补测试 job**：`cargo test --lib`（带/不带 `--features vulkan` 各一），
   失败重跑一次再判定——先把「有测试」这件事建立起来，重试策略随后再加
5. **方法论固化**：在 `docs/develop/OPTIMIZATION.md` 或本文档里写明
   **禁止用单轮全量回归做回归判定**，需要至少 3 轮且失败集合稳定才可下结论

### TODO-027: FFN 尾部的 norm/gate/residual 融合上设备 — ❌ 未完成，实现已丢弃，但四个坑要记住

目标：`readback_norm_and_residual_rows` 在每个 block 末尾把投影结果下载到 host、
做 rms_norm + 门控残差加、再传回设备。512×512 下每 block 每 site 是 59 MB 下行 +
59 MB 上行，一个 step 走两次。融合成单个 kernel 可以省掉其中一次。

**实现已完成到「自检通过」这一步，但接进主路径后成图是坏的（PSNR 10.03 dB，
改动前 GPU↔CPU 是 33.5 dB），因此没有提交。** kernel 本身在生产规模下是逐位正确的，
自检钉住了这一点；错的是它周围的接线。下面四个坑每一个都真实造成过可观测的失败，
而且**每一个的失败方式都不是「报错了」，而是「继续跑出一个看起来合理的错结果」**。

#### 坑 1：等长的两个 bound buffer 会被 glslang 折叠成一个

gamma 和 gate 都是 `HIDDEN` 宽、从 slot 1 和 slot 2 绑定、offset push 常量都是 0、
读取长度相同——编译器因此证明两次读等价，把 gate buffer 从 SPIR-V 里整个删掉：

    $ spirv-dis shaders/bin/fused_norm_gate_residual.spv | grep Binding
    OpDecorate %arena Binding 0
    OpDecorate %weight Binding 1        # Gate 整个不见了

于是 shader 用 gamma 当 gate 读，产出一张 PSNR 9.9 dB 但**结构完整**的图。
试过把索引写成不同表达式（`channel = offset % pc.width` 而不是 `offset`），
**无效**——折叠与表达式无关。最后把两个都改成从 arena 读、用各自的 word offset 才解决。

> **教训**：新增一个 bound buffer 后，用 `spirv-dis ... | grep Binding` 确认它还在。
> 这一步不能省，也不能靠肉眼 review shader。

#### 坑 2：`ArenaRegion.size` 是字节，不是元素数

`Layout::build` 里写 `gate_tanh: take(HIDDEN)?` 而其他行宽区域写的是
`rows * HIDDEN * 4`。`take` 收的是元素数，所以 3840 元素被登记成 3840 字节，
而 `write_f32` 要 15360 字节。结果每层都失败并 fallback：

    [GPU] Z-Image block 0 fell back to the CPU: ... host write region
          offset=8785664 size=3840 cannot hold 15360 bytes

denoise 从 41 s 变成 **3176 s**（77 倍）。自检里没暴露，因为自检用了一个
`region(slot, elements) -> size: elements * 4` 的闭包，**而 session 里是手写的字面量**。

> **教训**：这类错误自检抓不到，因为自检和生产用了两套不同的写法。让它们共用同一个分配函数。

#### 坑 3：`bind_buffers` 从一个永不回收的 pool 分配

在每 block 的调用里 `bind_buffers(&[])` 拿一个新 descriptor set。pool 是固定大小、
分配后不释放，所以第 19 层开始 `ERROR_OUT_OF_POOL_MEMORY`，之后每一层都 fallback。
profile 里表现为「前 18 层正常、之后全崩」。

现有的 `bind_norm` / `bind_modulation` 就是 per-(layer, kind) 缓存的，照抄即可。

> **教训**：`bind_*` 一律 per-layer 缓存。这在仓库里已有先例，不必重新发现。

#### 坑 4：`take_gpu_session` 会把 session **取走**

每步开头需要把残差上传到 arena，我这样写：

```rust
if let Some(mut session) = take_gpu_session(scratch, total_tokens, &mut gpu_enabled) {
    session.upload_residual(...)?;      // session 在这里被 drop
}                                       // 没有放回 scratch
```

`take_gpu_session` 的最后一句是 `scratch.gpu.remove(&rows)`——它**移除**而非借用。
作用域结束 session 被丢弃，循环里每次 `take_gpu_session` 都新建一个，于是
**残差永不跨层累积**，每层都从初始 latent 重新开始。

> **教训**：`take_gpu_session` / `put_gpu_session` 这对「借出」语义的函数，调用后必须归还。
> 名字里的 `take` 已经暗示了，但很容易读成「获取一个引用」。

#### 还有一个没定位的错误

修完上面四个，PSNR 仍是 10.03 dB。**没有找到根因**，不排除还有第五个接线问题。
没有继续猜下去——按前面 TODO-025 的教训，猜测式修复在这里已经失败过一次。

已确认与改动无关的对照：CPU 路径正常（459.7 s 出图正常）。

#### 已经确认正确、可以复用的部分

这些是这次调查的确定结论，即使实现被丢弃也有价值：

1. **参考必须是 `rms_norm.comp`，不是 `rms_norm_inplace`。**
   `rms_norm_inplace` 经由 `sum_sq_f32` 归约，而它返回 **f64** 并保持 64 位累加器
   （NEON 路径）。这是比任何 f32 GPU kernel 都更准的归约，**逐位一致根本达不到**——
   现有的 `rms_norm.comp` 从第一个元素就和它不一致。
   真正该对齐的是 `rms_norm.comp`：它是未融合路径实际跑的 kernel，
   也是残差流的来源。对齐它，融合才是数值上的 no-op。
   （本条修正了最初的假设，最初以为可以对齐 CPU 参考，结果自检永远过不了。）

2. **`tanh` 必须留在 host。** gate 是逐通道的（`HIDDEN` 宽，不是每行一个），
   而 GLSL 的 `tanh` 不保证和 Rust 的 `f32::tanh` 同位模式。在 host 算好再上传即可，
   代价只有 3840 次 tanh。

3. **无 AdaLN gate 的 refiner 用全 1 填充是数学等价的**：`add_modulated_residual`
   的 `None` 分支是 `tokens[i] += residual[i]`，而 `residual * 1.0 == residual`。

4. **数据流是自洽的**：`record_ffn` 之前会把 `attention` 重新上传到 `layout.x`，
   所以 FFN 输入不是 block 开头的那个陈旧值。

5. **normed 输出对主栈是死的**：`record_ffn` 直接吃 GPU 上的 `layout.x`，
   两个 site 写进 `attention` / `qkv` 的 normed 值，主栈都不读。
   真正需要回 host 的只有残差流。这决定了融合的价值上限。

#### 性能（实现本身是有效的，只是没跑对）

`w1 readback` 从 **1109 ms → 346 ms**。denoise 净持平（41.2 → 41.3 s），
因为省下的投影回读换成了残差回读——**除非残差也能跨层常驻**，
而那要求 session 在整个 step 生命周期内不被取走（见坑 4）。

所以这条路线的真实收益上限，取决于能否把残差真正留在设备上跨 34 层累积。
这需要一个比「每层 take 一次 session」更清晰的 session 所有权模型。

### TODO-028: f32 的 aarch64 NEON kernel 名不副实，BGE 在 ARM 上拿不到 SIMD 加速

在 `codex/bge-embedding-gguf` 上审 BGE embedding 支持时发现。BGE 的 decoder 变体
（Mistral / Qwen2 / Gemma2）走 `LlamaSession`（`src/models/decoder_embedding.rs`），
f32 权重落到 `src/ops/kernel/f32/neon.rs::matmul_f32_vs_f32_neon`。

**SIMD 路径是有的**（`forward_f32_rows_dispatch` 会分派过去，`dot_f32` 也有 NEON
实现），所以不是「缺 SIMD 路径」，而是**那个 kernel 基本不是 SIMD kernel**：

```rust
let product = vmulq_f32(
    vld1q_f32(weight_ptr.add(row_off + index)),
    vld1q_f32(input_ptr.add(index)),
);
total += vget_lane_f32(vget_low_f32(product), 0);   // 立刻拆回标量
total += vget_lane_f32(vget_low_f32(product), 1);
total += vget_lane_f32(vget_high_f32(product), 0);
total += vget_lane_f32(vget_high_f32(product), 1);
```

文件头写着「4-wide FMA per inner iteration」，但实现里**没有向量累加器、没有 FMA**：
每 4 个元素做一次 SIMD 乘，紧接着 4 次 `vget_lane` 把结果抽回标量相加。
相对纯标量只省了乘法，省不掉抽取开销和标量累加。

同仓库的姊妹 kernel 是对的：

| kernel | 向量累加器 |
|---|---|
| `f32/avx2.rs`（`avx2_matmul_packed!` 宏）| `_mm256_fmadd_ps` + 双累加器 |
| `bf16/neon.rs` | `vdupq_n_f32` + `vfmaq_f32` + `vaddvq_f32` |
| **`f32/neon.rs`** | **无**，逐 lane 抽回标量 |

`f32/neon.rs` 自己写着：

> `TODO-005: once a CI target exposes aarch64, validate against the scalar reference`

而本仓库 CI（见 TODO-026）**没有任何 aarch64 runner**，所以这个 kernel 从未被验证过。

#### 为什么写成这样（不能直接当 bug 改）

代码里有明确的动机：

> Keep the scalar left-to-right accumulation contract. A vector accumulator
> changes the rounding order enough to cross BF16 boundaries in the autoregressive
> depth decoder.

改成 `vfmaq_f32` 会改变求和结合律 → 数值不再与标量一致。所以这是**有意的精度选择
换来的性能放弃**，不是疏忽。

#### 影响面

- **BGE embedding 在 aarch64 上拿不到 f32 SIMD 加速**，性能接近标量
- 同样影响任何 f32 权重走这个 kernel 的 decoder 模型（本机是 aarch64）

#### 三个方案（未实施）

1. **照 `bf16/neon.rs` 用 `vfmaq_f32` + 4 累加器** —— 最快，但求和顺序改变，
   结果**不再 bit-exact**。embedding 的 cosine 相似度通常不受影响，但这是需要
   实测确认的判断，不是可以假定的。
2. **维持现状** —— 数值契约不动，aarch64 上没有 f32 SIMD 收益。
3. **分路径**：保留严格标量路径给自回归解码，另加一条 embedding 专用的 SIMD 路径。
   **这个方案最稳**——embedding 的容差天然比逐 token 解码宽，
   且不需要牺牲 decoder 的精度契约。

若要推进，先做实测：同一批文本比对三种方案的 embedding 余弦相似度与端到端
检索一致性，再决定。**不要在没有实测的情况下放宽 decoder 的求和契约。**

### TODO-029: BGE decoder embedding 路径的 logits 投影是死代码，且穿透读取内部 scratch

审 `codex/bge-embedding-gguf` 的 `bafc23c` 时发现。该提交把
`LlamaSession::forward_one_token` 的函数体抽成 `forward_input`，加了三个入口：

```rust
forward_one_token        → forward_input(token_id, None, None, /*project_logits*/ true)
forward_embedding        → forward_input(0, Some(emb), None, false)
forward_embedding_mrope  → forward_input(0, Some(emb), Some(mrope), false)
```

重构本身是对的（参数化优于复制三份），`mrope` 的 section 之和必须等于
`n_embd_head_k / 2` 这个校验也写对了。以下两点是「可以更好」，都不影响正确性。

#### 问题 1：`project_logits=false` 对 embedding 路径是死代码

`decoder_embedding.rs` 走的是 `session.prefill(&tokens, 1)`，而 `prefill` 是：

```rust
last_logits = self.forward_chunk(input, rows, base, is_last)?;
```

**不是** `forward_one_token`。`project_logits` 的实参是 `is_last`，即**最后一个 token
仍然会投影出完整 logits**（vocab 通常 32K–150K 宽），而 `decoder_embedding` 只要
`session.scratch.x`，这批 logits 算完直接丢弃。

新加的 `if project_logits { ... }` 包裹只对 `forward_one_token` 有意义，而那条路恒传
`true`。所以 embedding 场景省不下任何东西。

**要真正省掉**：需要让 `prefill` 的最后一个 chunk 也能传 `project_logits=false`，
即 `prefill` 增加一个「只要 hidden state」的入口。当前 API 表达不了这个意图。

**收益可量化**：每次 embedding 请求省一次 `n_embd × vocab` 的投影。0.6B 级模型
vocab 约 150K、n_embd 1024，即 1.5 亿次乘加，而 embedding 的实际计算量只有几层 decoder。

#### 问题 2：`decoder_embedding` 穿透读取 `session.scratch.x`

```rust
rms_norm_grouped(
    &session.scratch.x[..n_embd],   // 跨模块读 impl 的内部字段
    &session.weights.output_norm,
    ...
);
```

两处问题：

1. **封装**：`scratch` 是 `LlamaSession` 的内部状态，从 `decoder_embedding.rs` 直接
   穿透访问，读的是「forward 恰好停在某个中间态」的隐含约定。
2. **重复计算**：`forward_input` 内部已经算过 `rms_norm_grouped`（结果在 `normed`），
   embedding 侧又从 `scratch.x` 重算一遍同样的 norm。

**更干净的做法**：session 暴露一个 `last_hidden(&self) -> &[f32]`，
或者在 `forward_input` 尾部把 pooled 后的结果落到一个专门的字段。
既不用穿透 `scratch`，也不用重算 norm。

#### 与 TODO-028 的关系

`TODO-028` 记录了 f32 在 aarch64 上拿不到 SIMD 加速。
本条的「省掉 logits 投影」和它是同一个方向的两端：一个是**别做多余的计算**，
一个是**把要做的算快**。对 BGE 这类 embedding 负载，前者的收益可能更直接——
vocab 投影通常比 decoder 层本身更贵。

### TODO-030: Visualized BGE 的 embedding 构造复制了 `populate_embeddings`，且 CLS 位置是隐含假设

`bafc23c` 给 `bert_family` 加了视觉分支（`run_embedding_tokens_with_image`）。
核心重构是**对的**：`ec62356` 把 embedding LayerNorm 从 `populate_embeddings`
搬到 `forward_bert_layers` 开头，因为视觉路径要自己构造 `hidden`，而 LayerNorm 是
两条路共用的。搬完之后 text 与 visual 共享同一份 LayerNorm，并保留了
`bert.cpp:92` 的溯源。同一个提交里 `n_ctx_train` 的长度检查从 `token_ids.len()`
改成含图像行的 `n_tokens`，也是必要修正 —— 原来只查文本 token 数，加了图像会漏检。

以下两点是维护隐患，不影响当前正确性。

#### 问题 1：embedding 构造有两份实现

`populate_embeddings`（text）与 visual 分支的内联代码，各自实现了一遍
「token embedding + segment + position」：

```rust
// visual 分支内联手写
embedding_lookup(weights.token_embd, *id, h, weights.token_embd_ggml_type, row);
for j in 0..h {
    row[j] += segment.as_ref().map(|s| s[j]).unwrap_or(0.0);
    row[j] += position[t * h + j];
}
```

风险不在于现在会算错，而在于**改了一处另一处不会跟着变，而且不会有编译错误**。
不同 BGE 变体的 position/segment 规则未必相同（图像 token 是否加 segment、
segment 是否只加给文本），这类差异正是最容易只改一边的。

**建议**：抽出一个共用 helper，visual 只额外传「哪些行是图像」：

```rust
fn populate_embeddings_with_overrides(
    cfg: &BertConfig,
    weights: &BertWeights<'_>,
    token_ids: &[u32],
    image: Option<&[f32]>,
    image_rows: usize,
    hidden: &mut [f32],
) -> Result<(), String>
```

#### 问题 2：token 位置映射依赖未被断言的 CLS 假设

```rust
for (i, id) in token_ids.iter().enumerate() {
    let t = if i == 0 { 0 } else { i + image_rows };
    ...
}
```

`i == 0` 的特殊处理隐含假设 **token 0 永远是 CLS**，但代码里没有任何地方检查。
布局本身自洽（图像占 `1..=image_rows`，文本从 `image_rows + 1` 起），可一旦某个
tokenizer 不加 CLS、或加了别的 prefix token，映射就会静默错位 —— 不报错，
embedding 全错，且这类错误通常要到检索结果变差时才被发现。

**建议**：显式校验 token 数与图像行数的对应关系，并把「token 0 是 CLS」写成
一条可读的断言或注释约定，而不是只靠 `if i == 0` 这个分支表达。

#### 一个正面观察

`position` 表的校验写得很严，值得保留：

```rust
if info.dims.len() != 2 || info.dims[0] != h as u64 || info.dims[1] < n_tokens as u64 {
    return Err("invalid Visualized BGE position table".into());
}
```

dims 数量、宽度、以及**行数是否覆盖 `n_tokens`** 三者都查了 —— 这正是
「不查就会静默越界」的地方。

### TODO-031: Qwen3 resident Vulkan decode 在 GB10 上打不过 CPU，且默认根本不启用

排查「Qwen3 为什么没有 GPU 加速」时挖出三层问题，从表面到根因。

#### 1. 默认配置下 `--gpu` 是静默空操作（已修，a66705f）

Qwen3 唯一的 GPU 路径是常驻 Vulkan 解码器 `Qwen3VulkanSession`
（`src/vulkan/qwen3.rs`），它**要求 F16 KV**。而 CLI 默认是 F32
（`src/app/text/runtime.rs:50` 的 `defaults::KV_FORMAT = KvFormat::F32`），
于是 `src/models/qwen3/trunk/session.rs` 里 `compute_policy.context()` 那段
直接落到 `None` 分支，**不报错、不提示**，全程 CPU 跑完，日志里还照样打印
`[GPU] Vulkan device: NVIDIA GB10`。

```rust
} else if compute_policy == ComputePolicy::Vulkan {
    return Err("Qwen3 Vulkan currently requires F16 KV".into());
} else {
    None          // Auto + F32：请求被静默丢弃
}
```

`a66705f` 改成明确提示并给出正确 flag。**在此之前，本文件之前所有关于
「Qwen3 GPU 加速」的结论都是无效的** —— 那些数字测的是 CPU。

#### 2. 即使显式 `--kv-cache f16`，GPU 也赢不了

在 GB10（NVIDIA GB10，int8 dot product: supported）上实测，
同一台机器、同一 prompt（220 词）、`--threads 8`、等待 load<3、
多次重复取中位数：

| 模型 | 大小 | 生成 ratio (GPU/CPU) | 端到端 ratio | wall CPU | wall GPU |
|---|---|---|---|---|---|
| Qwen3-0.6B-Q8_0 | 0.6G | 0.46x | 0.34x | 2.8s | 8.7s |
| Qwen3-0.6B-Q4_K_M | 0.4G | 0.11x | 0.11x | 3.9s | 33.8s |
| Qwen3-0.6B-BF16 | 1.1G | 0.38x | 0.43x | 5.9s | 15.7s |
| Qwen3-4B-Q4_K_M | 2.3G | 1.08x | 0.14x | 15.7s | 113.4s |
| Qwen3-4B-Q8_0 | 4.0G | **1.08x** | 0.56x | 16.8s | 31.2s |
| Qwen3-8B-Q4_K_M | 4.7G | 0.19x | 0.11x | 35.1s | 398.5s |
| Qwen3-8B-Q8_0 | 8.1G | **1.02x** | 0.54x | 49.0s | 100.1s |
| Qwen3-14B-Q4_K_M | 8.4G | 0.97x | 0.56x | 72.7s | 138.0s |

两个独立的结论：

- **生成（steady state）在 4B 以上才勉强追平**，逐次 ratio 紧密聚在 1.0 附近
  （4B Q8_0: 1.28 / 1.00 / 0.98；8B Q8_0: 1.00 / 1.02 / 0.94），
  **不是加速，是持平**。
- **端到端在所有尺寸上都是输的**（0.11x ~ 0.59x）。因为常驻解码器要先
  付一次性启动成本：~0.4s 的 pipeline JIT + 把整个权重上传进显存。

启动成本拆解（Qwen3-0.6B-Q8_0，`RMI_SETUP_TRACE`）：

```
Qwen3Ops::new (pipeline/descriptor)  211.0 ms
权重上传+bind  633.5 MB               169.0 ms
core_contract 校验                     31.5 ms   ← 已是 OnceLock memoize
q8_0_activation_contract 校验           6.4 ms   ← 同上
```

契约校验那两项**已经**是 `OnceLock::get_or_init`，只跑一次，没有优化空间；
pipeline JIT 和权重上传是一次性 CLI 进程无法摊销的成本。

#### 3. 根因：dense GEMM 在 M 很小时被饿死

用 `RUST_GPU_SUBMIT_TRACE` 测 Qwen3-8B-Q8_0 的 prefill：
一次 prefill 只产生 ~36 次 submission（**分块是合理的**，不是提交过碎），
但每次 `wait` 平均 **188 ms**，而驱动调用 `submit` 只有 35 µs。
即整条链路的开销不在提交，而在 GPU 执行本身。

折算下来整趟 forward 对 8GB 权重只跑到约 **1.2 GB/s 有效带宽**；
而同一块卡上 LM head 那一类（`rows=1`, `n_out` 很大）能跑到 22 GB/s。
差别在于 **M（行数）**：prefill M≈60、decode M=1，对 GPU 来说都太小，
填不满 SMs，既算不出强度也吃不到带宽。

对比同机 CPU：8B prefill 11.3 t/s vs GPU 5.7 t/s —— **CPU 快一倍**。
这台机器的 CPU（含 AVX2/FMA 路径）在这个工作负载上就是比这条 Vulkan
dense 路径强。

#### 为什么先不写「按模型大小决定是否上 GPU」的阈值

本来想用上面的表定一个尺寸阈值，但数据不支持：**4B 以上生成只是持平，
没有出现真正的交叉点**，而且持平的区间里 e2e 依然全输。也就是说
「小模型不上 GPU」这个结论虽然成立，但真正的问题是
**常驻 dense 路径本身在这个硬件上就没赢过**，加阈值只能掩盖症状。

要真正修好，得动 dense GEMM kernel 本身（M 小的时候的 tiling / split-K），
那是重写 shader 的量级，不适合在没有回归基线的情况下顺手做。

#### 附带记录

- **Q4_K_M 在 GPU 上尤其糟**（8B: 0.19x，wall 398s）。
  `GpuMatmulMode::Dot`（GEMV/split-K 路径）只对 F16/BF16 存在，
  Q4_K 只能走 `Prepared` -> `Q4_K_MATMUL`，没有 rows=1 专用 kernel。
  这和 `7750eeb` 里 LFM2.5 MoE 的结论是同一件事的两个入口。
- **8B-Q8_0 的 GPU 运行峰值 RSS 约 18.7 GB**，超过 15 GiB 预算；
  一次性 CLI 同时持有 mmap 权重和显存副本是原因之一。

### TODO-032: Q4_K/Q5_K/Q6_K 在 GPU 上没有 DP4A kernel，8B Q4_K_M 比 CPU 慢 17.6 倍

接着 TODO-031 往下挖，发现 Q4_K 家族在 dense Vulkan 路径上缺少向量化 kernel，
这是 Q4_K_M 特别慢的**结构性**原因。

在 `src/vulkan/ops.rs` 的 operator 表里：

- **Q8_0** 有两个 kernel：
  - `Q8_MATMUL_GROUPED`，其 index 会被 `Q8_MATMUL_GROUPED_DP4A` 顶替
    （`force_dp4a && context.supports_integer_dot_product()`，约 1423 行）；
  - 外加 `Q8_MATMUL_TILED_DP4A`（`rows > 1` 时 `record_weight_matmul_rows`
    自动切到 `record_weight_matmul_tiled_rows`，约 1960 行）。
  两者都吃设备 int8 dot-product，8B Q8_0 生成能持平 CPU。

- **Q4_K / Q5_K / Q6_K** 只有**一个**标量 kernel（`q4_k_matmul.comp` 等，
  表项见约 392 行），既没有 dp4a 变体，也没有 tiled 变体。
  8B Q4_K_M 实测 prefill 0.4 t/s、生成 1.3 t/s，wall 45.8s，
  同机 CPU 2.6s —— **慢 17.6 倍**，是所有测过的组合里最差的。

`q4_k_matmul.comp` 主循环（每 block）：

```glsl
int value = int((group & 1u) == 0u ? packed & 15u : packed >> 4u);  // 解 4-bit nibble
int activation = q8(block * 256u + local);
partial[(local & 31u) >> 2u] += int(scale(slot, offset, group)) * activation * value;
```

单块累加值量级 = `scale(≤63) * activation(≤127) * value(≤15)`，
再在 8 个 lane 上求和，落到 int32。这种**块缩放的 int32 乘加不能直接套
Q8_0 的 dp4a 路径**（Q8_0 的量化把乘积压在 int8 内）。所以 Q4_K 想吃
DP4A，得写一个自己的「块缩放 + int32 提升」的 int8 dot kernel，或者
把激活缩放到能进 int8 的范围 —— 这是一个完整的 shader 项目，不适合随手做。

结论与 TODO-031 合并看：**GPU dense 路径目前只有 Q8_0 勉强可用（持平 CPU），
Q4_K 家族完全不能用（慢 17.6 倍）。** 在补出 Q4_K 的向量化 kernel 之前，
Q4_K 模型应该明确走 CPU，而不是靠一条经验阈值去猜。

### TODO-033: Q4_K split-K GEMV 试过，行度不够；真正的瓶颈是单字节权重访问效率

接着 TODO-032，真的去写了一个 rows==1 的 split-K GEMV（`q4_k_matmul.comp`
里新增分支：rows==1 时不让 64 个 lane 各领一个输出行，而是让 64 个 lane
合作同一个输出行，按 32 元素一个 scale/min 小组切 K，最后树形归约）。
256 元素的 Q4_K block 正好切成 8 个 32 元素组共享一个 scale 和一个 minimum，
所以 lane 之间不需要交换 scale，每个 lane 内部就能把该组的 dot 算完。

结果：

- **数值**：对真实权重（Qwen3-8B-Q4_K_M 的 blk.0.ffn_gate），
  GEMV 输出与已提交（对 CPU 位精确）的 batched 路径 **逐字节一致**。
- **性能**：rows=1 只从 2.33ms 降到 2.11ms（~10%），LFM2.5 MoE 端到端
  1.1 t/s → 1.2 t/s，**基本没动**。因此这个分支没有保留（回退了）。

**为什么并行度不是瓶颈。** 8B prefill 每次提交 GPU 等 188ms（vs 驱动
35µs），整趟 forward 对 8GB 权重只跑出 ~1.2 GB/s；而同卡 LM head
（rows=1、n_out 很大）能到 22 GB/s。差异在于**单次权重字节的访问效率**，
不是 lane 数：

```
Q4_K rows=1:  2.06 MB 权重，实测 2.11 ms  →  0.98 GB/s
理论下限:     若能到 25 GB/s，只需 0.08 ms   →  差 25 倍
```

根源在 shader 的 `weight_byte()` / `q8()`：一次取一个字节
（`w.data[index>>2] >> ((index&3)*8)` 再掩码），nibble 再逐个拆
（`packed & 15` / `packed >> 4`）。加多少 lane 都还是在做字节级提取。

**真正要做的**：把权重读取向量化——用 `uint4`/`int4` 一次搬 16 字节，
在寄存器里用位运算拆 nibble 和 scale，让一次内存事务服务 16 个元素。
这是重写 Q4_K 读取路径的量级，和 TODO-032 是同一个大工程，建议合并做。
在那之前：
- batched（rows>1）的 Q4_K 已经被 `7747f07` 的 shared-memory staging +
  scale 提取外提提速 2.3~2.6 倍（位精确）；
- rows=1 的 Q4_K（MoE decode、单 token）仍然只有 0.98 GB/s，
  继续走 `7750eeb` 的 CPU 门控。

### TODO-034: Z-Image GPU 路径罕见地不可复现（同 seed 约 1/20 次输出不同）

在 GB10 上交替跑同 seed 的 Z-Image 256×256 八步，**大约每 20 次有一次输出不同**。
幅度很小：mean|Δ| 1.2/255、PSNR 41.5 dB（不是坏图，是舍入级差异），
其余各次逐字节一致。main 与 unified 分支都出现过，unified 修复前后都出现，
所以**不是本次 push constant 特化引入的**。

`main` 的历史记录里已经有同类现象（`docs/usage/z-image.md`「host 写 → device 读的
barrier」一节）：arena 是 CPU 直接写、shader 直接读的映射内存，缺失
HOST→COMPUTE barrier 时能否看见写入只取决于速度；旧 kernel 每次 dispatch 约 700 ms，
写入总来得及被「看见」，tiled kernel 约 64 ms 就可能错过。该记录同时说 512×512 才会
复现、256×256 逐位一致——**现在 256 也会出现，说明 tiled 化之后窗口更窄了**。

需要做的事：

1. 在 `VulkanContext::host_write_barrier` 之外，找出**哪一类** host 写入没有对应的
   barrier。`zimage_projection_kernel_costs`（`src/vulkan/ops.rs`，`#[ignore]`）
   可以先复现到单 kernel：同一形状连续录制多次，比较读回的激活是否逐字节一致。
2. 目标是让 256 与 512 都逐字节可复现，再谈「GPU 与 CPU 逐位一致」。
   在此之前，任何「GPU 输出逐字节一致」的测试结论都要注明运行次数。

### TODO-035: 补偿 exp 的价值只在 decoder 侧，DiT 侧已确认不值得

试过给 `softmax.comp` 加 `exact_exp` push 位、让 Z-Image DiT 走单精度 Horner
（`exp_cheap`），结论是**回退**：

- 性能几乎没赚：attention 相位只从 17.6 s 到 16.7–17.1 s（512，4 步）。同一个
  shader 里编进两套实现后，热循环里那个 select 把省下的算术吃掉了。
- 数值代价可测：DiT GPU 输出与同分支 CPU 参考的 PSNR 从 37.66 dB 降到 31.62 dB
  （mean|Δ| 1.76 → 2.47，256×256 八步 seed 42）。

所以 DiT 保留补偿版。剩下的开放问题是 decoder 侧：`exp_accurate` 的注释说
「1 ULP 的 exp 误差可能翻转 F16 概率平局」，而 `attention_values` 在两种模式下都会把
概率 `round_f16_rte` 之后才累加——补偿把 exp 做到 F32 最后一位，紧接着就被 F16
舍入吃掉了。要回答的是：**decoder 路径真正需要的是 exp 的精度，还是「舍入到 F16
之后不再变」这个性质**。如果只需要后者，把 exp 的误差压到远小于 F16 间隔（例如
1e-4 相对）就够了，补偿项可以砍掉一半。

复现数据与完整推导见 `docs/develop/ZIMAGE_OPTIMIZATION.md`。

### TODO-036: Qwen3-4B Q8_0 过不了 model contract，而 0.6B 与 8B 能过

`test-unified3`（unified-compute）上用 `--gpu --kv-cache f16` 跑 Qwen3 的三个
Q8_0 尺寸，model contract（`src/models/qwen3/trunk/session.rs`，
abs 2e-3 + rel 2e-3）的结果**按尺寸分裂**：

| 模型 | contract | 实测（Generation t/s，同会话） |
|---|---|---|
| Qwen3-0.6B Q8_0 | 通过 | 25.6 / 25.7 / 26.6 |
| Qwen3-4B Q8_0 | **失败**：`model logit mismatch at 0: gpu=9.235686 cpu=9.315228 rel=0.008538938` | 走 CPU，10.6 / 11.6 t/s |
| Qwen3-8B Q8_0 | 通过 | 4.7 / 5.0 / 5.1 / 5.3 / 5.5 |

4B 的相对偏差 8.5e-3 是容差的 4.3 倍，于是整条 GPU 路径被拒并回退 CPU。
**同一个模型家族、同一种量化，只换尺寸就翻脸**，说明这不是「Q8 路径整体不精确」，
而是某个尺寸触发了特定形状的偏差（4B 的 hidden 2560 / 36 层与 8B 的 4096 / 36 层
在 n_in 上不整除某些 chunk 边界，是第一个该查的方向）。

补充两点，避免重复劳动：

- 这个失败**与 push constant 特化无关**：修复前后的 contract 数值逐位相同
  （gpu=9.235686 cpu=9.315228），greedy 输出也逐字相同。
- `main` 没有这道 contract 门，4B 直接跑 GPU，但只有 1.4 t/s generation，
  比 CPU 回退的 10.6 t/s 慢 7 倍——**门是合理的**，问题在于 4B 的 GPU 数值
  偏差太大而不是 GPU 太慢。

同时记录：unified 分支的 decoder 路径比 `main` 快 6 倍以上
（8B Q8_0 generation 5.1 t/s vs 0.8 t/s），这是该分支的目标，回归要盯住这条线。

### TODO-037: `--threads` 等于核数时 CPU 吞吐塌 29 倍

Qwen3-0.6B Q8_0，20 核 GB10，greedy，64 token：

| `--threads` | prefill t/s | generation t/s |
|---|---|---|
| 12 | 280–335 | 82.8–88.4 |
| 16 | 303–347 | 67.5–97.2 |
| 17 | 281 | 52.2 |
| 18 | 295 | 69.3 |
| 19 | 312 | 69.0 |
| **20** | **46–48** | **2.4** |

悬崖正好落在 `threads == nproc`，可复现 3/3，且不是偶发：20 线程的 prefill 也只有
1/6。默认线程数是 8，所以平时碰不到，只有显式 `--threads $(nproc)` 才会踩到。

机制（`src/core/thread_pool.rs`）：`ComputePool::new(n)` 起 `n - 1` 个 worker，
调用线程也参与，总参与者 = n。worker 在空闲时是 `spin_loop()` 忙等（CPU 模式，
GPU 模式才 yield），而 `compute()` 的收尾是「自旋等 `n_complete == n_threads`」。
在 20 核机器上再叠一个别的租户进程占着核，参与者一多，barrier 就开始等调度器，
一个 token 被放大到 0.4 s。

**已修**（`src/core/thread_pool.rs`）：等待循环与空闲 worker 都不再硬自旋。

1. 新增 `wait_while`：先自旋 `WAIT_SPINS` 次（覆盖一次 hand-off 的微秒级窗口），
   之后改成 `yield_now()`。原来的 `spin_loop()` 无限自旋会让每个参与者一直处于
   runnable 状态。
2. 空闲 worker 同样改成「有界自旋 + `yield_now()`」。这条更关键：
   `compute_with_chunks` 每次调用都新起 `n_threads - 1` 个线程，20 线程时进程里
   同时有 42 个线程挤在 20 核上，而常驻 worker 的热自旋正好在抢这些线程要用的核。
3. `cargo test --lib core::thread_pool` 的两个用例覆盖了 Drop 与 chunk 语义
   （`single_thread_pool_drops_without_spinning`、`chunks_begin_at_zero_and_execute_exactly_once`）。

修完的效果（Qwen3-0.6B Q8_0，`--threads 20`，同会话交替）：

| | generation t/s |
|---|---|
| 修复前 | 2.5 / 2.4 / 2.9 |
| 修复后 | 65.1 / 53.8 / 41.4 |

**20×**。顺带：`--threads 8` 的 CPU 生成从 41–56 t/s 到 83 t/s，
Z-Image 的 **CPU** 路径 denoise 从 139.7 s 降到 126.4 s（−9.5%），GPU 路径不变
（block profile 20.0 s vs 19.5 s，输出逐字节相同）。

没做的：把 worker 数夹到 `min(requested, nproc - 1)`。修完等待逻辑后
`--threads 20` 已经不再塌，这条就没有必要了——但默认线程数仍是 8，
因为 12–20 线程在 0.6B 上没有稳定收益。

顺带记一条同源结论：**Qwen3 的 GPU 常驻解码在这台机器上全面输给 CPU**
（0.6B：CPU 67–97 t/s vs GPU 26 t/s；8B：CPU 12.6–13.5 t/s vs GPU 5.1 t/s），
原因与 TODO-031 一致，只是这次把尺寸补全了。GPU 真正赢的是高算术强度的大矩阵
（Z-Image DiT 权重 1.3 GB/层、rows 数百），而 decode 是 rows=1 的 GEMV——正是
`docs/usage/z-image.md` 里点名会误导的那类形状。

### TODO-038: CPU/GPU 自动分发应该按实测选后端，而不是按模型大小

在 GB10 上把 Qwen3 的 Q8_0 从 0.6B 扫到 8B（同 prompt、greedy、16 线程，
线程池修复后的构建）：

| 模型 | 每 token 权重 | CPU gen t/s | GPU gen t/s | CPU prefill t/s | GPU prefill t/s |
|---|---|---|---|---|---|
| 0.6B | 0.54 GB | **121.2** | 52.8 | **346.9** | 25.9 |
| 4B | 3.74 GB | **23.1** | contract 失败→CPU | 1.3 | — |
| 8B | 7.57 GB | **11.1** | 9.2 | **14.0** | 4.3 |

**没有出现 GPU 超越的点**，但差距随尺寸收窄：0.6B 差 2.3×，8B 差 1.2×。

把两边换算成有效带宽（每 token 把整个权重集读一遍）：

| 模型 | CPU GB/s | GPU GB/s |
|---|---|---|
| 0.6B | 65 | 28 |
| 8B | 84 | 70 |

于是结论是：**GPU 解码是带宽受限的，而它只跑到 70 GB/s；CPU 在同一台机器上跑到 84 GB/s。**
`qwen3_decode_gemv_bandwidth`（`src/vulkan/ops.rs`，`#[ignore]`）在真实解码形状上
量到 lm_head 一次 661 MB 的 GEMV 最快 8.9 ms（74 GB/s），其余形状 10–52 GB/s，
小矩阵被每次 dispatch 的固定开销吃掉。

**但这个 GPU 数字是带偏的**：测量期间另一个租户占着 85% 的 GPU（root 的 python，
96 GB 显存），nvidia-smi 采样 79–92%。GPU 这边还没拿到空闲机器，所以
「GPU 在大模型上能不能反超」在这台机器上无法判定。

因此自动分发的策略应当是**实测驱动**，不是按尺寸猜：

1. **先量设备忙不忙**。本机 GPU 常驻 85% 利用率时，GPU 路径在任何尺寸都不该被选中。
   可以用首 token 的实测延迟对比 CPU 基线，落后就回退——现在只有「logit 逐位一致」
   的 contract 门，没有「谁更快」的门。
2. **把 setup 成本算进去**。host→device 上传实测只有 5–9 GB/s（PCIe 路径在这台
   统一内存机器上不快），8B 的 7.5 GB 权重上传就要 1 s 起步；一次性的 CLI 短跑
   永远摊不平，serving 才能。策略需要知道预期 token 数。
3. **contract 门先修**。4B 现在直接被拒（TODO-036），在数值对齐修好之前，
   任何尺寸相关的策略表都是空谈。

顺带记一条实测的坑：这台机器上任何「GPU 更快/更慢」的结论都必须附带
**当次测量的 GPU 利用率**，否则数据没有可比性——TODO-036/037/038 里的数字都受此影响。

### TODO-039: PyTorch 参照证明 GPU 路径不是「实现得不好」，是设备被占着

用原始 safetensors 权重跑 PyTorch（`tools/oracle/qwen3/pytorch_qwen3.py`，
bf16，greedy，64 token），和本仓库的两条路径同台对照：

| 模型 | PyTorch CUDA bf16 | 我们的 Vulkan GPU (Q8_0) | 我们的 CPU (Q8_0) | PyTorch CPU fp32 |
|---|---|---|---|---|
| Qwen3-0.6B | 24.7 t/s | **52.8 t/s** | 121.2 t/s | 2.4 t/s |
| Qwen3-8B | 4.2 t/s（15.6 GB/token） | **9.2 t/s**（7.6 GB/token） | 11.1 t/s | — |

**我们的 Vulkan 路径比同卡 PyTorch 快 2.1–2.2×**（Q8_0 把每 token 的字节数减半，
kernel 也够快），所以「GPU 输给 CPU」**不是实现问题**。

真正的原因是带宽比：GPU 有效读取 70 GB/s，CPU 84 GB/s，而 GPU 这 70 GB/s 是在
**另一个租户占着 85% 设备**的情况下拿到的（nvidia-smi 采样 79–92%）。GB10 的
DRAM 峰值在 273 GB/s 量级，设备空闲时 GPU 路径在 8B 上应该能到 20 t/s 以上，
也就是**会赢过 CPU 的 11.1 t/s**。

所以自动分发的正确形状是「按实测选」，而不是「按大小猜」：先量设备占用与首 token
延迟，落后 CPU 就回退。0.6B 上 GPU 无论如何都输（小模型权重装不进 cache，
CPU 直接吃 LPDDR），8B 以上在设备空闲时才可能赢——这两个边界都需要在
**独占设备**上重新测一遍。

顺带把 PyTorch 的输出也记下来：Q8_0 的 greedy 文本与 bf16 参考在十几个 token 后
就分叉（CPU 路径与 GPU 路径之间也分叉），这是量化 + 非逐位一致的正常表现，
不是 bug 的证据；要判对错得用同权重的 llama.cpp 参照（TODO-036 的 4B contract
失败同理）。

### TODO-040: Astronex GPU 路径在同 seed 下不可复现（GB10，2026-10-21）

同一个二进制、同一个 prompt、同一个 `--seed 42`，两次 GPU 跑出来的
256x144 视频逐帧 PSNR 只有 18.4–19.2 dB；把采样后的 latent 导出来
（`RMI_ASTRONEX_LATENT_OUT`）对比，61,440 个元素里有 61,439 个不同，
最大绝对差 1.78（latent 本身最大幅值 8.66）。差异在 VAE 之前就出现，
所以 VAE 和文本编码器都不是源头。

已经排除的：

- online attention kernel 与 BF16 投影 kernel **单独**反复调用都逐位一致
  （`examples/astronex_gpu_determinism.rs`）；
- 文本编码器里第一个出现差异的算子**永远是 GPU 投影**：输入哈希相同、
  权重相同，输出哈希不同（`examples/astronex_text_determinism.rs`）；
- 关掉 GPU matmul（`RUST_GPU_MIN_OUTPUTS`）或把 `ComputePool` 降到单线程，
  文本编码器就完全可复现 → CPU 路径和线程池自身的归约都不是源头；
- 差异是**间歇性**的：有些次编码逐位相同，而且首次出错的算子在几次运行之间会移动。

一个已知但未修的规范问题：`src/vulkan.rs` 的 `host_write_barrier` 是在
`begin_commands` 里记录的，位置在它本该覆盖的 host 写**之前**（权重上传与
输入写都在其后）。把屏障挪到 dispatch 之前能降低发生率但不能根除，
所以 host→device 可见性不是唯一原因。

下一步：在**独占设备**上重测（这台 GB10 有另一个租户占着 65 GB 和 79–92% 利用率），
确认是驱动/内存压力导致的偶发，还是我们自己的同步缺口。

### TODO-041: Astronex 视频输出依赖系统 `ffmpeg`

`src/models/diffusion/dreamx/media.rs` 直接 spawn `ffmpeg`，没有它
`--astronex` 跑完采样和 VAE 解码后在写视频时失败（`Start ffmpeg for DreamX
video: No such file or directory`）。这台机器没有系统 ffmpeg，验证时用的是
`imageio-ffmpeg` 装进仓库 venv 的静态构建（7.0.2，含 libx264）。
要么在 USAGE 里写明这个运行期依赖，要么在 `--out` 不支持时给出更早、更明确的报错。

### TODO-042: Astronex 权重契约的 pinned commit 不公开

转换器的形状契约来自 `references/astronex-world-contract`，钉在
`dec4ec502f7113a1ab07b78ae40a7ac1b88b2eeb`；这个 commit 不在
`github.com/Astronex-Robotics/Astronex-World` 的任何 ref 里，也不在
gouzil/rust-model-inference 的远端分支里。本地只能用下载到的 safetensors
自己生成 header，等于 denoiser/VAE 的形状 pin 变成自证（张量数
898/242/196、dtype、`model.*` 前缀和 UMT5 的硬编码形状仍然生效）。
要么把契约 header 随仓库分发，要么让转换器在缺契约时明确报「未钉住」而不是静默通过。

顺带记一条：官方图的 trace 在本机跑到第 1,438 个 checkpoint（整个文本编码器 +
denoiser 大部分）后停在 `NotImplementedError: Uncovered floating factory:
linspace.default` —— scalar oracle 的算子白名单没覆盖 `linspace`（solver 的
timestep 网格），不是权重或图的问题。要跑完整官方图 trace，得先给 oracle 补
`linspace` 的白名单条目。

### TODO-043: 对拍生成文本必须显式 `--temp 0`（默认是 0.6 采样）

排查 BF16/F32 Prepared 新路径的干扰面时（2026-10-22），前两批实验全部作废，
原因就是这个：CLI 的默认温度是 **0.6**（`src/app/cli/validate.rs:321`），
所以不带 `--temp 0` 的「同 seed 两次对比」其实是在比较两次独立采样。

它产生的假象很有说服力：同一二进制、同一 prompt，CPU 连续两次结果不同，
其中一次把澳大利亚首都答成 Melbourne（错误答案），一次答成 Canberra；
统计「与 CPU 参考的最长公共前缀」也随之变成噪声。补上 `--temp 0` 之后
CPU 侧逐位可复现（同 1 和 8 线程都一致），所有差异才变得可解释。

顺带记第二个坑：选测试对象时必须先静态确认它**真的走到被改的代码路径**。
Qwen3-0.6B-BF16 看起来是最干净的受害者（197 个纯 BF16 二维权重），实际上它
用 `with_bf16_input` 构造 kernel，mode 落在 `RoundedBf16`，根本不进新分支，
于是「输出一致」只证明了没变化。`dump_dispatch_log` 只在测试里调用，CLI 读不到，
只能从 `rounds_bf16_input()` 的来源反推哪些模型会命中。

排查结论：主力文本模型（Qwen3 / Qwen3.5 / Edge0 / ERNIE）全部不命中，
Z-Image 代码零改动且 3/3 逐位复现基线 md5，含 110 个 F32 二维权重的
occamy-1.0 在 `--temp 0` 下 GPU 与 CPU 逐位一致。尚未实测的是少数多模态
projector（ZDTaichu5 mmproj、LongCat）与几个 MoE gate。
