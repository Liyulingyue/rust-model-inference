# 技术设计规范：rust-model-inference

> **文档用途：** 本文档定义 `rust-model-inference` 引擎的设计契约、模块拓扑与可执行硬约束。参与重构或扩展项目的 AI Agent / 开发者必须严格遵守。
>
> 角色分工：本文档只写**跨模块适用的法则**与**模块地图**；子系统细节下沉到 `docs/develop/` 下的专门文档（KV cache、ggufrs、异构算力、并行 matmul 安全等），文末 §9 列出索引。

---

## 1. 核心设计初衷与目标

`rust-model-inference` 是一个针对端侧设备（Intel / AMD x86_64 AVX2+FMA、ARM64 NEON、APU、独立 GPU；macOS/iOS/Linux/Windows 跨平台）的**多模态混合量化**推理引擎。引擎同时支撑纯文本 LLM、语音 ASR、语音 TTS、视觉 LLM、扩散模型（文生图 / 图生视频 / 视频续写）、自动驾驶感知规划等任务，统一从一个 CLI 二进制和一个 axum HTTP 服务二进制入口派发。

### 关键工程痛点与破局策略

1. **摆脱 C++ 内存隐患：** 传统的 C++ 推理引擎在多模态/混合量化管道上极易因指针强转、悬垂引用或动态 `memcpy` 产生静默内存污染与段错误。
2. **面向 Agent 演进（Agent-Safe）：** 利用 Rust 编译期 Borrow Checker 与生命周期约束，赋予 AI Agent 安全重构代码的能力。代码通过编译，即数学证明其不存在数据竞争、野指针或越界。
3. **极致系统级性能：** 在 hot path 锁定 **f32 in-place Arena + SIMD 静态派发**；在存储层锁定 **mmap 零拷贝借用**；在算子层锁定**静态 enum 派发 + 预量化 Q8_0/Q8K 激活**。

---

## 2. 三大核心设计法则

### 法则一：存储层零拷贝借用（Zero-Copy Borrowed View）

* **规则：** GGUF / `.ggufrs` / `Vulkan` / `wgpu` 在加载层一律通过 `TensorSource` trait（`src/core/tensor.rs`）抽象，**裸字节切片 `&[u8]` / `&[f32]`** 流转，不复制权重。
* **范围限制：** 仅"GGUF 直读"路径真正零拷贝；`.ggufrs` 多组件容器在 `format/ggufrs.rs` 内部做段级零拷贝，跨段聚合仍可能拷贝。GPU 后端会把权重上传一次到设备内存（不算 hot path 复制）。
* **架构位置：** `src/core/{tensor,loader}.rs`、`src/format/ggufrs.rs`、`src/ops/kernel/quantized_tensor.rs` 的 `QuantizedTensor<'a>` borrowed enum。

### 法则二：交互层统一 `f32` In-Place Arena

* **规则：** 算子之间、Layer 之间、模态之间（Projector → LLM Layer）的特征向量交互，**必须且仅能使用预分配的 `ExecutionScratchpad` 切片 `&mut [f32]`**。KV cache 走独立的 `KvCache { F16, F32 }`（见 `src/core/scratchpad.rs`）。
* **物理合理性：** 用 `f32` 传递临时 activation 在位宽上有冗余，但消除了层间"实时重新量化"的开销，使数据 100% 在 L1/L2 cache 内 In-Place 覆写。
* **范围限制：** 该法则只约束 hot path（`forward` 函数内），**不**约束 `ComputePool`/`MemoryArena`/`BlockAllocator` 等一次性资源池的 `Box<[T]>` / `Vec<T>` 分配。文档 `forward` 一律禁止 `vec!`/`box!`/`format!`/`memcpy`；初始化期资源池自由分配。
* **Scratchpad 契约**（`src/core/scratchpad.rs:49-74`）：
  * 17 个 `Vec<f32>` 字段：`x, normed, q, k_new, v_new, attn_out, attn_proj, down_buf, gate_buf, up_buf, logits, q8_buf, scale_buf, q8k_buf, score_stride, scores`。
  * `max_n_in = (n_embd * 3).max(n_embd_q).max(n_ff)`；所有 forward 的局部 `max_n_in` 必须与此表达式同步。漂移触发 Windows `STATUS_HEAP_CORRUPTION` / `STATUS_ACCESS_VIOLATION`。
  * `gate_buf`/`up_buf` 需 `max(n_ff, n_embd * 3)`（dense FFN 与 shortconv `in_proj` 共用）。

### 法则三：算子层粒度混搭（Fine-Grained Hybrid Quantization）

* **规则：** 允许在**模态级**（ViT F16 / LLM Q4_K）、**层级**（Head/Tail 高精度 / Body 低精度）甚至**算子级**（Self-Attention Q8_0 / FFN Q4_K）自由混合精度。
* **实现：** 静态枚举 `QuantizedTensor<'a>` + 静态 `match` 派发到 `Kernel` trait；不允许出现虚函数表（vtable）寻址。同一权重在 hot path 上仅一次 `match`。
* **算子接口：** `src/ops/kernel/trait_.rs` 的 `Kernel` trait，hot path 入口是 `forward_prequantized(input_q8, input_scales, output, n_in, n_out, ith, nth)`。默认实现提供 `forward` / `forward_batched`，K-quant 通过 `forward_prepared` 走共享 Q8K activation。
* **Fuse 例外：** `QTensorOwned::fuse_vstack`（`src/ops/kernel/qtensor_owned.rs`）用于 FFN gate+up 融合，需要**拥有**权重复制——是这条法则下唯一的"非零拷贝"算子。

---

## 3. 模块拓扑与依赖方向

### 3.1 顶层模块地图（`src/` 真实结构）

```
src/
├── main.rs                # CLI 入口：5 种 DispatchMode 派发
├── lib.rs                 # 精选 re-export 入口
├── parity_trace.rs        # feature=parity-trace 下 SIMD/GPU vs scalar 对照
├── wgpu.rs                # wgpu 后端根模块（独立二进制内）
│
├── app/                   # 应用层：CLI 解析、模式入口、子应用
│   ├── cli.rs             # CliOptions（40+ flag）+ 各类 CliOptions 子结构
│   ├── text.rs            # 文本推理入口（chat template、token streaming）
│   ├── audio.rs           # ASR 入口
│   ├── tts.rs             # TTS 入口
│   ├── image.rs           # Diffusion 入口（Z-Image / PIG）
│   ├── dreamx.rs          # DreamX-Creator 入口
│   ├── qwen_drive.rs      # 自动驾驶入口
│   ├── omni.rs            # Qwen2.5-Omni / Qwen3-Omni 多模态入口
│   ├── dots.rs            # dots.tts / dots.tts.edit
│   ├── breeze.rs          # Breeze-TTS-2
│   ├── vibevoice.rs       # VibeVoice
│   ├── embedding.rs       # Embedding 入口
│   └── selftest.rs        # 空 --model 时的自检
│
├── core/                  # 基础域类型（不依赖 models/ops/app）
│   ├── tensor.rs          # GGMLType (30 variant), MetaValue*, TensorSource
│   ├── loader.rs          # ByteReader, GGUFLoader, model_config_from_source
│   ├── model.rs           # ModelGraph, QuantizedLinear (generic Layer 容器)
│   ├── tokenizer.rs       # BPETokenizer, EncodeOptions, StreamingDecoder
│   ├── tokenizer/         # 多 tokenizer 子实现
│   ├── memory.rs          # PagedKVBlock, BlockAllocator, KVCacheView, MemoryArena
│   ├── scratchpad.rs      # ExecutionScratchpad, KvCache (F16/F32), KvLifecycle
│   ├── thread_pool.rs     # ComputePool（BSP 自旋池）
│   ├── traits.rs          # Layer trait, ExecContext<'a>, ModelConfig
│   └── prefill.rs         # prefill batch 校验（CLI 默认 64）
│
├── format/                # 文件格式层（依赖 core，不依赖 models/app）
│   ├── ggufrs.rs          # .ggufrs 多组件容器读/写/校验（~4300 行）
│   └── load_plan.rs       # 异构设备 load plan（NUMA / tensor split）
│
├── ops/                   # 算子层（依赖 core，不依赖 models/app；backend 通过 cfg feature 进入）
│   ├── activation/        # silu / gelu / softmax 等
│   ├── dot.rs             # 通用 reduce + vec_dot
│   ├── embedding.rs       # 跨 dtype embedding lookup
│   ├── float.rs           # f16/bf16 SIMD 转换
│   ├── kernel/            # ★ matmul 内核层
│   │   ├── trait_.rs      # Kernel trait（hot path 入口）
│   │   ├── quantized_tensor.rs   # QuantizedTensor<'a> borrowed enum
│   │   ├── qtensor_owned.rs      # QTensorOwned（fuse 路径）
│   │   ├── simd_avx2.rs          # AVX2 公用工具
│   │   ├── f16/ f32/ bf16/       # 浮点权重子目录（mod/avx2/neon/scalar/avx2_q8/neon_q8）
│   │   ├── q4_0/ q4_1/ q8_0/     # 旧量化子目录（mod + SIMD + scalar + dispatch + parallel）
│   │   ├── q2_k.rs q3_k.rs q4_k.rs q5_k.rs q6_k.rs   # K-quant（多文件 scalar 路径，AVX2 在 quant/avx2_k.rs）
│   │   └── iq4_nl.rs iq4_xs.rs                       # I-quant（IQ4_NL / IQ4_XS kernel 入口）
│   ├── math/              # 数学工具
│   ├── matmul.rs          # matmul_tests
│   ├── norm.rs            # rms_norm / rms_norm_inplace / rms_unit_inplace
│   ├── quant/             # 量化辅助层（Q8_0 量化、K-quant AVX2 内核、IQ 表）
│   │   ├── mod.rs         # BlockQ8K、quantize_row_q8_k_into
│   │   ├── q8_0.rs        # Q8_0 量化
│   │   ├── avx2_k.rs      # AVX2 K-quant / IQ4_NL / IQ4_XS 内核（共享）
│   │   ├── neon_k.rs      # aarch64 NEON：IQ4_NL × Q8K（与 AVX2 同源 1 ULP drift）
│   │   ├── fuse.rs        # FFN 融合算子
│   │   ├── iq_tables.rs / iq_tables_data.rs  # IQ 表查表
│   ├── rope/              # RoPE（neox / mrope）
│   ├── sampling.rs        # top-k / top-p / temperature
│   ├── softmax.rs         # online softmax
│   └── ssm.rs             # Mamba2 SSM 算子（Nemotron-H）
│
├── models/                # 模型实现（依赖 ops/core/format）
│   ├── mod.rs             # 注册表
│   ├── llama/             # trunk only
│   ├── lfm2/              # trunk + vision.rs
│   ├── lfm25/             # trunk
│   ├── lfm2moe/           # trunk（MoE 32-choose-4, sigmoid gating）
│   ├── qwen3/             # trunk + asr/ + tts/ + vision/ + embedding.rs + hunyuan.rs + omni.rs + text.rs
│   ├── qwen35/            # trunk + vision/
│   ├── breeze/ dots/ spark/ vibevoice_asr/ gemma4/ nemotron_h/ qwen_drive/
│   └── diffusion/         # dreamx/ + z_image/ + pig.rs
│
├── vulkan/                # Vulkan GPU backend（feature=vulkan）
│   ├── ops.rs             # 通用 GPU 算子（matmul / RMSNorm / RoPE / attention / FFN）
│   ├── qwen3.rs           # Qwen3 Vulkan executor
│   └── qwen35.rs          # Qwen3.5 BF16 Vulkan executor
│
└── bin/                   # 二进制
    ├── server.rs          # axum HTTP server（992 行）
    ├── ggufrs.rs          # .ggufrs 导出工具
    ├── micro_bench.rs     # 性能基准
    ├── codec_test.rs      # codec 单元测试入口
    └── dump_{dims,emb,meta,tensors}.rs  # 诊断
```

### 3.2 依赖方向硬约束

```
       app/         →         core/        →         ops/        →         backend (vulkan/wgpu)
                          ↗ format/    ↘                                    (cfg feature)
                       models/
```

* `core/*` 不依赖 `models/*` / `ops/*` / `app/*` / `format/*`。
* `ops/*` 不依赖 `models/*` / `app/*`；`ops/*` 通过 cfg feature 可选依赖 `vulkan`/`wgpu`。
* `format/*` 依赖 `core/*`，不依赖 `models/*` / `app/*`（见 `format/mod.rs:9-13`）。
* `models/*` 依赖 `core/*`、`ops/*`、`format/*`，不依赖 `app/*`。
* `app/*` 依赖 `models/*`、`core/*`、`ops/*`、`format/*`。
* `vulkan.rs`/`wgpu.rs` 是 backend 模块（设备抽象），不属于 `ops/` 语义（见 `HETEROGENEOUS_COMPUTE.md`）。

---

## 4. 架构注册表（`src/main.rs` + `src/core/loader.rs`）

主模型代码当前认识这些 `general.architecture`：

| architecture | 主模型 | trunk 入口 | 备注 |
|---|---|---|---|
| `qwen2` | Qwen2 / Qwen2.5 / dots 内部 LLM / VibeVoice-ASR | `models::qwen3::trunk` 或专用 | 通用文本 |
| `qwen2vl` | Qwen2.5-Omni | 同上 + mmproj | 多模态 |
| `qwen3` | Qwen3 / Qwen3-Embedding | `models::qwen3::trunk` | 纯文本 + embedding |
| `qwen3vl` | Qwen3-VL / Qwen3-ASR | `models::qwen3::trunk` + vision/asr | 图像 / 音频 |
| `qwen3vlmoe` | Qwen3-Omni MoE | 同上 + mmproj | shared expert 显式拒绝 |
| `qwen35` | Qwen3.5 / Qwen3.8 / Qwen-Drive-VLM | `models::qwen35::trunk` + vision | dense + recurrent + SSM |
| `qwen3tts` | Qwen3-TTS | `models::qwen3::trunk` + tts | codec 后拼 |
| `llama` | Llama / MiniCPM5 / Nanbeige | `models::llama::trunk` | 通用文本 |
| `granite` | Granite | `models::llama::trunk` | 文本 |
| `hunyuan-dense` | Hunyuan-MT2 | `models::qwen3::trunk` 复用 | 多语言翻译 |
| `pig` | Z-Image Turbo | `models::diffusion::pig` | 文生图（diffusion） |
| `lfm2` | LFM2.5 / LFM2.5-VL | `models::lfm2::trunk` | shortconv + GQA |
| `lfm2moe` | LFM2-8B-A1B | `models::lfm2moe::trunk` | MoE hybrid |
| `nanbeige` | Nanbeige | `models::llama::trunk` 复用 | SPM tokenizer |
| `gemma4` | Gemma-4 E2B/E4B | `models::gemma4::trunk` | 文本 + 图像 + 音频 |
| `spark2_5` | Spark-X2.5 | `models::spark::trunk` | 文本 + thinking |
| `dreamx` | DreamX-Creator | `models::diffusion::dreamx` | 首帧驱动音视频 |

`clip` 是 mmproj 组件 architecture，不是可独立生成的主模型。

详细"具体型号 → 已验证格式 → 证据"见 [`docs/develop/SUPPORTED_MODELS.md`](SUPPORTED_MODELS.md) 与 [`docs/MODEL_LIST.md`](../MODEL_LIST.md)。

---

## 5. 二进制与 CLI 矩阵

### 5.1 二进制入口

| 二进制 | 路径 | 用途 |
|---|---|---|
| `rust-model-inference` | `src/main.rs`（470 行） | CLI 主入口；`DispatchMode` 派发 5 类子模式 |
| `rust-model-inference-server` | `src/bin/server.rs`（992 行） | axum HTTP 服务；流式 + 请求级动态媒体 |
| `rust-model-inference-ggufrs` | `src/bin/ggufrs.rs` | `.ggufrs` 导出工具 |
| `micro_bench` / `codec_test` / `dump_*` | `src/bin/` | 诊断与基准 |

启动期：`app::init_rayon_global_pool(n_threads)` 在 `main.rs:106` 调用，对齐 `ComputePool` 与 rayon global pool 的线程数。

### 5.2 CLI 矩阵（`src/app/cli.rs` 的 `CliOptions`）

> **热路径 flag（`run_inference` / `run_shared_inference` / `run_multimodal_with_video` 用）**

| flag | 含义 | 默认 |
|---|---|---|
| `--model <path>` | GGUF 或 .ggufrs 路径（必填，空时跑 selftest） | — |
| `--prompt "text"` | 输入文本 | `""` |
| `--threads N` | 线程数（影响 ComputePool + rayon） | `min(available, 8)` |
| `--max-tokens N` | 生成长度 | 128 |
| `--max-context N` | 最大上下文 cap | 8192（防 K2-Horizon 524288 撑爆 KV） |
| `--temp F` | 采样温度 | 0.6 |
| `--top-k N` / `--top-p F` | 采样参数 | — |
| `--seed N` | 随机种子 | — |
| `--thinking` | 启用 thinking 模式（Qwen3 / Spark） | false |
| `--bench` | 跳 chat template，原始 token 生成 | false |
| `--profile` | 打印 timing breakdown | false |
| `--kv-cache f16\|f32` | KV 精度 | `F32` |
| `--prefill-batch-size N` | prefill 批大小 | 64 |
| `--dump-logits` | dump logits 到 `/tmp/rust_logits.bin` | false |
| `--gpu` | 启用 Vulkan 后端 | false |

> **多模态 / 子应用 flag**

| flag | 触发入口 |
|---|---|
| `--mmproj <path>` | 多模态（vision / audio / mmproj 拼接） |
| `--image <path>` | 图像输入 |
| `--video <path>` | 视频输入 |
| `--audio <path>` | 音频输入（`qwen3vl`/`qwen2`→ASR；`gemma4`→多模态） |
| `--ref-audio` + `--ref-text` | TTS 声音克隆 |
| `--source-audio` + `--source-text` + `--target-text` | dots.tts.edit |
| `--tts` | TTS 子入口 |
| `--dreamx` | DreamX-Creator 子入口 |
| `--planner <path>` / `--perception <path>` | Qwen-Drive 自动驾驶 |
| `--scenes` / `--image-root` / `--frames` | Qwen-Drive 场景与帧 |
| `--vae` / `--text-encoder` | Z-Image 子入口 |
| `--steps` / `--resolution` / `--cfg-scale` / `--duration-seconds` / `--fps` | 扩散模型参数 |
| `--embedding` / `--embedding-output summary\|raw` | Embedding 入口 |
| `--language <code>` | TTS 语言（cn/en/ge/it/po/sp/ja/ko/fr/ru 等） |
| `--use-xvector` | dots.tts 说话人嵌入 |
| `--dry-run` / `--overwrite` / `--out` | 输出控制 |
| `--allow-memory-overcommit` | 允许越过 MAX_HEAD_DIM 等硬约束 |
| `--edit` | dots.tts.edit 模式 |

完整解析见 [`src/app/cli.rs:247`](../app/cli.rs)（2490 行）。

---

## 6. 文件格式层：`.ggufrs` 多组件容器

### 6.1 设计目标

`.ggufrs` 是引擎**自有**的多组件打包格式（GGUF Re-Segment），不是 llama.cpp 的交换格式：

* 把 LLM 主干 GGUF 与可选 mmproj GGUF 装进同一个文件
* 保留组件级 metadata 与原始 tensor 字节
* 64 KiB 段对齐；`GGUFRS_SEGMENT_ALIGNMENT` 常量定义在 `src/format/ggufrs.rs`
* 详尽物理布局见 [`docs/develop/GGUFRS.md`](GGUFRS.md)

### 6.2 入口与角色

```rust
pub use format::ggufrs::{
    export_ggufrs, open_model_source, ComponentInfo, ComponentRole, ExportOptions,
    GgufrsError, GgufrsFile, LoadedComponent, SegmentKind, GGUFRS_SEGMENT_ALIGNMENT,
    GGUFRS_VERSION,
};
```

* `open_model_source(path, role)` → `Box<dyn TensorSource>`：把 GGUF 或 .ggufrs 统一抽象为 `TensorSource` trait（`src/core/tensor.rs`）。
* `ComponentRole::Llm` / `ComponentRole::Mmproj`：CLI 传入 `--mmproj` 时同时打开两个 component。
* `LoadedComponent` / `SegmentKind`：分页视图。

### 6.3 异构 Load Plan

`src/format/load_plan.rs`（46 KB）实现 NUMA / tensor split 决策：

```rust
pub enum PlacementPolicy { LayerSplit, TensorSplit }
pub enum PlacementSlice { Whole, Rows { start: u64, end: u64 } }
pub struct Placement { component_id, segment_id, tensor_name, slice, segment_byte_range, device_id }
pub fn build_load_plan(...) -> Result<LoadPlan, ...>
pub fn load_logical_cpu(...) -> Result<LogicalCpuDeviceLoad, ...>
```

设计原则见 [`docs/develop/HETEROGENEOUS_COMPUTE.md`](HETEROGENEOUS_COMPUTE.md)。

---

## 7. 异构资源与 GPU 后端

### 7.1 GPU 后端矩阵

| 后端 | feature | 入口 | 覆盖范围 |
|---|---|---|---|
| AVX2+FMA | `target_arch="x86_64"`（默认） | `src/ops/kernel/{q4_0,q4_1,q8_0,bf16,f16,f32}/avx2*.rs`、`src/ops/quant/avx2_k.rs` | dense matmul + RMSNorm + RoPE |
| NEON | `target_arch="aarch64"`（默认） | 同上 + `*_neon*.rs` | dense matmul |
| Scalar | 兜底 | `src/ops/kernel/*/scalar.rs` | 全部 dtype + 全部算子 |
| Vulkan | `--features vulkan` | `src/vulkan/{ops,qwen3,qwen35}.rs`（290 KB） | Qwen3 dense（Q8_0/Q4_0/Q4_1/Q4_K/Q6_K/F16）；Qwen3.5 dense + recurrent + SSM（BF16） |
| wgpu | `--features wgpu` | `src/wgpu.rs` | 实验性 |

详见 [`docs/develop/VULKAN.md`](VULKAN.md) / [`VULKAN_INFERENCE_DESIGN.md`](VULKAN_INFERENCE_DESIGN.md) / [`VULKAN_INFERENCE_PLAN.md`](VULKAN_INFERENCE_PLAN.md) / [`HETEROGENEOUS_COMPUTE.md`](HETEROGENEOUS_COMPUTE.md)。

### 7.2 异构派发原则

"标量为底 + 统一入口 + 自动派发"：

* Scalar 是 floor，是 SIMD/GPU 的正确性参照（`parity_trace.rs` feature 是门禁）
* `models/` 只调用 op 级统一 API（`silu_inplace` / `matmul_q8_0_quantized_parallel_rows` 等），不出现 `*_avx2` / `*_vulkan` 具名符号
* 启动期探测：`has_avx2_fma()` / `has_neon()` / `get_vulkan_context()`

### 7.3 RAII 资源立即回收

`.ggufrs` 仅作为静态物理地址映射；**每个模型 runner 自带 `Drop` 释放**：

* `VisionEncoder`（`src/models/qwen3/vision/mod.rs` + `src/models/qwen35/vision/mod.rs`）：视觉/音频编码器计算完成后 `Drop` 触发释放，把显存让给 LLM KV cache 扩展
* LLM / Diffusion / TTS runner：各自独立硬件句柄，无跨阶段绑定

---

## 8. 多线程架构

### 8.1 双池模型（ComputePool + rayon）

代码注释（`src/core/thread_pool.rs:36-80`）明确写出当前形态与未来方向：

* **ComputePool**（`src/core/thread_pool.rs`）：自旋 + epoch-based BSP 调度；LLM prefill/decode 的唯一调度器；按 `(ith, nth)` 行分区闭包，无 work-stealing。
* **rayon global pool**（初始化：`src/app/cli.rs:222 init_rayon_global_pool`）：work-stealing；用于 audio conv chunk（`src/models/qwen3/asr/model.rs`）、vision patch encoding（`src/models/qwen3/vision/mod.rs` + `src/models/lfm2/vision.rs`）、qwen35 SSM（`src/models/qwen35/trunk/forward.rs`）。
* **未来统一方向**：迁移 audio/vision/qwen35 SSM → ComputePool（drop-in：`compute_with_chunks(n, f)`），**不**迁移 LLM；统一后可移除 rayon 依赖。

两池线程数在 `main.rs:102-106` 对齐到 `--threads` 解析值。

### 8.2 LLM 每层 BSP 流水线（以 qwen3 trunk 为例）

| Step | 执行者 | 操作 |
|---|---|---|
| 1 | `pool.compute()` | QKV matmul（3 个并行 matmul） |
| 2 | 主线程 | RoPE + Q/K norm + KV cache 写入（**单线程**：GQA 写-写竞争） |
| 3 | `pool.compute()` | Attention（online softmax，per-head 并行） |
| 4 | 主线程 | quantize attn_out → Q8_0 |
| 5 | `pool.compute()` | Wo 投影 + 残差加 + FFN norm + quantize → Gate+Up+SiLU |
| 6 | 主线程 | quantize gate_buf → Q8_0 |
| 7 | `pool.compute()` | Down 投影 |

> **GQA 正确性约束：** 当 `n_head_kv` 不能被线程数整除时，基于 Q head 范围推导的 KV head 范围会重叠。KV 写入必须单线程；Attention 只读 KV cache，无写竞争。

### 8.3 `compute_unchecked` 形式别名违规（Issue 4）

`Weight::quantize_and_matmul_with_scratch` 在每个 worker 内通过 `unsafe { std::slice::from_raw_parts_mut(output_ptr, n_out) }` 派生覆盖整段 `output` 的 `&mut [f32]`——形式上违反 Rust 别名规则（stacked borrows / tree borrows），实际依赖每个 kernel 按 `(ith, nth)` 计算 `[start, end)` disjoint 写入。详见 [`docs/develop/PARALLEL_MATMUL_SAFETY.md`](PARALLEL_MATMUL_SAFETY.md)。

### 8.4 ComputePool Epoch 竞态修复

Worker 的 `my_epoch` 必须从 `0` 初始化，**不能**读 `inner.epoch.load(Acquire)`，否则 start_barrier 后到 spin-loop 之间的延迟会让 worker 读到已递增的 epoch，跳过当前计算轮次。详见 `src/core/thread_pool.rs`。

---

## 9. 已实现 / 未实现清单（按 §10 风险表分层）

### 9.1 已落地（无需重新实现）

| 主题 | 落地点 |
|---|---|
| `.ggufrs` 多组件容器 | `src/format/ggufrs.rs`（~4300 行）+ `docs/develop/GGUFRS.md` |
| `QuantizedTensor` 枚举覆盖 Q4_K/Q5_K/Q6_K/Q2_K/Q3_K/IQ4_NL/IQ4_XS/F16/F32/BF16 | `src/ops/kernel/quantized_tensor.rs` |
| `Kernel` trait + 静态 `match` 派发 | `src/ops/kernel/trait_.rs` |
| Q8_K 预量化 K-quant 路径 | `src/ops/quant/avx2_k.rs` + `Kernel::forward_prepared` |
| FFN gate+up fuse | `QTensorOwned::fuse_vstack`（`qtensor_owned.rs`）+ `src/ops/quant/fuse.rs` |
| `ExecutionScratchpad` buffer 尺寸不变量契约 | `src/core/scratchpad.rs:49-74` |
| `KvCache` F16/F32 + `KvLifecycle` 生命周期 | `src/core/scratchpad.rs` + [`docs/develop/KV_CACHE_DESIGN.md`](KV_CACHE_DESIGN.md) |
| Paged KV (`PagedKVBlock` + `BlockAllocator` + `KVCacheView`) | `src/core/memory.rs` |
| Vulkan Qwen3 / Qwen3.5 executor | `src/vulkan/{qwen3,qwen35}.rs` |
| `parity_trace` 回归门禁 | `src/parity_trace.rs`（feature=parity-trace） |
| 双池模型与 rayon 全局池初始化 | `src/core/thread_pool.rs:36-80` + `src/app/cli.rs:222` |

### 9.2 进行中 / 未完成

| 主题 | 当前状态 | 参考 |
|---|---|---|
| Logits 输出投影优化（V·151936） | 内存带宽受限，fuse/分块尚未落地 | — |
| Q2_K / Q3_K SIMD | scalar 在 `src/ops/kernel/{q2_k,q3_k}.rs`；AVX2 仅 K-quant 路径 | — |
| IQ2_XXS / IQ2_XS / IQ3_XXS / IQ1_S / IQ3_S / IQ2_S / IQ1_M kernel | 仅在 `GGMLType` enum 注册，kernel 未实现 | `src/core/tensor.rs:32-45` |
| `lib.rs` glob re-export 清理 | `pub use ops::*;` / `pub use models::qwen3::*;` 仍残留 | [`docs/develop/REFACTOR_PLAN.md`](REFACTOR_PLAN.md) §3 |
| `format/ggufrs.rs` 拆分 read/write/validate | 仍是 ~4300 行单文件 | [`docs/develop/REFACTOR_PLAN.md`](REFACTOR_PLAN.md) §2 |
| Vulkan/wgpu 模块归属 | 仍在 `src/` 根目录；倾向 `src/backend/` | [`docs/develop/REFACTOR_PLAN.md`](REFACTOR_PLAN.md) §1 |
| 双池统一为 ComputePool | 方向确定，未执行 | `src/core/thread_pool.rs:62-79` |
| Nemotron-H 4B 正确推理 | first-coherent-output 已达成；与 llama.cpp oracle logits/argmax 仍有差异 | [`docs/develop/SUPPORTED_MODELS.md`](SUPPORTED_MODELS.md) |

---

## 10. Risk Assessment

| Risk | Severity | 当前缓解 / 跟踪 |
|---|---|---|
| `ExecutionScratchpad` 缓冲尺寸漂移 | High | `src/core/scratchpad.rs:49-74` 文档化不变量；漂移触发 Windows heap corruption |
| `compute_unchecked` 形式别名违规 | Medium | 实际 disjoint 写入；门禁见 [`PARALLEL_MATMUL_SAFETY.md`](PARALLEL_MATMUL_SAFETY.md) |
| GQA 多线程 KV 写竞争 | High | KV 写入单线程化（§8.2 Step 2） |
| ComputePool 大线程数 epoch 竞态 | High | worker `my_epoch` 从 0 初始化（§8.4） |
| GGUF metadata `context_length` 过大（如 K2-Horizon 524288） | High | `--max-context` 默认 8192（`src/app/cli.rs:91`） |
| Vulkan 不支持算子静默回退 | Medium | `text_encode` 失败时丢弃 GPU 结果用 CPU 重算（[`VULKAN.md`](VULKAN.md) §架构） |
| 双池线程数漂移 | Low | `main.rs:102-106` 启动期对齐 |
| 量化类型扩展 | Low | `QuantizedTensor` enum 天然支持；K-quant/I-quant 增量添加 |
| 多模态动态分辨率 | Medium | CLIP / LFM2-VL 等固定 block + 动态块数 |
| Nemotron-H Mamba2 scan 正确性 | Medium | 仍 Experimental；fixture + `tests/nemotron_h_parity.rs` 可用 `--include-ignored` 跑 |

---

## 11. 文档索引

本规范只写跨模块的设计契约；以下子系统的设计细节独立成文：

* [`MODEL_ORGANIZATION.md`](MODEL_ORGANIZATION.md) — 模型 `trunk/` + sibling 目录结构与依赖方向
* [`SUPPORTED_MODELS.md`](SUPPORTED_MODELS.md) — 每个具体型号的"已验证格式 + Oracle + 限制"矩阵
* [`KV_CACHE_DESIGN.md`](KV_CACHE_DESIGN.md) — KV cache 共享条件 + Ephemeral/Timed/Persistent 生命周期
* [`GGUFRS.md`](GGUFRS.md) — `.ggufrs` 物理布局
* [`HETEROGENEOUS_COMPUTE.md`](HETEROGENEOUS_COMPUTE.md) — "标量为底 + 后端注册表"原则
* [`VULKAN.md`](VULKAN.md) / [`VULKAN_INFERENCE_DESIGN.md`](VULKAN_INFERENCE_DESIGN.md) / [`VULKAN_INFERENCE_PLAN.md`](VULKAN_INFERENCE_PLAN.md) — Vulkan 后端覆盖范围与限制
* [`PARALLEL_MATMUL_SAFETY.md`](PARALLEL_MATMUL_SAFETY.md) — Issue 4：并行 matmul 形式别名违规
* [`OPTIMIZATION.md`](OPTIMIZATION.md) — 性能基线、已修复 bug、已验证无效方向
* [`LFM25_OPTIMIZATION.md`](LFM25_OPTIMIZATION.md) / [`LFM25_MOE_OPTIMIZATION.md`](LFM25_MOE_OPTIMIZATION.md) / [`ZIMAGE_OPTIMIZATION.md`](ZIMAGE_OPTIMIZATION.md) — 子模型/子方向优化记录
* [`WEIGHT_STORAGE_DESIGN.md`](WEIGHT_STORAGE_DESIGN.md) — 权重存储设计
* [`QWEN3_ASR.md`](QWEN3_ASR.md) — Qwen3-ASR 端到端
* [`REFERENCE_IMPLEMENTATIONS.md`](REFERENCE_IMPLEMENTATIONS.md) — pinned llama.cpp / 官方实现 commit 表
* [`REFACTOR_PLAN.md`](REFACTOR_PLAN.md) — 残余重构（GPU 归属、ggufrs 拆分、lib.rs re-export 清理）
* [`TODO.md`](TODO.md) / [`ISSUE.md`](ISSUE.md) — 进行中任务与开放问题
* [`docs/MODEL_LIST.md`](../MODEL_LIST.md) — 公开模型清单（下载链接 + 参考 commit）

---

## 12. 配套命令（快速参考）

```text
# 主 CLI
cargo run --release -- --model <path.gguf-or-ggufrs> --prompt "..." --threads N

# HTTP 服务
cargo run --release --bin rust-model-inference-server -- --addr 0.0.0.0:8080 --model ...

# 导出 .ggufrs
cargo run --release --bin rust-model-inference-ggufrs -- --llm <a.gguf> --mmproj <b.gguf> --out <c.ggufrs>

# 基准
cargo run --release --bin micro_bench -- ...

# Vulkan 后端
cargo run --release --features vulkan -- --model <...> --gpu

# 参考实现（pinned llama.cpp）
references/llama.cpp/build/bin/Release/llama-cli.exe -m <path.gguf> -p "..." -n N
```

构建：`cargo build --release`（`opt-level=3`, `lto=fat`, `codegen-units=1`）。
