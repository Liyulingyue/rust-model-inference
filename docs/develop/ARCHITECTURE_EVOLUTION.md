# 架构演进备忘：layer_template + feature gate

> **Status (2026-09-22)**：设计备忘，尚未实现。记录对模型架构支持体系
> 的中长期演进规划，涵盖半步图组装器、组件级调度、按需编译三个方向。

## 1. 问题：架构数线性增长

当前每个 `general.architecture` 对应一套手写 Rust forward 代码：

| 架构 | 代码量（估算） |
|---|---|
| qwen3 / qwen3vl | ~2000 行 |
| qwen35 | ~1800 行 |
| llama 家族（llama / k2-horizon / granite / nanbeige / qwen2_2） | ~1500 行 |
| gemma4 | ~1600 行 |
| lfm2 / lfm25 / lfm2moe | ~4000 行 |
| spark2_5 | ~1200 行 |
| nemotron_h | ~1100 行 |
| hunyuan-dense | ~800 行 |
| **合计** | **~14000 行** |

新架构接入 = 新写 ~1500 行。到 2030 年可能有 30+ 个架构，代码量不可持续。
大部分 trunk 之间是 copy-paste + 微调（norm 位置、激活函数、RoPE 变体）。

## 2. 半步图组装器（layer_template）

### 2.1 定位

**不是完整图执行器，而是组件级模板参数化。** 在 GGUFRS v2 的组件元数据里
增加可选的 `layer_template` 字段，让加载方按模板描述选择 forward 函数，
而不是按 `general.architecture` 字符串 match。

### 2.2 模板描述格式（草案）

```json
{
  "layer_template": {
    "type": "standard_transformer",
    "components": ["attn_norm", "attn_qkv", "rope", "attention", "attn_output",
                   "ffn_norm", "ffn_gate", "ffn_up", "silu_gated", "ffn_down"],
    "params": {
      "norm_type": "rms",
      "rope_type": "neox",
      "ffn_act": "silu_gated",
      "qk_norm": true,
      "block_count": 28
    },
    "layer_overrides": {
      "0-3": { "ffn_act": "gelu", "qk_norm": false }
    }
  }
}
```

### 2.3 已有 trunk 的模板分类

| 模板 | 覆盖的架构 | 当前代码量 | 合并后 |
|---|---|---|---|
| `standard_transformer` | qwen3, qwen35, llama, hunyuan, nanbeige, qwen2_2 | ~8000 行 | ~2000 行（1 份） |
| `standard_transformer_gelu` | gemma4 | ~1600 行 | 复用 + ~200 行差异分支 |
| `hybrid_attention_ssm` | lfm2, lfm25, lfm2moe | ~4000 行 | ~2000 行（1 份） |
| `mamba2_hybrid` | nemotron_h | ~1100 行 | ~1100 行 |
| `spark_moe` | spark2_5 | ~1200 行 | ~1200 行 |
| **合计** | 10 个架构 | ~14000 行 | **~7500 行** |

### 2.4 向后兼容

`layer_template` 是**可选字段**：

| 文件类型 | 有 layer_template | 处理方式 |
|---|---|---|
| GGUF（上游产出） | ❌ | 走当前路径：`general.architecture` → per-arch Rust 函数 |
| GGUFRS v2（有模板） | ✅ | 走新路径：`layer_template.type` → 模板 forward 函数 |
| GGUFRS v2（无模板） | ❌ | 兼容降级：走当前路径 |

两条路径共存——不分裂生态，不破坏现有 GGUF 兼容性。

## 3. 为什么不做完整图执行器

### 3.1 三种方案对比

| | 半步组装器 | 完整图执行器（ONNX 式） | 当前（per-arch 代码） |
|---|---|---|---|
| 冷启动开销 | < 1ms | ~100ms（图解析 + JIT） | 0 |
| 热路径分派 | 0（编译器 inline） | 0（函数指针，但无法 inline） | 0 |
| 热路径算子间 | 不落内存（跨函数 inline） | 落内存（op 边界阻断 inline） | 不落内存 |
| 量化融合 kernel | ✅ 手写融合 | ❌ custom op 是黑盒 | ✅ 手写融合 |
| 新架构接入成本 | 零（若模板已有） / 写模板（若新结构） | 零（若算子已有） / 写 custom op | ~1500 行 |
| 工程量 | 低（4-5 个模板函数） | 极高（JIT + 算子注册 + 内存规划） | 已完成 |

### 3.2 核心矛盾：量化与图执行器不可调和

block-level 量化（Q8_0 / Q4_K / IQ4_XS）的性能收益在**算子内部**——SIMD
指令选择、反量化与乘加融合、block 布局。图执行器的优化能力在**算子之间**——
算子融合、常量折叠、内存复用。

把量化塞进图执行器的标准算子有两条路，都走不通：

- **先 Dequant 再 MatMul** → 内存爆炸（600MB Q8_0 展开成 4GB F32）
- **Custom Op（`Q8_0MatMul`）** → 执行器对它是黑盒，无法融合，回到手写 kernel

所以 llama.cpp 和本项目选择"架构在代码里"——不是不会写图执行器，是图执行器
对量化场景的优化承诺无法兑现。

### 3.3 大模型加剧了这个矛盾

大模型是 memory-bound（算术强度 ~4 FLOP/byte），瓶颈在"读权重"不在"算"。
图执行器的自动算子融合主要省**计算开销**，对 memory-bound 场景收益有限。
唯一能省带宽的是"反量化 + 乘加融合"（权重只读一次），而这恰恰是图执行器
做不到的。

## 4. 组件级组装（远期方向）

### 4.1 从模板到组件

更激进的演进：不按 `general.architecture` 分派，按 **tensor prefix** 识别
组件，独立调度。

### 4.2 缝合怪模型示例

一个混合多厂商组件的模型可以自动支持：

```json
{
  "components": [
    {"name": "vision",  "tensor_prefix": "v.",          "type": "vit"},
    {"name": "audio",   "tensor_prefix": "a.",          "type": "qwen3a_audio"},
    {"name": "trunk",   "tensor_prefix": "blk.",        "type": "standard_transformer"},
    {"name": "decoder", "tensor_prefix": "decoder.ssm.", "type": "mamba2"}
  ],
  "data_flow": ["vision", "audio", "trunk", "decoder"]
}
```

加载方按 `data_flow` 顺序调用各组件模块。组件类型是有限的（~10-15 种覆盖
主流架构），但排列组合是无限的——新架构 = 已有组件的新组合，零代码接入。

### 4.3 组件类型清单（估算）

| 组件类型 | 用途 | 已有实现 |
|---|---|---|
| `standard_transformer` | 文本 LLM trunk | ✅ qwen3 / llama / gemma4 |
| `hybrid_attention_ssm` | attention + shortconv 混合 | ✅ lfm2 / lfm25 / lfm2moe |
| `mamba2` | Mamba2 SSM 层 | ✅ nemotron_h |
| `vit` | 视觉编码器（ViT） | ✅ qwen3vl / qwen35 vision |
| `qwen3a_audio` | Qwen3 音频编码器 | ✅ qwen3 asr |
| `funasr_encoder` | FunASR 语音编码器 | ✅ funasr |
| `clip_projector` | mmproj 投影器 | ✅ qwen35 / qwen3vl |
| `tts_talker` | TTS 解码器 | ✅ qwen3 tts |
| `dac_vocoder` | DAC 声码器 | ✅ qwen3 tts codec |
| `diffusion_dit` | 扩散模型 DiT | ✅ z-image / dreamx |
| `vae` | VAE 编/解码器 | ✅ z-image / dreamx |
| `spark_moe` | Spark MoE FFN | ✅ spark2_5 |

## 5. Feature gate（特性门控编译）

### 5.1 问题

随着架构增多，二进制无限膨胀。2030 年可能支持 30+ 个架构，但用户通常只用
2-3 个。不需要把所有架构的代码都编进 exe。

### 5.2 方案

用 Rust `#[cfg(feature = "...")]` 控制编译范围：

```toml
# Cargo.toml
[features]
default = ["qwen3", "qwen35", "llama", "gemma4"]

# 按架构门控
qwen3 = []
qwen35 = []
llama = []       # 含 k2-horizon / granite / nanbeige / qwen2_2
gemma4 = []
lfm2 = ["lfm25", "lfm2moe"]
lfm25 = []
lfm2moe = []
spark = []
nemotron_h = []
hunyuan = []
funasr = []

# 全量
all-models = ["qwen3", "qwen35", "llama", "gemma4", "lfm2", "spark",
              "nemotron_h", "hunyuan", "funasr"]
```

```rust
// models/mod.rs
#[cfg(feature = "qwen3")]  pub mod qwen3;
#[cfg(feature = "llama")]  pub mod llama;
// ...

// dispatcher
match arch {
    #[cfg(feature = "qwen3")]  "qwen3" => ...,
    #[cfg(feature = "llama")]  "llama" => ...,
    other => Err(format!("{other:?} not compiled in")),
}
```

### 5.3 使用方式

```bash
# 只编 Qwen3 — 最小二进制
cargo build --release --features qwen3

# 编主流 4 个 — default
cargo build --release

# 全量
cargo build --release --features all-models

# 按需组合
cargo build --release --features qwen3,gemma4,lfm2
```

### 5.4 估算效果

| 构建方式 | 估算二进制 | 架构覆盖 |
|---|---|---|
| `--features qwen3` | ~15 MB | 1 个 |
| default | ~25 MB | 4 个主流 |
| `--features all-models` | ~40 MB | 全部 |

**源码全保留，编译按需选。** 过时架构的源码留在仓库里（git 可查），但默认
不编进 exe。

### 5.5 与 layer_template 的关系

两个方案正交：

| 方案 | 解决什么 |
|---|---|
| layer_template | 减少需要写的代码量（10 个架构 → 4-5 个模板） |
| feature gate | 减少编进二进制的代码量（10 个架构 → 按需编译） |

可叠加：layer_template 减少模板数，feature gate 让用户只编译需要的模板。

## 6. 落地优先级

| 方向 | 优先级 | 理由 |
|---|---|---|
| feature gate | 高（可立即做） | 纯 Cargo.toml + `#[cfg]` 改动，零风险，立竿见影 |
| layer_template | 中（需设计） | 需要定义模板格式 + 重构 forward 函数，但收益大 |
| 组件级组装 | 低（远期） | 需要组件注册表 + data_flow 描述 + 接口标准化 |

建议先做 feature gate（1-2 小时工作量），再规划 layer_template（需要先验证
模板格式能覆盖所有现有架构的差异）。
