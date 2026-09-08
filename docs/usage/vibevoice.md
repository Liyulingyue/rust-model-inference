# VibeVoice ASR 用法

VibeVoice ASR（microsoft/VibeVoice-ASR-Streaming-7B）是 arch-qwen2 LLM +
`vibevoice_asr` mmproj 的流式 ASR 配对。代码侧入口
`src/app/vibevoice.rs::run_vibevoice_asr_cli`。

> 共用前置：构建 `cargo build --release --bin rust-model-inference`。
> 必须**同时**传 `--mmproj`（含 `vibevoice_asr` projector 元数据）和
> `--audio`（WAV）。

## 1. 推理

```bash
cargo run --release --bin rust-model-inference -- \
  --model models/VibeVoice-ASR-Streaming-7B-Q8_0.gguf \
  --mmproj models/mmproj-VibeVoice-ASR-Streaming-7B-F16.gguf \
  --audio models/sample.wav \
  --language en
```

CLI 路由（`src/app/audio.rs:14-30`）：

```rust
let is_vibevoice = crate::models::vibevoice_asr::is_vibevoice_asr_mmproj(probe.as_ref());
if is_vibevoice {
    return crate::app::vibevoice::run_vibevoice_asr_cli(options);
}
// 否则 fall through 到通用 ASR 路由
```

VibeVoice ASR 的 mmproj 通过 `clip.projector_type` 或 `general.architecture`
等于 `vibevoice_asr` 来识别（`src/models/vibevoice_asr/config.rs:67`）。

## 2. mmproj 元数据约定

`vibevoice_asr` mmproj 期望的关键 metadata（节选自
`src/models/vibevoice_asr/config.rs:123-200`）：

| 键 | 说明 |
|---|---|
| `vibevoice.sample_rate` | 24 kHz |
| `vibevoice.compress_ratio` | 3200 |
| `vibevoice.chunk_frames` | 22 |
| `vibevoice.lookahead_frames` | 4 |
| `vibevoice.llm_hidden_size` | 3584（与 Qwen2.5-7B decoder 对齐） |
| `vibevoice.acoustic.vae_dim` / `vibevoice.semantic.vae_dim` | 双 VAE latent 维度 |
| `vibevoice.encoder.n_filters` | conv stem 输出通道 |
| `vibevoice.encoder.ratios` / `vibevoice.encoder.depths` | 下采样比 + 每段 block 数（depths 长度 = ratios 长度 + 1） |
| `vibevoice.encoder.kernel_size` / `last_kernel_size` | conv kernel |
| `vibevoice.encoder.ffn_expansion` | FFN 扩展比 |
| `vibevoice.encoder.layernorm_eps` / `vibevoice.connector.eps` | LayerNorm eps |
| `vibevoice.acoustic.fix_std` / `vibevoice.acoustic.std_dist_type` | 声学 latent 标准化参数 |
| `vibevoice.encoder.pad_mode` | 仅支持 `constant` |
| `vibevoice.encoder.mixer_layer` | 仅支持已知 mixer |

非法配置（kernel_size / pad_mode / mixer_layer / std_dist_type / 维度不匹配）
在 config-load 阶段拒绝（见 `config.rs:166-205`）。

## 3. 已知约束

| 范围 | 行为 |
|---|---|
| 缺 `--mmproj` | 推理前报错：`VibeVoice ASR requires --mmproj` |
| 缺 `--audio` | 推理前报错：`VibeVoice ASR requires --audio` |
| 流式 | `transcribe_streaming` 是流式；chunk window + lookahead 由 metadata 决定 |
| `pad_mode != constant` | 加载拒绝：`Unsupported metadata: vibevoice.encoder.pad_mode {pad_mode:?}; only constant is supported` |

## 4. 与 llama.cpp / Oracle 的对齐

`docs/REFERENCE_IMPLEMENTATIONS.md` **没有** VibeVoice ASR 的 Pinned Oracle
（既无 pinned commit，也无 build 脚本，也无测试）。当前覆盖仅限
`src/models/vibevoice_asr/config.rs` 内的元数据校验与单元测试。

## 5. 相关源码索引

- `src/app/audio.rs` — ASR dispatcher（先探 mmproj，再决定走 vibevoice 还是 qwen3-asr）
- `src/app/vibevoice.rs` — CLI 入口
- `src/models/vibevoice_asr/` — 完整 Rust 实现：encoder、connector、LLM、generate
- `src/models/vibevoice_asr/config.rs` — 元数据校验
- `docs/REFERENCE_IMPLEMENTATIONS.md` — 当前无 pinned Oracle