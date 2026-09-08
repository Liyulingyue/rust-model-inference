# 模型支持清单

> 更新于 2026-09-08，DreamX-Creator 代码基线为 `1178200`。本清单以主 CLI `rust-model-inference` 为准。

相同的 `general.architecture` 只表示会进入同一条代码路径，不代表任意同架构 GGUF 都已确认可用。未在“具体型号”表中出现的模型，应先按 `Supported` 或 `Experimental` 看待，不能默认视为 `Verified`。

## 状态定义

| 状态 | 含义 |
|---|---|
| `Verified` | 已使用真实 GGUF 完成端到端运行；证据列会说明是否还做了 llama.cpp/Python Oracle 对齐。 |
| `Supported` | 架构、张量和运行路径已接入，并有代码或测试覆盖，但该具体型号尚无独立的真实 GGUF 端到端记录。 |
| `Experimental` | 已有入口或部分实现，但仍存在已知正确性缺口、失败记录或缺少可用模型验证。 |
| `Unsupported` | 当前明确未实现、受限制或会被代码拒绝。 |

`Verified` 不等于所有量化格式都已验证；只保证“已验证格式”列出的组合。

## 已确认的具体型号

| 型号 | GGUF architecture | 能力 | 所需组件 | 已验证格式 | 状态 | 证据 / 限制 |
|---|---|---|---|---|---|---|
| Qwen3-0.6B | `qwen3` | 文本生成 | 无 | Q8_0 | `Verified` | README 主路径和真实模型推理；其他 Qwen3 尺寸不自动继承此状态。 |
| Qwen3-Embedding-0.6B | `qwen3` | 文本 Embedding | `--embedding` | Q8_0 | `Verified` | [`tests/embedding_parity.rs`](tests/embedding_parity.rs) 覆盖 pinned llama.cpp 向量和位级对照。 |
| Qwen3-ASR-0.6B | `qwen3vl` | 语音识别 | Qwen3-ASR mmproj、WAV | Q8_0 LLM + Q8_0 mmproj | `Verified` | [`src/format/ggufrs.rs`](src/format/ggufrs.rs) 有固定文件哈希的 raw GGUF/GGUFRS 转写等价测试；仅支持 greedy 解码，不能同时传图像。 |
| Qwen3-TTS-12Hz-1.7B-Base | `qwen3tts` | TTS、参考音频声音克隆 | mmproj；克隆时还需参考 WAV/文本 | Q8_0 GGUF + mmproj | `Verified` | [`tests/qwen3_tts_reference.rs`](tests/qwen3_tts_reference.rs) 覆盖 pinned llama.cpp Oracle。 |
| dots.tts-base | `qwen2` LLM + `clip` mmproj | TTS | `dotstts` mmproj | 转换器产出的混合精度 GGUF + mmproj | `Verified` | [`tests/dots_tts_reference.rs`](tests/dots_tts_reference.rs) 覆盖 pinned Python Oracle 位级对照。 |
| dots.tts.edit | `qwen2` LLM + `clip` mmproj | 指令式语音编辑 | `dotstts` mmproj、源 WAV、编辑区间 | 转换器产出的混合精度 GGUF + mmproj | `Verified` | 与 Base 共用同一 Oracle 测试，覆盖 Edit 调度和完整波形链路。 |
| Qwen3.5-0.8B | `qwen35` | 文本、图像 | 图像需要 mmproj | Q8_0 LLM + F16 mmproj | `Verified` | 已有真实图文运行记录；未声明其他量化格式。 |
| Qwen3.5-2B | `qwen35` | 文本；图像路径已接入 | 图像需要匹配 mmproj | 实测 GGUF，量化后缀未固化 | `Verified` | `docs/TODO.md` 记录真实冒烟回归；未单独记录图像 Oracle。 |
| Qwen3.8-27B | `qwen35` | 文本、图像 | 图像需要 mmproj | 测试指定的 GGUF + mmproj | `Verified` | [`tests/qwen35_reference.rs`](tests/qwen35_reference.rs) 覆盖 pinned llama.cpp lossless checkpoints 和图像冒烟。Qwen3.8 是独立型号，不是 Qwen3-8B。 |
| Ornith-1.5-9B | `qwen35` | 文本生成 | 无 | 实测 GGUF，量化后缀未固化 | `Verified` | `docs/TODO.md` 记录 8/8 greedy token 与 llama.cpp 一致。 |
| MiniCPM5-1B | `llama` | 文本生成 | 无 | Q8_0 | `Verified` | `docs/TODO.md` 记录 8/8 greedy token 与 llama.cpp 一致。 |
| LFM2.5-1.2B-Instruct | `lfm2` | 文本生成 | 无 | Q8_0 | `Verified` | `docs/TODO.md` 记录 8/8 greedy token 与 llama.cpp 一致。 |
| LFM2-8B-A1B | `lfm2moe` | MoE 文本生成 | 无 | Q8_0 | `Verified` | 真实 GGUF 可完整生成；与 llama.cpp 前 6 个生成 token 一致，随后在 MoE 近平局处可能分叉。 |
| Spark-X2.5-1.7B | `spark2_5` | 文本生成、thinking | 无 | BF16 | `Verified` | 真实 GGUF 中英文和算术冒烟通过；尚未完成 XFllama.cpp token 级 Oracle 对齐。 |
| Spark-X2.5-4B | `spark2_5` | 文本生成、thinking | 无 | BF16 | `Verified` | 真实 GGUF 冒烟通过；当前 CPU 路径较慢，尚未完成严格 Oracle 对齐。 |
| Gemma 4 E2B | `gemma4` | 文本、图像、音频、图像+音频 | 任意媒体输入都需要 F16 mmproj | Q8_0 LLM + F16 mmproj | `Verified` | [`tests/gemma4_reference.rs`](tests/gemma4_reference.rs) 覆盖 pinned llama.cpp、文本及各媒体组合；不支持视频，要求 greedy 解码。 |
| Z-Image Turbo | `pig` | 文生图 | DiT、Qwen3 文本编码器、Flux VAE | Q8_0 DiT + Q8_0 文本编码器 + F16 VAE | `Verified` | [`tests/z_image_reference.rs`](tests/z_image_reference.rs) 覆盖 pinned Oracle 和 prompt 敏感性；当前范围是 CPU、512×512。 |

## DreamX-Creator CPU 验证证据

- 导出耗时 180.72 秒。main 文件 12,833,754,816 bytes、3,045 tensors、SHA-256 `84fe47b35fcb21552dbd51cae0ac511c52b1261c3d4845ff099c01bef597fd02`；mmproj 文件 15,564,058,592 bytes、1,086 tensors、SHA-256 `3cc74cd84edd7e22078f932e73b7b92f41c8094188058c1c830d613002b59523`。
- 真实 pair preflight 和 dry-run 通过；64×64、4 spatial tokens、0.2 秒、5 fps、1 step、Flash upsampler、LightVAE decoder 的原生 CPU 运行耗时 45.39 秒，maximum RSS 25,620,398,080 bytes，估算峰值 14.81 GiB/64 GiB。
- 产物已由 ffmpeg 探测：64×64 H.264 base video、48 kHz 单声道 PCM16 WAV、base mux、128×128 H.264 refined video 和 refined mux，共五个文件。
- 这是结构、手写算子及缩小端到端运行验证。当前主机无 CUDA，因此 [`tools/dreamx/dreamx_oracle_trace.py`](../tools/dreamx/dreamx_oracle_trace.py) 只验证了明确的无 CUDA 退出；未生成官方 Python checkpoints，也未运行完整时长/空间 token 数的官方 2K 推理。
- Rust refiner 当前使用固定 `[1000, 750, 500, 250]` timesteps 和 `sigma_start=0.6251`；上游还会执行 shifted-scheduler warping/filtering。该差异尚未通过 Oracle 消解，因此当前状态是 `Experimental`，不能据此声明官方 refiner 数值或质量对齐。

## 已接入但未达到 Verified 的范围

| 模型 / 范围 | GGUF architecture | 能力 | 所需组件 | 当前覆盖 | 状态 | 证据 / 限制 |
|---|---|---|---|---|---|---|
| 其他 Qwen3 文本 GGUF | `qwen3` | 文本生成 | 无 | 通用 metadata/tensor 分发 | `Supported` | 未逐个验证尺寸和量化组合；应为目标 GGUF 补一次真实推理。 |
| Qwen3-VL 0.6B / 2B 配置 | `qwen3vl` | 文本、图像、视频 | `qwen3vl_merger` mmproj | 两组主模型维度白名单、视觉编码器和 CLI 路由 | `Supported` | 当前代码接受 1024-dim 与 2048-dim 两组配置；没有独立的生成式 VL Oracle 记录。 |
| Qwen2.5-Omni 兼容 GGUF 对 | `qwen2vl` + `qwen2.5o` projector | 文本、图像、视频、音频 | 匹配 mmproj | 多媒体编码、投影和生成路径 | `Supported` | 架构级覆盖，不代表所有 Qwen2.5-Omni 尺寸均可用。 |
| Qwen3-Omni MoE 兼容 GGUF 对 | `qwen3vlmoe` + `qwen3vl_merger` projector | 文本、图像、视频、音频 | 匹配 mmproj | MoE、媒体投影和生成路径 | `Supported` | shared-expert 张量仍会被明确拒绝。 |
| Jina Embeddings v5 Omni retrieval | 带 `pooling_type` 的 Qwen-family arch | 文本、图像、视频、音频 Embedding | 媒体输入需要匹配 mmproj | pooling、媒体编码、CLI 参数和单元测试 | `Supported` | 当前仓库没有固定真实 GGUF/Oracle 的回归测试。 |
| LFM2.5-VL | `lfm2` | 图文生成 | SigLIP + LFM2 projector mmproj | 图像预处理、投影和生成路径 | `Supported` | 未在仓库中固定具体型号和真实 GGUF Oracle。 |
| Hunyuan-MT2 / Hunyuan Dense | `hunyuan-dense` | 文本生成 | 无 | 专用 prompt 和 Qwen3 trunk 分发 | `Supported` | README 旧名单只给出型号名；当前没有固定真实 GGUF 的回归证据。 |
| Granite 兼容文本 GGUF | `granite` | 文本生成 | 无 | 专用 prompt、attention/logit scaling | `Supported` | 架构路径已接入，但未固定一个具体 Granite 型号作为 E2E 回归。 |
| DreamX-Creator | `dreamx` + `clip`/`dreamx_creator` | 首帧驱动的同步音视频生成、2x 视频 refiner | `DreamX-Creator-Q8_0.gguf` + `mmproj-DreamX-Creator-BF16.gguf` | 真实导出、pair preflight、64×64/1 帧 CPU 全链路和五个媒体产物 | `Experimental` | base/refiner 路径可运行，但 CUDA Oracle 未执行，且 refiner shifted scheduler 尚未逐 checkpoint 对齐；精确证据见上方。 |
| 通用 Qwen2 文本 GGUF | `qwen2` | 文本生成 | 无 | 可进入 Qwen trunk；dots.tts 内部 LLM 已使用 | `Experimental` | 当前没有“任意 Qwen2 文本模型”保证，不能用 dots.tts 的内部成功替代通用验证。 |
| Dense LFM2 v2 | `lfm2` | 文本生成 | 无 | 保留专用 trunk 分发 | `Experimental` | `docs/TODO.md` 明确记录当前模型库没有对应 GGUF。 |
| Nanbeige | `nanbeige` | 文本生成 | 无 | SPM tokenizer 和 llama trunk 路由 | `Experimental` | 合入提交标题明确标注“未成功”，因此不能列为确定支持。 |

## 明确不支持或受限

| 范围 | 状态 | 当前行为 |
|---|---|---|
| 未注册的 `general.architecture` | `Unsupported` | 加载或模型配置阶段返回 `Unsupported architecture`。 |
| 不匹配当前两组维度的 Qwen3-VL 主模型 | `Unsupported` | 配置阶段返回 `Unsupported main-model configuration`。 |
| 带 shared experts 的 `qwen3vlmoe` | `Unsupported` | 权重加载明确返回 shared experts not supported。 |
| Qwen3-ASR + 图像，或非零 temperature | `Unsupported` | CLI 在推理前拒绝。 |
| Gemma 4 视频输入 | `Unsupported` | 多模态入口明确拒绝 `--video`。 |
| Z-Image Base、img2img、GPU 路径 | `Unsupported` | 当前仅实现 Z-Image Turbo 的原生 Rust CPU 文生图。 |
| DreamX-Creator GPU、未匹配 GGUF pair | `Unsupported` | DreamX 当前只走原生 CPU；pair ID、组件清单、版本或精度 metadata 不匹配会在加载阶段拒绝。 |

## 架构注册表

主模型代码当前认识这些 architecture：`qwen2`、`qwen2vl`、`qwen3`、`qwen3vl`、`qwen3vlmoe`、`qwen35`、`qwen3tts`、`llama`、`granite`、`hunyuan-dense`、`pig`、`lfm2`、`lfm2moe`、`nanbeige`、`gemma4`、`spark2_5`、`dreamx`。其中 `gemma4`、`spark2_5` 和 `dreamx` 使用各自的专用配置加载路径；`clip` 是 mmproj 组件架构，不是可独立生成的主模型。

服务端只覆盖其中较窄的一组运行模式。具体限制见 [README 的“服务端模式”](README.md#服务端模式)；模型是否出现在本清单，不代表它已经支持服务端流式输出或请求级动态媒体输入。

模型实现对照过的外部仓库、固定提交和 Oracle 入口见 [参考实现与 Oracle 清单](REFERENCE_IMPLEMENTATIONS.md)。
