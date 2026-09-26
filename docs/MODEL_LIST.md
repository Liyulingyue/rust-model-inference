# 模型清单

| 模型                          | 版本       | 任务类型             | 参考实现                              | 参考推理权重 | 备注     |
|-------------------------------|------------|----------------------|---------------------------------------|--------------|----------|
| Breeze-TTS-2                  | 3.5B          | TTS、指令控制、参考音频声音克隆 | 原始开发                             | https://huggingface.co/BreezeBlue/breeze-tts-2 | √ |
| dots.tts-base                 | base       | TTS                  | 原始开发 @ `32407a5`                  | https://huggingface.co/EvoAwaken-Workshop/dots-tts-base-gguf | 待核验 |
| dots.tts-edit                 | tts        | TTS                  | 原始开发 @ `32407a5`                  |              | 待核验 |
| DreamX-Creator                | 7B + 5B Refiner | 首帧驱动音视频生成 | AMAP-ML/DreamX-Creator | https://modelscope.cn/models/GD-ML/DreamX-Creator | CPU 缩小全链路已核验 |
| Fun-ASR                  | FunASR / SenseVoiceSmall / Paraformer  | ASR | FunASR llama.cpp @ `v0.2.6` | https://www.modelscope.cn/models/FunAudioLLM/Fun-ASR-Nano-GGUF / https://www.modelscope.cn/models/FunAudioLLM/fsmn-vad-GGUF / https://www.modelscope.cn/models/FunAudioLLM/SenseVoiceSmall-GGUF / https://www.modelscope.cn/models/FunAudioLLM/Paraformer-GGUF | √ |
| Falcon-H1                  | 1.5B / 3B | 文本                 | llama.cpp @ `171e8846b`                | https://modelscope.cn/models/unsloth/Falcon-H1-1.5B-Instruct-GGUF / https://modelscope.cn/models/unsloth/Falcon-H1-3B-Instruct-GGUF | √ (Experimental；1.5B/3B 同型（仅 n_embd/n_layer/n_head/n_ff/ssm_* 变），零代码改动；连贯 ChatML/raw 生成 + 逐位 tokenizer 对齐；张量级 Q8 matmul 余 ulp 误差) |
| Gemma-4                   | E2B / E4B / 12B          | 文本、图像、音频      | llama.cpp @ `3173a56` | https://www.modelscope.cn/models/unsloth/gemma-4-E2B-it-GGUF | √ |
| Granite-4.0                       | 1B           | 文本                 | llama.cpp                                      | https://www.modelscope.cn/models/unsloth/granite-4.0-1b-GGUF             | √ |
| Hy-MT2                       | 1.8B / 7B | 文本（翻译）                 | llama.cpp                   | https://www.modelscope.cn/models/fss618/Hy-MT2-1.8B-GGUF        | √ |
| Jina-Embeddings-v5-Omni       | Small Retrieval | 文本、图像、音频 Embedding | llama.cpp @ `b96806d` |              | Q8_0 文本模型 + F16 vision/audio mmproj 已逐位核验；音频限单个 30 秒块，[对照说明](../tools/oracle/jina_audio/README.md) |
| K2-Horizon                    | 0.9B / 3.7B / 7B         | 文本                 | MBZUAI-IFM/llama.cpp | https://modelscope.cn/models/IFM/K2-Horizon-7B-GGUF | √ |
| LFM2                          | 350M / 700M / 1.2B / 8B-A1B     | 文本          | llama.cpp                                      | https://www.modelscope.cn/models/unsloth/LFM2.5-8B-A1B-GGUF             | √ |
| LFM2.5                        | 230M / 1.2B / 1.2B-Thinking / 2.6B / 8B-A1B    | 文本                 | llama.cpp                             | https://www.modelscope.cn/models/unsloth/LFM2.5-1.2B-Instruct-GGUF             | √ |
| LFM2.5-VL                     | 450M / 1.6B / 3B       | 文本、图像        | llama.cpp                             | https://www.modelscope.cn/models/unsloth/LFM2.5-VL-3B-GGUF             | √ |
| MiniCPM5                      | 1B / 2B         | 文本                 |                                       | https://www.modelscope.cn/models/OpenBMB/MiniCPM5-1B-GGUF             | √ |
| Nanbeige4.2-3B | 3B | 文本 | llama.cpp @ `b96806d` | https://www.modelscope.cn/models/Abiray/Nanbeige4.2-3B-GGUF | √（Q8_0 标量/F32、NEON/F32/F16 KV；[验证范围](usage/llama.md#3-nanbeige42-3b)） |
| NeoHorse-1                   | 4B / 9B        | 文本                 | llama.cpp @ `b96806d`                 | https://huggingface.co/TokenRhythm/NeoHorse-1-4B-GGUF | √ |
| Ornith-1.5                    | 9B         | 文本                 |                                       |              | 待核验 |
| Qwen2.5-Omni                  | 3B           | 文本、音频、视频、图像（文本、音频输出）               | llama.cpp                                      | https://www.modelscope.cn/models/unsloth/Qwen2.5-Omni-3B-GGUF             | √ |
| Qwen2.5-VL                    | 3B           | 文本、图像               |                                       | https://www.modelscope.cn/models/unsloth/Qwen2.5-VL-3B-Instruct-GGUF             | √ |
| Qwen3                         | 0.6B       | 文本                 | llama.cpp                                      | https://www.modelscope.cn/models/unsloth/Qwen3-0.6B-GGUF             | √ |
| Qwen3-Reranker                 | 0.6B       | 跨编码器重排序（query/document 打分） | 复用 qwen3 trunk + `cls.output.weight` | https://modelscope.cn/models/ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF | √ (Experimental；`qwen3_rerank` CLI + `rust-model-server --model ...` 自动检测 `pooling_type=4` 并开 `/v1/rerank` HTTP 路由（Cohere/Jina 兼容 schema）；ChatML prompt；巴黎/光合基准验证相关 vs 不相关文档区分 >1.5 数量级；`tests/qwen3_rerank.rs` 单元测试 + `tests/qwen3_rerank_http.rs` 起 axum + curl 集成测试 3/3 过) |
| Qwen3-Embedding               | 0.6B       | Embedding            |                                       | https://www.modelscope.cn/models/Qwen/Qwen3-Embedding-0.6B-GGUF             | √ |
| Qwen3-ASR                     | 0.6B       | ASR                  | llama.cpp                                      |  https://www.modelscope.cn/models/ggml-org/Qwen3-ASR-0.6B-GGUF            | √ |
| Qwen3-Omni-MoE                |            | 多模态               |                                       |              | 待核验 |
| Qwen3-TTS                     | 12Hz-1.7B-Base | TTS        | llama.cpp @ `201e50c`                 |              | 待核验 |
| Qwen3-VL                      | 0.6B / 2B  | 多模态               |                                       |              | 待核验 |
| Qwen3.5                       | 0.8B / 2B  | 文本                 | llama.cpp @ `b96806d`                 |              | 待核验 |
| Qwen3.8                       | 27B        | 多模态               | llama.cpp @ `b96806d`                 |              | 待核验 |
| Qwen-Drive-1.0               | 4B         | 自动驾驶多模态感知 / 规划 | 官方实现 @ `28091c1`；llama.cpp @ `b96806d` | https://modelscope.cn/models/Qwen/Qwen-Drive-1.0-4B | BF16 VLM、mmproj、SFT/RL planner 与 F32 perception 已导出；Tokenizer、规划 checkpoint 和感知 BF16 算子逐位核验；CUDA 端到端感知尚未核验；[导出、哈希与限制](../tools/oracle/qwen_drive/README.md) |
| Spark-X2.5                    | 1.7B / 4B  | 文本                 | XHToken/llama.cpp                     | https://www.modelscope.cn/models/XHToken/Spark-X2.5-4B-GGUF             | √ |
| VibeVoice-ASR                 | 7B         | ASR                  |                                       |              | 待核验 |
| Z-Image                       | Turbo      | 文生图               | leejet/stable-diffusion.cpp @ `97d2990` |            | 待核验 |
| NVIDIA-Nemotron-3-Nano        | 4B         | 文本                 | llama.cpp @ `b96806d`               | https://www.modelscope.cn/models/unsloth/NVIDIA-Nemotron-3-Nano-4B-GGUF             | √ |
