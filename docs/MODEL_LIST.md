# 模型清单

| 模型                          | 版本       | 任务类型             | 参考实现                              | 参考推理权重 | 备注     |
|-------------------------------|------------|----------------------|---------------------------------------|--------------|----------|
| dots.tts-base                 | base       | TTS                  | 原始开发 @ `32407a5`                  | https://huggingface.co/EvoAwaken-Workshop/dots-tts-base-gguf | 待核验 |
| dots.tts-edit                 | tts        | TTS                  | 原始开发 @ `32407a5`                  |              | 待核验 |
| DreamX-Creator                | 7B + 5B Refiner | 首帧驱动音视频生成 | https://github.com/AMAP-ML/DreamX-Creator | https://modelscope.cn/models/GD-ML/DreamX-Creator | CPU 缩小全链路已核验 |
| Gemma-4                       | E2B        | 文本、图像、音频      | llama.cpp @ `3173a56`                 | https://www.modelscope.cn/models/unsloth/gemma-4-E2B-it-GGUF             | √ |
| Granite-4.0                       | 1B           | 文本                 | llama.cpp                                      | https://www.modelscope.cn/models/unsloth/granite-4.0-1b-GGUF             | √ |
| Hy-MT2                       | 1.8B | 文本                 | llama.cpp                   | https://www.modelscope.cn/models/fss618/Hy-MT2-1.8B-GGUF        | 待核验 |
| Jina-Embeddings-v5-Omni       |            | Embedding            |                                       |              | 待核验 |
| K2-Horizon                    | 7B         | 文本                 | https://github.com/MBZUAI-IFM/llama.cpp | https://modelscope.cn/models/IFM/K2-Horizon-7B-GGUF | BF16 已核验（ARM64 / x86_64） |
| LFM2                          | 8B-A1B     | 文本          | llama.cpp                                      | https://www.modelscope.cn/models/unsloth/LFM2.5-8B-A1B-GGUF             | √ |
| LFM2.5                        | 230M / 1.2B    | 文本                 | llama.cpp                             | https://www.modelscope.cn/models/unsloth/LFM2.5-1.2B-Instruct-GGUF             | √ |
| LFM2.5-VL                     | 450M / 3B       | 文本、图像        | llama.cpp                             | https://www.modelscope.cn/models/unsloth/LFM2.5-VL-3B-GGUF             | √ |
| MiniCPM5                      | 1B         | 文本                 |                                       |              | 待核验 |
| Nanbeige                      |            | 文本                 |                                       |              | 待核验 |
| NeoHorse-1                   | 4B         | 文本                 | llama.cpp @ `b96806d`                 | https://huggingface.co/TokenRhythm/NeoHorse-1-4B-GGUF | BF16 / F16 / Q8_0 已逐位核验；Q4_K_M / Q5_K_M 对齐标量量化 Oracle（ARM64、4 步）；[边界与命令](../tools/neohorse/README.md#neohorse-1-4b-官方-gguf-对比) |
| NeoHorse-1                   | 9B         | 文本                 | llama.cpp @ `b96806d`                 | https://huggingface.co/TokenRhythm/NeoHorse-1-9B | BF16 + NFC 已核验（ARM64 CPU、F32 KV、4 步逐位对齐）；[转换说明](../tools/neohorse/README.md)，发布权重不含 MTP |
| Ornith-1.5                    | 9B         | 文本                 |                                       |              | 待核验 |
| Qwen2.5-Omni                  |            | 多模态               |                                       |              | 待核验 |
| Qwen2.5-VL                    |            | 多模态               |                                       |              | 待核验 |
| Qwen3                         | 0.6B       | 文本                 | llama.cpp                                      | https://www.modelscope.cn/models/unsloth/Qwen3-0.6B-GGUF             | √ |
| Qwen3-Embedding               | 0.6B       | Embedding            |                                       |              | 待核验 |
| Qwen3-ASR                     | 0.6B       | ASR                  | llama.cpp                                      |  https://www.modelscope.cn/models/ggml-org/Qwen3-ASR-0.6B-GGUF            | √ |
| Qwen3-Omni-MoE                |            | 多模态               |                                       |              | 待核验 |
| Qwen3-TTS                     | 12Hz-1.7B-Base | TTS        | llama.cpp @ `201e50c`                 |              | 待核验 |
| Qwen3-VL                      | 0.6B / 2B  | 多模态               |                                       |              | 待核验 |
| Qwen3.5                       | 0.8B / 2B  | 文本                 | llama.cpp @ `b96806d`                 |              | 待核验 |
| Qwen3.8                       | 27B        | 多模态               | llama.cpp @ `b96806d`                 |              | 待核验 |
| Spark-X2.5                    | 1.7B / 4B  | 文本                 | XHToken/llama.cpp                     | https://www.modelscope.cn/models/XHToken/Spark-X2.5-4B-GGUF             | √ |
| VibeVoice-ASR                 | 7B         | ASR                  |                                       |              | 待核验 |
| Z-Image                       | Turbo      | 文生图               | leejet/stable-diffusion.cpp @ `97d2990` |            | 待核验 |
| NVIDIA-Nemotron-3-Nano        | 4B         | 文本                 |                                      |              | 待核验 |
