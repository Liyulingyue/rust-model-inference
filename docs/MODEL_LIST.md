# 模型清单

| 模型                          | 版本       | 任务类型             | 参考实现                              | 参考推理权重 | 备注     |
|-------------------------------|------------|----------------------|---------------------------------------|--------------|----------|
| dots.tts-base                 | base       | TTS                  | 原始开发 @ `32407a5`                  | https://huggingface.co/EvoAwaken-Workshop/dots-tts-base-gguf | 待核验 |
| dots.tts-edit                 | tts        | TTS                  | 原始开发 @ `32407a5`                  |              | 待核验 |
| DreamX-Creator                | 7B + 5B Refiner | 首帧驱动音视频生成 | https://github.com/AMAP-ML/DreamX-Creator | https://modelscope.cn/models/GD-ML/DreamX-Creator | CPU 缩小全链路已核验 |
| Gemma-4                       | E2B        | 文本、图像、音频      | llama.cpp @ `3173a56`                 |              | 待核验 |
| Granite                       |            | 文本                 |                                       |              | 待核验 |
| Hy-MT2                       | 1.8B | 文本                 | llama.cpp                   | https://www.modelscope.cn/models/fss618/Hy-MT2-1.8B-GGUF        | 待核验 |
| Jina-Embeddings-v5-Omni       |            | Embedding            |                                       |              | 待核验 |
| K2-Horizon                    | 7B         | 文本                 | https://github.com/MBZUAI-IFM/llama.cpp | https://modelscope.cn/models/IFM/K2-Horizon-7B-GGUF | BF16 已核验（ARM64 / x86_64） |
| LFM2                          | 8B-A1B     | 文本          |                                       |              | 待核验 |
| LFM2.5                        | 230M / 1.2B    | 文本                 | llama.cpp                             | https://www.modelscope.cn/models/unsloth/LFM2.5-1.2B-Instruct-GGUF             | √ |
| LFM2.5-VL                     | 450M / 3B       | 文本、图像        | llama.cpp                             |              | 待核验 |
| MiniCPM5                      | 1B         | 文本                 |                                       |              | 待核验 |
| Nanbeige                      |            | 文本                 |                                       |              | 待核验 |
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
| Spark-X2.5                    | 1.7B / 4B  | 文本                 | XHToken/llama.cpp                     | https://www.modelscope.cn/models/XHToken/Spark-X2.5-4B-GGUF             | 待核验 |
| VibeVoice-ASR                 | 7B         | ASR                  |                                       |              | 待核验 |
| Z-Image                       | Turbo      | 文生图               | leejet/stable-diffusion.cpp @ `97d2990` |            | 待核验 |
| NVIDIA-Nemotron-3-Nano        | 4B         | 文本                 |                                      |              | 待核验 |
