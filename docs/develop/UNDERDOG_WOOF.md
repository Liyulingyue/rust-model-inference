# Underdog Woof

英文 TTS，内置 `juno` / `mira` 两种声音，输出 24 kHz 单声道 PCM16 WAV。
单个 `underdog_woof` GGUF 包含文本嵌入、语音 transformer、flow head、
Mimi 解码器和发音词典，不需要 `--mmproj`。

当前状态为 **Experimental / 部分适配（GPU 验证阻塞、长句自由生成偏差待验收）**。
CPU 九种格式已做独立、同权重分阶段数值检查。Vulkan 投影路径已接入共享
`LinearExecutor`，但测试机 RADV/NAVI31 的共享设备路径出现 `device lost`，
Auto 回退 CPU，严格 Vulkan 失败；不能据此声称 GPU 已对齐或获得加速。
F32 长句 Mira 在教师强制时通过分阶段检查，但自由生成的波形相对误差为
`0.04634`，未通过预设 `0.002` 门槛。定位显示自回归反馈放大了浮点归约舍入；
官方实现自身的 ULP 扰动实验也复现了放大现象。原失败记录和阈值均保留，
尚未完成主观音质验收。
逐组合状态与实测记录见 [validation](../../tools/oracle/underdog_woof/validation.md)。

## 使用

在远端仓库内按 [Oracle README](../../tools/oracle/underdog_woof/README.md)
下载和转换模型，权重直接落在远端。运行：

```bash
target/release/rust-model-inference --tts \
  --model models/Underdog-Woof/woof-f32.gguf --threads 2 --compute cpu \
  --prompt 'Your order costs $12.50.' --voice juno --seed 0 --temp 0.5 --out speech.wav

target/release/rust-model-server --tts \
  --model models/Underdog-Woof/woof-f32.gguf --threads 2 --compute cpu \
  --host 127.0.0.1 --port 8080
curl http://127.0.0.1:8080/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"Hello world.","voice":"mira","seed":0,"temperature":0.5,"response_format":"wav"}' \
  --output speech.wav
```

CLI 默认 `voice=juno`、`seed=0`、`temperature=0.5`；seed 是 U32，temperature
范围 0–2。HTTP 接受同样参数，另有 `normalize`（默认 true）和
`response_format=wav|pcm`（pcm 为 24 kHz 单声道小端 S16）。服务器命令只设置
默认声音/seed/temperature，文本来自 HTTP；不接受显式 `--prompt` 或 `--out`。
模型同时加载后，`/v1/models` 给出可用于请求 `model` 字段的标识。

支持规则归一化、Unicode respelling、默认分句，以及静音/过长输出的三次重试与
最佳片段选择。每次请求独立初始化随机数与解码状态。原文最多 4096 字符；
归一化后最多 16384 字符，最多 128 个片段，每片段须容纳于 1100 位置上下文。
不支持声音克隆、语速控制、发音覆盖、其他语言模型、JEV 或 HTTP 流式输出。
内部八帧分块解码不代表流式协议支持。

## 精度和后端

转换器支持 F32/F16/BF16/Q8_0/Q4_0/Q4_1/Q4_K/Q5_K/Q6_K。
F32 指原始发布的 int8/FP16 权重精确反量化后的有效值；没有训练前量化的
全精度 checkpoint。K 格式中语音生成部分保持 F32，仅部分 decoder 矩阵量化。
所有格式的状态、KV 和累加保留 F32；F16 kernel 及块量化 kernel 对激活也有舍入。
正常 CPU 路径复用仓库 SIMD/线程池，量化矩阵保留原生量化 kernel。

F32 是默认建议；再次量化会改变输出，4bit 档的源权重损失尤其明显。
“同权重误差通过”不表示量化音质与 F32 相同，也不表示已经主观试听验收。
GPU 设计为投影 offload 与宿主归一化、RoPE、状态、卷积调度的混合执行，
不是整模型常驻设备。实测耗时和具体限制见
[performance.json](../../tools/oracle/underdog_woof/performance.json)。

同机 EPYC 9334、2 线程、F32、Juno 两句共 5.24 秒音频：三次暖运行中位数
Rust CLI 为 1.436 秒（含进程启动、加载），官方 NumPy 为 0.789 秒（仅合成，
另计构造 0.330 秒）。Auto 单次 5.292 秒且实际回退 CPU，严格 Vulkan 在
4.912 秒后失败；后二者不能作为 GPU 推理吞吐。两种 CPU 计时范围不同，
这些数字不构成同范围加速比。

## 许可

本适配派生自 **Underdog Woof，Copyright 2026 Conway**。
模型及其代码/转换版本受 [Underdog Non-Commercial License 1.0](../../tools/oracle/underdog_woof/licenses/LICENSE)
约束，原始 [NOTICE](../../tools/oracle/underdog_woof/licenses/NOTICE) 保留。
非商业使用须署名；应用提供本功能时应显示
[Voice by Underdog Woof](https://underdog.ai)，发布音频需在标题或描述中署名。
商业使用需另获上游许可，仓库 MIT 许可不替代这些条件。
