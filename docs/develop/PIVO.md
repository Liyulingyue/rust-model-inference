# PiVo-1.4M 状态与实现记录

> 本页只汇总状态、限制和实现沿革。逐检查点的数值证据、覆盖矩阵、性能表和复现
> 命令在 [oracle 记录](../../tools/oracle/pivo/README.md)；模型清单见
> [MODEL_LIST.md](../MODEL_LIST.md)。
>
> 更新于 2026-11-10。

## 入口

CLI 和 HTTP 都有。CLI：

```bash
rust-model-inference \
  --tts --model <pivo-F32.gguf> \
  --prompt "NASA is ready for the next mission." --voice Pi --out speech.wav
```

- `--voice Pi|Vo`：两个内置音色，默认 `Pi`。
- `--speed 0.5..2`：语速，默认 `1`。
- `--pronunciations '{"zzvo":["Z","IY1","V","OW0"]}'`：生僻词的 ARPAbet 发音覆盖。
- 这三个参数是 PiVo 专属，用在非 PiVo 模型上会明确报错退出。

HTTP：

```bash
rust-model-server --tts --model <pivo-F32.gguf> --host 127.0.0.1 --port 8080
curl -f http://127.0.0.1:8080/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"input":"Hello, my name is Vo.","voice":"Vo","speed":1,"response_format":"wav"}' --output vo.wav
```

`response_format` 是 `wav`（默认）或裸 PCM16 `pcm`。并发合成返回 429，参数错误
返回 400，schema 错误返回 422。

## 组件与权重

一个 GGUF 自带全部组件：文本前端、声学网络、FARGAN 声码器和 CMU 发音词典。
不需要 `--mmproj`，也不需要额外组件。

| 组件 | 张量数 | 格式 |
| --- | --- | --- |
| 声学网络（文本编码 + 声学解码 + 时长预测） | 87 | F32（其中 52 个矩阵可导出为 Q8_0） |
| FARGAN 声码器 | 22 | F32（矩阵同上可导出为 Q8_0） |
| 归一化、bias、逐通道统计 | 41 个一维张量 | F32（不量化） |
| 行长无法按 32 对齐的矩阵与卷积核 | 16 | F32（不量化） |

总计 109 个张量、1,401,602 个参数（1,401,562 个学习参数加 40 个归一化缓冲值）。

## 计算后端

CPU 与 Auto 都能产出有效音频；Auto 走 CPU，打印 `scope=pivo_encoder backend=Cpu`，
输出与 CPU 逐字节相同。

**严格 Vulkan（`--compute vulkan`）把声学编码器放到设备上**，打印
`scope=pivo_encoder backend=Vulkan`。只有编码器上设备，时长头、解码器和声码器
留在宿主：

- 声码器（FARGAN）把上一帧的输出喂回下一帧，逐帧严格顺序，上设备只会每帧多
  一次提交，不省任何计算（一条音频 92% 的时间在这个循环里）。
- 解码器喂给音高周期，而周期舍入余量只有 1.36e-6，比宿主与设备的差异还小，
  上设备必然翻周期、改变音频。
- 时长头输出的是取整后的帧数，最短 oracle 用例的舍入余量是 0.0024 帧，宿主与
  设备必须在对数时长上远好于这个余量，帧数才可能一致。

设备侧的 7 个 shader（`pivo_linear` / `pivo_rms_norm` / `pivo_rope` /
`pivo_attention` / `pivo_conv1d` / `pivo_swiglu` / `pivo_silu`）全部用 PiVo 自己的
布局写，因为共享 kernel 的布局不同，来回都要转置。权重上传一次常驻，激活值留在
arena 里，整条编码器录进一个命令缓冲。

数值上设备与宿主一致：编码器相对误差 4.61e-7，对数时长误差 4.77e-7，**帧数完全
相同**，波形 RMS 误差 7.41e-5 满刻度（oracle 门限 7.5e-4）。旋转位置编码的正弦
余弦表由宿主填写——设备的超越函数与宿主差几个 ULP，而时长会取整，这点差别足以
改变音频长度。

**但设备路径更慢**：编码器只占一条音频 3–12% 的时间，而设备每次 dispatch 大约
50 微秒、编码器需要约 70 次。所以 Auto 留在 CPU（也保住了与 CPU 逐字节一致的
不变量），只有严格 Vulkan 才走设备。

| 用例 | `--compute cpu`（HTTP 请求） | `--compute vulkan`（HTTP 请求） | 比值 |
| --- | --- | --- | --- |
| 21 token | 75.8 ms | 78.3 ms | 1.03× |
| 约 130 token | 561.7 ms | 630.7 ms | 1.12× |

同机性能（Intel Core Ultra 5 125H，x86_64，`release-fast`，冷进程到音频）：

| 用例 | Rust `--compute cpu` | 官方 ONNX Runtime（单线程） |
| --- | --- | --- |
| 1.87 秒音频 | 192.4 ms | 192.3 ms |
| 2.57 秒音频 | 219.9 ms | 240.2 ms |

RTF 约 0.09–0.10，都远快于实时。`--threads 1/2/4` 没有可测差异：模型只有
1.4M 参数，逐帧声码器循环占主导，线程池无事可并行。完整记录见
[performance.json](../../tools/oracle/pivo/performance.json)。

## 量化：Q8_0 导出存在，但音质不可用

转换器支持 `--quant q8_0`，文件从 9,252,704 字节缩到 6,988,192 字节
（-24.5%）。**但这个格式的音频不能用**：PiVo 的 FARGAN 反馈环会把权重的
1/127 量化格点放大成载波相位漂移，完整波形误差达 0.82 满刻度，逐帧时长最多
7.14%、音高周期最多 23.23% 与官方图不再一致。

反量化本身是精确的（52 个量化张量的最大误差正好是量化格点半 ULP），问题在
模型而不在量化器。详细数据和建议见
[validation.md](../../tools/oracle/pivo/validation.md#q80-export-the-audio-is-not-usable)。

因此：loader 保留该格式可读，CLI 打印 `format=pivo-q8_0-v1` 并警告音频不可用，
**实际使用请选 F32 导出**。

## 限制

- 仅英文、Pi/Vo 两个音色。
- 无 JEV、无 chat、无 embedding：这些路由在该后端上返回 404。
- 无参考音频克隆、无采样控制、无流式 HTTP。
- 仅 F32 可用（见上）。
- 设备路径只覆盖声学编码器，且实测比宿主慢；Auto 不走设备。
- 长时间文本的音质与听感测试仍未验证；设备路径只在一块 Intel 核显上测过，
  没有独显可用。
