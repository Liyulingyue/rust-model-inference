# Ternary Bonsai 2.5 27B

模型使用现有 `qwen35` 文本和图像入口。将 `MODEL` 设为官方解码器 GGUF 的路径，`MMPROJ` 设为同一发布仓的 BF16 或 Q8_0 投影器。以下命令在存放权重的推理主机执行。[权重和校验值](../../tools/oracle/ternary_bonsai/README.md)

```sh
cargo build --locked --profile release-fast --features vulkan \
  --bin rust-model-inference --bin rust-model-server
bin=target/release-fast
"$bin/rust-model-inference" --model "$MODEL" --compute vulkan --jinja --no-thinking \
  --prompt 'Name the capital of France.' --temperature 0 --max-tokens 16
"$bin/rust-model-inference" --model "$MODEL" --mmproj "$MMPROJ" --image red.png \
  --compute vulkan --jinja --no-thinking --temperature 0 --max-tokens 8 \
  --prompt 'What color is the image? Answer with one word.'
"$bin/rust-model-server" --model "$MODEL" --mmproj "$MMPROJ" --compute vulkan \
  --jinja --host 127.0.0.1 --port 8080 --max-context 512 --prefill-batch-size 16
```

`--compute cpu` 使用多线程 CPU，`auto` 自动选择设备。文本贪心生成可对 `*-mtp.gguf` 启用 `--mtp-draft 2`。图像输入走目标模型，CLI 图像命令省略草稿参数。HTTP 支持 `/v1/chat/completions`，图像使用 `image_url` 内容块，流式请求加 `"stream":true`。JEV 的文本和图像路由分别为 `/v1/jev/score`、`/v1/jev/image`。分组评分使用 `/v1/jev/grouped` 或 `/v1/jev/image_grouped`，CLI 在相同的 `--jev-question` 后为每组传入 `--jev-block` 和对应的 `--jev-option`。

```sh
curl -sS http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Name the capital of France."}],"temperature":0,"enable_thinking":false,"max_tokens":16}'
```

## 精度和性能

三种文件均直接加载。CPU 首次使用投影矩阵时，将三元码和原始 FP16 缩放无损重排为 Q4_0 并缓存，401 个 CPU 投影合计 13.41 GiB。词嵌入按 token 解包一行。Vulkan 逐矩阵重排并上传，最大的临时矩阵为 682.03 MiB 的输出头，上传完成即释放。设备侧保留投影权重、KV 和工作区。归一化 Hadamard、激活量化及矩阵乘法按误差阈值验收。

下表为提交 `9cb45549` 在同一台 EPYC 9334 / RADV NAVI31、8 个推理线程下的一次实测。每格先列 5 个输入 token 的首次 prefill 毫秒数，再列随后 3 次单 token decode 的平均吞吐。CPU 首次 prefill 含投影矩阵重排。Vulkan 的重排和上传发生在会话初始化期间，表中另列这段耗时。该主机同时运行其他推理任务，单次短提示的耗时会随设备负载变化。[全部逐步计时](../../tools/oracle/ternary_bonsai/performance.json)

| 权重 | Rust CPU，ms / tok/s | Rust 严格 Vulkan，ms / tok/s | Vulkan 初始化，s | 发布方 llama.cpp CPU，ms / tok/s |
|---|---:|---:|---:|---:|
| PTQ1_0 | 12,777 / 3.10 | 2,501 / 0.34 | 26.43 | 1,850 / 1.71 |
| PQ2_0 | 11,826 / 3.01 | 859 / 1.80 | 39.60 | 1,080 / 2.00 |
| Q2_0 | 11,241 / 2.76 | 3,781 / 0.38 | 31.61 | 1,469 / 1.95 |

Vulkan 每个文本位置实测 62 次提交。单行 decode 读取重排的投影权重，各层额外执行 Hadamard。发布方 CPU 直接读取紧凑三元块。各算子的耗时未单独采样。[对齐与回归记录](../../tools/oracle/ternary_bonsai/validation.md)
