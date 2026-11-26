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

`--compute cpu` 使用多线程 CPU，`auto` 自动选择设备。文本贪心生成可对 `*-mtp.gguf` 启用 `--mtp-draft 2`。图像输入走目标模型，CLI 图像命令省略草稿参数。HTTP 支持 `/v1/chat/completions`，图像使用 `image_url` 内容块，流式请求加 `"stream":true`。JEV 的文本和图像路由分别为 `/v1/jev/score`、`/v1/jev/image`。

```sh
curl -sS http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Name the capital of France."}],"temperature":0,"enable_thinking":false,"max_tokens":16}'
```

## 精度和性能

三种文件均直接加载。CPU 首次使用投影矩阵时，将三元码和原始 FP16 缩放无损重排为 Q4_0 并缓存，401 个 CPU 投影合计 13.41 GiB。词嵌入按 token 解包一行。Vulkan 逐矩阵重排并上传，最大的临时矩阵为 682.03 MiB 的输出头，上传完成即释放。设备侧保留投影权重、KV 和工作区。归一化 Hadamard、激活量化及矩阵乘法按误差阈值验收。

下表为同台 EPYC 9334 / RADV NAVI31 的实测。每格先列 5 个输入 token 的 prefill 毫秒数，再列随后 3 次单 token decode 的平均吞吐量。构建、重打包、上传和输出文件写入时间都在计时区间之外。单次短提示测量不代表长上下文吞吐。[逐步记录](../../tools/oracle/ternary_bonsai/performance.json)

| 权重 | Rust CPU，ms / tok/s | Rust Vulkan，ms / tok/s | 发布方 llama.cpp CPU，ms / tok/s |
|---|---:|---:|---:|
| PTQ1_0 | 1788 / 3.66 | 733 / 2.22 | 900 / 3.56 |
| PQ2_0 | 1723 / 3.07 | 789 / 2.19 | 454 / 4.22 |
| Q2_0 | 1812 / 2.85 | 718 / 2.19 | 701 / 4.18 |

Vulkan 每个位置实测 62 次提交。单行 decode 仍需读取重打包的全部投影权重，各层额外执行 Hadamard；发布方 CPU 直接读取紧凑的三元块。各部分耗时未单独采样。[数值、入口和回归范围](../../tools/oracle/ternary_bonsai/validation.md)
