# Z-Image 对齐与运行

Oracle 是 [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)
@pinned `97d2990807fe6d558e395f8764198d7c7e7b411c`：脚本按这个 commit 拉一份干净的检出，
打上本目录的 trace 补丁（把中间量导出来），编出 `sd-cli`。
CPU 侧的逐位对照由 `tests/z_image_reference.rs` 跑（它读 Oracle 导出的中间量，
和本仓库的 Rust 路径逐位比）。

- 权重、命令与逐位复现：`docs/develop/SUPPORTED_MODELS.md` 的「Z-Image Turbo」行。
- 性能快照与加速路线：`docs/develop/ZIMAGE_STATUS.md`（2026-10-06 的快照）。
- 设备路径的问题与证据：`docs/develop/TODO.md` 的 TODO-076（F16 档 NaN 与 F32 档加载）
  和 TODO-077（设备路径不可复现）。

## HTTP 入口

服务端已接入 `POST /v1/images/generations`（与 P-JiT 共用路由）。模型按**权重签名**判定
（`general.architecture=pig`，或张量名 `x_embedder.weight` + `cap_embedder.0.weight` +
`final_layer.linear.weight`），所以启动服务不需要额外的开关。

```bash
target/release/rust-model-inference --serve --host 127.0.0.1 --port 8099 \
  --model models/z-image-gguf-mine/z-image-turbo-q8_0.gguf \
  --text-encoder models/z-image-gguf-mine/qwen3_4b_f32-q8_0.gguf \
  --vae models/z-image-gguf-mine/pig_flux_vae_fp32-f16.gguf

curl -s -X POST http://127.0.0.1:8099/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"a red panda on a rock","size":"128x128","steps":2,"seed":42}'
```

请求字段：`prompt`（必填）、`size` 或 `width`/`height`（方形）、`steps`、`seed`；
`guidance` 被拒 —— Z-Image 没有 CFG 分支，带了会静默改变语义，所以直接报 400。
`n` 只接受 1，`response_format` 只接受 `b64_json`。返回
`{"created":…,"data":[{"b64_json":…}]}`，`b64_json` 是 PNG 的 base64。

实测（NVIDIA GB10，release 构建，128×128 / 2 步 / seed 42）：
HTTP 返回的 PNG 与同一权重的 CLI 输出**逐位相同**（md5 `86caef6e632c995511bf2395fc1933ad`）；
debug 与 release 构建的结果也逐位相同。

CLI 与 HTTP 走的是同一份模型代码：`DiffusionPipeline::generate_rgb`
（`src/models/diffusion/z_image/`），HTTP 层只把请求映射成 `ZImageOptions`，不碰计算。
