# HTTP 图片输入（OpenAI / Responses / Anthropic 兼容）

`/v1/chat/completions`、`/v1/responses`、`/v1/messages` 都接受**带图**的对话，
形状与 OpenAI 官方 API 一致。图片在服务端解码成 PNG/JPEG 字节，交给模型的视觉
投影器（`--mmproj`）编码成 embedding，再与文本 prompt 拼成同一条序列。

## 请求格式

三种协议各用自己的 content block（同一个文件里可以混着文字与图片）：

| 协议 | 端点 | 图片 block |
|---|---|---|
| OpenAI Chat Completions | `/v1/chat/completions` | `{"type":"image_url","image_url":{"url":"<来源>"}}` |
| OpenAI Responses | `/v1/responses` | `{"type":"input_image","image_url":"<来源>"}` |
| Anthropic Messages | `/v1/messages` | `{"type":"image","source":{"type":"base64","media_type":"image/png","data":"<base64>"}}` |

图片只能出现在 **user 轮**；放到 system / assistant / tool 轮会返回 400。一个请求
里可以带多张图。

## 来源形式

`<来源>`（以及 Anthropic 的 `source.data`）接受四种形式：

| 形式 | 示例 | 说明 |
|---|---|---|
| data URI | `data:image/png;base64,iVBORw0…` | 推荐；OpenAI SDK 传本地文件时也是这个形状 |
| 裸 base64 | `iVBORw0…` | Anthropic Messages 的 `source.data` 就是这个形状 |
| 本地路径 | `file:///path/to/a.png` 或 `/path/to/a.png` | 同机便利 |
| 远程 URL | `http(s)://…` | **默认关闭**，需 `--allow-remote-images` |

远程 URL 的护栏与 llama.cpp server 对齐：10 MB 上限、10 s 超时、最多 3 次重定向，
并且拒绝回环 / 私网 / 链路本地 / 未指定地址。默认关闭是因为让一个推理服务去拉任意
URL，等于给它开了一张内网探测（SSRF）的面。

## 前置条件

* 服务必须带 `--mmproj <path>`。没带时带图请求返回 400：
  `image input requires a vision projector; start the server with --mmproj <path>`。
* 投影器要与文本模型配套（Qwen3.5 / Qwen3-VL / Qwen2-VL 各用自己的 mmproj）。
* 视觉 token 有预算上限，超了返回 400 并提示先缩小图片。

## 限制

| 项 | 值 |
|---|---|
| 请求体上限 | 32 MB |
| 远程图片 | 10 MB / 10 s / ≤3 次重定向（仅 `--allow-remote-images` 开启时） |
| 远程目标 | 拒绝回环、私网、链路本地、未指定地址 |
| 视觉 token | 超过预算 → 400 |
| 解码格式 | 只接受 PNG / JPEG（其它格式 → 400） |

## 示例

```bash
# data URI（Chat Completions）
curl http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "{
    \"model\": \"Qwen3.5-2B-Q8_0\",
    \"messages\": [{\"role\": \"user\", \"content\": [
      {\"type\": \"text\", \"text\": \"描述这张图\"},
      {\"type\": \"image_url\", \"image_url\": {\"url\": \"data:image/png;base64,$(base64 -w0 a.png)\"}}
    ]}],
    \"max_tokens\": 128
  }"
```

```bash
# Anthropic Messages 的 base64 source
curl http://localhost:8080/v1/messages \
  -H 'Content-Type: application/json' \
  -H 'anthropic-version: 2023-06-01' \
  -d "{
    \"model\": \"Qwen3.5-2B-Q8_0\",
    \"max_tokens\": 128,
    \"messages\": [{\"role\": \"user\", \"content\": [
      {\"type\": \"image\", \"source\": {\"type\": \"base64\", \"media_type\": \"image/png\", \"data\": \"$(base64 -w0 a.png)\"}},
      {\"type\": \"text\", \"text\": \"描述这张图\"}
    ]}]
  }"
```

## 验证状态

* **协议层**：三种 block × 多种来源都有单测
  （`src/app/server/api/protocol.rs`、`src/app/server/api/image_input.rs`）。
* **服务层**（Qwen3.5-2B，**故意不带** `--mmproj`）：三种协议都返回 400
  `image input requires a vision projector…`；非图片 payload 返回
  `unsupported image format … expected PNG or JPEG`；远程 URL 返回
  `remote image URLs are disabled; start the server with --allow-remote-images …`
  —— 说明「解析 → 解码 → runtime 校验」整条链路是通的。
* **端到端**（真模型 + 配套 mmproj 的带图对话）尚未在本机验证：本地只有
  MiniCPM-V 4.7 的 mmproj（文本塔缺失）与 Qwen3.5-2B 的文本塔（mmproj 缺失），
  没有可配对的组合。哨兵
  `tests/cli_http_agreement.rs::cli_and_http_agree_on_image_input` 就是这条验证，
  配好 `RMI_AGREEMENT_MODEL` 与 `RMI_AGREEMENT_MMPROJ` 后即可运行
  （它会把 CLI 的 `--mmproj --image` 与 HTTP 的带图请求逐 token 对齐）。
