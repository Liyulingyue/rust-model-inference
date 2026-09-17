# DSpark speculative decoding

DSpark is available for CPU text generation with these verified pairs:

- Qwen3-4B + `deepseek-ai/dspark_qwen3_4b_block7`
- LFM2.5-1.2B-Instruct + its matching LiquidAI DSpark sidecar

```bash
cargo run --release --bin rust-model-inference -- \
  --model /path/to/target.gguf \
  --draft-model /path/to/dspark.gguf \
  --prompt "Hello" \
  --max-tokens 32 \
  --temp 0 \
  --spec-draft-n-max 7 \
  --spec-draft-conf-min 0
```

The target model remains authoritative: rejected draft tokens are replaced by
the target result. The initial implementation supports greedy decoding only,
uses CPU text inference, and rejects multimodal, embedding, GPU, interactive,
and non-zero-temperature modes. The sidecar metadata and target dimensions
must match; unsupported pairs fail during loading instead of falling back to a
similar architecture.

The verified artifacts are:

| Pair | Target SHA256 | Sidecar SHA256 |
|---|---|---|
| Qwen3-4B Q4_K_M | `7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5` | `f81a1877d6db00d1f8476d365d4c252e94ab27fa57bb481dd9bf0079fd276c97` |
| LFM2.5-1.2B-Instruct Q4_K_M | `b1b3de114215d9507409a662a501a631095a479a419584e8a2ded6304b19b4f5` | `5cf9bb2947638dd74a47b486b817f407831c0da420aeebb6973fb66c25af51e4` |

Both pairs were checked against llama.cpp commit
`84075273c82f7681d43436b692073cbd4ab15fe9` with one CPU thread, F32 KV,
greedy decoding, and a seven-token draft block. The draft hidden state, logits,
Markov bias/logits, confidence, generated token IDs, and acceptance blocks were
compared bit for bit. The real-artifact regression tests are in
`tests/dspark_reference.rs`.
