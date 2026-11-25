# Aion Instruct 2.3B

Aion uses Microsoft’s tokenizer and chat template and has a text-only decoder. The native GGUF carries both files, the layer attention schedule, the RoPE parameters and both generation stop IDs. See the [converter and runnable examples](../../tools/oracle/aion/README.md).

The six packed GGUF formats keep weights compressed and use F32 activations. CPU workers unpack one weight row at a time and reuse it across the batch with the shared SIMD dot product. The Vulkan kernel reads packed weights directly and accumulates in F32.

Strict Vulkan runs the Q/K/V, output, FFN and vocabulary projections, plus causal grouped-query attention and KV cache updates. Each layer retains its F32 or packed F16 cache across chunks and decode steps. A sliding layer reserves at most the window plus one prefill chunk minus one token. Auto uses the shared projection size threshold and can select host attention at session creation. RMSNorm, RoPE, residuals, SwiGLU and sampling run on the host. A device loss after KV has been populated requires a new session and prompt replay.

The executor uploads weights on first use within each request. The host exchanges activations with the device at projection and attention boundaries. End-to-end timings include these transfers and synchronization. See the [same-host measurements](../../tools/oracle/aion/performance.json) and [oracle and endpoint results](../../tools/oracle/aion/validation.md).

The Q8_0 measurement on the Apple M3 Max used F32 KV, a 56-token prompt and three teacher-forced decode steps. Each backend ran separately.

| Backend | First prefill (s) | Mean decode (ms/token) |
| --- | ---: | ---: |
| Rust CPU, 8 threads | 1.011 | 104.7 |
| Rust strict Vulkan | 1.004 | 75.8 |
| Microsoft PyTorch, CPU, 8 threads | 0.271 | 112.8 |
