# Aion Instruct 2.3B

Aion uses Microsoft’s tokenizer and chat template and has a text-only decoder. The native GGUF carries both files, the layer attention schedule, the RoPE parameters and both generation stop IDs. See the [converter and runnable examples](../../tools/oracle/aion/README.md).

The GPU path offloads Q/K/V, output projection, FFN and vocabulary projections through the shared Vulkan executor. RMSNorm, RoPE, attention, KV and sampling run on the CPU. Auto may use the CPU for small projections according to the shared size threshold. Strict Vulkan requires the requested projections to run on the device.

The executor uploads weights on its first use within a request and synchronizes at projection boundaries. End-to-end throughput depends on batch size, weight format and host-to-device traffic. See the [measured performance](../../tools/oracle/aion/performance.json) and [verification matrix](../../tools/oracle/aion/validation.md).
