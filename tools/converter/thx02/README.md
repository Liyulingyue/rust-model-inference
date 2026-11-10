# THX-02 converter

The converter handles ModelScope `HAL-X/THX-02` revision
`f17dca262dcd1d115e591424ce9cd012b51fc0db`. It verifies the full backbone and
readout SHA256, the configuration, tokenizer/codebook, every source tensor's
name/shape/type and payload boundaries before writing. It streams row chunks
and publishes a complete GGUF atomically without replacing an existing file.

Run from the repository root using the existing environment:

```sh
/Users/gouzi/Documents/git/rust-model-inference/.venv/bin/python \
  -m tools.converter.thx02.convert_thx02 \
  /Users/gouzi/Documents/git/rust-model-inference/models/THX-02 \
  /Users/gouzi/Documents/git/rust-model-inference/models/THX-02/THX-02-F32.gguf \
  --format f32
```

Supported exports are `f32`, `f16`, `bf16`, `q8_0`, `q4_0`, `q4_1`, `q4_k`,
`q5_k`, and `q6_k`, using the existing shared encoders. `q4_1` uses the shared
Q4_1 encoder, whose block layout follows the repository's runtime kernel
(`ops::kernel::q4_1`): an F16 scale, an F16 minimum and unsigned 0..15 nibbles
that the kernel reads back as `d * q + m`. The encoder is unit-tested against
an independent decoder; no THX-02 checkpoint has been converted with it, so it
carries no numerical evidence of its own.
`general.file_type` uses the standard unsigned GGUF FTYPE enum, independently
of individual tensor types; uniform K text exports use Q4_K_S/Q5_K_S/Q6_K.

Text embeddings, projections and readout follow `--format`. Norms, SSM vectors,
SSM convolution, learned vision positions and biases remain F32. Vision matrices
follow `f32`/`f16`/`bf16`/`q8_0`; vision patch kernels remain F32 with `q8_0`.
For Q4_0 and K-quants all vision weights remain F32 because the existing vision
loader does not support those block types. These files use mixed storage, as
recorded in `thx02.text_storage`, `thx02.vision_storage`, and
`thx02.float_storage`. A format's export check does not establish numerical or
device inference support.

The main GGUF uses `general.architecture=thx02`, `thx02.component=decision`,
`blk.*` text tensors, `cls.output.weight` with GGUF dimensions `[2560,255]`, and
bundled `clip.*` metadata with `v.*` / `mm.*` vision tensors. Source Conv3D patch
weights split into two temporal-frame kernels. The main output has 725 tensors
from 723 backbone tensors plus the readout and the extra patch frame.

`thx02.config`, `thx02.decision_config`, `thx02.joint_config` retain the complete
JSON configurations; `thx02.tokenizer_json` retains the complete tokenizer.
The discrete protocol is also exposed as `thx02.codes`, `thx02.code_token_ids`,
`thx02.protocol_version`, `thx02.attention_mode` and `thx02.temperature`.
`thx02.escape_tokens` derives from the decision encoder's official Transformers
special-token configuration, including `<|fim_suffix|>`; tokenizer added-token
`special` flags are not substituted for this set. Both files use these same
10 protocol-body escapes, excluding `<|im_start|>` and `<think>`, because the
official thinker payload also uses the decision encoder. The thinker tokenizer
and its own larger special-token set remain intact. Image preprocessing is pinned with
`clip.vision.image_min_pixels=65536` and
`clip.vision.image_max_pixels=1638400`.

The official recurrent implementation repeats each key head for two adjacent
value heads. The shared trunk selects key heads by value-head modulo. The
converter therefore permutes original value heads to
`[0,2,...,30,1,3,...,31]`, including the V segments of QKV/convolution, z,
alpha/beta, A-log/dt-bias and output input columns. Q/K stay unchanged. Metadata
pins `ssm.group_count=16`, `ssm.time_step_rank=32`, and
`thx02.recurrent_head_layout=qwen35-modulo-v1`. Norm offsets and `-exp(A_log)`
are applied in F32; the gated-delta norm is not offset.

Add `--thinker` and a separate output path to export a causal `qwen35` thinker.
This requires `thinker/adapter_model.safetensors`, `thinker/adapter_config.json`
and all `thinker/lm_config/{config,tokenizer,tokenizer_config}.json` files. The
rank-32/alpha-32 LoRA is merged in F32 and rounded to source BF16 before export,
matching official `thx02/build_thinker.py`. The same recurrent layout conversion
is then applied. The thinker has `thx02.component=thinker`, causal attention,
full `qwen35.tokenizer_json`, source configuration and adapter provenance. Pair
files by identical `thx02.source_revision` and `thx02.model_sha256`; the adapter
SHA256 is recorded separately. MTP weights are omitted because greedy causal
generation does not require that optional acceleration head.

Focused checks, with tiny payloads and source-header-only shape validation:

```sh
/Users/gouzi/Documents/git/rust-model-inference/.venv/bin/python \
  -m unittest tools.converter.thx02.test_convert_thx02 -q
```

Real checkpoint conversions and inference evidence are recorded in
`tools/oracle/thx02/validation.md`. Do not interpret the small exporter tests as
real-model numerical validation.
