# Sophon thinker and vision converter

Converts the thinker in ModelScope `AdventistsAI/Sophon-Realtime-32B-Tag`, revision
`990d6a74bbf15c1765a6f7736fa839f423fd5b59`. This is a converter contract, not
an assertion that every runtime capability or precision has passed inference.
The CLI validates the pinned configuration and complete indexed tensor inventory;
it does not authenticate the download. Verify the source revision and file hashes
before conversion.

Run from the repository root with its Python environment (NumPy is the only
production dependency):

```sh
./.venv/bin/python -m tools.converter.sophon_realtime.convert \
  models/Sophon-Realtime-32B-Tag \
  models/Sophon-Realtime-32B-Tag-GGUF/thinker-F32.gguf \
  --quant f32 \
  --mmproj models/Sophon-Realtime-32B-Tag-GGUF/mmproj-F32.gguf
```

The input can be either the release root or its `thinker/` directory. It requires
`config.json`, `model.safetensors.index.json` and all indexed shards,
`tokenizer.json`, and `chat_template.jinja`. The optional vision output also
requires `preprocessor_config.json`. The release's indexed shards already include
the complete vision tower; the duplicate standalone `visual.safetensors` is not
used. Audio and its projector are validated and partitioned, not exported as
language decoder tensors. `auxiliary-inventory.json` records their exact pinned
names, BF16 dtypes and shapes, together with the vision inventory.

`--quant` accepts `f32`, `f16`, `bf16`, `q8_0`, `q4_0`, `q4_1`, `q4_k`, `q5_k`,
and `q6_k`. All norms remain F32. Matrices use the existing shared GGUF quantizers;
a matrix whose input width is not block aligned remains F32. Conversion streams
bounded row groups instead of retaining the whole decoded 32B model. An F32
thinker alone needs about 130 GB; ensure sufficient destination space. K-quant
conversion uses eight bounded worker processes (at most sixteen 1 MiB row chunks
in flight), preserving deterministic tensor/row order. Other formats stay serial.
The existing Python reference encoders can still be slow at this scale.

The source reference is E4M3FN widened to F32 and multiplied by the per-channel
scale widened to F32. The source also dynamically quantizes input activations to
FP8. These GGUF outputs provide **weight-only** quantization and do not reproduce
that activation quantization. Source configuration and this semantic distinction
are embedded in every output. The thinker architecture is `sophon_realtime`,
with the decoder fields under `qwen3vl.*`; component and revision metadata allow
the runtime to check a thinker/vision pair.

Vision is F32 with `general.architecture=clip` and
`clip.projector_type=qwen3vl_merger`. It preserves the original 27-layer tower,
DeepStack layers `[8,16,24]`, pre-merge norm, source image budgets, and separate
matrices for the two temporal slices of the patch convolution.

Outputs are staged, directory-readback checked, and published with a no-overwrite
hard link. Failures clean up staged files. If creating a thinker/vision pair fails,
neither output remains. Publication across two files is not a transactional
filesystem operation: another process must wait for the command to finish before
opening a new pair. Existing files or dangling output symlinks are rejected.

Synthetic checks (PyTorch, safetensors and the installed `gguf` package are
independent test/oracle dependencies; conversion only needs NumPy):

```sh
./.venv/bin/python -m unittest tools.converter.sophon_realtime.test_convert
```

Checks cover all 256 E4M3FN codes (the two NaNs must fail), multiple channel scales,
all nine format dispatches, F32 scaled payloads, BF16 widening, exact tokenizer
added tokens/NFC/template, missing or malformed scales/tensors/components,
nonfinite values, configuration rejection, truncated shards/index mismatches,
vision temporal splitting/merger norm/DeepStack/pixel budgets, existing outputs,
and failure cleanup.

Audio encoder and Ultravox projector export:

```sh
./.venv/bin/python -m tools.converter.sophon_realtime.audio \
  models/Sophon-Realtime-32B-Tag \
  models/Sophon-Realtime-32B-Tag-GGUF/audio-F32.gguf --quant f32
```

All nine formats are supported. Norms, biases and convolution kernels remain F32;
K formats also retain F32 for matrices whose input width is not 256-aligned
(the audio hidden width is 896). The output records the requested format and
actual per-tensor GGML types. Original `audio_tower.*` and
`multi_modal_projector.*` names are preserved, together with the exact source
audio config and preprocessing JSON.

The audio source is **`thinker/audio/model.safetensors`**. The official pinned
`DuplexJevVisionForConditionalGeneration.load_weights` explicitly loads this
file after the indexed main checkpoint, overwriting the audio encoder/projector.
The main checkpoint's projector has different values and must not be substituted.
The standalone vision file, in contrast, has payloads identical to the indexed
vision tensors.

Independent sampled export checks:

```sh
./.venv/bin/python -m tools.converter.sophon_realtime.verify \
  models/Sophon-Realtime-32B-Tag \
  models/Sophon-Realtime-32B-Tag-GGUF/thinker-Q8_0.gguf --quant q8_0
```

The verifier handles thinker and audio components. It checks all vectors and
first/middle/last rows of every other tensor against independently loaded source
weights, uses the independent `gguf` payload decoders, and records degradation,
predeclared relative-RMSE budgets and SHA256. F32 must match exactly; F16/BF16
must additionally match independent PyTorch casts exactly. Preserved F32 tensors
must match the source exactly for every requested format, including norms and
K-unaligned audio matrices. The declared relative-RMSE budget gates both each
sampled tensor and the global result; a low-energy tensor cannot hide in the
aggregate. A tensor-budget failure names the tensor and its maximum absolute
error. These checks measure conversion/export quality, not inference or task
accuracy.

Native CosyVoice3 talker export (PyTorch is required for this `.pt` component):

```sh
./.venv/bin/python -m tools.converter.sophon_realtime.talker \
  models/Sophon-Realtime-32B-Tag \
  models/Sophon-Realtime-32B-Tag-GGUF/talker-F32.gguf --quant f32
./.venv/bin/python -m unittest tools.converter.sophon_realtime.test_talker
```

The input can be the release root or `talker/`. The converter authenticates the
pinned `llm.pt`, `cosyvoice3.yaml`, and `CosyVoice-BlankEN/{config.json,vocab.json,
merges.txt,tokenizer_config.json}` against their SHA256 hashes. It opens the
checkpoint using `torch.load(weights_only=True, mmap=True, map_location="cpu")`
and streams bounded row groups through the existing quantizers. All nine formats
are available; vectors, non-matrices and block-unaligned matrices remain F32.
Only a scalar subset of the YAML is read; its constructors are never executed.

The source has 293 F32 tensors. It is a Qwen2 trunk with 24 layers, hidden size
896, 14 query / 2 KV heads, head size 64, FFN size 4864, RoPE base 1e6, epsilon
1e-6, Q/K/V biases and no Q/K norms. The component architecture is
`sophon_realtime_talker`, with `qwen2.*` decoder metadata and explicit speech
configuration under `sophon_realtime.talker.*` and `talker_config_json`.

| Source | Export |
| --- | --- |
| `llm.model.model.embed_tokens.weight` | `token_embd.weight` (151936 × 896) |
| `llm.model.model.norm.weight` | `output_norm.weight` |
| `llm.model.model.layers.*` | Standard `blk.*` names, including Q/K/V biases |
| `llm_decoder.weight` | `output.weight` (6761 × 896 speech head) |
| `speech_embedding.weight` | `speech_embd.weight` (6761 × 896) |
| `llm.model.lm_head.weight` | Validated but omitted: the official speech forward bypasses this text head |

The 292 exported payloads preserve the native speech path. The output-head row
count is independent of the text tokenizer size, so a runtime must explicitly
compose the text embedding, decoder trunk and speech head/embedding. The omission
and actual per-tensor GGML types are recorded in metadata.

The original vocab JSON, tokenizer config JSON, merges text and YAML are retained
verbatim in `sophon_realtime.talker_*` metadata. The effective text tokenizer
includes the pinned CosyVoice3 wrapper's ordered special tokens: 151924 assigned
entries within 151936 text embedding rows. It uses Qwen2 preprocessing without
NFC normalization. Its text EOS/padding ID is 151643, following the wrapper's
override of the original tokenizer config. Speech controls are separate IDs:
SOS=6561, EOS=6562, task=6563, fill=6564; speech vocab size is 6761. Literal tags
such as `[breath]` and `<|endofprompt|>` must be parsed as special text tokens.
The retained chat template does not replace the official talker's direct prompt
and bistream scheduling. Full effective entry/ID/type agreement was checked
against the pinned official wrapper; runtime tokenization and speech inference
remain separate validation work.

Flow and HiFT native components use the same nine requested formats:

```sh
./.venv/bin/python -m tools.converter.sophon_realtime.code2wav \
  models/Sophon-Realtime-32B-Tag \
  models/Sophon-Realtime-32B-Tag-GGUF/flow-F32.gguf --component flow --quant f32
./.venv/bin/python -m tools.converter.sophon_realtime.code2wav \
  models/Sophon-Realtime-32B-Tag \
  models/Sophon-Realtime-32B-Tag-GGUF/hift-F32.gguf --component hift --quant f32
./.venv/bin/python -m unittest tools.converter.sophon_realtime.test_code2wav
```

The converter authenticates each `.pt` and `cosyvoice3.yaml` against the pinned
SHA256 values and checks the complete F32 name/shape/type contract. Flow exports
all 330 original names. HiFT materializes its 82 weight-norm `(g,v)` pairs with
`torch._weight_norm(v,g,dim=0)` in F32, replacing each pair by logical `.weight`
and exporting 246 tensors. Original g/v are absent; metadata records the mapping
and precision. Conv1d tensors retain PyTorch `[out,in,kernel]` order. In
particular, HiFT's upsampling layers are nearest-neighbor interpolation followed
by causal Conv1d, rather than transposed convolutions.

Convolutions, vectors, unaligned matrices and matrices below 65536 elements stay
F32. Flow's eligible 2-D matrices use the requested shared quantizer; HiFT's
payloads are consequently all F32 for all nine requested formats. Metadata
records requested versus actual types. Architectures/components are
`sophon_realtime_flow`/`flow` and `sophon_realtime_hift`/`hift`; normalized settings
are in `sophon_realtime.flow_config_json` or `sophon_realtime.hift_config_json`,
with the original YAML retained separately. The CLI offers no pin bypass.

Flow preserves its RoPE inverse-frequency buffer. The pinned x-transformers
reference rotates adjacent pairs; the tag launcher selects four Euler steps and
CFG=0, distinct from the YAML's original CFG=0.7. HiFT F32 materialization matches
the pinned vLLM-Omni F32 component contract. The older vendored generator casts
its F0 predictor to F64 before normalization, which has different rounding;
source-to-export effects must be measured separately from runtime error against
the same GGUF. Generated noise/phase state, inference schedules, waveform quality
and end-to-end runtime parity are outside conversion verification.
