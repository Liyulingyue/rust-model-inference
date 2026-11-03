# MiniCPM-V 4.7 parity oracle

Pinned oracle for `clip.projector_type = "minicpmv4_7"` (OpenBMB/MiniCPM-V-4.7-1B-gguf).

## Pin

| | |
|---|---|
| repo | `ggml-org/llama.cpp` |
| commit | `69f201a2051ea9b9e9b50c3cc56afd9e2ae64414` |
| provenance | merge commit of upstream PR [#29416](https://github.com/ggml-org/llama.cpp/pull/29416) *"[Model] Support MiniCPM-V 4.7"* (merged 2026-10-10) |

The repo's own `references/llama.cpp` **cannot** be used: its
`tools/mtmd/clip-impl.h` registers `minicpmv4_6` but not `minicpmv4_7`, so
`validate_mmproj_capabilities` rejects the mmproj before any compute happens.

Because a tarball checkout has no git metadata, the pin is verified by blob
hash of the three files that carry the contract:

| file | blob |
|---|---|
| `tools/mtmd/models/minicpmv.cpp` | `0e0fc0381148d1e62ce35fdc55107e89211730a8` |
| `tools/mtmd/mtmd.cpp` | `2c2b069361d5df81e35db59bfbe63d51aa011f38` |
| `tools/mtmd/clip.cpp` | `fccb00c3aa7002bbcdbc6e988bb9a747bc7de864` |

`build_minicpmv_oracle.sh` also asserts `clip-impl.h` registers `minicpmv4_7`,
which fails loudly on the wrong checkout instead of building something that
cannot load the model.

## Build

```sh
tools/oracle/minicpmv/build_minicpmv_oracle.sh          # -> /tmp/opencode/oracle
```

`GGML_NATIVE=OFF GGML_OPENMP=OFF` so the scalar path is what actually runs.

## Dumps

`minicpmv-parity-dump.patch` adds two env-gated dumps and touches no model
source:

* `RMI_MTMD_DUMP_DIR` — per-tile projector output (`tile_<n>.bin`) and the
  normalized pixels that feed the tower (`pixels_<n>.bin`).
* `RMI_MTMD_CB_DIR` — labelled graph checkpoints via the scheduler eval
  callback (currently not firing; see Known gaps).

## Compare

```sh
RMI_MTMD_DUMP_DIR=$DUMP llama-mtmd-cli \
  -m MiniCPM-V-4_7-1B-Q8_0.gguf --mmproj mmproj-model-f16.gguf \
  --image img.png -p x -n 1 --temp 0 --jinja

RMI_MINICPMV_MMPROJ=mmproj-model-f16.gguf RMI_MINICPMV_IMAGE=img.png \
RMI_MINICPMV_DUMP=$DUMP_RUST \
  cargo test --profile release-fast --test minicpmv_reference dump_tile -- --ignored --nocapture

tools/oracle/minicpmv/compare.py $DUMP $DUMP_RUST
```

**`--jinja` is mandatory on the oracle side.** `llama-mtmd-cli` defaults
`use_jinja = false`, and its fallback template emits a completely different
prompt (`user\n…\nassistant\n` with no `<|im_start|>` and no `<think>`). Comparing
against it produces a bogus "everything differs".

## Current result

```
pixels: 602112 values, 0 differ
projector: 65536 values compared, 65536 differ
PIXELS   OK
PROJECTOR DIFFER
```

* **Preprocessing is bit-exact.** The 448×448 overview resize (Triangle /
  bilinear) and the `(v/255 - mean)/std` normalisation agree on all 602112
  floats.
* **The vision tower diverges numerically** at `pos_embed`, by ~1e-6 relative
  (oracle `0.03810417652130127` vs Rust `0.03810374438762665`).

### Architectural agreement (what is actually pinned)

Bitwise equality is not the bar here; agreeing with the oracle about *what the
computation is* is. The checkpoint dump settles that:

| oracle checkpoint | floats | Rust |
|---|---|---|
| `layer_out-0` … `layer_out-6` | 1179648 (1024 tokens × 1152) | same |
| `vit_merger_attn_residual` | 1179648 | same |
| `vit_merger_ds_out` | 294912 (256 × 1152) | same |
| `layer_out-7` … `layer_out-26` | 294912 (256 × 1152) | same |
| `post_ln` | 294912 | same |
| `merger_out` | 65536 (64 × 1024) | same |

Seven layers before the merger and twenty after is the signature of
`insert_layer_id = 6`: `build_vit_layers(0, 7)`, merger, `build_vit_layers(7,
27)`. Wrong insertion point, or merging before the block instead of after it,
still yields finite embeddings, so
`tests/minicpmv_reference.rs::tower_shape_matches_the_oracle_checkpoint_dump`
asserts it rather than relying on a smoke run.
* Prompt assembly is verified structurally against the oracle's own debug log:
  both sides emit `user\n` + `<image_id>0</image_id>` + 65 image rows +
  `Describe.\nassistant\n<think>\n`, 81 tokens / rows total. `<image_id>N</image_id>`
  is injected by `mtmd_tokenizer` (`use_image_id`), **not** by the chat
  template, so a template-only implementation silently misses it.

## Fixed so far

`vit_merger` attends over **2×2** token windows (four tokens), not 4×4:

```cpp
window_idx[k++] = (2*wi    ) * pos_w + (2*wj    );
window_idx[k++] = (2*wi    ) * pos_w + (2*wj + 1);
window_idx[k++] = (2*wi + 1) * pos_w + (2*wj    );
window_idx[k++] = (2*wi + 1) * pos_w + (2*wj + 1);
```

Getting this wrong changes the output but is not sufficient to match.

## Stage-by-stage bisection

`GGML_SCHED_DEBUG` does not cover the clip graph, but the scheduler eval callback
does — once two things are right:

1. set it **unconditionally**, not inside an `else if (cb_eval == nullptr)`
   branch;
2. return `true` from the callback so the walk continues to the next node.

With those, `RMI_MTMD_CB_DIR` yields ~880 labelled checkpoints (`pos_embed`,
`attn_out-N`, `ffn_out-N`, `vit_merger_*`, `post_ln`, `merger_out`, ...).
`RMI_MINICPMV_STAGE_DIR` writes the matching `pos_embed.f32` on the Rust side.

Current bisection:

| stage | result |
|---|---|
| normalized pixels | 1179648 values, **0 differ** |
| `pos_embed` | **1178892 differ** — first divergence |

The `pos_embed` gap is small: oracle `0.03810417652130127` vs Rust
`0.03810374438762665`, i.e. ~1e-6 relative, so the structure is right and only
the accumulation differs.

### Hypotheses already ruled out

* **F32-product-then-widening vs widening-then-multiplying.** ggml's
  `ggml_vec_dot_f32` uses `(ggml_float)a[i] * (ggml_float)b[i]`, whereas the
  shared `crate::ops::dot_f32` computes the product in f32 and only then
  widens. Measured both: `0.03810374` (f32-first) vs `0.03810373` (widen-first)
  against the oracle's `0.03810418`. Neither matches, and they agree with each
  other, so this is **not** the cause.
* **im2col order in `ggml_conv_2d`.** Gathering the patch x-fastest (matching
  `ggml_conv_2d`'s kw-major im2col) instead of channel-major gave `0.00302`,
  two orders of magnitude worse — channel-major is correct here.

The next candidate is ggml's own summation shape: `ggml_vec_dot_f32` keeps
`GGML_F32_ARR` partial sums (4 by default) and adds them at the end, rather than
one sequential accumulator. That would need a local reimplementation inside
`encode_tile`, not a change to the shared `dot_f32`.

## Other known gaps

* Only `Q8_0` + `mmproj-model-f16` are covered.
* `MARKER_SLOT` (the KV slot the `<|image_pad|>` marker occupies, which sets the
  canvas origin `pos_0 - 1`) is still an assumption.
* Everything after `pos_embed` is untested: the split means the 27 ViT blocks,
  the `vit_merger` and the final projector have never been compared at all.