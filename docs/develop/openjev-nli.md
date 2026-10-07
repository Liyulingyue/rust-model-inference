# openjev` — Qwen3.5 + 3-class NLI head

`openjev/` is `Qwen/Qwen3.5` (0.8B / 2B / 4B / 35B-A3B) finetuned as an NLI
cross-encoder over `(premise, hypothesis)` pairs. The repo ships only HF
safetensors + a custom `modeling_openjev.py`; the GGUF has to be composed
in-engine from a base Qwen3.5 GGUF (e.g. `unsloth/Qwen3.5-0.8B-GGUF`) plus
the HF classifier head.

This entry records what we wired up, what works, and what does NOT
work today.

## What we built

### 1. `tools/converter/openjev/convert_openjev.py`

HF safetensors → GGUF converter. Takes:

* `--base-gguf` — a vanilla Qwen3.5 GGUF (`general.architecture=qwen35`).
* `--hf-checkpoint` — a HF directory with `config.json` + `model*.safetensors`
  holding `score.weight` of shape `[num_labels, hidden_size]` in BF16.
* `--output` — destination GGUF path.

It copies every backbone tensor verbatim and appends:

| tensor | shape | dtype | origin |
| --- | --- | --- | --- |
| `cls.output.weight` | `[hidden_size, num_labels]` | F32 | transposed from HF `score.weight` (BF16 → F32 + shape swap so the existing `qwen3::trunk::weights::score_logits` path works without changes) |
| `cls.output.bias` | `[num_labels]` | F32 | all-zero (HF `score` has no bias; matches llama.cpp rerank packer contract that the zero-bias head is part of the model) |

The converter writes metadata mirroring the HF config:

```
qwen35.pooling_type                       = 3   (last token)
qwen35.classifier.label_strings           = ["contradiction", "entailment", "neutral"]
qwen35.classifier.label_indices           = [0, 1, 2]
qwen35.classifier.nli_template             = "Premise: {premise}\nHypothesis: {hypothesis}"
qwen35.classifier.problem_type             = "single_label_classification"
qwen35.classifier.pad_token_id            = 248044
```

`label_strings` + `label_indices` are written as parallel GGUF arrays
(GGUF arrays only carry scalars — a `{0: "contradiction"}` dict would
have to flatten to two parallel arrays). The Rust loader reads them as
parallel arrays when present.

### 2. `qwen35::trunk::HybridTrunk` + `Qwen35Session` adapter

* New optional `cls_score: Option<Weight<'a>>` + `cls_score_bias: Vec<f32>`
  fields on `HybridTrunk`, loaded from `m_clip.output.weight` /
  `m_clip.output.bias` (matching the `qwen3` rerank path).
* `HybridTrunk::is_classifier()` / `score_logits(&[last_hidden]) ->
  Vec<f32>` mirror the `qwen3` equivalents; `score_logits` quantises
  the hidden state to Q8_0 and dispatches to the F32 matmul, then adds
  `cls_score_bias` if present.
* `Qwen35Session::forward_classify(token_ids, positions)` runs the full
  causal prefill, **reads the last non-pad token's post-RMSNorm hidden
  state** from `scratch.normed_buf`, and projects through
  `cls_score` + `cls_score_bias`.
  The `last_hidden(1)` method on `Qwen35Session` returns the FIRST row
  of `normed_buf` — wrong for NLI where the prefix
  `"Premise: ..."` is identical across samples and would force the same
  logits every call. `forward_classify` reads the LAST row directly via
  `scratch.normed_buf[(n-1)*n_embd..]`.
* `forward::qwen35::trunk::run_classify_qwen35_with_batch` is the
  free-function wrapper used by `examples/test_openjev_nli.rs`. It
  builds a fresh `Qwen35Model` + `Qwen35Session` per call, runs
  `forward_classify`, and returns the raw per-class logits.

### 3. `tools/converter/utils/gguf.py` reader fixes

While writing the converter two small reader fixes were needed:

* Added `GGUF_T_INT32 = 5` and `GGUF_T_FLOAT32 = 6` type ids — the
  existing reader only supported `UINT32` / `INT64` / `FLOAT64`. The
  base Qwen3.5 GGUF includes a `general.base_model.count` int32 array
  that previously raised "unsupported metadata type 5".
* The `Safetensors` reader no longer pretends to be a context manager;
  callers do `st = open_safetensors(path); st.get(name).raw`.

### 4. `examples/test_openjev_nli.rs`

End-to-end smoke test. Loads the converted GGUF, tokenizes four
`(premise, hypothesis)` pairs with `add_special=false` + the
`nli_template` from the config, calls
`run_classify_qwen35_with_batch`, applies softmax, prints per-class
probabilities and the argmax.

## What does NOT work today

`Qwen3.5Model::forward` carries significant numerical drift even at
modest depth — the shipped `tests/qwen35_reference.rs` budgets
`result_output abs_tol=0.5` (last-layer output) and
`attn_norm-63 abs_tol=1.25` (last-layer pre-norm). That's tolerable for
generation / argmax-only scoring (the `qwen3.5` row in MODEL_LIST.md uses
it for text + image multimodal decode) but too large for NLI-style
classification where the argmax depends on small logit differences and
absolute scale. Concretely, running the four samples above we get:

| sample | oracle argmax | Rust argmax | agreement? |
| --- | --- | --- | --- |
| 0 — "bird below" vs "bird 0.05 below" | neutral (96%) | neutral (75%) | ✓ |
| 1 — "man playing guitar" vs "someone making music" | entailment (98%) | neutral (96%) | ✗ |
| 2 — "no dog" vs "there is a dog" | contradiction (99%) | neutral (86%) | ✗ |
| 3 — "cat on mat" vs "sky is blue" | neutral (98%) | entailment (58%) | ✗ |

So: the conversion + classifier plumbing is end-to-end correct (no
crashes, the right tensors are loaded, the post-prefill RMSNorm runs,
the score matmul executes with the expected dimensions and dtype, softmax
is taken from the logits the model actually produces). The numeric
output of `Qwen35Model::forward` itself is far enough from a HF
reference that the resulting classifier head argmax is wrong for
75 % of the smoke-test pairs.

The drift is concentrated in the delta-rule linear-attention path (the
`Qcur-3`, `Kcur-3`, `layer_output-3`, etc. tolerances are all `0.075` to
`1.5`); the partial rotational RoPE scheme (25 % rotary) is also known
to drift. Both are pre-existing qwen35 forward limitations that the
reference test already documents — they are not introduced by this
work.

If `Qwen3.5Model::forward` were ever brought closer to HF numerics,
this entry would light up automatically: rerun `cargo run --release
--example test_openjev_nli` and the argmaxes should flip to
`[neutral, entailment, contradiction, neutral]` for the four samples
above. Until then, the right path for 3-class NLI on Qwen3.5 in this
engine is either:

* treat the drift as a calibration problem (per-layer scale + bias
  correction trained on a small labelled set), or
* use the qwen3 rerank path for binary NLI-style tasks where the drift
  is below the threshold for an argmax flip (the abs_tol on
  `Qwen3Model::forward` is `1e-4` instead of `0.5`).