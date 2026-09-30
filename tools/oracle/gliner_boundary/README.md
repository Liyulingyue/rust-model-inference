# GLiNER2.5 BoundaryExtractor oracles

Golden fixtures for `src/models/gliner_boundary/`, checked against the
reference Python in `target/gliner2-oracle/gliner2/models/boundary/`.

The model weights are **not** committed. Each script reads
`models/gliner2.5-base-v1/model.safetensors` + `config.json` (see
`.agents/skills/model-download/SKILL.md` for fetching them) and writes one
JSON fixture under `tests/fixtures/gliner2.5-base-v1/`. The matching Rust
test is env-gated on `RMI_GLINER2_5_BASE_V1_GGUF`.

```sh
export PYTHONPATH=target/gliner2-oracle
export RMI_GLINER2_5_BASE_V1_GGUF="$PWD/models/gliner2.5-base-v1/gliner2.5-base-v1-f32.gguf"

models/.venv/bin/python3 tools/oracle/gliner_boundary/dump_boundary_encoder.py
models/.venv/bin/python3 tools/oracle/gliner_boundary/dump_boundary_query_head.py
models/.venv/bin/python3 tools/oracle/gliner_boundary/dump_score_explicit_pairs.py
models/.venv/bin/python3 tools/oracle/gliner_boundary/dump_score_explicit_spans_full.py
models/.venv/bin/python3 tools/oracle/gliner_boundary/dump_boundary_attention_window.py
models/.venv/bin/python3 tools/oracle/gliner_boundary/dump_document_candidate_pool.py
models/.venv/bin/python3 tools/oracle/gliner_boundary/dump_shared_pool_scorer.py

cargo test --profile release-fast --test gliner2_5_base_v1_boundary_encoder_parity
cargo test --profile release-fast --test gliner2_5_base_v1_boundary_query_head_parity
cargo test --profile release-fast --test gliner2_5_base_v1_score_explicit_pairs_parity
cargo test --profile release-fast --test gliner2_5_base_v1_score_explicit_spans_full_parity
cargo test --profile release-fast --test gliner2_5_base_v1_boundary_attention_window_parity
cargo test --profile release-fast --test gliner2_5_base_v1_document_candidate_pool_parity
cargo test --profile release-fast --test gliner2_5_base_v1_shared_pool_scorer_parity
```

| Script | Fixture | Rust test | Max delta |
|---|---|---|---|
| `dump_boundary_encoder.py` | `boundary-encoder-golden.json` | `gliner2_5_base_v1_boundary_encoder_parity` | 1.907e-6 |
| `dump_boundary_query_head.py` | `boundary-query-head-golden.json` | `gliner2_5_base_v1_boundary_query_head_parity` | 3.338e-6 |
| `dump_score_explicit_pairs.py` | `score-explicit-pairs-golden.json` | `gliner2_5_base_v1_score_explicit_pairs_parity` | 2.384e-7 |
| `dump_score_explicit_spans_full.py` | `score-explicit-spans-full-golden.json` | `gliner2_5_base_v1_score_explicit_spans_full_parity` | 5.722e-6 |
| `dump_boundary_attention_window.py` | `boundary-attention-window-golden.json` | `gliner2_5_base_v1_boundary_attention_window_parity` | exact mask |
| `dump_document_candidate_pool.py` | `document-candidate-pool-golden.json` | `gliner2_5_base_v1_document_candidate_pool_parity` | 2.622e-6 |
| `dump_shared_pool_scorer.py` | `shared-pool-scorer-golden.json` | `gliner2_5_base_v1_shared_pool_scorer_parity` | 1.526e-5 |
| `dump_extract_spans_end_to_end.py` | `extract-spans-e2e-golden.json` | `gliner2_5_base_v1_extract_spans_e2e_parity` | 2.813e-5, `input_ids` exact |

`dump_score_explicit_spans_full.py` builds the reference `BoundaryHead`
directly from the checkpoint (`BoundaryHeadSettings(**config["boundary_head"])`,
`load_state_dict(strict=True)`) and calls the real
`BoundaryHead.score_explicit_spans`, so every feature flag in the fixture
comes from the checkpoint rather than from the script. Do not hand-build a
slightly-different config: that is how a partial port gets certified against
a config no released checkpoint uses. The earlier
`dump_pair_scorer_limited.py` did exactly that and was removed.

`common.py` holds the shared pieces: the reference `BoundaryHead` loaded from
the checkpoint (`load_state_dict(strict=True)`), the synthetic documents, and
`CASES`. Every oracle reads its feature flags from `config.json` via
`BoundaryHeadSettings` rather than hardcoding them.

## Choosing cases

`boundary_attention_window` is 128 for base-v1, so the local attention band
`|i - j| <= 128` only excludes keys once a document passes
`2 * 128 + 1 = 257` boundary positions. Every end-to-end fixture here uses 24
tokens or fewer, which means a port that simply dropped the window would still
be byte-exact against all of them. `dump_boundary_attention_window.py` exists
solely to cover that: it pins the mask for `n = 8` (band inactive) and
`n = 273` (band active).

The pool and scorer fixtures use a 24-token case as well, which is long enough
for a 25 x 25 Cartesian pairing pass and a full 192-slot pool.

## The end-to-end oracle

`dump_extract_spans_end_to_end.py` is the exception: it is the only one that
does not start from synthetic `text_states`. It runs the real
`SchemaTransformer`, a `transformers` DeBERTa-v3-base whose weights are the
checkpoint's *fine-tuned* `encoder.*`, the reference `BoundaryHead`, and
`decode_candidates` — then compares against the Rust tokenizer, prompt builder,
encoder, gather, pool, scorer and decode in one shot.

It needs `models/deberta-v3-base` (a separate `microsoft/deberta-v3-base`
download from ModelScope), because the reference loads the stock encoder and
then overwrites it from the checkpoint. Using the stock encoder produces logits
around -15 and no extractions at all — silently, with no error.

## What is not covered yet

No fixture for the entity classification head (`classifier.0` + ReLU +
`classifier.3`, `[C]` markers), relations, records, count or abstention.
