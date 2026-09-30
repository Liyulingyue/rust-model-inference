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

cargo test --profile release-fast --test gliner2_5_base_v1_boundary_encoder_parity
cargo test --profile release-fast --test gliner2_5_base_v1_boundary_query_head_parity
cargo test --profile release-fast --test gliner2_5_base_v1_score_explicit_pairs_parity
cargo test --profile release-fast --test gliner2_5_base_v1_score_explicit_spans_full_parity
```

| Script | Fixture | Rust test | Max delta |
|---|---|---|---|
| `dump_boundary_encoder.py` | `boundary-encoder-golden.json` | `gliner2_5_base_v1_boundary_encoder_parity` | 1.907e-6 |
| `dump_boundary_query_head.py` | `boundary-query-head-golden.json` | `gliner2_5_base_v1_boundary_query_head_parity` | 3.338e-6 |
| `dump_score_explicit_pairs.py` | `score-explicit-pairs-golden.json` | `gliner2_5_base_v1_score_explicit_pairs_parity` | 2.384e-7 |
| `dump_score_explicit_spans_full.py` | `score-explicit-spans-full-golden.json` | `gliner2_5_base_v1_score_explicit_spans_full_parity` | 5.722e-6 |

`dump_score_explicit_spans_full.py` builds the reference `BoundaryHead`
directly from the checkpoint (`BoundaryHeadSettings(**config["boundary_head"])`,
`load_state_dict(strict=True)`) and calls the real
`BoundaryHead.score_explicit_spans`, so every feature flag in the fixture
comes from the checkpoint rather than from the script. Do not hand-build a
slightly-different config: that is how a partial port gets certified against
a config no released checkpoint uses. The earlier
`dump_pair_scorer_limited.py` did exactly that and was removed.

## What is not covered yet

`gliner2.5-base-v1` sets `candidate_pool = "shared"`, so ordinary span
extraction runs `DocumentCandidatePool` + `SharedPoolScorer`
(`boundary/pool.py`), **not** `SparseBoundaryPairScorer`. The scripts above
only cover the `score_explicit_spans` path (entity classification, entity
attributes, joint-IE) plus the shared boundary encoder / query head it reuses.

Every fixture starts from synthetic `text_states` / `query_states`. There is
no tokenizer → encoder → decode end-to-end oracle yet.
