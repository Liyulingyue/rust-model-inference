"""Shared pieces for the GLiNER2.5 BoundaryExtractor oracles.

Every stage oracle needs the same three things: the real reference
``BoundaryHead`` built from the checkpoint, a deterministic synthetic
document, and the intermediate tensors that the stage under test consumes.
Keeping them here means the per-stage fixtures cannot drift apart in what
they feed the model.

The synthetic document is ``(arange(L * hidden) * 0.011).sin() - 1.0`` and
the queries are its first ``q_count`` rows, so a reader can regenerate the
inputs by hand; each fixture still stores them explicitly so a stage can be
regenerated and diffed on its own.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "target" / "gliner2-oracle") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "target" / "gliner2-oracle"))

from gliner2.configuration import BoundaryHeadSettings  # noqa: E402
from gliner2.models.boundary.model import BoundaryHead  # noqa: E402

HIDDEN_SIZE = 768

# ``c_count`` only applies to the explicit-span path, which takes
# caller-supplied candidates. The shared pool picks its own width
# (``pool_size``), so it ignores that key.
CASES = [
    # Two queries x 6 candidates over a short document, including an invalid
    # start==end, a reversed pair, and an out-of-range end.
    {"seq_len": 8, "valid_tokens": 6, "q_count": 2, "c_count": 6,
     "candidates": [[0, 2], [1, 4], [3, 6], [5, 5], [4, 2], [7, 8]]},
    # Three queries x 4 candidates, short document.
    {"seq_len": 5, "valid_tokens": 4, "q_count": 3, "c_count": 4,
     "candidates": [[0, 1], [1, 4], [2, 3], [4, 9]]},
    # Long enough that n_boundaries (25) stays under pool_boundary_top_k (32)
    # but the Cartesian pairing pass is 25 x 25 = 625 pairs, so the per-query
    # quota reservation and the deduplicating top-k actually do work.
    {"seq_len": 24, "valid_tokens": 20, "q_count": 2, "c_count": 4,
     "candidates": [[0, 3], [2, 7], [5, 19], [20, 24]]},
]


def model_dir() -> Path:
    return REPO_ROOT / "models" / "gliner2.5-base-v1"


def fixture_dir() -> Path:
    return REPO_ROOT / "tests" / "fixtures" / "gliner2.5-base-v1"


def build_head(dir_: Path | None = None) -> BoundaryHead:
    """The reference head, loaded from the checkpoint's own config + weights.

    ``load_state_dict`` is strict on purpose: a missing or extra key means this
    oracle would be scoring something other than the checkpoint. Feature flags
    come from ``config.json`` via ``BoundaryHeadSettings`` — never hand-written
    here. That distinction matters: the removed ``dump_pair_scorer_limited.py``
    built a scorer config no released checkpoint uses, and certified a partial
    port against it.
    """
    dir_ = dir_ or model_dir()
    config = json.loads((dir_ / "config.json").read_text())
    if config.get("architecture") != "boundary":
        raise ValueError(
            f"expected architecture='boundary', got {config.get('architecture')!r}"
        )
    settings = BoundaryHeadSettings(**config["boundary_head"])
    head = BoundaryHead(
        HIDDEN_SIZE, settings, query_dim=HIDDEN_SIZE,
        build_candidate_states=settings.enable_records,
    )
    state_dict = load_file(str(dir_ / "model.safetensors"))
    prefix = "boundary_head."
    sub = {
        key[len(prefix):]: value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }
    head.load_state_dict(sub, strict=True)
    head.eval()
    return head


def synthetic_states(seq_len: int) -> torch.Tensor:
    flat = (torch.arange(seq_len * HIDDEN_SIZE, dtype=torch.float32) * 0.011).sin() - 1.0
    return flat.view(1, seq_len, HIDDEN_SIZE)


def build_inputs(case: dict) -> dict:
    """Tokenize-free synthetic inputs plus the shared stage-1/2 outputs.

    Returns token states, masks, and the ``BoundaryEncoder`` /
    ``BoundaryQueryHead`` results that every later stage consumes. Queries
    reuse the document's first rows, matching the earlier stage oracles.
    """
    seq_len = case["seq_len"]
    q_count = case["q_count"]
    text_states = synthetic_states(seq_len)
    text_mask = torch.zeros(1, seq_len, dtype=torch.bool)
    text_mask[0, : case["valid_tokens"]] = True
    query_states = text_states[:, :q_count].contiguous()
    query_mask = torch.ones(1, q_count, dtype=torch.bool)
    return {
        "seq_len": seq_len,
        "q_count": q_count,
        "text_states": text_states,
        "text_mask": text_mask,
        "query_states": query_states,
        "query_mask": query_mask,
        "text_lengths": text_mask.sum(dim=1).long(),
    }


def run_head_stages(head: BoundaryHead, inputs: dict) -> dict:
    """``boundary_encoder`` + ``boundary_query_head`` over ``inputs``."""
    with torch.inference_mode():
        encoding = head.boundary_encoder(inputs["text_states"], inputs["text_mask"])
        marginals = head.boundary_query_head(
            encoding.states, encoding.mask,
            inputs["text_states"], inputs["text_mask"],
            inputs["query_states"], inputs["query_mask"],
        )
    return {"encoding": encoding, "marginals": marginals}


def explicit_indices(case: dict) -> torch.Tensor:
    """``[B, Q, C, 2]`` candidates, the same list for every query."""
    c_count = case["c_count"]
    q_count = case["q_count"]
    pairs = torch.tensor([case["candidates"]], dtype=torch.long)
    return (
        pairs.view(1, 1, c_count, 2)
        .expand(1, q_count, c_count, 2)
        .contiguous()
    )


def explicit_legal_mask(indices: torch.Tensor, inputs: dict) -> torch.Tensor:
    """``score_explicit_spans``'s legality rule, evaluated here so the fixture
    can record the exact mask both sides use."""
    text_lengths = inputs["text_lengths"]
    starts, ends = indices[..., 0], indices[..., 1]
    return (
        (starts >= 0)
        & (ends > starts)
        & (ends <= text_lengths.view(1, 1, 1))
        & inputs["query_mask"].unsqueeze(-1)
    )


def dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n")
    print(f"{path}")
