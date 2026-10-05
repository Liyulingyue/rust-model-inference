r"""Oracle for `_decode_choice_field` — the literal-enum decode, end to end.

`dump_choice_fields.py` pinned the *prompt* side: how the prefix renders and
where it lands. This one pins what the decode does with it, which is the half
the port still owes.

No released checkpoint ships a schema that uses `choices`, so there is no
pre-existing ground truth. But the reference will happily decode any schema that
declares one, so the fixture is produced by declaring `choices` and letting the
reference score it — the numbers come from the real encoder, the real head, and
the reference's own `_decode_choice_field`.

What this pins
--------------
`_decode_choice_field` (`engine.py:656-759`) is not a threshold-and-sort over
proposals. It scores each declared choice as an **explicit one-token span**
`(index, index + 1)` through `score_explicit_spans`, bypassing the candidate
pool entirely. Four consequences a port has to match:

1. **The span is the choice token, not the word in the document.** `index` comes
   from `_find_choice_idx`, which searches `text_tokens[:len_prefix]` only. The
   pair therefore covers the *prefix* row, and the reported value is the literal
   rather than a document surface. A port that scored the document occurrence
   would return a confidence for the wrong span while looking structurally
   correct.
2. **It is a sigmoid over a divided logit**, `sigmoid(logit / pair_temperature)`
   (`engine.py:703`) — the same temperature the span path uses, so
   `pair_temperature != 1.0` is observable.
3. **`dtype` picks the shape, and the shapes differ in more than arity.** `list`
   returns every choice at or above the field threshold, in *declaration* order
   (the `present` list, not score order). Scalar returns the `argmax` — or `None`
   when even the best is below the threshold. A scalar field never falls back to
   the first choice: it reports nothing.
4. **A repeated choice is skipped** (`engine.py:675-681`), and so is a choice
   absent from the prefix, so a schema listing a value the renderer dropped
   yields a shorter `present` list than it declared.

The prefix elements are matched **verbatim**, not re-split: `_find_choice_idx`
compares against `text_tokens[:len_prefix]`, and those entries are exactly what
`_build_classification_prefix` emitted. So a multi-word literal like
`"very happy"` is a single entry, matches as a single entry, and the scored span
`(index, index + 1)` covers the whole literal. It is *not* split into two tokens
and therefore is not lost — an earlier reading of this code assumed otherwise,
and `multi_token_choice_is_not_found` is the case that corrects it.
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "target" / "gliner2-oracle"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from transformers import AutoConfig, AutoModel, AutoTokenizer  # noqa: E402

from common import (  # noqa: E402
    DEFAULT_MODEL, build_head, dump_json, encoder_dir, fixture_dir,
)
from dump_extract_spans_end_to_end import (  # noqa: E402
    load_checkpoint_encoder,
)
from gliner2.inference.engine import BoundaryExtractor  # noqa: E402
from gliner2.processor import SchemaTransformer  # noqa: E402

TEXT = "Marie Curie was born in Warsaw and worked in Paris."


def schema_for(fields, name="trip", descriptions=None):
    """A structure group whose named fields are literal enums."""
    out = {
        "json_structures": [
            {name: {field: {"value": "", "choices": list(values)}
                    for field, values in fields.items()}}
        ],
    }
    if descriptions:
        out["json_descriptions"] = {name: dict(descriptions)}
    return out


def two_field_schema():
    return {
        "json_structures": [
            {
                "trip": {
                    "mood": {"value": "", "choices": ["happy", "sad"]},
                    "grade": {"value": "", "choices": ["a", "b", "c"]},
                }
            }
        ],
        "json_descriptions": {"trip": {"mood": "her mood", "grade": "her grade"}},
    }


THREE = ["happy", "sad", "neutral"]
DESCRIBED = {"mood": "her mood"}

# (name, schema, dtypes, threshold, note)
CASES = [
    (
        "list_dtype_three_choices",
        schema_for({"mood": THREE}, descriptions=DESCRIBED),
        {"mood": "list"},
        0.5,
        "The default dtype is `list`, so every choice at or above the threshold comes back, in declaration order rather than score order.",
    ),
    (
        "scalar_dtype_takes_argmax",
        schema_for({"mood": THREE}, descriptions=DESCRIBED),
        {"mood": "str"},
        0.5,
        "A scalar field returns the argmax, or nothing when the best choice is below the threshold.",
    ),
    (
        "zero_threshold_scalar_reports_something",
        schema_for({"mood": THREE}, descriptions=DESCRIBED),
        {"mood": "str"},
        0.0,
        "Threshold 0.0 clears every probability, so the scalar path returns its argmax rather than None, which separates the two branches without needing a different model.",
    ),
    (
        "unreachable_threshold_scalar_reports_nothing",
        schema_for({"mood": THREE}, descriptions=DESCRIBED),
        {"mood": "str"},
        1.0,
        "Threshold 1.0: sigmoid never reaches 1, so the scalar field reports nothing. A port that fell back to the first choice would pass every other case here.",
    ),
    (
        "unreachable_threshold_list_is_empty",
        schema_for({"mood": THREE}, descriptions=DESCRIBED),
        {"mood": "list"},
        1.0,
        "The same gate on the list branch yields an empty list, not None.",
    ),
    (
        "two_choice_fields_independent_dtypes",
        two_field_schema(),
        {"mood": "list", "grade": "str"},
        0.5,
        "Two choice fields with different dtypes in one group, each resolved against its own dtype.",
    ),
    (
        "repeated_choices_are_deduplicated",
        schema_for({"mood": ["happy", "happy", "sad"]}, descriptions=DESCRIBED),
        {"mood": "list"},
        0.5,
        "engine.py:675-681 skips a repeated literal, so `present` is shorter than the declared list and the duplicate is scored once.",
    ),
    (
        "uppercase_choices_match_case_insensitively",
        schema_for({"mood": ["Happy", "SAD"]}, descriptions=DESCRIBED),
        {"mood": "list"},
        0.5,
        "`_find_choice_idx` lower-cases both sides, and the reported value keeps the declared casing.",
    ),
    (
        "multi_token_choice_is_not_found",
        schema_for({"mood": ["very happy"]}, descriptions=DESCRIBED),
        {"mood": "list"},
        0.5,
        "A multi-word literal is ONE prefix entry, not two, so it matches verbatim and the (index, index+1) span covers the whole literal.",
    ),
    (
        "choice_also_appears_in_the_document",
        schema_for({"place": ["paris"]}, descriptions={"place": "where"}),
        {"place": "list"},
        0.5,
        "`paris` is both a declared choice and a document word. The lookup is prefix-only, so the scored span is the prefix row.",
    ),
    (
        "single_choice_list",
        schema_for({"mood": ["happy"]}, descriptions=DESCRIBED),
        {"mood": "list"},
        0.0,
        "One choice with a gate everything clears: the list branch has exactly one element, so the arity difference from the scalar branch is visible.",
    ),
    (
        "single_choice_scalar",
        schema_for({"mood": ["happy"]}, descriptions=DESCRIBED),
        {"mood": "str"},
        0.0,
        "The same schema as the previous case under `str`, which pins the shape difference at equal scores.",
    ),
]


def run_case(processor, encoder, head, schema, dtypes, threshold):
    batch = processor._collate_batch(
        [(TEXT, copy.deepcopy(schema))],
        max_len=None,
        error_policy="raise",
        build_targets=False,
    )
    with torch.inference_mode():
        hidden = encoder(
            input_ids=batch.input_ids, attention_mask=batch.attention_mask
        ).last_hidden_state
    width = hidden.shape[-1]

    def gather_routed(indices, mask):
        safe = indices.clamp(0, hidden.shape[1] - 1)
        states = hidden.gather(1, safe.unsqueeze(-1).expand(-1, -1, width))
        return states * mask.unsqueeze(-1).to(states.dtype)

    text_states = gather_routed(batch.text_word_indices, batch.text_word_mask)
    query_states = gather_routed(batch.query_marker_indices, batch.query_marker_mask)

    # `_decode_choice_field` is an instance method, but the only state it reads is
    # `self.boundary_head` and `self.boundary_settings`, so the unbound function is
    # called against a stand-in rather than loading a whole `BoundaryExtractor`
    # through `from_pretrained`.
    # `_decode_choice_field` reaches for the head, the settings, and
    # `_find_choice_idx` on `self`. The first two are data; the third is a plain
    # method that is a pure function of (choice, tokens), so it is bound from the
    # real class rather than re-implemented — the lookup under test has to be the
    # reference's.
    stand_in = SimpleNamespace(boundary_head=head, boundary_settings=head.settings)
    stand_in._find_choice_idx = BoundaryExtractor._find_choice_idx.__get__(stand_in)
    decode = BoundaryExtractor._decode_choice_field.__get__(stand_in)

    core = {
        "text_states": text_states,
        "text_mask": batch.text_word_mask,
        "query_states": query_states,
        "query_mask": batch.query_marker_mask,
    }

    from gliner2.models.boundary.model import _extractive_field_names

    query_names = []
    for tokens in batch.schema_tokens_list[0]:
        query_names.extend(_extractive_field_names(tokens))

    prefix = list(processor._build_classification_prefix(schema))
    prefix_length = len(prefix)
    # `_find_choice_idx` searches the prefix region of the *combined* stream.
    combined_prefix = prefix

    results = []
    for group in schema["json_structures"]:
        for parent, fields in group.items():
            for field, value in fields.items():
                if not (isinstance(value, dict) and "choices" in value):
                    continue
                query_id = query_names.index(field)

                found = []
                for choice in value["choices"]:
                    index = -1
                    for position, token in enumerate(combined_prefix):
                        if token.lower() == choice.lower():
                            index = position
                            break
                    found.append([choice, index])

                present = []
                seen = set()
                for choice, index in found:
                    if choice in seen or index < 0:
                        continue
                    seen.add(choice)
                    present.append((choice, index))

                if present:
                    pairs = torch.tensor(
                        [[index, index + 1] for _, index in present],
                        dtype=torch.long,
                    ).view(1, 1, len(present), 2)
                    with torch.inference_mode():
                        logits = head.score_explicit_spans(
                            text_states,
                            batch.text_word_mask,
                            query_states[:, query_id:query_id + 1],
                            batch.query_marker_mask[:, query_id:query_id + 1],
                            pairs,
                        )[0, 0]
                    probabilities = torch.sigmoid(
                        logits / head.settings.pair_temperature
                    )
                    raw_logits = [float(v) for v in logits]
                    probs = [float(v) for v in probabilities]
                else:
                    raw_logits = []
                    probs = []

                field_metadata = (schema.get("field_metadata") or {}).get(
                    f"{parent}.{field}", {}
                )
                configured = field_metadata.get("threshold")
                decoded = decode(
                    batch=batch,
                    sample_index=0,
                    core=core,
                    query_id=query_id,
                    choices=list(value["choices"]),
                    dtype=dtypes.get(field, "list"),
                    configured_threshold=configured,
                    default_threshold=threshold,
                    prefix_length=prefix_length,
                    include_confidence=True,
                )
                results.append({
                    "field": f"{parent}.{field}",
                    "query_id": query_id,
                    "dtype": dtypes.get(field, "list"),
                    "choice_lookup": found,
                    "present": [[c, i] for c, i in present],
                    "raw_logits": raw_logits,
                    "probabilities": probs,
                    "decoded": decoded,
                })
    return {
        "text": TEXT,
        "threshold": threshold,
        "schema": schema,
        "prefix_tokens": prefix,
        "prefix_length": prefix_length,
        "query_names": query_names,
        "dtypes": dtypes,
        "fields": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--out", type=Path,
        default=fixture_dir() / "choice-decode-golden.json",
    )
    args = parser.parse_args()

    base = encoder_dir(args.model)
    processor = SchemaTransformer(str(base), token_pooling="first", word_splitter=None)
    tokenizer = AutoTokenizer.from_pretrained(str(base))
    encoder = AutoModel.from_config(AutoConfig.from_pretrained(str(base)))
    load_checkpoint_encoder(encoder, args.model)
    encoder.eval()
    added = tokenizer.add_special_tokens(
        {"additional_special_tokens": SchemaTransformer.SPECIAL_TOKENS}
    )
    if added != len(SchemaTransformer.SPECIAL_TOKENS):
        raise ValueError(f"added {added} special tokens")
    head = build_head(model=args.model)

    payload = {
        "model": args.model,
        "note": (
            "No released checkpoint declares `choices`, so these are produced by "
            "declaring one and letting the reference decode it. `probabilities` come "
            "from `score_explicit_spans` over (index, index+1) in the prefix region "
            "- the candidate pool is bypassed entirely."
        ),
        "cases": [
            dict(name=name, note=note,
                 **run_case(processor, encoder, head, schema, dtypes, threshold))
            for name, schema, dtypes, threshold, note in CASES
        ],
    }
    dump_json(args.out, payload)


if __name__ == "__main__":
    main()
