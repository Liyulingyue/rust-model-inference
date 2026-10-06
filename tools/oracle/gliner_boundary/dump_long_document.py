r"""Oracle for long-document chunking and chunk-result merging.

`gliner2/inference/chunking.py` is model-agnostic: it splits a document into
overlapping word windows, shifts each chunk's local span offsets back to document
offsets, merges the per-chunk predictions, and finally strips the span metadata
the caller did not ask for. Every function here is pure, so the fixture needs no
GGUF and the test always runs.

What this pins
--------------
*Chunking* (`split_text_into_chunks`) has three rules that a plausible port gets
wrong:

- Windows are over **words**, and `start_char` / `end_char` come from the word
  tokens, so a chunk's text is a *slice of the original* rather than a
  re-join. `iter_word_offsets` asks the splitter for `lower=False`, so a chunk
  keeps the document's casing — the lower-casing happens later, in the model.
- The step is `chunk_size - chunk_overlap`, and the loop **breaks when
  `end_word` reaches the end** rather than stepping again, so the final chunk is
  not duplicated by a trailing empty window.
- An empty or word-free document still yields **one** chunk spanning the whole
  text, not zero chunks — otherwise `len(chunks) != len(chunk_results)` fires in
  `merge_chunk_results` for every empty document.

*Merging* (`merge_chunk_results`) is where the behaviour is type-dependent, and
each branch is a separate code path:

- A classification dict (`{label, confidence}`) merges to the **max confidence**,
  not the most common.
- **Bare strings merge by majority vote**, ties broken by *earliest* chunk —
  `Counter` plus `-index`, so the first occurrence wins a tie. This is the branch
  a port most often gets wrong, because it is not "first non-empty" and not
  "highest score".
- Lists concatenate and then dedupe; **span** items go through the overlap
  resolver and are re-sorted by `(start, end, text)`, while non-span items are
  deduplicated on a **confidence-insensitive** canonical key and the survivor is
  the higher-confidence one. A port that dedupes on the whole item would keep two
  copies of a relation seen in overlapping chunks with slightly different scores.
- The overlap policy defaults to `disallow` here, but `_dedupe_items` passes
  `default="allow"` to the resolver — two different defaults in one function.

*Stripping* (`_strip_span_metadata`) runs **after** merging and re-derives each
span's `text` from the remapped document offsets, so a surface is re-sliced from
the original text rather than carried from the chunk. An enum/choice field
(`text`+`confidence`, no offsets) collapses to the bare string when confidence is
off — a shape that does not exist in the non-long path.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "target" / "gliner2-oracle"))

from gliner2.inference.chunking import (  # noqa: E402
    merge_chunk_results, split_text_into_chunks,
)

SHORT = "Marie Curie worked in Paris. She was born in Warsaw."
LONG = " ".join(f"word{i}" for i in range(50))

# (name, text, chunk_size, chunk_overlap)
CHUNK_CASES = [
    ("fits_in_one_chunk", SHORT, 384, 64),
    ("exact_fit", SHORT, len(SHORT.split()), 0),
    ("two_windows", LONG, 20, 5),
    ("no_overlap", LONG, 20, 0),
    ("overlap_is_step", LONG, 20, 5),
    ("empty_text", "", 10, 2),
    ("whitespace_only", "     ", 10, 2),
    ("single_word", "hello", 10, 2),
    ("chunk_size_one", LONG, 1, 0),
    ("overlap_zero_size_one", LONG, 1, 1 - 1),
    # A window that does not divide evenly still terminates: the last chunk runs
    # to the end and the loop breaks.
    ("ragged_tail", LONG, 20, 7),
    # CJK, where a "word" is a character: word windows are character windows.
    ("cjk", "中华人民共和国万岁", 4, 2),
    # The document's casing must survive: iter_word_offsets asks for lower=False.
    ("casing_preserved", "Marie Curie WAS here", 3, 1),
]

# (name, per-chunk results, scalar_labels, include_confidence, include_spans)
#
# The value sits **directly** under its key, because that is what lets
# `_merge_values` see its type and take the matching branch. Nesting a value
# inside a list (as the real `json_structures` output does) sends every case down
# the concatenate-and-dedupe path instead, which pins none of the interesting
# branches. The offsets below are deliberately arbitrary — the reference
# re-slices each span's `text` from the document, so the `text` we supply is
# discarded and the fixture records what it computed instead.
def span(text, start, end, confidence):
    return {"text": text, "start": start, "end": end, "confidence": confidence}


MERGE_CASES = [
    (
        "classification_takes_max_confidence",
        [
            {"mood": {"label": "happy", "confidence": 0.3}},
            {"mood": {"label": "sad", "confidence": 0.9}},
        ],
        set(), True, False,
    ),
    (
        "bare_strings_take_the_majority",
        [
            # sad is the earliest *and* the majority, so a port that just took
            # the first chunk would agree. happy is the majority but arrives
            # second, so only a real count picks it.
            {"mood": "sad"},
            {"mood": "happy"},
            {"mood": "happy"},
        ],
        set(), False, False,
    ),
    (
        "majority_wins_over_an_early_lead",
        [
            {"mood": "a"},
            {"mood": "a"},
            {"mood": "b"},
            {"mood": "b"},
            {"mood": "b"},
        ],
        set(), False, False,
    ),
    (
        "bare_string_tie_goes_to_the_earliest_chunk",
        [
            {"mood": "happy"},
            {"mood": "sad"},
        ],
        set(), False, False,
    ),
    (
        "lists_concatenate_then_dedupe_spans",
        [
            {"people": [span("Marie", 0, 5, 0.4), span("Paris", 28, 33, 0.8)]},
            {"people": [span("Marie", 0, 5, 0.7)]},
        ],
        set(), True, True,
    ),
    (
        "duplicate_non_span_items_collapse_to_the_higher_confidence",
        [
            {"works_at": [
                {"head": "Marie", "head_start": 0, "head_end": 5,
                 "tail": "Paris", "tail_start": 22, "tail_end": 27,
                 "score": 0.55},
            ]},
            {"works_at": [
                {"head": "Marie", "head_start": 0, "head_end": 5,
                 "tail": "Paris", "tail_start": 22, "tail_end": 27,
                 "score": 0.91},
            ]},
        ],
        set(), False, False,
    ),
    (
        "scalar_entity_label_collapses_to_one_value",
        [
            {"people": [span("Marie", 0, 5, 0.6), span("Marie Curie", 0, 11, 0.9)]},
        ],
        {"people"}, True, True,
    ),
    (
        "strips_spans_when_not_requested",
        [
            {"city": [span("ignored", 6, 11, 0.8)]},
        ],
        set(), False, False,
    ),
    (
        "keeps_only_confidence_when_spans_are_off",
        [
            {"city": [span("ignored", 6, 11, 0.8)]},
        ],
        set(), True, False,
    ),
    (
        "enum_choice_field_collapses_to_a_bare_string",
        [
            {"mood": {"text": "happy", "confidence": 0.7}},
        ],
        set(), False, False,
    ),
    (
        "empty_values_everywhere",
        [
            {"city": []},
            {"city": []},
        ],
        set(), False, False,
    ),
    (
        "a_key_absent_from_some_chunks_still_appears",
        [
            {"city": [span("ignored", 6, 11, 0.8)]},
            {"person": [span("ignored", 0, 5, 0.7)]},
        ],
        set(), False, False,
    ),
    (
        "nested_dicts_merge_per_key",
        [
            {"trip": {"mood": "happy", "grade": "a"}},
            {"trip": {"mood": "sad", "grade": "a"}},
        ],
        set(), False, False,
    ),
    (
        "null_and_empty_are_skipped",
        [
            {"city": None},
            {"city": [span("ignored", 6, 11, 0.8)]},
        ],
        set(), False, False,
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path,
        default=REPO_ROOT / "tests" / "fixtures" / "gliner2.5-base-v1"
        / "long-document-golden.json",
    )
    args = parser.parse_args()

    chunk_cases = []
    for name, text, size, overlap in CHUNK_CASES:
        chunks = split_text_into_chunks(text, size, overlap)
        chunk_cases.append({
            "name": name,
            "text": text,
            "chunk_size": size,
            "chunk_overlap": overlap,
            "chunks": [
                {
                    "text": chunk.text,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                    "start_word": chunk.start_word,
                    "end_word": chunk.end_word,
                }
                for chunk in chunks
            ],
        })

    # Merging needs a document and a chunk list to remap against, one chunk per
    # result. A small window over `SHORT` yields exactly three chunks, so a
    # three-result case lines up with `chunks` and the offset remap is a no-op —
    # which is the case where a span's `text` is re-derived from the original.
    document = SHORT
    window = split_text_into_chunks(document, 3, 1)
    if len(window) < 5:
        raise SystemExit(
            f"expected at least 5 chunks from a size-3/overlap-1 window, got {len(window)}"
        )
    merge_cases = []
    for name, results, scalar, include_confidence, include_spans in MERGE_CASES:
        if len(results) > len(window):
            raise SystemExit(f"{name}: needs {len(results)} chunks, only {len(window)}")
        merged = merge_chunk_results(
            document,
            window[: len(results)],
            results,
            include_confidence=include_confidence,
            include_spans=include_spans,
            scalar_entity_labels=scalar,
        )
        merge_cases.append({
            "name": name,
            "_document": document,
            "_chunk_size": 3,
            "_chunk_overlap": 1,
            "chunk_results": results,
            "scalar_entity_labels": sorted(scalar),
            "include_confidence": include_confidence,
            "include_spans": include_spans,
            "merged": merged,
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"chunking": chunk_cases, "merging": merge_cases},
        ensure_ascii=False,
    ) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
