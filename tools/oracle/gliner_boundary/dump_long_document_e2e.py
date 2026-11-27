r"""Oracle for `batch_extract_long` — the long-document driver end to end.

`extract_long` and its nine siblings (`extract_entities_long`,
`classify_text_long`, `extract_json_long`, `extract_relations_long`, and the
`batch_` forms) are all thin wrappers over `batch_extract_long`, so this pins the
one thing they share rather than repeating the same case ten times.

The pipeline it pins
--------------------
1. `split_text_into_chunks` over the document, producing overlapping **word**
   windows. `chunk.text` is a slice of the original, so a chunk's local offsets
   are document offsets minus the chunk's `start_char`.
2. Every chunk goes through the ordinary `batch_extract` with
   `format_results=True, include_confidence=True, include_spans=True` — **both
   flags forced on regardless of what the caller asked for**.
3. `merge_chunk_results` merges a document's chunks, shifting offsets back and
   re-deriving surfaces from the document, then `strip_span_metadata` applies the
   caller's own flags.

Step 2 is the one that is easy to get wrong. `merge_chunk_results` walks
*formatted* JSON, so the per-chunk result must be shaped by `format_results`
before it is handed over, and with both flags on or the merge has no offsets to
shift. The caller's flags are applied once, at the end, by
`strip_span_metadata` — so a caller who asked for neither still gets the offsets
shifted correctly and only loses them at the very end.

What this pins
--------------
- **Offsets are document-relative, not chunk-relative.** A span found only in the
  second window still reports its position in the original text.
- **The caller's flags apply at the end.** `include_confidence=False` still
  produces correctly-positioned spans; the merge is unaffected.
- **A `choices` value survives the merge** with no offsets, because it never had
  any — `strip_span_metadata` has nothing to strip, and a driver that invented
  offsets for it would report a location in the document that the reference never
  claimed.
- **Scalar-dtype entities collapse to their best span and `null` when absent**,
  which is what `_scalar_entity_labels` tells the merge. Passing an empty label
  set makes the merge treat a scalar entity as a list, and the result differs in
  *type* (`[...]` versus an object or `null`), not merely in content.
- **An empty document still produces one chunk**, so the merge has something to
  return rather than failing on an empty slice list.

Cases deliberately vary only one of chunk_size/chunk_overlap/flags at a time, so a
failing case points at the knob rather than at the pipeline.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "target" / "gliner2-oracle"))

from gliner2 import AutoExtractor  # noqa: E402
from gliner2.inference.schema import Schema  # noqa: E402

from dump_raw_results import _build_view  # noqa: E402


def choices_schema():
    """A structure group with one `choices` field, built via the builder.

    `StructureBuilder.parent` is the *group name*, so chaining `.parent` yields a
    string and `batch_extract` then fails with `'str' object has no attribute
    'get'`. Holding the `Schema` and letting `build()` finish the builder is the
    shape that works.
    """
    schema = Schema()
    schema.structure("paper")
    schema._active_builder.field(
        "topic", dtype="str", choices=["physics", "chemistry"],
        description="the paper's topic",
    )
    return schema


ENTITY_SCHEMA = {
    "entities": ["person", "location"],
    "entity_descriptions": {"person": "a person", "location": "a place"},
}

# Long enough to need several windows at the sizes below, with a person and a
# place in the first window and another pair only in a later one, so a merge that
# forgets to shift offsets or drops a later chunk is visible.
LONG_TEXT = (
    "Marie Curie worked in Paris for many years. "
    "She shared a laboratory with Pierre Curie, her husband. "
    "Later the family moved to London, and she continued her research there. "
    "In 1903 she was awarded the Nobel Prize in Physics. "
    "Her notebooks survive in the archives at the University of Paris."
)

CASES = [
    # Defaults from the reference signature: chunk_size 384, chunk_overlap 64.
    ("default_windows", LONG_TEXT, ENTITY_SCHEMA, 0.3, 384, 64, True, True, "list"),
    # The caller's flags are applied at the end, so both-off must still shift
    # offsets correctly during the merge and only drop them afterwards.
    ("no_flags", LONG_TEXT, ENTITY_SCHEMA, 0.3, 384, 64, False, False, "list"),
    ("spans_only", LONG_TEXT, ENTITY_SCHEMA, 0.3, 384, 64, False, True, "list"),
    ("confidence_only", LONG_TEXT, ENTITY_SCHEMA, 0.3, 384, 64, True, False, "list"),
    # Small windows with no overlap: every window is disjoint, so a span in a
    # later window has a large offset shift.
    ("disjoint_windows", LONG_TEXT, ENTITY_SCHEMA, 0.3, 16, 0, True, True, "list"),
    # A one-word window with a heavy overlap: many chunks, each reporting the
    # same entity, which is what the dedup in the merge is for.
    ("heavy_overlap", LONG_TEXT, ENTITY_SCHEMA, 0.3, 8, 4, True, True, "list"),
    # Fewer words than one window: a single chunk, so the merge is a no-op and
    # the offsets are already document-relative.
    ("single_chunk", "Marie Curie worked in Paris.", ENTITY_SCHEMA, 0.3, 384, 64, True, True, "list"),
    # An empty document still yields one chunk, so this must not error.
    ("empty_document", "", ENTITY_SCHEMA, 0.3, 384, 64, True, True, "list"),
    # A scalar-dtype entity: the merge is told the label is scalar, which changes
    # the *type* it reports rather than only its content.
    (
        "scalar_entity_labels",
        LONG_TEXT,
        Schema().entities({"person": "a person", "location": "a place"}, dtype="str"),
        0.3,
        24,
        8,
        True,
        True,
        "list",
    ),
    # A `choices` field, on text where **every** window contains the winning
    # literal.
    #
    # The first version reused LONG_TEXT, whose opening 32-word window contains
    # neither "physics" nor "chemistry" — and there the two choices score within
    # the port's known f32 reduction-width drift of each other, so the argmax
    # flips and the *merged* payload differs. That is a real limitation (the same
    # drift is why the single-pass `choices` e2e runs at a 1e-1 tolerance) but it
    # is not what this case is for: this case exists to show that a choice
    # survives the merge without gaining a document location, and pinning that
    # needs a text where the choice is not a coin flip.
    (
        "choices_beside_spans",
        (
            "This paper studies physics in some detail. "
            "The physics of the reaction is described at length. "
            "Later sections revisit the physics measurement again. "
            "A closing note on physics closes the paper."
        ),
        choices_schema(),
        0.3,
        32,
        8,
        True,
        True,
        "list",
    ),
    # Relations, whose edges are `{head, tail}` objects whose offsets have to be
    # shifted by each side independently.
    (
        "relations",
        LONG_TEXT,
        {
            "entities": ["person", "location"],
            "entity_descriptions": {"person": "a person", "location": "a place"},
            "relations": [{"was_in": {"head": "person", "tail": "location"}}],
        },
        0.3,
        32,
        8,
        True,
        True,
        "list",
    ),
    # A classification rides along in the per-chunk formatted payload and comes
    # back per document, not merged across windows.
    (
        "classifications",
        "This paper is clearly about physics. "
        "It studies radioactivity in some detail. "
        "The experiments described here were repeated many times.",
        Schema().classification("topic", ["physics", "chemistry", "biology"]),
        0.3,
        16,
        4,
        True,
        True,
        "list",
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path,
        default=REPO_ROOT / "tests" / "fixtures" / "gliner2.5-base-v1"
        / "long-document-e2e-golden.json",
    )
    parser.add_argument("--view", type=Path, default=None)
    args = parser.parse_args()

    view = args.view or _build_view(REPO_ROOT)
    model = AutoExtractor.from_pretrained(str(view))

    cases = []
    for (
        name, text, schema, threshold, chunk_size, chunk_overlap,
        include_confidence, include_spans, _,
    ) in CASES:
        (merged,) = model.batch_extract_long(
            [text],
            [schema],
            threshold=threshold,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            include_confidence=include_confidence,
            include_spans=include_spans,
        )
        cases.append({
            "name": name,
            "text": text,
            "threshold": threshold,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "include_confidence": include_confidence,
            "include_spans": include_spans,
            "merged": merged,
        })
        print(f"  {name}: {json.dumps(merged)[:110]}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"cases": cases}, indent=1) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()