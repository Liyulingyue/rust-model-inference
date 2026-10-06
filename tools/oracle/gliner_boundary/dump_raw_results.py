r"""Oracle for the raw results dict — the layer before `format_results`.

`format_results` (see `dump_format_results.py`) is the last step of an extract
call. This is the step before it: the **raw** dict the decode stages assemble,
where a span prediction is still a 4-element sequence in the span case and an
offset-carrying dict in the structures case, and where the per-group key shapes
are still whatever each decoder built.

This is the ground truth for the `Extraction` -> raw-results converter that
F-3 接线 needs. The port's `extract()` returns a typed `Extraction` carrying
**word** offsets, while the reference's raw dict carries **character** offsets
(`start_map[start], end_map[end - 1]`, `runtime.py:846`), so the converter has
to translate coordinates as well as reshape values. Driving the real
`batch_extract(format_results=False)` pins both at once, which a hand-written
pure oracle could not.

What this pins
--------------
**The key shape per group type is not uniform, and each is easy to get wrong.**

- `entities` is a list holding **one** map from entity label to its spans. Every
  declared label appears, including any that found nothing.
- A `json_structures` group is keyed by its **own name** at top level (here
  `paper`), not by `json_structures`, and its value is a list holding one field
  map. The field values are
  the field values are **dicts carrying `text`/`confidence`/`start`/`end`** — not
  4-tuples. So the same span serializes as a sequence under `entities` and as a
  dict under `json_structures`, and the two struct formatters treat those
  differently.
- A relation type is keyed at top level with a list of `{head, tail}` objects,
  where each side is itself an offset-carrying dict. There is no `score` key on
  the edge: the confidence lives on each side, and both sides report the *same*
  number because the edge is scored once.
- A classification is a bare `(label, score)` **tuple** at top level, not a
  mapping. This is what makes it sniff as a relation in `format_results` unless
  the schema listed it in `classification_tasks`, and it is why the converter
  cannot guess — the schema has to be passed alongside.

**Group order follows the schema, and groups the schema did not declare are
absent entirely** rather than present-and-empty. A schema declaring only
relations produces no `entities` key at all.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "target" / "gliner2-oracle"))

from gliner2 import AutoExtractor  # noqa: E402

# The published checkpoint layout the loader expects: the fine-tuned weights at
# the top level and the *base* encoder's config under `encoder_config/`. The
# released `models/` tree keeps those in two directories, so the loader is
# pointed at a view that joins them rather than a copy of either.
DEFAULT_VIEW = Path("/tmp/opencode/gliner2ckpt")


def _build_view(repo_root: Path) -> Path:
    """Join the checkpoint and the base encoder config into one directory."""
    view = DEFAULT_VIEW
    if view.exists():
        return view
    view.mkdir(parents=True, exist_ok=True)
    checkpoint = repo_root / "models" / "gliner2.5-base-v1"
    encoder = repo_root / "models" / "deberta-v3-base"
    for name in ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"):
        link = view / name
        if not link.exists():
            link.symlink_to(checkpoint / name)
    encoder_config = view / "encoder_config"
    encoder_config.mkdir(exist_ok=True)
    for source in encoder.glob("*.json"):
        link = encoder_config / source.name
        if not link.exists():
            link.symlink_to(source)
    return view


def structure_schema(fields, dtypes=None, description=None, name="paper"):
    """A `json_structures` group with no `record_metadata`."""
    schema = {"json_structures": [{name: {field: [] for field in fields}}]}
    if description:
        schema["json_descriptions"] = {
            name: {field: f"{description} for {field}" for field in fields}
        }
    if dtypes:
        schema["field_metadata"] = {
            f"{name}.{field}": {"dtype": dtype} for field, dtype in dtypes.items()
        }
    return schema


CASES = [
    (
        "entities",
        "Marie Curie worked in Paris.",
        {
            "entities": ["person", "org"],
            "entity_descriptions": {"person": "a person", "org": "an organisation"},
        },
        0.3,
    ),
    (
        # A single entity label, so the one-map-per-call shape is visible without
        # a second key to confuse it.
        "entities_single_label",
        "Marie Curie worked in Paris.",
        {"entities": ["person"], "entity_descriptions": {"person": "a person"}},
        0.3,
    ),
    (
        # A second label that does not match the text semantically still reports
        # a span — `company` claims "Paris". So a label is not guaranteed empty
        # just because nothing of that kind is in the text; only the threshold
        # empties one, which the next case pins. I had named this
        # "a_declared_label_may_find_nothing" expecting an empty list, and the
        # model said otherwise.
        "entities_a_mismatched_label_still_reports_a_span",
        "Marie Curie worked in Paris.",
        {
            "entities": ["person", "company"],
            "entity_descriptions": {"person": "a person", "company": "a company"},
        },
        0.3,
    ),
    (
        # A threshold nothing clears. The group is present and every label maps
        # to an **empty list** — present-and-empty, not absent. The empty value
        # is a list here because the label exists; a group with no labels at all
        # is the case that differs, and `format_results` turns an empty
        # `entities` list into `{}`.
        "entities_nothing_clears_the_threshold",
        "Marie Curie worked in Paris.",
        {"entities": ["person"], "entity_descriptions": {"person": "a person"}},
        0.999,
    ),
    (
        # Offsets are **character** offsets into the original text, not word
        # indices: "Marie Curie" is words 0..2 but characters 0..11. The port
        # holds word offsets and has to translate, and this is the case that
        # catches a converter that forgets to.
        "entities_offsets_are_characters_not_words",
        "A meeting in Berlin with Dr. Kwame Nkrumah and Ann.",
        {
            "entities": ["person", "location"],
            "entity_descriptions": {"person": "a person", "location": "a place"},
        },
        0.3,
    ),
    (
        # Repeated whitespace and punctuation, so the character offsets and the
        # span text can disagree with a naive word-times-average guess.
        "entities_offsets_survive_punctuation_and_spacing",
        "Ada Lovelace, 1815 -- London;  Ada Lovelace again, London.",
        {
            "entities": ["person", "location"],
            "entity_descriptions": {"person": "a person", "location": "a place"},
        },
        0.3,
    ),
    (
        "structures",
        "Deep Learning by Yoshua Bengio.",
        structure_schema(
            ["title", "authors"],
            {"title": "str", "authors": "list"},
            description="a paper",
        ),
        0.3,
    ),
    (
        # The mixed dtypes. `name` is a `str` field and the output shows **two**
        # spans under it, so the "a scalar field binds only its first span"
        # rule is not what shapes this payload — the field value is a list
        # regardless of dtype, and the dtype only decides how a caller reads it
        # back. Worth pinning because the opposite is the natural assumption.
        "structures_mixed_dtypes",
        "Marie Curie worked in Paris with Pierre Curie in London.",
        structure_schema(
            ["name", "city"],
            {"name": "str", "city": "list"},
            description="a person and where they were",
            name="person",
        ),
        0.3,
    ),
    (
        "relations",
        "Marie Curie worked in Paris.",
        {
            "entities": ["person", "org"],
            "entity_descriptions": {"person": "a person", "org": "an organisation"},
            "relations": [{"worked_in": {"head": "person", "tail": "org"}}],
        },
        0.3,
    ),
    (
        # Relations and entities in one schema. The port must emit both keys, in
        # the schema's group order, rather than letting one displace the other.
        "relations_beside_entities",
        "Marie Curie worked in Paris and later in London.",
        {
            "entities": ["person", "org"],
            "entity_descriptions": {"person": "a person", "org": "an organisation"},
            "relations": [{"worked_in": {"head": "person", "tail": "org"}}],
        },
        0.3,
    ),
    (
        # A classification is a bare `(label, score)` tuple, which is exactly the
        # shape `format_results` sniffs as a relation. Only the schema's
        # `classification_tasks` keeps it out of `relation_extraction`, so the
        # converter must be handed that list rather than inferring it.
        "classifications",
        "This is a great paper.",
        {"classifications": [{"task": "sentiment", "labels": ["positive", "negative"]}]},
        0.3,
    ),
    (
        "classifications_multi_label",
        "This paper is about safety and alignment.",
        {
            "classifications": [
                {"task": "topics", "labels": ["ml", "safety", "alignment"], "multi_label": True}
            ]
        },
        0.3,
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path,
        default=REPO_ROOT / "tests" / "fixtures" / "gliner2.5-base-v1"
        / "raw-results-golden.json",
    )
    parser.add_argument(
        "--view", type=Path, default=None,
        help="joined checkpoint view; built from models/ when omitted",
    )
    args = parser.parse_args()

    view = args.view or _build_view(REPO_ROOT)
    model = AutoExtractor.from_pretrained(str(view))

    cases = []
    for name, text, schema, threshold in CASES:
        # `format_results=False` stops before the payload shaping, so this is
        # the raw dict the decode stages assembled. `include_confidence` and
        # `include_spans` are both on, which is the combination the long-document
        # path forces (`chunking.py` calls batch_extract with both).
        (raw,) = model.batch_extract(
            [text],
            [schema],
            threshold=threshold,
            format_results=False,
            include_confidence=True,
            include_spans=True,
        )
        cases.append({
            "name": name,
            "text": text,
            "schema": schema,
            "threshold": threshold,
            "raw": raw,
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"cases": cases}, indent=1) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
