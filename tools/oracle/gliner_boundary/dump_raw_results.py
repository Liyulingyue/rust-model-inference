r"""Oracle for the raw results dict — the layer before `format_results`.

`format_results` (see `dump_format_results.py`) is the last step of an extract
call. This is the step before it: the **raw** dict the decode stages assemble,
where predictions are offset-carrying dicts rather than the shaped strings and
label lists a caller sees, and where the per-group key shapes are still whatever
each decoder built.

This is the ground truth for the `Extraction` -> raw-results converter that
F-3 接线 needs. The port's `extract()` returns a typed `Extraction` carrying
**word** offsets, while the reference's raw dict carries **character** offsets
(`start_map[start], end_map[end - 1]`), so the converter has to translate
coordinates as well as reshape values. Driving the real
`batch_extract(format_results=False)` pins both at once, which a hand-written
pure oracle could not.

What this pins
--------------
**The key shape per group type is not uniform, and each is easy to get wrong.**

- `entities` is a list holding **one** map from entity label to its spans. Every
  declared label appears, including any that found nothing — and a label declared
  `dtype: "str"` is a **dict or `null`, not a list**: it collapses to its single
  best span, and to `null` when nothing cleared the threshold.
- A `json_structures` group is keyed by its **own name** at top level (here
  `paper`), not by `json_structures`, and its value is a list holding one field
  map. A span field's value is a list of offset-carrying dicts; a `choices` field
  has **no `start`/`end` at all**, because a choice has no document location.
- A relation type is keyed at top level with a list of `{head, tail}` objects.
  There is no `score` key on the edge: the confidence lives on each side, and
  **both sides report the same number** because the edge is scored once. Each side
  is built as `{text, start, end}` with `confidence` *added afterwards*, so its
  key order differs from a span's even though the content does not.
- A classification is a bare `(label, score)` **tuple** at top level, not a
  mapping. This is what makes it sniff as a relation in `format_results` unless
  the schema listed it in `classification_tasks`, and it is why the converter
  cannot guess — the schema has to be passed alongside.

**Group order follows the schema, and groups the schema did not declare are
absent entirely** rather than present-and-empty. A schema declaring only
relations produces no `entities` key at all.

**A plain-dict schema silently discards every piece of metadata.**
`_build_schema_dicts_and_metadata` (`runtime.py:345`) only reads
`entity_metadata`, `field_metadata`, `relation_metadata`, `field_orders` and
`entity_order` off a `Schema` **builder**; handed a dict it writes empty tables
and proceeds. So writing `entity_metadata: {person: {dtype: "str"}}` into a dict
schema neither raises nor applies — the label stays a list and the case passes
for the wrong reason. Three cases here were wrong in exactly that way before the
builder was used, which is why they are worth recording rather than deleting:
the failure is silent and the fixture looks fine.

`choices` has the same trap with a different cause: they live *inside* a field's
value (what `Schema.field` writes), not in a `field_metadata` side table. A
dict-schema `field_metadata` is dropped before the decoder sees it, so the field
decodes as an ordinary list field and scores **document spans** instead — a
result that looks plausible and is not the feature under test.

Cases needing metadata therefore go through the builder, and the fixture records
`built_with_schema_builder` so a reader knows which those are.
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


def choice_schema():
    """A `json_structures` group whose two fields are literal enums.

    `StructureBuilder.parent` is the *group name*, not the `Schema`, and
    `__getattr__` forwards unknown attributes to the parent schema after
    finishing the builder — so the builder has to be finished by a later call
    (`build()` does it) rather than by reaching for a parent. Holding the
    `Schema` in a variable and letting `build()` finish the builder is the shape
    that works, and it is what the case below uses.
    """
    schema = Schema()
    schema.structure("paper")
    schema._active_builder.field(
        "topic", dtype="str", choices=["ml", "safety", "vision"],
        description="the paper's topic",
    )
    schema._active_builder.field(
        "mood", dtype="list", choices=["positive", "negative"],
        description="the paper's mood",
    )
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
        # A `choices` field, built with the `Schema` builder because a plain
        # dict cannot express it. This took two attempts to get right, and the
        # first attempt is the useful part: `choices` written into a
        # `field_metadata` side table is **silently ignored**, and the field
        # decodes as an ordinary list field scoring document spans instead. So a
        # choice field is only a choice field when the choices sit *inside* the
        # field value, which is what the builder writes.
        #
        # The scores are the reference's choice scores, and a choice has no
        # document location — the shape below is what tells the converter that
        # `start`/`end` do not exist for this variant, and guessing otherwise
        # would invent offsets.
        "structures_choices_scalar_and_list",
        "The paper is clearly about safety and less about alignment.",
        choice_schema(),
        0.3,
    ),
    (
        # A scalar-dtype entity collapses to a single value instead of a list.
        # Also needs the builder: `entity_metadata` in a plain dict is dropped by
        # `_build_schema_dicts_and_metadata`, so the dtype silently stays `list`
        # and the case passes for the wrong reason.
        "entities_scalar_dtype_collapses_to_one_value",
        "Marie Curie worked in Paris.",
        Schema().entities({"person": "a person"}, dtype="str"),
        0.3,
    ),
    (
        # The same scalar dtype with a threshold nothing clears, so the collapse
        # lands on `None` — a different JSON type again, which
        # `format_entity_dict` turns into `null` downstream. Without this,
        # "collapses to one value" cannot tell "absent" from "present but null".
        "entities_scalar_dtype_with_nothing_to_report",
        "Marie Curie worked in Paris.",
        Schema().entities({"person": "a person"}, dtype="str"),
        0.999,
    ),
    (
        # The same schema with `include_confidence` and `include_spans` off. The
        # flag pair is a *fixture* dimension here rather than a case dimension:
        # `batch_extract` defaults them to False, so the shapes below are what the
        # single-document path produces unless the caller opts in — and the
        # long-document path, which is what this converter exists for, always
        # passes both. A converter that only handles both-on would look correct on
        # every other fixture and wrong on all of these.
        #
        # `_format_spans` has a fourth branch that returns the surface on its own,
        # so a span is a bare **string** here rather than a one-key object; a
        # scalar-dtype entity does the same; and a relation edge becomes a bare
        # `(head, tail)` pair.
        "entities_no_flags",
        "Marie Curie worked in Paris.",
        {
            "entities": ["person", "org"],
            "entity_descriptions": {"person": "a person", "org": "an organisation"},
        },
        0.3,
        False,
    ),
    (
        "structures_no_flags",
        "Deep Learning by Yoshua Bengio.",
        structure_schema(
            ["title", "authors"],
            {"title": "str", "authors": "list"},
            description="a paper",
        ),
        0.3,
        False,
    ),
    (
        "relations_no_flags",
        "Marie Curie worked in Paris.",
        {
            "entities": ["person", "org"],
            "entity_descriptions": {"person": "a person", "org": "an organisation"},
            "relations": [{"worked_in": {"head": "person", "tail": "org"}}],
        },
        0.3,
        False,
    ),
    (
        "entities_scalar_dtype_no_flags",
        "Marie Curie worked in Paris.",
        Schema().entities({"person": "a person"}, dtype="str"),
        0.3,
        False,
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
    for case in CASES:
        name, text, schema, threshold = case[:4]
        # `flags` defaults to both-on, which is what the long-document path forces
        # and what every earlier case in this file exercises.
        flags = case[4] if len(case) > 4 else True
        # `format_results=False` stops before the payload shaping, so this is
        # the raw dict the decode stages assembled. `include_confidence` and
        # `include_spans` are both on, which is the combination the long-document
        # path forces (`chunking.py` calls batch_extract with both).
        (raw,) = model.batch_extract(
            [text],
            [schema],
            threshold=threshold,
            format_results=False,
            include_confidence=flags,
            include_spans=flags,
        )
        builder = hasattr(schema, "build")
        # A builder and a dict reach the reference by different routes, and the
        # port needs the pieces of each. `build()` is the prompt-facing schema
        # the port parses into tasks; the metadata tables are what the builder
        # additionally carries and a dict cannot express. Both are recorded so a
        # test can feed the port exactly what the reference saw.
        prompt_schema = schema.build() if builder else schema
        if builder:
            # `build()` emits every container, so a schema that declared only a
            # structure group also carries `"entities": {}` and
            # `"classifications": []`. The reference reads those as "nothing
            # declared" — an empty dict has no keys to route — but the port's
            # parser rejects an empty `entities` outright, so recording the
            # empty containers would make the fixture unusable for the thing it
            # exists to test. They carry no routing information, so they are
            # dropped rather than special-cased on the consuming side.
            prompt_schema = {
                key: value
                for key, value in prompt_schema.items()
                if not (
                    key in ("entities", "relations", "classifications", "json_structures")
                    and not value
                )
            }
        metadata = {
            "entity_metadata": schema._entity_metadata if builder else {},
            "field_metadata": schema._field_metadata if builder else {},
            "entity_order": list(schema._entity_order) if builder else [],
            "classification_tasks": [
                task["task"] for task in prompt_schema.get("classifications", [])
            ],
            "structure_groups": [
                group_name
                for entry in prompt_schema.get("json_structures", [])
                for group_name in entry
            ],
        }
        cases.append({
            "name": name,
            "text": text,
            "schema": prompt_schema,
            "metadata": metadata,
            "built_with_schema_builder": builder,
            "threshold": threshold,
            "include_confidence": flags,
            "include_spans": flags,
            "raw": raw,
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"cases": cases}, indent=1) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
