"""Oracle for record-metadata normalization and spec compilation.

Dumps ``gliner2.processing.records.normalize_record_metadata`` and
``compile_record_specs`` over hand-written cases. Both are pure — they take a
metadata mapping plus a query layout and produce specs — so this needs no GGUF
and no model weights, and it always runs.

Why the validation half matters
-------------------------------
``normalize_record_metadata`` is not bookkeeping. It is where the reference
rejects the schemas that would otherwise crash or silently mis-decode much
later, and each rule has a distinct failure:

- ``mode`` missing: the group is *left alone* (legacy path), not defaulted. This
  is the one case that is a silent no-op rather than an error, which is exactly
  why it is worth a fixture case.
- ``natural`` without ``anchor``: ``RecordSpec.__post_init__`` raises.
- non-``natural`` *with* ``anchor``: also raises. So the anchor is not merely
  ignored in latent/anchorless mode, it is an error — a caller who set it is
  telling us they think this is a natural record.
- ``cardinality`` outside the four enum values: raises.
- anchor field gets ``required_one`` by default regardless of any declared
  dtype, because an instance must have its anchor.

``compile_record_specs`` then binds each field to a ``query_id`` from the layout,
which is the same id space relations use, and sorts fields by ``role_index``.

Cases
-----
``valid_*``: the three modes, each with scalar and list fields, plus a group with
no metadata at all (the legacy no-op).
``default_*``: cardinality defaults — anchor field, ``str`` dtype, and the
fall-through to ``zero_or_more``.
``error_*``: every rejection above, recorded by message so a relaxed check fails
here.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "target" / "gliner2-oracle") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "target" / "gliner2-oracle"))

from gliner2.models.base import QueryLayout, QuerySpec  # noqa: E402
from gliner2.processing.records import (  # noqa: E402
    compile_record_specs,
    normalize_record_metadata,
)


def layout(groups):
    """A query layout: `groups` is a list of (task_name, task_type, [role names])."""
    queries = []
    query_id = 0
    for task_index, (name, task_type, roles) in enumerate(groups):
        for role_index, role in enumerate(roles):
            queries.append(
                QuerySpec(
                    query_id=query_id,
                    task_index=task_index,
                    task_type=task_type,
                    task_name=name,
                    role_index=role_index,
                    role_name=role,
                )
            )
            query_id += 1
    return QueryLayout(queries=tuple(queries))


# A single `json_structures` group named "person" with four fields, which is
# enough to exercise anchor selection, scalar vs list cardinality and the
# role_index ordering.
PERSON = ("person", "json_structures", ["name", "employer", "role", "born"])

CASES = [
    {
        "name": "valid_natural",
        "groups": [PERSON],
        "field_dtypes": {"person": {"name": "str", "employer": "str", "role": "str",
                                    "born": "str"}},
        "record_metadata": {
            "person": {
                "mode": "natural",
                "anchor": "name",
                "fields": {
                    "employer": {"cardinality": "optional_one", "exclusive": True},
                    "role": {"cardinality": "zero_or_more"},
                },
            }
        },
    },
    {
        "name": "valid_latent",
        "groups": [PERSON],
        "field_dtypes": {"person": {"name": "str", "employer": "str", "role": "str",
                                    "born": "str"}},
        "record_metadata": {
            "person": {
                "mode": "latent",
                "fields": {
                    "name": {"cardinality": "required_one"},
                    "employer": {"cardinality": "required_one", "exclusive": True},
                    "role": {"cardinality": "one_or_more"},
                },
            }
        },
    },
    {
        "name": "valid_anchorless",
        "groups": [PERSON],
        "field_dtypes": {"person": {"name": "str"}},
        "record_metadata": {
            "person": {
                "mode": "anchorless",
                "occurrence_policy": "first",
            }
        },
    },
    {
        "name": "no_metadata_is_a_legacy_no_op",
        "groups": [PERSON],
        "field_dtypes": {"person": {"name": "str"}},
        "record_metadata": None,
    },
    {
        "name": "mode_absent_is_a_legacy_no_op",
        # An entry with no `mode` is explicitly *skipped*, not defaulted. So this
        # group gets no spec even though it is named.
        "groups": [PERSON],
        "field_dtypes": {"person": {"name": "str"}},
        "record_metadata": {"person": {"anchor": "name"}},
    },
    {
        "name": "metadata_for_an_absent_group",
        "groups": [PERSON],
        "field_dtypes": {"person": {"name": "str"}},
        "record_metadata": {"other": {"mode": "latent"}},
    },
    {
        "name": "defaults_anchor_dtype_and_fallback",
        # No `fields` block at all, so every cardinality comes from
        # `_default_cardinality`: the anchor is required_one, `str` dtypes are
        # optional_one, and the untyped field falls through to zero_or_more.
        "groups": [PERSON],
        "field_dtypes": {"person": {"name": "str", "employer": "str", "role": "str"}},
        "record_metadata": {"person": {"mode": "natural", "anchor": "name"}},
    },
    {
        "name": "explicit_beats_dtype",
        # `born` is declared "str" (which would default to optional_one) but the
        # metadata says one_or_more, which must win.
        "groups": [PERSON],
        "field_dtypes": {"person": {"name": "str", "born": "str"}},
        "record_metadata": {
            "person": {
                "mode": "natural",
                "anchor": "name",
                "fields": {"born": {"cardinality": "one_or_more"}},
            }
        },
    },
    {
        "name": "two_groups_keep_their_own_anchor",
        "groups": [
            PERSON,
            ("company", "json_structures", ["name", "founder", "city"]),
        ],
        "field_dtypes": {},
        "record_metadata": {
            "person": {"mode": "natural", "anchor": "name"},
            "company": {"mode": "natural", "anchor": "founder"},
        },
    },
    # --- rejections, recorded by message ---
    {
        "name": "error_natural_without_anchor",
        "groups": [PERSON],
        "field_dtypes": {},
        "record_metadata": {"person": {"mode": "natural"}},
    },
    {
        "name": "error_latent_with_anchor",
        "groups": [PERSON],
        "field_dtypes": {},
        "record_metadata": {"person": {"mode": "latent", "anchor": "name"}},
    },
    {
        "name": "error_unknown_mode",
        "groups": [PERSON],
        "field_dtypes": {},
        "record_metadata": {"person": {"mode": "greedy"}},
    },
    {
        "name": "error_bad_cardinality",
        "groups": [PERSON],
        "field_dtypes": {},
        "record_metadata": {
            "person": {
                "mode": "latent",
                "fields": {"employer": {"cardinality": "maybe"}},
            }
        },
    },
    {
        "name": "error_anchor_not_a_field",
        # The anchor names a field the group does not declare. Normalization
        # accepts it (it only checks that `anchor` is present), so the failure
        # lands in `compile_record_specs` looking for a matching field.
        "groups": [PERSON],
        "field_dtypes": {},
        "record_metadata": {"person": {"mode": "natural", "anchor": "nope"}},
    },
    {
        "name": "error_bad_occurrence_policy",
        "groups": [PERSON],
        "field_dtypes": {},
        "record_metadata": {
            "person": {"mode": "latent", "occurrence_policy": "whenever"}
        },
    },
    {
        "name": "error_metadata_not_a_mapping",
        "groups": [PERSON],
        "field_dtypes": {},
        "record_metadata": {"person": "natural"},
    },
]


def main() -> None:
    records = []
    for case in CASES:
        entry = {
            "name": case["name"],
            "groups": [
                {"name": name, "task_type": task_type, "roles": list(roles)}
                for name, task_type, roles in case["groups"]
            ],
            "field_dtypes": case["field_dtypes"],
            "record_metadata": case["record_metadata"],
        }
        query_layout = layout(case["groups"])
        error = None
        normalized: dict = {}
        specs: dict = {}
        try:
            normalized = normalize_record_metadata(
                case["record_metadata"], field_dtypes=case["field_dtypes"]
            )
            compiled = compile_record_specs(
                query_layout=query_layout,
                record_metadata=case["record_metadata"],
                field_dtypes=case["field_dtypes"],
            )
            specs = {
                str(key): {
                    "task_name": spec.task_name,
                    "task_type": spec.task_type,
                    "mode": spec.mode,
                    "anchor_query_id": spec.anchor_query_id,
                    "occurrence_policy": spec.occurrence_policy,
                    "fields": [
                        {
                            "query_id": field.query_id,
                            "name": field.name,
                            "role_index": field.role_index,
                            "cardinality": field.cardinality.value,
                            "is_anchor": field.is_anchor,
                            "exclusive": field.exclusive,
                        }
                        for field in spec.fields
                    ],
                }
                for key, spec in compiled.items()
            }
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
        entry["normalized"] = normalized
        entry["specs"] = specs
        entry["error"] = error
        records.append(entry)
        print(
            f"  {case['name']}: {len(specs)} spec(s)"
            f"{'  ERROR: ' + error if error else ''}",
            file=sys.stderr,
        )

    out = REPO_ROOT / "tests" / "fixtures" / "gliner2.5-base-v1" / "record-specs-golden.json"
    out.write_text(json.dumps({"cases": records}, indent=1) + "\n")
    print(f"wrote {out} ({len(records)} cases)", file=sys.stderr)


if __name__ == "__main__":
    main()
