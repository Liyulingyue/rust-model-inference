"""Oracle for the record decoder's assignment solver.

Dumps ``gliner2.training.matching.linear_sum_assignment`` on hand-written cost
matrices, plus the surrounding ``decode_group`` scalar-assignment arithmetic that
turns a solved matrix into per-instance field choices.

Why the tie-break is the whole point
------------------------------------
The scalar field assignment is a real optimisation problem whose optimum is
frequently **not unique**. Two candidates with equal probability produce two
optima of identical total cost, and *which* one the solver returns decides which
span a record field binds. So checking "the total cost is minimal" is not a test
at all — every solver passes that, including ones that return visibly different
spans. Every case here therefore asserts the exact ``(row, col)`` pairs.

Which solver
------------
The reference has two: SciPy's, when it imports, and its own deterministic
O(n^2 m) Jonker-Volgenant-style solver otherwise. The SciPy path adds a sub-ULP
lexicographic offset "so exact ties resolve reproducibly", so the two can
disagree on which optimum they return. This venv has no SciPy, so the reference
takes the internal path and that is what the fixture pins. See the module note
in ``src/models/gliner_boundary/matching.rs``.

Cases
-----
``unique``: every optimum is unique, so a greedy or wrong solver still passes.
``ties_*``: deliberately degenerate matrices where the optimum is not unique —
all-equal, diagonal-vs-off-diagonal, two disjoint optimal matchings, a plateau
of equal-cost rows. These are the cases that catch a solver with a different
tie-break.
``shape``: rectangular both ways, since the solver transposes a tall matrix and
has to swap the roles back.
``edge``: empty, single cell, a fully infinite row (which the reference turns
into a large finite sentinel rather than failing), and a NaN cell (which it
rejects outright — an optimality comparison against NaN is meaningless, so a
silent answer would be worse than an error).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "target" / "gliner2-oracle") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "target" / "gliner2-oracle"))

import torch  # noqa: E402

# `gliner2.training.matching` and `gliner2.models.boundary.records` import each
# other, so whichever is reached first is a partially-initialised module. Going
# through `common` (which imports the boundary model) establishes the order the
# package expects, the same way the other oracles in this directory do.
from common import fixture_dir  # noqa: E402

from gliner2.training.matching import linear_sum_assignment  # noqa: E402

# `nan` / `inf` are spelled as strings so the fixture stays valid JSON and so a
# case that is *meant* to be infinite cannot be silently rounded.
INF = "inf"
NEG_INF = "-inf"
NAN = "nan"

CASES = [
    # --- unique optimum: the easy baseline ---
    {
        "name": "unique_2x2",
        "cost": [[1.0, 2.0], [2.0, 1.0]],
    },
    {
        "name": "unique_3x3",
        "cost": [[9.0, 1.0, 5.0], [2.0, 8.0, 3.0], [4.0, 6.0, 7.0]],
    },
    # --- degenerate: non-unique optima, which is where solvers diverge ---
    {
        "name": "ties_all_equal_2x2",
        "cost": [[1.0, 1.0], [1.0, 1.0]],
    },
    {
        "name": "ties_all_equal_3x3",
        "cost": [[5.0] * 3 for _ in range(3)],
    },
    {
        "name": "ties_all_equal_4x4",
        "cost": [[2.0] * 4 for _ in range(4)],
    },
    # Two disjoint optimal matchings: (0,0)+(1,1) costs 2, (0,1)+(1,0) costs 4,
    # so this one is NOT tied at the optimum. Kept as a contrast to
    # `ties_disjoint_optimal` below, which makes the two optima equal.
    {
        "name": "diagonal_wins",
        "cost": [[1.0, 2.0], [2.0, 1.0]],
    },
    # (0,0)+(1,1) = 2 and (0,1)+(1,0) = 2: exactly two optimal matchings.
    {
        "name": "ties_disjoint_optimal",
        "cost": [[1.0, 1.0], [1.0, 1.0]],
    },
    # A plateau of rows that all prefer the same cheap column, so the second
    # row has to take the expensive one. Greedy-by-row picks the wrong pairing
    # when it commits the contested column to the wrong row.
    {
        "name": "ties_contested_column",
        "cost": [[0.0, 5.0, 5.0], [0.0, 5.0, 5.0], [9.0, 0.0, 9.0]],
    },
    # Column 0 is much cheaper for row 0, but rows 0 and 1 also tie exactly on
    # the other two columns, so the optimum is not unique in the tail.
    {
        "name": "ties_plateau_tail",
        "cost": [[0.0, 1.0, 1.0], [4.0, 1.0, 1.0], [4.0, 1.0, 1.0]],
    },
    # Two rows whose entire rows are identical, with a cheaper third row: the
    # optimum is unique on cost but the assignment of the identical rows is not.
    {
        "name": "ties_identical_rows",
        "cost": [[1.0, 3.0], [1.0, 3.0], [0.5, 0.5]],
    },
    # --- rectangular, both orientations ---
    {
        "name": "wide_2x4",
        "cost": [[4.0, 1.0, 9.0, 9.0], [9.0, 9.0, 2.0, 3.0]],
    },
    {
        "name": "tall_4x2",
        "cost": [[4.0, 9.0], [1.0, 9.0], [9.0, 2.0], [9.0, 3.0]],
    },
    {
        "name": "wide_1x3",
        "cost": [[3.0, 1.0, 2.0]],
    },
    {
        "name": "tall_3x1",
        "cost": [[3.0], [1.0], [2.0]],
    },
    {
        "name": "square_with_ties_3x3",
        "cost": [[1.0, 1.0, 8.0], [1.0, 1.0, 8.0], [8.0, 8.0, 1.0]],
    },
    # --- edge cases ---
    {"name": "empty_0x3", "cost": []},
    {"name": "empty_3x0", "cost": [[], [], []]},
    {"name": "single_cell", "cost": [[7.5]]},
    # A fully infinite row: the reference substitutes a large finite sentinel
    # so the search still admits a least-bad column instead of failing.
    {
        "name": "infinite_row",
        "cost": [[INF, INF, INF], [1.0, 2.0, 3.0], [3.0, 2.0, 1.0]],
    },
    {
        "name": "mixed_infinities",
        "cost": [[0.0, INF], [INF, 0.0], [1.0, 1.0]],
    },
    # NaN is rejected outright, not propagated: an optimality test against NaN
    # is meaningless, so a silent answer here would be worse than an error.
    {
        "name": "nan_cell_is_rejected",
        "cost": [[1.0, NAN], [2.0, 3.0]],
    },
]


def to_tensor(case: dict) -> torch.Tensor:
    rows = case["cost"]
    if not rows:
        return torch.zeros(0, 0, dtype=torch.float64)
    cols = len(rows[0])
    out = torch.zeros(len(rows), cols, dtype=torch.float64)
    for i, row in enumerate(rows):
        for j, value in enumerate(row):
            if value == INF:
                out[i, j] = float("inf")
            elif value == NEG_INF:
                out[i, j] = float("-inf")
            elif value == NAN:
                out[i, j] = float("nan")
            else:
                out[i, j] = float(value)
    return out


def main() -> None:
    try:
        import scipy  # noqa: F401
    except ImportError:
        pass
    else:
        print(
            "warning: scipy is installed, so the reference used its SciPy path; "
            "the Rust port implements the internal solver and may disagree on ties",
            file=sys.stderr,
        )

    records = []
    for case in CASES:
        cost = to_tensor(case)
        error = None
        try:
            rows_out, cols_out = linear_sum_assignment(cost)
        except Exception as exc:  # noqa: BLE001
            # The NaN case is the only one that should land here; recording the
            # message pins that the reference rejects rather than returns.
            rows_out, cols_out = torch.zeros(0, dtype=torch.long), torch.zeros(
                0, dtype=torch.long
            )
            error = str(exc)
        finite = cost[torch.isfinite(cost)]
        picked = cost[rows_out, cols_out] if len(rows_out) else cost.new_zeros(0)
        total = float(picked.sum()) if len(rows_out) else 0.0
        records.append({
            "name": case["name"],
            "rows": cost.shape[0],
            "cols": cost.shape[1],
            "cost": case["cost"],
            "row_ind": [int(v) for v in rows_out.tolist()],
            "col_ind": [int(v) for v in cols_out.tolist()],
            # `json.dumps` writes a bare `Infinity`, which is valid Python but
            # not valid JSON — serde_json refuses the file outright. Encode
            # non-finite totals the same way the cost cells are encoded.
            "total": total if total == total and abs(total) != float("inf") else (
                INF if total > 0 else NEG_INF
            ),
            "max_finite_abs": float(finite.abs().max()) if finite.numel() else 1.0,
            "error": error,
        })
        print(
            f"  {case['name']}: {list(zip(rows_out.tolist(), cols_out.tolist()))} "
            f"total={total}{'  ERROR: ' + error if error else ''}",
            file=sys.stderr,
        )

    out = fixture_dir() / "assignment-golden.json"
    out.write_text(json.dumps({"cases": records}, indent=1) + "\n")
    print(f"wrote {out} ({len(records)} cases)", file=sys.stderr)


if __name__ == "__main__":
    main()
