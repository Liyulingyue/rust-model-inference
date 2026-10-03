"""Golden for the `BoundaryAttentionBlock` local-window mask.

The boundary encoder's attention is restricted to `|i - j| <= window` when
``boundary_attention_window > 0`` (``encoding.py:122-131``). base-v1 sets it to
128, so the band only starts excluding keys once a document has more than
257 boundary positions (``2 * window + 1``). Every other fixture in this
directory uses a document of at most 24 tokens, so a port that simply omitted
the window would still be byte-exact against all of them. This fixture exists
to close that hole: it covers ``n = 8`` (window inactive) and ``n = 273``
(window active) and records, per query row, the exact set of allowed key
positions.

Recording the allowed set as ``(row, allowed_keys)`` rather than the raw
``n x n`` matrix keeps the fixture small while still pinning every mask entry.

Regenerate with:
    PYTHONPATH=target/gliner2-oracle \\
        models/.venv/bin/python3 \\
        tools/oracle/gliner_boundary/dump_boundary_attention_window.py
"""
import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "target" / "gliner2-oracle"))

from gliner2.models.boundary.encoding import BoundaryAttentionBlock  # noqa: E402

# n = 8 stays under 2 * 128 + 1 so the band covers everything; n = 273 is just
# past it, so the first and last rows lose keys.
SHAPES = [8, 273]


def allowed_keys(window: int, n: int) -> list[list[int]]:
    """Rebuild the reference mask and report the allowed keys per query row.

    Mirrors ``BoundaryAttention.forward`` with an all-valid boundary mask:
    ``allowed = mask[j] & |i - j| <= window``, then ``| eye``. The diagonal OR
    is unconditional in the reference (it is not gated on ``mask[i]``), so a
    padding query row still has exactly one legal key and cannot NaN.
    """
    block = BoundaryAttentionBlock(dim=1, num_heads=1, window=window, dropout=0.0)
    mask = torch.ones(1, n, dtype=torch.bool)
    positions = torch.arange(n)
    local = (positions.view(n, 1) - positions.view(1, n)).abs() <= window
    allowed = mask.view(1, 1, 1, n) & local.view(1, 1, n, n)
    allowed = allowed | torch.eye(n, dtype=torch.bool).view(1, 1, n, n)
    # Sanity: the block must accept this mask shape without complaint.
    block(torch.zeros(1, n, 1), mask)
    return [row.nonzero().flatten().tolist() for row in allowed[0, 0]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--window",
        type=int,
        default=128,
        help="boundary_attention_window from config.json (128 for base-v1)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "tests" / "fixtures" / "gliner2.5-base-v1"
        / "boundary-attention-window-golden.json",
    )
    args = parser.parse_args()

    fixture = {
        "window": args.window,
        "shapes": [
            {"n": n, "allowed": allowed_keys(args.window, n)} for n in SHAPES
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(fixture) + "\n")
    counts = {shape["n"]: sum(len(row) for row in shape["allowed"]) for shape in fixture["shapes"]}
    print(f"{args.out} (allowed entries: {counts})")


if __name__ == "__main__":
    main()
