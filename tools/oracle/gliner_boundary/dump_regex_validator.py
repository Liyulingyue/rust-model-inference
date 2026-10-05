"""Oracle for `RegexValidator` — the schema-level span filter.

``gliner2/inference/schema.py:27-55`` is a four-field post-processing filter over
one span's surface text::

    pattern: str
    mode:    "full" | "partial"   (default "full")
    exclude: bool = False
    flags:   int  = re.IGNORECASE  (default case-insensitive)

    validate(text) = (fullmatch | search)(text) is not None, negated if exclude

What this pins
--------------
A port has to reimplement this on a different regex engine, and the engines do
not agree. The cases below are the ones that actually diverge, each chosen
because ``regex`` in Rust is a documented *different implementation*, not a
binding to CPython's:

1. **Anchoring.** ``fullmatch`` is the whole string, not "search plus anchors".
   In Python ``$`` also matches immediately before a trailing newline, so
   ``re.fullmatch(r"a$", "a\\n")`` succeeds. Rust's ``$`` matches only at the end
   of the haystack, so the equivalent would fail; ``\\A(?:...)\\z`` is the only
   spelling that means the same thing in both. A trailing-newline surface is not
   hypothetical — ``_normalize_text`` appends a ``"."``.
2. **Case folding is not the same set.** Python's ``re.IGNORECASE`` on ``str``
   folds with the full Unicode case-folding table plus a few special cases
   (U+212A KELVIN folds to ``k``, U+017F LATIN SMALL LETTER LONG S folds to
   ``s``). Rust's ``(?i)`` uses Unicode simple case folding. They agree on most
   letters and disagree on exactly these.
3. **``.`` does not match a newline in either engine**, but ``(?s)``/``re.DOTALL``
   do, and a surface can contain one.
4. **Class shorthands**: ``\\w`` and ``\\d`` are Unicode-aware in both, and
   ``\\b`` is a word boundary in both, so those are safe.

``validate`` is a pure function of ``(pattern, mode, exclude, flags, text)``, so
the fixture is a truth table rather than a model run. The case surface
concentrates on span-shaped strings: names, identifiers, URLs, numbers, and the
hyphen/dot runs that ``_normalize_text`` and the word splitter produce.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "target" / "gliner2-oracle"))

from gliner2.inference.schema import RegexValidator  # noqa: E402

# (name, pattern, mode, exclude, flags, texts)
#   flags is spelled by name here so the fixture records the intent rather than
#   the integer, and the integer is written into the fixture for the port.
ICASE = re.IGNORECASE
CASE = 0

VALIDATORS = [
    # --- mode: full vs partial -------------------------------------------------
    ("full_exact_name", r"[A-Z][a-z]+", "full", False, ICASE,
     ["Marie", "marie curie", "MARIE", "M", "", "Jean-Luc"]),
    ("partial_exact_name", r"[A-Z][a-z]+", "partial", False, ICASE,
     ["Marie", "marie curie", "MARIE", "M", "", "Jean-Luc"]),
    # A pattern that matches the empty string: `full` keeps everything, and only
    # `exclude` can reject.
    ("full_matches_empty", r".*", "full", False, ICASE,
     ["", "a", "anything at all", "with\nnewline"]),
    # --- exclude --------------------------------------------------------------
    ("full_exclude_digits", r"\d+", "full", True, ICASE,
     ["123", "12a", "a12", "abc", "", "007"]),
    ("partial_exclude_digits", r"\d", "partial", True, ICASE,
     ["abc", "a1", "1", "", "no digits here", "x9y"]),
    # --- anchoring ------------------------------------------------------------
    # `$` before a trailing newline is the divergence: Python fullmatch accepts
    # "a\n", Rust's `$` would not.
    ("full_dollar_anchor", r"a$", "full", False, ICASE,
     ["a", "a\n", "ab", "b\na", ""]),
    ("full_caret_anchor", r"^a", "full", False, ICASE,
     ["a", "ab", "ba", "b\na", "\na"]),
    # Explicit absolute anchors, which both engines spell the same way.
    ("full_absolute_anchors", r"\Aa\z", "full", False, ICASE,
     ["a", "a\n", "ab", ""]),
    # --- case folding ---------------------------------------------------------
    # The Kelvin sign and long s are where Python and Rust disagree.
    ("full_kelvin", r"k", "full", False, ICASE,
     ["k", "K", "\u212a", "\u212b", "kk", "\u212a\u212a"]),
    # Kelvin with a *class* rather than a literal, which is the spelling a
    # user is likelier to write.
    ("full_kelvin_class", r"[k]", "full", False, ICASE,
     ["k", "K", "\u212a", "\u212b"]),
    # Angstrom sign has no case mapping at all, so it must not match even though
    # it looks like a decorated A.
    ("full_angstrom", r"a", "full", False, ICASE,
     ["a", "A", "\u212b", "\u00e5"]),
    # The long s folds to plain `s`, so the pattern has to be the bare letter;
    # a multi-letter pattern like `sigma` cannot show it.
    ("full_long_s", r"s", "full", False, ICASE,
     ["s", "S", "\u017f", "\u1e9e", "ss", "s\u017f"]),
    # Dotted capital I: Python folds it, and the lowercasing is two code points,
    # which is why the reference forbids lower-casing the source text first.
    ("full_dotted_i", r"i", "full", False, ICASE,
     ["i", "I", "\u0130", "\u0131", "i\u0307"]),
    # Case-sensitive: the same inputs must NOT match.
    ("full_case_sensitive", r"[a-z]+", "full", False, CASE,
     ["abc", "ABC", "Abc", "", "aBc"]),
    # --- dot and newlines -----------------------------------------------------
    ("partial_dot_no_newline", r"a.b", "partial", False, ICASE,
     ["axb", "a\nb", "a b", "ab"]),
    ("partial_dotall", r"a.b", "partial", False, ICASE | re.DOTALL,
     ["axb", "a\nb", "a b", "ab"]),
    ("full_dot_no_newline", r"a.b", "full", False, ICASE,
     ["axb", "a\nb", "a b", "ab"]),
    # --- shapes a span surface actually takes ---------------------------------
    ("full_url", r"https?://[^\s]+", "full", False, ICASE,
     ["https://a.dev", "http://a.dev/x?y=1", "ftp://a.dev", "a.dev"]),
    ("full_email", r"[^@\s]+@[^@\s]+\.[a-z]{2,}", "full", False, ICASE,
     ["a@b.co", "a@b", "not-an-email", "@b.co"]),
    ("full_number", r"\d+(?:\.\d+)?", "full", False, ICASE,
     ["42", "3.14", "1e5", "-1", ""]),
    # A trailing period, which `_normalize_text` appends, and a span boundary
    # that does not include it.
    ("full_trailing_period", r"[a-z]+\.", "full", False, ICASE,
     ["end.", "end", "end..", ".end"]),
    # Unicode identifiers, which is where `\w` matters.
    ("full_unicode_word", r"\w+", "full", False, ICASE,
     ["abc", "\u4e2d\u6587", "a\u0301", "", "a-b"]),
    # Hyphen and underscore runs, which the whitespace splitter keeps together
    # only for interior separators.
    ("full_slug", r"[a-z]+(?:-[a-z]+)+", "full", False, ICASE,
     ["well-known", "well", "well-known-state", "-well", "well-"]),
    # Word boundaries, which both engines spell the same way.
    ("partial_word_boundary", r"\bcat\b", "partial", False, ICASE,
     ["cat", "cats", "the cat sat", "cat.", "cat's", "concat"]),
    ("full_word_boundary", r"\bcat\b", "full", False, ICASE,
     ["cat", "cats", "the cat sat", "cat."]),
]

# Cases where a `full` validator and a `partial` one must disagree, so a port
# that ignores `mode` fails.
MODE_SENSITIVE = [
    ("mode_sensitive_plain", r"[A-Z][a-z]+", "Marie", "marie curie"),
    ("mode_sensitive_suffix", r"curie", "curie", "marie curie"),
    ("mode_sensitive_digit", r"\d+", "123", "a123"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path,
        default=REPO_ROOT / "tests" / "fixtures" / "gliner2.5-base-v1"
        / "regex-validator-golden.json",
    )
    args = parser.parse_args()

    cases = []
    for name, pattern, mode, exclude, flags, texts in VALIDATORS:
        validator = RegexValidator(
            pattern=pattern, mode=mode, exclude=exclude, flags=flags
        )
        cases.append({
            "name": name,
            "pattern": pattern,
            "mode": mode,
            "exclude": exclude,
            "ignore_case": bool(flags & ICASE),
            "dot_all": bool(flags & re.DOTALL),
            # The raw flags integer, so a port that wants to mirror the
            # reference's `flags` field exactly has it.
            "flags": int(flags),
            "results": [
                [text, bool(validator.validate(text))] for text in texts
            ],
        })

    mode_cases = []
    for name, pattern, full_text, partial_text in MODE_SENSITIVE:
        full = RegexValidator(pattern=pattern, mode="full")
        partial = RegexValidator(pattern=pattern, mode="partial")
        mode_cases.append({
            "name": name,
            "pattern": pattern,
            "full_matches": bool(full.validate(full_text)),
            "full_matches_partial_text": bool(full.validate(partial_text)),
            "partial_matches": bool(partial.validate(partial_text)),
        })

    # `mode` outside its two values, and an uncompilable pattern, are both
    # construction-time ValueErrors, so they cannot be reached from a decode.
    errors = {}
    for label, kwargs in [
        ("bad_mode", {"pattern": "a", "mode": "prefix"}),
        ("bad_pattern", {"pattern": "a(", "mode": "full"}),
    ]:
        try:
            RegexValidator(**kwargs)
            errors[label] = None
        except ValueError as error:
            errors[label] = str(error)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {
            "cases": cases,
            "mode_cases": mode_cases,
            "construction_errors": errors,
        },
        ensure_ascii=False,
    ) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
