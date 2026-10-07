r"""Oracle for the word splitters — `WhitespaceTokenSplitter` and `CharLevelSplitter`.

`gliner2/processing/word_splitter.py` defines both. Every boundary and span
decode indexes words, so the tokenizer's view of where words start and end is
load-bearing for span geometry, for `include_spans` character offsets, and for
long-document chunk boundaries.

What this pins
--------------
The two splitters differ on exactly one axis: which characters may share a
token.

    WhitespaceTokenSplitter  https?://… | www.… | email | @handle | \w+([-_]\w+)* | \S
    CharLevelSplitter        [A-Za-z0-9@._\-+]+ | \S

`\w` in the whitespace splitter is Unicode-aware, so it keeps `café` and CJK
runs whole. The char splitter's class is ASCII-only, so `café` splits into
`caf` + `é` and CJK splits per character. That is the point of it: a
whitespace splitter cannot find word boundaries in a language that does not
delimit them.

Four properties a port has to get right:

1. **Offsets index the original string.** Both splitters match against `text`
   and lower-case only the token value, because case folding can change length
   (`"İ".lower()` is `"i̇"`, two code points) and would corrupt every later
   offset. Lower-casing first and then measuring is the trap.
2. **`\S` is one code point, not one byte and not one grapheme.** So a
   non-ASCII character is one token whether it is one byte in UTF-8 or four,
   and an emoji is a single token. A port that iterates bytes splits every
   multi-byte character; a port that clusters graphemes merges `é` written as
   `e` + a combining accent into one token, which the reference does not do.
3. **The first alternative that matches at a position wins**, so the
   whitespace splitter's URL and email branches take precedence over the bare
   `\w+` branch — `http://a.b` is one token, and so is the `a.b` inside it if
   it were reached separately.
4. **An empty or whitespace-only input yields no tokens at all**, not an empty
   token.

The fixture records `(token, start, end)` for both splitters, so a port that
gets the values right but the offsets wrong still fails.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "target" / "gliner2-oracle"))

from gliner2.processing.word_splitter import (  # noqa: E402
    CharLevelSplitter, WhitespaceTokenSplitter, resolve_word_splitter,
)

# The whitespace splitter is what every boundary checkpoint runs, so its cases
# double as a regression guard on the existing port. The char splitter's cases
# are chosen so each of the four properties above is exercised.
TEXTS = [
    # Plain ASCII.
    "Marie Curie worked in Paris.",
    # The URL / email / handle branches, which must beat the bare `\w+` branch.
    "see https://ex.dev/a?b=c now",
    "mail me at a@b.co",
    "ping @handle_now",
    # `\w+([-_]\w+)*`: interior `-`/`_` stay attached, trailing ones do not.
    "well-known state-of-the_art co-operate",
    # Runs the char splitter must keep whole: every char here is in
    # `[A-Za-z0-9@._\-+]`.
    "a@b.co www.x.dev v1.2.3 x-1 +1-555-0100",
    # CJK, which the whitespace splitter keeps as one `\w+` run and the char
    # splitter breaks per character.
    "中华人民共和国",
    "东京は雨です",
    # Mixed script with no delimiter: the case the char splitter exists for.
    "Hello世界",
    "GPT-4模型发布",
    # Accented Latin: `\w` is Unicode-aware so the whitespace splitter keeps
    # `café` whole, while the char splitter's ASCII-only class splits it.
    "café au lait",
    "naïve résumé",
    # A decomposed accent, where the code point count differs from the grapheme
    # count. Both splitters see one token per code point. A port that clusters
    # graphemes would report one token here; the reference reports two, and it
    # does so for the *whitespace* splitter too because a combining mark is not
    # matched by `\w` either.
    "cafe\u0301",
    "cafe\u0301 noir",
    # Emoji and other symbols: four UTF-8 bytes, one code point, one token.
    "hi 👋🏽 there",
    "a ✓ b",
    # Punctuation between words: `\S` gives each its own token.
    "hi, there!",
    "(paren) [brack] {brace}",
    # Whitespace only, and empty: no tokens at all.
    "   ",
    "",
    # Tabs and newlines, which `\s` covers and `\S` therefore excludes.
    "one\ttwo\nthree",
    # Interior whitespace runs collapse to a boundary rather than a token.
    "a  b   c",
    # The trailing "." the normalizer appends when one is missing.
    "no trailing period",
    "has trailing period.",
    # A word longer than any subword budget, to show offsets do not drift.
    "supercalifragilisticexpialidocious",
    # Non-ASCII digits, which `\w` matches but the char splitter's class does not.
    "١٢٣٤ abc",
    # The two directions in which the `regex` crate's `\w` and Python's `\w`
    # disagree, which is why the whitespace pattern spells the class out as
    # `[\p{L}\p{N}_]` instead of using `\w`. Python's `\w` is "alphanumeric per
    # `str.isalnum()`, plus underscore" = categories L* and N* plus `_`; the
    # crate's is `[\p{Alphabetic}\p{M}\p{Nd}\p{Join_Control}\p{Pc}]`.
    #
    # Marks (Mn/Mc/Me) are `\w` to the crate and not to Python: U+0301 combining
    # acute, U+0903 Devanagari spacing vowel sign, U+20DD enclosing mark.
    "cafe\u0301 x\u0903 y\u20dd",
    # Nl and No are `\w` to Python and not to the crate: U+2160 Roman numeral
    # one, U+00BD vulgar fraction one half.
    "\u2160 \u00bd",
    # Join_Control is `\w` to the crate and not to Python: ZWJ U+200D. A
    # zero-width joiner is a format character, not alphanumeric.
    "a\u200db",
    # Cf is neither, but it is invisible, so this pins that a soft hyphen is a
    # `\S` token rather than being silently dropped.
    "a\u00adb",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path,
        default=REPO_ROOT / "tests" / "fixtures" / "gliner2.5-base-v1"
        / "word-splitter-golden.json",
    )
    args = parser.parse_args()

    whitespace = WhitespaceTokenSplitter()
    char_level = CharLevelSplitter()

    cases = []
    for text in TEXTS:
        cases.append({
            "text": text,
            "whitespace": [
                [token, start, end] for token, start, end in whitespace(text, True)
            ],
            "char": [
                [token, start, end] for token, start, end in char_level(text, True)
            ],
        })

    # `lower=False` must return the same offsets with the original casing, which
    # is the property that makes the "match first, lower after" ordering
    # observable rather than incidental.
    case_preserving = []
    for text in TEXTS:
        case_preserving.append([
            [token, start, end] for token, start, end in whitespace(text, False)
        ])

    # The resolver is the public entry point: names, classes and callables all
    # resolve, and an unknown name is a `ValueError` naming the supported set.
    resolved = {
        "whitespace": type(resolve_word_splitter("whitespace")).__name__,
        "char": type(resolve_word_splitter("char")).__name__,
        "default": type(resolve_word_splitter(None)).__name__,
        "whitespace_class": type(resolve_word_splitter(WhitespaceTokenSplitter)).__name__,
        "char_callable": type(
            resolve_word_splitter(lambda text, lower=True: iter(()))
        ).__name__,
    }
    try:
        resolve_word_splitter("nope")
        unknown = None
    except ValueError as error:
        unknown = str(error)

    payload = {
        "cases": cases,
        "whitespace_case_preserving": case_preserving,
        "resolve_word_splitter": resolved,
        "unknown_name_error": unknown,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(__import__("json").dumps(payload, ensure_ascii=False) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
