"""Dump reference SentencePiece encodings for the GLiNER2.5-Decide vocabulary.

Produces `tests/fixtures/gliner2-decide/spm-pieces.json`: a list of
`{"text", "pieces", "ids"}` that `tests/gliner2_spm_parity.rs` must reproduce
byte for byte. The corpus deliberately covers the cases that are easy to get
wrong: NFKC folding (`①`, `Ⅷ`, `™`, fullwidth digits), byte fallback for
characters outside the 128k training coverage, whitespace collapsing, and empty
input.

    models/.venv/bin/python tools/oracle/gliner2/dump_spm.py
"""
import argparse
import json
from pathlib import Path

import sentencepiece as spm

REPO_ROOT = Path(__file__).resolve().parents[3]

TEXTS = [
    # GLiNER's own vocabulary: the pieces the SchemaTransformer hands to the tokenizer.
    "(",
    ")",
    ",",
    "|",
    "intent",
    "answer: Did the treaty enter into force in 1992?",
    "order_status",
    "refund_request",
    "cancel_subscription",
    "card_pin_change: The customer wants a new PIN or the current PIN replaced",
    "Did the treaty enter into force in 1992?",
    "Guest in room 1408 says the AC has been out since yesterday",
    # The word splitter's alphabet.
    "my",
    "subscription",
    "renewed",
    "on",
    "april",
    "15",
    "for",
    "5,400",
    ".",
    "?",
    "!",
    "¥5,400",
    "e-mail",
    "user_name",
    "https://example.com/a?b=c",
    "www.example.com",
    "a@b.co",
    "@handle",
    "don't",
    "U.S.A.",
    "AC",
    "0",
    "10",
    "3.14",
    "-42",
    # Case, accents, punctuation, whitespace handling.
    "Hello",
    "HELLO",
    "café",
    "CAFÉ",
    "naïve",
    "  leading and   inner  spaces  ",
    "tab\there",
    "line\nbreak",
    "full-width　space",
    "a b",
    "— dash",
    "quote’s",
    "…ellipsis",
    "① circled",
    "Ⅷ roman",
    "™ trademark",
    "１２３ fullwidth digits",
    # Byte fallback territory: characters outside the 128k training coverage.
    "emoji 😀 here",
    "rare: ꓐ ꓱ ꓲ",
    "cjk: 日本語のテキスト",
    "mixed café 😀 123",
    "zero\u0000width",
    "combining a\u0301e\u0300",
    # Longer realistic sentences.
    "My subscription renewed on April 15 for 5,400 after the service was already down.",
    "Battery dies before lunch, but the keyboard and the screen are the best I have used.",
    "The central bank held rates and said inflation is still above target.",
    "She closed the ledger, blew out the lamp, and listened for the stair.",
    "The treaty was signed in Paris in 1992. It entered into force the following year.",
    # Degenerate inputs.
    "",
    " ",
    "   ",
    ".",
]


def main():
    arguments = argparse.ArgumentParser(description=__doc__)
    arguments.add_argument(
        "--model",
        type=Path,
        default=REPO_ROOT / "models/GLiNER2.5-Decide/spm.model",
    )
    arguments.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "tests/fixtures/gliner2-decide/spm-pieces.json",
    )
    options = arguments.parse_args()

    sp = spm.SentencePieceProcessor(model_file=str(options.model))
    rows = []
    for text in TEXTS:
        pieces = sp.encode(text, out_type=str)
        ids = sp.encode(text, out_type=int)
        assert [sp.id_to_piece(i) for i in ids] == pieces, text
        rows.append({"text": text, "pieces": pieces, "ids": ids})
    options.out.parent.mkdir(parents=True, exist_ok=True)
    options.out.write_text(json.dumps(rows, indent=1, ensure_ascii=False) + "\n")
    print(f"{options.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
