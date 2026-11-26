"""Render the deterministic page images the OCRia end-to-end test reads.

The three pages exercise the pipeline's discrete decisions rather than its
accuracy: a single-column receipt (the ordinary path), a two-column page with
a wide gutter (the column split), and the second page rotated a quarter turn
(the sideways retry). They are synthetic on purpose -- no third-party document
is redistributed with this repository -- and every coordinate is fixed, so a
regenerated page is byte-identical.

    python3 tools/oracle/ocria/make_pages.py tools/oracle/ocria/fixtures/pages

The oracle lines for each page are recorded next to them as `<page>.json` by
the pinned LiteParse engine; see README.md in this directory.
"""
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

#: Font search order: the DejaVu files a Linux CI box has, then the macOS
#: equivalents. `--font`/`--font-bold` override both. The committed pages were
#: rendered with the first pair that existed on the machine that recorded them;
#: regenerating with a different font changes the glyphs and therefore the
#: oracle JSON next to each page, which has to be re-recorded with it.
FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial.ttf",
]
FONT_BOLD_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
]


def first_existing(paths, override):
    """The font file to use: the override, else the first candidate present."""
    if override:
        return override
    for path in paths:
        if Path(path).is_file():
            return path
    raise SystemExit(f"no font found in {paths}; pass --font")
SIZE = (600, 800)


def receipt(font: str, font_bold: str) -> Image.Image:
    """One column, five blocks: the ordinary single-column read."""
    img = Image.new("RGB", SIZE, "white")
    d = ImageDraw.Draw(img)
    bold = ImageFont.truetype(font_bold, 34)
    body = ImageFont.truetype(font, 22)
    small = ImageFont.truetype(font, 19)
    d.text((60, 50), "CORNER STORE", font=bold, fill="black")
    d.text((60, 100), "12 Market Street", font=small, fill="black")
    d.text((60, 126), "Open 08:00 - 20:00", font=small, fill="black")
    for i, line in enumerate(
        [
            "Coffee beans      2    12.50",
            "Sourdough         1     4.20",
            "Olive oil         1     8.90",
            "Paper towels      3     6.75",
        ]
    ):
        d.text((60, 200 + i * 40), line, font=body, fill="black")
    d.text((60, 400), "TOTAL               32.35", font=body, fill="black")
    d.text((60, 440), "CARD                32.35", font=body, fill="black")
    d.text((60, 520), "Thank you for shopping", font=small, fill="black")
    return img


def two_column(font: str, font_bold: str) -> Image.Image:
    """Two columns with a gutter wider than the line spacing."""
    img = Image.new("RGB", SIZE, "white")
    d = ImageDraw.Draw(img)
    bold = ImageFont.truetype(font_bold, 26)
    body = ImageFont.truetype(font, 20)
    d.text((40, 40), "Annual Report", font=bold, fill="black")
    left = [
        "Revenue grew by",
        "eleven percent in",
        "the first half of",
        "the year, driven",
        "by the services",
        "segment.",
    ]
    right = [
        "Costs stayed flat",
        "against the plan,",
        "leaving room for",
        "the research team",
        "to expand in the",
        "next quarter.",
    ]
    for i, line in enumerate(left):
        d.text((40, 120 + i * 30), line, font=body, fill="black")
    for i, line in enumerate(right):
        d.text((340, 120 + i * 30), line, font=body, fill="black")
    for i in range(6):
        d.text((40, 420 + i * 30), left[i].upper(), font=body, fill="black")
        d.text((340, 420 + i * 30), right[i].upper(), font=body, fill="black")
    return img


def sideways(font: str, font_bold: str) -> Image.Image:
    """The receipt page a quarter turn clockwise: the sideways retry."""
    return receipt(font, font_bold).rotate(-90, expand=True)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?",
                        default="tools/oracle/ocria/fixtures/pages")
    parser.add_argument("--font", help="regular font file")
    parser.add_argument("--font-bold", help="bold font file")
    args = parser.parse_args()
    font = first_existing(FONT_CANDIDATES, args.font)
    font_bold = first_existing(FONT_BOLD_CANDIDATES, args.font_bold)
    print(f"font: {font}\nbold: {font_bold}")
    out = Path(args.directory)
    out.mkdir(parents=True, exist_ok=True)
    for name, image in [
        ("receipt", receipt(font, font_bold)),
        ("two_column", two_column(font, font_bold)),
        ("sideways", sideways(font, font_bold)),
    ]:
        path = out / f"{name}.png"
        image.save(path, optimize=True)
        print(f"{path} {image.size}")


if __name__ == "__main__":
    main()
