"""Render the Unlimited-OCR-2 oracle page fixtures.

The two pages are synthetic: a title and four lines of plain text on
white. They used to be committed as PNG binaries; this script produces
them instead, so the repository carries the recipe rather than the
asset.

The font is Liberation Sans, pinned by SHA-256 because the rasterised
result depends on it. Install it with `apt install fonts-liberation`.
Pillow is pinned in requirements.txt, since the encoder and the
rasteriser both come from it.

    python3 tools/oracle/unlimited_ocr/make_pages.py [--output DIR]

The script prints the SHA-256 of each page. The oracle records those
hashes in verification.json, so a difference against that record means
the font or the Pillow build changed and the recorded numbers no longer
describe the pages on disk.
"""
import argparse
import hashlib
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

#: Liberation Sans, as shipped by Debian's `fonts-liberation`.
FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/TTF/LiberationSans-Regular.ttf",
)
FONT_SHA256 = "4659bc0c58c5028dd488ec928d41d9265db43d9b669fc14ca8b0832daca7b144"

#: The two pages. Each line is (text, size, x, y) in pixels.
PAGES = {
    "page-1": {
        "title": ("UNLIMITED OCR TEST", 48, 60, 77),
        "body": (
            ("Invoice 2026-1028", 34, 60, 190),
            ("Item: Native Rust", 34, 60, 300),
            ("Quantity: 3", 34, 59, 410),
            ("Total: 42.50 USD", 34, 60, 520),
        ),
    },
    "page-2": {
        "title": ("PAGE TWO", 48, 60, 77),
        "body": (
            ("Numerical alignment", 34, 60, 190),
            ("Order: 8675309", 34, 60, 300),
            ("Status: completed", 34, 60, 410),
            ("Thank you.", 34, 60, 520),
        ),
    },
}
WIDTH, HEIGHT = 800, 1000
BACKGROUND = (255, 255, 255)
FOREGROUND = (0, 0, 0)


def font_path() -> Path:
    for candidate in FONT_CANDIDATES:
        path = Path(candidate)
        if path.exists():
            return path
    raise SystemExit(
        "Liberation Sans not found. Install it with `apt install fonts-liberation`, "
        f"or add its path to FONT_CANDIDATES. Looked in: {', '.join(FONT_CANDIDATES)}"
    )


def render(page: dict, font_file: Path) -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)
    title = page["title"]
    draw.text(
        (title[2], title[3]),
        title[0],
        font=ImageFont.truetype(str(font_file), title[1]),
        fill=FOREGROUND,
    )
    for text, size, x, y in page["body"]:
        draw.text((x, y), text, font=ImageFont.truetype(str(font_file), size), fill=FOREGROUND)
    return image


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "fixtures",
        help="directory to write page-1.png and page-2.png into",
    )
    args = parser.parse_args()
    file = font_path()
    digest = hashlib.sha256(file.read_bytes()).hexdigest()
    if digest != FONT_SHA256:
        print(
            f"warning: {file} has SHA-256 {digest}, expected {FONT_SHA256}; "
            "the rendered pages will differ from the recorded ones",
            file=sys.stderr,
        )
    args.output.mkdir(parents=True, exist_ok=True)
    for name, page in PAGES.items():
        image = render(page, file)
        target = args.output / f"{name}.png"
        image.save(target, format="PNG", optimize=False)
        print(f"{target}  {hashlib.sha256(target.read_bytes()).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
