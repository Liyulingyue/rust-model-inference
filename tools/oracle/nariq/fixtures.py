"""Generate the inputs for the NariQ extended checks on the machine holding the GGUFs."""

import argparse
from pathlib import Path
import struct
import subprocess
import zlib


def ledger(count, question):
    labels = ("maple", "cedar", "amber", "linen", "slate")
    lines = ["A library ledger lists the label of every entry."]
    for index in range(count):
        label = "orchid" if index == 17 else labels[index % len(labels)]
        lines.append(f"Entry {index:03d} holds the label {label}.")
    return "\n".join([*lines, "", question])


def spatial_png(path, width, height, blue_half):
    def chunk(name, data):
        return struct.pack(">I", len(data)) + name + data + struct.pack(">I", zlib.crc32(name + data))

    red, blue = bytes((231, 32, 24)), bytes((18, 68, 226))
    pixels = b"".join(
        b"\0" + b"".join(blue if blue_half(x, y) else red for x in range(width))
        for y in range(height)
    )
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(pixels))
        + chunk(b"IEND", b"")
    )


def video(path, colors):
    # Four frames at 2 fps. Each temporal patch sees a single solid color.
    frames = b"".join(bytes(color) * (96 * 96 * 2) for color in colors)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "96x96",
         "-r", "2", "-i", "pipe:0", "-an", "-c:v", "libx264", "-threads", "1", "-pix_fmt",
         "yuv420p", "-crf", "12", "-map_metadata", "-1", "-movflags", "+faststart", str(path)],
        input=frames,
        check=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--video", action="store_true", help="also generate the two MP4s with FFmpeg")
    args = parser.parse_args()
    dest = args.directory
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "continuation.txt").write_text(
        "The counting song goes on, one number per line:\nOne\nTwo\nThree\nFour\nFive\nSix\n"
    )
    (dest / "ledger.txt").write_text(ledger(128, "What is the label of entry 017? The label is"))
    for count in (360, 768):
        (dest / f"ledger-{count}.txt").write_text(
            ledger(count, "Which label belongs to entry 017? Reply with only its label.")
        )
    spatial_png(dest / "left.png", 192, 96, lambda x, y: x >= 96)
    spatial_png(dest / "top.png", 96, 192, lambda x, y: y < 96)
    if args.video:
        red, blue = (224, 24, 28), (20, 42, 225)
        video(dest / "red-blue.mp4", (red, blue))
        video(dest / "blue-red.mp4", (blue, red))


if __name__ == "__main__":
    main()
