#!/usr/bin/env python3
"""Compare ERNIE parity-trace sidecars with the pinned CPU Oracle (no tolerance)."""
import argparse
import json
import math
import re
import struct
from pathlib import Path


def compare(trace: Path, oracle: Path) -> None:
    matched = values = 0
    names = set()
    dit_layers, text_layers = {}, {}
    token_occurrence = 0
    reference_rows = list(map(json.loads, (oracle / "checkpoints.jsonl").read_text().splitlines()))
    reference = {(r["name"], r["occurrence"]): r for r in reference_rows}
    dit_order, text_order = [], []
    for row in map(json.loads, trace.read_text().splitlines()):
        name = row["name"]
        occurrence = row.get("occurrence", 0)
        transpose = False
        if "token_ids" in row:
            file = oracle / f"rmi.ernie.prompt_ids.{token_occurrence}.u32"
            token_occurrence += 1
            data = file.read_bytes()
            assert row["token_ids"] == list(struct.unpack(f"<{len(data)//4}I", data)), name
            matched += 1
            continue
        if name == "ernie_image.time_frequency":
            continue  # Covered separately by the fixed GGML timestep test.
        if name in {"ernie_image.text.norm", "ernie_image.text.attn_residual"}:
            if occurrence % 25:
                continue  # The Oracle names only each text graph's first block.
            occurrence //= 25
        if name == "ernie_image.text.block":
            occurrence //= 25
            text_layers.setdefault(occurrence, set()).add(row["layer"])
            if row["layer"] < 24:
                text_order.append((occurrence, row["layer"]))
            dest = f"rmi.ernie.text.block.{row['layer']}" if row["layer"] < 24 else "rmi.ernie.context"
        elif name == "ernie_image.block":
            occurrence //= 36
            dit_layers.setdefault(occurrence, set()).add(row["layer"])
            dit_order.append((occurrence, row["layer"]))
            dest = f"rmi.ernie.block.{row['layer']}"
        elif name == "ernie_image.initial_latent":
            dest = "rmi.ernie.input"
        elif name == "ernie_image.sample":
            dest = f"rmi.ernie.sample.{row['layer']}"
            occurrence = 0
        elif name.startswith("z_image.vae."):
            dest = name.replace("z_image.vae.", "rmi.ernie.vae.")
            transpose = name.rsplit(".", 1)[-1] in {"q", "k", "v", "attention_values"}
            if name.endswith("attention_values"):
                dest += "_linear"
        else:
            dest = name.replace("ernie_image.", "rmi.ernie.")
        suffix = f".{occurrence}" if occurrence else ""
        expected = (oracle / f"{dest}{suffix}.f32").read_bytes()
        actual = Path(row["binary_path"]).read_bytes()
        assert len(actual) == len(expected) == row["len"] * 4, (name, "shape/length")
        if name != "ernie_image.sample":
            assert math.prod(reference[(dest, occurrence)]["ne"]) == row["len"], (name, "Oracle shape")
        if transpose:
            # Oracle Linear is [pixels, channels]; shared VAE traces are CHW.
            bits = struct.unpack(f"<{len(expected)//4}I", expected)
            spatial = len(bits) // 512
            expected = struct.pack(f"<{len(bits)}I", *(bits[p * 512 + c] for c in range(512) for p in range(spatial)))
        if actual != expected:
            a = struct.unpack(f"<{len(actual)//4}I", actual)
            b = struct.unpack(f"<{len(expected)//4}I", expected)
            index = next(i for i, pair in enumerate(zip(a, b)) if pair[0] != pair[1])
            raise AssertionError(f"{name} layer={row.get('layer')} occurrence={occurrence} index={index}: {a[index]:08x} != {b[index]:08x}")
        names.add(name)
        matched += 1
        values += row["len"]
    assert "z_image.vae.rgb_channels" in names, "trace must include decoded RGB F32"
    assert all(layers == set(range(36)) for layers in dit_layers.values()), "incomplete DiT trace"
    assert all(layers == set(range(25)) for layers in text_layers.values()), "incomplete text trace"
    for prefix, order in [("block", dit_order), ("text.block", text_order)]:
        if order:
            pattern = re.compile(r"^rmi\.ernie\." + re.escape(prefix) + r"\.(\d+)$")
            expected_order = [(r["occurrence"], int(m[1])) for r in reference_rows if (m := pattern.fullmatch(r["name"]))]
            assert order == expected_order, (prefix, "checkpoint order/count")
    print(f"PASS: {matched} checkpoints, {values:,} F32 values match raw u32 bits")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("oracle", type=Path)
    args = parser.parse_args()
    compare(args.trace, args.oracle)
