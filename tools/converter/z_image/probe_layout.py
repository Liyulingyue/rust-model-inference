"""Probe: does the published Z-Image-Turbo safetensors layout map onto the GGUF
tensor names the Rust loader asks for?

The GGUF schema is fixed by the loader (src/models/diffusion/z_image/dit.rs): 453
tensors, 13 per transformer block, keyed as `layers.{i}.attention.qkv.weight`
and friends. The safetensors from Tongyi-MAI use Diffusers' own naming, which is
similar but not identical -- QKV is usually fused or split differently, and
norm/adaLN tensors are sometimes transposed. This script reads only safetensors
headers (no tensor payloads) so it can answer that question before committing to
a multi-hour conversion.
"""
import json
import re
import struct
import sys
from pathlib import Path


def read_header(path: Path) -> dict[str, list[int]]:
    """Return {tensor_name: dims} from a safetensors header, reading 8 bytes + JSON only."""
    with path.open("rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(length))
    return {
        name: info["shape"]
        for name, info in header.items()
        if name != "__metadata__"
    }


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "Z-Image-Turbo")
    shards = sorted((root / "transformer").glob("*.safetensors"))
    if not shards:
        print(f"no transformer safetensors under {root}/transformer")
        return 1

    names: dict[str, list[int]] = {}
    for shard in shards:
        print(f"header: {shard.name}", flush=True)
        for name, dims in read_header(shard).items():
            names[name] = dims
    print(f"\n{len(names)} tensors across {len(shards)} shard(s)\n")

    # Group by the block-indexed prefix so the shape of the naming is visible.
    indexed = re.compile(r"^(.*?)\.(\d+)\.(.*)$")
    groups: dict[str, set[str]] = {}
    for name in names:
        match = indexed.match(name)
        if match:
            groups.setdefault(match.group(1), set()).add(match.group(3))

    print("=== indexed tensor groups (prefix -> {suffixes}) ===")
    for prefix in sorted(groups):
        suffixes = sorted(groups[prefix])
        print(f"\n{prefix}.{{i}}  ({len(suffixes)} suffixes)")
        for suffix in suffixes:
            example = next(
                n for n in names if indexed.match(n) and n.startswith(f"{prefix}.")
                and indexed.match(n).group(3) == suffix
            )
            print(f"    {suffix:44} {names[example]}")

    print("\n=== non-indexed tensors ===")
    for name in sorted(names):
        if not indexed.match(name):
            print(f"    {name:56} {names[name]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
