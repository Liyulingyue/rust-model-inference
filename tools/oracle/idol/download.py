#!/usr/bin/env python3
"""Download only the pinned Idol components directly on the inference host."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


def verified(path, entry):
    if not path.is_file() or path.stat().st_size != entry["Size"]:
        return False
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest() == entry["Sha256"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("models"))
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture"
            entry = dict(Size=3, Sha256=hashlib.sha256(b"abc").hexdigest())
            assert not verified(path, entry)
            path.write_bytes(b"abc")
            assert verified(path, entry)
            path.write_bytes(b"bad")
            assert not verified(path, entry)
        print("PASS: download integrity checks")
        return
    root = args.root.resolve()
    manifest = json.loads(Path(__file__).with_name("manifest.json").read_text())
    for repo in manifest:
        destination = root / repo["directory"]
        missing = [entry["Path"] for entry in repo["files"]
                   if not verified(destination / entry["Path"], entry)]
        if missing and not args.verify_only:
            subprocess.run([str(root / ".venv/bin/modelscope"), "download",
                            "--model", repo["repo"], "--revision", repo["revision"],
                            *missing, "--local_dir", str(destination)], check=True, cwd=root)
        for entry in repo["files"]:
            path = destination / entry["Path"]
            if not verified(path, entry):
                raise ValueError(f"Missing or corrupt pinned component: {path}")
            print(f"verified {path.relative_to(root)} {entry['Sha256']}", flush=True)


if __name__ == "__main__":
    main()
