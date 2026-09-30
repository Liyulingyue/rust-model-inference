"""Replay pinned GLiNER2 requests through the scalar Oracle and Rust CLI."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from tools.oracle.gliner.compare import compare


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", choices=("refund", "multitask", "examples", "long-position"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    fixtures = Path(__file__).with_name("fixtures")
    names = [args.case] if args.case else ["refund", "multitask", "examples", "long-position"]
    for name in names:
        request = fixtures / f"{name}.json"
        payload = json.loads(request.read_text())
        official = args.output / f"{name}.official.json"
        rust = args.output / f"{name}.rust.jsonl"
        subprocess.run(
            [sys.executable, "-m", "tools.oracle.gliner.trace_official", "--model", str(args.model_dir),
             "--request", str(request), "--trace", str(official), "--scalar"],
            check=True,
        )
        env = {**os.environ, "RMI_SCALAR": "1", "RMI_PARITY_TRACE": str(rust)}
        command = ["target/release-fast/rust-model-inference", "--model", str(args.gguf), "--jev",
                   "--gliner2-decide", "--jev-context", payload["text"], "--gliner2-schema",
                   json.dumps(payload["tasks"], ensure_ascii=False), "--jev-output", "json"]
        with (args.output / f"{name}.result.jsonl").open("w") as output:
            subprocess.run(command, env=env, stdout=output, check=True)
        compare(official, rust)
        print(f"{name}: scalar parity passed", flush=True)


if __name__ == "__main__":
    main()
