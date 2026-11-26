#!/usr/bin/env python3
"""Diff this repository's GGUF tokenizer against the official `tokenizer.json`.

The GGUF carries only a BPE vocabulary plus the name of a pre-tokenization
regex (`tokenizer.ggml.pre`); the `tokenizer.json` shipped with the original
weights is the authority that llama.cpp's regex was transcribed from, and
HuggingFace `tokenizers` implements it directly. Diffing against it is a
stricter check than diffing token IDs against one llama.cpp build, and it needs
no weights, no compiler and no network.

Prerequisites
-------------
    pip install tokenizers            # into models/.venv, never the system Python

Usage
-----
    python3 tools/oracle/hy_mt3/tokenizer_parity.py <repo-root> <case-dir>

`<case-dir>` must contain, per model tag:

    <tag>.json    the official tokenizer.json
    <tag>.gguf    a tokenizer-only GGUF (metadata section, zero tensors)

The tokenizer-only GGUF keeps the check independent of the (multi-GB) weights.
It can be rebuilt from a full GGUF header with `tools/oracle/hy_mt3/head_to_tokenizer_gguf.py`.
The Rust side is driven through `cargo run --example tokenize_text`, which
prints the IDs for a prompt file exactly as the CLI would encode them.
"""
import json
import os
import pathlib
import subprocess
import sys
import tempfile

# Deliberately broad: digits, CJK/kana/Hangul, combining marks, emoji,
# contractions, every ASCII punctuation class, leading/trailing whitespace and
# the two empty-ish edge cases. Each one has caught a real tokenizer bug in this
# repository at some point, so they stay.
CASES = [
    ("raw_prompt", "将以下文本翻译为英语，注意只需要输出翻译后的结果，不要额外解释：\n\n机器学习是人工智能的一个分支。"),
    ("digits", "价格是12345元，订单号6124c78e。"),
    ("digits_long", "1234567890 和 007 以及 3.14159"),
    ("mixed", "Hello, 世界! iPhone 15 Pro 的价格是 ¥7,999.00 —— 含税吗？"),
    ("chat_control", "<｜hy_User｜>你好<｜hy_Assistant｜>"),
    ("combining", "e\u0301 a\u0300 n\u0303 café naïve"),
    ("emoji", "🚀🫠 emoji 混合 中文 ✨"),
    ("whitespace", "  leading\n\nnewlines\t\ttabs   spaces  "),
    ("contractions", "It's a test, don't we've they'll I'm he'd"),
    ("punctuation", "a!b?c,d.e:f;g'h\"i(j)k[l]m{n}o<p>q@r#s$t%u^v&w*x|y~z"),
    ("cjk_kana", "日本語のテスト と 中文测试 と 한글"),
    ("empty", ""),
    ("single_char", "a"),
    ("newline_only", "\n"),
]

TAGS = ("1.8B", "7B", "30B")


def hf_ids(path: pathlib.Path, text: str):
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(path)).encode(text, add_special_tokens=False).ids


def rust_ids(repo: pathlib.Path, gguf: pathlib.Path, text: str):
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as handle:
        handle.write(text)
        prompt_path = handle.name
    try:
        out = subprocess.run(
            [
                "cargo",
                "run",
                "-q",
                "--release",
                "--example",
                "tokenize_text",
                "--",
                str(gguf),
                prompt_path,
                "0",  # add_special
                "1",  # parse_special
            ],
            cwd=repo,
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )
    finally:
        pathlib.Path(prompt_path).unlink(missing_ok=True)
    if out.returncode != 0:
        raise SystemExit(f"rust tokenizer failed:\n{out.stderr[-2000:]}")
    return json.loads(out.stdout)["ids"]


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    repo = pathlib.Path(sys.argv[1]).resolve()
    case_dir = pathlib.Path(sys.argv[2]).resolve()
    failures = 0
    for tag in TAGS:
        gguf = case_dir / f"{tag}.gguf"
        spec = case_dir / f"{tag}.json"
        if not gguf.exists() or not spec.exists():
            print(f"== {tag}: SKIP (missing {gguf.name} or {spec.name})")
            continue
        print(f"== {tag} ==")
        for name, text in CASES:
            expected = hf_ids(spec, text)
            actual = rust_ids(repo, gguf, text)
            if expected != actual:
                failures += 1
                print(f"  DIFF {name:<14} tokenizer.json={expected}")
                print(f"       {'':<14} rust            ={actual}")
            else:
                print(f"  OK   {name:<14} n={len(expected)}")
    print(f"\n{'PASS' if failures == 0 else f'FAIL ({failures} case(s))'}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
