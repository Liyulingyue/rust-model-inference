#!/usr/bin/env python3
"""Self-test for check_moe.py using synthetic manifests.

Worth doing before the one real run: the script is the thing that decides
whether the MoE matches, and a script that reports PASS on broken input is
worse than no script. Covers identical, differing values, reordered routing,
and a name present on one side only.
"""

import json
import os
import struct
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
CHECK = os.path.join(HERE, "check_moe.py")


def write_trace(tmp, tag, records):
    """records: [(name, layer, step, [floats])] -> manifest path"""
    manifest = os.path.join(tmp, tag + ".jsonl")
    with open(manifest, "w", encoding="utf-8") as fh:
        for name, layer, step, values in records:
            binary = os.path.join(tmp, "%s.%s.%s.%s.f32" % (tag, name, layer, step))
            with open(binary, "wb") as bf:
                bf.write(struct.pack("<%df" % len(values), *values))
            fh.write(json.dumps({
                "name": name, "layer": layer, "step": step,
                "shape": [len(values)], "len": len(values),
                "binary_path": binary,
            }) + "\n")
    return manifest


def run(oracle, rust):
    proc = subprocess.run(
        [sys.executable, CHECK, oracle, rust],
        capture_output=True, text=True,
    )
    return proc.returncode, proc.stdout + proc.stderr


CASES = []


def case(name):
    def wrap(fn):
        CASES.append((name, fn))
        return fn
    return wrap


@case("identical values -> exit 0 and EXACT")
def _(tmp):
    recs = [("ffn_moe_logits", 0, 0, [1.0, 2.0, 3.0]),
            ("ffn_moe_topk", 0, 0, [67.0, 140.0, 206.0, 3.0])]
    rc, out = run(write_trace(tmp, "a", recs), write_trace(tmp, "b", recs))
    assert rc == 0, out
    assert "bit-identical" in out, out
    assert "IDENTICAL" in out, out


@case("one value differs -> exit 1 and reports the index")
def _(tmp):
    a = [("ffn_moe_logits", 0, 0, [1.0, 2.0, 3.0])]
    b = [("ffn_moe_logits", 0, 0, [1.0, 2.5, 3.0])]
    rc, out = run(write_trace(tmp, "a", a), write_trace(tmp, "b", b))
    assert rc == 1, out
    assert "DIFF" in out, out
    assert "[1]" in out, out


@case("routing same set but reordered -> reported, not silently passed")
def _(tmp):
    a = [("ffn_moe_topk", 0, 0, [67.0, 140.0, 206.0])]
    b = [("ffn_moe_topk", 0, 0, [140.0, 67.0, 206.0])]
    rc, out = run(write_trace(tmp, "a", a), write_trace(tmp, "b", b))
    assert rc == 1, out
    assert "SAME SET, DIFFERENT ORDER" in out, out


@case("different experts -> reported")
def _(tmp):
    a = [("ffn_moe_topk", 0, 0, [1.0, 2.0])]
    b = [("ffn_moe_topk", 0, 0, [1.0, 9.0])]
    rc, out = run(write_trace(tmp, "a", a), write_trace(tmp, "b", b))
    assert rc == 1, out
    assert "DIFFERENT EXPERTS" in out, out


@case("1-ulp difference is still DIFF, not a tolerance pass")
def _(tmp):
    base = 0.1
    nudged = struct.unpack("<f", struct.pack("<I", bits(base) + 1))[0]
    a = [("ffn_moe_probs", 0, 0, [base])]
    b = [("ffn_moe_probs", 0, 0, [nudged])]
    rc, out = run(write_trace(tmp, "a", a), write_trace(tmp, "b", b))
    assert rc == 1, out
    assert "DIFF" in out, out


@case("rust-only name is surfaced, not skipped")
def _(tmp):
    a = [("ffn_moe_logits", 0, 0, [1.0])]
    b = [("ffn_moe_logits", 0, 0, [1.0]), ("ffn_moe_input", 0, 0, [1.0, 2.0])]
    rc, out = run(write_trace(tmp, "a", a), write_trace(tmp, "b", b))
    assert rc == 0, out
    reported = [line for line in out.splitlines()
                if "rust only" in line and "ffn_moe_input" in line]
    assert reported, out
    assert "known Rust-only" in reported[0], out


@case("oracle-only name is surfaced")
def _(tmp):
    a = [("ffn_moe_logits", 0, 0, [1.0]), ("ffn_moe_weights_scaled", 0, 0, [1.0])]
    b = [("ffn_moe_logits", 0, 0, [1.0])]
    rc, out = run(write_trace(tmp, "a", a), write_trace(tmp, "b", b))
    reported = [line for line in out.splitlines()
                if "oracle only" in line and "ffn_moe_weights_scaled" in line]
    assert reported, out


@case("layer suffix in the name still joins a bare oracle name")
def _(tmp):
    # The oracle writes bare names plus a layer field; the Rust forward.rs
    # embeds the layer in the name. Without normalisation this pair matches
    # nothing and the checkpoint looks absent from the oracle.
    a = [("attn_post_norm", 3, 0, [1.0, 2.0])]
    b = [("attn_post_norm-3", 3, 0, [1.0, 2.0])]
    rc, out = run(write_trace(tmp, "a", a), write_trace(tmp, "b", b))
    assert rc == 0, out
    assert "joined: 1" in out, out


@case("null step joins step 0")
def _(tmp):
    # checkpoint_rows writes "step": null; the oracle writes 0 for one token.
    a = [("l_out", 0, 0, [1.0])]
    b = [("l_out-0", 0, None, [1.0])]
    rc, out = run(write_trace(tmp, "a", a), write_trace(tmp, "b", b))
    assert rc == 0, out
    assert "joined: 1" in out, out


@case("length mismatch is reported, not truncated away")
def _(tmp):
    a = [("ffn_moe_logits", 0, 0, [1.0, 2.0, 3.0])]
    b = [("ffn_moe_logits", 0, 0, [1.0])]
    rc, out = run(write_trace(tmp, "a", a), write_trace(tmp, "b", b))
    assert rc == 1, out
    assert "LENGTH" in out, out


def bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def write_trace_vocab(tmp, tag, logits):
    """A result_output-shaped trace with a chosen logit vector."""
    return write_trace(tmp, tag, [("result_output", None, None, logits)])


# -- token-agreement verdicts -----------------------------------------------
#
# A greedy token can agree while the two implementations disagree by more than
# the top-2 margin. That is not a pass, so the verdict has three states and the
# marginal one has to be its own.

def _token_case(name, oracle_logits, rust_logits, expect):
    def body(tmp):
        rc, out = run(write_trace_vocab(tmp, "a", oracle_logits),
                      write_trace_vocab(tmp, "b", rust_logits))
        assert expect in out, out
        if expect == "SAME TOKEN (margin":
            assert "only just" not in out, out
        if expect == "SAME TOKEN, but only just":
            assert rc == 1, out
    CASES.append((name, body))


_token_case(
    "clear argmax win is a pass",
    [1.0, 5.0, 4.0], [1.0, 5.02, 3.9], "SAME TOKEN (margin")

_token_case(
    "agreeing token inside the noise band is not a pass",
    [1.0, 5.0, 4.98], [1.0, 5.4, 4.5], "SAME TOKEN, but only just")

_token_case(
    "flipped argmax is a failure",
    [1.0, 5.0, 4.0], [1.0, 4.0, 5.0], "DIFFERENT TOKEN")


def main():
    failed = 0
    for name, fn in CASES:
        with tempfile.TemporaryDirectory() as tmp:
            try:
                fn(tmp)
                print("  ok    %s" % name)
            except AssertionError as err:
                failed += 1
                print("  FAIL  %s" % name)
                for line in str(err).splitlines()[:12]:
                    print("        %s" % line)
    print()
    print("%d/%d cases passed" % (len(CASES) - failed, len(CASES)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
