#!/usr/bin/env python3
"""Compare the qwen35moe oracle trace against the Rust RMI_PARITY_TRACE.

Both sides write JSONL manifests whose records carry `name`, `layer`, `step`
and `binary_path` to a raw little-endian f32 file. This joins the two on
(name, layer, step) and reports, per checkpoint, whether the raw bits are
identical, and if not how far apart they are.

Three things this deliberately does:

* It reports names present on only one side instead of skipping them. A silently
  ignored checkpoint is a hole that looks like a pass.
* `ffn_moe_topk` is compared as integers and also checked for equal set *and*
  order. "The same experts were selected" has to mean the same ids in the same
  order; matching a multiset while the order differs would still change the
  accumulation.
* It prints a verdict, not just numbers. The point is to know whether a
  divergence is real, not to eyeball a table.

Usage:
    python3 check_moe.py $ORACLE_MANIFEST $RUST_MANIFEST
"""

import json
import math
import struct
import sys
from collections import defaultdict

# Checkpoints whose whole purpose is a decision rather than a value.
DECISION_NAMES = {"ffn_moe_topk"}

# The Rust side emits this and the oracle has no counterpart: there is no
# callback node for the block input. Naming it here keeps it from looking like
# an unexplained missing file.
RUST_ONLY_KNOWN = {"ffn_moe_input"}


def normalise_layer(layer):
    """`-1` and absent mean the same thing: a checkpoint with no layer.

    The oracle writes -1 for whole-model nodes (result_norm, result_output);
    `parity_trace::checkpoint` writes null. Joining them as-is matches nothing,
    which again looks like the checkpoint is missing rather than spelled
    differently.
    """
    return -1 if layer is None else layer


def normalise(name, layer):
    """Strip a trailing "-<layer>" from a checkpoint name.

    The two trace writers disagree on where the layer goes. The oracle writes
    ggml's own `cb(tensor, "name", il)` convention: bare name, layer in its own
    field. `src/models/qwen35/trunk/trace.rs` follows that, but the older
    checkpoints in `forward.rs` embed it -- `attn_post_norm-3`. Joining the raw
    names silently matches nothing, which reads as "the oracle has no such
    checkpoint" rather than as a spelling difference, so normalise instead.

    Only strips when the suffix equals the record's own layer, so a name that
    genuinely ends in digits is left alone.
    """
    if layer is None:
        return name
    suffix = "-%d" % layer
    if name.endswith(suffix):
        return name[: -len(suffix)]
    return name


def load_manifest(path):
    """-> {(name, layer, step): record}

    `step` is normalised to 0 when absent. `parity_trace::checkpoint_rows`
    writes `"step": null` because it takes no step argument, while the oracle
    and `trace::layer_1d` write 0 for a single-token run. Those are the same
    position. Normalising rather than skipping is safe here because the
    duplicate-key check below runs *after* normalisation: if collapsing two
    records onto one key would lose a distinction, this raises instead of
    silently picking a winner.
    """
    out = {}
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as err:
                raise SystemExit("%s:%d is not JSON: %s" % (path, lineno, err))
            if "binary_path" not in rec:
                continue  # metadata-only record (prompt ids, layer flags)
            step = rec.get("step")
            key = (normalise(rec["name"], rec.get("layer")),
                   normalise_layer(rec.get("layer")), 0 if step is None else step)
            if key in out:
                raise SystemExit(
                    "%s:%d duplicate key %r. Two records for one (name, layer, step) "
                    "means one side recorded the same checkpoint twice; joining on it "
                    "would be ambiguous." % (path, lineno, key)
                )
            out[key] = rec
    return out


def load_f32(rec):
    with open(rec["binary_path"], "rb") as fh:
        raw = fh.read()
    count = len(raw) // 4
    return list(struct.unpack("<%df" % count, raw[: count * 4]))


def bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def compare(name, lhs, rhs):
    if len(lhs) != len(rhs):
        return {
            "status": "LENGTH",
            "detail": "oracle %d values, rust %d" % (len(lhs), len(rhs)),
        }
    differing = 0
    max_abs = 0.0
    max_rel = 0.0
    first = None
    for i, (a, b) in enumerate(zip(lhs, rhs)):
        if bits(a) != bits(b):
            differing += 1
            delta = abs(a - b) if math.isfinite(a) and math.isfinite(b) else float("inf")
            if delta > max_abs:
                max_abs = delta
                scale = max(abs(a), abs(b))
                if scale > 0:
                    max_rel = max(max_rel, delta / scale)
            if first is None:
                first = (i, a, b)
    return {
        "status": "EXACT" if differing == 0 else "DIFF",
        "differing": differing,
        "len": len(lhs),
        "max_abs": max_abs,
        "max_rel": max_rel,
        "first": first,
    }


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__.strip())
    oracle = load_manifest(sys.argv[1])
    rust = load_manifest(sys.argv[2])

    keys = sorted(set(oracle) & set(rust), key=lambda k: (k[0], k[1] if k[1] is not None else -1))
    only_oracle = sorted(set(oracle) - set(rust))
    only_rust = sorted(set(rust) - set(oracle))

    print("oracle records: %d, rust records: %d, joined: %d"
          % (len(oracle), len(rust), len(keys)))
    print()

    failures = []
    by_name = defaultdict(list)

    for key in keys:
        name = key[0]
        result = compare(name, load_f32(oracle[key]), load_f32(rust[key]))
        by_name[name].append((key, result))
        if result["status"] == "EXACT":
            print("  %-30s layer=%-3s step=%-3s EXACT  (%d values)"
                  % (name, key[1], key[2], result["len"]))
        elif result["status"] == "LENGTH":
            print("  %-30s layer=%-3s step=%-3s LENGTH %s"
                  % (name, key[1], key[2], result["detail"]))
            failures.append((key, result))
        else:
            idx, a, b = result["first"]
            print("  %-30s layer=%-3s step=%-3s DIFF   %d/%d values, max_abs=%.3e max_rel=%.3e"
                  % (name, key[1], key[2], result["differing"], result["len"],
                     result["max_abs"], result["max_rel"]))
            print("      first: [%d] oracle=%.9g rust=%.9g" % (idx, a, b))
            failures.append((key, result))

    print()
    print("== routing decision ==")
    for key in keys:
        if key[0] not in DECISION_NAMES:
            continue
        ids = [int(v) for v in load_f32(oracle[key])]
        mine = [int(v) for v in load_f32(rust[key])]
        same_order = ids == mine
        same_set = sorted(ids) == sorted(mine)
        verdict = "IDENTICAL" if same_order else (
            "SAME SET, DIFFERENT ORDER" if same_set else "DIFFERENT EXPERTS")
        print("  layer=%-3s %s" % (key[1], verdict))
        print("      oracle: %s" % ids)
        print("      rust  : %s" % mine)
        if not same_order:
            failures.append((key, {"status": "ROUTING"}))

    if ("result_output", -1, 0) in keys:
        oracle_logits = load_f32(oracle[("result_output", -1, 0)])
        rust_logits = load_f32(rust[("result_output", -1, 0)])
        oracle_order = sorted(range(len(oracle_logits)), key=lambda i: -oracle_logits[i])
        rust_order = sorted(range(len(rust_logits)), key=lambda i: -rust_logits[i])
        oracle_token = oracle_order[0]
        rust_token = rust_order[0]
        agree = oracle_token == rust_token
        # The margin, not just the agreement. "Same token" on one prompt is one
        # data point; a top-1/top-2 gap far larger than the disagreement in the
        # logits is what says the answer is not a coin flip.
        oracle_margin = oracle_logits[oracle_order[0]] - oracle_logits[oracle_order[1]]
        rust_margin = rust_logits[rust_order[0]] - rust_logits[rust_order[1]]
        spread = max(abs(a - b) for a, b in zip(oracle_logits, rust_logits))
        print("== does it emit the same token? ==")
        print("  oracle argmax %d (%.6f), runner-up %d (%.6f), margin %.4f"
              % (oracle_token, oracle_logits[oracle_token],
                 oracle_order[1], oracle_logits[oracle_order[1]], oracle_margin))
        print("  rust   argmax %d (%.6f), runner-up %d (%.6f), margin %.4f"
              % (rust_token, rust_logits[rust_token],
                 rust_order[1], rust_logits[rust_order[1]], rust_margin))
        print("  worst logit disagreement %.4f, vs an oracle margin of %.4f"
              % (spread, oracle_margin))
        if not agree:
            print("  DIFFERENT TOKEN")
            failures.append((("result_output", -1, 0), {"status": "TOKEN"}))
        elif spread > oracle_margin:
            # Same argmax, but the top-2 are closer together than the two
            # implementations disagree. The agreement is then a coincidence of
            # this prompt, not a property of the model: a neighbouring input
            # would flip it. Reporting this as a plain pass is the failure mode
            # -- "7/7 SAME TOKEN" reads as verified and means the opposite.
            print("  SAME TOKEN, but only just -- the disagreement is %.1fx the"
                  " top-2 margin, so the winner is not decided by the model"
                  % (spread / oracle_margin))
            failures.append((("result_output", -1, 0), {"status": "TOKEN_MARGINAL"}))
        else:
            print("  SAME TOKEN (margin %.4f is %.1fx the disagreement)"
                  % (oracle_margin, oracle_margin / spread))
        print()

    if only_oracle or only_rust:
        for label, keys in (("oracle only", only_oracle), ("rust only", only_rust)):
            if not keys:
                continue
            by_name = defaultdict(list)
            for key in keys:
                by_name[key[0]].append(key)
            for name in sorted(by_name):
                layers = sorted({(k[1] if k[1] is not None else -1) for k in by_name[name]})
                known = " (known Rust-only)" if name in RUST_ONLY_KNOWN else ""
                # Summarise rather than list: the Rust side runs all 40 layers
                # while the oracle dumps a few, so a per-record dump is hundreds
                # of lines that bury the joined results.
                print("  %-12s %-30s %d record(s), layers %s%s"
                      % (label, name, len(by_name[name]),
                         ("%d..%d" % (layers[0], layers[-1])) if len(layers) > 3
                         else str(layers), known))

    print()
    if failures:
        print("VERDICT: %d checkpoint(s) differ." % len(failures))
        return 1
    print("VERDICT: all joined checkpoints are bit-identical.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
