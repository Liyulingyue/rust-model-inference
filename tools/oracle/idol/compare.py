#!/usr/bin/env python3
"""Compare Idol's existing parity-trace JSONL + little-endian F32 files."""
import argparse
from array import array
import json
import math
from pathlib import Path
import sys
import tempfile


def records(path):
    path = Path(path)
    result = []
    seen = set()
    occurrences = {}
    for line in path.read_text().splitlines():
        record = json.loads(line)
        key = (record.get("name"), record.get("occurrence"))
        shape = record.get("shape")
        if key[1] != occurrences.get(key[0], 0):
            raise ValueError(f"Nonsequential checkpoint occurrence: {key}")
        if (not isinstance(key[0], str) or not key[0].startswith("idol.")
                or type(key[1]) is not int or key[1] < 0 or key in seen):
            raise ValueError(f"Invalid or duplicate checkpoint: {key}")
        if not isinstance(shape, list) or not shape or any(type(d) is not int or d <= 0 for d in shape):
            raise ValueError(f"Invalid checkpoint shape: {key} {shape}")
        size = math.prod(shape)
        if record.get("len", size) != size or record.get("finite") is not True:
            raise ValueError(f"Invalid length or nonfinite checkpoint: {key}")
        binary = Path(record["binary_path"])
        if not binary.is_absolute():
            binary = path.parent / binary
        if binary.stat().st_size != size * 4:
            raise ValueError(f"Binary length disagrees with shape: {key}")
        values = array("f")
        values.frombytes(binary.read_bytes())
        if sys.byteorder != "little":
            values.byteswap()
        if not all(math.isfinite(v) for v in values):
            raise ValueError(f"Nonfinite binary checkpoint: {key}")
        record["values"] = values
        result.append(record)
        seen.add(key)
        occurrences[key[0]] = key[1] + 1
    if not result:
        raise ValueError("Empty parity trace")
    return result


def metrics(reference, native):
    if len(reference) != len(native) or not reference:
        raise ValueError("Metric inputs must have the same positive length")
    differences = [float(a) - float(b) for a, b in zip(reference, native)]
    aa = math.fsum(float(v) ** 2 for v in reference)
    bb = math.fsum(float(v) ** 2 for v in native)
    rmse = math.sqrt(math.fsum(v * v for v in differences) / len(reference))
    cosine = math.fsum(float(a) * float(b) for a, b in zip(reference, native)) / math.sqrt(aa * bb) if aa and bb else float(aa == bb)
    return dict(max_abs=max(map(abs, differences)), rmse=rmse,
                relative_rmse=rmse / max(math.sqrt(aa / len(reference)), 1e-8),
                cosine=max(-1.0, min(1.0, cosine)))


def compare(oracle, native, gates):
    left, right = records(oracle), records(native)
    if len(left) != len(right):
        raise ValueError(f"Checkpoint counts differ: oracle={len(left)}, native={len(right)}")
    results = []
    for a, b in zip(left, right):
        identity = lambda r: (r["name"], r["occurrence"], r["shape"])
        if identity(a) != identity(b):
            raise ValueError(f"Checkpoint name/order/shape mismatch: {identity(a)} != {identity(b)}")
        gate = next((g for g in gates if g.get("name") == a["name"] or ("prefix" in g and a["name"].startswith(g["prefix"]))), None)
        if gate is None:
            raise ValueError(f"No predeclared gate for {a['name']}")
        values = metrics(a["values"], b["values"])
        absolute_limit = gate["atol"] + gate["rtol"] * max(map(abs, a["values"]))
        passed = (values["max_abs"] <= absolute_limit
                  and values["relative_rmse"] <= gate["relative_rmse"]
                  and values["cosine"] >= gate["cosine"])
        results.append(dict(name=a["name"], occurrence=a["occurrence"], shape=a["shape"],
                            **values, absolute_limit=absolute_limit, gate=gate, passed=passed))
    return results


def self_test():
    assert metrics([1., 2.], [1., 2.]) == dict(max_abs=0., rmse=0., relative_rmse=0., cosine=1.)
    assert metrics([0.], [0.])["cosine"] == 1.
    assert metrics([1.], [-1.])["cosine"] == -1.
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        raw = root / "value.f32"
        raw.write_bytes(bytes.fromhex("0000803f00000040"))
        checkpoint = dict(name="idol.test", occurrence=0, shape=[2], len=2, finite=True, binary_path=str(raw))
        left, right = root / "a.jsonl", root / "b.jsonl"
        left.write_text(json.dumps(checkpoint) + "\n")
        right.write_text(json.dumps(checkpoint) + "\n")
        gate = [dict(prefix="idol.", atol=0., rtol=0., relative_rmse=0., cosine=1.)]
        assert compare(left, right, gate)[0]["passed"]
        exact = [dict(name="idol.test", atol=0., rtol=0., relative_rmse=0., cosine=1.)] + gate
        assert compare(left, right, exact)[0]["gate"]["name"] == "idol.test"
        for change in [dict(shape=[1, 2]), dict(name="idol.other"), dict(finite=False), dict(shape=[3])]:
            right.write_text(json.dumps(checkpoint | change) + "\n")
            try:
                compare(left, right, gate)
            except ValueError:
                pass
            else:
                raise AssertionError(f"Mismatch was accepted: {change}")
        raw.write_bytes(bytes.fromhex("0000c07f00000040"))
        try:
            records(left)
        except ValueError:
            pass
        else:
            raise AssertionError("NaN was accepted")
    print("PASS: comparison self-checks (metrics, shape, ordering, finite binary)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("oracle", nargs="?")
    parser.add_argument("native", nargs="?")
    parser.add_argument("--gates", type=Path, default=Path(__file__).with_name("gates.json"))
    parser.add_argument("--report", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.oracle or not args.native:
        parser.error("oracle and native JSONL paths are required")
    results = compare(args.oracle, args.native, json.loads(args.gates.read_text())["gates"])
    if args.report:
        args.report.write_text(json.dumps(dict(passed=all(r["passed"] for r in results), checkpoints=results), indent=2) + "\n")
    failures = [r for r in results if not r["passed"]]
    for r in failures[:10]:
        print(f"FAIL {r['name']}: max_abs={r['max_abs']:.6g}, RMSE={r['rmse']:.6g}, relative_RMSE={r['relative_rmse']:.6g}, cosine={r['cosine']:.9g}")
    print(f"{'FAIL' if failures else 'PASS'}: {len(results)} checkpoints, {len(failures)} failed; finite/shape/name/order validated")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
