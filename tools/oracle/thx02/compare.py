#!/usr/bin/env python3
"""Check exact encoding and the predeclared THX-02 per-format numerical tolerances."""
import argparse
import json
from pathlib import Path

import numpy as np


def items(path):
    value = json.loads(path.read_text())
    return value if isinstance(value, list) else [value]


def metrics(actual, expected):
    a, b = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    assert a.shape == b.shape and a.size and np.isfinite(a).all() and np.isfinite(b).all(), (a.shape, b.shape)
    error = a - b
    return {"max_abs": float(np.abs(error).max()), "rmse": float(np.sqrt(np.mean(error * error)))}


def probabilities(case):
    result = []
    for answer in case["answers"].values():
        result.extend(answer["probabilities"].values() if "probabilities" in answer else [answer["noul"]])
    return result


def answer_contract(case):
    return [(key, answer["type"], list(answer.get("probabilities", {})), answer.get("legend"))
            for key, answer in case["answers"].items()]


def top_label(answer):
    """The label a decision reads: the argmax of the answer probabilities.

    A "noul" answer carries a single number instead of a label set, so it has
    no argmax; only the numeric drift applies to it.
    """
    if "probabilities" not in answer:
        return None
    values = answer["probabilities"]
    if not values:
        return None
    return max(values, key=lambda label: values[label])


def probability_drift(actual, expected):
    """Largest absolute probability difference over the labels of one answer."""
    labels = set(actual.get("probabilities", {})) | set(expected.get("probabilities", {}))
    if not labels:
        drift = abs(float(actual.get("noul", 0.0)) - float(expected.get("noul", 0.0)))
        return drift
    return max(abs(float(actual["probabilities"].get(label, 0.0))
                   - float(expected["probabilities"].get(label, 0.0))) for label in labels)


def decision_report(actual, expected):
    """What an operator would notice, next to the numerical acceptance gates.

    The predeclared limits measure equivalence with the reference, not
    usability: a block-quantized format can miss the full-vocabulary logits
    gate while still returning the same label in the same order, which is
    what the caller actually sees. This report answers the separate question
    "does the model still make the same decision?", and it never relaxes the
    acceptance gate.
    """
    rows = []
    for a, b in zip(actual, expected):
        flipped, drift = [], 0.0
        for key, expected_answer in b["answers"].items():
            actual_answer = a["answers"].get(key)
            if actual_answer is None:
                flipped.append(key)
                continue
            expected_label = top_label(expected_answer)
            if expected_label is not None and top_label(actual_answer) != expected_label:
                flipped.append(key)
            drift = max(drift, probability_drift(actual_answer, expected_answer))
        rows.append({"flipped_answers": flipped, "decisions_identical": not flipped,
                     "max_probability_drift": drift})
    return rows


#: Usability criterion, deliberately separate from the acceptance gate: the
#: decisions must not flip and the probabilities may not move by more than
#: this. It answers "can this checkpoint be used", not "is this checkpoint
#: numerically equivalent to the reference".
USABLE_PROBABILITY_DRIFT = 0.01


def compare(actual, expected, fmt, usable_drift=USABLE_PROBABILITY_DRIFT):
    limits = (.02, .005, .005) if fmt == "F32" else (.04, .01, .01) if fmt == "F16" else (.10, .025, .025)
    assert len(actual) == len(expected), (len(actual), len(expected))
    rows = []
    for index, (a, b) in enumerate(zip(actual, expected)):
        exact = {key: a[key] == b[key] for key in ("prompt", "input_ids", "positions", "markers")}
        exact["answer_order_and_labels"] = answer_contract(a) == answer_contract(b)
        logits = metrics(a["logits"], b["logits"])
        probability = metrics(probabilities(a), probabilities(b))
        passed = all(exact.values()) and logits["max_abs"] <= limits[0] and logits["rmse"] <= limits[1] and probability["max_abs"] <= limits[2]
        rows.append({"case": index, "exact": exact, "logits": logits, "probabilities": probability,
                     "passed": passed, "actual_seconds": a.get("seconds"), "reference_seconds": b.get("joint_seconds", b.get("seconds")), "device_stats": a.get("device_stats")})
    decisions = decision_report(actual, expected)
    usability = {
        "decisions_identical": all(row["decisions_identical"] for row in decisions),
        "flipped_answers": sorted({key for row in decisions for key in row["flipped_answers"]}),
        "max_probability_drift": max((row["max_probability_drift"] for row in decisions), default=0.0),
        "usable_probability_drift": usable_drift,
    }
    usability["usable"] = usability["decisions_identical"] and usability["max_probability_drift"] <= usable_drift
    return {"format": fmt, "limits": dict(zip(("logits_max_abs", "logits_rmse", "probabilities_max_abs"), limits)),
            "passed": all(row["passed"] for row in rows), "cases": rows,
            "usability": usability, "decisions": decisions}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rust", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--format", required=True, choices=("F32", "F16", "BF16", "Q8_0", "Q4_0", "Q4_K", "Q5_K", "Q6_K"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--actual-cases", nargs="+", type=int, help="explicit actual indices when the reference ran a subset")
    parser.add_argument("--quality-reference", type=Path, help="report full-precision drift separately, without reusing implementation limits")
    parser.add_argument("--usable-probability-drift", type=float, default=USABLE_PROBABILITY_DRIFT,
                        help="usability criterion for the decision report (default: %(default)s); it does not change the acceptance gate")
    args = parser.parse_args()
    actual, expected = items(args.rust), items(args.oracle)
    if args.actual_cases is not None:
        actual = [actual[i] for i in args.actual_cases]
    report = compare(actual, expected, args.format, args.usable_probability_drift)
    report.update(actual=str(args.rust.resolve()), oracle=str(args.oracle.resolve()))
    report["actual_cases"] = args.actual_cases
    if args.quality_reference:
        quality = items(args.quality_reference)
        if args.actual_cases is not None:
            quality = [quality[i] for i in args.actual_cases]
        assert len(actual) == len(quality)
        report["full_precision_drift"] = [{"case": i, "logits": metrics(a["logits"], b["logits"]), "probabilities": metrics(probabilities(a), probabilities(b))} for i, (a, b) in enumerate(zip(actual, quality))]
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "format": args.format, "cases": len(actual),
                      "usable": report["usability"]["usable"],
                      "decisions_identical": report["usability"]["decisions_identical"],
                      "max_probability_drift": report["usability"]["max_probability_drift"],
                      "output": str(args.output)}, ensure_ascii=False))
    assert report["passed"], "THX-02 tolerance or discrete-contract failure; see report"


if __name__ == "__main__":
    main()
