#!/usr/bin/env python3
"""Check that the recorded Idol validation reports name the geometry they cover.

The tolerances in the reports were measured on the diagnostic fixture of
``oracle.py``, which runs on a 64 pixel image, and the released model runs
on a 1024 pixel image. A report that says ``passed`` without naming its
geometry reads as a statement about the released model, so every report has
to carry the geometry it was measured on.

The check runs over the reports that are checked into the repository, so a
report that loses its geometry fails the build.
"""
import json
from pathlib import Path
import unittest

# The reports that state a measurement, and the geometry it was measured on.
REPORTS = {
    "numerical-validation.json": {"kind": "diagnostic", "image_side": 64},
    "device-validation.json": {"kind": "diagnostic", "image_side": 64},
}

ROOT = Path(__file__).parent


def check(path: Path, expected: dict) -> list:
    """Returns the problems of one report, an empty list when it is fine."""
    if not path.exists():
        return [f"{path.name}: missing"]
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return [f"{path.name}: unreadable: {error}"]
    if not isinstance(report, dict):
        return [f"{path.name}: the report is not an object"]
    geometry = report.get("geometry")
    if not isinstance(geometry, dict):
        return [f"{path.name}: the report does not name the geometry it covers"]
    problems = []
    for key, value in expected.items():
        if geometry.get(key) != value:
            problems.append(
                f"{path.name}: geometry.{key} is {geometry.get(key)!r}, expected {value!r}"
            )
    note = geometry.get("note")
    if not isinstance(note, str) or "production geometry" not in note:
        problems.append(
            f"{path.name}: geometry.note does not say that the production geometry is not covered"
        )
    return problems


class ReportGeometryTest(unittest.TestCase):
    def test_the_checked_in_reports_name_their_geometry(self):
        problems = []
        for name, expected in REPORTS.items():
            problems.extend(check(ROOT / name, expected))
        self.assertEqual(problems, [])

    def test_a_report_without_a_geometry_is_rejected(self):
        with self.subTest("a report that does not name its geometry"):
            problems = check(
                ROOT / "numerical-validation.json",
                REPORTS["numerical-validation.json"],
            )
            self.assertEqual(problems, [])

            stripped = ROOT / "numerical-validation.json"
            report = json.loads(stripped.read_text(encoding="utf-8"))
            report.pop("geometry")
            temporary = stripped.with_suffix(".stripped.json")
            temporary.write_text(json.dumps(report), encoding="utf-8")
            try:
                problems = check(
                    temporary, REPORTS["numerical-validation.json"]
                )
                self.assertEqual(len(problems), 1)
                self.assertIn("does not name the geometry", problems[0])
            finally:
                temporary.unlink()

    def test_a_report_with_the_wrong_geometry_is_rejected(self):
        stripped = ROOT / "numerical-validation.json"
        report = json.loads(stripped.read_text(encoding="utf-8"))
        report["geometry"]["image_side"] = 1024
        temporary = stripped.with_suffix(".wrong.json")
        temporary.write_text(json.dumps(report), encoding="utf-8")
        try:
            problems = check(temporary, REPORTS["numerical-validation.json"])
            self.assertEqual(len(problems), 1)
            self.assertIn("image_side", problems[0])
        finally:
            temporary.unlink()


if __name__ == "__main__":
    unittest.main()
