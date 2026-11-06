"""Small contract checks for the response comparator; no model dependencies."""
import unittest

from compare_requests import compare


class ResponseComparisonTests(unittest.TestCase):
    def test_float_tolerance_does_not_hide_schema_or_offset_changes(self):
        expected = {"score": 0.5, "start": 4}
        self.assertTrue(compare(expected, {"score": 0.50001, "start": 4}, 1e-4, 0)["passed"])
        self.assertFalse(compare(expected, {"score": 0.5, "start": 5}, 1, 1)["passed"])
        self.assertFalse(compare(expected, {"score": 0.5}, 1, 1)["passed"])

    def test_booleans_and_nonfinite_values_are_not_floating_results(self):
        for actual in [True, float("nan"), float("inf")]:
            self.assertFalse(compare({"score": 1.0}, {"score": actual}, 0, 0)["passed"])
        self.assertFalse(compare({"score": float("nan")}, {"score": 1.0}, 0, 0)["passed"])

    def test_error_codes_are_exact_but_diagnostic_text_can_differ(self):
        expected = {"error": "invalid_question", "message": "reference"}
        self.assertTrue(compare(expected, {**expected, "message": "native"}, 0, 0)["passed"])
        self.assertFalse(compare(expected, {**expected, "error": "other"}, 0, 0)["passed"])


if __name__ == "__main__":
    unittest.main()
