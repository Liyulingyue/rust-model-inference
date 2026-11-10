import copy
import unittest
from unittest.mock import patch

import numpy as np

from .compare import compare
from .compare_thinker import compare as compare_thinker


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.case = dict(prompt="p", input_ids=[1], positions=[[0, 0, 0, 0]],
                         markers=[0], logits=[[0., 1.]],
                         answers={"q": {"type": "choice", "probabilities": {"a": .4, "b": .6}}})

    def test_same_case_passes(self):
        self.assertTrue(compare([self.case], [self.case], "F32")["passed"])

    def test_equal_values_with_different_labels_fail(self):
        other = copy.deepcopy(self.case)
        other["answers"]["q"]["probabilities"] = {"b": .4, "a": .6}
        self.assertFalse(compare([other], [self.case], "F32")["passed"])

    def test_rmse_failure_is_not_hidden_by_probability_success(self):
        other = copy.deepcopy(self.case)
        other["logits"][0][0] += .05
        self.assertFalse(compare([other], [self.case], "Q8_0")["passed"])

    def test_a_block_format_can_miss_the_gate_and_still_be_usable(self):
        other = copy.deepcopy(self.case)
        other["logits"][0][0] += .05
        report = compare([other], [self.case], "Q8_0")
        # The acceptance gate is not relaxed ...
        self.assertFalse(report["passed"])
        # ... but the decision itself did not move.
        self.assertTrue(report["usability"]["decisions_identical"])
        self.assertEqual(report["usability"]["flipped_answers"], [])
        self.assertAlmostEqual(report["usability"]["max_probability_drift"], 0.0)
        self.assertTrue(report["usability"]["usable"])

    def test_a_flipped_label_is_reported_as_not_usable(self):
        other = copy.deepcopy(self.case)
        other["answers"]["q"]["probabilities"] = {"a": .55, "b": .45}
        report = compare([other], [self.case], "Q8_0")
        self.assertFalse(report["usability"]["decisions_identical"])
        self.assertEqual(report["usability"]["flipped_answers"], ["q"])
        self.assertFalse(report["usability"]["usable"])

    def test_probability_drift_beyond_the_usability_limit_is_reported(self):
        other = copy.deepcopy(self.case)
        other["answers"]["q"]["probabilities"] = {"a": .42, "b": .58}
        report = compare([other], [self.case], "Q8_0")
        self.assertTrue(report["passed"])          # inside the acceptance gate
        self.assertTrue(report["usability"]["decisions_identical"])
        self.assertAlmostEqual(report["usability"]["max_probability_drift"], .02, places=6)
        self.assertFalse(report["usability"]["usable"])
        # The usability limit is a declared parameter, not a fixed constant.
        relaxed = compare([other], [self.case], "Q8_0", usable_drift=.05)
        self.assertTrue(relaxed["usability"]["usable"])


class ThinkerComparisonTests(unittest.TestCase):
    def test_format_limits_and_discrete_contract(self):
        actual = dict(prompt='p', input_ids=[1], positions=[[0, 0, 0, 0]], generated_ids=[2],
                      steps=[dict(token_id=2, logits=np.zeros(248320))])
        expected = dict(actual, steps=[dict(token_id=2, logits_path='unused.npy')])
        actual['steps'][0]['logits'][0] = .05
        with patch('tools.oracle.thx02.compare_thinker.np.load', return_value=np.zeros(248320)):
            self.assertFalse(compare_thinker(actual, expected)['passed'])
            self.assertTrue(compare_thinker(actual, expected, 'BF16')['passed'])
            actual['steps'][0]['logits'].fill(.03)
            self.assertFalse(compare_thinker(actual, expected, 'BF16')['passed'])
            actual['steps'][0]['logits'].fill(0.)
            actual['generated_ids'] = [3]
            self.assertFalse(compare_thinker(actual, expected, 'BF16')['passed'])


if __name__ == "__main__":
    unittest.main()
