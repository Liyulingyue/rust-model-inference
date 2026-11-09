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
