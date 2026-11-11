import unittest
import numpy as np
from tools.oracle.pjit.compare import metrics


class ComparisonTests(unittest.TestCase):
    def test_metrics_and_invalid_outputs(self):
        result = metrics([1., 2.], [1., 2.])
        self.assertEqual(result['max_abs'], 0.)
        self.assertAlmostEqual(result['cosine'], 1.)
        self.assertGreater(metrics([1., 3.], [1., 2.])['nrmse'], 0.)
        for actual, expected in [([], []), ([1.], [1., 2.]), ([np.nan], [1.])]:
            with self.assertRaises(ValueError):
                metrics(actual, expected)


if __name__ == '__main__':
    unittest.main()
