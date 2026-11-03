from pathlib import Path
import tempfile
import unittest

from tools.oracle.timestral.benchmark import backend_evidence, cases, compare


class BenchmarkTest(unittest.TestCase):
    def test_forecasts_and_workload_shapes(self):
        forecast = dict(quantile_levels=[0.5], quantiles=[[[1.0]]])
        self.assertTrue(compare(forecast, forecast)['passed'])
        self.assertFalse(compare(forecast, dict(forecast, quantiles=[[[2.0]]]))['passed'])
        with self.assertRaises(AssertionError):
            compare(forecast, dict(forecast, quantiles=[[[float('nan')]]]))
        inputs = cases()
        self.assertEqual(len(inputs['max_context']['context']), 32768)
        self.assertEqual([len(row) for row in inputs['multivariate8']['context']], [512] * 8)
        self.assertEqual([len(row) for row in inputs['covariates']['known_covariates']], [608] * 2)

    def test_gpu_evidence_rejects_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / 'stderr.log'
            log.write_text('compute scope=timestral.transformer backend=Vulkan\n')
            self.assertEqual(backend_evidence(log, 'auto')['backends'], ['Vulkan'])
            for text in ('', 'compute scope=timestral.transformer backend=Cpu\n'):
                log.write_text(text)
                with self.assertRaises(AssertionError):
                    backend_evidence(log, 'vulkan')


if __name__ == '__main__':
    unittest.main()
