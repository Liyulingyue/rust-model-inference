"""CLI dispatch for real-weight thinker validation across exported precisions."""
import contextlib
import io
from pathlib import Path
import unittest
from unittest.mock import call, patch

from tools.oracle.thx02 import verify_thinking


class ThinkerValidationCliTests(unittest.TestCase):
    def invoke(self, *arguments):
        with patch('sys.argv', ['verify_thinking', '--model-dir', '.', *arguments]), \
             patch.object(verify_thinking, 'check', return_value={'passed': True}) as check, \
             contextlib.redirect_stdout(io.StringIO()):
            verify_thinking.main()
        return check

    def test_default_preserves_f16_and_three_backends(self):
        check = self.invoke()
        self.assertEqual(check.call_args_list, [
            call(Path.cwd(), (Path('target/release-fast')).resolve(), backend, 'F16')
            for backend in ('cpu', 'auto', 'vulkan')
        ])

    def test_requested_precisions_reach_each_requested_backend(self):
        check = self.invoke('--formats', 'BF16', 'Q8_0', '--computes', 'cpu', 'vulkan')
        self.assertEqual(check.call_args_list, [
            call(Path.cwd(), (Path('target/release-fast')).resolve(), backend, storage)
            for storage in ('BF16', 'Q8_0') for backend in ('cpu', 'vulkan')
        ])

    def test_unknown_precision_is_rejected_before_inference(self):
        errors = io.StringIO()
        with contextlib.redirect_stderr(errors), self.assertRaises(SystemExit) as error:
            self.invoke('--formats', 'Q4_1')
        self.assertEqual(error.exception.code, 2)
        self.assertIn('invalid choice', errors.getvalue())


if __name__ == '__main__':
    unittest.main()
