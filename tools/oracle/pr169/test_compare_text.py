import json
from pathlib import Path
import struct
import tempfile
import unittest

from tools.oracle.pr169.compare_text import compare


class CompareTests(unittest.TestCase):
    def test_complete_trace_and_reject_missing_corrupt_or_divergent_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace, log = root / 'rust.jsonl', root / 'oracle.log'
            log.write_text('RMI_PROMPT [1,2]\nRMI_GREEDY 1\nRMI_GREEDY 1\n')
            (root / 'index.tsv').write_text('0000\tresult_output\t2, 1, 1, 1\n0001\tresult_output\t2, 1, 1, 1\n')
            for index in range(2):
                (root / f'{index:04}.f32').write_bytes(struct.pack('<ff', 1.0, 2.0))
            rows = [dict(name='prompt_ids', token_ids=[1, 2])]
            rows += [dict(name='result_output', shape=[2], len=2,
                          binary_path=str(root / '0000.f32')) for _ in range(3)]
            trace.write_text('\n'.join(map(json.dumps, rows)))
            self.assertEqual(compare(trace, root, log)['generated_ids'], [1, 1])
            with self.assertRaisesRegex(AssertionError, 'teacher-forced history'):
                compare(trace, root, log, teacher_forced=True)
            forced = dict(name='teacher_forced_ids', token_ids=[1, 2, 1])
            trace.write_text('\n'.join(map(json.dumps, rows + [forced])))
            self.assertTrue(compare(trace, root, log, teacher_forced=True)['teacher_forced'])
            trace.write_text('\n'.join(map(json.dumps, rows[:-1])))
            with self.assertRaisesRegex(AssertionError, 'checkpoint count'):
                compare(trace, root, log)
            trace.write_text('\n'.join(map(json.dumps, rows)))
            for payload, message in [(b'', 'payload'), (struct.pack('<ff', float('nan'), 2.0), 'nonfinite'),
                                     (struct.pack('<ff', 3.0, 2.0), 'greedy')]:
                (root / '0001.f32').write_bytes(payload)
                with self.assertRaisesRegex(AssertionError, message):
                    compare(trace, root, log)
            (root / '0001.f32').write_bytes(struct.pack('<ff', 1.0, 3.0))
            with self.assertRaisesRegex(AssertionError, 'relative L2'):
                compare(trace, root, log)


if __name__ == '__main__':
    unittest.main()
