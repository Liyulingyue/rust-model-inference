import json
from pathlib import Path
import struct
import tempfile
import unittest

from tools.oracle.holo4.compare import compare


class CompareTests(unittest.TestCase):
    def test_graph_checks_embedding_bits_shapes_counts_and_numerical_errors(self):
        from tools.oracle.holo4.compare import compare_graph
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows, index = [], []
            names = ['model.input_embed', 'l_out-0', 'l_out-3', 'l_out-39', 'result_norm', 'result_output']
            for step in range(3):
                payload = root / f'row{step}.f32'
                payload.write_bytes(struct.pack('<ff', 1.0 + step, 2.0 + step))
                for name in names:
                    number = f'{len(index):04}'
                    (root / f'{number}.f32').write_bytes(payload.read_bytes())
                    index.append([number, name, '2, 1, 1, 1'])
                    if name == 'model.input_embed':
                        if step == 0:
                            rust_path = root / 'input.rust'
                            rust_path.write_bytes(struct.pack('<ffff', 1.0, 2.0, 2.0, 3.0))
                            rows.append(dict(name='qwen35.embedding', shape=[2, 2], len=4, binary_path=str(rust_path)))
                    else:
                        rows.append(dict(name=name, shape=[2] if name in ['result_norm', 'result_output'] else [1, 2],
                                         len=2, binary_path=str(payload)))
            self.assertTrue(compare_graph(rows, index, root, 2, 2, 0.03)['model.input_embed'][0]['bitwise_equal'])
            (root / 'index.tsv').write_text('\n'.join('\t'.join(entry) for entry in index))
            trace, log = root / 'rust.jsonl', root / 'oracle.log'
            log.write_text('RMI_PROMPT [1,2]\nRMI_GREEDY 1\nRMI_GREEDY 1\n')
            joined = rows + [dict(name='qwen35.prompt_ids', token_ids=[1, 2]), dict(name='qwen35.greedy_token_ids', token_ids=[1, 1])]
            trace.write_text('\n'.join(map(json.dumps, joined)))
            self.assertIn('checkpoints', compare(trace, root, log, graph=True))
            with self.assertRaisesRegex(AssertionError, 'count'):
                compare_graph(rows[:-1], index, root, 2, 2, 0.03)
            rows[-2]['shape'] = [3]
            with self.assertRaisesRegex(AssertionError, 'shape'):
                compare_graph(rows, index, root, 2, 2, 0.03)
            rows[-2]['shape'] = [2]
            layer = next(row for row in rows if row['name'] == 'l_out-0')
            layer['shape'] = [2, 1]
            with self.assertRaisesRegex(AssertionError, 'shape'):
                compare_graph(rows, index, root, 2, 2, 0.03)
            layer['shape'] = [1, 2]
            payload = Path(rows[0]['binary_path'])
            payload.write_bytes(struct.pack('<Ifff', 0x3f800001, 2.0, 1.0, 2.0))
            with self.assertRaisesRegex(AssertionError, 'embedding bits'):
                compare_graph(rows, index, root, 2, 2, 0.03)

    def test_complete_trace_and_reject_missing_corrupt_or_divergent_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace, log = root / 'rust.jsonl', root / 'oracle.log'
            log.write_text('RMI_PROMPT [1,2]\nRMI_GREEDY 1\nRMI_GREEDY 1\n')
            (root / 'index.tsv').write_text('0000\tresult_output\t2, 1, 1, 1\n0001\tresult_output\t2, 1, 1, 1\n0002\tresult_output\t2, 1, 1, 1\n')
            for index in range(3):
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
