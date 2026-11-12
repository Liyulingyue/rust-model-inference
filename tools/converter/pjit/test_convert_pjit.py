import unittest
from pathlib import Path
import subprocess
import tempfile
from unittest.mock import patch
from tools.converter.pjit.convert_pjit import shapes, validate_config, tensor_layout
from tools.converter.pjit.convert_text import convert, LLAMA_CPP_COMMIT

class ContractTests(unittest.TestCase):
    def test_inventory_and_convolution_layout(self):
        expected = shapes()
        self.assertEqual(sum(__import__('math').prod(s) for s in expected.values()), 2184091606)
        self.assertEqual(tensor_layout('patch_embed.proj1.weight', expected['patch_embed.proj1.weight']), (3072, 1024))
        self.assertEqual(tensor_layout('head.modulation', (1, 2, 2944)), (5888,))
        self.assertEqual(expected['blocks.21.ffn.w13.weight'], (10752, 2944))
    def test_rejects_wrong_config(self):
        with self.assertRaisesRegex(ValueError, 'architecture'):
            validate_config({'architecture': {'dim': 1}})

    def test_text_export_does_not_publish_failed_output(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / 'text.gguf'
            def fail(command, **kwargs):
                Path(command[-1]).write_bytes(b'partial')
                raise subprocess.CalledProcessError(1, command)
            with patch('subprocess.check_output', return_value=LLAMA_CPP_COMMIT), patch('subprocess.run', side_effect=fail):
                with self.assertRaises(subprocess.CalledProcessError):
                    convert(root, output, root)
            self.assertEqual(list(Path(root).iterdir()), [])

    def test_text_export_never_overwrites_racing_output(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / 'text.gguf'
            def race(command, **kwargs):
                Path(command[-1]).write_bytes(b'new')
                output.write_bytes(b'existing')
            with patch('subprocess.check_output', return_value=LLAMA_CPP_COMMIT), patch('subprocess.run', side_effect=race):
                with self.assertRaises(FileExistsError):
                    convert(root, output, root)
            self.assertEqual(output.read_bytes(), b'existing')

if __name__ == '__main__': unittest.main()
