import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import d3_oracle


class CausalContractTests(unittest.TestCase):
    def test_untied_checkpoint_without_lm_head_is_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'config.json').write_text(json.dumps({
                'tie_word_embeddings': True, 'text_config': {'tie_word_embeddings': False}}))
            (root / 'vela3_config.json').write_text(json.dumps({'definitions': {'writer': 'names'}}))
            args = SimpleNamespace(base=root, vela=root, causal=True, synthetic=False)
            with patch.object(d3_oracle, '_run_trace') as forward:
                with self.assertRaisesRegex(ValueError, 'requires tied embeddings'):
                    d3_oracle.run_trace(args)
                forward.assert_not_called()

    def test_tied_and_configured_grafted_heads_reach_the_trace(self):
        cases = [({'text_config': {'tie_word_embeddings': True}}, 'base'),
                 ({'tie_word_embeddings': True}, 'base'),
                 ({'text_config': {'tie_word_embeddings': False}}, 'grafted_head')]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for config, writer in cases:
                with self.subTest(config=config, writer=writer):
                    (root / 'config.json').write_text(json.dumps(config))
                    (root / 'vela3_config.json').write_text(json.dumps({'definitions': {'writer': writer}}))
                    args = SimpleNamespace(base=root, vela=root, causal=True, synthetic=False)
                    with patch.object(d3_oracle, '_run_trace', return_value='trace') as forward:
                        self.assertEqual(d3_oracle.run_trace(args), 'trace')
                        forward.assert_called_once()

    def test_synthetic_causal_fixture_needs_no_existing_checkpoint(self):
        with patch.object(d3_oracle, '_run_trace', return_value='fixture'):
            self.assertEqual(d3_oracle.run_trace(SimpleNamespace(causal=True, synthetic=True)), 'fixture')
