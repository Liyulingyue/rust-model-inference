import unittest

import torch

from .bundle import voice_tensors


class VoiceTests(unittest.TestCase):
    def fixture(self):
        return dict(speech_token=torch.tensor([[1, 2]], dtype=torch.int32),
                    speech_feat=torch.arange(320, dtype=torch.float32).reshape(1, 4, 80),
                    embedding=torch.ones(1, 192), hop=10)

    def test_voice_transposes_only_mel_and_preserves_discrete_ids(self):
        result = voice_tensors(self.fixture())
        self.assertEqual(result["mel"].shape, (80, 4))
        self.assertEqual(result["mel"][1].tolist(), [1, 81, 161, 241])
        self.assertEqual(result["tokens"].tolist(), [1, 2])

    def test_voice_rejects_bad_contracts(self):
        for key, value in [("hop", 1), ("embedding", torch.zeros(1, 192)),
                           ("speech_token", torch.tensor([[6561, 2]], dtype=torch.int32)),
                           ("speech_feat", torch.zeros(1, 3, 80))]:
            state = self.fixture()
            state[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                voice_tensors(state)


if __name__ == "__main__":
    unittest.main()
