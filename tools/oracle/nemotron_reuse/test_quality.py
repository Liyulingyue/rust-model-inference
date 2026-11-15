import unittest

import numpy as np

from quality import paired_speech, waveform_error


class PairedSpeechTest(unittest.TestCase):
    def test_pairs_preserve_pcm16_length_seed_and_snr(self):
        source_rate = 22050
        t = np.arange(5 * source_rate) / source_rate
        speech = 0.2 * np.sin(2 * np.pi * 317 * t)
        for rate in [8000, 16000]:
            for snr in [0, 10]:
                clean, noisy = paired_speech(speech, source_rate, rate, snr, 17)
                self.assertEqual(clean.shape, (3 * rate,))
                np.testing.assert_array_equal(noisy, paired_speech(speech, source_rate, rate, snr, 17)[1])
                np.testing.assert_array_equal(noisy * 32768, (noisy * 32768).astype(np.int16))
                measured = 10 * np.log10(np.mean(clean ** 2) / np.mean((noisy - clean) ** 2))
                self.assertAlmostEqual(float(measured), snr, delta=0.01)
                self.assertLess(waveform_error(noisy, clean)['relative_rmse'], 1.01)


if __name__ == '__main__':
    unittest.main()
