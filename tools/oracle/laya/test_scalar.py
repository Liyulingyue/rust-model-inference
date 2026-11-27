"""Small scalar-contract check, without model weights or a test framework."""
import tempfile
from pathlib import Path
import unittest

import torch
from scalar import Scalar


class ScalarContract(unittest.TestCase):
    def test_sequential_f32_and_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            scalar = Scalar(Path(tmp))
            with torch.inference_mode(), scalar:
                # A widened or reassociated dot gives 1 instead of 0.
                x = torch.tensor([[1e8, 1, -1e8]], dtype=torch.float32)
                self.assertEqual(torch.nn.functional.linear(x, torch.ones(1,3)).item(), 0)
                x = torch.tensor([[1.,3.]])
                y = torch.nn.functional.layer_norm(x, [2], torch.ones(2), eps=0)
                self.assertEqual(y.tolist(), [[-1.,1.]])
                self.assertEqual(torch.softmax(torch.zeros(1,2), -1).tolist(), [[.5,.5]])
                q = torch.zeros(1,1,1,2)
                k = torch.zeros(1,1,2,2)
                v = torch.tensor([[[[2.,4.],[1e8,1e8]]]])
                mask = torch.tensor([[[[True, False]]]])
                y = torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=mask)
                self.assertEqual(y.tolist(), [[[[2.,4.]]]])
                with self.assertRaisesRegex(NotImplementedError, "Uncovered floating"):
                    torch.linalg.vector_norm(x)


if __name__ == "__main__":
    unittest.main()
