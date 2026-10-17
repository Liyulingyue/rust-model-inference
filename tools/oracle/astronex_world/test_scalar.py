"""Small independent arithmetic contracts; no real weights or Rust code."""
import tempfile
from pathlib import Path
import unittest

import torch
from scalar import Scalar


class ScalarContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.scalar = Scalar(Path(cls.temp.name))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_sequential_reductions_and_convolution(self):
        with torch.inference_mode(), self.scalar:
            x = torch.tensor([[1e8,1.,-1e8]],dtype=torch.float32)
            self.assertEqual(torch.nn.functional.linear(x,torch.ones(1,3)).item(),0)
            self.assertEqual(x.sum().item(),0)
            self.assertEqual((x@torch.ones(3,1)).item(),0)
            x = x.reshape(1,3,1,1,1)
            out = torch.nn.functional.conv3d(x,torch.ones(1,3,1,1,1),torch.tensor([5.]))
            self.assertEqual(out.item(),5)
            self.assertEqual(torch.nn.functional.layer_norm(torch.tensor([[1.,3.]]),[2],eps=0).tolist(),[[-1.,1.]])
            self.assertEqual(torch.tensor([[3.,4.]]).norm(dim=-1).tolist(),[5.])

    def test_materialized_attention_and_mask(self):
        with torch.inference_mode(), self.scalar:
            q = torch.zeros(1,1,1,2)
            k = torch.zeros(1,1,2,2)
            v = torch.tensor([[[[2.,4.],[10.,20.]]]])
            a = self.scalar.attention(q,k,v)
            self.assertEqual(a.tolist(),[[[[6.,12.]]]])
            mask = torch.tensor([[[[True,False]]]])
            a = torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=mask)
            self.assertEqual(a.tolist(),[[[[2.,4.]]]])
            self.assertEqual(torch.softmax(torch.zeros(2,2),-1).tolist(),[[.5,.5],[.5,.5]])

    def test_intrinsic_f64_complex_rope(self):
        with torch.inference_mode(), self.scalar:
            x = torch.tensor([[1.,2.]],dtype=torch.float64)
            z = torch.view_as_complex(x)
            radius = torch.ones(1,dtype=torch.float64)
            rotated = z*torch.polar(radius,torch.zeros(1,dtype=torch.float64))
            self.assertEqual(torch.view_as_real(rotated).tolist(),[[1.,2.]])
            with self.assertRaisesRegex(ValueError, "F64 arithmetic is reserved"):
                torch.tensor([10000.],dtype=torch.float64)**torch.tensor([0.],dtype=torch.float64)

    def test_projective_four_by_four_transform(self):
        with torch.inference_mode(), self.scalar:
            matrix = torch.eye(4).reshape(1,1,4,4)
            matrix[0,0,0,3] = 2.0
            features = torch.zeros(1,1,1,1,1,4)
            features[...,3] = 1.0
            out = torch.einsum("bcij,bncpkj->bncpki", matrix, features)
            self.assertEqual(out.tolist(), [[[[[[2., 0., 0., 1.]]]]]])

    def test_fail_closed_and_no_lossy_casts(self):
        with torch.inference_mode(), self.scalar:
            x = torch.eye(2)
            with self.assertRaisesRegex(NotImplementedError,"Uncovered floating"):
                torch.linalg.det(x)
            with self.assertRaisesRegex(ValueError,"Forbidden lossy"):
                x.to(torch.bfloat16)
            with self.assertRaisesRegex(NotImplementedError,"floating factory"):
                torch.randn(1)


if __name__ == "__main__":
    unittest.main()
