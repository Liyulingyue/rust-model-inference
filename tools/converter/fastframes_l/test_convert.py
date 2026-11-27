import unittest
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from tools.converter.fastframes_l.convert_fastframes_l import mapped_tensors, encode_chunks, FORMATS

class ConversionTests(unittest.TestCase):
    def test_parallel_k_quantization_preserves_encoded_bytes(self):
        value = np.linspace(-2.,3.,3*256,dtype=np.float32).reshape(3,256)
        expected = b''.join(encode_chunks('matrix',value,'q4_k')[2])
        with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
            actual = b''.join(encode_chunks('matrix',value,'q4_k',pool)[2])
        self.assertEqual(actual,expected)

    def test_grouped_heads_are_transposed_and_split(self):
        value = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2,3,4)
        out = dict(mapped_tensors('kv_refiner', 'heads_grouped.gemms.0.dK_w', value))
        np.testing.assert_array_equal(out['refiner.heads.1.dK.weight'], value[1].T)
        bias = value[:, :1, :]
        out = dict(mapped_tensors('kv_refiner', 'heads_grouped.gemms.0.dK_b', bias))
        np.testing.assert_array_equal(out['refiner.heads.0.dK.bias'], bias[0,0])

    def test_patch_and_pointwise_flatten_but_depthwise_stays_5d(self):
        for component, name, shape in [('dit','patch_embedding.weight',(4,16,1,2,2)), ('pixel_decoder','pixel_head.entry_pw.weight',(4,8,1,1,1))]:
            out = list(mapped_tensors(component,name,np.ones(shape,dtype=np.float32)))
            self.assertEqual(out[0][1].shape,(shape[0],int(np.prod(shape[1:]))))
        out = list(mapped_tensors('pixel_decoder','pixel_head.conv_low.0.dw.weight',np.ones((4,1,3,3,3))))
        self.assertEqual(out[0][1].ndim,5)

    def test_formats_stream_exact_sizes_and_keep_vectors_f32(self):
        for fmt in FORMATS:
            kind, size, chunks = encode_chunks('matrix', np.ones((3,256),dtype=np.float32), fmt)
            self.assertEqual(sum(len(c) for c in chunks),size)
            self.assertEqual(kind, {'f32':0,'f16':1,'bf16':30,'q8_0':8,'q4_0':2,'q4_1':3,'q4_k':12,'q5_k':13,'q6_k':14}[fmt])
            kind, size, chunks = encode_chunks('bias',np.ones(5,dtype=np.float32),fmt)
            self.assertEqual((kind,size),(0,20))
            kind, size, chunks = encode_chunks('text.blocks.0.pos_embedding.embedding.weight', np.ones((32,64),dtype=np.float32),fmt)
            self.assertEqual((kind,size),(0,32*64*4))

if __name__ == '__main__': unittest.main()
