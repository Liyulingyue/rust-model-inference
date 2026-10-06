import json
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from tools.converter.mage_flow.convert_mage_flow import _expected_shapes, _vae_shapes, _validate_tensors, _add_tensor, VARIANTS
from tools.converter.utils.gguf import GgufWriter, open_safetensors, read_gguf_tensor_bytes, read_gguf_directory

class MageFlowConverterTest(unittest.TestCase):
    def test_raw_bf16_payload_survives_export(self):
        with tempfile.TemporaryDirectory() as raw:
            root=Path(raw);raw_bits=struct.pack('<6H',0x3f80,0xbf80,0x8000,0x0001,0x3eab,0x7f7f)
            header=json.dumps({'w':{'dtype':'BF16','shape':[2,3],'data_offsets':[0,12]}}).encode()
            source=root/'source.safetensors';source.write_bytes(struct.pack('<Q',len(header))+header+raw_bits)
            source=open_safetensors(source);_validate_tensors(source,{'w':(2,3)})
            out=root/'out.gguf';writer=GgufWriter(out);_add_tensor(writer,source,'w');writer.write()
            self.assertEqual(read_gguf_tensor_bytes(out,'w'),raw_bits)
            self.assertEqual(read_gguf_directory(out)[1]['w'][:2],(30,(3,2)))
    def test_rejects_lossy_wrong_shape_overlapping_and_truncated_weights(self):
        shapes={'a':(2,), 'b':(2,)}
        h={'a':{'dtype':'BF16','shape':[2],'data_offsets':[0,4]},'b':{'dtype':'BF16','shape':[2],'data_offsets':[4,8]}}
        source=SimpleNamespace(header=h,data_offset=8,file_size=16)
        self.assertEqual(_validate_tensors(source,shapes),shapes)
        for key,value,pattern in [('dtype','F16','must remain BF16'),('shape',[1,2],'expected shape'),('data_offsets',[2,6],'overlapping')]:
            old=h['b'][key];h['b'][key]=value
            with self.assertRaisesRegex(ValueError,pattern):_validate_tensors(source,shapes)
            h['b'][key]=old
        source.file_size=15
        with self.assertRaisesRegex(ValueError,'truncated'):_validate_tensors(source,shapes)
    def test_released_contracts_and_variants(self):
        self.assertEqual(len(_expected_shapes()),397)
        self.assertEqual(len(_vae_shapes()),728)
        self.assertEqual(set(VARIANTS),{'base','flow','turbo','edit-base','edit','edit-turbo'})

if __name__=='__main__':unittest.main()
