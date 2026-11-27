"""Check every GGUF tensor's name/shape/type/payload against the pinned safetensors."""
import argparse
import mmap
from pathlib import Path

from tools.converter.mage_flow.convert_qwen3vl_vision import Reader
from tools.converter.mage_vl.convert_mage_vl import TEXT_TYPES, encode_text_tensor, map_text_name
from tools.converter.utils.gguf import _read_gguf, GGML_F32, gguf_dims


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path)
    p.add_argument('text',type=Path)
    p.add_argument('vision',type=Path)
    p.add_argument('--text-type',choices=TEXT_TYPES,required=True)
    a = p.parse_args()
    reader = Reader(a.source)
    counts = [0,0]
    for idx,path in enumerate([a.text,a.vision]):
        metadata,tensors = _read_gguf(path)
        with path.open('rb') as source,mmap.mmap(source.fileno(),0,access=mmap.ACCESS_READ) as data:
            for name in reader.owners:
                vision = name.startswith('visual.')
                if int(vision)!=idx:
                    continue
                mapped = name if vision else map_text_name(name)
                expected_type, expected = ((GGML_F32, reader.f32(name).tobytes()) if vision
                                           else encode_text_tensor(reader, name, a.text_type))
                dtype,shape,size,offset = tensors[mapped]
                assert shape==gguf_dims(reader.shape(name)),name
                assert dtype==expected_type,name
                assert data[offset:offset+size]==expected,name
                counts[idx] += 1
            assert len(tensors)==counts[idx]
    assert counts==[399,297],counts
    print('all 696 tensor shapes/types/payloads match the pinned export',a.text_type)


if __name__=='__main__':
    main()
