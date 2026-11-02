"""Check every GGUF tensor's name/shape/type/payload against the pinned safetensors."""
import argparse
import mmap
from pathlib import Path

from tools.converter.mage_flow.convert_qwen3vl_vision import Reader
from tools.converter.mage_vl.convert_mage_vl import map_text_name
from tools.converter.utils.gguf import _read_gguf, GGML_BF16, GGML_F32, gguf_dims


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path)
    p.add_argument('text',type=Path)
    p.add_argument('vision',type=Path)
    p.add_argument('--text-type',choices=['bf16','f32'],required=True)
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
                widen = vision or a.text_type=='f32' or len(reader.shape(name))==1
                expected = reader.f32(name).tobytes() if widen else reader.raw(name)
                dtype,shape,size,offset = tensors[mapped]
                assert shape==gguf_dims(reader.shape(name)),name
                assert dtype==(GGML_F32 if widen else GGML_BF16),name
                assert data[offset:offset+size]==expected,name
                counts[idx] += 1
            assert len(tensors)==counts[idx]
    assert counts==[399,297],counts
    print('all 696 tensor shapes/types/payloads match source exactly',a.text_type)


if __name__=='__main__':
    main()
