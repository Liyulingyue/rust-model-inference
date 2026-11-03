"""Validate real source hashes, full tensor inventories and converted payloads."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from tools.converter.flux3.convert_flux3 import dit_shapes, text_shapes, vae_shapes, text_tensor_name
from tools.converter.utils.gguf import _read_gguf, open_safetensors

if not __debug__:
    raise RuntimeError('FLUX3 verification requires assertions (no -O/PYTHONOPTIMIZE)')

REVISION='337337621b8526e120f2275d250242f4a6bedd75'
HASHES={
    'model.safetensors':'c3e9813cdd049c831c836b2e9db61150e49a93fc1a4c63b256494642fee0bd7c',
    'ae.safetensors':'868fe7b343cc8f3a19dbcfcafbc3d5f888802be3f89bd81b65b3621a066ce8f3',
    'text_encoder/model-00001-of-00002.safetensors':'30a01a0556622645a3cce87b655bbbbbc1f170c196099f1b666c93202c3339a9',
    'text_encoder/model-00002-of-00002.safetensors':'046296a2a387efb43b0c997d5833c789604d168834f6e0d3064bf7bb13d002a6',
}
CONFIG_HASHES={
    'config.json':'c4fc638a9618e8ddc25f013c009be4d2aa9bd631d89dcf45149a377dd472c67d',
    'text_encoder/config.json':'edac7703329133edfc53e46ac0081835144c99d7eebf28b71c732694d435224d',
    'text_encoder/tokenizer.json':'a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7',
    'text_encoder/chat_template.json':'6f8a6a55027e3da5160105556cda5dd69f6423f1c32645f6730d32de7773d0c4',
}


def sha(path):
    with path.open('rb') as source:return hashlib.file_digest(source,'sha256').hexdigest()


def verify(root, output):
    report=dict(hf_revision=REVISION,sources={},components={})
    for name, expected in (HASHES | CONFIG_HASHES).items():
        path=root/name
        actual=sha(path)
        assert actual==expected, f'source SHA256 mismatch: {name}'
        report['sources'][name]=dict(sha256=actual,bytes=path.stat().st_size)
    for component, expected in [('dit',dit_shapes()),('text',text_shapes()),('vae',vae_shapes())]:
        path=output/f'flux3-{component}.gguf'
        metadata,directory=_read_gguf(path)
        assert metadata.get('general.architecture') == {'dit':'flux3','text':'qwen3vl','vae':'flux3_vae'}[component], component
        if component=='dit': assert metadata.get('flux3.flow_map') is True
        if component=='text':
            assert metadata.get('qwen3vl.rope.freq_base')==5000000.0
            assert metadata.get('qwen3vl.rope.dimension_sections')==[24,20,20,0]
            assert np.float32(metadata.get('qwen3vl.attention.layer_norm_rms_epsilon'))==np.float32(1e-6)
            tokenizer=json.loads((root/'text_encoder/tokenizer.json').read_text())
            assert metadata.get('tokenizer.ggml.merges')==tokenizer['model']['merges']
            assert metadata.get('tokenizer.ggml.normalizer.nfc') is True
            tokens=metadata['tokenizer.ggml.tokens']
            types=metadata['tokenizer.ggml.token_type']
            assert len(tokens)==len(types)==151936
            for piece,index in tokenizer['model']['vocab'].items(): assert tokens[index]==piece
            for entry in tokenizer['added_tokens']:
                assert tokens[entry['id']]==entry['content']
                assert types[entry['id']]==(3 if entry['special'] else 4)
        source_names=['model.safetensors'] if component=='dit' else ['ae.safetensors'] if component=='vae' else list(HASHES)[2:]
        inventory={}
        lossless=rounded=0
        error=norm=0.0
        with path.open('rb') as gguf:
            for source_name in source_names:
                tensors=open_safetensors(root/source_name)
                with tensors.path.open('rb') as source:
                    for original,info in tensors.header.items():
                        if original in ('__metadata__','bn.num_batches_tracked'):continue
                        name=text_tensor_name(original) if component=='text' else original
                        if name is None:continue
                        inventory[name]=tuple(info['shape'])
                        mapped=name.removeprefix('decoder.') if name.startswith('decoder.post_quant_conv.') else name
                        kind,dims,length,offset=directory[mapped]
                        assert tuple(dims)==tuple(info['shape'][::-1]), name
                        source.seek(tensors.data_offset+info['data_offsets'][0])
                        gguf.seek(offset)
                        if info['dtype']=='BF16' or len(info['shape'])!=4:
                            assert kind==(30 if info['dtype']=='BF16' else 0), name
                            remaining=length
                            while remaining:
                                size=min(4<<20,remaining)
                                a,b=source.read(size),gguf.read(size)
                                assert len(a)==size and a==b, f'lossless payload mismatch: {name}'
                                remaining-=size
                            lossless+=1
                        else:
                            assert kind==1 and info['dtype']=='F32',name
                            values=np.frombuffer(source.read(length*2),dtype='<f4')
                            raw=gguf.read(length)
                            assert raw==values.astype('<f2').tobytes(), f'F16 rounding mismatch: {name}'
                            converted=np.frombuffer(raw,dtype='<f2').astype(np.float64)
                            assert np.isfinite(converted).all(),name
                            error+=float(np.square(converted-values).sum())
                            norm+=float(np.square(values.astype(np.float64)).sum())
                            rounded+=1
        assert inventory==expected, f'{component} source inventory mismatch'
        assert len(directory)==len(expected), f'{component} GGUF inventory mismatch'
        report['components'][component]=dict(sha256=sha(path),bytes=path.stat().st_size,tensors=len(directory),lossless_payloads=lossless,f16_payloads=rounded,f16_relative_l2=(error/norm)**0.5 if norm else 0.0,architecture=metadata['general.architecture'])
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model_dir',type=Path)
    parser.add_argument('gguf_dir',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    report=verify(args.model_dir,args.gguf_dir)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print('PASS: source hashes, all tensor shapes/types, all payloads')


if __name__=='__main__':main()
