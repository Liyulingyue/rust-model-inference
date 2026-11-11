from pathlib import Path
import json,numpy as np,torch
from gguf import GGMLQuantizationType
from gguf.quants import dequantize
from tools.converter.utils.gguf import _read_gguf
from tools.oracle.arktts.same_weight import prepared_linears
torch.set_num_threads(4)
import argparse
parser = argparse.ArgumentParser(description="Export independently decoded same-input ArkTTS projection fixtures")
parser.add_argument("model_dir", type=Path, help="multiq directory containing per-format GGUF pairs")
root=parser.parse_args().model_dir.resolve()
for fmt in ('q8_0','q4_0','q4_1','bf16','f32','f16','q4_k','q5_k','q6_k'):
 rows=[]
 for component,names in [('', ['layers.0.attention.wqkv.weight','layers.0.feed_forward.w2.weight','fast_output.weight']),('codec-', ['quantizer.post_module.layers.0.attention.wqkv.weight','quantizer.post_module.layers.0.feed_forward.w2.weight','quantizer.upsample.0.1.pwconv1.weight'])]:
  path=root/fmt/f'Audio8-TTS-Realistic-Air-{component}{fmt.upper()}.gguf'
  meta,ts=_read_gguf(path)
  with path.open('rb') as source:
   for name in names:
    kind,dims,length,offset=ts[name];source.seek(offset);raw=source.read(length)
    W=dequantize(np.frombuffer(raw,dtype=np.uint8),GGMLQuantizationType(kind)).reshape(tuple(reversed(dims))).copy()
    weight=torch.from_numpy(W);minimum=None
    if kind==3:
     blocks=np.frombuffer(raw,dtype=np.uint8).reshape(dims[1],-1,20)
     minimum=torch.from_numpy(np.ascontiguousarray(blocks[:,:,2:4]).view('<f2')[:,:,0].astype(np.float32))
    X=np.random.default_rng(728).normal(size=(3,dims[0])).astype(np.float32)*np.float32(.25)
    with prepared_linears({weight.data_ptr():(kind,minimum)}):
     Y=torch.nn.functional.linear(torch.from_numpy(X),weight).numpy()
    rows.append(dict(gguf=str(path),tensor=name,n_in=dims[0],input=X.reshape(-1).tolist(),output=Y.reshape(-1).tolist(),relative_limit=.00005,embedding=W[1].tolist()))
 (root/fmt/'projections.json').write_text(json.dumps(rows))
 print(fmt,flush=True)
