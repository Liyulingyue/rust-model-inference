"""Pinned official P-JiT forward, with hooks; run only in the remote repository venv.

No source in the Oracle checkout is modified. Optional GGUF reload compares the
same stored weights, independently decoded by gguf-py, against PyTorch.
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

PIN='95403933f3727abe947977c75915a53292e1d841'

def main():
    import torch
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--oracle',type=Path,required=True);p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--qwen',type=Path)
    p.add_argument('--gguf',type=Path);p.add_argument('--width',type=int,default=33);p.add_argument('--height',type=int,default=35)
    p.add_argument('--activation-quant',action='store_true',help='independent Q8 activation QDQ matching shared prepared CPU kernels')
    p.add_argument('--text-f32',action='store_true');p.add_argument('--device',default='cuda');p.add_argument('--native-bf16',action='store_true')
    p.add_argument('--prompt',default='A red cube on a white table.');p.add_argument('--fixture',type=Path);p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--repeat',type=int,default=1);p.add_argument('--no-checkpoints',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if not 1<=a.repeat<=100:raise ValueError('repeat must be 1..100')
    import subprocess
    revision=subprocess.check_output(['git','-C',str(a.oracle),'rev-parse','HEAD'],text=True).strip()
    if revision!=PIN:raise ValueError(f'wrong Oracle commit {revision}')
    sys.path.insert(0,str(a.oracle.resolve()))
    from pjit_21k import PyramidJiT,set_attention_backend,QwenTextEncoder
    set_attention_backend('sdpa');torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=False
    if a.fixture:
        spec=json.loads((a.fixture/'input.json').read_text());a.width=spec['width'];a.height=spec['height']
        image=torch.from_numpy(np.fromfile(a.fixture/'image.f32',dtype='<f4').copy()).reshape(1,3,1,a.height,a.width).to(a.device)
        text=torch.from_numpy(np.fromfile(a.fixture/'text.f32',dtype='<f4').copy()).reshape(1,-1,7680).to(a.device)
    else:
        image=torch.randn((1,3,1,a.height,a.width),generator=torch.Generator().manual_seed(42))*4
        image=image.to(torch.bfloat16).float().to(a.device)
        if a.qwen:
            encoder=QwenTextEncoder(str(a.qwen),device=torch.device(a.device))
            if a.text_f32:encoder.model.float()
            text=encoder([a.prompt])[0].float()[None]
            formatted=encoder.tokenizer.apply_chat_template([{'role':'user','content':a.prompt}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
            ids=encoder.tokenizer(formatted, truncation=True, max_length=256)['input_ids']
            (a.output/'tokens.json').write_text(json.dumps(ids))
            del encoder;torch.cuda.empty_cache()
        else:text=torch.randn((1,7,7680),generator=torch.Generator().manual_seed(43)).to(a.device)*0.2
        image.cpu().numpy().astype('<f4').tofile(a.output/'image.f32')
        text.cpu().numpy().astype('<f4').tofile(a.output/'text.f32')
        (a.output/'input.json').write_text(json.dumps(dict(width=a.width,height=a.height,timestep=750)))
    if a.prepare_only:return
    length=text.shape[1];img_rows=math.ceil(a.width/32)*math.ceil(a.height/32)
    padded=torch.zeros((1,256,7680),device=a.device);padded[:,:length]=text
    model=PyramidJiT.from_pretrained(str(a.weights),device=a.device)
    if a.gguf:
        import gguf
        from gguf.quants import dequantize
        reader=gguf.GGUFReader(str(a.gguf));state=model.state_dict()
        if len(reader.tensors)!=len(state) or {t.name for t in reader.tensors}!=set(state):
            raise ValueError('GGUF must contain the exact released DiT parameter inventory')
        with torch.no_grad():
            for tensor in reader.tensors:
                values=dequantize(tensor.data,tensor.tensor_type).copy()
                state[tensor.name].copy_(torch.from_numpy(values).reshape(state[tensor.name].shape))
        if a.activation_quant:
            from tools.oracle.thx02.oracle import activation_roundtrip
            def quantize(kind):
                def hook(module,args):
                    value=args[0];values=value.detach().float().cpu().numpy()
                    if isinstance(module,torch.nn.Conv3d) and module.kernel_size==(1,1,1):
                        values=values.transpose(0,2,3,4,1)
                        values=activation_roundtrip(values,2 if kind==3 else kind).transpose(0,4,1,2,3)
                    elif isinstance(module,torch.nn.Conv3d):
                        b,c,f,h,w=values.shape
                        patches=values.reshape(b,c,f,h//32,32,w//32,32).transpose(0,2,3,5,1,4,6).reshape(-1,3072)
                        patches=activation_roundtrip(patches,2 if kind==3 else kind)
                        values=patches.reshape(b,f,h//32,w//32,c,32,32).transpose(0,4,1,2,5,3,6).reshape(b,c,f,h,w)
                    else:values=activation_roundtrip(values,2 if kind==3 else kind)
                    return (torch.from_numpy(np.ascontiguousarray(values)).to(value.device),*args[1:])
                return hook
            kinds={t.name:int(t.tensor_type) for t in reader.tensors}
            for name,module in model.named_modules():
                kind=kinds.get(name+'.weight')
                if isinstance(module,(torch.nn.Linear,torch.nn.Conv3d)) and kind in (2,3,8,12,13,14):
                    module.register_forward_pre_hook(quantize(kind))
    def save(name,values):
        values.detach().float().cpu().contiguous().numpy().astype('<f4').tofile(a.output/(name+'.f32'))
    def hook(name,rows):
        def emit(module,args,value):save(name,value[0,:rows])
        return emit
    def patch_hook(module,args,value):save('pjit.patch',value.flatten(2).transpose(1,2)[0])
    if not a.no_checkpoints:
        for stack,count,rows in [('image_refiner',2,img_rows),('text_refiner',2,length),('blocks',22,img_rows+length)]:
            for i in range(count):getattr(model,stack)[i].register_forward_hook(hook(f'{stack}.{i}',rows))
        model.patch_embed.register_forward_hook(patch_hook)
    lengths=torch.tensor([length],device=a.device,dtype=torch.int32)
    timestep=torch.tensor([spec['timestep'] if a.fixture else 750],device=a.device)
    times=[]
    with torch.inference_mode(),torch.autocast(device_type=a.device,dtype=torch.bfloat16,enabled=a.native_bf16):
        for _ in range(a.repeat):
            torch.cuda.synchronize() if a.device=='cuda' else None
            started=time.perf_counter()
            output=model(image,padded,lengths,timestep)
            torch.cuda.synchronize() if a.device=='cuda' else None
            times.append(time.perf_counter()-started)
    save('pjit.prediction',output[0,:,0])
    report=dict(oracle_commit=revision,device=a.device,seconds=times[-1],seconds_samples=times,checkpoints=not a.no_checkpoints,native_bf16=a.native_bf16,width=a.width,height=a.height,text_tokens=length,gguf=str(a.gguf) if a.gguf else None,activation_quant=a.activation_quant)
    (a.output/'run.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))

if __name__=='__main__':main()
