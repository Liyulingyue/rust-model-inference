"""Pinned official FastFrames oracle, with explicit backend/arithmetic controls.

The published recipe hardcodes NVIDIA cuDNN SDPA. On AMD, backend-only mode
retains BF16 casts but uses PyTorch's available SDPA backend. Controlled mode widens effective
weights, activations and cache to F32, matching RMI's documented compute recipe.
No pinned source file is edited.
"""
import argparse
import importlib
import importlib.metadata
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoTokenizer
from tools.oracle.fastframes_l.streaming_weights import StreamingWeights
import ftfy
import html


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--model-dir',type=Path,required=True)
    parser.add_argument('--text-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gguf',type=Path,help='independent GGUF dequantization oracle for the effective exported weights')
    parser.add_argument('--noise',type=Path,help='shared initial + three transition noise tensors, raw LE F32')
    parser.add_argument('--width',type=int,default=16)
    parser.add_argument('--height',type=int,default=16)
    parser.add_argument('--frames',type=int,default=5)
    parser.add_argument('--prompt',default='A red ball rolls on a table.')
    parser.add_argument('--seed',type=int,default=17)
    parser.add_argument('--threads',type=int,default=24)
    parser.add_argument('--recipe',choices=['controlled','published'],default='controlled')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--benchmark',action='store_true',help='disable checkpoint copies and writes for timing')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    def sha256(path):
        digest=hashlib.sha256()
        with Path(path).open('rb') as stream:
            for chunk in iter(lambda:stream.read(8*1024*1024),b''):digest.update(chunk)
        return digest.hexdigest()
    wrappers={name:sha256(Path(__file__).parent/name) for name in ['run_reference.py','streaming_weights.py','effective_weights.py']}
    pin=json.loads((Path(__file__).parent/'reference-files.json').read_text())
    for name,expected in pin['sha256'].items():
        if sha256(args.reference/name)!=expected:raise ValueError(f'oracle source differs from pin: {name}')
    for name,expected in pin['tokenizer_sha256'].items():
        if sha256(args.text_dir/'google/umt5-xxl'/name)!=expected:raise ValueError(f'tokenizer asset differs from pin: {name}')
    contract=json.loads((Path(__file__).resolve().parents[3]/'src/models/diffusion/fastframes_l/contract.json').read_text())
    hashes={}
    for directory,section in [(args.model_dir,'source'),(args.text_dir,'text_source')]:
        for name,expected in contract[section]['hashes'].items():
            digest=sha256(directory/name)
            if digest!=expected:raise ValueError(f'checkpoint differs from pin: {name}')
            hashes[name]=digest
    gguf_hash=sha256(args.gguf) if args.gguf else None
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision('highest')
    sys.path.insert(0,str(args.reference.resolve()))
    from fastframes.wan.causal_model import CausalWanModel,FastOps,allocate_kv_cache
    from fastframes.wan.model import WAN_CONFIGS
    from fastframes.wan.t5 import umt5_xxl_encoder
    from fastframes.models.kv_refinement import KVRefinementNetwork
    from fastframes.models.pixel_decoder import BridgedPixelDecoder,align_frames
    from fastframes.wan.scheduler import FlowMatchScheduler
    controlled=args.recipe=='controlled'
    dtype=torch.float32 if controlled else torch.bfloat16
    device=torch.device(args.device)
    attention_module=importlib.import_module('fastframes.wan.attention')
    causal_module=importlib.import_module('fastframes.wan.causal_model')
    model_module=importlib.import_module('fastframes.wan.model')
    refiner_module=importlib.import_module('fastframes.models.kv_refinement')
    def sdpa(q,k,v,causal=False,dropout_p=0.,**kwargs):
        out_dtype=q.dtype
        q,k,v=[u.transpose(1,2).to(dtype) for u in (q,k,v)]
        return F.scaled_dot_product_attention(q,k,v,is_causal=causal,dropout_p=dropout_p).transpose(1,2).contiguous().to(out_dtype)
    attention_module.attention=causal_module.attention=model_module.attention=sdpa
    model_module.cross_attention=lambda q,k,v:sdpa(q,k,v)
    refiner_module._sdpa=lambda q,k,v:F.scaled_dot_product_attention(q,k,v)
    # Meta construction avoids random initialization of 21B parameters.
    with torch.device('meta'):
        dit=CausalWanModel(WAN_CONFIGS['Wan2.1-T2V-14B'])
        refiner=KVRefinementNetwork(dim=5120,num_heads=40,head_dim=128,num_blocks=40)
        decoder=BridgedPixelDecoder()
        text=umt5_xxl_encoder(device='meta')
    states=[load_file(args.model_dir/'fastframes-l/dit.safetensors'),load_file(args.model_dir/'fastframes-l/kv_refiner.safetensors'),load_file(args.model_dir/'fastframes-l/pixel_decoder.safetensors'),torch.load(args.text_dir/'models_t5_umt5-xxl-enc-bf16.pth',map_location='cpu',mmap=True,weights_only=True)]
    modules=[dit,refiner,decoder,text]
    weights=StreamingWeights(states,device,dtype,args.gguf)
    weights.initialize(modules)
    formats=weights.formats
    tokenizer=AutoTokenizer.from_pretrained(args.text_dir/'google/umt5-xxl',local_files_only=True)
    prompt=' '.join(html.unescape(html.unescape(ftfy.fix_text(args.prompt))).split())
    encoding=tokenizer([prompt],padding='max_length',truncation=True,max_length=512,return_tensors='pt')
    ids=encoding.input_ids.to(device);mask=encoding.attention_mask.to(device)
    count=int(mask.sum())
    (args.output/'tokens.json').write_text(json.dumps(ids[0,:count].tolist())+'\n')
    records=[];occurrences={}
    def save(name,value):
        if args.benchmark:return
        value=value.detach().float().cpu().contiguous().numpy()
        occurrence=occurrences.get(name,0);occurrences[name]=occurrence+1
        path=args.output/f'{name}.{occurrence}.f32';value.tofile(path)
        records.append({'name':name,'occurrence':occurrence,'shape':list(value.shape),'file':path.name})
    for i,block in enumerate(text.blocks):
        block.register_forward_hook(lambda m,a,v,i=i:save(f'fastframes.text.{i}',v[:,:count]))
    for i,block in enumerate(dit.blocks):
        block.register_forward_hook(lambda m,a,v,i=i:save(f'fastframes.dit.{i}',v))
    for name in ['conv_low.0','conv_low.1','conv_high.0']:
        block=decoder.pixel_head.get_submodule(name)
        block.register_forward_hook(lambda m,a,v,name=name:save('decoder.pixel_head.'+name,v.permute(0,2,3,4,1)))
    decoder.pixel_head.fc_out.register_forward_hook(lambda m,a,v:save('fastframes.pixel_tokens',v))
    h,w=args.height//8,args.width//8
    frames=(args.frames+6)//4
    fast=FastOps(dit,h,w,frames)
    if formats:
        from tools.oracle.fastframes_l.effective_weights import install_prepared_activations
        weights.refresh=install_prepared_activations(modules,fast,formats,weights.tensors)
    weights.attach_blocks(dit,text,fast)
    scheduler=FlowMatchScheduler(shift=5.)
    generator=torch.Generator(device='cpu').manual_seed(args.seed)
    shape=(1,frames,16,h,w)
    if args.noise:
        values=np.fromfile(args.noise,dtype='<f4')
        if values.size!=4*np.prod(shape) or not np.isfinite(values).all():raise ValueError('invalid shared noise shape/values')
        noise=torch.from_numpy(values.reshape(4,*shape))
    else:
        noise=torch.randn((4,*shape),generator=generator,dtype=torch.float32)
        if not controlled:noise=noise.to(torch.bfloat16).float()
    noise.numpy().tofile(args.output/'noise.f32')
    noise=noise.to(device=device,dtype=dtype)
    if device.type=='cuda':torch.cuda.synchronize()
    start=time.perf_counter()
    autocast=torch.autocast(device.type,dtype=torch.bfloat16,enabled=not controlled)
    with torch.inference_mode(),autocast:
        embeds=text(ids,mask);embeds=embeds[:,:count]
        padded=F.pad(embeds,(0,0,0,512-count));save('fastframes.text',padded)
        save('fastframes.noise',noise)
        tc=weights.text_cache(dit,embeds)
        cache=allocate_kv_cache(dit,frames,h*w//4,dtype=dtype,device=device)
        times={}
        for t in [1000.,937.,833.,625.,0.]:
            ts=torch.tensor([[t]],device=device)
            times[t]=(ts,*dit.time_conditioning(ts,dtype))
        features=[];mods=[]
        def predict(x,t,frame):
            ts,e0,eh=times[t]
            save('fastframes.patch',fast.patch_embed(x))
            flow,feature=dit.forward_frame(x,e0,eh,tc,cache,frame,fast=fast)
            save('fastframes.flow',flow)
            return scheduler.flow_to_x0(flow.flatten(0,1),x.flatten(0,1),ts.flatten()).unflatten(0,(1,1)),feature,eh
        for frame in range(frames):
            schedule=[1000.,937.,833.,625.] if frame==0 else [1000.,833.]
            x=noise[0,:,frame:frame+1]
            for step,t in enumerate(schedule):
                x0,feature,eh=predict(x,t,frame);save('fastframes.x0',x0)
                if step+1<len(schedule):
                    next_t=torch.tensor([schedule[step+1]],device=device)
                    x=scheduler.add_noise(x0.flatten(0,1),noise[step+1,:,frame],next_t).unflatten(0,(1,1))
            features.append(feature);mods.append(eh)
            if frame==0: predict(x0,0.,frame)
            else:
                emb=refiner.embed_timestep(833.,1,device,dtype)
                stem=fast.patch_embed(x0) if controlled else dit.patch_embedding(x0.permute(0,2,1,3,4).contiguous()).flatten(2).transpose(1,2)
                phi=refiner.compute_phi(stem,emb)
                # Production trace is after the head's phi projection.
                save('fastframes.refiner.phi',refiner.heads_grouped.phi_proj['768'](phi))
                begin,end=frame*h*w//4,(frame+1)*h*w//4
                nk=torch.stack([c['k'][:,begin:end].reshape(h*w//4,5120) for c in cache]);nv=torch.stack([c['v'][:,begin:end].reshape(h*w//4,5120) for c in cache])
                k,v=refiner.repair_all(nk,nv,phi,emb)
                for i,c in enumerate(cache):
                    c['k'][:,begin:end].copy_(k[i].reshape(1,h*w//4,40,128));c['v'][:,begin:end].copy_(v[i].reshape(1,h*w//4,40,128))
                    save(f'fastframes.refiner.{i}.k',k[i]);save(f'fastframes.refiner.{i}.v',v[i])
        grid=torch.tensor([[frames,h//2,w//2]],device=device)
        pixels=align_frames(decoder(torch.cat(features,dim=1),torch.cat(mods,dim=1),grid),args.frames)
        save('fastframes.pixels',pixels)
        rgb=((pixels.float().clamp(-1,1)+1)*127.5).round().to(torch.uint8).permute(0,1,3,4,2).cpu().numpy();rgb.tofile(args.output/'rgb.u8')
    if device.type=='cuda':torch.cuda.synchronize()
    elapsed=time.perf_counter()-start
    if any(sha256(Path(__file__).parent/name)!=digest for name,digest in wrappers.items()):
        raise ValueError('oracle wrapper changed during execution; rerun with stable sources')
    manifest={
        'oracle_commit':pin['commit'],'reference_files_verified':True,
        'wrapper_sha256':wrappers,
        'tokenizer_sha256':pin['tokenizer_sha256'],
        'checkpoint_sha256':hashes,'gguf_sha256':gguf_hash,
        'noise_sha256':sha256(args.output/'noise.f32'),
        'timing_kind':'generation_including_rgb_transfer_write_and_streamed_weights' if args.benchmark else 'traced_including_checkpoint_io_and_streamed_weights',
        'weight_loading':'on-demand text/DiT blocks; release to meta after each call',
        'recipe':args.recipe,'device':str(device),'torch':torch.__version__,
        'numpy':np.__version__,'ftfy':ftfy.__version__,
        'dependencies':{name:importlib.metadata.version(name) for name in ['transformers','tokenizers','gguf','safetensors']},
        'elapsed_seconds':elapsed,
        'args':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        'checkpoints':records,
    }
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'elapsed_seconds':elapsed,'recipe':args.recipe,'device':str(device)}),flush=True)

if __name__=='__main__':main()
