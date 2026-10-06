"""Generate a scalar trace from the clean pinned Microsoft graph; compare every u32 bit."""
import argparse
import collections
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

def selected_records(path):
    values=[json.loads(s) for s in Path(path).read_text().splitlines()]
    if any(v["name"].startswith("mage_flow.text.") for v in values):
        return [v for v in values if v["name"].startswith(("mage_flow.","hidden_sequence."))]
    return values

COMMIT='76bec2bb3818863f470de7e867c2dc7f1d0bfd83'

def compare(left,right):
    a=selected_records(left);b=selected_records(right)
    assert len(a)==len(b),(len(a),len(b))
    total=0
    for x,y in zip(a,b):
        assert (x['name'],x['shape'],x['occurrence'])==(y['name'],y['shape'],y['occurrence']),(x,y)
        xx=np.fromfile(x['binary_path'],dtype='<u4');yy=np.fromfile(y['binary_path'],dtype='<u4')
        assert xx.size==yy.size==int(np.prod(x['shape'])),x['name']
        bad=np.flatnonzero(xx!=yy)
        if bad.size:
            i=int(bad[0]);raise AssertionError(f"First divergence {x['name']} occurrence {x['occurrence']} [{i}]: {xx[i]:08x} != {yy[i]:08x}; {bad.size}/{xx.size} different")
        total+=xx.size
    print(f'PASS: {len(a)} checkpoints, {total} F32 values, every raw u32 bit matches')

class Trace:
    def __init__(self,output):self.output=output;self.records=[];self.counts=collections.Counter()
    def emit(self,name,value,shape=None,prefix="mage_flow."):
        name=prefix+name;value=value.detach().float().cpu().contiguous().numpy();count=self.counts[name];self.counts[name]+=1
        path=self.output/f'{name}.{count}.f32';value.tofile(path)
        self.records.append(dict(name=name,shape=shape or [1,value.size],occurrence=count,binary_path=str(path.resolve())))
    def save(self):
        (self.output/'trace.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in self.records))

def vae(args):
    import torch
    from scalar import Scalar
    torch.set_num_threads(1);torch.backends.mkldnn.enabled=False
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False);trace=Trace(output)
    spec=importlib.util.spec_from_file_location('mage_vae',args.oracle/'mage_flow/models/modules/mage_vae.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    class UnfoldedVae(module.MageVAE):
        def _freeze_adaln_cache(self):pass
    model=UnfoldedVae(str(args.model/'vae/diffusion_pytorch_model.safetensors'),sample_posterior=False).float().eval()
    for name,m in model.named_modules():
        if name.startswith('dconv_encoder.'):source='student.'+name
        elif name.startswith('decoder_model.'):source='pipeline.'+name.removeprefix('decoder_model.')
        else:continue
        if isinstance(m,(torch.nn.Linear,torch.nn.Conv2d,module.DiCoBlock,module._EncoderDiCoBlock)):
            m.register_forward_hook(lambda m,a,result,name=source:trace.emit('vae.'+name,result))
    raw=np.fromfile(args.input,dtype='<f4');scalar=Scalar(output)
    with torch.inference_mode(),scalar:
        if args.mode=='vae-encode':
            result=model.encode(torch.from_numpy(raw.copy()).reshape(1,3,args.height,args.width));trace.emit('vae.mean',result)
        else:
            result=model.decode(torch.from_numpy(raw.copy()).reshape(1,128,args.height,args.width));trace.emit('vae.decoded',result)
    trace.save();(output/'scalar.json').write_text(json.dumps(dict(commit=COMMIT,torch=torch.__version__,calls=scalar.calls,flags=scalar.flags),indent=2))
    print(f'{len(trace.records)} Oracle checkpoints saved')

def dit(args):
    import torch
    from safetensors.torch import load_file
    from scalar import Scalar
    torch.set_num_threads(1)
    sys.path.insert(0,str(args.oracle))
    from mage_flow.models.mage_flow import MageFlow,MageFlowParams
    from mage_flow.models.modules import mage_layers
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False);trace=Trace(output);scalar=Scalar(output)
    params=MageFlowParams(128,128,2560,3072,24,12,[16,56,56],False)
    with torch.device('meta'):model=MageFlow(params)
    weights=load_file(str(args.model/'transformer/diffusion_pytorch_model.safetensors'))
    model.load_state_dict({k:v.float() for k,v in weights.items()},assign=True);del weights
    model.eval()
    with scalar:model.pos_embed=mage_layers.MageFlowEmbedRope(theta=10000,axes_dim=[16,56,56],scale_rope=True)
    def emit(name,value):trace.emit(name,value,[value.numel()//value.shape[-1],value.shape[-1]])
    current=[None];state={}
    for name,m in model.named_modules():
        if isinstance(m,(torch.nn.Linear,mage_layers.RMSNorm)):
            m.register_forward_hook(lambda m,a,r,name=name:emit(name,r))
        if name=='time_text_embed.time_proj':m.register_forward_hook(lambda m,a,r:emit('time_proj',r))
        if name=='norm_out':m.register_forward_hook(lambda m,a,r:emit('norm_out',r))
    original_rope=mage_layers.apply_rotary_emb_mageflow
    def rope(x,freq):
        result=original_rope(x,freq);index=state.get('rope_index',0);state['rope_index']=index+1
        emit(current[0]+('.attn.rope_q' if index%2==0 else '.attn.rope_k'),result.flatten(-2));return result
    mage_layers.apply_rotary_emb_mageflow=rope
    def attention(q,k,v,**kwargs):
        assert kwargs['causal'] is False and kwargs['dropout_p']==0
        assert kwargs['cu_seqlens_q'].tolist()==[0,q.shape[0]]
        a,b,c=[t.permute(1,0,2).contiguous() for t in (q,k,v)];y=torch.empty_like(a)
        scalar.call('mage_attention',a,b,c,y,a.shape[0],a.shape[1],b.shape[1],a.shape[2],1.0/(a.shape[2]**.5),1,0,0)
        result=y.permute(1,0,2).contiguous();emit(current[0]+'.attn.context',result.flatten(-2));return result
    mage_layers.flash_attn_varlen_func=attention
    for i,block in enumerate(model.transformer_blocks):
        prefix=f'transformer_blocks.{i}'
        def begin(m,a,kw,prefix=prefix):
            current[0]=prefix;state.clear();state['img']=kw['hidden_states'];state['txt']=kw['encoder_hidden_states']
        block.register_forward_pre_hook(begin,with_kwargs=True)
        block.img_mod.register_forward_hook(lambda m,a,r:state.__setitem__('im',r))
        block.txt_mod.register_forward_hook(lambda m,a,r:state.__setitem__('tm',r))
        def attn_begin(m,a,kw,prefix=prefix):
            emit(prefix+'.img_modulated',kw['hidden_states']);emit(prefix+'.txt_modulated',kw['encoder_hidden_states'])
        block.attn.register_forward_pre_hook(attn_begin,with_kwargs=True)
        def attn_end(m,a,r,prefix=prefix):
            state['img_after']=state['img']+state['im'][:,6144:9216].unsqueeze(1)*r[0]
            state['txt_after']=state['txt']+state['tm'][:,6144:9216].unsqueeze(1)*r[1]
            emit(prefix+'.img_after_attn',state['img_after']);emit(prefix+'.txt_after_attn',state['txt_after'])
        block.attn.register_forward_hook(attn_end)
        for stream in ('img','txt'):
            mlp=getattr(block,stream+'_mlp')
            mlp.net[0].register_forward_hook(lambda m,a,r,prefix=prefix,stream=stream:emit(prefix+'.'+stream+'_mlp.gelu',r))
            def mlp_end(m,a,r,prefix=prefix,stream=stream):
                params=state['im' if stream=='img' else 'tm'];value=state[stream+'_after']+params[:,15360:18432].unsqueeze(1)*r
                emit(prefix+'.'+stream,value)
            mlp.register_forward_hook(mlp_end)
    shapes=[tuple(map(int,s.split('x'))) for s in args.shapes.split(',')]
    raw=np.fromfile(args.input,dtype='<f4');context=np.fromfile(args.context,dtype='<f4')
    try:
        with torch.inference_mode(),scalar:
            image=torch.from_numpy(raw.copy()).reshape(1,-1,128)
            def forward(image,context,sigma):
                return model(image,context,torch.tensor([sigma],dtype=torch.float32),img_shapes=[[(1,h,w) for h,w in shapes]],img_cu_seqlens=torch.tensor([0,raw.size//128],dtype=torch.int32),txt_cu_seqlens=torch.tensor([0,context.numel()//2560],dtype=torch.int32))
            context=torch.from_numpy(context.copy()).reshape(1,-1,2560)
            if args.mode=='sample':
                from mage_flow.pipeline import build_scheduler
                scheduler=build_scheduler(args.steps,device='cpu')
                target=shapes[0][0]*shapes[0][1]
                trace.emit('sample.sigmas',scheduler.sigmas)
                negative=torch.from_numpy(np.fromfile(args.negative_context,dtype='<f4').copy()).reshape(1,-1,2560) if args.negative_context else None
                assert args.cfg==1 or negative is not None
                for step,t in enumerate(scheduler.timesteps):
                    velocity=forward(image,context,scheduler.sigmas[step].item())
                    if args.cfg>1:
                        unc=forward(image,negative,scheduler.sigmas[step].item());velocity=unc+args.cfg*(velocity-unc)
                    trace.emit('sample.velocity',velocity[:,:target,:])
                    stepped=scheduler.step(velocity[:,:target,:],t,image[:,:target,:],return_dict=False)[0]
                    image=torch.cat([stepped,image[:,target:,:]],dim=1)
                    trace.emit('sample.latent',stepped)
                result=image[:,:target,:]
            else:result=forward(image,context,args.sigma)
    finally:trace.save()
    (output/'scalar.json').write_text(json.dumps(dict(commit=COMMIT,torch=torch.__version__,calls=scalar.calls,flags=scalar.flags),indent=2))
    print(f'{len(trace.records)} Oracle checkpoints saved')

def hf_component(args):
    import torch
    from safetensors import safe_open
    from transformers import AutoConfig,AutoTokenizer
    from transformers.models.qwen3_vl import modeling_qwen3_vl as hf
    from scalar import Scalar
    torch.set_num_threads(1);torch.backends.mkldnn.enabled=False
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False);trace=Trace(output)
    kind=args.mode;scalar=Scalar(output,kind);config=AutoConfig.from_pretrained(args.model/'text_encoder',local_files_only=True)
    cfg=config.vision_config if kind=='vision' else config.text_config;cfg._attn_implementation='eager'
    with torch.device('meta'):model=(hf.Qwen3VLVisionModel if kind=='vision' else hf.Qwen3VLTextModel)(cfg)
    prefix='model.visual.' if kind=='vision' else 'model.language_model.'
    state={}
    for shard in sorted((args.model/'text_encoder').glob('model-*.safetensors')):
        with safe_open(shard,framework='pt',device='cpu') as f:
            for key in f.keys():
                if key.startswith(prefix):state[key.removeprefix(prefix)]=f.get_tensor(key).float()
    model.load_state_dict(state,assign=True,strict=True);del state;model.eval()
    with scalar:
        if kind=='vision':model.rotary_pos_emb=hf.Qwen3VLVisionRotaryEmbedding(32)
        else:model.rotary_emb.inv_freq=model.rotary_emb.rope_init_fn(cfg,device='cpu')[0]
    hidden=[];first={}
    if kind=='text':
        original_rope=hf.apply_rotary_pos_emb
        def traced_rope(*a,**kw):
            q,k=original_rope(*a,**kw)
            if 'Qcur-0' not in first:
                first['Qcur-0']=q.transpose(1,2).contiguous().detach().clone()
                first['Kcur-0']=k.transpose(1,2).contiguous().detach().clone()
            return q,k
        hf.apply_rotary_pos_emb=traced_rope
        original_attention=hf.eager_attention_forward
        def traced_attention(*a,**kw):
            result=original_attention(*a,**kw)
            if 'kqv_out-0' not in first:first['kqv_out-0']=result[0].detach().clone()
            return result
        hf.eager_attention_forward=traced_attention
        for label,part in [('model.input_embed',model.embed_tokens),('attn_norm-0',model.layers[0].input_layernorm),('q_norm-0',model.layers[0].self_attn.q_norm),('k_norm-0',model.layers[0].self_attn.k_norm),('ffn_out-0',model.layers[0].mlp)]:
            part.register_forward_hook(lambda m,a,r,label=label:first.__setitem__(label,r.detach().clone()))
    if kind=='vision':
        for i,block in enumerate(model.blocks):block.register_forward_hook(lambda m,a,r,i=i:trace.emit(f'vision.blocks.{i}',r))
        for i,merger in enumerate(model.deepstack_merger_list):merger.register_forward_hook(lambda m,a,r,i=i:trace.emit(f'vision.deepstack.{i}',r))
        def norm_input(m,a):trace.emit('vision.input',a[0])
        model.blocks[0].norm1.register_forward_pre_hook(norm_input)
        raw=np.fromfile(args.input,dtype='<f4').reshape(args.height,args.width,3)
        # HF processor groups patches into spatial merge blocks; temporal still frames repeat twice.
        pixels=torch.from_numpy(raw.copy()).permute(2,0,1)
        ph,pw=args.height//16,args.width//16
        patches=pixels.reshape(3,ph//2,2,16,pw//2,2,16).permute(1,4,2,5,0,3,6).reshape(-1,3,16,16)
        patches=patches.unsqueeze(2).expand(-1,-1,2,-1,-1).contiguous().reshape(-1,1536)
        try:
            with torch.inference_mode(),scalar:
                result,deepstack=model(patches,torch.tensor([[1,ph,pw]]));trace.emit('vision.output',result)
        finally:trace.save()
    else:
        tokenizer=AutoTokenizer.from_pretrained(args.model/'text_encoder',local_files_only=True)
        sys.path.insert(0,str(args.oracle));from mage_flow.models import utils as module
        template=module.PROMPT_TEMPLATE['mage-flow-edit' if args.references else 'mage-flow']
        body=''.join(f'Image {i}: <|vision_start|><|image_pad|><|vision_end|>' for i in range(1,args.references+1))+args.prompt
        ids=tokenizer.encode(template['template'].format(body),add_special_tokens=False)
        if args.references:
            features=torch.from_numpy(np.fromfile(args.reference_embeddings,dtype='<f4').copy()).reshape(-1,2560)
            ds=torch.from_numpy(np.fromfile(args.reference_deepstack,dtype='<f4').copy()).reshape(3,-1,2560)
            ids=[v for token in ids for v in ([151655]*features.shape[0] if token==151655 else [token])]
        trace.emit('text.tokens',torch.tensor(ids,dtype=torch.float32))
        for i,block in enumerate(model.layers):
            def layer_hook(m,a,r,i=i):
                value=r[0] if isinstance(r,tuple) else r
                if args.references and i<3:
                    value=value.clone();value[mask]+=ds[i].repeat(args.references,1)
                hidden.append((i,value.detach().clone()))
            block.register_forward_hook(layer_hook)
        try:
            with torch.inference_mode(),scalar:
                kwargs={}
                if args.references:
                    embeddings=model.embed_tokens(torch.tensor([ids]));mask=torch.tensor([ids])==151655
                    embeddings[mask]=features.repeat(args.references,1)
                    kwargs=dict(inputs_embeds=embeddings,visual_pos_masks=mask,deepstack_visual_embeds=[d.repeat(args.references,1) for d in ds])
                else:kwargs=dict(input_ids=torch.tensor([ids]))
                result=model(**kwargs,position_ids=torch.arange(len(ids)).reshape(1,-1),use_cache=False).last_hidden_state
        finally:
            for label,value in first.items():np.save(output/(label+'.npy'),value.numpy())
            for token in range(len(ids)):
                for layer,value in hidden:trace.emit(f'hidden_sequence.layer.{layer}',value[:,token,:],[1,2560],prefix='')
            if 'result' in locals():trace.emit('text.context',result[:,template['start_idx']:,:])
            trace.save()
    (output/'scalar.json').write_text(json.dumps(dict(commit=COMMIT,torch=torch.__version__,calls=scalar.calls,flags=scalar.flags),indent=2));print(f'{len(trace.records)} Oracle checkpoints saved')

def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='mode',required=True)
    cmp=sub.add_parser('compare');cmp.add_argument('oracle');cmp.add_argument('rust')
    for mode in ('vae-encode','vae-decode'):
        p=sub.add_parser(mode);p.add_argument('--oracle',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--input',type=Path,required=True);p.add_argument('--height',type=int,required=True);p.add_argument('--width',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    for mode in ('dit','sample'):
        p=sub.add_parser(mode);p.add_argument('--oracle',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--input',type=Path,required=True);p.add_argument('--context',type=Path,required=True);p.add_argument('--shapes',required=True);p.add_argument('--sigma',type=float,default=.5);p.add_argument('--output',type=Path,required=True)
        if mode=='sample':
            p.add_argument('--steps',type=int,default=4);p.add_argument('--cfg',type=float,default=1);p.add_argument('--negative-context',type=Path)
    for kind in ('text','vision'):
        p=sub.add_parser(kind);p.add_argument('--oracle',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
        if kind=='text':
            p.add_argument('--prompt',required=True);p.add_argument('--references',type=int,default=0);p.add_argument('--reference-embeddings',type=Path);p.add_argument('--reference-deepstack',type=Path)
        else:p.add_argument('--input',type=Path,required=True);p.add_argument('--height',type=int,required=True);p.add_argument('--width',type=int,required=True)
    args=parser.parse_args()
    if args.mode=='compare':return compare(args.oracle,args.rust)
    commit=subprocess.check_output(['git','-C',str(args.oracle),'rev-parse','HEAD'],text=True).strip()
    assert commit==COMMIT,commit
    assert not subprocess.check_output(['git','-C',str(args.oracle),'status','--porcelain'],text=True),'Oracle must remain clean'
    if args.mode in ('text','vision'):hf_component(args)
    elif args.mode in ('dit','sample'):dit(args)
    else:vae(args)

if __name__=='__main__':main()
