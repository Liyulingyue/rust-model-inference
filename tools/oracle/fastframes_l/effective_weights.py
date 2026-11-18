"""Independent GGUF weight and prepared-activation semantics for the oracle."""
import numpy as np
import torch
import torch.nn.functional as F

QUANTIZED={2,3,8,12,13,14}


def activation_qdq(value,kind):
    if kind==1:return value.half().to(value.dtype)
    if kind not in QUANTIZED:return value
    block=256 if kind in {12,13,14} else 32
    x=value.float().reshape(*value.shape[:-1],-1,block)
    if block==32:
        maximum=x.abs().amax(dim=-1,keepdim=True)
        inverse=torch.where(maximum>0,127./maximum,torch.zeros_like(maximum))
        scale=(maximum/127.).half().float()
    else:
        index=x.abs().argmax(dim=-1,keepdim=True)
        maximum=x.gather(-1,index)
        inverse=torch.where(maximum!=0,-127./maximum,torch.zeros_like(maximum))
        scale=torch.where(inverse!=0,1./inverse,torch.zeros_like(inverse))
    return (torch.round(x*inverse).clamp(-128,127)*scale).reshape(value.shape).to(value.dtype)


def q4_1_sum_delta(value):
    x=value.float().reshape(*value.shape[:-1],-1,32)
    maximum=x.abs().amax(dim=-1)
    inverse=torch.where(maximum>0,127./maximum,torch.zeros_like(maximum))
    total=torch.round(x*inverse.unsqueeze(-1)).clamp(-128,127).sum(dim=-1)
    scale=maximum/127.
    return (total*scale).half().float()-total*scale.half().float()


def install_prepared_activations(modules,fast,formats,tensors=None):
    pointers={}
    minimums={}
    cached={}
    for prefix,module in zip(['dit','refiner','decoder','text'],modules):
        for name,submodule in module.named_modules():
            if isinstance(submodule,torch.nn.Conv3d) and submodule.kernel_size==(1,1,1):
                def pointwise(x,module=submodule):
                    x=x.permute(0,2,3,4,1)
                    y=F.linear(x,module.weight.reshape(module.out_channels,module.in_channels),module.bias)
                    return y.permute(0,4,1,2,3).contiguous()
                submodule.forward=pointwise
    def refresh():
        pointers.clear()
        minimums.clear()
        current={}
        def minimum(name,device):
            tensor=tensors[name]
            array=tensor.data.reshape(-1,20)[:,2:4].copy().view('<f2')
            return torch.from_numpy(array.astype(np.float32).reshape(int(tensor.shape[1]),-1)).to(device)
        def bind(names,weight,grouped=False):
            names=(names,) if isinstance(names,str) else tuple(names)
            pointers[weight.data_ptr()]=formats[names[0]]
            if formats[names[0]]==3:
                key=(names,weight.data_ptr())
                if key in cached:
                    value=cached[key]
                else:
                    parts=[minimum(name,weight.device) for name in names]
                    value=torch.stack(parts).transpose(-1,-2) if grouped else torch.cat(parts)
                current[key]=minimums[weight.data_ptr()]=value
        for prefix,module in zip(['dit','refiner','decoder','text'],modules):
            for name,submodule in module.named_modules():
                if isinstance(submodule,(torch.nn.Linear,torch.nn.Conv3d)) and not submodule.weight.is_meta:
                    bind(f'{prefix}.{name}.weight',submodule.weight)
            if prefix=='refiner':
                for name,parameter in module.named_parameters():
                    if name.startswith('heads_grouped.gemms.0.') and name.endswith('_w'):
                        field=name.rsplit('.',1)[-1][:-2]
                        bind([f'refiner.heads.{i}.{field}.weight' for i in range(parameter.shape[0])],parameter,True)
        for layer,qkv in enumerate(fast.qkv):
            if qkv is not None and not qkv[0].is_meta:
                bind([f'dit.blocks.{layer}.self_attn.{name}.weight' for name in ['q','k','v']],qkv[0])
        bind('dit.patch_embedding.weight',fast.patch_weight)
        cached.clear()
        cached.update(current)
    refresh()
    original_linear=F.linear
    original_baddbmm=torch.baddbmm
    def linear(value,weight,bias=None):
        result=original_linear(activation_qdq(value,pointers.get(weight.data_ptr(),0)),weight,bias)
        if weight.data_ptr() in minimums:
            result+=original_linear(q4_1_sum_delta(value),minimums[weight.data_ptr()]).to(result.dtype)
        return result
    def baddbmm(input,batch1,batch2,*,beta=1,alpha=1,out=None):
        result=original_baddbmm(input,activation_qdq(batch1,pointers.get(batch2.data_ptr(),0)),batch2,beta=beta,alpha=alpha,out=out)
        if batch2.data_ptr() in minimums:
            result+=alpha*torch.matmul(q4_1_sum_delta(batch1),minimums[batch2.data_ptr()]).to(result.dtype)
        return result
    F.linear=linear
    torch.baddbmm=baddbmm
    return refresh
