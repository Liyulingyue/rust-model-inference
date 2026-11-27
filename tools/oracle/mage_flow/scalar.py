"""Pinned official graph; Torch provides tensor layout, C provides scalar arithmetic."""
import ctypes as ct
import shutil
import subprocess
from pathlib import Path
import sys
import torch
import importlib.util
_spec=importlib.util.spec_from_file_location("laya_scalar",Path(__file__).resolve().parents[1]/"laya/scalar.py")
_base=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_base)
BaseScalar=_base.Scalar

class Scalar(BaseScalar):
    def __init__(self,output,kind="mage"):
        self.kind=kind;self.bmm_calls=0
        torch.utils._python_dispatch.TorchDispatchMode.__init__(self)
        output.mkdir(parents=True,exist_ok=True)
        cc=shutil.which("clang") or shutil.which("gcc")
        if not cc:raise RuntimeError("A C compiler is required")
        flags=["-O2","-ffp-contract=off",*( ["-fno-vectorize","-fno-slp-vectorize"] if "clang" in cc else ["-fno-tree-vectorize"])]
        library=output/"mage_scalar.so"
        subprocess.run([cc,*flags,"-shared","-fPIC",str(Path(__file__).with_suffix(".c")),"-lm","-o",str(library)],check=True)
        self.lib=ct.CDLL(str(library.resolve()));self.calls={};self.attention_trace=None
        p,n,f,i=ct.c_void_p,ct.c_size_t,ct.c_float,ct.c_int
        for name,signature in {
            "linear":[p,p,p,p,n,n,n],"unary":[p,p,n,i,f],"binary":[p,p,p,n,i],
            "norm":[p,p,p,p,n,n,f],"softmax":[p,p,n,n],"sum_rows":[p,p,n,n],
            "attention":[p,p,p,p,p,n,n,n,n,f],
            "mage_attention":[p,p,p,p,n,n,n,n,f,i,i,i],"mage_linear64":[p,p,p,p,n,n,n],"mage_softmax64":[p,p,n,n],
            "mage_unary":[p,p,n,i],"mage_mean":[p,p,n,n],
            "mage_norm":[p,p,p,p,p,p,n,n,f],"mage_group_norm":[p,p,p,p,n,n,n,n,f],
            "mage_linspace":[f,f,p,n],"mage_conv":[p,p,p,p]+[n]*12,"mage_complex_mul":[p,p,p,n],
        }.items():
            fn=getattr(self.lib,name);fn.argtypes=signature;fn.restype=None
        self.flags=flags
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        kw=kwargs or {};name=str(func);x=args[0] if args else None
        tensors=[a for a in args if isinstance(a,torch.Tensor) and (a.is_floating_point() or a.is_complex())]
        if any(a.device.type!='cpu' or a.dtype not in (torch.float32,torch.complex64) for a in tensors):
            raise ValueError(f'Scalar Oracle requires CPU F32/complex64: {name}')
        if name=="aten.rsub.Scalar":
            assert kw.get("alpha",1)==1;return self.binary(args[1],x,1)
        if name in ("aten.add.Scalar","aten.sub.Scalar","aten.mul.Scalar","aten.div.Scalar"):
            return self.binary(x,args[1],{"aten.add.Scalar":0,"aten.sub.Scalar":1,"aten.mul.Scalar":2,"aten.div.Scalar":3}[name])
        if name=="aten.add_.Tensor":
            assert kw.get('alpha',1)==1
            x.copy_(self.binary(x,args[1],0));return x
        if name=="aten.outer.default":return self.binary(args[0].unsqueeze(-1),args[1].unsqueeze(0),2)
        if name=="aten.linear.default" and self.kind=="vision":
            a,w=args[:2];bias=args[2] if len(args)>2 else None;y=torch.empty((*a.shape[:-1],w.shape[0]))
            self.call("mage_linear64",a.contiguous(),w.contiguous(),bias,y,a.numel()//w.shape[1],w.shape[1],w.shape[0]);return y
        if name in ("aten._softmax.default","aten.softmax.int") and self.kind in ("vision","text"):
            assert args[1] in (-1,x.ndim-1);y=torch.empty_like(x);self.call("mage_softmax64",x.contiguous(),y,x.numel()//x.shape[-1],x.shape[-1]);return y
        if name=="aten._softmax.default":
            return super().__torch_dispatch__(torch.ops.aten.softmax.int,types,(x,args[1]),{})
        if name in ("aten.bmm.default","aten.matmul.default") and self.kind in ("vision","text"):
            a,b=args;position=(a.ndim>=3 and a.shape[0]==3 and a.shape[-1]==1)
            if not position:self.bmm_calls+=1
            wide=not position and self.kind=="text" and self.bmm_calls%2==1
            batch=torch.broadcast_shapes(a.shape[:-2],b.shape[:-2]);a=a.expand(*batch,*a.shape[-2:]).contiguous()
            w=b.transpose(-1,-2).expand(*batch,b.shape[-1],b.shape[-2]).contiguous()
            y=torch.empty((*batch,a.shape[-2],b.shape[-1]));aa=a.reshape(-1,*a.shape[-2:]);ww=w.reshape(-1,*w.shape[-2:]);yy=y.reshape(-1,*y.shape[-2:])
            for j in range(aa.shape[0]):self.call("mage_linear64" if wide else "linear",aa[j],ww[j],None,yy[j],a.shape[-2],a.shape[-1],b.shape[-1])
            return y
        if name=="aten.conv3d.default" and self.kind=="vision":
            a,w,b=args[:3];assert a.shape[2:]==w.shape[2:] and a.shape[1]==w.shape[1]
            y=torch.empty((a.shape[0],w.shape[0],1,1,1));self.call("mage_linear64",a.contiguous(),w.contiguous(),b,y,a.shape[0],w[0].numel(),w.shape[0]);return y
        if name=="aten.linspace.default":
            start,end,count=args;y=torch.empty(count,dtype=kw.get("dtype",torch.float32))
            assert y.dtype==torch.float32 and y.device.type=="cpu"
            self.call("mage_linspace",start,end,y,count);return y
        if name=="aten.mul.Tensor" and any(isinstance(v,torch.Tensor) and v.is_complex() for v in args):
            a,b=torch.broadcast_tensors(*args);a,b=a.contiguous(),b.contiguous();y=torch.empty_like(a)
            assert a.dtype==b.dtype==torch.complex64
            self.call("mage_complex_mul",a,b,y,a.numel());return y
        if name=="aten.polar.default":
            amp,angle=args;return torch.complex(self.binary(amp,self.unary(angle,4),2),self.binary(amp,self.unary(angle,3),2))
        if name=="aten.addmm.default":
            bias,a,w=args;assert kw.get("alpha",1)==kw.get("beta",1)==1
            return self.__torch_dispatch__(torch.ops.aten.linear.default,types,(a,w.T,bias),{})
        if name in ("aten.native_layer_norm.default","aten.layer_norm.default"):
            x=x.contiguous();dims=args[1];assert list(dims)==[x.shape[-1]]
            weight=args[2] if len(args)>2 else None;bias=args[3] if len(args)>3 else None
            eps=args[4] if len(args)>4 else kw.get("eps",1e-5)
            y=torch.empty_like(x);shape=(*x.shape[:-1],1);means=torch.empty(shape);scales=torch.empty(shape)
            self.call("mage_norm",x,weight,bias,y,means,scales,x.numel()//x.shape[-1],x.shape[-1],eps)
            return (y,means,scales) if "native" in name else y
        if name=="aten.group_norm.default":
            x,groups=args[:2];w=args[2] if len(args)>2 else None;b=args[3] if len(args)>3 else None;eps=args[4] if len(args)>4 else 1e-5
            return self.__torch_dispatch__(torch.ops.aten.native_group_norm.default,types,(x,w,b,x.shape[0],x.shape[1],x.numel()//(x.shape[0]*x.shape[1]),groups,eps),{})[0]
        if name=="aten.native_group_norm.default":
            x,w,b,batches,c,plane,groups,eps=args;y=torch.empty_like(x)
            self.call("mage_group_norm",x.contiguous(),w,b,y,batches,c,plane,groups,eps)
            # Only output is consumed by GroupNorm; stats are intentionally inaccessible.
            return y,torch.empty((batches,groups)),torch.empty((batches,groups))
        if name=="aten.conv2d.default":
            x,w=args[:2];b=args[2] if len(args)>2 else None
            stride=args[3] if len(args)>3 else [1,1];pad=args[4] if len(args)>4 else [0,0]
            dilation=args[5] if len(args)>5 else [1,1];groups=args[6] if len(args)>6 else 1
            return self.__torch_dispatch__(torch.ops.aten.convolution.default,types,(x,w,b,stride,pad,dilation,False,[0,0],groups),{})
        if name=="aten.convolution.default":
            x,w,b,stride,pad,dilation,transposed,outpad,groups=args
            assert not transposed and list(dilation)==[1,1] and list(outpad)==[0,0]
            batches,ci,h,width=x.shape;co,icg,kh,kw=w.shape
            assert icg==ci//groups
            y=torch.empty((batches,co,(h+2*pad[0]-kh)//stride[0]+1,(width+2*pad[1]-kw)//stride[1]+1))
            self.call("mage_conv",x.contiguous(),w.contiguous(),b,y,batches,ci,h,width,co,kh,kw,*stride,*pad,groups)
            return y
        if name=="aten.pad.default":
            if not x.is_floating_point():return func(*args,**kw)
            assert args[2]=="replicate"
            return torch.ops.aten.replication_pad2d.default(args[0],args[1])
        if name=="aten.mean.dim":
            dims=args[1];dims=[d%x.ndim for d in dims];keep=args[2] if len(args)>2 else kw.get("keepdim",False)
            other=[d for d in range(x.ndim) if d not in dims];a=x.permute(other+dims).contiguous();d=1
            for dim in dims:d*=x.shape[dim]
            shape=[1 if j in dims else x.shape[j] for j in range(x.ndim)] if keep else [x.shape[j] for j in other]
            y=torch.empty(shape,dtype=x.dtype);self.call("mage_mean",a,y,x.numel()//d,d);return y
        if name in ("aten._adaptive_avg_pool2d.default","aten.adaptive_avg_pool2d.default"):
            assert list(args[1])==[1,1]
            y=torch.empty((*x.shape[:-2],1,1));self.call("mage_mean",x.contiguous(),y,x.numel()//(x.shape[-2]*x.shape[-1]),x.shape[-2]*x.shape[-1]);return y
        unary={"aten.rsqrt.default":0,"aten.silu.default":1,"aten.sigmoid.default":2,"aten.sqrt.default":5}
        if name=="aten.pow.Tensor_Scalar" and args[1]==2:unary[name]=3
        if name=="aten.gelu.default" and (args[1] if len(args)>1 else kw.get("approximate","none"))=="tanh":unary[name]=4
        if name in unary:
            y=torch.empty_like(x);self.call("mage_unary",x.contiguous(),y,x.numel(),unary[name]);return y
        if name in ("aten.mm.default","aten.bmm.default"):
            return super().__torch_dispatch__(torch.ops.aten.matmul.default,types,args,kw)
        if name=="aten.pow.Tensor_Scalar" and args[1]==-1:return self.unary(x,7)
        if name in ("aten.clamp.default","aten.clamp.Tensor"):
            low=args[1] if len(args)>1 else kw.get("min");high=args[2] if len(args)>2 else kw.get("max")
            return torch.where(x<low,low,torch.where(x>high,high,x))
        if name.removeprefix("aten.") in {"to.device","type_as.default","numpy_T.default","repeat_interleave.self_Tensor","repeat.default","_local_scalar_dense.default","full_like.default","ones_like.default","flip.default","flatten.using_ints","view_as_real.default","view_as_complex.default","_to_copy.default","t.default","split.Tensor","split_with_sizes.default","squeeze.default","replication_pad2d.default","im2col.default","col2im.default","_unsafe_view.default","as_strided.default","index.Tensor","index_put_.default","index_copy_.default","fill_.Scalar","new_zeros.default","new_empty.default","resize_.default","complex.default","lt.Scalar","gt.Scalar","where.ScalarSelf","where.ScalarOther","where.Scalar","where.self","eq.Scalar","eq.Tensor","isfinite.default"}:
            return func(*args,**kw)
        return super().__torch_dispatch__(func,types,args,kw)
