"""Pinned graph interpreter: torch copies/layout, independent scalar C arithmetic.

F32 is the strict graph policy; F64/complex128 is restricted to the intrinsic
official time-frequency/RoPE path. Any unsupported arithmetic fails closed.
"""
import ctypes as ct
from contextlib import contextmanager
import math
from pathlib import Path
import subprocess
import sys

import torch
from torch.utils._python_dispatch import TorchDispatchMode

FLAGS = ["-O2", "-ffp-contract=off", "-fno-vectorize", "-fno-slp-vectorize"]


def tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from tensors(item)
    elif isinstance(value, dict):
        yield from tensors(list(value.values()))


class Scalar(TorchDispatchMode):
    def __init__(self, output):
        super().__init__()
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        library = output / ("scalar.dylib" if sys.platform == "darwin" else "scalar.so")
        subprocess.run(["clang", *FLAGS, "-shared", "-fPIC", str(Path(__file__).with_suffix(".c")),
                        "-lm", "-o", str(library)], check=True)
        self.lib = ct.CDLL(str(library.resolve()))
        p, n, i = ct.c_void_p, ct.c_size_t, ct.c_int
        for suffix, f in (("f32", ct.c_float), ("f64", ct.c_double)):
            for name, signature in {
                "linear": [p,p,p,p,n,n,n], "unary": [p,p,n,i,f],
                "binary": [p,p,p,n,i], "reduce": [p,p,n,n,i],
                "norm": [p,p,p,p,p,p,n,n,f], "softmax": [p,p,n,n],
            }.items():
                fn = getattr(self.lib, f"{name}_{suffix}")
                fn.argtypes, fn.restype = signature, None
        for name, signature in {
            "complex_mul": [p,p,p,n], "polar": [p,p,p,n],
            "attention": [p,p,p,p,p,n,n,n,n,n,ct.c_float],
            "conv3d": [p,p,p,p,p], "project_f32": [p,p,p,n,n,n,n,n,n],
            "solve_f32": [p,p,p,n],
        }.items():
            fn = getattr(self.lib, name)
            fn.argtypes, fn.restype = signature, None
        self.calls = {}
        self.operators = {}
        self._allow_f64 = 0

    @contextmanager
    def f64_scope(self):
        """Permit the official time-frequency helper's intrinsic F64 math."""
        self._allow_f64 += 1
        try:
            yield
        finally:
            self._allow_f64 -= 1

    def call(self, name, *args):
        self.calls[name] = self.calls.get(name, 0) + 1
        getattr(self.lib, name)(*(ct.c_void_p(a.data_ptr()) if isinstance(a, torch.Tensor)
                                 else a for a in args))

    def numeric(self, name, dtype, *args):
        if dtype not in (torch.float32, torch.float64):
            raise ValueError(f"Scalar arithmetic dtype {dtype}: {name}")
        if dtype == torch.float64 and not self._allow_f64:
            raise ValueError("F64 arithmetic is reserved for official complex RoPE; use F32 graph tensors")
        self.call(name + ("_f64" if dtype == torch.float64 else "_f32"), *args)

    def unary(self, x, op, a=0):
        x = x.contiguous()
        y = torch.empty_like(x)
        self.numeric("unary", x.dtype, x, y, x.numel(), op, a)
        return y

    def binary(self, x, z, op):
        dtype = x.dtype if isinstance(x, torch.Tensor) else z.dtype
        if isinstance(x, torch.Tensor) and isinstance(z, torch.Tensor):
            dtype = torch.promote_types(x.dtype, z.dtype)
        x, z = torch.broadcast_tensors(torch.as_tensor(x, dtype=dtype), torch.as_tensor(z, dtype=dtype))
        x, z = x.contiguous(), z.contiguous()
        y = torch.empty_like(x)
        if dtype == torch.complex128:
            if op != 2:
                raise NotImplementedError("Only complex128 RoPE multiplication is supported")
            self.call("complex_mul", x, z, y, x.numel())
        else:
            self.numeric("binary", dtype, x, z, y, x.numel(), op)
        return y

    def linear(self, x, w, bias=None):
        x, w = x.contiguous(), w.contiguous()
        if x.shape[-1] != w.shape[-1] or x.dtype != w.dtype:
            raise ValueError("Incompatible scalar linear inputs")
        y = torch.empty((*x.shape[:-1], w.shape[0]), dtype=x.dtype)
        self.numeric("linear", x.dtype, x, w, None if bias is None else bias.contiguous(),
                     y, x.numel() // x.shape[-1], x.shape[-1], w.shape[0])
        return y

    def matmul(self, a, b):
        av, bv = a.ndim == 1, b.ndim == 1
        if av: a = a.unsqueeze(-2)
        if bv: b = b.unsqueeze(-1)
        batch = torch.broadcast_shapes(a.shape[:-2], b.shape[:-2])
        a = a.expand(*batch, *a.shape[-2:]).contiguous()
        w = b.transpose(-1,-2).expand(*batch, b.shape[-1], b.shape[-2]).contiguous()
        y = torch.empty((*batch, a.shape[-2], b.shape[-1]), dtype=a.dtype)
        aa, ww, yy = a.reshape(-1,*a.shape[-2:]), w.reshape(-1,*w.shape[-2:]), y.reshape(-1,*y.shape[-2:])
        for j in range(aa.shape[0]):
            self.numeric("linear", a.dtype, aa[j], ww[j], None, yy[j], a.shape[-2], a.shape[-1], b.shape[-1])
        if av: y = y.squeeze(-2)
        if bv: y = y.squeeze(-1)
        return y

    def reduce(self, x, dims, keep, op):
        dims = list(range(x.ndim)) if dims is None else [dims] if isinstance(dims, int) else list(dims)
        dims = sorted({d % x.ndim for d in dims})
        if not dims: dims = list(range(x.ndim))
        other = [d for d in range(x.ndim) if d not in dims]
        shape = [x.shape[d] for d in other]
        d = math.prod(x.shape[k] for k in dims)
        if d == 0:
            raise ValueError("Empty scalar floating reduction")
        xx = x.permute(other+dims).contiguous()
        y = torch.empty(shape, dtype=x.dtype)
        self.numeric("reduce", x.dtype, xx, y, y.numel(), d, op)
        if keep:
            y = y.reshape([1 if k in dims else x.shape[k] for k in range(x.ndim)])
        return y

    def softmax(self, x, dim):
        dim %= x.ndim
        xx = x.movedim(dim, -1).contiguous()
        y = torch.empty_like(xx)
        self.numeric("softmax", x.dtype, xx, y, xx.numel()//xx.shape[-1], xx.shape[-1])
        return y.movedim(-1,dim)

    def attention(self, q, k, v, mask=None, scale=None):
        if any(t.dtype != torch.float32 or t.device.type != "cpu" for t in (q,k,v)):
            raise ValueError("Strict attention requires CPU F32")
        batch = torch.broadcast_shapes(q.shape[:-2],k.shape[:-2],v.shape[:-2])
        q,k,v = [t.expand(*batch,*t.shape[-2:]).contiguous() for t in (q,k,v)]
        nq,d,nk,dv = q.shape[-2],q.shape[-1],k.shape[-2],v.shape[-1]
        if mask is not None:
            if mask.dtype == torch.bool:
                mask = torch.where(mask,0.0,-float("inf"))
            mask = mask.expand(*batch,nq,nk).contiguous()
        y = torch.empty((*batch,nq,dv),dtype=q.dtype)
        self.call("attention",q,k,v,mask,y,math.prod(batch),nq,nk,d,dv,
                  1.0/math.sqrt(d) if scale is None else scale)
        return y

    def conv(self, x, w, b, stride, padding, dilation, groups):
        ndim = x.ndim-2
        if ndim not in (2,3) or x.dtype != torch.float32:
            raise NotImplementedError("Only scalar F32 conv2d/conv3d is covered")
        if ndim == 2:
            x,w = x.unsqueeze(2),w.unsqueeze(2)
            stride,padding,dilation = [1,*stride],[0,*padding],[1,*dilation]
        out = [(x.shape[k+2]+2*padding[k]-dilation[k]*(w.shape[k+2]-1)-1)//stride[k]+1 for k in range(3)]
        y = torch.empty((x.shape[0],w.shape[0],*out),dtype=x.dtype)
        s = ct.c_size_t * 22
        spec = s(*x.shape,w.shape[0],*w.shape[2:],*out,*stride,*padding,*dilation,groups)
        self.call("conv3d",x.contiguous(),w.contiguous(),None if b is None else b.contiguous(),y,spec)
        return y.squeeze(2) if ndim == 2 else y

    def einsum(self, equation, args):
        if equation == "...ij,...jk->...ik":
            return self.matmul(*args)
        if equation == "...ij,...j->...i":
            return self.matmul(args[0],args[1].unsqueeze(-1)).squeeze(-1)
        if equation == "bcij,bncpkj->bncpki":
            matrix, feats = args
            if matrix.dtype != torch.float32 or feats.dtype != torch.float32:
                raise ValueError("PRoPE projection requires F32")
            b,n,c,p,k,d = feats.shape
            if matrix.shape != (b,c,d,d):
                raise ValueError(f"PRoPE matrix/feature shape mismatch: {matrix.shape} {feats.shape}")
            y = torch.empty_like(feats)
            self.call("project_f32", matrix.contiguous(), feats.contiguous(), y,
                      b,n,c,p,k,d)
            return y
        if equation == "i,j->ij":
            return self.binary(args[0][:,None],args[1][None,:],2)
        if equation == "k,bkc...->bc...":
            weights, values = args
            weighted = self.binary(weights.reshape(1,-1,*([1]*(values.ndim-2))),values,2)
            return self.reduce(weighted,1,False,0)
        if "," not in equation and "->" in equation:
            src,dst = equation.split("->")
            if set(src)==set(dst) and len(set(src))==len(src):
                return args[0].permute([src.index(c) for c in dst])
        raise NotImplementedError(f"Uncovered floating einsum: {equation}")

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kw = kwargs or {}
        name = str(func).removeprefix("aten.")
        values = list(tensors((args,kw)))
        floats = [t for t in values if t.is_floating_point() or t.is_complex()]
        self.operators[name] = self.operators.get(name,0)+1
        if not floats:
            # Factories create constants or layout-only uninitialized storage.
            if name.startswith(("rand", "normal", "linspace", "logspace")):
                raise NotImplementedError(f"Uncovered floating factory: {name}")
            return func(*args,**kw)
        if any(t.device.type!="cpu" or t.dtype not in (torch.float32,torch.float64,torch.complex128) for t in floats):
            raise ValueError(f"Strict Oracle requires CPU F32/F64/complex128: {name}")
        x = args[0]
        if name == "linear.default":
            return self.linear(x,args[1],args[2] if len(args)>2 else None)
        if name in ("matmul.default","mm.default","bmm.default","mv.default","dot.default"):
            return self.matmul(*args[:2])
        if name == "addmm.default":
            y = self.matmul(args[1],args[2])
            alpha,beta = kw.get("alpha",1),kw.get("beta",1)
            if alpha!=1: y=self.binary(y,alpha,2)
            return y if beta==0 else self.binary(y,self.binary(x,beta,2) if beta!=1 else x,0)
        if name in ("layer_norm.default","native_layer_norm.default"):
            dims,w,b,eps = args[1:5]
            d = math.prod(dims); xx=x.contiguous(); y=torch.empty_like(xx)
            stats_shape=(*x.shape[:-len(dims)],*([1]*len(dims)))
            mean,inv = torch.empty(stats_shape,dtype=x.dtype),torch.empty(stats_shape,dtype=x.dtype)
            self.numeric("norm",x.dtype,xx,None if w is None else w.contiguous(),None if b is None else b.contiguous(),
                         y,mean,inv,x.numel()//d,d,eps)
            return (y,mean,inv) if name.startswith("native_") else y
        if name in ("convolution.default","_convolution.default"):
            if args[6] or any(args[7]):
                raise NotImplementedError("Transposed scalar convolution")
            return self.conv(*args[:6],args[8])
        if name in ("conv2d.default","conv3d.default"):
            n=x.ndim-2
            rest=list(args[2:])+[None,[1]*n,[0]*n,[1]*n,1]
            b=args[2] if len(args)>2 else None
            return self.conv(x,args[1],b,args[3] if len(args)>3 else [1]*n,
                             args[4] if len(args)>4 else [0]*n,args[5] if len(args)>5 else [1]*n,
                             args[6] if len(args)>6 else 1)
        if name == "einsum.default":
            return self.einsum(x,args[1])
        if name == "outer.default":
            return self.binary(x[:,None],args[1][None,:],2)
        if name == "polar.default":
            xx,z=torch.broadcast_tensors(x,args[1]);xx,z=xx.contiguous(),z.contiguous()
            if xx.dtype!=torch.float64: raise ValueError("Official RoPE polar must be F64")
            y=torch.empty_like(xx,dtype=torch.complex128);self.call("polar",xx,z,y,xx.numel());return y
        unary={"neg.default":0,"exp.default":1,"log.default":2,"sin.default":3,"cos.default":4,
               "tanh.default":5,"sqrt.default":6,"rsqrt.default":7,"reciprocal.default":10,
               "abs.default":11,"floor.default":12,"expm1.default":13,"sigmoid.default":14,
               "silu.default":15,"silu_.default":15,"gelu.default":16}
        if name in unary:
            op = 17 if name=="gelu.default" and kw.get("approximate",args[1] if len(args)>1 else "none")=="tanh" else unary[name]
            y=self.unary(x,op)
            return x.copy_(y) if "_." in name else y
        if name in ("pow.Tensor_Scalar","pow.Scalar"):
            return self.unary(x,8,args[1]) if name.endswith("Tensor_Scalar") else self.unary(args[1],9,x)
        binary={"add.Tensor":0,"add.Scalar":0,"add_.Tensor":0,"add_.Scalar":0,
                "sub.Tensor":1,"sub.Scalar":1,"sub_.Tensor":1,"sub_.Scalar":1,
                "rsub.Tensor":1,"rsub.Scalar":1,
                "mul.Tensor":2,"mul.Scalar":2,"mul_.Tensor":2,"mul_.Scalar":2,
                "div.Tensor":3,"div.Scalar":3,"div_.Tensor":3,"div_.Scalar":3,
                "maximum.default":4,"minimum.default":5,"pow.Tensor_Tensor":6}
        if name in binary:
            z=args[1];alpha=kw.get("alpha",1)
            if alpha!=1:z=self.binary(z,alpha,2) if isinstance(z,torch.Tensor) else z*alpha
            if name.startswith("rsub."):
                y=self.binary(z,x,1)
            else:
                y=self.binary(x,z,binary[name])
            return x.copy_(y) if "_." in name else y
        if name in ("clamp.default","clamp_.default","clamp_min.default","clamp_max.default"):
            low=args[1] if len(args)>1 else kw.get("min")
            high=args[2] if len(args)>2 else kw.get("max")
            if name=="clamp_max.default":high,low=low,None
            y=x
            if low is not None:y=self.binary(y,low,4)
            if high is not None:y=self.binary(y,high,5)
            return x.copy_(y) if name=="clamp_.default" else y
        if name=="relu.default":return self.binary(x,0,4)
        if name in ("softmax.int","_softmax.default"):
            return self.softmax(x,args[1])
        reductions={"sum.dim_IntList":0,"sum.default":0,"mean.dim":1,"mean.default":1,
                    "amax.default":2,"amin.default":3}
        if name in reductions:
            return self.reduce(x,args[1] if len(args)>1 else None,args[2] if len(args)>2 else kw.get("keepdim",False),reductions[name])
        if name=="linalg_vector_norm.default":
            if len(args)>1 and args[1]!=2:raise NotImplementedError("Only p=2 norm")
            return self.reduce(x,args[2] if len(args)>2 else None,args[3] if len(args)>3 else False,4)
        if name=="linalg_solve.default":
            if x.ndim!=2 or x.shape[0]!=x.shape[1] or args[1].ndim!=1 or x.dtype!=torch.float32:
                raise NotImplementedError("Only small UniPC vector solve")
            y=torch.empty_like(args[1]);self.call("solve_f32",x.contiguous(),args[1].contiguous(),y,x.shape[0]);return y
        if name=="scaled_dot_product_attention.default":
            dropout=args[4] if len(args)>4 else kw.get("dropout_p",0)
            causal=args[5] if len(args)>5 else kw.get("is_causal",False)
            if dropout or causal or kw.get("enable_gqa",False):raise NotImplementedError("Use explicit attention masks")
            return self.attention(*args[:3],args[3] if len(args)>3 else kw.get("attn_mask"),kw.get("scale"))
        layout={"slice.Tensor","select.int","view.default","_unsafe_view.default","reshape.default","flatten.using_ints","flatten.default","permute.default",
                "transpose.int","t.default","movedim.int","unsqueeze.default","squeeze.dim","squeeze.default","squeeze.dims","unflatten.int",
                "expand.default","expand_as.default","unbind.int","split.Tensor","split_with_sizes.default","chunk.default",
                "clone.default","contiguous.default","copy_.default","detach.default","detach_.default","alias.default",
                "lift_fresh.default","lift_fresh_copy.default","_to_copy.default","type_as.default","to.dtype","to.dtype_layout","to.device",
                "cat.default","concat.default","stack.default","repeat.default","index.Tensor","index_put_.default","index_select.default",
                "broadcast_tensors.default",
                "gather.default","embedding.default","masked_fill.Scalar","masked_fill_.Scalar","where.self",
                "fill_.Scalar","fill_.Tensor","zero_.default","empty_like.default","zeros_like.default","ones_like.default","new_zeros.default",
                "new_empty.default","new_full.default","view_as_complex.default","view_as_real.default","resolve_conj.default",
                "resolve_neg.default","_local_scalar_dense.default","item.default","equal.default","eq.Tensor","eq.Scalar",
                "ne.Tensor","ne.Scalar","lt.Tensor","lt.Scalar","le.Tensor","le.Scalar","gt.Tensor","gt.Scalar","ge.Tensor",
                "ge.Scalar","isfinite.default","isinf.default","isnan.default","any.default","all.default",
                "constant_pad_nd.default","upsample_nearest2d.default","upsample_nearest3d.default",
                "pad.default",
                "_upsample_nearest_exact2d.default","_upsample_nearest_exact3d.default"}
        if name in layout:
            if name in ("_to_copy.default","to.dtype","to.dtype_layout"):
                dtype=kw.get("dtype",args[1] if len(args)>1 and isinstance(args[1],torch.dtype) else None)
                if dtype in (torch.float16,torch.bfloat16):raise ValueError(f"Forbidden lossy graph cast: {dtype}")
            return func(*args,**kw)
        if name=="dropout.default" and (args[1]==0 or not args[2]):return x
        raise NotImplementedError(f"Uncovered floating operator: aten.{name}")
