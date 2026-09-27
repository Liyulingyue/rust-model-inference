"""Run the official graph through independent scalar F32 primitives.

Torch handles indexing/layout only. Every floating arithmetic operator is
intercepted; an unknown arithmetic operator fails instead of using torch kernels.
"""
import ctypes as ct
from pathlib import Path
import subprocess

import torch
from torch.utils._python_dispatch import TorchDispatchMode

FLAGS = ["-O2", "-ffp-contract=off", "-fno-vectorize", "-fno-slp-vectorize"]


class Scalar(TorchDispatchMode):
    def __init__(self, output):
        super().__init__()
        output.mkdir(parents=True, exist_ok=True)
        library = output / "scalar.dylib"
        subprocess.run(["clang", *FLAGS, "-shared", "-fPIC",
                        str(Path(__file__).with_suffix(".c")), "-o", str(library)], check=True)
        self.lib = ct.CDLL(str(library.resolve()))
        p, n, f, i = ct.c_void_p, ct.c_size_t, ct.c_float, ct.c_int
        for name, signature in {
            "linear": [p,p,p,p,n,n,n], "unary": [p,p,n,i,f],
            "binary": [p,p,p,n,i], "norm": [p,p,p,p,n,n,f],
            "softmax": [p,p,n,n], "sum_rows": [p,p,n,n],
            "attention": [p,p,p,p,p,n,n,n,n,f],
        }.items():
            fn = getattr(self.lib, name)
            fn.argtypes, fn.restype = signature, None
        self.calls = {}
        self.attention_trace = None

    def call(self, name, *args):
        self.calls[name] = self.calls.get(name, 0)+1
        getattr(self.lib, name)(*(ct.c_void_p(a.data_ptr()) if isinstance(a, torch.Tensor)
                                 else a for a in args))

    def unary(self, x, op, a=0):
        x = x.contiguous()
        y = torch.empty_like(x)
        self.call("unary", x, y, x.numel(), op, a)
        return y

    def binary(self, x, z, op):
        x, z = torch.broadcast_tensors(torch.as_tensor(x, dtype=torch.float32),
                                       torch.as_tensor(z, dtype=torch.float32))
        x, z = x.contiguous(), z.contiguous()
        y = torch.empty_like(x)
        self.call("binary", x, z, y, x.numel(), op)
        return y

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kw = kwargs or {}
        name = str(func)
        floats = [a for a in args if isinstance(a, torch.Tensor) and a.is_floating_point()]
        if not floats:
            return func(*args, **kw)
        if any(a.dtype != torch.float32 or a.device.type != "cpu" for a in floats):
            raise ValueError(f"Scalar Oracle requires CPU F32: {name}")
        x = args[0]
        if name == "aten.linear.default":
            x, w = x.contiguous(), args[1].contiguous()
            b = args[2].contiguous() if len(args)>2 and args[2] is not None else None
            y = torch.empty((*x.shape[:-1], w.shape[0]), dtype=x.dtype)
            self.call("linear", x, w, b, y, x.numel()//w.shape[1], w.shape[1], w.shape[0])
            return y
        if name == "aten.layer_norm.default":
            dims, w = args[1:3]
            b = args[3] if len(args)>3 else kw.get("bias")
            eps = args[4] if len(args)>4 else kw.get("eps", 1e-5)
            assert list(dims) == [x.shape[-1]] and w is not None
            x = x.contiguous(); y = torch.empty_like(x)
            self.call("norm", x, w.contiguous(), b.contiguous() if b is not None else None,
                      y, x.numel()//x.shape[-1], x.shape[-1], eps)
            return y
        if name == "aten.scaled_dot_product_attention.default":
            q, k, v = [a.contiguous() for a in args[:3]]
            mask = args[3] if len(args)>3 else kw.get("attn_mask")
            dropout = args[4] if len(args)>4 else kw.get("dropout_p", 0)
            causal = args[5] if len(args)>5 else kw.get("is_causal", False)
            assert dropout == 0 and not causal and not kw.get("enable_gqa", False)
            nq, d, nk = q.shape[-2], q.shape[-1], k.shape[-2]
            if mask is not None:
                if mask.dtype == torch.bool:
                    mask = torch.where(mask, 0.0, -float("inf"))
                mask = mask.expand(*q.shape[:-2], nq, nk).contiguous()
            y = torch.empty_like(q)
            first_encoder = self.calls.get("attention", 0) % 24 == 0
            if first_encoder and self.attention_trace:
                packed = torch.cat([t.transpose(1,2).reshape(t.shape[0], nq, -1)
                                    for t in (q,k,v)], -1)
                self.attention_trace("laya.encoder.0.rope", packed)
            self.call("attention", q, k, v, mask, y, q.numel()//(nq*d), nq, nk, d,
                      kw.get("scale") or 1.0/(d**0.5))
            if first_encoder and self.attention_trace:
                self.attention_trace("laya.encoder.0.context", y.transpose(1,2).reshape(y.shape[0],nq,-1))
            return y
        if name == "aten.matmul.default":
            a, b = args
            assert a.ndim >= 2 and b.ndim >= 2
            batch = torch.broadcast_shapes(a.shape[:-2], b.shape[:-2])
            a = a.expand(*batch, *a.shape[-2:]).contiguous()
            w = b.transpose(-1,-2).expand(*batch, b.shape[-1], b.shape[-2]).contiguous()
            y = torch.empty((*batch, a.shape[-2], b.shape[-1]), dtype=a.dtype)
            aa, ww, yy = a.reshape(-1,*a.shape[-2:]), w.reshape(-1,*w.shape[-2:]), y.reshape(-1,*y.shape[-2:])
            for j in range(aa.shape[0]):
                self.call("linear", aa[j], ww[j], None, yy[j], a.shape[-2], a.shape[-1], b.shape[-1])
            return y
        unary = {"aten.neg.default":0, "aten.exp.default":1, "aten.log.default":2,
                 "aten.sin.default":3, "aten.cos.default":4, "aten.gelu.default":5,
                 "aten.clamp_min.default":6, "aten.relu.default":6,
                 "aten.reciprocal.default":7}
        if name in unary:
            if name == "aten.gelu.default":
                assert kw.get("approximate", "none") == "none"
            return self.unary(x, unary[name], args[1] if name == "aten.clamp_min.default" else 0)
        if name == "aten.pow.Scalar":
            return self.unary(args[1], 8, args[0])
        binary = {"aten.add.Tensor":0, "aten.sub.Tensor":1,
                  "aten.mul.Tensor":2, "aten.div.Tensor":3}
        if name in binary:
            assert kw.get("alpha", 1) == 1
            return self.binary(x, args[1], binary[name])
        if name in ("aten.softmax.int", "aten.sum.dim_IntList"):
            dim = args[1]
            if isinstance(dim, list):
                assert len(dim) == 1
                dim = dim[0]
            assert dim in (-1, x.ndim-1)
            x = x.contiguous()
            if name == "aten.softmax.int":
                y = torch.empty_like(x)
                self.call("softmax", x, y, x.numel()//x.shape[-1], x.shape[-1])
            else:
                keep = args[2] if len(args)>2 else kw.get("keepdim", False)
                y = torch.empty((*x.shape[:-1],1) if keep else x.shape[:-1], dtype=x.dtype)
                self.call("sum_rows", x, y, y.numel(), x.shape[-1])
            return y
        # No floating arithmetic: copies, views, index selection, comparisons.
        layout = {
            "slice.Tensor", "to.dtype", "to.dtype_layout", "unsqueeze.default",
            "transpose.int", "view.default", "contiguous.default", "expand.default",
            "unbind.int", "chunk.default", "select.int", "masked_fill.Scalar",
            "squeeze.dim", "embedding.default", "masked_fill_.Scalar", "unflatten.int",
            "reshape.default", "permute.default", "lift_fresh.default", "detach_.default",
            "gather.default", "detach.default", "topk.default", "clone.default",
            "copy_.default", "cat.default", "stack.default", "empty_like.default",
            "zeros_like.default", "alias.default", "resolve_conj.default", "resolve_neg.default",
            "item.default",
        }
        if name.removeprefix("aten.") in layout:
            return func(*args, **kw)
        if name == "aten.dropout.default" and (args[1] == 0 or not args[2]):
            return x
        raise NotImplementedError(f"Uncovered floating operator: {name}")

    def reset_rope(self, encoder):
        from transformers.models.modernbert.modeling_modernbert import ModernBertRotaryEmbedding
        for module in encoder.modules():
            if isinstance(module, ModernBertRotaryEmbedding):
                for layer_type in module.layer_types:
                    inv, scale = module.compute_default_rope_parameters(module.config, "cpu", layer_type=layer_type)
                    setattr(module, f"{layer_type}_inv_freq", inv)
                    setattr(module, f"{layer_type}_attention_scaling", scale)
