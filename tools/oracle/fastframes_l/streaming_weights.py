"""Materialize official modules one block at a time on memory-limited hosts."""
import numpy as np
import torch
import gguf


class StreamingWeights:
    def __init__(self, states, device, dtype, gguf_path=None):
        self.states = dict(zip(['dit', 'refiner', 'decoder', 'text'], states))
        self.device, self.dtype = device, dtype
        self.reader = gguf.GGUFReader(str(gguf_path)) if gguf_path else None
        self.tensors = {t.name: t for t in self.reader.tensors} if self.reader else {}
        self.formats = {name: int(t.tensor_type) for name, t in self.tensors.items()}
        self.refresh = lambda: None

    def value(self, component, name):
        original = self.states[component][name]
        if not self.reader:
            return original.to(device=self.device, dtype=self.dtype)
        grouped = 'heads_grouped.gemms.0.'
        if component == 'refiner' and name.startswith(grouped):
            field = name[len(grouped):]
            suffix = '.weight' if field.endswith('_w') else '.bias' if field.endswith('_b') else ''
            field = field[:-2] if suffix else field
            parts = []
            for layer in range(original.shape[0]):
                tensor = self.tensors[f'refiner.heads.{layer}.{field}{suffix}']
                shape = (original.shape[2], original.shape[1]) if suffix == '.weight' else (original.shape[2],)
                array = gguf.quants.dequantize(tensor.data, tensor.tensor_type).reshape(shape)
                part = torch.from_numpy(np.array(array, dtype=np.float32, copy=True))
                parts.append(part.T if suffix == '.weight' else part)
            value = torch.stack(parts).reshape(original.shape)
        else:
            key = f'{component}.{name}'
            # FF-S's converter prefixes the plain decoder's keys with
            # ``pixel_head.``; FF-L's bridged decoder names them directly.
            if component == 'decoder' and key not in self.tensors:
                alternative = f'decoder.pixel_head.{name}'
                if alternative in self.tensors:
                    key = alternative
            tensor = self.tensors[key]
            array = gguf.quants.dequantize(tensor.data, tensor.tensor_type).reshape(tuple(original.shape))
            value = torch.from_numpy(np.array(array, dtype=np.float32, copy=True))
        return value.to(device=self.device, dtype=self.dtype)

    def load(self, module, component, prefix='', deferred=()):
        state = {name: self.value(component, prefix + name) for name in module.state_dict()
                 if not name.startswith(deferred)}
        result = module.load_state_dict(state, strict=not deferred, assign=True)
        if result.unexpected_keys or any(not k.startswith(deferred) for k in result.missing_keys):
            raise ValueError(f'incomplete streamed state: {result}')
        module.eval().requires_grad_(False)

    def initialize(self, modules):
        for component, module in zip(self.states, modules):
            self.load(module, component, deferred=('blocks.',) if component in {'dit', 'text'} else ())

    def attach_blocks(self, dit, text, fast):
        """Stream per-block weights; ``fast`` (fused QKV/patch views) is optional."""
        for component, model in [('dit', dit), ('text', text)]:
            for index, block in enumerate(model.blocks):
                def before(module, args, component=component, index=index):
                    self.load(module, component, f'blocks.{index}.')
                    if component == 'dit' and fast is not None:
                        sa = module.self_attn
                        fast.qkv[index] = (torch.cat([sa.q.weight, sa.k.weight, sa.v.weight]).detach(),
                                           torch.cat([sa.q.bias, sa.k.bias, sa.v.bias]).detach())
                    self.refresh()

                def after(module, args, output, component=component, index=index):
                    if component == 'dit' and fast is not None:
                        fast.qkv[index] = None
                    module.to_empty(device='meta')

                block.register_forward_pre_hook(before)
                block.register_forward_hook(after)

    def text_cache(self, dit, embeds):
        context = dit.embed_text(embeds)
        cache = []
        for index, block in enumerate(dit.blocks):
            module = block.cross_attn
            self.load(module, 'dit', f'blocks.{index}.cross_attn.')
            self.refresh()
            k, v = module.text_kv(context)
            cache.append({'k': k, 'v': v})
            module.to_empty(device='meta')
        return cache
