"""Pinned Transformers Oracle, optionally using independently decoded GGUF weights."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
import transformers.utils

SOURCE_SHA = "20b43f36ad15623d5a16cbc99847ec1b32dcb1e5ee81716682da40933fc2c7f9"


def save(path, value):
    if isinstance(value, tuple):
        value = value[0]
    value = value.detach().float().cpu().contiguous().numpy()
    value.astype('<f4').tofile(path)


def load_text_gguf(model, path):
    """Independently decode exported weights with upstream gguf, keeping HF kernels."""
    import gguf

    contract = json.loads((path.parent/'source-tensors.json').read_text())
    parameters = {'model.'+name: value for name, value in model.model.language_model.named_parameters()}
    parameters['lm_head.weight'] = model.lm_head.weight
    tensors = {tensor.name: tensor for tensor in gguf.GGUFReader(path).tensors}
    assert len(parameters) == len(tensors) == 399
    assert {contract[name]['gguf_name'] for name in parameters} == tensors.keys()
    with torch.no_grad():
        for name, parameter in parameters.items():
            tensor = tensors[contract[name]['gguf_name']]
            assert tuple(reversed(tensor.shape)) == tuple(parameter.shape), name
            values = gguf.quants.dequantize(tensor.data, tensor.tensor_type)
            assert values.size == parameter.numel() and np.isfinite(values).all(), name
            parameter.copy_(torch.from_numpy(values.copy().reshape(tuple(parameter.shape))))
    return {'path': str(path.resolve()), 'bytes': path.stat().st_size,
            'tensors': len(tensors), 'decoder': 'upstream gguf.quants.dequantize'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('model', type=Path)
    p.add_argument('out', type=Path)
    p.add_argument('--steps', type=int, default=8)
    p.add_argument('--templates-only', action='store_true')
    p.add_argument('--text-gguf', type=Path, help='Independent same-weight quantized Oracle')
    a = p.parse_args()
    assert hashlib.sha256((a.model/'modeling_llava_onevision2.py').read_bytes()).hexdigest() == SOURCE_SHA
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    # This host has flash_attn without its GPU extension; use the release's CPU SDPA fallback.
    transformers.utils.is_flash_attn_2_available = lambda: False
    processor = AutoProcessor.from_pretrained(a.model, trust_remote_code=True, local_files_only=True)
    a.out.mkdir(parents=True, exist_ok=True)
    samples = ['', 'Hello, world!', '中文 / café 🙂\n\t  spaces', '<|im_start|>user\nHi<|im_end|>']
    (a.out/'tokenizer.json').write_text(json.dumps([
        {'text': s, 'ids': processor.tokenizer.encode(s, add_special_tokens=False)} for s in samples], ensure_ascii=False, indent=2))
    template_cases = [
        {'messages':[{'role':'system','content':'Be brief.'},{'role':'user','content':'查上海天气'}],
         'tools':[{'type':'function','function':{'name':'weather','description':'天气','parameters':{'type':'object','properties':{'city':{'type':'string'},'scale':{'type':'number','minimum':1e-6,'maximum':1e16}}}}}], 'thinking':False},
        {'messages':[{'role':'user','content':[{'type':'image'},{'type':'text','text':'First'}]},
                     {'role':'assistant','content':'A'},
                     {'role':'user','content':[{'type':'image'},{'type':'text','text':'Second'}]}], 'tools':[], 'thinking':True},
        {'messages':[{'role':'user','content':'天气?'},
                     {'role':'assistant','content':'','tool_calls':[{'type':'function','function':{'name':'weather','arguments':{'city':'上海','scale':1e-7}}}]},
                     {'role':'tool','content':'Sunny'}], 'tools':[], 'thinking':False},
    ]
    for case in template_cases:
        case['ids'] = processor.tokenizer.apply_chat_template(case['messages'],tools=case['tools'],
            tokenize=True,add_generation_prompt=True,enable_thinking=case['thinking'])
    (a.out/'templates.json').write_text(json.dumps(template_cases,ensure_ascii=False,indent=2))
    if a.templates_only:
        return
    model, loading = AutoModelForCausalLM.from_pretrained(
        a.model, trust_remote_code=True, dtype=torch.float32,
        attn_implementation='eager', output_loading_info=True, local_files_only=True)
    assert not any(loading.get(k) for k in ['missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs']), loading
    model.eval()
    (a.out/'loading.json').write_text(json.dumps(loading, indent=2))
    weights = load_text_gguf(model, a.text_gguf) if a.text_gguf else {'source': 'pinned BF16 widened to F32'}
    (a.out/'weights.json').write_text(json.dumps(weights, indent=2))
    for name, size, thinking in [('text', None, True), ('square', (64,64), False),
                                  ('wide', (96,64), False), ('resize', (83,57), False)]:
        out = a.out/name
        out.mkdir(exist_ok=True)
        images = []
        content = 'What is 2 + 2? Answer briefly.'
        if size:
            w,h = size
            y,x = np.mgrid[:h,:w]
            rgb = np.stack([(x*3+y*7)%256, (x*11+y*2)%256, (x*5+y*13)%256], axis=-1).astype('uint8')
            images = [Image.fromarray(rgb)]
            images[0].save(out/'image.png')
            content = [{'type':'image'}, {'type':'text','text':'Describe the colors in this image briefly.'}]
        messages = [{'role':'user', 'content':content}]
        text = processor.tokenizer.apply_chat_template(messages, tools=[], tokenize=False,
            add_generation_prompt=True, enable_thinking=thinking)
        raw_ids = processor.tokenizer.encode(text, add_special_tokens=False)
        inputs = processor(text=[text], images=images or None, return_tensors='pt')
        if images:
            inputs['patch_positions'] = torch.cat([torch.cartesian_prod(
                torch.arange(int(t)),torch.arange(int(h)),torch.arange(int(w)))
                for t,h,w in inputs['image_grid_thw']])
        (out/'prompt.txt').write_text(text)
        case = {'messages':messages, 'tools':[], 'thinking':thinking, 'raw_ids':raw_ids,
                'input_ids':inputs['input_ids'][0].tolist(), 'steps':a.steps,
                'images':['image.png'] if images else []}
        if images:
            save(out/'pixels.f32', inputs['pixel_values'])
            case['grid'] = inputs['image_grid_thw'].tolist()
        (out/'case.json').write_text(json.dumps(case, ensure_ascii=False, indent=2))
        hooks = []
        visual = model.model.visual
        for tag, module in [('patch', visual.embeddings), ('pre', visual.layernorm_pre),
                            *[(f'vit{i}', layer) for i,layer in enumerate(visual.encoder.layers)],
                            ('projected', visual.merger)]:
            hooks.append(module.register_forward_hook(lambda m,args,result,tag=tag: save(out/f'{tag}.f32', result)))
        # Preserve final-token rows before cache continuation overwrites the hooks.
        for i in [0,17,35]:
            hooks.append(model.model.language_model.layers[i].register_forward_hook(
                lambda m,args,result,i=i: save(out/f'lm{i}.f32', (result[0] if isinstance(result,tuple) else result)[:,-1,:])))
        hooks.append(model.model.language_model.norm.register_forward_hook(
            lambda m,args,result: save(out/'norm.f32', result[:,-1,:])))
        with torch.inference_mode():
            start = time.perf_counter()
            output = model(**inputs, use_cache=True, logits_to_keep=1)
            prefill_seconds = time.perf_counter() - start
            for hook in hooks:
                hook.remove()
            generated = []
            decode_seconds = 0.0
            length = inputs['input_ids'].shape[1]
            for step in range(a.steps):
                logits = output.logits[0,-1]
                save(out/f'logits{step}.f32', logits)
                token = int(logits.argmax())
                generated.append(token)
                if step+1 < a.steps:
                    start = time.perf_counter()
                    output = model(input_ids=torch.tensor([[token]]),
                        position_ids=torch.tensor([[length+step]]),
                        attention_mask=torch.ones(1,length+step+1,dtype=torch.long),
                        cache_position=torch.tensor([length+step]),
                        past_key_values=output.past_key_values, use_cache=True, logits_to_keep=1)
                    decode_seconds += time.perf_counter() - start
        (out/'greedy.json').write_text(json.dumps(generated))
        (out/'timing.json').write_text(json.dumps({'threads': 8, 'prefill_seconds': prefill_seconds,
            'decode_seconds': decode_seconds, 'decode_steps': max(a.steps-1, 0),
            'text': processor.tokenizer.decode(generated)}, indent=2))
        print(name, generated, processor.tokenizer.decode(generated), flush=True)


if __name__ == '__main__':
    main()
