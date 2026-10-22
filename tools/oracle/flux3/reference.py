"""CPU Oracle for BFL bfc8889a. Run from repository root; see README.md."""
import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoConfig, AutoTokenizer
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

COMMIT = 'bfc8889a4497f846416f253a3972a1941ac698b9'
SOURCE_HASHES = {
    'dit.py': 'd4c0d2aea5282ae63ac32a1f3029a738ddab951bb1033d6f3a2e9bd192c89934',
    'autoencoder.py': '960e2a98046bcb78f2d36fbbd4848db726f14810c7887485f7e2682346f95189',
    'sampling.py': '22be0d823fb6a2c5bc993c77fe0836ca729219d34cc08e86d01a398e04bdcdf5',
    'pipeline.py': 'd28d60ca680241f7b5ae1b90d83eb6218af5ea51b91ee69214bd266ec10bccb1',
    'text_encoder.py': '3bce784f43b8e8d9eee0dab75d928006398efd80e77f9f179bba6ddd28ffe74b',
}
LAYERS = (4, 8, 12, 16, 20, 24, 28, 32)


class Trace:
    def __init__(self, directory):
        directory.mkdir(parents=True, exist_ok=False)
        self.directory, self.records, self.counts = directory, [], {}

    def write(self, name, tensor):
        values = tensor.detach().float().cpu().contiguous().numpy()
        occurrence = self.counts.get(name, 0)
        self.counts[name] = occurrence + 1
        path = self.directory / f'{name}.{occurrence}.f32'
        values.astype('<f4').tofile(path)
        self.records.append(dict(name=name, occurrence=occurrence, shape=list(values.shape), binary_path=str(path.resolve())))
        (self.directory / 'trace.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in self.records))


def text_encode(root, prompt, dtype, trace):
    tokenizer = AutoTokenizer.from_pretrained(root / 'text_encoder', local_files_only=True)
    tokenizer.padding_side = 'right'
    rendered = tokenizer.apply_chat_template([dict(role='user', content=prompt)], tokenize=False, add_generation_prompt=True)
    real = len(tokenizer(rendered, truncation=True, max_length=4096)['input_ids'])
    length = min((real + 79)//80 * 80, 4096)
    inputs = tokenizer(rendered, return_tensors='pt', padding='max_length', truncation=True, max_length=length)
    trace.write('flux3.token_ids', inputs['input_ids'][0])
    config = AutoConfig.from_pretrained(root / 'text_encoder', local_files_only=True).text_config
    config._attn_implementation = 'eager'
    with torch.device('meta'):
        model = Qwen3VLTextModel(config)
    weights = {}
    for shard in sorted((root/'text_encoder').glob('model-*.safetensors')):
        with safe_open(shard, framework='pt') as source:
            for name in source.keys():
                if name.startswith('model.language_model.'):
                    weights[name.removeprefix('model.language_model.')] = source.get_tensor(name).to(dtype)
    model.load_state_dict(weights, strict=True, assign=True)
    # Non-persistent rotary buffers are absent from the checkpoint's state dict.
    model.rotary_emb = type(model.rotary_emb)(config, device='cpu')
    model.eval()
    del weights
    captured = {}
    def capture(index):
        def hook(module, args, output):
            captured[index] = output.detach()
        return hook
    for index in LAYERS:
        model.layers[index-1].register_forward_hook(capture(index))
    mask = inputs['attention_mask']
    positions = (mask.cumsum(-1)-1).masked_fill(mask == 0, 1)
    model(input_ids=inputs['input_ids'], attention_mask=mask, position_ids=positions[None].expand(4,-1,-1), use_cache=False)
    tokens = torch.stack([captured[k] for k in LAYERS], dim=1).permute(0,2,1,3).reshape(length,-1).to(torch.bfloat16)
    trace.write('flux3.text', tokens)
    return tokens


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model_dir', type=Path)
    parser.add_argument('output_dir', type=Path)
    parser.add_argument('--oracle-source', type=Path, required=True)
    parser.add_argument('--prompt', default='A red cube on a white table.')
    parser.add_argument('--side', type=int, default=64)
    parser.add_argument('--steps', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--dtype', choices=['f32','bf16'], default='f32')
    parser.add_argument('--reference', type=Path, action='append', default=[])
    parser.add_argument('--text-only', action='store_true')
    parser.add_argument('--vae-only', action='store_true')
    args = parser.parse_args()
    for name, expected in SOURCE_HASHES.items():
        actual = hashlib.sha256((args.oracle_source/'src/flux3'/name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'{name} does not match BFL {COMMIT}')
    sys.path.insert(0, str(args.oracle_source/'src'))
    from flux3.autoencoder import AutoEncoder, latent_ids
    from flux3.dit import DiT, Packed, MODES
    from flux3.sampling import timesteps
    torch.set_num_threads(args.threads)
    dtype = torch.float32 if args.dtype == 'f32' else torch.bfloat16
    trace = Trace(args.output_dir)
    (args.output_dir/'run.json').write_text(json.dumps(dict(arguments=vars(args), oracle_commit=COMMIT, torch=torch.__version__, numpy=np.__version__, transformers=__import__('transformers').__version__), default=str, indent=2)+'\n')
    with torch.inference_mode():
        if args.vae_only:
            ae = AutoEncoder().eval()
            ae.load_state_dict(load_file(str(args.model_dir/'ae.safetensors')), strict=True)
            side = args.side
            image = np.fromfunction(lambda y,x,c: (x*3+y*5+c*17)%256, (side,side,3)).astype(np.uint8)
            pixels = torch.from_numpy(image).permute(2,0,1).float()/127.5-1
            tokens, _ = ae.encode(pixels)
            trace.write('flux3.reference.0',tokens.to(torch.bfloat16))
            h = side//16
            noise = torch.randn((h*h,128),generator=torch.Generator().manual_seed(args.seed)).to(torch.bfloat16).float()
            trace.write('flux3.noise',noise)
            z = noise.reshape(h,h,128).permute(2,0,1)[None]
            z = z*(ae.bn.running_var.reshape(1,-1,1,1)+1e-4).sqrt()+ae.bn.running_mean.reshape(1,-1,1,1)
            z = z.reshape(1,32,2,2,h,h).permute(0,1,4,2,5,3).reshape(1,32,h*2,h*2)
            trace.write('flux3.vae_input',z[0])
            rgb = ae.decoder(z)[0]
            trace.write('z_image.vae.rgb_channels',rgb.reshape(side,side,3))
            Image.fromarray(((rgb.clamp(-1,1)+1)*127.5).to(torch.uint8).permute(1,2,0).numpy()).save(args.output_dir/'output.png')
            return
        text = text_encode(args.model_dir, args.prompt, dtype, trace)
        gc.collect()
        if args.text_only:
            return
        ae = AutoEncoder().eval()
        ae.load_state_dict(load_file(str(args.model_dir/'ae.safetensors')), strict=True)
        refs, ref_ids = [], []
        for index, path in enumerate(args.reference):
            image = torch.from_numpy(np.array(Image.open(path).convert('RGB'))).permute(2,0,1).float()/127.5-1
            tokens, ids = ae.encode(image)
            tokens, ids = tokens.to(torch.bfloat16), ids.float()
            ids[:,1:3] *= 0.5
            ids[:,3] = -(1+index)
            trace.write(f'flux3.reference.{index}', tokens)
            refs.append(tokens); ref_ids.append(ids)
        h = args.side//16
        noise = torch.randn((h*h,128), generator=torch.Generator().manual_seed(args.seed)).to(torch.bfloat16).float()
        trace.write('flux3.noise', noise)
        image_positions = latent_ids(h,h).float(); image_positions[:,1:3] *= 0.5
        text_positions = torch.zeros(text.shape[0],4); text_positions[:,3] = torch.arange(text.shape[0])
        data = dict(image=noise, image_cond=torch.cat(refs) if refs else torch.zeros(0,128), text=text.float())
        ids = dict(image=image_positions, image_cond=torch.cat(ref_ids) if refs else torch.zeros(0,4), text=text_positions)
        packed = Packed(data=data, ids=ids, timesteps={m:torch.zeros(len(data[m]), dtype=dtype) for m in MODES}, mode_indices={m:[(0,len(data[m]))] for m in MODES})
        with torch.device('meta'):
            dit = DiT(flow_map=True)
        weights = {k:v.to(dtype) for k,v in load_file(str(args.model_dir/'model.safetensors')).items()}
        dit.load_state_dict(weights, strict=True, assign=True); del weights
        dit.eval()
        def hook(name):
            return lambda module, inputs, output: trace.write(name, output)
        dit.emb_in['image'].register_forward_hook(hook('flux3.image.input'))
        if refs:
            dit.emb_in['image_cond'].register_forward_hook(hook('flux3.image_cond.input'))
        dit.txt_in.register_forward_hook(hook('flux3.txt.input'))
        for name, module in dit.named_modules():
            if name.startswith(('content_mode_blocks.', 'txt_mode_blocks.', 'single_blocks.')) and name.split('.')[-1].isdigit():
                if 'image_cond' not in name or refs:
                    module.register_forward_hook(hook(name))
        schedule = timesteps(args.steps,16.0)
        for step, (current, target) in enumerate(zip(schedule[:-1],schedule[1:])):
            packed.timesteps['image'].fill_(current.item())
            packed.data = {m:v.to(dtype) for m,v in packed.data.items()}
            velocity = dit(packed, timesteps_ref=target.reshape(1)).float()
            trace.write('flux3.velocity', velocity)
            noise = noise + (target-current)*velocity
            packed.data['image'] = noise
            trace.write(f'flux3.latent.{step}',noise)
        del dit
        gc.collect()
        shape = (h,h,128)
        z = noise.reshape(shape).permute(2,0,1)[None]
        z = z * (ae.bn.running_var.reshape(1,-1,1,1)+1e-4).sqrt() + ae.bn.running_mean.reshape(1,-1,1,1)
        z = z.reshape(1,32,2,2,h,h).permute(0,1,4,2,5,3).reshape(1,32,2*h,2*h)
        trace.write('flux3.vae_input',z[0])
        rgb = ae.decoder(z)[0]
        # Shared VAE trace labels H,W,C but its payload is planar (legacy schema).
        trace.write('z_image.vae.rgb_channels',rgb.reshape(args.side,args.side,3))
        pixels = ((rgb.clamp(-1,1)+1)*127.5).to(torch.uint8).permute(1,2,0).numpy()
        Image.fromarray(pixels).save(args.output_dir/'output.png')


if __name__ == '__main__':
    main()
