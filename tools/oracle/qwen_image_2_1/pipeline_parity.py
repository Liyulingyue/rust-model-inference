#!/usr/bin/env python3
"""Real CLI vs pinned CPU Oracle: text, single/multiple references, CFG, Euler, RGBA."""
import argparse
from functools import partial
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import time
import zlib
from compare import png_rgba

if not __debug__:
    raise RuntimeError("Bitwise verification requires Python assertions (no -O/PYTHONOPTIMIZE)")

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def fixture(root, index, w, h):
    pixels = bytes(v for y in range(h) for x in range(w) for v in (x*255//(w-1), y*255//(h-1), (x+y+index*31)%256, 255 if x < w//2 else 160))
    def chunk(tag, payload):
        return struct.pack('>I', len(payload))+tag+payload+struct.pack('>I', zlib.crc32(tag+payload)&0xffffffff)
    png = root/f'reference-{index}.png'
    png.write_bytes(b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',w,h,8,6,0,0,0))+chunk(b'IDAT',zlib.compress(b''.join(b'\0'+pixels[y*w*4:(y+1)*w*4] for y in range(h))))+chunk(b'IEND',b''))
    rgba = root/f'reference-{index}.f32'
    rgba.write_bytes(b''.join(struct.pack('<f', pixels[4*i+c]/255) for c in range(4) for i in range(w*h)))
    f32 = lambda v: struct.unpack('<f', struct.pack('<f', v))[0]
    rgba.with_suffix('.vae.f32').write_bytes(b''.join(struct.pack('<f', f32(f32(f32(pixels[4*i+c]/255)*2)-1)) for c in range(4) for i in range(w*h)))
    return png, rgba, w, h


def run(command, trace, scalar, log):
    env = dict(os.environ)
    env.pop('RMI_SCALAR', None)
    env.pop('RMI_PARITY_FILTER', None)
    if scalar:
        env['RMI_SCALAR'] = '1'
    for key in ['RMI_PARITY_TRACE', 'QWEN_IMAGE_2_1_ORACLE_TRACE']:
        if trace is None:
            env.pop(key, None)
        else:
            env[key] = str(trace)
    started = time.monotonic()
    with log.open('w') as f:
        subprocess.run(list(map(str, command)), env=env, stdout=f, stderr=subprocess.STDOUT, check=True, cwd=REPO)
    return round(time.monotonic()-started, 3)


def check(mode, rust, oracle, *extra, relative_l2=None):
    tolerance = [] if relative_l2 is None else ['--relative-l2',str(relative_l2)]
    result = json.loads(subprocess.check_output([sys.executable,str(HERE/'compare.py'),mode,str(rust),str(oracle),*map(str,extra),*tolerance], cwd=REPO, text=True))
    print(json.dumps({k:v for k,v in result.items() if k != 'metrics'}), flush=True)
    return result


def named(trace, name, index=0):
    return Path([json.loads(s)['binary_path'] for s in trace.read_text().splitlines() if json.loads(s)['name']==name][index])


def split_text(trace, root):
    groups, group = [], []
    for line in trace.read_text().splitlines():
        row = json.loads(line)
        if row['name'] == 'qwen.text.tokens':
            if group: groups.append(group)
            group = []
        if row['name'].startswith(('qwen.text.', 'hidden_sequence.layer.')):
            group.append(line)
    if group: groups.append(group)
    paths = []
    for i, lines in enumerate(groups):
        path = root/f'rust-text-{i}.jsonl'
        path.write_text('\n'.join(lines)+'\n')
        paths.append(path)
    return paths


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dit')
    p.add_argument('--text')
    p.add_argument('--mmproj')
    p.add_argument('--vae')
    p.add_argument('--rust-bin', default=str(REPO/'target/release-fast/rust-model-inference'))
    p.add_argument('--production-rust-bin', help='also check the same CLI with a normal build, without tracing')
    p.add_argument('--oracle-bin', required=True, help='qwen-image-2-1-components from build_oracle.sh')
    p.add_argument('--out', default=str(REPO/'target/qwen-image-2-1-validation'))
    p.add_argument('--cases', nargs='+', choices=['context','text','edit1','edit2'], default=['text','edit1','edit2'])
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--steps', type=int)
    p.add_argument('--cfg', type=float)
    p.add_argument('--turbo', action='store_true', help='official fixed eight-step schedule and CFG=1 defaults')
    p.add_argument('--turbo-model-index', default=str(REPO/'tests/fixtures/qwen_image_2_1_turbo/model_index.json'))
    p.add_argument('--relative-l2', type=float, help='numerical error bound for reduced-precision computations')
    p.add_argument('--optimized', action='store_true', help='leave Rust optimizations enabled; select the reference build with --oracle-bin')
    args = p.parse_args()
    model = 'Qwen-Image-2.1-Turbo' if args.turbo else 'Qwen-Image-2.1'
    args.dit = args.dit or os.environ.get('QWEN_IMAGE_2_1_DIT', str(REPO/f'models/{model}-GGUF/{model.lower()}-Q8_0.gguf'))
    args.vae = args.vae or str(REPO/f'models/{model}/vae/diffusion_pytorch_model.safetensors')
    args.text = args.text or str(REPO/'models/Qwen3-VL-8B-Instruct-GGUF'/('Qwen3-VL-8B-Instruct-Q8_0.gguf' if args.turbo else 'Qwen3VL-8B-Instruct-Q8_0.gguf'))
    args.mmproj = args.mmproj or str(REPO/'models/Qwen3-VL-8B-Instruct-GGUF'/('mmproj-F16.gguf' if args.turbo else 'mmproj-Qwen3VL-8B-Instruct-F16.gguf'))
    args.steps = args.steps if args.steps is not None else (8 if args.turbo else 3)
    args.cfg = args.cfg if args.cfg is not None else (1.0 if args.turbo else 3.25)
    if 'context' in args.cases and args.cfg != 1:
        p.error('the explicit context case requires CFG 1')
    compare_stage = partial(check, relative_l2=args.relative_l2)
    root = Path(args.out).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.turbo:
        sigmas = json.loads(Path(args.turbo_model_index).read_text())['sample_sigmas'] + [0.0]
        assert args.steps == 8 and len(sigmas) == 9 and sigmas[0] == 1 and sigmas[-1] == 0
        schedule = root/'checkpoint-sigmas.f32'
        schedule.write_bytes(struct.pack('<9f',*sigmas))
        os.environ['QWEN_IMAGE_2_1_TURBO_SIGMAS'] = str(schedule)
    fixtures = [fixture(root,0,64,32), fixture(root,1,32,64)]
    context = root/'synthetic-context.f32'
    context.write_bytes(struct.pack('<8192f', *[((i % 101)-50)/100 for i in range(8192)]))
    results = []
    for case in args.cases:
        dst = root/case
        dst.mkdir(exist_ok=False)  # Never append a rerun to old checkpoint files.
        nref = {'context':0, 'text':0, 'edit1':1, 'edit2':2}[case]
        prompt = {'context':'','text':'A blue cat. 蓝猫 🐈','edit1':'Make it blue.','edit2':'Combine the two references. 蓝色 🐈'}[case]
        negative = '' if case == 'edit2' else 'low quality'
        rust = dst/'rust.jsonl'
        command = [args.rust_bin,'--model',args.dit,'--vae',args.vae,'--width','64','--height','32','--steps',args.steps,'--cfg',args.cfg,'--seed','42','--threads',args.threads,'--out',dst/'output.png']
        command += ['--qwen-context-file',context] if case == 'context' else ['--text-encoder',args.text,'--prompt',prompt,'--negative-prompt',negative]
        if args.turbo: command.append('--qwen-image-turbo')
        if nref:
            command += ['--mmproj',args.mmproj,'--image',fixtures[0][0]]
            for image in fixtures[1:nref]: command += ['--reference',image[0]]
        production_command = command.copy()
        print(f'{case}: real Rust CLI', flush=True)
        elapsed = run(command,rust,not args.optimized,dst/'rust.log')
        if case == 'context':
            assert named(rust,'qwen.input.context').read_bytes() == context.read_bytes(), 'explicit conditioning F32 input'
        rust_texts = split_text(rust,dst)
        expected_texts = 0 if case == 'context' else (1 if args.cfg == 1 else 2)
        assert len(rust_texts) == expected_texts
        oracle_texts = []
        stages = []
        for i, text in enumerate([prompt,negative][:expected_texts]):
            oracle = dst/f'oracle-text-{i}.jsonl'
            command = [args.oracle_bin,'text',args.text,text,args.threads]
            if nref:
                command.append(args.mmproj)
                for _,rgba,w,h in fixtures[:nref]: command += [rgba,w,h]
            run(command,oracle,not args.optimized,dst/f'oracle-text-{i}.log')
            stages.append(compare_stage('text',rust_texts[i],oracle,'--references',nref))
            if nref and i == 0: stages.append(compare_stage('vision',rust,oracle,'--references',nref))
            oracle_texts.append(oracle)
        oracle_latents = []
        for i,(_,rgba,w,h) in enumerate(fixtures[:nref]):
            oracle = dst/f'oracle-encode-{i}.jsonl'
            normalized = rgba.with_suffix('.vae.f32')
            assert named(rust,'qwen.vae.input',i).read_bytes() == normalized.read_bytes(), 'independent planar RGBA preprocessing'
            run([args.oracle_bin,'vae_encode',args.vae,normalized,w,h,args.threads],oracle,not args.optimized,dst/f'oracle-encode-{i}.log')
            stages.append(compare_stage('vae_encode',rust,oracle,'--reference',i))
            oracle_latents += [named(oracle,'qwen.vae.encoded'),w//16,h//16]
        noise = dst/'oracle-noise.jsonl'
        run([args.oracle_bin,'noise',42,4,2],noise,not args.optimized,dst/'oracle-noise.log')
        assert named(noise,'qwen.sample.noise').read_bytes() == named(rust,'qwen.sample.noise').read_bytes(), 'seeded MT19937 noise bits'
        oracle = dst/'oracle-sample.jsonl'
        positive_context = context if case == 'context' else named(oracle_texts[0],'qwen.text.context')
        negative_context = context if case == 'context' else named(oracle_texts[-1],'qwen.text.context')
        command = [args.oracle_bin,'sample_turbo' if args.turbo else 'sample',args.dit,named(noise,'qwen.sample.noise'),positive_context,negative_context,4,2,args.steps,args.cfg,args.threads]
        if nref: command += [named(oracle_texts[0],'qwen.text.image_slots'),named(oracle_texts[-1],'qwen.text.image_slots'),*oracle_latents]
        run(command,oracle,not args.optimized,dst/'oracle-sample.log')
        stages.append(compare_stage('sample',rust,oracle,'--evaluations',args.steps*(1 if args.cfg == 1 else 2)))
        decoded = dst/'oracle-decode.jsonl'
        run([args.oracle_bin,'vae',args.vae,named(oracle,'qwen.sample.latent',args.steps-1),4,2,args.threads],decoded,not args.optimized,dst/'oracle-decode.log')
        stages.append(compare_stage('vae',rust,decoded,'--png',dst/'output.png'))
        result = dict(case=case,scalar=not args.optimized,width=64,height=32,steps=args.steps,cfg=args.cfg,references=nref,rust_seconds=elapsed,stages=stages)
        if args.production_rust_bin:
            production_command[0] = args.production_rust_bin
            production_png = dst/'production.png'
            production_command[production_command.index('--out')+1] = production_png
            result['production_seconds'] = run(production_command,None,not args.optimized,dst/'production.log')
            assert png_rgba(production_png) == png_rgba(dst/'output.png'), 'normal build RGBA pixels'
        results.append(result)
        (root/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
        print(f'{case}: all checkpoints and PNG comparisons passed', flush=True)


if __name__ == '__main__':
    main()
