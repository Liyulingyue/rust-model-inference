#!/usr/bin/env python3
"""Trace a pinned local d3 model or compare its checkpoints with native Rust.

.venv/bin/python tools/oracle/vela3/d3_oracle.py trace --base models/d3-lite \
    --vela models/vela3-lite --adapter span --text 'A short prompt' --out .codex/d3-oracle.jsonl
.venv/bin/python tools/oracle/vela3/d3_oracle.py compare .codex/d3-oracle.jsonl .codex/d3-rust.jsonl

Build the native trace harness with:
  cargo +1.98.1 test --profile release-fast --lib --features parity-trace --no-run
Then pass the reported lib-test executable to `suite --native PATH --base ... --vela ... --out ...`.
`suite --dtype float32` keeps the real BF16 weight values but disables native activation rounding;
this isolates graph correctness without writing another multi-GB checkpoint. Float32 suites use
pointwise atol=5e-4, rtol=1e-4. BF16 text suites use max_abs=0.75, RMSE<=0.06, cosine>=0.9995 per
activation checkpoint; logits use atol=0.1, rtol=0.02 and require the same argmax. Tokens and mRoPE
always match exactly. Every threshold can be overridden explicitly and is saved with the report.
For a small four-step causal greedy check, use `suite --synthetic --causal --generation-steps 4`
with a scratch --base directory, --vela models/vela3-lite, --dtype float32 and --atol 2e-6 --rtol 1e-6.
For the turbo prefix mask, use `suite --synthetic --state-tokens 2 --adapters` with the same small
FP32 fixture. Synthetic traces use the saved pinned runtime when available and record its actual SHA.
For a real 27B text checkpoint, add `--dtype float32 --stream-f32-weights` to keep stored weights
BF16 and temporarily cast one decoder layer at a time; compute and comparison remain F32.

The reused image tower has FP32 activations and F16-lookup GELU. It is an approximate path: a real
lite image trace does not meet the BF16 text envelope. `--images FILE --vision-detail` records the
projection plus every vision layer to make that limitation measurable, rather than hiding it behind
a looser default. `vision-fixture` reproduces the independent tiny Transformers vision golden.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
import sys
from collections import defaultdict
from pathlib import Path


def load_module(path):
    name = f"vela_oracle_{path.stem}_{hashlib.sha256(str(path).encode()).hexdigest()[:12]}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


class Trace:
    def __init__(self, path):
        self.path = path.absolute()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.files = self.path.with_suffix('.f32')
        self.files.mkdir(exist_ok=True)
        self.stream = self.path.open('w', encoding='utf-8')
        self.count = defaultdict(int)

    def record(self, record):
        self.stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
        self.stream.flush()

    def emit(self, name, value, layer=None):
        import numpy as np
        values = value.detach().float().cpu().contiguous().numpy()
        if values.ndim >= 2 and values.shape[0] == 1:
            values = values[0]
        occurrence = self.count[name]
        self.count[name] += 1
        binary = self.files / f'{name}.{occurrence:04}.f32'
        values.astype('<f4', copy=False).tofile(binary)
        self.record(dict(name=name, layer=layer, occurrence=occurrence,
                         shape=list(values.shape), len=int(values.size),
                         finite=bool(np.isfinite(values).all()), binary_path=str(binary)))


def synthetic(root, dtype, prefix=False):
    """A small full-graph fixture using the official transformers model classes."""
    import torch
    from safetensors.torch import save_file
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
    root.mkdir(parents=True, exist_ok=True)
    # Preserve turbo's 6:1 GQA and 3:1 delta-head groups in the prefix fixture.
    config = Qwen3_5TextConfig(hidden_size=4, intermediate_size=8, num_hidden_layers=2,
        num_attention_heads=6 if prefix else 2, num_key_value_heads=1, head_dim=2, vocab_size=8,
        linear_num_key_heads=1, linear_num_value_heads=3 if prefix else 2, linear_key_head_dim=2,
        linear_value_head_dim=2, linear_conv_kernel_dim=4, rms_norm_eps=1e-6,
        max_position_embeddings=32, layer_types=['linear_attention', 'full_attention'],
        rope_parameters={'rope_type': 'default', 'rope_theta': 10000.,
                         'partial_rotary_factor': 1., 'mrope_section': [1, 0, 0]},
        tie_word_embeddings=True, eos_token_id=7, attention_bias=False)
    config._attn_implementation = 'sdpa'
    model = Qwen3_5TextModel(config).to(dtype=dtype).eval()

    def weight(name, shape):
        n = 1
        for d in shape:
            n *= d
        salt = sum(name.encode())
        return ((torch.arange(n) + salt) % 17 - 8).float().mul_(0.025).reshape(shape)

    tensors = {f'language_model.{name}': weight(f'language_model.{name}', p.shape).to(dtype)
               for name, p in model.state_dict().items()}
    model.load_state_dict({n.removeprefix('language_model.'): p for n, p in tensors.items()})
    save_file(tensors, root / 'model.safetensors')
    (root / 'config.json').write_text(json.dumps(dict(model_type='qwen3_5',
        text_config=config.to_dict(), tie_word_embeddings=True)))
    (root / 'decision_config.json').write_text(json.dumps(dict(format_version=1, pooling='last',
        readout_dtype='float32', attention_mode='prefix_bidir' if prefix else 'noncausal_full_attention',
        temperature=1.)))
    save_file({'weight': weight('weight', [255, 4])}, root / 'readout.safetensors')
    lora = {}
    for module in ['layers.0.linear_attn.in_proj_qkv',
                   'layers.1.self_attn.q_proj', 'layers.1.mlp.down_proj']:
        projection = model.get_submodule(module)
        lora[module + '.A'] = weight(module + '.A', [2, projection.in_features])
        lora[module + '.B'] = weight(module + '.B', [projection.out_features, 2])
    save_file(lora, root / 'span_lora.safetensors', metadata={'r': '2', 'alpha': '3'})
    (root / 'vela3_config.json').write_text(json.dumps(dict(adapters={'span': {
        'file': 'span_lora.safetensors', 'r': 2, 'alpha': 3, 'modules': 3}})))
    return model


def run_trace(args):
    if args.causal and not args.synthetic:
        config = json.loads((args.base / 'config.json').read_text())
        vela = json.loads((args.vela / 'vela3_config.json').read_text())
        tied = config.get('text_config', {}).get('tie_word_embeddings')
        if not isinstance(tied, bool):
            tied = config.get('tie_word_embeddings')
        if tied is not True and vela.get('definitions', {}).get('writer') != 'grafted_head':
            raise ValueError('causal tracing requires tied embeddings or the configured grafted LM head')
    with contextlib.ExitStack() as scope:
        return _run_trace(args, scope)


def _run_trace(args, scope):
    import torch
    import transformers
    from safetensors.torch import load_file
    torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)
    os.environ['D3_FAST'] = '0'
    base, vela = args.base.absolute(), args.vela.absolute()
    dtype = torch.float32 if args.dtype == 'float32' else torch.bfloat16
    if args.stream_f32_weights and (args.dtype != 'float32' or args.synthetic or args.images):
        raise ValueError('stream-f32-weights requires a real float32 text trace')
    storage_dtype = torch.bfloat16 if args.stream_f32_weights else dtype
    checked_bf16_tensors = None
    if args.stream_f32_weights:
        from stream_f32 import require_bf16_checkpoint
        checked_bf16_tensors = require_bf16_checkpoint(base)
    if args.images and (args.synthetic or args.causal):
        raise ValueError('image tracing requires a local checkpoint and a noncausal prompt pass')
    if args.generation_steps is not None and (not args.causal or not 1 <= args.generation_steps <= 40):
        raise ValueError('generation_steps requires --causal and 1..40 greedy steps')
    if args.synthetic:
        model = synthetic(base, dtype, prefix=args.state_tokens is not None)
        vela = base
        runtime_name = 'd3-turbo' if args.state_tokens is not None else 'd3-lite'
        runtime_root = Path('models') / runtime_name
        runtime_path = runtime_root / 'pinned' / 'd3_runtime.py'
        if not runtime_path.is_file():
            runtime_path = runtime_root / 'd3_runtime.py'
        runtime = load_module(runtime_path.absolute())
        ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        input_text = None
    else:
        from transformers import AutoTokenizer
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model
        runtime = load_module(base / 'd3_runtime.py')
        tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
        input_text = args.text
        if args.ids:
            values = json.loads(Path(args.ids).read_text())
            if isinstance(values, dict):
                values = values['token_ids']
            ids = torch.tensor([values], dtype=torch.long)
        else:
            ids = tokenizer(args.text, add_special_tokens=False, return_tensors='pt')['input_ids']
        backbone = Qwen3_5Model.from_pretrained(base, dtype=storage_dtype, attn_implementation='sdpa',
                                              local_files_only=True).eval()
        model = backbone.language_model
    unexpanded_ids = ids[0].tolist()
    vela_config = json.loads((vela / 'vela3_config.json').read_text())
    adapters = load_module((args.vela if not args.synthetic else Path('models/vela3-lite')).absolute() / 'vela3_adapters.py')
    if args.adapter:
        spec = vela_config['adapters'][args.adapter]
        info = adapters.attach(model, args.adapter, vela / spec['file'])
        if any(info[field] != spec[field] for field in ['r', 'alpha', 'modules']):
            raise ValueError(f'adapter {args.adapter} does not match vela3_config.json: {info}')
    if args.stream_f32_weights:
        from stream_f32 import stream_f32
        scope.enter_context(stream_f32(model))
    trace = Trace(args.out)
    source_paths = [base / 'config.json', base / 'decision_config.json', base / 'd3_runtime.py',
                    base / 'preprocessor_config.json', base / 'MODEL_MANIFEST.json',
                    base / 'readout.safetensors', vela / 'vela3_adapters.py', vela / 'vela3_config.json',
                    Path(runtime.__file__)]
    if args.stream_f32_weights:
        source_paths.append(Path(__file__).with_name('stream_f32.py'))
    if args.adapter:
        source_paths.append(vela / vela_config['adapters'][args.adapter]['file'])
    if args.causal and vela_config.get('definitions', {}).get('writer') == 'grafted_head':
        source_paths.append(vela / vela_config['definitions']['lm_head_file'])
    trace.record(dict(kind='oracle', torch=torch.__version__, transformers=transformers.__version__,
        dtype=str(dtype), storage_dtype=str(storage_dtype), stream_f32_weights=args.stream_f32_weights,
        checked_bf16_source_tensors=checked_bf16_tensors,
        adapter=args.adapter, text=input_text, unexpanded_token_ids=unexpanded_ids,
        images=[str(p.absolute()) for p in args.images], state_tokens=args.state_tokens,
        causal=args.causal, generation_steps=args.generation_steps,
        sources={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths if p.is_file()}))
    hooks = []
    for index, layer in enumerate(model.layers):
        hooks.append(layer.input_layernorm.register_forward_hook(
            lambda module, inputs, output, i=index: trace.emit('vela3.d3.input_norm', output, i)))
        hooks.append(layer.post_attention_layernorm.register_forward_pre_hook(
            lambda module, inputs, i=index: trace.emit('vela3.d3.attention_residual', inputs[0], i)))
        hooks.append(layer.register_forward_hook(
            lambda module, inputs, output, i=index: trace.emit('vela3.d3.layer', output, i)))
    encoder = getattr(runtime, 'PromptEncoder', None)
    mask = torch.ones_like(ids)
    embeds = model.embed_tokens(ids)
    mrope = None
    state_tokens = args.state_tokens
    if args.images:
        from PIL import Image
        import torchvision
        processor = runtime.load_processor(base)
        runtime.linearize_patch_embed(backbone)
        hooks.append(backbone.visual.patch_embed.register_forward_hook(
            lambda module, inputs, output: trace.emit('omni.vision.patch_bias', output.unsqueeze(0))))
        hooks.append(backbone.visual.blocks[0].register_forward_pre_hook(
            lambda module, inputs: trace.emit('omni.vision.inp_pos_emb', inputs[0].unsqueeze(0))))
        for index, block in enumerate(backbone.visual.blocks):
            hooks.append(block.register_forward_hook(
                lambda module, inputs, output, i=index: trace.emit('omni.vision.layer_out', output.unsqueeze(0), i)))
        # An explicit ID fixture contains one image token per image before processor expansion.
        image_text = tokenizer.decode(unexpanded_ids, skip_special_tokens=False) if args.ids else args.text
        inputs = processor(text=[image_text], images=[Image.open(p).convert('RGB') for p in args.images],
                           padding=True, return_tensors='pt')
        ids, mask = inputs['input_ids'], inputs['attention_mask']
        embeds = model.embed_tokens(ids)
        pooled = backbone.get_image_features(inputs['pixel_values'], inputs['image_grid_thw'],
                                             return_dict=True).pooler_output
        for feature in pooled:
            trace.emit('vela3.d3.vision.projected', feature.unsqueeze(0))
        features = torch.cat(pooled, dim=0).to(embeds.dtype)
        slots, _ = backbone.get_placeholder_mask(ids, inputs_embeds=embeds, image_features=features)
        embeds = embeds.masked_scatter(slots, features)
        mrope = backbone.compute_3d_position_ids(input_ids=ids, image_grid_thw=inputs['image_grid_thw'],
            video_grid_thw=None, inputs_embeds=embeds, attention_mask=mask,
            past_key_values=None, mm_token_type_ids=inputs['mm_token_type_ids'])
        trace.record(dict(kind='image_processor', processor=type(processor.image_processor).__name__,
            torchvision=torchvision.__version__, grid_thw=inputs['image_grid_thw'].tolist()))
        if state_tokens is not None:
            counts = iter((inputs['image_grid_thw'].prod(dim=1) // processor.image_processor.merge_size**2).tolist())
            state_tokens = sum(next(counts) if i == backbone.config.image_token_id else 1
                               for i in unexpanded_ids[:state_tokens])
    if args.causal:
        definition = vela_config.get('definitions', {})
        if definition.get('writer') == 'grafted_head':
            head = load_file(vela / definition['lm_head_file'])['lm_head.weight'].to(dtype)
        else:
            head = model.embed_tokens.weight
    else:
        head = load_file(base / 'readout.safetensors')['weight'].float()
        temperature = json.loads((base / 'decision_config.json').read_text()).get('temperature', 1.)
    generated = []
    for _ in range(args.generation_steps or 1):
        trace.record(dict(name='vela3.d3.tokens', shape=[ids.numel()], len=ids.numel(), token_ids=ids[0].tolist()))
        if mrope is not None:
            trace.emit('vela3.d3.positions', mrope[:, 0, :].T.unsqueeze(0))
        trace.emit('vela3.d3.embedding', embeds)
        with adapters.active(args.adapter):
            if args.causal:
                hidden = model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
            elif encoder is not None:
                extra = {}
                gated_full = args.adapter and vela_config.get('decisions', {}).get('attention') == 'noncausal'
                if state_tokens is not None and not gated_full:
                    extra['blocks'] = runtime.StateBlocks.of(ids.shape[-1], [ids.shape[-1]], [state_tokens])
                if mrope is not None:
                    extra['mrope'] = mrope
                hidden = encoder(model)(embeds, mask, **extra)
            else:
                runtime.enable_noncausal_full_attention(model)
                hidden = model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
        trace.emit('vela3.d3.hidden', hidden)
        if args.causal:
            logits = torch.nn.functional.linear(
                hidden[:, -1], head.float() if args.stream_f32_weights else head).float()
            trace.emit('vela3.d3.generation_logits', logits)
        else:
            logits = (hidden[:, -1].float() @ head.T)[0, :args.options] / temperature
            trace.emit('vela3.d3.logits', logits)
        if args.generation_steps is not None:
            token = int(logits.argmax())
            if token == model.config.eos_token_id:
                break
            generated.append(token)
            ids = torch.cat([ids, torch.tensor([[token]], dtype=torch.long)], dim=-1)
            mask = torch.ones_like(ids)
            embeds = model.embed_tokens(ids)
    if args.generation_steps is not None:
        trace.record(dict(name='vela3.d3.generated_tokens', shape=[len(generated)],
                          len=len(generated), token_ids=generated))
    for hook in hooks:
        hook.remove()
    print(json.dumps(dict(output=str(args.out), tokens=ids.numel(), generated_token_ids=generated,
                          logits=logits.flatten()[:8].tolist())))


def comparison(args):
    import numpy as np

    def records(path):
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        vision_names = {'omni.vision.patch_bias', 'omni.vision.inp_pos_emb', 'omni.vision.layer_out'}
        return [r for r in rows if 'name' in r and (r['name'].startswith('vela3.d3.') or
                (args.vision_detail and r['name'] in vision_names))]

    expected, actual = records(args.oracle), records(args.native)
    if len(expected) != len(actual):
        raise SystemExit(f'checkpoint count mismatch: oracle {len(expected)}, native {len(actual)}')
    reports = []
    for a, b in zip(expected, actual):
        if (a['name'], a.get('layer'), a['shape']) != (b['name'], b.get('layer'), b['shape']):
            raise SystemExit(f'checkpoint order/shape mismatch: {a} vs {b}')
        if 'token_ids' in a:
            if a['token_ids'] != b.get('token_ids'):
                raise SystemExit('token IDs differ')
            continue
        x = np.fromfile(a['binary_path'], dtype='<f4')
        y = np.fromfile(b['binary_path'], dtype='<f4')
        if x.size != a['len'] or y.size != b['len']:
            raise SystemExit('checkpoint payload length differs from shape')
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise SystemExit(f"nonfinite checkpoint: {a['name']}")
        error = np.abs(x - y)
        rmse = float(np.sqrt(np.mean(error * error)))
        cosine = float(np.dot(x, y) / max(float(np.linalg.norm(x) * np.linalg.norm(y)), 1e-30))
        maximum = float(error.max(initial=0))
        if a['name'] == 'vela3.d3.positions':
            good = bool(np.array_equal(x, y))
        elif args.profile == 'bf16' and not a['name'].endswith('logits'):
            good = maximum <= args.max_abs and rmse <= args.max_rmse and cosine >= args.min_cosine
        else:
            good = bool(np.allclose(x, y, atol=args.atol, rtol=args.rtol))
        report = dict(name=a['name'], layer=a.get('layer'), count=x.size, passed=good,
            max_abs=maximum, rmse=rmse, cosine=cosine)
        if a['name'].endswith('logits'):
            report.update(oracle_argmax=int(x.argmax()), native_argmax=int(y.argmax()))
            if args.profile == 'bf16':
                report['passed'] = report['passed'] and report['oracle_argmax'] == report['native_argmax']
        reports.append(report)
    return dict(profile=args.profile, atol=args.atol, rtol=args.rtol,
        envelope=dict(max_abs=args.max_abs, max_rmse=args.max_rmse, min_cosine=args.min_cosine)
            if args.profile == 'bf16' else None,
        passed=all(r['passed'] for r in reports), checks=reports)


def compare(args):
    result = comparison(args)
    print(json.dumps(result, indent=2, allow_nan=False))
    if not result['passed']:
        raise SystemExit(1)


def suite(args):
    """Run the same local short prompt through the official and native full graphs."""
    import subprocess
    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    for adapter in ([None] if args.causal else [None, *args.adapters]):
        name = adapter or 'base'
        oracle = (args.out / f'{name}-oracle.jsonl').absolute()
        native = (args.out / f'{name}-native.jsonl').absolute()
        command = [sys.executable, str(Path(__file__).absolute()), 'trace',
            '--base', str(args.base), '--vela', str(args.vela), '--text', args.text,
            '--dtype', args.dtype, '--out', str(oracle)]
        if adapter:
            command += ['--adapter', adapter]
        if args.ids:
            command += ['--ids', str(args.ids)]
        if args.state_tokens is not None:
            command += ['--state-tokens', str(args.state_tokens)]
        if args.causal:
            command += ['--causal']
        if args.generation_steps is not None:
            command += ['--generation-steps', str(args.generation_steps)]
        if args.synthetic:
            command += ['--synthetic']
        if args.stream_f32_weights:
            command += ['--stream-f32-weights']
        for image in args.images:
            command += ['--images', str(image)]
        with (args.out / f'{name}-oracle.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        metadata = json.loads(oracle.read_text().splitlines()[0])
        environment = os.environ.copy()
        for variable in ['ADAPTER', 'IMAGES', 'STATE', 'CAUSAL', 'STEPS', 'F32']:
            environment.pop(f'RMI_VELA3_D3_{variable}', None)
        environment.pop('RMI_PARITY_FILTER', None)
        environment.update(RMI_VELA3_D3_BASE=str(args.base.absolute()),
            RMI_VELA3_D3_ROOT=str((args.base if args.synthetic else args.vela).absolute()),
            RMI_VELA3_D3_IDS=json.dumps(metadata['unexpanded_token_ids']),
            RMI_PARITY_TRACE=str(native))
        if adapter:
            environment['RMI_VELA3_D3_ADAPTER'] = adapter
        if args.images:
            environment['RMI_VELA3_D3_IMAGES'] = json.dumps([str(p.absolute()) for p in args.images])
        if args.state_tokens is not None:
            environment['RMI_VELA3_D3_STATE'] = str(args.state_tokens)
        if args.dtype == 'float32':
            environment['RMI_VELA3_D3_F32'] = '1'
        if args.causal:
            environment['RMI_VELA3_D3_CAUSAL'] = '1'
        if args.generation_steps is not None:
            environment['RMI_VELA3_D3_STEPS'] = str(args.generation_steps)
        native.unlink(missing_ok=True)
        with (args.out / f'{name}-native.log').open('w') as log:
            subprocess.run([str(args.native.absolute()), 'models::vela3::d3::tests::trace_local_d3_checkpoint',
                '--ignored', '--exact', '--nocapture'], env=environment,
                stdout=log, stderr=subprocess.STDOUT, check=True)
        options = argparse.Namespace(oracle=oracle, native=native, profile=args.profile or
            ('bf16' if args.dtype == 'bfloat16' else 'pointwise'),
            atol=args.atol if args.atol is not None else (0.1 if args.dtype == 'bfloat16' else 5e-4),
            rtol=args.rtol if args.rtol is not None else (0.02 if args.dtype == 'bfloat16' else 1e-4),
            max_abs=args.max_abs, max_rmse=args.max_rmse, min_cosine=args.min_cosine)
        options.vision_detail = args.vision_detail
        report = comparison(options)
        (args.out / f'{name}-comparison.json').write_text(json.dumps(report, indent=2, allow_nan=False))
        summary = dict(adapter=adapter, passed=report['passed'],
            max_abs=max(r['max_abs'] for r in report['checks']),
            max_rmse=max(r['rmse'] for r in report['checks']),
            min_cosine=min(r['cosine'] for r in report['checks']),
            logits=report['checks'][-1])
        if args.generation_steps is not None:
            summary['generated_token_ids'] = next(json.loads(line)['token_ids'] for line in native.read_text().splitlines()
                if json.loads(line).get('name') == 'vela3.d3.generated_tokens')
        results.append(summary)
        print(json.dumps(summary, allow_nan=False), flush=True)
    (args.out / 'summary.json').write_text(json.dumps(results, indent=2, allow_nan=False))
    if not all(result['passed'] for result in results):
        raise SystemExit(1)


def comparison_options(parser, *, suite_mode=False):
    parser.add_argument('--profile', choices=['pointwise', 'bf16'], default=None if suite_mode else 'pointwise')
    parser.add_argument('--atol', type=float, default=None if suite_mode else 0.1)
    parser.add_argument('--rtol', type=float, default=None if suite_mode else 0.02)
    parser.add_argument('--max-abs', type=float, default=0.75)
    parser.add_argument('--max-rmse', type=float, default=0.06)
    parser.add_argument('--min-cosine', type=float, default=0.9995)
    parser.add_argument('--vision-detail', action='store_true')


def vision_fixture(args):
    import torch
    import transformers
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    config = Qwen3_5VisionConfig(hidden_size=8, intermediate_size=16, depth=1, num_heads=2,
        out_hidden_size=4, num_position_embeddings=4, patch_size=2, spatial_merge_size=2,
        temporal_patch_size=2, in_channels=3, hidden_act='gelu_pytorch_tanh',
        deepstack_visual_indexes=[], rope_parameters={'rope_type': 'axial', 'rope_theta': 10000.})
    config._attn_implementation = 'sdpa'
    model = Qwen3_5VisionModel(config).float().eval()
    weights = {}
    for name, parameter in model.state_dict().items():
        salt = sum(f'visual.{name}'.encode())
        weights[name] = ((torch.arange(parameter.numel()) + salt) % 17 - 8).float().mul_(0.025).reshape(parameter.shape)
    model.load_state_dict(weights)
    rgb = (torch.arange(8 * 4 * 3) % 17).float().div_(8).sub_(1).reshape(4, 8, 3)
    patches = []
    for x in [0, 4]:
        for dy in [0, 2]:
            for dx in [0, 2]:
                patch = rgb[dy:dy + 2, x + dx:x + dx + 2].permute(2, 0, 1)
                patches.append(patch[:, None].expand(3, 2, 2, 2).reshape(-1))
    output = model(torch.stack(patches), grid_thw=torch.tensor([[1, 2, 4]]), return_dict=True)
    print(json.dumps(dict(torch=torch.__version__, transformers=transformers.__version__,
        projection=output.pooler_output.flatten().tolist(),
        pre_merger_first16=output.last_hidden_state.flatten()[:16].tolist()), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    trace = commands.add_parser('trace')
    trace.add_argument('--base', type=Path, required=True)
    trace.add_argument('--vela', type=Path, required=True)
    trace.add_argument('--text', default='State:\nAlice works at Acme.\nQuestion:\nWhere does Alice work?')
    trace.add_argument('--ids', type=Path)
    trace.add_argument('--images', type=Path, action='append', default=[])
    trace.add_argument('--adapter')
    trace.add_argument('--state-tokens', type=int)
    trace.add_argument('--causal', action='store_true')
    trace.add_argument('--generation-steps', type=int)
    trace.add_argument('--options', type=int, default=3)
    trace.add_argument('--threads', type=int, default=1)
    trace.add_argument('--dtype', choices=['float32', 'bfloat16'], default='bfloat16')
    trace.add_argument('--synthetic', action='store_true')
    trace.add_argument('--stream-f32-weights', action='store_true',
                       help='load BF16 weights and temporarily cast one layer for F32 text diagnostics')
    trace.add_argument('--out', type=Path, required=True)
    trace.set_defaults(run=run_trace)
    cmp = commands.add_parser('compare')
    cmp.add_argument('oracle', type=Path)
    cmp.add_argument('native', type=Path)
    comparison_options(cmp)
    cmp.set_defaults(run=compare)
    batch = commands.add_parser('suite', help='reproduce local full-graph oracle/native parity with explicit thresholds')
    batch.add_argument('--base', type=Path, required=True)
    batch.add_argument('--vela', type=Path, required=True)
    batch.add_argument('--native', type=Path, required=True, help='parity-trace lib test executable')
    batch.add_argument('--text', default='Alice works at Acme.')
    batch.add_argument('--ids', type=Path)
    batch.add_argument('--images', type=Path, action='append', default=[])
    batch.add_argument('--state-tokens', type=int)
    batch.add_argument('--causal', action='store_true')
    batch.add_argument('--generation-steps', type=int)
    batch.add_argument('--synthetic', action='store_true')
    batch.add_argument('--stream-f32-weights', action='store_true',
                       help='load BF16 weights and temporarily cast one layer for F32 text diagnostics')
    batch.add_argument('--dtype', choices=['float32', 'bfloat16'], default='bfloat16')
    batch.add_argument('--adapters', nargs='*', default=['span', 'decision'])
    batch.add_argument('--out', type=Path, required=True)
    comparison_options(batch, suite_mode=True)
    batch.set_defaults(run=suite)
    fixture = commands.add_parser('vision-fixture', help='reproduce the independent tiny FP32 vision golden')
    fixture.set_defaults(run=vision_fixture)
    args = parser.parse_args()
    args.run(args)


if __name__ == '__main__':
    main()
