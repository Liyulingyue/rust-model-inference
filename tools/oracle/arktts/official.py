"""Run the pinned, unmodified Audio8 Transformers implementation on CPU FP32."""
from __future__ import annotations

import argparse
import json
import sys
import types
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers.generation import MaxLengthCriteria, StoppingCriteriaList


def main(model_dir: Path, output: Path, frames: int, main_gguf=None, codec_gguf=None, history=None, suite="original"):
    torch.set_num_threads(4)
    output.mkdir(parents=True, exist_ok=False)
    package = types.ModuleType("arktts_official")
    package.__path__ = [str(model_dir.resolve())]
    sys.modules[package.__name__] = package
    from arktts_official.configuration_arktts import ArkttsConfig
    from arktts_official.modeling_arktts import ArkttsModel
    from arktts_official.processing_arktts import ArkttsProcessor

    config = ArkttsConfig(**json.loads((model_dir / "config.json").read_text()))
    config._name_or_path = str(model_dir.resolve())
    model = ArkttsModel(config).eval().float()
    model.load_state_dict(load_file(model_dir / "model.safetensors"), strict=True)
    processor = ArkttsProcessor.from_pretrained(model_dir, trust_remote_code=True, local_files_only=True)
    codec = model.load_codec(device="cpu", dtype=torch.float32)
    contracts = {}
    if bool(main_gguf) != bool(codec_gguf):
        raise ValueError("same-weight Oracle requires both GGUF components")
    if main_gguf:
        from tools.oracle.arktts.same_weight import load, prepared_linears
        main_contracts, bundle = load(model, main_gguf, "arktts")
        codec_contracts, codec_bundle = load(codec, codec_gguf, "arktts_codec")
        if bundle != codec_bundle:
            raise ValueError("GGUF checkpoint bundles differ")
        contracts.update(main_contracts)
        contracts.update(codec_contracts)
    histories = json.loads(Path(history).read_text())["cases"] if history else None
    timings = []
    records = []
    codec_phase = {"value": "decode"}
    trace_enabled = {"value": True}

    def record(name, value, layer=None, step=None):
        if not trace_enabled["value"]:
            return
        if codec_phase["value"] == "encode" and (name.startswith("quantizer.post_module") or name in {"arktts.codec.quantized", "arktts.codec.post", "arktts.codec.upsampled"}):
            return
        value = value.detach().float().cpu().contiguous().numpy()
        path = output / f"{len(records):05d}.f32"
        value.astype("<f4").tofile(path)
        records.append({"name": name, "layer": layer, "step": step, "shape": list(value.shape), "binary_path": str(path.resolve())})

    for name, layers in [("layers", model.layers), ("fast_layers", model.fast_layers)]:
        for index, layer in enumerate(layers):
            def hook(module, inputs, result, name=name, index=index):
                start = int(inputs[3][0])
                if index == 0:
                    record(name + ".input", inputs[0][0], step=start)
                record(name, result[0], layer=index, step=start)
            layer.register_forward_hook(hook)

    context = {"step": 0}
    slow_step = model._slow_step
    def slow(*args, **kwargs):
        result = slow_step(*args, **kwargs)
        record("arktts.slow.logits", result[0][0], step=context["step"])
        context["step"] += 1
        return result
    model._slow_step = slow
    fast_step = model._fast_step
    def fast(hidden, position):
        result = fast_step(hidden, position)
        if position:
            record("arktts.fast.logits", result[0], layer=position, step=context["step"] - 1)
        return result
    model._fast_step = fast
    generate_codebooks = model._generate_codebooks
    def codebooks(hidden, semantic, *args, **kwargs):
        # The official loop computes then discards FastAR on EOS; Rust exits before that work.
        enabled = trace_enabled["value"]
        trace_enabled["value"] = enabled and not bool((semantic == config.eos_token_id).all())
        try:
            return generate_codebooks(hidden, semantic, *args, **kwargs)
        finally:
            trace_enabled["value"] = enabled
    model._generate_codebooks = codebooks

    for name, module in codec.named_modules():
        if module.__class__.__name__ == "ArkttsCodecWindowTransformer":
            for index, layer in enumerate(module.layers):
                def hook(_module, inputs, result, name=name, index=index):
                    if index == 0:
                        record(name + ".layers.input", inputs[0][0], step=0)
                    record(name + ".layers", result[0], layer=index, step=0)
                layer.register_forward_hook(hook)
    def channels(value):
        return value[0].transpose(0, 1)
    codec.encoder.register_forward_hook(lambda m, i, o: record("arktts.codec.encoder", channels(o)))
    codec.quantizer.pre_module.register_forward_hook(lambda m, i, o: record("arktts.codec.pre", channels(o)))
    codec.quantizer.post_module.register_forward_pre_hook(lambda m, i: record("arktts.codec.quantized", channels(i[0])))
    codec.quantizer.post_module.register_forward_hook(lambda m, i, o: record("arktts.codec.post", channels(o)))
    codec.quantizer.upsample.register_forward_hook(lambda m, i, o: record("arktts.codec.upsampled", channels(o)))
    for index in range(4):
        codec.decoder.model[index + 1].register_forward_hook(lambda m, i, o, index=index: record("arktts.codec.decoder", channels(o), layer=index))

    cases = []
    with torch.inference_mode(), (prepared_linears(contracts) if main_gguf else nullcontext()):
        length, first, second = (6145, 0.027, 0.093) if suite == "validation" else (4097, 0.031, 0.071)
        reference_audio = (0.15 * np.sin(np.arange(length, dtype=np.float32) * np.float32(first))
                           + 0.04 * np.cos(np.arange(length, dtype=np.float32) * np.float32(second))).astype("<f4")
        reference_audio.tofile(output / "reference.f32")
        # Keep generation and codec traces in the same order as the Rust test.
        trace_enabled["value"] = False
        began = time.perf_counter()
        reference_codes, _ = codec.encode(torch.from_numpy(reference_audio)[None, None])
        timings.append({"phase": "codec_encode", "samples": length, "seconds": time.perf_counter() - began})
        trace_enabled["value"] = True
        reference_codes = reference_codes[0].transpose(0, 1).tolist()
        specifications = [
            ("english", "Hello world.", None),
            ("unicode", "  你好，世界！\tCafé\u0301  日本語。\n", None),
            ("reference", "欢迎使用语音合成。", "  A short reference.\n"),
        ] if suite == "original" else [
            ("english", "One, two, three.", None),
            ("unicode", "  今天气真好，Café！\t日本語。\n", None),
            ("reference", "模型量化验证。", "A different reference."),
        ]
        for name, text, ref_text in specifications:
            fixed_history = next((c for c in histories if c["name"] == name), None) if histories else None
            prompt_codes = fixed_history["reference_codes"] if fixed_history else reference_codes
            codes = torch.tensor(prompt_codes).T if ref_text else None
            inputs = processor(text=text, reference_text=ref_text, reference_codes=codes)
            prompt, mask = model._prepare_prompt(**inputs)
            context["step"] = 0
            greedy = None
            if main_gguf:
                trace_enabled["value"] = False
                began = time.perf_counter()
                greedy = model.generate(input_ids=prompt, attention_mask=mask, max_new_tokens=frames,
                                        do_sample=False, temperature=1.0)[0].T.tolist()
                timings.append({"phase": "generate", "name": name, "frames": len(greedy),
                                "seconds": time.perf_counter() - began})
                trace_enabled["value"] = True
                context["step"] = 0
            class ForceHistory:
                def __init__(self, semantic):
                    self.semantic = semantic
                def __call__(self, ids, scores):
                    target = fixed_history["frames"][context["step"] - 1][0 if self.semantic else ids.shape[-1]]
                    if self.semantic:
                        target += config.semantic_begin_id
                    result = torch.full_like(scores, -torch.inf)
                    result[:, target] = 0
                    return result
            forcing = {"logits_processor": [ForceHistory(True)], "codebook_logits_processor": [ForceHistory(False)]} if fixed_history else {}
            result = model.generate(input_ids=prompt, attention_mask=mask, max_new_tokens=frames, **forcing,
                                    do_sample=False, temperature=1.0, top_k=50, top_p=0.95,
                                    stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(prompt.shape[-1] + frames)]))
            cases.append({"name": name, "text": text, "reference_text": ref_text,
                          "reference_codes": prompt_codes if ref_text else None, "greedy_frames": greedy,
                          "prompt": prompt[0].T.tolist(), "frames": result[0].T.tolist()})
        fixed = [[0, 0, 1, 2, 3, 4, 5, 6, 1023, 4095],
                 [4095, 5000, 999, 31, 1, 26, 100, 200, 300, 400],
                 [1234, 25, 15, 8, 9, 10, 11, 12, 13, 14]]
        if suite == "validation":
            fixed = [[2345, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                     [3456, 5000, 999, 8, 17, 26, 135, 244, 353, 462],
                     [123, 78, 56, 14, 33, 28, 12, 41, 19, 42]]
        trace_enabled["value"] = False
        began = time.perf_counter()
        codec.decode(torch.tensor(fixed).T[None])
        timings.append({"phase": "codec_decode", "frames": len(fixed), "seconds": time.perf_counter() - began})
        trace_enabled["value"] = True
        waveform = codec.decode(torch.tensor(fixed).T[None])[0, 0]
        record("arktts.codec.audio", waveform)
        waveform.numpy().astype("<f4").tofile(output / "waveform.f32")
        codec_phase["value"] = "encode"
        encoded, _ = codec.encode(torch.from_numpy(reference_audio)[None, None])
        encoded = encoded[0].T.tolist()
    from tools.oracle.arktts.compare import TOLERANCES, precision_profile
    codec_types = {kind for kind, _ in codec_contracts.values()} if codec_gguf else set()
    profile = precision_profile(codec_types)
    _, _, _, waveform_max, waveform_rmse = TOLERANCES[profile]
    (output / "performance.json").write_text(json.dumps(timings, indent=2))
    (output / "cases.json").write_text(json.dumps({"max_frames": frames,
        "waveform_max_abs": waveform_max, "waveform_rmse": waveform_rmse,
        "exact_reference_codes": not main_gguf, "exact_generated_codes": not main_gguf,
        "cases": cases, "fixed_codes": fixed, "reference_codes": encoded}, ensure_ascii=False, indent=2))
    (output / "trace.json").write_text(json.dumps(records, ensure_ascii=False, indent=2))
    print(json.dumps({"cases": len(cases), "frames": [len(c["frames"]) for c in cases], "checkpoints": len(records), "samples": waveform.numel()}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--main-gguf", type=Path)
    parser.add_argument("--codec-gguf", type=Path)
    parser.add_argument("--history", type=Path, help="fixed-history cases.json for lossy checkpoint comparisons")
    parser.add_argument("--suite", choices=("original", "validation"), default="original")
    args = parser.parse_args()
    main(args.model_dir, args.output, args.frames, args.main_gguf, args.codec_gguf, args.history, args.suite)
