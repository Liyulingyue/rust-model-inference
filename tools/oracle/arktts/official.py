"""Run the pinned, unmodified Audio8 Transformers implementation on CPU FP32."""
from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers.generation import MaxLengthCriteria, StoppingCriteriaList


def main(model_dir: Path, output: Path, frames: int):
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
    with torch.inference_mode():
        reference_audio = (0.15 * np.sin(np.arange(4097, dtype=np.float32) * np.float32(0.031))
                           + 0.04 * np.cos(np.arange(4097, dtype=np.float32) * np.float32(0.071))).astype("<f4")
        reference_audio.tofile(output / "reference.f32")
        # Keep generation and codec traces in the same order as the Rust test.
        trace_enabled["value"] = False
        reference_codes, _ = codec.encode(torch.from_numpy(reference_audio)[None, None])
        trace_enabled["value"] = True
        reference_codes = reference_codes[0].transpose(0, 1).tolist()
        for name, text, ref_text in [
            ("english", "Hello world.", None),
            ("unicode", "  你好，世界！\tCafé\u0301  日本語。\n", None),
            ("reference", "欢迎使用语音合成。", "  A short reference.\n"),
        ]:
            codes = torch.tensor(reference_codes).T if ref_text else None
            inputs = processor(text=text, reference_text=ref_text, reference_codes=codes)
            prompt, mask = model._prepare_prompt(**inputs)
            context["step"] = 0
            result = model.generate(input_ids=prompt, attention_mask=mask, max_new_tokens=frames,
                                    do_sample=False, temperature=1.0, top_k=50, top_p=0.95,
                                    stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(prompt.shape[-1] + frames)]))
            cases.append({"name": name, "text": text, "reference_text": ref_text,
                          "reference_codes": reference_codes if ref_text else None,
                          "prompt": prompt[0].T.tolist(), "frames": result[0].T.tolist()})
        fixed = [[0, 0, 1, 2, 3, 4, 5, 6, 1023, 4095],
                 [4095, 5000, 999, 31, 1, 26, 100, 200, 300, 400],
                 [1234, 25, 15, 8, 9, 10, 11, 12, 13, 14]]
        waveform = codec.decode(torch.tensor(fixed).T[None])[0, 0]
        record("arktts.codec.audio", waveform)
        waveform.numpy().astype("<f4").tofile(output / "waveform.f32")
        codec_phase["value"] = "encode"
        encoded, _ = codec.encode(torch.from_numpy(reference_audio)[None, None])
        encoded = encoded[0].T.tolist()
    (output / "cases.json").write_text(json.dumps({"max_frames": frames, "cases": cases, "fixed_codes": fixed, "reference_codes": encoded}, ensure_ascii=False, indent=2))
    (output / "trace.json").write_text(json.dumps(records, ensure_ascii=False, indent=2))
    print(json.dumps({"cases": len(cases), "frames": [len(c["frames"]) for c in cases], "checkpoints": len(records), "samples": waveform.numel()}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--frames", type=int, default=4)
    args = parser.parse_args()
    main(args.model_dir, args.output, args.frames)
