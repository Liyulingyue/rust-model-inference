"""Run the pinned official remote-code model in F32 and capture every Loop visit."""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoConfig,
    AutoModelForImageTextToText,
    AutoTokenizer,
    AutoProcessor,
)
from tools.oracle.sensenova_loop.quantized_reference import host_device_map, install_activation_quantization, materialize_cpu, replace_projections

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("model", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--steps", type=int, default=8)
parser.add_argument("--threads", type=int, default=16)
parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
parser.add_argument("--host-inputs", action="store_true", help="Keep F32 embeddings, vision and output projection on CPU to fit decoder VRAM")
parser.add_argument("--gguf", type=Path, help="Use independently dequantized projections of this GGUF")
parser.add_argument("--activation-quantization", action="store_true", help="Include native block activation QDQ in the independent same-GGUF reference")
parser.add_argument(
    "--image", type=Path, help="Run one image fixture instead of text fixtures"
)
args = parser.parse_args()
if args.activation_quantization and not args.gguf:
    parser.error("--activation-quantization requires --gguf")
args.output.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(args.threads)
torch.manual_seed(0)
config = AutoConfig.from_pretrained(
    args.model, trust_remote_code=True, local_files_only=True
)
device_map = host_device_map(args.device) if args.host_inputs and args.device == "cuda" else args.device
model = AutoModelForImageTextToText.from_pretrained(
    args.model,
    config=config,
    trust_remote_code=True,
    local_files_only=True,
    dtype=torch.float32,
    device_map=device_map,
    attn_implementation="eager",
).eval()
tokenizer = AutoTokenizer.from_pretrained(
    args.model, local_files_only=True, trust_remote_code=True
)
text = model.model.language_model
if args.host_inputs and args.device == "cuda":
    # Detaching offload hooks can restore CUDA parameters; place them before copying GGUF weights.
    for module in (text.embed_tokens, model.model.visual, model.lm_head):
        materialize_cpu(module)
    embedding_forward = text.embed_tokens.forward
    def cpu_embeddings(ids):
        return embedding_forward(ids.cpu()).to(args.device)
    text.embed_tokens.forward = cpu_embeddings
    head_forward = model.lm_head.forward
    def cpu_head(hidden):
        return head_forward(hidden.cpu()).to(args.device)
    model.lm_head.forward = cpu_head
    vision_forward = model.model.visual.forward
    def cpu_vision(*inputs, **kwargs):
        inputs = tuple(value.cpu() if isinstance(value, torch.Tensor) else value for value in inputs)
        kwargs = {key: value.cpu() if isinstance(value, torch.Tensor) else value for key, value in kwargs.items()}
        output = vision_forward(*inputs, **kwargs)
        output.pooler_output = output.pooler_output.to(args.device)
        return output
    model.model.visual.forward = cpu_vision
execution = {
    "decoder_device": args.device, "host_inputs": args.host_inputs,
    "embedding_devices": sorted({str(p.device) for p in text.embed_tokens.parameters()}),
    "vision_devices": sorted({str(p.device) for p in model.model.visual.parameters()}),
    "output_head_devices": sorted({str(p.device) for p in model.lm_head.parameters()}),
    "device_map": device_map,
    "dtype": "float32", "attention": "eager", "threads": args.threads, "seed": 0,
}
(args.output / "execution.json").write_text(json.dumps(execution, indent=2))
if args.gguf:
    provenance = replace_projections(model, args.gguf)
    if args.activation_quantization:
        provenance["activation_quantization"] = install_activation_quantization(model, args.gguf)
    else:
        provenance["activation_quantization"] = "float32"
    execution["activation_quantization"] = provenance["activation_quantization"]
    (args.output / "execution.json").write_text(json.dumps(execution, indent=2))
    (args.output / "precision-reference.json").write_text(json.dumps(provenance, indent=2))
steps = []
visits = []


def record(_module, _inputs, output):
    values = output.detach().float().cpu().numpy()
    visits.append(values.reshape(values.shape[1], -1).copy())


hooks = [layer.register_forward_hook(record) for layer in text.layers]
fixtures = []
texts = ["Hello!", "你好，介绍一下你自己。", "Cafe\u0301  \t\n<|im_end|>🙂"]
if args.image:
    texts = ["What color is this image?"]
for case, prompt in enumerate(texts):
    content = prompt
    if args.image:
        content = [
            {"type": "image", "image": str(args.image)},
            {"type": "text", "text": prompt},
        ]
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    ids = tokenizer.encode(rendered, add_special_tokens=False)
    image_inputs = {}
    if args.image:
        processor = AutoProcessor.from_pretrained(
            args.model, local_files_only=True, trust_remote_code=True
        )
        image_inputs = dict(
            processor(
                text=[rendered],
                images=[Image.open(args.image).convert("RGB")],
                return_tensors="pt",
            )
        )
        ids = image_inputs.pop("input_ids")[0].tolist()
        image_inputs.pop("attention_mask", None)
        image_inputs = {key: value.to(args.device) for key, value in image_inputs.items()}
        # Preserve the processor-expanded prompt for exact native tokenizer checks.
        rendered = tokenizer.decode(ids, skip_special_tokens=False)
    fixture = {
        "prompt": prompt,
        "rendered": rendered,
        "token_ids": ids,
        "plain_ids": tokenizer.encode(prompt, add_special_tokens=False),
        "generated": [],
        "steps": [],
    }
    if args.image:
        fixture["image"] = str(args.image.resolve())
        fixture["image_grid_thw"] = image_inputs["image_grid_thw"].tolist()

        def vision_output(_module, _inputs, output):
            np.save(
                args.output / "vision.npy",
                output.pooler_output.detach().float().cpu().numpy(),
            )

        vision_hook = model.model.visual.register_forward_hook(vision_output)
    tokens = torch.tensor([ids], device=args.device)
    cache = None
    for step in range(args.steps if case == 0 else 1):
        visits.clear()
        started = time.perf_counter()
        with torch.inference_mode():
            result = model(
                input_ids=tokens,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
                **(image_inputs if step == 0 else {}),
            )
        cache = result.past_key_values
        next_id = int(result.logits[0, -1].argmax())
        elapsed = time.perf_counter() - started
        for visit, values in enumerate(visits):
            np.save(args.output / f"case{case}.step{step}.visit{visit}.npy", values)
        logits = result.logits[0, -1].float().cpu().numpy()
        np.save(args.output / f"case{case}.step{step}.logits.npy", logits)
        fixture["steps"].append(
            {
                "rows": tokens.shape[1],
                "visits": [list(v.shape) for v in visits],
                "argmax": next_id,
                "elapsed_s": elapsed,
            }
        )
        fixture["generated"].append(next_id)
        tokens = torch.tensor([[next_id]], device=args.device)
    fixture["text"] = tokenizer.decode(fixture["generated"])
    fixtures.append(fixture)
    (args.output / "fixtures.json").write_text(
        json.dumps(fixtures, ensure_ascii=False, indent=2)
    )
    print(case, fixture["generated"], repr(fixture["text"]), flush=True)
    if args.image:
        vision_hook.remove()
for hook in hooks:
    hook.remove()
if args.device == "cuda":
    execution["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
(args.output / "execution.json").write_text(json.dumps(execution, indent=2))
