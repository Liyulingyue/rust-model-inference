"""Run the pinned official remote-code model in F32 and capture every Loop visit."""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
from transformers import (
    AutoConfig,
    AutoModelForImageTextToText,
    AutoTokenizer,
    AutoProcessor,
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("model", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--steps", type=int, default=8)
parser.add_argument("--threads", type=int, default=16)
parser.add_argument(
    "--image", type=Path, help="Run one image fixture instead of text fixtures"
)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(args.threads)
torch.manual_seed(0)
config = AutoConfig.from_pretrained(
    args.model, trust_remote_code=True, local_files_only=True
)
model = AutoModelForImageTextToText.from_pretrained(
    args.model,
    config=config,
    trust_remote_code=True,
    local_files_only=True,
    dtype=torch.float32,
    attn_implementation="eager",
).eval()
tokenizer = AutoTokenizer.from_pretrained(
    args.model, local_files_only=True, trust_remote_code=True
)
text = model.model.language_model
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
        from PIL import Image

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
                output.pooler_output.detach().float().numpy(),
            )

        vision_hook = model.model.visual.register_forward_hook(vision_output)
    tokens = torch.tensor([ids])
    cache = None
    for step in range(args.steps if case == 0 else 1):
        visits.clear()
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
        for visit, values in enumerate(visits):
            np.save(args.output / f"case{case}.step{step}.visit{visit}.npy", values)
        logits = result.logits[0, -1].float().numpy()
        np.save(args.output / f"case{case}.step{step}.logits.npy", logits)
        fixture["steps"].append(
            {
                "rows": tokens.shape[1],
                "visits": [list(v.shape) for v in visits],
                "argmax": next_id,
            }
        )
        fixture["generated"].append(next_id)
        tokens = torch.tensor([[next_id]])
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
