"""Independent same-GGUF Qwen2/CosyVoice3 talker logits and bistream Oracle.

Uses gguf-py's decoder, Transformers' Qwen2Model and the pinned upstream
tokenizer/_need_text/tag_clean sources. No Rust or converter math is imported.
"""

from __future__ import annotations

import argparse
import ast
import inspect
import json
import logging
from pathlib import Path
import re
import tempfile
import time
from types import SimpleNamespace

import numpy as np

from .oracle import GATES, meets_gate, metrics, reference_weight, sha256
from .code2wav_oracle import RuntimeActivations

REVISION = "990d6a74bbf15c1765a6f7736fa839f423fd5b59"
PROMPT = "You are a helpful assistant.<|endofprompt|>"
SPEECH, SOS, EOS, TASK, FILL = 6561, 6561, 6562, 6563, 6564
SOURCE_PINS = {
    "CosyVoice-BlankEN/config.json": "168aa1bd401abc3bc262ba15ba4e499627a8b4e006e9d050b47c22de20660185",
    "CosyVoice-BlankEN/tokenizer_config.json": "482bd979881423375ca5414e4e0d94cd7c5349dbb17fffd46b4d36d71e62a1bc",
    "CosyVoice-BlankEN/vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
    "CosyVoice-BlankEN/merges.txt": "ac8ff86a72bee70828fbc1119bc4398c6f3a9a6e490d7b0dbe917be025478bd0",
}
REFERENCE_PINS = {
    "talker_tag.py": "3e223da391de4dde0267acdd854b8c0943bb0e872127c01ab79808a930bdf81a",
    "proc.py": "0492a716cf48f7ccabb6ec14f4ca8b00fe330de3d41b9092483faa652a134cd1",
}
TOKENIZER_PIN = "40cc63949fda0f05aea4cedef715ee63a1dabf73d9e2b7e1eeeb50cf1c6ff423"


def pinned_ast(path, expected):
    if sha256(path) != expected:
        raise ValueError(f"reference does not match the pinned source: {path}")
    return ast.parse(path.read_text())


def load_tag_reference(directory):
    """Execute only official scheduling/cleaning nodes, excluding vLLM imports."""
    import torch

    namespace = {"torch": torch, "re": re, "logger": logging.getLogger(__name__)}
    talker = pinned_ast(directory / "talker_tag.py", REFERENCE_PINS["talker_tag.py"])
    state = next(
        node
        for node in talker.body
        if isinstance(node, ast.ClassDef) and node.name == "_St"
    )
    cls = next(
        node
        for node in talker.body
        if isinstance(node, ast.ClassDef) and node.name == "SophonTagTalker"
    )
    need_text = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "_need_text"
    )
    processing = pinned_ast(directory / "proc.py", REFERENCE_PINS["proc.py"])
    clean = [
        node
        for node in processing.body
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == "tag_clean"
            or isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id in {"POINT", "SPAN", "CLOSE"}
                for t in node.targets
            )
        )
    ]
    module = ast.Module(body=[state, need_text, *clean], type_ignores=[])
    exec(compile(module, str(directory), "exec"), namespace)
    return namespace


def load_tokenizer(path, source):
    """Load the original wrapper verbatim, without importing vLLM or its package."""
    import torch
    from functools import cache
    from transformers import AutoTokenizer

    module = pinned_ast(source, TOKENIZER_PIN)
    classes = [
        node
        for node in module.body
        if isinstance(node, ast.ClassDef)
        and node.name in {"CosyVoice2Tokenizer", "CosyVoice3Tokenizer"}
    ]
    namespace = {"torch": torch, "AutoTokenizer": AutoTokenizer, "cache": cache}
    exec(
        compile(ast.Module(body=classes, type_ignores=[]), str(source), "exec"),
        namespace,
    )
    return namespace["CosyVoice3Tokenizer"](str(path), skip_special_tokens=True)


class Schedule:
    """Complete-text bistream driver; upstream _need_text chooses exact feed rows."""

    def __init__(self, reference, text_ids, text_embedding, speech_embedding, last_fed):
        self.state = reference["_St"](None)
        self.state.text = text_ids
        self.state.fin = True
        self.state.last_fed = last_fed
        self.speech = speech_embedding
        self.reference = reference
        self.owner = SimpleNamespace(
            MIX=[5, 15],
            TASK=TASK,
            _temb=lambda ids: text_embedding[ids],
            _semb=lambda ids: speech_embedding[ids],
        )

    @property
    def phase(self):
        return self.state.phase

    def forced(self):
        st = self.state
        return FILL if st.phase == "bi" and len(st.out) == st.next_fill else None

    def accept(self, token):
        import torch

        st = self.state
        control = EOS if st.phase == "final" else FILL
        if (
            st.done_eos
            or type(token) is not int
            or token < 0
            or (token >= SPEECH and token != control)
        ):
            raise ValueError(f"invalid teacher token {token} in phase {st.phase}")
        if self.forced() is not None and token != self.forced():
            raise ValueError("teacher overrides a forced fill")
        if token == FILL:
            st.next_fill = len(st.out) + 16
        st.out.append(token)
        if token == EOS:
            st.done_eos = True
            return None
        if token == FILL:
            self.reference["_need_text"](self.owner, st)
            embeddings = torch.cat(st.queue, dim=0)
            st.queue = []
        else:
            embeddings = self.speech[token : token + 1]
        st.last_fed = embeddings[-1:]
        return embeddings


def tensor_name(name):
    top = {
        "token_embd.weight": "embed_tokens.weight",
        "output_norm.weight": "norm.weight",
    }
    if name in top:
        return top[name]
    match = re.fullmatch(r"blk\.(\d+)\.([a-z_]+)\.(weight|bias)", name)
    suffixes = {
        "attn_norm": "input_layernorm",
        "ffn_norm": "post_attention_layernorm",
        "attn_q": "self_attn.q_proj",
        "attn_k": "self_attn.k_proj",
        "attn_v": "self_attn.v_proj",
        "attn_output": "self_attn.o_proj",
        "ffn_gate": "mlp.gate_proj",
        "ffn_up": "mlp.up_proj",
        "ffn_down": "mlp.down_proj",
    }
    if (
        not match
        or match[2] not in suffixes
        or (match[3] == "bias" and match[2] not in {"attn_q", "attn_k", "attn_v"})
    ):
        raise ValueError(f"unexpected talker tensor: {name}")
    return f"layers.{int(match[1])}.{suffixes[match[2]]}.{match[3]}"


def load_model(reader, config, output):
    import torch
    from gguf.quants import dequantize
    from transformers import Qwen2Model

    state, separate = {}, {}
    with tempfile.TemporaryFile(prefix="talker-reference-", dir=output) as scratch:
        for tensor in reader.tensors:
            weight = torch.from_numpy(reference_weight(tensor, dequantize, scratch))
            target = (
                separate
                if tensor.name in {"speech_embd.weight", "output.weight"}
                else state
            )
            name = tensor.name if target is separate else tensor_name(tensor.name)
            if name in target:
                raise ValueError(f"duplicate talker tensor: {name}")
            target[name] = weight
    with torch.device("meta"):
        model = Qwen2Model(config)
    model.load_state_dict(state, strict=True, assign=True)
    # RoPE frequencies are nonpersistent buffers, so assign=True cannot materialize them.
    model.rotary_emb = type(model.rotary_emb)(config)
    model.eval()
    speech, head = separate["speech_embd.weight"], separate["output.weight"]
    expected = (6761, config.hidden_size)
    if tuple(speech.shape) != expected or tuple(head.shape) != expected:
        raise ValueError("invalid talker speech embedding/output shape")
    return model, speech, head


def trace_inputs(trace, prompt_ids, text_ids):
    def ids(name):
        records = [row for row in trace if row["name"] == f"sophon.talker.{name}"]
        if len(records) != 1:
            raise ValueError(f"expected one Rust talker {name} record")
        return records[0]["token_ids"]

    if ids("prompt_ids") != prompt_ids or ids("text_ids") != text_ids:
        raise ValueError("Rust and official talker tokenizer IDs differ")
    teacher = ids("sampled_tokens")
    checkpoints = [row for row in trace if row["name"] == "sophon.talker.logits"]
    if not teacher or len(checkpoints) != len(teacher):
        raise ValueError("Rust talker logits and complete sampled token counts differ")
    for step, row in enumerate(checkpoints):
        if row.get("step") != step or row.get("shape") != [6761]:
            raise ValueError(
                "Rust talker checkpoints are not contiguous vocabulary rows"
            )
    return teacher, checkpoints


def masked_greedy(logits, phase, forced=None):
    if forced is not None:
        return forced
    scores = np.array(logits, copy=True)
    control = EOS if phase == "final" else FILL
    scores[SPEECH:] = -np.inf
    scores[control] = logits[control]
    return int(np.argmax(scores))


def greedy_margin(logits, phase):
    control = EOS if phase == "final" else FILL
    allowed = np.concatenate([logits[:SPEECH], logits[control : control + 1]])
    top = np.partition(allowed, -2)[-2:]
    return float(top.max() - top.min())


def reference_steps(
    model,
    speech,
    head,
    reference,
    text_ids,
    prompt_ids,
    teacher,
    steps,
    activation=None,
):
    import torch
    from transformers.cache_utils import DynamicCache

    initial = torch.cat([speech[SOS : SOS + 1], model.embed_tokens.weight[prompt_ids]])
    schedule = Schedule(
        reference, text_ids, model.embed_tokens.weight, speech, initial[-1:]
    )
    cache, current, position = DynamicCache(config=model.config), initial, 0
    for step in range(steps):
        phase, forced = schedule.phase, schedule.forced()
        positions = torch.arange(position, position + len(current))[None]
        started = time.perf_counter()
        output = model(
            inputs_embeds=current[None],
            position_ids=positions,
            past_key_values=cache,
            use_cache=True,
        )
        head_input = output.last_hidden_state[0, -1]
        if activation is not None:
            head_input = activation.apply(head_input, "output.weight")
        logits = torch.nn.functional.linear(head_input, head).float().detach().numpy()
        if logits.shape != (6761,) or not np.isfinite(logits).all():
            raise ValueError("nonfinite or malformed official talker logits")
        selected = (
            teacher[step]
            if teacher is not None
            else masked_greedy(logits, phase, forced)
        )
        following = schedule.accept(selected)
        record = {
            "step": step,
            "phase": phase,
            "forced": forced,
            "selected_token": selected,
            "reference_greedy_id": masked_greedy(logits, phase, forced),
            "greedy_margin": (
                None if forced is not None else greedy_margin(logits, phase)
            ),
            "input_rows": len(current),
            "input_start_position": position,
            "seconds": time.perf_counter() - started,
        }
        yield logits, record
        if following is None:
            if teacher is not None and step + 1 != len(teacher):
                raise ValueError("teacher tokens follow EOS")
            break
        position += len(current)
        current = following


def run(args):
    import gguf
    import torch
    import transformers
    from gguf.quants import dequantize
    from transformers import Qwen2Config, Qwen2Model

    if args.steps < 1 or args.threads < 1 or not args.text.strip():
        raise ValueError("steps/threads must be positive and text must be nonempty")
    torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)
    torch.manual_seed(args.seed)
    model_dir = args.model_dir / "talker"
    for path, expected in SOURCE_PINS.items():
        if sha256(model_dir / path) != expected:
            raise ValueError(f"source does not match pinned release: {path}")
    reference = load_tag_reference(args.tag_reference)
    tokenizer = load_tokenizer(
        model_dir / "CosyVoice-BlankEN", args.tokenizer_reference
    )
    text = reference["tag_clean"](args.text, True)
    text_ids, prompt_ids = tokenizer.encode(
        text, allowed_special="all"
    ), tokenizer.encode(PROMPT, allowed_special="all")
    if not text_ids:
        raise ValueError("text is empty after official tag cleaning")
    config = Qwen2Config(
        **json.loads((model_dir / "CosyVoice-BlankEN/config.json").read_text())
    )
    config._attn_implementation = "eager"
    if (
        len(prompt_ids) + len(text_ids) + args.steps * 2 + 8
        > config.max_position_embeddings
    ):
        raise ValueError("talker context budget exceeded")
    reader = gguf.GGUFReader(str(args.gguf), mode="c")
    for key, expected in {
        "general.architecture": "sophon_realtime_talker",
        "sophon_realtime.component": "talker",
        "sophon_realtime.source_revision": REVISION,
        "sophon_realtime.requested_quant": args.precision,
    }.items():
        field = reader.fields.get(key)
        if field is None or bytes(field.parts[field.data[0]]).decode() != expected:
            raise ValueError(f"GGUF is not the pinned talker: {key}")
    trace = (
        [
            json.loads(line)
            for line in args.rust_trace.read_text().splitlines()
            if line.strip()
        ]
        if args.rust_trace
        else []
    )
    teacher, checkpoints = (
        trace_inputs(trace, prompt_ids, text_ids) if trace else (None, [])
    )
    if args.teacher_tokens:
        supplied = json.loads(args.teacher_tokens.read_text())
        if not isinstance(supplied, list) or not supplied:
            raise ValueError("teacher JSON must be a nonempty list including fill/EOS")
        if teacher is not None and teacher != supplied:
            raise ValueError("teacher JSON and Rust sampled token IDs differ")
        teacher = supplied
    steps = min(args.steps, len(teacher)) if teacher is not None else args.steps
    args.output.mkdir(parents=True, exist_ok=False)
    model, speech, head = load_model(reader, config, args.output)
    kinds = {tensor.name: int(tensor.tensor_type) for tensor in reader.tensors}
    names = {
        tensor_name(tensor.name): tensor.name
        for tensor in reader.tensors
        if tensor.name not in {"speech_embd.weight", "output.weight"}
    }
    activation = (
        RuntimeActivations(model, kinds, names)
        if args.activation == "runtime"
        else None
    )
    records = []
    for logits, record in reference_steps(
        model, speech, head, reference, text_ids, prompt_ids, teacher, steps, activation
    ):
        logits.astype("<f4", copy=False).tofile(
            args.output / f"logits-{record['step']}.f32"
        )
        if checkpoints:
            actual = np.fromfile(
                checkpoints[record["step"]]["binary_path"], dtype="<f4"
            )
            comparison = metrics(actual, logits)
            record.update(
                rust_metrics=comparison,
                numerical_gate_pass=bool(meets_gate(comparison, args.precision)),
                rust_greedy_id=masked_greedy(actual, record["phase"], record["forced"]),
            )
            record["greedy_match"] = (
                record["rust_greedy_id"] == record["reference_greedy_id"]
            )
        records.append(record)
        print(json.dumps(record), flush=True)
    result = {
        "semantics": "independent same-GGUF F32 compute/F32 KV, teacher-forced complete-text official bistream schedule",
        "sampling": "teacher-forced when supplied; otherwise masked greedy; seed does not promise cross-framework sample identity",
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "threads": args.threads,
        "seed": args.seed,
        "transformers_source_sha256": sha256(Path(inspect.getfile(Qwen2Model))),
        "gguf_decoder_sha256": sha256(Path(inspect.getfile(dequantize))),
        "gguf_sha256": sha256(args.gguf),
        "source_revision": REVISION,
        "source_pins": SOURCE_PINS,
        "tag_reference_pins": REFERENCE_PINS,
        "tokenizer_reference_sha256": TOKENIZER_PIN,
        "text": args.text,
        "cleaned_text": text,
        "text_ids": text_ids,
        "prompt": PROMPT,
        "prompt_ids": prompt_ids,
        "teacher_tokens": teacher,
        "records": records,
        "precision": args.precision,
        "activation": args.activation,
        "activation_projections": (
            activation.description if activation is not None else []
        ),
        "tensor_types": {
            name: sum(t.tensor_type.name == name for t in reader.tensors)
            for name in sorted({t.tensor_type.name for t in reader.tensors})
        },
        "gate": dict(
            zip(("maximum_relative_rmse", "minimum_cosine"), GATES[args.precision])
        ),
        "parity_pass": (
            all(row["numerical_gate_pass"] and row["greedy_match"] for row in records)
            if trace
            else None
        ),
    }
    (args.output / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2)
    )
    if activation is not None:
        activation.close()
    if result["parity_pass"] is False:
        raise SystemExit(
            "talker numerical or masked-greedy gate failed; inspect result.json"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--tag-reference",
        type=Path,
        default=Path("target/sophon/reference/tag/plugin/sophon_realtime"),
    )
    parser.add_argument(
        "--tokenizer-reference",
        type=Path,
        default=Path(
            "target/sophon/vllm-omni/vllm_omni/model_executor/models/cosyvoice3/tokenizer.py"
        ),
    )
    parser.add_argument("--rust-trace", type=Path)
    parser.add_argument("--teacher-tokens", type=Path)
    parser.add_argument("--text", default="你好，我是你的语音助手。[laughter]")
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--precision", choices=list(GATES), default="f32")
    parser.add_argument("--activation", choices=["f32", "runtime"], default="f32")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
