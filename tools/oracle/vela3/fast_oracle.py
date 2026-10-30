#!/usr/bin/env python3
"""Export Vela3-fast numerical references from the pinned local official snapshot.

No model or tokenizer downloads occur. Run in the repository's .venv, with the
unmodified ModelScope files and complete safetensors already in --model.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path


def make_tiny_snapshot(source: Path, destination: Path):
    """Use official model code with deterministic small weights, including nonzero LoRA."""
    import copy
    import shutil
    import torch
    from safetensors.torch import save_file
    destination.mkdir(parents=True, exist_ok=True)
    for file in source.iterdir():
        if file.suffix in (".py", ".json") and file.name != "repo-files.json":
            shutil.copyfile(file, destination / file.name)
    cfg = json.loads((destination / "config.json").read_text())
    cfg.update(hidden_size=8, intermediate_size=12, num_hidden_layers=2, num_attention_heads=2,
               num_key_value_heads=1, head_dim=4, attention_windows=[2, None], rope_theta=[10000., 2560000.],
               rmi_fixture="synthetic weights; official modeling and readout code")
    cfg["decision"]["proj_dim"] = 4
    (destination / "config.json").write_text(json.dumps(cfg))
    sys.path.insert(0, str(destination))
    from transformers import AutoConfig
    from vela3_encoder import DecisionEncoder
    from vela3_schema import SchemaNet, inject_lora
    torch.manual_seed(123)
    config = AutoConfig.from_pretrained(str(destination), trust_remote_code=True, local_files_only=True)
    base = DecisionEncoder(config, 4).eval()
    save_file(base.state_dict(), str(destination / "model.safetensors"))
    files = ("span", "router", "relation", "ner", "span_sets", "evidence", "domain", "factcheck", "feedback_df")
    for i, name in enumerate(files):
        torch.manual_seed(234 + i)
        net = SchemaNet(copy.deepcopy(base.encoder), 8, 4, span=True)
        inject_lora(net, 2, 4)
        with torch.no_grad():
            for key, param in net.named_parameters():
                if "lora_" in key:
                    param.normal_(0, .05)
            net.log_tau.fill_(-1.1)
            net.log_tau_span.fill_(-.8)
        state = {key: value.contiguous() for key, value in net.state_dict().items()
                 if not key.startswith("encoder.") or ".lora_" in key or key.startswith("encoder.layers.1.")}
        for key, value in list(state.items()):
            if key.startswith("encoder.layers.1.") and ".lora_" not in key:
                state[key] = value + .001
        save_file(state, str(destination / f"{name}_adapter.safetensors"),
                  metadata={"lora_rank": "2", "lora_alpha": "4", "lora_layers": "all"})


def synthetic_request(root: Path, window: int | None):
    if window:
        return {"state": {"user": "Alice met 北京 Acme. " * 90, "context": "Acme is a company.",
                          "answer": "Alice works at Acme."},
                "questions": {"long_span": {"type": "span", "target": "user", "threshold": 0.,
                                           "criteria": {"person": "A person name", "company": "A company name"}}}}
    questions = {
        "decision": {"type": "choice", "instructions": "Who is mentioned?",
                     "criteria": {"human": "A person", "company": None}},
        "noul": {"type": "noul", "instructions": "Does it mention a company?"},
        "entities": {"type": "span", "criteria": ["person", "company"], "threshold": 0.},
        "relation": {"type": "span", "instructions": "Which spans are related to Alice by these relations?",
                     "criteria": ["employer", "location"], "threshold": 0.},
        "evidence": {"type": "span", "criteria": ["evidence"], "target": "context", "threshold": 0.},
        "pii": {"type": "span", "criteria": ["EMAIL_ADDRESS"], "target": "user", "threshold": 0.},
        "halu": {"type": "span", "criteria": ["unsupported"], "threshold": 0.},
        "toxic": {"type": "span", "criteria": ["Violence"], "threshold": 0.},
        "described": {"type": "span", "criteria": {"person": "A person name", "organization": "A company name"}, "threshold": 0.},
    }
    grouped = 0
    for text, schema in json.loads((root / "router_questions.json").read_text())["questions"].items():
        task = schema["id"] in ("domain", "factcheck", "feedback")
        if task or (schema["type"] == "set" and "router_set" not in questions):
            key = schema["id"] if task else "router_set"
        elif schema["type"] == "choice" and grouped < 2:
            key = "grouped_" + schema["id"]
            grouped += 1
        else:
            continue
        questions[key] = {"type": schema["type"], "instructions": text,
                          "criteria": dict(list(schema["options"].items())[:3])}
    return {"state": {"user": "Alice works at Acme. Contact a@b.org. [https://x.org/a).] ꔀꔁ.", "context": "Alice joined Acme in 2025.",
                      "answer": "Alice works at Acme."}, "questions": questions}


def adapter_files(root: Path):
    """All configured files, including adapters the official loader can silently omit."""
    cfg = json.loads((root / "vela3_config.json").read_text())
    files = {entry["file"] for entry in cfg.get("adapters", {}).values() if entry.get("file")}
    for entry in cfg.get("router", {}).get("task_adapters", {}).values():
        files.update(entry.get("files") or [entry["file"]])
    files.update(cfg.get("span", {}).get("task_adapters", {}).values())
    for entry in (cfg.get("span", {}).get("evidence", {}), cfg.get("span", {}).get("set_adapter", {})):
        if entry.get("adapter") or entry.get("file"):
            files.add(entry.get("adapter") or entry["file"])
    return sorted(files)


def span_only_model(base, root: Path):
    """Use official span methods while unrelated router weights are still absent."""
    import vela3_encoder as official
    import vela3_schema as schema
    model = official.Vela3Encoder.__new__(official.Vela3Encoder)
    model.decision = base
    model.cfg = json.loads((root / "vela3_config.json").read_text())
    model.name = model.cfg["name"]
    model._VS = schema
    model.router_q, model.router_task = {}, {}
    def engine(file, max_len=8192):
        return schema.SchemaEngine(root, schema.build_adapter(base, root / file, "cpu"), "cpu", max_len=max_len)
    model.span = engine("span_adapter.safetensors")
    evidence = model.cfg["span"]["evidence"]
    evidence_file = evidence.get("adapter")
    model.span_ev = engine(evidence_file, evidence.get("max_len", 32768)) if evidence_file and (root / evidence_file).is_file() else model.span
    model.task = {task: engine(file) for task, file in model.cfg["span"].get("task_adapters", {}).items() if (root / file).is_file()}
    model.entity_route = model.cfg["span"].get("entity_route", "ner")
    sets = model.cfg["span"].get("set_adapter")
    model.span_set = engine(sets["file"]) if sets and (root / sets["file"]).is_file() else None
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--request", type=Path, help="System One JSON request; synthetic mode supplies a route-coverage request")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--synthetic", type=Path, help="create deterministic small weights here using the official --model source snapshot")
    parser.add_argument("--span-window", type=int, help="Oracle test override for all ordinary span engine windows (use a synthetic fixture)")
    parser.add_argument("--span-only", action="store_true", help="run official span methods without loading unrelated router weights")
    parser.add_argument("--system-one", action="store_true", help="also run all official adapter routes")
    parser.add_argument("--require-all-adapters", action="store_true", help="fail before loading if any configured adapter file is absent; use with --system-one")
    args = parser.parse_args()
    root = args.model.resolve()
    required = ("config.json", "tokenizer.json", "modeling_ultrabert.py", "vela3_encoder.py")
    if not args.synthetic:
        required += ("model.safetensors",)
    for file in required:
        if not (root / file).is_file():
            parser.error(f"missing local snapshot file: {root / file}")
    import os
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if args.synthetic:
        make_tiny_snapshot(root, args.synthetic.resolve())
        root = args.synthetic.resolve()
    configured_adapters = adapter_files(root)
    if args.require_all_adapters:
        if not args.system_one or args.span_only:
            parser.error("--require-all-adapters requires --system-one without --span-only")
        missing = [name for name in configured_adapters if not (root / name).is_file()]
        if missing:
            parser.error("missing configured adapter files: " + ", ".join(missing))
    sys.path.insert(0, str(root))
    import torch
    import vela3_encoder as official
    if not args.request and not args.synthetic:
        parser.error("--request is required for official checkpoint weights")
    request = json.loads(args.request.read_text()) if args.request else synthetic_request(root, args.span_window)
    if args.synthetic and args.span_window:
        runtime = json.loads((root / "vela3_config.json").read_text())
        runtime["span"]["window"]["max_len"] = args.span_window
        (root / "vela3_config.json").write_text(json.dumps(runtime))
    torch.set_num_threads(1)
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    model = official.DecisionEncoder.load(root, device="cpu")
    records = []
    for key, question in request["questions"].items():
        if question.get("type") not in ("choice", "noul"):
            continue
        if question.get("type") == "choice" and isinstance(question.get("criteria"), list):
            question = dict(question, criteria={str(n): None for n in question["criteria"]})
        record = model.encode(request.get("state"), question)
        x = torch.tensor([record["ids"]], dtype=torch.long)
        hidden = model.encoder(input_ids=x, attention_mask=torch.ones_like(x)).last_hidden_state
        logits = model._logits(hidden, [record])[0]
        records.append(dict(key=key, record=record, hidden=hidden[0].float().tolist(), logits=logits.float().tolist()))
    decision_questions = {k: q for k, q in request["questions"].items() if q.get("type") in ("choice", "noul")}
    for key, q in list(decision_questions.items()):
        if q.get("type") == "choice" and isinstance(q.get("criteria"), list):
            decision_questions[key] = dict(q, criteria={str(n): None for n in q["criteria"]})
    answers, input_tokens = model.predict(request.get("state"), decision_questions)
    result = dict(snapshot="vllm-sr/vela3-fast@8485c157", request=request, records=records,
                  decision=dict(model="vllm-sr/vela3-fast", answers=answers, usage=dict(input_tokens=input_tokens, output_tokens=0)),
                  reference=dict(torch=torch.__version__, dtype="float32", device="cpu", attention="sdpa",
                                 weights=json.loads((root / "config.json").read_text()).get("rmi_fixture", "official checkpoint"),
                                 adapter_files_present=[name for name in configured_adapters if (root / name).is_file()],
                                 require_all_adapters=args.require_all_adapters))
    if args.system_one or args.synthetic or args.span_only:
        if args.span_only:
            if any(q.get("type") != "span" for q in request["questions"].values()):
                parser.error("--span-only requires only span questions")
            model = span_only_model(model, root)
            result["reference"]["mode"] = "official span methods; unrelated router weights are not loaded"
        else:
            del model
            model = official.Vela3Encoder.load(root, device="cpu")
        if args.span_window:
            if args.span_window <= 0:
                parser.error("--span-window must be positive")
            for engine in [model.span, *model.task.values(), model.span_set]:
                if engine is not None:
                    engine.max_len = args.span_window
            result["reference"]["span_window_override"] = args.span_window
        result["system_one"] = model.system_one(request.get("state"), request["questions"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    summary = dict(fixture=str(args.out), records=len(records), input_tokens=input_tokens,
                   adapter_files_present=result["reference"]["adapter_files_present"])
    if "system_one" in result:
        summary["system_one_input_tokens"] = result["system_one"]["usage"]["input_tokens"]
        summary["heads"] = {key: answer.get("head", "decision")
                            for key, answer in result["system_one"]["answers"].items()}
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
