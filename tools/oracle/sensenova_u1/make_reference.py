#!/usr/bin/env python3
"""Dump deterministic reference tensors from the OFFICIAL SenseNova-U1 oracle.

This is the "golden" generator for the Rust port. It loads the model exactly the
way ``examples/t2i/inference.py`` does (``load_model_and_tokenizer_for_format``)
and then replays ``NEOChatModel.t2i_generate`` **verbatim** -- the body below is a
line-for-line copy of
``src/sensenova_u1/models/neo_unify/modeling_neo_chat.py::t2i_generate`` with
``capture(...)`` calls inserted at the boundaries a Rust implementation needs to
check against.  Keeping the arithmetic identical (same op order, same dtypes,
same RNG call) is the whole point: ``--verify-png`` re-renders the returned
tensor through the official ``_to_pil`` path so the dump can be diffed against a
plain ``examples/t2i/inference.py`` run.

Contract (fixed; see ``CONTRACT`` below)
---------------------------------------
prompt  : "A studio photograph of a red apple on a white table, soft daylight."
size    : 512 x 512      steps: 4      seed: 42
cfg     : scale 4.0, norm "none", interval (0.0, 1.0)
schedule: timestep_shift 3.0, "standard" (the oracle hardcodes it), t_eps 0.02
method  : Euler, batch_size 1, cfg_batching True (BatchedCFGCache)

Outputs (``--out_dir``)
-----------------------
``manifest.json``            everything below, with shapes/dtypes/slices
``state_dict_keys.json``     state_dict names + shapes + dtypes + quant types
``*.npy``                    float32 (or int64 for ids) tensors, see manifest
``final.png``                the official ``_to_pil`` render of the final tensor
``prompt.json``              exact query strings + token ids + special-token ids

Usage
-----
  /root/sn_oracle_venv/bin/python make_reference.py \
      --model_path /root/models/SenseNova-U1.6-8B-MoT-GGUF/SenseNova-U1.6-8B-MoT-Q8_0.gguf \
      --model_format gguf --device cpu --dtype float32 \
      --out_dir /root/oracle_out/ref512
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

# --------------------------------------------------------------------------- #
# Fixed reference contract.  Do not edit without bumping REFERENCE_VERSION.
# --------------------------------------------------------------------------- #
REFERENCE_VERSION = 1

PROMPT = "A studio photograph of a red apple on a white table, soft daylight."

CONTRACT = {
    "prompt": PROMPT,
    "width": 512,
    "height": 512,
    "num_steps": 4,
    "seed": 42,
    "cfg_scale": 4.0,
    "cfg_norm": "none",
    "cfg_interval": (0.0, 1.0),
    "timestep_shift": 3.0,
    "t_eps": 0.02,
    "batch_size": 1,
    "method": "euler",
    "cfg_batching": True,
    "think_mode": False,
    "attn_backend": "sdpa",
    # Slices applied to oversized tensors (recorded in the manifest).
    "kv_seq_slice": 64,
    "kv_layers": [0, 1, 21, 41],
}

NORM_MEAN = (0.5, 0.5, 0.5)
NORM_STD = (0.5, 0.5, 0.5)

IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"


# --------------------------------------------------------------------------- #
# Capture plumbing
# --------------------------------------------------------------------------- #
class Capture:
    """Accumulates tensors as float32 CPU numpy arrays keyed by file stem."""

    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.entries: list[dict] = []

    def put(self, stem: str, tensor, *, role: str, note: str | None = None, sl: str | None = None):
        if tensor is None:
            self.entries.append({"file": None, "name": stem, "value": None, "role": role, "note": "None (skipped)"})
            return None
        if isinstance(tensor, torch.Tensor):
            arr = tensor.detach().to("cpu").numpy()
            if arr.dtype in (np.float64,):
                arr = arr.astype(np.float32)
        else:
            arr = np.asarray(tensor)
        path = self.out_dir / f"{stem}.npy"
        np.save(path, arr)
        self.entries.append(
            {
                "file": path.name,
                "role": role,
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "slice": sl,
                "note": note,
                "nbytes": int(arr.nbytes),
            }
        )
        return arr

    def info(self, name: str, value, note: str = ""):
        self.entries.append({"file": None, "name": name, "value": value, "note": note})


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, torch.dtype):
        return str(o)
    if isinstance(o, tuple):
        return list(o)
    return repr(o)


# --------------------------------------------------------------------------- #
# Verbatim replay of NEOChatModel.t2i_generate with capture points
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def run_t2i_contract(mnc, model, tokenizer, contract, cap: Capture):
    """Mirror of modeling_neo_chat.py::t2i_generate (non-think path)."""
    m = model

    cfg_scale = contract["cfg_scale"]
    cfg_norm = contract["cfg_norm"]
    cfg_interval = tuple(contract["cfg_interval"])
    timestep_shift = contract["timestep_shift"]
    num_steps = contract["num_steps"]
    image_size = (contract["width"], contract["height"])  # (W, H), as t2i_generate
    batch_size = contract["batch_size"]
    seed = contract["seed"]
    t_eps = contract["t_eps"]
    cfg_batching = contract["cfg_batching"]
    enable_timestep_shift = True

    assert m.concat_time_token_num == 0
    assert cfg_norm in ["cfg_zero_star", "global", "none", "channel"]
    merge_size = int(1 / m.downsample_ratio)

    m.config.t_eps = t_eps
    question_condition = f"{contract['prompt']}"

    think_text = ""
    needs_cfg = cfg_scale > 1

    # think_mode is False -> '<think>\n\n</think>\n\n' + IMG_START_TOKEN
    think_content = "<think>\n\n</think>\n\n" + IMG_START_TOKEN
    query_condition = m._build_t2i_query(
        question_condition, system_message=mnc.SYSTEM_MESSAGE_FOR_GEN, append_text=think_content
    )
    query_uncondition = m._build_t2i_query("", append_text=IMG_START_TOKEN) if needs_cfg else None

    input_ids_condition, indexes_condition, attention_mask_condition_prefix = m._build_t2i_text_inputs(
        tokenizer, query_condition
    )
    if query_uncondition is not None:
        input_ids_uncondition, indexes_uncondition, attention_mask_uncondition_prefix = m._build_t2i_text_inputs(
            tokenizer, query_uncondition
        )
    else:
        input_ids_uncondition = indexes_uncondition = attention_mask_uncondition_prefix = None

    token_h = image_size[1] // (m.patch_size * merge_size)
    token_w = image_size[0] // (m.patch_size * merge_size)

    indexes_image_condition = m._build_t2i_image_indexes(
        token_h, token_w, indexes_condition.shape[1], device=input_ids_condition.device
    )
    indexes_image_uncondition = (
        m._build_t2i_image_indexes(token_h, token_w, indexes_uncondition.shape[1], device=input_ids_uncondition.device)
        if indexes_uncondition is not None
        else None
    )

    cap.info("token_h", token_h)
    cap.info("token_w", token_w)
    cap.info("merge_size", merge_size)
    cap.info("patch_size", int(m.patch_size))
    cap.info("downsample_ratio", float(m.downsample_ratio))
    cap.info("image_token_num", int(token_h * token_w))
    cap.info("concat_time_token_num", int(m.concat_time_token_num))

    # ---- prompt / tokenizer dump -------------------------------------------
    tmpl = mnc.get_conv_template(m.template)
    special = {
        "img_start": IMG_START_TOKEN,
        "img_end": IMG_END_TOKEN,
        "img_context": IMG_CONTEXT_TOKEN,
        "im_start": "<|im_start|>",
        "im_end": "<|im_end|>",
    }
    special_ids = {k: int(tokenizer.convert_tokens_to_ids(v)) for k, v in special.items()}
    # single-token probes for the tokens a Rust tokenizer must get right
    for tok in ["<|im_start|>", "<|im_end|>", "<img>", "</img>", "<IMG_CONTEXT>", "<|endoftext|>"]:
        tid = tokenizer.convert_tokens_to_ids(tok)
        special_ids[f"id:{tok}"] = int(tid) if tid is not None else None

    cap.put("input_ids_cond", input_ids_condition, role="int64 token ids of the conditional query")
    cap.put("input_ids_uncond", input_ids_uncondition, role="int64 token ids of the unconditional query")
    cap.put("indexes_cond", indexes_condition, role="int64 (3, L) [t, h, w] position indexes, conditional prefix")
    cap.put("indexes_uncond", indexes_uncondition, role="int64 (3, L) [t, h, w] position indexes, unconditional prefix")
    cap.put(
        "indexes_image_cond",
        indexes_image_condition,
        role="int64 (3, token_h*token_w) image-token position indexes, conditional branch",
    )
    cap.put(
        "indexes_image_uncond",
        indexes_image_uncondition,
        role="int64 (3, token_h*token_w) image-token position indexes, unconditional branch",
    )

    prompt_doc = {
        "contract_prompt": contract["prompt"],
        "question_condition": question_condition,
        "query_condition": query_condition,
        "query_uncondition": query_uncondition,
        "think_content_suffix": think_content,
        "chat_template": m.template,
        "system_message_for_gen": mnc.SYSTEM_MESSAGE_FOR_GEN,
        "system_message_attr": m.system_message,
        "roles": list(tmpl.roles),
        "sep": tmpl.sep,
        "sep_style": str(tmpl.sep_style),
        "special_token_ids": special_ids,
        "input_ids_condition": input_ids_condition[0].tolist(),
        "input_ids_uncondition": input_ids_uncondition[0].tolist() if input_ids_uncondition is not None else None,
        "prefix_len_condition": int(input_ids_condition.shape[1]),
        "prefix_len_uncondition": int(input_ids_uncondition.shape[1]) if input_ids_uncondition is not None else None,
    }
    # Canonical filename required by docs/develop/SENSENOVA_U1.md §6.
    (cap.out_dir / "prompt_tokens.json").write_text(json.dumps(prompt_doc, indent=2, ensure_ascii=False))

    # ---- prefix forward ----------------------------------------------------
    t0 = time.time()
    past_key_values_condition, prefix_hidden_states = m._t2i_prefix_forward(
        input_ids_condition, indexes_condition, attention_mask_condition_prefix
    )
    prefix_cond_s = time.time() - t0
    device = prefix_hidden_states.device
    dtype = prefix_hidden_states.dtype

    past_key_values_uncondition = None
    if input_ids_uncondition is not None:
        t0 = time.time()
        past_key_values_uncondition, prefix_hidden_states_uncond = m._t2i_prefix_forward(
            input_ids_uncondition, indexes_uncondition, attention_mask_uncondition_prefix
        )
        prefix_uncond_s = time.time() - t0
        del prefix_hidden_states_uncond
    else:
        prefix_uncond_s = 0.0

    cap.info("prefix_forward_seconds_cond", round(prefix_cond_s, 3))
    cap.info("prefix_forward_seconds_uncond", round(prefix_uncond_s, 3))
    cap.info("kv_dtype", str(dtype))

    # ---- prefix KV capture (BEFORE expand / prepare_flash_kv_cache) --------
    kv_slice = contract["kv_seq_slice"]
    layers = past_key_values_condition.layers
    cap.info("num_kv_layers", len(layers))
    lk0 = layers[0].keys
    cap.info("kv_layer0_raw_shape", list(lk0.shape))
    cap.info("kv_layer0_num_kv_heads", int(lk0.shape[1]) if lk0.dim() == 4 else None)
    cap.info(
        "kv_all_layer_shapes",
        [
            {
                "layer": i,
                "keys": list(l.keys.shape) if l.keys is not None else None,
                "values": list(l.values.shape) if l.values is not None else None,
            }
            for i, l in enumerate(layers)
        ],
        note="metadata only (no payload); layout [B, num_kv_heads, seq, head_dim] before prepare_flash_kv_cache",
    )
    for li in contract["kv_layers"]:
        if li >= len(layers):
            cap.info(f"kv_layer{li}", "MISSING (fewer layers)")
            continue
        k = layers[li].keys
        v = layers[li].values
        s = kv_slice if kv_slice is not None else k.shape[-2]
        s = min(int(s), int(k.shape[-2]))
        note = f"prefix KV cache after prefix forward, sliced to first {s} of {int(k.shape[-2])} sequence positions"
        cap.put(f"prefix_kv_layer{li:02d}_k", k[:, :, :s, :], role=f"layer {li} key", note=note, sl=f"[:, :, :{s}, :]")
        cap.put(f"prefix_kv_layer{li:02d}_v", v[:, :, :s, :], role=f"layer {li} value", note=note, sl=f"[:, :, :{s}, :]")
    cap.put(
        "prefix_last_hidden_state_cond",
        prefix_hidden_states,
        role="language_model.model(...).last_hidden_state for the conditional prefix (full, no slice)",
    )
    del prefix_hidden_states

    # ---- batch expand + flash cache prep (verbatim) ------------------------
    for layer_idx in range(len(past_key_values_condition.layers)):
        past_key_values_condition.layers[layer_idx].keys = past_key_values_condition.layers[layer_idx].keys.expand(
            batch_size, *past_key_values_condition.layers[layer_idx].keys.shape[1:]
        )
        past_key_values_condition.layers[layer_idx].values = past_key_values_condition.layers[layer_idx].values.expand(
            batch_size, *past_key_values_condition.layers[layer_idx].values.shape[1:]
        )
        if past_key_values_uncondition is not None:
            past_key_values_uncondition.layers[layer_idx].keys = past_key_values_uncondition.layers[
                layer_idx
            ].keys.expand(batch_size, *past_key_values_uncondition.layers[layer_idx].keys.shape[1:])
            past_key_values_uncondition.layers[layer_idx].values = past_key_values_uncondition.layers[
                layer_idx
            ].values.expand(batch_size, *past_key_values_uncondition.layers[layer_idx].values.shape[1:])

    mnc.prepare_flash_kv_cache(past_key_values_condition, current_len=token_h * token_w, batch_size=batch_size)
    if past_key_values_uncondition is not None:
        mnc.prepare_flash_kv_cache(past_key_values_uncondition, current_len=token_h * token_w, batch_size=batch_size)

    # ---- initial noise (verbatim: same RNG call, same seed) ----------------
    grid_h = image_size[1] // m.patch_size
    grid_w = image_size[0] // m.patch_size
    grid_hw = torch.tensor([[grid_h, grid_w]] * batch_size, device=device)

    noise_scale = m.noise_scale
    if m.noise_scale_mode in ("resolution", "dynamic", "dynamic_sqrt"):
        base = float(m.noise_scale_base_image_seq_len)
        scale = math.sqrt((grid_h * grid_w) / (merge_size**2) / base)
        noise_scale = scale * float(m.noise_scale)
        if m.noise_scale_mode == "dynamic_sqrt":
            noise_scale = math.sqrt(noise_scale)
    noise_scale = min(noise_scale, m.noise_scale_max_value)
    cap.info("noise_scale_mode", str(m.noise_scale_mode))
    cap.info("noise_scale_base_image_seq_len", float(m.noise_scale_base_image_seq_len))
    cap.info("noise_scale_max_value", float(m.noise_scale_max_value))
    cap.info("noise_scale", float(noise_scale))
    cap.info("grid_h", int(grid_h))
    cap.info("grid_w", int(grid_w))

    generator = torch.Generator(device).manual_seed(seed)
    image_prediction = noise_scale * torch.randn(
        (batch_size, 3, image_size[1], image_size[0]), device=device, dtype=dtype, generator=generator
    )
    cap.put(
        "noise_init",
        image_prediction,
        role="image_prediction right after noise_scale * randn(...) (normalized space, NOT pixels)",
    )

    attention_mask_condition = {"full_attention": None}
    cfg_cache = None
    if (
        cfg_batching
        and needs_cfg
        and not m.training
        and isinstance(m.language_model, (mnc.Qwen3ForCausalLM, mnc.Qwen3MoeForCausalLM))
    ):
        attention = m.language_model.model.layers[0].self_attn
        if all(
            rope.rope_type in (None, "default")
            for rope in (attention.rotary_emb, attention.rotary_emb_hw)
        ):
            cfg_cache = mnc.BatchedCFGCache((past_key_values_condition, past_key_values_uncondition))
    cap.info("cfg_batching_effective", cfg_cache is not None)

    timesteps = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
    timesteps_raw = timesteps.clone()
    if enable_timestep_shift:
        timesteps = m._apply_time_schedule(timesteps, token_h * token_w, timestep_shift)
    cap.put("timesteps", timesteps, role="shifted timestep schedule actually fed to the model (num_steps+1,)")
    cap.put("timesteps_raw", timesteps_raw, role="pre-shift linspace(0,1,num_steps+1)")
    cap.info("time_schedule_attr_after", str(m.time_schedule))

    step_times = []
    for step_i in range(num_steps):
        ts = time.time()
        t = timesteps[step_i]
        t_next = timesteps[step_i + 1]

        z = m.patchify(image_prediction, m.patch_size * merge_size)
        image_input = m.patchify(image_prediction, m.patch_size, channel_first=True)
        image_embeds = m.extract_feature(
            image_input.view(batch_size * grid_h * grid_w, -1), gen_model=True, grid_hw=grid_hw
        ).view(batch_size, token_h * token_w, -1)
        t_expanded = t.expand(batch_size * token_h * token_w)
        timestep_embeddings = m.fm_modules["timestep_embedder"](t_expanded).view(batch_size, token_h * token_w, -1)
        if m.add_noise_scale_embedding:
            noise_scale_tensor = torch.full_like(t_expanded, noise_scale / m.noise_scale_max_value)
            noise_embeddings = m.fm_modules["noise_scale_embedder"](noise_scale_tensor).view(
                batch_size, token_h * token_w, -1
            )
            timestep_embeddings = timestep_embeddings + noise_embeddings
        image_embeds = image_embeds + timestep_embeddings

        v_pred_condition = None
        v_pred_uncondition = None
        if t >= cfg_interval[0] and t <= cfg_interval[1] and cfg_scale > 1:
            v_pred_condition, v_pred_uncondition = m._t2i_predict_cfg(
                image_embeds,
                (indexes_image_condition, indexes_image_uncondition),
                (past_key_values_condition, past_key_values_uncondition),
                t,
                z,
                token_h * token_w,
                cfg_cache=cfg_cache,
                timestep_embeddings=timestep_embeddings,
                image_size=image_size,
            )
            if cfg_norm == "cfg_zero_star":
                positive_flat = v_pred_condition.view(batch_size, -1)
                negative_flat = v_pred_uncondition.view(batch_size, -1)
                alpha = mnc.optimized_scale(positive_flat, negative_flat)
                alpha = alpha.view(batch_size, *([1] * (len(v_pred_condition.shape) - 1)))
                alpha = alpha.to(positive_flat.dtype)
                if step_i <= 0:
                    v_pred = v_pred_condition * 0.0
                else:
                    v_pred = v_pred_uncondition * alpha + cfg_scale * (v_pred_condition - v_pred_uncondition * alpha)
            else:
                v_pred = v_pred_uncondition + cfg_scale * (v_pred_condition - v_pred_uncondition)
                if cfg_norm == "global":
                    norm_v_condition = torch.norm(v_pred_condition, dim=(1, 2), keepdim=True)
                    norm_v_cfg = torch.norm(v_pred, dim=(1, 2), keepdim=True)
                    scale = (norm_v_condition / (norm_v_cfg + 1e-8)).clamp(min=0, max=1.0)
                    v_pred = v_pred * scale
                elif cfg_norm == "channel":
                    norm_v_condition = torch.norm(v_pred_condition, dim=-1, keepdim=True)
                    norm_v_cfg = torch.norm(v_pred, dim=-1, keepdim=True)
                    scale = (norm_v_condition / (norm_v_cfg + 1e-8)).clamp(min=0, max=1.0)
                    v_pred = v_pred * scale
        else:
            v_pred = m._t2i_predict_v(
                image_embeds,
                indexes_image_condition,
                attention_mask_condition,
                past_key_values_condition,
                t,
                z,
                image_token_num=token_h * token_w,
                timestep_embeddings=timestep_embeddings,
                image_size=image_size,
            )

        z_next = z + (t_next - t) * v_pred
        image_prediction = m.unpatchify(z_next, m.patch_size * merge_size, image_size[1], image_size[0])

        # ---- per-step capture ---------------------------------------------
        tag = f"step{step_i}"
        cap.put(f"{tag}_t", t, role=f"step {step_i} t (shifted schedule)")
        cap.put(f"{tag}_t_next", t_next, role=f"step {step_i} t_next (shifted schedule)")
        cap.put(f"{tag}_z", z, role=f"step {step_i} patchified latent z BEFORE the Euler step")
        cap.put(f"{tag}_v_pred", v_pred, role=f"step {step_i} v_pred AFTER CFG combine (cfg_scale={cfg_scale})")
        if v_pred_condition is not None:
            cap.put(f"{tag}_v_pred_cond", v_pred_condition, role=f"step {step_i} raw conditional velocity")
            cap.put(f"{tag}_v_pred_uncond", v_pred_uncondition, role=f"step {step_i} raw unconditional velocity")
        cap.put(f"{tag}_z_next", z_next, role=f"step {step_i} z + (t_next - t) * v_pred")
        cap.put(
            f"{tag}_v_denominator",
            (1 - t).clamp_min(t_eps),
            role=f"step {step_i} divisor predict_v uses: v_pred = (x_pred - z) / (1-t).clamp_min(t_eps)",
        )
        if step_i == 0:
            cap.put(
                f"{tag}_image_embeds",
                image_embeds,
                role="step 0 embeddings fed to the denoiser = vision_mot_gen(patchify(x,16,channel_first)) "
                "+ timestep_embedder(t) + noise_scale_embedder(noise_scale/16)",
            )
            cap.info(
                "step0_derivable_inputs",
                "image_input and timestep_embeddings are NOT dumped separately because they are exactly "
                "reconstructible: image_input = patchify(noise_init, 16, channel_first=True) -> "
                "(1, grid_h*grid_w, 768); timestep_embeddings = timestep_embedder(t) + "
                "noise_scale_embedder(noise_scale/16) -> (1, token_h*token_w, 4096).",
            )
        step_times.append(round(time.time() - ts, 3))
        print(f"[ref] step {step_i}: t={float(t):.6f} t_next={float(t_next):.6f} "
              f"z{tuple(z.shape)} |v|={float(v_pred.abs().mean()):.6f} ({step_times[-1]}s)", flush=True)

    mnc.clear_flash_kv_cache(past_key_values_condition)
    if past_key_values_uncondition is not None:
        mnc.clear_flash_kv_cache(past_key_values_uncondition)

    m.last_think_content = think_text
    cap.info("step_seconds", step_times)
    return image_prediction


# --------------------------------------------------------------------------- #
# Official PNG render path (copied from examples/t2i/inference.py)
# --------------------------------------------------------------------------- #
def _denorm(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(NORM_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(NORM_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x * std + mean).clamp(0, 1)


def to_pil(batch: torch.Tensor):
    from PIL import Image

    arr = _denorm(batch.float()).permute(0, 2, 3, 1).cpu().numpy()
    arr = (arr * 255.0).round().astype(np.uint8)
    return [Image.fromarray(a) for a in arr]


# --------------------------------------------------------------------------- #
# state_dict inventory
# --------------------------------------------------------------------------- #
def _row(name: str, t) -> dict:
    row = {"name": name}
    try:
        row["shape"] = [int(d) for d in t.shape]
    except Exception:
        row["shape"] = None
    try:
        row["dtype"] = str(t.dtype)
    except Exception:
        row["dtype"] = None
    row["type"] = type(t).__name__
    qt = getattr(t, "quant_type", None)
    if qt is not None:
        row["quant_type"] = str(qt)
    try:
        row["device"] = str(t.device)
    except Exception:
        pass
    try:
        row["numel"] = int(t.numel())
    except Exception:
        row["numel"] = None
    return row


def dump_state_dict(model, path: Path):
    rows, params, gguf_linears, meta_like = [], [], [], []
    try:
        sd = model.state_dict()
    except Exception:
        traceback.print_exc()
        sd = {}
    for name, t in sd.items():
        row = _row(name, t)
        rows.append(row)
        if row.get("device") == "meta" or row.get("type") == "Tensor" and row.get("numel") is None:
            meta_like.append(name)
    for name, p in model.named_parameters():
        params.append(_row(name, p))
    for mod_name, mod in model.named_modules():
        if type(mod).__name__ == "GGUFLinear":
            w = getattr(mod, "weight", None)
            gguf_linears.append(
                {
                    "module": mod_name,
                    "weight_shape": [int(d) for d in w.shape] if w is not None else None,
                    "weight_type": type(w).__name__ if w is not None else None,
                    "quant_type": str(getattr(w, "quant_type", None)) if w is not None else None,
                }
            )
    doc = {
        "state_dict_count": len(rows),
        "param_count": len(params),
        "gguf_linear_count": len(gguf_linears),
        "meta_like_count": len(meta_like),
        "state_dict": rows,
        "named_parameters": params,
        "gguf_linears": gguf_linears,
    }
    path.write_text(json.dumps(doc, indent=1))
    return len(rows), len(params)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--model_format", default="gguf", choices=["auto", "bf16", "fp8", "int8-convrot", "gguf"])
    ap.add_argument("--gguf_checkpoint", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--skip-state-dict", action="store_true")
    ap.add_argument("--verify-png", default=None, help="existing PNG from examples/t2i/inference.py to compare against")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import sensenova_u1
    from sensenova_u1.models.neo_unify import modeling_neo_chat as mnc
    from sensenova_u1.utils import load_model_and_tokenizer_for_format

    sensenova_u1.set_attn_backend(CONTRACT["attn_backend"])
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    print(f"[ref] attn_backend requested={CONTRACT['attn_backend']} effective={sensenova_u1.effective_attn_backend()!r}")
    print(f"[ref] loading {args.model_path} format={args.model_format} dtype={args.dtype} device={args.device}", flush=True)
    t_load = time.time()
    model, tokenizer = load_model_and_tokenizer_for_format(
        args.model_path,
        model_format=args.model_format,
        dtype=dtype,
        device=args.device,
        gguf_checkpoint=args.gguf_checkpoint,
        for_offload=False,
    )
    load_s = time.time() - t_load
    print(f"[ref] model loaded in {load_s:.1f}s", flush=True)

    cap = Capture(out_dir)
    cap.info("load_seconds", round(load_s, 3))

    state_dict_count = param_count = None
    if not args.skip_state_dict:
        state_dict_count, param_count = dump_state_dict(model, out_dir / "state_dict_keys.json")
        print(f"[ref] state_dict: {state_dict_count} keys, {param_count} params", flush=True)

    t_gen = time.time()
    final = run_t2i_contract(mnc, model, tokenizer, CONTRACT, cap)
    gen_s = time.time() - t_gen
    print(f"[ref] generation done in {gen_s:.1f}s", flush=True)

    cap.put("final_image_prediction", final, role="the tensor t2i_generate returns (normalized space, [-1,1]-ish)")
    pil = to_pil(final)[0]
    pil.save(out_dir / "final.png")

    verify = None
    if args.verify_png:
        from PIL import Image

        ref = Image.open(args.verify_png).convert("RGB")
        mine = pil.convert("RGB")
        if ref.size != mine.size:
            verify = {"match": False, "reason": f"size {ref.size} != {mine.size}"}
        else:
            a = np.asarray(ref, dtype=np.int16)
            b = np.asarray(mine, dtype=np.int16)
            d = np.abs(a - b)
            verify = {
                "reference_png": args.verify_png,
                "match_exact": bool((d == 0).all()),
                "max_abs_diff_u8": int(d.max()),
                "mean_abs_diff_u8": float(d.mean()),
                "pct_pixels_diff": float((d.max(axis=-1) > 0).mean() * 100.0),
            }
            print(f"[ref] verify vs {args.verify_png}: {verify}", flush=True)

    manifest = {
        "reference_version": REFERENCE_VERSION,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "contract": CONTRACT,
        "model_path": args.model_path,
        "model_format": args.model_format,
        "device": args.device,
        "dtype": args.dtype,
        "attn_backend_effective": str(sensenova_u1.effective_attn_backend()),
        "load_seconds": round(load_s, 3),
        "generate_seconds": round(gen_s, 3),
        "versions": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_hip": getattr(torch.version, "hip", None),
            "numpy": np.__version__,
        },
        "oracle_versions": _oracle_versions(),
        "state_dict_count": state_dict_count,
        "param_count": param_count,
        "tensors": cap.entries,
        "total_npy_bytes": sum(e.get("nbytes", 0) for e in cap.entries if e.get("file")),
        "verify_png": verify,
        "notes": [
            "All .npy are float32 unless dtype says int64.",
            "Latents/tensors are in the model's normalized image space (mean=std=0.5), not [0,1] pixels.",
            "z is patchified with patch_size*merge_size = 32 -> (1, token_h*token_w, 3*32*32=3072).",
            "z_next[i] is bit-identical to z[i+1] (the oracle reuses the same variable); z_next[last] equals "
            "patchify(final_image_prediction, 32).",
            "predict_v returns v_pred = (x_pred - z) / (1 - t).clamp_min(config.t_eps), where x_pred is the "
            "ConvDecoder output; x_pred is therefore exactly recoverable as v_pred * v_denominator + z.",
            "KV dumps are sliced on the sequence axis and are taken BEFORE the oracle's "
            "prepare_flash_kv_cache() transpose, i.e. layout is [B, num_kv_heads, seq, head_dim].",
            "The oracle's neo1_0 template emits NO system block when system_message == '' (see "
            "conversation.py get_prompt / SeparatorStyle.MPT), so the unconditional query starts directly "
            "with '<|im_start|>user\\n'; an empty *user* message still emits role+sep. This differs from the "
            "rendering sketch in docs/develop/SENSENOVA_U1.md §5 which always shows a system block.",
            "cfg_batching=True takes the oracle's batched path (predict_cfg_batched: concat [cond, uncond] "
            "then .chunk(2)), which is the official CLI default; a serial-CFG port may see tiny GEMM-order "
            "differences.",
            "t2i_generate hardcodes self.time_schedule='standard' inside _apply_time_schedule, so the config's "
            "time_schedule='dynamic' and base_shift/max_shift are dead code in this release.",
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=_json_default))
    total = manifest["total_npy_bytes"]
    print(f"[ref] wrote {len([e for e in cap.entries if e.get('file')])} tensors, {total/1e6:.1f} MB -> {out_dir}", flush=True)
    return 0


def _oracle_versions() -> dict:
    out = {}
    try:
        import importlib.metadata as md

        for pkg in ("sensenova-u1", "transformers", "accelerate", "diffusers", "gguf", "safetensors", "sentencepiece", "pillow"):
            try:
                out[pkg] = md.version(pkg)
            except Exception:
                out[pkg] = None
    except Exception:
        pass
    try:
        import subprocess

        out["git_commit"] = (
            subprocess.check_output(["git", "-C", "/tmp/sn_oracle", "rev-parse", "HEAD"], text=True).strip()
        )
        out["git_describe"] = (
            subprocess.check_output(["git", "-C", "/tmp/sn_oracle", "log", "--oneline", "-1"], text=True).strip()
        )
    except Exception:
        pass
    return out


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
