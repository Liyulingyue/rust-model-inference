"""Run and compare the pinned Astronex-World Python graph.

The Oracle is deliberately a small runner around the read-only official
checkout.  `Scalar` owns every floating operator that is compared.  Missing
weights, a dirty/wrong reference checkout, lossy casts, unknown operators, or
an empty/ambiguous trace are hard errors.

Typical commands (from the repository root)::

  .venv/bin/python tools/oracle/astronex_world/trace_official.py self-test
  .venv/bin/python tools/oracle/astronex_world/trace_official.py fixture out/oracle
  .venv/bin/python tools/oracle/astronex_world/trace_official.py trace --weights models/Astronex-World out/oracle
  .venv/bin/python tools/oracle/astronex_world/trace_official.py compare out/oracle out/native

Real runs require complete denoiser, tokenizer/text-encoder, and VAE files.
The fixture exercises the same strict C primitives and trace contract without
pretending to be a real-weight end-to-end result.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Iterable

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
REFERENCE = REPO / "references" / "astronex-world"
REFERENCE_COMMIT = "27584bf1a89da01ef35a03b48d83bc818d2adf2b"
CONTRACT_COMMIT = "dec4ec502f7113a1ab07b78ae40a7ac1b88b2eeb"

sys.path.insert(0, str(HERE))
from scalar import FLAGS, Scalar  # noqa: E402


def _guard_reference(path: Path = REFERENCE) -> str:
    if not (path / ".git").exists():
        raise RuntimeError(f"missing official checkout: {path}")
    commit = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    if commit != REFERENCE_COMMIT:
        raise RuntimeError(f"official source must be {REFERENCE_COMMIT}, got {commit}")
    dirty = subprocess.check_output(["git", "-C", str(path), "status", "--porcelain"], text=True)
    if dirty.strip():
        raise RuntimeError("official source checkout is dirty; use a clean isolated clone")
    return commit


def _guard_contract(path: Path | None) -> str | None:
    if path is None:
        return None
    commit = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    if commit != CONTRACT_COMMIT:
        raise RuntimeError(f"contract source must be {CONTRACT_COMMIT}, got {commit}")
    dirty = subprocess.check_output(["git", "-C", str(path), "status", "--porcelain"], text=True)
    if dirty.strip():
        raise RuntimeError("contract source checkout is dirty")
    return commit


class Trace:
    """Append-only raw F32/token trace with occurrence and shape contracts."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.records: list[dict] = []
        self.counts: collections.Counter[str] = collections.Counter()

    def emit(self, name: str, value, *, kind: str = "f32") -> None:
        if kind == "token_ids":
            ids = [int(x) for x in value]
            if not ids:
                raise RuntimeError(f"empty token checkpoint: {name}")
            occurrence = self.counts[name]
            self.counts[name] += 1
            self.records.append({"name": name, "occurrence": occurrence,
                                 "kind": kind, "shape": [len(ids)], "values": ids})
            return
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name}: checkpoint is not a tensor")
        if value.numel() == 0:
            raise RuntimeError(f"empty floating checkpoint: {name}")
        if value.dtype != torch.float32:
            raise RuntimeError(f"{name}: expected F32 checkpoint, got {value.dtype}")
        data = value.detach().contiguous().cpu().numpy()
        if not np.isfinite(data).all():
            raise RuntimeError(f"{name}: non-finite value")
        occurrence = self.counts[name]
        self.counts[name] += 1
        path = self.root / f"{len(self.records):06d}.{name.replace('/', '_')}.{occurrence}.f32"
        data.astype("<f4", copy=False).tofile(path)
        self.records.append({"name": name, "occurrence": occurrence,
                             "kind": kind, "shape": list(data.shape),
                             "dtype": "f32", "binary_path": str(path.resolve()),
                             "count": int(data.size)})

    def finish(self, *, metadata: dict | None = None) -> None:
        if not self.records:
            raise RuntimeError("trace has no checkpoints")
        # A second identical occurrence is valid (cache path); an empty name or
        # non-monotone occurrence is never valid and usually means a bad hook.
        for rec in self.records:
            if not rec.get("name") or rec.get("occurrence", -1) < 0:
                raise RuntimeError(f"invalid checkpoint record: {rec}")
            if rec["kind"] == "f32":
                p = Path(rec["binary_path"])
                if not p.is_file() or p.stat().st_size != rec["count"] * 4:
                    raise RuntimeError(f"invalid checkpoint binary: {rec}")
        (self.root / "trace.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in self.records))
        payload = {"checkpoints": len(self.records), "counts": dict(self.counts),
                   "arithmetic": "independent scalar C; F32 graph; official F64 RoPE only",
                   "compiler_flags": FLAGS, **(metadata or {})}
        (self.root / "environment.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def _tensor_output(value):
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for v in value:
            if isinstance(v, torch.Tensor):
                return v
    if isinstance(value, dict):
        return _tensor_output(list(value.values()))
    return None


def _hook(trace: Trace, name: str):
    def callback(_module, _inputs, output):
        value = _tensor_output(output)
        if value is not None and value.dtype == torch.float32:
            trace.emit(name, value)
    return callback


def _patch_scalar_attention(scalar: Scalar):
    """Patch attention and the official F64 time-frequency helper.

    The graph remains F32 everywhere except the official
    ``sinusoidal_embedding_1d`` implementation, whose source explicitly
    promotes its position/frequency calculation to F64.  Keep that narrow
    exception inside ``Scalar.f64_scope`` while still making every other F64
    operation fail closed.
    """
    import models.attention as attention_module
    import models.causal_model as causal_module
    import models.wan_model as wan_module
    original = (attention_module.attention, getattr(causal_module, "attention", None),
                attention_module.flash_attention, wan_module.flash_attention,
                wan_module.sinusoidal_embedding_1d,
                causal_module.sinusoidal_embedding_1d)

    def strict_attention(q, k, v, **kwargs):
        if any(t.dtype != torch.float32 for t in (q, k, v)):
            raise ValueError("Oracle attention requires F32 q/k/v; BF16 downcast escaped")
        # Official Wan layout is [B, L, H, D], while the scalar C primitive is
        # [B, H, L, D]. The causal cache path supplies already-valid prefixes;
        # q_lens/k_lens masks are therefore unnecessary here and would only add
        # a non-official masking operation.
        qh, kh, vh = (t.transpose(1, 2).contiguous() for t in (q, k, v))
        out = scalar.attention(qh, kh, vh, scale=kwargs.get("softmax_scale"))
        return out.transpose(1, 2).contiguous()

    attention_module.attention = strict_attention
    causal_module.attention = strict_attention
    attention_module.flash_attention = strict_attention
    wan_module.flash_attention = strict_attention

    original_wan_sinusoidal = wan_module.sinusoidal_embedding_1d
    original_causal_sinusoidal = causal_module.sinusoidal_embedding_1d

    def strict_wan_sinusoidal(dim, position):
        with scalar.f64_scope():
            return original_wan_sinusoidal(dim, position)

    def strict_causal_sinusoidal(dim, position):
        with scalar.f64_scope():
            return original_causal_sinusoidal(dim, position)

    wan_module.sinusoidal_embedding_1d = strict_wan_sinusoidal
    causal_module.sinusoidal_embedding_1d = strict_causal_sinusoidal
    return original


def _restore_scalar_attention(original):
    import models.attention as attention_module
    import models.causal_model as causal_module
    import models.wan_model as wan_module
    (attention_module.attention, causal_module.attention,
     attention_module.flash_attention, wan_module.flash_attention,
     wan_module.sinusoidal_embedding_1d,
     causal_module.sinusoidal_embedding_1d) = original


def _module_hooks(module, trace: Trace, prefix: str, limit: int | None = None):
    handles = []
    for name, child in module.named_modules():
        if not name:
            continue
        # Keep the trace useful and bounded: every official block output plus
        # the arithmetic-heavy children. The full module list is in the model
        # state contract and would repeat unchanged container outputs.
        leaf = name.split(".")[-1]
        if child.__class__.__name__ in {"Linear", "Conv3d", "UMT5LayerNorm", "WanRMSNorm", "WanLayerNorm"} or leaf in {
            "self_attn", "cross_attn", "ffn", "DenseReluDense", "encoder", "decoder", "quant_conv", "post_quant_conv",
        }:
            handles.append(child.register_forward_hook(_hook(trace, f"{prefix}.{name}")))
            if limit is not None and len(handles) >= limit:
                break
    return handles


def _load_indexed_state(weights: Path, model: torch.nn.Module) -> tuple[int, int]:
    """Load complete safetensors shards one at a time and verify exact keys."""
    index = weights / "model.safetensors.index.json"
    if not index.is_file():
        raise RuntimeError(f"missing model.safetensors.index.json under {weights}")
    index_data = json.loads(index.read_text())
    weight_map = index_data.get("weight_map", {})
    if not weight_map:
        raise RuntimeError("empty model weight map")
    shards = sorted(set(weight_map.values()))
    for name in shards:
        path = weights / name
        if not path.is_file() or path.name.endswith(".incomplete"):
            raise RuntimeError(f"incomplete/missing weight shard: {path}")
    from safetensors.torch import load_file
    expected = set(model.state_dict())
    loaded: set[str] = set()
    unexpected: set[str] = set()
    for shard in shards:
        state = load_file(str(weights / shard), device="cpu")
        # Converter keeps the `model.` wrapper used by the official wrapper.
        state = {k.removeprefix("model."): v for k, v in state.items()}
        unknown = set(state) - expected
        unexpected |= unknown
        state = {k: v for k, v in state.items() if k in expected}
        model.load_state_dict(state, strict=False)
        loaded |= set(state)
        del state
    missing = expected - loaded
    if missing or unexpected:
        raise RuntimeError(f"weight key contract failed: missing={len(missing)} unexpected={len(unexpected)}")
    return len(loaded), len(shards)


def _build_causal(weights: Path):
    _guard_reference()
    sys.path.insert(0, str(REFERENCE))
    from models.causal_model import CausalWanModel
    from models.action import add_action_parameters
    from models.prope import add_prope_parameters
    config_path = weights / "transformer" / "config.json"
    if not config_path.is_file():
        raise RuntimeError(f"missing denoiser config: {config_path}")
    from torch.utils._python_dispatch import _disable_current_modes
    # Construction/load is setup. Keeping it outside Scalar prevents model
    # initialisers from masquerading as graph arithmetic; the forward below is
    # still entirely intercepted.
    with _disable_current_modes():
        config = CausalWanModel.load_config(str(weights / "transformer"))
        model = CausalWanModel.from_config(config, local_attn_size=20, sink_size=4)
        add_prope_parameters(model, zero_init=False)
        add_action_parameters(model, action_dim=64, num_embodiments=32)
        model.add_action_head(64)
        loaded, shards = _load_indexed_state(weights, model)
        model.float().eval()
    return model, loaded, shards


def _trace_tokens_text(weights: Path, trace: Trace, prompt: str, scalar: Scalar):
    sys.path.insert(0, str(REFERENCE))
    from models.wan22_components import Wan22TextEncoder
    root = weights
    tokenizer_dir = root / "tokenizer"
    text_dir = root / "text_encoder"
    if not tokenizer_dir.is_dir() or not text_dir.is_dir():
        raise RuntimeError(f"missing tokenizer/text_encoder under {root}")
    from torch.utils._python_dispatch import _disable_current_modes
    with _disable_current_modes():
        encoder = Wan22TextEncoder(root=root, dtype=torch.float32).float().eval()
        ids = encoder.tokenizer(prompt, padding="max_length", max_length=512,
                                truncation=True, add_special_tokens=True, return_tensors="pt").input_ids[0]
    trace.emit("astronex.text.token_ids", ids.tolist(), kind="token_ids")
    handles = _module_hooks(encoder.text_encoder, trace, "astronex.text", limit=400)
    with torch.inference_mode(), scalar:
        out = encoder([prompt])["prompt_embeds"].float()
    for h in handles:
        h.remove()
    trace.emit("astronex.text.output", out)


def _trace_text_small(trace: Trace, scalar: Scalar):
    """Exercise UMT5's gated ``gelu_new`` and relative attention on fixed IDs."""
    from transformers import UMT5Config, UMT5EncoderModel
    from torch.utils._python_dispatch import _disable_current_modes
    with _disable_current_modes():
        model = UMT5EncoderModel(UMT5Config(
            vocab_size=32, d_model=8, d_kv=2, d_ff=16,
            num_layers=1, num_heads=2, dropout_rate=0.0,
        )).float().eval()
    ids = torch.tensor([[1, 2, 3, 0]], dtype=torch.long)
    mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.long)
    trace.emit("astronex.text.token_ids", ids[0].tolist(), kind="token_ids")
    with torch.inference_mode(), scalar:
        out = model(ids, attention_mask=mask).last_hidden_state.float()
        trace.emit("astronex.text.output", out)


def _cache(batch: int, frames: int, frame_tokens: int, heads: int, dim: int,
           blocks: int = 30, device="cpu"):
    n = 20 * frame_tokens
    return [{"k": torch.zeros(batch,n,heads,dim), "v": torch.zeros(batch,n,heads,dim),
             "global_end_index": torch.tensor([0]), "local_end_index": torch.tensor([0])} for _ in range(blocks)]


def _trace_denoiser(model, trace: Trace, scalar: Scalar, context: torch.Tensor,
                    frames: int = 8, latent_hw: tuple[int,int] = (2,2)):
    channels = int(model.in_dim)
    h,w = latent_hw
    # Fixed tiny inputs are deliberate: 32x32 pixel inputs become 2x2 latent
    # cells after the 16x VAE compression and one 2x2 Wan patch per frame.
    n_values = 1 * channels * frames * h * w
    # Integer arange is layout/input construction, then the explicit scalar
    # divide is part of the traced F32 graph. Random factories are forbidden so
    # the same Oracle run cannot silently choose a different noise tensor.
    x = (torch.arange(n_values, dtype=torch.int64).reshape(1, channels, frames, h, w)
         .to(torch.float32) - 512.0) / 1000.0
    t = torch.full((1, frames), 100, dtype=torch.long)
    cameras = torch.eye(4).reshape(1,1,4,4).repeat(1,frames,1,1)
    Ks = torch.eye(3).reshape(1,1,3,3).repeat(1,frames,1,1)
    actions = torch.zeros((1,frames,64))
    actions[...,0] = torch.arange(frames)
    embodiment = torch.zeros((1,), dtype=torch.long)
    seq = frames * (h // 2) * (w // 2)
    heads, dim = model.num_heads, model.dim // model.num_heads
    cache = _cache(1, frames, (h//2)*(w//2), heads, dim, blocks=len(model.blocks))
    cross = [{"k":torch.zeros(1,512,heads,dim),"v":torch.zeros(1,512,heads,dim),"is_init":False} for _ in model.blocks]
    prope = _cache(1, frames, (h//2)*(w//2), heads, dim, blocks=len(model.blocks))
    handles = _module_hooks(model, trace, "astronex.denoiser", limit=3000)
    trace.emit("astronex.denoiser.input", x)
    original_attention = _patch_scalar_attention(scalar)
    try:
        with torch.inference_mode(), scalar:
            out = model(x, t=t, context=[context[0]], seq_len=seq,
                        kv_cache=cache, crossattn_cache=cross, current_start=0,
                        cache_start=0, viewmats=cameras, Ks=Ks,
                        prope_kv_cache=prope, actions=actions,
                        embodiment_id=embodiment)
            trace.emit("astronex.denoiser.output.cache0", out.float())
            # A second chunk exercises cache persistence and camera/action slicing.
            x2 = torch.tensor(x.detach().cpu().numpy() * 0.5 + 0.125)
            t2 = torch.full((1,frames), 50, dtype=torch.long)
            out2 = model(x2, t=t2, context=[context[0]], seq_len=seq,
                         kv_cache=cache, crossattn_cache=cross, current_start=seq,
                         cache_start=seq, viewmats=cameras, Ks=Ks,
                         prope_kv_cache=prope, actions=actions,
                         embodiment_id=embodiment)
            trace.emit("astronex.denoiser.output.cache1", out2.float())
    finally:
        _restore_scalar_attention(original_attention)
    for hnd in handles:
        hnd.remove()


def _trace_bidirectional_small(trace: Trace, scalar: Scalar):
    """Exercise the official non-causal Wan graph for the synthetic fixture."""
    sys.path.insert(0, str(REFERENCE))
    from models.action import add_action_parameters
    from models.prope import add_prope_parameters
    from models.wan_model import WanModel
    # Module initialisation uses random floating factories by design; it is
    # setup, not graph arithmetic, and must happen before Scalar dispatch.
    from torch.utils._python_dispatch import _disable_current_modes
    with _disable_current_modes():
        model = WanModel(
            model_type="t2v", patch_size=(1, 2, 2), text_len=4,
            in_dim=4, dim=8, ffn_dim=16, freq_dim=4, text_dim=8,
            out_dim=4, num_heads=2, num_layers=1, qk_norm=True,
            cross_attn_norm=True,
        ).float().eval()
        add_prope_parameters(model, zero_init=False)
        add_action_parameters(model, action_dim=64, num_embodiments=2)
    x = (torch.arange(4 * 2 * 2 * 2, dtype=torch.int64).reshape(4, 2, 2, 2)
         .to(torch.float32) - 64.0) / 1000.0
    cameras = torch.eye(4).reshape(1, 1, 4, 4).repeat(1, 2, 1, 1)
    Ks = torch.eye(3).reshape(1, 1, 3, 3).repeat(1, 2, 1, 1)
    context = [torch.zeros(4, 8, dtype=torch.float32)]
    actions = torch.zeros((1, 2, 64), dtype=torch.float32)
    original_attention = _patch_scalar_attention(scalar)
    try:
        with torch.inference_mode(), scalar:
            out = model([x], t=torch.tensor([100]), context=context, seq_len=2,
                        viewmats=cameras, Ks=Ks, actions=actions,
                        embodiment_id=torch.zeros(1, dtype=torch.long))
            trace.emit("astronex.denoiser.output.bidirectional", out.float())
    finally:
        _restore_scalar_attention(original_attention)


def _trace_vae(weights: Path, trace: Trace, scalar: Scalar):
    sys.path.insert(0, str(REFERENCE))
    from models.wan22_components import Wan22VAEWrapper
    vae_dir = weights / "vae"
    if not vae_dir.is_dir():
        raise RuntimeError(f"missing VAE directory: {vae_dir}")
    from torch.utils._python_dispatch import _disable_current_modes
    with _disable_current_modes():
        vae = Wan22VAEWrapper(root=weights, dtype=torch.float32).float().eval()
    # 32x32 pixels, 8 causal latent frames -> 29 decoded pixel frames after
    # the causal temporal compression. Fixed values make this stage repeatable.
    pixel = torch.linspace(-1, 1, 1*3*8*32*32, dtype=torch.float32).reshape(1,3,8,32,32)
    handles = _module_hooks(vae.model, trace, "astronex.vae", limit=1000)
    with torch.inference_mode(), scalar:
        latent = vae.encode_to_latent(pixel)
        trace.emit("astronex.vae.latent", latent.float())
        decoded = vae.decode_to_pixel(latent)
        trace.emit("astronex.vae.pixel", decoded.float())
    for h in handles:
        h.remove()


def _trace_vae_small(trace: Trace, scalar: Scalar):
    """Exercise the official AutoencoderKLWan encode/decode with tiny weights."""
    from diffusers import AutoencoderKLWan
    from torch.utils._python_dispatch import _disable_current_modes
    with _disable_current_modes():
        model = AutoencoderKLWan(
            base_dim=4, decoder_base_dim=4, z_dim=2, dim_mult=[1],
            num_res_blocks=1, attn_scales=[], temperal_downsample=[],
            in_channels=3, out_channels=3, scale_factor_temporal=1,
            scale_factor_spatial=1).float().eval()
    pixel = (torch.arange(3 * 2 * 4 * 4, dtype=torch.int64)
             .reshape(1, 3, 2, 4, 4).to(torch.float32) - 100.0) / 100.0
    with torch.inference_mode(), scalar:
        latent = model.encode(pixel).latent_dist.mode().float()
        trace.emit("astronex.vae.latent", latent)
        decoded = model.decode(latent).sample.float()
        trace.emit("astronex.vae.pixel", decoded)


def _trace_unipc(trace: Trace, scalar: Scalar):
    """Run the official 8-step UniPC state machine on fixed F32 tensors."""
    sys.path.insert(0, str(REFERENCE))
    from models.fm_solvers_unipc import FlowUniPCMultistepScheduler
    scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=1000, solver_order=2,
                                            shift=1, predict_x0=True)
    scheduler.set_timesteps(8, device="cpu", shift=5.0)
    sample = torch.tensor([[[[[0.25, -0.5], [0.75, -1.0]]]]], dtype=torch.float32)
    # Keep model output deterministic while still making every predictor step
    # nonzero. No random factory is allowed inside Scalar.
    model_output = torch.tensor([[[[[0.125, 0.25], [-0.375, 0.5]]]]], dtype=torch.float32)
    for i, timestep in enumerate(scheduler.timesteps):
        with torch.inference_mode(), scalar:
            trace.emit(f"astronex.unipc.sample.{i}", sample)
            sample = scheduler.step(model_output, timestep, sample, return_dict=False)[0].float()
    trace.emit("astronex.unipc.sample.final", sample)


def _fixture(output: Path):
    _guard_reference()
    trace = Trace(output)
    with torch.inference_mode(), Scalar(output / "compile") as scalar:
        x = torch.tensor([[1e8, 1., -1e8]], dtype=torch.float32)
        w = torch.tensor([[1., 1., 1.]], dtype=torch.float32)
        trace.emit("fixture.linear", torch.nn.functional.linear(x,w))
        trace.emit("fixture.norm", torch.nn.functional.layer_norm(torch.tensor([[1.,3.]],dtype=torch.float32), [2], eps=0))
        q = torch.zeros(1,1,1,2); k = torch.zeros(1,1,2,2)
        v = torch.tensor([[[[2.,4.],[10.,20.]]]])
        trace.emit("fixture.attention", torch.nn.functional.scaled_dot_product_attention(q,k,v))
        z = torch.tensor([0, 2, 1, 3], dtype=torch.float32).reshape(1,1,1,2,2)
        trace.emit("fixture.conv3d", torch.nn.functional.conv3d(z, torch.ones(1,1,1,1,1)))
        _trace_bidirectional_small(trace, scalar)
        _trace_vae_small(trace, scalar)
        _trace_text_small(trace, scalar)
        _trace_unipc(trace, scalar)
    trace.emit("fixture.token_ids", [0, 17, 3], kind="token_ids")
    trace.finish(metadata={"kind":"synthetic-official-operator-fixture"})


def _read_trace(root: Path) -> list[dict]:
    path = root / "trace.jsonl"
    if not path.is_file():
        raise RuntimeError(f"missing trace: {path}")
    records = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    if not records:
        raise RuntimeError(f"empty trace: {path}")
    seen = set()
    for rec in records:
        key = (rec.get("name"), rec.get("occurrence"))
        if key in seen:
            raise RuntimeError(f"duplicate checkpoint occurrence: {key}")
        seen.add(key)
        if rec.get("kind") == "f32":
            p = Path(rec.get("binary_path", ""))
            if not p.is_file() or p.stat().st_size != int(rec.get("count", -1))*4:
                raise RuntimeError(f"missing/truncated binary for {key}")
    return records


def compare(oracle: Path, native: Path):
    left, right = _read_trace(oracle), _read_trace(native)
    if len(left) != len(right):
        raise AssertionError(f"checkpoint counts differ: {len(left)} != {len(right)}")
    for a,b in zip(left,right):
        contract = (a.get("name"),a.get("occurrence"),a.get("kind"),a.get("shape"))
        if contract != (b.get("name"),b.get("occurrence"),b.get("kind"),b.get("shape")):
            raise AssertionError(f"checkpoint contract differs: {a} != {b}")
        if a["kind"] == "token_ids":
            if a.get("values") != b.get("values"):
                raise AssertionError(f"{a['name']}: token ids differ")
            continue
        x,y=np.fromfile(a["binary_path"],dtype="<u4"),np.fromfile(b["binary_path"],dtype="<u4")
        if x.size != y.size or x.size != int(np.prod(a["shape"])):
            raise AssertionError(f"{a['name']} occurrence={a['occurrence']}: binary count differs")
        bad=np.flatnonzero(x!=y)
        if bad.size:
            i=int(bad[0]);raise AssertionError(
                f"first divergence {a['name']} occurrence={a['occurrence']} index={i}: "
                f"oracle=0x{int(x[i]):08x}, native=0x{int(y[i]):08x}; {bad.size}/{x.size} differ")
    print(f"PASS: {len(left)} checkpoints, names/order/shapes/counts/token IDs and all F32 bits match")


def trace_real(weights: Path, output: Path, prompt: str, contract: Path | None):
    commit = _guard_reference(); contract_commit = _guard_contract(contract)
    if not weights.is_dir(): raise RuntimeError(f"missing weights directory: {weights}")
    trace = Trace(output)
    with torch.inference_mode(), Scalar(output / "compile") as scalar:
        _trace_tokens_text(weights, trace, prompt, scalar)
        model, loaded, shards = _build_causal(weights)
        text_out = trace.records[-1]
        context = torch.from_numpy(np.fromfile(text_out["binary_path"],dtype="<f4")).reshape(text_out["shape"])
        _trace_denoiser(model, trace, scalar, context)
        _trace_vae(weights, trace, scalar)
        _trace_unipc(trace, scalar)
    trace.finish(metadata={"official_commit":commit,"contract_commit":contract_commit,
                           "weights":str(weights.resolve()),"weights_loaded":loaded,
                           "weight_shards":shards,"prompt":prompt})


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__); sub=ap.add_subparsers(dest="command",required=True)
    sub.add_parser("self-test")
    f=sub.add_parser("fixture");f.add_argument("output",type=Path)
    t=sub.add_parser("trace");t.add_argument("output",type=Path);t.add_argument("--weights",type=Path,default=REPO/"models"/"Astronex-World");t.add_argument("--prompt",default="A camera moves through a quiet room.");t.add_argument("--contract",type=Path,default=None)
    c=sub.add_parser("compare");c.add_argument("oracle",type=Path);c.add_argument("native",type=Path)
    args=ap.parse_args(argv)
    if args.command=="self-test":
        import unittest
        suite=unittest.defaultTestLoader.discover(str(HERE),pattern="test*.py")
        raise SystemExit(0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1)
    if args.command=="fixture": _fixture(args.output);print(f"fixture trace: {args.output}");return
    if args.command=="compare": compare(args.oracle,args.native);return
    trace_real(args.weights,args.output,args.prompt,args.contract)


if __name__ == "__main__": main()
