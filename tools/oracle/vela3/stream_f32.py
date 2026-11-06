"""Temporary F32 execution of a BF16 Qwen3.5 text model, one decoder layer at a time.

Use ``with stream_f32(backbone.language_model):`` around embedding, PromptEncoder
and trace calls. Attach LoRA before entering. For a causal head outside the model,
call ``F.linear(hidden.float(), head.float())`` separately. This is an exclusive,
no-gradient, short text diagnostic; it does not stream vision or cached decoding.
Run this file with --self-test for an exact small-model comparison.
"""
from __future__ import annotations

import contextlib
import functools
import json
from pathlib import Path

import torch


def require_bf16_checkpoint(root):
    """Reject source tensors that BF16 loading would silently round."""
    from safetensors import safe_open

    root = Path(root)
    index = root / "model.safetensors.index.json"
    shards = (sorted(set(json.loads(index.read_text())["weight_map"].values()))
              if index.is_file() else ["model.safetensors"])
    count = 0
    for shard in shards:
        with safe_open(root / shard, framework="pt", device="cpu") as tensors:
            for name in tensors.keys():
                dtype = tensors.get_slice(name).get_dtype()
                if dtype != "BF16":
                    raise ValueError(f"stream-f32-weights requires BF16 source tensors: {shard}:{name} is {dtype}")
                count += 1
    return count


@contextlib.contextmanager
def _float_tensors(module):
    saved, converted = [], {}
    try:
        for child in module.modules():
            for table in (child._parameters, child._buffers):
                for name, original in tuple(table.items()):
                    if original is None or not original.is_floating_point() or original.dtype == torch.float32:
                        continue
                    saved.append((table, name, original))
                    key = id(original)
                    if key not in converted:
                        value = original.float()
                        converted[key] = (
                            torch.nn.Parameter(value, requires_grad=original.requires_grad)
                            if isinstance(original, torch.nn.Parameter) else value
                        )
                    # Replacing an existing value preserves registration order and aliases.
                    table[name] = converted[key]
        yield
    finally:
        for table, name, original in reversed(saved):
            table[name] = original


@contextlib.contextmanager
def stream_f32(model):
    """Keep stored tensors intact; existing forward hooks observe F32 outputs."""
    saved = []

    def wrap(module, embedding=False):
        original = module.forward
        saved.append((module, "forward" in vars(module), original))

        @functools.wraps(original)
        def forward(*args, **kwargs):
            if embedding:
                return original(*args, **kwargs).float()
            with _float_tensors(module):
                return original(*args, **kwargs)

        module.forward = forward

    with torch.no_grad():
        try:
            wrap(model.embed_tokens, embedding=True)
            for layer in model.layers:
                wrap(layer)
            wrap(model.norm)
            wrap(model.rotary_emb)
            yield model
        finally:
            for module, had_override, original in reversed(saved):
                if had_override:
                    module.forward = original
                else:
                    del module.forward


def _load_module(path, name):
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def self_test(runtime_path, adapters_path):
    import copy
    import tempfile

    import transformers
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

    torch.manual_seed(20261030)
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    torch.use_deterministic_algorithms(True)
    runtime = _load_module(runtime_path, "stream_f32_runtime")
    adapters = _load_module(adapters_path, "stream_f32_adapters")
    config = Qwen3_5Config(text_config=dict(
        hidden_size=16, intermediate_size=24, num_hidden_layers=4,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8, vocab_size=32,
        linear_num_key_heads=1, linear_num_value_heads=2, linear_key_head_dim=8,
        linear_value_head_dim=8, linear_conv_kernel_dim=4, max_position_embeddings=128,
        layer_types=["linear_attention", "full_attention"] * 2,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.,
                         "partial_rotary_factor": 1., "mrope_section": [1, 1, 2]},
    ), vision_config=dict(depth=1, hidden_size=16, intermediate_size=24, num_heads=2,
                          patch_size=2, temporal_patch_size=2, spatial_merge_size=1,
                          out_hidden_size=16, num_position_embeddings=16))
    config._attn_implementation = "sdpa"
    with tempfile.TemporaryDirectory(prefix="stream-f32-", dir=Path(__file__).parent) as tmp:
        initial = Qwen3_5Model(config).bfloat16().eval()
        initial.save_pretrained(tmp)
        assert require_bf16_checkpoint(tmp) == len(initial.state_dict())
        from safetensors.torch import save_file
        mixed = Path(tmp) / "mixed"
        mixed.mkdir()
        save_file({"a": torch.tensor([1.001], dtype=torch.float32)}, mixed / "model.safetensors")
        try:
            require_bf16_checkpoint(mixed)
        except ValueError as error:
            assert "is F32" in str(error)
        else:
            raise AssertionError("mixed source precision was silently rounded")
        del initial
        stored = Qwen3_5Model.from_pretrained(
            tmp, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True,
        ).eval()
    lm = stored.language_model
    for path in ["layers.0.linear_attn.in_proj_qkv", "layers.1.self_attn.q_proj", "layers.3.mlp.down_proj"]:
        parent, attr = path.rsplit(".", 1)
        parent = lm.get_submodule(parent)
        base = getattr(parent, attr)
        wrapper = adapters.MultiLoRA(base)
        wrapper.add("test", torch.randn(2, base.in_features) * .03,
                    torch.randn(base.out_features, 2) * .03, 1.5)
        setattr(parent, attr, wrapper)
    reference = copy.deepcopy(stored).float()
    ids = torch.tensor([[1, 5, 9, 2, 7, 11, 3, 6, 4]])
    mask = torch.ones_like(ids)
    head = torch.randn(7, 16)
    tables = [(table, tuple(table.items())) for module in stored.modules()
              for table in (module._parameters, module._buffers)]
    originals = [(value, value.dtype, value.data_ptr()) for _, items in tables
                 for _, value in items if value is not None]
    forwards = [(module, vars(module).get("forward")) for module in stored.modules()]
    dtypes = {name: parameter.dtype for name, parameter in lm.named_parameters()}
    peak_converted = 0

    def restored():
        for table, items in tables:
            assert tuple(table) == tuple(name for name, _ in items), "registration order changed"
            assert all(table[name] is value for name, value in items), "tensor reference changed"
        assert all(value.dtype == dtype and value.data_ptr() == pointer
                   for value, dtype, pointer in originals), "original storage/dtype changed"
        assert all(vars(module).get("forward") is original for module, original in forwards)

    def active_layer(prefix):
        nonlocal peak_converted
        converted = 0
        assert lm.dtype == torch.bfloat16
        for name, parameter in lm.named_parameters():
            expected = torch.float32 if prefix is not None and name.startswith(prefix) else dtypes[name]
            assert parameter.dtype == expected, (prefix, name, parameter.dtype, expected)
            if expected == torch.float32 and dtypes[name] != torch.float32:
                converted += parameter.numel() * 4
        peak_converted = max(peak_converted, converted)

    def capture(backbone, mode, adapter, streaming):
        text = backbone.language_model
        outputs, hooks = {}, []

        def save(name):
            def hook(module, args, output):
                outputs[name] = output.detach().clone()
            return hook

        # Install observers first: forward wrapping must not depend on hook order.
        hooks.append(text.embed_tokens.register_forward_hook(save("embedding")))
        for index, layer in enumerate(text.layers):
            hooks.append(layer.register_forward_hook(save(f"layer.{index}")))
            if streaming:
                hooks.append(layer.input_layernorm.register_forward_pre_hook(
                    lambda module, args, i=index: active_layer(f"layers.{i}.")))
        if streaming:
            hooks.append(text.norm.register_forward_pre_hook(lambda module, args: active_layer(None)))
        try:
            with (stream_f32(text) if streaming else contextlib.nullcontext()), adapters.active(adapter):
                embeds = text.embed_tokens(ids)
                assert embeds.dtype == torch.float32
                if mode == "causal":
                    hidden = text(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
                else:
                    kwargs = {"blocks": runtime.StateBlocks.of(ids.shape[-1], [ids.shape[-1]], [4])} if mode == "prefix" else {}
                    hidden = runtime.PromptEncoder(text)(embeds, mask, **kwargs)
                outputs["hidden"] = hidden.clone()
                outputs["logits"] = torch.nn.functional.linear(hidden[:, -1].float(), head)
                outputs["causal_head"] = torch.nn.functional.linear(
                    hidden[:, -1].float(), text.embed_tokens.weight.float())
        finally:
            for hook in hooks:
                hook.remove()
        return outputs

    cases, checkpoints = [], 0
    for mode in ["noncausal", "prefix", "causal"]:
        for adapter in [None, "test"]:
            expected = capture(reference, mode, adapter, False)
            actual = capture(stored, mode, adapter, True)
            restored()
            assert expected.keys() == actual.keys()
            for name, value in actual.items():
                assert value.dtype == torch.float32
                assert torch.equal(value, expected[name]), (
                    mode, adapter, name, (value - expected[name]).abs().max().item())
                checkpoints += 1
            cases.append({"mode": mode, "adapter": adapter, "exact": True})

    class InjectedFailure(RuntimeError):
        pass

    def fail(module, args):
        active_layer("layers.3.")
        raise InjectedFailure("during an F32 decoder layer")

    hook = lm.layers[3].self_attn.q_proj.register_forward_pre_hook(fail)
    try:
        try:
            capture(stored, "noncausal", "test", True)
        except InjectedFailure:
            pass
        else:
            raise AssertionError("injected exception was swallowed")
    finally:
        hook.remove()
    restored()
    again = capture(stored, "noncausal", "test", True)
    expected = capture(reference, "noncausal", "test", False)
    assert all(torch.equal(value, expected[name]) for name, value in again.items())
    restored()

    # Conversion can fail before the context manager reaches yield (e.g. OOM).
    from unittest.mock import patch
    target = [p for p in lm.layers[0].parameters() if p.dtype == torch.bfloat16][1]
    original_float = torch.Tensor.float

    def fail_float(value, *args, **kwargs):
        if value is target:
            raise InjectedFailure("during parameter conversion")
        return original_float(value, *args, **kwargs)

    try:
        with patch.object(torch.Tensor, "float", fail_float):
            capture(stored, "noncausal", "test", True)
    except InjectedFailure:
        pass
    else:
        raise AssertionError("conversion exception was swallowed")
    restored()
    return dict(passed=True, torch=torch.__version__, transformers=transformers.__version__,
                cases=cases, exact_checkpoints=checkpoints, max_absolute_error=0.,
                original_tensor_references_and_order=True, exception_cleanup=True,
                partial_conversion_cleanup=True,
                source_precision_guard=True,
                reentry_after_exception=True, model_dtype="torch.bfloat16",
                maximum_temporary_f32_parameter_bytes=peak_converted,
                scope="text; no cache; no vision computation")


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", required=True)
    parser.add_argument("--runtime", type=Path, default=Path("models/d3-turbo/pinned/d3_runtime.py"))
    parser.add_argument("--adapters", type=Path, default=Path("models/vela3-turbo/vela3_adapters.py"))
    args = parser.parse_args()
    os.environ["D3_FAST"] = "0"
    print(json.dumps(self_test(args.runtime.resolve(), args.adapters.resolve()), indent=2))
