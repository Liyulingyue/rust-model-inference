"""Check projection mapping/chunking without downloading a model."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("gguf")

from tools.converter.utils.gguf import GGML_F32, GgufWriter, gguf_dims
from tools.converter.utils.quantize_gguf import BLOCK_FORMATS, quantize_tensor
from tools.oracle.sensenova_loop.quantized_reference import host_device_map, install_activation_quantization, materialize_cpu, replace_projections


def test_host_device_map_never_onloads_host_modules(monkeypatch):
    accelerate = pytest.importorskip("accelerate")
    import accelerate.hooks as hooks
    from torch import nn

    model = nn.Module()
    model.model = nn.Module()
    model.model.visual = nn.Linear(4, 4)
    model.model.language_model = nn.Module()
    text = model.model.language_model
    text.embed_tokens = nn.Embedding(8, 4)
    text.layers = nn.ModuleList([nn.ModuleDict({
        name: nn.Linear(4, 4) for name in ("attn", "mhc_attn", "mhc_ffn")
    })])
    text.norm = nn.LayerNorm(4)
    text.rotary_emb = nn.Module()
    text.rotary_emb.register_buffer("inv_freq", torch.ones(2), persistent=False)
    model.lm_head = nn.Linear(4, 8, bias=False)
    modules = dict(model.named_modules())
    prefixes = {id(module): name for name, module in modules.items()}
    mapping = host_device_map("cuda")
    params = dict(model.named_parameters())
    locations = {
        name: [device for prefix, device in mapping.items() if name.startswith(prefix + ".")]
        for name in params
    }
    assert all(len(matches) == 1 for matches in locations.values())
    host = {name for name, devices in locations.items() if devices == ["cpu"]}
    gpu = set(params) - host
    expected = {name: params[name].detach().clone() for name in host}
    requests = []
    original = hooks.set_module_tensor_to_device

    def record(module, name, device, *args, **kwargs):
        prefix = prefixes[id(module)]
        requests.append(((prefix + "." if prefix else "") + name, str(device)))
        if str(device).startswith("cuda"):
            return  # Record migration intent without allocating CUDA memory.
        return original(module, name, device, *args, **kwargs)

    monkeypatch.setattr(hooks, "set_module_tensor_to_device", record)
    accelerate.dispatch_model(model, mapping, main_device="cuda")
    for name, device in mapping.items():
        if device == "cpu":
            materialize_cpu(modules[name])
    moved = {name for name, device in requests if device.startswith("cuda")}
    assert not host & moved
    assert gpu <= moved
    assert "model.language_model.rotary_emb.inv_freq" in moved
    for name, param in model.named_parameters():
        assert param.device.type == "cpu"
        if name in host:
            assert torch.equal(param, expected[name])
    with torch.no_grad():
        model.lm_head.weight.fill_(0.5)
        assert torch.equal(model.lm_head(torch.ones(1, 4)), torch.full((1, 8), 2.0))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_host_inputs_materialize_offloaded_parameters_on_cpu(device):
    accelerate = pytest.importorskip("accelerate")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm device required")
    module = torch.nn.Linear(4, 3).to(device)
    expected = module.weight.detach().cpu().clone()
    accelerate.cpu_offload(module, execution_device=torch.device(device))
    assert module.weight.is_meta
    materialize_cpu(module)
    assert module.weight.device.type == "cpu"
    assert torch.equal(module.weight, expected)


class Model:
    def __init__(self):
        self.projection = torch.nn.Parameter(torch.zeros(600, 256))
        self.router = torch.nn.Parameter(torch.full((8, 512), 7.0))

    def named_parameters(self):
        return iter([
            ("model.language_model.embed_tokens.weight", self.projection),
            ("model.language_model.layers.4.mhc_attn.weight", self.router),
        ])


def write_reference(path, fmt, *, extra=False):
    values = np.broadcast_to(
        ((np.arange(600, dtype=np.float32) % 31 - 15) / 8)[:, None], (600, 256)
    ).copy()
    kind, payload = (GGML_F32, values.tobytes()) if fmt == "f32" else quantize_tensor(
        "token_embd.weight", values.tobytes(), GGML_F32, (256, 600), fmt
    )
    writer = GgufWriter(path)
    writer.add_meta("general.architecture", "sensenova_loop")
    writer.add_tensor("token_embd.weight", kind, gguf_dims(values.shape), payload)
    if extra:
        writer.add_tensor("output.weight", GGML_F32, (256, 1), values[:1].tobytes())
    writer.write()
    return values


@pytest.mark.parametrize("fmt", ["f32", *sorted(BLOCK_FORMATS)])
def test_reference_loads_all_rows_and_preserves_mhc(tmp_path, fmt):
    path = tmp_path / f"{fmt}.gguf"
    expected = write_reference(path, fmt)
    model = Model()
    provenance = replace_projections(model, path)
    np.testing.assert_allclose(model.projection.detach().numpy(), expected, atol=0.002, rtol=0.002)
    assert torch.all(model.router == 7)
    assert provenance["projections"] == 1 and len(provenance["sha256"]) == 64


def test_reference_rejects_partial_mapping_and_wrong_shape(tmp_path):
    path = tmp_path / "extra.gguf"
    write_reference(path, "f32", extra=True)
    with pytest.raises(ValueError, match="unmapped GGUF projections"):
        replace_projections(Model(), path)
    path = tmp_path / "shape.gguf"
    write_reference(path, "f32")
    model = Model()
    model.projection = torch.nn.Parameter(torch.zeros(599, 256))
    with pytest.raises(ValueError, match="invalid reference projection"):
        replace_projections(model, path)


def test_reference_rejects_meta_parameter(tmp_path):
    path = tmp_path / "meta.gguf"
    write_reference(path, "f32")
    model = Model()
    model.projection = torch.nn.Parameter(torch.empty(600, 256, device="meta"))
    with pytest.raises(ValueError, match="unmaterialized reference projection token_embd.weight"):
        replace_projections(model, path)


@pytest.mark.parametrize("fmt", ["q8_0", "q4_0", "q4_1", "q4_k", "q5_k", "q6_k"])
def test_activation_reference_wraps_output_head_and_keeps_other_modules(tmp_path, fmt):
    import gguf
    values = np.stack([np.full(256, 1.25, np.float32), np.full(256, -2.5, np.float32)])
    kind, payload = quantize_tensor("output.weight", values.tobytes(), GGML_F32, (256, 2), fmt)
    path = tmp_path / (fmt + ".gguf")
    writer = GgufWriter(path)
    writer.add_tensor("output.weight", kind, (256, 2), payload)
    writer.write()
    model = torch.nn.Module()
    model.lm_head = torch.nn.Linear(256, 2, bias=False)
    model.unrelated = torch.nn.Linear(256, 2, bias=False)
    decoded = gguf.quants.dequantize(np.frombuffer(payload, dtype=np.uint8), kind).reshape(2, 256)
    with torch.no_grad():
        model.lm_head.weight.copy_(torch.from_numpy(decoded.copy()))
    original = model.unrelated.forward
    description = install_activation_quantization(model, path)
    assert [entry["tensor"] for entry in description] == ["output.weight"]
    assert model.unrelated.forward == original
    input = torch.ones(2, 256)
    input[:, ::32] = 127.03125
    input[:, 1::32] = 0.5
    actual = model.lm_head(input=input)
    assert actual.shape == (2, 2) and torch.isfinite(actual).all()
    assert not torch.equal(actual, torch.nn.functional.linear(input, model.lm_head.weight))
    if fmt == "q4_1":
        # Constant Q4_1 rows contain only their minimum; each block contributes
        # its separately F16-rounded Q8_1 sum, not the sum of rounded Q8 scales.
        scale = np.float32(127.03125) / np.float32(127)
        block_sum = np.float32(157) * scale
        expected = np.float32(np.float16(block_sum)) * 8 * np.array([1.25, -2.5])
        np.testing.assert_allclose(actual.detach().numpy(), np.broadcast_to(expected, (2, 2)), rtol=1e-6)
