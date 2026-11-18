"""Export stage fixtures from pinned, read-only official NumPy Underdog-Woof.

Each checkpoint is a little-endian F32 file described by manifest.json.
Text is already normalized/respelt. --gguf independently dequantizes that
file with the upstream gguf package before restoring the official model.
"""
import argparse
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from tools.converter.underdog_woof.convert_underdog_woof import CONTRACT, sha256, verify_sources


def official_runtime(directory):
    verify_sources(directory)
    path = Path(directory) / "underdog_woof/runtime.py"
    spec = importlib.util.spec_from_file_location("_pinned_underdog_woof_runtime", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def restore_gguf(state, path):
    """Reverse export layouts using upstream gguf's independent dequantizer."""
    from gguf import GGUFReader
    from gguf.quants import dequantize

    reader = GGUFReader(str(path))
    for key, expected in {
        "general.architecture": CONTRACT["architecture"],
        "underdog_woof.source_revision": CONTRACT["source"]["revision"],
    }.items():
        if reader.fields[key].contents() != expected:
            raise ValueError(f"same-weight Oracle expected {key}={expected!r}")
    if {tensor.name for tensor in reader.tensors} != set(CONTRACT["tensors"]):
        raise ValueError("same-weight GGUF tensor inventory mismatch")
    tensors = {}
    for tensor in reader.tensors:
        shape = CONTRACT["tensors"][tensor.name]
        if list(reversed(tensor.shape.tolist())) != shape:
            raise ValueError(f"same-weight GGUF shape mismatch: {tensor.name}")
        array = np.ascontiguousarray(dequantize(tensor.data, tensor.tensor_type).reshape(shape), dtype=np.float32)
        if not np.isfinite(array).all():
            raise ValueError(f"non-finite same-weight GGUF tensor: {tensor.name}")
        tensors[tensor.name] = array

    def linear(name):
        result = {"w": tensors[name + ".weight"]}
        if name + ".bias" in tensors:
            result["b"] = tensors[name + ".bias"]
        return result

    speech, decoder = state["speech"], state["decoder"]
    for name in ("ce", "ada", "lat_in", "h_inp", "h_out"):
        speech[name] = linear("speech." + name)
    for i, layer in enumerate(speech["layers"]):
        for j, name in enumerate(("qkv", "o", "fc1", "fc2")):
            layer[j] = linear(f"speech.layers.{i}.{name}")
    for i, layer in enumerate(speech["hblocks"]):
        for j, name in enumerate(("fc1", "fc2")):
            layer[j] = linear(f"speech.hblocks.{i}.{name}")
    speech["text_emb"], speech["bos"] = tensors["speech.text_emb.weight"], tensors["speech.bos"]
    state["voices"] = {name: tensors["voices." + name] for name in CONTRACT["voices"]}
    state["mean"], state["std"] = tensors["latent.mean"], tensors["latent.std"]
    decoder["inp"] = [tensors["decoder.inp.weight"], tensors["decoder.inp.bias"]]
    decoder["inv_freq"] = tensors["decoder.inv_freq"]
    for i, layer in enumerate(decoder["tf"]):
        for name in ("qkv", "o", "fc1", "fc2"):
            layer[name] = tensors[f"decoder.tf.{i}.{name}.weight"]
        for name in ("ln1", "ln2"):
            layer[name] = [tensors[f"decoder.tf.{i}.{name}.{part}"] for part in ("weight", "bias")]

    def conv(name, config):
        return {"w": np.stack([tensors[f"{name}.tap.{j}.weight"]
                               for j in range(config["kernel_size"])], axis=2),
                "b": tensors[name + ".bias"]}

    for i, config in enumerate(CONTRACT["seanet"]):
        kind, name = config["kind"], f"decoder.seanet.{i}"
        if kind == "conv":
            decoder["seanet"][i][1] = conv(name, config)
        elif kind == "convt":
            matrix = tensors[name + ".weight"]
            weight = matrix.T.reshape(config["in_channels"], config["kernel_size"], config["out_channels"])
            decoder["seanet"][i][1] = {"w": np.ascontiguousarray(weight.transpose(0, 2, 1)),
                                       "b": tensors[name + ".bias"], "stride": config["stride"]}
        elif kind == "res":
            decoder["seanet"][i][1] = [conv(f"{name}.{j}", layer) for j, layer in enumerate(config["layers"])]
    return {"path": str(Path(path).resolve()), "sha256": sha256(path),
            "format": reader.fields["underdog_woof.format"].contents(),
            "independent_dequantization": "gguf.quants.dequantize"}


def public_model(directory, *, voice="juno", temperature=0.5, gguf=None):
    """Load the unmodified public wrapper, including normalization and retry selection."""
    directory = Path(directory).resolve()
    verify_sources(directory)
    name = "_pinned_underdog_woof_public"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, directory / "underdog_woof/__init__.py",
            submodule_search_locations=[str(directory / "underdog_woof")])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    module = sys.modules[name]
    runtime = sys.modules[name + ".runtime"]
    tts = module.Woof(weights=str(directory / "woof_int8.npz"), voice=voice, temp=temperature)
    same_weight = None
    if gguf is not None:
        with np.load(directory / "woof_int8.npz", allow_pickle=False) as npz:
            state = runtime.unflatten(npz)
        same_weight = restore_gguf(state, gguf)
        tts.model.speech = runtime.Speech(state["speech"])
        tts.model.dec = runtime.Decoder(state["decoder"])
        tts.model.mean, tts.model.std = state["mean"], state["std"]
        tts.model.voice = state["voices"][voice]
    return tts, same_weight


def export_public(directory, output, *, text, voice="juno", seed=0, temperature=0.5, gguf=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if not 0 <= seed < 2 ** 32 or not 0 <= temperature <= 2:
        raise ValueError("invalid public request seed or temperature")
    tts, same_weight = public_model(directory, voice=voice, temperature=temperature, gguf=gguf)
    takes = []
    original_latents = tts.model.latents

    def latents(piece, **options):
        result = original_latents(piece, **options)
        takes.append({"text": piece, "frames": len(result)})
        return result

    tts.model.latents = latents
    started = time.perf_counter()
    waveform = tts(text, seed=seed)
    elapsed = time.perf_counter() - started
    if not len(waveform) or not np.isfinite(waveform).all():
        raise ValueError("invalid public Oracle output")
    output.mkdir(parents=True)
    waveform.astype("<f4").tofile(output / "waveform.f32")
    tts.save(str(output / "reference.wav"), waveform)
    result = {"schema": "underdog-woof-public-v1", "source": CONTRACT["source"],
              "text": text, "voice": voice, "seed": seed, "temperature": temperature,
              "same_weight_gguf": same_weight, "takes": takes,
              "samples": len(waveform), "seconds": elapsed,
              "waveform_sha256": sha256(output / "waveform.f32")}
    (output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return output / "manifest.json"


def export(directory, output, *, text="Hi.", voice="juno", seed=0, frames=16,
           temperature=0.5, gguf=None, force_latents=None, eos_threshold=None):
    directory, output = Path(directory), Path(output)
    if output.exists():
        raise FileExistsError(output)
    if voice not in CONTRACT["voices"] or frames < 1 or frames > CONTRACT["config"]["max_frames"]:
        raise ValueError("invalid voice or fixture frame count")
    if not np.isfinite(temperature) or temperature < 0 or not 0 <= seed < 2 ** 32:
        raise ValueError("invalid fixture temperature or seed")
    if eos_threshold is not None and (not 0 < eos_threshold < 1 or force_latents is not None):
        raise ValueError("natural EOS requires a threshold in (0, 1) and no forced latents")
    force = None
    if force_latents is not None:
        force = np.fromfile(force_latents, dtype="<f4")
        if force.size != frames * CONTRACT["config"]["speech"]["latent"] or not np.isfinite(force).all():
            raise ValueError("forced latents must contain exactly frames * latent finite F32 values")
        force = force.reshape(frames, CONTRACT["config"]["speech"]["latent"])
    runtime = official_runtime(directory)
    with np.load(directory / "woof_int8.npz", allow_pickle=False) as npz:
        state = runtime.unflatten(npz)
    same_weight = None if gguf is None else restore_gguf(state, gguf)
    ids = runtime.encode_text(text)
    speech = runtime.Speech(state["speech"])
    conditioning = state["voices"][voice].astype(np.float32)
    if len(ids) + len(conditioning) + frames + 1 > speech.max_len:
        raise ValueError("fixture exceeds official speech context")
    checkpoints = {}

    def record(name, value):
        array = np.ascontiguousarray(value, dtype="<f4")
        if not np.isfinite(array).all():
            raise ValueError(f"non-finite Oracle checkpoint: {name}")
        checkpoints[name] = array.copy()

    record("speech.prefix_input", np.concatenate([conditioning, speech.text_emb[ids], speech.bos[None]], axis=0))
    original_prefix, original_block = speech._prefix_block, speech._block1
    original_cond, original_head = speech._cond, speech._head
    counters = {"condition": 0, "head": 0}

    def prefix(layer, value, count):
        result = original_prefix(layer, value, count)
        record(f"speech.prefix.{layer}", result)
        return result

    def block(layer, value, position):
        frame = position - (len(conditioning) + len(ids) + 1)
        if layer == 0:
            record(f"speech.block_input.{frame}", value)
        result = original_block(layer, value, position)
        record(f"speech.block.{frame}.{layer}", result)
        return result

    def cond(value):
        index = counters["condition"]
        result, eos = original_cond(value)
        record(f"speech.condition.{index}", result)
        record(f"speech.eos.{index}", np.array([eos], dtype=np.float32))
        counters["condition"] += 1
        return result, eos

    def head(value, noise):
        frame = counters["head"]
        record(f"speech.noise.{frame}", noise)
        result = original_head(value, noise)
        record(f"speech.head.{frame}", result)
        counters["head"] += 1
        return result

    speech._prefix_block, speech._block1 = prefix, block
    speech._cond, speech._head = cond, head
    runtime.RNG.seed(seed)
    started = time.perf_counter()
    # The official minimum-frame option keeps EOS disabled for fixed-length numerical probes.
    latents = speech.generate(conditioning, ids, max_frames=frames, temp=temperature,
                              eos_thresh=0.9 if eos_threshold is None else eos_threshold,
                              min_frames_per_char=(frames + 1) / len(ids) if eos_threshold is None else 0.0,
                              force=force)
    speech_seconds = time.perf_counter() - started
    if eos_threshold is None and len(latents) != frames:
        raise ValueError("official fixed-frame Oracle returned fewer frames")
    record("latents", latents)
    decoder_latents = latents if force is None else force
    record("decoder_latents", decoder_latents)
    denormalized = decoder_latents * state["std"].astype(np.float32) + state["mean"].astype(np.float32)
    record("denormalized_latents", denormalized)
    decoder = runtime.Decoder(state["decoder"])
    started = time.perf_counter()
    waveform = decoder(denormalized)
    decoder_seconds = time.perf_counter() - started
    record("waveform", waveform)

    decoder = runtime.Decoder(state["decoder"])
    decoder_input = denormalized @ decoder.inp_w + decoder.inp_b
    record("decoder.input", decoder_input)
    value = decoder.transformer(decoder_input)
    record("decoder.transformer", value)
    for kind, operation in decoder.ops:
        if kind == "res":
            operation[0].reset()
            operation[1].reset()
        elif operation is not None:
            operation.reset()
    for i, operation in enumerate(decoder.ops):
        value = runtime.Decoder._run(value, [operation])
        record(f"decoder.seanet.{i}", value)
    record("waveform_full_sentence", value[:, 0])
    stage_max_abs = float(np.max(np.abs(value[:, 0] - waveform)))
    output.mkdir(parents=True, exist_ok=False)
    try:
        manifest = {
            "schema": "underdog-woof-oracle-v1", "source": CONTRACT["source"],
            "text": text, "text_ids": ids, "voice": voice, "seed": seed,
            "frames": len(latents), "max_frames": frames, "temperature": temperature,
            "eos_disabled": eos_threshold is None, "eos_threshold": eos_threshold,
            "forced_latents": None if force_latents is None else {
                "path": str(Path(force_latents).resolve()), "sha256": sha256(force_latents)},
            "same_weight_gguf": same_weight,
            "performance_seconds": {"speech": speech_seconds, "decoder": decoder_seconds},
            "decoder_chunk_vs_full_sentence_max_abs": stage_max_abs, "checkpoints": {},
        }
        for name, value in checkpoints.items():
            filename = name + ".f32"
            (output / filename).write_bytes(value.tobytes())
            manifest["checkpoints"][name] = {"shape": list(value.shape), "dtype": "f32-le",
                                               "file": filename, "sha256": sha256(output / filename)}
        runtime.write_wav(str(output / "reference.wav"), waveform)
        manifest["audio"] = {"file": "reference.wav", "sha256": sha256(output / "reference.wav"),
                             "sample_rate": runtime.SR, "samples": len(waveform)}
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    except BaseException:
        shutil.rmtree(output)
        raise
    return output / "manifest.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--text", default="Hi.", help="already normalized/respelt text")
    parser.add_argument("--voice", choices=CONTRACT["voices"], default="juno")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--gguf", type=Path)
    parser.add_argument("--public", action="store_true",
                        help="exercise the official public normalizer/splitting/retry pipeline")
    parser.add_argument("--force-latents", type=Path,
                        help="frames * 32 little-endian F32 values for teacher forcing and fixed-input decoding")
    parser.add_argument("--eos-threshold", type=float,
                        help="enable natural EOS instead of generating exactly --frames frames")
    args = parser.parse_args()
    if args.public:
        if args.force_latents is not None or args.eos_threshold is not None:
            parser.error("--public cannot use forced latents or an EOS override")
        print(export_public(args.directory, args.output, text=args.text, voice=args.voice,
                            seed=args.seed, temperature=args.temperature, gguf=args.gguf))
        return
    print(export(args.directory, args.output, text=args.text, voice=args.voice, seed=args.seed,
                 frames=args.frames, temperature=args.temperature, gguf=args.gguf,
                 force_latents=args.force_latents, eos_threshold=args.eos_threshold))


if __name__ == "__main__":
    main()
