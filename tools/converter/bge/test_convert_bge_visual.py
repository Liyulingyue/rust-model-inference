import json
import os
import base64
import socket
import time
import urllib.request
import urllib.error
import struct
import subprocess
import tempfile
import zlib
import unittest
from pathlib import Path

import numpy as np

from tools.converter.bge.test_convert_bge import (
    _write_safetensors,
    _rewrite_safetensors,
    _decoder_files,
    _bert_files,
    _write_torch_bin,
    _Tensor,
    _Storage,
)
from tools.converter.bge.convert_bge import TensorStore, _decoder_mapping, convert
from tools.converter.bge.convert_bge_visual import (
    convert_visual,
    clip_shapes,
    vision_shapes,
    screenshot_shapes,
)
from tools.converter.utils.gguf import read_gguf_directory


def write_image(path: Path):
    def chunk(tag, body):
        return (
            struct.pack(">I", len(body))
            + tag
            + body
            + struct.pack(">I", zlib.crc32(tag + body))
        )

    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">2I5B", 4, 4, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress((b"\0" + bytes([180, 40, 80]) * 4) * 4))
        + chunk(b"IEND", b"")
    )


def clip_files(root: Path):
    root.mkdir()
    config = dict(
        model_type="clip",
        projection_dim=32,
        text_config=dict(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            vocab_size=4,
            max_position_embeddings=8,
            hidden_act="quick_gelu",
        ),
        vision_config=dict(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            image_size=4,
            patch_size=2,
            hidden_act="quick_gelu",
        ),
    )
    (root / "config.json").write_text(json.dumps(config))
    tokenizer = dict(
        version="1.0",
        truncation=None,
        padding=None,
        added_tokens=[],
        normalizer=None,
        pre_tokenizer=dict(type="Whitespace"),
        post_processor=dict(
            type="TemplateProcessing",
            single=[
                {"SpecialToken": {"id": "[CLS]", "type_id": 0}},
                {"Sequence": {"id": "A", "type_id": 0}},
                {"SpecialToken": {"id": "[SEP]", "type_id": 0}},
            ],
            pair=[],
            special_tokens={
                "[CLS]": {"id": "[CLS]", "ids": [1], "tokens": ["[CLS]"]},
                "[SEP]": {"id": "[SEP]", "ids": [2], "tokens": ["[SEP]"]},
            },
        ),
        decoder=None,
        model=dict(
            type="WordLevel",
            vocab={"[UNK]": 0, "[CLS]": 1, "[SEP]": 2, "hello": 3},
            unk_token="[UNK]",
        ),
    )
    (root / "tokenizer.json").write_text(json.dumps(tokenizer))
    config["text_config"]["eos_token_id"] = 2
    (root / "config.json").write_text(json.dumps(config))
    (root / "preprocessor_config.json").write_text(
        json.dumps(
            dict(
                size=4,
                crop_size=4,
                image_mean=[0.1, 0.2, 0.3],
                image_std=[0.5, 0.5, 0.5],
            )
        )
    )
    rng = np.random.default_rng(42)
    tensors = {}
    for name, shape in clip_shapes(config).items():
        values = rng.normal(0, 0.1, shape).astype("<f4")
        if "layer_norm" in name or "layernorm" in name:
            values = (
                np.ones(shape, dtype="<f4")
                if name.endswith("weight")
                else np.zeros(shape, dtype="<f4")
            )
        tensors[name] = (shape, values.tobytes())
    tensors["logit_scale"] = ((), struct.pack("<f", 2.6592))
    _write_safetensors(root / "model.safetensors", tensors)
    return config


def decoder_visual_files(root: Path, screenshot: bool):
    kind = "qwen2" if screenshot else "mistral"
    _decoder_files(root, model_type=kind, hidden=256)
    text = json.loads((root / "config.json").read_text(encoding="utf-8"))
    text["max_position_embeddings"] = 4096
    chars = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    byte_chars = chars.copy()
    next_id = 0
    for i in range(256):
        if i not in chars:
            chars.append(i)
            byte_chars.append(256 + next_id)
            next_id += 1
    vocab = {chr(c): i for i, c in enumerate(byte_chars)}
    specials = [
        "<|endoftext|>",
        "<|im_start|>",
        "<|im_end|>",
        "<|vision_start|>",
        "<|image_pad|>",
        "<|vision_end|>",
        "<image>",
    ]
    vocab.update({token: len(vocab) + i for i, token in enumerate(specials)})
    (root / "tokenizer.json").write_text(
        json.dumps(
            dict(
                model=dict(type="BPE", vocab=vocab, merges=[]),
                added_tokens=[
                    dict(id=vocab[t], content=t, special=True) for t in specials
                ],
            )
        )
    )
    (root / "tokenizer_config.json").write_text(
        json.dumps(
            dict(
                eos_token="<|im_end|>",
                pad_token="<|endoftext|>",
                add_bos_token=False,
                add_eos_token=False,
            )
        )
    )
    text["vocab_size"] = len(vocab) + 3
    store = TensorStore(root)
    tensors = {}
    for name in _decoder_mapping(kind, 1).values():
        original = name.removeprefix("model.")
        tensor = store[original]
        tensors[name] = (tensor.shape, store.read(tensor))
    rng = np.random.default_rng(101)
    tensors["model.embed_tokens.weight"] = (
        (text["vocab_size"], 256),
        rng.normal(0, 0.1, (text["vocab_size"], 256)).astype("<f4").tobytes(),
    )
    if screenshot:
        text.update(num_attention_heads=2, num_key_value_heads=1, head_dim=128)
        vision = dict(
            hidden_size=32,
            intermediate_size=64,
            depth=2,
            num_heads=4,
            patch_size=2,
            out_hidden_size=256,
            fullatt_block_indexes=[1],
            window_size=112,
        )
        config = dict(
            text,
            model_type="qwen2_5_vl",
            image_token_id=vocab["<|image_pad|>"],
            vision_config=vision,
            rope_scaling=dict(mrope_section=[16, 24, 24]),
        )
        shapes = screenshot_shapes(config)
        processor = dict(
            min_pixels=64,
            max_pixels=256,
            patch_size=2,
            temporal_patch_size=2,
            merge_size=2,
            image_mean=[0.1, 0.2, 0.3],
            image_std=[0.5, 0.5, 0.5],
        )
    else:
        vision = dict(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            image_size=4,
            patch_size=2,
            hidden_act="quick_gelu",
        )
        config = dict(
            model_type="llava_next",
            text_config=text,
            vision_config=vision,
            image_token_index=vocab["<image>"],
            vision_feature_layer=-2,
            image_grid_pinpoints=[[8, 8]],
            vision_feature_select_strategy="default",
        )
        shapes = vision_shapes("vision_tower.vision_model", vision)
        for name, shape in [("linear_1", (256, 32)), ("linear_2", (256, 256))]:
            shapes[f"multi_modal_projector.{name}.weight"] = shape
            shapes[f"multi_modal_projector.{name}.bias"] = (shape[0],)
        shapes["image_newline"] = (256,)
        tensors = {("language_model." + name): value for name, value in tensors.items()}
        processor = dict(image_mean=[0.1, 0.2, 0.3], image_std=[0.5, 0.5, 0.5])
    for name, shape in shapes.items():
        values = rng.normal(0, 0.03, shape).astype("<f4")
        if "norm" in name or ".ln_" in name:
            values = (
                np.ones(shape, dtype="<f4")
                if name.endswith("weight")
                else np.zeros(shape, dtype="<f4")
            )
        tensors[name] = (shape, values.tobytes())
    (root / "config.json").write_text(json.dumps(config))
    (root / "preprocessor_config.json").write_text(json.dumps(processor))
    _write_safetensors(root / "model.safetensors", tensors)
    return config


def visualized_files(root: Path):
    text = root / "text"
    _bert_files(text)
    store = TensorStore(text)
    config = json.loads((text / "config.json").read_text(encoding="utf-8"))
    config.update(
        hidden_size=32,
        intermediate_size=64,
        max_position_embeddings=64,
        layer_norm_eps=1e-6,
    )
    (text / "config.json").write_text(json.dumps(config))
    rng = np.random.default_rng(73)
    tensors = {}
    for name in store.names():
        shape = store[name].shape
        shape = tuple(32 if x == 4 else 64 if x == 8 else x for x in shape)
        if name == "embeddings.position_embeddings.weight":
            shape = (64, 32)
        renamed = (
            "bge_embeddings." + name.removeprefix("embeddings.")
            if name.startswith("embeddings.")
            else "bge_" + name
        )
        tensors[renamed] = shape
    p = "model_visual.visual"
    h = 32
    ff = 64
    b = p + ".blocks.0"
    tensors.update(
        {
            p + ".cls_token": (1, 1, h),
            p + ".pos_embed": (1, 5, h),
            p + ".patch_embed.proj.weight": (h, 3, 2, 2),
            p + ".patch_embed.proj.bias": (h,),
            "visual_proj.weight": (h, h),
            "visual_proj.bias": (h,),
        }
    )
    for norm, width in [
        ("norm1", h),
        ("norm2", h),
        ("attn.inner_attn_ln", h),
        ("mlp.ffn_ln", ff),
    ]:
        for suffix in ["weight", "bias"]:
            tensors[f"{b}.{norm}.{suffix}"] = (width,)
    for name in ["q_proj", "k_proj", "v_proj"]:
        tensors[f"{b}.attn.{name}.weight"] = (h, h)
    for name in ["q_bias", "v_bias"]:
        tensors[f"{b}.attn.{name}"] = (h,)
    tensors[f"{b}.attn.proj.weight"] = (h, h)
    tensors[f"{b}.attn.proj.bias"] = (h,)
    for name, shape in [("w1", (ff, h)), ("w2", (ff, h)), ("w3", (h, ff))]:
        tensors[f"{b}.mlp.{name}.weight"] = shape
        tensors[f"{b}.mlp.{name}.bias"] = (shape[0],)
    for kind in ["cos", "sin"]:
        tensors[f"{b}.attn.rope.freqs_{kind}"] = (4, 8)
    state = {}
    for index, (name, shape) in enumerate(tensors.items()):
        values = rng.normal(0, 0.03, shape).astype("<f4")
        if "Norm" in name or "norm" in name or "ln" in name:
            values = (
                np.ones(shape, dtype="<f4")
                if name.endswith("weight")
                else np.zeros(shape, dtype="<f4")
            )
        if "freqs_" in name:
            values = np.full(shape, 1 if name.endswith("cos") else 0, dtype="<f4")
        stride = tuple(int(np.prod(shape[i + 1 :])) for i in range(len(shape)))
        state[name] = _Tensor(
            _Storage(str(index), values.tobytes(), "F32"), shape, stride
        )
    checkpoint = root / "visualized.pth"
    _write_torch_bin(checkpoint, state)
    return checkpoint, text


class BgeVisualConverterTest(unittest.TestCase):
    @unittest.skipUnless(
        os.getenv("BGE_RUST_BIN"), "set BGE_RUST_BIN to exercise actual CLI"
    )
    def test_llava_text_input_preserves_the_existing_decoder_output(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model = root / "visual"
            config = decoder_visual_files(model, False)
            plain = root / "text"
            plain.mkdir()
            (plain / "config.json").write_text(json.dumps(config["text_config"]))
            for name in [
                "tokenizer.json",
                "tokenizer.model",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "vocab.json",
                "merges.txt",
            ]:
                if (model / name).exists():
                    (plain / name).write_bytes((model / name).read_bytes())
            store = TensorStore(model)
            tensors = {
                name.removeprefix("language_model."): (
                    store[name].shape,
                    store.read(store[name]),
                )
                for name in store.names()
                if name.startswith("language_model.")
            }
            _write_safetensors(plain / "model.safetensors", tensors)
            for quant in ["F32", "Q8_0"]:
                text_gguf, visual_gguf = (
                    root / f"text-{quant}.gguf",
                    root / f"visual-{quant}.gguf",
                )
                convert(plain, text_gguf, quant)
                convert_visual(model, visual_gguf, quant)
                outputs = []
                for path, prompt in [
                    (text_gguf, "[INST]  hello \n [/INST]"),
                    (visual_gguf, "hello"),
                ]:
                    result = subprocess.run(
                        [
                            os.environ["BGE_RUST_BIN"],
                            "--model",
                            str(path),
                            "--embedding",
                            "--embedding-output",
                            "raw",
                            "--kv-cache",
                            "f32",
                            "--threads",
                            "1",
                            "--prompt",
                            prompt,
                        ],
                        capture_output=True,
                        text=True,
                        check=True,
                        env={**os.environ, "RMI_SCALAR": "1"},
                    )
                    outputs.append(
                        result.stdout.split("embedding_raw:", 1)[1].splitlines()[0]
                    )
                self.assertEqual(*outputs)

    @unittest.skipUnless(
        os.getenv("BGE_SERVER_BIN"), "set BGE_SERVER_BIN to exercise actual HTTP"
    )
    def test_self_contained_visual_http_needs_no_projector_and_rejects_audio(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model = root / "clip"
            clip_files(model)
            output = root / "clip.gguf"
            convert_visual(model, output)
            image = root / "sample.png"
            write_image(image)
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            with (root / "server.log").open("w") as log:
                server = subprocess.Popen(
                    [
                        os.environ["BGE_SERVER_BIN"],
                        "--model",
                        str(output),
                        "--embedding",
                        "--host",
                        "127.0.0.1",
                        "--port",
                        str(port),
                    ],
                    stdout=log,
                    stderr=log,
                )
                try:
                    base = f"http://127.0.0.1:{port}"
                    for _ in range(100):
                        try:
                            urllib.request.urlopen(base + "/health", timeout=1)
                            break
                        except OSError:
                            self.assertIsNone(server.poll())
                            time.sleep(0.1)

                    def post(payload):
                        req = urllib.request.Request(
                            base + "/v1/embeddings",
                            data=json.dumps(payload).encode(),
                            headers={"Content-Type": "application/json"},
                        )
                        return urllib.request.urlopen(req, timeout=10)

                    payload = dict(
                        image=base64.b64encode(image.read_bytes()).decode(),
                        prompt="hello",
                    )
                    with post(payload) as response:
                        values = json.load(response)["data"][0]["embedding"]
                    self.assertEqual(len(values), 32)
                    self.assertTrue(np.isfinite(values).all())
                    self.assertAlmostEqual(float(np.linalg.norm(values)), 1.0, places=5)
                    with self.assertRaises(urllib.error.HTTPError) as error:
                        post(dict(audio="AA=="))
                    self.assertEqual(error.exception.code, 400)
                    error.exception.close()
                finally:
                    server.terminate()
                    server.wait(timeout=10)

    @unittest.skipUnless(
        os.getenv("BGE_RUST_BIN"), "set BGE_RUST_BIN to exercise actual CLI"
    )
    def test_visualized_pytorch_checkpoint_cli_and_image_position_capacity(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            checkpoint, text = visualized_files(root)
            image = root / "sample.png"
            write_image(image)
            for quant in ["F32", "Q8_0"]:
                output = root / f"{quant}.gguf"
                convert_visual(checkpoint, output, quant, text_model=text)
                for args in [
                    ["--prompt", "v1"],
                    ["--image", str(image)],
                    ["--prompt", "v1", "--image", str(image)],
                ]:
                    proc = subprocess.run(
                        [
                            os.environ["BGE_RUST_BIN"],
                            "--model",
                            str(output),
                            "--embedding",
                            "--embedding-output",
                            "raw",
                            *args,
                        ],
                        capture_output=True,
                        text=True,
                    )
                    self.assertEqual(proc.returncode, 0, proc.stderr)
                    vec = [
                        float(x)
                        for x in proc.stdout.split("embedding_raw:", 1)[1].split()
                    ]
                    self.assertEqual(len(vec), 32)
                    self.assertTrue(np.isfinite(vec).all())
                    self.assertAlmostEqual(float(np.linalg.norm(vec)), 1.0, places=5)
                proc = subprocess.run(
                    [
                        os.environ["BGE_RUST_BIN"],
                        "--model",
                        str(output),
                        "--embedding",
                        "--image",
                        str(image),
                        "--prompt",
                        " ".join(["v1"] * 64),
                    ],
                    capture_output=True,
                    text=True,
                )
                self.assertNotEqual(proc.returncode, 0)
                self.assertIn("context", proc.stderr.lower())

    def test_decoder_visual_inventory_and_padded_vocabulary(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for screenshot in [False, True]:
                model = root / str(screenshot)
                config = decoder_visual_files(model, screenshot)
                for quant in ["F32", "Q8_0"]:
                    output = root / f"{screenshot}-{quant}.gguf"
                    convert_visual(model, output, quant)
                    meta, tensors = read_gguf_directory(output)
                    self.assertEqual(
                        meta["general.architecture"],
                        "qwen2vl" if screenshot else "llama",
                    )
                    self.assertTrue(
                        meta["bge.screenshot" if screenshot else "bge.llava_next"]
                    )
                    self.assertEqual(len(meta["tokenizer.ggml.tokens"]), 266)
                    self.assertTrue(
                        meta["tokenizer.ggml.tokens"][-1].startswith("<|reserved_")
                    )
                    if screenshot:
                        self.assertEqual(tensors["v.patch_embd.weight"][0], 0)
                        self.assertEqual(
                            tensors["v.patch_embd.weight.1"][1], (2, 2, 3, 32)
                        )
                        self.assertEqual(
                            meta["qwen2vl.rope.dimension_sections"], [16, 24, 24, 0]
                        )

    @unittest.skipUnless(
        os.getenv("BGE_RUST_BIN"), "set BGE_RUST_BIN to exercise actual CLI"
    )
    def test_decoder_visual_cli_and_missing_image_validation(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            image = root / "image.png"
            write_image(image)
            for screenshot in [False, True]:
                model = root / str(screenshot)
                decoder_visual_files(model, screenshot)
                for quant in ["F32", "Q8_0"]:
                    output = root / f"{screenshot}-{quant}.gguf"
                    convert_visual(model, output, quant)
                    for args in [
                        ["--prompt", "hello"],
                        ["--image", str(image)],
                        ["--prompt", "hello", "--image", str(image)],
                    ]:
                        proc = subprocess.run(
                            [
                                os.environ["BGE_RUST_BIN"],
                                "--model",
                                str(output),
                                "--embedding",
                                "--embedding-output",
                                "raw",
                                "--threads",
                                "1",
                                *args,
                            ],
                            capture_output=True,
                            text=True,
                        )
                        self.assertEqual(proc.returncode, 0, proc.stderr)
                        vec = [
                            float(x)
                            for x in proc.stdout.split("embedding_raw:", 1)[1].split()
                        ]
                        self.assertEqual(len(vec), 256)
                        self.assertTrue(np.isfinite(vec).all())
                        self.assertAlmostEqual(
                            float(np.linalg.norm(vec)), 1.0, places=5
                        )
                    prompt = (
                        "<|im_start|>system\nX<|im_end|>\n<|im_start|>user\n<|image_pad|><|im_end|>\n<|im_start|>assistant\n<|endoftext|>"
                        if screenshot
                        else "[INST] <image> [/INST]"
                    )
                    proc = subprocess.run(
                        [
                            os.environ["BGE_RUST_BIN"],
                            "--model",
                            str(output),
                            "--embedding",
                            "--prompt",
                            prompt,
                        ],
                        capture_output=True,
                        text=True,
                    )
                    self.assertNotEqual(proc.returncode, 0)
                    self.assertIn("image", proc.stderr)

    def test_clip_tensor_inventory_and_tokenizer_are_self_contained(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model = root / "clip"
            config = clip_files(model)
            for quant in ["F32", "Q8_0"]:
                output = root / f"{quant}.gguf"
                convert_visual(model, output, quant)
                meta, tensors = read_gguf_directory(output)
                self.assertEqual(meta["general.architecture"], "bge-vl-clip")
                self.assertEqual(
                    json.loads(meta["bge.tokenizer_json"]),
                    json.loads((model / "tokenizer.json").read_text(encoding="utf-8")),
                )
                self.assertEqual(set(tensors), set(clip_shapes(config)))
                self.assertEqual(
                    tensors["visual_projection.weight"][0], 0 if quant == "F32" else 8
                )
                self.assertEqual(
                    tensors["vision_model.embeddings.patch_embedding.weight"][0], 0
                )

    def test_clip_does_not_drop_unknown_weights_or_accept_wrong_shapes(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model = root / "clip"
            clip_files(model)
            _rewrite_safetensors(model / "model.safetensors", extra="unknown.weight")
            with self.assertRaisesRegex(ValueError, "unsupported.*tensors"):
                convert_visual(model, root / "bad.gguf")
            self.assertFalse((root / "bad.gguf").exists())
            other = root / "wrong-shape"
            clip_files(other)
            store = TensorStore(other)
            tensors = {
                name: (store[name].shape, store.read(store[name]))
                for name in store.names()
            }
            name = "visual_projection.weight"
            tensors[name] = ((16, 64), tensors[name][1])
            _write_safetensors(other / "model.safetensors", tensors)
            with self.assertRaisesRegex(ValueError, "shape.*expected"):
                convert_visual(other, root / "wrong-shape.gguf")
            self.assertFalse((root / "wrong-shape.gguf").exists())

    @unittest.skipUnless(
        os.getenv("BGE_RUST_BIN"), "set BGE_RUST_BIN to exercise actual CLI"
    )
    def test_clip_cli_handles_text_image_and_sum_before_normalizing(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model = root / "clip"
            clip_files(model)
            image = root / "image.png"
            write_image(image)
            for quant in ["F32", "Q8_0"]:
                output = root / f"{quant}.gguf"
                convert_visual(model, output, quant)

                def run(args):
                    proc = subprocess.run(
                        [
                            os.environ["BGE_RUST_BIN"],
                            "--model",
                            str(output),
                            "--embedding",
                            "--embedding-output",
                            "raw",
                            "--threads",
                            "1",
                            *args,
                        ],
                        capture_output=True,
                        text=True,
                    )
                    self.assertEqual(proc.returncode, 0, proc.stderr)
                    vec = [
                        float(x)
                        for x in proc.stdout.split("embedding_raw:", 1)[1].split()
                    ]
                    self.assertEqual(len(vec), 32)
                    self.assertTrue(np.isfinite(vec).all())
                    self.assertAlmostEqual(float(np.linalg.norm(vec)), 1.0, places=5)
                    return vec

                text = run(["--prompt", "hello"])
                pic = run(["--image", str(image)])
                fusion = run(["--prompt", "hello", "--image", str(image)])
                self.assertNotEqual(text, pic)
                self.assertNotEqual(fusion, text)
                self.assertNotEqual(fusion, pic)
