import base64
import json
import struct
import tempfile
import unittest
from unittest.mock import Mock, patch
from pathlib import Path

import numpy as np

from convert_dots_tts import (
    GGML_BF16,
    GGML_F32,
    GGML_I64,
    GGML_Q8_0,
    GgufWriter,
    Tensor,
    _export_open_model,
    _fold_weight_norm_dim0_f32,
    _output_paths,
    _require_tensor,
    _tensor_nbytes,
    bf16_bytes_to_q8_0,
    emit_tensor,
    export_model,
    quantize_q8_0,
    read_gguf_directory,
    read_gguf_tensor_bytes,
    validate_quant,
    validate_variant,
)


class ExportContractTest(unittest.TestCase):
    def test_complete_export_quantizes_both_files_and_preserves_mixed_precision(self):
        def tensor(name, shape, dtype="BF16"):
            count = int(np.prod(shape))
            values = ((np.arange(count, dtype=np.float32) % 37 - 18) / 19).reshape(shape)
            if dtype == "BF16":
                raw = (values.view(np.uint32) >> 16).astype("<u2").tobytes()
            elif dtype == "I64":
                raw = struct.pack("<q", 7)
            else:
                raw = values.astype("<f4").tobytes()
            return Tensor(name, dtype, shape, raw)

        core = {}
        core["llm.model.embed_tokens.weight"] = tensor("embed", (4, 32))
        core["llm.model.norm.weight"] = tensor("norm", (32,))
        shapes = {
            "hidden_proj": (32, 32), "latent_proj": (32, 16),
            "coordinate_proj": (32, 16), "xvec_proj.0": (32, 32),
            "xvec_proj.1": (32,), "eos_proj.0": (32, 32), "eos_proj.2": (2, 32),
            "patch_encoder.ds_proj": (16, 16, 2),
            "patch_encoder.in_proj": (32, 16), "patch_encoder.out_proj": (32, 64),
            "velocity_field_predictor.input_layer": (32, 32),
            "velocity_field_predictor.time_embedder.mlp.0": (32, 256),
            "velocity_field_predictor.time_embedder.mlp.2": (32, 32),
            "velocity_field_predictor.output_layer.adaLN_modulation.1": (64, 32),
            "velocity_field_predictor.output_layer.linear": (16, 32),
        }
        for name, shape in shapes.items():
            core[name + ".weight"] = tensor(name + ".weight", shape)
            core[name + ".bias"] = tensor(name + ".bias", (shape[0],))
        speaker = {
            "model.conv.weight": tensor("conv", (2, 32, 1, 1), "F32"),
            "model.small.weight": tensor("small", (32, 1, 3, 3), "F32"),
            "model.norm.weight": tensor("norm", (32,), "F32"),
            "model.norm.num_batches_tracked": tensor("counter", (), "I64"),
            "resample.kernel": tensor("filter", (1, 32), "F32"),
        }
        vocoder = {
            "conv.weight_g": Tensor("g", "F32", (2, 1, 1), struct.pack("<2f", 1, 2)),
            "conv.weight_v": tensor("v", (2, 16, 2), "F32"),
            "small.weight_g": Tensor("g", "F32", (4, 1, 1), struct.pack("<4f", 1, 2, 3, 4)),
            "small.weight_v": tensor("v", (4, 3, 1), "F32"),
            "plain.weight": tensor("plain", (2, 32, 1), "F32"),
            "lstm.weight_ih_l0": tensor("ih", (8, 32), "F32"),
            "lstm.weight_hh_l0": tensor("hh", (8, 32), "F32"),
            "lstm.bias_ih_l0": tensor("bias", (8,), "F32"),
            "upsample.filter": tensor("filter", (1, 1, 32), "F32"),
        }
        sources = [Mock(header=data, tensor=data.__getitem__) for data in (core, speaker, vocoder)]
        llm_cfg = dict(num_hidden_layers=0, hidden_size=32, num_attention_heads=2,
                       num_key_value_heads=2, max_position_embeddings=128,
                       intermediate_size=64, rms_norm_eps=1e-6, vocab_size=4)
        cfg = dict(patch_size=4, latent_dim=16, campplus_embedding_size=32,
                   vocoder=dict(downsample_rates=[2, 2], sample_rate=24000),
                   PatchEncoder=dict(hidden_size=32, ffn_hidden_size=64, num_layers=0),
                   DiT=dict(hidden_size=32, ffn_hidden_size=64, num_layers=0, num_heads=2))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, data in (("llm_config", llm_cfg), ("config", cfg),
                               ("tokenizer_config", {}), ("vocab", dict(a=0, b=1, c=2)),
                               ("added_tokens", {"<eos>": 3})):
                (root / f"{name}.json").write_text(json.dumps(data))
            (root / "merges.txt").write_text("a b\n")
            for quant in ("bf16", "q8_0"):
                with self.subTest(quant=quant), patch("convert_dots_tts.load_latent_stats",
                        return_value=dict(mean=[0.] * 128, var=[1.] * 128)):
                    llm_path, mmproj_path = _export_open_model(root, "base", root, False, quant, *sources)
                llm_meta, llm = read_gguf_directory(llm_path)
                mm_meta, mm = read_gguf_directory(mmproj_path)
                self.assertEqual(llm_meta["general.file_type"], 7 if quant == "q8_0" else 32)
                self.assertEqual(mm_meta["general.file_type"], llm_meta["general.file_type"])
                if quant == "q8_0":
                    self.assertEqual(mm_meta["general.quantization_version"], 2)
                else:
                    self.assertNotIn("general.quantization_version", mm_meta)
                matrix_type = GGML_Q8_0 if quant == "q8_0" else GGML_BF16
                self.assertEqual(llm["token_embd.weight"][0], matrix_type)
                self.assertEqual(llm["output_norm.weight"][0], GGML_F32)
                self.assertEqual(read_gguf_tensor_bytes(llm_path, "token_embd.weight"),
                                 read_gguf_tensor_bytes(llm_path, "output.weight"))
                self.assertEqual(mm["dotstts.dit.input_layer.weight"][0], matrix_type)
                self.assertEqual(mm["dotstts.hidden_proj.weight"][0], matrix_type)
                patch_entry = mm["dotstts.patch_encoder.ds_proj.weight"]
                self.assertEqual(patch_entry[:2], (matrix_type, (32, 16) if quant == "q8_0" else (2, 16, 16)))
                self.assertEqual(mm["dotstts.latent_proj.weight"][0], GGML_BF16)
                self.assertEqual(mm["dotstts.xvec_proj.1.weight"][0], GGML_BF16)
                self.assertEqual(mm["dotstts.hidden_proj.bias"][0], GGML_BF16)
                self.assertEqual(mm["dotstts.latent_stats.mean"][0], GGML_F32)
                for name in ("conv", "small"):
                    _, folded = _fold_weight_norm_dim0_f32(vocoder[name + ".weight_g"].raw,
                        vocoder[name + ".weight_v"].raw, vocoder[name + ".weight_g"].shape[0])
                    expected = quantize_q8_0(np.frombuffer(folded, dtype="<f4")) if quant == "q8_0" and name == "conv" else folded
                    self.assertEqual(read_gguf_tensor_bytes(mmproj_path, f"dotstts.vocoder.{name}.weight"), expected)
                for source_name in ("plain.weight", "lstm.weight_ih_l0", "lstm.weight_hh_l0"):
                    expected = vocoder[source_name].raw
                    if quant == "q8_0":
                        expected = quantize_q8_0(np.frombuffer(expected, dtype="<f4"))
                    self.assertEqual(read_gguf_tensor_bytes(mmproj_path, "dotstts.vocoder." + source_name), expected)
                self.assertEqual(mm["dotstts.speaker.conv.weight"][:2],
                    (GGML_Q8_0, (32, 2)) if quant == "q8_0" else (GGML_F32, (1, 1, 32, 2)))
                for name, source_name in (("speaker.small.weight", "model.small.weight"),
                                          ("speaker.norm.weight", "model.norm.weight"),
                                          ("speaker.resample_kernel", "resample.kernel")):
                    self.assertEqual(mm["dotstts." + name][0], GGML_F32)
                    self.assertEqual(read_gguf_tensor_bytes(mmproj_path, "dotstts." + name), speaker[source_name].raw)
                self.assertEqual(mm["dotstts.speaker.norm.num_batches_tracked"], (GGML_I64, (1,), 8))
                self.assertEqual(mm["dotstts.vocoder.upsample.filter"][0], GGML_F32)
                self.assertFalse(any(name.endswith((".weight_g", ".weight_v")) for name in mm))

    def test_weight_norm_fold_matches_pinned_torch_short_rows_bitwise(self):
        g_bits = (
            0x3EFF9355, 0x3F131C5A, 0x3F10961D, 0x3F1F8408,
            0x3EE5DE22, 0x3F01ACA1, 0x3EF24F51, 0x3FBF3BE8,
            0x3F29967F, 0x3F260836, 0x3F1FF717, 0x3F19EF22,
        )
        v_bits = (
            0x3E5EF79F, 0x3E42FAD1, 0x3DA03D88, 0x3E23DE04,
            0x3DE0DB4B, 0x39D95976, 0x3DBB09CE, 0x3E33FD66,
            0x3E67CD2D, 0x3DD88A29, 0x3E6E329C, 0x3E3AC183,
            0x3DE97A93, 0x3DD14DCB, 0x3DF6DFAB, 0x3DD9B752,
            0x3E078FFF, 0x3DDFF0D3, 0xBE1C1B66, 0xBE10AC05,
            0xBE1E8CC6, 0xBDF5F698, 0x3B8811B1, 0x3DF40404,
            0xBDC982B7, 0xBE0F34B2, 0xBE101B75, 0xBE92BE7A,
            0xBD9E74DD, 0xBC198808, 0x3EA36E58, 0xBC0AF22F,
            0x3E8A4AFB, 0xBDB53CE6, 0xBDDC154D, 0xBDC47DEB,
        )
        expected_weight = (
            0x3EB9B734, 0x3EA26770, 0x3E0577FA, 0x3EF29C05,
            0x3EA673ED, 0x3AA0E544, 0x3E2F9886, 0x3EA8FA81,
            0x3ED99ED7, 0x3E56E6E9, 0x3EEC656E, 0x3EB957EE,
            0x3E867B40, 0x3E711D26, 0x3E8E3260, 0x3E888599,
            0x3EAA033C, 0x3E8C6CD4, 0xBE8E8674, 0xBE8415C4,
            0xBE90C16B, 0xBF87B7E9, 0x3D16294B, 0x3F86A4CD,
            0xBE96B1F9, 0xBED62FC3, 0xBED788E7, 0xBF2036DA,
            0xBE2D0078, 0xBCA79FE7, 0x3EF42D30, 0xBC4F9827,
            0x3ECE9E57, 0xBEA12528, 0xBEC3AF15, 0xBEAEB540,
        )
        raw = lambda words: struct.pack(f"<{len(words)}I", *words)

        _, actual_weight = _fold_weight_norm_dim0_f32(
            raw(g_bits), raw(v_bits), len(g_bits)
        )

        self.assertEqual(struct.unpack("<36I", actual_weight), expected_weight)

    def test_weight_norm_norms_match_pinned_torch_generator2_fixture_bitwise(self):
        v_raw = base64.b64decode(
            """
            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
            """
        )
        expected_norm = base64.b64decode(
            """
            0zKRPhEYpz74+q0+fSCwPiolnz5BXLQ+y8qaPlc7rT4XdY0+422xPhFBqD61xNg+rpqEPl2wmz5ZFhQ/a6G7PquZ
            tz5gabI+eQ7KPtCKsj7xxbg++0XDPvb2qT7R6iQ/
            """
        )

        actual_norm, actual_weight = _fold_weight_norm_dim0_f32(
            struct.pack("<24f", *([1.0] * 24)), v_raw, 24
        )

        expected_weight = base64.b64decode(
            """
            FjkivNwAUjwH/Xo9N/ssPV3xCb48Bm+9NsQsPscc2D21YEY85EVGPZTBjz1czFU9BeezvAAw8j1cnQ8+Ksj2PD6U97nXa6A8A2Bm
                        PXM8OD1wh/y8B2WYPavTDj5e71I9s9Q8PeqpMjykNCy9kiOWO8fhgT694CM/3AR+vort/b7E0kw9jJ3YvT8ww7306J08U2AhPR2m
                        Gj05zei9OzjivQmiiTxmqMW8Ts+DPUwJKj0ZzoY9A9cvvanFfr3RRQu6zSWAvOSOir2u55S9hPkbvSMGKr6v8jy+KMl4vjcReb40
                        0B69U2yHvcG6C72Ns8c8sb5EvVJmjr2Cbzi9SONRvNV8LL1FRoC97FokvczfZDw7pPC8meeFvWXRV7143pO8ZxuUPRUL5T3tAeY9
                        zXecPGWDsr7nnz2+j3e1vM6kLr3CG2M+pf9PPmqMWz4yEqo9LtUhPoS4iD7gbJ4+EbOBPsupo73EeKK9zLicvZXLUb0YIy8+eL5v
                        PjoRhD47oQA+id8jPlUyNT7hCiQ+jBQfPm1EQD6eIhM+IUdDPnwOGz70w0A+bc43Pn2HJz7Ikeo9KdoKPpRcCD6QdtM9baeGPd2M
                        Oj6CHT4+QckgPpR4ED6zzhs+cT0bPtFvBD7Ms8U9dJcSvgEZD76mtg2+1yEOvmXnkDx42XK7VejQvSWu270A6Ey+cbQnvkBoHL4R
                        Ni6+u5vrvURGx73OHZy9/vSdvTuAQD40gj0++HIsPpmDHT4W7wm+nDfqvZ3i6722//e9SuMXPu4SQz0SZ3w9HlY2vCxlij76jLQ9
                        kTrFPUklMz2qDGs9DQKjvFy/5rxkPVq88AA3PmWt3D3JAaa9MneJvh4aST1lEJo85lhQPTDcMD2QdEU+HNXtPVeck7y8iiq+yEXX
                        vaPa9bwDS3G96gvEvcRmtr5E7Ls8T50yvnwy9b62sy++9OB+vct4z7xq+MY9TfmYvnqdLb4ke8q99F2xvSYzRD6VQ4y8nO4/Pave
                        Vz3y+yS+5P2RvU8uKb1LbSw9JSyvvf2swL2fPbe9mrLQvZVzEb4gIii+Ta86vpYEJb7485G94g+MvY8ZpL1EOqG9jg6Nvf6rnr0t
                        vJu9Ij2qvSnC7r3I3P29aYTkvbiI6L2cVma9yb97vasQgr2ShoW96y8+PsibTj6wxE8+o6BEPmX4IDxczIW8s5eUPUW6zjyD6YE+
                        vwCMPsMhjz7o8HQ+0w7Au86lZzwKna08XwnJPH5uEr5gphC+0EkUvij4E75Awlw+iiZ3PkKDfz5XKXk+nUebvYoBpr0grZG9eM3C
                        vV56Fb5K2Ru+v5P8vXeHK75S7Gu9+OtPvc5Al70JLbm9jWuMvW/zn73PFqe9oDzPvQKH370sldC9NlTXvVDi9b0s3lG9kL51vV/S
                        ir1hyZq9WbMzPvxRSD6+glg+g7ZbPmdEpD0nLpA9YUC6PH3SIT3v5Ho+n36GPjBGkT42t40+GYWOOhwtoDw8H8I8qGzqPNLhCr4f
                        9fu9sv/vvcZZBb6YvWc+kDR6Piulgz4uRYk++dh9Pc+NDzwa+7k8bk7BPI60Hj66jO89+1VKvXN2C75e4j09qF6KPNJvBzwOLLo9
                        g2HiPO39vL3F3w++OwRVvbZ1pTyeHfA75zs6O/+iCz0/4IQ9KVQEvR4Wu70YFUC9/Dq7u4I3kD3U5jU+wS8qPu9xtr3UPqe+idj8
                        O7qj7D53yGW9lyU9Ppnktz6XZpU+PvokvjtdB76hMD88rWXgPWVBJj2ZndI8aPozvP54BD0Xtlm9kGfMPc1JlD6Z4I0+saIsPkHv
                        Cj4mqqQ9QL0IvZtXaT7VW0k+4q45PszjJTy32yw+H/K1Pft7YL3KBBG++E8zPmWv8j2xfZu93y1cvg67CD6DKsg9j5Gyu5bJm72d
                        xkw+uf0ePpyfrbvcUhS+YguxvU8QM711GUM6LpqYPbIyvj2G9hw988rJvkK7yr7d+hm+wYXBvflalzyaJ0A+j98XvuCZD75A+wO+
                        WPhhvcTrSD71WdA9UhwFPB3RS73CsdS9O1+WveTUOLys5fQ9ecEMPQz5HTtxm+W7aAKgvLEvDrylDjO9hAS9PX5mvz3OW467EZNU
                        O9k8xzy83rm8DRZLvb6ykT3JzPI92Dz0vP7vhjyHvhO88qA2vG8EhrxN+iC9OqUgPZC28z1oas88ZilEPQOiJj0g/049dX6vPebA
                        Aj4Bpjc/z83WvKIcEb+/3Ww95QZTva+6xbyqb9Y97rn0vAtXmTwJwQC+tCgnvtxF3zyruQA81PMLvIMep7wwXhk9tNvBvKMLz7vD
                        N5k9fSYjPoWBJD6fvx8+Fp0uPm8J0D1mQss93BchPr+JCz7nIT0++0FAPreHRz6HFCM+GMbpPb428D1uvfE9O4LTPQCXKz55kzc+
                        EVYoPsVRKT6F+wg+xU0HPqUyCj42ovY9upIJvr3/C76crCW+ydorvpTHGb0lf4O8PXyHu9emjb1UvDu+2AVHvqY/T75aGGO+T0aT
                        vVmvdL1ARj+9+f88vaylID5deTs+5WEmPq8tPT4MMxG+9TcFvoyt+71Ttxi+jrb0PDQiij2twrA8OtDTPfuLID2hJbQ9bHa0vB41
                        PD2qSSU9JH0/PR6EpD2s5AQ+C0k5PUZHU7wBBHM91dRePi9Ulj0mx6g9hHKWPYk86T1wMaE90UENPc9CFT3QMSI+7FmCvFiJK701
                        dh29lWnEvfnluDw7DKu+mb+kPtSsNj+40sG8oT1LPbP+irwseUi+m2MgvTlBAr5H3jQ8iQU6Pbf/wLxKmZk9QfSdPXs+oz1rdAU7
                        FB8CPQYVHj2vkKq9fUYAPBK4kj3M+I09mZKnPRP6XL1A2sm9k3TFvL5HyjzZ5bg9iCcDPjjJ2j16xhM+Ill1PpLelz6dSLE+EqfL
                        PsEEMrySXUQ9e3ctPaWpFT1naCk+bhRPPrCyhT4mrJE+ZU2FvZT7nb2PkJq9oSWIvY5xMr56wG6+v2iBvgQBdb7JL4O9VqiZvYXh
                        jr0/lHa9WtdmPbDYQT0fWCo9ND8qPVdaz7yiiy69N4u+OtWqnT1kfuO82Zckvak9Eb1AhP68TPaLvDDdRLyx4Sk7g04xvK4Gvr1y
                        1k88lwISPhI7UbuEn2w88kxbPJNVCLy28Am8cMT3PBUJ0T1Ssp88aGbKvWQFo7y1fsC87bbMvIdZDjv2pQw7pwGkPUoiRD12BFi9
                        pYgRPe3EED37CAQ9XiFFPXFOFD8w0vI+g1j8vmKThr5E84o8nt2tvV/TezyWc8s9zW6GPF2iC70SpSG+0WtTvfGMdjyyXfe8stmJ
                        vHj4VbzD8Oo8ehNQvWa5T7tRFpQ9lr6/Ot1kXjyhphe8waIovRkisLxc9Os98P1YPX9w0L1nRng9yjODPMrMML0mLPs8OoqkPbhD
                        QT0lybW9OwqNvZ1k7ToMpsE79i8GveqxIzzcWkc9pnR8PYz2Pb00QoC9xcirPMUYCzwYNwE9L9L/PPtpGT9KFGW+A8MTv/7/tz4K
                        yF69R6kcvWue0D1iT549YpTJPDVUrL3sPMS9bKOXPZtm87v1A2E6prQgvD4VzricVci8EI0UvcoAbz2Qcqw92AGRvUMTIT0MCgI+
                        mRQ1PqV8v73eVto7ku2UPYoFeTwQi2U8iYL+PWqpZT4Gpq49krcpPpmsrj5LOLg+s05NPvEWZ733Acc8lysTPorDFj7iBQo+7q9/
                        Pjr7oD78Zls+OgMuvZlNw72x2ue93Kclvr065Ttv/tc9lsTVvFUYFL5FveG9MvEtvuJoM76cFCe+4tcfvfmjLr0GTIq903yNvW8f
                        i71sOhO6UGMxPZuA5jwTj/O7tHLYvYkxSr5pgRm+chWkvYO3lL3G0mq96hievepMAL6vpPy9T1XwvXbLAL7Mtnm9YB41vT41gL0E
                        vYe9ZgGqva26o73k8py9O4C9vdmY1r2G/+O9wOKcvbf7v71T5Za9RDeNvQURmb3x/5S92bZWPjJoRj7KIGA+xHpPPu5eSj2qtHc9
                        NLRRPZHs8T2P5Y4+JzuRPksOjz4J74g+GEvHPCPAKT2pNDM98k4oPSW3s71WtPG9mG3pvXSUyb1+5oI+nAGHPm48gz56ZIM+kNoq
                        Pp+WMD6Wqh0+VeEyPrDJTD5JSkE+sVr9Pff2Gj7kFjQ+mskuPogMNz6/Qy4+ksLlPT7a5D2znfE9pDflPWSmNj6ATTw+xPMwPthr
                        RD4GLAM+1pj0PfIgAD58Y+o984LnvW2qC77AhwS+RxUfvnSISr0mehq8DldbPYLYAz2oIgK+0CnKvRuTD74rfD++GuatvUjEnr2p
                        eYW98x+mvYTFYj6se1Q+bS9aPsoXXD5flNC99YSxvR6CtL0WlQe+r3QWPoLuLD57CCA+mCQzPiVUDj6DWcw99IggPnOmUT6O0Tg+
                        9N47PhRyRT4uzSM+XpsGPuXDDj67uAo+A9ngPdEKMj4L6zc+mF8sPqVhOD4qjQ4+7JgQPhrdEz49VPU9DmwjvqNk371Ae/K9AYER
                        vvkMlrz4lBI9607NPKO9Ar2ACC6+s/Ejvt+SKL6AlS6+G+n7vRSB6b354dK9G5bbvcjKNT5TyTQ+mlIFPhljMT6kxgG+2tvqvZh9
                        8b3+QgS+R/AJPsGcHj6K0RY+P1RXPMhISLu0p6Y9BcmbPPrVDL01iRA+3UEFPg+pvDwsZTC9h+ydPsNSeD6tXho+Wz3UPaLLkj0E
                        w6897/7DPdKnKr0bR1A+Zp5NPtrI+D2S5lA9gPwavgtn/72hgd69CbyOvUHm7b0wSUW+Q1KFvq8og74BxIa+CH+DvqxiX77iw8a9
                        w9aDvOyszrwpMBK9SDyLvGIz0D2FmM09pcSiPchZpL0pDzq+sT0/vn6mMr4oo+y9MZRAPkNVPz5VMSM+9eM2Pq01Cj7haEU+8hcr
                        PrpvHT7mQUk+rFI7PhGUOz658zc+IETsPfcJ5j2F5+k9dTPfPZ8pPz4KfT4+IQgwPuv7Pj5jygY+pon5PXCG/j24WOs9jVQTvmwa
                        571fXwa+Fv72vVRjsDwdXom67BydPD8BiT3pARy+3gImvtLCAL72mDa+ZfDQvS2yvr1sWpq9MZepvQwJRj6HkzA+77ouPip1RD48
                        o/G9JbHPvQWiy72daQC+Sjh9PY4m0ryFzF28k9sjve9HIz6fmNo9IKVIvfbTC77jKtc8s3dGvZGC2b3yS269ZmHQPJEklL3rb0S+
                        uc91vq1hTjwpAPK8fbc5vfg9Vb3n4p094XTnvMjTF74DfUC++0WAvCPrVT2ULT8+yBdiPoMo4L1Iq5G+rXoavhOUdz3SNbu97WUb
                        Pj/toz5NgLU+NvhvvovYNr7dfUG9g9RLPRPpZD1Eycq8VI10vWmA07xwgsy9UzwLPYEceD50R6w+5jSwu88VzD2Wc9c9RU4fPtCg
                        Bb4Y4Fo9+x/3PXTQFT5fatY9NvxBPpcPgj77KoQ+DG15PSaQAT6XFlE+87J1Pvb7WD0wpfs9dS4YPiQzXT7ITTw9U43RPQoeNj7M
                        0F8+TSjAvRyK972Bhiu+UmVQvm7r2z2ra+o9mfimPVYtCj7R2/C9SVY4vtSVN744MHy+4YRKPUuoRjor8m+9Fl+nvRwHBr31EOE9
                        VpUlPsKQND7tqOC85tewvZOpD76KazS+cVqkvcpj2b3gpM29wYndvY2wE74fPh6+ZXd2vtP3gL4kI4u96yeLvQ9hob2tn6e9rE+T
                        vb2KvL1RI8O9twrPvcCjBL53Whe+Ou8Avi6j+r20smi9EjmZvesBqr2oq669+oILPvzXND6INBE+6axEPr3zAr59uBO+9On2PVAL
                        Wz3hrT8+yjyXPiJeTz5oI+Y9/FNFvEqBEj33yoM9lpmMPcvLF77biyC+ZrM3vlkYMr7c/CQ+14RwPpGvgT6ahFk+7mAEvBUOLDxa
                        EQw88A7BPYUNAT2BQ488ENYxvRxnyj0RTbg7QpaZPS7eID5jNTE++c0hPsxGiz4Vmsw+4/D4PmatOL107YA7JPs+PTKYhz1Fk+89
                        aLsnPjGUqD5qZc8+Ah2LvZyrqr1Hj1K9FCetvdirhL0Ko9m8/lQ/vfgofL3KkIO99gTEvW3sxr0SnOm9/xfPvMBggbudDpK7CIZZ
                        vAY78DzgDB+9QVTFu0G7yjsn8py95Cexvcz7lb2PtNa9
            """
        )
        self.assertEqual(actual_norm, expected_norm)
        self.assertEqual(actual_weight, expected_weight)

    def test_export_model_discovers_and_wires_component_pairs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model"
            out = root / "out"
            model.mkdir()
            out.mkdir()
            required = (
                "model.safetensors", "speaker_encoder.safetensors", "vocoder.safetensors",
                "llm_config.json", "config.json", "tokenizer_config.json", "vocab.json",
                "added_tokens.json", "merges.txt", "latent_stats.pt",
            )
            for name in required:
                (model / name).touch()
            sources = [Mock(), Mock(), Mock()]
            expected = (out / "dots-tts-base-BF16.gguf", out / "dots-tts-base-mmproj-BF16.gguf")
            with patch("convert_dots_tts.open_safetensors", side_effect=sources) as opened:
                with patch("convert_dots_tts._export_open_model", return_value=expected) as inner:
                    actual = export_model(model, "base", out, False, "bf16")
            self.assertEqual([call.args[0] for call in opened.call_args_list], [
                model.resolve() / "model.safetensors",
                model.resolve() / "speaker_encoder.safetensors",
                model.resolve() / "vocoder.safetensors",
            ])
            inner.assert_called_once_with(
                model.resolve(), "base", out.resolve(), False, "bf16", *sources
            )
            self.assertIs(actual, expected)
            for source in sources:
                source.close.assert_called_once_with()

    def test_output_paths_include_primary_precision(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            self.assertEqual(_output_paths(out, "base", "bf16"), (
                out / "dots-tts-base-BF16.gguf",
                out / "dots-tts-base-mmproj-BF16.gguf",
            ))
            self.assertEqual(_output_paths(out, "edit", "q8_0"), (
                out / "dots-tts-edit-Q8_0.gguf",
                out / "dots-tts-edit-mmproj-Q8_0.gguf",
            ))
            with self.assertRaises(ValueError):
                _output_paths(out, "base", "q4_k")

    def test_weight_norm_norms_match_pinned_torch_for_row_lengths_1_through_32(self):
        expected_norms = (
            0x3F78151D, 0x3F7A7195, 0x3F53592C, 0x3F5C45B7,
            0x3FA9DCFF, 0x3FC6370E, 0x3FC0D729, 0x3FC3D0C0,
            0x3FE64F06, 0x40026F1F, 0x40078230, 0x40090849,
            0x400D6EDE, 0x401F3351, 0x4025698C, 0x4024817C,
            0x4025C3BB, 0x40300E62, 0x4038BA66, 0x4036AE90,
            0x403303BD, 0x4035A2E7, 0x4040EBF6, 0x4042FD71,
            0x4042DBB5, 0x4043D4BB, 0x404F688A, 0x4056B2E3,
            0x4055E0C8, 0x4056B2E3, 0x405FCE69, 0x4068E18C,
        )
        for length, expected in enumerate(expected_norms, start=1):
            values = [
                ((index * 37 + length * 11) % 211 - 105) / 97.0
                for index in range(length)
            ]
            norm, _ = _fold_weight_norm_dim0_f32(
                struct.pack("<f", 1.0),
                struct.pack(f"<{length}f", *values),
                1,
            )
            self.assertEqual(struct.unpack("<I", norm)[0], expected, f"length={length}")

    def test_bf16_bits_and_clip_metadata_survive_readback(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "tiny.gguf"
            writer = GgufWriter(path)
            writer.add_meta("general.architecture", "clip")
            writer.add_meta("clip.has_audio_encoder", True)
            writer.add_meta("clip.has_gen_audio_encoder", True)
            writer.add_meta("clip.audio.projector_type", "dotstts_spkenc")
            writer.add_meta("clip.gen.audio.projector_type", "dotstts_gen")
            bf16 = bytes.fromhex("803f20c0")
            writer.add_tensor("bf16.weight", GGML_BF16, (2,), bf16)
            writer.add_tensor("f32.bias", GGML_F32, (1,), bytes.fromhex("0000803f"))
            writer.write()

            metadata, tensors = read_gguf_directory(path)
            self.assertEqual(metadata["general.architecture"], "clip")
            self.assertEqual(metadata["clip.gen.audio.projector_type"], "dotstts_gen")
            self.assertEqual(tensors["bf16.weight"], (GGML_BF16, (2,), 4))
            self.assertEqual(read_gguf_tensor_bytes(path, "bf16.weight"), bf16)

    def test_duplicate_names_and_implicit_overwrite_are_rejected(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "tiny.gguf"
            writer = GgufWriter(path)
            writer.add_meta("general.architecture", "clip")
            with self.assertRaises(ValueError):
                writer.add_meta("general.architecture", "clip")
            writer.add_tensor("x", GGML_F32, (1,), bytes(4))
            with self.assertRaises(ValueError):
                writer.add_tensor("x", GGML_F32, (1,), bytes(4))
            writer.write()
            with self.assertRaises(FileExistsError):
                writer.write()
            writer.write(overwrite=True)

    def test_variant_is_explicitly_bounded(self):
        self.assertEqual(validate_variant("base"), "base")
        self.assertEqual(validate_variant("edit"), "edit")
        with self.assertRaises(ValueError):
            validate_variant("experimental")

    def test_quant_is_explicitly_bounded(self):
        self.assertEqual(validate_quant("bf16"), "bf16")
        self.assertEqual(validate_quant("q8_0"), "q8_0")
        with self.assertRaises(ValueError):
            validate_quant("q4_k")

    def test_q8_0_rounds_half_away_from_zero_and_encodes_zero_block(self):
        values = np.zeros(64, dtype=np.float32)
        values[:5] = [-127.0, -0.5, 0.5, 1.5, 127.0]
        raw = quantize_q8_0(values)
        blocks = np.frombuffer(raw, dtype=np.uint8).reshape(2, 34)
        self.assertEqual(blocks[0, :2].copy().view(np.float16)[0], np.float16(1.0))
        self.assertEqual(blocks[0, 2:7].view(np.int8).tolist(), [-127, -1, 1, 2, 127])
        self.assertEqual(blocks[1, :2].copy().view(np.float16)[0], np.float16(0.0))
        self.assertEqual(blocks[1, 2:].view(np.int8).tolist(), [0] * 32)

    def test_q8_0_matches_pinned_llama_cpp_bytes(self):
        # llama.cpp b96806d96061049a5b574269b049bf6241d63d46,
        # ggml/src/ggml-quants.c quantize_row_q8_0_ref, compiled without fast-math.
        values = np.zeros((4, 32), dtype=np.float32)
        values[0, :8] = [-1, 1, 0.50003, -0.50003, 0.011813, -0.011813, 0.999, -0.999]
        values[1, :7] = [-127, 127, -0.5, 0.5, 1.5, -1.5,
                         np.nextafter(np.float32(0.5), np.float32(0))]
        values[2] = np.arange(-16, 16, dtype=np.float32) * np.float32(1e-7)
        expected = bytes.fromhex(
            "0820817f40c002fe7f81000000000000000000000000000000000000000000000000"
            "003c817fff0102fe0000000000000000000000000000000000000000000000000000"
            "000081899199a1a9b1b9c0c8d0d8e0e8f0f8000810182028303840474f575f676f77"
            "00000000000000000000000000000000000000000000000000000000000000000000"
        )
        self.assertEqual(quantize_q8_0(values), expected)
        self.assertEqual(quantize_q8_0(values.astype(">f4")), expected)

    def test_q8_0_rejects_nonfinite_and_unrepresentable_scales(self):
        for bad in (float("nan"), float("inf"), -float("inf"), 1e10, 1e-40):
            values = np.zeros(32, dtype=np.float32)
            values[0] = bad
            with self.subTest(value=bad), self.assertRaises(ValueError):
                quantize_q8_0(values)

    def test_q8_0_rejects_payload_not_multiple_of_block(self):
        with self.assertRaisesRegex(ValueError, "not a multiple of block size"):
            quantize_q8_0(np.zeros(33, dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "odd byte length"):
            bf16_bytes_to_q8_0(b"\x00\x00\x80")

    def test_tensor_nbytes_handles_block_quants_and_rejects_unaligned(self):
        self.assertEqual(_tensor_nbytes(GGML_Q8_0, (32,)), 34)
        self.assertEqual(_tensor_nbytes(GGML_Q8_0, (1024, 1024)), 1024 * 1024 // 32 * 34)
        self.assertEqual(_tensor_nbytes(GGML_F32, (3,)), 12)
        self.assertEqual(_tensor_nbytes(GGML_BF16, (4,)), 8)
        with self.assertRaisesRegex(ValueError, "not a multiple of block size"):
            _tensor_nbytes(GGML_Q8_0, (33,))
        with self.assertRaisesRegex(ValueError, "row width"):
            _tensor_nbytes(GGML_Q8_0, (16, 2))

    def test_q8_0_payload_survives_atomic_gguf_readback(self):
        with tempfile.TemporaryDirectory() as raw_dir:
            path = Path(raw_dir) / "tiny.gguf"
            payload = quantize_q8_0(np.arange(32, dtype=np.float32))
            writer = GgufWriter(path)
            writer.add_meta("general.architecture", "qwen2")
            writer.add_meta("general.file_type", 7)
            writer.add_tensor("weight", GGML_Q8_0, (32, 1), payload)
            writer.write()

            metadata, tensors = read_gguf_directory(path)
            self.assertEqual(metadata["general.file_type"], 7)
            self.assertEqual(tensors, {"weight": (GGML_Q8_0, (32, 1), 34)})
            self.assertEqual(read_gguf_tensor_bytes(path, "weight"), payload)

    def test_llm_norm_must_be_bf16(self):
        norm = Tensor("llm.model.norm.weight", "F16", (4,), bytes(8))
        with self.assertRaisesRegex(ValueError, "expected BF16"):
            _require_tensor(norm, "BF16", (4,))

    def test_mapped_tensor_shape_must_match_config_dimensions(self):
        weight = Tensor("hidden_proj.weight", "BF16", (1024, 1024), bytes(2 * 1024 * 1024))
        with self.assertRaisesRegex(ValueError, "expected shape"):
            _require_tensor(weight, "BF16", (1024, 1536))
        truncated = Tensor("weight", "BF16", (2, 32), bytes(64))
        with self.assertRaisesRegex(ValueError, "64 bytes, expected 128"):
            emit_tensor(GgufWriter(Path("unused.gguf")), "hidden_proj.weight", truncated, "q8_0")


if __name__ == "__main__":
    unittest.main()
