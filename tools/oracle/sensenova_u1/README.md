# SenseNova-U1.6-8B-MoT — Python oracle (reference generator)

Standing up the **official** reference implementation so the Rust port in
`src/models/diffusion/sensenova_u1/` can be diffed numerically against it.

Contract source of truth: [`docs/develop/SENSENOVA_U1.md`](../../../docs/develop/SENSENOVA_U1.md).

> **STATUS: the oracle ran successfully; the Rust port does not match it yet.**
> Reference tensors and a verified PNG exist (the replay driver reproduces the
> official CLI's PNG bit-for-bit, `match_exact: true`). The port is proven exact
> on the tokenizer, timesteps, initial noise, vision tower, both embedders and
> the ConvDecoder, and correct on the prefix (understanding) pass — but the
> generation-branch forward still diverges. Full measured results, the
> localisation table and the next steps are in
> [`validation.md`](validation.md).

---

## 1. Pinned versions

| item | value |
| --- | --- |
| oracle | `github.com/OpenSenseNova/SenseNova-U1` @ **`d2d9bd1`** |
| weights | `SenseNova/SenseNova-U1.6-8B-MoT-GGUF` (ModelScope) |
| Q8_0 | `SenseNova-U1.6-8B-MoT-Q8_0.gguf`, **20,695,981,472 B (20.7 GB)**, sha256 `097ca40a…4806` |
| Q6_K | 16,620,178,848 B (16.6 GB) |
| Q4_0 | 12,281,421,216 B (12.3 GB) |
| remote python | `/opt/venv/bin/python3`, CPython 3.12.3 |
| torch | `2.10.0+rocm7.2.4.git3d3aa833`, `torch.version.hip = 7.2.53211` |
| GPU | 1 × gfx1100 (Radeon PRO W7900, 48 GB) visible to torch out of 8 host GPUs |
| transformers | 5.24.0 | accelerate | 1.16.0 | numpy | 2.4.6 |
| diffusers | 0.43.0 (GGUF dequant-on-forward path) | gguf | 0.19.0 | sentencepiece | 0.2.3 |

> **Q8_0 is 20.7 GB, not ~9 GB.** Only 589 of 1116 tensors are quantized; the
> remaining 527 (embeddings, norms, convs, timestep MLPs) stay at full
> precision. Budget ~21 GB RSS for weights alone before any activations.

## 2. Remote environment recipe

`bootstrap_remote.sh` is the idempotent one-shot; the essentials:

```bash
# 1. venv. NOTE: `python -m venv --system-site-packages` from /opt/venv does NOT
#    inherit torch, because the new venv's base_prefix is /usr, not /opt/venv.
#    Bridge it with a .pth instead of letting pip near torch.
/opt/venv/bin/python3 -m venv --system-site-packages /root/sn_oracle_venv
echo /opt/venv/lib/python3.12/site-packages \
  > /root/sn_oracle_venv/lib/python3.12/site-packages/_opt_venv_bridge.pth

# 2. oracle, editable, no deps. pyproject pins torch==2.8.0 / torchvision==0.23.0
#    / pillow==12.0.0; honouring those would replace the ROCm torch with a CPU
#    wheel, so every install here is --no-deps.
cd /tmp/sn_oracle
/root/sn_oracle_venv/bin/pip install -e . --no-deps
/root/sn_oracle_venv/bin/pip install --no-deps gguf diffusers sentencepiece httpx tqdm packaging

# 3. prove torch survived
/root/sn_oracle_venv/bin/python -c "import torch; assert torch.version.hip; print(torch.__version__)"
```

Verified after install:

```
torch 2.10.0+rocm7.2.4.git3d3aa833  hip 7.2.53211  cuda_available True  devices 1
sensenova_u1 OK  /tmp/sn_oracle/src/sensenova_u1/__init__.py
diffusers 0.43.0  gguf 0.19.0  sentencepiece 0.2.3
from diffusers.quantizers.gguf.utils import GGUFParameter, SUPPORTED_GGUF_QUANT_TYPES   # OK
from diffusers import GGUFQuantizationConfig; from diffusers.quantizers.gguf import GGUFQuantizer   # OK
```

The last three matter: the oracle's `utils/gguf_loader.py` imports exactly
those symbols, and `diffusers` moved `GGUFLinear` between releases — 0.43.0 has
it under `diffusers.quantizers.gguf`, which is where the quantizer hook puts it,
so the oracle's own `n_gguf_linear` self-check is the thing to watch in the log
(`[gguf] N GGUFLinear modules active`).

## 3. Commands

### 3.1 Official CLI smoke test

```bash
cd /tmp/sn_oracle
/root/sn_oracle_venv/bin/python examples/t2i/inference.py \
  --model_path /root/models/SenseNova-U1.6-8B-MoT-GGUF/SenseNova-U1.6-8B-MoT-Q8_0.gguf \
  --model_format gguf \
  --prompt "A studio photograph of a red apple on a white table, soft daylight." \
  --width 512 --height 512 --num_steps 4 --seed 42 \
  --device cpu --dtype float32 --attn_backend sdpa \
  --output /root/oracle_out/smoke.png --profile
```

### 3.2 Reference tensors

```bash
/root/sn_oracle_venv/bin/python make_reference.py \
  --model_path /root/models/SenseNova-U1.6-8B-MoT-GGUF/SenseNova-U1.6-8B-MoT-Q8_0.gguf \
  --model_format gguf --device cpu --dtype float32 \
  --out_dir /root/oracle_out/ref512 \
  --verify-png /root/oracle_out/smoke.png
```

`--verify-png` diffs `make_reference.py`'s own `_to_pil` render against the CLI's
PNG (exact-match flag, max/mean abs u8 diff, % differing pixels) and records the
result in `manifest.json`. **That comparison is what proves the captured replay
is numerically the same as the official example** — without it the dump is just
"some other implementation".

### 3.3 Everything in one shot

```bash
bash run_reference.sh      # gguf header dump -> CLI smoke -> reference -> tar
```

## 4. The fixed reference contract

`make_reference.py` hardcodes `CONTRACT` (`REFERENCE_VERSION = 1`):

| field | value |
| --- | --- |
| prompt | `A studio photograph of a red apple on a white table, soft daylight.` |
| width × height | 512 × 512 |
| num_steps | 4 |
| seed | 42 |
| cfg_scale | 4.0 |
| cfg_norm | `none` |
| cfg_interval | (0.0, 1.0) |
| timestep_shift | 3.0 |
| t_eps | 0.02 (the `t2i_generate` default; the CLI never overrides it) |
| cfg_batching | True (CLI default) |
| attn_backend | `sdpa` |
| dtype | `float32` |

`make_reference.py` is a **line-for-line replay** of
`src/sensenova_u1/models/neo_unify/modeling_neo_chat.py::t2i_generate` with
`cap.put(...)` calls inserted at the boundaries a Rust implementation needs.
Same op order, same dtypes, same single RNG call. It is not a reimplementation
of the algorithm — that would defeat the purpose.

## 5. What gets dumped

Per-step tensors are `(1, token_h*token_w, 3*32*32) = (1, 256, 3072)` at
512×512 (3.15 MB each); the whole set is ~82 MB.

| file | shape | what it is |
| --- | --- | --- |
| `prompt_tokens.json` | — | **canonical** (doc §6): both query strings verbatim, both int64 id lists, `special_token_ids`, template/roles/sep |
| `input_ids_cond.npy` / `input_ids_uncond.npy` | `(1, L)` int64 | token ids of the conditional / unconditional query |
| `indexes_cond.npy` / `indexes_uncond.npy` | `(3, L)` int64 | `(t,h,w)` M-RoPE positions of the text prefix |
| `indexes_image_cond.npy` / `indexes_image_uncond.npy` | `(3, 256)` int64 | `(t,h,w)` positions of the image tokens (`t` constant = prefix len, `h = i//token_w`, `w = i%token_w`) |
| `noise_init.npy` | `(1,3,512,512)` f32 | `noise_scale * randn(...)` with `torch.Generator('cpu').manual_seed(42)`, `noise_scale = 2.0` |
| `timesteps.npy` / `timesteps_raw.npy` | `(5,)` f32 | shifted schedule / raw `linspace(0,1,5)` |
| `prefix_kv_layer{00,01,21,41}_{k,v}.npy` | `(1, 8, 64, 128)` f32 | prefix KV **after** the prefix forward, **before** `prepare_flash_kv_cache`, sliced to the first 64 of L sequence positions |
| `prefix_last_hidden_state_cond.npy` | `(1, L, 4096)` f32 | `language_model.model(...).last_hidden_state` for the conditional prefix (full) |
| `step{i}_t.npy`, `step{i}_t_next.npy` | scalar f32 | schedule entries |
| `step{i}_z.npy` | `(1,256,3072)` f32 | latent **before** the Euler step |
| `step{i}_v_pred_cond.npy`, `step{i}_v_pred_uncond.npy` | `(1,256,3072)` f32 | raw branch velocities (isolates CFG from the denoiser) |
| `step{i}_v_pred.npy` | `(1,256,3072)` f32 | after CFG combine |
| `step{i}_z_next.npy` | `(1,256,3072)` f32 | `z + (t_next - t) * v_pred` |
| `step{i}_v_denominator.npy` | scalar f32 | `(1-t).clamp_min(t_eps)`, the divisor in `predict_v` |
| `step0_image_embeds.npy` | `(1,256,4096)` f32 | denoiser input = `vision_mot_gen(...) + timestep_embedder(t) + noise_scale_embedder(noise_scale/16)` |
| `final_image_prediction.npy` | `(1,3,512,512)` f32 | the tensor `t2i_generate` returns |
| `final.png` | — | official `_to_pil` render |
| `state_dict_keys.json` | — | state_dict + named_parameters + GGUFLinear inventory |
| `manifest.json` | — | shapes/dtypes/slices/versions/timings/`--verify-png` verdict |

Useful identities recorded in the manifest:

* `z_next[i] ≡ z[i+1]` bit-for-bit (the oracle reuses one variable).
* `x_pred = v_pred * v_denominator + z` — the ConvDecoder output is exactly
  recoverable, so the head can be validated without an extra dump.
* `image_input = patchify(noise_init, 16, channel_first=True)` → `(1,1024,768)`.

## 6. Status

**Blocked.** The remote stopped accepting SSH at ~09:37 Asia/Shanghai, right
after the first smoke run was launched:

```
$ ssh -p 35879 root@36.150.116.206 'echo ALIVE'
ssh: connect to host 36.150.116.206 port 35879: Connection refused

$ nc -z 36.150.116.206 35879   ->  Connection refused
$ nc -z 36.150.116.206 22      ->  Connection refused
$ ping 36.150.116.206          ->  3/3 replies, ~65 ms
```

Host alive, no sshd anywhere. Per the `amd-cloud-gpu-test` skill this is the
Radeon Cloud **port rotation**: every Template restart gets a new forwarded
port on the same fixed IP, and the old one dies. The Template also wipes
`/root` and `/opt` unless a persistent volume is attached, so the 20.7 GB Q8_0
GGUF, `/root/sn_oracle_venv` and `/tmp/sn_oracle` may all be gone.

**To resume:** re-launch the Template, re-run
`.../SSH-Radeon-Cloud/.../start-ssh-server.sh` in the JupyterLab terminal, then
give the new `Port` to this workspace and run:

```bash
scp -P <newport> -i ~/.ssh/gouzi-remote tools/oracle/sensenova_u1/*.py \
    tools/oracle/sensenova_u1/*.sh root@36.150.116.206:/root/sn_scripts/
ssh -p <newport> -i ~/.ssh/gouzi-remote root@36.150.116.206 \
    'bash /root/sn_scripts/bootstrap_remote.sh && bash /root/sn_scripts/run_reference.sh'
```

## 7. Facts pinned offline (independent of the remote run)

Derived from upstream `d2d9bd1` source plus the pinned tokenizer files, so they
are already usable by the Rust port:

### 7.1 Tokenizer

`Qwen2Tokenizer` (BPE, `vocab.json` + `merges.txt`). Base vocab 151643 + 293
added tokens = **151936** = `config.vocab_size`; the added-token ids are exactly
151643..151935 with no gaps.

| token | id |
| --- | --- |
| `<|endoftext|>` (pad) | **151643** |
| `<|im_start|>` | **151644** |
| `<|im_end|>` (eos) | **151645** |
| `<IMG_CONTEXT>` | **151669** |
| `<img>` | **151670** |
| `</img>` | **151671** |

`bos_token = None`, `add_bos_token = False`, `split_special_tokens = False`,
`model_max_length = 12288`.

### 7.2 Chat template — `neo1_0`, hand-rolled, *not* the HF `chat_template`

`src/sensenova_u1/models/neo_unify/conversation.py`:

```python
system_template = '<|im_start|>system\n{system_message}'
roles           = ('<|im_start|>user\n', '<|im_start|>assistant\n')
sep_style       = SeparatorStyle.MPT
sep             = '<|im_end|>\n'
```

`get_prompt()` for MPT:

```python
ret = '' if system_prompt == '' else system_prompt + self.sep
for i, (role, message) in enumerate(self.messages):
    if message:
        ret += role + message + self.sep
    else:
        ret += role + self.sep if (i != len(self.messages) and message is not None) else role
```

so with `think_content = '<think>\n\n</think>\n\n<img>'`:

* **conditional** =
  `<|im_start|>system\n` + `SYSTEM_MESSAGE_FOR_GEN` + `<|im_end|>\n`
  + `<|im_start|>user\n` + prompt + `<|im_end|>\n`
  + `<|im_start|>assistant\n` + `<think>\n\n</think>\n\n<img>`
* **unconditional** =
  `<|im_start|>user\n<|im_end|>\n<|im_start|>assistant\n<img>`

> ⚠️ **The unconditional string is a real trap.** `_build_t2i_query("")` leaves
> `system_message = ''`, so `get_prompt()` emits **no system block at all** — but
> the *empty user message* still hits the `ret += role + self.sep` branch, so
> `<|im_start|>user\n<|im_end|>\n` is present. The rendering sketch in
> `docs/develop/SENSENOVA_U1.md` §5 shows a system block unconditionally, which
> would give a different token sequence. The oracle wins; `prompt_tokens.json`
> pins the truth.

### 7.3 Schedule and noise

`_apply_time_schedule` **hardcodes** `self.time_schedule = "standard"` before
branching, so the config's `time_schedule="dynamic"` and
`base_shift`/`max_shift`/`time_shift_type` are dead code in this release.
`standard` = `σ ← shift·σ / (1 + (shift-1)·σ)`, then `t ← 1 - σ`.

For `num_steps=4, shift=3.0`:

| i | linspace | σ=1-t | shifted σ | **t** |
| --- | --- | --- | --- | --- |
| 0 | 0.00 | 1.00 | 1.000 | **0.00** |
| 1 | 0.25 | 0.75 | 0.900 | **0.10** |
| 2 | 0.50 | 0.50 | 0.750 | **0.25** |
| 3 | 0.75 | 0.25 | 0.500 | **0.50** |
| 4 | 1.00 | 0.00 | 0.000 | **1.00** |

Note step 0 has `t = 0`, so `t >= cfg_interval[0]` is true and CFG *is* applied
on the first step.

`noise_scale`: `mode="resolution"`, `base_image_seq_len=64`, `noise_scale=1.0`,
`max_value=16.0`. At 512×512, `grid_h=grid_w=32`, `merge_size=2`, so
`scale = sqrt((32*32)/2²/64) = sqrt(1024/4/64) = sqrt(4) = 2.0` → `noise_scale = 2.0`, and
`x = 2.0 * randn((1,3,512,512), generator=torch.Generator('cpu').manual_seed(42))`
in **float32**.

### 7.4 Latent layout

`patchify(x, 32)` (channel-last, `nhwpqc`) → `(1, 256, 3072)`, i.e. 3·32·32
values per token in `(patch_h, patch_w, channel)` order.
`patchify(x, 16, channel_first=True)` (`nhwcpq`) → `(1, 1024, 768)` for the
vision tower, i.e. `(channel, patch_h, patch_w)`.

### 7.5 `predict_v`

```python
v_pred = (x_pred - z) / (1 - t).clamp_min(self.config.t_eps)   # t_eps = 0.02
```

with `x_pred` the `ConvDecoder` (pixel head) output; the LLM call uses
`update_cache=False` and `image_gen_indicators = ones(bool)` to route through
the `*_mot_gen` projections.

## 8. Files here

| file | purpose |
| --- | --- |
| `make_reference.py` | reference-tensor driver (captured replay of `t2i_generate`) |
| `run_reference.sh` | remote one-shot: gguf header → CLI smoke → reference → tar |
| `bootstrap_remote.sh` | remote one-shot: venv + oracle + deps, idempotent |
| `compare_naming.py` | GGUF tensor names vs oracle `state_dict()` 3-way diff |
| `gguf_dump.py` | dependency-free GGUF header/tensor dumper |
| `reference/` | the dumped artifacts (copied back from the remote) |
