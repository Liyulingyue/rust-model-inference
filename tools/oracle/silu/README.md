# SiLU oracle

A standalone C program that reproduces ggml's ARM `ggml_vec_silu_f32`
(`ggml/src/ggml-cpu/vec.h`) verbatim, so the repository's SiLU can be
compared against it **bit for bit**.

## Why this exists

ggml's vectorized SiLU does not use libm's `expf` but the polynomial
approximation in `ggml_v_expf`. The two differ by up to one ULP, and in the
Nemotron-3-Nano 30B SSM branch that one ULP is enough to flip an 8-bit
Q8_K activation level in the Q4_K projection that follows, which turns a
1e-7 relative difference into a 4e-4 one. So the polynomial has to be
right, not just close.

Two transcription details are easy to get wrong and both were:

* **`vfmaq_f32(a, b, c)` is `a + b*c`** (ACLE argument order), and
  `vfmsq_f32` likewise. Reading it as `a*b + c` inverts every FMA.
* **`vdupq_n_u32(0x7f000000)` is a u32 literal, not a float one.**
  `0x7f000000` reinterpreted as a float is a NaN pattern, so
  `vdupq_n_f32(0x7f000000)` would add `0x4EFE0000` (the rounding of
  2139095040.0f) instead. With the correct u32 literal `s1` is `0x01000000`
  (a denormal), so `exp(-inf)` comes out as exactly zero -- which is what
  the softmax needs for its masked (`-inf`) columns. Getting this wrong
  makes every masked column contribute a huge finite value instead.

## Build

```sh
cc -O2 -march=armv8.2-a+dotprod -o /tmp/opencode/silu_oracle \
    tools/oracle/silu/silu_oracle.c
```

## usage

```sh
silu_oracle <in.f32> <out.f32>
```

Both files hold `N` little-endian f32 values; the second receives ggml's
SiLU of the first.

## Cross-check

`tests/silu_ggml_reference.rs` builds the oracle on demand, feeds it a
deterministic corpus and compares the repository's `silu_ggml` output bit
for bit. It skips (rather than fails) when no C compiler is available or
the host is not ARM.
