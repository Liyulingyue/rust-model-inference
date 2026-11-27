# Q5_0 x Q8_0 dot-product oracle

A standalone C program that transcribes ggml's ARM NEON
`ggml_vec_dot_q5_0_q8_0` (`ggml/src/ggml-cpu/arch/arm/quants.c`) so the
repository's Q5_0 kernel can be compared against it **bit for bit**.

## Why this exists

The 30B's `ssm_in.weight` is Q5_0 while the 4B's is Q8_0. The 4B's SSM
branch matched the upstream oracle to 1 ULP while the 30B's diverged by
~1e-4, which pointed at the quantized-weight path rather than at the
Mamba-2 arithmetic. This tool is where that got pinned down.

The divergence turned out to be structural, not a rounding constant:

* ggml handles **two blocks per iteration** into two separate 4-lane F32
  accumulators (`sumv0` / `sumv1`), then reduces them with `vaddvq_f32`.
  A single scalar accumulator summing left-to-right is algebraically
  identical but rounds differently.
* the per-lane products are **integer dot products widened to F32** and
  accumulated with `vmlaq_n_f32` (an FMA), not integer sums multiplied by
  the scale afterwards.
* the high bit goes through ggml's `table_b2b_1` lookup, which yields
  `(!bit) << 4` so the `-16` bias is folded into the subtraction.

The per-lane grouping of the integer dot has to match too: ggml
accumulates each lane separately, and its own comment on the
`ggml_vdotq_s32` fallback says "different per-lane grouping -- do not use
when individual lane values matter". That is why both sides use the native
`vdotq_s32`, which requires `dotprod`.

## Build

```sh
cc -O2 -march=armv8.2-a+dotprod -o /tmp/opencode/q5_0_oracle \
    tools/oracle/q5_0/q5_0_oracle.c
```

`-march=armv8.2-a+dotprod` is not optional. The repository picks its kernel
at *runtime* (it detects `dotprod` and uses the native `vdotq_s32`), while
ggml picks at *compile* time; if the two run different instructions the
comparison is meaningless. `i8mm` is deliberately left off so ggml stays on
the NEON path the repository mirrors.

## usage

```sh
q5_0_oracle <weights.bin> <activations.bin> <out.bin> <n_out> <n_in>
```

| file | layout |
|---|---|
| `weights.bin` | `n_out` rows of `n_in` elements, Q5_0 (22 bytes / 32) |
| `activations.bin` | `n_in` Q8_0-quantized values (32 bytes / 32) plus the per-block F16 scales appended after the payload |
| `out.bin` | `n_out` F32 dot products |

## Cross-check

`tests/q5_0_reference.rs` builds the oracle on demand, feeds it a
deterministic corpus and compares the repository's kernel output bit for
bit. It skips (rather than fails) when no C compiler is available or the
host is not ARM.
