# YuE2 VAE decode: what is and is not the bottleneck

The decode is 155 s of a ~537 s render (29%), second only to the NAR solve.
The instinct is to go after it, so this file records the five hypotheses that
were measured and ruled out, and the one thing that was never explained.

## Shape of the decoder

Six decoder blocks, widths 2048/1024/512/256/128/64, upsampling 64 -> 921568
samples, latent 960 frames -> 38.4 s of 48 kHz stereo audio. Every residual is
**two** convolutions, not one:

```
block1  conv in=1024 out=1024 k=7 dil=1 pad=3   frames=5760 length=5760
        conv in=1024 out=1024 k=1 dil=1 pad=0   frames=5760 length=5760
        conv in=1024 out=1024 k=7 dil=3 pad=9
        conv in=1024 out=1024 k=1
        conv in=1024 out=1024 k=7 dil=9 pad=27
        conv in=1024 out=1024 k=1
```

Measured split, 20 threads, 960 frames (see `YUE2_VAE_TIMING=1`):

| stage | time | share |
| --- | --- | --- |
| residuals | 120.4 s | 77% |
| up convs | 33.9 s | 22% |
| snake activations | 2.4 s | 1.5% |
| final snake | 1.9 s | |

## Ruled out

Every number below came from a standalone probe replicating the shipped
access pattern, not from a model run.

1. **Serial snake activations are not a tail worth parallelizing.** 2.4 s of
   157 s. Each block's snake runs *before* its upsample, so it still sees the
   latent-sized tensor; the audio-sized work happens in the convs.

2. **Thread contention.** Scaling is near linear: 1 -> 20 threads gives
   3.8 -> 53.8 GOP/s on the k=7 conv, 81% of ideal. There is no cache
   thrashing to reclaim.

3. **ic blocking.** Counter to the expectation, larger blocks are strictly
   faster: 18.8 GOP/s at block_ic=4 rising to 185.9 at block_ic=256 (i.e. no
   blocking). The chunking only adds loop overhead.

4. **Weight indexing style.** Slicing a per-channel view before the inner
   loop versus indexing the full array with `(oc*input + ic)*kernel + tap`:
   53.7 versus 52.9 GOP/s.

5. **Dilation.** The decoder uses dilation [1, 3, 9] per block, but 53.4 /
   53.1 / 50.7 GOP/s across the three -- the non-unit tap spacing costs
   nothing measurable.

6. **The `k % 2048` modulus in the conv's accumulator-flush branch.** A
   microbenchmark says it is genuinely expensive: 0.47 s versus 0.10 s over
   2e8 iterations, so 4.7x on the arithmetic it guards. Hoisting it out of the
   inner loop and hoisting the loop-invariant branch condition changed the
   decode from 155.5 s to 150.4 s, and the block-1 conv from 3.25 s to
   3.40 s -- i.e. nothing, inside the run-to-run noise of this machine. The
   change was reverted rather than kept as complexity that buys nothing.

## The pointwise conv is the slowest per FLOP, and still does not matter

The `k=1` conv at the end of every residual runs at 28.0 GOP/s against the
k=7 conv's 53.8, because `for time { for ic { input[ic*len + time] } }` has no
tap reuse at all: each output walks `ch` locations that are `len` floats
apart, one cache line each. But it is 12.1 GFLOP against the k=7 conv's 84.6,
so the three pointwise convs in a block are ~12% of the block's FLOP. Driving
them to zero would save roughly 1.3 s per block.

## What is unexplained

The probe predicts block 1's residuals at 6.1 s; the shipped code takes
12.2 s. Half the cost is still unattributed. The two agree on shape, dilation,
channel count, thread count and weight-indexing style, so the gap is in
something the probe does not model. Until that gap is closed, any proposed
change is a guess.

## Method note

This file exists because every hypothesis above was a guess that measurement
rejected, several of them after a code change was already written. The
sequence that worked was: measure the shape, build a probe that replicates the
real access pattern, sweep one variable, and only then touch the kernel. An
earlier version of this investigation also miscounted FLOPs by dropping a
dimension and concluded that wide blocks were 15x slower than narrow ones --
they are all within 10% of each other at ~24 GOP/s.
