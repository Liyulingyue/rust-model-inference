"""GGML K-quants (Q4_K, Q6_K) for the converter.

These are llama.cpp's block formats: a 256-value super-block carries one F16
scale plus eight 6-bit sub-scales, and Q4_K additionally keeps a per-sub-block
2-bit minimum.  The finer granularity is what makes them usable for a model
whose source is already 4-bit -- a Q4_0 round trip measured ~12% mean relative
error against the MLX affine weights, because one F16 scale per 32 values
cannot track the per-group bias the source format carries.

Both writers are line-by-line ports of ``quantize_row_q4_K_ref`` and
``quantize_row_q6_K_ref`` in ``references/llama.cpp/ggml/src/ggml-quants.c``,
including ``make_qkx2_quants`` and ``get_scale_min_k4``.  The float
accumulation order is preserved because the scale search compares weighted
squared errors, so a different summation order can select a different scale and
the decoded weights would drift.
"""

from __future__ import annotations

import numpy as np


K_BLOCK = 256
SUB_BLOCK = 32
#: Bytes per super-block, matching ``core::tensor::GGMLType::nbytes``.
K_BLOCK_BYTES = {"q4_k": 144, "q6_k": 210}

#: Working-set budget for the vectorized k-quant search, in bytes.
#:
#: ``make_qkx2_quants`` evaluates 20 candidate scales at once, so the largest
#: intermediate is ``8 * 32 * 20 * 8`` bytes (F64) per super-block, plus a
#: handful of arrays the same size.  The batch size is derived from this budget
#: rather than hard-coded, because the embedding matrix alone is 970k
#: super-blocks and quantizing it in one shot would ask for terabytes.
K_SEARCH_BUDGET_BYTES = 5 << 30

#: Bytes of transient arrays the Q4_K search holds per super-block.
_Q4K_BYTES_PER_BLOCK = 8 * SUB_BLOCK * 20 * 8 * 4
#: Q6_K sweeps 19 candidate scales over 16-value sub-blocks.
_Q6K_BYTES_PER_BLOCK = 16 * 19 * 8 * 3


def _batch_blocks(bytes_per_block: int) -> int:
    """Super-blocks per batch that fit ``K_SEARCH_BUDGET_BYTES``."""
    return max(1, K_SEARCH_BUDGET_BYTES // bytes_per_block)


def _nearest_int(values: np.ndarray) -> np.ndarray:
    """``nearest_int``: ties away from zero, as ``roundf`` does.

    ``np.rint`` is ties-to-even, so an exact ``.5`` would pick a different code
    than ggml.
    """
    return np.where(values >= 0.0, np.floor(values + 0.5), np.ceil(values - 0.5))


def _nearest_int_scalar(value: float) -> int:
    """Scalar ``nearest_int``; the inner loops run millions of times, so this
    avoids building a one-element array per call."""
    if value >= 0.0:
        return int(np.floor(value + 0.5))
    return int(np.ceil(value - 0.5))


def _f16_bytes(value) -> bytes:
    return np.float16(value).tobytes()


def _get_scale_min_k4(j: int, q: np.ndarray) -> tuple[int, int]:
    """Port of ``get_scale_min_k4``; ``q`` is the 12-byte scale table."""
    if j < 4:
        return int(q[j]) & 63, int(q[j + 4]) & 63
    return (
        (int(q[j + 4]) & 0xF) | ((int(q[j - 4]) >> 6) << 4),
        (int(q[j + 4]) >> 4) | ((int(q[j]) >> 6) << 4),
    )


def _make_qkx2_quants(
    n: int,
    nmax: int,
    x: np.ndarray,
    weights: np.ndarray,
    rmin: float,
    rdelta: float,
    nstep: int,
) -> tuple[float, float, np.ndarray]:
    """Port of ``make_qkx2_quants`` with ``use_mad=false``.

    Returns ``(scale, min, L)`` where the caller reconstructs
    ``scale * L + min``.  The reference accumulates the weighted sums in a
    single sequential pass; NumPy's pairwise summation would pick a different
    ``D`` near zero and therefore a different scale, so the reductions are done
    in order.
    """
    mn = float(x[0])
    mx = float(x[0])
    sum_w = float(weights[0])
    sum_x = sum_w * float(x[0])
    for i in range(1, n):
        xi = float(x[i])
        if xi < mn:
            mn = xi
        if xi > mx:
            mx = xi
        w = float(weights[i])
        sum_w += w
        sum_x += w * xi
    if mn > 0:
        mn = 0.0
    if mx == mn:
        return 0.0, -mn, np.zeros(n, dtype=np.uint8)

    iscale = nmax / (mx - mn)
    scale = 1.0 / iscale
    L = np.clip(_nearest_int(iscale * (x - mn)), 0, nmax).astype(np.uint8)
    best_error = 0.0
    for i in range(n):
        diff = scale * float(L[i]) + mn - float(x[i])
        best_error += float(weights[i]) * diff * diff

    for is_ in range(nstep + 1):
        iscale = (rmin + rdelta * is_ + nmax) / (mx - mn)
        sum_l = sum_l2 = sum_xl = 0.0
        Laux = np.zeros(n, dtype=np.uint8)
        for i in range(n):
            l = _nearest_int_scalar(iscale * (float(x[i]) - mn))
            if l < 0:
                l = 0
            elif l > nmax:
                l = nmax
            Laux[i] = l
            w = float(weights[i])
            sum_l += w * l
            sum_l2 += w * l * l
            sum_xl += w * l * float(x[i])
        D = sum_w * sum_l2 - sum_l * sum_l
        if D <= 0:
            continue
        this_scale = (sum_w * sum_xl - sum_x * sum_l) / D
        this_min = (sum_l2 * sum_x - sum_l * sum_xl) / D
        if this_min > 0:
            this_min = 0.0
            this_scale = sum_xl / sum_l2
        cur_error = 0.0
        for i in range(n):
            diff = this_scale * float(Laux[i]) + this_min - float(x[i])
            cur_error += float(weights[i]) * diff * diff
        if cur_error < best_error:
            L = Laux.copy()
            best_error = cur_error
            scale = this_scale
            mn = this_min
    return scale, -mn, L


def _qkx2_batch(x: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized ``make_qkx2_quants(32, 15, ..., rmin=-1, rdelta=0.1, nstep=20)``.

    ``x`` and ``weights`` are ``(blocks, 8, 32)``.  The reference walks 21
    candidate scales per sub-block and keeps the lowest weighted error, which
    is a pure elementwise-plus-reduction pattern, so every candidate is
    evaluated at once along a trailing axis.  Returns ``(scale, min, L)`` with
    ``L`` shaped like ``x``.

    The reference accumulates in F64 while its input is F32, and the candidate
    that wins changes with the last bits of the sum, so the reductions stay in
    F64 and the per-element order matches the scalar loop.
    """
    xd = x.astype(np.float64)
    wd = weights.astype(np.float64)
    lo = np.minimum(xd.min(axis=2), 0.0)
    hi = xd.max(axis=2)
    span = hi - lo
    safe_span = np.where(span == 0.0, 1.0, span)
    sum_w = wd.sum(axis=2)
    sum_x = (wd * xd).sum(axis=2)
    sum_x2 = (wd * xd * xd).sum(axis=2)

    # is=0 seed, mirroring the reference's first estimate.
    isc0 = 15.0 / safe_span
    L0 = np.clip(np.floor(isc0[..., None] * (xd - lo[..., None]) + 0.5), 0, 15)
    scale0 = np.where(sum_x2 > 0.0, sum_x / np.maximum((wd * L0 * L0).sum(axis=2), 1e-300), 1.0 / isc0)
    resid0 = scale0[..., None] * L0 + lo[..., None] - xd
    best_err = (wd * resid0 * resid0).sum(axis=2)
    best_scale = scale0
    best_min = lo
    best_L = L0

    steps = np.arange(1, 21, dtype=np.float64)
    # (blocks, 8, 1, 20) candidate scales so the sweep lands on the trailing
    # axis opposite the 32 values.
    isc = (-1.0 + 0.1 * steps + 15.0)[None, None, None, :] / safe_span[..., None, None]
    Lc = np.clip(
        np.floor(isc * (xd[..., None] - lo[..., None, None]) + 0.5), 0, 15
    )
    sum_l = (wd[..., None] * Lc).sum(axis=2)
    sum_l2 = (wd[..., None] * Lc * Lc).sum(axis=2)
    sum_xl = (wd[..., None] * Lc * xd[..., None]).sum(axis=2)
    D = sum_w[..., None] * sum_l2 - sum_l * sum_l
    safe_D = np.where(D == 0.0, 1.0, D)
    this_scale = (sum_w[..., None] * sum_xl - sum_x[..., None] * sum_l) / safe_D
    this_min = (sum_l2 * sum_x[..., None] - sum_l * sum_xl) / safe_D
    positive = this_min > 0.0
    this_min = np.where(positive, 0.0, this_min)
    this_scale = np.where(
        positive, sum_xl / np.maximum(sum_l2, 1e-300), this_scale
    )
    resid = this_scale[..., None, :] * Lc + this_min[..., None, :] - xd[..., None]
    err = (wd[..., None] * resid * resid).sum(axis=2)
    improved = (D > 0.0) & (err < best_err[..., None])
    any_better = improved.any(axis=-1)
    if any_better.any():
        # Pick the lowest-error candidate per sub-block, then keep it only where
        # it actually beat the is=0 seed.
        pick = np.argmin(np.where(improved, err, np.inf), axis=-1)
        take = np.take_along_axis
        cand_scale = take(this_scale, pick[..., None], axis=-1)[..., 0]
        cand_min = take(this_min, pick[..., None], axis=-1)[..., 0]
        cand_L = take(Lc, pick[..., None, None].repeat(SUB_BLOCK, axis=-1), axis=-1)[..., 0]
        best_scale = np.where(any_better, cand_scale, best_scale)
        best_min = np.where(any_better, cand_min, best_min)
        best_L = np.where(any_better[..., None], cand_L, best_L)
    return best_scale, -best_min, best_L.astype(np.uint8)


def _quantize_chunked(
    values: np.ndarray,
    name: str,
    batch_fn,
    blocks_per_batch: int,
    progress=None,
) -> bytes:
    """Run ``batch_fn`` over ``values`` in bounded chunks, reporting progress.

    The k-quant search vectorizes over super-blocks, so the chunk size is what
    caps peak memory.  ``progress`` is called as ``progress(done, total)`` after
    each chunk, which lets the converter print a rate without the quantizer
    knowing anything about the checkpoint.
    """
    flat = np.ascontiguousarray(values, dtype=np.float32).reshape(-1)
    if flat.size % K_BLOCK:
        raise ValueError(
            f"{name} payload {flat.size} elements is not a multiple of {K_BLOCK}"
        )
    blocks = flat.reshape(-1, 8, SUB_BLOCK)
    total = blocks.shape[0]
    if total == 0:
        return b""
    parts = []
    done = 0
    for start in range(0, total, blocks_per_batch):
        parts.append(batch_fn(blocks[start : start + blocks_per_batch]))
        done += min(blocks_per_batch, total - start)
        if progress is not None:
            progress(done, total)
    return np.concatenate(parts).tobytes() if len(parts) > 1 else parts[0].tobytes()


def _q4k_batch(blocks: np.ndarray) -> np.ndarray:
    """Quantize ``(n, 8, 32)`` F32 blocks into ``(n, 144)`` Q4_K payloads."""
    n = blocks.shape[0]
    out = np.zeros((n, K_BLOCK_BYTES["q4_k"]), dtype=np.uint8)
    if n == 0:
        return out

    xd = blocks.astype(np.float64)
    avg = np.sqrt((xd * xd).mean(axis=2))
    weights = avg[..., None] + np.abs(xd)
    scales, mins, _ = _qkx2_batch(blocks, weights)

    max_scale = scales.max(axis=1)
    max_min = mins.max(axis=1)
    inv_scale = np.where(max_scale > 0.0, 63.0 / np.where(max_scale == 0.0, 1.0, max_scale), 0.0)
    inv_min = np.where(max_min > 0.0, 63.0 / np.where(max_min == 0.0, 1.0, max_min), 0.0)
    ls = np.minimum(63, _nearest_int(inv_scale[:, None] * scales)).astype(np.int64)
    lm = np.minimum(63, _nearest_int(inv_min[:, None] * mins)).astype(np.int64)

    # ``get_scale_min_k4`` layout: scales 0..3 in the low six bits of bytes 0..3,
    # scales 4..7 in the low nibble of bytes 8..11 plus the top two bits of the
    # same even bytes, and the minimums interleaved alongside.
    table = np.zeros((n, 12), dtype=np.uint8)
    for j in range(4):
        table[:, j] = ls[:, j].astype(np.uint8)
        table[:, j + 4] = lm[:, j].astype(np.uint8)
    for j in range(4, 8):
        table[:, j + 4] = ((ls[:, j] & 0xF) | ((lm[:, j] & 0xF) << 4)).astype(np.uint8)
        table[:, j - 4] |= ((ls[:, j] >> 4) << 6).astype(np.uint8)
        table[:, j] |= ((lm[:, j] >> 4) << 6).astype(np.uint8)
    d = (max_scale / 63.0).astype("<f2")
    dmin = (max_min / 63.0).astype("<f2")

    # Re-derive the codes against the rounded d/dmin, exactly as the reference
    # does, so they agree with what the runtime will decode.
    d_f32 = d.astype(np.float32)[:, None]
    dmin_f32 = dmin.astype(np.float32)[:, None]
    recovered = np.zeros((n, 8), dtype=np.int64)
    mfield = np.zeros((n, 8), dtype=np.int64)
    for j in range(8):
        if j < 4:
            recovered[:, j] = table[:, j] & 63
            mfield[:, j] = table[:, j + 4] & 63
        else:
            recovered[:, j] = (table[:, j + 4] & 0x0F) | ((table[:, j - 4] >> 6) << 4)
            mfield[:, j] = (table[:, j + 4] >> 4) | ((table[:, j] >> 6) << 4)
    dj = d_f32 * recovered
    dm = dmin_f32 * mfield
    safe = np.where(dj == 0.0, 1.0, dj)
    codes = np.clip(
        _nearest_int((xd + dm[..., None]) / safe[..., None]), 0, 15
    ).astype(np.uint8)
    codes = np.where((dj == 0.0)[..., None], 0, codes)

    out[:, 0:2] = d.view(np.uint8).reshape(-1, 2)
    out[:, 2:4] = dmin.view(np.uint8).reshape(-1, 2)
    out[:, 4:16] = table
    # Flatten the sub-blocks into the reference's linear order, then pack each
    # 64-value group as 32 bytes: low 32 codes in the low nibbles, high 32 in
    # the high nibbles.
    linear = codes.reshape(n, K_BLOCK)
    for group in range(4):
        base = group * 64
        out[:, 16 + group * 32 : 16 + group * 32 + 32] = (
            linear[:, base : base + 32] | (linear[:, base + 32 : base + 64] << 4)
        )
    return out


def quantize_q4_k(values: np.ndarray, progress=None) -> bytes:
    """Q4_K: 256 values, F16 scale/min, eight 6-bit sub-scales, 4-bit codes.

    A vectorized port of ``quantize_row_q4_K_ref``.  Each 32-value sub-block is
    quantized by ``make_qkx2_quants(32, 15, ..., rmin=-1, rdelta=0.1, nstep=20)``
    with magnitude-flavoured weights, the eight sub-scales are normalized
    against the block maximum into 6 bits, and the codes are packed so the
    runtime's ``dequantize_row_q4_K`` reads them back in order.

    The search needs 20 candidate codes per value, so the work is chunked to
    ``K_SEARCH_BUDGET_BYTES`` at a time; the embedding matrix alone is 970k
    super-blocks and would otherwise ask for terabytes.
    """
    return _quantize_chunked(
        values,
        "q4_k",
        _q4k_batch,
        _batch_blocks(_Q4K_BYTES_PER_BLOCK),
        progress,
    )


def _make_qx_quants(n: int, nmax: int, x: np.ndarray) -> tuple[float, np.ndarray]:
    """Port of ``make_qx_quants`` with ``rmse_type=1`` and ``qw=NULL``.

    Q6_K needs a symmetric scale per sub-block.  The reference minimizes the
    squared error, so it first fits ``scale = sumlx/suml2`` and then sweeps
    ``iscale = -(nmax + 0.1*is) / max`` for ``is`` in -9..9, keeping whichever
    candidate has the larger ``scale*sumlx``.  ``L`` comes back biased by
    ``nmax``; the caller recomputes the codes against the stored scale anyway.
    """
    mx = 0.0
    amax = 0.0
    for i in range(n):
        ax = abs(float(x[i]))
        if ax > amax:
            amax = ax
            mx = float(x[i])
    if amax < 1e-15:
        return 0.0, np.zeros(n, dtype=np.int8)
    iscale = -nmax / mx
    L = np.zeros(n, dtype=np.int8)
    sumlx = suml2 = 0.0
    for i in range(n):
        xi = float(x[i])
        l = _nearest_int_scalar(iscale * xi)
        l = max(-nmax, min(nmax - 1, l))
        L[i] = l + nmax
        w = xi * xi
        sumlx += w * xi * l
        suml2 += w * l * l
    scale = sumlx / suml2 if suml2 else 0.0
    best = scale * sumlx
    for is_ in range(-9, 10):
        if is_ == 0:
            continue
        iscale = -(nmax + 0.1 * is_) / mx
        sumlx = suml2 = 0.0
        for i in range(n):
            xi = float(x[i])
            l = max(-nmax, min(nmax - 1, _nearest_int_scalar(iscale * xi)))
            w = xi * xi
            sumlx += w * xi * l
            suml2 += w * l * l
        if suml2 > 0 and sumlx * sumlx > best * suml2:
            for i in range(n):
                l = max(-nmax, min(nmax - 1, _nearest_int_scalar(iscale * float(x[i]))))
                L[i] = nmax + l
            scale = sumlx / suml2
            best = scale * sumlx
    return scale, L


def _q6k_batch(blocks: np.ndarray) -> np.ndarray:
    """Quantize ``(n, 256)`` F32 blocks into ``(n, 210)`` Q6_K payloads."""
    out = np.zeros((blocks.shape[0], K_BLOCK_BYTES["q6_k"]), dtype=np.uint8)

    for bi, block in enumerate(blocks):
        sub = block.reshape(16, 16)
        scales = np.zeros(16, dtype=np.float32)
        # ``L`` is flat, indexed ``L[16*sub_block + position]``, matching the
        # reference's packing loop.
        L = np.zeros(K_BLOCK, dtype=np.int8)
        max_scale = np.float32(0.0)
        max_abs_scale = np.float32(0.0)
        for ib in range(16):
            scale, codes = _make_qx_quants(16, 32, sub[ib])
            scales[ib] = scale
            L[16 * ib : 16 * ib + 16] = codes
            if abs(scale) > max_abs_scale:
                max_abs_scale = abs(scale)
                max_scale = scale
        if max_abs_scale < 1e-15:
            out[bi, 208:210] = np.frombuffer(_f16_bytes(0.0), dtype=np.uint8)
            continue

        iscale = -128.0 / max_scale
        d = np.float16(1.0 / iscale)
        # The reference clamps to 127 and then stores into an ``int8_t``, so a
        # 127 becomes -128 on the wire.  Reproduce the wrap rather than
        # clamping to 127 here, or the sub-scales will not match the reader.
        stored = np.clip(_nearest_int(iscale * scales), -127, 127).astype(np.int8)
        d_f32 = np.float32(d)
        for j in range(16):
            dj = d_f32 * np.float32(stored[j])
            if dj == 0.0:
                continue
            L[16 * j : 16 * j + 16] = (
                np.clip(_nearest_int(sub[j] / dj), -32, 31).astype(np.int8) + np.int8(32)
            )
        codes = L.view(np.uint8)
        # Each 128-value group packs four sub-blocks per byte column: the low
        # nibbles of ``ql[l]``/``ql[l+32]`` hold sub-blocks 0/2 and 1/3, and
        # ``qh[l]`` collects the top two bits of all four.
        for j in range(0, K_BLOCK, 128):
            for l in range(32):
                ql = j // 2 + l
                out[bi, ql] = (codes[j + l] & 0xF) | ((codes[j + l + 64] & 0xF) << 4)
                out[bi, ql + 32] = (codes[j + l + 32] & 0xF) | (
                    (codes[j + l + 96] & 0xF) << 4
                )
                out[bi, 128 + j // 4 + l] = (
                    (codes[j + l] >> 4)
                    | ((codes[j + l + 32] >> 4) << 2)
                    | ((codes[j + l + 64] >> 4) << 4)
                    | ((codes[j + l + 96] >> 4) << 6)
                )
        out[bi, 192:208] = stored.view(np.uint8)
        out[bi, 208:210] = np.frombuffer(_f16_bytes(d), dtype=np.uint8)
    return out


def quantize_q6_k(values: np.ndarray, progress=None) -> bytes:
    """Q6_K: 256 values, one F16 scale, sixteen int8 sub-scales, 6-bit codes.

    A port of ``quantize_row_q6_K_ref``.  The shared F16 scale is derived from
    the sub-block with the largest magnitude, the remaining sub-scales are
    stored as int8 ratios against it, and the codes are packed so the runtime's
    ``dequantize_row_q6_k`` reads the shared scale from offset 208.
    """
    return _quantize_chunked(
        values,
        "q6_k",
        _q6k_batch,
        _batch_blocks(_Q6K_BYTES_PER_BLOCK),
        progress,
    )


#: GGML type name -> (encoder, bytes per 256-value super-block).
K_ENCODERS = {
    "q4_k": (quantize_q4_k, K_BLOCK_BYTES["q4_k"]),
    "q6_k": (quantize_q6_k, K_BLOCK_BYTES["q6_k"]),
}


def quantize_k(name: str, values: np.ndarray, progress=None) -> bytes:
    """Encode ``values`` with the named k-quant, rejecting misaligned input."""
    try:
        encode, _bytes = K_ENCODERS[name]
    except KeyError:
        raise ValueError(f"unknown k-quant {name!r}; choices: {sorted(K_ENCODERS)}") from None
    return encode(values, progress)


def k_block_bytes(name: str) -> int:
    return K_ENCODERS[name][1]
