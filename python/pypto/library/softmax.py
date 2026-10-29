# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Compensated row Softmax with explicit FP16 input/output boundaries."""

from typing import Any

import pypto.language as pl


@pl.jit.inline
def _pair_add(
    ah: pl.Scalar[pl.FP32],
    al: pl.Scalar[pl.FP32],
    bh: pl.Scalar[pl.FP32],
    bl: pl.Scalar[pl.FP32],
) -> tuple[pl.Scalar[pl.FP32], pl.Scalar[pl.FP32]]:
    total = ah + bh
    v = total - ah
    error = (ah - (total - v)) + (bh - v)
    tail = error + (al + bl)
    hi = total + tail
    return hi, tail - (hi - total)


@pl.jit.inline
def _pair_multiply(
    ah: pl.Scalar[pl.FP32],
    al: pl.Scalar[pl.FP32],
    bh: pl.Scalar[pl.FP32],
    bl: pl.Scalar[pl.FP32],
) -> tuple[pl.Scalar[pl.FP32], pl.Scalar[pl.FP32]]:
    ca = 4097.0 * ah
    cb = 4097.0 * bh
    a_high = ca - (ca - ah)
    b_high = cb - (cb - bh)
    a_low = ah - a_high
    b_low = bh - b_high
    product = ah * bh
    error = ((a_high * b_high - product) + a_high * b_low + a_low * b_high) + a_low * b_low
    tail = error + (ah * bl + al * bh)
    hi = product + tail
    return hi, tail - (hi - product)


@pl.jit.inline
def _pair_exponential(x: pl.Scalar[pl.FP32]) -> tuple[pl.Scalar[pl.FP32], pl.Scalar[pl.FP32]]:
    high = pl.const(0.0, pl.FP32)
    low = pl.const(0.0, pl.FP32)
    # Row max subtraction makes x nonpositive. Terms below -80 cannot affect
    # FP16 normalization when the denominator contains at least exp(0) == 1.
    if x >= -80.0:
        n: pl.Scalar[pl.INT32] = pl.cast(x * 1.4426950408889634 - 0.5, pl.INT32)
        nf = pl.cast(-n, pl.FP32)
        mh, ml = _pair_multiply(nf, 0.0, 0.6931471824645996, -1.904654323148236e-9)
        rh, rl = _pair_add(x, 0.0, mh, ml)
        # Degree-13 Taylor coefficients retain each FP32 coefficient residual.
        ph = pl.const(1.6059044372074283e-10, pl.FP32)
        pt = pl.const(-5.3525265115627256e-18, pl.FP32)
        mh, ml = _pair_multiply(ph, pt, rh, rl)
        ph, pt = _pair_add(mh, ml, 2.0876755879584152e-09, 1.1082839809204342e-16)
        mh, ml = _pair_multiply(ph, pt, rh, rl)
        ph, pt = _pair_add(mh, ml, 2.5052107943679403e-08, 4.4176230446483665e-16)
        mh, ml = _pair_multiply(ph, pt, rh, rl)
        ph, pt = _pair_add(mh, ml, 2.755731998149713e-07, -7.5751122090511949e-15)
        mh, ml = _pair_multiply(ph, pt, rh, rl)
        ph, pt = _pair_add(mh, ml, 2.7557318844628753e-06, 3.7935712242972291e-14)
        mh, ml = _pair_multiply(ph, pt, rh, rl)
        ph, pt = _pair_add(mh, ml, 2.4801587642286904e-05, -3.4069960936668198e-13)
        mh, ml = _pair_multiply(ph, pt, rh, rl)
        ph, pt = _pair_add(mh, ml, 0.00019841270113829523, -2.7255968749334558e-12)
        mh, ml = _pair_multiply(ph, pt, rh, rl)
        ph, pt = _pair_add(mh, ml, 0.0013888889225199819, -3.3631094437103215e-11)
        mh, ml = _pair_multiply(ph, pt, rh, rl)
        ph, pt = _pair_add(mh, ml, 0.008333333767950535, -4.3461720333759502e-10)
        mh, ml = _pair_multiply(ph, pt, rh, rl)
        ph, pt = _pair_add(mh, ml, 0.0416666679084301, -1.2417634698280722e-09)
        mh, ml = _pair_multiply(ph, pt, rh, rl)
        ph, pt = _pair_add(mh, ml, 0.1666666716337204, -4.9670538793122887e-09)
        mh, ml = _pair_multiply(ph, pt, rh, rl)
        ph, pt = _pair_add(mh, ml, 0.5, 0.0)
        mh, ml = _pair_multiply(ph, pt, rh, rl)
        ph, pt = _pair_add(mh, ml, 1.0, 0.0)
        mh, ml = _pair_multiply(ph, pt, rh, rl)
        ph, pt = _pair_add(mh, ml, 1.0, 0.0)
        scale = pl.const(1.0, pl.FP32)
        for _step in pl.range(0, -n):
            scale = scale * 0.5
        high = ph * scale
        low = pt * scale
    return high, low


def fp16_softmax_row(columns: int) -> Any:
    """Build an inline compensated Softmax over one finite FP16 row.

    Args:
        columns: Positive multiple of sixteen, at most 1024.

    Returns:
        Public JIT inline function from FP16 ``[1, columns]`` to the same
        shape. Compensation preserves exponential and normalization residuals
        before the final FP16 conversion; it is more costly than vector exp.

    Raises:
        ValueError: The row does not satisfy the static aligned profile.
    """
    if type(columns) is not int or columns <= 0 or columns % 16 or columns > 1024:
        raise ValueError(f"FP16 compensated Softmax requires 16..1024 aligned columns, got {columns!r}")

    @pl.jit.inline
    def softmax(scores: pl.Tile[[1, columns], pl.FP16]) -> pl.Tile[[1, columns], pl.FP16]:
        values = pl.cast(scores, pl.FP32, mode="none")
        maximum = pl.tile.read(values, [0, 0])
        for col in pl.range(1, columns):
            current = pl.tile.read(values, [0, col])
            if current > maximum:
                # Keep the assignment in the scalar DSL: Python max and
                # conditional expressions are not supported parser operations.
                maximum = pl.cast(current, pl.FP32)
        exp_high = pl.tile.create([1, columns], pl.FP32)
        exp_low = pl.tile.create([1, columns], pl.FP32)
        total_high = pl.const(0.0, pl.FP32)
        total_low = pl.const(0.0, pl.FP32)
        for col in pl.range(0, columns):
            current = pl.tile.read(values, [0, col])
            eh, el = _pair_exponential(current - maximum)
            pl.tile.write(exp_high, [0, col], eh)
            pl.tile.write(exp_low, [0, col], el)
            total_high, total_low = _pair_add(total_high, total_low, eh, el)
        for col in pl.range(0, columns):
            eh = pl.tile.read(exp_high, [0, col])
            el = pl.tile.read(exp_low, [0, col])
            quotient = eh / total_high
            mh, ml = _pair_multiply(total_high, total_low, -quotient, 0.0)
            rh, rl = _pair_add(eh, el, mh, ml)
            nh, nl = _pair_add(quotient, 0.0, (rh + rl) / total_high, 0.0)
            pl.tile.write(values, [0, col], nh + nl)
        return pl.cast(values, pl.FP16, mode="none")

    return softmax


def fp32_softmax_tile(rows: int, columns: int) -> Any:
    """Build inline FP32 row Softmax with vector exp and explicit scratch.

    Args:
        rows: Positive static row count.
        columns: Positive multiple of eight, satisfying the selected backend's
            tile and memory limits. No workspace is written to global memory.

    Returns:
        Inline function from an FP32 tile to the same shape. Unlike the
        compensated FP16 variant, vector exp and reductions use FP32 precision.
    """
    if type(rows) is not int or rows <= 0 or type(columns) is not int or columns <= 0 or columns % 8:
        raise ValueError(f"FP32 Softmax requires positive rows and aligned columns, got {(rows, columns)!r}")

    @pl.jit.inline
    def softmax(value: pl.Tile[[rows, columns], pl.FP32]) -> pl.Tile[[rows, columns], pl.FP32]:
        max_scratch = pl.tile.create([rows, columns], pl.FP32)
        maximum = pl.tile.row_max(value, max_scratch)
        shifted = pl.sub(value, pl.tile.row_expand(value, maximum))
        exponent = pl.exp(shifted)
        sum_scratch = pl.tile.create([rows, columns], pl.FP32)
        total = pl.tile.row_sum(exponent, sum_scratch)
        return pl.div(exponent, pl.tile.row_expand(exponent, total))

    return softmax
