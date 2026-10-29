# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Public finite-FP16 SiLU with separate sigmoid/product rounding."""

from typing import Any

import pypto.language as pl


def fp16_silu_constants(rows: int, cols: int = 16) -> Any:
    """Return inline SiLU with caller-owned zero/offset/scale/one/epsilon tiles.

    Constants are FP32 tiles with values 0, 15/2048, 8192, 1 and 2^-23.
    Keeping them outside a caller's work loop avoids repeated vector fills.

    Args:
        rows: Static row count. N16 supports 64, 128, 160 or 256 rows;
            N32 supports 128 rows and N80 supports 1 or 13 rows.
        cols: Static column count, 16 (default), 32 or 80.

    Returns:
        Inline function taking an FP16 ``[rows, cols]`` tile and the five
        same-shaped FP32 constants. The sigmoid is rounded to FP16 before
        multiplication, preserving separate sigmoid/product rounding.

    Raises:
        ValueError: The row count is outside the supported static profile.
    """
    if (
        type(rows) is not int
        or type(cols) is not int
        or (rows, cols) not in ((64, 16), (128, 16), (160, 16), (256, 16), (128, 32), (1, 80), (13, 80))
    ):
        raise ValueError("FP16 SiLU requires a supported static N16, N32 or N80 tile")

    @pl.jit.inline
    def activate(
        value: pl.Tile[[rows, cols], pl.FP16],
        zero: pl.Tile[[rows, cols], pl.FP32],
        offset: pl.Tile[[rows, cols], pl.FP32],
        scale: pl.Tile[[rows, cols], pl.FP32],
        one: pl.Tile[[rows, cols], pl.FP32],
        epsilon: pl.Tile[[rows, cols], pl.FP32],
    ) -> pl.Tile[[rows, cols], pl.FP16]:
        x = pl.cast(value, pl.FP32, mode="none")
        negative = pl.sub(zero, x)
        exponent = pl.exp(negative)
        # Nonzero FP16 distance from -15/2048 is at least 2^-18,
        # so 8192 * distance^2 is at least the 2^-23 correction.
        delta = pl.add(x, offset)
        squared = pl.mul(delta, delta)
        correction = pl.tile.maximum(zero, pl.sub(epsilon, pl.mul(squared, scale)))
        corrected = pl.add(exponent, correction)
        sigmoid = pl.div(one, pl.add(corrected, one))
        sigmoid_half = pl.cast(sigmoid, pl.FP16, mode="none")
        # Two finite FP16 factors have an exact FP32 product; direct FP16
        # multiplication retains the same final rounding and signed zeros.
        return pl.mul(value, sigmoid_half)

    return activate


def fp16_silu(rows: int, cols: int = 16) -> Any:
    """Build finite FP16 SiLU with separate sigmoid/product rounding.

    Args:
        rows: Static row count. N16 supports 64, 128, 160 or 256 rows;
            N32 supports 128 rows and N80 supports 1 or 13 rows.
        cols: Static column count, 16 (default), 32 or 80.

    Returns:
        Inline function from an FP16 ``[rows, cols]`` tile to the same shape,
        allocating local constants. Use ``fp16_silu_constants`` to retain
        those constants across a caller-owned work loop.

    Raises:
        ValueError: The row count is outside the supported static profile.
    """
    activate_core = fp16_silu_constants(rows, cols)

    @pl.jit.inline
    def activate(value: pl.Tile[[rows, cols], pl.FP16]) -> pl.Tile[[rows, cols], pl.FP16]:
        zero = pl.tile.full([rows, cols], pl.FP32, 0.0)
        offset = pl.tile.full([rows, cols], pl.FP32, 15.0 / 2048)
        scale = pl.tile.full([rows, cols], pl.FP32, 8192.0)
        one = pl.tile.full([rows, cols], pl.FP32, 1.0)
        epsilon = pl.tile.full([rows, cols], pl.FP32, 0.00000011920928955078125)
        return activate_core(value, zero, offset, scale, one, epsilon)

    return activate
