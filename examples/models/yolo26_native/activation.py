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


def fp16_silu(rows: int) -> Any:
    """Return an inline M64/M128/M256 by N16 activation for coupled 310P cores."""
    if type(rows) is not int or rows not in (64, 128, 256):
        raise ValueError("FP16 SiLU requires an M64/M128/M256 by N16 tile")

    @pl.jit.inline
    def activate(value: pl.Tile[[rows, 16], pl.FP16]) -> pl.Tile[[rows, 16], pl.FP16]:
        x = pl.cast(value, pl.FP32, mode="none")
        zero = pl.tile.full([rows, 16], pl.FP32, 0.0)
        negative = pl.sub(zero, x)
        exponent = pl.exp(negative)
        offset = pl.tile.full([rows, 16], pl.FP32, 15.0 / 2048)
        # Nonzero FP16 distance from -15/2048 is at least 2^-18,
        # so 8192 * distance^2 is at least the 2^-23 correction.
        delta = pl.add(x, offset)
        squared = pl.mul(delta, delta)
        scale = pl.tile.full([rows, 16], pl.FP32, 8192.0)
        one = pl.tile.full([rows, 16], pl.FP32, 1.0)
        epsilon = pl.tile.full([rows, 16], pl.FP32, 0.00000011920928955078125)
        correction = pl.tile.maximum(zero, pl.sub(epsilon, pl.mul(squared, scale)))
        corrected = pl.add(exponent, correction)
        sigmoid = pl.div(one, pl.add(corrected, one))
        sigmoid_half = pl.cast(sigmoid, pl.FP16, mode="none")
        sigmoid = pl.cast(sigmoid_half, pl.FP32, mode="none")
        return pl.cast(pl.mul(x, sigmoid), pl.FP16, mode="none")

    return activate
