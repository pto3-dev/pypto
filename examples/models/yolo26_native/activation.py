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
        # Finite FP16 neighbors around -15/2048 are at least 2^-18 apart.
        # The scaled distance is exactly a one/zero mask on this input domain,
        # correcting the hardware exp rounding before narrowing the sigmoid.
        delta = pl.add(x, offset)
        magnitude = pl.tile.maximum(delta, pl.sub(zero, delta))
        scale = pl.tile.full([rows, 16], pl.FP32, 524288.0)
        one = pl.tile.full([rows, 16], pl.FP32, 1.0)
        mask = pl.tile.maximum(zero, pl.sub(one, pl.mul(magnitude, scale)))
        epsilon = pl.tile.full([rows, 16], pl.FP32, 0.00000011920928955078125)
        corrected = pl.add(exponent, pl.mul(mask, epsilon))
        sigmoid = pl.div(one, pl.add(corrected, one))
        sigmoid_half = pl.cast(sigmoid, pl.FP16, mode="none")
        sigmoid = pl.cast(sigmoid_half, pl.FP32, mode="none")
        return pl.cast(pl.mul(x, sigmoid), pl.FP16, mode="none")

    return activate
