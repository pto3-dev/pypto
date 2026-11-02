# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Finite FP16 spatial pooling composed from existing public operations."""

from typing import Any

import pypto.language as pl


def fp16_max_pool2d_window(
    height: int,
    width: int,
    channels: int,
    *,
    kernel_size: int = 5,
    source_stride: int | None = None,
    source_offset: int = 0,
) -> Any:
    """Build a finite FP16 NHWC max-pooling window with same padding.

    Args:
        height: Positive input height.
        width: Positive input width.
        channels: C16-aligned channel count, at most 1024.
        kernel_size: Odd window width/height in 1..31; stride is one.
        source_stride: Physical input row width, defaults to channels.
        source_offset: C16-aligned channel origin inside each physical row.

    Returns:
        Public JIT inline helper accepting the two-dimensional input Tensor
        and a bounded output row/column. It returns one [1, channels] Tile.
        The caller owns scheduling, output allocation and writeback.

    Raises:
        ValueError: Geometry cannot describe a bounded aligned window.

    Inputs must be finite FP16 values. The seed is -65504, and valid neighbors
    are visited in lexical row/column order. Keeping operand order preserves
    the existing per-lane maximum contract, including signed-zero ties.
    """
    geometry = (height, width, channels, kernel_size)
    if (
        any(type(value) is not int for value in geometry)
        or min(height, width, channels) <= 0
        or channels % 16
        or channels > 1024
        or not 1 <= kernel_size <= 31
        or kernel_size % 2 == 0
    ):
        raise ValueError(f"pooling requires positive H/W, C16<=1024 and an odd 1..31 window, got {geometry}")
    stride = channels if source_stride is None else source_stride
    if (
        type(stride) is not int
        or type(source_offset) is not int
        or stride <= 0
        or source_offset < 0
        or stride % 16
        or source_offset % 16
        or source_offset + channels > stride
        or (stride - channels) // 16 > 65535
        or height * width * stride > 0x7FFFFFFF
    ):
        raise ValueError(
            f"pooling requires a bounded C16 source view, got stride={stride}, offset={source_offset}"
        )
    pixels, radius = height * width, kernel_size // 2

    @pl.jit.inline
    def window(
        source: pl.Tensor[[pixels, stride], pl.FP16],
        row: pl.Scalar[pl.INDEX],
        column: pl.Scalar[pl.INDEX],
    ) -> pl.Tile[[1, channels], pl.FP16]:
        # Scalar clamps keep every DMA inside the image instead of reading
        # padding. One full channel tile replaces separate C16 work items.
        pooled = pl.tile.full([1, channels], pl.FP16, -65504.0)
        # Tile values cross both dynamic loop boundaries explicitly; a local
        # reassignment alone does not define an IR loop-carried result.
        for sy, (row_maximum,) in pl.range(
            pl.max(row - radius, 0), pl.min(row + radius + 1, height), init_values=(pooled,)
        ):
            row_result = row_maximum
            for sx, (column_maximum,) in pl.range(
                pl.max(column - radius, 0), pl.min(column + radius + 1, width), init_values=(row_result,)
            ):
                value = pl.load(source, [sy * width + sx, source_offset], [1, channels])
                row_result = pl.yield_(pl.maximum(column_maximum, value))
            pooled = pl.yield_(row_result)
        return pooled

    return window
