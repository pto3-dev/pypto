# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Public FP16 convolution bias, activation and bounded output writeback."""

from typing import Any

import pypto.language as pl

from .activation import fp16_silu_constants


def fp16_conv_epilogue(rows: int, *, silu: bool = True) -> Any:
    """Build one N16 bias/cast/activation epilogue from existing public ops.

    Args:
        rows: Static M16-aligned row count. With SiLU, use an N16 profile
            supported by ``pypto.library.fp16_silu_constants``.
        silu: Apply separately rounded FP16 sigmoid/product after bias and
            FP16 convolution rounding. False returns the rounded convolution.

    Returns:
        Inline helper accepting merged FP32 values, an FP32 ``[1, 16]``
        bias, and five caller-owned FP32 constant tiles (zero, offset, scale,
        one and epsilon). Keep constants outside the spatial work loop.
        The helper does not allocate constants, perform DMA or store output.

    Raises:
        ValueError: Geometry or activation is outside the supported profile.
    """
    if type(rows) is not int or rows < 16 or rows % 16 or type(silu) is not bool:
        raise ValueError(
            f"convolution epilogue requires positive M16 rows and boolean SiLU, got {rows}, {silu}"
        )
    if silu:
        activate = fp16_silu_constants(rows)

        @pl.jit.inline
        def finish(
            value: pl.Tile[[rows, 16], pl.FP32],
            bias: pl.Tile[[1, 16], pl.FP32],
            zero: pl.Tile[[rows, 16], pl.FP32],
            offset: pl.Tile[[rows, 16], pl.FP32],
            scale: pl.Tile[[rows, 16], pl.FP32],
            one: pl.Tile[[rows, 16], pl.FP32],
            epsilon: pl.Tile[[rows, 16], pl.FP32],
        ):
            biased = pl.add(value, pl.tile.col_expand(value, bias))
            rounded = pl.cast(biased, pl.FP16, mode="none")
            return activate(rounded, zero, offset, scale, one, epsilon)

        return finish

    @pl.jit.inline
    def finish(
        value: pl.Tile[[rows, 16], pl.FP32],
        bias: pl.Tile[[1, 16], pl.FP32],
        zero: pl.Tile[[rows, 16], pl.FP32],
        offset: pl.Tile[[rows, 16], pl.FP32],
        scale: pl.Tile[[rows, 16], pl.FP32],
        one: pl.Tile[[rows, 16], pl.FP32],
        epsilon: pl.Tile[[rows, 16], pl.FP32],
    ):
        return pl.cast(pl.add(value, pl.tile.col_expand(value, bias)), pl.FP16, mode="none")

    return finish


def fp16_conv_writeback(rows: int, output_rows: int, output_columns: int, *, silu: bool = True) -> Any:
    """Store an ordered N16 column through bounded 64/32/16-row epilogues.

    Keep the Cube tile large while reusing five M64 FP32 constants across
    the output loop and across both N16 columns of an N32 product. Only a
    completed FP16 result is written to GM. The input is a whole, already
    merged Vec column: no partial-row Acc alias is required.

    Args:
        rows: M16-aligned height of the merged FP32 Vec column.
        output_rows: Static row capacity of the destination Tensor.
        output_columns: C16-aligned physical output row stride.
        silu: Preserve the model's separate FP16 sigmoid/product rounding.

    Returns:
        Inline writer taking the merged column, a ``[1,16]`` FP32 bias,
        five ``[64,16]`` FP32 constants (zero/offset/scale/one/epsilon), the
        destination Tensor and bounded starting row/column indices. The
        caller owns scheduling, constants, bias loading and output bounds.
        Smaller chunks alias prefixes of the same constants; no refill or
        intermediate GM tensor is introduced.

    Raises:
        ValueError: Geometry cannot describe a bounded C16 output column.
    """
    geometry = (rows, output_rows, output_columns)
    if (
        any(type(value) is not int for value in geometry)
        or rows < 16
        or rows % 16
        or output_rows < rows
        or output_columns < 16
        or output_columns % 16
        or output_rows * output_columns > 0xFFFFFFFF
        or rows * 16 * 4 > 131072
        or type(silu) is not bool
    ):
        raise ValueError(f"convolution writeback requires bounded M16/C16 geometry, got {geometry}, {silu}")

    # Explicit Vec parameter types preserve slice memory inheritance before
    # inlining; no partial-row Acc reinterpretation or materialization is used.
    def build(first: int) -> Any:
        remaining = rows - first
        count = 64 if remaining >= 64 else 32 if remaining >= 32 else 16
        finish = fp16_conv_epilogue(count, silu=silu)

        @pl.jit.inline
        def activate(
            value: pl.Tile[[rows, 16], pl.FP32, pl.Mem.Vec],
            bias: pl.Tile[[1, 16], pl.FP32],
            zero: pl.Tile[[64, 16], pl.FP32, pl.Mem.Vec],
            offset: pl.Tile[[64, 16], pl.FP32, pl.Mem.Vec],
            scale: pl.Tile[[64, 16], pl.FP32, pl.Mem.Vec],
            one: pl.Tile[[64, 16], pl.FP32, pl.Mem.Vec],
            epsilon: pl.Tile[[64, 16], pl.FP32, pl.Mem.Vec],
        ):
            return finish(
                pl.tile.slice(value, [count, 16], [first, 0]),
                bias,
                pl.tile.slice(zero, [count, 16], [0, 0]),
                pl.tile.slice(offset, [count, 16], [0, 0]),
                pl.tile.slice(scale, [count, 16], [0, 0]),
                pl.tile.slice(one, [count, 16], [0, 0]),
                pl.tile.slice(epsilon, [count, 16], [0, 0]),
            )

        if count == remaining:

            @pl.jit.inline
            def write(
                value: pl.Tile[[rows, 16], pl.FP32, pl.Mem.Vec],
                bias: pl.Tile[[1, 16], pl.FP32],
                zero: pl.Tile[[64, 16], pl.FP32, pl.Mem.Vec],
                offset: pl.Tile[[64, 16], pl.FP32, pl.Mem.Vec],
                scale: pl.Tile[[64, 16], pl.FP32, pl.Mem.Vec],
                one: pl.Tile[[64, 16], pl.FP32, pl.Mem.Vec],
                epsilon: pl.Tile[[64, 16], pl.FP32, pl.Mem.Vec],
                output: pl.Tensor[[output_rows, output_columns], pl.FP16],
                row: pl.Scalar[pl.INDEX],
                column: pl.Scalar[pl.INDEX],
            ):
                result = activate(value, bias, zero, offset, scale, one, epsilon)
                return pl.store(result, [row + first, column], output)

            return write

        # Recursion is in this Python factory, not the device program. Each
        # statically bounded inline call becomes another final output store.
        write_rest = build(first + count)

        @pl.jit.inline
        def write(
            value: pl.Tile[[rows, 16], pl.FP32, pl.Mem.Vec],
            bias: pl.Tile[[1, 16], pl.FP32],
            zero: pl.Tile[[64, 16], pl.FP32, pl.Mem.Vec],
            offset: pl.Tile[[64, 16], pl.FP32, pl.Mem.Vec],
            scale: pl.Tile[[64, 16], pl.FP32, pl.Mem.Vec],
            one: pl.Tile[[64, 16], pl.FP32, pl.Mem.Vec],
            epsilon: pl.Tile[[64, 16], pl.FP32, pl.Mem.Vec],
            output: pl.Tensor[[output_rows, output_columns], pl.FP16],
            row: pl.Scalar[pl.INDEX],
            column: pl.Scalar[pl.INDEX],
        ):
            result = activate(value, bias, zero, offset, scale, one, epsilon)
            output = pl.store(result, [row + first, column], output)
            return write_rest(value, bias, zero, offset, scale, one, epsilon, output, row, column)

        return write

    return build(0)
