# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Public ordered N32 pointwise convolutions with fixed supported geometries."""

from typing import Any

import pypto.language as pl
from pypto.library import fp16_silu_constants


def ordered_pointwise_products(rows: int, k: int) -> Any:
    """Share the four ordered N32 partials and their N16 writeback panels."""

    @pl.jit.inline
    def ordered_products(
        left: pl.Tile[[rows, k], pl.FP16],
        w0: pl.Tile[[k, 32], pl.FP16],
        w1: pl.Tile[[k, 32], pl.FP16],
        w2: pl.Tile[[k, 32], pl.FP16],
        w3: pl.Tile[[k, 32], pl.FP16],
    ) -> tuple[pl.Tile[[rows, 16], pl.FP32], pl.Tile[[rows, 16], pl.FP32]]:
        p0 = pl.matmul(left, w0)
        low = pl.tile.move(
            pl.tile.slice(p0, [rows, 16], [0, 0]),
            pl.MemorySpace.Vec,
            blayout=pl.TileLayout.row_major,
            slayout=pl.TileLayout.none_box,
        )
        high = pl.tile.move(
            pl.tile.slice(p0, [rows, 16], [0, 16]),
            pl.MemorySpace.Vec,
            blayout=pl.TileLayout.row_major,
            slayout=pl.TileLayout.none_box,
        )
        p1 = pl.matmul(left, w1)
        low = pl.add(low, pl.tile.slice(p1, [rows, 16], [0, 0]))
        high = pl.add(high, pl.tile.slice(p1, [rows, 16], [0, 16]))
        p2 = pl.matmul(left, w2)
        low = pl.add(low, pl.tile.slice(p2, [rows, 16], [0, 0]))
        high = pl.add(high, pl.tile.slice(p2, [rows, 16], [0, 16]))
        p3 = pl.matmul(left, w3)
        low = pl.add(low, pl.tile.slice(p3, [rows, 16], [0, 0]))
        high = pl.add(high, pl.tile.slice(p3, [rows, 16], [0, 16]))
        return low, high

    return ordered_products


def ordered_pointwise_32(height: int = 160, cin: int = 32, source_stride: int = 0, source_offset: int = 0):
    if (height, cin) not in ((160, 32), (80, 32), (40, 64)):
        raise ValueError("unsupported ordered N32 pointwise geometry")
    stride = source_stride or cin
    if stride % 16 or source_offset % 16 or source_offset < 0 or source_offset + cin > stride:
        raise ValueError("ordered N32 pointwise requires a bounded aligned source view")
    rows = 64 if height == 40 else 160
    pixels = height * height
    k = cin * 4
    input_size = pixels * stride
    output_size = pixels * 32
    weight_size = 4 * k * 32
    activate = fp16_silu_constants(rows)

    ordered_products = ordered_pointwise_products(rows, k)

    @pl.jit.incore
    def compute(
        metadata: pl.Tensor[[32], pl.FP32],
        x: pl.Tensor[[input_size], pl.FP16],
        weight: pl.Tensor[[weight_size], pl.FP16],
        bias: pl.Tensor[[32], pl.FP32],
        output: pl.Out[pl.Tensor[[output_size], pl.FP16]],
        zero: pl.Tensor[[16], pl.FP16],
    ):
        source = pl.tensor.view(x, [pixels, stride])
        weights = pl.tensor.view(weight, [k * 4, 32])
        biases = pl.tensor.view(bias, [1, 32])
        destination = pl.tensor.view(output, [pixels, 32])
        w0 = pl.load(weights, [0, 0], [k, 32], target_memory=pl.MemorySpace.Mat)
        w1 = pl.load(weights, [k, 0], [k, 32], target_memory=pl.MemorySpace.Mat)
        w2 = pl.load(weights, [k * 2, 0], [k, 32], target_memory=pl.MemorySpace.Mat)
        w3 = pl.load(weights, [k * 3, 0], [k, 32], target_memory=pl.MemorySpace.Mat)
        b0 = pl.load(biases, [0, 0], [1, 16])
        b1 = pl.load(biases, [0, 16], [1, 16])
        zero_v = pl.tile.full([rows, 16], pl.FP32, 0.0)
        offset_v = pl.tile.full([rows, 16], pl.FP32, 15.0 / 2048)
        scale_v = pl.tile.full([rows, 16], pl.FP32, 8192.0)
        one_v = pl.tile.full([rows, 16], pl.FP32, 1.0)
        epsilon_v = pl.tile.full([rows, 16], pl.FP32, 0.00000011920928955078125)
        if pl.tile.get_block_num() == 1:
            for iteration in pl.pipeline(pixels // rows, stage=2):
                work = iteration
                row = work * rows
                left = pl.tile.create(
                    [rows, k],
                    pl.FP16,
                    target_memory=pl.MemorySpace.Left,
                    blayout=pl.TileLayout.row_major,
                    slayout=pl.TileLayout.row_major,
                )
                for block in pl.range(0, cin // 16):
                    source_panel = pl.load(
                        source,
                        [row, source_offset + block * 16],
                        [rows, 16],
                        target_memory=pl.MemorySpace.Mat,
                    )
                    for replica in pl.range(0, 4):
                        left = pl.tile.assemble(left, source_panel, [0, (block * 4 + replica) * 16])
                low, high = ordered_products(left, w0, w1, w2, w3)
                value = low
                value = pl.add(value, pl.tile.col_expand(value, b0))
                result = pl.cast(value, pl.FP16, mode="none")
                destination = pl.store(
                    activate(result, zero_v, offset_v, scale_v, one_v, epsilon_v), [row, 0], destination
                )
                value = high
                value = pl.add(value, pl.tile.col_expand(value, b1))
                result = pl.cast(value, pl.FP16, mode="none")
                destination = pl.store(
                    activate(result, zero_v, offset_v, scale_v, one_v, epsilon_v), [row, 16], destination
                )
        else:
            for work in pl.range(pl.tile.get_block_idx(), pixels // rows, pl.tile.get_block_num()):
                row = work * rows
                left = pl.tile.create(
                    [rows, k],
                    pl.FP16,
                    target_memory=pl.MemorySpace.Left,
                    blayout=pl.TileLayout.row_major,
                    slayout=pl.TileLayout.row_major,
                )
                for block in pl.range(0, cin // 16):
                    source_panel = pl.load(
                        source,
                        [row, source_offset + block * 16],
                        [rows, 16],
                        target_memory=pl.MemorySpace.Mat,
                    )
                    for replica in pl.range(0, 4):
                        left = pl.tile.assemble(left, source_panel, [0, (block * 4 + replica) * 16])
                low, high = ordered_products(left, w0, w1, w2, w3)
                value = low
                value = pl.add(value, pl.tile.col_expand(value, b0))
                result = pl.cast(value, pl.FP16, mode="none")
                destination = pl.store(
                    activate(result, zero_v, offset_v, scale_v, one_v, epsilon_v), [row, 0], destination
                )
                value = high
                value = pl.add(value, pl.tile.col_expand(value, b1))
                result = pl.cast(value, pl.FP16, mode="none")
                destination = pl.store(
                    activate(result, zero_v, offset_v, scale_v, one_v, epsilon_v), [row, 16], destination
                )
        return output

    return compute
