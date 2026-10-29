# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Public 80x80 C80 depthwise convolution with ordered FP32 accumulation."""

import pypto.language as pl
from pypto.library import fp16_silu_constants


def public_depthwise_80():  # noqa: PLR0915
    """Build the stride-one, padding-one, four-way accumulation SiLU kernel."""
    act13 = fp16_silu_constants(13, 80)
    act1 = fp16_silu_constants(1, 80)

    # Keep the nine ordered taps explicit, including both guarded image edges.
    @pl.jit.incore
    def compute(  # noqa: PLR0915
        metadata: pl.Tensor[[32], pl.FP32],
        x: pl.Tensor[[512000], pl.FP16],
        weight: pl.Tensor[[720], pl.FP16],
        bias: pl.Tensor[[80], pl.FP32],
        output: pl.Out[pl.Tensor[[512000], pl.FP16]],
    ):
        src = pl.tensor.view(x, [6400, 80])
        ws = pl.tensor.view(weight, [9, 80])
        bs = pl.tensor.view(bias, [1, 80])
        dst = pl.tensor.view(output, [6400, 80])
        bz = pl.load(bs, [0, 0], [1, 80])
        shape = pl.tile.full([13, 80], pl.FP32, 0.0)
        b13 = pl.tile.col_expand(shape, bz)
        w0 = pl.cast(pl.load(ws, [0, 0], [1, 80]), pl.FP32, mode="none")
        v0 = pl.tile.col_expand(shape, w0)
        w1 = pl.cast(pl.load(ws, [1, 0], [1, 80]), pl.FP32, mode="none")
        v1 = pl.tile.col_expand(shape, w1)
        w2 = pl.cast(pl.load(ws, [2, 0], [1, 80]), pl.FP32, mode="none")
        v2 = pl.tile.col_expand(shape, w2)
        w3 = pl.cast(pl.load(ws, [3, 0], [1, 80]), pl.FP32, mode="none")
        v3 = pl.tile.col_expand(shape, w3)
        w4 = pl.cast(pl.load(ws, [4, 0], [1, 80]), pl.FP32, mode="none")
        v4 = pl.tile.col_expand(shape, w4)
        w5 = pl.cast(pl.load(ws, [5, 0], [1, 80]), pl.FP32, mode="none")
        v5 = pl.tile.col_expand(shape, w5)
        w6 = pl.cast(pl.load(ws, [6, 0], [1, 80]), pl.FP32, mode="none")
        v6 = pl.tile.col_expand(shape, w6)
        w7 = pl.cast(pl.load(ws, [7, 0], [1, 80]), pl.FP32, mode="none")
        v7 = pl.tile.col_expand(shape, w7)
        w8 = pl.cast(pl.load(ws, [8, 0], [1, 80]), pl.FP32, mode="none")
        v8 = pl.tile.col_expand(shape, w8)
        zero13 = pl.tile.full([13, 80], pl.FP32, 0.0)
        offset13 = pl.tile.full([13, 80], pl.FP32, 15.0 / 2048)
        scale13 = pl.tile.full([13, 80], pl.FP32, 8192.0)
        one13 = pl.tile.full([13, 80], pl.FP32, 1.0)
        epsilon13 = pl.tile.full([13, 80], pl.FP32, 0.00000011920928955078125)
        zero1 = pl.tile.full([1, 80], pl.FP32, 0.0)
        offset1 = pl.tile.full([1, 80], pl.FP32, 15.0 / 2048)
        scale1 = pl.tile.full([1, 80], pl.FP32, 8192.0)
        one1 = pl.tile.full([1, 80], pl.FP32, 1.0)
        epsilon1 = pl.tile.full([1, 80], pl.FP32, 0.00000011920928955078125)
        for row in pl.range(pl.tile.get_block_idx(), 80, pl.tile.get_block_num()):
            sy0 = pl.max(row + (-1), 0)
            sy2 = pl.min(row + 1, 79)
            for group in pl.range(0, 6):
                col = 1 + group * 13
                # Three full-width slices share each contiguous 15-pixel stripe.
                top_window = pl.load(src, [sy0 * 80 + col + (-1), 0], [15, 80])
                mid_window = pl.load(src, [row * 80 + col + (-1), 0], [15, 80])
                bottom_window = pl.load(src, [sy2 * 80 + col + (-1), 0], [15, 80])
                p0_13 = pl.tile.full([13, 80], pl.FP32, 0.0)
                p1_13 = pl.tile.full([13, 80], pl.FP32, 0.0)
                p2_13 = pl.tile.full([13, 80], pl.FP32, 0.0)
                p3_13 = pl.tile.full([13, 80], pl.FP32, 0.0)
                if row > 0:
                    a0_13 = pl.cast(pl.tile.slice(top_window, [13, 80], [0, 0]), pl.FP32, mode="none")
                    p0_13 = pl.add(p0_13, pl.mul(a0_13, v0))
                    a1_13 = pl.cast(pl.tile.slice(top_window, [13, 80], [1, 0]), pl.FP32, mode="none")
                    p1_13 = pl.add(p1_13, pl.mul(a1_13, v1))
                    a2_13 = pl.cast(pl.tile.slice(top_window, [13, 80], [2, 0]), pl.FP32, mode="none")
                    p2_13 = pl.add(p2_13, pl.mul(a2_13, v2))
                a3_13 = pl.cast(pl.tile.slice(mid_window, [13, 80], [0, 0]), pl.FP32, mode="none")
                p3_13 = pl.add(p3_13, pl.mul(a3_13, v3))
                a4_13 = pl.cast(pl.tile.slice(mid_window, [13, 80], [1, 0]), pl.FP32, mode="none")
                p0_13 = pl.add(p0_13, pl.mul(a4_13, v4))
                a5_13 = pl.cast(pl.tile.slice(mid_window, [13, 80], [2, 0]), pl.FP32, mode="none")
                p1_13 = pl.add(p1_13, pl.mul(a5_13, v5))
                if row < 79:
                    a6_13 = pl.cast(pl.tile.slice(bottom_window, [13, 80], [0, 0]), pl.FP32, mode="none")
                    p2_13 = pl.add(p2_13, pl.mul(a6_13, v6))
                    a7_13 = pl.cast(pl.tile.slice(bottom_window, [13, 80], [1, 0]), pl.FP32, mode="none")
                    p3_13 = pl.add(p3_13, pl.mul(a7_13, v7))
                    a8_13 = pl.cast(pl.tile.slice(bottom_window, [13, 80], [2, 0]), pl.FP32, mode="none")
                    p0_13 = pl.add(p0_13, pl.mul(a8_13, v8))
                total13 = pl.add(pl.add(pl.add(p0_13, p1_13), p2_13), p3_13)
                rounded13 = pl.cast(pl.add(total13, b13), pl.FP16, mode="none")
                result13 = act13(rounded13, zero13, offset13, scale13, one13, epsilon13)
                dst = pl.store(result13, [row * 80 + col, 0], dst)
            p0_1 = pl.tile.full([1, 80], pl.FP32, 0.0)
            p1_1 = pl.tile.full([1, 80], pl.FP32, 0.0)
            p2_1 = pl.tile.full([1, 80], pl.FP32, 0.0)
            p3_1 = pl.tile.full([1, 80], pl.FP32, 0.0)
            if row > 0:
                a1_1 = pl.cast(pl.load(src, [sy0 * 80 + 0, 0], [1, 80]), pl.FP32, mode="none")
                p1_1 = pl.add(p1_1, pl.mul(a1_1, w1))
                a2_1 = pl.cast(pl.load(src, [sy0 * 80 + 0 + (1), 0], [1, 80]), pl.FP32, mode="none")
                p2_1 = pl.add(p2_1, pl.mul(a2_1, w2))
            a4_1 = pl.cast(pl.load(src, [row * 80 + 0, 0], [1, 80]), pl.FP32, mode="none")
            p0_1 = pl.add(p0_1, pl.mul(a4_1, w4))
            a5_1 = pl.cast(pl.load(src, [row * 80 + 0 + (1), 0], [1, 80]), pl.FP32, mode="none")
            p1_1 = pl.add(p1_1, pl.mul(a5_1, w5))
            if row < 79:
                a7_1 = pl.cast(pl.load(src, [sy2 * 80 + 0, 0], [1, 80]), pl.FP32, mode="none")
                p3_1 = pl.add(p3_1, pl.mul(a7_1, w7))
                a8_1 = pl.cast(pl.load(src, [sy2 * 80 + 0 + (1), 0], [1, 80]), pl.FP32, mode="none")
                p0_1 = pl.add(p0_1, pl.mul(a8_1, w8))
            total1 = pl.add(pl.add(pl.add(p0_1, p1_1), p2_1), p3_1)
            rounded1 = pl.cast(pl.add(total1, bz), pl.FP16, mode="none")
            result1 = act1(rounded1, zero1, offset1, scale1, one1, epsilon1)
            dst = pl.store(result1, [row * 80 + 0, 0], dst)
            p0_1 = pl.tile.full([1, 80], pl.FP32, 0.0)
            p1_1 = pl.tile.full([1, 80], pl.FP32, 0.0)
            p2_1 = pl.tile.full([1, 80], pl.FP32, 0.0)
            p3_1 = pl.tile.full([1, 80], pl.FP32, 0.0)
            if row > 0:
                a0_1 = pl.cast(pl.load(src, [sy0 * 80 + 79 + (-1), 0], [1, 80]), pl.FP32, mode="none")
                p0_1 = pl.add(p0_1, pl.mul(a0_1, w0))
                a1_1 = pl.cast(pl.load(src, [sy0 * 80 + 79, 0], [1, 80]), pl.FP32, mode="none")
                p1_1 = pl.add(p1_1, pl.mul(a1_1, w1))
            a3_1 = pl.cast(pl.load(src, [row * 80 + 79 + (-1), 0], [1, 80]), pl.FP32, mode="none")
            p3_1 = pl.add(p3_1, pl.mul(a3_1, w3))
            a4_1 = pl.cast(pl.load(src, [row * 80 + 79, 0], [1, 80]), pl.FP32, mode="none")
            p0_1 = pl.add(p0_1, pl.mul(a4_1, w4))
            if row < 79:
                a6_1 = pl.cast(pl.load(src, [sy2 * 80 + 79 + (-1), 0], [1, 80]), pl.FP32, mode="none")
                p2_1 = pl.add(p2_1, pl.mul(a6_1, w6))
                a7_1 = pl.cast(pl.load(src, [sy2 * 80 + 79, 0], [1, 80]), pl.FP32, mode="none")
                p3_1 = pl.add(p3_1, pl.mul(a7_1, w7))
            total1 = pl.add(pl.add(pl.add(p0_1, p1_1), p2_1), p3_1)
            rounded1 = pl.cast(pl.add(total1, bz), pl.FP16, mode="none")
            result1 = act1(rounded1, zero1, offset1, scale1, one1, epsilon1)
            dst = pl.store(result1, [row * 80 + 79, 0], dst)
        return output

    return compute
