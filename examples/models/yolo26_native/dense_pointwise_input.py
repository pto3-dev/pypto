# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Experimental dense four-phase input windows using existing public PyPTO ops."""

from typing import Any

import pypto.language as pl

from .operators import ConvPlan


def dense_pointwise_image_cache(plan: ConvPlan, rows: int) -> Any:
    """Stage one input window once into four channel-phase NZ images.

    Phase ``p`` contains original channels ``p, p+4, ...`` in ascending order.
    Every phase width is padded to C16, and its padded lanes are zero. A source
    C64 panel is loaded once and shared by four fixed-mask gathers. Full panels
    use the existing two-stage pipeline; a C16/C32/C48 tail is zero-padded in UB
    before selection, so no read crosses the logical input-channel boundary.

    The returned inline helper accepts the original source Tensor and its
    bounded first pixel, and returns four independent ``[rows, ceil(C/64)*16]``
    Mat tiles. It performs no Cube reduction, activation, or GM output store.
    Consumers own those operations and can retain the images across output
    channel panels. This experimental factory does not change model selection;
    device precision and performance remain acceptance gates for dense Cube K.
    """
    if (
        not plan.four_way_accumulation
        or plan.kernel != 1
        or plan.stride != 1
        or plan.padding != 0
        or plan.cin % 16
        or plan.cin > 512
        or type(rows) is not int
        or rows < 16
        or rows > 224
        or rows % 16
        or rows > plan.height * plan.width
    ):
        raise ValueError(
            "dense pointwise images require ordered C16 inputs up to 512 and bounded M16 rows up to 224"
        )
    pixels, channels = plan.height * plan.width, plan.cin
    source_stride, source_offset = plan.input_row_stride, plan.source_offset
    full, tail = divmod(channels, 64)
    has_full, has_tail = full > 0, tail > 0
    phase_columns = (channels + 63) // 64 * 16

    @pl.jit.inline
    def stage(
        source: pl.Tensor[[pixels, source_stride], pl.FP16],
        first: pl.Scalar[pl.INDEX],
    ):
        image0 = pl.tile.create([rows, phase_columns], pl.FP16, target_memory=pl.Mem.Mat)
        image1 = pl.tile.create([rows, phase_columns], pl.FP16, target_memory=pl.Mem.Mat)
        image2 = pl.tile.create([rows, phase_columns], pl.FP16, target_memory=pl.Mem.Mat)
        image3 = pl.tile.create([rows, phase_columns], pl.FP16, target_memory=pl.Mem.Mat)
        if has_full:
            for block in pl.pipeline(full, stage=2):
                values = pl.load(source, [first, source_offset + block * 64], [rows, 64])
                phase0 = pl.tile.gather_mask(values, pl.tile.MaskPattern.P0001)
                phase1 = pl.tile.gather_mask(values, pl.tile.MaskPattern.P0010)
                phase2 = pl.tile.gather_mask(values, pl.tile.MaskPattern.P0100)
                phase3 = pl.tile.gather_mask(values, pl.tile.MaskPattern.P1000)
                image0 = pl.tile.assemble(image0, phase0, [0, block * 16])
                image1 = pl.tile.assemble(image1, phase1, [0, block * 16])
                image2 = pl.tile.assemble(image2, phase2, [0, block * 16])
                image3 = pl.tile.assemble(image3, phase3, [0, block * 16])
        if has_tail:
            # Fill before inserting only real channels: masks select the
            # original channel phases and defined zeros in every padded lane.
            padded = pl.tile.full([rows, 64], pl.FP16, 0.0)
            tail_values = pl.load(source, [first, source_offset + full * 64], [rows, tail])
            padded = pl.tile.assemble(padded, tail_values, [0, 0])
            phase0 = pl.tile.gather_mask(padded, pl.tile.MaskPattern.P0001)
            phase1 = pl.tile.gather_mask(padded, pl.tile.MaskPattern.P0010)
            phase2 = pl.tile.gather_mask(padded, pl.tile.MaskPattern.P0100)
            phase3 = pl.tile.gather_mask(padded, pl.tile.MaskPattern.P1000)
            image0 = pl.tile.assemble(image0, phase0, [0, full * 16])
            image1 = pl.tile.assemble(image1, phase1, [0, full * 16])
            image2 = pl.tile.assemble(image2, phase2, [0, full * 16])
            image3 = pl.tile.assemble(image3, phase3, [0, full * 16])
        return image0, image1, image2, image3

    return stage
