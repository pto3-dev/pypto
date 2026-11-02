# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Public channel-view copying through existing load/store and pipeline ops."""

from dataclasses import dataclass
from typing import Any

import pypto.language as pl

from .layout import VectorOp, VectorPlan


def copy_kernel(plan: VectorPlan) -> Any:
    """Copy aligned FP16 NHWC channel views without arithmetic or conversion."""
    if plan.operation != VectorOp.COPY:
        raise ValueError("public channel copy requires a COPY plan")
    if plan.channels > 4096:
        raise ValueError("public channel copy supports at most 4096 channels")
    if any(
        (stride - plan.channels) // 16 > 65535 for stride in (plan.source_stride, plan.destination_stride)
    ):
        raise ValueError("public channel copy requires bounded DMA row gaps")
    pixels, channels = plan.height * plan.width, plan.channels
    ss, ds = plan.source_stride, plan.destination_stride
    so, dest = plan.source_offset, plan.destination_offset
    ni, no = plan.input_size, plan.output_size
    rows = 16
    while rows * channels > 8192 or rows > pixels:
        rows //= 2

    def panel(count: int) -> Any:
        @pl.jit.inline
        def copy_panel(
            first: pl.Scalar[pl.INDEX], source: pl.Tensor[[ni], pl.FP16], output: pl.Tensor[[no], pl.FP16]
        ) -> None:
            source_view = pl.tensor.view(source, [pixels, ss])
            destination_view = pl.tensor.view(output, [pixels, ds])
            value = pl.load(source_view, [first, so], [count, channels])
            pl.store(value, [first, dest], destination_view)

        return copy_panel

    def grid(count: int) -> Any:
        full, tail = divmod(pixels, count)
        transfer = panel(count)
        finish = panel(tail if tail else 1)
        tail_first = full * count if tail else 0

        @pl.jit.inline
        def copy_grid(source: pl.Tensor[[ni], pl.FP16], output: pl.Tensor[[no], pl.FP16]) -> None:
            # Independent batch images see grid one. A1 lowers this pipeline to
            # replicated static banks; multi-core panels have disjoint owners.
            if pl.tile.get_block_num() == 1:
                for block in pl.pipeline(full, stage=2):
                    transfer(block * count, source, output)
            else:
                for block in pl.range(pl.tile.get_block_idx(), full, pl.tile.get_block_num()):
                    transfer(block * count, source, output)
            if tail > 0:
                if pl.tile.get_block_idx() == full % pl.tile.get_block_num():
                    finish(tail_first, source, output)

        return copy_grid

    def dispatch(count: int) -> Any:
        transfer = grid(count)
        if count == 1:
            return transfer
        smaller = dispatch(count // 2)
        panels = (pixels + count - 1) // count

        @pl.jit.inline
        def choose_grid(source: pl.Tensor[[ni], pl.FP16], output: pl.Tensor[[no], pl.FP16]) -> None:
            # Keep the largest 16-KiB-or-smaller panel that exposes enough work
            # for the actual grid, matching native COPY's occupancy decision.
            if pl.tile.get_block_num() <= panels:
                transfer(source, output)
            else:
                smaller(source, output)

        return choose_grid

    transfer = dispatch(rows)

    @pl.jit.incore
    def channel_copy(
        metadata: pl.Tensor[[32], pl.FP32],
        source: pl.Tensor[[ni], pl.FP16],
        second: pl.Tensor[[16], pl.FP16],
        output: pl.Out[pl.Tensor[[no], pl.FP16]],
    ):
        transfer(source, output)
        return output

    return channel_copy


@dataclass(frozen=True)
class PublicCopyPlan(VectorPlan):
    """Existing four-Tensor vector ABI with raw-bit public channel copying."""

    def jit_kernel(self) -> Any:
        return copy_kernel(self)
