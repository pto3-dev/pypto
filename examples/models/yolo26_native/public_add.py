# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Experimental public residual addition with aligned NHWC channel views."""

from dataclasses import dataclass
from typing import Any

import pypto.language as pl

from .layout import VectorOp, VectorPlan


def add_kernel(plan: VectorPlan) -> Any:
    """Use existing load/add/store and pipeline operations for FP16 ADD."""
    if plan.operation != VectorOp.ADD:
        raise ValueError("public residual addition requires an ADD plan")
    if plan.channels > 1024:
        raise ValueError("public residual addition supports at most 1024 channels")
    if any(
        (stride - plan.channels) // 16 > 65535
        for stride in (plan.source_stride, plan.second_stride, plan.destination_stride)
    ):
        raise ValueError("public residual addition requires bounded DMA row gaps")
    pixels, channels = plan.height * plan.width, plan.channels
    source_stride, second_stride, destination_stride = (
        plan.source_stride,
        plan.second_stride,
        plan.destination_stride,
    )
    source_offset, second_offset, destination_offset = (
        plan.source_offset,
        plan.second_offset,
        plan.destination_offset,
    )
    rows = 16
    while rows * channels > 4096 or rows > pixels:
        rows //= 2
    full, tail = divmod(pixels, rows)
    # Inline helpers are type-checked before dead branches disappear. Give the
    # unused zero-tail specialization a valid origin inside the tensor too.
    tail_first = full * rows if tail else 0
    ni, nb, no = plan.input_size, plan.second_size, plan.output_size

    def panel(count: int) -> Any:
        @pl.jit.inline
        def add_panel(
            first: pl.Scalar[pl.INDEX],
            left: pl.Tensor[[ni], pl.FP16],
            right: pl.Tensor[[nb], pl.FP16],
            destination: pl.Tensor[[no], pl.FP16],
        ) -> None:
            left_view = pl.tensor.view(left, [pixels, source_stride])
            right_view = pl.tensor.view(right, [pixels, second_stride])
            output_view = pl.tensor.view(destination, [pixels, destination_stride])
            a = pl.load(left_view, [first, source_offset], [count, channels])
            b = pl.load(right_view, [first, second_offset], [count, channels])
            pl.store(pl.add(a, b), [first, destination_offset], output_view)

        return add_panel

    add_full = panel(rows)
    add_tail = panel(tail if tail else 1)

    @pl.jit.incore
    def residual_add(
        metadata: pl.Tensor[[32], pl.FP32],
        source: pl.Tensor[[ni], pl.FP16],
        second: pl.Tensor[[nb], pl.FP16],
        output: pl.Out[pl.Tensor[[no], pl.FP16]],
    ):
        # Batch replay provides one logical block per independent image. Keep
        # its two-stage DMA pipeline and a disjoint, strided multi-block path.
        if pl.tile.get_block_num() == 1:
            for block in pl.pipeline(full, stage=2):
                add_full(block * rows, source, second, output)
        else:
            for block in pl.range(pl.tile.get_block_idx(), full, pl.tile.get_block_num()):
                add_full(block * rows, source, second, output)
        if tail > 0:
            if pl.tile.get_block_idx() == 0:
                add_tail(tail_first, source, second, output)
        return output

    return residual_add


@dataclass(frozen=True)
class PublicAddPlan(VectorPlan):
    """Existing vector ABI with a public FP16 residual-add implementation."""

    def jit_kernel(self) -> Any:
        return add_kernel(self)
