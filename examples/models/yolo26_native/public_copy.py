# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Experimental aligned channel-slice COPY using existing public PyPTO ops."""

from dataclasses import dataclass
from typing import Any

import pypto.language as pl

from .layout import VectorOp, VectorPlan


def copy_kernel(plan: VectorPlan) -> Any:
    """Build an aligned COPY that must be dispatched with one logical block."""
    if plan.operation != VectorOp.COPY:
        raise ValueError("public copy requires a COPY plan")
    pixels = plan.height * plan.width
    if pixels % 16 or plan.channels > 512:
        raise ValueError("public copy requires whole 16-pixel tiles and at most 512 channels")
    source_stride, destination_stride = plan.source_stride, plan.destination_stride
    source_offset, destination_offset, channels = plan.source_offset, plan.destination_offset, plan.channels
    ni, nb, no = plan.input_size, plan.second_size, plan.output_size

    @pl.jit.incore
    def copy(
        metadata: pl.Tensor[[32], pl.FP32],
        source: pl.Tensor[[ni], pl.FP16],
        second: pl.Tensor[[nb], pl.FP16],
        output: pl.Out[pl.Tensor[[no], pl.FP16]],
    ):
        src = pl.tensor.view(source, [pixels, source_stride])
        dst = pl.tensor.view(output, [pixels, destination_stride])
        for tile in pl.pipeline(pixels // 16, stage=2):
            values = pl.load(src, [tile * 16, source_offset], [16, channels])
            dst = pl.store(values, [tile * 16, destination_offset], dst)
        return output

    return copy


@dataclass(frozen=True)
class PublicCopyPlan(VectorPlan):
    """Existing vector ABI using a logical single-core public copy kernel."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.operation != VectorOp.COPY:
            raise ValueError("public copy requires a COPY plan")
        if self.height * self.width % 16 or self.channels > 512:
            raise ValueError("public copy requires whole 16-pixel tiles and at most 512 channels")

    def jit_kernel(self) -> Any:
        return copy_kernel(self)
