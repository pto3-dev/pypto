# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Compact public PyPTO classifier projection with resident input and weights."""

from dataclasses import asdict, dataclass, replace
from typing import Any

import pypto.language as pl
import torch

from .operators import ConvPlan


def pointwise_projection(pixels: int) -> Any:
    """Share an M80 Left operand across all five compact classifier panels."""
    if pixels not in (400, 1600, 6400):
        raise ValueError("classifier projection requires a 20/40/80 square grid")
    elements = pixels * 80

    @pl.jit.inline
    def project_panel(
        left: pl.Tile[[80, 80], pl.FP16],
        weight: pl.Tile[[80, 16], pl.FP16],
        bias: pl.Tile[[1, 16], pl.FP32],
    ) -> pl.Tile[[80, 16], pl.FP16]:
        acc = pl.matmul(left, weight)
        value = pl.add(acc, pl.tile.col_expand(acc, bias))
        return pl.cast(value, pl.FP16, mode="none")

    @pl.jit.incore
    def convolution(
        metadata: pl.Tensor[[32], pl.FP32],
        x: pl.Tensor[[elements], pl.FP16],
        weight: pl.Tensor[[6400], pl.FP16],
        bias: pl.Tensor[[80], pl.FP32],
        output: pl.Out[pl.Tensor[[elements], pl.FP16]],
        zero: pl.Tensor[[16], pl.FP16],
    ):
        source = pl.tensor.view(x, [pixels, 80])
        weights = pl.tensor.view(weight, [400, 16])
        biases = pl.tensor.view(bias, [1, 80])
        destination = pl.tensor.view(output, [pixels, 80])
        # Keep all five constant weight/bias tiles live across the spatial loop.
        # M80 divides every production grid, so no tail or zero MAC is needed.
        w0 = pl.load(weights, [0, 0], [80, 16], target_memory=pl.MemorySpace.Mat)
        w1 = pl.load(weights, [80, 0], [80, 16], target_memory=pl.MemorySpace.Mat)
        w2 = pl.load(weights, [160, 0], [80, 16], target_memory=pl.MemorySpace.Mat)
        w3 = pl.load(weights, [240, 0], [80, 16], target_memory=pl.MemorySpace.Mat)
        w4 = pl.load(weights, [320, 0], [80, 16], target_memory=pl.MemorySpace.Mat)
        b0 = pl.load(biases, [0, 0], [1, 16])
        b1 = pl.load(biases, [0, 16], [1, 16])
        b2 = pl.load(biases, [0, 32], [1, 16])
        b3 = pl.load(biases, [0, 48], [1, 16])
        b4 = pl.load(biases, [0, 64], [1, 16])
        for work in pl.range(pl.tile.get_block_idx(), pixels // 80, pl.tile.get_block_num()):
            row = work * 80
            a = pl.load(source, [row, 0], [80, 80], target_memory=pl.MemorySpace.Mat)
            left = pl.tile.move(
                a, pl.MemorySpace.Left, blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.row_major
            )
            # Explicit cache variables select resident operands without dynamic
            # tuple indexing; the inline helper owns the shared epilogue.
            rounded = project_panel(left, w0, b0)
            destination = pl.store(rounded, [row, 0], destination)
            rounded = project_panel(left, w1, b1)
            destination = pl.store(rounded, [row, 16], destination)
            rounded = project_panel(left, w2, b2)
            destination = pl.store(rounded, [row, 32], destination)
            rounded = project_panel(left, w3, b3)
            destination = pl.store(rounded, [row, 48], destination)
            rounded = project_panel(left, w4, b4)
            destination = pl.store(rounded, [row, 64], destination)
        return output

    return convolution


def _classifier_geometry_supported(plan: ConvPlan) -> bool:
    return (
        plan.height == plan.width
        and plan.height in (20, 40, 80)
        and plan.cin == plan.cout == plan.input_row_stride == 80
        and plan.kernel == plan.stride == 1
        and plan.padding == plan.source_offset == 0
        and not plan.silu
    )


@dataclass(frozen=True)
class ClassifierProjectionPlan(ConvPlan):
    """Compact public PyPTO implementation of the three classifier projections."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if not _classifier_geometry_supported(self) or self.four_way_accumulation:
            raise ValueError("public classifier projection requires its fixed dense 80-channel profile")

    def metadata(self, offset: int = 0, *, weight_l2_offset: int = 0) -> torch.Tensor:
        if (
            type(offset) is not int
            or offset != 0
            or type(weight_l2_offset) is not int
            or weight_l2_offset != 0
        ):
            raise ValueError("public classifier projection requires resident views without metadata offsets")
        return super().metadata()

    def jit_kernel(self, output_padding: int = 0) -> Any:
        if type(output_padding) is not int or output_padding != 0:
            raise ValueError("public classifier output guards must use a resident tensor view")
        return pointwise_projection(self.oh * self.ow)


def select_classifier_projection(plan: ConvPlan, *, weight_l2_bypass: bool) -> ConvPlan:
    if not weight_l2_bypass and _classifier_geometry_supported(plan):
        return ClassifierProjectionPlan(**asdict(replace(plan, four_way_accumulation=False)))
    return plan
