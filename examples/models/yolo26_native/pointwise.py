# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Public ordered 1x1 convolution for the 160-square CI48/CO64 profile."""

from dataclasses import asdict, dataclass
from typing import Any

import pypto.language as pl
import torch
from pypto.library import fp16_silu_constants

from .operators import ConvPlan
from .pointwise32 import ordered_pointwise_32


def ordered_pointwise_convolution() -> Any:
    """Reuse K192 Left and N32 weights across two ordered output panels."""
    rows = 160
    activate = fp16_silu_constants(rows)

    @pl.jit.inline
    def ordered_products(
        left: pl.Tile[[rows, 192], pl.FP16],
        w0: pl.Tile[[192, 32], pl.FP16],
        w1: pl.Tile[[192, 32], pl.FP16],
        w2: pl.Tile[[192, 32], pl.FP16],
        w3: pl.Tile[[192, 32], pl.FP16],
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

    @pl.jit.incore
    def compute(
        metadata: pl.Tensor[[32], pl.FP32],
        x: pl.Tensor[[1228800], pl.FP16],
        weight: pl.Tensor[[49152], pl.FP16],
        bias: pl.Tensor[[64], pl.FP32],
        output: pl.Out[pl.Tensor[[1638400], pl.FP16]],
        zero: pl.Tensor[[16], pl.FP16],
    ):
        source = pl.tensor.view(x, [25600, 48])
        weights = pl.tensor.view(weight, [1536, 32])
        biases = pl.tensor.view(bias, [1, 64])
        destination = pl.tensor.view(output, [25600, 64])
        w0 = pl.load(weights, [0, 0], [192, 32], target_memory=pl.MemorySpace.Mat)
        w1 = pl.load(weights, [192, 0], [192, 32], target_memory=pl.MemorySpace.Mat)
        w2 = pl.load(weights, [384, 0], [192, 32], target_memory=pl.MemorySpace.Mat)
        w3 = pl.load(weights, [576, 0], [192, 32], target_memory=pl.MemorySpace.Mat)
        w4 = pl.load(weights, [768, 0], [192, 32], target_memory=pl.MemorySpace.Mat)
        w5 = pl.load(weights, [960, 0], [192, 32], target_memory=pl.MemorySpace.Mat)
        w6 = pl.load(weights, [1152, 0], [192, 32], target_memory=pl.MemorySpace.Mat)
        w7 = pl.load(weights, [1344, 0], [192, 32], target_memory=pl.MemorySpace.Mat)
        b0 = pl.load(biases, [0, 0], [1, 16])
        b1 = pl.load(biases, [0, 16], [1, 16])
        b2 = pl.load(biases, [0, 32], [1, 16])
        b3 = pl.load(biases, [0, 48], [1, 16])
        zero_v = pl.tile.full([rows, 16], pl.FP32, 0.0)
        offset_v = pl.tile.full([rows, 16], pl.FP32, 15.0 / 2048)
        scale_v = pl.tile.full([rows, 16], pl.FP32, 8192.0)
        one_v = pl.tile.full([rows, 16], pl.FP32, 1.0)
        epsilon_v = pl.tile.full([rows, 16], pl.FP32, 0.00000011920928955078125)
        if pl.tile.get_block_num() == 1:
            for iteration in pl.pipeline(160, stage=2):
                work = iteration
                row = work * rows
                left = pl.tile.create(
                    [rows, 192],
                    pl.FP16,
                    target_memory=pl.MemorySpace.Left,
                    blayout=pl.TileLayout.row_major,
                    slayout=pl.TileLayout.row_major,
                )
                for block in pl.range(0, 3):
                    source_panel = pl.load(
                        source, [row, block * 16], [rows, 16], target_memory=pl.MemorySpace.Mat
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
                low, high = ordered_products(left, w4, w5, w6, w7)
                value = pl.add(low, pl.tile.col_expand(low, b2))
                result = pl.cast(value, pl.FP16, mode="none")
                destination = pl.store(
                    activate(result, zero_v, offset_v, scale_v, one_v, epsilon_v), [row, 32], destination
                )
                value = pl.add(high, pl.tile.col_expand(high, b3))
                result = pl.cast(value, pl.FP16, mode="none")
                destination = pl.store(
                    activate(result, zero_v, offset_v, scale_v, one_v, epsilon_v), [row, 48], destination
                )
        else:
            for work in pl.range(pl.tile.get_block_idx(), 25600 // rows, pl.tile.get_block_num()):
                row = work * rows
                left = pl.tile.create(
                    [rows, 192],
                    pl.FP16,
                    target_memory=pl.MemorySpace.Left,
                    blayout=pl.TileLayout.row_major,
                    slayout=pl.TileLayout.row_major,
                )
                for block in pl.range(0, 3):
                    source_panel = pl.load(
                        source, [row, block * 16], [rows, 16], target_memory=pl.MemorySpace.Mat
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
                low, high = ordered_products(left, w4, w5, w6, w7)
                value = pl.add(low, pl.tile.col_expand(low, b2))
                result = pl.cast(value, pl.FP16, mode="none")
                destination = pl.store(
                    activate(result, zero_v, offset_v, scale_v, one_v, epsilon_v), [row, 32], destination
                )
                value = pl.add(high, pl.tile.col_expand(high, b3))
                result = pl.cast(value, pl.FP16, mode="none")
                destination = pl.store(
                    activate(result, zero_v, offset_v, scale_v, one_v, epsilon_v), [row, 48], destination
                )
        return output

    return compute


def _ordered_pointwise_geometry(plan: ConvPlan) -> bool:
    return (
        plan.height == plan.width == 160
        and plan.cin == plan.input_row_stride == 48
        and plan.cout == 64
        and plan.kernel == plan.stride == 1
        and plan.padding == plan.source_offset == 0
        and plan.silu
        and plan.four_way_accumulation
    )


@dataclass(frozen=True)
class PublicOrderedPointwisePlan(ConvPlan):
    """Fixed ordered reduction with shared Left, L1 weights and UB constants."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if not _ordered_pointwise_geometry(self):
            raise ValueError("public ordered pointwise requires 160x160 CI48/CO64 with SiLU")

    @property
    def weight_size(self) -> int:
        return 49152

    def pack_weight(self, weight: torch.Tensor) -> torch.Tensor:
        if list(weight.shape) != [64, 48, 1, 1] or weight.device.type != "cpu":
            raise ValueError("public ordered pointwise requires CPU [64,48,1,1] weights")
        dense = weight.half().reshape(64, 48)
        packed = torch.zeros((2, 4, 192, 32), dtype=torch.float16)
        # One live K row per fractal preserves each scalar partial's order.
        for panel in range(2):
            for partial in range(4):
                for term in range(12):
                    channel = term * 4 + partial
                    packed[panel, partial, term * 16 + channel % 16] = dense[
                        panel * 32 : (panel + 1) * 32, channel
                    ]
        return packed.reshape(-1)

    def metadata(self, offset: int = 0, *, weight_l2_offset: int = 0) -> torch.Tensor:
        if (
            type(offset) is not int
            or offset != 0
            or type(weight_l2_offset) is not int
            or weight_l2_offset != 0
        ):
            raise ValueError("public ordered pointwise requires resident views without metadata offsets")
        return super().metadata()

    def jit_kernel(self, output_padding: int = 0) -> Any:
        if type(output_padding) is not int or output_padding != 0:
            raise ValueError("public ordered pointwise guards must use a resident tensor view")
        return ordered_pointwise_convolution()


def _ordered_pointwise_32_geometry(plan: ConvPlan) -> bool:
    return (
        plan.height == plan.width == 160
        and plan.cin == plan.input_row_stride == 32
        and plan.cout == 32
        and plan.kernel == plan.stride == 1
        and plan.padding == plan.source_offset == 0
        and plan.silu
        and plan.four_way_accumulation
    )


@dataclass(frozen=True)
class PublicOrderedPointwise32Plan(ConvPlan):
    """N32 ordered pointwise with single-core pipeline and general fallback."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if not _ordered_pointwise_32_geometry(self):
            raise ValueError("public N32 pointwise requires 160x160 CI32/CO32 with SiLU")

    @property
    def weight_size(self) -> int:
        return 16384

    def pack_weight(self, weight: torch.Tensor) -> torch.Tensor:
        if list(weight.shape) != [32, 32, 1, 1] or weight.device.type != "cpu":
            raise ValueError("public N32 pointwise requires CPU [32,32,1,1] weights")
        dense = weight.half().reshape(32, 32)
        packed = torch.zeros((4, 128, 32), dtype=torch.float16)
        for partial in range(4):
            for term in range(8):
                channel = term * 4 + partial
                packed[partial, term * 16 + channel % 16, :] = dense[:, channel]
        return packed.reshape(-1)

    def metadata(self, offset: int = 0, *, weight_l2_offset: int = 0) -> torch.Tensor:
        if (
            type(offset) is not int
            or offset != 0
            or type(weight_l2_offset) is not int
            or weight_l2_offset != 0
        ):
            raise ValueError("public N32 pointwise requires resident views without metadata offsets")
        return super().metadata()

    def jit_kernel(self, output_padding: int = 0) -> Any:
        if type(output_padding) is not int or output_padding != 0:
            raise ValueError("public N32 pointwise guards must use a resident tensor view")
        return ordered_pointwise_32()


def select_ordered_pointwise(plan: ConvPlan, *, weight_l2_bypass: bool) -> ConvPlan:
    if not weight_l2_bypass:
        if _ordered_pointwise_32_geometry(plan):
            return PublicOrderedPointwise32Plan(**asdict(plan))
        if _ordered_pointwise_geometry(plan):
            return PublicOrderedPointwisePlan(**asdict(plan))
    return plan
