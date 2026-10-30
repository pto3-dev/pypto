# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Static FP16 attention products with FP32 cube accumulation on 310P."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pypto.language as pl
import torch


def _pv400_kernel() -> Any:
    """Compute 80 rows per tile without changing the K16 accumulation order."""

    @pl.jit.incore
    def matrix(
        metadata: pl.Tensor[[32], pl.FP32],
        a: pl.Tensor[[160000], pl.FP16],
        b: pl.Tensor[[25600], pl.FP16],
        output: pl.Out[pl.Tensor[[25600], pl.FP16]],
    ):
        left = pl.tensor.view(a, [400, 400])
        right = pl.tensor.view(b, [400, 64])
        destination = pl.tensor.view(output, [400, 64])
        # Nested affine loops keep DMA offsets provably bounded on A1.
        # M=80 divides all 400 rows; K=16 preserves the native MAD order.
        for row_work in pl.range(pl.tile.get_block_idx(), 5, pl.tile.get_block_num()):
            for column_work in pl.range(0, 4):
                row = row_work * 80
                column = column_work * 16
                lhs = pl.load(left, [row, 0], [80, 16], target_memory=pl.MemorySpace.Mat)
                rhs = pl.load(right, [0, column], [16, 16], target_memory=pl.MemorySpace.Mat)
                acc = pl.matmul(lhs, rhs)
                for step in pl.range(1, 25):
                    lhs = pl.load(left, [row, step * 16], [80, 16], target_memory=pl.MemorySpace.Mat)
                    rhs = pl.load(right, [step * 16, column], [16, 16], target_memory=pl.MemorySpace.Mat)
                    acc = pl.matmul_acc(acc, lhs, rhs)
                rounded = pl.cast(acc, pl.FP16, mode="none")
                pl.store(rounded, [row, column], destination)
        return output

    return matrix


def _qk400_kernel() -> Any:
    """Reuse two Q panels per output row, with ordered K16 accumulation."""

    @pl.jit.incore
    def matrix(
        metadata: pl.Tensor[[32], pl.FP32],
        a: pl.Tensor[[12800], pl.FP16],
        b: pl.Tensor[[12800], pl.FP16],
        output: pl.Out[pl.Tensor[[160000], pl.FP16]],
    ):
        query = pl.tensor.view(a, [400, 32])
        key = pl.tensor.view(b, [400, 32])
        destination = pl.tensor.view(output, [400, 400])
        for row in pl.range(pl.tile.get_block_idx(), 25, pl.tile.get_block_num()):
            q0 = pl.load(query, [row * 16, 0], [16, 16], target_memory=pl.Mem.Mat)
            q1 = pl.load(query, [row * 16, 16], [16, 16], target_memory=pl.Mem.Mat)
            for column in pl.range(25):
                k0 = pl.load(key, [column * 16, 0], [16, 16], target_memory=pl.Mem.Mat)
                k1 = pl.load(key, [column * 16, 16], [16, 16], target_memory=pl.Mem.Mat)
                acc = pl.matmul(q0, pl.tile.transpose_view(k0))
                acc = pl.matmul_acc(acc, q1, pl.tile.transpose_view(k1))
                values = pl.tile.move(
                    acc, pl.Mem.Vec, blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.none_box
                )
                destination = pl.store(
                    pl.cast(values, pl.FP16, mode="none"), [row * 16, column * 16], destination
                )
        return output

    return matrix


@dataclass(frozen=True)
class MatrixPlan:
    """Aligned row-major product; optionally interpret B as [N,K]."""

    m: int
    n: int
    k: int
    transpose_b: bool = False
    linear_accumulation: bool = False
    experimental_public_qk: bool = False

    def __post_init__(self) -> None:
        if type(self.experimental_public_qk) is not bool:
            raise TypeError("experimental public QK must be a boolean")
        if type(self.linear_accumulation) is not bool:
            raise TypeError("linear accumulation must be a boolean")
        if self.experimental_public_qk and (
            self.m,
            self.n,
            self.k,
            self.transpose_b,
            self.linear_accumulation,
        ) != (400, 400, 32, True, False):
            raise ValueError("experimental public QK requires 400x400x32 transposed ordered product")
        if any(type(v) is not int or v <= 0 or v % 16 for v in (self.m, self.n, self.k)):
            raise ValueError("matrix dimensions must be positive multiples of sixteen")
        if max(self.m * self.k, self.k * self.n, self.m * self.n) > 0x7FFFFFFF:
            raise ValueError("matrix geometry exceeds the signed indexing range")

    def metadata(self) -> torch.Tensor:
        """Encode static geometry without reading activation values."""
        return torch.tensor([self.m, self.n, self.k, int(self.transpose_b)] + [0] * 28, dtype=torch.float32)

    def jit_kernel(self) -> Any:
        """Use public PyPTO for production PV; retain native entries for other profiles."""
        if self.experimental_public_qk and (
            self.m,
            self.n,
            self.k,
            self.transpose_b,
            self.linear_accumulation,
        ) == (400, 400, 32, True, False):
            return _qk400_kernel()
        if (self.m, self.n, self.k, self.transpose_b, self.linear_accumulation) == (
            400,
            64,
            400,
            False,
            False,
        ):
            return _pv400_kernel()
        na, nb, no = self.m * self.k, self.n * self.k, self.m * self.n
        source = Path(__file__).with_name("matmul_linear.cpp" if self.linear_accumulation else "matmul.cpp")

        @pl.jit.extern(core_type="aic", source=source)
        def matrix(
            metadata: pl.Tensor[[32], pl.FP32],
            a: pl.Tensor[[na], pl.FP16],
            b: pl.Tensor[[nb], pl.FP16],
            output: pl.Out[pl.Tensor[[no], pl.FP16]],
        ): ...

        return matrix
