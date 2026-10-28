# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Fixed-shape attention Softmax plans with explicit FP16 boundaries."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pypto.language as pl
import torch


def _softmax_kernel(cols: int) -> Any:
    @pl.jit.incore
    def softmax(x: pl.Tensor[[16, cols], pl.FP32], output: pl.Out[pl.Tensor[[16, cols], pl.FP32]]):
        a = pl.load(x, [0, 0], [16, cols])
        max_scratch = pl.tile.create([16, cols], dtype=pl.FP32)
        maximum = pl.tile.row_max(a, max_scratch)
        shifted = pl.sub(a, pl.tile.row_expand(a, maximum))
        exponent = pl.exp(shifted)
        sum_scratch = pl.tile.create([16, cols], dtype=pl.FP32)
        total = pl.tile.row_sum(exponent, sum_scratch)
        value = pl.div(exponent, pl.tile.row_expand(exponent, total))
        pl.store(value, [0, 0], output)
        return output

    return softmax


@dataclass(frozen=True)
class Softmax16Plan:
    """Evaluate a fixed 16x16 finite FP16 score matrix entirely on 310P."""

    def metadata(self) -> torch.Tensor:
        """Keep the native plan ABI uniform; shape is encoded in the kernel."""
        return torch.zeros(32, dtype=torch.float32)

    def jit_kernel(self) -> Any:
        """Return the public external entry for this fixed shape."""
        source = Path(__file__).with_name("softmax_pair.cpp")

        @pl.jit.extern(core_type="aic", source=source)
        def softmax(
            metadata: pl.Tensor[[32], pl.FP32],
            scores: pl.Tensor[[256], pl.FP16],
            output: pl.Out[pl.Tensor[[256], pl.FP16]],
        ): ...

        return softmax


@dataclass(frozen=True)
class Softmax400Plan:
    """Keep 400 attention rows and FP16 casts in one PyPTO device task.

    Each core processes disjoint 16-row blocks using the original FP32
    reduction shape and separately narrowed FP16 result. Batch execution
    supplies one independent image per core through the public batch worker.
    """

    def metadata(self) -> torch.Tensor:
        """Retain the uniform native plan ABI; geometry is fixed in the kernel."""
        return torch.zeros(32, dtype=torch.float32)

    def jit_kernel(self) -> Any:
        """Compile the persistent reduction through public PyPTO operators."""

        @pl.jit.incore
        def softmax(
            metadata: pl.Tensor[[32], pl.FP32],
            scores: pl.Tensor[[160000], pl.FP16],
            output: pl.Out[pl.Tensor[[160000], pl.FP16]],
        ):
            matrix = pl.tensor.view(scores, [400, 400])
            destination = pl.tensor.view(output, [400, 400])
            for work in pl.range(pl.tile.get_block_idx(), 25, pl.tile.get_block_num()):
                packed = pl.load(matrix, [work * 16, 0], [16, 400])
                value = pl.cast(packed, pl.FP32, mode="none")
                max_scratch = pl.tile.create([16, 400], dtype=pl.FP32)
                maximum = pl.tile.row_max(value, max_scratch)
                shifted = pl.sub(value, pl.tile.row_expand(value, maximum))
                exponent = pl.exp(shifted)
                sum_scratch = pl.tile.create([16, 400], dtype=pl.FP32)
                total = pl.tile.row_sum(exponent, sum_scratch)
                normalized = pl.div(exponent, pl.tile.row_expand(exponent, total))
                rounded = pl.cast(normalized, pl.FP16, mode="none")
                destination = pl.store(rounded, [work * 16, 0], destination)
            return output

        return softmax
