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
from typing import Any

import pypto.language as pl
import torch
from pypto.library import fp16_softmax_row, fp32_softmax_tile


def _softmax_kernel(cols: int) -> Any:
    normalize = fp32_softmax_tile(16, cols)

    @pl.jit.incore
    def softmax(x: pl.Tensor[[16, cols], pl.FP32], output: pl.Out[pl.Tensor[[16, cols], pl.FP32]]):
        a = pl.load(x, [0, 0], [16, cols])
        value = normalize(a)
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
        """Compile compensated row arithmetic exclusively through public PyPTO."""
        normalize = fp16_softmax_row(16)

        @pl.jit.incore
        def softmax(
            metadata: pl.Tensor[[32], pl.FP32],
            scores: pl.Tensor[[256], pl.FP16],
            output: pl.Out[pl.Tensor[[256], pl.FP16]],
        ):
            matrix = pl.tensor.view(scores, [16, 16])
            destination = pl.tensor.view(output, [16, 16])
            for row in pl.range(pl.tile.get_block_idx(), 16, pl.tile.get_block_num()):
                value = pl.load(matrix, [row, 0], [1, 16])
                normalized = normalize(value)
                destination = pl.store(normalized, [row, 0], destination)
            return output

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
        normalize = fp32_softmax_tile(16, 400)

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
                normalized = normalize(value)
                rounded = pl.cast(normalized, pl.FP16, mode="none")
                destination = pl.store(rounded, [work * 16, 0], destination)
            return output

        return softmax
