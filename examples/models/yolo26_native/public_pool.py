# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Public 5x5 max pooling with the existing four-Tensor vector ABI."""

from dataclasses import dataclass
from typing import Any

import pypto.language as pl
from pypto.library import fp16_max_pool2d_window

from .layout import VectorOp, VectorPlan


@dataclass(frozen=True)
class PublicPoolPlan(VectorPlan):
    """Compose a public spatial window without model-local C++ arithmetic."""

    kernel_size: int = 5

    def jit_kernel(self) -> Any:
        if self.operation != VectorOp.MAXPOOL:
            raise ValueError("public pooling requires a MAXPOOL plan")
        if (self.destination_stride - self.channels) // 16 > 65535:
            raise ValueError("public pooling requires a bounded destination DMA row gap")
        window = fp16_max_pool2d_window(
            self.height,
            self.width,
            self.channels,
            kernel_size=self.kernel_size,
            source_stride=self.source_stride,
            source_offset=self.source_offset,
        )
        pixels, width = self.height * self.width, self.width
        ni, nb, no = self.input_size, self.second_size, self.output_size
        source_stride, destination_stride = self.source_stride, self.destination_stride
        destination_offset = self.destination_offset

        @pl.jit.incore
        def max_pool(
            metadata: pl.Tensor[[32], pl.FP32],
            source: pl.Tensor[[ni], pl.FP16],
            second: pl.Tensor[[nb], pl.FP16],
            output: pl.Out[pl.Tensor[[no], pl.FP16]],
        ):
            source_rows = pl.tensor.view(source, [pixels, source_stride])
            output_rows = pl.tensor.view(output, [pixels, destination_stride])
            # Independent batch images see logical grid one. Multi-core calls
            # distribute pixels rather than reloading a window for each C16.
            for pixel in pl.range(pl.tile.get_block_idx(), pixels, pl.tile.get_block_num()):
                pooled = window(source_rows, pixel // width, pixel % width)
                pl.store(pooled, [pixel, destination_offset], output_rows)
            return output

        return max_pool
