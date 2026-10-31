# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Public dense casts and FP32-scaled attention conversion."""

from dataclasses import dataclass
from typing import Any

import pypto.language as pl

from .layout import _CASTS, VectorOp, VectorPlan


def conversion_kernel(plan: VectorPlan) -> Any:
    """Preserve the native conversion ABI and FP32 multiply boundary."""
    if plan.operation not in _CASTS:
        raise ValueError(f"public conversion requires a dense conversion operation, got {plan.operation}")
    length = plan.height
    columns = 4096 if length >= 4096 else 256
    full, tail = divmod(length, columns)
    input_type = pl.FP32 if plan.operation == VectorOp.FLOAT_TO_HALF else pl.FP16
    output_type = pl.FP32 if plan.operation == VectorOp.HALF_TO_FLOAT else pl.FP16

    def panel(count: int) -> Any:
        if plan.operation == VectorOp.FLOAT_TO_HALF:

            @pl.jit.inline
            def transform(values: pl.Tile[[1, count], pl.FP32], scale: pl.Scalar[pl.FP32]):
                return pl.cast(values, pl.FP16, mode="none")

        elif plan.operation == VectorOp.HALF_TO_FLOAT:

            @pl.jit.inline
            def transform(values: pl.Tile[[1, count], pl.FP16], scale: pl.Scalar[pl.FP32]):
                widened = pl.cast(values, pl.FP32, mode="none")
                return pl.mul(widened, scale)

        else:

            @pl.jit.inline
            def transform(values: pl.Tile[[1, count], pl.FP16], scale: pl.Scalar[pl.FP32]):
                widened = pl.cast(values, pl.FP32, mode="none")
                scaled = pl.mul(widened, scale)
                return pl.cast(scaled, pl.FP16, mode="none")

        @pl.jit.inline
        def convert_panel(
            first: pl.Scalar[pl.INDEX],
            scale: pl.Scalar[pl.FP32],
            source: pl.Tensor[[length], input_type],
            destination: pl.Tensor[[length], output_type],
        ) -> None:
            source_view = pl.tensor.view(source, [1, length])
            output_view = pl.tensor.view(destination, [1, length])
            values = pl.load(source_view, [0, first], [1, count])
            result = transform(values, scale)
            pl.store(result, [0, first], output_view)

        return convert_panel

    convert_full = panel(columns)
    convert_tail = panel(tail if tail else 256)
    tail_first = full * columns if tail else 0

    @pl.jit.incore
    def dense_conversion(
        metadata: pl.Tensor[[32], pl.FP32],
        source: pl.Tensor[[length], input_type],
        second: pl.Tensor[[16], pl.FP16],
        output: pl.Out[pl.Tensor[[length], output_type]],
    ):
        constants = pl.load(pl.tensor.view(metadata, [1, 32]), [0, 0], [1, 32])
        scale = pl.tile.read(constants, [0, 15])
        # One logical block owns a batch image. Public pipeline provides its
        # two-stage panels; the multi-block path owns disjoint full panels.
        if pl.tile.get_block_num() == 1:
            for block in pl.pipeline(full, stage=2):
                convert_full(block * columns, scale, source, output)
        else:
            for block in pl.range(pl.tile.get_block_idx(), full, pl.tile.get_block_num()):
                convert_full(block * columns, scale, source, output)
        if tail > 0:
            if pl.tile.get_block_idx() == 0:
                convert_tail(tail_first, scale, source, output)
        return output

    return dense_conversion


@dataclass(frozen=True)
class PublicConversionPlan(VectorPlan):
    """Dense vector ABI implemented using public cast, multiply and DMA."""

    def jit_kernel(self) -> Any:
        return conversion_kernel(self)
