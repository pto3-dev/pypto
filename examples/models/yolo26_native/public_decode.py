# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Public PyPTO box/sigmoid decoding with the native FP16 rounding contract."""

from dataclasses import dataclass
from typing import Any

import pypto.language as pl
import torch

from .detection import DecodePlan


def _panel_rows(plan: DecodePlan) -> int:
    pixels = plan.height * plan.width
    return 8 if pixels % 8 == 0 else 1


def decode_kernel(plan: DecodePlan) -> Any:
    """Compose existing public arithmetic, scalar access, assembly and gather."""
    height, width, stride = plan.height, plan.width, float(plan.stride)
    pixels, offset, total = height * width, plan.anchor_offset, plan.total_anchors
    rows = _panel_rows(plan)
    index_count = (rows + 1) * 96
    nb, nc, no = pixels * 16, pixels * 80, total * 96

    def panel(count: int) -> Any:
        @pl.jit.inline
        def decode_panel(
            first: pl.Scalar[pl.INDEX],
            positions: pl.Tile[[count, 96], pl.INT32],
            boxes: pl.Tensor[[nb], pl.FP16],
            scores: pl.Tensor[[nc], pl.FP16],
            output: pl.Tensor[[no], pl.FP32],
        ) -> None:
            box_view = pl.tensor.view(boxes, [pixels, 16])
            score_view = pl.tensor.view(scores, [pixels, 80])
            output_view = pl.tensor.view(output, [total, 96])
            box = pl.cast(pl.load(box_view, [first, 0], [count, 16]), pl.FP32, mode="none")
            score = pl.cast(pl.load(score_view, [first, 0], [count, 80]), pl.FP32, mode="none")
            zero = pl.tile.full([count, 80], pl.FP32, 0.0)
            one = pl.tile.full([count, 80], pl.FP32, 1.0)
            negative = pl.sub(zero, score)
            exponential = pl.exp(negative)
            denominator = pl.add(exponential, one)
            sigmoid = pl.div(one, denominator)
            narrowed = pl.cast(sigmoid, pl.FP16, mode="none")
            probabilities = pl.cast(narrowed, pl.FP32, mode="none")
            # Aligned insertion puts 80 scores at column 16. Gather shifts
            # them to column 4 and selects the zero suffix for record padding.
            packed = pl.tile.full([count, 112], pl.FP32, 0.0)
            packed = pl.tile.assemble(packed, probabilities, [0, 16])
            for local in pl.range(count):
                # Every half cast is an observable native coordinate rounding.
                ax = pl.cast(
                    pl.cast(pl.cast(pl.cast((first + local) % width, pl.INT32), pl.FP32) + 0.5, pl.FP16),
                    pl.FP32,
                )
                ay = pl.cast(
                    pl.cast(pl.cast(pl.cast((first + local) // width, pl.INT32), pl.FP32) + 0.5, pl.FP16),
                    pl.FP32,
                )
                left = pl.cast(pl.cast(ax - pl.tile.read(box, [local, 0]), pl.FP16), pl.FP32)
                top = pl.cast(pl.cast(ay - pl.tile.read(box, [local, 1]), pl.FP16), pl.FP32)
                right = pl.cast(pl.cast(ax + pl.tile.read(box, [local, 2]), pl.FP16), pl.FP32)
                bottom = pl.cast(pl.cast(ay + pl.tile.read(box, [local, 3]), pl.FP16), pl.FP32)
                pl.tile.write(packed, [local, 0], pl.cast(pl.cast(left * stride, pl.FP16), pl.FP32))
                pl.tile.write(packed, [local, 1], pl.cast(pl.cast(top * stride, pl.FP16), pl.FP32))
                pl.tile.write(packed, [local, 2], pl.cast(pl.cast(right * stride, pl.FP16), pl.FP32))
                pl.tile.write(packed, [local, 3], pl.cast(pl.cast(bottom * stride, pl.FP16), pl.FP32))
            decoded = pl.gather(packed, index=positions)
            pl.tensor.assemble(output_view, decoded, [offset + first, 0])

        return decode_panel

    decode_full = panel(rows)
    decode_one = panel(1)

    @pl.jit.incore
    def public_decode(
        metadata: pl.Tensor[[index_count], pl.INT32],
        boxes: pl.Tensor[[nb], pl.FP16],
        scores: pl.Tensor[[nc], pl.FP16],
        output: pl.Out[pl.Tensor[[no], pl.FP32]],
    ):
        # Match the native panel choice for the actual logical core grid.
        # Load immutable indices once per core, outside its panel loop.
        index_view = pl.tensor.view(metadata, [rows + 1, 96])
        if pl.tile.get_block_num() <= pixels // rows:
            positions = pl.load(index_view, [0, 0], [rows, 96])
            for block in pl.range(pl.tile.get_block_idx(), pixels // rows, pl.tile.get_block_num()):
                decode_full(block * rows, positions, boxes, scores, output)
        else:
            positions_one = pl.load(index_view, [rows, 0], [1, 96])
            for pixel in pl.range(pl.tile.get_block_idx(), pixels, pl.tile.get_block_num()):
                decode_one(pixel, positions_one, boxes, scores, output)
        return output

    return public_decode


@dataclass(frozen=True)
class PublicDecodePlan(DecodePlan):
    """Existing four-tensor decode ABI with immutable element-index metadata."""

    def metadata(self) -> torch.Tensor:
        """Pack full-panel and one-anchor index tables for setup-time sharing."""
        indices = []
        for count in (_panel_rows(self), 1):
            for row in range(count):
                for column in range(96):
                    source_column = column if column < 4 else column + 12
                    indices.append(row * 112 + source_column)
        return torch.tensor(indices, dtype=torch.int32)

    def jit_kernel(self) -> Any:
        return decode_kernel(self)
