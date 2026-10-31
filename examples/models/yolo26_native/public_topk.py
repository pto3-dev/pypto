# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Public anchor-pruning component for YOLO26's exact two-stage TopK.

The aligned 304-pair intermediate is internal; consumers use exactly its first
300 anchors. This component does not replace the final 300x6 detection kernel.
"""

from dataclasses import dataclass
from typing import Any

import pypto.language as pl
import torch

from .detection import TopKPlan

# One flattened class-index table, then 32 maxima positions and anchor IDs.
_MAXIMA_OFFSET = 32 * 80
_LABELS_OFFSET = _MAXIMA_OFFSET + 32


def _expanded_class_maxima(rows: int) -> Any:
    @pl.jit.inline
    def class_maxima(
        records: pl.Tile[[rows, 96], pl.FP32],
        positions: pl.Tile[[1, rows * 80], pl.INT32],
    ):
        # Boxes and twelve padding fields must never enter anchor selection.
        # Gather only the eighty probabilities, then expose rows without a copy.
        # Explicit Tile gather keeps this entire component in the Tile domain.
        # A1 needs two INT32 byte-offset scratch words per FP32 output value.
        offsets = pl.create_tile([1, rows * 160], pl.INT32)
        gathered = pl.tile.gather(records, positions, offsets)
        classes = pl.tile.reshape(gathered, [rows, 80])
        scratch = pl.create_tile([rows, 80], pl.FP32)
        maxima = pl.tile.row_max(classes, scratch)
        # Reduction columns are packed. Expand through the public broadcast
        # instead of treating their storage as an ordinary aligned row Tile.
        expanded = pl.tile.full([rows, 16], pl.FP32, 0.0)
        return pl.tile.row_expand(expanded, maxima)

    return class_maxima


def _anchor_prefix(anchors: int) -> Any:
    """Build inline anchor pruning for reuse inside the complete TopK kernel."""
    TopKPlan(anchors)
    panels, tail = divmod(anchors, 32)
    padded = (anchors + 31) // 32 * 32
    table_count = _LABELS_OFFSET + padded
    full_maxima = _expanded_class_maxima(32)
    tail_maxima = _expanded_class_maxima(8)

    @pl.jit.inline
    def merge_panel(
        current: pl.Tile[[1, 608], pl.FP32],
        values: pl.Tile[[1, 32], pl.FP32],
        labels: pl.Tile[[1, 32], pl.INT32],
    ):
        # Identity extraction reinterprets the immutable label words. It avoids
        # adding integer arithmetic or a model-specific index-generation op.
        indices = pl.tile.gather_mask(labels, pl.tile.MaskPattern.P1111, output_dtype=pl.UINT32)
        sorted_pairs = pl.tile.sort32(values, indices)
        scratch = pl.create_tile([1, 8 * (608 + 64)], pl.FP32)
        return pl.tile.mrgsort(current, sorted_pairs, tmp=scratch, output_cols=608)

    # Shape specialization happens in Python: a zero tail must not construct
    # an out-of-bounds partition even in an unreachable device branch.
    if tail:

        @pl.jit.inline
        def finish_tail(
            current: pl.Tile[[1, 608], pl.FP32],
            metadata: pl.Tensor[[table_count], pl.INT32],
            dense: pl.Tensor[[anchors * 96], pl.FP32],
        ):
            table = pl.tensor.view(metadata, [1, table_count])
            records = pl.tensor.view(dense, [anchors, 96])
            # Packed FP32 reduction columns need at least eight physical rows.
            # Reduce groups of eight real rows; fewer than eight residual rows
            # use scalar maxima instead of an unallocatable 1x1 packed Tile.
            tail_positions = pl.load(table, [0, 0], [1, 640])
            tail_values = pl.tile.full([1, 32], pl.FP32, -1.0)
            for group in pl.range(tail // 8):
                tail_records = pl.load(records, [panels * 32 + group * 8, 0], [8, 96])
                tail_expanded = tail_maxima(tail_records, tail_positions)
                # Pack at most 24 maxima through existing scalar Tile access.
                # This avoids offset-gather workspace and dynamic panel insertion
                # for the small tail; the large main panels stay vectorized.
                for row in pl.range(8):
                    maximum = pl.tile.read(tail_expanded, [row, 0])
                    pl.tile.write(tail_values, [0, group * 8 + row], maximum)
            for local in pl.range(tail // 8 * 8, tail):
                residual_record = pl.load(records, [panels * 32 + local, 0], [1, 96])
                maximum = pl.tile.read(residual_record, [0, 4])
                for column, (best,) in pl.range(5, 84, init_values=(maximum,)):
                    value = pl.tile.read(residual_record, [0, column])
                    if value > best:
                        larger = pl.yield_(value)
                    else:
                        larger = pl.yield_(best)
                    row_maximum = pl.yield_(larger)
                pl.tile.write(tail_values, [0, local], row_maximum)
            labels = pl.load(table, [0, _LABELS_OFFSET + panels * 32], [1, 32])
            return merge_panel(current, tail_values, labels)
    else:

        @pl.jit.inline
        def finish_tail(
            current: pl.Tile[[1, 608], pl.FP32],
            metadata: pl.Tensor[[table_count], pl.INT32],
            dense: pl.Tensor[[anchors * 96], pl.FP32],
        ):
            return current

    @pl.jit.inline
    def select_anchors(
        metadata: pl.Tensor[[table_count], pl.INT32],
        dense: pl.Tensor[[anchors * 96], pl.FP32],
    ):
        table = pl.tensor.view(metadata, [1, table_count])
        records = pl.tensor.view(dense, [anchors, 96])
        class_positions = pl.load(table, [0, 0], [1, _MAXIMA_OFFSET])
        maximum_positions = pl.load(table, [0, _MAXIMA_OFFSET], [1, 32])
        # Decoded probabilities are in [0,1], as in the native heap contract.
        # -1 seeds cannot displace a real anchor, even when every score is zero.
        seed = pl.tile.full([1, 608], pl.FP32, -1.0)
        for block, (current,) in pl.range(panels, init_values=(seed,)):
            panel_records = pl.load(records, [block * 32, 0], [32, 96])
            panel_expanded = full_maxima(panel_records, class_positions)
            panel_offsets = pl.create_tile([1, 64], pl.INT32)
            panel_values = pl.tile.gather(panel_expanded, maximum_positions, panel_offsets)
            panel_labels = pl.load(table, [0, _LABELS_OFFSET + block * 32], [1, 32])
            selected = merge_panel(current, panel_values, panel_labels)
            result = pl.yield_(selected)
        finished = finish_tail(result, metadata, dense)
        return finished

    return select_anchors


def anchor_prefix_kernel(anchors: int) -> Any:
    """Expose the inline stage for independent first-stage qualification.

    The complete detector can call the inline helper and retain its result in
    UB. This wrapper's one store is only the intermediate validation boundary.
    """
    select_anchors = _anchor_prefix(anchors)
    table_count = _LABELS_OFFSET + (anchors + 31) // 32 * 32

    @pl.jit.incore
    def public_anchor_prefix(
        metadata: pl.Tensor[[table_count], pl.INT32],
        dense: pl.Tensor[[anchors * 96], pl.FP32],
        prefix: pl.Out[pl.Tensor[[608], pl.FP32]],
    ):
        result = select_anchors(metadata, dense)
        destination = pl.tensor.view(prefix, [1, 608])
        pl.store(result, [0, 0], destination)
        return prefix

    return public_anchor_prefix


@dataclass(frozen=True)
class AnchorPrefixPlan:
    """Internal first-stage candidate; one image and one logical core per call."""

    anchors: int

    def __post_init__(self) -> None:
        TopKPlan(self.anchors)

    def metadata(self) -> torch.Tensor:
        """Build immutable gather positions and monotonic original-anchor IDs."""
        positions = [row * 96 + 4 + column for row in range(32) for column in range(80)]
        positions.extend(row * 16 for row in range(32))
        positions.extend(range(self.anchors))
        positions.extend([0x7FFFFFFF] * (-self.anchors % 32))
        return torch.tensor(positions, dtype=torch.int32)

    def jit_kernel(self) -> Any:
        return anchor_prefix_kernel(self.anchors)
