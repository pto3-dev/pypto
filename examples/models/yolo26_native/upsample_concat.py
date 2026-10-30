# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Public nearest-neighbor/concat fusion with no intermediate GM tensor."""

from collections import Counter
from dataclasses import asdict, dataclass, replace
from typing import Any

import pypto.language as pl
import torch

from .layout import ConcatPlan, VectorOp, VectorPlan


@dataclass(frozen=True)
class UpsampleConcatPlan:
    """Join a 2x nearest-neighbor source and an aligned same-size skip source."""

    height: int
    width: int
    up_channels: int
    skip_channels: int
    up_stride: int
    skip_stride: int
    up_offset: int = 0
    skip_offset: int = 0
    upsample_first: bool = True

    def __post_init__(self) -> None:
        if type(self.upsample_first) is not bool:
            raise TypeError("upsample_first must be a boolean")
        # Reuse the existing geometry, channel-view and indexing contracts.
        VectorPlan(
            VectorOp.UPSAMPLE,
            2 * self.height,
            2 * self.width,
            self.up_channels,
            self.up_stride,
            self.up_channels,
            self.up_offset,
            input_height=self.height,
            input_width=self.width,
        )
        up = (self.up_channels, self.up_channels, 0)
        skip = (self.skip_channels, self.skip_stride, self.skip_offset)
        ConcatPlan(2 * self.height, 2 * self.width, (up, skip) if self.upsample_first else (skip, up))
        if max(self.up_channels, self.skip_channels) > 1024:
            raise ValueError("public upsample-concat supports at most 1024 channels per source")
        if (self.up_stride - self.up_channels) // 16 > 65535:
            raise ValueError("public upsample-concat requires bounded input DMA row gaps")

    @property
    def rows(self) -> int:
        """Keep every source panel inside one input row and a bounded UB bank."""
        rows = 16
        while rows > 1 and (self.width % rows or rows * (self.up_channels + 4 * self.skip_channels) > 8192):
            rows //= 2
        return rows

    @property
    def input_sizes(self) -> tuple[int, int]:
        return (self.height * self.width * self.up_stride, 4 * self.height * self.width * self.skip_stride)

    @property
    def output_size(self) -> int:
        return 4 * self.height * self.width * (self.up_channels + self.skip_channels)

    def metadata(self) -> torch.Tensor:
        """Encode only static geometry; activation values never visit the host."""
        return torch.tensor(
            [
                self.height,
                self.width,
                self.up_channels,
                self.skip_channels,
                self.up_stride,
                self.skip_stride,
                self.up_offset,
                self.skip_offset,
                int(self.upsample_first),
                self.rows,
                *([0] * 22),
            ],
            dtype=torch.float32,
        )

    def jit_kernel(self) -> Any:
        """Use zero-copy row views, vector broadcast and strided output stores."""
        ih, iw = self.height, self.width
        ca, cb, rows = self.up_channels, self.skip_channels, self.rows
        sa, sb = self.up_stride, self.skip_stride
        oa, ob = self.up_offset, self.skip_offset
        co, ow = ca + cb, 2 * iw
        pixels, output_pixels = ih * iw, 4 * ih * iw
        chunks = iw // rows
        work_items = ih * chunks
        na, nb = self.input_sizes
        no = self.output_size
        ac, bc = (0, ca) if self.upsample_first else (cb, 0)

        @pl.jit.inline
        def join_panel(
            work: pl.Scalar[pl.INDEX],
            low: pl.Tensor[[na], pl.FP16],
            skip: pl.Tensor[[nb], pl.FP16],
            destination: pl.Tensor[[no], pl.FP16],
        ) -> None:
            low_view = pl.tensor.view(low, [pixels, sa])
            skip_view = pl.tensor.view(skip, [output_pixels, sb])
            dst = pl.tensor.view(destination, [output_pixels, co])
            first = work // chunks * 2 * ow + work % chunks * 2 * rows
            source = pl.load(low_view, [work * rows, oa], [rows, ca])
            top = pl.load(skip_view, [first, ob], [2 * rows, cb])
            bottom = pl.load(skip_view, [first + ow, ob], [2 * rows, cb])
            # Static row slices alias the resident input panel. Broadcast stays
            # on PIPE_V; no UB concatenation copy or intermediate GM is needed.
            for row in pl.unroll(rows):
                value = pl.tile.slice(source, [1, ca], [row, 0])
                shape = pl.tile.create([2, ca], dtype=pl.FP16)
                pair = pl.tile.col_expand(shape, value)
                pl.store(pair, [first + row * 2, ac], dst)
                pl.store(pair, [first + ow + row * 2, ac], dst)
            pl.store(top, [first, bc], dst)
            pl.store(bottom, [first + ow, bc], dst)

        @pl.jit.incore
        def upsample_concat(
            metadata: pl.Tensor[[32], pl.FP32],
            low: pl.Tensor[[na], pl.FP16],
            skip: pl.Tensor[[nb], pl.FP16],
            output: pl.Out[pl.Tensor[[no], pl.FP16]],
        ):
            if pl.tile.get_block_num() == 1:
                for work in pl.pipeline(work_items, stage=2):
                    join_panel(work, low, skip, output)
            else:
                for work in pl.range(pl.tile.get_block_idx(), work_items, pl.tile.get_block_num()):
                    join_panel(work, low, skip, output)
            return output

        return upsample_concat


def _storage_key(tensor: Any) -> tuple[int, ...]:
    """Include all resident views in fanout checks, not just Python identities."""
    if hasattr(tensor, "_owner") and hasattr(tensor, "_id"):
        return (id(tensor._owner), tensor._id)
    # CPU graph doubles retain their root through parent; production resident
    # tensors instead share an owner/allocation ID even for offset views.
    while getattr(tensor, "parent", None) is not None:
        tensor = tensor.parent
    return (id(tensor),)


def _fusion_plan(up: Any, concat: Any, producer: Any, consumer: Any) -> UpsampleConcatPlan | None:
    if not isinstance(up, VectorPlan) or up.operation != VectorOp.UPSAMPLE:
        return None
    if not isinstance(concat, ConcatPlan) or len(concat.sources) != 2:
        return None
    if (up.height, up.width) != (concat.height, concat.width):
        return None
    if (up.destination_stride, up.destination_offset) != (up.channels, 0):
        return None
    if len(producer.arguments) != 4 or len(consumer.arguments) != 6 or producer.blocks != consumer.blocks:
        return None
    positions = [i for i in (0, 1) if consumer.arguments[i + 1] is producer.arguments[-1]]
    if len(positions) != 1:
        return None
    position = positions[0]
    if concat.sources[position] != (up.channels, up.channels, 0):
        return None
    cb, sb, ob = concat.sources[1 - position]
    if max(up.channels, cb) > 1024 or (up.source_stride - up.channels) // 16 > 65535:
        return None
    return UpsampleConcatPlan(
        up.input_height,
        up.input_width,
        up.channels,
        cb,
        up.source_stride,
        sb,
        up.source_offset,
        ob,
        position == 0,
    )


def fuse_upsample_concat(runner: Any) -> None:
    """Replace adjacent single-use routing pairs while preserving feature views.

    The original trace is already topologically ordered. No task is moved:
    replacing one adjacent pair keeps every other dependency and task intact.
    Count aliases across the full trace before deleting an intermediate writer.
    Debug features become channel views of the existing concat destination.
    """
    from .executor import Launch  # noqa: PLC0415

    reads = Counter(_storage_key(arg) for task in runner.trace for arg in task.arguments[:-1])
    writes = Counter(_storage_key(task.arguments[-1]) for task in runner.trace)
    result = []
    index = 0
    while index < len(runner.trace):
        producer = runner.trace[index]
        consumer = runner.trace[index + 1] if index + 1 < len(runner.trace) else None
        plan = _fusion_plan(
            runner._launch_plans.get(id(producer)),
            runner._launch_plans.get(id(consumer)),
            producer,
            consumer,
        )
        if plan is None:
            result.append(producer)
            index += 1
            continue
        temporary = producer.arguments[-1]
        key = _storage_key(temporary)
        views = [(k, v) for k, v in runner.features.items() if _storage_key(v.data) == key]
        # A distinct resident view needs offset-aware rebasing. Conservatively
        # leave it untouched; normal Feature.channel_slice retains data identity.
        if (
            reads[key] != 1
            or writes[key] != 1
            or any(
                v.data is not temporary
                or (v.height, v.width, v.row_stride) != (2 * plan.height, 2 * plan.width, plan.up_channels)
                or not 0 <= v.offset < v.offset + v.channels <= plan.up_channels
                for _, v in views
            )
        ):
            result.append(producer)
            index += 1
            continue
        destination = consumer.arguments[-1]
        skip = consumer.arguments[2 if plan.upsample_first else 1]
        low = producer.arguments[1]
        if _storage_key(destination) in {_storage_key(low), _storage_key(skip), key}:
            result.append(producer)
            index += 1
            continue
        kernel = runner._prepare(plan.jit_kernel(), repr(plan))
        metadata = runner.session.upload(plan.metadata())
        fused = Launch(kernel, (metadata, low, skip, destination), producer.blocks, "route:upsample-concat")
        result.append(fused)
        del runner._launch_plans[id(producer)]
        del runner._launch_plans[id(consumer)]
        runner._launch_plans[id(fused)] = plan
        channel = 0 if plan.upsample_first else plan.skip_channels
        for k, view in views:
            runner.features[k] = replace(
                view,
                data=destination,
                row_stride=plan.up_channels + plan.skip_channels,
                offset=channel + view.offset,
            )
        # Keep scalar audit data only, so removed Launch objects do not retain
        # otherwise-unused intermediate buffers through an audit reference.
        runner._upsample_concat_fusions.append(
            {
                "original_index": index,
                "plan": asdict(plan),
                "removed_gm_bytes": 4 * plan.height * plan.width * plan.up_channels * 4,
            }
        )
        index += 2
    runner.trace = result
