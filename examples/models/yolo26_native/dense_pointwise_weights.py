# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Constant phase-major weights and public-PyPTO resident L1 staging."""

from typing import Any

import pypto.language as pl
import torch

from .dense_pointwise_input import _dense_pointwise_phase_columns
from .operators import ConvPlan


def pack_dense_pointwise_weights(plan: ConvPlan, weight: torch.Tensor) -> torch.Tensor:
    """Pack constant CPU weights for dense four-phase Cube consumers.

    Like ConvPlan.pack_weight, this is one-time CPU preprocessing before
    upload. The returned contiguous FP16 Tensor has shape ``[4*Kp, co]``,
    where ``Kp=ceil(cin/64)*16``. Its element ``[p*Kp+k, n]`` is the original
    FP16 weight ``[n, 4*k+p, 0, 0]``; padded input/output channels are zero.
    It is a separate experimental layout, not the existing external-kernel ABI.

    Args:
        plan: Ordered 1x1 geometry with C16 input channels up to 512.
        weight: CPU floating weights with shape ``[cout, cin, 1, 1]``.

    Returns:
        Detached phase-major FP16 weights. The source is never modified.

    Raises:
        ValueError: Geometry, source shape, device or dtype is unsupported.
    """
    phase_columns = _dense_pointwise_phase_columns(plan)
    if (
        not isinstance(weight, torch.Tensor)
        or list(weight.shape) != [plan.cout, plan.cin, 1, 1]
        or weight.device.type != "cpu"
        or not weight.is_floating_point()
    ):
        raise ValueError("dense pointwise weights require CPU floating [cout, cin, 1, 1] storage")
    # Preserve the existing FP16 rounding boundary once. All later operations
    # only copy lanes; no reduction, quantization or runtime fallback occurs.
    source = weight.detach().to(torch.float16).reshape(plan.cout, plan.cin)
    # A caller may select another default device for inference. Constant
    # preprocessing must keep its destination on the validated CPU source.
    packed = torch.zeros((4, phase_columns, plan.co), dtype=torch.float16, device="cpu")
    for phase in range(4):
        packed[phase, : plan.cin // 4, : plan.cout] = source[:, phase::4].transpose(0, 1)
    return packed.reshape(4 * phase_columns, plan.co)


def dense_pointwise_weight_cache(
    plan: ConvPlan, columns: int | None = None, *, bank_bytes: int | None = None
) -> Any:
    """Load a bounded output-channel window of all four weight phases to L1.

    Args:
        plan: Geometry for the phase-major constant weight Tensor.
        columns: C16-aligned cache width; defaults to all padded outputs.
        bank_bytes: Optional byte budget for each of two Vec staging banks.
            None retains direct GM-to-Mat loads. The budget must be a static
            multiple of 512, at least Kp*16*2 and at most 32768 bytes. Staged
            columns divide the cache width, so every DMA is a full C16 panel.

    Returns:
        Inline helper accepting ``weights[4*Kp, co]`` and a bounded,
        C16-aligned first output column. It returns four disjoint
        ``[Kp, columns]`` Mat tiles, reusable across spatial windows.
        Consumers own the combined L1 budget with input images; the normal
        planner and native compilation still enforce aggregate capacity.

    Raises:
        ValueError: Cache width, L1 capacity or Vec bank budget is unsupported.
    """
    phase_rows = _dense_pointwise_phase_columns(plan)
    channels = plan.co
    columns = channels if columns is None else columns
    if type(columns) is not int or columns < 16 or columns % 16 or columns > channels:
        raise ValueError("dense pointwise weight cache requires a bounded static C16 width")
    if 4 * phase_rows * columns * 2 > 1048576:
        raise ValueError("dense pointwise weight cache exceeds the 1 MiB L1 budget")
    weight_rows = 4 * phase_rows

    if bank_bytes is not None:
        minimum = phase_rows * 16 * 2
        if type(bank_bytes) is not int or bank_bytes < minimum or bank_bytes > 32768 or bank_bytes % 512:
            raise ValueError(
                "dense pointwise weight bank requires a static 512-byte budget "
                f"in [{minimum},32768], got {bank_bytes!r}"
            )
        panel_columns = min(columns, bank_bytes // (phase_rows * 2) // 16 * 16)
        # A divisor gives fixed-shape transfers without a separate tail
        # pipeline or reading beyond a caller-owned output-channel window.
        while columns % panel_columns:
            panel_columns -= 16
        panels = columns // panel_columns

        @pl.jit.inline
        def stage(
            weights: pl.Tensor[[weight_rows, channels], pl.FP16],
            first: pl.Scalar[pl.INDEX],
        ):
            # Stacking phases along columns keeps each returned phase a
            # supported full-height Mat column view. Row-stacked phases would
            # require unsupported multi-C16 Mat row views or L1-to-L1 copies.
            resident = pl.tile.create([phase_rows, 4 * columns], pl.FP16, target_memory=pl.Mem.Mat)
            # One pipeline reuses two Vec banks for every phase. Flattening the
            # destination columns also avoids four loop-carried Mat branches.
            for panel in pl.pipeline(4 * panels, stage=2):
                phase = panel // panels
                column = panel % panels * panel_columns
                values = pl.load(weights, [phase * phase_rows, first + column], [phase_rows, panel_columns])
                resident = pl.tile.assemble(resident, values, [0, panel * panel_columns])
            weight0 = pl.tile.slice(resident, [phase_rows, columns], [0, 0])
            weight1 = pl.tile.slice(resident, [phase_rows, columns], [0, columns])
            weight2 = pl.tile.slice(resident, [phase_rows, columns], [0, 2 * columns])
            weight3 = pl.tile.slice(resident, [phase_rows, columns], [0, 3 * columns])
            return weight0, weight1, weight2, weight3

        return stage

    @pl.jit.inline
    def stage(
        weights: pl.Tensor[[weight_rows, channels], pl.FP16],
        first: pl.Scalar[pl.INDEX],
    ):
        weight0 = pl.load(weights, [0, first], [phase_rows, columns], target_memory=pl.Mem.Mat)
        weight1 = pl.load(weights, [phase_rows, first], [phase_rows, columns], target_memory=pl.Mem.Mat)
        weight2 = pl.load(weights, [2 * phase_rows, first], [phase_rows, columns], target_memory=pl.Mem.Mat)
        weight3 = pl.load(weights, [3 * phase_rows, first], [phase_rows, columns], target_memory=pl.Mem.Mat)
        return weight0, weight1, weight2, weight3

    return stage
