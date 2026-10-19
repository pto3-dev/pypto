# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Native YOLO26 box/sigmoid decoding and two-stage detection TopK."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pypto.language as pl
import torch


@dataclass(frozen=True)
class DecodePlan:
    """One 80-class feature level decoded into dense FP32 [anchors,96]."""

    height: int
    width: int
    stride: int
    anchor_offset: int
    total_anchors: int

    def __post_init__(self) -> None:
        values = (self.height, self.width, self.stride, self.anchor_offset, self.total_anchors)
        if (
            any(type(v) is not int or v < 0 for v in values)
            or min(self.height, self.width, self.total_anchors) <= 0
        ):
            raise ValueError(
                "detection geometry requires positive dimensions and a nonnegative anchor offset"
            )
        if (
            self.stride not in (8, 16, 32)
            or self.anchor_offset + self.height * self.width > self.total_anchors
        ):
            raise ValueError("detection stride or destination anchor interval is invalid")
        if self.total_anchors * 96 > 0x7FFFFFFF:
            raise ValueError("detection geometry exceeds the signed indexing range")

    def metadata(self) -> torch.Tensor:
        """Create static constants, including the feature level's anchor offset."""
        return torch.tensor(
            [self.height, self.width, self.stride, self.anchor_offset] + [0] * 28, dtype=torch.float32
        )

    def jit_kernel(self) -> Any:
        """Declare aligned FP16 box/class buffers and a dense output buffer."""
        nb, nc, no = self.height * self.width * 16, self.height * self.width * 80, self.total_anchors * 96
        source = Path(__file__).with_name("decode.cpp")

        @pl.jit.extern(core_type="aic", source=source)
        def decode(
            metadata: pl.Tensor[[32], pl.FP32],
            boxes: pl.Tensor[[nb], pl.FP16],
            scores: pl.Tensor[[nc], pl.FP16],
            output: pl.Out[pl.Tensor[[no], pl.FP32]],
        ): ...

        return decode


@dataclass(frozen=True)
class TopKPlan:
    """Select 300 detections after pruning to 300 candidate anchors."""

    anchors: int

    def __post_init__(self) -> None:
        if type(self.anchors) is not int or not 300 <= self.anchors <= 0x7FFFFFFF // 96:
            raise ValueError("detection TopK requires at least 300 anchors within the tensor indexing range")

    def metadata(self) -> torch.Tensor:
        """Encode the static anchor count; selection itself runs on the device."""
        return torch.tensor([self.anchors] + [0] * 31, dtype=torch.float32)

    def jit_kernel(self) -> Any:
        """Declare the dense input and flattened [300,6] final detections."""
        ni = self.anchors * 96
        source = Path(__file__).with_name("topk.cpp")

        @pl.jit.extern(core_type="aic", source=source)
        def topk(
            metadata: pl.Tensor[[32], pl.FP32],
            dense: pl.Tensor[[ni], pl.FP32],
            detections: pl.Out[pl.Tensor[[1800], pl.FP32]],
        ): ...

        return topk
