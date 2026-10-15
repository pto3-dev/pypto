# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Native depthwise 3x3 convolution for aligned YOLO26 feature maps."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pypto.language as pl
import torch


@dataclass(frozen=True)
class DepthwisePlan:
    """Static NHWC FP16 depthwise convolution, padding one, bias/SiLU."""

    height: int
    width: int
    channels: int
    stride: int = 1
    silu: bool = True

    def __post_init__(self) -> None:
        if any(type(v) is not int or v <= 0 for v in (self.height, self.width, self.channels, self.stride)):
            raise ValueError("depthwise geometry requires positive integer dimensions")
        if self.channels % 16 or self.stride not in (1, 2):
            raise ValueError("depthwise channels must align to sixteen and stride must be one or two")
        if max(self.input_size, self.output_size) > 0x7FFFFFFF:
            raise ValueError("depthwise geometry exceeds the signed indexing range")

    @property
    def oh(self) -> int:
        return (self.height - 1) // self.stride + 1

    @property
    def ow(self) -> int:
        return (self.width - 1) // self.stride + 1

    @property
    def input_size(self) -> int:
        return self.height * self.width * self.channels

    @property
    def output_size(self) -> int:
        return self.oh * self.ow * self.channels

    def metadata(self) -> torch.Tensor:
        """Encode the validated static launch geometry."""
        return torch.tensor(
            [self.height, self.width, self.channels, self.stride, self.oh, self.ow, 0, int(self.silu)]
            + [0] * 24,
            dtype=torch.float32,
        )

    def pack_weight(self, weight: torch.Tensor) -> torch.Tensor:
        """Pack constant CPU [C,1,3,3] weights as [3,3,C]."""
        if tuple(weight.shape) != (self.channels, 1, 3, 3) or weight.device.type != "cpu":
            raise ValueError("depthwise weights require CPU [channels,1,3,3]")
        return weight[:, 0].permute(1, 2, 0).to(torch.float16).contiguous().reshape(-1)

    def jit_kernel(self) -> Any:
        """Declare exact resident input, constant and output buffer sizes."""
        ni, nw, nb, no = self.input_size, self.channels * 9, self.channels, self.output_size
        source = Path(__file__).with_name("depthwise.cpp")

        @pl.jit.extern(core_type="aic", source=source)
        def depthwise(
            metadata: pl.Tensor[[32], pl.FP32],
            x: pl.Tensor[[ni], pl.FP16],
            weight: pl.Tensor[[nw], pl.FP16],
            bias: pl.Tensor[[nb], pl.FP32],
            output: pl.Out[pl.Tensor[[no], pl.FP16]],
        ): ...

        return depthwise
