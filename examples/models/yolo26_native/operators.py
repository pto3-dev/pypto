# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Typed external PTO3 operators for a device-only YOLO26 execution graph."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pypto.language as pl
import torch


@dataclass(frozen=True)
class ConvPlan:
    """Static geometry for aligned NHWC FP16 convolution and bias/SiLU."""

    height: int
    width: int
    cin: int
    cout: int
    kernel: int = 3
    stride: int = 1
    padding: int = 1
    silu: bool = True
    four_way_accumulation: bool = False

    def __post_init__(self) -> None:
        if type(self.four_way_accumulation) is not bool:
            raise TypeError("four-way accumulation must be a boolean")
        if any(
            type(v) is not int
            for v in (self.height, self.width, self.cin, self.cout, self.kernel, self.stride, self.padding)
        ):
            raise TypeError("convolution geometry requires integer dimensions")
        if min(self.height, self.width, self.cin, self.cout) <= 0 or max(self.cin, self.cout) > 1024:
            raise ValueError("convolution dimensions must be positive with channels <= 1024")
        if (
            self.kernel not in (1, 3)
            or self.stride not in (1, 2)
            or not 0 <= self.padding <= self.kernel // 2
        ):
            raise ValueError("convolution supports kernel 1/3, stride 1/2 and bounded symmetric padding")
        if self.oh <= 0 or self.ow <= 0 or self.oh * self.ow % 16:
            raise ValueError("convolution output pixel count must be a positive multiple of sixteen")
        if max(self.input_size, self.output_size, self.weight_size) > 0xFFFFFFFF:
            raise ValueError("convolution geometry exceeds the tensor ABI")

    @property
    def ci(self) -> int:
        return (self.cin + 15) // 16 * 16

    @property
    def co(self) -> int:
        return (self.cout + 15) // 16 * 16

    @property
    def oh(self) -> int:
        return (self.height + 2 * self.padding - self.kernel) // self.stride + 1

    @property
    def ow(self) -> int:
        return (self.width + 2 * self.padding - self.kernel) // self.stride + 1

    @property
    def input_size(self) -> int:
        return self.height * self.width * self.ci

    @property
    def output_size(self) -> int:
        return self.oh * self.ow * self.co

    @property
    def weight_size(self) -> int:
        if self.four_way_accumulation:
            return self.co * self.reduction_size
        return self.co * self.ci * self.kernel * self.kernel

    @property
    def reduction_size(self) -> int:
        """Return the logical CHW reduction length padded to four terms."""
        return (self.cin * self.kernel * self.kernel + 3) // 4 * 4

    def metadata(self, offset: int = 0) -> torch.Tensor:
        """Encode static launch geometry; no input-dependent host arithmetic."""
        if type(offset) is not int or offset < 0 or offset % 16:
            raise ValueError("convolution output offset must be a nonnegative multiple of sixteen")
        values = [
            self.height,
            self.width,
            self.cin,
            self.cout,
            self.kernel,
            self.stride,
            self.padding,
            self.oh,
            self.ow,
            self.ci,
            self.co,
            int(self.silu),
            offset,
        ]
        return torch.tensor(values + [0] * (32 - len(values)), dtype=torch.float32)

    def pack_weight(self, weight: torch.Tensor) -> torch.Tensor:
        """Pack constant CPU weights into sixteen-by-sixteen cube panels."""
        if (
            list(weight.shape) != [self.cout, self.cin, self.kernel, self.kernel]
            or weight.device.type != "cpu"
        ):
            raise ValueError("convolution weight geometry/device mismatch")
        if self.four_way_accumulation:
            packed = torch.zeros((self.co, self.reduction_size), dtype=torch.float16)
            packed[: self.cout, : self.cin * self.kernel * self.kernel] = weight.half().reshape(self.cout, -1)
            return (
                packed.reshape(self.co // 16, 16, self.reduction_size)
                .transpose(1, 2)
                .contiguous()
                .reshape(-1)
            )
        padded = torch.zeros((self.co, self.ci, self.kernel, self.kernel), dtype=torch.float16)
        padded[: self.cout, : self.cin] = weight.to(torch.float16)
        return (
            padded.reshape(self.co // 16, 16, self.ci // 16, 16, self.kernel, self.kernel)
            .permute(0, 4, 5, 2, 3, 1)
            .contiguous()
            .reshape(-1)
        )

    def jit_kernel(self, output_padding: int = 0) -> Any:
        """Declare the external kernel with its exact static allocation sizes."""
        if type(output_padding) is not int or output_padding < 0 or output_padding % 16:
            raise ValueError("convolution output padding must be a nonnegative multiple of sixteen")
        ni, nw, nb, no = self.input_size, self.weight_size, self.co, self.output_size + 2 * output_padding
        source = Path(__file__).with_name("conv_four_way.cpp" if self.four_way_accumulation else "conv.cpp")

        @pl.jit.extern(core_type="aic", source=source)
        def convolution(
            metadata: pl.Tensor[[32], pl.FP32],
            x: pl.Tensor[[ni], pl.FP16],
            weight: pl.Tensor[[nw], pl.FP16],
            bias: pl.Tensor[[nb], pl.FP32],
            output: pl.Out[pl.Tensor[[no], pl.FP16]],
            zero: pl.Tensor[[16], pl.FP16],
        ): ...

        return convolution
