# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Static native layout, routing, pooling and attention-conversion plans."""

from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any

import pypto.language as pl
import torch


class VectorOp(IntEnum):
    PACK = 0
    COPY = 1
    UPSAMPLE = 2
    ADD = 3
    MAXPOOL = 4
    FLOAT_TO_HALF = 5
    HALF_TO_FLOAT = 6
    SCALE_HALF = 7


_CASTS = (VectorOp.FLOAT_TO_HALF, VectorOp.HALF_TO_FLOAT, VectorOp.SCALE_HALF)


@dataclass(frozen=True)
class VectorPlan:
    """Bounded tensor geometry, with channel slices represented by offsets."""

    operation: VectorOp
    height: int
    width: int = 1
    channels: int = 16
    source_stride: int = 16
    destination_stride: int = 16
    source_offset: int = 0
    destination_offset: int = 0
    input_height: int = 0
    input_width: int = 0
    second_stride: int = 16
    second_offset: int = 0
    scale: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.operation, VectorOp):
            raise ValueError("operation must be a VectorOp")
        geometry = (
            self.height,
            self.width,
            self.channels,
            self.source_stride,
            self.destination_stride,
            self.source_offset,
            self.destination_offset,
            self.input_height,
            self.input_width,
            self.second_stride,
            self.second_offset,
        )
        if (
            any(type(v) is not int or v < 0 for v in geometry)
            or min(self.height, self.width, self.channels) <= 0
        ):
            raise ValueError("vector geometry requires nonnegative integers and positive dimensions")
        if self.operation in _CASTS:
            if self.height % 256 or self.width != 1:
                raise ValueError("dense conversion length must be a positive multiple of 256")
        elif self.operation == VectorOp.PACK:
            if self.height * self.width % 16 or not 1 <= self.channels <= 16 or self.destination_stride != 16:
                raise ValueError("input packing requires groups of 16 pixels and 1..16 input channels")
        else:
            self._validate_layout()
        if max(self.input_size, self.output_size, self.second_size) > 0x7FFFFFFF:
            raise ValueError("vector geometry exceeds the signed indexing range")

    def _validate_layout(self) -> None:
        if any(
            v % 16
            for v in (
                self.channels,
                self.source_stride,
                self.destination_stride,
                self.source_offset,
                self.destination_offset,
                self.second_stride,
                self.second_offset,
            )
        ):
            raise ValueError("native feature-map routing requires aligned channel strides and offsets")
        if (
            self.source_offset + self.channels > self.source_stride
            or self.destination_offset + self.channels > self.destination_stride
        ):
            raise ValueError("channel slice exceeds its row stride")
        if self.operation == VectorOp.ADD and self.second_offset + self.channels > self.second_stride:
            raise ValueError("second channel slice exceeds its row stride")
        if self.operation == VectorOp.UPSAMPLE and (
            self.height != 2 * self.input_height or self.width != 2 * self.input_width
        ):
            raise ValueError("nearest upsampling requires output dimensions twice the input")

    @property
    def input_size(self) -> int:
        if self.operation in _CASTS:
            return self.height
        if self.operation == VectorOp.PACK:
            return self.height * self.width * self.channels
        if self.operation == VectorOp.UPSAMPLE:
            return self.input_height * self.input_width * self.source_stride
        return self.height * self.width * self.source_stride

    @property
    def output_size(self) -> int:
        return self.height if self.operation in _CASTS else self.height * self.width * self.destination_stride

    @property
    def second_size(self) -> int:
        return self.height * self.width * self.second_stride if self.operation == VectorOp.ADD else 16

    def metadata(self) -> torch.Tensor:
        """Encode constants only; model activation values stay on the device."""
        values = [
            int(self.operation),
            self.height,
            self.width,
            self.channels,
            self.source_stride,
            self.destination_stride,
            self.source_offset,
            self.destination_offset,
            self.input_height,
            self.input_width,
            self.second_stride,
            self.second_offset,
            0,
            0,
            0,
            self.scale,
        ]
        return torch.tensor(values + [0] * 16, dtype=torch.float32)

    def jit_kernel(self) -> Any:
        """Declare exact buffer sizes and conversion dtypes for public compile."""
        ni, nb, no = self.input_size, self.second_size, self.output_size
        ti = pl.FP32 if self.operation in (VectorOp.PACK, VectorOp.FLOAT_TO_HALF) else pl.FP16
        to = pl.FP32 if self.operation == VectorOp.HALF_TO_FLOAT else pl.FP16
        source = Path(__file__).with_name("vector.cpp")

        @pl.jit.extern(core_type="aic", source=source)
        def vector(
            metadata: pl.Tensor[[32], pl.FP32],
            x: pl.Tensor[[ni], ti],
            second: pl.Tensor[[nb], pl.FP16],
            output: pl.Out[pl.Tensor[[no], to]],
        ): ...

        return vector


@dataclass(frozen=True)
class ConcatPlan:
    """Concatenate two to four aligned NHWC source views in one native task."""

    height: int
    width: int
    sources: tuple[tuple[int, int, int], ...]

    def __post_init__(self) -> None:
        if type(self.height) is not int or type(self.width) is not int or min(self.height, self.width) <= 0:
            raise ValueError("concat requires positive integer spatial dimensions")
        if not 2 <= len(self.sources) <= 4:
            raise ValueError("native concat requires two to four source views")
        for channels, stride, offset in self.sources:
            if any(type(v) is not int or v < 0 or v % 16 for v in (channels, stride, offset)):
                raise ValueError("concat source channels, strides and offsets must be aligned integers")
            if channels == 0 or offset + channels > stride or (stride - channels) // 16 > 65535:
                raise ValueError("concat source view must fit its row and the 16-bit DMA gap")
        if self.channels > 4096 or self.height * self.width > 2**24:
            raise ValueError("concat geometry exceeds its UB or exact metadata budget")
        if max(self.input_sizes, default=0) > 0x7FFFFFFF or self.output_size > 0x7FFFFFFF:
            raise ValueError("concat allocation exceeds the signed indexing range")

    @property
    def channels(self) -> int:
        return sum(channels for channels, _, _ in self.sources)

    @property
    def input_sizes(self) -> tuple[int, ...]:
        return tuple(self.height * self.width * stride for _, stride, _ in self.sources)

    @property
    def output_size(self) -> int:
        return self.height * self.width * self.channels

    def metadata(self) -> torch.Tensor:
        """Encode bounded copy geometry, without host activation arithmetic."""
        padded = self.sources + ((0, 0, 0),) * (4 - len(self.sources))
        values = [self.height * self.width, self.channels, len(self.sources), 0]
        values.extend(v[0] for v in padded)
        values.extend(v[1] for v in padded)
        values.extend(v[2] for v in padded)
        return torch.tensor(values + [0] * 16, dtype=torch.float32)

    def jit_kernel(self) -> Any:
        """Expose every read source to the device graph's dependency analysis."""
        n0, n1, n2, n3 = self.input_sizes + (16,) * (4 - len(self.sources))
        no = self.output_size
        source = Path(__file__).with_name("concat.cpp")

        @pl.jit.extern(core_type="aic", source=source)
        def concatenate(
            metadata: pl.Tensor[[32], pl.FP32],
            x0: pl.Tensor[[n0], pl.FP16],
            x1: pl.Tensor[[n1], pl.FP16],
            x2: pl.Tensor[[n2], pl.FP16],
            x3: pl.Tensor[[n3], pl.FP16],
            output: pl.Out[pl.Tensor[[no], pl.FP16]],
        ): ...

        return concatenate
