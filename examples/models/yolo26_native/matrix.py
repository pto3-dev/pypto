# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Static FP16 attention products with FP32 cube accumulation on 310P."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pypto.language as pl
import torch


@dataclass(frozen=True)
class MatrixPlan:
    """Aligned row-major product; optionally interpret B as [N,K]."""

    m: int
    n: int
    k: int
    transpose_b: bool = False
    linear_accumulation: bool = False

    def __post_init__(self) -> None:
        if type(self.linear_accumulation) is not bool:
            raise TypeError("linear accumulation must be a boolean")
        if any(type(v) is not int or v <= 0 or v % 16 for v in (self.m, self.n, self.k)):
            raise ValueError("matrix dimensions must be positive multiples of sixteen")
        if max(self.m * self.k, self.k * self.n, self.m * self.n) > 0x7FFFFFFF:
            raise ValueError("matrix geometry exceeds the signed indexing range")

    def metadata(self) -> torch.Tensor:
        """Encode static geometry without reading activation values."""
        return torch.tensor([self.m, self.n, self.k, int(self.transpose_b)] + [0] * 28, dtype=torch.float32)

    def jit_kernel(self) -> Any:
        """Return the public external JIT entry with exact flattened sizes."""
        na, nb, no = self.m * self.k, self.n * self.k, self.m * self.n
        source = Path(__file__).with_name("matmul_linear.cpp" if self.linear_accumulation else "matmul.cpp")

        @pl.jit.extern(core_type="aic", source=source)
        def matrix(
            metadata: pl.Tensor[[32], pl.FP32],
            a: pl.Tensor[[na], pl.FP16],
            b: pl.Tensor[[nb], pl.FP16],
            output: pl.Out[pl.Tensor[[no], pl.FP16]],
        ): ...

        return matrix
