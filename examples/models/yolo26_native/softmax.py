# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Compensated native FP16 softmax for sixteen attention positions."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pypto.language as pl
import torch


@dataclass(frozen=True)
class Softmax16Plan:
    """Evaluate a fixed 16x16 finite FP16 score matrix entirely on 310P."""

    def metadata(self) -> torch.Tensor:
        """Keep the native plan ABI uniform; shape is encoded in the kernel."""
        return torch.zeros(32, dtype=torch.float32)

    def jit_kernel(self) -> Any:
        """Return the public external entry for this fixed shape."""
        source = Path(__file__).with_name("softmax_pair.cpp")

        @pl.jit.extern(core_type="aic", source=source)
        def softmax(
            metadata: pl.Tensor[[32], pl.FP32],
            scores: pl.Tensor[[256], pl.FP16],
            output: pl.Out[pl.Tensor[[256], pl.FP16]],
        ): ...

        return softmax


@dataclass(frozen=True)
class Softmax400Plan:
    """Keep 400 attention rows and FP16 casts in one native device task.

    Each core processes disjoint 16-row blocks using the original FP32
    reduction shape and separately narrowed FP16 result. Batch execution
    supplies one independent image per core through the public batch worker.
    """

    def metadata(self) -> torch.Tensor:
        """Retain the uniform native plan ABI; geometry is fixed in the kernel."""
        return torch.zeros(32, dtype=torch.float32)

    def jit_kernel(self) -> Any:
        """Return the fixed 400x400 FP16 external entry."""
        source = Path(__file__).with_name("softmax_matrix.cpp")

        @pl.jit.extern(core_type="aic", source=source)
        def softmax(
            metadata: pl.Tensor[[32], pl.FP32],
            scores: pl.Tensor[[160000], pl.FP16],
            output: pl.Out[pl.Tensor[[160000], pl.FP16]],
        ): ...

        return softmax
