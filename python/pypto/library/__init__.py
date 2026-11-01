# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Reusable algorithms composed exclusively from public PyPTO operations."""

from .activation import fp16_silu, fp16_silu_constants
from .convolution import fp16_conv_epilogue, fp16_conv_writeback
from .softmax import fp16_softmax_row, fp32_softmax_tile

__all__ = [
    "fp16_conv_epilogue",
    "fp16_conv_writeback",
    "fp16_silu",
    "fp16_silu_constants",
    "fp16_softmax_row",
    "fp32_softmax_tile",
]
