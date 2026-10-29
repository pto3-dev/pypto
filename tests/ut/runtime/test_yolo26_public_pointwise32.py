# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Selection and weight packing for the public N32 pointwise profile."""

from dataclasses import replace

import pytest
import torch

from examples.models.yolo26_native.operators import ConvPlan
from examples.models.yolo26_native.pointwise import PublicOrderedPointwise32Plan, select_ordered_pointwise


def test_pointwise32_selection_and_weight_packing():
    original = ConvPlan(160, 160, 32, 32, 1, 1, 0, True, True)
    selected = select_ordered_pointwise(original, weight_l2_bypass=False)
    assert isinstance(selected, PublicOrderedPointwise32Plan)
    assert select_ordered_pointwise(original, weight_l2_bypass=True) is original
    for change in (
        {"height": 80},
        {"cout": 48},
        {"silu": False},
        {"four_way_accumulation": False},
        {"source_stride": 48, "source_offset": 16},
    ):
        plan = replace(original, **change)
        assert select_ordered_pointwise(plan, weight_l2_bypass=False) is plan
    weight = torch.arange(32 * 32, dtype=torch.float32).reshape(32, 32, 1, 1)
    packed = selected.pack_weight(weight).reshape(4, 128, 32)
    expected = torch.zeros_like(packed)
    dense = weight.half().reshape(32, 32)
    for channel in range(32):
        partial = channel % 4
        term = channel // 4
        expected[partial, term * 16 + channel % 16] = dense[:, channel]
    assert torch.equal(packed, expected)
    with pytest.raises(ValueError, match="resident views"):
        selected.metadata(weight_l2_offset=512)
    with pytest.raises(ValueError, match="resident tensor view"):
        selected.jit_kernel(output_padding=16)
