# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The diagnostic reference contract includes dtype, shape and exact bits."""

import numpy as np
import pytest

from examples.models.yolo26_native.chip_profile import _require_reference_array


def test_accepts_identical_bits():
    value = np.array([0.0, 1.5], dtype=np.float16)
    _require_reference_array(value, value.copy(), "same")


@pytest.mark.parametrize("change", ["zero_sign", "dtype", "shape"])
def test_rejects_value_equal_but_different_representation(change):
    value = np.array([0.0, 1.0], dtype=np.float16)
    expected = value.copy()
    if change == "zero_sign":
        expected[0] = -0.0
    elif change == "dtype":
        expected = expected.astype(np.float32)
    else:
        expected = expected.reshape(1, 2)
    with pytest.raises(AssertionError, match="strict reference"):
        _require_reference_array(value, expected, change)
