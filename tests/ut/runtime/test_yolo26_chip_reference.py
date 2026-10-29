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

from examples.models.yolo26_native.chip_profile import (
    _combined_source,
    _require_reference_array,
    _validate_observer_trace,
)


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


@pytest.mark.parametrize("generated", [False, True])
def test_combines_native_and_generated_entries(tmp_path, generated):
    declaration = (
        'extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry('
        if generated
        else 'extern "C" AICORE void kernel_entry('
    )
    source = tmp_path / "kernel.cpp"
    source.write_text("#include <cstdint>\n" + declaration + "__gm__ int64_t* args) {}\n")
    combined = _combined_source([source])
    assert "AICORE void op_0(" in combined
    assert "op0::op_0(args)" in combined
    assert 'extern "C"' not in combined.split("namespace op0 {", 1)[1].split("}\n", 1)[0]


@pytest.mark.parametrize("tasks,blocks", [(183, {1, 2, 4, 8}), (216, {1, 8}), (256, {8})])
def test_observer_accepts_current_and_older_graph_sizes(tasks, blocks):
    _validate_observer_trace(tasks, blocks)


@pytest.mark.parametrize("tasks,blocks", [(0, {8}), (2**31, {8}), (183, set()), (183, {1, 9})])
def test_observer_rejects_invalid_wire_or_dispatch(tasks, blocks):
    with pytest.raises(ValueError, match="bounded nonempty graph"):
        _validate_observer_trace(tasks, blocks)


def test_combined_sources_keep_each_logical_grid(tmp_path):
    source = tmp_path / "kernel.cpp"
    source.write_text('extern "C" AICORE void kernel_entry(__gm__ int64_t* args) { get_block_num(); }')
    combined = _combined_source([source, source], [4, 8])
    assert "#define get_block_num() 4\nnamespace op0" in combined
    assert "#define get_block_num() 8\nnamespace op1" in combined
    assert combined.count("#undef get_block_num") == 2
    assert "op0::op_0(args)" in combined and "op1::op_1(args)" in combined


@pytest.mark.parametrize("grids", [[], [0], [9]])
def test_combined_sources_reject_invalid_logical_grids(tmp_path, grids):
    with pytest.raises(ValueError, match="logical grid"):
        _combined_source([tmp_path / "unused.cpp"], grids)


def test_combines_inline_native_entry(tmp_path):
    source = tmp_path / "spatial.cpp"
    source.write_text(
        'extern "C" AICORE inline __attribute__((always_inline)) void kernel_entry(__gm__ int64_t* args) {}'
    )
    assert "AICORE void op_0(" in _combined_source([source], [4])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
