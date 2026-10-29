# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
# EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Compile-time bounds for 310P N32 accumulator panels."""

import pypto.language as pl
import pytest
from pypto import ir
from pypto.backend import BackendType


@pytest.mark.parametrize("rows, offset, valid", [(32, 16, True), (16, 0, False), (32, 1, False)])
def test_310p_n32_accumulator_panel(tmp_path, rows, offset, valid):
    @pl.jit.incore
    def gemm(
        a: pl.Tensor[[32, 64], pl.FP16],
        b: pl.Tensor[[64, 32], pl.FP16],
        out: pl.Out[pl.Tensor[[rows, 16], pl.FP32]],
    ):
        lhs = pl.load(a, [0, 0], [32, 64], target_memory=pl.Mem.Mat)
        rhs = pl.load(b, [0, 0], [64, 32], target_memory=pl.Mem.Mat)
        acc = pl.matmul(lhs, rhs)
        panel = pl.tile.slice(acc, [rows, 16], [0, offset])
        out = pl.store(panel, [0, 0], out)
        return out

    def compile_program():
        return ir.compile(
            gemm.specialize(),
            backend_type=BackendType.Ascend310P,
            output_dir=str(tmp_path),
            skip_ptoas=True,
            dump_passes=False,
        )

    if valid:
        compile_program()
        source = next(tmp_path.rglob("*.pto")).read_text()
        assert "pto.tmatmul" in source
        assert "pto.subview" in source
    else:
        with pytest.raises((RuntimeError, ValueError), match="accumulator slice"):
            compile_program()
