# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Code generation probe for 400x32 attention QK with per-row Q reuse."""

import pypto.language as pl
import pytest
from pypto import ir
from pypto.backend import BackendType


def test_qk_row_reuse_codegen(tmp_path):
    @pl.jit.incore
    def kernel(
        query: pl.Tensor[[400, 32], pl.FP16],
        key: pl.Tensor[[400, 32], pl.FP16],
        output: pl.Out[pl.Tensor[[400, 400], pl.FP16]],
    ):
        for row in pl.range(pl.tile.get_block_idx(), 25, pl.tile.get_block_num()):
            q0 = pl.load(query, [row * 16, 0], [16, 16], target_memory=pl.Mem.Mat)
            q1 = pl.load(query, [row * 16, 16], [16, 16], target_memory=pl.Mem.Mat)
            for column in pl.range(25):
                k0 = pl.load(key, [column * 16, 0], [16, 16], target_memory=pl.Mem.Mat)
                k1 = pl.load(key, [column * 16, 16], [16, 16], target_memory=pl.Mem.Mat)
                acc = pl.matmul(q0, pl.tile.transpose_view(k0))
                acc = pl.matmul_acc(acc, q1, pl.tile.transpose_view(k1))
                values = pl.tile.move(
                    acc, pl.Mem.Vec, blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.none_box
                )
                output = pl.store(pl.cast(values, pl.FP16, mode="none"), [row * 16, column * 16], output)
        return output

    ir.compile(
        kernel.specialize(), backend_type=BackendType.Ascend310P, output_dir=str(tmp_path), skip_ptoas=True
    )
    code = "\n".join(path.read_text() for path in tmp_path.rglob("*.pto"))
    assert "pto.tmatmul" in code and "pto.tmatmul.acc" in code
    assert "pto.get_block_idx" in code and "pto.get_block_num" in code
    row_loop = code.split("scf.for %row", 1)[1]
    outside_column, inside_column = row_loop.split("scf.for %column", 1)
    assert outside_column.count("pto.tload ") == 2  # Q panels loaded once per row.
    assert inside_column.count("pto.tload ") == 2  # K panels loaded per column.


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
