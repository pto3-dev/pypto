# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Probe a public tile transpose view for the 310P QK pattern."""

import pypto.language as pl
import pytest
from pypto import ir
from pypto.backend import BackendType


def test_public_qk_k32_codegen(tmp_path):
    @pl.jit.incore
    def kernel(
        query: pl.Tensor[[16, 32], pl.FP16],
        key: pl.Tensor[[16, 32], pl.FP16],
        output: pl.Out[pl.Tensor[[16, 16], pl.FP16]],
    ):
        q0 = pl.load(query, [0, 0], [16, 16], target_memory=pl.Mem.Mat)
        k0 = pl.load(key, [0, 0], [16, 16], target_memory=pl.Mem.Mat)
        acc = pl.matmul(q0, pl.tile.transpose_view(k0))
        q1 = pl.load(query, [0, 16], [16, 16], target_memory=pl.Mem.Mat)
        k1 = pl.load(key, [0, 16], [16, 16], target_memory=pl.Mem.Mat)
        acc = pl.matmul_acc(acc, q1, pl.tile.transpose_view(k1))
        value = pl.tile.move(acc, pl.Mem.Vec, blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.none_box)
        return pl.store(pl.cast(value, pl.FP16, mode="none"), [0, 0], output)

    ir.compile(
        kernel.specialize(), backend_type=BackendType.Ascend310P, output_dir=str(tmp_path), skip_ptoas=True
    )
    code = "\n".join(p.read_text() for p in tmp_path.rglob("*.pto"))
    assert "pto.tmatmul" in code and "pto.tmatmul.acc" in code


def test_transpose_view_rejects_larger_mat(tmp_path):
    @pl.jit.incore
    def kernel(
        source: pl.Tensor[[32, 16], pl.FP16],
        output: pl.Out[pl.Tensor[[16, 32], pl.FP16]],
    ):
        value = pl.load(source, [0, 0], [32, 16], target_memory=pl.Mem.Mat)
        return pl.store(pl.tile.transpose_view(value), [0, 0], output)

    with pytest.raises(RuntimeError, match="Ascend310P transpose_view requires a full FP16 16x16 Mat"):
        ir.compile(
            kernel.specialize(),
            backend_type=BackendType.Ascend310P,
            output_dir=str(tmp_path),
            skip_ptoas=True,
        )


def test_transpose_view_rejects_vec(tmp_path):
    @pl.jit.incore
    def kernel(
        source: pl.Tensor[[16, 16], pl.FP16],
        output: pl.Out[pl.Tensor[[16, 16], pl.FP16]],
    ):
        value = pl.load(source, [0, 0], [16, 16], target_memory=pl.Mem.Vec)
        return pl.store(pl.tile.transpose_view(value), [0, 0], output)

    with pytest.raises(RuntimeError, match="Ascend310P transpose_view requires a full FP16 16x16 Mat"):
        ir.compile(
            kernel.specialize(),
            backend_type=BackendType.Ascend310P,
            output_dir=str(tmp_path),
            skip_ptoas=True,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
