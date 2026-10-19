# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Native A1 sum/max reductions through public JIT compile/call/reload."""

import ctypes

import pypto.language as pl
import pytest
import torch
from pypto import ir
from pypto.backend import BackendType


@pytest.fixture(scope="module", autouse=True)
def require_310p_hardware():
    try:
        acl = ctypes.CDLL("libascendcl.so")
    except OSError:
        pytest.skip("Ascend runtime is unavailable")
    acl.aclrtGetSocName.restype = ctypes.c_char_p
    acl.aclrtGetSocName.argtypes = []
    soc = acl.aclrtGetSocName()
    if not soc or not soc.startswith(b"Ascend310P"):
        pytest.skip("This native regression requires Ascend310P hardware")


def _reduction_kernel(kind, dtype, rows, cols):
    if dtype == pl.FP16 and kind == "row_sum":

        @pl.jit.incore
        def row_sum_fp16(
            x: pl.Tensor[[rows, cols], pl.FP16],
            output: pl.Tensor[[rows + 32, 1], pl.FP16],
        ):
            a = pl.load(x, [0, 0], [rows, cols])
            tmp = pl.tile.create([rows, cols], dtype=pl.FP16)
            result = pl.tile.row_sum(a, tmp)
            pl.store(result, [16, 0], output)

        return row_sum_fp16
    elif dtype == pl.FP16 and kind == "row_max":

        @pl.jit.incore
        def row_max_fp16(
            x: pl.Tensor[[rows, cols], pl.FP16],
            output: pl.Tensor[[rows + 32, 1], pl.FP16],
        ):
            a = pl.load(x, [0, 0], [rows, cols])
            tmp = pl.tile.create([rows, cols], dtype=pl.FP16)
            result = pl.tile.row_max(a, tmp)
            pl.store(result, [16, 0], output)

        return row_max_fp16
    elif dtype == pl.FP16 and kind == "col_sum":

        @pl.jit.incore
        def col_sum_fp16(
            x: pl.Tensor[[rows, cols], pl.FP16],
            output: pl.Tensor[[1, cols + 32], pl.FP16],
        ):
            a = pl.load(x, [0, 0], [rows, cols])
            result = pl.tile.col_sum(a)
            pl.store(result, [0, 16], output)

        return col_sum_fp16
    elif dtype == pl.FP16 and kind == "col_max":

        @pl.jit.incore
        def col_max_fp16(
            x: pl.Tensor[[rows, cols], pl.FP16],
            output: pl.Tensor[[1, cols + 32], pl.FP16],
        ):
            a = pl.load(x, [0, 0], [rows, cols])
            result = pl.tile.col_max(a)
            pl.store(result, [0, 16], output)

        return col_max_fp16
    elif dtype == pl.FP32 and kind == "row_sum":

        @pl.jit.incore
        def row_sum_fp32(
            x: pl.Tensor[[rows, cols], pl.FP32],
            output: pl.Tensor[[rows + 32, 1], pl.FP32],
        ):
            a = pl.load(x, [0, 0], [rows, cols])
            tmp = pl.tile.create([rows, cols], dtype=pl.FP32)
            result = pl.tile.row_sum(a, tmp)
            pl.store(result, [16, 0], output)

        return row_sum_fp32
    elif dtype == pl.FP32 and kind == "row_max":

        @pl.jit.incore
        def row_max_fp32(
            x: pl.Tensor[[rows, cols], pl.FP32],
            output: pl.Tensor[[rows + 32, 1], pl.FP32],
        ):
            a = pl.load(x, [0, 0], [rows, cols])
            tmp = pl.tile.create([rows, cols], dtype=pl.FP32)
            result = pl.tile.row_max(a, tmp)
            pl.store(result, [16, 0], output)

        return row_max_fp32
    elif dtype == pl.FP32 and kind == "col_sum":

        @pl.jit.incore
        def col_sum_fp32(
            x: pl.Tensor[[rows, cols], pl.FP32],
            output: pl.Tensor[[1, cols + 32], pl.FP32],
        ):
            a = pl.load(x, [0, 0], [rows, cols])
            result = pl.tile.col_sum(a)
            pl.store(result, [0, 16], output)

        return col_sum_fp32
    elif dtype == pl.FP32 and kind == "col_max":

        @pl.jit.incore
        def col_max_fp32(
            x: pl.Tensor[[rows, cols], pl.FP32],
            output: pl.Tensor[[1, cols + 32], pl.FP32],
        ):
            a = pl.load(x, [0, 0], [rows, cols])
            result = pl.tile.col_max(a)
            pl.store(result, [0, 16], output)

        return col_max_fp32
    raise ValueError(f"Unsupported reduction kind/dtype: {kind}/{dtype}")


@pytest.mark.parametrize("dtype", [pl.FP16, pl.FP32])
@pytest.mark.parametrize("kind", ["row_sum", "row_max", "col_sum", "col_max"])
@pytest.mark.parametrize(("rows", "cols"), [(16, 16), (16, 80), (32, 256)])
def test_310p_axis_reductions(tmp_path, dtype, kind, rows, cols):
    _check_public_reduction(tmp_path, dtype, kind, rows, cols)


@pytest.mark.parametrize("dtype", [pl.FP16, pl.FP32])
@pytest.mark.parametrize(
    ("kind", "rows", "cols"),
    [
        ("row_sum", 16, 400),
        ("row_max", 16, 400),
        ("col_sum", 80, 80),
        ("col_max", 80, 80),
    ],
)
def test_310p_yolo26_reduction_profiles(tmp_path, dtype, kind, rows, cols):
    _check_public_reduction(tmp_path, dtype, kind, rows, cols)


def _check_public_reduction(tmp_path, dtype, kind, rows, cols):
    kernel = _reduction_kernel(kind, dtype, rows, cols)
    compiled = ir.compile(
        kernel.specialize(), backend_type=BackendType.Ascend310P, output_dir=str(tmp_path), dump_passes=False
    )
    assert len(list(tmp_path.rglob("*.pto"))) == 1
    reloaded = ir.CompiledProgram.from_dir(compiled.output_dir)
    torch_dtype = torch.float16 if dtype == pl.FP16 else torch.float32
    row = kind.startswith("row")
    axis = 1 if row else 0
    for seed, callable_program in [(157, compiled), (163, reloaded)]:
        generator = torch.Generator().manual_seed(seed)
        low, high = (-8, 0) if seed == 157 else (-8, 9)
        x = (torch.randint(low, high, (rows, cols), generator=generator).float() / 16).to(torch_dtype)
        if seed == 163 and dtype == pl.FP32:
            x = torch.randn((rows, cols), generator=generator)
        shape = (rows + 32, 1) if row else (1, cols + 32)
        output = torch.full(shape, 7.0, dtype=torch_dtype)
        callable_program(x, output)
        expected = (
            x.amax(dim=axis, keepdim=True)
            if kind.endswith("max")
            else x.float().sum(dim=axis, keepdim=True).to(torch_dtype)
        )
        actual = output[16 : rows + 16, :] if row else output[:, 16 : cols + 16]
        torch.testing.assert_close(
            actual, expected, rtol=0 if dtype == pl.FP16 else 0.00001, atol=0 if dtype == pl.FP16 else 0.00002
        )
        before = output[:16, :] if row else output[:, :16]
        after = output[rows + 16 :, :] if row else output[:, cols + 16 :]
        assert torch.equal(before, torch.full_like(before, 7.0))
        assert torch.equal(after, torch.full_like(after, 7.0))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
