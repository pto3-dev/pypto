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


def _broadcast_kernel(dtype, kind, rows, cols):
    if dtype == pl.FP16 and kind == "row_sum":

        @pl.jit.incore
        def row_sum_fp16(
            x: pl.Tensor[[rows, cols], pl.FP16],
            output: pl.Tensor[[rows, cols + 32], pl.FP16],
        ):
            a = pl.load(x, [0, 0], [rows, cols])
            tmp = pl.tile.create([rows, cols], dtype=pl.FP16)
            reduced = pl.tile.row_sum(a, tmp)
            result = pl.tile.row_expand(a, reduced)
            pl.store(result, [0, 16], output)

        return row_sum_fp16
    if dtype == pl.FP16 and kind == "row_max":

        @pl.jit.incore
        def row_max_fp16(
            x: pl.Tensor[[rows, cols], pl.FP16],
            output: pl.Tensor[[rows, cols + 32], pl.FP16],
        ):
            a = pl.load(x, [0, 0], [rows, cols])
            tmp = pl.tile.create([rows, cols], dtype=pl.FP16)
            reduced = pl.tile.row_max(a, tmp)
            result = pl.tile.row_expand(a, reduced)
            pl.store(result, [0, 16], output)

        return row_max_fp16
    if dtype == pl.FP16 and kind == "col_sum":

        @pl.jit.incore
        def col_sum_fp16(
            x: pl.Tensor[[rows, cols], pl.FP16],
            output: pl.Tensor[[rows, cols + 32], pl.FP16],
        ):
            a = pl.load(x, [0, 0], [rows, cols])
            reduced = pl.tile.col_sum(a)
            result = pl.tile.col_expand(a, reduced)
            pl.store(result, [0, 16], output)

        return col_sum_fp16
    if dtype == pl.FP16 and kind == "col_max":

        @pl.jit.incore
        def col_max_fp16(
            x: pl.Tensor[[rows, cols], pl.FP16],
            output: pl.Tensor[[rows, cols + 32], pl.FP16],
        ):
            a = pl.load(x, [0, 0], [rows, cols])
            reduced = pl.tile.col_max(a)
            result = pl.tile.col_expand(a, reduced)
            pl.store(result, [0, 16], output)

        return col_max_fp16
    if dtype == pl.FP32 and kind == "row_sum":

        @pl.jit.incore
        def row_sum_fp32(
            x: pl.Tensor[[rows, cols], pl.FP32],
            output: pl.Tensor[[rows, cols + 32], pl.FP32],
        ):
            a = pl.load(x, [0, 0], [rows, cols])
            tmp = pl.tile.create([rows, cols], dtype=pl.FP32)
            reduced = pl.tile.row_sum(a, tmp)
            result = pl.tile.row_expand(a, reduced)
            pl.store(result, [0, 16], output)

        return row_sum_fp32
    if dtype == pl.FP32 and kind == "row_max":

        @pl.jit.incore
        def row_max_fp32(
            x: pl.Tensor[[rows, cols], pl.FP32],
            output: pl.Tensor[[rows, cols + 32], pl.FP32],
        ):
            a = pl.load(x, [0, 0], [rows, cols])
            tmp = pl.tile.create([rows, cols], dtype=pl.FP32)
            reduced = pl.tile.row_max(a, tmp)
            result = pl.tile.row_expand(a, reduced)
            pl.store(result, [0, 16], output)

        return row_max_fp32
    if dtype == pl.FP32 and kind == "col_sum":

        @pl.jit.incore
        def col_sum_fp32(
            x: pl.Tensor[[rows, cols], pl.FP32],
            output: pl.Tensor[[rows, cols + 32], pl.FP32],
        ):
            a = pl.load(x, [0, 0], [rows, cols])
            reduced = pl.tile.col_sum(a)
            result = pl.tile.col_expand(a, reduced)
            pl.store(result, [0, 16], output)

        return col_sum_fp32
    if dtype == pl.FP32 and kind == "col_max":

        @pl.jit.incore
        def col_max_fp32(
            x: pl.Tensor[[rows, cols], pl.FP32],
            output: pl.Tensor[[rows, cols + 32], pl.FP32],
        ):
            a = pl.load(x, [0, 0], [rows, cols])
            reduced = pl.tile.col_max(a)
            result = pl.tile.col_expand(a, reduced)
            pl.store(result, [0, 16], output)

        return col_max_fp32
    raise ValueError(f"Unsupported broadcast: {kind}/{dtype}")


@pytest.mark.parametrize("dtype", [pl.FP16, pl.FP32])
@pytest.mark.parametrize("kind", ["row_sum", "row_max", "col_sum", "col_max"])
@pytest.mark.parametrize(("rows", "cols"), [(16, 16), (16, 80), (16, 400), (80, 80)])
def test_310p_reduction_broadcast(tmp_path, dtype, kind, rows, cols):
    compiled = ir.compile(
        _broadcast_kernel(dtype, kind, rows, cols).specialize(),
        backend_type=BackendType.Ascend310P,
        output_dir=str(tmp_path),
        dump_passes=False,
    )
    reloaded = ir.CompiledProgram.from_dir(compiled.output_dir)
    torch_dtype = torch.float16 if dtype == pl.FP16 else torch.float32
    for seed, program in [(181, compiled), (191, reloaded)]:
        generator = torch.Generator().manual_seed(seed)
        x = (torch.randint(-16, 17, (rows, cols), generator=generator).float() / 32).to(torch_dtype)
        output = torch.full((rows, cols + 32), 7.0, dtype=torch_dtype)
        program(x, output)
        axis = 1 if kind.startswith("row") else 0
        expected = (
            x.amax(dim=axis, keepdim=True)
            if kind.endswith("max")
            else x.float().sum(dim=axis, keepdim=True).to(torch_dtype)
        )
        assert torch.equal(output[:, 16 : cols + 16], expected.expand(rows, cols))
        assert torch.equal(output[:, :16], torch.full_like(output[:, :16], 7.0))
        assert torch.equal(output[:, cols + 16 :], torch.full_like(output[:, cols + 16 :], 7.0))


def _stable_softmax_kernel(rows, cols, column=False):
    if column:

        @pl.jit.incore
        def column_softmax(
            x: pl.Tensor[[rows, cols], pl.FP32],
            output: pl.Tensor[[rows, cols + 32], pl.FP32],
        ):
            a = pl.load(x, [0, 0], [rows, cols])
            maximum = pl.tile.col_max(a)
            shifted = pl.sub(a, pl.tile.col_expand(a, maximum))
            exponential = pl.exp(shifted)
            total = pl.tile.col_sum(exponential)
            result = pl.div(exponential, pl.tile.col_expand(exponential, total))
            pl.store(result, [0, 16], output)

        return column_softmax

    @pl.jit.incore
    def row_softmax(
        x: pl.Tensor[[rows, cols], pl.FP32],
        output: pl.Tensor[[rows, cols + 32], pl.FP32],
    ):
        a = pl.load(x, [0, 0], [rows, cols])
        tmp_max = pl.tile.create([rows, cols], dtype=pl.FP32)
        maximum = pl.tile.row_max(a, tmp_max)
        shifted = pl.sub(a, pl.tile.row_expand(a, maximum))
        exponential = pl.exp(shifted)
        tmp_sum = pl.tile.create([rows, cols], dtype=pl.FP32)
        total = pl.tile.row_sum(exponential, tmp_sum)
        result = pl.div(exponential, pl.tile.row_expand(exponential, total))
        pl.store(result, [0, 16], output)

    return row_softmax


@pytest.mark.parametrize(
    ("rows", "cols", "column"), [(16, 16, False), (16, 80, False), (16, 400, False), (80, 80, True)]
)
def test_310p_stable_softmax(tmp_path, rows, cols, column):
    compiled = ir.compile(
        _stable_softmax_kernel(rows, cols, column).specialize(),
        backend_type=BackendType.Ascend310P,
        output_dir=str(tmp_path),
        dump_passes=False,
    )
    axis = 0 if column else 1
    for seed, program in [(193, compiled), (197, ir.CompiledProgram.from_dir(compiled.output_dir))]:
        generator = torch.Generator().manual_seed(seed)
        x = torch.randn((rows, cols), generator=generator) * 4
        # Offset stresses max-subtraction stability; a constant slice checks ties.
        x += 1000 if seed == 193 else -1000
        if column:
            x[:, 0] = x[0, 0]
        else:
            x[0, :] = x[0, 0]
        output = torch.full((rows, cols + 32), 7.0)
        program(x, output)
        actual = output[:, 16 : cols + 16]
        torch.testing.assert_close(actual, torch.softmax(x, dim=axis), rtol=0.0003, atol=0.000003)
        torch.testing.assert_close(
            actual.sum(dim=axis), torch.ones(cols if column else rows), rtol=0.0001, atol=0.00001
        )
        assert torch.equal(output[:, :16], torch.full_like(output[:, :16], 7.0))
        assert torch.equal(output[:, cols + 16 :], torch.full_like(output[:, cols + 16 :], 7.0))
