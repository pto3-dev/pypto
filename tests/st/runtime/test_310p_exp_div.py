# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Native A1 exponential/division and tensor-form SiLU through public APIs."""

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


def _activation_kernel(op, dtype, cols):
    if op == "exp":

        @pl.jit.incore
        def exp_kernel(
            x: pl.Tensor[[1, cols], dtype],
            auxiliary: pl.Tensor[[1, cols], dtype],
            ones: pl.Tensor[[1, cols], dtype],
            output: pl.Tensor[[1, cols], dtype],
        ):
            a = pl.load(x, [0, 0], [1, cols])
            result = pl.exp(a)
            pl.store(result, [0, 0], output)

        return exp_kernel
    if op == "div":

        @pl.jit.incore
        def div_kernel(
            x: pl.Tensor[[1, cols], dtype],
            auxiliary: pl.Tensor[[1, cols], dtype],
            ones: pl.Tensor[[1, cols], dtype],
            output: pl.Tensor[[1, cols], dtype],
        ):
            a = pl.load(x, [0, 0], [1, cols])
            b = pl.load(auxiliary, [0, 0], [1, cols])
            result = pl.div(a, b)
            pl.store(result, [0, 0], output)

        return div_kernel

    @pl.jit.incore
    def silu_kernel(
        x: pl.Tensor[[1, cols], dtype],
        auxiliary: pl.Tensor[[1, cols], dtype],
        ones: pl.Tensor[[1, cols], dtype],
        output: pl.Tensor[[1, cols], dtype],
    ):
        a = pl.load(x, [0, 0], [1, cols])
        zero = pl.load(auxiliary, [0, 0], [1, cols])
        one = pl.load(ones, [0, 0], [1, cols])
        negative = pl.sub(zero, a)
        exponential = pl.exp(negative)
        denominator = pl.add(exponential, one)
        result = pl.div(a, denominator)
        pl.store(result, [0, 0], output)

    return silu_kernel


@pytest.mark.parametrize("dtype", [pl.FP16, pl.FP32])
@pytest.mark.parametrize(
    ("op", "cols"),
    [
        ("exp", 16),
        ("exp", 80),
        ("exp", 16336),
        ("div", 16),
        ("div", 80),
        ("div", 8192),
        ("silu", 16),
        ("silu", 80),
        ("silu", 2048),
    ],
)
def test_310p_exp_div_silu(tmp_path, dtype, op, cols):
    kernel = _activation_kernel(op, dtype, cols)
    compiled = ir.compile(
        kernel.specialize(), backend_type=BackendType.Ascend310P, output_dir=str(tmp_path), dump_passes=False
    )
    assert len(list(tmp_path.rglob("*.pto"))) == 1
    reloaded = ir.CompiledProgram.from_dir(compiled.output_dir)
    torch_dtype = torch.float16 if dtype == pl.FP16 else torch.float32
    for seed, callable_program in [(113, compiled), (127, reloaded)]:
        generator = torch.Generator().manual_seed(seed)
        x = (torch.rand((1, cols), generator=generator) * 16 - 8).to(torch_dtype)
        ones = torch.ones_like(x)
        if op == "div":
            auxiliary = (torch.rand((1, cols), generator=generator) * 4 + 0.25).to(torch_dtype)
            auxiliary[:, ::2] *= -1
            expected = x / auxiliary
        elif op == "exp":
            auxiliary = torch.zeros_like(x)
            expected = x.exp()
        else:
            auxiliary = torch.zeros_like(x)
            expected = torch.nn.functional.silu(x.float()).to(torch_dtype)
        output = torch.full_like(x, float("nan"))
        callable_program(x, auxiliary, ones, output)
        torch.testing.assert_close(
            output,
            expected,
            rtol=0.003 if dtype == pl.FP16 else 0.0001,
            atol=0.00002 if dtype == pl.FP16 else 0.000002,
        )


@pytest.mark.parametrize("dtype", [pl.FP16, pl.FP32])
@pytest.mark.parametrize("cols", [16, 80, 2048])
def test_310p_silu_scalar_broadcast(tmp_path, dtype, cols):
    if dtype == pl.FP16:

        @pl.jit.incore
        def silu_fp16(x: pl.Tensor[[1, cols], pl.FP16], output: pl.Tensor[[1, cols], pl.FP16]):
            a = pl.load(x, [0, 0], [1, cols])
            zero = pl.tile.full([1, cols], dtype=pl.FP16, value=0.0)
            one = pl.tile.full([1, cols], dtype=pl.FP16, value=1.0)
            negative = pl.sub(zero, a)
            exponential = pl.exp(negative)
            denominator = pl.add(exponential, one)
            result = pl.div(a, denominator)
            pl.store(result, [0, 0], output)

        kernel = silu_fp16
    else:

        @pl.jit.incore
        def silu_fp32(x: pl.Tensor[[1, cols], pl.FP32], output: pl.Tensor[[1, cols], pl.FP32]):
            a = pl.load(x, [0, 0], [1, cols])
            zero = pl.tile.full([1, cols], dtype=pl.FP32, value=0.0)
            one = pl.tile.full([1, cols], dtype=pl.FP32, value=1.0)
            negative = pl.sub(zero, a)
            exponential = pl.exp(negative)
            denominator = pl.add(exponential, one)
            result = pl.div(a, denominator)
            pl.store(result, [0, 0], output)

        kernel = silu_fp32

    compiled = ir.compile(
        kernel.specialize(), backend_type=BackendType.Ascend310P, output_dir=str(tmp_path), dump_passes=False
    )
    reloaded = ir.CompiledProgram.from_dir(compiled.output_dir)
    torch_dtype = torch.float16 if dtype == pl.FP16 else torch.float32
    for seed, callable_program in [(131, compiled), (137, reloaded)]:
        generator = torch.Generator().manual_seed(seed)
        x = (torch.rand((1, cols), generator=generator) * 16 - 8).to(torch_dtype)
        output = torch.full_like(x, float("nan"))
        callable_program(x, output)
        expected = torch.nn.functional.silu(x.float()).to(torch_dtype)
        torch.testing.assert_close(
            output,
            expected,
            rtol=0.003 if dtype == pl.FP16 else 0.0001,
            atol=0.00002 if dtype == pl.FP16 else 0.000002,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
