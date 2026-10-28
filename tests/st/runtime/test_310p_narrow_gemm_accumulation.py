# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Native 310P accumulation through the public compile/call/reload APIs."""

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


@pytest.mark.parametrize("rows", [16, 32, 128, 1024])
@pytest.mark.parametrize("init_cond", [None, False, True])
def test_310p_narrow_gemm_accumulation(tmp_path, rows, init_cond):
    @pl.program
    class Accumulation:
        @pl.function(type=pl.FunctionType.InCore)
        def compute(
            self,
            lhs0: pl.Tensor[[rows, 16], pl.FP16],
            rhs0: pl.Tensor[[16, 16], pl.FP16],
            lhs1: pl.Tensor[[rows, 16], pl.FP16],
            rhs1: pl.Tensor[[16, 16], pl.FP16],
            output: pl.Tensor[[rows, 16], pl.FP32],
        ):
            a0 = pl.load(lhs0, [0, 0], [rows, 16], target_memory=pl.MemorySpace.Mat)
            b0 = pl.load(rhs0, [0, 0], [16, 16], target_memory=pl.MemorySpace.Mat)
            acc = pl.matmul(a0, b0)
            a1 = pl.load(lhs1, [0, 0], [rows, 16], target_memory=pl.MemorySpace.Mat)
            b1 = pl.load(rhs1, [0, 0], [16, 16], target_memory=pl.MemorySpace.Mat)
            acc = pl.matmul_acc(acc, a1, b1, init_cond=init_cond)
            acc = pl.matmul_acc(acc, a0, b0)
            pl.store(acc, [0, 0], output)

    compiled = ir.compile(
        Accumulation,
        backend_type=BackendType.Ascend310P,
        output_dir=str(tmp_path),
        dump_passes=False,
    )
    assert len(list(tmp_path.rglob("*.pto"))) == 1
    reloaded = ir.CompiledProgram.from_dir(compiled.output_dir)
    for seed, callable_program in [(73, compiled), (83, reloaded)]:
        generator = torch.Generator().manual_seed(seed)
        if seed == 73:
            lhs0 = torch.randint(-4, 5, (rows, 16), generator=generator).half()
            rhs0 = torch.randint(-4, 5, (16, 16), generator=generator).half()
            lhs1 = torch.randint(-4, 5, (rows, 16), generator=generator).half()
            rhs1 = torch.randint(-4, 5, (16, 16), generator=generator).half()
        else:
            lhs0 = torch.randn((rows, 16), generator=generator).half()
            rhs0 = torch.randn((16, 16), generator=generator).half()
            lhs1 = torch.randn((rows, 16), generator=generator).half()
            rhs1 = torch.randn((16, 16), generator=generator).half()
        output = torch.full((rows, 16), float("nan"))
        callable_program(lhs0, rhs0, lhs1, rhs1, output)
        first = lhs0.float() @ rhs0.float()
        second = lhs1.float() @ rhs1.float()
        expected = first + second if init_cond else 2 * first + second
        torch.testing.assert_close(
            output, expected, rtol=1e-5 if seed == 83 else 0, atol=2e-5 if seed == 83 else 0
        )


@pytest.mark.parametrize("rows", [16, 32, 128, 512, 1024])
def test_310p_tiled_gemm_32(tmp_path, rows):
    @pl.program
    class TiledGemm32:
        @pl.function(type=pl.FunctionType.InCore)
        def compute(
            self,
            lhs: pl.Tensor[[rows, 32], pl.FP16],
            rhs: pl.Tensor[[32, 32], pl.FP16],
            output: pl.Tensor[[rows, 32], pl.FP32],
        ):
            a0 = pl.load(lhs, [0, 0], [rows, 16], target_memory=pl.MemorySpace.Mat)
            a1 = pl.load(lhs, [0, 16], [rows, 16], target_memory=pl.MemorySpace.Mat)
            b00 = pl.load(rhs, [0, 0], [16, 16], target_memory=pl.MemorySpace.Mat)
            b10 = pl.load(rhs, [16, 0], [16, 16], target_memory=pl.MemorySpace.Mat)
            c0 = pl.matmul(a0, b00)
            c0 = pl.matmul_acc(c0, a1, b10)
            pl.store(c0, [0, 0], output)
            b01 = pl.load(rhs, [0, 16], [16, 16], target_memory=pl.MemorySpace.Mat)
            b11 = pl.load(rhs, [16, 16], [16, 16], target_memory=pl.MemorySpace.Mat)
            c1 = pl.matmul(a0, b01)
            c1 = pl.matmul_acc(c1, a1, b11)
            pl.store(c1, [0, 16], output)

    compiled = ir.compile(
        TiledGemm32,
        backend_type=BackendType.Ascend310P,
        output_dir=str(tmp_path),
        dump_passes=False,
    )
    assert len(list(tmp_path.rglob("*.pto"))) == 1
    reloaded = ir.CompiledProgram.from_dir(compiled.output_dir)
    for seed, callable_program in [(73, compiled), (83, reloaded)]:
        generator = torch.Generator().manual_seed(seed)
        if seed == 73:
            lhs = torch.randint(-4, 5, (rows, 32), generator=generator).half()
            rhs = torch.randint(-4, 5, (32, 32), generator=generator).half()
        else:
            lhs = torch.randn((rows, 32), generator=generator).half()
            rhs = torch.randn((32, 32), generator=generator).half()
        output = torch.full((rows, 32), float("nan"))
        callable_program(lhs, rhs, output)
        torch.testing.assert_close(
            output,
            lhs.float() @ rhs.float(),
            rtol=1e-5 if seed == 83 else 0,
            atol=2e-5 if seed == 83 else 0,
        )


@pytest.mark.parametrize(("rows", "reduction"), [(32, 32), (32, 144), (224, 144), (16, 2048)])
def test_310p_public_multi_k_gemm(tmp_path, rows, reduction):
    @pl.jit.incore
    def compute(
        lhs: pl.Tensor[[rows, reduction], pl.FP16],
        rhs: pl.Tensor[[reduction, 16], pl.FP16],
        output: pl.Out[pl.Tensor[[rows, 16], pl.FP32]],
    ):
        a = pl.load(lhs, [0, 0], [rows, reduction], target_memory=pl.MemorySpace.Mat)
        b = pl.load(rhs, [0, 0], [reduction, 16], target_memory=pl.MemorySpace.Mat)
        product = pl.matmul(a, b)
        pl.store(product, [0, 0], output)
        return output

    compiled = ir.compile(
        compute.specialize(),
        backend_type=BackendType.Ascend310P,
        output_dir=str(tmp_path),
        dump_passes=False,
    )
    assert not compute._external_source_paths()
    generated = next(tmp_path.rglob("*.pto")).read_text()
    assert generated.count("pto.tmatmul ") == 1
    assert "tpush" not in generated and "tpop" not in generated
    reloaded = ir.CompiledProgram.from_dir(compiled.output_dir)
    generator = torch.Generator().manual_seed(317)
    for program in (compiled, reloaded):
        # Dyadic products and sums stay exactly representable in FP32. Cube
        # reduction order is not assumed equal to a serial arbitrary-FP16 sum.
        lhs = (torch.randint(-8, 9, (rows, reduction), generator=generator).float() / 16).half()
        rhs = (torch.randint(-8, 9, (reduction, 16), generator=generator).float() / 16).half()
        output = torch.full((rows, 16), float("nan"), dtype=torch.float32)
        program(lhs, rhs, output)
        assert torch.equal(output, lhs.float() @ rhs.float())


@pytest.mark.parametrize("init_cond", [False, True])
def test_310p_multi_k_strided_accumulation(tmp_path, init_cond):
    rows, reduction = 32, 144

    @pl.jit.incore
    def compute(
        lhs0: pl.Tensor[[rows, reduction + 32], pl.FP16],
        rhs0: pl.Tensor[[reduction, 48], pl.FP16],
        lhs1: pl.Tensor[[rows, reduction + 32], pl.FP16],
        rhs1: pl.Tensor[[reduction, 48], pl.FP16],
        output: pl.Out[pl.Tensor[[rows, 48], pl.FP32]],
    ):
        a0 = pl.load(lhs0, [0, 16], [rows, reduction], target_memory=pl.MemorySpace.Mat)
        b0 = pl.load(rhs0, [0, 16], [reduction, 16], target_memory=pl.MemorySpace.Mat)
        acc = pl.matmul(a0, b0)
        a1 = pl.load(lhs1, [0, 16], [rows, reduction], target_memory=pl.MemorySpace.Mat)
        b1 = pl.load(rhs1, [0, 16], [reduction, 16], target_memory=pl.MemorySpace.Mat)
        acc = pl.matmul_acc(acc, a1, b1, init_cond=init_cond)
        acc = pl.matmul_acc(acc, a0, b0)
        pl.store(acc, [0, 16], output)
        return output

    compiled = ir.compile(
        compute.specialize(), backend_type=BackendType.Ascend310P, output_dir=str(tmp_path), dump_passes=False
    )
    generator = torch.Generator().manual_seed(318)
    for program in (compiled, ir.CompiledProgram.from_dir(compiled.output_dir)):
        left = [
            (torch.randint(-8, 9, (rows, reduction + 32), generator=generator).float() / 16).half()
            for _ in range(2)
        ]
        right = [
            (torch.randint(-8, 9, (reduction, 48), generator=generator).float() / 16).half() for _ in range(2)
        ]
        output = torch.full((rows, 48), 7.0, dtype=torch.float32)
        program(left[0], right[0], left[1], right[1], output)
        products = [a[:, 16:-16].float() @ b[:, 16:32].float() for a, b in zip(left, right, strict=True)]
        expected = products[0] + products[1] if init_cond else 2 * products[0] + products[1]
        assert torch.equal(output[:, 16:32], expected)
        assert torch.all(output[:, :16] == 7) and torch.all(output[:, 32:] == 7)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
