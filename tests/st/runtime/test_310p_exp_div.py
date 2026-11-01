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
from pypto.library import fp16_silu, fp16_silu_constants
from pypto.runtime import A1ResidentSession


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


@pytest.mark.parametrize("dtype", [pl.FP16, pl.FP32])
@pytest.mark.parametrize(("rows", "cols"), [(1, 80), (80, 16)])
def test_310p_public_maximum(tmp_path, dtype, rows, cols):
    @pl.jit.incore
    def maximum(
        lhs: pl.Tensor[[rows, cols], dtype],
        rhs: pl.Tensor[[rows, cols], dtype],
        output: pl.Out[pl.Tensor[[rows, cols + 32], dtype]],
    ):
        a = pl.load(lhs, [0, 0], [rows, cols])
        b = pl.load(rhs, [0, 0], [rows, cols])
        value = pl.tile.maximum(a, b)
        pl.store(value, [0, 16], output)
        return output

    assert not maximum._external_source_paths()
    compiled = ir.compile(
        maximum.specialize(),
        backend_type=BackendType.Ascend310P,
        output_dir=str(tmp_path),
        dump_passes=False,
    )
    torch_dtype = torch.float16 if dtype == pl.FP16 else torch.float32
    for seed, program in [(977, compiled), (983, ir.CompiledProgram.from_dir(compiled.output_dir))]:
        generator = torch.Generator().manual_seed(seed)
        lhs = torch.randn((rows, cols), generator=generator).to(torch_dtype)
        rhs = torch.randn((rows, cols), generator=generator).to(torch_dtype)
        # Include selected finite subnormals and zeros in the exact comparison.
        lhs[0, 0], rhs[0, 0] = torch.finfo(torch_dtype).tiny / 2, 0
        lhs[0, 1], rhs[0, 1] = 0, -torch.finfo(torch_dtype).tiny / 2
        output = torch.full((rows, cols + 32), 7.0, dtype=torch_dtype)
        program(lhs, rhs, output)
        torch.testing.assert_close(output[:, 16:-16], torch.maximum(lhs, rhs), rtol=0, atol=0)
        assert torch.all(output[:, :16] == 7) and torch.all(output[:, -16:] == 7)


@pytest.mark.parametrize("rows", [16, 32, 64, 128, 256])
@pytest.mark.parametrize("blocks", [1, 8])
def test_310p_public_fp16_silu_all_finite_patterns(tmp_path, rows, blocks):
    activate = fp16_silu(rows)

    @pl.jit.incore
    def compute(
        x: pl.Tensor[[65536], pl.FP16],
        output: pl.Out[pl.Tensor[[65536], pl.FP16]],
    ):
        source = pl.tensor.view(x, [4096, 16])
        destination = pl.tensor.view(output, [4096, 16])
        for work in pl.range(pl.tile.get_block_idx(), 4096 // rows, pl.tile.get_block_num()):
            value = pl.load(source, [work * rows, 0], [rows, 16])
            result = activate(value)
            destination = pl.store(result, [work * rows, 0], destination)
        return output

    assert not compute._external_source_paths()
    compiled = ir.compile(
        compute.specialize(),
        backend_type=BackendType.Ascend310P,
        output_dir=str(tmp_path),
        dump_passes=False,
    )
    patterns = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.float16)
    values = patterns[torch.isfinite(patterns)]
    assert values.numel() == 63488
    original = torch.cat((values, torch.zeros(65536 - values.numel(), dtype=torch.float16)))
    with A1ResidentSession(compiled) as session:
        kernels = [session.prepare(p) for p in (compiled, ir.CompiledProgram.from_dir(compiled.output_dir))]
        data = session.empty((65536,), torch.float16)
        guard = session.upload(torch.full((65568,), 7.0, dtype=torch.float16))
        output = guard.view((65536,), 16)
        for seed, kernel in zip([0, 991], kernels, strict=True):
            order = (
                torch.arange(65536)
                if seed == 0
                else torch.randperm(65536, generator=torch.Generator().manual_seed(seed))
            )
            x = original[order].contiguous()
            expected = x * x.sigmoid()
            data.copy_from(x)
            kernel(data, output, blocks=blocks)
            assert torch.equal(output.to_cpu().view(torch.int16), expected.view(torch.int16))
            full = guard.to_cpu()
            assert torch.all(full[:16] == 7) and torch.all(full[-16:] == 7)


@pytest.mark.parametrize("rows", [16, 32, 64, 128, 160, 256])
@pytest.mark.parametrize("blocks", [1, 8])
def test_310p_public_fp16_silu_resident_constants(tmp_path, rows, blocks):
    activate = fp16_silu_constants(rows)
    elements = ((65536 + rows * 16 - 1) // (rows * 16)) * rows * 16
    image_rows = elements // 16

    @pl.jit.incore
    def compute(
        x: pl.Tensor[[elements], pl.FP16],
        output: pl.Out[pl.Tensor[[elements], pl.FP16]],
    ):
        source = pl.tensor.view(x, [image_rows, 16])
        destination = pl.tensor.view(output, [image_rows, 16])
        zero = pl.tile.full([rows, 16], pl.FP32, 0.0)
        offset = pl.tile.full([rows, 16], pl.FP32, 15.0 / 2048)
        scale = pl.tile.full([rows, 16], pl.FP32, 8192.0)
        one = pl.tile.full([rows, 16], pl.FP32, 1.0)
        epsilon = pl.tile.full([rows, 16], pl.FP32, 0.00000011920928955078125)
        for work in pl.range(pl.tile.get_block_idx(), image_rows // rows, pl.tile.get_block_num()):
            value = pl.load(source, [work * rows, 0], [rows, 16])
            result = activate(value, zero, offset, scale, one, epsilon)
            destination = pl.store(result, [work * rows, 0], destination)
        return output

    assert not compute._external_source_paths()
    compiled = ir.compile(
        compute.specialize(),
        backend_type=BackendType.Ascend310P,
        output_dir=str(tmp_path),
        dump_passes=False,
    )
    patterns = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.float16)
    finite = patterns[torch.isfinite(patterns)]
    assert finite.numel() == 63488
    original = torch.cat((finite, torch.zeros(elements - finite.numel(), dtype=torch.float16)))
    with A1ResidentSession(compiled) as session:
        kernels = [session.prepare(p) for p in (compiled, ir.CompiledProgram.from_dir(compiled.output_dir))]
        assert all(len(kernel._params) == 2 for kernel in kernels)
        data = session.empty((elements,), torch.float16)
        guard = session.upload(torch.full((elements + 32,), 7.0, dtype=torch.float16))
        output = guard.view((elements,), 16)
        for seed, kernel in zip([0, 991], kernels, strict=True):
            order = (
                torch.arange(elements)
                if seed == 0
                else torch.randperm(elements, generator=torch.Generator().manual_seed(seed))
            )
            x = original[order].contiguous()
            data.copy_from(x)
            kernel(data, output, blocks=blocks)
            assert torch.equal(output.to_cpu().view(torch.int16), (x * x.sigmoid()).view(torch.int16))
            full = guard.to_cpu()
            assert torch.all(full[:16] == 7) and torch.all(full[-16:] == 7)


@pytest.mark.parametrize("dtype,torch_dtype", [(pl.FP16, torch.float16), (pl.FP32, torch.float32)])
@pytest.mark.parametrize("rows", [64, 160, 256])
@pytest.mark.parametrize("blocks", [1, 8])
def test_310p_public_vector_reshape(tmp_path, dtype, torch_dtype, rows, blocks):
    count = rows * 16
    total = 8 * count

    @pl.jit.incore
    def compute(x: pl.Tensor[[total], dtype], out: pl.Out[pl.Tensor[[total], dtype]]):
        source = pl.tensor.view(x, [8 * rows, 16])
        destination = pl.tensor.view(out, [8 * rows, 16])
        for work in pl.range(pl.tile.get_block_idx(), 8, pl.tile.get_block_num()):
            value = pl.load(source, [work * rows, 0], [rows, 16])
            flat = pl.tile.reshape(value, [1, count])
            doubled = pl.add(flat, flat)
            restored = pl.tile.reshape(doubled, [rows, 16])
            destination = pl.store(restored, [work * rows, 0], destination)
        return out

    assert not compute._external_source_paths()
    compiled = ir.compile(
        compute.specialize(), backend_type=BackendType.Ascend310P, output_dir=str(tmp_path), dump_passes=False
    )
    with A1ResidentSession(compiled) as session:
        kernels = [session.prepare(p) for p in (compiled, ir.CompiledProgram.from_dir(compiled.output_dir))]
        data = session.empty((total,), torch_dtype)
        guards = session.upload(torch.full((total + 32,), 7, dtype=torch_dtype))
        output = guards.view((total,), 16)
        for seed, kernel in zip([1033, 1039], kernels, strict=True):
            generator = torch.Generator().manual_seed(seed)
            values = torch.randint(-256, 256, (total,), generator=generator).to(torch_dtype) / 8
            data.copy_from(values)
            kernel(data, output, blocks=blocks)
            assert torch.equal(output.to_cpu(), values * 2)
            assert torch.equal(data.to_cpu(), values)
            saved = guards.to_cpu()
            assert torch.all(saved[:16] == 7) and torch.all(saved[-16:] == 7)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
