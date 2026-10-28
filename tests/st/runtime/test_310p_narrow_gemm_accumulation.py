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
from pypto.runtime import A1L2Worker, A1ResidentSession


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


def _vec_to_mat_gemm(rows, reduction):
    @pl.jit.incore
    def compute(
        lhs: pl.Tensor[[rows, reduction], pl.FP16],
        rhs: pl.Tensor[[reduction, 16], pl.FP16],
        output: pl.Out[pl.Tensor[[rows, 16], pl.FP32]],
    ):
        # Repeated local lifetimes exercise UB/L1 reuse dependencies, including
        # the MTE3 reader before a later vector write reuses the same UB bytes.
        for repeat in pl.range(0, 3):
            a = pl.load(lhs, [0, 0], [rows, reduction], target_memory=pl.MemorySpace.Vec)
            doubled = pl.add(a, a)
            ma = pl.tile.move(doubled, pl.MemorySpace.Mat)
            b = pl.load(rhs, [0, 0], [reduction, 16], target_memory=pl.MemorySpace.Vec)
            mb = pl.tile.move(b, pl.MemorySpace.Mat)
            product = pl.matmul(ma, mb)
            product = pl.matmul_acc(product, ma, mb)
            pl.store(product, [0, 0], output)
        return output

    return compute


@pytest.mark.parametrize("rows,reduction", [(16, 16), (32, 32), (224, 144), (16, 2048)])
def test_310p_vec_to_mat_gemm(tmp_path, rows, reduction):
    compute = _vec_to_mat_gemm(rows, reduction)
    assert not compute._external_source_paths()
    compiled = ir.compile(
        compute.specialize(),
        backend_type=BackendType.Ascend310P,
        output_dir=str(tmp_path),
        dump_passes=False,
    )
    generator = torch.Generator().manual_seed(619)
    with A1ResidentSession(compiled) as session:
        kernels = [session.prepare(p) for p in (compiled, ir.CompiledProgram.from_dir(compiled.output_dir))]
        lhs = session.empty((rows, reduction), torch.float16)
        rhs = session.empty((reduction, 16), torch.float16)
        guard = session.upload(torch.full((rows * 16 + 32,), 7.0, dtype=torch.float32))
        output = guard.view((rows, 16), 16)
        for kernel in kernels:
            a = (torch.randint(-8, 9, (rows, reduction), generator=generator).float() / 16).half()
            b = (torch.randint(-8, 9, (reduction, 16), generator=generator).float() / 16).half()
            expected = 4 * (a.float() @ b.float())
            lhs.copy_from(a)
            rhs.copy_from(b)
            kernel(lhs, rhs, output, blocks=1)
            # Dyadic products and all sums are exact in FP32; no tolerance is used.
            assert torch.equal(output.to_cpu(), expected)
            full = guard.to_cpu()
            assert torch.all(full[:16] == 7) and torch.all(full[-16:] == 7)


def test_310p_vec_to_mat_independent_images(tmp_path):
    compiled = ir.compile(
        _vec_to_mat_gemm(32, 32).specialize(),
        backend_type=BackendType.Ascend310P,
        output_dir=str(tmp_path),
        dump_passes=False,
    )
    generator = torch.Generator().manual_seed(719)
    with A1L2Worker(compiled) as worker:
        kernel = worker.prepare_independent(ir.CompiledProgram.from_dir(compiled.output_dir), 8)
        lhs = [worker.empty((32, 32), torch.float16) for _ in range(8)]
        rhs = [worker.empty((32, 16), torch.float16) for _ in range(8)]
        guards = [worker.upload(torch.full((544,), 7.0, dtype=torch.float32)) for _ in range(8)]
        outputs = [guard.view((32, 16), 16) for guard in guards]
        arguments = tuple(item for values in zip(lhs, rhs, outputs, strict=True) for item in values)
        graph = worker.register_graph([(kernel, arguments, 8)])
        assert graph.task_count == 1
        for replay in range(2):
            expected = []
            for a, b in zip(lhs, rhs, strict=True):
                left = (torch.randint(-8, 9, (32, 32), generator=generator).float() / 16).half()
                right = (torch.randint(-8, 9, (32, 16), generator=generator).float() / 16).half()
                expected.append(4 * (left.float() @ right.float()))
                a.copy_from(left)
                b.copy_from(right)
            graph()
            for out, reference, guard in zip(outputs, expected, guards, strict=True):
                assert torch.equal(out.to_cpu(), reference)
                full = guard.to_cpu()
                assert torch.all(full[:16] == 7) and torch.all(full[-16:] == 7)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def _left_panel_gemm(rows, replicate):
    row_offset = 0 if replicate else 16
    panel_rows = rows if replicate else 16

    @pl.jit.incore
    def compute(
        initial: pl.Tensor[[rows, 64], pl.FP16],
        panel: pl.Tensor[[panel_rows, 16], pl.FP16],
        rhs: pl.Tensor[[64, 16], pl.FP16],
        output: pl.Out[pl.Tensor[[rows, 16], pl.FP32]],
    ):
        # Reload and overwrite across lifetimes to exercise MTE1 readers,
        # Cube consumers and reuse of the source Mat and destination Left.
        for repeat in pl.range(0, 3):
            original = pl.load(initial, [0, 0], [rows, 64], target_memory=pl.MemorySpace.Mat)
            left = pl.tile.move(
                original,
                pl.MemorySpace.Left,
                blayout=pl.TileLayout.row_major,
                slayout=pl.TileLayout.row_major,
            )
            source = pl.load(panel, [0, 0], [panel_rows, 16], target_memory=pl.MemorySpace.Mat)
            if replicate:
                for column in pl.range(0, 4):
                    left = pl.tile.assemble(left, source, [0, column * 16])
            else:
                left = pl.tile.assemble(left, source, [row_offset, 16])
            right = pl.load(rhs, [0, 0], [64, 16], target_memory=pl.MemorySpace.Mat)
            product = pl.matmul(left, right)
            pl.store(product, [0, 0], output)
        return output

    return compute


@pytest.mark.parametrize("rows,replicate", [(32, False), (32, True), (112, True)])
def test_310p_left_panel_insert(tmp_path, rows, replicate):
    compute = _left_panel_gemm(rows, replicate)
    assert not compute._external_source_paths()
    compiled = ir.compile(
        compute.specialize(),
        backend_type=BackendType.Ascend310P,
        output_dir=str(tmp_path),
        dump_passes=False,
    )
    generated = next(tmp_path.rglob("*.pto")).read_text()
    assert "pto.tinsert" in generated
    generator = torch.Generator().manual_seed(953)
    panel_rows = rows if replicate else 16
    with A1ResidentSession(compiled) as session:
        kernels = [session.prepare(p) for p in (compiled, ir.CompiledProgram.from_dir(compiled.output_dir))]
        original = session.empty((rows, 64), torch.float16)
        panel = session.empty((panel_rows, 16), torch.float16)
        rhs = session.empty((64, 16), torch.float16)
        guard = session.upload(torch.full((rows * 16 + 32,), 7.0, dtype=torch.float32))
        output = guard.view((rows, 16), 16)
        for kernel in kernels:
            a, p, b = [
                (torch.randint(-8, 9, shape, generator=generator).float() / 16).half()
                for shape in [(rows, 64), (panel_rows, 16), (64, 16)]
            ]
            expected_left = a.clone()
            if replicate:
                expected_left = p.repeat(1, 4)
            else:
                expected_left[16:32, 16:32] = p
            expected = expected_left.float() @ b.float()
            original.copy_from(a)
            panel.copy_from(p)
            rhs.copy_from(b)
            kernel(original, panel, rhs, output, blocks=1)
            # All dyadic products/sums are exact; every untouched Left element
            # contributes to the patch case and the output has GM canaries.
            assert torch.equal(output.to_cpu(), expected)
            full = guard.to_cpu()
            assert torch.all(full[:16] == 7) and torch.all(full[-16:] == 7)


def test_310p_left_panel_independent_images(tmp_path):
    compiled = ir.compile(
        _left_panel_gemm(32, True).specialize(),
        backend_type=BackendType.Ascend310P,
        output_dir=str(tmp_path),
        dump_passes=False,
    )
    generator = torch.Generator().manual_seed(957)
    with A1L2Worker(compiled) as worker:
        kernel = worker.prepare_independent(ir.CompiledProgram.from_dir(compiled.output_dir), 8)
        originals = [worker.empty((32, 64), torch.float16) for _ in range(8)]
        panels = [worker.empty((32, 16), torch.float16) for _ in range(8)]
        rights = [worker.empty((64, 16), torch.float16) for _ in range(8)]
        guards = [worker.upload(torch.full((544,), 7.0, dtype=torch.float32)) for _ in range(8)]
        outputs = [guard.view((32, 16), 16) for guard in guards]
        arguments = tuple(
            item for values in zip(originals, panels, rights, outputs, strict=True) for item in values
        )
        graph = worker.register_graph([(kernel, arguments, 8)])
        assert graph.task_count == 1
        for replay in range(2):
            expected = []
            for original, panel, right in zip(originals, panels, rights, strict=True):
                a, p, b = [
                    (torch.randint(-8, 9, shape, generator=generator).float() / 16).half()
                    for shape in [(32, 64), (32, 16), (64, 16)]
                ]
                original.copy_from(a)
                panel.copy_from(p)
                right.copy_from(b)
                expected.append(p.repeat(1, 4).float() @ b.float())
            graph()
            for result, oracle, guard in zip(outputs, expected, guards, strict=True):
                assert torch.equal(result.to_cpu(), oracle)
                full = guard.to_cpu()
                assert torch.all(full[:16] == 7) and torch.all(full[-16:] == 7)


def test_310p_inline_tile_epilogue(tmp_path):
    @pl.jit.inline
    def epilogue(
        left: pl.Tile[[32, 32], pl.FP16],
        right: pl.Tile[[32, 16], pl.FP16],
        bias: pl.Tile[[1, 16], pl.FP32],
    ) -> pl.Tile[[32, 16], pl.FP16]:
        product = pl.matmul(left, right)
        return pl.cast(pl.add(product, pl.tile.col_expand(product, bias)), pl.FP16, mode="none")

    @pl.jit.incore
    def compute(
        lhs: pl.Tensor[[32, 32], pl.FP16],
        rhs: pl.Tensor[[32, 16], pl.FP16],
        bias: pl.Tensor[[1, 16], pl.FP32],
        output: pl.Out[pl.Tensor[[32, 16], pl.FP16]],
    ):
        left = pl.load(lhs, [0, 0], [32, 32], target_memory=pl.MemorySpace.Mat)
        right = pl.load(rhs, [0, 0], [32, 16], target_memory=pl.MemorySpace.Mat)
        vbias = pl.load(bias, [0, 0], [1, 16])
        rounded = epilogue(left, right, vbias)
        pl.store(rounded, [0, 0], output)
        return output

    assert not compute._external_source_paths()
    compiled = ir.compile(
        compute.specialize(),
        backend_type=BackendType.Ascend310P,
        output_dir=str(tmp_path),
        dump_passes=False,
    )
    generator = torch.Generator().manual_seed(1061)
    with A1ResidentSession(compiled) as session:
        calls = [session.prepare(p) for p in (compiled, ir.CompiledProgram.from_dir(compiled.output_dir))]
        left = session.empty((32, 32), torch.float16)
        right = session.empty((32, 16), torch.float16)
        bias = session.empty((1, 16), torch.float32)
        guard = session.upload(torch.full((544,), 7.0, dtype=torch.float16))
        output = guard.view((32, 16), 16)
        for call in calls:
            a, b, c = [
                torch.randint(-8, 9, shape, generator=generator).float() / 16
                for shape in [(32, 32), (32, 16), (1, 16)]
            ]
            left.copy_from(a.half())
            right.copy_from(b.half())
            bias.copy_from(c)
            call(left, right, bias, output, blocks=1)
            assert torch.equal(output.to_cpu(), (a @ b + c).half())
            full = guard.to_cpu()
            assert torch.all(full[:16] == 7) and torch.all(full[-16:] == 7)
