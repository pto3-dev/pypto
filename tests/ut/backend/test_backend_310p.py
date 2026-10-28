# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""310P compiler selection, independent capabilities, and runtime boundaries."""

import json
from pathlib import Path

import pypto.language as pl
import pytest
from pypto import backend, codegen, ir
from pypto.backend import BackendType
from pypto.runtime import ExecutionMode, RunConfig


@pytest.fixture(autouse=True)
def restore_backend_selection():
    """Preserve the caller's global backend selection across rejection tests."""
    previous = backend.get_backend_type() if backend.is_backend_configured() else None
    backend.reset_for_testing()
    yield
    backend.reset_for_testing()
    if previous is not None:
        backend.set_backend_type(previous)


def test_310p_is_a_distinct_target():
    assert BackendType.__members__["Ascend310P"] == BackendType.Ascend310P
    assert BackendType.Ascend310P != BackendType.Ascend910B
    assert BackendType.Ascend310P != BackendType.Ascend950
    assert BackendType.Ascend910B.value == 0
    assert BackendType.Ascend950.value == 1


def test_310p_does_not_return_an_a3_backend():
    target = backend.get_backend_instance(BackendType.Ascend310P)
    assert target is backend.Backend310P.instance()
    assert target.get_type_name() == "310P"
    assert target.get_handler().get_pto_target_arch() == "a1"
    assert target.get_handler().get_extra_ptoas_flags() == ["--pto-arch", "a1"]


def test_selection_is_idempotent():
    backend.set_backend_type(BackendType.Ascend310P)
    backend.set_backend_type(BackendType.Ascend310P)
    assert backend.get_backend_type() == BackendType.Ascend310P


def test_rejected_selection_preserves_an_existing_backend():
    backend.set_backend_type(BackendType.Ascend950)
    with pytest.raises(ValueError, match="cannot change to Ascend310P"):
        backend.set_backend_type(BackendType.Ascend310P)
    assert backend.get_backend_type() == BackendType.Ascend950


def test_runtime_310p_onboard_selection():
    config = RunConfig(arch=BackendType.Ascend310P, execution_mode=ExecutionMode.ONBOARD)
    assert config.platform == "a1"
    assert config.backend_type == BackendType.Ascend310P
    assert RunConfig(platform="a1").execution_mode is ExecutionMode.ONBOARD
    with pytest.raises(ValueError, match="Ascend310P simulation"):
        RunConfig(arch=BackendType.Ascend310P, execution_mode=ExecutionMode.SIM)


@pytest.mark.parametrize("platform", ["a2a3", "a2a3sim", "a5", "a5sim"])
def test_compile_rejects_310p_before_creating_artifacts(tmp_path, platform):
    output_dir = tmp_path / "must_not_exist"
    program = ir.Program([], "unsupported_310p", ir.Span.unknown())
    with pytest.raises(ValueError, match="Ascend310P requires platform"):
        ir.compile(
            program,
            backend_type=BackendType.Ascend310P,
            platform=platform,
            output_dir=str(output_dir),
            skip_ptoas=True,
        )
    assert not output_dir.exists()
    assert not backend.is_backend_configured()


@pytest.mark.parametrize("platform", ["a2a3", "a2a3sim", "a5", "a5sim"])
@pytest.mark.parametrize("target_keyword", ["arch", "backend_type"])
def test_legacy_runtime_platform_cannot_hide_310p(platform, target_keyword):
    with pytest.raises(ValueError, match="Ascend310P requires platform"):
        if target_keyword == "arch":
            RunConfig(platform=platform, arch=BackendType.Ascend310P)
        else:
            RunConfig(platform=platform, backend_type=BackendType.Ascend310P)


@pytest.mark.parametrize(
    ("backend_type", "available"),
    [(BackendType.Ascend910B, True), (BackendType.Ascend950, True), (BackendType.Ascend310P, True)],
)
def test_availability_query_does_not_configure_a_backend(backend_type, available):
    assert backend.is_backend_available(backend_type) is available
    assert not backend.is_backend_configured()


def test_conservative_310p_soc_and_routes():
    target = backend.Backend310P.instance()
    for space, size in [
        (ir.MemorySpace.Vec, 128 * 1024),
        (ir.MemorySpace.Mat, 512 * 1024),
        (ir.MemorySpace.Left, 64 * 1024),
        (ir.MemorySpace.Right, 64 * 1024),
        (ir.MemorySpace.Acc, 256 * 1024),
        (ir.MemorySpace.Bias, 0),
    ]:
        assert target.get_mem_size(space) == size
    assert target.get_core_count(ir.CoreType.CUBE) == 8
    assert target.get_core_count(ir.CoreType.VECTOR) == 7
    assert target.find_mem_path(ir.MemorySpace.DDR, ir.MemorySpace.Vec) == [
        ir.MemorySpace.DDR,
        ir.MemorySpace.Vec,
    ]
    assert target.find_mem_path(ir.MemorySpace.Acc, ir.MemorySpace.DDR) == [
        ir.MemorySpace.Acc,
        ir.MemorySpace.Vec,
        ir.MemorySpace.DDR,
    ]
    assert target.find_mem_path(ir.MemorySpace.DDR, ir.MemorySpace.Mat) == [
        ir.MemorySpace.DDR,
        ir.MemorySpace.Mat,
    ]
    assert target.get_handler().requires_coupled_incore_dispatch()
    assert target.get_handler().supports_acc_to_gm_via_vec(pl.FP32, pl.FP32)
    assert not target.get_handler().supports_acc_to_gm_dtype(pl.FP32)
    assert not target.get_handler().supports_acc_to_gm_via_vec(pl.FP32, pl.FP16)
    assert not backend.Backend910B.instance().get_handler().requires_coupled_incore_dispatch()
    assert not backend.Backend950.instance().get_handler().requires_coupled_incore_dispatch()
    handler = target.get_handler()
    assert not handler.requires_runtime_subblock_bridge()
    assert not handler.requires_no_split_dual_aiv_dispatch()
    with pytest.raises(ValueError, match="no supported simulator"):
        handler.get_default_sim_platform()


def _vector_add_program(dtype=pl.FP16):
    @pl.program
    class VectorAdd:
        @pl.function(type=pl.FunctionType.InCore)
        def add(
            self,
            lhs: pl.Tensor[[16, 128], dtype],
            rhs: pl.Tensor[[16, 128], dtype],
            out: pl.Tensor[[16, 128], dtype],
        ):
            a = pl.load(lhs, [0, 0], [16, 128])
            b = pl.load(rhs, [0, 0], [16, 128])
            pl.store(pl.add(a, b), [0, 0], out)

    return VectorAdd


def test_310p_fp16_code_generation():
    backend.set_backend_type(BackendType.Ascend310P)
    VectorAdd = _vector_add_program()
    transformed = ir.PassManager.get_strategy(ir.OptimizationStrategy.Default).run_passes(VectorAdd)
    result = codegen.PTOCodegen().generate(transformed)
    source = result if isinstance(result, str) else "".join(result.values())
    assert 'pto.target_arch = "a1"' in source
    assert "pto.tload" in source
    assert "pto.tadd" in source
    assert "pto.tstore" in source
    assert "pto.set_ffts" not in source


def test_compile_310p_defaults_to_a1(tmp_path):
    compiled = ir.compile(
        _vector_add_program(),
        backend_type=BackendType.Ascend310P,
        output_dir=str(tmp_path),
        skip_ptoas=True,
        dump_passes=False,
    )
    assert compiled.platform == "a1"
    sources = list(tmp_path.rglob("*.pto"))
    assert sources
    assert all('pto.target_arch = "a1"' in path.read_text() for path in sources)


def test_310p_rejects_unsupported_bf16_data():
    backend.set_backend_type(BackendType.Ascend310P)

    @pl.program
    class VectorBF16:
        @pl.function(type=pl.FunctionType.InCore)
        def load(self, src: pl.Tensor[[16, 128], pl.BF16], dst: pl.Tensor[[16, 128], pl.BF16]):
            tile = pl.load(src, [0, 0], [16, 128])
            pl.store(tile, [0, 0], dst)

    transformed = ir.PassManager.get_strategy(ir.OptimizationStrategy.Default).run_passes(VectorBF16)
    with pytest.raises(ValueError, match="Ascend310P initial vector profile requires FP16 or FP32"):
        codegen.PTOCodegen().generate(transformed)


def test_310p_does_not_inherit_unvalidated_vector_operations():
    backend.set_backend_type(BackendType.Ascend310P)

    @pl.program
    class VectorAbs:
        @pl.function(type=pl.FunctionType.InCore)
        def abs(
            self,
            lhs: pl.Tensor[[16, 128], pl.FP16],
            rhs: pl.Tensor[[16, 128], pl.FP16],
            out: pl.Tensor[[16, 128], pl.FP16],
        ):
            a = pl.load(lhs, [0, 0], [16, 128])
            pl.store(pl.abs(a), [0, 0], out)

    transformed = ir.PassManager.get_strategy(ir.OptimizationStrategy.Default).run_passes(VectorAbs)
    with pytest.raises(ValueError, match="No codegen registered for operation: tile.abs"):
        codegen.PTOCodegen().generate(transformed)


def test_310p_fp32_public_compilation(tmp_path):
    compiled = ir.compile(
        _vector_add_program(pl.FP32),
        backend_type=BackendType.Ascend310P,
        output_dir=str(tmp_path),
        skip_ptoas=True,
        dump_passes=False,
    )
    assert compiled.platform == "a1"
    sources = list(tmp_path.rglob("*.pto"))
    assert sources
    text = "".join(path.read_text() for path in sources)
    assert 'pto.target_arch = "a1"' in text
    assert "16x128xf32" in text or "dtype=f32" in text
    assert "pto.tadd" in text


def test_legacy_runtime_arch_cannot_hide_310p():
    with pytest.raises(ValueError, match="A1 architecture"):
        RunConfig(arch=BackendType.Ascend910B, backend_type=BackendType.Ascend310P)


def _sub_mul_program(dtype):
    @pl.program
    class SubMul:
        @pl.function(type=pl.FunctionType.InCore)
        def compute(
            self,
            lhs: pl.Tensor[[16, 128], dtype],
            rhs: pl.Tensor[[16, 128], dtype],
            out: pl.Tensor[[16, 128], dtype],
        ):
            a = pl.load(lhs, [0, 0], [16, 128])
            b = pl.load(rhs, [0, 0], [16, 128])
            difference = pl.sub(a, b)
            product = pl.mul(difference, b)
            pl.store(product, [0, 0], out)

    return SubMul


@pytest.mark.parametrize("dtype", [pl.FP16, pl.FP32])
def test_310p_subtraction_and_multiplication_are_registered(dtype):
    backend.set_backend_type(BackendType.Ascend310P)
    transformed = ir.PassManager.get_strategy(ir.OptimizationStrategy.Default).run_passes(
        _sub_mul_program(dtype)
    )
    result = codegen.PTOCodegen().generate(transformed)
    source = result if isinstance(result, str) else "".join(result.values())
    assert "pto.tsub" in source
    assert "pto.tmul" in source


def _strided_row_program(dtype):
    @pl.program
    class StridedRow:
        @pl.function(type=pl.FunctionType.InCore)
        def compute(
            self,
            lhs: pl.Tensor[[16, 256], dtype],
            rhs: pl.Tensor[[16, 256], dtype],
            out: pl.Tensor[[16, 256], dtype],
        ):
            a = pl.load(lhs, [0, 32], [16, 128])
            b = pl.load(rhs, [0, 32], [16, 128])
            total = pl.add(a, b)
            pl.store(total, [0, 32], out)

    return StridedRow


@pytest.mark.parametrize("dtype", [pl.FP16, pl.FP32])
def test_310p_strided_partition_retains_parent_geometry(dtype):
    backend.set_backend_type(BackendType.Ascend310P)
    transformed = ir.PassManager.get_strategy(ir.OptimizationStrategy.Default).run_passes(
        _strided_row_program(dtype)
    )
    result = codegen.PTOCodegen().generate(transformed)
    source = result if isinstance(result, str) else "".join(result.values())
    assert "pto.partition_view" in source
    assert "pto.tload" in source
    assert "pto.tstore" in source
    assert "256 : index" in source
    assert "32 : index" in source


def _masked_tail_program(dtype):
    cols = 16 if dtype == pl.FP16 else 8

    @pl.program
    class MaskedTail:
        @pl.function(type=pl.FunctionType.InCore)
        def compute(
            self,
            lhs: pl.Tensor[[1, cols], dtype],
            rhs: pl.Tensor[[1, cols], dtype],
            out: pl.Tensor[[1, cols], dtype],
        ):
            a = pl.load(lhs, [0, 0], [1, cols])
            b = pl.load(rhs, [0, 0], [1, cols])
            difference = pl.sub(a, b)
            product = pl.mul(difference, b)
            total = pl.add(product, a)
            pl.store(total, [0, 0], out)

    return MaskedTail


@pytest.mark.parametrize("dtype", [pl.FP16, pl.FP32])
def test_310p_generates_one_dma_block_vector_tail(dtype):
    backend.set_backend_type(BackendType.Ascend310P)
    transformed = ir.PassManager.get_strategy(ir.OptimizationStrategy.Default).run_passes(
        _masked_tail_program(dtype)
    )
    result = codegen.PTOCodegen().generate(transformed)
    source = result if isinstance(result, str) else "".join(result.values())
    assert "pto.tsub" in source
    assert "pto.tmul" in source
    assert "pto.tadd" in source
    assert "pto.tstore" in source


def _single_fractal_gemm_program(dtype=pl.FP16, rows=16):
    @pl.program
    class SingleFractalGemm:
        @pl.function(type=pl.FunctionType.InCore)
        def compute(
            self,
            lhs: pl.Tensor[[rows, 16], dtype],
            rhs: pl.Tensor[[16, 16], dtype],
            out: pl.Tensor[[rows, 16], pl.FP32],
        ):
            a = pl.load(lhs, [0, 0], [rows, 16], target_memory=pl.MemorySpace.Mat)
            b = pl.load(rhs, [0, 0], [16, 16], target_memory=pl.MemorySpace.Mat)
            product = pl.matmul(a, b)
            pl.store(product, [0, 0], out)

    return SingleFractalGemm


def test_310p_gemm_keeps_one_coupled_kernel_and_explicit_writeback(tmp_path):
    compiled = ir.compile(
        _single_fractal_gemm_program(),
        backend_type=BackendType.Ascend310P,
        output_dir=str(tmp_path),
        skip_ptoas=True,
        dump_passes=False,
    )
    pto_sources = list(tmp_path.rglob("*.pto"))
    assert len(pto_sources) == 1
    source = pto_sources[0].read_text()
    assert "pto.tmatmul" in source
    assert "acc_to_vec_mode<single_mode_vec0>" in source
    assert "tpush" not in source
    assert "tpop" not in source
    assert not (tmp_path / "orchestration").exists()
    assert compiled.platform == "a1"


@pytest.mark.parametrize(
    ("dtype", "rows", "message"),
    [
        (pl.FP32, 16, "FP16 inputs"),
        (pl.FP16, 17, "fractal boxes"),
    ],
)
def test_310p_gemm_rejects_unvalidated_profiles_before_ptoas(tmp_path, dtype, rows, message):
    with pytest.raises(RuntimeError, match=message):
        ir.compile(
            _single_fractal_gemm_program(dtype, rows),
            backend_type=BackendType.Ascend310P,
            output_dir=str(tmp_path),
            skip_ptoas=True,
            dump_passes=False,
        )


def _single_fractal_gemm_bias_program(rows=16):
    @pl.program
    class GemmBias:
        @pl.function(type=pl.FunctionType.InCore)
        def compute(
            self,
            lhs: pl.Tensor[[rows, 16], pl.FP16],
            rhs: pl.Tensor[[16, 16], pl.FP16],
            bias: pl.Tensor[[rows, 16], pl.FP32],
            out: pl.Tensor[[rows, 16], pl.FP32],
            raw: pl.Tensor[[rows, 16], pl.FP32],
        ):
            a = pl.load(lhs, [0, 0], [rows, 16], target_memory=pl.MemorySpace.Mat)
            b = pl.load(rhs, [0, 0], [16, 16], target_memory=pl.MemorySpace.Mat)
            bias_tile = pl.load(bias, [0, 0], [rows, 16])
            product = pl.matmul(a, b)
            pl.store(product, [0, 0], raw)
            pl.store(pl.add(product, bias_tile), [0, 0], out)

    return GemmBias


def test_310p_gemm_writeback_is_planned_before_vector_consumers(tmp_path):
    ir.compile(
        _single_fractal_gemm_bias_program(),
        backend_type=BackendType.Ascend310P,
        output_dir=str(tmp_path),
        skip_ptoas=True,
        dump_passes=False,
    )
    sources = list(tmp_path.rglob("*.pto"))
    assert len(sources) == 1
    source = sources[0].read_text()
    assert source.count("pto.tmatmul ") == 1
    assert source.count("acc_to_vec_mode<single_mode_vec0>") == 1
    assert "pto.tadd" in source
    assert source.count("pto.tstore ") == 2
    assert "tpush" not in source
    assert "tpop" not in source


def _floating_conversion_program(dtype, cols=16, mode="none"):
    converted_dtype = pl.FP32 if dtype == pl.FP16 else pl.FP16

    @pl.program
    class FloatingConversion:
        @pl.function(type=pl.FunctionType.InCore)
        def compute(
            self,
            source: pl.Tensor[[1, cols], dtype],
            destination: pl.Tensor[[1, cols], converted_dtype],
        ):
            tile = pl.load(source, [0, 0], [1, cols])
            converted = pl.cast(tile, converted_dtype, mode=mode)
            pl.store(converted, [0, 0], destination)

    return FloatingConversion


@pytest.mark.parametrize("dtype", [pl.FP16, pl.FP32])
@pytest.mark.parametrize("cols", [16, 80, 16336])
def test_310p_floating_conversion_codegen(tmp_path, dtype, cols):
    ir.compile(
        _floating_conversion_program(dtype, cols),
        backend_type=BackendType.Ascend310P,
        output_dir=str(tmp_path),
        skip_ptoas=True,
        dump_passes=False,
    )
    sources = list(tmp_path.rglob("*.pto"))
    assert len(sources) == 1
    source = sources[0].read_text()
    assert "pto.tcvt" in source
    assert "round_mode NONE" in source
    assert "tpush" not in source
    assert "tpop" not in source


@pytest.mark.parametrize("mode", ["round", "rint", "floor", "ceil", "trunc", "odd"])
def test_310p_floating_conversion_rejects_unvalidated_rounding(tmp_path, mode):
    with pytest.raises(RuntimeError, match="mode='none'"):
        ir.compile(
            _floating_conversion_program(pl.FP32, mode=mode),
            backend_type=BackendType.Ascend310P,
            output_dir=str(tmp_path),
            skip_ptoas=True,
            dump_passes=False,
        )


def _gemm_half_output_program(rows=16):
    @pl.program
    class GemmHalfOutput:
        @pl.function(type=pl.FunctionType.InCore)
        def compute(
            self,
            lhs: pl.Tensor[[rows, 16], pl.FP16],
            rhs: pl.Tensor[[16, 16], pl.FP16],
            output: pl.Tensor[[rows, 16], pl.FP16],
        ):
            a = pl.load(lhs, [0, 0], [rows, 16], target_memory=pl.MemorySpace.Mat)
            b = pl.load(rhs, [0, 0], [16, 16], target_memory=pl.MemorySpace.Mat)
            product = pl.matmul(a, b)
            pl.store(pl.cast(product, pl.FP16, mode="none"), [0, 0], output)

    return GemmHalfOutput


def test_310p_gemm_conversion_stages_to_vec_before_narrowing(tmp_path):
    ir.compile(
        _gemm_half_output_program(),
        backend_type=BackendType.Ascend310P,
        output_dir=str(tmp_path),
        skip_ptoas=True,
        dump_passes=False,
    )
    sources = list(tmp_path.rglob("*.pto"))
    assert len(sources) == 1
    source = sources[0].read_text()
    assert "pto.tmatmul" in source
    assert "acc_to_vec_mode<single_mode_vec0>" in source
    assert "pto.tcvt" in source
    assert "tpush" not in source
    assert "tpop" not in source


def test_310p_floating_conversion_rejects_explicit_scratch(tmp_path):
    @pl.program
    class CastWithScratch:
        @pl.function(type=pl.FunctionType.InCore)
        def compute(
            self,
            source: pl.Tensor[[1, 16], pl.FP32],
            destination: pl.Tensor[[1, 16], pl.FP16],
        ):
            tile = pl.load(source, [0, 0], [1, 16])
            converted = pl.tile.cast(tile, pl.FP16, mode="none", tmp=tile)
            pl.store(converted, [0, 0], destination)

    with pytest.raises(RuntimeError, match=r"without\s*\|?\s*scratch"):
        ir.compile(
            CastWithScratch,
            backend_type=BackendType.Ascend310P,
            output_dir=str(tmp_path),
            skip_ptoas=True,
            dump_passes=False,
        )


@pytest.mark.parametrize("rows", [32, 128, 1024, 2048])
def test_310p_narrow_gemm_row_profile_codegen(tmp_path, rows):
    ir.compile(
        _single_fractal_gemm_program(rows=rows),
        backend_type=BackendType.Ascend310P,
        output_dir=str(tmp_path),
        skip_ptoas=True,
        dump_passes=False,
    )
    sources = list(tmp_path.rglob("*.pto"))
    assert len(sources) == 1
    source = sources[0].read_text()
    assert "pto.tmatmul" in source
    assert "acc_to_vec_mode<single_mode_vec0>" in source
    assert "tpush" not in source
    assert "tpop" not in source


@pytest.mark.parametrize(
    ("rows", "error_type", "message"),
    [
        (1, RuntimeError, "full aligned"),
        (17, RuntimeError, "fractal boxes"),
        (2049, ValueError, "Vec buffer usage"),
        (2064, ValueError, "Vec buffer usage"),
    ],
)
def test_310p_narrow_gemm_rejects_unvalidated_rows(tmp_path, rows, error_type, message):
    with pytest.raises(error_type, match=message):
        ir.compile(
            _single_fractal_gemm_program(rows=rows),
            backend_type=BackendType.Ascend310P,
            output_dir=str(tmp_path),
            skip_ptoas=True,
            dump_passes=False,
        )


@pytest.mark.parametrize("rows", [32, 128])
def test_310p_narrow_gemm_row_profile_keeps_vector_consumers_coupled(tmp_path, rows):
    for name, program in [
        ("bias", _single_fractal_gemm_bias_program(rows)),
        ("half", _gemm_half_output_program(rows)),
    ]:
        folder = tmp_path / name
        ir.compile(
            program,
            backend_type=BackendType.Ascend310P,
            output_dir=str(folder),
            skip_ptoas=True,
            dump_passes=False,
        )
        sources = list(folder.rglob("*.pto"))
        assert len(sources) == 1
        source = sources[0].read_text()
        assert "pto.tmatmul" in source
        assert "acc_to_vec_mode<single_mode_vec0>" in source
        assert "tpush" not in source
        assert "tpop" not in source
        assert ("pto.tadd" if name == "bias" else "pto.tcvt") in source


def _narrow_gemm_accumulation_program(rows=16, init_cond=None):
    @pl.program
    class NarrowGemmAccumulation:
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

    return NarrowGemmAccumulation


@pytest.mark.parametrize("rows", [16, 32, 128, 1024])
@pytest.mark.parametrize("init_cond", [None, False, True])
def test_310p_narrow_gemm_accumulation_codegen(tmp_path, rows, init_cond):
    ir.compile(
        _narrow_gemm_accumulation_program(rows, init_cond),
        backend_type=BackendType.Ascend310P,
        output_dir=str(tmp_path),
        skip_ptoas=True,
        dump_passes=False,
    )
    sources = list(tmp_path.rglob("*.pto"))
    assert len(sources) == 1
    source = sources[0].read_text()
    assert source.count("pto.tmatmul.acc") == (1 if init_cond else 2)
    assert "acc_to_vec_mode<single_mode_vec0>" in source
    assert "tpush" not in source
    assert "tpop" not in source


def _tiled_gemm_32_program(rows=16):
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

    return TiledGemm32


@pytest.mark.parametrize("rows", [16, 32, 128, 512, 1024])
def test_310p_strided_matrix_panels_codegen(tmp_path, rows):
    ir.compile(
        _tiled_gemm_32_program(rows),
        backend_type=BackendType.Ascend310P,
        output_dir=str(tmp_path),
        skip_ptoas=True,
        dump_passes=False,
    )
    sources = list(tmp_path.rglob("*.pto"))
    assert len(sources) == 1
    source = sources[0].read_text()
    assert source.count("pto.tmatmul.acc") == 2
    assert source.count("acc_to_vec_mode<single_mode_vec0>") == 2
    assert "tpush" not in source
    assert "tpop" not in source


def _exp_div_program(dtype, cols=80, high_precision=False):
    @pl.program
    class ExpDiv:
        @pl.function(type=pl.FunctionType.InCore)
        def compute(
            self,
            lhs: pl.Tensor[[1, cols], dtype],
            rhs: pl.Tensor[[1, cols], dtype],
            output: pl.Tensor[[1, cols], dtype],
        ):
            a = pl.load(lhs, [0, 0], [1, cols])
            b = pl.load(rhs, [0, 0], [1, cols])
            exponential = pl.exp(a)
            result = pl.div(exponential, b, high_precision=high_precision)
            pl.store(result, [0, 0], output)

    return ExpDiv


@pytest.mark.parametrize("dtype", [pl.FP16, pl.FP32])
@pytest.mark.parametrize("cols", [16, 80, 2048])
def test_310p_exp_div_codegen(tmp_path, dtype, cols):
    ir.compile(
        _exp_div_program(dtype, cols),
        backend_type=BackendType.Ascend310P,
        output_dir=str(tmp_path),
        skip_ptoas=True,
        dump_passes=False,
    )
    sources = list(tmp_path.rglob("*.pto"))
    assert len(sources) == 1
    assert "pto.texp" in sources[0].read_text()
    assert "pto.tdiv" in sources[0].read_text()


@pytest.mark.parametrize("dtype", [pl.FP16, pl.FP32])
def test_310p_div_rejects_high_precision(tmp_path, dtype):
    with pytest.raises(RuntimeError, match="Ascend310P division supports default precision only"):
        ir.compile(
            _exp_div_program(dtype, high_precision=True),
            backend_type=BackendType.Ascend310P,
            output_dir=str(tmp_path),
            skip_ptoas=True,
            dump_passes=False,
        )


def _scalar_broadcast_program(dtype, cols=80, value=-2.5):
    @pl.program
    class ScalarBroadcast:
        @pl.function(type=pl.FunctionType.InCore)
        def compute(self, output: pl.Tensor[[1, cols], dtype]):
            filled = pl.tile.full([1, cols], dtype=dtype, value=value)
            pl.store(filled, [0, 0], output)

    return ScalarBroadcast


@pytest.mark.parametrize("dtype", [pl.FP16, pl.FP32])
@pytest.mark.parametrize("cols", [16, 80, 16336])
def test_310p_scalar_broadcast_codegen(tmp_path, dtype, cols):
    ir.compile(
        _scalar_broadcast_program(dtype, cols),
        backend_type=BackendType.Ascend310P,
        output_dir=str(tmp_path),
        skip_ptoas=True,
        dump_passes=False,
    )
    sources = list(tmp_path.rglob("*.pto"))
    assert len(sources) == 1
    assert "pto.texpands" in sources[0].read_text()


def _axis_reduction_program(dtype, kind, rows=16, cols=80):
    if kind == "row_sum":

        @pl.program
        class RowSum:
            @pl.function(type=pl.FunctionType.InCore)
            def compute(self, x: pl.Tensor[[rows, cols], dtype], output: pl.Tensor[[rows, 1], dtype]):
                a = pl.load(x, [0, 0], [rows, cols])
                tmp = pl.tile.create([rows, cols], dtype=dtype)
                result = pl.tile.row_sum(a, tmp)
                pl.store(result, [0, 0], output)

        return RowSum

    elif kind == "row_max":

        @pl.program
        class RowMax:
            @pl.function(type=pl.FunctionType.InCore)
            def compute(self, x: pl.Tensor[[rows, cols], dtype], output: pl.Tensor[[rows, 1], dtype]):
                a = pl.load(x, [0, 0], [rows, cols])
                tmp = pl.tile.create([rows, cols], dtype=dtype)
                result = pl.tile.row_max(a, tmp)
                pl.store(result, [0, 0], output)

        return RowMax

    elif kind == "col_sum":

        @pl.program
        class ColSum:
            @pl.function(type=pl.FunctionType.InCore)
            def compute(self, x: pl.Tensor[[rows, cols], dtype], output: pl.Tensor[[1, cols], dtype]):
                a = pl.load(x, [0, 0], [rows, cols])
                result = pl.tile.col_sum(a)
                pl.store(result, [0, 0], output)

        return ColSum

    elif kind == "col_max":

        @pl.program
        class ColMax:
            @pl.function(type=pl.FunctionType.InCore)
            def compute(self, x: pl.Tensor[[rows, cols], dtype], output: pl.Tensor[[1, cols], dtype]):
                a = pl.load(x, [0, 0], [rows, cols])
                result = pl.tile.col_max(a)
                pl.store(result, [0, 0], output)

        return ColMax
    raise ValueError(f"Unsupported reduction kind: {kind}")


@pytest.mark.parametrize("dtype", [pl.FP16, pl.FP32])
@pytest.mark.parametrize("kind", ["row_sum", "row_max", "col_sum", "col_max"])
@pytest.mark.parametrize(("rows", "cols"), [(16, 16), (16, 80), (32, 256), (16, 400), (80, 80)])
def test_310p_axis_reduction_codegen(tmp_path, dtype, kind, rows, cols):
    ir.compile(
        _axis_reduction_program(dtype, kind, rows, cols),
        backend_type=BackendType.Ascend310P,
        output_dir=str(tmp_path),
        skip_ptoas=True,
        dump_passes=False,
    )
    sources = list(tmp_path.rglob("*.pto"))
    assert len(sources) == 1
    assert f"pto.t{kind.replace('_', '')}" in sources[0].read_text()


@pytest.mark.parametrize("dtype", [pl.FP16, pl.FP32])
def test_310p_column_sum_rejects_binary_scratch(tmp_path, dtype):
    @pl.program
    class BinaryColumnSum:
        @pl.function(type=pl.FunctionType.InCore)
        def compute(self, x: pl.Tensor[[16, 80], dtype], output: pl.Tensor[[1, 80], dtype]):
            a = pl.load(x, [0, 0], [16, 80])
            tmp = pl.tile.create([16, 80], dtype=dtype)
            result = pl.tile.col_sum(a, tmp)
            pl.store(result, [0, 0], output)

    with pytest.raises(RuntimeError, match="column sum supports the sequential form"):
        ir.compile(
            BinaryColumnSum,
            backend_type=BackendType.Ascend310P,
            output_dir=str(tmp_path),
            skip_ptoas=True,
            dump_passes=False,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


@pytest.mark.parametrize("dtype", [pl.FP16, pl.FP32])
@pytest.mark.parametrize("row", [False, True])
def test_310p_reduction_broadcast_codegen(tmp_path, dtype, row):
    if row:

        @pl.jit.incore
        def row_kernel(
            x: pl.Tensor[[16, 80], dtype],
            scratch: pl.Tensor[[16, 80], dtype],
            output: pl.Tensor[[16, 80], dtype],
        ):
            a = pl.load(x, [0, 0], [16, 80])
            tmp = pl.load(scratch, [0, 0], [16, 80])
            reduced = pl.tile.row_max(a, tmp)
            result = pl.tile.row_expand(a, reduced)
            pl.store(result, [0, 0], output)

        kernel = row_kernel
    else:

        @pl.jit.incore
        def col_kernel(x: pl.Tensor[[16, 80], dtype], output: pl.Tensor[[16, 80], dtype]):
            a = pl.load(x, [0, 0], [16, 80])
            reduced = pl.tile.col_sum(a)
            result = pl.tile.col_expand(a, reduced)
            pl.store(result, [0, 0], output)

        kernel = col_kernel
    ir.compile(
        kernel.specialize(),
        backend_type=BackendType.Ascend310P,
        output_dir=str(tmp_path),
        skip_ptoas=True,
        dump_passes=False,
    )
    sources = list(tmp_path.rglob("*.pto"))
    assert len(sources) == 1
    assert ("pto.trowexpand" if row else "pto.tcolexpand") in sources[0].read_text()


def test_static_spmd_emits_one_a1_grid(tmp_path):
    """Public SPMD retains one dispatch and uses bounded physical identity."""

    @pl.jit
    def packed(
        x: pl.Tensor[[128, 16], pl.FP32],
        y: pl.Tensor[[128, 16], pl.FP32],
        out: pl.Out[pl.Tensor[[128, 16], pl.FP32]],
    ):
        for block in pl.spmd(8):
            a = pl.load(x, [block * 16, 0], [16, 16])
            b = pl.load(y, [block * 16, 0], [16, 16])
            out = pl.store(pl.add(a, b), [block * 16, 0], out)
        return out

    result = ir.compile(
        packed.specialize(),
        backend_type=BackendType.Ascend310P,
        output_dir=str(tmp_path),
        skip_ptoas=True,
        dump_passes=False,
    )
    manifest = json.loads((Path(result.output_dir) / "a1_spmd.json").read_text())
    assert manifest["blocks"] == 8 and manifest["task_count"] == 1
    assert manifest["argument_count"] == 3
    source = next(tmp_path.rglob("*.pto")).read_text()
    assert "pto.get_block_idx" in source


@pytest.mark.parametrize("dtype", [pl.FP16, pl.FP32])
def test_310p_flat_tensor_view_preserves_dma_geometry(dtype):
    @pl.program
    class Viewed:
        @pl.function(type=pl.FunctionType.InCore)
        def compute(self, source: pl.Tensor[[512], dtype], out: pl.Tensor[[512], dtype]):
            matrix = pl.tensor.view(source, [32, 16])
            destination = pl.tensor.view(out, [32, 16])
            value = pl.load(matrix, [16, 0], [16, 16])
            destination = pl.store(value, [16, 0], destination)
            return out

    backend.set_backend_type(BackendType.Ascend310P)
    transformed = ir.PassManager.get_strategy(ir.OptimizationStrategy.Default).run_passes(Viewed)
    result = codegen.PTOCodegen().generate(transformed)
    source = result if isinstance(result, str) else "".join(result.values())
    assert "pto.make_tensor_view" in source
    assert "pto.tload" in source and "pto.tstore" in source
    assert "tensor.view" not in source
