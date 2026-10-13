# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""310P compiler selection, independent capabilities, and runtime boundaries."""

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
        RunConfig(platform=platform, **{target_keyword: BackendType.Ascend310P})


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
    with pytest.raises(ValueError, match="path"):
        target.find_mem_path(ir.MemorySpace.Acc, ir.MemorySpace.DDR)
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
