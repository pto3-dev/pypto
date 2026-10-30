# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Public integer DMA compilation, real Tensor ABI and instruction semantics."""

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pypto.language as pl
import pytest
from pypto import ir
from pypto.backend import BackendType
from pypto.backend._a1_spmd import _single_call
from pypto.runtime.pto_isa import pto_isa_include_dir


def storage_kernel(
    dtype: Any,
    rows: int,
    columns: int,
    tile_rows: int,
    source_stride: int,
    destination_stride: int,
    source_column: int,
    destination_column: int,
) -> Any:
    rounds = rows // tile_rows

    @pl.jit.inline
    def panel(
        block: pl.Scalar[pl.INDEX],
        source: pl.Tensor[[rows * source_stride], dtype],
        destination: pl.Tensor[[rows * destination_stride], dtype],
    ):
        source_view = pl.tensor.view(source, [rows, source_stride])
        destination_view = pl.tensor.view(destination, [rows, destination_stride])
        data = pl.load(source_view, [block * tile_rows, source_column], [tile_rows, columns])
        pl.store(data, [block * tile_rows, destination_column], destination_view)
        return destination

    @pl.jit.incore
    def storage_copy(
        source: pl.Tensor[[rows * source_stride], dtype],
        destination: pl.Out[pl.Tensor[[rows * destination_stride], dtype]],
    ):
        if pl.tile.get_block_num() == 1:
            for block in pl.pipeline(rounds, stage=2):
                destination = panel(block, source, destination)
        else:
            for block in pl.range(pl.tile.get_block_idx(), rounds, pl.tile.get_block_num()):
                destination = panel(block, source, destination)
        return destination

    return storage_copy


def spmd_storage_kernel(dtype: Any) -> Any:
    @pl.jit
    def storage_spmd(
        source: pl.Tensor[[64 * 32], dtype],
        destination: pl.Out[pl.Tensor[[64 * 40], dtype]],
    ):
        for block in pl.spmd(8):
            source_view = pl.tensor.view(source, [64, 32])
            destination_view = pl.tensor.view(destination, [64, 40])
            data = pl.load(source_view, [block * 8, 8], [8, 8])
            pl.store(data, [block * 8, 16], destination_view)
        return destination

    return storage_spmd


# Full/odd panel counts, independent GM strides, nonzero columns, a short row
# panel and the native wide-gap DMA fallback. All fit the uint32 Tensor ABI.
_GEOMETRIES = [
    (64, 8, 8, 8, 8, 0, 0),
    (40, 8, 8, 32, 40, 8, 16),
    (24, 32, 8, 64, 96, 16, 32),
    (3, 8, 3, 24, 40, 8, 16),
    (16, 8, 8, 8 + 65536 * 8, 24, 0, 8),
]
_CASES = [(dtype, geometry, False) for dtype in (pl.INT32, pl.UINT32) for geometry in _GEOMETRIES] + [
    (dtype, (64, 8, 8, 32, 40, 8, 16), True) for dtype in (pl.INT32, pl.UINT32, pl.FP32)
]


def _dtype_info(dtype: Any) -> tuple[str, str, str]:
    if dtype == pl.INT32:
        return "int32", "int32_t", "INT32"
    if dtype == pl.UINT32:
        return "uint32", "uint32_t", "UINT32"
    assert dtype == pl.FP32
    return "fp32", "float", "FLOAT32"


@pytest.fixture(scope="module", params=_CASES)
def compiled_storage(request, tmp_path_factory):
    assert shutil.which("ptoas"), "requires PTOAS for complete public compilation"
    dtype, geometry, spmd = request.param
    output = tmp_path_factory.mktemp("integer-dma")
    kernel = spmd_storage_kernel(dtype) if spmd else storage_kernel(dtype, *geometry)
    compiled = ir.compile(
        kernel.specialize(),
        backend_type=BackendType.Ascend310P,
        output_dir=str(output),
    )
    sources = list((output / "kernels").glob("*/*.cpp"))
    assert len(sources) == 1
    order = [0, 1]
    if spmd:
        manifest = json.loads((output / "a1_spmd.json").read_text())
        assert manifest["blocks"] == 8 and manifest["task_count"] == 1
        assert manifest["argument_count"] == 2
        assert (output / manifest["kernel"]).resolve() == sources[0].resolve()
        order = manifest["argument_order"]
        assert sorted(order) == [0, 1]
    metadata = json.loads((output / "compiled_meta.json").read_text())
    expected = _dtype_info(dtype)[0]
    assert [param["dtype"] for param in metadata["params"]] == [expected, expected]
    assert metadata["params"][0]["direction"] == "In"
    assert metadata["params"][1]["direction"] in {"Out", "InOut"}
    return dtype, geometry, sources[0], compiled, spmd, order


def _includes() -> list[str]:
    runtime = Path(__file__).resolve().parents[3] / "runtime"
    return [
        f"-I{path}"
        for path in (
            pto_isa_include_dir(),
            runtime / "src/a1/platform/include",
            runtime / "src/common",
            runtime / "src/common/task_interface",
            runtime / "src/common/platform/include",
        )
    ]


def test_public_integer_dma_cpu_tensor_abi(compiled_storage, tmp_path):
    compiler = shutil.which("clang++-19")
    assert compiler, "requires clang++-19 for CPU instruction semantics"
    dtype, geometry, generated, _compiled, spmd, order = compiled_storage
    names = (
        "ROWS",
        "COLUMNS",
        "TILE_ROWS",
        "SOURCE_STRIDE",
        "DESTINATION_STRIDE",
        "SOURCE_COLUMN",
        "DESTINATION_COLUMN",
    )
    definitions = [f"-DDMA_{name}={value}" for name, value in zip(names, geometry, strict=True)]
    definitions += [f"-DDMA_ARG0={order[0]}", f"-DDMA_ARG1={order[1]}"]
    _name, ctype, native_dtype = _dtype_info(dtype)
    definitions += [f"-DDMA_CTYPE={ctype}", f"-DDMA_DTYPE=DataType::{native_dtype}"]
    source = tmp_path / "oracle.cpp"
    source.write_text(
        "#include <cstdint>\n"
        + generated.read_text()
        + "\n"
        + Path(__file__).with_name("integer_dma_oracle.cpp").read_text()
    )
    binary = tmp_path / "oracle"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O1",
            "-D__CPU_SIM",
            "-D__DAV_CUBE__",
            "-D__DAV_VEC__",
            *definitions,
            *_includes(),
            str(source),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    for cores in (8,) if spmd else (1, 3, 8):
        subprocess.run(
            [str(binary)],
            check=True,
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "PTO_CPU_SIM_NUM_CORES": str(cores),
                "PTO_CPU_SIM_TRACE_ENABLE": "0",
                "OMP_NUM_THREADS": "1",
            },
        )


def test_public_integer_dma_native_compile(compiled_storage, tmp_path):
    compiler = shutil.which("ccec")
    assert compiler, "requires CCEC for offline dav-m200 compilation"
    _dtype, _geometry, source, _compiled, _spmd, _order = compiled_storage
    subprocess.run(
        [
            compiler,
            "-c",
            "-O2",
            "-x",
            "cce",
            "-std=c++17",
            "--cce-aicore-only",
            "--cce-aicore-arch=dav-m200",
            *_includes(),
            str(source),
            "-o",
            str(tmp_path / "native.o"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("dtype", [pl.INT32, pl.UINT32])
def test_integer_arithmetic_remains_rejected(tmp_path, dtype):
    @pl.program
    class IntegerAdd:
        @pl.function(type=pl.FunctionType.InCore)
        def compute(self, src: pl.Tensor[[8, 8], dtype], out: pl.Tensor[[8, 8], dtype]):
            a = pl.load(src, [0, 0], [8, 8])
            pl.store(pl.add(a, a), [0, 0], out)

    with pytest.raises(RuntimeError, match="initial vector profile"):
        ir.compile(IntegerAdd, backend_type=BackendType.Ascend310P, output_dir=str(tmp_path), skip_ptoas=True)


def test_single_grid_bridge_keeps_void_calls_fail_closed():
    span = ir.Span.unknown()
    call = ir.Call(ir.GlobalVar("storage_kernel"), [], ir.VoidType(), span)
    statement = ir.EvalStmt(call, span)
    found = _single_call(statement)
    assert found is not None and found.same_as(call)
    assert _single_call(ir.SeqStmts([statement, statement], span)) is None
    barrier = ir.EvalStmt(ir.create_op_call("system.bar_all", [], span), span)
    assert _single_call(ir.SeqStmts([statement, barrier], span)) is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
