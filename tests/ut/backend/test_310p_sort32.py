# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Existing public sort32 on A1: planner scratch, exact pairs and real Tensor ABI."""

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pypto.language as pl
import pytest
from pypto import ir, passes
from pypto.backend import BackendType
from pypto.runtime.pto_isa import pto_isa_include_dir


def sort32_kernel(
    rows: int,
    columns: int,
    scratch: int | None = None,
    *,
    mode: str = "tile",
    dtype: Any = pl.FP32,
    index_dtype: Any = pl.UINT32,
) -> Any:
    panels = 8 if mode == "spmd" else 5
    total_rows = rows * panels
    input_stride = columns + 16
    output_columns = columns * (4 if dtype == pl.FP16 else 2)
    output_stride = output_columns + 16

    if scratch is None:

        @pl.jit.inline
        def sort_values(
            values: pl.Tile[[rows, columns], dtype], positions: pl.Tile[[rows, columns], index_dtype]
        ) -> pl.Tile[[rows, output_columns], dtype]:
            return pl.sort32(values, positions)
    else:

        @pl.jit.inline
        def sort_values(
            values: pl.Tile[[rows, columns], dtype], positions: pl.Tile[[rows, columns], index_dtype]
        ) -> pl.Tile[[rows, output_columns], dtype]:
            workspace = pl.create_tile([1, scratch], dtype)
            return pl.tile.sort32(values, positions, tmp=workspace)

    if mode == "spmd":

        @pl.jit
        def sort_spmd(
            source: pl.Tensor[[total_rows * input_stride], dtype],
            indices: pl.Tensor[[total_rows * input_stride], index_dtype],
            output: pl.Out[pl.Tensor[[total_rows * output_stride], dtype]],
        ):
            for block in pl.spmd(8):
                source_view = pl.tensor.view(source, [total_rows, input_stride])
                index_view = pl.tensor.view(indices, [total_rows, input_stride])
                output_view = pl.tensor.view(output, [total_rows, output_stride])
                values = pl.load(source_view, [block * rows, 8], [rows, columns])
                positions = pl.load(index_view, [block * rows, 8], [rows, columns])
                result = sort_values(values, positions)
                pl.store(result, [block * rows, 8], output_view)
            return output

        return sort_spmd

    if mode == "pipeline":

        @pl.jit.incore
        def sort_pipeline(
            source: pl.Tensor[[total_rows * input_stride], dtype],
            indices: pl.Tensor[[total_rows * input_stride], index_dtype],
            output: pl.Out[pl.Tensor[[total_rows * output_stride], dtype]],
        ):
            source_view = pl.tensor.view(source, [total_rows, input_stride])
            index_view = pl.tensor.view(indices, [total_rows, input_stride])
            output_view = pl.tensor.view(output, [total_rows, output_stride])
            for block in pl.pipeline(panels, stage=2):
                values = pl.load(source_view, [block * rows, 8], [rows, columns])
                positions = pl.load(index_view, [block * rows, 8], [rows, columns])
                result = sort_values(values, positions)
                pl.store(result, [block * rows, 8], output_view)
            return output

        return sort_pipeline

    if mode == "tensor":

        @pl.jit.incore
        def sort_tensor(
            source: pl.Tensor[[total_rows * input_stride], dtype],
            indices: pl.Tensor[[total_rows * input_stride], index_dtype],
            output: pl.Out[pl.Tensor[[total_rows * output_stride], dtype]],
        ):
            source_view = pl.tensor.view(source, [total_rows, input_stride])
            index_view = pl.tensor.view(indices, [total_rows, input_stride])
            output_view = pl.tensor.view(output, [total_rows, output_stride])
            for block in pl.range(pl.tile.get_block_idx(), panels, pl.tile.get_block_num()):
                values = pl.tensor.slice(source_view, [rows, columns], [block * rows, 8])
                positions = pl.tensor.slice(index_view, [rows, columns], [block * rows, 8])
                result = pl.sort32(values, positions)
                pl.tensor.assemble(output_view, result, [block * rows, 8])
            return output

        return sort_tensor

    @pl.jit.incore
    def sort_tile(
        source: pl.Tensor[[total_rows * input_stride], dtype],
        indices: pl.Tensor[[total_rows * input_stride], index_dtype],
        output: pl.Out[pl.Tensor[[total_rows * output_stride], dtype]],
    ):
        source_view = pl.tensor.view(source, [total_rows, input_stride])
        index_view = pl.tensor.view(indices, [total_rows, input_stride])
        output_view = pl.tensor.view(output, [total_rows, output_stride])
        for block in pl.range(pl.tile.get_block_idx(), panels, pl.tile.get_block_num()):
            values = pl.load(source_view, [block * rows, 8], [rows, columns])
            positions = pl.load(index_view, [block * rows, 8], [rows, columns])
            result = sort_values(values, positions)
            pl.store(result, [block * rows, 8], output_view)
        return output

    return sort_tile


def test_public_sort32_implicit_default_compile(tmp_path):
    ir.compile(
        sort32_kernel(1, 32).specialize(), backend_type=BackendType.Ascend310P, output_dir=str(tmp_path)
    )
    sources = list((tmp_path / "kernels").glob("*/*.cpp"))
    assert len(sources) == 1
    assert "TSORT32(" in sources[0].read_text()


_CASES = [
    (1, 32, None, "tile", passes.MemoryPlanner.PYPTO),
    (3, 160, None, "tile", passes.MemoryPlanner.PYPTO),
    (8, 96, 1536, "tile", passes.MemoryPlanner.PYPTO),
    (1, 512, None, "tile", passes.MemoryPlanner.PYPTO),
    (2, 512, 4096, "tile", passes.MemoryPlanner.DSA_RP),
    (3, 160, 1040, "tile", passes.MemoryPlanner.PTOAS),
    (3, 160, None, "tile", passes.MemoryPlanner.PTOAS),
    (3, 96, None, "tensor", passes.MemoryPlanner.PYPTO),
    (3, 96, None, "tensor", passes.MemoryPlanner.PTOAS),
    (3, 96, None, "pipeline", passes.MemoryPlanner.PYPTO),
    (3, 96, None, "spmd", passes.MemoryPlanner.PYPTO),
]


@pytest.fixture(scope="module", params=_CASES)
def compiled_sort(request, tmp_path_factory):
    rows, columns, scratch, mode, planner = request.param
    output = tmp_path_factory.mktemp("public-sort32")
    ir.compile(
        sort32_kernel(rows, columns, scratch, mode=mode).specialize(),
        backend_type=BackendType.Ascend310P,
        memory_planner=planner,
        output_dir=str(output),
    )
    sources = list((output / "kernels").glob("*/*.cpp"))
    assert len(sources) == 1
    code = sources[0].read_text()
    assert "TSORT32(" in code
    assert "wait_flag(PIPE_MTE2, PIPE_V," in code
    assert "wait_flag(PIPE_S, PIPE_MTE3," in code
    assert "TSETVALIDSHAPE(" not in code
    metadata = json.loads((output / "compiled_meta.json").read_text())
    assert [param["dtype"] for param in metadata["params"]] == ["fp32", "uint32", "fp32"]
    order = [0, 1, 2]
    panels = 8 if mode == "spmd" else 5
    if mode == "spmd":
        manifest = json.loads((output / "a1_spmd.json").read_text())
        assert manifest["blocks"] == 8 and manifest["task_count"] == 1
        order = manifest["argument_order"]
        assert sorted(order) == [0, 1, 2]
    return rows, columns, panels, mode, sources[0], order


def _includes() -> list[str]:
    runtime = Path(__file__).resolve().parents[3] / "runtime"
    return [
        f"-I{p}"
        for p in (
            pto_isa_include_dir(),
            runtime / "src/a1/platform/include",
            runtime / "src/common",
            runtime / "src/common/task_interface",
            runtime / "src/common/platform/include",
        )
    ]


def test_public_sort32_cpu_tensor_abi(compiled_sort, tmp_path):
    compiler = shutil.which("clang++-19")
    assert compiler, "requires clang++-19 for CPU instruction semantics"
    rows, columns, panels, mode, generated, order = compiled_sort
    source = tmp_path / "oracle.cpp"
    source.write_text(
        "#include <cstdint>\n"
        + generated.read_text()
        + "\n"
        + Path(__file__).with_name("sort32_oracle.cpp").read_text()
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
            f"-DSORT_ROWS={rows}",
            f"-DSORT_COLS={columns}",
            f"-DSORT_PANELS={panels}",
            *[f"-DSORT_ARG{i}={value}" for i, value in enumerate(order)],
            *_includes(),
            str(source),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    cores = (8,) if mode == "spmd" else (1,) if mode == "pipeline" else (1, 3, 8)
    for count in cores:
        subprocess.run(
            [str(binary)],
            check=True,
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "PTO_CPU_SIM_NUM_CORES": str(count),
                "PTO_CPU_SIM_TRACE_ENABLE": "0",
                "OMP_NUM_THREADS": "1",
            },
        )


def test_public_sort32_native_compile(compiled_sort, tmp_path):
    compiler = shutil.which("ccec")
    assert compiler, "requires CCEC for offline dav-m200 compilation"
    source = compiled_sort[4]
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


@pytest.mark.parametrize(
    "columns,scratch,dtype,index_dtype,diagnostic",
    [
        (48, 512, pl.FP32, pl.UINT32, "32-value"),
        (32, 504, pl.FP32, pl.UINT32, "512"),
        (32, 512, pl.FP16, pl.UINT32, "FP32"),
        (32, 512, pl.FP32, pl.INT32, "UINT32"),
    ],
)
def test_public_sort32_rejects_invalid_profile(tmp_path, columns, scratch, dtype, index_dtype, diagnostic):
    with pytest.raises(RuntimeError, match=diagnostic):
        ir.compile(
            sort32_kernel(1, columns, scratch, dtype=dtype, index_dtype=index_dtype).specialize(),
            backend_type=BackendType.Ascend310P,
            output_dir=str(tmp_path),
        )
