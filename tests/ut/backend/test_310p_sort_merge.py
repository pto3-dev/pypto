# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Resident public sort32-to-merge composition, exact pairs and native offline only."""

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


def sort_merge_kernel(columns: int, output_columns: int, *, extract_fields: bool = False) -> Any:
    panels = 5
    input_stride, output_stride = columns + 16, output_columns + 16
    input_count, output_count = panels * input_stride, panels * output_stride
    group_output = output_columns if columns == 128 else 256

    @pl.jit.inline
    def merge_group(pairs: pl.Tile[[1, 256], pl.FP32]):
        # One reshape exposes the four sorted 32-value runs as full rows.
        # Row slices are read-only storage aliases, never GM round trips.
        runs = pl.tile.reshape(pairs, [4, 64])
        a = pl.tile.slice(runs, [1, 64], [0, 0])
        b = pl.tile.slice(runs, [1, 64], [1, 0])
        c = pl.tile.slice(runs, [1, 64], [2, 0])
        d = pl.tile.slice(runs, [1, 64], [3, 0])
        scratch = pl.create_tile([1, 2048], pl.FP32)
        return pl.mrgsort(a, b, c, d, tmp=scratch, output_cols=group_output)

    if columns == 128:

        @pl.jit.inline
        def merge_all(pairs: pl.Tile[[1, 256], pl.FP32]):
            return merge_group(pairs)
    else:

        @pl.jit.inline
        def merge_all(pairs: pl.Tile[[1, 1024], pl.FP32]):
            groups = pl.tile.reshape(pairs, [4, 256])
            first = merge_group(pl.tile.slice(groups, [1, 256], [0, 0]))
            second = merge_group(pl.tile.slice(groups, [1, 256], [1, 0]))
            third = merge_group(pl.tile.slice(groups, [1, 256], [2, 0]))
            fourth = merge_group(pl.tile.slice(groups, [1, 256], [3, 0]))
            scratch = pl.create_tile([1, 8192], pl.FP32)
            return pl.tile.mrgsort(first, second, third, fourth, tmp=scratch, output_cols=output_columns)

    if extract_fields:
        field_columns = output_columns // 2
        field_stride = field_columns + 16
        field_count = panels * field_stride

        @pl.jit.incore
        def resident_sort_extract(
            source: pl.Tensor[[input_count], pl.FP32],
            indices: pl.Tensor[[input_count], pl.UINT32],
            values: pl.Out[pl.Tensor[[field_count], pl.FP32]],
            positions: pl.Out[pl.Tensor[[field_count], pl.UINT32]],
        ):
            source_view = pl.tensor.view(source, [panels, input_stride])
            indices_view = pl.tensor.view(indices, [panels, input_stride])
            values_view = pl.tensor.view(values, [panels, field_stride])
            positions_view = pl.tensor.view(positions, [panels, field_stride])
            for panel in pl.range(pl.tile.get_block_idx(), panels, pl.tile.get_block_num()):
                data = pl.load(source_view, [panel, 8], [1, columns])
                locations = pl.load(indices_view, [panel, 8], [1, columns])
                pairs = pl.sort32(data, locations)
                merged = merge_all(pairs)
                selected = pl.tile.gather_mask(merged, pl.tile.MaskPattern.P0101)
                selected_indices = pl.tile.gather_mask(
                    merged, pl.tile.MaskPattern.P1010, output_dtype=pl.UINT32
                )
                pl.store(selected, [panel, 8], values_view)
                pl.store(selected_indices, [panel, 8], positions_view)
            return values, positions

        return resident_sort_extract

    @pl.jit.incore
    def resident_sort_merge(
        source: pl.Tensor[[input_count], pl.FP32],
        indices: pl.Tensor[[input_count], pl.UINT32],
        output: pl.Out[pl.Tensor[[output_count], pl.FP32]],
    ):
        values_view = pl.tensor.view(source, [panels, input_stride])
        indices_view = pl.tensor.view(indices, [panels, input_stride])
        output_view = pl.tensor.view(output, [panels, output_stride])
        for panel in pl.range(pl.tile.get_block_idx(), panels, pl.tile.get_block_num()):
            values = pl.load(values_view, [panel, 8], [1, columns])
            positions = pl.load(indices_view, [panel, 8], [1, columns])
            pairs = pl.sort32(values, positions)
            result = merge_all(pairs)
            pl.store(result, [panel, 8], output_view)
        return output

    return resident_sort_merge


_CASES = [
    (128, 256, passes.MemoryPlanner.PYPTO),
    (128, 80, passes.MemoryPlanner.PTOAS),
    (512, 600, passes.MemoryPlanner.PYPTO),
    (512, 608, passes.MemoryPlanner.DSA_RP),
    (512, 600, passes.MemoryPlanner.PTOAS),
    (512, 608, passes.MemoryPlanner.PTOAS),
]


@pytest.fixture(scope="module", params=_CASES)
def compiled_sort_merge(request, tmp_path_factory):
    columns, output_columns, planner = request.param
    directory = tmp_path_factory.mktemp("resident-sort-merge")
    ir.compile(
        sort_merge_kernel(columns, output_columns).specialize(),
        backend_type=BackendType.Ascend310P,
        memory_planner=planner,
        output_dir=str(directory),
    )
    sources = list((directory / "kernels").glob("*/*.cpp"))
    assert len(sources) == 1
    code = sources[0].read_text()
    assert code.count("TSORT32(") == 1
    assert code.count("TMRGSORT<") == (1 if columns == 128 else 5)
    assert code.count("TLOAD(") == 2 and code.count("TSTORE(") == 1
    assert "TSETVALIDSHAPE(" not in code
    metadata = json.loads((directory / "compiled_meta.json").read_text())
    assert [p["dtype"] for p in metadata["params"]] == ["fp32", "uint32", "fp32"]
    return columns, output_columns, sources[0]


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


def test_resident_sort_merge_cpu(compiled_sort_merge, tmp_path):
    columns, output_columns, generated = compiled_sort_merge
    compiler = shutil.which("clang++-19")
    assert compiler, "requires clang++-19 for CPU instruction semantics"
    source = tmp_path / "oracle.cpp"
    source.write_text(
        "#include <cstdint>\n"
        + generated.read_text()
        + "\n"
        + Path(__file__).with_name("sort_merge_oracle.cpp").read_text()
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
            f"-DSORT_COLS={columns}",
            f"-DCHAIN_OUTPUT_COLS={output_columns}",
            *_includes(),
            str(source),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    for count in (1, 3, 8):
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


def test_resident_sort_merge_native_compile(compiled_sort_merge, tmp_path):
    compiler = shutil.which("ccec")
    assert compiler, "requires CCEC for offline dav-m200 compilation"
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
            str(compiled_sort_merge[2]),
            "-o",
            str(tmp_path / "native.o"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


_EXTRACT_CASES = [
    (128, 256, passes.MemoryPlanner.PYPTO),
    (128, 80, passes.MemoryPlanner.PTOAS),
    (512, 608, passes.MemoryPlanner.PYPTO),
    (512, 608, passes.MemoryPlanner.DSA_RP),
    (512, 608, passes.MemoryPlanner.PTOAS),
]


@pytest.fixture(scope="module", params=_EXTRACT_CASES)
def compiled_sort_extract(request, tmp_path_factory):
    columns, output_columns, planner = request.param
    directory = tmp_path_factory.mktemp("resident-sort-extract")
    ir.compile(
        sort_merge_kernel(columns, output_columns, extract_fields=True).specialize(),
        backend_type=BackendType.Ascend310P,
        memory_planner=planner,
        output_dir=str(directory),
    )
    sources = list((directory / "kernels").glob("*/*.cpp"))
    assert len(sources) == 1
    code = sources[0].read_text()
    assert code.count("TSORT32(") == 1
    assert code.count("TMRGSORT<") == (1 if columns == 128 else 5)
    assert code.count("TGATHER<") == 2
    assert code.count("TLOAD(") == 2 and code.count("TSTORE(") == 2
    assert "TSETVALIDSHAPE(" not in code
    metadata = json.loads((directory / "compiled_meta.json").read_text())
    assert [p["dtype"] for p in metadata["params"]] == ["fp32", "uint32", "fp32", "uint32"]
    return columns, output_columns, sources[0]


def test_resident_sort_extract_cpu(compiled_sort_extract, tmp_path):
    columns, output_columns, generated = compiled_sort_extract
    compiler = shutil.which("clang++-19")
    assert compiler, "requires clang++-19 for exact composition oracle"
    source = tmp_path / "oracle.cpp"
    source.write_text(
        "#include <cstdint>\n"
        + generated.read_text()
        + "\n"
        + Path(__file__).with_name("sort_extract_oracle.cpp").read_text()
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
            f"-DSORT_COLS={columns}",
            f"-DEXTRACT_PAIRS={output_columns // 2}",
            *_includes(),
            str(source),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    for count in (1, 3, 8):
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


def test_resident_sort_extract_native(compiled_sort_extract, tmp_path):
    compiler = shutil.which("ccec")
    assert compiler, "requires CCEC for dav-m200 offline composition validation"
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
            str(compiled_sort_extract[2]),
            "-o",
            str(tmp_path / "native.o"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
