# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Existing public multi-list merge on A1, generated ABI and resident composition."""

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


def merge_pair_kernel() -> Any:
    @pl.jit.incore
    def merge_pair(
        source0: pl.Tensor[[1, 64], pl.FP32],
        source1: pl.Tensor[[1, 96], pl.FP32],
        output: pl.Out[pl.Tensor[[1, 160], pl.FP32]],
    ):
        first = pl.load(source0, [0, 0], [1, 64])
        second = pl.load(source1, [0, 0], [1, 96])
        scratch = pl.create_tile([1, 1280], pl.FP32)
        result = pl.tile.mrgsort(first, second, tmp=scratch, output_cols=160)
        pl.store(result, [0, 0], output)
        return output

    return merge_pair


def test_public_merge_default_compile(tmp_path: Path) -> None:
    ir.compile(
        merge_pair_kernel().specialize(), backend_type=BackendType.Ascend310P, output_dir=str(tmp_path)
    )
    kernels = list((tmp_path / "kernels").glob("*/*.cpp"))
    assert len(kernels) == 1
    assert "TMRGSORT<" in kernels[0].read_text()


def merge_kernel(
    widths: tuple[int, ...],
    output_columns: int,
    *,
    mode: str = "tile",
    scratch_columns: int | None = None,
    dtype: Any = pl.FP32,
) -> Any:
    ways = len(widths)
    w0, w1, w2, w3 = widths + (32,) * (4 - ways)
    panels = 8 if mode == "spmd" else 5
    input_stride = max(widths) + 16
    output_stride = output_columns + 16
    input_count = panels * 4 * input_stride
    output_count = panels * output_stride
    scratch_count = sum(widths) * 8 if scratch_columns is None else scratch_columns

    # Fixed-arity wrappers only select the existing public merge overload.
    # All kernel execution, memory planning and synchronization stay shared.
    if ways == 2:

        @pl.jit.inline
        def merge_values(source: pl.Tensor[[input_count], dtype], block: pl.Scalar[pl.INDEX]):
            view = pl.tensor.view(source, [panels * 4, input_stride])
            a = pl.load(view, [block * 4, 8], [1, w0])
            b = pl.load(view, [block * 4 + 1, 8], [1, w1])
            scratch = pl.create_tile([1, scratch_count], dtype)
            return pl.tile.mrgsort(a, b, tmp=scratch, output_cols=output_columns)
    elif ways == 3:

        @pl.jit.inline
        def merge_values(source: pl.Tensor[[input_count], dtype], block: pl.Scalar[pl.INDEX]):
            view = pl.tensor.view(source, [panels * 4, input_stride])
            a = pl.load(view, [block * 4, 8], [1, w0])
            b = pl.load(view, [block * 4 + 1, 8], [1, w1])
            c = pl.load(view, [block * 4 + 2, 8], [1, w2])
            scratch = pl.create_tile([1, scratch_count], dtype)
            return pl.mrgsort(a, b, c, tmp=scratch, output_cols=output_columns)
    else:

        @pl.jit.inline
        def merge_values(source: pl.Tensor[[input_count], dtype], block: pl.Scalar[pl.INDEX]):
            view = pl.tensor.view(source, [panels * 4, input_stride])
            a = pl.load(view, [block * 4, 8], [1, w0])
            b = pl.load(view, [block * 4 + 1, 8], [1, w1])
            c = pl.load(view, [block * 4 + 2, 8], [1, w2])
            d = pl.load(view, [block * 4 + 3, 8], [1, w3])
            scratch = pl.create_tile([1, scratch_count], dtype)
            return pl.tile.mrgsort(a, b, c, d, tmp=scratch, output_cols=output_columns)

    @pl.jit.inline
    def merge_panel(
        source: pl.Tensor[[input_count], dtype],
        output: pl.Tensor[[output_count], dtype],
        block: pl.Scalar[pl.INDEX],
    ):
        target = pl.tensor.view(output, [panels, output_stride])
        result = merge_values(source, block)
        pl.store(result, [block, 8], target)
        return output

    if mode == "spmd":

        @pl.jit
        def merge_spmd(
            source: pl.Tensor[[input_count], dtype], output: pl.Out[pl.Tensor[[output_count], dtype]]
        ):
            for block in pl.spmd(8):
                output = merge_panel(source, output, block)
            return output

        return merge_spmd
    if mode == "pipeline":

        @pl.jit.incore
        def merge_pipeline(
            source: pl.Tensor[[input_count], dtype], output: pl.Out[pl.Tensor[[output_count], dtype]]
        ):
            for block in pl.pipeline(panels, stage=2):
                output = merge_panel(source, output, block)
            return output

        return merge_pipeline

    @pl.jit.incore
    def merge_tile(source: pl.Tensor[[input_count], dtype], output: pl.Out[pl.Tensor[[output_count], dtype]]):
        for block in pl.range(pl.tile.get_block_idx(), panels, pl.tile.get_block_num()):
            output = merge_panel(source, output, block)
        return output

    return merge_tile


_CASES = [
    ((64, 96), 8, "tile", passes.MemoryPlanner.PYPTO),
    ((64, 96), 80, "tile", passes.MemoryPlanner.PYPTO),
    ((64, 96), 160, "tile", passes.MemoryPlanner.DSA_RP),
    ((32, 64, 96), 96, "tile", passes.MemoryPlanner.PYPTO),
    ((32, 64, 96), 192, "tile", passes.MemoryPlanner.DSA_RP),
    ((32, 64, 96, 128), 160, "tile", passes.MemoryPlanner.PTOAS),
    ((32, 64, 96, 128), 320, "tile", passes.MemoryPlanner.PTOAS),
    ((256, 256, 256, 256), 600, "tile", passes.MemoryPlanner.PYPTO),
    ((64, 96), 160, "pipeline", passes.MemoryPlanner.PYPTO),
    ((64, 64, 64, 64), 128, "spmd", passes.MemoryPlanner.PYPTO),
]


@pytest.fixture(scope="module", params=_CASES)
def compiled_merge(request, tmp_path_factory):
    widths, columns, mode, planner = request.param
    directory = tmp_path_factory.mktemp("public-merge")
    ir.compile(
        merge_kernel(widths, columns, mode=mode).specialize(),
        backend_type=BackendType.Ascend310P,
        memory_planner=planner,
        output_dir=str(directory),
    )
    sources = list((directory / "kernels").glob("*/*.cpp"))
    assert len(sources) == 1
    code = sources[0].read_text()
    # Stage-two lowering emits a double-step body and its odd tail in one kernel.
    copies = 3 if mode == "pipeline" else 1
    assert code.count("TMRGSORT<") == copies
    assert code.count("TLOAD(") == copies * len(widths)
    assert code.count("TSTORE(") == copies
    assert "wait_flag(PIPE_MTE2, PIPE_V," in code
    assert "set_flag(PIPE_MTE2, PIPE_V," in code
    assert "wait_flag(PIPE_S, PIPE_MTE3," in code
    assert "TSETVALIDSHAPE(" not in code
    metadata = json.loads((directory / "compiled_meta.json").read_text())
    assert [p["dtype"] for p in metadata["params"]] == ["fp32", "fp32"]
    order = [0, 1]
    panels = 8 if mode == "spmd" else 5
    if mode == "spmd":
        manifest = json.loads((directory / "a1_spmd.json").read_text())
        assert manifest["blocks"] == 8 and manifest["task_count"] == 1
        order = manifest["argument_order"]
        assert sorted(order) == [0, 1]
    return widths, columns, mode, panels, sources[0], order


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


def test_public_merge_cpu_tensor_abi(compiled_merge, tmp_path):
    widths, columns, mode, panels, generated, order = compiled_merge
    compiler = shutil.which("clang++-19")
    assert compiler, "requires clang++-19 for CPU instruction semantics"
    source = tmp_path / "oracle.cpp"
    source.write_text(
        "#include <cstdint>\n"
        + generated.read_text()
        + "\n"
        + Path(__file__).with_name("mrgsort_oracle.cpp").read_text()
    )
    binary = tmp_path / "oracle"
    padded_widths = widths + (32,) * (4 - len(widths))
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O1",
            "-D__CPU_SIM",
            "-D__DAV_CUBE__",
            "-D__DAV_VEC__",
            f"-DMERGE_WAYS={len(widths)}",
            f"-DMERGE_COLS={columns}",
            f"-DMERGE_PANELS={panels}",
            *[f"-DMERGE_WIDTH{i}={w}" for i, w in enumerate(padded_widths)],
            *[f"-DMERGE_ARG{i}={value}" for i, value in enumerate(order)],
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


def test_public_merge_native_compile(compiled_merge, tmp_path):
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
            str(compiled_merge[4]),
            "-o",
            str(tmp_path / "native.o"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    "widths,columns,scratch,dtype,diagnostic",
    [
        ((24, 32), 56, 448, pl.FP32, "16-pair"),
        ((64, 96), 80, 1272, pl.FP32, "eight-field"),
        ((64, 96), 80, 1280, pl.FP16, "FP32"),
        ((64, 96), 10, 1280, pl.FP32, "32-byte-aligned"),
    ],
)
def test_public_merge_rejects_invalid_profile(tmp_path, widths, columns, scratch, dtype, diagnostic):
    with pytest.raises((RuntimeError, ValueError), match=diagnostic):
        ir.compile(
            merge_kernel(widths, columns, scratch_columns=scratch, dtype=dtype).specialize(),
            backend_type=BackendType.Ascend310P,
            output_dir=str(tmp_path),
        )
