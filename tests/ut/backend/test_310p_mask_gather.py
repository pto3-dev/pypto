# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Public A1 fixed-mask field extraction, exact generated code and offline only."""

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


def mask_pair_kernel() -> Any:
    @pl.jit.incore
    def mask_pair(
        source: pl.Tensor[[1, 608], pl.FP32],
        scores: pl.Out[pl.Tensor[[1, 304], pl.FP32]],
        indices: pl.Out[pl.Tensor[[1, 304], pl.UINT32]],
    ):
        pairs = pl.load(source, [0, 0], [1, 608])
        values = pl.tile.gather_mask(pairs, pl.tile.MaskPattern.P0101)
        positions = pl.tile.gather_mask(pairs, pl.tile.MaskPattern.P1010, output_dtype=pl.UINT32)
        pl.store(values, [0, 0], scores)
        pl.store(positions, [0, 0], indices)
        return scores, indices

    return mask_pair


def test_public_mask_default_compile(tmp_path: Path) -> None:
    ir.compile(mask_pair_kernel().specialize(), backend_type=BackendType.Ascend310P, output_dir=str(tmp_path))
    sources = list((tmp_path / "kernels").glob("*/*.cpp"))
    assert len(sources) == 1
    code = sources[0].read_text()
    assert code.count("TGATHER<") == 2
    assert code.count("TLOAD(") == 1 and code.count("TSTORE(") == 2


def mask_kernel(dtype_name: str, rows: int, columns: int, pattern_a: int, pattern_b: int, mode: str = "tile"):
    dtype = getattr(pl, dtype_name)
    index_dtype = pl.FP16 if dtype_name == "FP16" else pl.UINT32
    factor = 2 if pattern_a <= 2 else 4 if pattern_a <= 6 else 1
    output_columns = columns // factor
    panels = 8 if mode == "spmd" else 5
    total_rows = panels * (rows + 2)
    input_stride, output_stride = columns + 32, output_columns + 32
    input_count, output_count = total_rows * input_stride, total_rows * output_stride

    @pl.jit.inline
    def panel(
        source: pl.Tensor[[input_count], dtype],
        values: pl.Tensor[[output_count], dtype],
        indices: pl.Tensor[[output_count], index_dtype],
        block: pl.Scalar[pl.INDEX],
    ):
        source_view = pl.tensor.view(source, [total_rows, input_stride])
        value_view = pl.tensor.view(values, [total_rows, output_stride])
        index_view = pl.tensor.view(indices, [total_rows, output_stride])
        tile = pl.load(source_view, [block * (rows + 2) + 1, 16], [rows, columns])
        selected = pl.tile.gather_mask(tile, pattern_a)
        positions = pl.tile.gather_mask(tile, pattern_b, output_dtype=index_dtype)
        pl.store(selected, [block * (rows + 2) + 1, 16], value_view)
        pl.store(positions, [block * (rows + 2) + 1, 16], index_view)
        return values, indices

    if mode == "spmd":

        @pl.jit
        def mask_spmd(
            source: pl.Tensor[[input_count], dtype],
            values: pl.Out[pl.Tensor[[output_count], dtype]],
            indices: pl.Out[pl.Tensor[[output_count], index_dtype]],
        ):
            for block in pl.spmd(8):
                values, indices = panel(source, values, indices, block)
            return values, indices

        return mask_spmd
    if mode == "pipeline":

        @pl.jit.incore
        def mask_pipeline(
            source: pl.Tensor[[input_count], dtype],
            values: pl.Out[pl.Tensor[[output_count], dtype]],
            indices: pl.Out[pl.Tensor[[output_count], index_dtype]],
        ):
            for block in pl.pipeline(panels, stage=2):
                values, indices = panel(source, values, indices, block)
            return values, indices

        return mask_pipeline

    @pl.jit.incore
    def mask_tile(
        source: pl.Tensor[[input_count], dtype],
        values: pl.Out[pl.Tensor[[output_count], dtype]],
        indices: pl.Out[pl.Tensor[[output_count], index_dtype]],
    ):
        for block in pl.range(pl.tile.get_block_idx(), panels, pl.tile.get_block_num()):
            values, indices = panel(source, values, indices, block)
        return values, indices

    return mask_tile


mask_cases = [
    (dtype, 3, 128, a, b, "tile", passes.MemoryPlanner.PYPTO)
    for dtype in ("FP16", "FP32", "INT32", "UINT32")
    for a, b in ((1, 2), (3, 6), (4, 5), (7, 7))
]
mask_cases = mask_cases + [
    ("FP32", 1, 608, 1, 2, "tile", passes.MemoryPlanner.PTOAS),
    ("FP32", 1, 608, 1, 2, "tile", passes.MemoryPlanner.DSA_RP),
    ("FP16", 3, 160, 1, 2, "tile", passes.MemoryPlanner.PTOAS),
    ("FP32", 3, 128, 1, 2, "pipeline", passes.MemoryPlanner.PYPTO),
    ("FP32", 3, 128, 1, 2, "spmd", passes.MemoryPlanner.PYPTO),
    ("FP32", 1, 16416, 3, 6, "tile", passes.MemoryPlanner.PTOAS),
    ("FP16", 1, 32832, 4, 5, "tile", passes.MemoryPlanner.PYPTO),
]


@pytest.fixture(scope="module", params=mask_cases)
def compiled_mask(request, tmp_path_factory):
    dtype, rows, columns, a, b, mode, planner = request.param
    directory = tmp_path_factory.mktemp("public-mask")
    ir.compile(
        mask_kernel(dtype, rows, columns, a, b, mode).specialize(),
        backend_type=BackendType.Ascend310P,
        memory_planner=planner,
        output_dir=str(directory),
    )
    sources = list((directory / "kernels").glob("*/*.cpp"))
    assert len(sources) == 1
    code = sources[0].read_text()
    # The two-step pipeline has an unrolled pair plus its odd tail, all in one kernel.
    copies = 3 if mode == "pipeline" else 1
    assert code.count("TLOAD(") == copies
    assert code.count("TGATHER<") == copies * 2
    assert code.count("TSTORE(") == copies * 2
    assert "wait_flag(PIPE_MTE2, PIPE_V," in code
    assert "wait_flag(PIPE_V, PIPE_MTE3," in code
    assert "PIPE_MTE2, PIPE_S," not in code
    assert "TSETVALIDSHAPE(" not in code
    metadata = json.loads((directory / "compiled_meta.json").read_text())
    index_dtype = "fp16" if dtype == "FP16" else "uint32"
    assert [p["dtype"] for p in metadata["params"]] == [dtype.lower(), dtype.lower(), index_dtype]
    order = [0, 1, 2]
    if mode == "spmd":
        manifest = json.loads((directory / "a1_spmd.json").read_text())
        assert manifest["blocks"] == 8 and manifest["task_count"] == 1
        order = manifest["argument_order"]
        assert sorted(order) == [0, 1, 2]
    return dtype, rows, columns, a, b, mode, sources[0], order


def _includes():
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


def test_public_mask_cpu(compiled_mask, tmp_path):
    dtype, rows, columns, a, b, mode, generated, order = compiled_mask
    compiler = shutil.which("clang++-19")
    assert compiler, "requires clang++-19 for raw-bit CPU oracle"
    source = tmp_path / "oracle.cpp"
    source.write_text(
        "#include <cstdint>\n"
        + generated.read_text()
        + "\n"
        + Path(__file__).with_name("mask_gather_oracle.cpp").read_text()
    )
    binary = tmp_path / "oracle"
    ctype = {"FP16": "half", "FP32": "float", "INT32": "int32_t", "UINT32": "uint32_t"}[dtype]
    index_type = "half" if dtype == "FP16" else "uint32_t"
    abi_dtype = {"FP16": "FLOAT16", "FP32": "FLOAT32", "INT32": "INT32", "UINT32": "UINT32"}[dtype]
    definitions = [
        f"-DMASK_TYPE={ctype}",
        f"-DMASK_DATA_TYPE={abi_dtype}",
        f"-DMASK_INDEX_TYPE={index_type}",
        f"-DMASK_ROWS={rows}",
        f"-DMASK_COLS={columns}",
        f"-DMASK_A={a}",
        f"-DMASK_B={b}",
        f"-DMASK_PANELS={8 if mode == 'spmd' else 5}",
        *(f"-DMASK_ARG_{i}={position}" for i, position in enumerate(order)),
    ]
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


def test_public_mask_native_offline(compiled_mask, tmp_path):
    compiler = shutil.which("ccec")
    assert compiler, "requires CCEC for dav-m200 offline validation"
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
            str(compiled_mask[6]),
            "-o",
            str(tmp_path / "native.o"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_public_mask_rejects_unaligned_extracted_row(tmp_path):
    # 300 scalar fields need padding to 304 for the A1 32-byte row contract.
    @pl.jit.incore
    def unaligned(source: pl.Tensor[[1, 600], pl.FP32], output: pl.Out[pl.Tensor[[1, 300], pl.FP32]]):
        data = pl.load(source, [0, 0], [1, 600])
        selected = pl.tile.gather_mask(data, pl.tile.MaskPattern.P0101)
        pl.store(selected, [0, 0], output)
        return output

    with pytest.raises((RuntimeError, ValueError), match="gather_mask requires static"):
        ir.compile(unaligned.specialize(), backend_type=BackendType.Ascend310P, output_dir=str(tmp_path))
