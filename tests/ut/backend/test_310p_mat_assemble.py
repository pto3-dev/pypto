# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Public UB-to-L1 assembly, partial preservation and pipeline buffer reuse."""

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pypto.language as pl
import pytest
from pypto import ir, passes
from pypto.backend import BackendType
from pypto.runtime.pto_isa import pto_isa_include_dir


@dataclass(frozen=True)
class Assembly:
    rows: int
    columns: int
    panel: int
    pipeline: bool = True
    patch_rows: int = 0
    row: int = 0
    column: int = 0


_CASES = [
    Assembly(16, 16, 16, False),
    Assembly(32, 80, 16),
    Assembly(32, 96, 32),
    Assembly(224, 48, 16),
    Assembly(1024, 32, 16),
    Assembly(16, 128, 128, False),
    Assembly(48, 64, 16, False, 16, 16, 16),
    Assembly(32, 64, 32, False, 16, 16, 32),
]


def _kernel(case: Assembly) -> Any:
    m, k, panel = case.rows, case.columns, case.panel
    patch_rows, row, column = case.patch_rows, case.row, case.column

    @pl.jit.inline
    def finish(
        matrix: pl.Tile[[m, k], pl.FP16, pl.Mem.Mat],
        weights: pl.Tensor[[k, 16], pl.FP16],
        output: pl.Tensor[[m, 16], pl.FP32],
    ):
        left = pl.tile.move(
            matrix, pl.Mem.Left, blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.row_major
        )
        right = pl.load(weights, [0, 0], [k, 16], target_memory=pl.Mem.Mat)
        acc = pl.matmul(left, right)
        values = pl.tile.move(
            acc, pl.Mem.Vec, blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.none_box
        )
        return pl.store(values, [0, 0], output)

    @pl.jit.inline
    def write_panel(
        matrix: pl.Tile[[m, k], pl.FP16, pl.Mem.Mat],
        source: pl.Tensor[[m, k], pl.FP16],
        index: pl.Scalar[pl.INDEX],
    ):
        values = pl.load(source, [0, index * panel], [m, panel])
        return pl.tile.assemble(matrix, values, [0, index * panel])

    if patch_rows:

        @pl.jit.incore
        def assemble_mat(
            seed: pl.Tensor[[m, k], pl.FP16],
            patch: pl.Tensor[[patch_rows, panel], pl.FP16],
            weights: pl.Tensor[[k, 16], pl.FP16],
            out: pl.Out[pl.Tensor[[m, 16], pl.FP32]],
        ):
            matrix = pl.load(seed, [0, 0], [m, k], target_memory=pl.Mem.Mat)
            values = pl.load(patch, [0, 0], [patch_rows, panel])
            matrix = pl.tile.assemble(matrix, values, [row, column])
            return finish(matrix, weights, out)

    elif case.pipeline:

        @pl.jit.incore
        def assemble_mat(
            source: pl.Tensor[[m, k], pl.FP16],
            weights: pl.Tensor[[k, 16], pl.FP16],
            out: pl.Out[pl.Tensor[[m, 16], pl.FP32]],
        ):
            matrix = pl.create_tile([m, k], pl.FP16, target_memory=pl.Mem.Mat)
            for index in pl.pipeline(k // panel, stage=2):
                matrix = write_panel(matrix, source, index)
            return finish(matrix, weights, out)

    else:

        @pl.jit.incore
        def assemble_mat(
            source: pl.Tensor[[m, k], pl.FP16],
            weights: pl.Tensor[[k, 16], pl.FP16],
            out: pl.Out[pl.Tensor[[m, 16], pl.FP32]],
        ):
            matrix = pl.create_tile([m, k], pl.FP16, target_memory=pl.Mem.Mat)
            for index in pl.range(k // panel):
                matrix = write_panel(matrix, source, index)
            return finish(matrix, weights, out)

    return assemble_mat


@pytest.mark.parametrize(
    "planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.PTOAS, passes.MemoryPlanner.DSA_RP]
)
@pytest.mark.parametrize("case", _CASES)
def test_assemble_mat_tensor_abi(tmp_path, pass_verification_instruments, case, planner):
    with passes.PassContext(pass_verification_instruments, memory_planner=planner):
        ir.compile(
            _kernel(case).specialize(),
            backend_type=BackendType.Ascend310P,
            output_dir=str(tmp_path),
        )
    sources = list((tmp_path / "kernels").glob("*/*.cpp"))
    assert len(sources) == 1
    code = sources[0].read_text()
    assert "TINSERT(" in code and "TMATMUL(" in code
    for source, destination in (("MTE2", "MTE3"), ("MTE3", "MTE1")):
        assert f"set_flag(PIPE_{source}, PIPE_{destination}," in code
        assert f"wait_flag(PIPE_{source}, PIPE_{destination}," in code
    if case.pipeline:
        # Both source banks must be loaded before the first insertion. Counting
        # distinct Vec destinations excludes the independent weight Mat load.
        vector_names = set(re.findall(r"Tile<TileType::Vec, half,[^;]+> (\w+);", code))
        loads = re.finditer(r"TLOAD\((\w+),", code[: code.index("TINSERT(")])
        assert len({load.group(1) for load in loads if load.group(1) in vector_names}) >= 2
    params = json.loads((tmp_path / "compiled_meta.json").read_text())["params"]
    shapes = [[case.rows, case.columns]]
    if case.patch_rows:
        shapes.append([case.patch_rows, case.panel])
    shapes += [[case.columns, 16], [case.rows, 16]]
    assert [p["shape"] for p in params] == shapes
    assert [p["dtype"] for p in params] == ["fp16"] * (len(shapes) - 1) + ["fp32"]
    runtime = Path(__file__).resolve().parents[3] / "runtime"
    includes = [
        f"-I{p}"
        for p in (
            pto_isa_include_dir(),
            runtime / "src/a1/platform/include",
            runtime / "src/common",
            runtime / "src/common/task_interface",
            runtime / "src/common/platform/include",
        )
    ]
    definitions = {
        "M": case.rows,
        "K": case.columns,
        "PARTIAL": int(bool(case.patch_rows)),
        "PR": case.patch_rows,
        "PC": case.panel,
        "R0": case.row,
        "C0": case.column,
    }
    source = tmp_path / "oracle.cpp"
    source.write_text(
        "#include <cstdint>\n" + code + "\n" + Path(__file__).with_name("mat_assemble_oracle.cpp").read_text()
    )
    cpu, native = shutil.which("clang++-19"), shutil.which("ccec")
    assert cpu and native, "requires driver-free CPU and native compilers"
    binary = tmp_path / "oracle"
    subprocess.run(
        [
            cpu,
            "-std=c++20",
            "-O1",
            "-ffp-contract=off",
            "-D__CPU_SIM",
            "-D__DAV_CUBE__",
            "-D__DAV_VEC__",
            *[f"-DAM_{name}={value}" for name, value in definitions.items()],
            *includes,
            str(source),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [str(binary)],
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PTO_CPU_SIM_NUM_CORES": "1",
            "PTO_CPU_SIM_TRACE_ENABLE": "0",
            "OMP_NUM_THREADS": "1",
        },
    )
    subprocess.run(
        [
            native,
            "-c",
            "-O2",
            "-x",
            "cce",
            "-std=c++17",
            "--cce-aicore-only",
            "--cce-aicore-arch=dav-m200",
            *includes,
            str(sources[0]),
            "-o",
            str(tmp_path / "native.o"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("row,column,patch_rows", [(1, 0, 16), (0, 1, 16), (0, 0, 1)])
def test_assemble_mat_rejects_unaligned_panels(tmp_path, row, column, patch_rows):
    with pytest.raises(RuntimeError, match="Vec-to-Mat"):
        ir.compile(
            _kernel(Assembly(32, 64, 16, False, patch_rows, row, column)).specialize(),
            backend_type=BackendType.Ascend310P,
            output_dir=str(tmp_path),
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
