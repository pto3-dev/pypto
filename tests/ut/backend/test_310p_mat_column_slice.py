# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Resident NZ weight-column views feeding N16/N32 Cube operations."""

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pypto.language as pl
import pytest
from pypto import ir
from pypto.backend import BackendType
from pypto.language.parser.diagnostics import InvalidOperationError
from pypto.runtime.pto_isa import pto_isa_include_dir


def mat_column_kernel(
    m: int,
    k: int,
    parent_n: int,
    n: int,
    column: int,
    rounds: int,
    *,
    explicit_move: bool = False,
) -> Any:
    if explicit_move:

        @pl.jit.inline
        def select(parent: pl.Tile[[k, parent_n], pl.FP16]):
            view = pl.tile.slice(parent, [k, n], [0, column])
            return pl.tile.move(view, pl.Mem.Right)
    else:

        @pl.jit.inline
        def select(parent: pl.Tile[[k, parent_n], pl.FP16]):
            return pl.tile.slice(parent, [k, n], [0, column])

    if n == 16:

        @pl.jit.inline
        def finish(
            acc: pl.Tile[[m, 16], pl.FP32],
            output: pl.Tensor[[rounds * m, n], pl.FP32],
            row: pl.Scalar[pl.INDEX],
        ):
            return pl.store(acc, [row, 0], output)
    else:

        @pl.jit.inline
        def finish(
            acc: pl.Tile[[m, 32], pl.FP32],
            output: pl.Tensor[[rounds * m, n], pl.FP32],
            row: pl.Scalar[pl.INDEX],
        ):
            low = pl.tile.slice(acc, [m, 16], [0, 0])
            high = pl.tile.slice(acc, [m, 16], [0, 16])
            output = pl.store(low, [row, 0], output)
            return pl.store(high, [row, 16], output)

    @pl.jit.inline
    def product(
        index: pl.Scalar[pl.INDEX],
        left: pl.Tile[[m, k], pl.FP16],
        weights: pl.Tensor[[rounds * k, parent_n], pl.FP16],
        output: pl.Tensor[[rounds * m, n], pl.FP32],
    ):
        parent = pl.load(weights, [index * k, 0], [k, parent_n], target_memory=pl.Mem.Mat)
        window = select(parent)
        acc = pl.matmul(left, window)
        return finish(acc, output, index * m)

    @pl.jit.incore
    def mat_columns(
        a: pl.Tensor[[m, k], pl.FP16],
        weights: pl.Tensor[[rounds * k, parent_n], pl.FP16],
        out: pl.Out[pl.Tensor[[rounds * m, n], pl.FP32]],
    ):
        source = pl.load(a, [0, 0], [m, k], target_memory=pl.Mem.Mat)
        left = pl.tile.move(
            source, pl.Mem.Left, blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.row_major
        )
        if pl.tile.get_block_num() == 1:
            for index in pl.pipeline(rounds, stage=2):
                out = product(index, left, weights, out)
        else:
            for index in pl.range(pl.tile.get_block_idx(), rounds, pl.tile.get_block_num()):
                out = product(index, left, weights, out)
        return out

    return mat_columns


_CASES = [
    (16, 16, 48, 16, 32, 3),
    (32, 64, 96, 32, 16, 5),
    (64, 128, 128, 32, 96, 5),
    (32, 256, 64, 32, 32, 2),
]


@pytest.fixture(
    scope="module",
    params=[(x, False) for x in _CASES] + [(x, True) for x in _CASES[:2]],
    ids=lambda pair: f"m{pair[0][0]}k{pair[0][1]}n{pair[0][3]}-offset{pair[0][4]}-explicit{pair[1]}",
)
def compiled_columns(request, tmp_path_factory):
    if not shutil.which("ptoas"):
        pytest.skip("requires PTOAS for complete public compilation")
    output = tmp_path_factory.mktemp("mat-columns")
    geometry, explicit_move = request.param
    result = ir.compile(
        mat_column_kernel(*geometry, explicit_move=explicit_move).specialize(),
        backend_type=BackendType.Ascend310P,
        output_dir=str(output),
    )
    assert isinstance(result, ir.CompiledProgram)
    sources = list((output / "kernels").glob("*/mat_columns.cpp"))
    assert len(sources) == 1
    return geometry, sources[0]


def _includes():
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


def test_mat_columns_cpu_tensor_abi(compiled_columns, tmp_path, *, left=False):
    compiler = shutil.which("clang++-19")
    if not compiler:
        pytest.skip("requires clang++-19 for CPU instruction semantics")
    values, generated = compiled_columns
    source = tmp_path / "oracle.cpp"
    source.write_text(
        "#include <cstdint>\n"
        + generated.read_text()
        + "\n"
        + Path(__file__).with_name("mat_column_slice_oracle.cpp").read_text()
    )
    keys = ("M", "K", "PN", "N", "COL", "ROUNDS")
    binary = tmp_path / "oracle"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O1",
            "-D__CPU_SIM",
            "-D__DAV_CUBE__",
            "-D__DAV_VEC__",
            f"-DMC_LEFT={int(left)}",
            *[f"-DMC_{key}={value}" for key, value in zip(keys, values, strict=True)],
            *_includes(),
            str(source),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    for cores in (1, 3, 8, 32):
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


def test_mat_columns_offline_native_compile(compiled_columns, tmp_path):
    compiler = shutil.which("ccec")
    if not compiler:
        pytest.skip("requires CCEC for offline dav-m200 compilation")
    _, generated = compiled_columns
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
            str(generated),
            "-o",
            str(tmp_path / "native.o"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def mat_left_column_kernel() -> Any:
    @pl.jit.incore
    def mat_columns(
        a: pl.Tensor[[32, 96], pl.FP16],
        weights: pl.Tensor[[64, 16], pl.FP16],
        out: pl.Out[pl.Tensor[[32, 16], pl.FP32]],
    ):
        parent = pl.load(a, [0, 0], [32, 96], target_memory=pl.Mem.Mat)
        view = pl.tile.slice(parent, [32, 64], [0, 16])
        left = pl.tile.move(
            view, pl.Mem.Left, blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.row_major
        )
        right = pl.load(weights, [0, 0], [64, 16], target_memory=pl.Mem.Mat)
        acc = pl.matmul(left, right)
        return pl.store(acc, [0, 0], out)

    return mat_columns


def test_mat_columns_left_cpu_and_native(tmp_path):
    if not all(shutil.which(name) for name in ("ptoas", "clang++-19", "ccec")):
        pytest.skip("requires PTOAS, clang++-19 and CCEC")
    output = tmp_path / "left"
    result = ir.compile(
        mat_left_column_kernel().specialize(), backend_type=BackendType.Ascend310P, output_dir=str(output)
    )
    assert isinstance(result, ir.CompiledProgram)
    generated = list((output / "kernels").glob("*/mat_columns.cpp"))
    assert len(generated) == 1
    # Reuse the independent harness with an explicit operand-side selector.
    test_mat_columns_cpu_tensor_abi(((32, 64, 96, 16, 16, 1), generated[0]), tmp_path, left=True)
    test_mat_columns_offline_native_compile((None, generated[0]), tmp_path)


def unsupported_slice_kernel(row: int, column: int, height: int) -> Any:
    @pl.jit.incore
    def unsupported(
        a: pl.Tensor[[16, height], pl.FP16],
        weights: pl.Tensor[[64, 96], pl.FP16],
        out: pl.Out[pl.Tensor[[16, 32], pl.FP32]],
    ):
        parent = pl.load(weights, [0, 0], [64, 96], target_memory=pl.Mem.Mat)
        view = pl.tile.slice(parent, [height, 32], [row, column])
        right = pl.tile.move(view, pl.Mem.Right)
        left = pl.load(a, [0, 0], [16, height], target_memory=pl.Mem.Mat)
        acc = pl.matmul(left, right)
        low = pl.tile.slice(acc, [16, 16], [0, 0])
        high = pl.tile.slice(acc, [16, 16], [0, 16])
        out = pl.store(low, [0, 0], out)
        return pl.store(high, [0, 16], out)

    return unsupported


@pytest.mark.parametrize(
    "row,column,height,error,diagnostic",
    [
        (0, 1, 64, RuntimeError, "Ascend310P Mat slices"),
        (16, 16, 32, RuntimeError, "Ascend310P Mat slices"),
        (0, 80, 64, InvalidOperationError, "tile.slice reads past"),
    ],
)
def test_mat_columns_reject_unsafe_views(tmp_path, row, column, height, error, diagnostic):
    # Frontend exact-window bounds may reject before backend layout validation.
    # Both stages must keep rejecting these unsafe requests.
    with pytest.raises(error, match=diagnostic):
        ir.compile(
            unsupported_slice_kernel(row, column, height).specialize(),
            backend_type=BackendType.Ascend310P,
            skip_ptoas=True,
            output_dir=str(tmp_path),
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
