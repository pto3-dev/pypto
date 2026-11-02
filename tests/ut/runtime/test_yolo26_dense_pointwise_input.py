# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Dense phase images: independent logical-channel reference and Cube consumers."""

import json
import os
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any

import pypto.language as pl
import pytest
from pypto import ir, passes
from pypto.backend import BackendType
from pypto.runtime.pto_isa import pto_isa_include_dir

from examples.models.yolo26_native.dense_pointwise_input import dense_pointwise_image_cache
from examples.models.yolo26_native.operators import ConvPlan


def _plan(channels: int) -> ConvPlan:
    return ConvPlan(16, 16, channels, 16, 1, 1, 0, False, True, channels + 32, 16)


def _kernel(plan: ConvPlan, rows: int) -> Any:
    stage = dense_pointwise_image_cache(plan, rows)
    pixels, stride = plan.height * plan.width, plan.input_row_stride
    columns = (plan.cin + 63) // 64 * 16
    output_rows = 4 * rows + 32

    @pl.jit.inline
    def consume(
        image: pl.Tile[[rows, columns], pl.FP16, pl.Mem.Mat],
        weights: pl.Tile[[columns, columns], pl.FP16, pl.Mem.Mat],
        output: pl.Tensor[[output_rows, columns], pl.FP32],
        phase: pl.Scalar[pl.INDEX],
    ):
        left = pl.tile.move(
            image, pl.Mem.Left, blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.row_major
        )
        # Each resident phase image is reused across its N16 identity panels.
        # This checks the complete NZ layout through a real Cube consumer.
        for panel in pl.range(columns // 16):
            right = pl.tile.slice(weights, [columns, 16], [0, panel * 16])
            values = pl.tile.matmul(left, right)
            output = pl.store(values, [16 + phase * rows, panel * 16], output)
        return output

    @pl.jit.incore
    def phase_images(
        source: pl.Tensor[[pixels, stride], pl.FP16],
        weights: pl.Tensor[[columns, columns], pl.FP16],
        output: pl.Out[pl.Tensor[[output_rows, columns], pl.FP32]],
    ):
        image0, image1, image2, image3 = stage(source, 16)
        identity = pl.load(weights, [0, 0], [columns, columns], target_memory=pl.Mem.Mat)
        output = consume(image0, identity, output, 0)
        output = consume(image1, identity, output, 1)
        output = consume(image2, identity, output, 2)
        output = consume(image3, identity, output, 3)
        return output

    return phase_images


_CASES = [
    (16, 16),
    (32, 32),
    (48, 48),
    (64, 64),
    (112, 80),
    (128, 128),
    (176, 192),
    (224, 256),
    (112, 320),
    (112, 336),
    (224, 512),
]


@pytest.mark.parametrize("rows,channels", _CASES)
@pytest.mark.parametrize(
    "planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.PTOAS, passes.MemoryPlanner.DSA_RP]
)
def test_dense_phase_input_cpu_native(tmp_path, pass_verification_instruments, rows, channels, planner):
    plan = _plan(channels)
    kernel = _kernel(plan, rows)
    assert not kernel._external_source_paths()
    with passes.PassContext(pass_verification_instruments, memory_planner=planner):
        ir.compile(kernel.specialize(), backend_type=BackendType.Ascend310P, output_dir=str(tmp_path))
    sources = list((tmp_path / "kernels").glob("*/*.cpp"))
    assert len(sources) == 1
    code = sources[0].read_text()
    assert "TGATHER<" in code and "TINSERT(" in code and "TMATMUL(" in code
    columns = (channels + 63) // 64 * 16
    params = json.loads((tmp_path / "compiled_meta.json").read_text())["params"]
    assert [param["dtype"] for param in params] == ["fp16", "fp16", "fp32"]
    assert [param["shape"] for param in params] == [
        [256, channels + 32],
        [columns, columns],
        [4 * rows + 32, columns],
    ]
    runtime = Path(__file__).resolve().parents[3] / "runtime"
    includes = [
        f"-I{path}"
        for path in (
            pto_isa_include_dir(),
            runtime / "src/a1/platform/include",
            runtime / "src/common",
            runtime / "src/common/task_interface",
            runtime / "src/common/platform/include",
        )
    ]
    cpu, native = shutil.which("clang++-19"), shutil.which("ccec")
    assert cpu and native
    oracle = tmp_path / "oracle.cpp"
    oracle.write_text(
        "#include <cstdint>\n"
        + code
        + "\n"
        + Path(__file__).with_name("dense_pointwise_input_oracle.cpp").read_text()
    )
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
            f"-DDP_ROWS={rows}",
            f"-DDP_CHANNELS={channels}",
            *includes,
            str(oracle),
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
            "PTO_CPU_SIM_L1_BYTES": "1048576",
            "PTO_CPU_SIM_UB_BYTES": "131072",
            "PTO_CPU_SIM_L0C_BYTES": "262144",
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


@pytest.mark.parametrize("rows", [0, 15, True, 240, 272])
def test_dense_phase_input_rejects_invalid_rows(rows):
    with pytest.raises(ValueError, match="dense pointwise images require"):
        dense_pointwise_image_cache(_plan(64), rows)


@pytest.mark.parametrize(
    "change", [{"four_way_accumulation": False}, {"kernel": 3, "padding": 1}, {"cin": 3}]
)
def test_dense_phase_input_rejects_invalid_plan(change):
    with pytest.raises(ValueError, match="dense pointwise images require"):
        dense_pointwise_image_cache(replace(_plan(64), **change), 16)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
