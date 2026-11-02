# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Full public pooling compilation, unchanged ABI and exact CPU semantics."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from pypto import ir, passes
from pypto.backend import BackendType
from pypto.library import fp16_max_pool2d_window
from pypto.runtime import a1_runner
from pypto.runtime.pto_isa import ensure_pto_isa_root, pto_isa_include_dir

from examples.models.yolo26_native.layout import VectorOp
from examples.models.yolo26_native.public_pool import PublicPoolPlan

_PLANS = [
    PublicPoolPlan(VectorOp.MAXPOOL, 1, 1, 16, 16, 16),
    PublicPoolPlan(VectorOp.MAXPOOL, 4, 4, 128, 128, 128),
    PublicPoolPlan(VectorOp.MAXPOOL, 20, 20, 128, 128, 128),
    PublicPoolPlan(VectorOp.MAXPOOL, 3, 7, 80, 128, 144, 16, 32),
    PublicPoolPlan(VectorOp.MAXPOOL, 1, 17, 16, 32, 48, 16, 16),
    PublicPoolPlan(VectorOp.MAXPOOL, 7, 1, 32, 48, 64, 16, 32),
    PublicPoolPlan(VectorOp.MAXPOOL, 3, 5, 1024, 1024, 1024),
    *[PublicPoolPlan(VectorOp.MAXPOOL, 3, 7, 32, 48, 64, 16, 32, kernel_size=k) for k in (1, 7, 31)],
]


def _run(command, **kwargs):
    result = subprocess.run(command, check=False, capture_output=True, text=True, **kwargs)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def _include_flags():
    runtime = Path(__file__).resolve().parents[3] / "runtime"
    paths = (
        pto_isa_include_dir(),
        runtime / "src/a1/platform/include",
        runtime / "src/common",
        runtime / "src/common/task_interface",
        runtime / "src/common/platform/include",
    )
    return [f"-I{path}" for path in paths]


@pytest.mark.parametrize(
    "plan", _PLANS, ids=lambda p: f"h{p.height}-w{p.width}-c{p.channels}-k{p.kernel_size}"
)
@pytest.mark.parametrize(
    "planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.PTOAS, passes.MemoryPlanner.DSA_RP]
)
def test_public_pool_cpu_native(tmp_path, pass_verification_instruments, plan, planner):
    assert all(shutil.which(name) for name in ("ptoas", "clang++-19", "ccec"))
    jit = plan.jit_kernel()
    assert not jit._external_source_paths()
    with passes.PassContext(pass_verification_instruments, memory_planner=planner):
        ir.compile(jit.specialize(), backend_type=BackendType.Ascend310P, output_dir=str(tmp_path))
    generated = list((tmp_path / "kernels").glob("*/max_pool.cpp"))
    assert len(generated) == 1
    code = generated[0].read_text()
    assert "TMAX" in code and "vector.cpp" not in code
    metadata = json.loads((tmp_path / "compiled_meta.json").read_text())
    assert [p["shape"] for p in metadata["params"]] == [
        [32],
        [plan.input_size],
        [plan.second_size],
        [plan.output_size],
    ]
    assert [p["dtype"] for p in metadata["params"]] == ["fp32", "fp16", "fp16", "fp16"]
    source = tmp_path / "oracle.cpp"
    source.write_text(
        "#include <cstdint>\n" + code + "\n" + Path(__file__).with_name("public_pool_oracle.cpp").read_text()
    )
    defines = {
        "POOL_HEIGHT": plan.height,
        "POOL_WIDTH": plan.width,
        "POOL_CHANNELS": plan.channels,
        "POOL_SOURCE_STRIDE": plan.source_stride,
        "POOL_DESTINATION_STRIDE": plan.destination_stride,
        "POOL_SOURCE_OFFSET": plan.source_offset,
        "POOL_DESTINATION_OFFSET": plan.destination_offset,
        "POOL_KERNEL": plan.kernel_size,
    }
    binary = tmp_path / "oracle"
    _run(
        [
            "clang++-19",
            "-std=c++20",
            "-O1",
            "-ffp-contract=off",
            "-D__CPU_SIM",
            "-D__DAV_CUBE__",
            "-D__DAV_VEC__",
            *[f"-D{name}={value}" for name, value in defines.items()],
            *_include_flags(),
            str(source),
            "-o",
            str(binary),
        ]
    )
    for cores in (1, 3, 8, 32):
        _run(
            [str(binary)],
            env={
                **os.environ,
                "PTO_CPU_SIM_NUM_CORES": str(cores),
                "PTO_CPU_SIM_TRACE_ENABLE": "0",
                "OMP_NUM_THREADS": "1",
            },
        )
    _run(
        [
            "ccec",
            "-c",
            "-O2",
            "-x",
            "cce",
            "-std=c++17",
            "--cce-aicore-only",
            "--cce-aicore-arch=dav-m200",
            *_include_flags(),
            str(generated[0]),
            "-o",
            str(tmp_path / "native.o"),
        ]
    )
    assert (tmp_path / "native.o").read_bytes()[:4] == b"\x7fELF"
    if (plan.height, plan.width, plan.channels) == (20, 20, 128):
        # Production batch eight uses distinct Tensor signatures with logical
        # grid one per image. Compile the actual runtime wrapper offline too.
        image = a1_runner._compile_image(
            tmp_path,
            generated[0],
            ensure_pto_isa_root(),
            independent_instances=8,
            argument_count=4,
        )
        assert image.startswith(b"\x7fELF")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"height": True},
        {"width": 0},
        {"channels": 17},
        {"channels": 1040},
        {"kernel_size": 0},
        {"kernel_size": 2},
        {"kernel_size": 33},
        {"source_stride": 16, "source_offset": 16},
        {"source_offset": -16},
        {"source_stride": 1048592},
        {"source_stride": 1048576, "height": 2048},
    ],
)
def test_public_pool_rejects_invalid_geometry(kwargs):
    arguments = dict(height=4, width=4, channels=16)
    arguments.update(kwargs)
    with pytest.raises(ValueError, match="pooling"):
        fp16_max_pool2d_window(**arguments)


def test_public_pool_rejects_different_vector_operation():
    with pytest.raises(ValueError, match="MAXPOOL"):
        PublicPoolPlan(VectorOp.COPY, 1).jit_kernel()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
