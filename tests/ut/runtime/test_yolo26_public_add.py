# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Full public compilation, runtime ABI and bitwise CPU checks for residual ADD."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest
from pypto import ir
from pypto.backend import BackendType
from pypto.runtime.pto_isa import pto_isa_include_dir

from examples.models.yolo26_native.layout import VectorOp
from examples.models.yolo26_native.public_add import PublicAddPlan

# Whole/odd tile counts, a partial final row panel, a single short panel,
# three distinct row strides, production geometry and the bounded wide row gap.
_CASES = [
    PublicAddPlan(VectorOp.ADD, 20, 20, 128, 128, 128, second_stride=128),
    PublicAddPlan(VectorOp.ADD, 1, 31, 80, 128, 160, 16, 48, second_stride=144, second_offset=32),
    PublicAddPlan(VectorOp.ADD, 1, 7, 16, 16, 16),
    PublicAddPlan(VectorOp.ADD, 1, 17, 1024, 1024, 1024, second_stride=1024),
    PublicAddPlan(VectorOp.ADD, 1, 2048, 64, 64, 64, second_stride=64),
    PublicAddPlan(VectorOp.ADD, 1, 1, 16, 1048576, 16),
    # All remaining distinct residual layouts captured from the real 640 model;
    # the 20x20 C128 contiguous layout is already the first case above.
    PublicAddPlan(VectorOp.ADD, 160, 160, 16, 32, 16, 16, 0, second_stride=16),
    PublicAddPlan(VectorOp.ADD, 80, 80, 32, 64, 32, 32, 0, second_stride=32),
    PublicAddPlan(VectorOp.ADD, 40, 40, 32, 32, 32, second_stride=32),
    PublicAddPlan(VectorOp.ADD, 20, 20, 64, 64, 64, second_stride=64),
    PublicAddPlan(VectorOp.ADD, 20, 20, 256, 256, 256, second_stride=256),
    PublicAddPlan(VectorOp.ADD, 20, 20, 128, 256, 128, 128, 0, second_stride=128),
    PublicAddPlan(VectorOp.ADD, 80, 80, 16, 16, 16),
]


@pytest.fixture(
    scope="module", params=_CASES, ids=lambda p: f"p{p.height * p.width}-c{p.channels}-s{p.source_stride}"
)
def compiled_add(request, tmp_path_factory):
    if not shutil.which("ptoas"):
        pytest.skip("requires PTOAS for complete public compilation")
    plan = request.param
    root = tmp_path_factory.mktemp("public-add")
    ir.compile(plan.jit_kernel().specialize(), backend_type=BackendType.Ascend310P, output_dir=str(root))
    # The A1 runtime consumes only final wrappers under kernels/, not the
    # similarly named intermediate C++ emitted into ptoas/.
    sources = list((root / "kernels").glob("*/residual_add.cpp"))
    assert len(sources) == 1
    return plan, sources[0]


def _include_flags():
    runtime = Path(__file__).resolve().parents[3] / "runtime"
    paths = (
        pto_isa_include_dir(),
        runtime / "src/a1/platform/include",
        runtime / "src/common",
        runtime / "src/common/task_interface",
        runtime / "src/common/platform/include",
    )
    return [f"-I{p}" for p in paths]


def test_public_add_cpu_abi(compiled_add, tmp_path):
    compiler = shutil.which("clang++-19")
    if not compiler:
        pytest.skip("requires clang++-19 for CPU instruction semantics")
    plan, generated = compiled_add
    source = tmp_path / "oracle.cpp"
    source.write_text(
        "#include <cstdint>\n"
        + generated.read_text()
        + "\n"
        + Path(__file__).with_name("public_add_oracle.cpp").read_text()
    )
    defines = {
        "ADD_PIXELS": plan.height * plan.width,
        "ADD_CHANNELS": plan.channels,
        "ADD_SOURCE_STRIDE": plan.source_stride,
        "ADD_SECOND_STRIDE": plan.second_stride,
        "ADD_DESTINATION_STRIDE": plan.destination_stride,
        "ADD_SOURCE_OFFSET": plan.source_offset,
        "ADD_SECOND_OFFSET": plan.second_offset,
        "ADD_DESTINATION_OFFSET": plan.destination_offset,
    }
    binary = tmp_path / "oracle"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O1",
            "-ffp-contract=off",
            "-D__CPU_SIM",
            "-D__DAV_CUBE__",
            "-D__DAV_VEC__",
            *[f"-D{k}={v}" for k, v in defines.items()],
            *_include_flags(),
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


def test_public_add_native_compile(compiled_add, tmp_path):
    compiler = shutil.which("ccec")
    if not compiler:
        pytest.skip("requires CCEC for offline dav-m200 compilation")
    _, source = compiled_add
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
            *_include_flags(),
            str(source),
            "-o",
            str(tmp_path / "native.o"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    "plan,message",
    [
        (PublicAddPlan(VectorOp.COPY, 1), "requires an ADD plan"),
        (PublicAddPlan(VectorOp.ADD, 1, 1, 1040, 1040, 1040, second_stride=1040), "at most 1024 channels"),
        (PublicAddPlan(VectorOp.ADD, 1, 1, 16, 1048592, 16), "bounded DMA row gaps"),
    ],
)
def test_public_add_rejects_unsupported_profile(plan, message):
    with pytest.raises(ValueError, match=message):
        plan.jit_kernel()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
