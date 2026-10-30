# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Bitwise Tensor-ABI checks for public nearest-neighbor/concat fusion."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest
from pypto import ir
from pypto.backend import BackendType
from test_yolo26_public_add import _include_flags

from examples.models.yolo26_native.upsample_concat import UpsampleConcatPlan

# All model geometries, reverse order, odd/sliced sources and ABI boundaries.
_CASES = [
    UpsampleConcatPlan(h, w, ca, cb, ca, cb)
    for h, w, ca, cb in [(20, 20, 256, 128), (40, 40, 128, 128), (4, 4, 256, 128), (8, 8, 128, 128)]
] + [
    UpsampleConcatPlan(3, 5, 80, 48, 112, 96, 16, 32, False),
    UpsampleConcatPlan(1, 1, 16, 16, 16, 16),
    UpsampleConcatPlan(1, 3, 1024, 1024, 1056, 1072, 16, 32, False),
    UpsampleConcatPlan(1, 1, 16, 16, 1048576, 1048576),
]


@pytest.fixture(
    scope="module", params=_CASES, ids=lambda p: f"{p.height}x{p.width}-c{p.up_channels}-{p.skip_channels}"
)
def compiled_fusion(request, tmp_path_factory):
    if not shutil.which("ptoas"):
        pytest.skip("requires PTOAS for complete public compilation")
    plan = request.param
    output = tmp_path_factory.mktemp("upsample-concat")
    result = ir.compile(
        plan.jit_kernel().specialize(), backend_type=BackendType.Ascend310P, output_dir=str(output)
    )
    assert isinstance(result, ir.CompiledProgram)
    sources = list((output / "kernels").glob("*/upsample_concat.cpp"))
    assert len(sources) == 1
    return plan, sources[0]


def test_fusion_cpu_tensor_abi(compiled_fusion, tmp_path):
    compiler = shutil.which("clang++-19")
    if not compiler:
        pytest.skip("requires clang++-19 for CPU instruction semantics")
    plan, generated = compiled_fusion
    source = tmp_path / "oracle.cpp"
    source.write_text(
        "#include <cstdint>\n"
        + generated.read_text()
        + "\n"
        + Path(__file__).with_name("public_upsample_concat_oracle.cpp").read_text()
    )
    values = dict(
        H=plan.height,
        W=plan.width,
        CA=plan.up_channels,
        CB=plan.skip_channels,
        SA=plan.up_stride,
        SB=plan.skip_stride,
        OA=plan.up_offset,
        OB=plan.skip_offset,
        FIRST=int(plan.upsample_first),
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
            *[f"-DUC_{key}={value}" for key, value in values.items()],
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


def test_fusion_offline_native_compile(compiled_fusion, tmp_path):
    compiler = shutil.which("ccec")
    if not compiler:
        pytest.skip("requires CCEC for offline dav-m200 compilation")
    _, generated = compiled_fusion
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
            str(generated),
            "-o",
            str(tmp_path / "native.o"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
