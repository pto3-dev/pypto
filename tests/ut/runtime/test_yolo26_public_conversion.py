# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Exact dense conversion ABI, rounding, pipeline and native compilation."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from pypto import ir, passes
from pypto.backend import BackendType
from pypto.runtime.pto_isa import pto_isa_include_dir

from examples.models.yolo26_native.layout import _CASTS, VectorOp
from examples.models.yolo26_native.public_conversion import PublicConversionPlan

_PROFILES = (
    [(op, length, passes.MemoryPlanner.PYPTO) for op in _CASTS for length in (256, 768, 12800, 63488)]
    + [
        (op, 12800, planner)
        for op in _CASTS
        for planner in (passes.MemoryPlanner.PTOAS, passes.MemoryPlanner.DSA_RP)
    ]
    + [(op, 4352, passes.MemoryPlanner.PYPTO) for op in (VectorOp.FLOAT_TO_HALF, VectorOp.HALF_TO_FLOAT)]
)


@pytest.mark.parametrize("operation,length,planner", _PROFILES)
def test_public_conversion_exact(tmp_path, pass_verification_instruments, operation, length, planner):
    plan = PublicConversionPlan(operation, length)
    directory = tmp_path / "compiled"
    with passes.PassContext(pass_verification_instruments, memory_planner=planner):
        ir.compile(
            plan.jit_kernel().specialize(), backend_type=BackendType.Ascend310P, output_dir=str(directory)
        )
    sources = list((directory / "kernels").glob("*/*.cpp"))
    assert len(sources) == 1
    code = sources[0].read_text()
    assert "TCVT(" in code
    assert ("TMULS(" in code) == (operation != VectorOp.FLOAT_TO_HALF)
    params = json.loads((directory / "compiled_meta.json").read_text())["params"]
    assert [p["dtype"] for p in params] == [
        "fp32",
        "fp32" if operation == VectorOp.FLOAT_TO_HALF else "fp16",
        "fp16",
        "fp32" if operation == VectorOp.HALF_TO_FLOAT else "fp16",
    ]
    assert [p["shape"] for p in params] == [[32], [length], [16], [length]]
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
    cpu, native = shutil.which("clang++-19"), shutil.which("ccec")
    assert cpu and native, "requires CPU instruction semantics and driver-free CCEC"
    source = tmp_path / "oracle.cpp"
    source.write_text(
        "#include <cstdint>\n"
        + code
        + "\n"
        + Path(__file__).with_name("public_conversion_oracle.cpp").read_text()
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
            f"-DCONVERSION_OP={int(operation)}",
            f"-DCONVERSION_LENGTH={length}",
            *includes,
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


@pytest.mark.parametrize("operation", [VectorOp.COPY, VectorOp.ADD, VectorOp.PACK])
def test_public_conversion_rejects_other_operations(operation):
    with pytest.raises(ValueError, match="public conversion requires a dense conversion operation"):
        PublicConversionPlan(operation, 256).jit_kernel()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
