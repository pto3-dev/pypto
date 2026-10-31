# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Exact first-stage anchor pruning through public PyPTO only."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from pypto import ir, passes
from pypto.backend import BackendType
from pypto.runtime.pto_isa import pto_isa_include_dir

from examples.models.yolo26_native.public_topk import AnchorPrefixPlan

_PROFILES = [
    (anchors, passes.MemoryPlanner.PYPTO) for anchors in (300, 304, 319, 320, 321, 327, 328, 336, 1344, 8400)
] + [
    (anchors, planner)
    for anchors in (336, 8400)
    for planner in (passes.MemoryPlanner.PTOAS, passes.MemoryPlanner.DSA_RP)
]


@pytest.mark.parametrize("anchors,planner", _PROFILES)
def test_public_anchor_prefix_exact(tmp_path, pass_verification_instruments, anchors, planner):
    plan = AnchorPrefixPlan(anchors)
    directory = tmp_path / "compiled"
    with passes.PassContext(pass_verification_instruments, memory_planner=planner):
        ir.compile(
            plan.jit_kernel().specialize(), backend_type=BackendType.Ascend310P, output_dir=str(directory)
        )
    sources = list((directory / "kernels").glob("*/*.cpp"))
    assert len(sources) == 1
    generated = sources[0]
    code = generated.read_text()
    assert "TSORT32(" in code and "TMRGSORT<" in code and "TROWMAX(" in code
    assert code.count("TSTORE(") == 1
    params = json.loads((directory / "compiled_meta.json").read_text())["params"]
    assert [p["dtype"] for p in params] == ["int32", "fp32", "fp32"]
    assert params[-1]["shape"] == [608]
    metadata = tmp_path / "indices.bin"
    metadata.write_bytes(plan.metadata().numpy().tobytes())
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
    assert cpu and native, "requires CPU semantics and driver-free native compilation"
    oracle = tmp_path / "oracle.cpp"
    oracle.write_text(
        "#include <cstdint>\n"
        + code
        + "\n"
        + Path(__file__).with_name("public_anchor_prefix_oracle.cpp").read_text()
    )
    binary = tmp_path / "oracle"
    subprocess.run(
        [
            cpu,
            "-std=c++20",
            "-O1",
            "-D__CPU_SIM",
            "-D__DAV_CUBE__",
            "-D__DAV_VEC__",
            f"-DTOPK_ANCHORS={anchors}",
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
        [str(binary), str(metadata)],
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
