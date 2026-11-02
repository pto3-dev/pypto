# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Exact COPY semantics through real descriptors and all A1 memory planners."""

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from pypto import ir, passes
from pypto.backend import BackendType
from pypto.runtime import a1_runner
from pypto.runtime.pto_isa import ensure_pto_isa_root, pto_isa_include_dir

from examples.models.yolo26_native.layout import VectorOp
from examples.models.yolo26_native.public_copy import PublicCopyPlan

# Exact stride-256 attention channel views captured from the real 128/640
# model graphs. Each shape reuses one COPY factory; offsets are channels.
_PRODUCTION_VIEWS = ((32, 0), (32, 32), (64, 64), (32, 128), (32, 160), (64, 192))
_PRODUCTION_PLANS = [
    PublicCopyPlan(VectorOp.COPY, side, side, channels, 256, channels, offset, 0)
    for side in (4, 20)
    for channels, offset in _PRODUCTION_VIEWS
]
_PRODUCTION_BATCH_PLANS = [plan for plan in _PRODUCTION_PLANS if plan.height == 20]

_PLANS = [
    PublicCopyPlan(VectorOp.COPY, 1, 1, 16, 16, 16),
    PublicCopyPlan(VectorOp.COPY, 1, 31, 80, 128, 160, 16, 48),
    PublicCopyPlan(VectorOp.COPY, 1, 17, 1024, 1024, 1024),
    PublicCopyPlan(VectorOp.COPY, 1, 3, 4096, 4112, 4128, 16, 32),
    PublicCopyPlan(VectorOp.COPY, 1, 65, 32, 32, 32),
    # Every FP16 bit pattern traverses an active copied lane in this profile.
    PublicCopyPlan(VectorOp.COPY, 1, 4096, 16, 16, 16),
    PublicCopyPlan(VectorOp.COPY, 20, 20, 128, 128, 128),
    PublicCopyPlan(VectorOp.COPY, 1, 1, 16, 1048576, 16),
] + _PRODUCTION_PLANS


@pytest.mark.parametrize("plan", _PLANS, ids=lambda p: f"p{p.height * p.width}-c{p.channels}")
@pytest.mark.parametrize(
    "planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.PTOAS, passes.MemoryPlanner.DSA_RP]
)
def test_public_copy_cpu_native(tmp_path, pass_verification_instruments, plan, planner):
    assert all(shutil.which(x) for x in ("clang++-19", "ccec", "ptoas"))
    jit = plan.jit_kernel()
    assert not jit._external_source_paths()
    with passes.PassContext(pass_verification_instruments, memory_planner=planner):
        ir.compile(jit.specialize(), backend_type=BackendType.Ascend310P, output_dir=str(tmp_path))
    kernels = list((tmp_path / "kernels").glob("*/channel_copy.cpp"))
    assert len(kernels) == 1
    generated = kernels[0]
    metadata = json.loads((tmp_path / "compiled_meta.json").read_text())
    assert [(p["shape"], p["dtype"]) for p in metadata["params"]] == [
        ([32], "fp32"),
        ([plan.input_size], "fp16"),
        ([16], "fp16"),
        ([plan.output_size], "fp16"),
    ]
    runtime = Path(__file__).resolve().parents[3] / "runtime"
    includes = [
        f"-I{x}"
        for x in (
            pto_isa_include_dir(),
            runtime / "src/a1/platform/include",
            runtime / "src/common",
            runtime / "src/common/task_interface",
            runtime / "src/common/platform/include",
        )
    ]
    oracle = tmp_path / "oracle.cpp"
    oracle.write_text(
        "#include <cstdint>\n"
        + generated.read_text()
        + "\n"
        + Path(__file__).with_name("public_copy_oracle.cpp").read_text()
    )
    defines = dict(
        COPY_PIXELS=plan.height * plan.width,
        COPY_CHANNELS=plan.channels,
        COPY_SOURCE_STRIDE=plan.source_stride,
        COPY_DESTINATION_STRIDE=plan.destination_stride,
        COPY_SOURCE_OFFSET=plan.source_offset,
        COPY_DESTINATION_OFFSET=plan.destination_offset,
    )
    binary = tmp_path / "oracle"
    subprocess.run(
        [
            "clang++-19",
            "-std=c++20",
            "-O1",
            "-D__CPU_SIM",
            "-D__DAV_CUBE__",
            "-D__DAV_VEC__",
            *[f"-D{k}={v}" for k, v in defines.items()],
            *includes,
            str(oracle),
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
            "ccec",
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
    assert (tmp_path / "native.o").read_bytes().startswith(b"\x7fELF")
    batch_wrapper = (plan.height, plan.width, plan.channels) == (
        20,
        20,
        128,
    ) or plan in _PRODUCTION_BATCH_PLANS
    if batch_wrapper:
        image = a1_runner._compile_image(
            tmp_path, generated, ensure_pto_isa_root(), independent_instances=8, argument_count=4
        )
        assert image.startswith(b"\x7fELF")

    # Persist coverage by exact geometry and planner, rather than relying on
    # pytest's shortened parameter IDs or counting repeated kernel calls.
    (tmp_path / "copy-production-profile.json").write_text(
        json.dumps(
            dict(
                geometry=[
                    plan.height,
                    plan.width,
                    plan.channels,
                    plan.source_stride,
                    plan.destination_stride,
                    plan.source_offset,
                    plan.destination_offset,
                ],
                planner=str(planner),
                production=plan in _PRODUCTION_PLANS,
                independent_instances=8 if batch_wrapper else 1,
                output_word_comparisons=plan.output_size * 2 * 4,
                generated_sha256=hashlib.sha256(generated.read_bytes()).hexdigest(),
            )
        )
    )


@pytest.mark.parametrize(
    "plan,message",
    [
        (PublicCopyPlan(VectorOp.ADD, 1), "requires a COPY plan"),
        (PublicCopyPlan(VectorOp.COPY, 1, 1, 4112, 4112, 4112), "at most 4096 channels"),
        (PublicCopyPlan(VectorOp.COPY, 1, 1, 16, 1048592, 16), "bounded DMA row gaps"),
    ],
)
def test_public_copy_rejects_unsupported_plan(plan, message):
    with pytest.raises(ValueError, match=message):
        plan.jit_kernel()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
