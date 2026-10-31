# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Generated public decode kernel: native rounding, ABI and offline compilation."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from pypto import ir
from pypto.backend import BackendType
from pypto.runtime.pto_isa import pto_isa_include_dir

from examples.models.yolo26_native.public_decode import PublicDecodePlan

_CASES = [
    PublicDecodePlan(1, 1, 8, 3, 9),
    PublicDecodePlan(1, 7, 16, 3, 19),
    PublicDecodePlan(1, 8, 8, 3, 21),
    PublicDecodePlan(4, 4, 32, 7, 29),
    PublicDecodePlan(8, 8, 16, 256, 336),
    PublicDecodePlan(16, 16, 8, 0, 336),
    PublicDecodePlan(80, 80, 8, 0, 8400),
    PublicDecodePlan(40, 40, 16, 6400, 8400),
    PublicDecodePlan(20, 20, 32, 8000, 8400),
]


@pytest.fixture(scope="module", params=_CASES, ids=lambda p: f"h{p.height}-w{p.width}-s{p.stride}")
def compiled_decode(request, tmp_path_factory):
    assert shutil.which("ptoas"), "requires PTOAS for full public compilation"
    plan = request.param
    output = tmp_path_factory.mktemp("public-decode")
    ir.compile(plan.jit_kernel().specialize(), backend_type=BackendType.Ascend310P, output_dir=str(output))
    sources = list((output / "kernels").glob("*/public_decode.cpp"))
    assert len(sources) == 1
    metadata = output / "indices.bin"
    metadata.write_bytes(plan.metadata().numpy().tobytes())
    signature = json.loads((output / "compiled_meta.json").read_text())["params"]
    assert [p["dtype"] for p in signature] == ["int32", "fp16", "fp16", "fp32"]
    return plan, sources[0], metadata


def _includes() -> list[str]:
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


def test_public_decode_compiles(compiled_decode):
    _, source, _ = compiled_decode
    code = source.read_text()
    assert "TGATHER(" in code
    assert "TEXP(" in code and "TDIV(" in code


def test_public_decode_cpu_abi(compiled_decode, tmp_path):
    compiler = shutil.which("clang++-19")
    assert compiler, "requires clang++-19 for CPU instruction semantics"
    plan, generated, metadata = compiled_decode
    source = tmp_path / "oracle.cpp"
    source.write_text(
        "#include <cstdint>\n"
        + generated.read_text()
        + "\n"
        + Path(__file__).with_name("public_decode_oracle.cpp").read_text()
    )
    definitions = {
        "DECODE_HEIGHT": plan.height,
        "DECODE_WIDTH": plan.width,
        "DECODE_STRIDE": plan.stride,
        "DECODE_OFFSET": plan.anchor_offset,
        "DECODE_TOTAL": plan.total_anchors,
        "DECODE_TABLE_COUNT": plan.metadata().numel(),
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
            *[f"-D{k}={v}" for k, v in definitions.items()],
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
            [str(binary), str(metadata)],
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


def test_public_decode_native_compile(compiled_decode, tmp_path):
    compiler = shutil.which("ccec")
    assert compiler, "requires CCEC for offline dav-m200 compilation"
    _, source, _ = compiled_decode
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
            str(source),
            "-o",
            str(tmp_path / "native.o"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
