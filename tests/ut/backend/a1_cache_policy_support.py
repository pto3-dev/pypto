# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Shared CPU/native support for the A1 load cache-policy regression matrix."""

import os
import shutil
import subprocess
from pathlib import Path

import pypto.language as pl
from pypto.runtime import a1_runner
from pypto.runtime.pto_isa import pto_isa_include_dir


def capture_entries(monkeypatch, directory):
    """Delegate native compilation unchanged and preserve each temporary entry."""
    compiler_type, launcher_type = a1_runner._load_tools()
    captured = []

    class CapturingCompiler:
        def __init__(self, platform):
            self.compiler = compiler_type(platform)

        def compile_single_launch(self, source, isa, **kwargs):
            saved = directory / f"entry_{len(captured)}.cpp"
            saved.write_bytes(Path(source).read_bytes())
            captured.append(saved)
            return self.compiler.compile_single_launch(source, isa, **kwargs)

    monkeypatch.setattr(a1_runner, "_load_tools", lambda: (CapturingCompiler, launcher_type))
    return captured


def run_cpu_oracle(entry, directory, dtype, *, rows, blocks, order=(0, 1), bypass=True, instances=1):
    """Run the actual outer entry against disjoint raw-word alias regions."""
    directory.mkdir(exist_ok=True)
    compiler = shutil.which("clang++-19")
    assert compiler, "requires clang++-19 for the raw-byte CPU oracle"
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
    position = [pl.FP16, pl.FP32, pl.INT32, pl.UINT32].index(dtype)
    ctype, tag = [("half", "FLOAT16"), ("float", "FLOAT32"), ("int32_t", "INT32"), ("uint32_t", "UINT32")][
        position
    ]
    source = directory / "oracle.cpp"
    source.write_text(
        entry.read_text() + "\n" + Path(__file__).with_name("a1_cache_policy_oracle.cpp").read_text()
    )
    binary = directory / "oracle"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O1",
            "-D__CPU_SIM",
            "-D__DAV_CUBE__",
            "-D__DAV_VEC__",
            "-D__global__=",
            f"-DCACHE_CTYPE={ctype}",
            f"-DCACHE_DTYPE=DataType::{tag}",
            f"-DCACHE_BYPASS={int(bypass)}",
            f"-DCACHE_ROWS={rows}",
            f"-DCACHE_INSTANCES={instances}",
            f"-DCACHE_BLOCKS={blocks}",
            f"-DCACHE_ARG0={order[0]}",
            f"-DCACHE_ARG1={order[1]}",
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
            "PTO_CPU_SIM_NUM_CORES": str(blocks),
            "PTO_CPU_SIM_TRACE_ENABLE": "0",
            "OMP_NUM_THREADS": "1",
        },
    )
