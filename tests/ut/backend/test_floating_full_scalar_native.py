# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Bitwise/native coverage for floating fill constants on the 310P public API."""

import os
import re
import shutil
import subprocess
from pathlib import Path

import pypto.language as pl
import pytest
from pypto import ir, passes
from pypto.backend import BackendType
from pypto.runtime.pto_isa import pto_isa_include_dir


def _kernel(rows, dtype, zero):
    if zero:

        @pl.jit.incore
        def kernel(output: pl.Out[pl.Tensor[[rows, 16], dtype]]):
            value = pl.tile.full([rows, 16], dtype, 0.0)
            return pl.store(value, [0, 0], output)

    else:

        @pl.jit.incore
        def kernel(output: pl.Out[pl.Tensor[[rows, 16], dtype]]):
            value = pl.tile.full([rows, 16], dtype, 0.125)
            return pl.store(value, [0, 0], output)

    return kernel


@pytest.mark.parametrize("rows", [16, 256])
@pytest.mark.parametrize("dtype", [pl.FP16, pl.FP32])
@pytest.mark.parametrize("zero", [False, True])
def test_floating_fill_cpu_native(tmp_path, pass_verification_instruments, rows, dtype, zero):
    with passes.PassContext(pass_verification_instruments):
        ir.compile(
            _kernel(rows, dtype, zero).specialize(),
            backend_type=BackendType.Ascend310P,
            output_dir=str(tmp_path),
        )
    sources = list((tmp_path / "kernels").glob("*/*.cpp"))
    assert len(sources) == 1
    code = sources[0].read_text()
    assert code.count("TEXPANDS(") == 1 and code.count("TSTORE(") == 1
    scalar_type, runtime_type = ("half", "FLOAT16") if dtype == pl.FP16 else ("float", "FLOAT32")
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
    cpu, native, readelf = shutil.which("clang++-19"), shutil.which("ccec"), shutil.which("readelf")
    assert cpu and native and readelf
    oracle = tmp_path / "oracle.cpp"
    oracle.write_text(
        "#include <cstdint>\n"
        + code
        + "\n"
        + Path(__file__).with_name("floating_full_scalar_oracle.cpp").read_text()
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
            f"-DFS_ROWS={rows}",
            f"-DFS_TYPE={scalar_type}",
            f"-DFS_DTYPE={runtime_type}",
            f"-DFS_VALUE={'0.0f' if zero else '0.125f'}",
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
    relocations = subprocess.run(
        [readelf, "-rW", str(tmp_path / "native.o")], check=True, capture_output=True, text=True
    ).stdout
    assert not re.search(r"Relocation section '\.rel(a)?\.text", relocations)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
