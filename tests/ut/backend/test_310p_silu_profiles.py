# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Exhaustive public SiLU for small spatial tiles and retained row profiles."""

import os
import re
import shutil
import subprocess
from pathlib import Path

import pypto.language as pl
import pytest
import torch
from pypto import ir, passes
from pypto.backend import BackendType
from pypto.library import fp16_silu, fp16_silu_constants
from pypto.runtime.pto_isa import pto_isa_include_dir


def _silu_kernel(rows, resident):
    elements = ((65536 + rows * 16 - 1) // (rows * 16)) * rows * 16
    image_rows = elements // 16
    if resident:
        activate = fp16_silu_constants(rows)

        @pl.jit.incore
        def kernel(
            source: pl.Tensor[[image_rows, 16], pl.FP16],
            output: pl.Out[pl.Tensor[[image_rows, 16], pl.FP16]],
        ):
            zero = pl.tile.full([rows, 16], pl.FP32, 0.0)
            offset = pl.tile.full([rows, 16], pl.FP32, 15.0 / 2048)
            scale = pl.tile.full([rows, 16], pl.FP32, 8192.0)
            one = pl.tile.full([rows, 16], pl.FP32, 1.0)
            epsilon = pl.tile.full([rows, 16], pl.FP32, 0.00000011920928955078125)
            for work in pl.range(pl.tile.get_block_idx(), image_rows // rows, pl.tile.get_block_num()):
                value = pl.load(source, [work * rows, 0], [rows, 16])
                result = activate(value, zero, offset, scale, one, epsilon)
                output = pl.store(result, [work * rows, 0], output)
            return output

    else:
        activate = fp16_silu(rows)

        @pl.jit.incore
        def kernel(
            source: pl.Tensor[[image_rows, 16], pl.FP16],
            output: pl.Out[pl.Tensor[[image_rows, 16], pl.FP16]],
        ):
            for work in pl.range(pl.tile.get_block_idx(), image_rows // rows, pl.tile.get_block_num()):
                value = pl.load(source, [work * rows, 0], [rows, 16])
                output = pl.store(activate(value), [work * rows, 0], output)
            return output

    return kernel, elements


@pytest.mark.parametrize("rows", [16, 32, 64, 128, 160, 256])
@pytest.mark.parametrize("resident", [False, True])
@pytest.mark.parametrize(
    "planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.PTOAS, passes.MemoryPlanner.DSA_RP]
)
def test_silu_finite_patterns_cpu_native(tmp_path, pass_verification_instruments, rows, resident, planner):
    kernel, elements = _silu_kernel(rows, resident)
    assert not kernel._external_source_paths()
    with passes.PassContext(pass_verification_instruments, memory_planner=planner):
        ir.compile(kernel.specialize(), backend_type=BackendType.Ascend310P, output_dir=str(tmp_path))
    sources = list((tmp_path / "kernels").glob("*/*.cpp"))
    assert len(sources) == 1
    code = sources[0].read_text()
    assert code.count("TEXP(") == 1 and code.count("TDIV(") == 1 and code.count("TSTORE(") == 1
    patterns = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.float16)
    finite = patterns[torch.isfinite(patterns)]
    assert finite.numel() == 63488
    original = torch.cat((finite, torch.zeros(elements - finite.numel(), dtype=torch.float16)))
    files = [tmp_path / name for name in ("input.bin", "expected.bin")]
    # The same separate FP16 sigmoid/product reference as the existing device
    # tests. Exercise sorted and shuffled patterns with all finite bit values.
    inputs, references = bytearray(), bytearray()
    for seed in (0, 991):
        order = (
            torch.arange(elements)
            if not seed
            else torch.randperm(elements, generator=torch.Generator().manual_seed(seed))
        )
        values = original[order].contiguous()
        inputs.extend(values.numpy().tobytes())
        references.extend((values * values.sigmoid()).numpy().tobytes())
    for path, data in zip(files, (inputs, references)):
        path.write_bytes(data)
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
    cpu, native, readelf = shutil.which("clang++-19"), shutil.which("ccec"), shutil.which("readelf")
    assert cpu and native and readelf
    oracle = tmp_path / "oracle.cpp"
    oracle.write_text(
        "#include <cstdint>\n"
        + code
        + "\n"
        + Path(__file__).with_name("silu_profiles_oracle.cpp").read_text()
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
            f"-DSILU_ELEMENTS={elements}",
            *includes,
            str(oracle),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    for cores in (1, 8):
        subprocess.run(
            [str(binary), *map(str, files)],
            check=True,
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "PTO_CPU_SIM_NUM_CORES": str(cores),
                "PTO_CPU_SIM_TRACE_ENABLE": "0",
                "OMP_NUM_THREADS": "1",
                "PTO_CPU_SIM_UB_BYTES": "131072",
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


@pytest.mark.parametrize("factory", [fp16_silu, fp16_silu_constants])
@pytest.mark.parametrize(
    "rows,columns",
    [
        (0, 16),
        (1, 16),
        (13, 16),
        (15, 16),
        (17, 16),
        (48, 16),
        (1, 32),
        (32, 32),
        (64, 80),
        (True, 80),
        (16.0, 16),
        (16, True),
    ],
)
def test_silu_profile_gate_preserves_unsupported_shapes(factory, rows, columns):
    with pytest.raises(ValueError, match="supported static"):
        factory(rows, columns)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
