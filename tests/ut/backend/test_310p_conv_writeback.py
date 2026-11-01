# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Public convolution epilogues preserve rounding, row slices and GM guards."""

import importlib.util
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pypto.language as pl
import pytest
import torch
from pypto import ir, passes
from pypto.backend import BackendType
from pypto.library import fp16_conv_epilogue, fp16_conv_writeback
from pypto.runtime.pto_isa import pto_isa_include_dir

_CASES = [
    (16, 32, 16, False),
    (32, 80, 32, False),
    (64, 32, 0, False),
    (112, 80, 64, False),
    (128, 48, 16, False),
    (224, 64, 32, False),
    (256, 80, 16, False),
    (64, 48, 16, True),
    (128, 64, 32, True),
    (256, 80, 16, True),
]


def _epilogue_kernel(rows, columns, column, direct, silu):
    output_rows = rows + 32
    constants = rows if direct else 64
    if direct:
        finish = fp16_conv_epilogue(rows, silu=silu)

        @pl.jit.inline
        def execute(
            value: pl.Tile[[rows, 16], pl.FP32, pl.Mem.Vec],
            bias: pl.Tile[[1, 16], pl.FP32],
            zero: pl.Tile[[rows, 16], pl.FP32, pl.Mem.Vec],
            offset: pl.Tile[[rows, 16], pl.FP32, pl.Mem.Vec],
            scale: pl.Tile[[rows, 16], pl.FP32, pl.Mem.Vec],
            one: pl.Tile[[rows, 16], pl.FP32, pl.Mem.Vec],
            epsilon: pl.Tile[[rows, 16], pl.FP32, pl.Mem.Vec],
            output: pl.Tensor[[output_rows, columns], pl.FP16],
            row: pl.Scalar[pl.INDEX],
            out_column: pl.Scalar[pl.INDEX],
        ):
            return pl.store(finish(value, bias, zero, offset, scale, one, epsilon), [row, out_column], output)

    else:
        execute = fp16_conv_writeback(rows, output_rows, columns, silu=silu)

    @pl.jit.incore
    def kernel(
        source: pl.Tensor[[rows, 16], pl.FP32],
        bias: pl.Tensor[[1, 16], pl.FP32],
        output: pl.Out[pl.Tensor[[output_rows, columns], pl.FP16]],
    ):
        zero = pl.tile.full([constants, 16], pl.FP32, 0.0)
        offset = pl.tile.full([constants, 16], pl.FP32, 15.0 / 2048)
        scale = pl.tile.full([constants, 16], pl.FP32, 8192.0)
        one = pl.tile.full([constants, 16], pl.FP32, 1.0)
        epsilon = pl.tile.full([constants, 16], pl.FP32, 0.00000011920928955078125)
        value = pl.load(source, [0, 0], [rows, 16], target_memory=pl.Mem.Vec)
        vbias = pl.load(bias, [0, 0], [1, 16], target_memory=pl.Mem.Vec)
        return execute(value, vbias, zero, offset, scale, one, epsilon, output, 16, column)

    return kernel


@pytest.mark.parametrize("case", _CASES)
@pytest.mark.parametrize("silu", [False, True])
@pytest.mark.parametrize(
    "planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.PTOAS, passes.MemoryPlanner.DSA_RP]
)
def test_public_conv_epilogue_cpu_native(tmp_path, pass_verification_instruments, case, silu, planner):
    rows, columns, column, direct = case
    kernel = _epilogue_kernel(*case, silu)
    assert not kernel._external_source_paths()
    with passes.PassContext(pass_verification_instruments, memory_planner=planner):
        ir.compile(kernel.specialize(), backend_type=BackendType.Ascend310P, output_dir=str(tmp_path))
    sources = list((tmp_path / "kernels").glob("*/*.cpp"))
    assert len(sources) == 1
    code = sources[0].read_text()
    chunks = 1 if direct else rows // 64 + (rows % 64) // 32 + (rows % 32) // 16
    assert code.count("TSTORE(") == chunks
    assert code.count("TEXP(") == (chunks if silu else 0)
    values_data, bias_data, expected_data = bytearray(), bytearray(), bytearray()
    index = torch.arange(rows * 16, dtype=torch.int64)
    for dataset in range(3):
        bias = (torch.arange(16, dtype=torch.float32) * 7 % 17 - 8) / 37
        bias[1] = 0
        if dataset == 1:
            bits = index * 7919 % 63488
            bits = torch.where(bits < 31744, bits, (bits - 31744) | 0x8000)
            value = bits.to(torch.uint16).view(torch.float16).float()
        else:
            value = ((index * 13 % 129) - 64).float() / (32 if dataset == 0 else 4096)
            if dataset == 2:
                value += torch.where(index % 2 != 0, 2.0**-24, -(2.0**-24))
        value = value.reshape(rows, 16)
        value[0, 0] = -15.0 / 2048 - bias[0]
        value[0, 1] = -0.0
        rounded = (value + bias).half()
        expected = rounded * rounded.sigmoid() if silu else rounded
        values_data.extend(value.numpy().tobytes())
        bias_data.extend(bias.numpy().tobytes())
        expected_data.extend(expected.numpy().tobytes())
    files = [tmp_path / name for name in ("source.bin", "bias.bin", "expected.bin")]
    for path, data in zip(files, (values_data, bias_data, expected_data)):
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
        + Path(__file__).with_name("conv_writeback_oracle.cpp").read_text()
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
            f"-DCW_ROWS={rows}",
            f"-DCW_COLUMNS={columns}",
            f"-DCW_COLUMN={column}",
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
        [str(binary), *map(str, files)],
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PTO_CPU_SIM_NUM_CORES": "1",
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


@pytest.mark.parametrize("rows", [0, True, 16.0, 17])
def test_conv_epilogue_rejects_bad_rows(rows):
    with pytest.raises(ValueError, match="convolution epilogue requires"):
        fp16_conv_epilogue(rows)


@pytest.mark.parametrize(
    "geometry",
    [
        (0, 32, 16),
        (17, 32, 16),
        (32, 16, 32),
        (32, 64, 15),
        (True, 64, 16),
        (32, 64, 16.0),
        (4096, 4096, 16),
        (16, 2**32, 16),
    ],
)
def test_conv_writeback_rejects_bad_geometry(geometry):
    with pytest.raises(ValueError, match="convolution writeback requires"):
        fp16_conv_writeback(*geometry)


@pytest.mark.parametrize("rows", [16, 32, 112, 224])
@pytest.mark.parametrize("silu", [False, True])
def test_device_writeback_fixture_offline_compile(tmp_path, monkeypatch, rows, silu):
    # Compile the exact future device-test kernel without invoking its hardware
    # fixture. Stop before constructing a session; never mock numerical results.
    path = Path(__file__).resolve().parents[2] / "st/runtime/test_310p_exp_div.py"
    name = "pypto_conv_writeback_offline_fixture"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    calls = []

    class DeviceNotOpened(Exception):
        pass

    def stop_device(compiled):
        calls.append(compiled)
        raise DeviceNotOpened

    def deny_loader(*_args, **_kwargs):
        raise AssertionError("device discovery is forbidden in offline validation")

    monkeypatch.setattr(module, "A1ResidentSession", stop_device)
    monkeypatch.setattr(module, "ctypes", SimpleNamespace(CDLL=deny_loader))
    with pytest.raises(DeviceNotOpened):
        module.test_310p_public_conv_writeback_strided(tmp_path, rows, silu)
    assert len(calls) == 1
    assert len(list((tmp_path / "kernels").glob("*/*.cpp"))) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
