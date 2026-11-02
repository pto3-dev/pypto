# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Dense weight layouts, exact four-stream Cube results and executed GM payload."""

import json
import os
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pypto.language as pl
import pytest
import test_yolo26_public_add_trace as host_trace
import torch
from pypto import ir, passes
from pypto.backend import BackendType
from pypto.runtime.pto_isa import pto_isa_include_dir

from examples.models.yolo26_native.dense_pointwise_input import dense_pointwise_image_cache
from examples.models.yolo26_native.dense_pointwise_weights import (
    dense_pointwise_weight_cache,
    pack_dense_pointwise_weights,
)
from examples.models.yolo26_native.operators import ConvPlan

# Register the existing CPU-only fixture under a local pytest name.
real_model = host_trace.model


def _plan(channels: int, outputs: int) -> ConvPlan:
    return ConvPlan(32, 32, channels, outputs, 1, 1, 0, False, True, channels + 32, 16)


def _assert_packing(plan: ConvPlan, original: torch.Tensor, packed: torch.Tensor) -> None:
    phase_rows = (plan.cin + 63) // 64 * 16
    assert packed.dtype == torch.float16 and packed.device.type == "cpu"
    assert list(packed.shape) == [4 * phase_rows, plan.co]
    assert packed.is_contiguous() and not packed.requires_grad
    # Independent flat destination-index equation; do not reproduce the
    # implementation's per-phase strided transpose/copy operations.
    index = np.arange(packed.numel(), dtype=np.int64)
    phase = index // (phase_rows * plan.co)
    channel = (index // plan.co % phase_rows) * 4 + phase
    output = index % plan.co
    valid = (channel < plan.cin) & (output < plan.cout)
    source = original.detach().half().reshape(plan.cout, plan.cin).numpy().view(np.uint16)
    expected = np.zeros(packed.numel(), dtype=np.uint16)
    expected[valid] = source[output[valid], channel[valid]]
    assert np.array_equal(packed.numpy().reshape(-1).view(np.uint16), expected)


@pytest.mark.parametrize("channels,outputs", [(16, 19), (32, 64), (48, 80), (80, 129), (336, 96), (512, 128)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_dense_weight_packing_bits(channels, outputs, dtype):
    plan = _plan(channels, outputs)
    bits = torch.arange(channels * outputs, dtype=torch.int64).remainder(65536).to(torch.uint16)
    original = bits.view(torch.float16).reshape(outputs, channels, 1, 1).to(dtype)
    before = original.clone()
    packed = pack_dense_pointwise_weights(plan, original)
    _assert_packing(plan, original, packed)
    assert np.array_equal(original.numpy().view(np.uint8), before.numpy().view(np.uint8))


def test_dense_weight_packing_ignores_default_device():
    plan = _plan(16, 19)
    original = torch.arange(19 * 16, dtype=torch.float32, device="cpu").reshape(19, 16, 1, 1)
    original.requires_grad_(True)
    # Meta exercises a non-CPU default without querying/initializing hardware.
    # The input's device and gradient ownership must stay with the caller.
    with torch.device("meta"):
        packed = pack_dense_pointwise_weights(plan, original)
    _assert_packing(plan, original, packed)
    assert original.requires_grad and original.device.type == "cpu"


def test_dense_weight_packing_real_weights(real_model, tmp_path):
    checked = []
    for name, layer in real_model.named_modules():
        if not isinstance(layer, torch.nn.Conv2d) or layer.kernel_size != (1, 1):
            continue
        if layer.in_channels % 16 or layer.in_channels > 512:
            continue
        plan = _plan(layer.in_channels, layer.out_channels)
        original = layer.weight.detach()
        before = original.clone()
        packed = pack_dense_pointwise_weights(plan, original)
        _assert_packing(plan, original, packed)
        assert torch.equal(original.view(torch.uint16), before.view(torch.uint16))
        checked.append(
            {"name": name, "channels": plan.cin, "outputs": plan.cout, "shape": list(packed.shape)}
        )
    assert checked
    (tmp_path / "real_weight_packing.json").write_text(json.dumps(checked, indent=2) + "\n")


def _kernel(plan: ConvPlan, rows: int, columns: int, first: int) -> Any:
    image_cache = dense_pointwise_image_cache(plan, rows)
    weight_cache = dense_pointwise_weight_cache(plan, columns)
    phase_rows = (plan.cin + 63) // 64 * 16
    pixels, stride, outputs = plan.height * plan.width, plan.input_row_stride, plan.co
    output_rows = 8 * rows + 32
    weight_rows = 4 * phase_rows

    @pl.jit.inline
    def consume(
        image: pl.Tile[[rows, phase_rows], pl.FP16, pl.Mem.Mat],
        weight: pl.Tile[[phase_rows, columns], pl.FP16, pl.Mem.Mat],
        output: pl.Tensor[[output_rows, columns], pl.FP32],
        at: pl.Scalar[pl.INDEX],
    ):
        left = pl.tile.move(
            image, pl.Mem.Left, blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.row_major
        )
        for panel in pl.range(columns // 16):
            right = pl.tile.slice(weight, [phase_rows, 16], [0, panel * 16])
            value = pl.tile.matmul(left, right)
            output = pl.store(value, [16 + at * rows, panel * 16], output)
        return output

    @pl.jit.incore
    def phase_products(
        source: pl.Tensor[[pixels, stride], pl.FP16],
        weights: pl.Tensor[[weight_rows, outputs], pl.FP16],
        output: pl.Out[pl.Tensor[[output_rows, columns], pl.FP32]],
    ):
        weight0, weight1, weight2, weight3 = weight_cache(weights, first)
        # Retain all four weight phases across two distinct spatial windows.
        # The CPU observer counts real TLOAD requests, not source-code sites.
        for window in pl.range(2):
            image0, image1, image2, image3 = image_cache(source, 16 + window * rows)
            output = consume(image0, weight0, output, window * 4)
            output = consume(image1, weight1, output, window * 4 + 1)
            output = consume(image2, weight2, output, window * 4 + 2)
            output = consume(image3, weight3, output, window * 4 + 3)
        return output

    return phase_products


_CASES = [
    (16, 16, 19, 32, 0),
    (32, 32, 64, 32, 16),
    (48, 48, 80, 64, 16),
    (64, 64, 128, 64, 32),
    (112, 80, 96, 32, 64),
    (128, 128, 129, 32, 112),
    (176, 192, 64, 64, 0),
    (224, 256, 128, 32, 64),
    (112, 320, 96, 32, 32),
    (112, 336, 80, 32, 48),
    (224, 512, 256, 256, 0),
    (16, 512, 1024, 32, 992),
]


def _datasets(plan: ConvPlan):
    images, packed, original = [], [], []
    for dataset in range(3):

        def finite(count: int, multiplier: int):
            index = torch.arange(count, dtype=torch.int64)
            if dataset == 2:
                bits = index * multiplier % 63488
                bits = torch.where(bits < 31744, bits, (bits - 31744) | 0x8000)
                return bits.to(torch.uint16).view(torch.float16)
            return ((index * multiplier % 101 - 50).float() / (16 if dataset == 0 else 37)).half()

        image = finite(plan.input_size, 73)
        weight = finite(plan.cout * plan.cin, 41).reshape(plan.cout, plan.cin, 1, 1)
        images.append(image.numpy().tobytes())
        packed.append(pack_dense_pointwise_weights(plan, weight).numpy().tobytes())
        original.append(weight.numpy().tobytes())
    return b"".join(images), b"".join(packed), b"".join(original)


@pytest.mark.parametrize("case", _CASES)
@pytest.mark.parametrize(
    "planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.PTOAS, passes.MemoryPlanner.DSA_RP]
)
def test_dense_weight_cache_cpu_native(tmp_path, pass_verification_instruments, case, planner):
    rows, channels, outputs, columns, first = case
    plan = _plan(channels, outputs)
    kernel = _kernel(plan, rows, columns, first)
    assert not kernel._external_source_paths()
    with passes.PassContext(pass_verification_instruments, memory_planner=planner):
        ir.compile(kernel.specialize(), backend_type=BackendType.Ascend310P, output_dir=str(tmp_path))
    sources = list((tmp_path / "kernels").glob("*/*.cpp"))
    assert len(sources) == 1
    code = sources[0].read_text()
    assert "TEXTRACT(" in code and "TMATMUL(" in code
    phase_rows = (channels + 63) // 64 * 16
    params = json.loads((tmp_path / "compiled_meta.json").read_text())["params"]
    assert [p["dtype"] for p in params] == ["fp16", "fp16", "fp32"]
    assert [p["shape"] for p in params] == [
        [1024, channels + 32],
        [4 * phase_rows, plan.co],
        [8 * rows + 32, columns],
    ]
    files = [tmp_path / name for name in ("images.bin", "packed.bin", "original.bin")]
    for path, data in zip(files, _datasets(plan), strict=True):
        path.write_bytes(data)
    support = Path(__file__).with_name("dense_pointwise_weights_oracle.cpp").read_text()
    oracle = tmp_path / "oracle.cpp"
    oracle.write_text(
        "#define DP_PROBE_ONLY\n" + support + "\n#undef DP_PROBE_ONLY\n" + code + "\n#undef TLOAD\n" + support
    )
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
    assert cpu and native
    binary = tmp_path / "oracle"
    definitions = {"ROWS": rows, "CHANNELS": channels, "OUTPUTS": outputs, "COLS": columns, "FIRST": first}
    subprocess.run(
        [
            cpu,
            "-std=c++20",
            "-O1",
            "-ffp-contract=off",
            "-D__CPU_SIM",
            "-D__DAV_CUBE__",
            "-D__DAV_VEC__",
            *[f"-DDW_{key}={value}" for key, value in definitions.items()],
            *includes,
            str(oracle),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = subprocess.run(
        [str(binary), *map(str, files)],
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PTO_CPU_SIM_NUM_CORES": "1",
            "PTO_CPU_SIM_TRACE_ENABLE": "0",
            "OMP_NUM_THREADS": "1",
            "PTO_CPU_SIM_L1_BYTES": "1048576",
            "PTO_CPU_SIM_UB_BYTES": "131072",
            "PTO_CPU_SIM_L0C_BYTES": "262144",
        },
    )
    expected = [2 * rows * channels * 2, 4 * phase_rows * columns * 2, 2 * ((channels + 63) // 64), 4]
    assert [list(map(int, line.split())) for line in result.stdout.splitlines()] == [expected] * 3
    (tmp_path / "payload.json").write_text(json.dumps({"datasets": 3, "observations": [expected] * 3}) + "\n")
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


@pytest.mark.parametrize("columns", [0, -16, 1, 15, 80, True, 16.0])
def test_dense_weight_cache_rejects_invalid_columns(columns):
    with pytest.raises(ValueError, match="weight cache requires"):
        dense_pointwise_weight_cache(_plan(64, 64), columns)


@pytest.mark.parametrize(
    "source",
    [
        torch.empty(16, 64),
        torch.zeros(16, 64, 1, 1, dtype=torch.int32),
        torch.empty(16, 64, 1, 1, device="meta"),
    ],
)
def test_dense_weight_packer_rejects_invalid_storage(source):
    with pytest.raises(ValueError, match="weights require CPU floating"):
        pack_dense_pointwise_weights(_plan(64, 16), source)


def test_dense_weights_reject_unordered_geometry():
    plan = replace(_plan(64, 16), four_way_accumulation=False)
    with pytest.raises(ValueError, match="dense pointwise images require"):
        pack_dense_pointwise_weights(plan, torch.zeros(16, 64, 1, 1))
    with pytest.raises(ValueError, match="dense pointwise images require"):
        dense_pointwise_weight_cache(plan)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
