# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Exercise public 310P two-stage Mat loads and GEMM with a tail."""

import re

import pypto.language as pl
import pytest
import torch
from pypto import ir
from pypto.backend import BackendType
from pypto.runtime import A1ResidentSession
from yolo26_native_test_utils import require_310p


@pytest.mark.parametrize("tiles", [1, 5, 6])
def test_310p_public_mat_pipeline(tmp_path, tiles):
    require_310p()
    rows = tiles * 16

    @pl.jit.incore
    def compute(
        a: pl.Tensor[[rows, 16], pl.FP16],
        b: pl.Tensor[[16, 16], pl.FP16],
        out: pl.Out[pl.Tensor[[rows, 16], pl.FP32]],
    ):
        weight = pl.load(b, [0, 0], [16, 16], target_memory=pl.Mem.Mat)
        for i in pl.pipeline(0, tiles, stage=2):
            source = pl.load(a, [i * 16, 0], [16, 16], target_memory=pl.Mem.Mat)
            product = pl.matmul(source, weight)
            out = pl.store(product, [i * 16, 0], out)
        return out

    compiled = ir.compile(
        compute.specialize(), backend_type=BackendType.Ascend310P, output_dir=str(tmp_path), dump_passes=False
    )
    if tiles == 5:
        pto = next(tmp_path.rglob("*.pto")).read_text()
        mat_addresses = set(re.findall(r"pto\.alloc_tile addr = (%\w+).*loc=mat", pto))
        assert {"%c512_i64", "%c1024_i64"} <= mat_addresses
        cpp = next((tmp_path / "kernels").rglob("*.cpp")).read_text()
        assert cpp[: cpp.index("TMOV(")].count("TLOAD(") >= 3
    with A1ResidentSession(compiled) as session:
        a_device = session.empty((rows, 16), torch.float16)
        b_device = session.empty((16, 16), torch.float16)
        guard = session.upload(torch.full((rows * 16 + 32,), -7.0, dtype=torch.float32))
        output = guard.view((rows, 16), 16)
        kernel = session.prepare(compiled)
        for seed in (1637, 1657):
            generator = torch.Generator().manual_seed(seed)
            a = (torch.randint(-8, 9, (rows, 16), generator=generator).float() / 16).half()
            b = (torch.randint(-8, 9, (16, 16), generator=generator).float() / 16).half()
            a_device.copy_from(a)
            b_device.copy_from(b)
            kernel(a_device, b_device, output, blocks=1)
            assert torch.equal(output.to_cpu(), a.float() @ b.float())
            assert torch.equal(a_device.to_cpu(), a)
            assert torch.equal(b_device.to_cpu(), b)
            whole = guard.to_cpu()
            assert torch.all(whole[:16] == -7) and torch.all(whole[-16:] == -7)
