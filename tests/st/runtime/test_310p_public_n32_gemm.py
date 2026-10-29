# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
# EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Verify public N32 matmul and accumulator panel writes on Ascend 310P."""

import pypto.language as pl
import torch
from pypto import ir
from pypto.backend import BackendType
from pypto.runtime import A1ResidentSession
from yolo26_native_test_utils import require_310p


def test_310p_public_n32_gemm_panels(tmp_path):
    require_310p()

    @pl.jit.incore
    def gemm(
        a: pl.Tensor[[32, 64], pl.FP16],
        b: pl.Tensor[[64, 32], pl.FP16],
        out: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
    ):
        ma = pl.load(a, [0, 0], [32, 64], target_memory=pl.Mem.Mat)
        mb = pl.load(b, [0, 0], [64, 32], target_memory=pl.Mem.Mat)
        acc = pl.matmul(ma, mb)
        lo = pl.tile.slice(acc, [32, 16], [0, 0])
        hi = pl.tile.slice(acc, [32, 16], [0, 16])
        out = pl.store(lo, [0, 0], out)
        out = pl.store(hi, [0, 16], out)
        return out

    compiled = ir.compile(
        gemm.specialize(), backend_type=BackendType.Ascend310P, output_dir=str(tmp_path), dump_passes=False
    )
    with A1ResidentSession(compiled) as session:
        a_dev = session.empty((32, 64), torch.float16)
        b_dev = session.empty((64, 32), torch.float16)
        guard = session.upload(torch.full((1024 + 32,), -7.0, dtype=torch.float32))
        output = guard.view((32, 32), 16)
        kernel = session.prepare(compiled)
        for seed in (1601, 1607):
            generator = torch.Generator().manual_seed(seed)
            a = (torch.randint(-8, 9, (32, 64), generator=generator).float() / 16).half()
            b = (torch.randint(-8, 9, (64, 32), generator=generator).float() / 16).half()
            a_dev.copy_from(a)
            b_dev.copy_from(b)
            kernel(a_dev, b_dev, output, blocks=1)
            assert torch.equal(output.to_cpu(), a.float() @ b.float())
            assert torch.equal(a_dev.to_cpu(), a)
            assert torch.equal(b_dev.to_cpu(), b)
            whole = guard.to_cpu()
            assert torch.all(whole[:16] == -7) and torch.all(whole[-16:] == -7)
