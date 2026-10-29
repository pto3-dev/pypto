# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Public 310P depthwise convolution with exact four-way FP32 accumulation."""

import pytest
import torch
import torch.nn.functional as F
from pypto import ir
from pypto.backend import BackendType
from pypto.runtime import A1ResidentSession
from yolo26_native_test_utils import load_plan, require_310p


def test_public_depthwise_selection():
    plan = load_plan("depthwise").DepthwisePlan(80, 80, 80, 1, True, True)
    assert not plan.jit_kernel()._external_source_paths()
    for geometry in [
        (80, 80, 80, 1, False, True),
        (80, 80, 80, 1, True, False),
        (80, 80, 80, 2, True, True),
        (40, 40, 80, 1, True, True),
    ]:
        assert load_plan("depthwise").DepthwisePlan(*geometry).jit_kernel()._external_source_paths()


@pytest.mark.parametrize("blocks", [1, 8])
def test_public_depthwise_four_way_exact(tmp_path, blocks):
    require_310p()
    plan = load_plan("depthwise").DepthwisePlan(80, 80, 80, 1, True, True)
    compiled = ir.compile(
        plan.jit_kernel().specialize(), backend_type=BackendType.Ascend310P, output_dir=str(tmp_path)
    )
    with A1ResidentSession(compiled) as session:
        kernel = session.prepare(compiled)
        meta = session.upload(plan.metadata())
        xdev = session.empty((plan.input_size,), torch.float16)
        wdev = session.empty((720,), torch.float16)
        bdev = session.empty((80,), torch.float32)
        guard = session.upload(torch.full((plan.output_size + 32,), 7.0, dtype=torch.float16))
        output = guard.view((plan.output_size,), 16)
        for seed in (3271, 3277):
            generator = torch.Generator().manual_seed(seed)
            x = (torch.randn((1, 80, 80, 80), generator=generator) * 0.25).half()
            weight = (torch.randn((80, 1, 3, 3), generator=generator) * 0.125).half()
            bias = (torch.randn(80, generator=generator) * 0.1).half()
            padded = F.pad(x.float(), (1, 1, 1, 1))
            sums = [torch.zeros((1, 80, 80, 80)) for _ in range(4)]
            for pos in range(9):
                ky, kx = divmod(pos, 3)
                sums[pos % 4] += padded[:, :, ky : ky + 80, kx : kx + 80] * weight[:, 0, ky, kx].float().view(
                    1, 80, 1, 1
                )
            expected = (((sums[0] + sums[1]) + sums[2]) + sums[3] + bias.float().view(1, 80, 1, 1)).half()
            expected = expected * expected.sigmoid()
            xdev.copy_from(x[0].permute(1, 2, 0).contiguous().reshape(-1))
            wdev.copy_from(plan.pack_weight(weight))
            bdev.copy_from(bias.float())
            output.copy_from(torch.full((plan.output_size,), float("nan"), dtype=torch.float16))
            kernel(meta, xdev, wdev, bdev, output, blocks=blocks)
            actual = output.to_cpu().reshape(80, 80, 80).permute(2, 0, 1).unsqueeze(0)
            assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
            full = guard.to_cpu()
            assert torch.all(full[:16] == 7) and torch.all(full[-16:] == 7)
