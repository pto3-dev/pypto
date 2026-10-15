# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Native resident depthwise convolution, bias and SiLU."""

import pytest
import torch
import torch.nn.functional as F
from pypto import ir
from pypto.backend import BackendType
from pypto.runtime import A1ResidentSession
from yolo26_native_test_utils import load_plan, require_310p


@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("silu", [False, True])
def test_310p_native_depthwise(tmp_path, stride, silu):
    require_310p()
    plan = load_plan("depthwise").DepthwisePlan(8, 8, 32, stride, silu)
    compiled = ir.compile(
        plan.jit_kernel().specialize(),
        backend_type=BackendType.Ascend310P,
        output_dir=str(tmp_path),
        dump_passes=False,
    )
    with A1ResidentSession(compiled) as session:
        kernel = session.prepare(ir.CompiledProgram.from_dir(compiled.output_dir))
        dx = session.empty((plan.input_size,), torch.float16)
        dw = session.empty((plan.channels * 9,), torch.float16)
        db = session.empty((plan.channels,), torch.float32)
        guard = session.upload(torch.full((plan.output_size + 32,), 7.0, dtype=torch.float16))
        output = guard.view((plan.output_size,), 16)
        meta = session.upload(plan.metadata())
        for seed in (269, 271):
            g = torch.Generator().manual_seed(seed)
            x = (torch.randn((1, plan.channels, plan.height, plan.width), generator=g) * 0.25).half()
            w = (torch.randn((plan.channels, 1, 3, 3), generator=g) * 0.125).half()
            bias = (torch.randn(plan.channels, generator=g) * 0.1).half()
            expected = F.conv2d(x, w, bias, padding=1, stride=stride, groups=plan.channels)
            if silu:
                expected = F.silu(expected)
            dx.copy_from(x[0].permute(1, 2, 0).contiguous().reshape(-1))
            dw.copy_from(plan.pack_weight(w))
            db.copy_from(bias.float())
            kernel(meta, dx, dw, db, output, blocks=8)
            actual = output.to_cpu().reshape(plan.oh, plan.ow, plan.channels).permute(2, 0, 1).unsqueeze(0)
            torch.testing.assert_close(actual, expected, rtol=0.003, atol=0.002)
            full = guard.to_cpu()
            assert torch.equal(full[:16], torch.full((16,), 7.0, dtype=torch.float16))
            assert torch.equal(full[-16:], torch.full((16,), 7.0, dtype=torch.float16))


def test_depthwise_rejects_unaligned_channels():
    with pytest.raises(ValueError, match="sixteen"):
        load_plan("depthwise").DepthwisePlan(8, 8, 17)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
