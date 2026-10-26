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


@pytest.mark.parametrize("channels", [32, 64, 80, 128, 256])
@pytest.mark.parametrize("four_way", [False, True])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("silu", [False, True])
def test_310p_native_depthwise(tmp_path, stride, silu, four_way, channels):
    require_310p()
    plan = load_plan("depthwise").DepthwisePlan(8, 8, channels, stride, silu, four_way)
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
                expected = expected * torch.sigmoid(expected)
            dx.copy_from(x[0].permute(1, 2, 0).contiguous().reshape(-1))
            dw.copy_from(plan.pack_weight(w))
            db.copy_from(bias.float())
            kernel(meta, dx, dw, db, output, blocks=8)
            actual = output.to_cpu().reshape(plan.oh, plan.ow, plan.channels).permute(2, 0, 1).unsqueeze(0)
            if four_way and not silu:
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            else:
                torch.testing.assert_close(actual, expected, rtol=0.003, atol=0.002)
            full = guard.to_cpu()
            assert torch.equal(full[:16], torch.full((16,), 7.0, dtype=torch.float16))
            assert torch.equal(full[-16:], torch.full((16,), 7.0, dtype=torch.float16))


def test_depthwise_rejects_unaligned_channels():
    with pytest.raises(ValueError, match="sixteen"):
        load_plan("depthwise").DepthwisePlan(8, 8, 17)


@pytest.mark.parametrize("channels", [16, 64, 80, 128, 256])
def test_310p_depthwise_four_way_preserves_cancellation(tmp_path, channels):
    require_310p()
    plan = load_plan("depthwise").DepthwisePlan(4, 4, channels, silu=False, four_way_accumulation=True)
    compiled = ir.compile(
        plan.jit_kernel().specialize(),
        backend_type=BackendType.Ascend310P,
        output_dir=str(tmp_path),
        dump_passes=False,
    )
    x = torch.full((1, channels, 4, 4), 4096.0, dtype=torch.float16)
    weight = torch.zeros((channels, 1, 3, 3), dtype=torch.float16)
    weight[:, 0, 0, 0] = 4096.0
    weight[:, 0, 0, 1] = 2.0**-12
    weight[:, 0, 1, 1] = -4096.0
    expected = F.conv2d(x.double(), weight.double(), padding=1, groups=channels).half()
    assert torch.all(expected[:, :, 1:-1, 1:-1] == 1)
    output = torch.empty(plan.output_size, dtype=torch.float16)
    compiled(
        plan.metadata(),
        x[0].permute(1, 2, 0).contiguous().reshape(-1),
        plan.pack_weight(weight),
        torch.zeros(channels),
        output,
    )
    actual = output.reshape(4, 4, channels).permute(2, 0, 1).unsqueeze(0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "shape", [(80, 80, 64, 1), (80, 80, 80, 1), (41, 37, 128, 2), (21, 19, 256, 1), (2, 288, 80, 1)]
)
@pytest.mark.parametrize("blocks", [1, 8])
@pytest.mark.parametrize("silu", [False, True])
def test_310p_depthwise_resident_rows_exact(tmp_path, shape, blocks, silu):
    """Keep row-ring wrap, stripe halos, stride-two and oversized fallbacks exact."""
    require_310p()
    h, w, channels, stride = shape
    plan = load_plan("depthwise").DepthwisePlan(h, w, channels, stride, silu, True)
    g = torch.Generator().manual_seed(331)
    x = (torch.randn((1, channels, h, w), generator=g) * 0.25).half()
    weight = (torch.randn((channels, 1, 3, 3), generator=g) * 0.125).half()
    bias = (torch.randn(channels, generator=g) * 0.1).half()
    # Explicit four-part FP32 reduction preserves the kernel's arithmetic
    # contract independently of CPU convolution's shape-dependent algorithms.
    padded = F.pad(x.float(), (1, 1, 1, 1))
    sums = [torch.zeros((1, channels, plan.oh, plan.ow)) for _ in range(4)]
    for pos in range(9):
        ky, kx = divmod(pos, 3)
        patch = padded[:, :, ky : ky + plan.oh * stride : stride, kx : kx + plan.ow * stride : stride]
        sums[pos % 4] += patch * weight[:, 0, ky, kx].float().view(1, channels, 1, 1)
    expected = (((sums[0] + sums[1]) + sums[2]) + sums[3] + bias.float().view(1, channels, 1, 1)).half()
    if silu:
        expected = expected * torch.sigmoid(expected)
    compiled = ir.compile(
        plan.jit_kernel().specialize(),
        backend_type=BackendType.Ascend310P,
        output_dir=str(tmp_path),
        dump_passes=False,
    )
    with A1ResidentSession(compiled) as session:
        kernel = session.prepare(ir.CompiledProgram.from_dir(compiled.output_dir))
        guard = session.upload(torch.full((plan.output_size + 32,), 7.0, dtype=torch.float16))
        output = guard.view((plan.output_size,), 16)
        arguments = [
            session.upload(t)
            for t in (
                plan.metadata(),
                x[0].permute(1, 2, 0).contiguous().reshape(-1),
                plan.pack_weight(weight),
                bias.float(),
            )
        ]
        kernel(*arguments, output, blocks=blocks)
        actual = output.to_cpu().reshape(plan.oh, plan.ow, channels).permute(2, 0, 1).unsqueeze(0)
        assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
        full = guard.to_cpu()
        assert torch.all(full[:16] == 7) and torch.all(full[-16:] == 7)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
