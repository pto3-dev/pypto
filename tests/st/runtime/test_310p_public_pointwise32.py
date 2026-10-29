# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Strict 310P comparison of public N32 pointwise against native and CPU."""

import pytest
import torch
from pypto import ir
from pypto.backend import BackendType
from pypto.runtime import A1ResidentSession
from yolo26_native_test_utils import require_310p

from examples.models.yolo26_native.operators import ConvPlan
from examples.models.yolo26_native.pointwise import PublicOrderedPointwise32Plan


@pytest.mark.parametrize("blocks", [1, 8])
def test_310p_public_pointwise32(tmp_path, blocks):
    require_310p()
    native = ConvPlan(160, 160, 32, 32, 1, 1, 0, True, True)
    public = PublicOrderedPointwise32Plan(160, 160, 32, 32, 1, 1, 0, True, True)
    plans = {"native": native, "public": public}
    compiled = {
        name: ir.compile(
            plan.jit_kernel().specialize(),
            backend_type=BackendType.Ascend310P,
            output_dir=str(tmp_path / name),
            dump_passes=False,
        )
        for name, plan in plans.items()
    }
    assert not public.jit_kernel()._external_source_paths()
    with A1ResidentSession(compiled["native"]) as session:
        kernels = {name: session.prepare(program) for name, program in compiled.items()}
        source = session.empty((native.input_size,), torch.float16)
        weights = {name: session.empty((plan.weight_size,), torch.float16) for name, plan in plans.items()}
        bias_device = session.empty((32,), torch.float32)
        metadata = session.upload(native.metadata())
        zero = session.upload(torch.zeros(16, dtype=torch.float16))
        guards = {
            name: session.upload(torch.full((native.output_size + 32,), 7.0, dtype=torch.float16))
            for name in plans
        }
        outputs = {name: guard.view((native.output_size,), 16) for name, guard in guards.items()}
        for seed in (1777, 1783):
            generator = torch.Generator().manual_seed(seed)
            x = (torch.randn((25600, 32), generator=generator) * 0.25).half()
            weight = (torch.randn((32, 32), generator=generator) * 0.125).half()
            bias = (torch.randn(32, generator=generator) * 0.1).half().float()
            source.copy_from(x.flatten())
            bias_device.copy_from(bias)
            for name, plan in plans.items():
                weights[name].copy_from(plan.pack_weight(weight[:, :, None, None]))
                outputs[name].copy_from(torch.full((native.output_size,), float("nan"), dtype=torch.float16))
                kernels[name](
                    metadata, source, weights[name], bias_device, outputs[name], zero, blocks=blocks
                )
            result = outputs["public"].to_cpu()
            assert torch.equal(result.view(torch.int16), outputs["native"].to_cpu().view(torch.int16))
            partial = torch.zeros((4, 25600, 32), dtype=torch.float32)
            for channel in range(32):
                partial[channel % 4] += x[:, channel].float()[:, None] * weight[:, channel].float()[None, :]
            rounded = (((partial[0] + partial[1]) + partial[2]) + partial[3] + bias).half()
            expected = rounded * rounded.sigmoid()
            assert torch.equal(result.view(torch.int16), expected.flatten().view(torch.int16))
            assert torch.equal(source.to_cpu(), x.flatten())
            for guard in guards.values():
                whole = guard.to_cpu()
                assert torch.all(whole[:16] == 7) and torch.all(whole[-16:] == 7)
