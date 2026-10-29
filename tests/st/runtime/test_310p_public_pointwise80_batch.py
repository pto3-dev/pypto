# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Check full writes on independent single-core batch graph replays."""

import pytest
import torch
from pypto import ir
from pypto.backend import BackendType
from pypto.runtime import A1L2Worker
from yolo26_native_test_utils import require_310p

from examples.models.yolo26_native.operators import ConvPlan
from examples.models.yolo26_native.pointwise import PublicOrderedPointwise32Plan


@pytest.mark.parametrize("source_stride,source_offset", [(0, 0), (64, 0), (64, 32)])
def test_310p_public_pointwise80_independent_batch(tmp_path, source_stride, source_offset):
    require_310p()
    native = ConvPlan(80, 80, 32, 32, 1, 1, 0, True, True, source_stride, source_offset)
    public = PublicOrderedPointwise32Plan(80, 80, 32, 32, 1, 1, 0, True, True, source_stride, source_offset)
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
    generator = torch.Generator().manual_seed(1801)
    weight = (torch.randn((32, 32), generator=generator) * 0.125).half()
    bias = (torch.randn(32, generator=generator) * 0.1).half().float()
    with A1L2Worker(compiled["native"]) as worker:
        inputs = [worker.empty((native.input_size,), torch.float16) for _ in range(8)]
        weights = {
            name: worker.upload(plan.pack_weight(weight[:, :, None, None])) for name, plan in plans.items()
        }
        bias_device = worker.upload(bias)
        metadata = worker.upload(native.metadata())
        zero = worker.upload(torch.zeros(16, dtype=torch.float16))
        guards = [
            worker.upload(torch.full((native.output_size + 32,), 7.0, dtype=torch.float16)) for _ in range(8)
        ]
        outputs = [guard.view((native.output_size,), 16) for guard in guards]
        graphs = {}
        for name, program in compiled.items():
            kernel = worker.prepare_independent(program, 8)
            args = tuple(
                value
                for i in range(8)
                for value in (metadata, inputs[i], weights[name], bias_device, outputs[i], zero)
            )
            graphs[name] = worker.register_graph([(kernel, args, 8)])
        for seed in (1811, 1823):
            generator = torch.Generator().manual_seed(seed)
            source = [
                (torch.randn((native.input_size,), generator=generator) * 0.25).half() for _ in range(8)
            ]
            for device, data in zip(inputs, source, strict=True):
                device.copy_from(data)
            reference = None
            for name in ("native", "public"):
                for output in outputs:
                    output.copy_from(torch.full((native.output_size,), float("nan"), dtype=torch.float16))
                graphs[name]()
                actual = [value.to_cpu() for value in outputs]
                if reference is None:
                    reference = actual
                else:
                    for expected, result in zip(reference, actual, strict=True):
                        assert torch.equal(result.view(torch.int16), expected.view(torch.int16))
                for guard in guards:
                    whole = guard.to_cpu()
                    assert torch.all(whole[:16] == 7) and torch.all(whole[-16:] == 7)
            for device, data in zip(inputs, source, strict=True):
                assert torch.equal(device.to_cpu(), data)
