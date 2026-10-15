# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Native resident input packing, routing, residuals, pooling and casts."""

import pytest
import torch
import torch.nn.functional as F
from pypto import ir
from pypto.backend import BackendType
from pypto.runtime import A1ResidentSession
from yolo26_native_test_utils import load_plan, require_310p


def _plan(name):
    module = load_plan("layout")
    op = module.VectorOp
    constructors = {
        "pack": lambda: module.VectorPlan(op.PACK, 8, 8, channels=3),
        "copy": lambda: module.VectorPlan(op.COPY, 4, 4, 16, 48, 64, 16, 32),
        "upsample": lambda: module.VectorPlan(op.UPSAMPLE, 8, 8, 32, 32, 32, input_height=4, input_width=4),
        "add": lambda: module.VectorPlan(op.ADD, 4, 4, 16, 48, 32, 16, 16, second_stride=32),
        "maxpool": lambda: module.VectorPlan(op.MAXPOOL, 4, 4, 32, 32, 32),
        "float_to_half": lambda: module.VectorPlan(op.FLOAT_TO_HALF, 512),
        "half_to_float": lambda: module.VectorPlan(op.HALF_TO_FLOAT, 160000, scale=0.125),
        "scale_half": lambda: module.VectorPlan(op.SCALE_HALF, 512, scale=32**-0.5),
    }
    return constructors[name]()


def _expected(plan, name, x, b):
    result = torch.full(
        (plan.output_size,), 7.0, dtype=torch.float32 if name == "half_to_float" else torch.float16
    )
    if name == "pack":
        result = result.reshape(plan.height, plan.width, 16)
        result.zero_()
        result[:, :, : plan.channels] = (
            x.reshape(plan.channels, plan.height, plan.width).permute(1, 2, 0).half()
        )
    elif name in ("float_to_half", "half_to_float", "scale_half"):
        result = x.half() if name == "float_to_half" else (x.float() * plan.scale).to(result.dtype)
    else:
        source = x.reshape(-1, plan.source_stride)[:, plan.source_offset : plan.source_offset + plan.channels]
        if name == "upsample":
            source = (
                source.reshape(plan.input_height, plan.input_width, plan.channels)
                .repeat_interleave(2, 0)
                .repeat_interleave(2, 1)
                .reshape(-1, plan.channels)
            )
        elif name == "add":
            source = (
                source
                + b.reshape(-1, plan.second_stride)[
                    :, plan.second_offset : plan.second_offset + plan.channels
                ]
            )
        elif name == "maxpool":
            source = (
                F.max_pool2d(
                    source.reshape(plan.height, plan.width, plan.channels).permute(2, 0, 1).unsqueeze(0),
                    5,
                    1,
                    2,
                )[0]
                .permute(1, 2, 0)
                .reshape(-1, plan.channels)
            )
        result.reshape(-1, plan.destination_stride)[
            :, plan.destination_offset : plan.destination_offset + plan.channels
        ] = source
    return result.reshape(-1)


@pytest.mark.parametrize(
    "name", ["pack", "copy", "upsample", "add", "maxpool", "float_to_half", "half_to_float", "scale_half"]
)
def test_310p_native_layout(tmp_path, name):
    require_310p()
    plan = _plan(name)
    compiled = ir.compile(
        plan.jit_kernel().specialize(),
        backend_type=BackendType.Ascend310P,
        output_dir=str(tmp_path),
        dump_passes=False,
    )
    dtype = torch.float32 if name in ("pack", "float_to_half") else torch.float16
    output_dtype = torch.float32 if name == "half_to_float" else torch.float16
    with A1ResidentSession(compiled) as session:
        kernel = session.prepare(ir.CompiledProgram.from_dir(compiled.output_dir))
        dx = session.empty((plan.input_size,), dtype)
        db = session.empty((plan.second_size,), torch.float16)
        guard = session.upload(torch.full((plan.output_size + 32,), 7.0, dtype=output_dtype))
        out = guard.view((plan.output_size,), 16)
        metadata = session.upload(plan.metadata())
        for seed in (257, 263):
            g = torch.Generator().manual_seed(seed)
            x = torch.randn(plan.input_size, generator=g).to(dtype)
            b = torch.randn(plan.second_size, generator=g).half()
            dx.copy_from(x)
            db.copy_from(b)
            out.copy_from(torch.full((plan.output_size,), 7.0, dtype=output_dtype))
            kernel(metadata, dx, db, out, blocks=8)
            torch.testing.assert_close(out.to_cpu(), _expected(plan, name, x, b), rtol=0, atol=0)
            actual = guard.to_cpu()
            assert torch.equal(actual[:16], torch.full((16,), 7.0, dtype=output_dtype))
            assert torch.equal(actual[-16:], torch.full((16,), 7.0, dtype=output_dtype))


def test_layout_rejects_invalid_slices_and_upsampling():
    module = load_plan("layout")
    with pytest.raises(ValueError, match="slice"):
        module.VectorPlan(module.VectorOp.COPY, 4, 4, 16, 16, 16, 16)
    with pytest.raises(ValueError, match="twice"):
        module.VectorPlan(module.VectorOp.UPSAMPLE, 8, 8, input_height=3, input_width=4)
    with pytest.raises(ValueError, match="256"):
        module.VectorPlan(module.VectorOp.FLOAT_TO_HALF, 255)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
