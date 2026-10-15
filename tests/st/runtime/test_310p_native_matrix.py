# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Native resident QK and probability/value matrix products."""

import pytest
import torch
import torch.nn.functional as F
from pypto import ir
from pypto.backend import BackendType
from pypto.runtime import A1ResidentSession
from yolo26_native_test_utils import load_plan, require_310p


@pytest.mark.parametrize(
    "shape,transpose_b",
    [((16, 16, 32), True), ((400, 400, 32), True), ((16, 64, 16), False), ((400, 64, 400), False)],
)
def test_310p_native_matrix(tmp_path, shape, transpose_b):
    require_310p()
    plan = load_plan("matrix").MatrixPlan(*shape, transpose_b=transpose_b)
    compiled = ir.compile(
        plan.jit_kernel().specialize(),
        backend_type=BackendType.Ascend310P,
        output_dir=str(tmp_path),
        dump_passes=False,
    )
    with A1ResidentSession(compiled) as session:
        kernel = session.prepare(ir.CompiledProgram.from_dir(compiled.output_dir))
        da = session.empty((plan.m * plan.k,), torch.float16)
        db = session.empty((plan.n * plan.k,), torch.float16)
        guard = session.upload(torch.full((plan.m * plan.n + 32,), 7.0, dtype=torch.float16))
        output = guard.view((plan.m * plan.n,), 16)
        metadata = session.upload(plan.metadata())
        for seed in (277, 281):
            g = torch.Generator().manual_seed(seed)
            a = (torch.randn((plan.m, plan.k), generator=g) * 0.25).half()
            b = (
                torch.randn((plan.n, plan.k) if transpose_b else (plan.k, plan.n), generator=g) * 0.25
            ).half()
            expected = a @ (b.T if transpose_b else b)
            da.copy_from(a.reshape(-1))
            db.copy_from(b.reshape(-1))
            kernel(metadata, da, db, output, blocks=8)
            torch.testing.assert_close(
                output.to_cpu().reshape(plan.m, plan.n), expected, rtol=0.003, atol=0.002
            )
            full = guard.to_cpu()
            assert torch.equal(full[:16], torch.full((16,), 7.0, dtype=torch.float16))
            assert torch.equal(full[-16:], torch.full((16,), 7.0, dtype=torch.float16))


@pytest.mark.parametrize(
    "shape,transpose_b", [((16, 16, 32), True), ((32, 32, 48), True), ((16, 32, 32), False)]
)
def test_310p_linear_matrix_exact_batched_golden(tmp_path, shape, transpose_b):
    require_310p()
    plan = load_plan("matrix").MatrixPlan(*shape, transpose_b=transpose_b, linear_accumulation=True)
    compiled = ir.compile(
        plan.jit_kernel().specialize(),
        backend_type=BackendType.Ascend310P,
        output_dir=str(tmp_path),
        dump_passes=False,
    )
    with A1ResidentSession(compiled) as session:
        kernel = session.prepare(ir.CompiledProgram.from_dir(compiled.output_dir))
        da = session.empty((plan.m * plan.k,), torch.float16)
        db = session.empty((plan.n * plan.k,), torch.float16)
        guard = session.upload(torch.full((plan.m * plan.n + 32,), 7.0, dtype=torch.float16))
        output = guard.view((plan.m * plan.n,), 16)
        metadata = session.upload(plan.metadata())
        for seed in (347, 349):
            g = torch.Generator().manual_seed(seed)
            a = (torch.randn((plan.m, plan.k), generator=g) * 0.25).half()
            b = (
                torch.randn((plan.n, plan.k) if transpose_b else (plan.k, plan.n), generator=g) * 0.25
            ).half()
            # Preserve Q's transposed CPU layout; 2D mm can choose another reduction.
            cpu_a = a.T.contiguous().T
            cpu_b = b.T if transpose_b else b
            if transpose_b:
                cpu_b = cpu_b.contiguous()
            expected = (cpu_a[None, None] @ cpu_b[None, None])[0, 0]
            da.copy_from(a.reshape(-1))
            db.copy_from(b.reshape(-1))
            kernel(metadata, da, db, output, blocks=8)
            torch.testing.assert_close(output.to_cpu().reshape(plan.m, plan.n), expected, rtol=0, atol=0)
            full = guard.to_cpu()
            assert torch.equal(full[:16], torch.full((16,), 7.0, dtype=torch.float16))
            assert torch.equal(full[-16:], torch.full((16,), 7.0, dtype=torch.float16))


def test_matrix_rejects_unaligned_shapes():
    with pytest.raises(ValueError, match="sixteen"):
        load_plan("matrix").MatrixPlan(16, 17, 32)


def test_310p_native_convolution_accumulator_reuse(tmp_path):
    require_310p()
    plan = load_plan("operators").ConvPlan(16, 16, 16, 16, kernel=1, padding=0, silu=False)
    compiled = ir.compile(
        plan.jit_kernel().specialize(),
        backend_type=BackendType.Ascend310P,
        output_dir=str(tmp_path),
        dump_passes=False,
    )
    with A1ResidentSession(compiled) as session:
        kernel = session.prepare(compiled)
        dx = session.empty((plan.input_size,), torch.float16)
        dw = session.empty((plan.weight_size,), torch.float16)
        db = session.upload(torch.zeros(plan.co))
        output = session.empty((plan.output_size,), torch.float16)
        zero = session.upload(torch.zeros(16, dtype=torch.float16))
        metadata = session.upload(plan.metadata())
        for seed in (331, 337):
            g = torch.Generator().manual_seed(seed)
            x = (torch.randn((1, 16, 16, 16), generator=g) * 0.25).half()
            w = (torch.randn((16, 16, 1, 1), generator=g) * 0.125).half()
            dx.copy_from(x[0].permute(1, 2, 0).contiguous().reshape(-1))
            dw.copy_from(plan.pack_weight(w))
            kernel(metadata, dx, dw, db, output, zero, blocks=8)
            actual = output.to_cpu().reshape(16, 16, 16).permute(2, 0, 1).unsqueeze(0)
            torch.testing.assert_close(actual, F.conv2d(x, w), rtol=0.003, atol=0.002)


@pytest.mark.parametrize("case", ["random", "wide", "midpoint"])
def test_310p_softmax16_compensated_fp16(tmp_path, case):
    require_310p()
    plan = load_plan("softmax").Softmax16Plan()
    compiled = ir.compile(
        plan.jit_kernel().specialize(),
        backend_type=BackendType.Ascend310P,
        output_dir=str(tmp_path),
        dump_passes=False,
    )
    # This row's exact probability at column five lies just below an FP16 midpoint.
    boundary = torch.tensor(
        [
            7.1171875,
            6.703125,
            6.3671875,
            7.14453125,
            6.48828125,
            6.11328125,
            6.26953125,
            6.63671875,
            6.02734375,
            4.9375,
            5.1328125,
            6.02734375,
            4.85546875,
            4.16796875,
            4.3984375,
            5.21484375,
        ],
        dtype=torch.float16,
    )
    with A1ResidentSession(compiled) as session:
        kernel = session.prepare(ir.CompiledProgram.from_dir(compiled.output_dir))
        metadata = session.upload(plan.metadata())
        dx = session.empty((256,), torch.float16)
        guard = session.upload(torch.full((288,), 7.0, dtype=torch.float16))
        output = guard.view((256,), 16)
        for seed in (353, 359):
            g = torch.Generator().manual_seed(seed)
            x = (torch.randn((16, 16), generator=g) * 3).half()
            if case == "midpoint":
                x = boundary[torch.randperm(16, generator=g)].repeat(16, 1)
            elif case == "wide":
                x = (x * 128).half()
                x[0].zero_()
                x[1, 0] = 65504
                x[1, 1:] = -65504
            expected = torch.softmax(x.double(), -1).half()
            dx.copy_from(x.reshape(-1))
            kernel(metadata, dx, output, blocks=8)
            actual = output.to_cpu().reshape(16, 16)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert torch.isfinite(actual).all()
            full = guard.to_cpu()
            assert torch.equal(full[:16], torch.full((16,), 7.0, dtype=torch.float16))
            assert torch.equal(full[-16:], torch.full((16,), 7.0, dtype=torch.float16))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
