# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""PTO3 external convolution: native gathering, cube accumulation and SiLU."""

import ctypes
import importlib.util
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from pypto import ir
from pypto.backend import BackendType
from pypto.runtime import A1ResidentSession
from yolo26_native_test_utils import load_plan, require_310p

from examples.models.yolo26_native.executor import _runtime_l2_cache_offset


def _operators():
    path = Path(__file__).resolve().parents[3] / "examples/models/yolo26_native/operators.py"
    spec = importlib.util.spec_from_file_location("yolo26_native_operators", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _pack_input(plan, x):
    packed = torch.full((plan.height, plan.width, plan.input_row_stride), float("nan"), dtype=torch.float16)
    packed[:, :, plan.source_offset : plan.source_offset + plan.ci] = 0
    packed[:, :, plan.source_offset : plan.source_offset + plan.cin] = x[0].permute(1, 2, 0)
    return packed


@pytest.mark.parametrize(
    "shape,source_layout",
    [
        ((8, 8, 3, 16, 3, 2, 1), (0, 0)),
        ((8, 8, 32, 32, 3, 1, 1), (0, 0)),
        ((4, 4, 16, 16, 1, 1, 0), (0, 0)),
        ((4, 4, 64, 4, 1, 1, 0), (0, 0)),
        ((8, 8, 32, 32, 3, 1, 1), (64, 16)),
        ((4, 4, 16, 16, 1, 1, 0), (48, 16)),
        ((8, 8, 64, 128, 3, 1, 1), (96, 16)),
        ((16, 16, 64, 128, 3, 2, 1), (96, 16)),
        ((8, 8, 512, 113, 3, 1, 1), (544, 16)),
        ((16, 16, 64, 64, 3, 2, 1), (96, 16)),
        ((8, 8, 128, 16, 3, 1, 1), (160, 16)),
        ((8, 8, 16, 16, 3, 1, 1), (48, 16)),
    ],
)
@pytest.mark.parametrize("silu", [False, True])
@pytest.mark.parametrize("launch_blocks", [1, 8])
def test_310p_native_convolution(tmp_path, shape, source_layout, silu, launch_blocks):
    try:
        acl = ctypes.CDLL("libascendcl.so")
    except OSError:
        pytest.skip("Ascend runtime unavailable")
    acl.aclrtGetSocName.restype = ctypes.c_char_p
    if not acl.aclrtGetSocName().startswith(b"Ascend310P"):
        pytest.skip("requires Ascend310P")
    plan = _operators().ConvPlan(
        *shape, silu=silu, source_stride=source_layout[0], source_offset=source_layout[1]
    )
    program = plan.jit_kernel(output_padding=16)
    compiled = ir.compile(
        program.specialize(), backend_type=BackendType.Ascend310P, output_dir=str(tmp_path), dump_passes=False
    )
    for seed, call in [(229, compiled), (233, ir.CompiledProgram.from_dir(compiled.output_dir))]:
        g = torch.Generator().manual_seed(seed)
        x = (torch.randn((1, plan.cin, plan.height, plan.width), generator=g) * 0.25).half()
        w = (torch.randn((plan.cout, plan.cin, plan.kernel, plan.kernel), generator=g) * 0.125).half()
        bias = (torch.randn(plan.cout, generator=g) * 0.1).half()
        expected = F.conv2d(x, w, bias, stride=plan.stride, padding=plan.padding)
        if silu:
            expected = expected * torch.sigmoid(expected)
        packed = _pack_input(plan, x)
        packed_bias = torch.zeros(plan.co, dtype=torch.float32)
        packed_bias[: plan.cout] = bias.float()
        output = torch.full((plan.output_size + 32,), 7.0, dtype=torch.float16)
        arguments = (
            plan.metadata(offset=16),
            packed.reshape(-1),
            plan.pack_weight(w),
            packed_bias,
            output,
            torch.zeros(16, dtype=torch.float16),
        )
        if launch_blocks == 1:
            call(*arguments)
        else:
            with A1ResidentSession(call) as session:
                resident = tuple(session.upload(value) for value in arguments)
                session.prepare(call)(*resident, blocks=launch_blocks)
                output.copy_(resident[4].to_cpu())
        actual = (
            output[16:-16].reshape(plan.oh, plan.ow, plan.co)[:, :, : plan.cout].permute(2, 0, 1).unsqueeze(0)
        )
        torch.testing.assert_close(actual, expected, rtol=0.003, atol=0.002)
        assert torch.equal(output[:16], torch.full((16,), 7.0, dtype=torch.float16))
        assert torch.equal(output[-16:], torch.full((16,), 7.0, dtype=torch.float16))


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(height=0),
        dict(stride=3),
        dict(cout=1025),
        dict(width=3),
        dict(source_stride=17),
        dict(source_offset=16),
        dict(source_offset=-16),
    ],
)
def test_convolution_rejects_invalid_geometry(kwargs):
    shape = dict(height=4, width=4, cin=16, cout=16, kernel=1, padding=0)
    shape.update(kwargs)
    with pytest.raises(ValueError):
        _operators().ConvPlan(**shape)


def test_convolution_rejects_invalid_output_alignment():
    plan = _operators().ConvPlan(4, 4, 16, 16, kernel=1, padding=0)
    for offset in (-16, 1):
        with pytest.raises(ValueError, match="offset"):
            plan.metadata(offset)
        with pytest.raises(ValueError, match="padding"):
            plan.jit_kernel(offset)


@pytest.mark.parametrize(
    "cin,kernel",
    [
        (1, 1),
        (2, 1),
        (3, 1),
        (3, 3),
        (16, 1),
        (17, 3),
        (18, 1),
        (19, 3),
        (32, 1),
        (32, 3),
        (33, 3),
        (512, 1),
        (513, 1),
    ],
)
def test_310p_four_way_convolution_exact_chw_reduction(tmp_path, cin, kernel):
    require_310p()
    plan = _operators().ConvPlan(4, 4, cin, 16, kernel, 1, kernel // 2, False, True)
    compiled = ir.compile(
        plan.jit_kernel(output_padding=16).specialize(),
        backend_type=BackendType.Ascend310P,
        output_dir=str(tmp_path),
        dump_passes=False,
    )
    for seed, call in [(331, compiled), (337, ir.CompiledProgram.from_dir(compiled.output_dir))]:
        generator = torch.Generator().manual_seed(seed)
        x = torch.randn((1, cin, 4, 4), generator=generator).half()
        weight = torch.randn((16, cin, kernel, kernel), generator=generator).half()
        bias = torch.randn(16, generator=generator).half()
        if cin >= 512:
            # Binary-exact products isolate the cache boundary from CPU summation order.
            x = (x * 8).round() / 8
            weight = (weight * 8).round() / 8
        expected = F.conv2d(x, weight, bias, padding=kernel // 2)
        packed = torch.zeros((4, 4, plan.ci), dtype=torch.float16)
        packed[:, :, :cin] = x[0].permute(1, 2, 0)
        output = torch.full((plan.output_size + 32,), 7.0, dtype=torch.float16)
        call(
            plan.metadata(16),
            packed.reshape(-1),
            plan.pack_weight(weight),
            bias.float(),
            output,
            torch.zeros(16, dtype=torch.float16),
        )
        actual = output[16:-16].reshape(4, 4, 16).permute(2, 0, 1).unsqueeze(0)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert torch.equal(output[:16], torch.full((16,), 7.0, dtype=torch.float16))
        assert torch.equal(output[-16:], torch.full((16,), 7.0, dtype=torch.float16))


@pytest.mark.parametrize(
    "shape",
    [
        (8, 6, 17, 17, 3, 1),
        (8, 8, 3, 32, 3, 2),
        (8, 8, 33, 48, 1, 1),
        (4, 4, 16, 160, 1, 1),
        (128, 128, 17, 17, 3, 2),
        (8, 8, 17, 32, 3, 1, 64, 16),
        (8, 8, 33, 48, 1, 1, 80, 16),
    ],
)
def test_310p_four_way_spatial_channels_and_stride(tmp_path, shape):
    require_310p()
    height, width, cin, cout, kernel, stride = shape[:6]
    source_stride, source_offset = shape[6:] if len(shape) == 8 else (0, 0)
    plan = _operators().ConvPlan(
        height,
        width,
        cin,
        cout,
        kernel,
        stride,
        kernel // 2,
        False,
        True,
        source_stride=source_stride,
        source_offset=source_offset,
    )
    compiled = ir.compile(
        plan.jit_kernel(output_padding=16).specialize(),
        backend_type=BackendType.Ascend310P,
        output_dir=str(tmp_path),
        dump_passes=False,
    )
    for seed, call in [(367, compiled), (373, ir.CompiledProgram.from_dir(compiled.output_dir))]:
        g = torch.Generator().manual_seed(seed)
        x = torch.randn((1, cin, height, width), generator=g).half()
        w = torch.randn((cout, cin, kernel, kernel), generator=g).half()
        b = torch.randn(cout, generator=g).half()
        if height == 128:
            # Binary-exact products isolate the oversized-image fallback.
            x = (x * 8).round() / 8
            w = (w * 8).round() / 8
        expected = F.conv2d(x, w, b, stride=stride, padding=kernel // 2)
        packed = _pack_input(plan, x)
        bias = torch.zeros(plan.co)
        bias[:cout] = b.float()
        output = torch.full((plan.output_size + 32,), 7.0, dtype=torch.float16)
        call(
            plan.metadata(16),
            packed.reshape(-1),
            plan.pack_weight(w),
            bias,
            output,
            torch.zeros(16, dtype=torch.float16),
        )
        actual = output[16:-16].reshape(plan.oh, plan.ow, plan.co)
        torch.testing.assert_close(
            actual[:, :, :cout].permute(2, 0, 1).unsqueeze(0), expected, rtol=0, atol=0
        )
        assert torch.all(actual[:, :, cout:] == 0)
        assert torch.equal(output[:16], torch.full((16,), 7.0, dtype=torch.float16))
        assert torch.equal(output[-16:], torch.full((16,), 7.0, dtype=torch.float16))


@pytest.mark.parametrize("cin,kernel", [(3, 3), (17, 3), (32, 1), (513, 1), (256, 3)])
def test_310p_four_way_weight_l2_bypass(tmp_path, cin, kernel):
    require_310p()
    ci = (cin + 15) // 16 * 16
    plan = _operators().ConvPlan(
        8,
        8,
        cin,
        17,
        kernel,
        1,
        kernel // 2,
        four_way_accumulation=True,
        source_stride=ci + 32,
        source_offset=16,
    )
    compiled = ir.compile(
        plan.jit_kernel(output_padding=16).specialize(),
        backend_type=BackendType.Ascend310P,
        output_dir=str(tmp_path),
        dump_passes=False,
    )
    generator = torch.Generator().manual_seed(823)
    x = torch.randn((1, cin, 8, 8), generator=generator).half()
    weight = torch.randn((17, cin, kernel, kernel), generator=generator).half()
    bias = torch.zeros(plan.co)
    bias[:17] = torch.randn(17, generator=generator).half().float()
    with A1ResidentSession(compiled, 0) as session:
        offset = _runtime_l2_cache_offset(0)
        call = session.prepare(compiled)
        inputs = tuple(
            session.upload(t)
            for t in (
                _pack_input(plan, x).reshape(-1),
                plan.pack_weight(weight),
                bias,
            )
        )
        output = session.upload(torch.full((plan.output_size + 32,), 7.0, dtype=torch.float16))
        zero = session.upload(torch.zeros(16, dtype=torch.float16))
        actual = []
        for active in (0, offset):
            meta = session.upload(plan.metadata(16, weight_l2_offset=active))
            call(meta, *inputs, output, zero, blocks=8)
            actual.append(output.to_cpu())
        assert torch.equal(actual[0].view(torch.int16), actual[1].view(torch.int16))
        assert torch.all(actual[1][:16] == 7) and torch.all(actual[1][-16:] == 7)
        assert torch.all(actual[1][16:-16].reshape(8, 8, plan.co)[:, :, 17:] == 0)


@pytest.mark.parametrize(
    "height,width,cin,cout",
    [
        (4, 4, 16, 17),
        (4, 8, 32, 128),
        (8, 8, 512, 128),
        (16, 16, 48, 17),
        (16, 32, 32, 17),
        (32, 32, 48, 17),
        (4, 4, 512, 128),
        (4, 8, 512, 128),
        (16, 16, 512, 128),
        (3, 16, 512, 128),
        (6, 16, 512, 128),
        (12, 16, 512, 128),
        (4, 4, 512, 256),
        (16, 16, 512, 129),
        (8, 16, 16, 256),
    ],
)
@pytest.mark.parametrize("silu", [False, True])
def test_310p_ordered_pointwise_cube_exact_tiles(tmp_path, height, width, cin, cout, silu):
    require_310p()
    plan = _operators().ConvPlan(
        height, width, cin, cout, 1, 1, 0, silu, True, source_stride=cin + 32, source_offset=16
    )
    compiled = ir.compile(
        plan.jit_kernel(output_padding=16).specialize(),
        backend_type=BackendType.Ascend310P,
        output_dir=str(tmp_path),
        dump_passes=False,
    )
    for seed, call in [(991, compiled), (997, ir.CompiledProgram.from_dir(compiled.output_dir))]:
        generator = torch.Generator().manual_seed(seed)
        x = torch.randn((1, cin, height, width), generator=generator).half()
        # Include finite FP16 subnormals and both signs of zero inside the source view.
        x[0, 0, 0, :4] = torch.tensor([2**-24, -(2**-24), 0.0, -0.0], dtype=torch.float16)
        weight = torch.randn((cout, cin, 1, 1), generator=generator).half()
        bias = torch.zeros(plan.co)
        bias[:cout] = torch.randn(cout, generator=generator).half().float()
        partials = torch.zeros((4, height, width, cout))
        for channel in range(cin):
            partials[channel % 4].add_(x[0, channel].float().unsqueeze(-1) * weight[:, channel, 0, 0].float())
        expected = (((partials[0] + partials[1]) + partials[2]) + partials[3] + bias[:cout]).half()
        if silu:
            expected = expected * torch.sigmoid(expected)
        output = torch.full((plan.output_size + 32,), 7.0, dtype=torch.float16)
        call(
            plan.metadata(16),
            _pack_input(plan, x).reshape(-1),
            plan.pack_weight(weight),
            bias,
            output,
            torch.zeros(16, dtype=torch.float16),
        )
        actual = output[16:-16].reshape(height, width, plan.co)
        torch.testing.assert_close(actual[:, :, :cout], expected, rtol=0, atol=0)
        assert torch.all(actual[:, :, cout:] == 0)
        assert torch.all(output[:16] == 7) and torch.all(output[-16:] == 7)


@pytest.mark.parametrize("height,cin,stride", [(16, 17, 2), (8, 513, 1)])
def test_310p_eight_pixel_spatial_convolution_guards(tmp_path, height, cin, stride):
    require_310p()
    plan = _operators().ConvPlan(
        height,
        height,
        cin,
        17,
        3,
        stride,
        1,
        False,
        True,
        source_stride=(cin + 15) // 16 * 16 + 32,
        source_offset=16,
    )
    compiled = ir.compile(
        plan.jit_kernel(output_padding=16).specialize(),
        backend_type=BackendType.Ascend310P,
        output_dir=str(tmp_path),
        dump_passes=False,
    )
    for seed, call in [(1009, compiled), (1013, ir.CompiledProgram.from_dir(compiled.output_dir))]:
        generator = torch.Generator().manual_seed(seed)
        # Binary-exact products isolate the long reduction and scratch boundary.
        x = (torch.randn((1, cin, height, height), generator=generator) * 8).round().half() / 8
        weight = (torch.randn((17, cin, 3, 3), generator=generator) * 8).round().half() / 8
        bias = torch.zeros(plan.co)
        bias[:17] = torch.randn(17, generator=generator).half().float()
        expected = F.conv2d(x, weight, bias[:17].half(), stride=stride, padding=1)
        output = torch.full((plan.output_size + 32,), 7.0, dtype=torch.float16)
        call(
            plan.metadata(16),
            _pack_input(plan, x).reshape(-1),
            plan.pack_weight(weight),
            bias,
            output,
            torch.zeros(16, dtype=torch.float16),
        )
        actual = output[16:-16].reshape(plan.oh, plan.ow, plan.co)
        torch.testing.assert_close(actual[:, :, :17].permute(2, 0, 1).unsqueeze(0), expected, rtol=0, atol=0)
        assert torch.all(actual[:, :, 17:] == 0)
        assert torch.all(output[:16] == 7) and torch.all(output[-16:] == 7)


@pytest.mark.parametrize(
    "height,width,cin,cout,stride",
    [
        (4, 4, 3, 17, 1),
        (4, 4, 4, 17, 1),
        (4, 8, 12, 32, 1),
        (4, 8, 17, 128, 1),
        (8, 8, 64, 128, 1),
        (32, 32, 32, 32, 2),
        (128, 128, 3, 16, 2),
        (8, 8, 128, 128, 1),
        (4, 4, 225, 17, 1),
        (64, 24, 17, 17, 1),
    ],
)
@pytest.mark.parametrize("silu", [False, True])
def test_310p_ordered_spatial_cube_exact_tiles(tmp_path, height, width, cin, cout, stride, silu):
    require_310p()
    plan = _operators().ConvPlan(
        height,
        width,
        cin,
        cout,
        3,
        stride,
        1,
        silu,
        True,
        source_stride=(cin + 15) // 16 * 16 + 32,
        source_offset=16,
    )
    assert plan.ordered_spatial_cube
    _check_ordered_spatial_cube_exact(tmp_path, plan, silu)


def _check_ordered_spatial_cube_exact(tmp_path, plan, silu, *, launch_blocks=1):
    height, width, cin, cout, stride = plan.height, plan.width, plan.cin, plan.cout, plan.stride
    compiled = ir.compile(
        plan.jit_kernel(output_padding=16).specialize(),
        backend_type=BackendType.Ascend310P,
        output_dir=str(tmp_path),
        dump_passes=False,
    )
    for seed, call in [(1031, compiled), (1033, ir.CompiledProgram.from_dir(compiled.output_dir))]:
        generator = torch.Generator().manual_seed(seed)
        x = (torch.randn((1, cin, height, width), generator=generator) * 8).round().half() / 8
        weight = (torch.randn((cout, cin, 3, 3), generator=generator) * 8).round().half() / 8
        bias = torch.zeros(plan.co)
        bias[:cout] = torch.randn(cout, generator=generator).half().float()
        expected = F.conv2d(x, weight, bias[:cout].half(), stride=stride, padding=1)
        if silu:
            expected = expected * torch.sigmoid(expected)
        output = torch.full((plan.output_size + 32,), 7.0, dtype=torch.float16)
        arguments = (
            plan.metadata(16),
            _pack_input(plan, x).reshape(-1),
            plan.pack_weight(weight),
            bias,
            output,
            torch.zeros(16, dtype=torch.float16),
        )
        if launch_blocks == 1:
            call(*arguments)
        else:
            with A1ResidentSession(call) as session:
                resident = tuple(session.upload(value) for value in arguments)
                session.prepare(call)(*resident, blocks=launch_blocks)
                output.copy_(resident[4].to_cpu())
        actual = output[16:-16].reshape(plan.oh, plan.ow, plan.co)
        torch.testing.assert_close(
            actual[:, :, :cout].permute(2, 0, 1).unsqueeze(0), expected, rtol=0, atol=0
        )
        assert torch.all(actual[:, :, cout:] == 0)
        assert torch.all(output[:16] == 7) and torch.all(output[-16:] == 7)


@pytest.mark.parametrize(
    "height,width,cin,cout",
    [
        (4, 4, 226, 113),
        (4, 4, 228, 113),
        (8, 8, 256, 128),
        (4, 8, 257, 128),
        (4, 4, 1024, 113),
        (8, 16, 256, 128),
    ],
)
def test_310p_streamed_spatial_cube_exact_tiles(tmp_path, height, width, cin, cout):
    require_310p()
    plan = _operators().ConvPlan(
        height,
        width,
        cin,
        cout,
        3,
        1,
        1,
        False,
        True,
        source_stride=(cin + 15) // 16 * 16 + 32,
        source_offset=16,
    )
    assert plan.streamed_spatial_cube and not plan.ordered_spatial_cube
    _check_ordered_spatial_cube_exact(tmp_path, plan, False)


@pytest.mark.parametrize(
    "height,width,cin,cout",
    [
        (4, 4, 64, 17),
        (4, 8, 32, 128),
        (8, 8, 64, 128),
        (16, 16, 32, 128),
        (4, 4, 228, 128),
        (4, 4, 4, 17),
        (4, 4, 12, 32),
        (4, 4, 16, 128),
        (4, 4, 512, 128),
        (4, 4, 228, 16),
        (4, 4, 256, 16),
        (4, 4, 512, 16),
        (4, 4, 1024, 16),
        (8, 16, 64, 49),
        (8, 8, 128, 113),
        (4, 4, 128, 256),
        (16, 16, 12, 32),
        (8, 8, 228, 128),
        (4, 4, 64, 256),
        (4, 4, 12, 241),
        (32, 32, 16, 32),
        (16, 16, 64, 63),
        (17, 16, 32, 128),
        (33, 16, 12, 16),
        (64, 32, 16, 16),
        (8, 8, 12, 113),
        (8, 8, 512, 128),
    ],
)
def test_310p_spatial_cube_random_four_partials(tmp_path, height, width, cin, cout):
    """Check cancellation and operand reuse against independent CHW partials."""
    _check_spatial_cube_random_four_partials(tmp_path, height, width, cin, cout)


def test_310p_spatial_cube_random_chw_tail(tmp_path):
    """Keep the final three CHW products in partial zero, matching the CPU convolution."""
    _check_spatial_cube_random_four_partials(tmp_path, 32, 32, 3, 16, chw_tail=True)


def _check_spatial_cube_random_four_partials(
    tmp_path, height, width, cin, cout, *, chw_tail=False, launch_blocks=1
):
    require_310p()
    plan = _operators().ConvPlan(
        height,
        width,
        cin,
        cout,
        3,
        1,
        1,
        False,
        True,
        source_stride=(cin + 15) // 16 * 16 + 32,
        source_offset=16,
    )
    compiled = ir.compile(
        plan.jit_kernel(output_padding=16).specialize(),
        backend_type=BackendType.Ascend310P,
        output_dir=str(tmp_path),
        dump_passes=False,
    )
    for seed, call in [(433, compiled), (439, ir.CompiledProgram.from_dir(compiled.output_dir))]:
        g = torch.Generator().manual_seed(seed)
        x = torch.randn((1, cin, height, width), generator=g).half()
        x[0, 0, 0, :4] = torch.tensor([2**-24, -(2**-24), 0.0, -0.0], dtype=torch.float16)
        weight = torch.randn((cout, cin, 3, 3), generator=g).half()
        bias = torch.zeros(plan.co)
        bias[:cout] = torch.randn(cout, generator=g).half().float()
        patches = F.unfold(x.float(), 3, padding=1).transpose(1, 2).reshape(-1, cin * 9)
        terms = weight.float().reshape(cout, -1)
        partials = torch.zeros((4, height * width, cout))
        for term in range(cin * 9):
            partial = 0 if chw_tail and term >= cin * 9 // 4 * 4 else term % 4
            partials[partial].add_(patches[:, term, None] * terms[:, term])
        expected = (((partials[0] + partials[1]) + partials[2]) + partials[3] + bias[:cout]).half()
        if chw_tail:
            cpu = F.conv2d(x, weight, bias[:cout].half(), padding=1)
            assert torch.equal(
                expected.view(torch.int16),
                cpu.permute(0, 2, 3, 1).reshape_as(expected).contiguous().view(torch.int16),
            )
        output = torch.full((plan.output_size + 32,), 7.0, dtype=torch.float16)
        arguments = (
            plan.metadata(16),
            _pack_input(plan, x).reshape(-1),
            plan.pack_weight(weight),
            bias,
            output,
            torch.zeros(16, dtype=torch.float16),
        )
        if launch_blocks == 1:
            call(*arguments)
        else:
            with A1ResidentSession(call) as session:
                resident = tuple(session.upload(value) for value in arguments)
                session.prepare(call)(*resident, blocks=launch_blocks)
                output.copy_(resident[4].to_cpu())
        actual = output[16:-16].reshape(height * width, plan.co)
        assert torch.equal(actual[:, :cout].contiguous().view(torch.int16), expected.view(torch.int16))
        assert torch.all(actual[:, cout:] == 0)
        assert torch.all(output[:16] == 7) and torch.all(output[-16:] == 7)


@pytest.mark.parametrize(
    "height,width,cin,cout",
    [
        (8, 8, 64, 128),
        (16, 16, 32, 128),
        (8, 16, 64, 49),
        (8, 8, 512, 128),
        (32, 32, 16, 32),
        (17, 16, 32, 128),
        (8, 8, 12, 113),
    ],
)
def test_310p_spatial_resident_band_random_four_partials(tmp_path, height, width, cin, cout):
    """Retain exact sums with eight cores, halo bands, source views and fallback grids."""
    _check_spatial_cube_random_four_partials(tmp_path, height, width, cin, cout, launch_blocks=8)


@pytest.mark.parametrize(
    "height,width,cin,cout,stride,silu",
    [
        (32, 32, 64, 64, 2, True),
        (16, 16, 128, 128, 2, True),
        (8, 8, 128, 16, 1, True),
        (64, 32, 16, 16, 1, False),
    ],
)
def test_310p_spatial_resident_band_strided_tiles(tmp_path, height, width, cin, cout, stride, silu):
    """Keep top/bottom halos and the existing dynamic C16 row-band schedule exact."""
    require_310p()
    plan = _operators().ConvPlan(
        height,
        width,
        cin,
        cout,
        3,
        stride,
        1,
        silu,
        True,
        source_stride=(cin + 15) // 16 * 16 + 32,
        source_offset=16,
    )
    _check_ordered_spatial_cube_exact(tmp_path, plan, silu, launch_blocks=8)


@pytest.mark.parametrize(
    "height,width,cin,cout,stride",
    [
        (640, 640, 3, 16, 2),
        (320, 320, 16, 32, 2),
        (160, 160, 64, 64, 2),
        (80, 80, 128, 128, 2),
        (80, 80, 64, 64, 2),
        (80, 80, 64, 16, 1),
    ],
)
@pytest.mark.parametrize("launch_blocks", [1, 8])
def test_310p_windowed_spatial_cube_production_geometry(
    tmp_path, height, width, cin, cout, stride, launch_blocks
):
    """Check real large-image halos, source views and output guards exactly."""
    require_310p()
    plan = _operators().ConvPlan(
        height,
        width,
        cin,
        cout,
        3,
        stride,
        1,
        False,
        True,
        source_stride=(cin + 15) // 16 * 16 + 32,
        source_offset=16,
    )
    assert plan.windowed_spatial_cube
    _check_ordered_spatial_cube_exact(tmp_path, plan, False, launch_blocks=launch_blocks)


@pytest.mark.parametrize("cin,cout", [(64, 16), (128, 17)])
@pytest.mark.parametrize("launch_blocks", [1, 8])
def test_310p_windowed_spatial_cube_random_four_partials(tmp_path, cin, cout, launch_blocks):
    """Keep arbitrary FP16 products bitwise exact across cropped multi-C0 halos."""
    plan = _operators().ConvPlan(80, 80, cin, cout, 3, 1, 1, False, True)
    assert plan.windowed_spatial_cube
    _check_spatial_cube_random_four_partials(tmp_path, 80, 80, cin, cout, launch_blocks=launch_blocks)


@pytest.mark.parametrize(
    "height,width,cin,cout,launch_blocks",
    [(88, 80, 64, 33, 3), (88, 80, 64, 33, 8), (80, 32, 256, 17, 8)],
)
def test_310p_windowed_spatial_cube_ragged_stripes(tmp_path, height, width, cin, cout, launch_blocks):
    """Mask ragged spatial stripes; also reuse input with streamed weight banks."""
    plan = _operators().ConvPlan(height, width, cin, cout, 3, 1, 1, False, True)
    assert plan.windowed_spatial_cube
    _check_spatial_cube_random_four_partials(tmp_path, height, width, cin, cout, launch_blocks=launch_blocks)


@pytest.mark.parametrize("kind", ["cube", "pointwise", "spatial", "four_way_fallback", "depthwise"])
def test_310p_fp16_silu_all_finite_patterns(tmp_path, kind):
    """Match CPU FP16 sigmoid/multiply boundaries across every finite bit pattern."""
    require_310p()
    patterns = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.float16)
    finite = torch.isfinite(patterns)
    assert int(finite.sum()) == 63488
    # Identity convolutions start their sum at +0; normalize both signed zeros.
    values = torch.where(finite & (patterns != 0), patterns, torch.zeros_like(patterns))
    packed = values.reshape(64, 64, 16)
    if kind == "depthwise":
        plan = load_plan("depthwise").DepthwisePlan(64, 64, 16, silu=True, four_way_accumulation=True)
        weight = torch.zeros((16, 1, 3, 3), dtype=torch.float16)
        weight[:, 0, 1, 1] = 1
    else:
        stride = 2 if kind == "four_way_fallback" else 1
        kernel = 3 if kind == "spatial" else 1
        height = width = 128 if stride == 2 else 64
        plan = _operators().ConvPlan(height, width, 16, 16, kernel, stride, kernel // 2, True, kind != "cube")
        weight = torch.zeros((16, 16, kernel, kernel), dtype=torch.float16)
        weight[torch.arange(16), torch.arange(16), kernel // 2, kernel // 2] = 1
        if stride == 2:
            packed = packed.repeat_interleave(2, 0).repeat_interleave(2, 1)
    compiled = ir.compile(
        plan.jit_kernel().specialize(),
        backend_type=BackendType.Ascend310P,
        output_dir=str(tmp_path),
        dump_passes=False,
    )
    with A1ResidentSession(compiled) as session:
        call = session.prepare(ir.CompiledProgram.from_dir(compiled.output_dir))
        guard = session.upload(torch.full((plan.output_size + 32,), 7.0, dtype=torch.float16))
        output = guard.view((plan.output_size,), 16)
        arguments = [
            session.upload(t)
            for t in (plan.metadata(), packed.reshape(-1), plan.pack_weight(weight), torch.zeros(16))
        ]
        arguments.append(output)
        if kind != "depthwise":
            arguments.append(session.upload(torch.zeros(16, dtype=torch.float16)))
        call(*arguments, blocks=8)
        expected = values * torch.sigmoid(values)
        assert torch.equal(output.to_cpu().view(torch.int16), expected.view(torch.int16))
        actual = guard.to_cpu()
        assert torch.all(actual[:16] == 7) and torch.all(actual[-16:] == 7)


@pytest.mark.parametrize("launch_blocks", [1, 8])
@pytest.mark.parametrize(
    "height,width,cin,cout",
    [(16, 16, 20, 64), (40, 40, 32, 32), (40, 40, 128, 256), (80, 80, 128, 128)],
)
def test_310p_spatial_wide_panels_large_grid_exact(tmp_path, height, width, cin, cout, launch_blocks):
    """Share im2col operands across N32 while retaining all four ordered sums.

    Include a partial C0, resident and streamed weights, bounded input windows,
    strided source views and a spatial grid that is ragged across eight cores.
    The independent random oracle also checks restored artifacts and guards.
    """
    _check_spatial_cube_random_four_partials(tmp_path, height, width, cin, cout, launch_blocks=launch_blocks)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
