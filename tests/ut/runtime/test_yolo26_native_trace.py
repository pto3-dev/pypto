# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Keep native YOLO26 replay free from silent host-operator fallbacks."""

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from examples.models.yolo26_native.executor import (
    Feature,
    Launch,
    NativeYOLO26,
    RejectHostOperators,
    _runtime_l2_cache_offset,
)
from examples.models.yolo26_native.operators import ConvPlan
from examples.models.yolo26_native.precision import FP16SiLU, prepare_fp16_model


def test_native_trace_allows_host_dispatch_without_tensor_arithmetic():
    seen = []

    def kernel(value: Any, *, blocks: int) -> None:
        seen.append((value, blocks))

    native = NativeYOLO26.__new__(NativeYOLO26)
    native.trace = [Launch(kernel, ("resident-tensor",), 8, "test")]
    guard = RejectHostOperators()
    with guard:
        native.replay()
    assert seen == [("resident-tensor", 8)]
    assert guard.calls == 0


def test_native_trace_rejects_host_tensor_arithmetic():
    tensor = torch.ones(4)

    def kernel(*, blocks: int) -> None:
        torch.add(tensor, blocks)

    native = NativeYOLO26.__new__(NativeYOLO26)
    native.trace = [Launch(kernel, (), 1, "invalid-fallback")]
    guard = RejectHostOperators()
    with pytest.raises(RuntimeError, match="host ATen operation"), guard:
        native.replay()
    assert guard.calls == 1


def test_native_unknown_module_fails_without_calling_cpu_forward():
    class Unsupported(torch.nn.Module):
        def forward(self, _value):
            raise AssertionError("CPU forward must never run")

    native = NativeYOLO26.__new__(NativeYOLO26)
    with pytest.raises(ValueError, match="unsupported native YOLO26 module"):
        native._module(Unsupported(), None)


def test_fp16_silu_preserves_sigmoid_narrowing_boundary():
    x = torch.linspace(-8, 8, 4096).half()
    expected = (x.float() * torch.sigmoid(x.float()).half().float()).half()
    actual = FP16SiLU()(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.any(actual != torch.nn.functional.silu(x))


@pytest.mark.parametrize("affine,tracked", [(False, True), (True, False)])
def test_fp16_preparation_rejects_batchnorm_without_constants(affine, tracked):
    class Invalid(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(16, 16, 1)
            self.bn = torch.nn.BatchNorm2d(16, affine=affine, track_running_stats=tracked)

        def fuse(self, *, verbose):
            raise AssertionError("invalid BatchNorm must fail before fusion")

    with pytest.raises(ValueError, match="affine BatchNorm with tracked"):
        prepare_fp16_model(Invalid().eval())


@pytest.mark.parametrize("bypass", [False, True])
def test_native_convolution_consumes_source_view_without_copy(monkeypatch, bypass):
    native = NativeYOLO26.__new__(NativeYOLO26)
    native.four_way_accumulation = True
    native.weight_l2_bypass = bypass
    native.device_id = 0
    native.weight_l2_cache_offset = 0
    native.trace = []
    native.session = SimpleNamespace(upload=lambda value: value)
    monkeypatch.setattr(native, "_prepare", lambda *_args: "prepared-convolution")
    monkeypatch.setattr(native, "_empty", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(native, "_dummy", lambda: object())

    def reject_compaction(_value):
        raise AssertionError("ordinary convolution must consume its source view")

    monkeypatch.setattr(native, "_compact", reject_compaction)
    queried = []

    def offset(device):
        queried.append(device)
        return 1 << 44

    monkeypatch.setattr("examples.models.yolo26_native.executor._runtime_l2_cache_offset", offset)
    source = object()
    view = Feature(source, 4, 4, 16, 64, 16)
    result = native._convolution(torch.nn.Conv2d(16, 16, 1).eval(), view, False)
    assert len(native.trace) == 1 and native.trace[0].label == "convolution"
    assert native.trace[0].arguments[1] is source
    assert native.trace[0].arguments[0][13:15].tolist() == [64.0, 16.0]
    assert queried == ([0] if bypass else [])
    assert native.trace[0].arguments[0][16:20].tolist() == ([0, 0, 4096, 0] if bypass else [0] * 4)
    assert (result.height, result.width, result.channels, result.row_stride, result.offset) == (
        4,
        4,
        16,
        16,
        0,
    )


@pytest.mark.parametrize("offset", [0, 1 << 44, 0xFEDCBA9876543210])
def test_convolution_weight_offset_round_trips_without_fp32_loss(offset):
    plan = ConvPlan(4, 4, 16, 16, 1, 1, 0, four_way_accumulation=True)
    words = plan.metadata(weight_l2_offset=offset)[16:20]
    assert sum(int(word) << (16 * i) for i, word in enumerate(words)) == offset


@pytest.mark.parametrize("offset", [True, -1, 1 << 64, 1.0])
def test_convolution_rejects_invalid_weight_l2_offset(offset):
    plan = ConvPlan(4, 4, 16, 16, 1, 1, 0, four_way_accumulation=True)
    with pytest.raises(ValueError, match="uint64"):
        plan.metadata(weight_l2_offset=offset)


def test_cube_convolution_rejects_weight_l2_offset():
    plan = ConvPlan(4, 4, 16, 16, 1, 1, 0)
    with pytest.raises(ValueError, match="four-way"):
        plan.metadata(weight_l2_offset=1 << 44)


@pytest.mark.parametrize(
    "current,status,offset", [(0, 0, 1 << 44), (1, 0, 1 << 44), (0, 107001, 0), (0, 0, 0)]
)
def test_weight_l2_query_uses_visible_runtime_device(monkeypatch, current, status, offset):
    seen = []

    def get_device(pointer):
        pointer._obj.value = current
        return 0

    def get_offset(device, pointer):
        seen.append(device)
        pointer._obj.value = offset
        return status

    library = SimpleNamespace(aclrtGetDevice=get_device, rtGetL2CacheOffset=get_offset)
    monkeypatch.setattr("examples.models.yolo26_native.executor.ctypes.CDLL", lambda _name: library)
    if current == 0 and status == 0 and offset:
        assert _runtime_l2_cache_offset(0) == offset
    else:
        with pytest.raises(RuntimeError):
            _runtime_l2_cache_offset(0)
    assert seen == ([] if current != 0 else [0])


@pytest.mark.parametrize(
    "cin,kernel,stride,ordered,selected",
    [
        (16, 1, 1, True, True),
        (512, 1, 1, True, True),
        (528, 1, 1, True, False),
        (17, 1, 1, True, False),
        (32, 3, 1, True, False),
        (32, 1, 2, True, False),
        (32, 1, 1, False, False),
    ],
)
def test_ordered_pointwise_cube_bounds(cin, kernel, stride, ordered, selected):
    plan = ConvPlan(8, 8, cin, 17, kernel, stride, kernel // 2, four_way_accumulation=ordered)
    assert plan.ordered_pointwise_cube is selected
    packed = plan.pack_weight(torch.ones((17, cin, kernel, kernel)))
    assert packed.numel() == plan.weight_size
    if selected:
        assert packed.numel() == plan.co * cin
        assert torch.count_nonzero(packed) == 17 * cin


@pytest.mark.parametrize("row_stride,selected", [(1048576, True), (1048592, False)])
def test_ordered_pointwise_cube_bounds_dma_source_stride(row_stride, selected):
    plan = ConvPlan(4, 4, 16, 16, 1, 1, 0, four_way_accumulation=True, source_stride=row_stride)
    assert plan.ordered_pointwise_cube is selected


@pytest.mark.parametrize(
    "height,cin,stride,selected",
    [
        (4, 3, 1, True),
        (32, 64, 2, True),
        (128, 3, 2, True),
        (4, 112, 1, True),
        (4, 113, 1, True),
        (4, 225, 1, True),
        (4, 226, 1, False),
        (8, 256, 1, False),
    ],
)
def test_ordered_spatial_cube_bounds(height, cin, stride, selected):
    plan = ConvPlan(height, height, cin, 17, 3, stride, 1, four_way_accumulation=True)
    assert plan.ordered_spatial_cube is selected
    assert plan.pack_weight(torch.ones((17, cin, 3, 3))).numel() == plan.co * plan.reduction_size


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
