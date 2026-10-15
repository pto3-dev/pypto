# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Keep native YOLO26 replay free from silent host-operator fallbacks."""

from typing import Any

import pytest
import torch

from examples.models.yolo26_native.executor import Launch, NativeYOLO26, RejectHostOperators
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
