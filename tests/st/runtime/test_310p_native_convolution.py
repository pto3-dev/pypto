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


def _operators():
    path = Path(__file__).resolve().parents[3] / "examples/models/yolo26_native/operators.py"
    spec = importlib.util.spec_from_file_location("yolo26_native_operators", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "shape",
    [(8, 8, 3, 16, 3, 2, 1), (8, 8, 32, 32, 3, 1, 1), (4, 4, 16, 16, 1, 1, 0), (4, 4, 64, 4, 1, 1, 0)],
)
@pytest.mark.parametrize("silu", [False, True])
def test_310p_native_convolution(tmp_path, shape, silu):
    try:
        acl = ctypes.CDLL("libascendcl.so")
    except OSError:
        pytest.skip("Ascend runtime unavailable")
    acl.aclrtGetSocName.restype = ctypes.c_char_p
    if not acl.aclrtGetSocName().startswith(b"Ascend310P"):
        pytest.skip("requires Ascend310P")
    plan = _operators().ConvPlan(*shape, silu=silu)
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
            expected = F.silu(expected)
        packed = torch.zeros((plan.height, plan.width, plan.ci), dtype=torch.float16)
        packed[:, :, : plan.cin] = x[0].permute(1, 2, 0)
        packed_bias = torch.zeros(plan.co, dtype=torch.float32)
        packed_bias[: plan.cout] = bias.float()
        output = torch.full((plan.output_size + 32,), 7.0, dtype=torch.float16)
        call(
            plan.metadata(offset=16),
            packed.reshape(-1),
            plan.pack_weight(w),
            packed_bias,
            output,
            torch.zeros(16, dtype=torch.float16),
        )
        actual = (
            output[16:-16].reshape(plan.oh, plan.ow, plan.co)[:, :, : plan.cout].permute(2, 0, 1).unsqueeze(0)
        )
        torch.testing.assert_close(actual, expected, rtol=0.003, atol=0.002)
        assert torch.equal(output[:16], torch.full((16,), 7.0, dtype=torch.float16))
        assert torch.equal(output[-16:], torch.full((16,), 7.0, dtype=torch.float16))


@pytest.mark.parametrize("kwargs", [dict(height=0), dict(stride=3), dict(cout=1025), dict(width=3)])
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
