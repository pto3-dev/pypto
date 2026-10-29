# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Exhaustive finite FP16 SiLU for depthwise convolution tiles."""

import pypto.language as pl
import pytest
import torch
from pypto import ir
from pypto.backend import BackendType
from pypto.library import fp16_silu_constants
from pypto.runtime import A1ResidentSession
from yolo26_native_test_utils import require_310p


@pytest.mark.parametrize("rows", [1, 13])
@pytest.mark.parametrize("blocks", [1, 8])
def test_310p_silu_n80_all_finite(tmp_path, rows, blocks):
    require_310p()
    cols = 80
    elements = ((65536 + rows * cols - 1) // (rows * cols)) * rows * cols
    image_rows = elements // cols
    activate = fp16_silu_constants(rows, cols)

    @pl.jit.incore
    def compute(x: pl.Tensor[[elements], pl.FP16], output: pl.Out[pl.Tensor[[elements], pl.FP16]]):
        source = pl.tensor.view(x, [image_rows, cols])
        destination = pl.tensor.view(output, [image_rows, cols])
        zero = pl.tile.full([rows, cols], pl.FP32, 0.0)
        offset = pl.tile.full([rows, cols], pl.FP32, 15.0 / 2048)
        scale = pl.tile.full([rows, cols], pl.FP32, 8192.0)
        one = pl.tile.full([rows, cols], pl.FP32, 1.0)
        epsilon = pl.tile.full([rows, cols], pl.FP32, 0.00000011920928955078125)
        for work in pl.range(pl.tile.get_block_idx(), image_rows // rows, pl.tile.get_block_num()):
            value = pl.load(source, [work * rows, 0], [rows, cols])
            destination = pl.store(
                activate(value, zero, offset, scale, one, epsilon), [work * rows, 0], destination
            )
        return output

    compiled = ir.compile(compute.specialize(), backend_type=BackendType.Ascend310P, output_dir=str(tmp_path))
    patterns = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.float16)
    finite = patterns[torch.isfinite(patterns)]
    assert finite.numel() == 63488
    values = torch.cat((finite, torch.zeros(elements - finite.numel(), dtype=torch.float16)))
    with A1ResidentSession(compiled) as session:
        kernel = session.prepare(compiled)
        x = session.upload(values)
        guard = session.upload(torch.full((elements + 32,), 7.0, dtype=torch.float16))
        output = guard.view((elements,), 16)
        kernel(x, output, blocks=blocks)
        actual = output.to_cpu()
        assert torch.equal(actual.view(torch.int16), (values * values.sigmoid()).view(torch.int16))
        full = guard.to_cpu()
        assert torch.all(full[:16] == 7) and torch.all(full[-16:] == 7)


@pytest.mark.parametrize("rows,cols", [(1, 16), (13, 16), (1, 32), (64, 80), (True, 80)])
def test_silu_rejects_unsupported_shapes(rows, cols):
    with pytest.raises(ValueError, match="supported static"):
        fp16_silu_constants(rows, cols)
