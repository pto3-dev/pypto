# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Full-width Vec row slices retain their parent and pipeline dependencies."""

import pypto.language as pl
import pytest
import torch
from pypto import ir
from pypto.backend import BackendType
from pypto.runtime import A1ResidentSession
from yolo26_native_test_utils import require_310p


@pytest.mark.parametrize("row", [0, 1, 2])
@pytest.mark.parametrize("blocks", [1, 8])
@pytest.mark.parametrize(
    "dtype,torch_dtype,int_dtype",
    [(pl.FP16, torch.float16, torch.int16), (pl.FP32, torch.float32, torch.int32)],
)
def test_310p_public_vec_row_slice(tmp_path, row, blocks, dtype, torch_dtype, int_dtype):
    require_310p()

    @pl.jit.incore
    def kernel(x: pl.Tensor[[120, 80], dtype], out: pl.Out[pl.Tensor[[104, 80], dtype]]):
        for work in pl.range(pl.tile.get_block_idx(), 8, pl.tile.get_block_num()):
            parent = pl.load(x, [work * 15, 0], [15, 80])
            view = pl.tile.slice(parent, [13, 80], [row, 0])
            result = pl.add(view, view)
            out = pl.store(result, [work * 13, 0], out)
        return out

    compiled = ir.compile(kernel.specialize(), backend_type=BackendType.Ascend310P, output_dir=str(tmp_path))
    with A1ResidentSession(compiled) as session:
        call = session.prepare(compiled)
        source = session.empty((120, 80), torch_dtype)
        guard = session.upload(torch.full((104 * 80 + 32,), 7.0, dtype=torch_dtype))
        out = guard.view((104, 80), 16)
        for seed in (3301, 3307):
            values = torch.randn((120, 80), generator=torch.Generator().manual_seed(seed)).to(torch_dtype)
            selected = values.reshape(8, 15, 80)[:, row : row + 13, :].reshape(104, 80)
            expected = selected + selected
            source.copy_from(values)
            out.copy_from(torch.full((104, 80), float("nan"), dtype=torch_dtype))
            call(source, out, blocks=blocks)
            assert torch.equal(out.to_cpu().view(int_dtype), expected.view(int_dtype))
            assert torch.equal(source.to_cpu().view(int_dtype), values.view(int_dtype))
            full = guard.to_cpu()
            assert torch.all(full[:16] == 7) and torch.all(full[-16:] == 7)


def test_310p_vec_slice_rejects_partial_columns(tmp_path):
    @pl.jit.incore
    def kernel(x: pl.Tensor[[15, 80], pl.FP32], out: pl.Out[pl.Tensor[[13, 64], pl.FP32]]):
        parent = pl.load(x, [0, 0], [15, 80])
        view = pl.tile.slice(parent, [13, 64], [0, 0])
        out = pl.store(view, [0, 0], out)
        return out

    with pytest.raises(RuntimeError, match="Vec slices require a bounded full-width static"):
        ir.compile(
            kernel.specialize(),
            backend_type=BackendType.Ascend310P,
            output_dir=str(tmp_path),
            skip_ptoas=True,
        )
