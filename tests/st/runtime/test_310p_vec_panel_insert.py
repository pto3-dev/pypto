# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A1 Vec panel insertion preserves untouched columns and bit patterns."""

import pypto.language as pl
import pytest
import torch
from pypto import ir
from pypto.backend import BackendType
from pypto.runtime import A1ResidentSession
from yolo26_native_test_utils import require_310p


@pytest.mark.parametrize("blocks", [1, 8])
@pytest.mark.parametrize("both", [False, True])
@pytest.mark.parametrize(
    "dtype,torch_dtype,int_dtype",
    [(pl.FP16, torch.float16, torch.int16), (pl.FP32, torch.float32, torch.int32)],
)
def test_310p_vec_insert(tmp_path, blocks, both, dtype, torch_dtype, int_dtype):
    require_310p()

    @pl.jit.incore
    def kernel(
        x: pl.Tensor[[1024, 32], dtype],
        initial: pl.Tensor[[1024, 32], dtype],
        out: pl.Out[pl.Tensor[[1024, 32], dtype]],
    ):
        for work in pl.range(pl.tile.get_block_idx(), 8, pl.tile.get_block_num()):
            target = pl.load(initial, [work * 128, 0], [128, 32])
            left = pl.load(x, [work * 128, 0], [128, 16])
            target = pl.tile.assemble(target, left, [0, 0])
            if both:
                right = pl.load(x, [work * 128, 16], [128, 16])
                target = pl.tile.assemble(target, right, [0, 16])
            out = pl.store(target, [work * 128, 0], out)
        return out

    compiled = ir.compile(kernel.specialize(), backend_type=BackendType.Ascend310P, output_dir=str(tmp_path))
    with A1ResidentSession(compiled) as session:
        call = session.prepare(compiled)
        x = session.empty((1024, 32), torch_dtype)
        initial = session.empty((1024, 32), torch_dtype)
        guard = session.upload(torch.full((1024 * 32 + 32,), 7.0, dtype=torch_dtype))
        out = guard.view((1024, 32), 16)
        for seed in (4101, 4109):
            generator = torch.Generator().manual_seed(seed)
            values = torch.randn((1024, 32), generator=generator).to(torch_dtype)
            original = torch.randn((1024, 32), generator=generator).to(torch_dtype)
            expected = original.clone()
            expected[:, :16] = values[:, :16]
            if both:
                expected[:, 16:] = values[:, 16:]
            x.copy_from(values)
            initial.copy_from(original)
            out.copy_from(torch.full((1024, 32), float("nan"), dtype=torch_dtype))
            call(x, initial, out, blocks=blocks)
            assert torch.equal(out.to_cpu().view(int_dtype), expected.view(int_dtype))
            assert torch.equal(initial.to_cpu().view(int_dtype), original.view(int_dtype))
            full = guard.to_cpu()
            assert torch.all(full[:16] == 7) and torch.all(full[-16:] == 7)


def test_310p_vec_insert_rejects_unaligned_offset(tmp_path):
    @pl.jit.incore
    def kernel(
        x: pl.Tensor[[128, 16], pl.FP32],
        initial: pl.Tensor[[128, 32], pl.FP32],
        out: pl.Out[pl.Tensor[[128, 32], pl.FP32]],
    ):
        source = pl.load(x, [0, 0], [128, 16])
        target = pl.load(initial, [0, 0], [128, 32])
        target = pl.tile.assemble(target, source, [0, 1])
        return pl.store(target, [0, 0], out)

    with pytest.raises(RuntimeError, match="Vec panel insertion requires bounded static"):
        ir.compile(
            kernel.specialize(),
            backend_type=BackendType.Ascend310P,
            output_dir=str(tmp_path),
            skip_ptoas=True,
        )
