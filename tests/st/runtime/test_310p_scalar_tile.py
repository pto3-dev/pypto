# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""310P scalar UB reads/writes interoperate with vector work and DMA."""

import pypto.language as pl
import pytest
import torch
from pypto import ir
from pypto.backend import BackendType
from pypto.library import fp16_softmax_row
from pypto.runtime import A1ResidentSession
from yolo26_native_test_utils import require_310p


@pytest.mark.parametrize("blocks", [1, 8])
def test_310p_scalar_tile_roundtrip(tmp_path, blocks):
    require_310p()

    @pl.jit.incore
    def compute(x: pl.Tensor[[16, 16], pl.FP32], out: pl.Out[pl.Tensor[[16, 16], pl.FP32]]):
        for row in pl.range(pl.tile.get_block_idx(), 16, pl.tile.get_block_num()):
            value = pl.load(x, [row, 0], [1, 16])
            doubled = pl.add(value, value)
            scalar = pl.tile.read(doubled, [0, 1])
            pl.tile.write(doubled, [0, 0], scalar)
            # This vector consumer must wait for the S-pipe write above.
            result = pl.add(doubled, doubled)
            pl.store(result, [row, 0], out)
        return out

    compiled = ir.compile(
        compute.specialize(), backend_type=BackendType.Ascend310P, output_dir=str(tmp_path), dump_passes=False
    )
    with A1ResidentSession(compiled) as session:
        kernel = session.prepare(ir.CompiledProgram.from_dir(compiled.output_dir))
        dx = session.empty((16, 16), torch.float32)
        guard = session.upload(torch.full((288,), 7.0))
        output = guard.view((16, 16), 16)
        for seed in (1051, 1061):
            generator = torch.Generator().manual_seed(seed)
            values = torch.randn((16, 16), generator=generator)
            dx.copy_from(values)
            kernel(dx, output, blocks=blocks)
            expected = values * 4
            expected[:, 0] = values[:, 1] * 4
            torch.testing.assert_close(output.to_cpu(), expected, rtol=0, atol=0)
            torch.testing.assert_close(dx.to_cpu(), values, rtol=0, atol=0)
            full = guard.to_cpu()
            assert torch.all(full[:16] == 7) and torch.all(full[-16:] == 7)


@pytest.mark.parametrize("columns,blocks", [(16, 1), (32, 8), (64, 8), (1024, 8)])
def test_310p_public_compensated_softmax_rows(tmp_path, columns, blocks):
    """The reusable algorithm also handles rows beyond the model's 16 columns."""
    require_310p()
    normalize = fp16_softmax_row(columns)

    @pl.jit.incore
    def compute(x: pl.Tensor[[7, columns], pl.FP16], out: pl.Out[pl.Tensor[[7, columns], pl.FP16]]):
        for row in pl.range(pl.tile.get_block_idx(), 7, pl.tile.get_block_num()):
            values = pl.load(x, [row, 0], [1, columns])
            result = normalize(values)
            pl.store(result, [row, 0], out)
        return out

    assert not compute._external_source_paths()
    compiled = ir.compile(
        compute.specialize(), backend_type=BackendType.Ascend310P, output_dir=str(tmp_path), dump_passes=False
    )
    with A1ResidentSession(compiled) as session:
        kernel = session.prepare(ir.CompiledProgram.from_dir(compiled.output_dir))
        dx = session.empty((7, columns), torch.float16)
        guard = session.upload(torch.full((7 * columns + 32,), 7.0, dtype=torch.float16))
        output = guard.view((7, columns), 16)
        for seed in (1063, 1069, 1087):
            generator = torch.Generator().manual_seed(seed)
            values = (torch.randn((7, columns), generator=generator) * 5).half()
            values[0].zero_()
            values[1, 0] = 65504
            values[1, 1:] = -65504
            dx.copy_from(values)
            kernel(dx, output, blocks=blocks)
            torch.testing.assert_close(
                output.to_cpu(), torch.softmax(values.double(), -1).half(), rtol=0, atol=0
            )
            full = guard.to_cpu()
            assert torch.all(full[:16] == 7) and torch.all(full[-16:] == 7)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
