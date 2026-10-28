# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Public SPMD dispatch, block-indexed DMA and dependent 310P grids."""

import json

import pypto.language as pl
import pytest
import torch
from pypto import ir
from pypto.backend import BackendType
from pypto.runtime import A1L2Worker
from yolo26_native_test_utils import require_310p


def make_grid(blocks: int):
    @pl.jit
    def packed(
        x: pl.Tensor[[128, 32], pl.FP32],
        y: pl.Tensor[[128, 32], pl.FP32],
        out: pl.Out[pl.Tensor[[128, 32], pl.FP32]],
    ):
        for block in pl.spmd(blocks):
            a = pl.load(x, [block * 16, 16], [16, 16])
            b = pl.load(y, [block * 16, 16], [16, 16])
            out = pl.store(pl.add(a, b), [block * 16, 16], out)
        return out

    return packed


@pytest.mark.parametrize("blocks", [1, 2, 4, 8])
def test_310p_public_spmd_dependent_grids(tmp_path, blocks):
    require_310p()
    compiled = ir.compile(
        make_grid(blocks).specialize(),
        backend_type=BackendType.Ascend310P,
        output_dir=str(tmp_path / "grid"),
        dump_passes=False,
    )
    manifest = json.loads((tmp_path / "grid" / "a1_spmd.json").read_text())
    assert manifest["blocks"] == blocks and manifest["task_count"] == 1
    loaded = ir.CompiledProgram.from_dir(compiled.output_dir)
    # Exercise the ordinary compiled-program call as well as resident dispatch.
    direct_x = torch.arange(4096, dtype=torch.float32).reshape(128, 32) / 16
    direct_y = torch.ones_like(direct_x) * 0.5
    direct_out = torch.full_like(direct_x, -7.25)
    loaded(direct_x, direct_y, direct_out)
    direct_expected = torch.full_like(direct_x, -7.25)
    direct_expected[: blocks * 16, 16:] = direct_x[: blocks * 16, 16:] + 0.5
    assert torch.equal(direct_out, direct_expected)
    with A1L2Worker(loaded) as worker:
        x = torch.arange(4096, dtype=torch.float32).reshape(128, 32) / 16
        y = torch.ones_like(x) * 0.5
        sentinel = torch.full_like(x, -7.25)
        dx, dy = worker.upload(x), worker.upload(y)
        middle, output = worker.upload(sentinel), worker.upload(sentinel)
        kernel = worker.prepare(loaded)
        with pytest.raises(ValueError, match="blocks"):
            kernel(dx, dy, middle, blocks=9)
        first = worker.prepare_spmd(loaded, dx, dy, middle)
        assert first.task_count == 1
        graph = worker.register_graph(
            [(kernel, (dx, dy, middle), blocks), (kernel, (middle, dy, output), blocks)]
        )
        assert graph.task_count == 2
        for offset in (0.0, 4.0):
            dx.copy_from(x + offset)
            graph()
            expected = sentinel.clone()
            expected[: blocks * 16, 16:] = x[: blocks * 16, 16:] + offset + 1.0
            assert torch.equal(output.to_cpu(), expected)
        with pytest.raises(ValueError, match="virtualized"):
            worker.prepare_independent(loaded, 8)


@pytest.mark.parametrize("blocks", [1, 8])
def test_310p_spmd_packs_uneven_worklist(tmp_path, blocks):
    require_310p()

    @pl.jit
    def packed(x: pl.Tensor[[400, 16], pl.FP32], out: pl.Out[pl.Tensor[[400, 16], pl.FP32]]):
        for block in pl.spmd(blocks):
            for work in pl.range(block, 25, pl.tile.get_block_num()):
                tile = pl.load(x, [work * 16, 0], [16, 16])
                bias = pl.tile.full([16, 16], pl.FP32, 0.5)
                out = pl.store(pl.add(tile, bias), [work * 16, 0], out)
        return out

    compiled = ir.compile(
        packed.specialize(),
        backend_type=BackendType.Ascend310P,
        output_dir=str(tmp_path / "worklist"),
        dump_passes=False,
    )
    with A1L2Worker(compiled) as worker:
        x = torch.arange(6400, dtype=torch.float32).reshape(400, 16) / 16
        dx = worker.upload(x)
        output = worker.empty((400, 16), torch.float32)
        graph = worker.prepare_spmd(compiled, dx, output)
        assert graph.task_count == 1
        graph()
        assert torch.equal(output.to_cpu(), x + 0.5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
