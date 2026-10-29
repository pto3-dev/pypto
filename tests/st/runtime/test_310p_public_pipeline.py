# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""310P public pipeline: independent UB slots, tail handling and exact results."""

import pypto.language as pl
import pytest
import torch
from pypto import ir
from pypto.backend import BackendType
from pypto.runtime import A1ResidentSession
from yolo26_native_test_utils import require_310p


def _pipeline_copy(tiles: int):
    rows = tiles * 16

    @pl.jit.incore
    def compute(x: pl.Tensor[[rows, 16], pl.FP32], out: pl.Out[pl.Tensor[[rows, 16], pl.FP32]]):
        for i in pl.pipeline(0, tiles, stage=2):
            value = pl.load(x, [i * 16, 0], [16, 16])
            out = pl.store(pl.add(value, value), [i * 16, 0], out)
        return out

    return compute


@pytest.mark.parametrize("tiles", [1, 5, 6])
def test_310p_public_two_stage_pipeline(tmp_path, tiles):
    require_310p()
    compiled = ir.compile(
        _pipeline_copy(tiles).specialize(),
        backend_type=BackendType.Ascend310P,
        output_dir=str(tmp_path),
        dump_passes=False,
    )
    with A1ResidentSession(compiled) as session:
        values = (torch.arange(tiles * 256, dtype=torch.float32) - 500) / 16
        source = session.upload(values)
        guard = session.upload(torch.full((tiles * 256 + 32,), -7.0, dtype=torch.float32))
        output = guard.view((tiles * 16, 16), 16)
        kernel = session.prepare(compiled)
        for delta in (0.0, 4.0):
            source.copy_from(values + delta)
            kernel(source.view((tiles * 16, 16)), output, blocks=1)
            assert torch.equal(output.to_cpu().reshape(-1), (values + delta) * 2)
            assert torch.equal(source.to_cpu(), values + delta)
            whole = guard.to_cpu()
            assert torch.all(whole[:16] == -7) and torch.all(whole[-16:] == -7)


def test_310p_public_two_static_slots(tmp_path):
    require_310p()
    scratch = pl.MemRef(slots=2)

    @pl.program
    class TwoSlots:
        @pl.function(type=pl.FunctionType.InCore)
        def compute(self, x: pl.Tensor[[32, 16], pl.FP32], out: pl.Out[pl.Tensor[[32, 16], pl.FP32]]):
            first: pl.Tile[[16, 16], pl.FP32, scratch[0], pl.Mem.Vec] = pl.load(x, [0, 0], [16, 16])
            second: pl.Tile[[16, 16], pl.FP32, scratch[1], pl.Mem.Vec] = pl.load(x, [16, 0], [16, 16])
            out = pl.store(first, [0, 0], out)
            out = pl.store(second, [16, 0], out)
            return out

    compiled = ir.compile(
        TwoSlots, backend_type=BackendType.Ascend310P, output_dir=str(tmp_path), dump_passes=False
    )
    with A1ResidentSession(compiled) as session:
        values = torch.arange(512, dtype=torch.float32).reshape(32, 16)
        source = session.upload(values)
        output = session.empty((32, 16), torch.float32)
        session.prepare(compiled)(source, output, blocks=1)
        assert torch.equal(output.to_cpu(), values)
        assert torch.equal(source.to_cpu(), values)


@pytest.mark.parametrize("slots,tiles", [(2, 17), (4, 32)])
def test_310p_public_runtime_slots(tmp_path, slots, tiles):
    require_310p()
    rows = tiles * 16
    scratch = pl.MemRef(slots=slots)

    @pl.program
    class RotatingSlots:
        @pl.function(type=pl.FunctionType.InCore)
        def compute(self, x: pl.Tensor[[rows, 16], pl.FP32], out: pl.Out[pl.Tensor[[rows, 16], pl.FP32]]):
            for i in pl.range(tiles):
                value: pl.Tile[[16, 16], pl.FP32, scratch[i % slots], pl.Mem.Vec] = pl.load(
                    x, [i * 16, 0], [16, 16]
                )
                out = pl.store(value, [i * 16, 0], out)
            return out

    compiled = ir.compile(
        RotatingSlots, backend_type=BackendType.Ascend310P, output_dir=str(tmp_path), dump_passes=False
    )
    with A1ResidentSession(compiled) as session:
        source = session.empty((rows, 16), torch.float32)
        guard = session.upload(torch.full((rows * 16 + 32,), -7.0, dtype=torch.float32))
        output = guard.view((rows, 16), 16)
        kernel = session.prepare(compiled)
        for seed in (541, 547, 557):
            generator = torch.Generator().manual_seed(seed)
            values = torch.randint(-2048, 2048, (rows, 16), generator=generator).float()
            source.copy_from(values)
            kernel(source, output, blocks=1)
            assert torch.equal(output.to_cpu(), values)
            assert torch.equal(source.to_cpu(), values)
            full = guard.to_cpu()
            assert torch.all(full[:16] == -7) and torch.all(full[-16:] == -7)


def test_310p_jit_inline_runtime_slots(tmp_path):
    require_310p()

    @pl.jit.incore
    def compute(x: pl.Tensor[[80, 16], pl.FP32], out: pl.Out[pl.Tensor[[80, 16], pl.FP32]]):
        for i in pl.range(5):
            value: pl.Tile[[16, 16], pl.FP32, pl.MemRef("scratch", slots=2)[i % 2], pl.Mem.Vec] = pl.load(
                x, [i * 16, 0], [16, 16]
            )
            out = pl.store(value, [i * 16, 0], out)
        return out

    compiled = ir.compile(
        compute.specialize(), backend_type=BackendType.Ascend310P, output_dir=str(tmp_path), dump_passes=False
    )
    with A1ResidentSession(compiled) as session:
        source = session.empty((80, 16), torch.float32)
        guard = session.upload(torch.full((80 * 16 + 32,), -7.0, dtype=torch.float32))
        output = guard.view((80, 16), 16)
        kernel = session.prepare(compiled)
        for seed in (563, 569):
            generator = torch.Generator().manual_seed(seed)
            values = torch.randint(-2048, 2048, (80, 16), generator=generator).float()
            source.copy_from(values)
            kernel(source, output, blocks=1)
            assert torch.equal(output.to_cpu(), values)
            assert torch.equal(source.to_cpu(), values)
            full = guard.to_cpu()
            assert torch.all(full[:16] == -7) and torch.all(full[-16:] == -7)
