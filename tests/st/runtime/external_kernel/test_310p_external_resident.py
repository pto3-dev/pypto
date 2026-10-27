# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Public standalone external-source execution on Ascend310P."""

import ctypes
from pathlib import Path

import pypto.language as pl
import pytest
import torch
from pypto import ir
from pypto.backend import BackendType
from pypto.runtime import A1L2Worker, A1ResidentSession


def test_external_compile_and_call(tmp_path):
    try:
        acl = ctypes.CDLL("libascendcl.so")
    except OSError:
        pytest.skip("Ascend runtime unavailable")
    acl.aclrtGetSocName.restype = ctypes.c_char_p
    if not acl.aclrtGetSocName().startswith(b"Ascend310P"):
        pytest.skip("requires Ascend310P")
    source = Path(__file__).parent / "kernels/a1/resident_add.cpp"

    @pl.jit.extern(core_type="aic", source=source)
    def native(
        a: pl.Tensor[[1, 256], pl.FP32],
        b: pl.Tensor[[1, 256], pl.FP32],
        out: pl.Out[pl.Tensor[[1, 256], pl.FP32]],
    ): ...

    compiled = ir.compile(
        native.specialize(), backend_type=BackendType.Ascend310P, output_dir=str(tmp_path), dump_passes=False
    )
    assert (tmp_path / "a1_external_kernel.json").is_file()
    assert len(list((tmp_path / "kernels").glob("*/*.cpp"))) == 1
    x = torch.arange(256).reshape(1, 256).float() / 16
    y = x.flip([1]).contiguous()
    out = torch.zeros_like(x)
    compiled(x, y, out)
    assert torch.equal(out, x + y)
    reloaded = ir.CompiledProgram.from_dir(compiled.output_dir)
    reloaded(x * 2, y, out)
    assert torch.equal(out, x * 2 + y)

    with A1ResidentSession(compiled) as session:
        add = session.prepare(compiled)
        da, db = session.upload(x), session.upload(y)
        dc = session.empty((1, 256), torch.float32)
        add(da, db, dc)
        add(dc, db, dc)
        assert torch.equal(dc.to_cpu(), x + 2 * y)
        da.copy_from(x * 2)
        reloaded_add = session.prepare(reloaded)
        reloaded_add(da, db, dc)
        assert torch.equal(dc.to_cpu(), x * 2 + y)
        with pytest.raises(ValueError, match="signature"):
            add(da.view((16, 16)), db, dc)
    with pytest.raises(RuntimeError, match="closed"):
        add(da, db, dc)


@pytest.mark.parametrize("instances", [2, 3, 8])
@pytest.mark.parametrize("kind", ["extern", "generated"])
def test_independent_instances_dependent_graph(tmp_path, instances, kind):
    """Each physical core must use its own complete signature, including views.

    The extern body offsets by logical block_idx; a missing virtual grid would
    corrupt guards on cores above zero. Two dependent tasks and changing inputs
    expose stale descriptors, cross-instance aliasing and incorrect graph order.
    """
    try:
        acl = ctypes.CDLL("libascendcl.so")
    except OSError:
        pytest.skip("Ascend runtime unavailable")
    acl.aclrtGetSocName.restype = ctypes.c_char_p
    if not acl.aclrtGetSocName().startswith(b"Ascend310P"):
        pytest.skip("requires Ascend310P")
    source = Path(__file__).parent / "kernels/a1/resident_add.cpp"

    @pl.jit.extern(core_type="aic", source=source)
    def native(
        a: pl.Tensor[[1, 256], pl.FP32],
        b: pl.Tensor[[1, 256], pl.FP32],
        out: pl.Out[pl.Tensor[[1, 256], pl.FP32]],
    ): ...

    if kind == "generated":

        @pl.jit.incore
        def generated(
            a: pl.Tensor[[1, 256], pl.FP32],
            b: pl.Tensor[[1, 256], pl.FP32],
            out: pl.Out[pl.Tensor[[1, 256], pl.FP32]],
        ):
            lhs = pl.load(a, [0, 0], [1, 256])
            rhs = pl.load(b, [0, 0], [1, 256])
            result = pl.add(lhs, rhs)
            out = pl.store(result, [0, 0], out)
            return out

        native = generated

    compiled = ir.compile(
        native.specialize(), backend_type=BackendType.Ascend310P, output_dir=str(tmp_path), dump_passes=False
    )
    with A1L2Worker(compiled) as worker:
        kernel = worker.prepare_independent(ir.CompiledProgram.from_dir(compiled.output_dir), instances)
        state = []
        for image in range(instances):
            x = torch.arange(256).float().reshape(1, 256) / 16 + image
            b = torch.full_like(x, (image + 1) / 8)
            guards = [worker.upload(torch.full((288,), 7.0)) for _ in range(2)]
            mid, out = [g.view((1, 256), 16) for g in guards]
            state.append((x, b, worker.upload(x), worker.upload(b), mid, out, guards))
        calls = []
        for stage in range(2):
            arguments = tuple(
                arg
                for row in state
                for arg in (row[2] if stage == 0 else row[4], row[3], row[4] if stage == 0 else row[5])
            )
            calls.append((kernel, arguments, instances))
        graph = worker.register_graph(calls)
        assert graph.task_count == 2
        for replay in range(2):
            for x, _b, dx, *_rest in state:
                dx.copy_from(x + replay)
            graph()
            for x, b, _dx, _db, mid, out, guards in state:
                assert torch.equal(mid.to_cpu(), (x + replay) + b)
                assert torch.equal(out.to_cpu(), ((x + replay) + b) + b)
                for guard in guards:
                    actual = guard.to_cpu()
                    assert torch.all(actual[:16] == 7) and torch.all(actual[-16:] == 7)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
