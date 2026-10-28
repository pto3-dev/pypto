# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Preserve row-softmax bit patterns while coalescing tasks and FP16 casts."""

import pytest
import torch
from pypto import ir
from pypto.backend import BackendType
from pypto.runtime import A1L2Worker, A1ResidentSession
from yolo26_native_test_utils import load_plan, require_310p

from examples.models.yolo26_native.softmax import Softmax400Plan, _softmax_kernel


@pytest.mark.parametrize("blocks", [1, 8])
def test_310p_persistent_softmax_matches_parent_rows(tmp_path, blocks):
    """Exercise uneven row ownership, changed inputs, artifact reload and view guards."""
    require_310p()
    plan = load_plan("softmax").Softmax400Plan()
    jit = plan.jit_kernel()
    assert not jit._external_source_paths()
    compiled = ir.compile(
        jit.specialize(),
        backend_type=BackendType.Ascend310P,
        output_dir=str(tmp_path / "persistent"),
        dump_passes=False,
    )
    parent = ir.compile(
        _softmax_kernel(400).specialize(),
        backend_type=BackendType.Ascend310P,
        output_dir=str(tmp_path / "parent"),
        dump_passes=False,
    )
    generator = torch.Generator().manual_seed(317)
    cases = (
        (torch.randn((400, 400), generator=generator) * 3).half(),
        torch.zeros((400, 400), dtype=torch.float16),
        torch.linspace(-80, 80, 400).half().repeat(400, 1),
        torch.eye(400, dtype=torch.float16) * 80,
    )
    with A1ResidentSession(compiled) as session:
        kernel = session.prepare(ir.CompiledProgram.from_dir(compiled.output_dir))
        old = session.prepare(parent)
        metadata = session.upload(plan.metadata())
        source = session.empty((160000,), torch.float16)
        floats = session.empty((160000,), torch.float32)
        parent_output = session.empty((160000,), torch.float32)
        guard = session.upload(torch.full((160032,), 7.0, dtype=torch.float16))
        output = guard.view((160000,), 16)
        for scores in cases:
            source.copy_from(scores.reshape(-1))
            floats.copy_from(scores.float().reshape(-1))
            for row in range(0, 400, 16):
                old(floats.view((16, 400), row * 400), parent_output.view((16, 400), row * 400), blocks=1)
            kernel(metadata, source, output, blocks=blocks)
            # The parent row kernel and its separate FP16 narrowing are the
            # exact reference; no numerical tolerance is added by this feature.
            expected = parent_output.to_cpu().half()
            assert torch.equal(output.to_cpu().view(torch.int16), expected.view(torch.int16))
            full = guard.to_cpu()
            assert torch.all(full[:16] == 7) and torch.all(full[-16:] == 7)


def test_310p_pypto_softmax_independent_images(tmp_path):
    """Keep the production eight-image dispatch and output views bitwise exact."""
    require_310p()
    plan = Softmax400Plan()
    compiled = ir.compile(
        plan.jit_kernel().specialize(),
        backend_type=BackendType.Ascend310P,
        output_dir=str(tmp_path / "pypto"),
        dump_passes=False,
    )
    reference = ir.compile(
        _softmax_kernel(400).specialize(),
        backend_type=BackendType.Ascend310P,
        output_dir=str(tmp_path / "reference"),
        dump_passes=False,
    )
    generator = torch.Generator().manual_seed(5317)
    with A1L2Worker(compiled) as worker:
        packed = worker.prepare_independent(ir.CompiledProgram.from_dir(compiled.output_dir), 8)
        old = worker.prepare(reference)
        metadata = worker.upload(plan.metadata())
        inputs = [worker.empty((160000,), torch.float16) for _ in range(8)]
        guards = [worker.upload(torch.full((160032,), 7.0, dtype=torch.float16)) for _ in range(8)]
        outputs = [guard.view((160000,), 16) for guard in guards]
        arguments = tuple(
            item for source, out in zip(inputs, outputs, strict=True) for item in (metadata, source, out)
        )
        graph = worker.register_graph([(packed, arguments, 8)])
        assert graph.task_count == 1
        floats = worker.empty((400, 400), torch.float32)
        normalized = worker.empty((400, 400), torch.float32)
        for _round in range(2):
            scores = [
                (torch.randn((400, 400), generator=generator) * (image + 1)).half() for image in range(8)
            ]
            for source, values in zip(inputs, scores, strict=True):
                source.copy_from(values.reshape(-1))
            graph()
            for values, out, guard in zip(scores, outputs, guards, strict=True):
                floats.copy_from(values.float())
                for row in range(0, 400, 16):
                    old(floats.view((16, 400), row * 400), normalized.view((16, 400), row * 400), blocks=1)
                expected = normalized.to_cpu().half().reshape(-1)
                assert torch.equal(out.to_cpu().view(torch.int16), expected.view(torch.int16))
                full = guard.to_cpu()
                assert torch.all(full[:16] == 7) and torch.all(full[-16:] == 7)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
