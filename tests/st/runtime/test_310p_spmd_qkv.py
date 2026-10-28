# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Require public-SPMD QKV packing to preserve the native parent bit patterns."""

import pytest
import torch
from pypto import ir
from pypto.backend import BackendType
from pypto.runtime import A1L2Worker
from yolo26_native_test_utils import require_310p

from examples.models.yolo26_native.layout import VectorOp, VectorPlan
from examples.models.yolo26_native.spmd_qkv import pack_attention8


def test_310p_spmd_qkv_matches_native_routing(tmp_path):
    require_310p()
    compiled = ir.compile(
        pack_attention8.specialize(),
        backend_type=BackendType.Ascend310P,
        output_dir=str(tmp_path / "spmd"),
        dump_passes=False,
    )
    with A1L2Worker(compiled) as worker:
        dummy = worker.upload(torch.zeros(16, dtype=torch.float16))
        arguments, expected, guards = [], [], []
        # Shift columns per image so all FP16 payload classes reach Q, K and V.
        for image in range(8):
            bits = (torch.arange(400 * 256, dtype=torch.int32) + image * 32).remainder(65536).to(torch.int16)
            source = worker.upload(bits.view(torch.float16))
            arguments.append(source.view((400, 256)))
            image_expected = []
            for head in range(2):
                reference = []
                for channels, offset in ((32, head * 128), (32, head * 128 + 32), (64, head * 128 + 64)):
                    plan = VectorPlan(VectorOp.COPY, 20, 20, channels, 256, channels, offset)
                    parent = ir.compile(
                        plan.jit_kernel().specialize(),
                        backend_type=BackendType.Ascend310P,
                        output_dir=str(tmp_path / f"copy-{image}-{offset}"),
                        dump_passes=False,
                    )
                    output = worker.empty((400 * channels,), torch.float16)
                    worker.prepare(parent)(worker.upload(plan.metadata()), source, dummy, output, blocks=1)
                    reference.append(output)
                scale = VectorPlan(VectorOp.SCALE_HALF, 400 * 32, scale=32**-0.5)
                parent = ir.compile(
                    scale.jit_kernel().specialize(),
                    backend_type=BackendType.Ascend310P,
                    output_dir=str(tmp_path / f"scale-{image}-{head}"),
                    dump_passes=False,
                )
                scaled = worker.empty((400 * 32,), torch.float16)
                worker.prepare(parent)(worker.upload(scale.metadata()), reference[0], dummy, scaled, blocks=1)
                image_expected.extend((scaled, reference[1], reference[2]))
            expected.extend(image_expected)
            for output, columns in zip(image_expected, (32, 32, 64, 32, 32, 64), strict=True):
                guard = worker.upload(torch.full((400 * columns + 32,), 7.0, dtype=torch.float16))
                guards.append(guard)
                arguments.append(guard.view((400, columns), 16))
        graph = worker.prepare_spmd(ir.CompiledProgram.from_dir(compiled.output_dir), *arguments)
        assert graph.task_count == 1
        for _ in range(2):
            graph()
            for guard, parent in zip(guards, expected, strict=True):
                actual = guard.to_cpu()
                assert torch.all(actual[:16] == 7) and torch.all(actual[-16:] == 7)
                assert torch.equal(actual[16:-16].view(torch.int16), parent.to_cpu().view(torch.int16))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
