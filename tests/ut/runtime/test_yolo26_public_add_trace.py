# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Real-weight host graph and production-shape compile checks for public ADD."""

import os
import shutil
from pathlib import Path

import pytest
import torch
from pypto import ir
from pypto.backend import BackendType

from examples.models.yolo26_native.executor import NativeYOLO26, PublicAddYOLO26
from examples.models.yolo26_native.executor_batch import NativeYOLO26Batch
from examples.models.yolo26_native.layout import VectorOp
from examples.models.yolo26_native.precision import prepare_fp16_model
from examples.models.yolo26_native.public_add import PublicAddPlan


class Memory:
    def __init__(self, count, dtype, contents=None, parent=None):
        self.count, self.dtype, self.contents, self.parent = count, dtype, contents, parent

    def to_cpu(self):
        assert self.contents is not None
        return self.contents

    def view(self, shape):
        return Memory(int(torch.tensor(shape).prod()), self.dtype, parent=self)


class Kernel:
    def __init__(self, key):
        self._params = (key,)


class Worker:
    def upload(self, tensor):
        return Memory(tensor.numel(), tensor.dtype, tensor.clone())

    def empty(self, shape, dtype):
        return Memory(shape[0], dtype)

    def prepare_independent(self, artifact, count):
        assert count == 8
        return Kernel(artifact)

    def prepare(self, artifact):
        return Kernel(artifact)

    def register_graph(self, _graph):
        return lambda: None

    def close(self):
        pass


def _aliases(trace):
    names = {}
    return [
        [(names.setdefault(id(arg), len(names)), arg.count, str(arg.dtype)) for arg in task.arguments]
        for task in trace
    ]


@pytest.fixture
def model():
    weights = Path(os.environ.get("YOLO26_WEIGHTS", ""))
    if not weights.is_file():
        pytest.skip("set YOLO26_WEIGHTS for the real-model host trace")
    YOLO = pytest.importorskip("ultralytics").YOLO
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        result = YOLO(str(weights)).model.eval()
        result.model[-1].end2end = True
        yield prepare_fp16_model(result)
    finally:
        torch.set_num_threads(previous)


@pytest.fixture
def mock_prepare(monkeypatch):
    holder = {}

    def prepare(self, _jit, key):
        if self.session is None:
            self.session = holder["worker"]
        if key not in self._prepared:
            kernel = Kernel(key)
            self._prepared[key] = kernel
            self._artifacts[kernel] = key
        return self._prepared[key]

    monkeypatch.setattr(NativeYOLO26, "_prepare", prepare)
    return holder


def _compare_graphs(baseline, candidate, signatures):
    assert [(t.label, t.blocks) for t in baseline.trace] == [(t.label, t.blocks) for t in candidate.trace]
    assert _aliases(baseline.trace) == _aliases(candidate.trace)
    changed = [
        (a, b)
        for a, b in zip(baseline.trace, candidate.trace, strict=True)
        if a.kernel._params != b.kernel._params
    ]
    adds = [t for t in candidate.trace if t.label == "vector:ADD"]
    assert len(changed) == len(adds) > 0
    for old, new in changed:
        assert old.label == new.label == "vector:ADD"
        assert "PublicAddPlan(" in str(new.kernel._params)
        assert len(new.arguments) == signatures * 4
        for image in range(signatures):
            index = image * 4
            torch.testing.assert_close(
                old.arguments[index].contents, new.arguments[index].contents, rtol=0, atol=0
            )
        assert len({id(new.arguments[i * 4]) for i in range(signatures)}) == 1
        for offset in (1, 2, 3):
            assert len({id(new.arguments[i * 4 + offset]) for i in range(signatures)}) == signatures
    return adds


def test_public_add_batch_graph_and_production_compile(monkeypatch, mock_prepare, model, tmp_path):
    if not shutil.which("ptoas"):
        pytest.skip("requires PTOAS for production-shape public compilation")
    from examples.models.yolo26_native import spmd_qkv  # noqa: PLC0415

    plans = {}
    for spmd in (False, True):
        # Mock only the SPMD artifact while recording graphs. Restore the
        # compiler before compiling every actual residual-add parameter group.
        with monkeypatch.context() as context:
            context.setattr(spmd_qkv.ir, "compile", lambda *_a, **_k: "mock-spmd-artifact")
            mock_prepare["worker"] = Worker()
            baseline = NativeYOLO26Batch(model, 640, tmp_path / "baseline", spmd_attention_pack=spmd)
            mock_prepare["worker"] = Worker()
            candidate = NativeYOLO26Batch(
                model, 640, tmp_path / "candidate", spmd_attention_pack=spmd, experimental_public_add=True
            )
            adds = _compare_graphs(baseline, candidate, 8)
            for task in adds:
                values = task.arguments[0].contents.tolist()
                plan = PublicAddPlan(
                    VectorOp.ADD,
                    int(values[1]),
                    int(values[2]),
                    int(values[3]),
                    int(values[4]),
                    int(values[5]),
                    int(values[6]),
                    int(values[7]),
                    second_stride=int(values[10]),
                    second_offset=int(values[11]),
                )
                torch.testing.assert_close(plan.metadata(), task.arguments[0].contents, rtol=0, atol=0)
                plans[repr(plan)] = plan
            baseline.close()
            candidate.close()
    assert plans
    for index, plan in enumerate(plans.values()):
        output = tmp_path / "production" / str(index)
        compiled = ir.compile(
            plan.jit_kernel().specialize(), backend_type=BackendType.Ascend310P, output_dir=str(output)
        )
        assert isinstance(compiled, ir.CompiledProgram)
        assert len(list((output / "kernels").glob("*/residual_add.cpp"))) == 1


@pytest.mark.parametrize("size", [128, 640])
def test_public_add_single_image_graph(mock_prepare, model, tmp_path, size):
    mock_prepare["worker"] = Worker()
    baseline = NativeYOLO26(
        model, size, tmp_path / "baseline", four_way_accumulation=True, precise_small_softmax=True
    )
    mock_prepare["worker"] = Worker()
    candidate = PublicAddYOLO26(
        model, size, tmp_path / "candidate", four_way_accumulation=True, precise_small_softmax=True
    )
    _compare_graphs(baseline, candidate, 1)
    baseline.close()
    candidate.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
