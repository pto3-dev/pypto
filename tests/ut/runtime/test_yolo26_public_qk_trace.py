# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host-only batch graph checks for experimental public QK."""

import os
from pathlib import Path

import pytest
import torch

from examples.models.yolo26_native.executor import NativeYOLO26
from examples.models.yolo26_native.executor_batch import NativeYOLO26Batch
from examples.models.yolo26_native.precision import prepare_fp16_model


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
        pytest.skip("set YOLO26_WEIGHTS to run the real-model host trace")
    YOLO = pytest.importorskip("ultralytics").YOLO
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        result = YOLO(str(weights)).model.eval()
        result.model[-1].end2end = True
        yield prepare_fp16_model(result)
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize("spmd", [False, True])
def test_public_qk_preserves_batch_graph(monkeypatch, tmp_path, model, spmd):
    from examples.models.yolo26_native import spmd_qkv  # noqa: PLC0415

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
    monkeypatch.setattr(spmd_qkv.ir, "compile", lambda *_args, **_kwargs: "mock-spmd-artifact")
    holder["worker"] = Worker()
    baseline = NativeYOLO26Batch(model, 640, tmp_path / "baseline", spmd_attention_pack=spmd)
    holder["worker"] = Worker()
    candidate = NativeYOLO26Batch(
        model, 640, tmp_path / "public", spmd_attention_pack=spmd, experimental_public_qk=True
    )
    assert [(task.label, task.blocks) for task in baseline.trace] == [
        (task.label, task.blocks) for task in candidate.trace
    ]
    assert _aliases(baseline.trace) == _aliases(candidate.trace)
    changed = [
        (old, new)
        for old, new in zip(baseline.trace, candidate.trace, strict=True)
        if old.kernel._params != new.kernel._params
    ]
    assert len(changed) == 4
    for old, new in changed:
        assert old.label == new.label == "matrix"
        assert "m=400, n=400, k=32, transpose_b=True" in str(new.kernel._params)
        assert "experimental_public_qk=True" in str(new.kernel._params)
        assert len(new.arguments) == 32


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
