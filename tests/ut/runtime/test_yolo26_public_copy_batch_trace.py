# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Validate COPY opt-in against a real batch-eight model trace on the host."""

import os
from pathlib import Path

import pytest
import torch

from examples.models.yolo26_native.executor import NativeYOLO26
from examples.models.yolo26_native.executor_batch import NativeYOLO26Batch
from examples.models.yolo26_native.precision import prepare_fp16_model


class Memory:
    def __init__(self, count, dtype):
        self.count, self.dtype = count, dtype


class Kernel:
    def __init__(self, key):
        self._params = (key,)


class Worker:
    def __init__(self):
        self.graphs = []

    def upload(self, tensor):
        return Memory(tensor.numel(), tensor.dtype)

    def empty(self, shape, dtype):
        return Memory(shape[0], dtype)

    def prepare_independent(self, artifact, count):
        assert count == 8
        return Kernel(artifact)

    def register_graph(self, graph):
        self.graphs.append(graph)
        return lambda: None

    def close(self):
        pass


def normalize_aliases(trace):
    names = {}
    normalized = []
    for task in trace:
        row = []
        for arg in task.arguments:
            name = names.setdefault(id(arg), len(names))
            row.append((name, arg.count, str(arg.dtype)))
        normalized.append(row)
    return normalized


@pytest.fixture
def single_torch_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_real_batch_copy_preserves_graph(monkeypatch, tmp_path, single_torch_thread):
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
    weights = Path(os.environ.get("YOLO26_WEIGHTS", ""))
    if not weights.is_file():
        pytest.skip("set YOLO26_WEIGHTS to run the real-model host trace")
    YOLO = pytest.importorskip("ultralytics").YOLO
    model = YOLO(str(weights)).model.eval()
    model.model[-1].end2end = True
    model = prepare_fp16_model(model)
    holder["worker"] = Worker()
    baseline = NativeYOLO26Batch(model, 640, tmp_path / "baseline")
    holder["worker"] = Worker()
    candidate = NativeYOLO26Batch(model, 640, tmp_path / "public", experimental_public_copy=True)

    assert [(t.label, t.blocks) for t in baseline.trace] == [(t.label, t.blocks) for t in candidate.trace]
    assert normalize_aliases(baseline.trace) == normalize_aliases(candidate.trace)
    copy_tasks = [t for t in candidate.trace if t.label == "vector:COPY"]
    assert len({t.kernel._params for t in copy_tasks}) == 6
    assert all(t.blocks == 8 and len(t.arguments) == 32 for t in copy_tasks)
    assert all(t.blocks == 1 for image in candidate.images for t in image.trace if t.label == "vector:COPY")
    for old, new in zip(baseline.trace, candidate.trace, strict=True):
        if old.label != "vector:COPY":
            assert old.kernel._params == new.kernel._params


def test_real_batch_copy_with_spmd_attention(monkeypatch, tmp_path, single_torch_thread):
    from examples.models.yolo26_native import spmd_qkv  # noqa: PLC0415

    class TraceMemory(Memory):
        def __init__(self, count, dtype, contents=None, parent=None):
            super().__init__(count, dtype)
            self.contents = contents
            self.parent = parent

        def to_cpu(self):
            assert self.contents is not None
            return self.contents

        def view(self, shape):
            return TraceMemory(int(torch.tensor(shape).prod()), self.dtype, parent=self)

    class TraceWorker(Worker):
        def upload(self, tensor):
            return TraceMemory(tensor.numel(), tensor.dtype, tensor.clone())

        def empty(self, shape, dtype):
            return TraceMemory(shape[0], dtype)

        def prepare(self, artifact):
            return Kernel(artifact)

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
    weights = Path(os.environ.get("YOLO26_WEIGHTS", ""))
    if not weights.is_file():
        pytest.skip("set YOLO26_WEIGHTS to run the real-model host trace")
    YOLO = pytest.importorskip("ultralytics").YOLO
    model = YOLO(str(weights)).model.eval()
    model.model[-1].end2end = True
    model = prepare_fp16_model(model)
    holder["worker"] = TraceWorker()
    baseline = NativeYOLO26Batch(model, 640, tmp_path / "baseline-spmd", spmd_attention_pack=True)
    holder["worker"] = TraceWorker()
    candidate = NativeYOLO26Batch(
        model, 640, tmp_path / "public-spmd", spmd_attention_pack=True, experimental_public_copy=True
    )

    assert [(task.label, task.blocks) for task in baseline.trace] == [
        (task.label, task.blocks) for task in candidate.trace
    ]
    assert normalize_aliases(baseline.trace) == normalize_aliases(candidate.trace)
    assert sum(task.label == "spmd:attention-pack" for task in candidate.trace) == 2
    # The SPMD helper attaches this metadata dynamically to the batch.
    replacements = getattr(candidate, "_spmd_attention_replacements")
    assert len(replacements) == 2
    assert all(len(task.arguments) == 56 for task in candidate.trace if task.label == "spmd:attention-pack")
    packed = [task for task in candidate.trace if task.label == "spmd:attention-pack"]
    for task, replacement in zip(packed, replacements, strict=True):
        start = replacement["parent_indices"][0]
        for image_index, image in enumerate(candidate.images):
            group = image.trace[start : start + 14]
            args = task.arguments[image_index * 7 : (image_index + 1) * 7]
            assert args[0].parent is group[0].arguments[1]
            for actual, index in zip(args[1:], (3, 1, 2, 10, 8, 9), strict=True):
                assert actual.parent is group[index].arguments[-1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
