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
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from pypto import ir
from pypto.backend import BackendType

from examples.models.yolo26_native.executor import (
    Feature,
    Launch,
    NativeYOLO26,
    PublicAddYOLO26,
    PublicUpsampleConcatYOLO26,
)
from examples.models.yolo26_native.executor_batch import NativeYOLO26Batch
from examples.models.yolo26_native.layout import ConcatPlan, VectorOp, VectorPlan
from examples.models.yolo26_native.precision import prepare_fp16_model
from examples.models.yolo26_native.public_add import PublicAddPlan
from examples.models.yolo26_native.upsample_concat import UpsampleConcatPlan, fuse_upsample_concat


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


def _compare_fused_graphs(baseline, candidate, signatures):
    # Keep a bijection across every retained task, rather than only checking
    # labels: shared weights and all producer/consumer aliases must survive.
    mapping, inverse = {}, {}

    def bind(old, new):
        assert (old.count, old.dtype) == (new.count, new.dtype)
        assert mapping.setdefault(id(old), new) is new
        assert inverse.setdefault(id(new), old) is old
        if old.contents is not None:
            torch.testing.assert_close(old.contents, new.contents, rtol=0, atol=0)

    i = 0
    count = 0
    feature_views = {}
    for task in candidate.trace:
        old = baseline.trace[i]
        if task.label != "route:upsample-concat":
            assert (old.label, old.blocks, old.kernel._params) == (
                task.label,
                task.blocks,
                task.kernel._params,
            )
            assert len(old.arguments) == len(task.arguments)
            for a, b in zip(old.arguments, task.arguments, strict=True):
                bind(a, b)
            i += 1
            continue
        concat = baseline.trace[i + 1]
        assert (old.label, concat.label) == ("vector:UPSAMPLE", "concat")
        assert old.blocks == concat.blocks == task.blocks
        assert len(task.arguments) == 4 * signatures
        assert len({id(task.arguments[4 * k]) for k in range(signatures)}) == 1
        for position in (1, 2, 3):
            assert len({id(task.arguments[4 * k + position]) for k in range(signatures)}) == signatures
        for k in range(signatures):
            u, c, f = (
                old.arguments[k * 4 : k * 4 + 4],
                concat.arguments[k * 6 : k * 6 + 6],
                task.arguments[k * 4 : k * 4 + 4],
            )
            assert c[1] is u[-1]  # Both real model concat sites use upsample first.
            uv = [int(v) for v in u[0].contents.tolist()]
            cv = [int(v) for v in c[0].contents.tolist()]
            plan = UpsampleConcatPlan(uv[8], uv[9], uv[3], cv[5], uv[4], cv[9], uv[6], cv[13])
            torch.testing.assert_close(plan.metadata(), f[0].contents, rtol=0, atol=0)
            assert "UpsampleConcatPlan(" in str(task.kernel._params)
            bind(u[1], f[1])
            bind(c[2], f[2])
            bind(c[-1], f[-1])
            feature_views[id(u[-1])] = (f[-1], plan.up_channels + plan.skip_channels)
        i += 2
        count += 1
    assert i == len(baseline.trace)
    assert count == 2 and len(candidate.trace) == len(baseline.trace) - 2
    old_images = getattr(baseline, "images", [baseline])
    new_images = getattr(candidate, "images", [candidate])
    for old_image, new_image in zip(old_images, new_images, strict=True):
        assert old_image.features.keys() == new_image.features.keys()
        assert len(new_image._upsample_concat_fusions) == 2
        for key, old_feature in old_image.features.items():
            new_feature = new_image.features[key]
            assert (old_feature.height, old_feature.width, old_feature.channels) == (
                new_feature.height,
                new_feature.width,
                new_feature.channels,
            )
            if id(old_feature.data) in feature_views:
                data, stride = feature_views[id(old_feature.data)]
                assert new_feature.data is data and new_feature.row_stride == stride
                assert new_feature.offset == old_feature.offset
            else:
                bind(old_feature.data, new_feature.data)
                assert (old_feature.row_stride, old_feature.offset) == (
                    new_feature.row_stride,
                    new_feature.offset,
                )
    return sum(r["removed_gm_bytes"] for image in new_images for r in image._upsample_concat_fusions)


@pytest.mark.parametrize("size", [128, 640])
def test_public_upsample_concat_single_graph(mock_prepare, model, tmp_path, size):
    mock_prepare["worker"] = Worker()
    baseline = NativeYOLO26(
        model, size, tmp_path / "baseline", four_way_accumulation=True, precise_small_softmax=True
    )
    mock_prepare["worker"] = Worker()
    candidate = PublicUpsampleConcatYOLO26(
        model, size, tmp_path / "candidate", four_way_accumulation=True, precise_small_softmax=True
    )
    removed = _compare_fused_graphs(baseline, candidate, 1)
    assert removed == (4915200 if size == 640 else 196608)
    baseline.close()
    candidate.close()


@pytest.mark.parametrize("spmd", [False, True])
@pytest.mark.parametrize("public_add", [False, True])
def test_public_upsample_concat_batch_graph(monkeypatch, mock_prepare, model, tmp_path, spmd, public_add):
    from examples.models.yolo26_native import spmd_qkv  # noqa: PLC0415

    monkeypatch.setattr(spmd_qkv.ir, "compile", lambda *_a, **_k: "mock-spmd-artifact")
    mock_prepare["worker"] = Worker()
    baseline = NativeYOLO26Batch(
        model, 640, tmp_path / "baseline", spmd_attention_pack=spmd, experimental_public_add=public_add
    )
    mock_prepare["worker"] = Worker()
    candidate = NativeYOLO26Batch(
        model,
        640,
        tmp_path / "candidate",
        spmd_attention_pack=spmd,
        experimental_public_add=public_add,
        experimental_upsample_concat=True,
    )
    assert _compare_fused_graphs(baseline, candidate, 8) == 39321600
    assert len(candidate.trace) == (167 if spmd else 181)
    baseline.close()
    candidate.close()


def _small_fusion_graph(reverse=False):
    up = VectorPlan(VectorOp.UPSAMPLE, 2, 2, 16, 16, 16, input_height=1, input_width=1)
    sources = ((16, 16, 0), (32, 48, 16))
    concat = ConcatPlan(2, 2, sources[::-1] if reverse else sources)
    worker = Worker()
    low, skip, temporary, destination = (worker.empty((n,), torch.float16) for n in (16, 192, 64, 192))
    dummy = worker.upload(torch.zeros(16, dtype=torch.float16))
    producer = Launch(
        Kernel("up"), (worker.upload(up.metadata()), low, dummy, temporary), 8, "vector:UPSAMPLE"
    )
    inputs = (skip, temporary) if reverse else (temporary, skip)
    consumer = Launch(
        Kernel("concat"), (worker.upload(concat.metadata()), *inputs, dummy, dummy, destination), 8, "concat"
    )
    return SimpleNamespace(
        trace=[producer, consumer],
        features={7: Feature(temporary, 2, 2, 16, 16)},
        _launch_plans={id(producer): up, id(consumer): concat},
        session=worker,
        _prepare=lambda _jit, key: Kernel(key),
        _upsample_concat_fusions=[],
    )


@pytest.mark.parametrize("reverse", [False, True])
def test_fusion_preserves_debug_channel_view(reverse):
    runner = _small_fusion_graph(reverse)
    old_temporary = runner.trace[0].arguments[-1]
    runner.features[8] = runner.features[7].channel_slice(4, 8)
    fuse_upsample_concat(runner)
    assert len(runner.trace) == 1
    task = runner.trace[0]
    plan = runner._launch_plans[id(task)]
    assert plan.upsample_first is not reverse
    assert task.arguments[-1] is not old_temporary
    assert runner._upsample_concat_fusions[0]["removed_gm_bytes"] == 256
    # Debug reads must still see the logical upsample and nested channel slice,
    # even though the standalone upsample buffer is no longer written.
    destination = task.arguments[-1]
    destination.contents = torch.arange(192, dtype=torch.float16)
    base = destination.contents.reshape(2, 2, 48)
    start = 32 if reverse else 0
    for key, offset, channels in [(7, 0, 16), (8, 4, 8)]:
        feature = runner.features[key]
        assert feature.data is destination and feature.row_stride == 48
        assert feature.offset == start + offset
        expected = (
            base[:, :, start + offset : start + offset + channels].permute(2, 0, 1).unsqueeze(0).contiguous()
        )
        torch.testing.assert_close(feature.to_cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "hazard", ["reader", "alias_reader", "writer", "distinct_feature_view", "nonadjacent"]
)
def test_fusion_retains_intermediate_with_other_users(hazard):
    runner = _small_fusion_graph()
    old = runner.trace.copy()
    temporary = old[0].arguments[-1]
    other = Memory(64, torch.float16)
    if hazard == "distinct_feature_view":
        runner.features[8] = Feature(temporary.view((64,)), 2, 2, 16, 16)
    elif hazard == "nonadjacent":
        runner.trace.insert(1, Launch(Kernel("unrelated"), (other, other), 8, "unrelated"))
    elif hazard == "writer":
        runner.trace.insert(0, Launch(Kernel("writer"), (other, temporary), 8, "writer"))
    else:
        argument = temporary.view((64,)) if hazard == "alias_reader" else temporary
        runner.trace.append(Launch(Kernel("reader"), (argument, other), 8, "reader"))
    expected = runner.trace.copy()
    fuse_upsample_concat(runner)
    assert len(runner.trace) == len(expected)
    assert all(a is b for a, b in zip(runner.trace, expected, strict=True))
    assert runner.features[7].data is temporary
    assert runner._upsample_concat_fusions == []


@pytest.mark.parametrize("alias_reader", [False, True])
def test_fusion_counts_actual_resident_allocation_aliases(alias_reader):
    from simpler_setup.a1_single_launch import A1DeviceTensor, A1SingleLaunch  # noqa: PLC0415

    runner = _small_fusion_graph()
    producer, consumer = runner.trace
    # Bypass the owner constructor: descriptors need its identity, while
    # initializing a session would load ACL and open an NPU device.
    owner = A1SingleLaunch.__new__(A1SingleLaunch)
    resident = A1DeviceTensor(owner, 17, (64,), torch.float16, 128)
    producer = replace(producer, arguments=(*producer.arguments[:-1], resident))
    consumer = replace(consumer, arguments=(consumer.arguments[0], resident, *consumer.arguments[2:]))
    runner.trace = [producer, consumer]
    runner._launch_plans = {
        id(producer): VectorPlan(VectorOp.UPSAMPLE, 2, 2, 16, 16, 16, input_height=1, input_width=1),
        id(consumer): ConcatPlan(2, 2, ((16, 16, 0), (32, 48, 16))),
    }
    runner.features[7] = replace(runner.features[7], data=resident)
    if alias_reader:
        alias = resident.view((16,), offset_elements=16)
        assert alias is not resident
        runner.trace.append(Launch(Kernel("reader"), (alias, Memory(16, torch.float16)), 8, "reader"))
    fuse_upsample_concat(runner)
    assert len(runner.trace) == (3 if alias_reader else 1)
    assert len(runner._upsample_concat_fusions) == (0 if alias_reader else 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
